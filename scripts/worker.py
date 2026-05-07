#!/usr/bin/env python3
"""Nakshatra worker (M5) — gRPC server that wraps a llama-nakshatra-worker
daemon subprocess running our patched libllama.

The daemon owns the heavy state (model, KV cache); this Python process owns
the gRPC surface. Each Forward RPC is one round-trip to the daemon over
stdin/stdout.

CLI:
  --port           gRPC listen port
  --sub-gguf       path to the sub-GGUF this worker holds
  --mode           first | middle | last (matches the cluster config)
  --layer-start, --layer-end
                   declared by the cluster config; reported via Info
  --model-id       human-readable model id (matches cluster config)
  --daemon-bin     path to llama-nakshatra-worker binary
  --n-ctx          context length cap (default 256)
"""
import argparse
import struct
import subprocess
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc


CMD_TOKEN_DECODE = 1
CMD_EMBD_DECODE  = 2
CMD_INFO         = 3


class DaemonClient:
    """Manages a long-lived llama-nakshatra-worker subprocess over stdin/stdout."""

    def __init__(self, daemon_bin: str, sub_gguf: str, mode: str, n_ctx: int, n_threads: int = 0):
        self.proc = subprocess.Popen(
            [daemon_bin, sub_gguf, mode, str(n_ctx), str(n_threads)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.lock = threading.Lock()
        # Drain stderr in a background thread so it doesn't fill up.
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # Wait for the "[daemon] ready" line so info+forward are valid.
        self._wait_ready()

    def _drain_stderr(self):
        for line in iter(self.proc.stderr.readline, b""):
            sys.stderr.write(f"[daemon] {line.decode('utf-8', 'replace')}")
            sys.stderr.flush()

    def _wait_ready(self, timeout: float = 60.0):
        # Daemon prints to stderr; we wait until it has loaded the model.
        # Simple heuristic: send INFO and wait for response.
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                _, _ = self.call(CMD_INFO, 0, b"")
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"daemon exited rc={self.proc.returncode}")
                time.sleep(0.5)
        raise TimeoutError("daemon never became ready")

    def call(self, cmd: int, n_tokens: int, payload: bytes, start_pos: int = 0, flags: int = 0):
        with self.lock:
            hdr = struct.pack("<IIIII", cmd, n_tokens, start_pos, flags, len(payload))
            self.proc.stdin.write(hdr + payload)
            self.proc.stdin.flush()
            head = self.proc.stdout.read(8)
            if len(head) != 8:
                raise RuntimeError(f"short read from daemon (got {len(head)} bytes)")
            status, plen = struct.unpack("<II", head)
            data = self.proc.stdout.read(plen) if plen else b""
            return status, data

    def info(self):
        s, p = self.call(CMD_INFO, 0, b"")
        if s != 0 or len(p) < 24:
            raise RuntimeError(f"info call failed status={s} len={len(p)}")
        layer_start, layer_end, n_embd, has_embd, has_lm, n_vocab = struct.unpack("<6i", p[:24])
        return dict(layer_start=layer_start, layer_end=layer_end, n_embd=n_embd,
                    has_token_embd=bool(has_embd), has_lm_head=bool(has_lm), n_vocab=n_vocab)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class WorkerServicer(pb_grpc.NakshatraServicer):
    def __init__(self, daemon: DaemonClient, mode: str, layer_start: int, layer_end: int, model_id: str):
        self.daemon = daemon
        self.mode = mode
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.model_id = model_id
        self.daemon_info = daemon.info()
        print(f"[worker] daemon info: {self.daemon_info}", flush=True)
        self.n_embd = self.daemon_info["n_embd"]

    def Info(self, request, context):
        return pb.InfoResponse(
            protocol_version="0.1.0",
            backend="llamacpp-cpu-patched",
            model_id=self.model_id,
            model_content_hash=b"\x00" * 32,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            hidden_size=self.n_embd,
            wire_dtype="f32",
            kv_cache_tokens_free=256,
            has_token_embd=(self.mode == "first"),
            has_lm_head=(self.mode == "last"),
        )

    def Forward(self, request, context):
        n = request.n_tokens
        flags = 0x1 if request.keep_kv else 0x0
        start_pos = int(request.start_pos)
        if request.has_token_ids:
            if len(request.hidden_in) != n * 4:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"hidden_in size mismatch for token_ids mode")
                return pb.ForwardResponse()
            status, resp = self.daemon.call(CMD_TOKEN_DECODE, n, request.hidden_in,
                                             start_pos=start_pos, flags=flags)
        else:
            expected = n * self.n_embd * 4
            if len(request.hidden_in) != expected:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"hidden_in size mismatch: got {len(request.hidden_in)}, expected {expected}")
                return pb.ForwardResponse()
            status, resp = self.daemon.call(CMD_EMBD_DECODE, n, request.hidden_in,
                                             start_pos=start_pos, flags=flags)

        if status != 0 or len(resp) < 4:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"daemon decode failed status={status} resp_len={len(resp)}")
            return pb.ForwardResponse()

        # Strip the 4-byte rtype prefix; payload is hidden_state OR int32 token id.
        rtype = struct.unpack("<I", resp[:4])[0]
        payload = resp[4:]
        # Caller (chain client) knows what mode this worker is in via Info; the
        # bytes returned are either hidden state or a single int32 token id.
        return pb.ForwardResponse(hidden_out=payload)

    def Inference(self, request_iterator, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Inference streaming will arrive after multi-token testing on Forward")
        return iter([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5500)
    ap.add_argument("--sub-gguf", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["first", "middle", "last"], required=True)
    ap.add_argument("--layer-start", type=int, required=True)
    ap.add_argument("--layer-end", type=int, required=True)
    ap.add_argument("--model-id", type=str, default="nakshatra-v0.1")
    ap.add_argument("--daemon-bin", type=str, default="/home/prithvi/llama.cpp/build/bin/llama-nakshatra-worker")
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--n-threads", type=int, default=0,
                    help="threads for llama_decode; 0 = let llama.cpp pick a default")
    args = ap.parse_args()

    print(f"[worker] spawning daemon: {args.daemon_bin} {args.sub_gguf} {args.mode} {args.n_ctx} threads={args.n_threads}", flush=True)
    daemon = DaemonClient(args.daemon_bin, args.sub_gguf, args.mode, args.n_ctx, args.n_threads)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = WorkerServicer(daemon, args.mode, args.layer_start, args.layer_end, args.model_id)
    pb_grpc.add_NakshatraServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{args.port}")
    print(f"[worker] M5 listening on :{args.port}  mode={args.mode}  layers=[{args.layer_start},{args.layer_end})  model={args.model_id}", flush=True)
    server.start()
    try:
        server.wait_for_termination()
    finally:
        daemon.close()


if __name__ == "__main__":
    main()
