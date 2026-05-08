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
import hashlib
import json
import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from urllib import request as urlrequest, error as urlerror

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc


CMD_TOKEN_DECODE = 1
CMD_EMBD_DECODE  = 2
CMD_INFO         = 3


class DaemonClient:
    """Manages a long-lived llama-nakshatra-worker subprocess over stdin/stdout."""

    def __init__(self, daemon_bin: str, sub_gguf: str, mode: str, n_ctx: int, n_threads: int = 0, n_gpu_layers: int = 0):
        self.proc = subprocess.Popen(
            [daemon_bin, sub_gguf, mode, str(n_ctx), str(n_threads), str(n_gpu_layers)],
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


def register_with_pillar(pillar_url: str, payload: dict, log_prefix: str = "[worker]"):
    """POST to <pillar_url>/peer. Best-effort: log on failure, never raise."""
    try:
        req = urlrequest.Request(
            f"{pillar_url.rstrip('/')}/peer",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=5) as resp:
            body = resp.read().decode()
            print(f"{log_prefix} registered with pillar: {body}", flush=True)
            return True
    except (urlerror.URLError, OSError, TimeoutError) as e:
        print(f"{log_prefix} pillar registration failed ({pillar_url}): {e}", flush=True)
        return False


def heartbeat_loop(pillar_url: str, payload: dict, interval: float = 30.0,
                   stop_event: threading.Event = None):
    """Re-register with pillar every `interval` seconds. Run as daemon thread."""
    stop_event = stop_event or threading.Event()
    while not stop_event.is_set():
        if stop_event.wait(timeout=interval):
            break
        register_with_pillar(pillar_url, payload, log_prefix="[heartbeat]")


def detect_ram_gb() -> float:
    """Best-effort total RAM detection (stdlib only). Returns 0.0 on failure."""
    sys_name = platform.system()
    try:
        if sys_name == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        elif sys_name == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2).decode().strip()
            return int(out) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def detect_disk_avail_gb(path: str = "/") -> float:
    """Free disk space at `path` in GB."""
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return 0.0


def detect_cpu_model() -> str:
    sys_name = platform.system()
    try:
        if sys_name == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys_name == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=2)
            return out.decode().strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def sha256_of_file(path: str) -> str:
    """Stream SHA-256 of a file. ~5s for 8 GB on SSD, fine for one-time startup cost."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    ap.add_argument("--n-gpu-layers", type=int, default=0,
                    help="layers to offload to GPU; 0 = CPU only, 99 = all on GPU")
    ap.add_argument("--pillar-url", type=str, default="",
                    help="Sthambha pillar URL (e.g. http://umbrel:7777). If set, "
                         "worker registers self + sends heartbeat every 30s.")
    ap.add_argument("--public-address", type=str, default="",
                    help="Address other peers should use to reach this worker "
                         "(default: hostname:port). Override for special routing.")
    ap.add_argument("--node-id", type=str, default="",
                    help="Stable node identifier for the registry. Defaults to "
                         "hostname-port (e.g. mac3-2-5530).")
    # Phase 3.5 — hardware declarations. Operator-declared (network trusts you).
    ap.add_argument("--gpu-vendor", type=str, default="",
                    help="GPU vendor string e.g. AMD / NVIDIA / Apple / Intel.")
    ap.add_argument("--gpu-model", type=str, default="",
                    help="GPU model string e.g. 'Radeon Pro 5700 XT'.")
    ap.add_argument("--gpu-vram-gb", type=float, default=0.0,
                    help="Total GPU VRAM in GB. 0 = no GPU declared.")
    ap.add_argument("--gpu-backend", type=str, default="cpu",
                    help="Inference backend: rocm/cuda/metal/vulkan/cpu.")
    ap.add_argument("--vram-offered-gb", type=float, default=-1.0,
                    help="VRAM you're offering to the network. Default = gpu-vram-gb.")
    ap.add_argument("--ram-offered-gb", type=float, default=-1.0,
                    help="RAM offered to the network. Default = half of system RAM.")
    ap.add_argument("--cpu-threads-offered", type=int, default=0,
                    help="CPU threads offered. Default = --n-threads value.")
    ap.add_argument("--disk-for-cache-gb", type=float, default=0.0,
                    help="Disk space available for layer cache.")
    ap.add_argument("--skip-sha256", action="store_true",
                    help="Skip SHA-256 of sub-GGUF (for fast restarts; cached_files lacks hash).")
    args = ap.parse_args()

    print(f"[worker] spawning daemon: {args.daemon_bin} {args.sub_gguf} {args.mode} {args.n_ctx} threads={args.n_threads} gpu_layers={args.n_gpu_layers}", flush=True)
    daemon = DaemonClient(args.daemon_bin, args.sub_gguf, args.mode, args.n_ctx, args.n_threads, args.n_gpu_layers)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = WorkerServicer(daemon, args.mode, args.layer_start, args.layer_end, args.model_id)
    pb_grpc.add_NakshatraServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{args.port}")
    print(f"[worker] M5 listening on :{args.port}  mode={args.mode}  layers=[{args.layer_start},{args.layer_end})  model={args.model_id}", flush=True)
    server.start()

    # Pillar registration (Phase 3b). Best-effort — worker still serves
    # requests if the pillar is unreachable (back-compat with static YAML
    # cluster configs). Heartbeat thread keeps the registry view fresh.
    stop_event = threading.Event()
    heartbeat_thread = None
    if args.pillar_url:
        public_addr = args.public_address or f"{socket.gethostname()}:{args.port}"
        node_id = args.node_id or f"{socket.gethostname()}-{args.port}"

        # Phase 3.5: compute SHA-256 of the sub-GGUF (one-time at startup)
        sub_gguf_sha256 = ""
        sub_gguf_size = 0
        try:
            sub_gguf_size = os.path.getsize(args.sub_gguf)
            if not args.skip_sha256:
                t0 = time.time()
                print(f"[worker] computing sha256 of {args.sub_gguf} ({sub_gguf_size/1e9:.1f} GB)…", flush=True)
                sub_gguf_sha256 = sha256_of_file(args.sub_gguf)
                print(f"[worker] sha256={sub_gguf_sha256[:16]}... ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[worker] sub-GGUF inspection failed (continuing without hash): {e}", flush=True)

        # Hardware auto-detection (best-effort) + operator overrides
        ram_total = detect_ram_gb()
        disk_avail = detect_disk_avail_gb(os.path.dirname(args.sub_gguf) or "/")
        cpu_model = detect_cpu_model()
        cpu_threads = os.cpu_count() or 0

        gpus = []
        if args.gpu_vram_gb > 0:
            gpus.append({
                "vendor": args.gpu_vendor or "unknown",
                "model": args.gpu_model or "unknown",
                "vram_total_gb": args.gpu_vram_gb,
                "backend": args.gpu_backend,
            })
        hardware = {
            "platform": platform.system().lower(),
            "arch": platform.machine(),
            "cpu_model": cpu_model,
            "cpu_threads": cpu_threads,
            "ram_total_gb": ram_total,
            "disk_avail_gb": disk_avail,
            "gpus": gpus,
        }

        # Budget: operator declares; sensible defaults
        vram_offered = args.vram_offered_gb if args.vram_offered_gb >= 0 else args.gpu_vram_gb
        ram_offered = args.ram_offered_gb if args.ram_offered_gb >= 0 else max(ram_total / 2, 0.0)
        cpu_offered = args.cpu_threads_offered or args.n_threads or cpu_threads
        budget = {
            "vram_offered_gb": vram_offered,
            "ram_offered_gb": ram_offered,
            "cpu_threads_offered": cpu_offered,
            "disk_for_cache_gb": args.disk_for_cache_gb,
        }

        cached_files = []
        if sub_gguf_size > 0:
            cached_files.append({
                "model_id": args.model_id,
                "model_sha256": sub_gguf_sha256,
                "layer_start": args.layer_start,
                "layer_end": args.layer_end,
                "size_bytes": sub_gguf_size,
                "file_path": str(Path(args.sub_gguf).resolve()),
            })

        register_payload = {
            "node_id": node_id,
            "node_type": "compute",
            "address": public_addr,
            "layer_offerings": [{
                "model_id": args.model_id,
                "model_sha256": sub_gguf_sha256,
                "layer_start": args.layer_start,
                "layer_end": args.layer_end,
            }],
            "hardware": hardware,
            "budget": budget,
            "cached_files": cached_files,
        }
        register_with_pillar(args.pillar_url, register_payload)
        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            args=(args.pillar_url, register_payload, 30.0, stop_event),
            daemon=True,
        )
        heartbeat_thread.start()
        print(f"[worker] heartbeat → {args.pillar_url} every 30s as {node_id} @ {public_addr}", flush=True)

    try:
        server.wait_for_termination()
    finally:
        stop_event.set()
        daemon.close()


if __name__ == "__main__":
    main()
