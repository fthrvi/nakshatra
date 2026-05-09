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
import collections
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from concurrent import futures
from http.server import HTTPServer, BaseHTTPRequestHandler
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
        # Stderr buffer (Phase 3.6): keeps last N lines so we can verify what
        # the daemon ACTUALLY did (e.g., GPU offload count) vs. what we asked.
        self.stderr_lines = collections.deque(maxlen=500)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # Wait for the "[daemon] ready" line so info+forward are valid.
        self._wait_ready()

    def _drain_stderr(self):
        for line in iter(self.proc.stderr.readline, b""):
            text = line.decode("utf-8", "replace")
            self.stderr_lines.append(text)
            sys.stderr.write(f"[daemon] {text}")
            sys.stderr.flush()

    def gpu_offload_status(self) -> dict:
        """Parse stderr buffer for what the daemon ACTUALLY did with the GPU.

        Returns:
          {
            "n_offloaded": int,    # N from "offloaded N/M layers to GPU"
            "total_layers": int,   # M from same
            "uses_gpu": bool,      # n_offloaded > 0
            "backend_hints": [],   # any backend names spotted in stderr
            "log_lines": [],       # relevant excerpt for diagnostics
          }
        """
        n_offloaded = 0
        total_layers = 0
        uses_gpu = False
        backend_hints = set()
        relevant = []

        # Pattern: "load_tensors: offloaded N/M layers to GPU"
        offload_re = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers", re.IGNORECASE)
        # Backend signature lines llama.cpp prints (Metal/ROCm/CUDA buffers)
        backend_signals = (
            ("metal",   re.compile(r"\bMetal\b|ggml-metal|MPS", re.IGNORECASE)),
            ("rocm",    re.compile(r"\bROCm\b|HIP\b|hipMalloc|amdhip", re.IGNORECASE)),
            ("cuda",    re.compile(r"\bCUDA\b|cudaMalloc|cuBLAS",     re.IGNORECASE)),
            ("vulkan",  re.compile(r"\bVulkan\b|vk_buffer|MoltenVK",  re.IGNORECASE)),
            ("cpu",     re.compile(r"\bCPU\b\s+(KV|compute|output)\s+buffer", re.IGNORECASE)),
        )

        for line in self.stderr_lines:
            m = offload_re.search(line)
            if m:
                n_offloaded = int(m.group(1))
                total_layers = int(m.group(2))
                uses_gpu = n_offloaded > 0
                relevant.append(line.rstrip())
            for name, rx in backend_signals:
                if rx.search(line):
                    backend_hints.add(name)
                    if name != "cpu" or "compute buffer" in line.lower():
                        relevant.append(line.rstrip())

        return {
            "n_offloaded": n_offloaded,
            "total_layers": total_layers,
            "uses_gpu": uses_gpu,
            "backend_hints": sorted(backend_hints),
            "log_lines": relevant[:20],
        }

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


_FILE_SERVER_DIR = ""


class FileServerHandler(BaseHTTPRequestHandler):
    """Phase-4 file server. Serves files from this worker's sub-GGUF directory
    over HTTP with byte-range support, so other peers can bootstrap their
    layer range from us instead of needing manual scp.

    Endpoints:
        GET /file/<basename>     — sends the file (full or Range-restricted)
        GET /                    — info ping ("worker file server")
    """
    server_version = "NakshatraFileServer/0.1"

    def log_message(self, format, *args):
        # Quiet by default; uncomment for debugging
        # sys.stderr.write(f"[fileserver] {self.address_string()} {format % args}\n")
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/health":
            body = b'{"server":"nakshatra-file","status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if not self.path.startswith("/file/"):
            self.send_error(404, "not found")
            return

        # Sanitize: only allow simple basenames, no traversal
        filename = self.path[len("/file/"):]
        if "/" in filename or ".." in filename or filename.startswith("."):
            self.send_error(400, "bad filename")
            return
        path = os.path.join(_FILE_SERVER_DIR, filename)
        if not os.path.isfile(path):
            self.send_error(404, "file not found")
            return

        size = os.path.getsize(path)
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            try:
                spec = range_header[6:].strip()
                start_str, _, end_str = spec.partition("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else size - 1
                if start < 0 or end >= size or start > end:
                    self.send_error(416, "bad range")
                    return
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except Exception as e:
                self.send_error(400, f"range error: {e}")
            return

        # Full-file send
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile, 8 * 1024 * 1024)
        except (BrokenPipeError, ConnectionResetError):
            pass


class _ThreadingHTTPServer(HTTPServer):
    """Allow multiple concurrent fetches (without this, parallel byte-range
    requests would serialize through one handler thread)."""
    daemon_threads = True
    def process_request(self, request, client_address):
        threading.Thread(
            target=self._handle_request_thread, args=(request, client_address),
            daemon=True,
        ).start()
    def _handle_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            pass
        finally:
            try: self.shutdown_request(request)
            except Exception: pass


def start_file_server(serving_dir: str, port: int):
    """Start the Phase-4 file server in a background thread."""
    global _FILE_SERVER_DIR
    _FILE_SERVER_DIR = str(Path(serving_dir).resolve())
    server = _ThreadingHTTPServer(("0.0.0.0", port), FileServerHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[fileserver] listening on :{port}, serving from {_FILE_SERVER_DIR}", flush=True)


def fetch_sub_gguf_from_peer(pillar_url: str, model_id: str,
                              layer_start: int, layer_end: int,
                              dest_path: str,
                              own_node_id: str = "") -> str:
    """Query Sthambha for a peer with the requested file and download it.

    Returns the local path on success; raises RuntimeError on failure.
    Verifies SHA-256 against the pillar's recorded hash if present.

    Skips candidates whose node_id matches `own_node_id` (don't fetch
    from yourself). Retries the next candidate on connection failure
    so a stale-but-still-listed peer doesn't block the bootstrap.
    """
    # 1. Ask the pillar for the file index
    files_url = f"{pillar_url.rstrip('/')}/files?model={model_id}"
    try:
        with urlrequest.urlopen(files_url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(f"could not query pillar at {files_url}: {e}")

    # 2. Find candidate online peers holding (model_id, layer_start, layer_end)
    candidates = [
        f for f in data.get("files", [])
        if f.get("model_id") == model_id
        and int(f.get("layer_start", -1)) == layer_start
        and int(f.get("layer_end", -1)) == layer_end
        and f.get("is_online")
        and f.get("node_id") != own_node_id  # don't fetch from yourself
    ]
    if not candidates:
        raise RuntimeError(
            f"no online peer holds {model_id} layers [{layer_start},{layer_end})"
            f" — file not available on the network"
        )

    # 3. Try each candidate in order; on connection failure, fall through
    last_error = None
    for chosen in candidates:
        addr = chosen.get("address", "")
        if ":" not in addr:
            last_error = f"malformed address {addr!r}"
            continue
        host, _, port_str = addr.rpartition(":")
        grpc_port = int(port_str)
        file_port = grpc_port + 1000  # convention: file server on grpc_port + 1000
        expected_sha = chosen.get("model_sha256", "")
        src_basename = Path(chosen.get("file_path", "")).name or f"w-{layer_start}-{layer_end}.gguf"

        fetch_url = f"http://{host}:{file_port}/file/{src_basename}"
        print(f"[fetch] trying {chosen.get('node_id')} → {fetch_url}", flush=True)

        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path + ".tmp"
        h = hashlib.sha256()
        bytes_received = 0
        t0 = time.time()
        try:
            with urlrequest.urlopen(fetch_url, timeout=60) as resp:
                content_length = int(resp.headers.get("Content-Length", "0"))
                with open(tmp_path, "wb") as out:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        h.update(chunk)
                        bytes_received += len(chunk)
                        if bytes_received and bytes_received % (200 * 1024 * 1024) < 8 * 1024 * 1024:
                            pct = (bytes_received / content_length * 100) if content_length else 0
                            elapsed = time.time() - t0
                            rate_mbps = (bytes_received / 1e6) / max(elapsed, 0.001)
                            print(f"[fetch]   {bytes_received/1e9:.1f}/{content_length/1e9:.1f} GB ({pct:.0f}%) at {rate_mbps:.1f} MB/s",
                                  flush=True)
        except Exception as e:
            print(f"[fetch] {chosen.get('node_id')} failed ({e}); trying next candidate", flush=True)
            try: os.unlink(tmp_path)
            except Exception: pass
            last_error = str(e)
            continue

        # Verify SHA-256 if the pillar gave us one
        actual_sha = h.hexdigest()
        if expected_sha and actual_sha != expected_sha:
            print(f"[fetch] {chosen.get('node_id')} sha mismatch (got {actual_sha[:12]}, expected {expected_sha[:12]}); trying next", flush=True)
            try: os.unlink(tmp_path)
            except Exception: pass
            last_error = "sha mismatch"
            continue

        # Atomic move into place
        os.rename(tmp_path, dest_path)
        elapsed = time.time() - t0
        print(f"[fetch] saved {dest_path} ({bytes_received:,} bytes, sha={actual_sha[:12]}..., "
              f"{elapsed:.1f}s, {(bytes_received/1e6)/max(elapsed,0.001):.1f} MB/s) from {chosen.get('node_id')}", flush=True)
        return dest_path

    raise RuntimeError(f"all {len(candidates)} candidate peers failed; last error: {last_error}")


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
    ap.add_argument("--file-server-port", type=int, default=0,
                    help="Phase 4: HTTP file-server port for peer-to-peer fetch. "
                         "Default = grpc port + 1000 (e.g. 5530 → 6530).")
    ap.add_argument("--no-file-server", action="store_true",
                    help="Disable the Phase-4 HTTP file server (this worker won't "
                         "let peers fetch its sub-GGUF).")
    ap.add_argument("--auto-fetch", action="store_true",
                    help="If --sub-gguf doesn't exist, query --pillar-url for who "
                         "has the file and download it before spawning the daemon.")
    args = ap.parse_args()

    # Phase 4: if sub-GGUF is missing AND we have a pillar to ask, fetch it
    # before doing anything else. This is what lets a fresh machine bootstrap
    # without manual scp.
    if not os.path.exists(args.sub_gguf):
        if args.auto_fetch and args.pillar_url:
            print(f"[worker] sub-gguf missing at {args.sub_gguf}; auto-fetching from peer", flush=True)
            own_node_id = args.node_id or f"{socket.gethostname()}-{args.port}"
            try:
                fetch_sub_gguf_from_peer(
                    args.pillar_url, args.model_id,
                    args.layer_start, args.layer_end,
                    args.sub_gguf,
                    own_node_id=own_node_id,
                )
            except Exception as e:
                sys.exit(f"[worker] auto-fetch failed: {e}")
        else:
            sys.exit(f"[worker] sub-gguf does not exist: {args.sub_gguf} "
                     f"(use --auto-fetch with --pillar-url to bootstrap from a peer)")

    print(f"[worker] spawning daemon: {args.daemon_bin} {args.sub_gguf} {args.mode} {args.n_ctx} threads={args.n_threads} gpu_layers={args.n_gpu_layers}", flush=True)
    daemon = DaemonClient(args.daemon_bin, args.sub_gguf, args.mode, args.n_ctx, args.n_threads, args.n_gpu_layers)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = WorkerServicer(daemon, args.mode, args.layer_start, args.layer_end, args.model_id)
    pb_grpc.add_NakshatraServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{args.port}")
    print(f"[worker] M5 listening on :{args.port}  mode={args.mode}  layers=[{args.layer_start},{args.layer_end})  model={args.model_id}", flush=True)
    server.start()

    # Phase 4: file server lets peers fetch this worker's sub-GGUF over HTTP
    # byte-range. New peers joining the cluster don't need their files
    # pre-shipped — they query the pillar's file index, find a peer that has
    # the file, and download it. Convention: file-server port = grpc port + 1000.
    file_server_port = args.file_server_port or (args.port + 1000)
    if not args.no_file_server:
        try:
            start_file_server(serving_dir=str(Path(args.sub_gguf).parent),
                              port=file_server_port)
        except Exception as e:
            print(f"[fileserver] failed to start: {e} (peers won't be able to fetch from this worker)", flush=True)
            file_server_port = 0
    else:
        file_server_port = 0

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

        # Phase 3.6 — verify declared GPU backend against daemon reality.
        # The daemon's stderr tells us what actually happened on model load
        # (it prints "offloaded N/M layers to GPU"). If we declared a GPU
        # backend on the CLI but the daemon offloaded 0 layers, the binary
        # was almost certainly built without that backend. We DOWNGRADE the
        # declaration to "cpu" before posting to the pillar — better to be
        # truthful than to lie about capability.
        offload_status = daemon.gpu_offload_status()
        actual_backend = args.gpu_backend
        declared_gpu = (args.gpu_vram_gb > 0 and args.gpu_backend != "cpu")
        if declared_gpu and not offload_status["uses_gpu"]:
            print(f"[worker] WARNING: declared --gpu-backend={args.gpu_backend} "
                  f"but daemon offloaded {offload_status['n_offloaded']}/"
                  f"{offload_status['total_layers']} layers — daemon binary likely "
                  f"lacks {args.gpu_backend} support; downgrading registration to cpu",
                  flush=True)
            actual_backend = "cpu"
        elif declared_gpu:
            print(f"[worker] verified: daemon offloaded "
                  f"{offload_status['n_offloaded']}/{offload_status['total_layers']} "
                  f"layers via {args.gpu_backend} (backends seen in log: "
                  f"{offload_status['backend_hints']})", flush=True)

        gpus = []
        if args.gpu_vram_gb > 0:
            gpus.append({
                "vendor": args.gpu_vendor or "unknown",
                "model": args.gpu_model or "unknown",
                "vram_total_gb": args.gpu_vram_gb,
                "backend": actual_backend,
                "actual_layers_offloaded": offload_status["n_offloaded"],
                "total_layers_loaded": offload_status["total_layers"],
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
