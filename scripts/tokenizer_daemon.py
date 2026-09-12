"""tokenizer_daemon.py — a resident vocab-only tokenizer, so client.py's per-request subprocess
spawn doesn't also pay a ~0.3-0.5s Llama(vocab_only=True) reload on every single chat turn.

Measured 2026-09-12 (nakshatra-unconscious.service, DeepSeek-R1-Distill-Llama-8B):
    import llama_cpp:        ~0.12s
    Llama(vocab_only) load:  ~0.21s
paid on EVERY request by client.py's tokenize_local(), on top of the subprocess spawn itself.
This is the smaller, lower-risk half of that finding — it caches ONLY the tokenizer, leaving
client.py's decode loop (recovery, speculative decode, streaming, push mode — the genuinely
fragile, heavily-branched ~800 lines this session found 3 separate bugs in) completely untouched.
The other half (a fully resident client holding warm gRPC channels too, avoiding the subprocess
spawn itself) is a bigger, separate redesign — see reference_nakshatra_client_resident_coordinator
in trisul memory for that design, deliberately NOT attempted in the same session as this.

Protocol: a Unix domain socket, one JSON object per line, request -> response, request:
    {"op": "tokenize", "text": "<utf-8 str>", "add_bos": true, "special": true}
        -> {"ids": [1, 2, 3, ...]}
    {"op": "detokenize", "ids": [1, 2, 3]}
        -> {"text": "<utf-8 str, invalid bytes replaced>"}
Any error -> {"error": "<message>"}. Never crashes the connection on a bad request.

Usage: tokenizer_daemon.py <model_path> --socket <path> [--idle-timeout-s N]
Exits after --idle-timeout-s (default 600) with no connections — scale-to-zero, matching every
other resident piece of this stack. The caller (nakshatra_serve.py) is responsible for spawning
it on demand and treating "socket doesn't exist / connect fails" as "not running yet, start it".
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
import threading
import time


class _IdleTracker:
    """Tracks the last time any connection was active. main() polls this to self-exit."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_active = time.monotonic()
        self._active_conns = 0

    def conn_opened(self):
        with self._lock:
            self._active_conns += 1
            self._last_active = time.monotonic()

    def conn_closed(self):
        with self._lock:
            self._active_conns -= 1
            self._last_active = time.monotonic()

    def idle_seconds(self) -> float:
        with self._lock:
            if self._active_conns > 0:
                return 0.0
            return time.monotonic() - self._last_active


def _make_handler(llama, idle: _IdleTracker):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            idle.conn_opened()
            try:
                for line in self.rfile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        req = json.loads(line)
                        resp = _handle_request(llama, req)
                    except Exception as e:
                        resp = {"error": f"{type(e).__name__}: {e}"}
                    self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
                    self.wfile.flush()
            finally:
                idle.conn_closed()

    return Handler


def _handle_request(llama, req: dict) -> dict:
    op = req.get("op")
    if op == "tokenize":
        text = req["text"]
        ids = llama.tokenize(text.encode("utf-8"),
                             add_bos=bool(req.get("add_bos", True)),
                             special=bool(req.get("special", True)))
        return {"ids": list(ids)}
    if op == "detokenize":
        ids = [int(i) for i in req["ids"]]
        text = llama.detokenize(ids).decode("utf-8", errors="replace")
        return {"text": text}
    if op == "ping":
        return {"ok": True}
    return {"error": f"unknown op {op!r}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str, help="full GGUF path (vocab_only load)")
    ap.add_argument("--socket", type=str, required=True, help="Unix socket path to listen on")
    ap.add_argument("--idle-timeout-s", type=float, default=600.0,
                    help="exit after this long with zero active connections (default 600)")
    args = ap.parse_args()

    from llama_cpp import Llama
    llama = Llama(model_path=args.model_path, vocab_only=True, verbose=False)

    # A stale socket file from a previous, uncleanly-killed run must not block bind().
    try:
        os.unlink(args.socket)
    except OSError:
        pass

    idle = _IdleTracker()
    server = socketserver.ThreadingUnixStreamServer(args.socket, _make_handler(llama, idle))
    server.daemon_threads = True
    os.chmod(args.socket, 0o600)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[tokenizer-daemon] ready: {args.model_path} socket={args.socket} "
          f"idle_timeout_s={args.idle_timeout_s}", flush=True)

    try:
        while True:
            time.sleep(5.0)
            if idle.idle_seconds() >= args.idle_timeout_s:
                print(f"[tokenizer-daemon] idle {args.idle_timeout_s}s — exiting", flush=True)
                break
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        try:
            os.unlink(args.socket)
        except OSError:
            pass


if __name__ == "__main__":
    main()
