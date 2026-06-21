"""slice_server.py — read-only HTTP server that lets mesh peers pull the GGUF
slices this node already holds. The transport half of the slice-acquisition
layer (slice_fetch.py): a node that lacks a slice pulls it from a peer running
this server over the WireGuard mesh — the cheap, local-network source that beats
the WAN/HF origin (especially on WiFi).

Endpoints:
  GET /slices            → JSON {"slices": [{"name","bytes"}...]}  (the directory:
                           lets a peer/roster discover what this node holds)
  GET /slice/<filename>  → the raw GGUF bytes (streamed)

Read-only and path-traversal-safe (only basenames under the served dir, only
*.gguf). Bind to the mesh interface; slices are public, hash-verified on arrival.
"""
from __future__ import annotations
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

_CHUNK = 8 << 20


def _safe_name(name: str) -> str | None:
    """Accept only a bare *.gguf basename — no path separators, no traversal."""
    name = unquote(name)
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        return None
    if os.path.basename(name) != name or not name.endswith(".gguf"):
        return None
    return name


def make_handler(slices_dir: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass  # quiet by default

        def do_GET(self):
            if self.path == "/slices":
                return self._list()
            if self.path.startswith("/slice/"):
                return self._send(self.path[len("/slice/"):])
            self.send_error(404, "not found")

        def _list(self):
            out = []
            try:
                for n in os.listdir(slices_dir):
                    if n.endswith(".gguf"):
                        out.append({"name": n,
                                    "bytes": os.path.getsize(os.path.join(slices_dir, n))})
            except OSError:
                pass
            body = json.dumps({"slices": out}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send(self, raw_name: str):
            name = _safe_name(raw_name)
            if name is None:
                return self.send_error(400, "bad name")
            path = os.path.join(slices_dir, name)
            if not os.path.isfile(path):
                return self.send_error(404, "no such slice")
            try:
                size = os.path.getsize(path)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                with open(path, "rb") as f:
                    while True:
                        b = f.read(_CHUNK)
                        if not b:
                            break
                        self.wfile.write(b)
            except (OSError, BrokenPipeError):
                pass
    return Handler


def serve(slices_dir: str, host: str = "0.0.0.0", port: int = 8077):
    httpd = ThreadingHTTPServer((host, port), make_handler(slices_dir))
    return httpd  # caller runs .serve_forever() (or in a thread)


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.nakshatra/slices")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8077
    print(f"[slice-server] serving {d} on :{port}", flush=True)
    serve(d, port=port).serve_forever()
