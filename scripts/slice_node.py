"""slice_node.py — run a mesh slice node: serve this node's GGUF slices over HTTP
AND heartbeat-publish them to the shared TTL directory so peers can discover and
pull them. Stop the process and the TTL evicts the node automatically (no manual
deregister). The live arm of the placement directory.

Usage:
  python slice_node.py --slices ~/.nakshatra/slices --node 10.42.0.3:8077 \
      --dir /mnt/mesh/slice-directory --port 8077 --interval 60
"""
from __future__ import annotations
import argparse
import os
import threading
import time

import slice_server
import slice_directory


def run(slices_dir: str, node: str, dir_path: str, port: int = 8077,
        interval: float = 60.0, host: str = "0.0.0.0",
        log=print):
    """Start the HTTP slice server + a heartbeat that republishes this node's
    slices to `dir_path` every `interval` s. Returns the httpd (caller may
    .shutdown()). Both run in daemon threads."""
    httpd = slice_server.serve(slices_dir, host=host, port=port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log(f"[slice-node] serving {slices_dir} on {host}:{port} as {node}")

    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            try:
                files = [n for n in os.listdir(slices_dir) if n.endswith(".gguf")]
                slice_directory.publish_to_dir(dir_path, node, files)
            except Exception as e:
                log(f"[slice-node] heartbeat failed: {e}")
            stop.wait(interval)
    threading.Thread(target=heartbeat, daemon=True).start()
    httpd._heartbeat_stop = stop   # let a caller stop the heartbeat too
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=os.path.expanduser("~/.nakshatra/slices"))
    ap.add_argument("--node", required=True, help="advertised host:port of this server")
    ap.add_argument("--dir", required=True, help="shared directory path (mesh fs)")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    run(args.slices, args.node, args.dir, port=args.port,
        interval=args.interval, host=args.host)
    print("[slice-node] running; Ctrl-C to stop (TTL evicts on exit)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
