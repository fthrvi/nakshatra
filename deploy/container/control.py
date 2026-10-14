#!/usr/bin/env python3
"""Small client for the in-container supervisor control socket."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time


def request(payload: dict) -> dict:
    path = os.environ.get("NAK_SUPERVISOR_SOCKET", "/tmp/nakshatra-supervisor.sock")
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(10)
        client.connect(path)
        client.sendall(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "update"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = request({"action": args.action})
        requested_update = args.action == "update"
        if requested_update and result.get("ok"):
            while True:
                time.sleep(0.2)
                result = request({"action": "status"})
                if not result.get("updating"):
                    break
        if not args.quiet:
            print(json.dumps(result, indent=1, sort_keys=True))
        return 0 if result.get("ok") and (not requested_update or result.get("update_ok") is True) else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if not args.quiet:
            print(f"supervisor unavailable: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
