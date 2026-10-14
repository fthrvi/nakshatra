#!/usr/bin/env python3
"""The narrow systemctl surface used by the signed release inside the container."""
from __future__ import annotations

import json
import os
import socket
import sys


def _request(action: str, **fields) -> dict:
    path = os.environ.get("NAK_SUPERVISOR_SOCKET", "/tmp/nakshatra-supervisor.sock")
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(30)
        client.connect(path)
        client.sendall(json.dumps({"action": action, **fields}, separators=(",", ":")).encode() + b"\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


def _services(args: list[str]) -> list[str]:
    return [arg.removesuffix(".service") for arg in args if not arg.startswith("-")]


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    args = [arg for arg in args if arg != "--user"]
    quiet = "--quiet" in args
    args = [arg for arg in args if arg != "--quiet"]
    if not args:
        return 1
    command, rest = args[0], args[1:]
    if command == "is-system-running":
        if not quiet:
            print("running")
        return 0
    try:
        if command == "daemon-reload":
            result = _request("reload")
        elif command in {"enable", "disable", "restart"}:
            names = _services(rest)
            result = _request(command, services=names, now="--now" in rest)
        elif command in {"is-active", "is-enabled"}:
            names = _services(rest)
            if len(names) != 1:
                return 1
            result = _request(command, service=names[0])
            if not quiet and result.get("stdout"):
                print(result["stdout"])
        elif command == "show":
            names = _services(rest)
            if len(names) != 1:
                return 1
            result = _request("show", service=names[0])
            if result.get("stdout"):
                print(result["stdout"])
        else:
            if not quiet:
                print(f"systemctl container shim does not implement {command!r}", file=sys.stderr)
            return 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if not quiet:
            print(f"systemctl container supervisor unavailable: {exc}", file=sys.stderr)
        return 1
    if not result.get("ok") and not quiet and result.get("error"):
        print(result["error"], file=sys.stderr)
    return 0 if result.get("ok") else int(result.get("returncode", 1))


if __name__ == "__main__":
    raise SystemExit(main())
