#!/usr/bin/env python3
"""Verify the two-file join bootstrap from an invite line, then run it without a shell."""
from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


def parse_invite_line(line: str) -> tuple[str, str, str, str, str]:
    """Return installer URL/hash, releasekit URL/hash, and invite code.

    Only the exact one-line grammar emitted by ``nak invite`` is accepted.  In
    particular, no downloaded bytes or user-provided shell syntax is executed.
    """
    try:
        words = shlex.split(line, posix=True)
    except ValueError as exc:
        raise ValueError(f"malformed invite line: {exc}") from exc
    if len(words) != 26 or words[:9] != [
        "mkdir", "-p", "~/nak-join", "&&", "cd", "~/nak-join", "&&", "curl", "-fsSO"
    ] or words[10] != "-O" or words[12:15] != [
        "&&", "printf", "%s  install.py\\n%s  releasekit.py\\n"
    ] or words[17:25] != [
        "|", "sha256sum", "-c", "--quiet", "&&", "python3", "install.py", "join"
    ]:
        raise ValueError("invite line is not the signed-installer form emitted by `nak invite`")
    install_url, kit_url, install_sha, kit_sha, invite = words[9], words[11], words[15], words[16], words[25]
    if not re.fullmatch(r"[0-9a-f]{64}", install_sha) or not re.fullmatch(r"[0-9a-f]{64}", kit_sha):
        raise ValueError("invite line has an invalid installer fingerprint")
    if not re.fullmatch(r"nki1\.[A-Za-z0-9_-]+", invite) or len(invite) > 8192:
        raise ValueError("invite line has an invalid Nakshatra invite")
    parsed = [urllib.parse.urlsplit(url) for url in (install_url, kit_url)]
    if any(p.scheme not in {"http", "https"} or not p.hostname or p.username or p.password or p.query or p.fragment
           for p in parsed):
        raise ValueError("installer URLs must be plain http(s) URLs without credentials, query, or fragment")
    if parsed[0].path.rsplit("/", 1)[-1] != "install.py" or parsed[1].path.rsplit("/", 1)[-1] != "releasekit.py":
        raise ValueError("invite line does not name the two expected installer files")
    if install_url.removesuffix("install.py") != kit_url.removesuffix("releasekit.py"):
        raise ValueError("installer files do not come from the same release directory")
    return install_url, install_sha, kit_url, kit_sha, invite


def _download_verified(url: str, expected: str, destination: Path) -> None:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected:
        destination.unlink(missing_ok=True)
        raise ValueError(f"{destination.name} does not match the fingerprint in the invite line")


def main() -> int:
    line = sys.stdin.readline(16385).rstrip("\n")
    if not line or sys.stdin.read(1):
        print("error: expected exactly one invite line on stdin", file=sys.stderr)
        return 2
    try:
        install_url, install_sha, kit_url, kit_sha, invite = parse_invite_line(line)
        with tempfile.TemporaryDirectory(prefix="nak-join-") as work:
            directory = Path(work)
            _download_verified(install_url, install_sha, directory / "install.py")
            _download_verified(kit_url, kit_sha, directory / "releasekit.py")
            env = dict(os.environ)
            prefix = env.get("NAK_NODE_PREFIX", str(Path.home() / ".nakshatra-node"))
            return subprocess.run(
                [sys.executable, str(directory / "install.py"), "--prefix", prefix, "join", invite],
                cwd=directory,
                env=env,
                check=False,
            ).returncode
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
