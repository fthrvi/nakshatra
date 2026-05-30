#!/usr/bin/env python3
"""Cluster-wide daemon version-lock smoke (sprint Phase B).

SSHs each cluster machine, runs ``llama-nakshatra-worker-fabric
--version``, and asserts every binary reports the same
``NAKSHATRA_FABRIC_SHA``. Closes the long-standing daemon-skew
finding from the 2026-05-21 retro (May 8 Linux build vs May 13 Mac
build silently disagreeing on slice acceptance).

Run after ``experiments/v0.0/build-fabric.sh`` to verify the deploy:

    bash experiments/v0.0/build-fabric.sh
    python scripts/smoke_daemon_version.py

Exit 0 when all hosts report the same SHA; non-zero on any
disagreement, missing binary, or SSH failure.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Dict, Optional


DEFAULT_HOSTS = (
    "node-a",
    "node-e",
    "node-b",
    "node-c",
    "node-d",
)

# Match the worker_daemon.cpp --version output:
#   nakshatra-fabric-worker
#     sha        <hex>
#     built_on   <host>
#     built_at   <date> <time>
_SHA_LINE = re.compile(r"^\s*sha\s+(\S+)\s*$", re.MULTILINE)
_HOST_LINE = re.compile(r"^\s*built_on\s+(\S+)\s*$", re.MULTILINE)
_TIME_LINE = re.compile(r"^\s*built_at\s+(.+?)\s*$", re.MULTILINE)


def query(host: str) -> Optional[Dict[str, str]]:
    """SSH the host + run the daemon's --version. Returns a parsed
    dict, or None on any failure."""
    # PATH dance covers macOS non-interactive SSH stripping
    # /usr/local/bin. The daemon binary lives at the canonical
    # ~/llama.cpp/build/bin/ path on every cluster machine.
    cmd = [
        "ssh", "-o", "ConnectTimeout=8", host,
        'PATH="$PATH:/usr/local/bin" $HOME/llama.cpp/build/bin/'
        'llama-nakshatra-worker-fabric --version',
    ]
    try:
        out = subprocess.check_output(
            cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {host}: rc={e.returncode}\n    {e.output.strip()[:200]}",
              file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  ✗ {host}: ssh timeout", file=sys.stderr)
        return None

    sha_m = _SHA_LINE.search(out)
    host_m = _HOST_LINE.search(out)
    time_m = _TIME_LINE.search(out)
    if not (sha_m and host_m and time_m):
        print(f"  ✗ {host}: unparseable --version output:\n    {out.strip()}",
              file=sys.stderr)
        return None
    return {
        "sha": sha_m.group(1),
        "built_on": host_m.group(1),
        "built_at": time_m.group(1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hosts", nargs="*",
                    help="cluster hosts to query (default: full 5-machine set)")
    ap.add_argument("--expected-sha",
                    help="if set, assert every host's sha matches this")
    args = ap.parse_args()
    hosts = tuple(args.hosts) if args.hosts else DEFAULT_HOSTS

    print(f"[smoke] querying {len(hosts)} cluster machines for daemon SHA "
          f"+ build stamp\n")
    results: Dict[str, Optional[Dict[str, str]]] = {}
    for h in hosts:
        results[h] = query(h)

    print()
    print(f"{'host':35s}  {'sha':10s}  {'built_on':20s}  built_at")
    print(f"{'-' * 35}  {'-' * 10}  {'-' * 20}  {'-' * 20}")
    for h, info in results.items():
        if info is None:
            print(f"{h:35s}  (failed)")
        else:
            print(f"{h:35s}  {info['sha']:10s}  {info['built_on']:20s}  "
                  f"{info['built_at']}")
    print()

    failed = [h for h, info in results.items() if info is None]
    if failed:
        print(f"[smoke] ✗ {len(failed)} host(s) failed to report a version: "
              f"{', '.join(failed)}", file=sys.stderr)
        return 1

    shas = {info["sha"] for info in results.values() if info}
    if len(shas) != 1:
        print(f"[smoke] ✗ SHA DRIFT detected — {len(shas)} distinct shas "
              f"across the cluster: {sorted(shas)}", file=sys.stderr)
        print("[smoke]   re-run experiments/v0.0/build-fabric.sh to reconcile",
              file=sys.stderr)
        return 2

    sha = next(iter(shas))
    if args.expected_sha and sha != args.expected_sha:
        print(f"[smoke] ✗ cluster SHA {sha} != expected {args.expected_sha}",
              file=sys.stderr)
        return 3

    print(f"[smoke] ✓ all {len(hosts)} cluster machines locked to "
          f"SHA {sha}")
    print(f"[smoke] daemon-skew finding from 2026-05-21 retro is "
          f"CLOSED for the fabric path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
