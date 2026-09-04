#!/usr/bin/env python3
"""Provision a node in a CLEAN container and judge what actually happened.

    ./test-onboard-e2e.py --image ubuntu:24.04 --script deploy/provision-worker.sh
    ./test-onboard-e2e.py --report-file captured.txt        # judge a saved run, no docker

⚠️⚠️ THE TEST PHASE 1 NEVER HAD. `provision-worker.sh` shipped `--n-gpu-layers 0` for weeks:
machines with real GPUs served on CPU, every health check passed because the daemon genuinely
WAS running, and the installer printed the right thing the whole time. Nothing compared what
it PRINTED with what it RAN.

So this compares them. The container gathers (`onboard-report.sh`), and a pure function
judges (`onboardcheck.onboard_succeeded`) — the same split as the join phases, for the same
reason: the judging half is testable without docker, and `--report-file` is how.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from containerplan import container_argv          # noqa: E402
from onboardcheck import onboard_succeeded        # noqa: E402
from reportparse import parse_report              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="ubuntu:24.04")
    ap.add_argument("--script", default="")
    ap.add_argument("--report-file", default="",
                    help="judge a previously captured run; runs no container at all")
    ap.add_argument("--memory-gb", type=int, default=8)
    ap.add_argument("--timeout-s", type=int, default=3600)
    a = ap.parse_args()

    if a.report_file:
        stdout = Path(a.report_file).read_text()
    else:
        if not a.script:
            print("need --script (or --report-file)", file=sys.stderr)
            return 2
        try:
            argv = container_argv(a.image, str(Path(a.script).resolve()),
                                  memory_gb=a.memory_gb, timeout_s=a.timeout_s)
        except ValueError as e:
            print(f"refused to plan the container: {e}", file=sys.stderr)
            return 2
        print("$ " + " ".join(argv), file=sys.stderr)
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=a.timeout_s + 60)
            stdout = p.stdout
        except FileNotFoundError:
            print("docker not found — use --report-file to judge a captured run",
                  file=sys.stderr)
            return 2
        except subprocess.TimeoutExpired:
            print("FAIL: the container did not finish in time", file=sys.stderr)
            return 1

    report, problems = parse_report(stdout)
    if problems:
        # ⚠️ "No report" is NOT "provisioning failed". The container died before it could
        # speak, which usually points at the harness rather than the installer — and telling
        # an operator their installer is broken when the harness is, costs a day.
        print("could not read the container's report:", file=sys.stderr)
        for p in problems:
            print(f"  · {p}", file=sys.stderr)
        return 2

    ok, why = onboard_succeeded(report)
    if ok:
        print(f"PASS — daemon pid {report.get('daemon_pid')}, "
              f"accel={report.get('detected_accel')}, argv proves GPU offload")
        return 0
    print("FAIL — the node provisioned but is not serving correctly:", file=sys.stderr)
    for w in why:
        print(f"  · {w}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
