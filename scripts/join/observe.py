"""The ONLY place in the join path that touches the world.

Every phase is a pure decision; this is what feeds them. Keeping all I/O behind one function
is what makes `nakshatra join --dry-run` a real test rather than a demo: swap this for a dict
and the entire six-phase sequence runs on a laptop with no GPU, no network and no coordinator.

⚠️ An observation that FAILS returns nothing rather than raising. A fact we could not take is
not a fact against the node — the phase decides whether it can proceed without it, and it
will say "not observed" rather than guessing. That distinction is why `selfcheck` reports
"not observed" separately from "observed and wrong": they call for different fixes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any, Dict

from acceldetect import detect_accel
from compat import compatible          # noqa: F401  (used by the admit facts)
from joincode import decode_join
from procargs import serve_args


def _run(cmd: list[str], timeout: int = 20) -> str:
    """Best-effort capture. A probe that is absent is not an error — a box without
    `nvidia-smi` is a box without NVIDIA, which is a fact, not a failure."""
    if not shutil.which(cmd[0]):
        return ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception:
        return ""


def _listeners() -> list[dict]:
    """Who holds which port, from `ss`. ⚠️ Parsed defensively: this feeds a decision, and a
    format change in `ss` must degrade to "no listeners seen", never to a crash."""
    out, rows = _run(["ss", "-lntp"]), []
    for line in out.splitlines()[1:]:
        m = re.search(r":(\d+)\s", line)
        if not m:
            continue
        pid = re.search(r"pid=(\d+)", line)
        name = re.search(r'users:\(\("([^"]+)"', line)
        rows.append({"port": int(m.group(1)), "addr": "",
                     "pid": int(pid.group(1)) if pid else None,
                     "name": name.group(1) if name else None})
    return rows


def observe(phase: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """Gather what `phase` is about to need. Called by the orchestrator BEFORE each phase."""
    if phase == "admit":
        out: Dict[str, Any] = {"now": int(__import__("time").time())}
        code = facts.get("join_code")
        if isinstance(code, str) and code:
            try:
                out["join"] = decode_join(code, now=out["now"])
            except ValueError as e:
                # ⚠️ Deliberately not re-raised and deliberately not carrying the code: the
                # phase will refuse for want of `join`, and the reason reaches the operator
                # without the credential reaching the log.
                out["join_decode_error"] = str(e)
        return out

    if phase == "preflight":
        accel = detect_accel(nvidia_smi=_run(["nvidia-smi"]), rocminfo=_run(["rocminfo"]),
                             vulkaninfo=_run(["vulkaninfo", "--summary"]),
                             sysfs_drm=os.listdir("/sys/class/drm") if os.path.isdir("/sys/class/drm") else [])
        st = os.statvfs(facts.get("work_dir") or os.path.expanduser("~"))
        return {"accel": accel, "free_bytes": st.f_bavail * st.f_frsize,
                "listeners": _listeners()}

    if phase == "serve":
        pid = facts.get("daemon_pid")
        return {"running_argv": serve_args(pid)} if isinstance(pid, int) and pid > 0 else {}

    return {}


def load_observations(path: str) -> Dict[str, Any]:
    """⚠️ THE DRY-RUN DOOR. Observations from a JSON file instead of from the machine, so the
    whole sequence can be exercised — including every refusal path — without a GPU, a network
    or a coordinator. A join path you can only test by actually joining is one nobody tests."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("observations file must be a JSON object")
    return data
