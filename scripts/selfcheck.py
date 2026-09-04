"""selfcheck.py — what a freshly joined node must PROVE before it says "ready".

⚠️⚠️ THE CHECK THIS EXISTS FOR. A one-command installer shipped `--n-gpu-layers 0` for weeks,
so machines with real GPUs served on CPU — and every health check passed, because the daemon
genuinely was running. "Is it up?" was the wrong question; "is it using the card?" was the
right one, and nothing asked it. A GPU node serving zero layers on the GPU looks identical to
a working node from outside.

⚠️ EVERY ABSENT FACT IS A FAILURE, NEVER A PASS. `facts.get("model_loaded")` returning None
must not read as falsy-so-skip or truthy-so-fine: a fact the setup agent did not OBSERVE is a
fact not ESTABLISHED. The two are reported differently — "not observed" versus "observed and
wrong" — because they call for different fixes, and collapsing them sends an operator to
inspect a value that was never taken.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

GPU_ACCELS = frozenset({"cuda", "rocm", "vulkan"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

#: The facts a node must supply. Absent ⇒ "not observed" ⇒ not ready.
REQUIRED = ("daemon_pid", "accel", "serve_ngl", "identity_pubkey",
            "model_loaded", "answered_probe_ms")


def node_ready(facts: Any) -> Tuple[bool, List[str]]:
    """Return (ready, reasons). Every failing reason, not just the first — an operator
    fixing a node wants the whole list, not one round trip per problem."""
    if not isinstance(facts, dict):
        return False, ["facts is not a dict — nothing was observed"]

    reasons: List[str] = []
    missing = {k for k in REQUIRED if k not in facts}
    for k in REQUIRED:
        if k in missing:
            reasons.append(f"{k} not observed")

    pid = facts.get("daemon_pid")
    if "daemon_pid" not in missing and not (isinstance(pid, int)
                                            and not isinstance(pid, bool) and pid > 0):
        reasons.append("daemon_pid not a positive int — no serving process")

    accel = facts.get("accel")
    ngl = facts.get("serve_ngl")
    ngl_ok = isinstance(ngl, int) and not isinstance(ngl, bool)
    if "serve_ngl" not in missing and not ngl_ok:
        reasons.append("serve_ngl not an int")
    # ⚠️ accel and serve_ngl are judged TOGETHER and neither substitutes for the other:
    # without accel you cannot know whether 0 is correct or catastrophic, so an absent accel
    # is its own reason rather than a licence to skip the ngl rule.
    if "accel" not in missing and "serve_ngl" not in missing and ngl_ok:
        if accel in GPU_ACCELS and ngl <= 0:
            reasons.append(f"serve_ngl is {ngl} on accel={accel} — a GPU node serving no "
                           "layers on the GPU; the daemon is up and the card is idle")
        elif accel == "cpu" and ngl != 0:
            reasons.append(f"serve_ngl is {ngl} on accel=cpu — claiming offload with no GPU")
        elif accel not in GPU_ACCELS and accel != "cpu":
            reasons.append(f"accel {accel!r} is not one of cuda/rocm/vulkan/cpu")

    pub = facts.get("identity_pubkey")
    if "identity_pubkey" not in missing and not (isinstance(pub, str) and _HEX64.match(pub)):
        # ⚠️ Not cosmetic: an unregistered node can serve but can never be PAID. That is not
        # "ready", it is donating, and the operator should be told before the work starts.
        reasons.append("identity_pubkey not 64 lowercase hex chars — node could serve but "
                       "could never be credited")

    if "model_loaded" not in missing and facts.get("model_loaded") is not True:
        reasons.append("model_loaded is not True")

    ms = facts.get("answered_probe_ms")
    if "answered_probe_ms" not in missing:
        if not isinstance(ms, (int, float)) or isinstance(ms, bool):
            reasons.append("answered_probe_ms not a number")
        elif not (0 < ms < 60000):
            reasons.append(f"answered_probe_ms {ms} outside (0, 60000) — it did not answer")

    return (not reasons), reasons
