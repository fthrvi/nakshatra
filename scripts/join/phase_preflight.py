"""Join phase 2: can this box actually serve? A pure decision over observations.

⚠️ KV COMES OFF THE TOP, BEFORE LAYERS. Sizing layers first and hoping the KV cache fits is
how a node reports 32 layers, accepts the job, and OOMs on the first long prompt — and the
failure lands on the REQUESTER, who waited, not on the node that mis-declared.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# ⚠️ CALL THE MODULES, DO NOT RESTATE THEM. `room_for`, `port_conflicts` and
# `serving_capacity` are already pure functions with their own adversarial tests — the disk
# reserve, the sub-1024 privileges case, and KV-before-layers all live there and were verified
# there. Re-deriving them here would give the repo two answers to the same question, and the
# one that drifts is always the copy nobody tested.
from capability import serving_capacity
from diskcheck import room_for
from portcheck import port_conflicts

GPU_ACCELS = frozenset({"cuda", "rocm", "vulkan"})
DISK_RESERVE = 0.05
NEEDED = ("accel", "free_bytes", "slice_bytes", "port", "listeners",
          "vram_bytes", "n_layers", "model_bytes_per_layer", "ctx_tokens", "kv_bytes_per_token")


def _num(v: Any) -> bool:
    """⚠️ `facts` comes from a machine we do not control. Every comparison needs this first:
    a phase that raises is a phase the orchestrator cannot use, and `'x' < 1.0` is a TypeError
    that took two attempts to stop happening."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def preflight(facts: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    if not isinstance(facts, dict):
        return False, ["facts is not a dict — nothing was observed"], {}
    problems: List[str] = []
    missing = [k for k in NEEDED if k not in facts]
    problems += [f"{k} not observed" for k in missing]

    free, need = facts.get("free_bytes"), facts.get("slice_bytes")
    if _num(free) and _num(need):
        ok, why = room_for(int(free), int(need), reserve=DISK_RESERVE)
        if not ok:
            # ⚠️ `room_for` says "need 953 MiB more" — correct, and it does not say OF WHAT.
            # The phase owns the context because it is the layer that knows: an operator
            # reading a list of problems needs each one to stand alone.
            problems.append(f"disk: {why}")
    elif "free_bytes" not in missing and "slice_bytes" not in missing:
        problems.append("free_bytes/slice_bytes are not numbers")

    port, listeners = facts.get("port"), facts.get("listeners")
    if isinstance(port, int) and not isinstance(port, bool):
        free_port, why = port_conflicts(port, listeners if isinstance(listeners, list) else [])
        if not free_port:
            problems.append(why)
    elif "port" not in missing:
        problems.append("port is not an int")

    vram, nl = facts.get("vram_bytes"), facts.get("n_layers")
    per, ctx = facts.get("model_bytes_per_layer"), facts.get("ctx_tokens")
    kvb = facts.get("kv_bytes_per_token")
    updates: Dict[str, Any] = {}
    if all(_num(x) for x in (vram, nl, per, ctx, kvb)):
        cap = serving_capacity(int(vram), model_bytes_per_layer=int(per), n_layers=int(nl),
                               ctx_tokens=int(ctx), kv_bytes_per_token=int(kvb),
                               headroom=DISK_RESERVE)
        if cap["max_layers"] <= 0:
            problems.append(cap["reason"])
        else:
            updates = {"max_layers": cap["max_layers"],
                       "holds_full_model": cap["holds_full_model"]}
    elif not any(k in missing for k in ("vram_bytes", "n_layers", "model_bytes_per_layer",
                                        "ctx_tokens", "kv_bytes_per_token")):
        problems.append("capacity observations are not all numbers")

    return (not problems), problems, updates
