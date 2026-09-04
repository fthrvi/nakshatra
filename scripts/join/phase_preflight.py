"""Join phase 2: can this box actually serve? A pure decision over observations.

⚠️ KV COMES OFF THE TOP, BEFORE LAYERS. Sizing layers first and hoping the KV cache fits is
how a node reports 32 layers, accepts the job, and OOMs on the first long prompt — and the
failure lands on the REQUESTER, who waited, not on the node that mis-declared.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
        usable = free * (1.0 - DISK_RESERVE)
        if usable < need:
            short = (need - usable) / (1 << 20)
            problems.append(f"not enough disk: {short:.0f} MiB short after a "
                            f"{DISK_RESERVE:.0%} reserve")
    elif "free_bytes" not in missing and "slice_bytes" not in missing:
        problems.append("free_bytes/slice_bytes are not numbers")

    port = facts.get("port")
    if _num(port) and isinstance(port, int):
        if port < 1024:
            # ⚠️ A distinct problem from "in use" — it calls for privileges, not a different
            # port, and merging the two sends an operator to the wrong fix.
            problems.append(f"port {port} is below 1024 and needs elevated privileges")
        elif port > 65535:
            problems.append(f"port {port} is out of range")
        listeners = facts.get("listeners")
        if isinstance(listeners, list):
            for l in listeners:
                if isinstance(l, dict) and l.get("port") == port:
                    who = f"{l.get('name') or 'unknown'} (pid {l.get('pid')})"
                    # ⚠️ Name the holder: "8080 in use" starts a hunt; "held by llama-server
                    # (pid 4242)" is usually the previous run of this very daemon.
                    problems.append(f"port {port} is already held by {who}")
                    break
    elif "port" not in missing:
        problems.append("port is not an int")

    vram, nl = facts.get("vram_bytes"), facts.get("n_layers")
    per, ctx = facts.get("model_bytes_per_layer"), facts.get("ctx_tokens")
    kvb = facts.get("kv_bytes_per_token")
    updates: Dict[str, Any] = {}
    if all(_num(x) for x in (vram, nl, per, ctx, kvb)) and per > 0 and nl > 0:
        usable = vram * (1.0 - DISK_RESERVE)
        remaining = usable - (ctx * kvb)          # ⚠️ KV first, always
        layers = int(remaining // per) if remaining > 0 else 0
        if remaining <= 0:
            problems.append(f"KV cache for {int(ctx)} tokens needs "
                            f"{ctx * kvb / (1 << 20):.0f} MiB, more than this card has "
                            f"after reserve — reduce the context, not the model")
        elif layers <= 0:
            problems.append("holds zero layers after the KV cache is reserved")
        else:
            updates = {"max_layers": min(layers, int(nl)),
                       "holds_full_model": layers >= nl}
    elif not any(k in missing for k in ("vram_bytes", "n_layers", "model_bytes_per_layer",
                                        "ctx_tokens", "kv_bytes_per_token")):
        problems.append("capacity observations are not all numbers")

    return (not problems), problems, updates
