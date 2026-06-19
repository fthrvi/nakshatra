"""
unconscious_lifecycle.py — the 3-state lifecycle for Prithvi's "unconscious brain".

Prithvi's unconscious (the DeepSeek-R1 reasoner the conscious mind escalates to via think_deeper)
should behave like a roaming organ, NOT a pinned service:

    DISABLED          operator turned it off            → nothing summoned; think_deeper degrades
    IDLE_NO_HARDWARE  enabled, but no OWNED accelerator → nothing summoned; waits for hardware
                      is currently present on the mesh
    READY             enabled + owned accelerator(s)    → place the chain across them & serve
                      present                             (scale-to-zero when not thinking)

"Owned" is the trust-tier `self` — Prithvi's unconscious only ever runs on *your* hardware, never a
stranger's. (Foreign nodes are excluded structurally: the unconscious model's min trust tier is the
strictest, so the identity firewall in worker_join.eligible_workers already drops non-self nodes.
This module makes that an explicit, testable lifecycle and adds the enabled toggle + idle state, so
the service no longer CRASHES when no hardware is present — it just idles.)

This is PURE policy + tiny file I/O — no GPU, no transport. The summon path (RosterWorkerController)
consults `decide_placement` before launching, and the hub reads `write_status`'s JSON to show/flip
the state. Default-safe: with no flag file, `read_enabled` returns True and callers opt INTO the
lifecycle gate explicitly, so the live service's behavior is unchanged until deliberately wired.

v1 SCOPE (honest): "owned hardware present" means the self-tier accelerator nodes listed in the
admission roster (peers.tsv) — which GROWS when a node joins — NOT a live heartbeat. There is no
liveness/capability heartbeat yet, so a node that is rostered-but-down still reads as present until
a connect fails (the firewall/recovery handles that downstream). A real presence feed (and the
join→roster persist link that records a newly-joined node's detected kind/vram) is the next layer.
"""
from __future__ import annotations

import enum
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import worker_join as wj  # noqa: E402  (same dir; the identity firewall lives here)


class UnconsciousState(str, enum.Enum):
    DISABLED = "disabled"
    IDLE_NO_HARDWARE = "idle"
    READY = "ready"


@dataclass
class Placement:
    """The lifecycle decision: the state, the owned nodes chosen to host the chain (empty unless
    READY), their aggregate accelerator memory (ADVISORY only — see decide_placement), the
    firewall's reject reasons (for the UI to explain an IDLE), and a human reason."""
    state: UnconsciousState
    nodes: list = field(default_factory=list)      # list[WorkerStanding] when READY, else []
    total_vram_mb: int = 0
    reason: str = ""
    rejected: list = field(default_factory=list)   # [{worker, reason, ...}] from the firewall

    @property
    def is_ready(self) -> bool:
        return self.state == UnconsciousState.READY


def _cap(w) -> dict:
    return getattr(w, "capabilities", None) or {}


def decide_placement(enabled: bool, standings: Sequence, *, model_id: str,
                     eligible_fn: Optional[Callable] = None, min_vram_mb: int = 0,
                     min_tier_fn: Optional[Callable[[str], str]] = None,
                     rank: Optional[dict] = None) -> Placement:
    """Classify the unconscious's lifecycle state from the toggle + the nodes currently in the
    roster, using THE SAME identity firewall the summon path uses.

    The gate is `worker_join.eligible_workers(model_id, standings, …)` — trust-tier (rank ≥ the
    model's min tier; the unconscious model resolves to `self`, so strangers are excluded) + the
    capability floor (explicit `cpu-worker` rejected) + an optional per-node `min_vram_mb`. Reusing
    it (rather than reimplementing a tier/vram filter) GUARANTEES the lifecycle agrees with
    `plan_chain`: whatever decide_placement calls READY, plan_chain will serve; whatever it calls
    IDLE, plan_chain would have raised PermissionError on — so IDLE is the graceful pre-check that
    keeps the service from crashing when no owned hardware is present.

    `eligible_fn(model, standings)->(eligible, rejected)` is injectable for tests; default is the
    live firewall (loads the admission policy). `min_tier_fn`/`rank` let callers/tests run the
    firewall without a control plane. vram is ADVISORY (reported, not gated) — pipeline-parallel
    holds a per-node slice, not the whole model, so a sum-of-vram gate is misleading; a real floor
    belongs per-node inside `eligible_workers` via `min_vram_mb`.
    """
    if not enabled:
        return Placement(UnconsciousState.DISABLED, [], 0, "operator disabled")

    if eligible_fn is None:
        eligible_fn = lambda m, ws: wj.eligible_workers(
            m, list(ws), min_tier_fn=min_tier_fn, rank=rank, min_vram_mb=min_vram_mb)
    eligible, rejected = eligible_fn(model_id, standings)
    total = sum(int(_cap(w).get("vram_mb") or 0) for w in eligible)

    if not eligible:
        return Placement(UnconsciousState.IDLE_NO_HARDWARE, [], 0,
                         "no owned accelerator is eligible (identity firewall + capability floor)",
                         rejected=list(rejected))
    return Placement(UnconsciousState.READY, list(eligible), total,
                     f"{len(eligible)} eligible owned node(s)" +
                     (f", {total}MB aggregate" if total else ""),
                     rejected=list(rejected))


# ── the operator toggle (a tiny file the hub writes; the controller reads) ───────────────────────
_TRUE = {"1", "on", "enabled", "true", "yes"}
_FALSE = {"0", "off", "disabled", "false", "no"}


def read_enabled(flag_path: os.PathLike | str, *, default: bool = True) -> bool:
    """Read the operator toggle. ABSENT file → `default` (True) so the live unconscious is unaffected
    until someone deliberately disables it. Unrecognized content → default."""
    try:
        txt = Path(flag_path).read_text(encoding="utf-8").strip().lower()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return default
    except OSError:
        return default
    if txt in _TRUE:
        return True
    if txt in _FALSE:
        return False
    return default


def set_enabled(flag_path: os.PathLike | str, value: bool) -> None:
    """Flip the toggle (what the hub button does). Atomic-ish write."""
    p = Path(flag_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("1\n" if value else "0\n", encoding="utf-8")
    tmp.replace(p)


def write_status(status_path: os.PathLike | str, placement: Placement, *,
                 model_id: str = "", min_tier: str = "", now: Optional[float] = None) -> dict:
    """Write the lifecycle status JSON the hub (infra.html) reads to show state + placement + WHY.
    Returns the dict it wrote (handy for tests)."""
    doc = {
        "model_id": model_id,
        "state": placement.state.value,
        "reason": placement.reason,
        "min_tier": min_tier,                 # the trust floor strangers can't cross (self)
        "total_vram_mb": placement.total_vram_mb,
        "nodes": [{
            "id": getattr(w, "node_id", "?"),
            "tier": getattr(w, "tier", "?"),
            "kind": _cap(w).get("kind") or "?",
            "vram_mb": int(_cap(w).get("vram_mb") or 0),
            "gpu": _cap(w).get("gpu") or _cap(w).get("backend") or "?",
        } for w in placement.nodes],
        "rejected": placement.rejected,       # why candidate nodes were excluded (for an IDLE)
        "updated": now if now is not None else time.time(),
    }
    p = Path(status_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(p)
    return doc


# Default control-plane paths (the hub writes the flag; the controller writes the status).
DEFAULT_FLAG_PATH = str(Path.home() / ".nakshatra" / "unconscious.enabled")
DEFAULT_STATUS_PATH = "/run/prithvi/unconscious/status.json"
