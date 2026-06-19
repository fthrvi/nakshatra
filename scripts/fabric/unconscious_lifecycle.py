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
"""
from __future__ import annotations

import enum
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence


class UnconsciousState(str, enum.Enum):
    DISABLED = "disabled"
    IDLE_NO_HARDWARE = "idle"
    READY = "ready"


@dataclass
class Placement:
    """The lifecycle decision: the state, the owned nodes chosen to host the chain (empty unless
    READY), their aggregate accelerator memory, and a human reason for the audit/UI."""
    state: UnconsciousState
    nodes: list = field(default_factory=list)      # list[WorkerStanding] when READY, else []
    total_vram_mb: int = 0
    reason: str = ""

    @property
    def is_ready(self) -> bool:
        return self.state == UnconsciousState.READY


def _cap(w) -> dict:
    return getattr(w, "capabilities", None) or {}


def _is_owned(w, require_tier: Optional[str]) -> bool:
    if not getattr(w, "admitted", True):
        return False
    return require_tier is None or getattr(w, "tier", None) == require_tier


def _is_usable_accelerator(w) -> bool:
    """A node can host a layer slice if it has an accelerator. We reject ONLY an explicit
    `cpu-worker` (no accelerator — too slow to be a node). A node that advertises no `kind` at all
    (the legacy roster, older workers) is assumed usable — so enabling the lifecycle never demotes
    the already-running unconscious whose roster carries no capabilities."""
    kind = _cap(w).get("kind")
    return kind != "cpu-worker"


def decide_placement(enabled: bool, standings: Sequence, *,
                     min_vram_mb: int = 0, require_tier: Optional[str] = "self") -> Placement:
    """Classify the unconscious's lifecycle state from the toggle + the live node standings.

    `standings` are WorkerStandings for nodes currently present on the mesh, each ideally carrying
    capabilities {kind, vram_mb}. `require_tier` ("self") restricts placement to OWNED nodes; pass
    None to disable the tier filter (e.g. when the caller already firewalled by tier). `min_vram_mb`
    is an aggregate floor — pipeline-parallel splits the model across nodes, so the sum of owned
    accelerator memory is the first-order "does the model fit across the chain" check (KV/context
    overhead not modelled here — keep a margin). Only enforced when real vram numbers are present.

    Defence-in-depth: READY does NOT bypass the identity firewall — the actual summon still runs
    plan_chain/eligible_workers, which re-checks trust tier and the capability floor.
    """
    if not enabled:
        return Placement(UnconsciousState.DISABLED, [], 0, "operator disabled")

    owned = [w for w in standings if _is_owned(w, require_tier)]
    usable = [w for w in owned if _is_usable_accelerator(w)]
    total = sum(int(_cap(w).get("vram_mb") or 0) for w in usable)

    if not usable:
        tier_note = f" (require_tier={require_tier!r})" if require_tier else ""
        return Placement(UnconsciousState.IDLE_NO_HARDWARE, [], 0,
                         f"no owned accelerator present{tier_note}")
    if min_vram_mb and total and total < min_vram_mb:
        return Placement(UnconsciousState.IDLE_NO_HARDWARE, [], total,
                         f"owned accelerator memory {total}MB < {min_vram_mb}MB needed for the model")
    return Placement(UnconsciousState.READY, list(usable), total,
                     f"{len(usable)} owned accelerator node(s), {total}MB aggregate")


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
                 model_id: str = "", now: Optional[float] = None) -> dict:
    """Write the lifecycle status JSON the hub (infra.html) reads to show state + placement.
    Returns the dict it wrote (handy for tests)."""
    doc = {
        "model_id": model_id,
        "state": placement.state.value,
        "reason": placement.reason,
        "total_vram_mb": placement.total_vram_mb,
        "nodes": [{
            "id": getattr(w, "node_id", "?"),
            "tier": getattr(w, "tier", "?"),
            "vram_mb": int(_cap(w).get("vram_mb") or 0),
            "gpu": _cap(w).get("gpu") or _cap(w).get("backend") or "?",
        } for w in placement.nodes],
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
