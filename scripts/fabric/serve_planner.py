"""
serve_planner — THE LAST MILE: turn admitted, firewall-eligible workers into a chain the live
serve actually consumes.

Everything upstream is built: a box presents a signed identity → `admission.admit()` rosters it at
a TRUST TIER → `worker_join.eligible_workers()` is the IDENTITY FIREWALL (a stranger's GPU is never
assigned Prithvi's sensitive layers). What was missing is the bridge from "who MAY serve" to a
concrete chain that `client.py` / `nakshatra_serve.py` runs:

    roster (peers.tsv)  →  eligible_workers(model)  [THE FIREWALL]  →  contiguous layer assignment
                        →  client.py cluster-YAML   →  live serve  →  a real token.

Prithvi's own steer (he asked for this): "wire up MY OWN peer to serve layers into the inference
chain." So the first proof is his SELF-tier node serving a self-only model — the strictest firewall
case — and only then does the same planner generalize to public models on stranger workers (the
firewall keeps his sensitive self on his own/trusted nodes either way).

Design notes:
  • Self-contained: the layer partition is a built-in CONTIGUOUS even split. `partition_fn` is
    injectable, so Sthambha's `plan_split` (compute/bandwidth-aware) can drop in later without a
    rewrite — but the planner needs no extra repo to run.
  • Composes, doesn't rebuild: `eligible_fn` defaults to `worker_join.eligible_workers` (which reads
    the live `infra/control-plane/admission.py` model policy); inject it to test without a control
    plane or GPU.
  • Does NOT slice GGUFs: dynamic per-assignment slicing is a separate packaging step. `slice_for`
    maps an assigned worker+range → its sub-GGUF path; the first proof injects the real pre-cut
    slices the live workers already serve.

The chain contract `client.py` enforces (see client.py `_setup_chain`): workers must contiguously
cover [0, num_blocks); the first worker carries token_embd (mode "first"), the last carries lm_head
(mode "last"). The built-in split honors this.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import worker_join as wj  # WorkerStanding + eligible_workers (the firewall)


# ── contiguous layer partition (the built-in planner) ───────────────────────────────────────────────
def contiguous_split(num_layers: int, n_workers: int) -> list[tuple[int, int, str]]:
    """Split [0, num_layers) into `n_workers` CONTIGUOUS ranges (remainder spread to the front),
    tagged first / middle / last to satisfy the chain contract. Returns [(start, end, mode), ...].

    A 1-worker chain (one box owns every layer) is "solo": it carries both token_embd and lm_head.
    client.py's stock partition check expects a first+last pair, so a solo chain needs a solo-capable
    worker — we tag it explicitly rather than silently mislabel it."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if n_workers <= 0:
        raise ValueError("no eligible workers to assign layers to")
    if n_workers > num_layers:
        raise ValueError(f"{n_workers} workers but only {num_layers} layers — can't give each a layer")
    base, extra = divmod(num_layers, n_workers)
    ranges: list[tuple[int, int, str]] = []
    start = 0
    for i in range(n_workers):
        size = base + (1 if i < extra else 0)
        end = start + size
        if n_workers == 1:
            mode = "solo"
        elif i == 0:
            mode = "first"
        elif i == n_workers - 1:
            mode = "last"
        else:
            mode = "middle"
        ranges.append((start, end, mode))
        start = end
    return ranges


def _default_slice_for(worker: wj.WorkerStanding, start: int, end: int, model_id: str) -> str:
    """Fallback sub-GGUF path. Dynamic slicing is a separate step; this is a stable template so the
    chain YAML is well-formed even before a slicer runs. The first proof injects real slice paths."""
    return str(Path.home() / ".nakshatra" / "models" / f"{model_id}-L{start}-{end}.gguf")


def _addr_port(w: wj.WorkerStanding) -> tuple[str, int]:
    """A worker's serving endpoint. WorkerStanding carries it in `capabilities` (address/port), which
    the worker self-describes on join; falls back to the admission `coord` if that's where it lives."""
    caps = w.capabilities or {}
    addr = caps.get("address") or caps.get("coord") or "127.0.0.1"
    port = int(caps.get("port") or 0)
    return addr, port


@dataclass
class ChainPlan:
    """The planner's output: the chain YAML dict client.py consumes, plus the firewall audit."""
    model_id: str
    chain: dict                                  # {model:{...}, workers:[...]} — the client.py YAML
    eligible: list = field(default_factory=list)  # node_ids that may serve this model
    rejected: list = field(default_factory=list)  # [{worker, reason, ...}] — firewall denials
    min_tier: str = ""                           # the model's required floor (the privacy lever)

    def to_report(self) -> dict:
        return {"model": self.model_id, "min_tier": self.min_tier,
                "eligible": list(self.eligible), "rejected": list(self.rejected),
                "assignment": [{"worker": w["id"], "address": w["address"], "port": w["port"],
                                "layer_range": w["layer_range"], "mode": w["mode"]}
                               for w in self.chain["workers"]]}


# ── the firewall-gated planner ────────────────────────────────────────────────────────────────────
def plan_chain(model_id: str, workers: Sequence[wj.WorkerStanding], *,
               num_layers: int, hidden_size: int, wire_dtype: str = "f32",
               eligible_fn: Optional[Callable] = None,
               partition_fn: Callable[[int, int], list] = contiguous_split,
               slice_for: Optional[Callable[[wj.WorkerStanding, int, int, str], str]] = None,
               min_tier_fn: Optional[Callable[[str], str]] = None,
               rank: Optional[dict] = None,
               place_fn: Optional[Callable] = None,
               max_stages: Optional[int] = None) -> ChainPlan:
    """Produce a chain YAML for `model_id` from the admitted `workers`, GATED BY THE IDENTITY
    FIREWALL. Workers whose trust tier is below the model's min tier are excluded (logged in
    `rejected`); the rest get contiguous layer assignments.

    eligible_fn(model, workers) -> (eligible, rejected) defaults to worker_join.eligible_workers,
    which reads the live admission model policy. Inject it (and/or min_tier_fn/rank) to run without
    a control plane. Raises if no worker survives the firewall (default-deny: refuse to serve rather
    than fall back to a lower-tier peer).

    `max_stages` (FEWER-HOP placement, the WAN win): cap the chain to at most this many workers (each
    then holds more layers). Every chain hop is a latency-bound round-trip over WAN, so 2 stages ~=
    2x faster than 3; set this when workers are high-latency + have spare memory. The kept workers are
    the FIRST `max_stages` of the (already RTT/preference-ordered) eligible list — so ordering upstream
    decides WHICH survive. Minimum 2 (a chain needs first + last). None = use all eligible (default)."""
    if eligible_fn is None:
        eligible, rejected = wj.eligible_workers(model_id, list(workers),
                                                 min_tier_fn=min_tier_fn, rank=rank)
    else:
        eligible, rejected = eligible_fn(model_id, list(workers))

    # the model's floor, for the audit (best-effort — never fail the plan just to label it)
    floor = ""
    try:
        if min_tier_fn is not None:
            floor = min_tier_fn(model_id)
        else:
            adm = wj._load_admission()
            floor = adm.model_min_tier(model_id, adm.load_models())
    except Exception:
        floor = ""

    if not eligible:
        raise PermissionError(
            f"identity firewall: no admitted worker is eligible to serve '{model_id}' "
            f"(min tier '{floor or '?'}'); refusing to serve rather than drop to a lower tier. "
            f"rejected={[r.get('reason') for r in rejected]}")

    # FEWER-HOP placement: cap stages for WAN (each kept worker holds more layers, but fewer
    # latency-bound round-trips). Keep the first max_stages (>=2; ordering upstream picks WHICH).
    if max_stages and len(eligible) > max(2, max_stages):
        eligible = eligible[:max(2, max_stages)]

    slice_for = slice_for or (lambda w, s, e, m: _default_slice_for(w, s, e, m))
    # SMART PLACEMENT (NKS_SMART_PLACEMENT): if a place_fn is injected and it returns a
    # capacity/wire-aware assignment over a CHOSEN SUBSET of eligible workers, use that
    # (route-whole = one solo worker; else a balanced split). If it returns falsy — model
    # un-placeable or no telemetry — fall back to the built-in even split. Placement never
    # fails the serve; worst case it's the current behavior.
    placed = place_fn(eligible, num_layers) if place_fn is not None else None
    if placed:
        used = list(placed)
    else:
        used = [(w, s, e, m)
                for w, (s, e, m) in zip(eligible, partition_fn(num_layers, len(eligible)))]
    workers_yaml = []
    for w, start, end, mode in used:
        addr, port = _addr_port(w)
        workers_yaml.append({
            "id": w.node_id, "address": addr, "port": port,
            "layer_range": [start, end], "mode": mode,
            "sub_gguf_path": slice_for(w, start, end, model_id),
        })
    chain = {"model": {"id": model_id, "hidden_size": hidden_size,
                       "num_blocks": num_layers, "wire_dtype": wire_dtype},
             "workers": workers_yaml}
    return ChainPlan(model_id=model_id, chain=chain,
                     eligible=[w.node_id for w in eligible], rejected=rejected, min_tier=floor)


# ── roster bridge: load the live admission roster → WorkerStandings → plan ──────────────────────────
def standings_from_roster(*, roster_loader: Optional[Callable[[], dict]] = None) -> list[wj.WorkerStanding]:
    """Turn the operator-curated admission roster (peers.tsv: pubkey·name·operator·tier·tenant·coord)
    into worker standings. Each rostered peer is a candidate serving worker at its `coord` endpoint;
    its admission tier is what the firewall gates on. Inject `roster_loader` to test."""
    if roster_loader is None:
        adm = wj._load_admission()
        roster_loader = adm.load_peers
    peers = roster_loader() or {}
    standings = []
    for pub, p in peers.items():
        coord = p.get("coord") or ""
        addr, _, port = coord.partition(":")
        standings.append(wj.WorkerStanding(
            node_id=p.get("name") or pub[:12], pubkey=pub, admitted=True,
            tier=p.get("tier", "stranger"), tenant=p.get("tenant"),
            capabilities={"address": addr or "127.0.0.1", "port": int(port) if port.isdigit() else 0}))
    return standings


if __name__ == "__main__":
    import argparse, yaml, json
    ap = argparse.ArgumentParser(description="firewall-gated chain planner: roster -> chain.yaml")
    ap.add_argument("--model", required=True, help="model id (its min tier is the firewall floor)")
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--hidden-size", type=int, required=True)
    ap.add_argument("--wire-dtype", default="f32")
    ap.add_argument("--out", help="write chain YAML here (default: stdout)")
    args = ap.parse_args()
    standings = standings_from_roster()
    plan = plan_chain(args.model, standings, num_layers=args.num_layers,
                      hidden_size=args.hidden_size, wire_dtype=args.wire_dtype)
    sys.stderr.write(json.dumps(plan.to_report(), indent=2) + "\n")
    out = yaml.safe_dump(plan.chain, sort_keys=False)
    if args.out:
        Path(args.out).write_text(out)
        sys.stderr.write(f"wrote {args.out}\n")
    else:
        print(out)
