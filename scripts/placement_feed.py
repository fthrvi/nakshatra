"""placement_feed.py — feed the placement planner REAL measured data (the missing
bridge). `placement.py` is a pure, unit-tested water-filling planner whose `Node`
carries a `tok_per_s` capacity term — but nothing ever populated it from the live
mesh, so in practice it fell back to VRAM-weighting (a fast 3060 and a slow one
looked identical). This module is the connective tissue: it JOINS two sources the
stack already has and emits the planner's inputs.

    static admission  (peers.tsv / standings: WHO may serve, at what tier, WHERE)
                 ⨝
    live telemetry    (pillar heartbeats: how FAST = recent_rpc_ms, how much
                       VRAM is LOCKED for the network = budget.vram_offered_gb,
                       how many layers it currently serves)
                 ↓
    placement.Node[]  +  rtt_ms matrix  +  addr map   →  placement.plan() → plan_to_chain()

Why `recent_rpc_ms` is the right capacity signal: it is the worker's *local daemon
compute time* for its slice of one decode step (DaemonClient.recent_rpc_ms in
worker.py — worker↔llama.cpp, NOT a network hop). So it measures pure per-node
compute, which is exactly what the water-filling objective `max_i(layers_i /
rate_i)` needs. The correct per-node `rate` is per-layer throughput:

    rate = 1000 · layers_served / recent_rpc_ms      [layers/sec]

(if a node serves L layers in `recent_rpc_ms` ms, it does 1000·L/ms layers per
second). When `layers_served` is unknown we fall back to `1000/recent_rpc_ms`
(relative speed ordering — still strictly better than VRAM). When `recent_rpc_ms`
is unknown (a freshly-joined peer with no data yet) we emit `tok_per_s=0`, which
`placement.py` already treats as "weight by VRAM" — matching client.py's existing
`_rpc_key` convention (no rpc data → defer to other signals).

No GPU, no network in the pure core (capacity/node/rtt builders) — deterministic +
unit-tested. The `*_from_peers` / `to_placement_inputs` loaders do the live join.

Cardinal rule (carried from the architecture note): the wire stays VISIBLE. The RTT
matrix this module builds is a first-class planner input precisely so a forced split
pays a measured, minimized wire cost — never an invisible one.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement  # the pure planner (Node, plan, plan_to_chain)


# ── pure core: telemetry → capacity → Node (deterministic, unit-tested) ──────────────

def capacity_tok_per_s(recent_rpc_ms: float, layers_served: Optional[int] = None) -> float:
    """Convert a worker's measured per-step daemon latency into a placement capacity
    (per-layer throughput, layers/sec). Returns 0.0 when latency is unknown/invalid
    so the planner falls back to VRAM-weighting (placement.py honors tok_per_s==0).

    layers_served known   → 1000 · layers_served / recent_rpc_ms   (true per-layer rate)
    layers_served unknown → 1000 / recent_rpc_ms                   (relative speed proxy)
    """
    try:
        ms = float(recent_rpc_ms)
    except (TypeError, ValueError):
        return 0.0
    if ms <= 0.0:
        return 0.0  # no data yet → defer to VRAM (matches client.py _rpc_key stand-in)
    if layers_served and layers_served > 0:
        return 1000.0 * float(layers_served) / ms
    return 1000.0 / ms


def make_node(name: str, vram_gb: float, recent_rpc_ms: float = 0.0,
              layers_served: Optional[int] = None) -> placement.Node:
    """Build a placement.Node from one node's static identity + live telemetry."""
    return placement.Node(
        name=name,
        vram_gb=float(vram_gb or 0.0),
        tok_per_s=capacity_tok_per_s(recent_rpc_ms, layers_served),
    )


def build_nodes(telemetry: Mapping[str, Mapping]) -> List[placement.Node]:
    """{node_name: {vram_gb, recent_rpc_ms, layers_served}} → [placement.Node].
    The keystone Phase-0 contract: the planner now runs on measured data."""
    nodes: List[placement.Node] = []
    for name, t in telemetry.items():
        nodes.append(make_node(
            name=name,
            vram_gb=t.get("vram_gb", 0.0),
            recent_rpc_ms=t.get("recent_rpc_ms", 0.0),
            layers_served=t.get("layers_served"),
        ))
    return nodes


def rtt_matrix(samples: Iterable[Tuple[str, str, float]]) -> Dict[Tuple[str, str], float]:
    """Fold (from, to, ms) RTT samples into the symmetric matrix placement.metro_clusters
    consumes. Both (a,b) and (b,a) are stored (placement looks up either order, but
    storing both keeps the matrix self-documenting). Last sample wins; self-pairs and
    non-positive RTTs are dropped (a 0ms link would collapse every cluster boundary)."""
    m: Dict[Tuple[str, str], float] = {}
    for a, b, ms in samples:
        if a == b:
            continue
        try:
            v = float(ms)
        except (TypeError, ValueError):
            continue
        if v <= 0.0:
            continue
        m[(a, b)] = v
        m[(b, a)] = v
    return m


# ── live join: pillar peer projection → telemetry (thin I/O-shaped adapters) ──────────

def _peer_vram_gb(peer: Mapping) -> float:
    """VRAM this peer has LOCKED for the network. Prefer the explicit offer
    (budget.vram_offered_gb — the operator's lease, exactly the 'lock your VRAM for
    the swarm' knob); fall back to summed physical GPU VRAM if no budget is declared."""
    budget = peer.get("budget") or {}
    offered = budget.get("vram_offered_gb")
    if offered:
        try:
            return float(offered)
        except (TypeError, ValueError):
            pass
    hw = peer.get("hardware") or {}
    total = 0.0
    for g in hw.get("gpus", []) or []:
        try:
            total += float(g.get("vram_total_gb") or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def _peer_layers_for_model(peer: Mapping, model_id: Optional[str]) -> Optional[int]:
    """Layers this peer currently serves for `model_id` (Σ end-start over its
    layer_offerings). Used to turn recent_rpc_ms into a true per-layer rate. None if
    the peer advertises nothing for this model (→ capacity falls back to the proxy)."""
    if not model_id:
        return None
    total = 0
    for o in peer.get("layer_offerings", []) or []:
        if o.get("model_id") == model_id:
            try:
                total += int(o["layer_end"]) - int(o["layer_start"])
            except (KeyError, TypeError, ValueError):
                continue
    return total or None


def peer_to_telemetry(peer: Mapping, model_id: Optional[str] = None) -> Dict:
    """One pillar peer-projection record → the telemetry dict build_nodes wants."""
    return {
        "vram_gb": _peer_vram_gb(peer),
        "recent_rpc_ms": peer.get("recent_rpc_ms", 0.0) or 0.0,
        "layers_served": _peer_layers_for_model(peer, model_id),
    }


def _peer_name(peer: Mapping) -> str:
    return peer.get("node_id") or peer.get("name") or (peer.get("pubkey") or "")[:12]


def telemetry_from_peers(peers: Iterable[Mapping],
                         model_id: Optional[str] = None) -> Dict[str, Dict]:
    """Live pillar peer projection (the heartbeat-fed list) → {name: telemetry}."""
    return {_peer_name(p): peer_to_telemetry(p, model_id) for p in peers if _peer_name(p)}


def fetch_pillar_peers(pillar_url: Optional[str], model_id: Optional[str] = None,
                       timeout: float = 10.0, *, node_id: Optional[str] = None,
                       key_path: Optional[str] = None) -> list:
    """Fetch the pillar's live peer projection (GET {pillar}/peers) — the dicts that carry
    recent_rpc_ms / budget.vram_offered_gb / layer_offerings.

    SIGNS the request with the worker's Ed25519 key (Sthambha-Ed25519) when one is available,
    so auth-required pillars (which 401 a bare GET) accept it; degrades to UNSIGNED otherwise,
    matching client.py's _fetch_spki_index. The URL stays plain `/peers` (no ?model= query) —
    it matches the existing /peers convention and sidesteps any query-in-canonical ambiguity;
    model filtering happens downstream in telemetry_from_peers(). node_id (the keyid) resolves
    param → NAKSHATRA_NODE_ID → hostname; the key loads from key_path → ~/.nakshatra/keys/
    worker.ed25519 (load-only — never creates a stray unregistered key). [] on ANY failure
    (incl. 401); never raises — placement must not break the serve."""
    if not pillar_url:
        return []
    import json as _json
    from urllib import request as _rq
    base, path = pillar_url.rstrip("/"), "/peers"
    headers = {}
    try:                                   # best-effort signed auth (Sthambha-Ed25519)
        import os as _os, socket as _sock
        from pathlib import Path as _P
        kp = _P(key_path) if key_path else (_P.home() / ".nakshatra" / "keys" / "worker.ed25519")
        if kp.exists():
            priv = kp.read_bytes()
            if len(priv) == 32:
                nid = node_id or _os.environ.get("NAKSHATRA_NODE_ID") or _sock.gethostname()
                import nakshatra_auth as _na
                hdr, _ = _na.build_signed_envelope(priv, nid, "GET", path, b"")
                headers["Authorization"] = hdr
    except Exception:
        pass                               # → unsigned fallback
    try:
        with _rq.urlopen(_rq.Request(base + path, headers=headers, method="GET"),
                         timeout=timeout) as r:
            data = _json.loads(r.read().decode("utf-8"))
        if isinstance(data, dict):
            return data.get("peers") or data.get("online") or []
        return data or []
    except Exception:
        return []


def to_placement_inputs(peers: Iterable[Mapping], *, model_id: Optional[str] = None,
                        rtt_samples: Optional[Iterable[Tuple[str, str, float]]] = None
                        ) -> Tuple[List[placement.Node], Dict[Tuple[str, str], float],
                                   Dict[str, Tuple[str, int]]]:
    """One call: live peers (+ optional RTT samples) → everything placement.plan needs.

    Returns (nodes, rtt_ms, node_addr) where node_addr maps name → (host, port) for
    plan_to_chain(). Apply the identity firewall (serve_planner.eligible_workers) to
    pick WHICH peers reach here — this only annotates capacity; it does not gate trust.
    """
    peers = list(peers)
    telem = telemetry_from_peers(peers, model_id)
    nodes = build_nodes(telem)
    rtt = rtt_matrix(rtt_samples or [])
    addr: Dict[str, Tuple[str, int]] = {}
    for p in peers:
        name = _peer_name(p)
        if not name:
            continue
        coord = p.get("coord") or p.get("address") or ""
        host, _, port = str(coord).partition(":")
        addr[name] = (host or "127.0.0.1", int(port) if port.isdigit() else 0)
    return nodes, rtt, addr


# ── serve adapter: placement.Plan → serve_planner assignment (the Phase-1 seam) ───────

def assignment_from_plan(plan, workers_by_name: Mapping, total_layers: int) -> list:
    """Map a placement.Plan onto serve_planner's assignment shape:
    [(worker, layer_start, layer_end, mode), ...]. Route-whole → one SOLO worker holding
    [0, total_layers). Split → the chosen cluster nodes in layer order (first/middle/last).
    Workers absent from `workers_by_name` are skipped (shouldn't happen). Returns [] if the
    plan can't be realized — the caller then falls back to the even split."""
    if getattr(plan, "whole_host", None):
        w = workers_by_name.get(plan.whole_host)
        return [(w, 0, total_layers, "solo")] if w is not None else []
    items = sorted(plan.splits.items(), key=lambda kv: kv[1][0])  # by layer_start
    n = len(items)
    out = []
    for i, (name, (a, b)) in enumerate(items):
        w = workers_by_name.get(name)
        if w is None:
            continue
        mode = "solo" if n == 1 else ("first" if i == 0 else ("last" if i == n - 1 else "middle"))
        out.append((w, a, b, mode))
    return out


def make_place_fn(*, model_gb: float, telemetry_of, rtt_samples=None, headroom_gb: float = 1.0,
                  name_of=lambda w: w.node_id):
    """Build a serve_planner `place_fn`: (workers, num_layers) -> [(worker,start,end,mode)] | None.

    Runs the real planner on MEASURED data: telemetry_of(worker) -> {vram_gb, recent_rpc_ms,
    layers_served} feeds capacity (route-whole when one node fits = 0 wire crossings; else a
    capacity- and RTT-aware split inside one low-RTT cluster). Returns None on any failure or
    un-placeable model so the caller falls back to the even split — placement never fails the
    serve. This is what plan_chain calls behind NKS_SMART_PLACEMENT."""
    rtt = rtt_matrix(rtt_samples or [])

    def place(workers, num_layers):
        try:
            telem = {name_of(w): telemetry_of(w) for w in workers}
            nodes = build_nodes(telem)
            p = placement.plan(model_gb=model_gb, total_layers=num_layers, nodes=nodes,
                               rtt_ms=rtt, headroom_gb=headroom_gb)
            return assignment_from_plan(p, {name_of(w): w for w in workers}, num_layers) or None
        except Exception:
            return None

    return place
