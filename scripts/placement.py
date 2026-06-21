"""placement.py — decide WHERE a model's layers run on the mesh (the big-model
prize, pure-Python core). Pairs with slice_directory (who *holds* what); this
decides who *should* hold what.

The cardinal rule (research 2026-06-21): route-don't-split when it fits one box,
split only when forced, and keep a forced split inside ONE low-RTT cluster so the
client pays one WAN hop, not K. This module is the planner; the lifecycle/roster
executes its output. No GPU, no network — deterministic + unit-tested.

Levers, in order of leverage:
  1. fits_whole → choose_whole_host  : 0 cross-node hops (the ~6× win we measured)
  2. metro_clusters                  : group nodes within an RTT threshold
  3. assign_spans                    : contiguous layer ranges, throughput-weighted,
                                       VRAM-capped, covering all layers
  4. plan                            : ties it together → whole-host OR a cluster split
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Node:
    name: str
    vram_gb: float
    tok_per_s: float = 0.0          # measured local decode throughput (capacity)
    # rtt_ms to other nodes is supplied separately as a matrix


def model_fits(node_vram_gb: float, model_gb: float, headroom_gb: float = 1.0) -> bool:
    """Whole model + KV/activation headroom fits one node's VRAM."""
    return node_vram_gb >= model_gb + headroom_gb


def choose_whole_host(nodes: List[Node], model_gb: float,
                      headroom_gb: float = 1.0) -> Optional[Node]:
    """The fastest node that can hold the WHOLE model (route-don't-split). None if
    no single node fits → a split is forced."""
    candidates = [n for n in nodes if model_fits(n.vram_gb, model_gb, headroom_gb)]
    if not candidates:
        return None
    return max(candidates, key=lambda n: (n.tok_per_s, n.vram_gb))


def metro_clusters(nodes: List[Node], rtt_ms: Dict[Tuple[str, str], float],
                   threshold_ms: float = 5.0) -> List[List[Node]]:
    """Union-find group nodes whose pairwise RTT ≤ threshold. A forced split
    should live entirely inside one cluster (one WAN cut for the client)."""
    parent = {n.name: n.name for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            r = rtt_ms.get((a.name, b.name), rtt_ms.get((b.name, a.name)))
            if r is not None and r <= threshold_ms:
                union(a.name, b.name)
    groups: Dict[str, List[Node]] = {}
    for n in nodes:
        groups.setdefault(find(n.name), []).append(n)
    # largest cluster first (most aggregate capacity)
    return sorted(groups.values(), key=lambda g: -sum(x.vram_gb for x in g))


def assign_spans(cluster: List[Node], total_layers: int, model_gb: float,
                 headroom_gb: float = 1.0) -> Optional[Dict[str, Tuple[int, int]]]:
    """Assign CONTIGUOUS layer ranges across a cluster, sized by each node's
    capacity (throughput-weighted, VRAM-capped), covering all layers in node
    order. Returns {node: (start, end)} or None if the cluster can't hold the
    model even split (aggregate VRAM too small)."""
    per_layer_gb = model_gb / total_layers
    # max layers each node can hold (VRAM cap)
    caps = {n.name: max(0, int((n.vram_gb - headroom_gb) / per_layer_gb))
            for n in cluster}
    if sum(caps.values()) < total_layers:
        return None  # even split doesn't fit this cluster
    # weight by throughput (fall back to vram) so faster nodes get more layers,
    # but never exceed the VRAM cap.
    weights = {n.name: (n.tok_per_s or n.vram_gb) for n in cluster}
    wsum = sum(weights.values()) or 1.0
    want = {n.name: weights[n.name] / wsum * total_layers for n in cluster}
    # round to ints, cap by VRAM, then fix the remainder so they sum to total
    alloc = {n.name: min(caps[n.name], int(round(want[n.name]))) for n in cluster}
    # distribute leftover layers to nodes with spare VRAM capacity
    remaining = total_layers - sum(alloc.values())
    order = sorted(cluster, key=lambda n: -(caps[n.name] - alloc[n.name]))
    i = 0
    while remaining != 0 and order:
        n = order[i % len(order)]
        if remaining > 0 and alloc[n.name] < caps[n.name]:
            alloc[n.name] += 1; remaining -= 1
        elif remaining < 0 and alloc[n.name] > 0:
            alloc[n.name] -= 1; remaining += 1
        i += 1
        if i > 10 * len(order) * (abs(remaining) + 1):
            break
    if remaining != 0:
        return None
    # lay out contiguous ranges (skip zero-width nodes)
    spans: Dict[str, Tuple[int, int]] = {}
    cur = 0
    for n in cluster:
        k = alloc[n.name]
        if k > 0:
            spans[n.name] = (cur, cur + k)
            cur += k
    return spans


def balanced_spans(cluster: List[Node], total_layers: int, model_gb: float,
                   headroom_gb: float = 1.0) -> Optional[Dict[str, Tuple[int, int]]]:
    """WATER-FILLING placement (shard's heterogeneous edge): partition contiguous
    layer ranges to MINIMIZE the slowest pipeline stage — `max_i (layers_i /
    tok_per_s_i)` — not just split proportionally. The pipeline's per-token time is
    its slowest stage, so equalizing stage-time (capacity-aware, VRAM-capped) is the
    right objective on mixed GPUs (a fast 4090 should hold more layers than a slow
    3060). Returns {node:(start,end)} contiguous + full coverage, or None if even
    the balanced split can't fit. Falls back to vram-weighting if throughput unknown.

    Method: binary-search the stage-time ceiling T; at T each node can serve
    floor(T * tok_per_s) layers (capped by VRAM); find the min T whose capacity
    covers all layers, then assign greedily up to each node's share."""
    per_layer_gb = model_gb / total_layers
    cap = {n.name: max(0, int((n.vram_gb - headroom_gb) / per_layer_gb)) for n in cluster}
    if sum(cap.values()) < total_layers:
        return None
    rate = {n.name: (n.tok_per_s if n.tok_per_s > 0 else n.vram_gb) for n in cluster}

    def layers_at(T: float) -> Dict[str, int]:
        # at stage-time ceiling T, node can do up to T*rate layers (VRAM-capped)
        return {n.name: min(cap[n.name], int(T * rate[n.name])) for n in cluster}

    lo, hi = 0.0, total_layers / min(rate.values()) + 1.0   # hi: slowest node does all
    for _ in range(60):                                      # binary search on T
        mid = (lo + hi) / 2
        if sum(layers_at(mid).values()) >= total_layers:
            hi = mid
        else:
            lo = mid
    alloc = layers_at(hi)
    # trim overshoot to exactly total_layers, dropping from the most-overloaded first
    extra = sum(alloc.values()) - total_layers
    for n in sorted(cluster, key=lambda n: -(alloc[n.name] / rate[n.name] if alloc[n.name] else 0)):
        if extra <= 0:
            break
        d = min(extra, alloc[n.name]); alloc[n.name] -= d; extra -= d
    if sum(alloc.values()) != total_layers:
        return None
    spans: Dict[str, Tuple[int, int]] = {}
    cur = 0
    for n in cluster:                                        # contiguous, node order
        k = alloc[n.name]
        if k > 0:
            spans[n.name] = (cur, cur + k); cur += k
    return spans


@dataclass
class Plan:
    whole_host: Optional[str] = None
    splits: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    cluster: List[str] = field(default_factory=list)
    reason: str = ""


def plan(model_gb: float, total_layers: int, nodes: List[Node],
         rtt_ms: "Optional[Dict[Tuple[str, str], float]]" = None,
         cluster_threshold_ms: float = 5.0,
         headroom_gb: float = 1.0) -> Plan:
    """Route-whole if any node fits; else split inside the best (largest-capacity)
    metro cluster. Raises ValueError if even the best cluster can't hold it."""
    host = choose_whole_host(nodes, model_gb, headroom_gb)
    if host is not None:
        return Plan(whole_host=host.name, reason="fits one box — route, don't split")
    rtt_ms = rtt_ms or {}
    for cluster in metro_clusters(nodes, rtt_ms, cluster_threshold_ms):
        # water-filling (stage-time-balanced) first; fall back to proportional
        spans = (balanced_spans(cluster, total_layers, model_gb, headroom_gb)
                 or assign_spans(cluster, total_layers, model_gb, headroom_gb))
        if spans is not None:
            return Plan(splits=spans, cluster=[n.name for n in cluster],
                        reason=f"split across {len(spans)} nodes in one cluster")
    raise ValueError("no node fits whole and no single cluster can hold the split")


def plan_to_chain(plan: Plan, *, model_id: str, model_hash: str, hidden_size: int,
                  total_layers: int, node_addr: Dict[str, Tuple[str, int]],
                  slices_dir: str, wire_dtype: str = "f32") -> dict:
    """Emit a client.py chain config from a Plan — makes the water-filling planner's
    output directly SERVABLE (the planner's hands). For a split, produces the
    `{model, workers:[{id,address,port,layer_range,mode,sub_gguf_path}]}` chain
    (mode = first/mid/last by layer position; slice path is content-addressed). For
    route-whole, returns `{route_to: host, port}` — no split chain needed."""
    if plan.whole_host:
        host, port = node_addr[plan.whole_host]
        return {"route_to": {"host": host, "port": port, "node": plan.whole_host}}
    workers = []
    for node, (a, b) in sorted(plan.splits.items(), key=lambda kv: kv[1][0]):
        host, port = node_addr[node]
        mode = "first" if a == 0 else ("last" if b == total_layers else "mid")
        workers.append({
            "id": f"plan-{node}",
            "address": host, "port": port,
            "layer_range": [a, b],
            "mode": mode,
            "sub_gguf_path": f"{slices_dir}/{model_id}@{model_hash}-L{a}-{b}.gguf",
        })
    return {"model": {"id": model_id, "hidden_size": hidden_size,
                      "num_blocks": total_layers, "wire_dtype": wire_dtype},
            "workers": workers}
