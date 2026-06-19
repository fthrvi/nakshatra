"""
topology_order.py — order a pipeline chain by measured inter-node latency.

Speed-stack finding #11 (docs/2026-06-19-speed-stack-plan.md, trisul research). In a
pipeline-parallel chain the per-token cost is the sum of inter-stage hops, and internet
latency is asymmetric and ignores geography — so ordering the SELECTED workers on the
*measured* latency mesh is a direct multiplier (shard reported ~4.5× from selection+order).

SCOPE BOUNDARY (important): this module orders an ALREADY-SELECTED set of workers with an
ALREADY-DECIDED layer split. It does NOT choose which workers or how many layers each — the
moment latency should influence *selection*, the right tool flips to DAG shortest-path
(EdgeShard/Parallax), not TSP. With the set fixed and only the sequence free, the problem is
exactly a fixed-depot asymmetric Hamiltonian cycle → Held-Karp is textbook-correct.

The objective (forward-only chain that returns to the coordinator):

    entry(coord -> w0)  +  Σ hop(wi -> wi+1)  +  return(w_last -> coord)

Two correctness rules baked in (both are silent 2×/wrong-cost bugs otherwise):
  • ONE-WAY cost, not RTT. Activations flow forward only; only the last worker returns the
    token. So `cost` is a ONE-WAY latency in ms. If you have a round-trip RTT matrix, convert
    it with `rtt_to_oneway(..., halve=True)` first — using RTT directly double-counts.
  • ASYMMETRIC-safe. cost[(a,b)] may differ from cost[(b,a)] (FabricLinkSnapshot is stored
    per directional link). Held-Karp on a directed cost is fine; the n>exact_max fallback uses
    Or-opt (node reinsertion), NEVER segment-reversal 2-opt (which is only valid on symmetric
    costs and would compute a wrong cost for reversed sub-paths).

Pure: no imports from worker/fabric/grpc. Fully unit-testable without a cluster.
The planner seam (assemble `cost` from FabricLinkSnapshot rows; reorder BEFORE tagging the
first/last embedding/lm_head roles) is a separate later diff.
"""
from __future__ import annotations

from itertools import permutations
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

# A directed one-way latency matrix: cost[(a, b)] = ms to send one-way from a to b.
CostMatrix = Dict[Tuple[Hashable, Hashable], float]
NodeId = Hashable


def rtt_to_oneway(rtt: CostMatrix, *, halve: bool = True) -> CostMatrix:
    """Convert a round-trip RTT matrix to the one-way cost this module expects.

    Pipeline hops are one-way, so a round-trip RTT over-counts by ~2×. `halve=True`
    (the default, matching Petals' rtt/2 convention) divides each entry by two. Pass
    halve=False only if your matrix is already one-way.
    """
    f = 0.5 if halve else 1.0
    return {k: v * f for k, v in rtt.items()}


def _edge(cost: CostMatrix, a: NodeId, b: NodeId) -> Optional[float]:
    """One-way cost a→b, or None if that directed edge wasn't measured."""
    v = cost.get((a, b))
    return float(v) if v is not None else None


def _have_all_edges(cost: CostMatrix, candidates: Sequence[NodeId],
                    coordinator: NodeId) -> bool:
    """Every directed edge Held-Karp needs: coord↔each, and each ordered candidate pair."""
    for w in candidates:
        if _edge(cost, coordinator, w) is None or _edge(cost, w, coordinator) is None:
            return False
    for a in candidates:
        for b in candidates:
            if a != b and _edge(cost, a, b) is None:
                return False
    return True


def tour_cost(order: Sequence[NodeId], cost: CostMatrix, coordinator: NodeId) -> float:
    """Total one-way latency of a chain: entry + forward hops + return."""
    if not order:
        return 0.0
    total = _edge(cost, coordinator, order[0])
    for a, b in zip(order, order[1:]):
        total += _edge(cost, a, b)
    total += _edge(cost, order[-1], coordinator)
    return total


def _order_exact(cost: CostMatrix, candidates: List[NodeId],
                 coordinator: NodeId) -> List[NodeId]:
    """Held-Karp: min-cost fixed-depot asymmetric Hamiltonian cycle.

    dp[S][j] = min cost of a path that starts at coord, visits exactly the candidate
    set S, and ends at candidate j ∈ S. Answer closes the cycle back to coord.
    O(2^n · n^2) — used for n ≤ exact_max (≈12 → ~50k states).
    """
    n = len(candidates)
    idx = {w: i for i, w in enumerate(candidates)}
    # dp keyed by (bitmask of visited candidates, last index) -> (cost, prev_index)
    dp: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for j, w in enumerate(candidates):
        dp[(1 << j, j)] = (_edge(cost, coordinator, w), -1)
    for size in range(2, n + 1):
        # iterate masks with `size` bits set
        for mask in range(1 << n):
            if bin(mask).count("1") != size:
                continue
            for j in range(n):
                if not (mask & (1 << j)):
                    continue
                prev_mask = mask ^ (1 << j)
                best = None
                for k in range(n):
                    if not (prev_mask & (1 << k)):
                        continue
                    pc = dp.get((prev_mask, k))
                    if pc is None:
                        continue
                    c = pc[0] + _edge(cost, candidates[k], candidates[j])
                    if best is None or c < best[0]:
                        best = (c, k)
                if best is not None:
                    dp[(mask, j)] = best
    full = (1 << n) - 1
    best_end, best_cost = -1, None
    for j in range(n):
        cell = dp.get((full, j))
        if cell is None:
            continue
        c = cell[0] + _edge(cost, candidates[j], coordinator)
        if best_cost is None or c < best_cost:
            best_cost, best_end = c, j
    # reconstruct
    order_idx: List[int] = []
    mask, j = full, best_end
    while j != -1:
        order_idx.append(j)
        _, prev = dp[(mask, j)]
        mask ^= (1 << j)
        j = prev
    order_idx.reverse()
    return [candidates[i] for i in order_idx]


def _nn_from(cost: CostMatrix, candidates: List[NodeId], seed: NodeId) -> List[NodeId]:
    """Nearest-neighbour chain seeded at `seed`, extending by min forward hop."""
    remaining = [w for w in candidates if w != seed]
    order = [seed]
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda w: _edge(cost, last, w))
        order.append(nxt)
        remaining.remove(nxt)
    return order


def _or_opt(order: List[NodeId], cost: CostMatrix, coordinator: NodeId) -> List[NodeId]:
    """Or-opt local search: relocate runs of length 1..3 to the best position; repeat
    until stable. Asymmetric-safe — it never REVERSES a segment (unlike 2-opt), so the
    recomputed cost is always correct on a directed matrix.
    """
    improved = True
    while improved:
        improved = False
        for seg_len in (1, 2, 3):
            for i in range(len(order) - seg_len + 1):
                seg = order[i:i + seg_len]
                rest = order[:i] + order[i + seg_len:]
                best_order, best_c = order, tour_cost(order, cost, coordinator)
                for j in range(len(rest) + 1):
                    cand = rest[:j] + seg + rest[j:]
                    c = tour_cost(cand, cost, coordinator)
                    if c < best_c - 1e-9:
                        best_order, best_c = cand, c
                if best_order is not order:
                    order = best_order
                    improved = True
    return order


def _order_heuristic(cost: CostMatrix, candidates: List[NodeId],
                     coordinator: NodeId) -> List[NodeId]:
    """Multi-start nearest-neighbour + Or-opt. Asymmetric-safe (no 2-opt reversal).

    A single NN seed can land far from optimal on a directed matrix, so we seed NN from
    EVERY candidate, Or-opt each, and keep the cheapest chain. n·(NN + Or-opt) — fine for
    the n>exact_max range (Held-Karp handles n≤12).
    """
    best_order, best_c = None, None
    for seed in candidates:
        order = _or_opt(_nn_from(cost, candidates, seed), cost, coordinator)
        c = tour_cost(order, cost, coordinator)
        if best_c is None or c < best_c:
            best_order, best_c = order, c
    return best_order


def order_chain(cost: CostMatrix, candidates: Sequence[NodeId], coordinator: NodeId,
                *, exact_max: int = 12) -> List[NodeId]:
    """Order `candidates` to minimise total one-way pipeline latency (see module docstring).

    cost        : directed ONE-WAY latency ms; cost[(a,b)] may differ from cost[(b,a)].
                  For an RTT matrix, pass rtt_to_oneway(rtt) instead.
    candidates  : the already-SELECTED workers to sequence (a list of ids).
    coordinator : the depot id (entry + return anchor).
    exact_max   : use Held-Karp exact for len(candidates) ≤ this, else NN+Or-opt.

    Returns a permutation of `candidates`. FALLBACK: if ≤1 candidate, or any directed edge
    the optimiser needs is missing from `cost`, returns `list(candidates)` UNCHANGED — the
    caller keeps its current order rather than the optimiser inventing latencies.
    """
    cands = list(candidates)
    if len(cands) <= 1:
        return cands
    if not _have_all_edges(cost, cands, coordinator):
        return cands
    if len(cands) <= exact_max:
        return _order_exact(cost, cands, coordinator)
    return _order_heuristic(cost, cands, coordinator)


def brute_force(cost: CostMatrix, candidates: Sequence[NodeId],
                coordinator: NodeId) -> List[NodeId]:
    """Exhaustive optimum — for tests only (n! ; keep n small)."""
    cands = list(candidates)
    best, best_c = cands, None
    for perm in permutations(cands):
        c = tour_cost(perm, cost, coordinator)
        if best_c is None or c < best_c:
            best, best_c = list(perm), c
    return best
