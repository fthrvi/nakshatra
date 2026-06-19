"""
Unit tests for RTT-aware pipeline ordering (scripts/fabric/topology_order.py).

Pure — no cluster, no GPU. Proves the optimiser minimises one-way chain latency and,
critically, the two traps the co-pilot flagged: directionality (asymmetric costs must be
honoured, not silently symmetrised) and the one-way-vs-RTT halving.
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "fabric"))
from topology_order import (  # noqa: E402
    order_chain, tour_cost, brute_force, rtt_to_oneway,
    _order_exact, _order_heuristic,
)

COORD = "coord"


def _rand_cost(nodes, rng, lo=1.0, hi=200.0):
    """Random directed (asymmetric) one-way matrix over nodes + COORD."""
    alln = [COORD] + list(nodes)
    return {(a, b): rng.uniform(lo, hi) for a in alln for b in alln if a != b}


# ---------------------------------------------------------------- obvious cases

def test_symmetric_obvious_optimum():
    # A and B near coord and each other; C far. Cheapest chain visits the cheap hops.
    cost = {
        ("coord", "A"): 1, ("A", "coord"): 1,
        ("coord", "B"): 2, ("B", "coord"): 2,
        ("coord", "C"): 50, ("C", "coord"): 50,
        ("A", "B"): 1, ("B", "A"): 1,
        ("A", "C"): 50, ("C", "A"): 50,
        ("B", "C"): 1, ("C", "B"): 1,
    }
    order = order_chain(cost, ["A", "B", "C"], COORD)
    # [A,B,C] and [C,B,A] are BOTH cost-53 optima → compare cost, not the exact perm.
    assert tour_cost(order, cost, COORD) == pytest.approx(
        tour_cost(brute_force(cost, ["A", "B", "C"], COORD), cost, COORD))
    # entry A(1) + A→B(1) + B→C(1) + return C(50) = 53 is the min
    assert tour_cost(order, cost, COORD) == pytest.approx(53.0)


def test_asymmetric_directionality_is_honoured():
    # The KEY regression: cost is directional. A→B is cheap but B→A is expensive.
    # A symmetric-assuming optimiser (or 2-opt segment reversal) would mis-cost the
    # reversed order and could pick B-before-A. The correct optimum respects direction.
    cost = {
        ("coord", "A"): 1, ("A", "coord"): 100,
        ("coord", "B"): 100, ("B", "coord"): 1,
        ("A", "B"): 1, ("B", "A"): 100,
    }
    order = order_chain(cost, ["A", "B"], COORD)
    # coord→A(1) + A→B(1) + B→coord(1) = 3  ≪  coord→B(100)+B→A(100)+A→coord(100)=300
    assert order == ["A", "B"]
    assert tour_cost(order, cost, COORD) == pytest.approx(3.0)
    assert tour_cost(["B", "A"], cost, COORD) == pytest.approx(300.0)


def test_degenerate_sizes():
    cost = {("coord", "A"): 5, ("A", "coord"): 5}
    assert order_chain(cost, [], COORD) == []
    assert order_chain(cost, ["A"], COORD) == ["A"]
    # n==2 still ordered
    cost2 = _rand_cost(["A", "B"], random.Random(1))
    o = order_chain(cost2, ["A", "B"], COORD)
    assert sorted(o) == ["A", "B"]
    assert o == brute_force(cost2, ["A", "B"], COORD)


# ---------------------------------------------------------------- exact optimality

@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_held_karp_matches_brute_force(n):
    rng = random.Random(100 + n)
    nodes = [f"w{i}" for i in range(n)]
    for _ in range(20):
        cost = _rand_cost(nodes, rng)
        got = _order_exact(cost, nodes, COORD)
        assert sorted(got) == sorted(nodes)               # valid permutation
        opt = brute_force(cost, nodes, COORD)
        # cost-optimal (ties allowed → compare costs, not the exact permutation)
        assert tour_cost(got, cost, COORD) == pytest.approx(
            tour_cost(opt, cost, COORD))


def test_order_chain_dispatches_exact_under_threshold():
    rng = random.Random(7)
    nodes = [f"w{i}" for i in range(6)]
    cost = _rand_cost(nodes, rng)
    assert tour_cost(order_chain(cost, nodes, COORD), cost, COORD) == pytest.approx(
        tour_cost(brute_force(cost, nodes, COORD), cost, COORD))


# ---------------------------------------------------------------- heuristic (Or-opt)

@pytest.mark.parametrize("n", [8, 9, 10])
def test_heuristic_valid_and_no_better_than_exact(n):
    # Heuristic must be a valid permutation and never beat the exact optimum; and on
    # these sizes (where exact is still tractable) stay within a sane bound of it.
    rng = random.Random(500 + n)
    nodes = [f"w{i}" for i in range(n)]
    for _ in range(10):
        cost = _rand_cost(nodes, rng)
        heur = _order_heuristic(cost, nodes, COORD)
        assert sorted(heur) == sorted(nodes)
        hc = tour_cost(heur, cost, COORD)
        ec = tour_cost(_order_exact(cost, nodes, COORD), cost, COORD)
        assert hc >= ec - 1e-9                # exact is optimal; heuristic can't beat it
        # Fully-random (non-metric) directed matrices are an adversarial worst case;
        # multi-start NN+Or-opt stays within a generous factor. Real RTT matrices are
        # near-metric and do far better. The contract under test is: valid + ≤ this bound.
        assert hc <= ec * 1.50 + 1e-6


def test_heuristic_path_used_above_threshold():
    # With exact_max small, a 5-node call takes the heuristic path and still returns valid.
    rng = random.Random(9)
    nodes = [f"w{i}" for i in range(5)]
    cost = _rand_cost(nodes, rng)
    o = order_chain(cost, nodes, COORD, exact_max=3)
    assert sorted(o) == sorted(nodes)


# ---------------------------------------------------------------- fallbacks & helpers

def test_missing_edge_falls_back_to_input_order():
    # Drop one directed edge the optimiser needs → return candidates UNCHANGED.
    nodes = ["A", "B", "C"]
    cost = _rand_cost(nodes, random.Random(3))
    del cost[("A", "C")]
    assert order_chain(cost, nodes, COORD) == nodes        # unchanged, no invention
    # also when a coord edge is missing
    cost2 = _rand_cost(nodes, random.Random(4))
    del cost2[("B", "coord")]
    assert order_chain(cost2, nodes, COORD) == nodes


def test_rtt_to_oneway_halves():
    rtt = {("coord", "A"): 100.0, ("A", "coord"): 80.0}
    ow = rtt_to_oneway(rtt)
    assert ow[("coord", "A")] == 50.0 and ow[("A", "coord")] == 40.0
    assert rtt_to_oneway(rtt, halve=False) == rtt
    # ordering on the halved matrix is consistent with ordering on RTT (scale-invariant),
    # but the absolute tour cost is half — the thing that matters for not double-counting.
    nodes = ["A", "B"]
    rtt_full = _scaled(_rand_cost(nodes, random.Random(11)), 1.0)
    ow_full = rtt_to_oneway(rtt_full)
    o_rtt = order_chain(rtt_full, nodes, COORD)
    o_ow = order_chain(ow_full, nodes, COORD)
    assert o_rtt == o_ow
    assert tour_cost(o_ow, ow_full, COORD) == pytest.approx(
        tour_cost(o_rtt, rtt_full, COORD) / 2)


def _scaled(cost, f):
    return {k: v * f for k, v in cost.items()}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
