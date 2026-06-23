#!/usr/bin/env python3
"""Unit tests for the wire-cost term in placement.py — scoring + wire-aware choose_split. No infra.

Run:  pytest scripts/test_placement_wire.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement as p


def N(name, vram, rate):
    return p.Node(name, vram, rate)


def test_stage_time_s_is_slowest_stage():
    spans = {"a": (0, 8), "b": (8, 16)}
    rate = {"a": 100.0, "b": 50.0}                 # a: 8/100=.08, b: 8/50=.16 → max .16
    assert abs(p.stage_time_s(spans, rate) - 0.16) < 1e-9


def test_wire_cost_s_sums_hop_latency():
    spans = {"a": (0, 8), "b": (8, 16), "c": (16, 24)}
    rtt = {("a", "b"): 10.0, ("b", "c"): 20.0}     # 2 hops → (10+20)/1000 = .03
    assert abs(p.wire_cost_s(spans, rtt) - 0.03) < 1e-9


def test_wire_cost_s_bandwidth_term():
    spans = {"a": (0, 8), "b": (8, 16)}
    rtt = {("a", "b"): 0.0}
    # 1 hop, bytes=1e9 over 1e9 bps → +1.0s
    assert abs(p.wire_cost_s(spans, rtt, bytes_per_token=1e9, bandwidth_bps=1e9) - 1.0) < 1e-6


def test_chain_cost_is_compute_plus_wire():
    spans = {"a": (0, 16), "b": (16, 32)}
    rate = {"a": 100.0, "b": 100.0}
    rtt = {("a", "b"): 170.0}
    # stage 16/100=.16 + wire .17 = .33
    assert abs(p.chain_cost_s(spans, rate, rtt) - 0.33) < 1e-9


def test_choose_split_fewer_hops_at_high_rtt():
    cluster = [N("a", 6, 100), N("b", 6, 100), N("c", 6, 100)]
    rtt = {pair: 170.0 for pair in [("a", "b"), ("b", "c"), ("a", "c")]}
    spans = p.choose_split(cluster, 32, 8.0, rtt)
    assert len(spans) == 2          # 1 hop @170ms beats 2 hops — fewer nodes wins


def test_choose_split_more_nodes_at_low_rtt():
    cluster = [N("a", 6, 100), N("b", 6, 100), N("c", 6, 100)]
    rtt = {pair: 1.0 for pair in [("a", "b"), ("b", "c"), ("a", "c")]}
    spans = p.choose_split(cluster, 32, 8.0, rtt)
    assert len(spans) == 3          # cheap wire → more nodes (lower stage-time) wins


def test_plan_wire_aware_uses_choose_split():
    nodes = [N("a", 6, 100), N("b", 6, 100), N("c", 6, 100)]
    rtt = {pair: 1.0 for pair in [("a", "b"), ("b", "c"), ("a", "c")]}
    pl = p.plan(8.0, 32, nodes, rtt_ms=rtt, bytes_per_token=4096 * 4)   # hidden 4096 × f32
    assert pl.whole_host is None and len(pl.splits) == 3                # low rtt → 3 nodes


def test_plan_default_unchanged_without_bytes():
    nodes = [N("a", 6, 100), N("b", 6, 100)]
    rtt = {("a", "b"): 1.0}
    pl = p.plan(8.0, 32, nodes, rtt_ms=rtt)        # bytes_per_token=0 → plain balanced_spans
    assert pl.whole_host is None and len(pl.splits) == 2


def test_plan_still_routes_whole_when_one_fits():
    nodes = [N("big", 24, 100), N("small", 6, 80)]
    pl = p.plan(8.0, 32, nodes, bytes_per_token=4096 * 4)   # wire params don't break route-whole
    assert pl.whole_host == "big"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
