#!/usr/bin/env python3
"""Unit tests for placement_feed.py — the bridge that feeds placement.py real
measured data. Pure/deterministic; no GPU, no network.

Run:  pytest scripts/test_placement_feed.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement
import placement_feed as pf


# ── capacity_tok_per_s: the core math ────────────────────────────────────────────────

def test_capacity_per_layer_rate():
    # 8 layers in 40ms → 1000*8/40 = 200 layers/s
    assert pf.capacity_tok_per_s(40.0, layers_served=8) == 200.0


def test_capacity_proxy_when_layers_unknown():
    # no layer count → relative speed proxy 1000/ms
    assert pf.capacity_tok_per_s(50.0) == 20.0
    assert pf.capacity_tok_per_s(50.0, layers_served=None) == 20.0


def test_capacity_zero_when_no_data():
    # unknown/invalid latency → 0.0 so placement falls back to VRAM weighting
    assert pf.capacity_tok_per_s(0.0) == 0.0
    assert pf.capacity_tok_per_s(-5.0) == 0.0
    assert pf.capacity_tok_per_s(None) == 0.0          # type: ignore[arg-type]
    assert pf.capacity_tok_per_s("nan-ish") == 0.0     # type: ignore[arg-type]


def test_faster_node_higher_capacity():
    # lower rpc_ms (faster) → strictly higher capacity, same layer count
    fast = pf.capacity_tok_per_s(20.0, layers_served=8)
    slow = pf.capacity_tok_per_s(80.0, layers_served=8)
    assert fast > slow


# ── make_node / build_nodes ──────────────────────────────────────────────────────────

def test_make_node_populates_capacity():
    n = pf.make_node("ijru", vram_gb=12.0, recent_rpc_ms=40.0, layers_served=8)
    assert isinstance(n, placement.Node)
    assert n.name == "ijru" and n.vram_gb == 12.0
    assert n.tok_per_s == 200.0


def test_build_nodes_from_telemetry():
    telem = {
        "hub":  {"vram_gb": 16.0, "recent_rpc_ms": 25.0, "layers_served": 16},
        "ijru": {"vram_gb": 12.0, "recent_rpc_ms": 50.0, "layers_served": 16},
        "new":  {"vram_gb": 24.0},  # freshly joined: no rpc data
    }
    nodes = {n.name: n for n in pf.build_nodes(telem)}
    assert nodes["hub"].tok_per_s > nodes["ijru"].tok_per_s     # hub faster
    assert nodes["new"].tok_per_s == 0.0                        # defers to VRAM
    assert nodes["new"].vram_gb == 24.0


# ── rtt_matrix ───────────────────────────────────────────────────────────────────────

def test_rtt_matrix_symmetric_and_clean():
    m = pf.rtt_matrix([("a", "b", 3.0), ("b", "c", 40.0), ("x", "x", 1.0), ("a", "d", 0.0)])
    assert m[("a", "b")] == 3.0 and m[("b", "a")] == 3.0   # symmetric
    assert m[("b", "c")] == 40.0
    assert ("x", "x") not in m                              # self-pair dropped
    assert ("a", "d") not in m                              # non-positive dropped


# ── live join: peer projection → telemetry ───────────────────────────────────────────

def test_peer_vram_prefers_offered_budget():
    peer = {"node_id": "hub",
            "budget": {"vram_offered_gb": 10.0},
            "hardware": {"gpus": [{"vram_total_gb": 16.0}]}}
    # offered (locked-for-network) wins over physical total
    assert pf._peer_vram_gb(peer) == 10.0


def test_peer_vram_falls_back_to_physical():
    peer = {"node_id": "hub", "hardware": {"gpus": [{"vram_total_gb": 16.0},
                                                    {"vram_total_gb": 8.0}]}}
    assert pf._peer_vram_gb(peer) == 24.0


def test_peer_layers_for_model():
    peer = {"node_id": "ijru", "layer_offerings": [
        {"model_id": "m1", "layer_start": 0, "layer_end": 8},
        {"model_id": "m1", "layer_start": 8, "layer_end": 16},
        {"model_id": "other", "layer_start": 0, "layer_end": 4}]}
    assert pf._peer_layers_for_model(peer, "m1") == 16
    assert pf._peer_layers_for_model(peer, "absent") is None
    assert pf._peer_layers_for_model(peer, None) is None


def test_telemetry_from_peers():
    peers = [
        {"node_id": "hub", "recent_rpc_ms": 25.0,
         "budget": {"vram_offered_gb": 14.0},
         "layer_offerings": [{"model_id": "m1", "layer_start": 0, "layer_end": 16}]},
        {"name": "ijru", "recent_rpc_ms": 50.0,
         "hardware": {"gpus": [{"vram_total_gb": 12.0}]},
         "layer_offerings": [{"model_id": "m1", "layer_start": 0, "layer_end": 16}]},
    ]
    telem = pf.telemetry_from_peers(peers, model_id="m1")
    assert telem["hub"]["vram_gb"] == 14.0
    assert telem["hub"]["layers_served"] == 16
    assert telem["ijru"]["vram_gb"] == 12.0


# ── end-to-end: the planner consumes the bridged data ────────────────────────────────

def test_planner_uses_measured_capacity_for_split():
    """Two nodes, neither fits the whole model → forced split. The FAST node must get
    MORE layers than the slow one — proving measured capacity (not VRAM) drove it.
    Both nodes have equal VRAM so ONLY tok_per_s can break the tie."""
    peers = [
        {"node_id": "fast", "recent_rpc_ms": 20.0, "coord": "10.0.0.1:5540",
         "budget": {"vram_offered_gb": 6.0},
         "layer_offerings": [{"model_id": "m", "layer_start": 0, "layer_end": 16}]},
        {"node_id": "slow", "recent_rpc_ms": 80.0, "coord": "10.0.0.2:5540",
         "budget": {"vram_offered_gb": 6.0},
         "layer_offerings": [{"model_id": "m", "layer_start": 0, "layer_end": 16}]},
    ]
    nodes, rtt, addr = pf.to_placement_inputs(
        peers, model_id="m", rtt_samples=[("fast", "slow", 2.0)])
    # model bigger than either node (6gb offered each, model 8gb) → must split,
    # and the two are in one low-RTT cluster (2ms ≤ 5ms threshold)
    plan = placement.plan(model_gb=8.0, total_layers=16, nodes=nodes, rtt_ms=rtt)
    assert plan.whole_host is None                  # forced split
    assert set(plan.splits) == {"fast", "slow"}
    fast_layers = plan.splits["fast"][1] - plan.splits["fast"][0]
    slow_layers = plan.splits["slow"][1] - plan.splits["slow"][0]
    assert fast_layers > slow_layers                # measured capacity drove the split
    # and it's directly servable
    chain = placement.plan_to_chain(
        plan, model_id="m", model_hash="abc", hidden_size=4096,
        total_layers=16, node_addr=addr, slices_dir="/tmp/slices")
    assert len(chain["workers"]) == 2


def test_planner_routes_whole_when_one_node_fits():
    """A node that fits the whole model → route-don't-split (0 wire crossings),
    and the FASTER fitting node wins."""
    peers = [
        {"node_id": "big-fast", "recent_rpc_ms": 20.0, "coord": "10.0.0.1:5540",
         "budget": {"vram_offered_gb": 24.0},
         "layer_offerings": [{"model_id": "m", "layer_start": 0, "layer_end": 16}]},
        {"node_id": "big-slow", "recent_rpc_ms": 90.0, "coord": "10.0.0.2:5540",
         "budget": {"vram_offered_gb": 24.0},
         "layer_offerings": [{"model_id": "m", "layer_start": 0, "layer_end": 16}]},
    ]
    nodes, rtt, addr = pf.to_placement_inputs(peers, model_id="m")
    plan = placement.plan(model_gb=8.0, total_layers=16, nodes=nodes, rtt_ms=rtt)
    assert plan.whole_host == "big-fast"            # route, and faster wins
    chain = placement.plan_to_chain(
        plan, model_id="m", model_hash="abc", hidden_size=4096,
        total_layers=16, node_addr=addr, slices_dir="/tmp/slices")
    assert "route_to" in chain and chain["route_to"]["node"] == "big-fast"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
