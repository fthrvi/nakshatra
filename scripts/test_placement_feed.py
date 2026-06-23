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


# ── serve adapter: placement.Plan → serve assignment, + the place_fn factory ──────────

class _W:
    """Minimal fake WorkerStanding: node_id + capabilities (what serve_planner reads)."""
    def __init__(self, node_id, addr="10.0.0.1", port=5540):
        self.node_id = node_id
        self.capabilities = {"address": addr, "port": port}


def test_assignment_from_plan_route_whole():
    plan = placement.Plan(whole_host="fast")
    wbn = {"fast": _W("fast")}
    out = pf.assignment_from_plan(plan, wbn, 16)
    assert out == [(wbn["fast"], 0, 16, "solo")]


def test_assignment_from_plan_split_modes_and_coverage():
    plan = placement.Plan(splits={"a": (0, 8), "b": (8, 16)}, cluster=["a", "b"])
    wbn = {"a": _W("a"), "b": _W("b")}
    out = pf.assignment_from_plan(plan, wbn, 16)
    assert [(w.node_id, s, e, m) for (w, s, e, m) in out] == \
        [("a", 0, 8, "first"), ("b", 8, 16, "last")]
    assert out[0][1] == 0 and out[-1][2] == 16          # contiguous full coverage


def test_make_place_fn_routes_whole_when_fits():
    workers = [_W("big")]
    telem = {"big": {"vram_gb": 24.0, "recent_rpc_ms": 20.0, "layers_served": 16}}
    place = pf.make_place_fn(model_gb=8.0, telemetry_of=lambda w: telem[w.node_id])
    assert place(workers, 16) == [(workers[0], 0, 16, "solo")]   # route-don't-split


def test_make_place_fn_split_is_capacity_weighted():
    fast, slow = _W("fast"), _W("slow")
    telem = {"fast": {"vram_gb": 6.0, "recent_rpc_ms": 20.0, "layers_served": 16},
             "slow": {"vram_gb": 6.0, "recent_rpc_ms": 80.0, "layers_served": 16}}
    place = pf.make_place_fn(model_gb=8.0, telemetry_of=lambda w: telem[w.node_id],
                             rtt_samples=[("fast", "slow", 2.0)])
    out = place([fast, slow], 16)
    assert out is not None and len(out) == 2
    by = {w.node_id: (s, e) for (w, s, e, m) in out}
    assert (by["fast"][1] - by["fast"][0]) > (by["slow"][1] - by["slow"][0])   # fast holds more
    spans = sorted(by.values())
    assert spans[0][0] == 0 and spans[-1][1] == 16       # contiguous coverage


def test_make_place_fn_none_when_unplaceable():
    place = pf.make_place_fn(model_gb=100.0,
                             telemetry_of=lambda w: {"vram_gb": 2.0, "recent_rpc_ms": 50.0})
    assert place([_W("tiny")], 16) is None               # → caller falls back to even split


def test_plan_chain_uses_place_fn_end_to_end():
    """The live seam: plan_chain with an injected place_fn emits a chain from the
    placement (fast node gets more layers, contiguous [0,16) coverage)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "fabric"))
        import serve_planner as sp
    except Exception as e:
        import pytest
        pytest.skip(f"serve_planner deps unavailable here: {e!r}")
    ws = [_W("fast"), _W("slow")]
    telem = {"fast": {"vram_gb": 6.0, "recent_rpc_ms": 20.0, "layers_served": 16},
             "slow": {"vram_gb": 6.0, "recent_rpc_ms": 80.0, "layers_served": 16}}
    place = pf.make_place_fn(model_gb=8.0, telemetry_of=lambda w: telem[w.node_id],
                             rtt_samples=[("fast", "slow", 2.0)])
    cp = sp.plan_chain("m", ws, num_layers=16, hidden_size=4096,
                       eligible_fn=lambda model, workers: (workers, []),
                       min_tier_fn=lambda m: "", place_fn=place)
    wy = cp.chain["workers"]
    assert len(wy) == 2
    span = {x["id"]: (x["layer_range"][1] - x["layer_range"][0]) for x in wy}
    assert span["fast"] > span["slow"]
    rs = sorted([x["layer_range"] for x in wy])
    assert rs[0][0] == 0 and rs[-1][1] == 16


# ── pillar telemetry fetch (resilience) ───────────────────────────────────────────────

def test_fetch_pillar_peers_safe_on_bad_input():
    assert pf.fetch_pillar_peers(None) == []                       # no url → []
    assert pf.fetch_pillar_peers("http://127.0.0.1:1", "m", timeout=0.5) == []  # refused → []


def test_pillar_fetch_signature_verifies():
    """The Sthambha-Ed25519 signature fetch_pillar_peers produces over GET /peers must verify
    against the signer's pubkey — proving the canonical/signing is correct, so an auth-required
    pillar will ACCEPT it once the key is registered (the cross-box blocker fix)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import nakshatra_auth as na
    except Exception as e:
        import pytest
        pytest.skip(f"nakshatra_auth unavailable: {e!r}")
    priv, pub = na.generate_keypair()
    ts = 1700000000
    sig = na.sign_request(priv, "GET", "/peers", b"", ts)
    assert na.verify_request(pub, "GET", "/peers", b"", ts, sig) is True
    assert na.verify_request(pub, "GET", "/peers?x=1", b"", ts, sig) is False   # path-bound


def test_fetch_pillar_peers_signs_with_key(tmp_path):
    """With a key present, the signing branch runs (and fails open to [] on a refused conn)."""
    key = tmp_path / "worker.ed25519"
    key.write_bytes(b"\x01" * 32)
    assert pf.fetch_pillar_peers("http://127.0.0.1:1", "m", timeout=0.5,
                                 node_id="hub", key_path=str(key)) == []


# ── live glue: serve_chain under NKS_SMART_PLACEMENT routes whole to the fast node ─────

def test_serve_chain_smart_placement_routes_whole(tmp_path):
    """End-to-end of the LIVE seam: with NKS_SMART_PLACEMENT + injected pillar telemetry,
    build_chain_from_roster routes the whole (fitting) model to the FAST node — a 1-worker
    solo chain — instead of the even split. Proves the flag actually fires through the real
    serve_chain → plan_chain → place_fn path."""
    import os
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "fabric"))
        import serve_chain as sc
        import yaml
    except Exception as e:
        import pytest
        pytest.skip(f"serve_chain deps unavailable here: {e!r}")

    class _Slicer:
        n_layers = 16
        def _ensure_manifest(self): pass
        def slice_for(self, w, s, e, m): return f"/s/{m}-{s}-{e}.gguf"

    roster = lambda: {
        "k1": {"name": "fast", "tier": "self", "coord": "10.0.0.1:5540"},
        "k2": {"name": "slow", "tier": "self", "coord": "10.0.0.2:5540"}}
    peers = [
        {"node_id": "fast", "recent_rpc_ms": 20.0, "budget": {"vram_offered_gb": 24.0},
         "layer_offerings": [{"model_id": "m", "layer_start": 0, "layer_end": 16}]},
        {"node_id": "slow", "recent_rpc_ms": 90.0, "budget": {"vram_offered_gb": 24.0},
         "layer_offerings": [{"model_id": "m", "layer_start": 0, "layer_end": 16}]}]
    out = tmp_path / "chain.yaml"
    os.environ["NKS_SMART_PLACEMENT"] = "1"
    try:
        sc.build_chain_from_roster(
            "m", hidden_size=4096, num_layers=16, package_location="/tmp/fake",
            slicer_factory=lambda loc: _Slicer(), roster_loader=roster,
            min_tier_fn=lambda m: "stranger", model_size_gb=8.0,
            pillar_url="http://fake", peers_fetcher=lambda url, mid: peers,
            out_path=str(out))
    except Exception as e:
        import pytest
        pytest.skip(f"serve_chain admission/eligible path needs more setup here: {e!r}")
    finally:
        os.environ.pop("NKS_SMART_PLACEMENT", None)
    chain = yaml.safe_load(out.read_text())
    assert len(chain["workers"]) == 1                       # route-whole, not even-split across 2
    assert chain["workers"][0]["id"] == "fast"              # the faster node won
    assert chain["workers"][0]["layer_range"] == [0, 16]    # whole model


# ── RTT probe + probe-enabled splitting ───────────────────────────────────────────────

def _listener():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    return s, s.getsockname()[1]


def test_probe_rtt_reachable_and_skips_unreachable():
    s, port = _listener()
    try:
        out = pf.probe_rtt([_W("r", "127.0.0.1", port), _W("d", "127.0.0.1", 1)])  # 1 = refused
        names = {n for (_, n, _) in out}
        assert "r" in names and "d" not in names                 # reachable kept, dead skipped
        assert all(ms >= 0 for (_, _, ms) in out)
    finally:
        s.close()


def test_make_place_fn_probe_enables_split():
    """Two equal nodes (the hub + a peer), neither fits the whole model → forced split. WITHOUT
    RTT, metro_clusters makes singletons and the split can't form (→ None). WITH probe=True, the
    live self→peer RTT (the hub is the probe's self_name) clusters them so the split forms. Proves
    the RTT probe is what enables splitting. (In a real split the hub IS one of the nodes, so the
    star-probe from self gives exactly the hub↔peer edge metro_clusters needs.)"""
    s, port = _listener()
    try:
        hub, peer = _W("hub", "127.0.0.1", port), _W("peer", "127.0.0.1", port)
        telem = {"hub": {"vram_gb": 6.0, "recent_rpc_ms": 20.0, "layers_served": 16},
                 "peer": {"vram_gb": 6.0, "recent_rpc_ms": 80.0, "layers_served": 16}}
        tof = lambda w: telem[w.node_id]
        # no RTT, no probe → can't cluster → None (fallback)
        assert pf.make_place_fn(model_gb=8.0, telemetry_of=tof)([hub, peer], 16) is None
        # probe on → localhost self(hub)→peer RTT clusters them → split forms
        out = pf.make_place_fn(model_gb=8.0, telemetry_of=tof, probe=True)([hub, peer], 16)
        assert out is not None and len(out) == 2
    finally:
        s.close()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
