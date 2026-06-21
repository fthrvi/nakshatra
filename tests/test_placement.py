"""placement — big-model placement planner, tested without GPU/network."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import placement as pl
import pytest


def test_route_whole_when_fits_picks_fastest():
    nodes = [pl.Node("slow", vram_gb=24, tok_per_s=20),
             pl.Node("fast", vram_gb=16, tok_per_s=70),
             pl.Node("tiny", vram_gb=8, tok_per_s=200)]
    p = pl.plan(model_gb=9.0, total_layers=32, nodes=nodes)
    assert p.whole_host == "fast"        # fits (16>9+1) and fastest among fitters
    assert not p.splits


def test_tiny_node_excluded_from_whole():
    nodes = [pl.Node("tiny", vram_gb=8, tok_per_s=200)]
    # 9GB model doesn't fit 8GB → no whole host, and a 1-node "cluster" must hold split
    with pytest.raises(ValueError):
        pl.plan(model_gb=9.0, total_layers=32, nodes=nodes)


def test_metro_clusters_group_by_rtt():
    nodes = [pl.Node("a", 12), pl.Node("b", 12), pl.Node("c", 12)]
    rtt = {("a", "b"): 2.0, ("a", "c"): 50.0, ("b", "c"): 50.0}
    clusters = pl.metro_clusters(nodes, rtt, threshold_ms=5)
    names = sorted(sorted(n.name for n in g) for g in clusters)
    assert ["a", "b"] in names and ["c"] in names


def test_forced_split_within_cluster_covers_all_layers():
    # 40GB model, no single node fits; two 24GB nodes in one cluster hold it split
    nodes = [pl.Node("g0", vram_gb=24, tok_per_s=50),
             pl.Node("g1", vram_gb=24, tok_per_s=50)]
    rtt = {("g0", "g1"): 2.0}
    p = pl.plan(model_gb=40.0, total_layers=80, nodes=nodes, rtt_ms=rtt)
    assert p.whole_host is None
    assert set(p.splits.keys()) == {"g0", "g1"}
    # contiguous + full coverage [0,80)
    spans = sorted(p.splits.values())
    assert spans[0][0] == 0 and spans[-1][1] == 80
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b == c                    # no gap, no overlap


def test_split_respects_vram_cap():
    # one big, one small node: small node must get fewer layers (VRAM-capped)
    nodes = [pl.Node("big", vram_gb=40, tok_per_s=50),
             pl.Node("small", vram_gb=13, tok_per_s=50)]
    rtt = {("big", "small"): 2.0}
    p = pl.plan(model_gb=44.0, total_layers=88, nodes=nodes, rtt_ms=rtt)
    # per-layer 0.5GB; small caps at (13-1)/0.5=24 layers; big takes the rest
    assert p.splits["small"][1] - p.splits["small"][0] <= 24
    total = sum(b - a for a, b in p.splits.values())
    assert total == 88


def test_split_fails_when_cluster_too_small():
    nodes = [pl.Node("a", vram_gb=8), pl.Node("b", vram_gb=8)]
    rtt = {("a", "b"): 2.0}
    with pytest.raises(ValueError):       # 40GB can't fit 16GB aggregate
        pl.plan(model_gb=40.0, total_layers=80, nodes=nodes, rtt_ms=rtt)


def test_prefers_split_in_one_cluster_not_across_wan():
    # 2 nodes near each other (hold the split) + 1 far node; planner uses the cluster
    nodes = [pl.Node("near0", 24, 50), pl.Node("near1", 24, 50), pl.Node("far", 24, 50)]
    rtt = {("near0", "near1"): 2.0, ("near0", "far"): 80.0, ("near1", "far"): 80.0}
    p = pl.plan(model_gb=40.0, total_layers=80, nodes=nodes, rtt_ms=rtt)
    assert set(p.cluster) == {"near0", "near1"}     # the low-RTT cluster
    assert "far" not in p.splits


def test_balanced_spans_faster_node_gets_more_layers():
    # fast node (3x throughput) should hold ~3x the layers — equalizing stage time
    nodes=[pl.Node("fast",vram_gb=40,tok_per_s=90), pl.Node("slow",vram_gb=40,tok_per_s=30)]
    sp=pl.balanced_spans(nodes, total_layers=80, model_gb=40.0)
    assert sp is not None
    fast=sp["fast"][1]-sp["fast"][0]; slow=sp["slow"][1]-sp["slow"][0]
    assert fast+slow==80
    assert fast>slow                                  # faster node carries more
    # stage times should be close (within ~20%): layers/rate
    tf, ts = fast/90, slow/30
    assert abs(tf-ts)/max(tf,ts) < 0.25


def test_balanced_spans_contiguous_full_coverage():
    nodes=[pl.Node("a",24,50), pl.Node("b",24,50), pl.Node("c",24,50)]
    sp=pl.balanced_spans(nodes, total_layers=60, model_gb=30.0)
    spans=sorted(sp.values())
    assert spans[0][0]==0 and spans[-1][1]==60
    for (a,b),(c,d) in zip(spans, spans[1:]): assert b==c


def test_balanced_spans_respects_vram_cap():
    # fast but tiny-VRAM node can't take its "fair" throughput share
    nodes=[pl.Node("fast_small",vram_gb=7,tok_per_s=200), pl.Node("big",vram_gb=60,tok_per_s=40)]
    sp=pl.balanced_spans(nodes, total_layers=80, model_gb=40.0)  # 0.5GB/layer
    assert sp is not None and sum(b-a for a,b in sp.values())==80
    # fast_small caps at (7-1)/0.5 = 12 layers despite high throughput
    assert sp.get("fast_small",(0,0))[1]-sp.get("fast_small",(0,0))[0] <= 12


def test_balanced_spans_too_small_returns_none():
    nodes=[pl.Node("a",8,50), pl.Node("b",8,50)]
    assert pl.balanced_spans(nodes, total_layers=80, model_gb=40.0) is None
