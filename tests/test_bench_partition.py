"""Unit tests for scripts/bench_partition.py — the greedy-vs-compute
partition benchmark harness. Covers the pure logic (plan build, YAML render,
slice paths, tok/s parse) so a bug doesn't surface only when the cluster is up
and a measurement window is burning."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import bench_partition as bp  # noqa: E402

CFG = {
    "model": {"id": "llama-3.3-70b", "num_layers": 80, "model_bytes": 42 * 10**9,
              "hidden_size": 8192, "wire_dtype": "f16", "full_gguf_path": "/m.gguf"},
    "hub": "fast", "slice_dir": "/tmp/bs", "prompt": "hi", "tokens": 8, "repeats": 1,
    "nodes": [
        {"id": "fast",  "address": "10.0.0.1", "port": 5530, "vram_gb": 16,
         "ram_gb": 64, "rpc_ms": 120, "vendor": "amd", "backend": "rocm"},
        {"id": "slowA", "address": "10.0.0.2", "port": 5531, "vram_gb": 16,
         "ram_gb": 64, "rpc_ms": 480, "vendor": "amd", "backend": "metal"},
        {"id": "slowB", "address": "10.0.0.3", "port": 5532, "vram_gb": 16,
         "ram_gb": 64, "rpc_ms": 460, "vendor": "amd", "backend": "metal"},
    ],
}


def _split(plan):
    return {s.peer_id: s.layer_end - s.layer_start for s in plan.slots}


def test_build_plan_greedy_vs_compute_both_cover_and_differ():
    g, c = bp.build_plan(CFG, "greedy"), bp.build_plan(CFG, "compute")
    assert sum(_split(g).values()) == 80
    assert sum(_split(c).values()) == 80
    assert _split(g) != _split(c)        # the whole point: assignment differs


def test_render_yaml_matches_plan_and_addresses():
    plan = bp.build_plan(CFG, "compute")
    y = bp.render_yaml(CFG, "compute", plan)
    assert y["model"]["id"] == "llama-3.3-70b"
    assert y["model"]["num_blocks"] == 80 and y["model"]["hidden_size"] == 8192
    addr = {n["id"]: (n["address"], n["port"]) for n in CFG["nodes"]}
    covered = 0
    for w, slot in zip(y["workers"], plan.slots):
        assert w["id"] == slot.peer_id
        assert w["layer_range"] == [slot.layer_start, slot.layer_end]
        assert (w["address"], w["port"]) == addr[slot.peer_id]
        assert w["sub_gguf_path"].endswith(".gguf")
        covered += slot.layer_end - slot.layer_start
    assert covered == 80
    for a, b in zip(plan.slots, plan.slots[1:]):   # contiguous
        assert a.layer_end == b.layer_start


def test_slice_path_deterministic_and_labeled():
    p1 = bp.slice_path(CFG, "greedy", 0, 0, 20)
    p2 = bp.slice_path(CFG, "greedy", 0, 0, 20)
    assert p1 == p2 and "greedy" in p1 and "L0-20" in p1


def test_toks_regex_parses_client_rate():
    m = bp._TOKS.search("[chain] generated 3 tokens in 14.20s  (0.21 tok/s)")
    assert m and float(m.group(3)) == 0.21
