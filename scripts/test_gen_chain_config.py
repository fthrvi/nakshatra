#!/usr/bin/env python3
"""Unit tests for gen_chain_config.py — route-whole / even-split YAML shape + contract. No infra.

Run:  pytest scripts/test_gen_chain_config.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_chain_config as g


def test_route_whole_is_one_solo_worker():
    c = g.route_whole("m", 4096, 32, ("hub", "127.0.0.1", 5540), "/sl")
    assert len(c["workers"]) == 1
    w = c["workers"][0]
    assert w["mode"] == "solo" and w["layer_range"] == [0, 32]
    assert w["id"] == "hub" and w["address"] == "127.0.0.1" and w["port"] == 5540
    assert c["model"] == {"id": "m", "hidden_size": 4096, "num_blocks": 32, "wire_dtype": "f32"}


def test_even_split_modes_and_full_coverage():
    c = g.even_split("m", 4096, 32, [("hub", "127.0.0.1", 5540), ("ijru", "127.0.0.1", 5571)], "/sl")
    w = c["workers"]
    assert [x["mode"] for x in w] == ["first", "last"]
    assert w[0]["layer_range"] == [0, 16] and w[1]["layer_range"] == [16, 32]
    # contiguous full coverage [0,32)
    assert w[0]["layer_range"][0] == 0 and w[-1]["layer_range"][1] == 32


def test_even_split_three_has_middle():
    c = g.even_split("m", 4096, 30, [("a", "h", 1), ("b", "h", 2), ("c", "h", 3)], "/sl")
    assert [x["mode"] for x in c["workers"]] == ["first", "middle", "last"]
    assert [x["layer_range"] for x in c["workers"]] == [[0, 10], [10, 20], [20, 30]]


def test_even_split_remainder_to_front():
    c = g.even_split("m", 4096, 33, [("a", "h", 1), ("b", "h", 2)], "/sl")  # 17 + 16
    assert [x["layer_range"] for x in c["workers"]] == [[0, 17], [17, 33]]


def test_build_chain_rejects_gap():
    import pytest
    with pytest.raises(ValueError):
        g.build_chain("m", 4096, 32, "f32",
                      [("a", "h", 1, 0, 8), ("b", "h", 2, 16, 32)], "/sl")  # gap 8..16


def test_build_chain_rejects_incomplete_coverage():
    import pytest
    with pytest.raises(ValueError):
        g.build_chain("m", 4096, 32, "f32", [("a", "h", 1, 0, 16)], "/sl")  # missing 16..32


def test_slice_path_is_content_addressed():
    c = g.route_whole("dsr1", 4096, 32, ("hub", "10.0.0.1", 5540), "/home/x/.nakshatra/slices")
    assert c["workers"][0]["sub_gguf_path"].endswith("dsr1@x-L0-32.gguf")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
