#!/usr/bin/env python3
"""Unit tests for bench_placement.py — parser, speedup, run_arm handling. No infra needed.

Run:  pytest scripts/test_bench_placement.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_placement as bp


def test_parse_tok_s_exact():
    assert bp.parse_tok_s("[chain] generated 64 tokens in 8.00s  (8.00 tok/s)") == \
        {"tokens": 64, "elapsed": 8.0, "tok_s": 8.0}


def test_parse_tok_s_embedded():
    out = "boot noise\n[chain] generated 130 tokens in 2.50s  (52.00 tok/s)\ntrailing"
    assert bp.parse_tok_s(out)["tok_s"] == 52.0


def test_parse_tok_s_none():
    assert bp.parse_tok_s("Error: connection refused") is None
    assert bp.parse_tok_s("") is None


def test_speedup():
    assert bp.speedup(40.0, 5.0) == 8.0          # route-whole 8x faster than the WAN split
    assert bp.speedup(5.0, 40.0) == 0.125
    assert bp.speedup(0, 5) is None and bp.speedup(5, 0) is None and bp.speedup(None, 5) is None


def test_run_arm_parses(monkeypatch):
    class _P:
        returncode = 0
        stdout = "[chain] generated 100 tokens in 4.0s  (25.0 tok/s)"
        stderr = ""
    monkeypatch.setattr(bp.subprocess, "run", lambda *a, **k: _P())
    r = bp.run_arm("route", "cfg.yaml", model_path="m.gguf", prompt="p", max_tokens=8, timeout=5)
    assert r["tok_s"] == 25.0 and r["tokens"] == 100


def test_run_arm_no_line_is_error(monkeypatch):
    class _P:
        returncode = 1
        stdout = ""
        stderr = "Error: UNAVAILABLE 127.0.0.1:5540 refused"
    monkeypatch.setattr(bp.subprocess, "run", lambda *a, **k: _P())
    r = bp.run_arm("split", "cfg.yaml", model_path="m.gguf", prompt="p", max_tokens=8, timeout=5)
    assert "error" in r and "tail" in r and "tok_s" not in r


def test_run_arm_timeout(monkeypatch):
    def _raise(*a, **k):
        raise bp.subprocess.TimeoutExpired(cmd="client", timeout=1)
    monkeypatch.setattr(bp.subprocess, "run", _raise)
    assert bp.run_arm("x", "c", model_path="m", prompt="p", max_tokens=8, timeout=1) == {"error": "timeout"}


def test_bench_computes_speedup(monkeypatch):
    # route-whole arm fast, split arm slow → speedup > 1
    seq = {"route_whole": "[chain] generated 100 tokens in 2.0s  (50.0 tok/s)",
           "split": "[chain] generated 100 tokens in 20.0s  (5.0 tok/s)"}
    def fake_run(name, cfg, **kw):
        return bp.parse_tok_s(seq[name])
    monkeypatch.setattr(bp, "run_arm", fake_run)
    res = bp.bench(route_config="r", split_config="s", model_path="m",
                   prompt="p", reps=2, max_tokens=8, timeout=5)
    assert res["arms"]["route_whole"]["median_tok_s"] == 50.0
    assert res["arms"]["split"]["median_tok_s"] == 5.0
    assert res["route_whole_speedup"] == 10.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
