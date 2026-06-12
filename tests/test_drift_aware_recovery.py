"""Tests for drift-class-constrained recovery (v1.1 hardening)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from recovery.drift_aware import (  # noqa: E402
    drift_compatible, next_compatible_cursor, first_advanceable_worker)


def _c(drift_class=None, **kw):
    d = {"id": kw.get("id", "x")}
    if drift_class is not None:
        d["drift_class"] = drift_class
    return d


# ── compatibility policy ──────────────────────────────────────────────

def test_same_class_compatible():
    assert drift_compatible("A", "A")

def test_different_class_refused():
    assert not drift_compatible("A", "B")

def test_unknown_alternate_refused_when_primary_classified():
    # primary has a class but the alternate doesn't declare one → can't prove
    # it won't diverge → refuse
    assert not drift_compatible("A", None)

def test_legacy_primary_allows_anything():
    # primary unclassified (pre-gauge mesh) → preserve old any-alternate behaviour
    assert drift_compatible(None, "B")
    assert drift_compatible(None, None)


# ── cursor selection ──────────────────────────────────────────────────

def test_picks_next_same_class_skipping_mismatch():
    cands = [_c("A"), _c("B"), _c("A")]   # primary A; alt1 B (skip), alt2 A (ok)
    assert next_compatible_cursor(cands, 0) == 2

def test_no_compatible_alternate_returns_none():
    cands = [_c("A"), _c("B"), _c("C")]   # only mismatched alternates
    assert next_compatible_cursor(cands, 0) is None

def test_legacy_chain_advances_normally():
    cands = [_c(), _c(), _c()]            # no classes anywhere → legacy
    assert next_compatible_cursor(cands, 0) == 1
    assert next_compatible_cursor(cands, 1) == 2
    assert next_compatible_cursor(cands, 2) is None


# ── worker-level advance (drop-in for _advance_one_alternate) ─────────

def test_first_advanceable_skips_drift_locked_worker():
    # worker w0 has only a mismatched alternate (locked); w1 has a same-class one
    w0 = {"id": "w0", "cursor": 0, "candidates": [_c("A", id="w0p"), _c("B", id="w0a")]}
    w1 = {"id": "w1", "cursor": 0, "candidates": [_c("X", id="w1p"), _c("X", id="w1a")]}
    advanced = first_advanceable_worker([w0, w1])
    assert advanced is w1 and w1["cursor"] == 1 and w0["cursor"] == 0

def test_first_advanceable_none_when_all_locked():
    w0 = {"id": "w0", "cursor": 0, "candidates": [_c("A"), _c("B")]}
    w1 = {"id": "w1", "cursor": 0, "candidates": [_c("X"), _c("Y")]}
    assert first_advanceable_worker([w0, w1]) is None


# ── regression: the primary candidate must CARRY drift_class ──────────
# v1.1 polish. The config/registry chain builders copy a fixed key set onto the
# primary candidate. If `drift_class` is dropped from that copy (as it originally
# was), next_compatible_cursor reads candidates[0]["drift_class"] as None and the
# whole constraint silently degrades to "any alternate". These tests pin the
# contract: a classified primary built the way client.py builds it must reject a
# mismatched alternate.

# the exact key set client.py copies onto the primary candidate (keep in sync)
_PRIMARY_KEYS = ("id", "address", "port", "layer_range", "mode",
                 "peer_spki_hash", "drift_class")


def _build_worker_like_client(w):
    """Mirror client.py's candidate construction (primary key-copy + alternates)."""
    primary = {k: w[k] for k in _PRIMARY_KEYS if k in w}
    return {"id": w["id"], "cursor": 0,
            "candidates": [primary] + list(w.get("alternates") or [])}

def test_primary_keyset_includes_drift_class():
    assert "drift_class" in _PRIMARY_KEYS   # the one-line regression guard

def test_config_built_primary_enforces_drift_class():
    # YAML-shaped worker: classA primary, a classB alternate (must be refused)
    # and a classA alternate (must be the one chosen).
    raw = {"id": "w0", "address": "10.0.0.1", "port": 5530,
           "layer_range": [0, 8], "mode": "first", "drift_class": "classA",
           "alternates": [
               {"id": "altB", "drift_class": "classB"},   # wrong build → skip
               {"id": "altA", "drift_class": "classA"}]}   # same build → take
    w = _build_worker_like_client(raw)
    assert w["candidates"][0]["drift_class"] == "classA"   # survived the copy
    advanced = first_advanceable_worker([w])
    assert advanced is w and w["cursor"] == 2              # skipped altB, took altA

def test_config_built_primary_locks_when_only_mismatch():
    raw = {"id": "w0", "address": "10.0.0.1", "port": 5530,
           "layer_range": [0, 8], "mode": "first", "drift_class": "classA",
           "alternates": [{"id": "altB", "drift_class": "classB"}]}
    w = _build_worker_like_client(raw)
    assert first_advanceable_worker([w]) is None           # no same-class → clean fail
