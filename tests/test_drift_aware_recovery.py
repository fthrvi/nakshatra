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
