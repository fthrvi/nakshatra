"""Tests for the drift-class conformance gauge (docs/cross-machine-validation.md §4).

Pure tests on the fingerprint contract — no inference needed. The
llama-cpp producer is exercised only where llama_cpp is installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from discovery.drift_gauge import (  # noqa: E402
    fingerprint_from_token_ids, classes_of, DriftFingerprint,
    GAUGE_VERSION, CANONICAL_PROMPT, GAUGE_TOKENS)


def test_same_sequence_same_fingerprint():
    a = fingerprint_from_token_ids("m", [12366, 13, 578, 469])
    b = fingerprint_from_token_ids("m", [12366, 13, 578, 469])
    assert a.fingerprint == b.fingerprint and a.same_class(b)


def test_one_token_flip_changes_class():
    # the cross-machine case: agree on token 1, diverge at token 4
    prithvi = fingerprint_from_token_ids("m", [12366, 13, 578, 469, 3168])   # Eiffel
    mac4 = fingerprint_from_token_ids("m", [12366, 13, 578, 6864, 315])      # capital
    assert prithvi.fingerprint != mac4.fingerprint
    assert not prithvi.same_class(mac4)


def test_different_model_never_same_class():
    a = fingerprint_from_token_ids("model-a", [1, 2, 3])
    b = fingerprint_from_token_ids("model-b", [1, 2, 3])  # identical ids, diff model
    assert not a.same_class(b)


def test_gauge_version_guard():
    a = fingerprint_from_token_ids("m", [1, 2, 3], gauge_version=1)
    b = fingerprint_from_token_ids("m", [1, 2, 3], gauge_version=2)
    assert not a.same_class(b)


def test_classes_of_groups_by_fingerprint():
    fps = [
        fingerprint_from_token_ids("m", [1, 2, 3]),   # class X
        fingerprint_from_token_ids("m", [1, 2, 3]),   # class X
        fingerprint_from_token_ids("m", [1, 2, 9]),   # class Y
    ]
    classes = classes_of(fps)
    assert len(classes) == 2
    assert sorted(len(v) for v in classes.values()) == [1, 2]


def test_fingerprint_is_stable_string():
    # a frozen golden value so an accidental change to the hashing is caught
    fp = fingerprint_from_token_ids("accept-1b", [12366, 13, 578], gauge_version=1)
    assert fp.fingerprint == __import__("hashlib").sha256(
        b"1\naccept-1b\n12366,13,578").hexdigest()


def test_metadata_fields():
    fp = fingerprint_from_token_ids("accept-1b", list(range(GAUGE_TOKENS)))
    assert fp.model_id == "accept-1b" and fp.gauge_version == GAUGE_VERSION
    assert fp.n_tokens == GAUGE_TOKENS and len(fp.fingerprint) == 64
    assert "accept-1b@gauge1:" in fp.short()
