"""Tests for Phase D of the worker hardening sprint (2026-05-20).

Covers:
  D1 — nakshatra_validation strict-type helpers
  D5+D7 — nakshatra_audit logger: write, rotate, dedup
  D6 — heartbeat backoff formula
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nakshatra_audit as au  # noqa: E402
import nakshatra_validation as v  # noqa: E402
import worker as w  # noqa: E402


# ── D1: as_strict_bool ───────────────────────────────────────────────


def test_d1_strict_bool_passes_literal_true_false():
    assert v.as_strict_bool(True) is True
    assert v.as_strict_bool(False) is False


def test_d1_strict_bool_rejects_truthy_strings():
    """The whole point — `bool('false')` is True, but strict_bool says no."""
    assert v.as_strict_bool("true") is False
    assert v.as_strict_bool("false") is False
    assert v.as_strict_bool("1") is False
    assert v.as_strict_bool("yes") is False


def test_d1_strict_bool_rejects_truthy_ints():
    assert v.as_strict_bool(1) is False
    assert v.as_strict_bool(0) is False


def test_d1_strict_bool_rejects_containers():
    assert v.as_strict_bool([1, 2, 3]) is False
    assert v.as_strict_bool({"key": "value"}) is False
    assert v.as_strict_bool(None) is False


def test_d1_strict_bool_default_overrides():
    assert v.as_strict_bool("any value", default=True) is True
    assert v.as_strict_bool(None, default=True) is True


# ── D1: as_safe_int ──────────────────────────────────────────────────


def test_d1_safe_int_passes_normal():
    assert v.as_safe_int(42) == 42
    assert v.as_safe_int("100") == 100
    assert v.as_safe_int(-5) == -5


def test_d1_safe_int_rejects_booleans():
    """Booleans are technically ints in Python; an attacker-supplied
    `True` shouldn't become `1` in an int field."""
    assert v.as_safe_int(True) == 0
    assert v.as_safe_int(False) == 0


def test_d1_safe_int_rejects_floats():
    assert v.as_safe_int(3.14) == 0
    assert v.as_safe_int(0.0) == 0


def test_d1_safe_int_rejects_garbage():
    assert v.as_safe_int(None) == 0
    assert v.as_safe_int("not a number") == 0
    assert v.as_safe_int([1, 2]) == 0


def test_d1_safe_int_clamps_to_bounds():
    assert v.as_safe_int(1000, lo=0, hi=100) == 100
    assert v.as_safe_int(-50, lo=0, hi=100) == 0
    assert v.as_safe_int(42, lo=0, hi=100) == 42


def test_d1_safe_int_custom_default():
    assert v.as_safe_int(None, default=99) == 99


# ── D1: as_safe_float ────────────────────────────────────────────────


def test_d1_safe_float_passes_finite():
    assert v.as_safe_float(3.14) == 3.14
    assert v.as_safe_float("2.71") == pytest.approx(2.71)
    assert v.as_safe_float(0) == 0.0


def test_d1_safe_float_rejects_nan_inf():
    assert v.as_safe_float(math.nan) == 0.0
    assert v.as_safe_float(math.inf) == 0.0
    assert v.as_safe_float(-math.inf) == 0.0
    assert v.as_safe_float("nan") == 0.0


def test_d1_safe_float_rejects_booleans():
    assert v.as_safe_float(True) == 0.0


def test_d1_safe_float_rejects_negative_when_disallowed():
    assert v.as_safe_float(-1.0, allow_negative=False) == 0.0
    assert v.as_safe_float(-1.0, allow_negative=True) == -1.0


# ── D1: as_str_enum ──────────────────────────────────────────────────


def test_d1_str_enum_accepts_allowed():
    assert v.as_str_enum("alpha", ("alpha", "beta"), default="beta") == "alpha"
    assert v.as_str_enum("beta", ("alpha", "beta"), default="alpha") == "beta"


def test_d1_str_enum_rejects_disallowed():
    assert v.as_str_enum("gamma", ("alpha", "beta"), default="alpha") == "alpha"
    assert v.as_str_enum("ALPHA", ("alpha",), default="alpha") == "alpha"  # case-sensitive


def test_d1_str_enum_rejects_non_strings():
    assert v.as_str_enum(None, ("alpha",), default="alpha") == "alpha"
    assert v.as_str_enum(1, ("1",), default="alpha") == "alpha"


# ── D1: as_bounded_hex ───────────────────────────────────────────────


def test_d1_bounded_hex_accepts_valid():
    h = "ab" * 32  # 64 chars
    assert v.as_bounded_hex(h, 64) == h


def test_d1_bounded_hex_canonicalises_lowercase():
    assert v.as_bounded_hex("AB" * 32, 64) == "ab" * 32


def test_d1_bounded_hex_empty_passes_as_empty():
    assert v.as_bounded_hex("", 64) == ""


def test_d1_bounded_hex_rejects_oversized():
    assert v.as_bounded_hex("a" * 65, 64) == ""


def test_d1_bounded_hex_rejects_non_hex():
    assert v.as_bounded_hex("z" * 64, 64) == ""
    assert v.as_bounded_hex("not-hex", 64) == ""


def test_d1_bounded_hex_rejects_non_string():
    assert v.as_bounded_hex(None, 64) == ""
    assert v.as_bounded_hex(123, 64) == ""


# ── D1: as_bounded_str ───────────────────────────────────────────────


def test_d1_bounded_str_passes_within_cap():
    assert v.as_bounded_str("hello", 100) == "hello"
    assert v.as_bounded_str("", 100) == ""


def test_d1_bounded_str_rejects_oversized():
    assert v.as_bounded_str("x" * 100, 50) == ""


def test_d1_bounded_str_counts_utf8_bytes():
    # 'é' is 2 UTF-8 bytes
    assert v.as_bounded_str("é", 2) == "é"
    assert v.as_bounded_str("é", 1) == ""


def test_d1_bounded_str_rejects_non_string():
    assert v.as_bounded_str(123, 100) == ""
    assert v.as_bounded_str(None, 100) == ""


# ── D1: as_str_list ──────────────────────────────────────────────────


def test_d1_str_list_passes_normal():
    assert v.as_str_list(["a", "b", "c"]) == ["a", "b", "c"]


def test_d1_str_list_filters_non_strings():
    assert v.as_str_list(["a", 1, "b", None, "c"]) == ["a", "b", "c"]


def test_d1_str_list_caps_item_count():
    assert v.as_str_list(["x"] * 200, max_items=3) == ["x", "x", "x"]


def test_d1_str_list_drops_oversized_items():
    items = ["short", "x" * 1000]
    assert v.as_str_list(items, max_item_bytes=10) == ["short"]


def test_d1_str_list_rejects_non_list():
    assert v.as_str_list("not a list") == []
    assert v.as_str_list(None) == []
    assert v.as_str_list({"k": "v"}) == []


# ── D5: AuditLogger writes JSONL ─────────────────────────────────────


def test_d5_audit_logger_writes_one_line_per_event(tmp_path):
    log = au.AuditLogger(tmp_path / "audit.jsonl")
    assert log.log("worker_started", port=5500) is True
    assert log.log("slice_spawned", task_id="abc") is True
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["event"] == "worker_started"
    assert rec1["port"] == 5500
    assert isinstance(rec1["ts"], int)


def test_d5_audit_logger_creates_parent_dir(tmp_path):
    nested = tmp_path / "deep" / "nest" / "audit.jsonl"
    log = au.AuditLogger(nested)
    log.log("test")
    assert nested.is_file()


def test_d5_audit_logger_stats(tmp_path):
    log = au.AuditLogger(tmp_path / "audit.jsonl")
    log.log("test_a")
    log.log("test_b")
    stats = log.stats()
    assert stats["writes"] == 2
    assert stats["rotations"] == 0
    assert stats["dedup_suppressed"] == 0
    assert stats["size_bytes"] > 0


# ── D5: rotation ─────────────────────────────────────────────────────


def test_d5_audit_logger_rotates_on_overflow(tmp_path):
    """File over max_bytes → renamed to .1; fresh file replaces."""
    path = tmp_path / "audit.jsonl"
    # Set the cap tiny so a few writes trigger rotation.
    log = au.AuditLogger(path, max_bytes=200)
    # Write enough to push over the cap
    for i in range(20):
        log.log(f"event_{i}", filler="x" * 50)
    rotated = path.parent / "audit.jsonl.1"
    assert rotated.exists()
    assert log.stats()["rotations"] >= 1


def test_d5_audit_logger_replaces_old_rotation(tmp_path):
    """Existing .1 is overwritten on rotation (we keep one rotated copy)."""
    path = tmp_path / "audit.jsonl"
    log = au.AuditLogger(path, max_bytes=100)
    for i in range(10):
        log.log(f"event_{i}", filler="x" * 50)
    first_rotated_size = (path.parent / "audit.jsonl.1").stat().st_size
    # Trigger another rotation
    for i in range(10, 20):
        log.log(f"event_{i}", filler="y" * 50)
    # Should still be one rotated file (with potentially-different size)
    rotated_files = list(path.parent.glob("audit.jsonl.*"))
    assert len(rotated_files) == 1


# ── D7: dedup for auth_failure events ────────────────────────────────


def test_d7_auth_failure_dedup_within_window(tmp_path):
    log = au.AuditLogger(tmp_path / "audit.jsonl", dedup_window_s=60.0)
    # First write: accepted
    assert log.log("auth_failure_grpc", ip="1.2.3.4", reason="bad sig") is True
    # Repeat with same (event, ip, reason): suppressed
    assert log.log("auth_failure_grpc", ip="1.2.3.4", reason="bad sig") is False
    assert log.log("auth_failure_grpc", ip="1.2.3.4", reason="bad sig") is False
    # Different reason: accepted
    assert log.log("auth_failure_grpc", ip="1.2.3.4", reason="stale ts") is True
    # Different IP: accepted
    assert log.log("auth_failure_grpc", ip="5.6.7.8", reason="bad sig") is True
    stats = log.stats()
    assert stats["dedup_suppressed"] == 2


def test_d7_dedup_window_expires(tmp_path):
    log = au.AuditLogger(tmp_path / "audit.jsonl", dedup_window_s=0.1)
    assert log.log("auth_failure_grpc", ip="1.2.3.4", reason="bad") is True
    assert log.log("auth_failure_grpc", ip="1.2.3.4", reason="bad") is False
    time.sleep(0.2)
    assert log.log("auth_failure_grpc", ip="1.2.3.4", reason="bad") is True


def test_d7_non_auth_events_never_deduplicated(tmp_path):
    log = au.AuditLogger(tmp_path / "audit.jsonl")
    # Repeated identical worker_started events all written
    for _ in range(5):
        assert log.log("worker_started", port=5500) is True
    assert log.stats()["writes"] == 5
    assert log.stats()["dedup_suppressed"] == 0


def test_d7_dedup_cap_evicts_oldest(tmp_path):
    log = au.AuditLogger(
        tmp_path / "audit.jsonl",
        dedup_window_s=300.0, dedup_cap=3,
    )
    # Fill the dedup map
    for i in range(3):
        log.log("auth_failure_grpc", ip=f"ip-{i}", reason="bad")
    # Overflow — oldest evicted
    log.log("auth_failure_grpc", ip="ip-3", reason="bad")
    # ip-0 should no longer be in dedup; re-emitting writes again
    assert log.log("auth_failure_grpc", ip="ip-0", reason="bad") is True


# ── D5: module-level singleton ───────────────────────────────────────


def test_d5_module_audit_no_op_without_init():
    """Reset singleton to None to simulate worker that didn't init."""
    au._AUDIT = None
    assert au.audit("test_event") is False


def test_d5_module_audit_after_init(tmp_path):
    au.init_audit(tmp_path / "audit.jsonl")
    assert au.audit("test_event", foo="bar") is True
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["event"] == "test_event"
    au._AUDIT = None  # cleanup for other tests


# ── D6: heartbeat backoff formula ────────────────────────────────────


class _FixedRng:
    def __init__(self, vals):
        self._vals = list(vals)
    def random(self):
        return self._vals.pop(0)


def test_d6_no_failures_returns_base_with_jitter():
    """consecutive_failures=0 → base * (1 ± jitter_fraction)."""
    # rng.random()=0.5 → jitter coeff = 1.0 (no change)
    rng = _FixedRng([0.5])
    assert w._next_heartbeat_interval(30.0, 0, rng=rng) == pytest.approx(30.0)


def test_d6_exponential_growth_per_failure():
    """Each failure roughly doubles the base before jitter."""
    rng = _FixedRng([0.5, 0.5, 0.5, 0.5])  # zero jitter
    assert w._next_heartbeat_interval(30.0, 1, rng=rng) == pytest.approx(60.0)
    assert w._next_heartbeat_interval(30.0, 2, rng=rng) == pytest.approx(120.0)
    assert w._next_heartbeat_interval(30.0, 3, rng=rng) == pytest.approx(240.0)


def test_d6_capped_at_max_interval():
    """Backoff caps at HEARTBEAT_MAX_INTERVAL_S (600s default)."""
    rng = _FixedRng([0.5])
    # Many failures → would be 30 * 2^20 > MAX, so clamped
    assert w._next_heartbeat_interval(30.0, 20, rng=rng) == pytest.approx(
        w.HEARTBEAT_MAX_INTERVAL_S
    )


def test_d6_jitter_keeps_within_25_percent():
    """rng.random() in [0,1] → jitter coeff in [1-0.25, 1+0.25]."""
    rng_low = _FixedRng([0.0])   # jitter coeff = 1 - 0.25 = 0.75
    rng_high = _FixedRng([1.0])  # jitter coeff = 1 + 0.25 = 1.25
    rng_zero = _FixedRng([0.5])  # jitter coeff = 1.0
    base = 30.0
    assert w._next_heartbeat_interval(base, 0, rng=rng_low) == pytest.approx(0.75 * base)
    assert w._next_heartbeat_interval(base, 0, rng=rng_high) == pytest.approx(1.25 * base)
    assert w._next_heartbeat_interval(base, 0, rng=rng_zero) == pytest.approx(base)


def test_d6_minimum_floor_is_one_second():
    """Even with absurd inputs, the floor is 1s — never spin-loop."""
    rng = _FixedRng([0.0])
    result = w._next_heartbeat_interval(0.1, 0, rng=rng)
    assert result >= 1.0


# ── D4 integration: pillar attestation_nonce_hex bounded-hex parse ───


def test_d4_attestation_nonce_strict_hex_via_validation():
    """The pillar-supplied nonce is now run through as_bounded_hex.
    Non-hex / oversized values collapse to empty (don't echo back)."""
    # Use the helper directly to lock the contract
    assert v.as_bounded_hex("a" * 64, 64) == "a" * 64
    assert v.as_bounded_hex("z" * 64, 64) == ""        # non-hex
    assert v.as_bounded_hex("a" * 65, 64) == ""        # oversized
    assert v.as_bounded_hex(None, 64) == ""            # non-string
    assert v.as_bounded_hex(12345, 64) == ""           # int — non-string
