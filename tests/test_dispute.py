import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from dispute import should_dispute


def test_below_min_checks_all_fail():
    # 2 checks, both fail, min_checks=3 → no dispute
    checks = [
        {"ok": False, "checked_at": 1, "reason": "timeout"},
        {"ok": False, "checked_at": 2, "reason": "timeout"},
    ]
    result = should_dispute(checks, min_checks=3)
    assert result == (False, "insufficient evidence: 2 of 3 checks")


def test_below_min_checks_some_fail():
    # 2 checks, 1 fails, min_checks=3 → no dispute
    checks = [
        {"ok": True, "checked_at": 1, "reason": "ok"},
        {"ok": False, "checked_at": 2, "reason": "timeout"},
    ]
    result = should_dispute(checks, min_checks=3)
    assert result == (False, "insufficient evidence: 2 of 3 checks")


def test_exactly_at_ratio_no_dispute():
    # 4 checks, 2 fail → 0.5, fail_ratio=0.5 → no dispute (boundary favors operator)
    checks = [
        {"ok": True, "checked_at": 1, "reason": "ok"},
        {"ok": False, "checked_at": 2, "reason": "timeout"},
        {"ok": True, "checked_at": 3, "reason": "ok"},
        {"ok": False, "checked_at": 4, "reason": "timeout"},
    ]
    result = should_dispute(checks, min_checks=3, fail_ratio=0.5)
    assert result == (False, "failure rate (0.50) not above threshold (0.50)")


def test_just_above_ratio_disputes():
    # 5 checks, 3 fail → 0.6 > 0.5 → dispute
    checks = [
        {"ok": True, "checked_at": 1, "reason": "ok"},
        {"ok": False, "checked_at": 2, "reason": "timeout"},
        {"ok": False, "checked_at": 3, "reason": "timeout"},
        {"ok": True, "checked_at": 4, "reason": "ok"},
        {"ok": False, "checked_at": 5, "reason": "timeout"},
    ]
    result = should_dispute(checks, min_checks=3, fail_ratio=0.5)
    assert result == (True, "3 of 5 checks failed (0.60 > 0.50)")


def test_all_pass_no_dispute():
    # 5 checks, all pass → 0.0 ≤ 0.5 → no dispute
    checks = [
        {"ok": True, "checked_at": 1, "reason": "ok"},
        {"ok": True, "checked_at": 2, "reason": "ok"},
        {"ok": True, "checked_at": 3, "reason": "ok"},
        {"ok": True, "checked_at": 4, "reason": "ok"},
        {"ok": True, "checked_at": 5, "reason": "ok"},
    ]
    result = should_dispute(checks, min_checks=3)
    assert result == (False, "failure rate (0.00) not above threshold (0.50)")


def test_duplicate_entries_collapse():
    # Same (checked_at, reason) repeated 5 times → counts as 1 check
    # So 1 check, 1 fail → 1.0 > 0.5 but only 1 < 3 → no dispute
    checks = [
        {"ok": False, "checked_at": 1, "reason": "timeout"},
        {"ok": False, "checked_at": 1, "reason": "timeout"},
        {"ok": False, "checked_at": 1, "reason": "timeout"},
        {"ok": False, "checked_at": 1, "reason": "timeout"},
        {"ok": False, "checked_at": 1, "reason": "timeout"},
    ]
    result = should_dispute(checks, min_checks=3)
    assert result == (False, "insufficient evidence: 1 of 3 checks")


def test_duplicate_entries_collapse_to_avoid_dispute():
    # 4 unique checks: 2 fail, 2 pass → 0.5 = 0.5 → no dispute
    # But if duplicates weren't collapsed, would be 5 checks with 3 fails → 0.6 > 0.5 → dispute
    checks = [
        {"ok": True, "checked_at": 1, "reason": "ok"},
        {"ok": False, "checked_at": 2, "reason": "timeout"},
        {"ok": True, "checked_at": 1, "reason": "ok"},  # duplicate
        {"ok": False, "checked_at": 2, "reason": "timeout"},  # duplicate
        {"ok": False, "checked_at": 2, "reason": "timeout"},  # duplicate
    ]
    result = should_dispute(checks, min_checks=3, fail_ratio=0.5)
    assert result == (False, "insufficient evidence: 2 of 3 checks")


def test_empty_list():
    result = should_dispute([])
    assert result == (False, "invalid checks: empty list")


def test_non_list_input():
    result = should_dispute("not a list")
    assert result == (False, "invalid checks: not a list")


def test_list_with_non_dict_entry():
    checks = [
        {"ok": True, "checked_at": 1, "reason": "ok"},
        "not a dict",
    ]
    result = should_dispute(checks)
    assert result == (False, "invalid checks: contains non-dict entry")


def test_custom_min_checks_and_fail_ratio():
    # 3 checks, 2 fail → 2/3 ≈ 0.6667 > 0.6 → dispute
    checks = [
        {"ok": True, "checked_at": 1, "reason": "ok"},
        {"ok": False, "checked_at": 2, "reason": "timeout"},
        {"ok": False, "checked_at": 3, "reason": "timeout"},
    ]
    result = should_dispute(checks, min_checks=2, fail_ratio=0.6)
    assert result == (True, "2 of 3 checks failed (0.67 > 0.60)")


def test_custom_min_checks_below_threshold():
    # 2 checks, 2 fail, min_checks=3 → no dispute
    checks = [
        {"ok": False, "checked_at": 1, "reason": "timeout"},
        {"ok": False, "checked_at": 2, "reason": "timeout"},
    ]
    result = should_dispute(checks, min_checks=3, fail_ratio=0.5)
    assert result == (False, "insufficient evidence: 2 of 3 checks")


def test_custom_min_checks_at_boundary():
    # 3 checks, 2 fail → 2/3 ≈ 0.6667, fail_ratio=2/3 → no dispute (not strictly greater)
    checks = [
        {"ok": True, "checked_at": 1, "reason": "ok"},
        {"ok": False, "checked_at": 2, "reason": "timeout"},
        {"ok": False, "checked_at": 3, "reason": "timeout"},
    ]
    result = should_dispute(checks, min_checks=3, fail_ratio=2/3)
    assert result == (False, "failure rate (0.67) not above threshold (0.67)")


def test_missing_checked_at_or_reason_fields():
    # Checks with missing fields should still be deduplicated and counted
    checks = [
        {"ok": True, "checked_at": 1},
        {"ok": False, "checked_at": 1, "reason": "timeout"},
        {"ok": False, "reason": "timeout"},
    ]
    # First two have same checked_at=1 but different reasons → different keys
    # Last one has no checked_at (None) and reason="timeout" → different key
    # So 3 unique checks, 2 fail → 2/3 ≈ 0.6667 > 0.5 → dispute
    result = should_dispute(checks, min_checks=3, fail_ratio=0.5)
    assert result == (True, "2 of 3 checks failed (0.67 > 0.50)")


def test_missing_ok_field():
    # Check with missing "ok" field → treated as not failed (ok is not False)
    checks = [
        {"checked_at": 1, "reason": "timeout"},
        {"ok": False, "checked_at": 2, "reason": "timeout"},
        {"ok": False, "checked_at": 3, "reason": "timeout"},
    ]
    result = should_dispute(checks, min_checks=3, fail_ratio=0.5)
    assert result == (True, "2 of 3 checks failed (0.67 > 0.50)")