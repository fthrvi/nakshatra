"""
Tests for creditlimit.py
"""

import pytest
from creditlimit import TIERS, may_consume


def test_tiers_strictly_increasing():
    """TIERS must be strictly increasing across stranger/known/trusted."""
    assert TIERS["stranger"] < TIERS["known"]
    assert TIERS["known"] < TIERS["trusted"]


def test_self_unlimited():
    """'self' tier must be unlimited (None)."""
    assert TIERS["self"] is None


def test_spend_within_limit_allowed():
    """A spend within the limit should be allowed."""
    # stranger limit is 100, so balance=50, cost=100 → new_balance=-50 ≥ -100
    allowed, why = may_consume(50, 100, "stranger")
    assert allowed is True
    assert "tier=stranger" in why
    assert "limit=100" in why


def test_spend_exceeding_limit_denied():
    """A spend that would exceed the limit should be denied."""
    # stranger limit is 100, so balance=50, cost=200 → new_balance=-150 < -100
    allowed, why = may_consume(50, 200, "stranger")
    assert allowed is False
    assert "would exceed limit" in why


def test_boundary_at_limit_allowed():
    """Exactly AT the limit (balance - cost == -limit) should be allowed."""
    # stranger limit is 100, so balance=0, cost=100 → new_balance=-100 ≥ -100
    allowed, why = may_consume(0, 100, "stranger")
    assert allowed is True
    assert "balance=0 → -100" in why


def test_unknown_tier_treated_as_stranger():
    """An unknown tier must be treated as 'stranger' (tightest limit)."""
    # Test with an unknown tier
    allowed_unknown, why_unknown = may_consume(0, 100, "unknown_tier")
    allowed_stranger, why_stranger = may_consume(0, 100, "stranger")
    
    assert allowed_unknown == allowed_stranger
    assert why_unknown == why_stranger


def test_cost_zero_allowed():
    """cost=0 should be allowed."""
    allowed, why = may_consume(100, 0, "stranger")
    assert allowed is True
    assert "cost ≤ 0" in why


def test_negative_cost_allowed():
    """Negative cost should be allowed."""
    allowed, why = may_consume(100, -50, "stranger")
    assert allowed is True
    assert "cost ≤ 0" in why


def test_non_int_balance():
    """Non-int balance should return (False, ...) with error."""
    allowed, why = may_consume("100", 50, "stranger")
    assert allowed is False
    assert "balance must be int" in why


def test_non_int_cost():
    """Non-int cost should return (False, ...) with error."""
    allowed, why = may_consume(100, "50", "stranger")
    assert allowed is False
    assert "cost must be int" in why


def test_positive_balance_spending_through_zero():
    """A positive balance spending down through zero should work if within limit."""
    # stranger limit is 100, so balance=50, cost=150 → new_balance=-100 ≥ -100
    allowed, why = may_consume(50, 150, "stranger")
    assert allowed is True
    assert "balance=50 → -100" in why


def test_never_raises():
    """The function should never raise an exception."""
    # Test various edge cases that might cause issues
    test_cases = [
        (None, 50, "stranger"),
        (100, None, "stranger"),
        (100, 50, None),
        ("abc", 50, "stranger"),
        (100, "abc", "stranger"),
        (100, 50, "unknown"),
        (100, 50, "self"),
        (100, 50, "trusted"),
        (100, 50, "known"),
        (0, 0, "stranger"),
        (-100, 50, "stranger"),
    ]
    
    for args in test_cases:
        try:
            may_consume(*args)
        except Exception:
            pytest.fail(f"may_consume raised on args: {args}")