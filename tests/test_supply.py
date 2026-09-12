import pytest
from supply import supply_problems, would_break_invariant


class TestSupplyProblems:
    def test_balanced_ledger_clean(self):
        """A balanced ledger should have no problems."""
        balances = {"alice": 100, "bob": -100}
        assert supply_problems(balances) == []
    
    def test_ledger_summing_to_positive_drift(self):
        """A ledger summing to +5 should report drift +5."""
        balances = {"alice": 100, "bob": -95}
        assert supply_problems(balances) == ["drift 5"]
    
    def test_ledger_summing_to_negative_drift(self):
        """A ledger summing to -5 should report drift -5."""
        balances = {"alice": 100, "bob": -105}
        assert supply_problems(balances) == ["drift -5"]
    
    def test_empty_ledger_clean(self):
        """An empty ledger should be fine (sum is 0)."""
        balances = {}
        assert supply_problems(balances) == []
    
    def test_non_int_balance(self):
        """A non-int balance should be reported."""
        balances = {"alice": "100", "bob": -100}
        problems = supply_problems(balances)
        assert "non-int balance for account 'alice'" in problems
    
    def test_empty_account_id(self):
        """An empty-string account id should be reported."""
        balances = {"": 100, "bob": -100}
        problems = supply_problems(balances)
        assert "empty account id" in problems
    
    def test_malformed_inputs_no_raise(self):
        """Malformed inputs should not raise."""
        # Non-string keys - should not raise, just skip or handle gracefully
        # Since the function signature says dict[str, int], non-string keys should be handled
        # The function should not raise on any input
        try:
            supply_problems({123: 100})
        except Exception:
            pytest.fail("supply_problems raised exception on non-string key")
        
        # None values
        try:
            supply_problems({"alice": None})
        except Exception:
            pytest.fail("supply_problems raised exception on None value")


class TestWouldBreakInvariant:
    def test_valid_settlement_not_breaking(self):
        """A valid settlement should not break the invariant."""
        balances = {"alice": 100, "bob": -100}
        credits = {"alice": 50}
        requester = "bob"
        assert would_break_invariant(balances, credits, requester) == (False, "")
    
    def test_negative_credit_breaking(self):
        """A negative credit should break the invariant."""
        balances = {"alice": 100, "bob": -100}
        credits = {"alice": -50}
        requester = "bob"
        assert would_break_invariant(balances, credits, requester) == (True, "negative credit for account 'alice'")
    
    def test_non_int_credit_breaking(self):
        """A non-int credit should break the invariant."""
        balances = {"alice": 100, "bob": -100}
        credits = {"alice": "50"}
        requester = "bob"
        assert would_break_invariant(balances, credits, requester) == (True, "non-int credit for account 'alice'")
    
    def test_empty_requester_breaking(self):
        """An empty requester should break the invariant."""
        balances = {"alice": 100, "bob": -100}
        credits = {"alice": 50}
        requester = ""
        assert would_break_invariant(balances, credits, requester) == (True, "invalid requester")
    
    def test_requester_also_creditee_netting_out(self):
        """When the requester is also a creditee, it should net out and NOT be flagged."""
        balances = {"alice": 100, "bob": -100}
        credits = {"bob": 50}  # bob is the requester and also gets credit
        requester = "bob"
        assert would_break_invariant(balances, credits, requester) == (False, "")
    
    def test_malformed_inputs_no_raise(self):
        """Malformed inputs should not raise."""
        # Non-string requester - should not raise
        try:
            would_break_invariant({"alice": 100}, {"bob": 50}, 123)
        except Exception:
            pytest.fail("would_break_invariant raised exception on non-string requester")
        
        # Non-dict credits - should not raise
        try:
            would_break_invariant({"alice": 100}, "credits", "bob")
        except Exception:
            pytest.fail("would_break_invariant raised exception on non-dict credits")
        
        # Non-dict balances - should not raise
        try:
            would_break_invariant("balances", {"bob": 50}, "bob")
        except Exception:
            pytest.fail("would_break_invariant raised exception on non-dict balances")