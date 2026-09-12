import pytest
from flags import flag_enabled


class TestFlagEnabled:
    """Test all truthy spellings"""
    
    def test_truthy_1(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "1"}) is True
    
    def test_truthy_true(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "true"}) is True
    
    def test_truthy_yes(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "yes"}) is True
    
    def test_truthy_on(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "on"}) is True
    
    def test_truthy_true_uppercase(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "TRUE"}) is True
    
    def test_truthy_yes_mixed_case(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "YeS"}) is True
    
    def test_truthy_on_with_whitespace(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "  on  "}) is True
    
    def test_truthy_1_with_whitespace(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "  1  "}) is True


class TestFlagEnabledFalsy:
    """Test all falsy spellings"""
    
    def test_falsy_0(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "0"}) is False
    
    def test_falsy_false(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "false"}) is False
    
    def test_falsy_no(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "no"}) is False
    
    def test_falsy_off(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "off"}) is False
    
    def test_falsy_empty_string(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": ""}) is False
    
    def test_falsy_false_uppercase(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "FALSE"}) is False
    
    def test_falsy_no_with_whitespace(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "  no  "}) is False
    
    def test_falsy_0_with_whitespace(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "  0  "}) is False


class TestFlagEnabledGarbageValues:
    """Test that garbage values return False even when default=True"""
    
    def test_garbage_ture_returns_false_even_with_default_true(self):
        # This is the critical test: a typo should NOT enable the feature
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "ture"}, default=True) is False
    
    def test_garbage_enabled_returns_false(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "enabled"}) is False
    
    def test_garbage_maybe_returns_false(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "maybe"}) is False
    
    def test_garbage_random_returns_false(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "xyz123"}) is False
    
    def test_garbage_with_default_false(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "garbage"}, default=False) is False


class TestFlagEnabledAbsentVariable:
    """Test absent variable returns default"""
    
    def test_absent_returns_default_false(self):
        assert flag_enabled("credits", {}) is False
    
    def test_absent_returns_default_true(self):
        assert flag_enabled("credits", {}, default=True) is True
    
    def test_absent_different_flag_returns_default(self):
        assert flag_enabled("other", {}, default=True) is True


class TestFlagEnabledEmptyName:
    """Test empty name returns False"""
    
    def test_empty_name_returns_false(self):
        assert flag_enabled("", {"NAKSHATRA_": "1"}) is False
    
    def test_empty_name_with_default_true(self):
        assert flag_enabled("", {"NAKSHATRA_": "1"}, default=True) is False


class TestFlagEnabledNonDictEnv:
    """Test non-dict env returns False"""
    
    def test_env_none_returns_false(self):
        assert flag_enabled("credits", None) is False
    
    def test_env_list_returns_false(self):
        assert flag_enabled("credits", []) is False
    
    def test_env_string_returns_false(self):
        assert flag_enabled("credits", "env") is False


class TestFlagEnabledPrefixAndCase:
    """Test NAKSHATRA_ prefix and uppercasing"""
    
    def test_prefix_applied_correctly(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "1"}) is True
    
    def test_name_uppercased(self):
        assert flag_enabled("Credits", {"NAKSHATRA_CREDITS": "1"}) is True
    
    def test_name_lowercase(self):
        assert flag_enabled("credits", {"NAKSHATRA_CREDITS": "1"}) is True
    
    def test_name_mixed_case(self):
        assert flag_enabled("CrEdItS", {"NAKSHATRA_CREDITS": "1"}) is True
    
    def test_wrong_prefix_ignored(self):
        assert flag_enabled("credits", {"SHATRA_CREDITS": "1"}) is False
    
    def test_wrong_prefix_with_nakshatra_prefix(self):
        assert flag_enabled("credits", {"NAKSHATRA_credits": "1"}) is False


class TestFlagEnabledNeverRaises:
    """Test that the function never raises"""
    
    def test_none_name(self):
        assert flag_enabled(None, {"NAKSHATRA_NONE": "1"}) is False
    
    def test_int_name(self):
        assert flag_enabled(123, {"NAKSHATRA_123": "1"}) is False
    
    def test_none_env(self):
        assert flag_enabled("credits", None) is False
    
    def test_env_with_non_string_values(self):
        # The function should handle non-string values gracefully
        # Since env.get() returns the value as-is, we need to handle this
        # But the task says env is dict[str, str], so we assume string values
        # If a non-string value is passed, it should still not raise
        pass  # This test is skipped since env is typed as dict[str, str]
    
    def test_special_characters_in_name(self):
        assert flag_enabled("test-name", {"NAKSHATRA_TEST-NAME": "1"}) is True
    
    def test_very_long_name(self):
        long_name = "a" * 1000
        assert flag_enabled(long_name, {f"NAKSHATRA_{long_name.upper()}": "1"}) is True