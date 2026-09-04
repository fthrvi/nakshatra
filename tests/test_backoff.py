import pytest
from backoff import delays, should_retry


class TestDelays:
    def test_growth_across_attempts_with_jitter_zero(self):
        """Test exponential growth across attempts 1-5 with jitter=0."""
        for attempt in range(1, 6):
            delay = delays(attempt, base=0.5, cap=60.0, jitter=0.0)
            expected = 0.5 * (2 ** (attempt - 1))
            assert delay == expected, f"Attempt {attempt}: expected {expected}, got {delay}"

    def test_cap_holds_at_large_attempts(self):
        """Test that cap holds at large attempt numbers."""
        for attempt in [10, 20, 30, 40, 50]:
            delay = delays(attempt, base=0.5, cap=60.0, jitter=0.0)
            assert delay == 60.0, f"Attempt {attempt}: expected cap 60.0, got {delay}"

    def test_jitter_zero_is_deterministic(self):
        """Test that jitter=0 is exactly deterministic."""
        delays_list = [delays(5, base=0.5, cap=60.0, jitter=0.0) for _ in range(10)]
        assert len(set(delays_list)) == 1, "jitter=0 should be deterministic"

    def test_jitter_extremes_with_stubbed_rand(self):
        """Test that stubbed rand returning 0.0 and 1.0 produce the two extremes."""
        # Stub rand returning 0.0 (lower bound)
        delay_low = delays(3, base=0.5, cap=60.0, jitter=1.0, rand=lambda: 0.0)
        # Stub rand returning 1.0 (upper bound)
        delay_high = delays(3, base=0.5, cap=60.0, jitter=1.0, rand=lambda: 1.0)

        # For attempt=3, base=0.5: exponential = 0.5 * 2^2 = 2.0
        # With jitter=1.0, factor range is [0.5, 1.5]
        # So low = 2.0 * 0.5 = 1.0, high = 2.0 * 1.5 = 3.0
        assert delay_low == 1.0, f"Expected lower bound 1.0, got {delay_low}"
        assert delay_high == 3.0, f"Expected upper bound 3.0, got {delay_high}"

    def test_jitter_extremes_inside_bounds(self):
        """Test that jittered delays are always in [0, cap]."""
        for attempt in range(1, 10):
            delay_low = delays(attempt, base=0.5, cap=60.0, jitter=1.0, rand=lambda: 0.0)
            delay_high = delays(attempt, base=0.5, cap=60.0, jitter=1.0, rand=lambda: 1.0)
            assert 0.0 <= delay_low <= 60.0, f"Attempt {attempt} low: {delay_low} out of bounds"
            assert 0.0 <= delay_high <= 60.0, f"Attempt {attempt} high: {delay_high} out of bounds"

    def test_attempt_zero_and_negative(self):
        """Test that attempt=0 and negative attempts return 0.0."""
        assert delays(0) == 0.0
        assert delays(-1) == 0.0
        assert delays(-10) == 0.0

    def test_negative_base_cap_jitter_clamped_to_zero(self):
        """Test that negative base/cap/jitter are clamped to 0."""
        # Negative base should clamp to 0, resulting in 0 delay
        assert delays(1, base=-1.0, cap=60.0, jitter=0.0) == 0.0
        # Negative cap should clamp to 0
        assert delays(1, base=0.5, cap=-1.0, jitter=0.0) == 0.0
        # Negative jitter should clamp to 0
        assert delays(1, base=0.5, cap=60.0, jitter=-1.0) == 0.5


class TestShouldRetry:
    def test_retryable_statuses(self):
        """Test each retryable status code."""
        retryable_statuses = [408, 425, 429, 500, 502, 503, 504]
        for status in retryable_statuses:
            assert should_retry(status) is True, f"Status {status} should be retryable"

    def test_non_retryable_statuses(self):
        """Test each non-retryable status code."""
        non_retryable_statuses = [400, 401, 403, 404, 409, 422]
        for status in non_retryable_statuses:
            assert should_retry(status) is False, f"Status {status} should not be retryable"

    def test_retryable_exception_names(self):
        """Test each retryable exception name."""
        retryable_exceptions = [
            "TimeoutError", "ConnectionError", "ConnectionResetError",
            "URLError", "socket.timeout"
        ]
        for exc_name in retryable_exceptions:
            assert should_retry(None, exc_name) is True, f"Exception {exc_name} should be retryable"

    def test_none_and_empty_gives_false(self):
        """Test that status=None and empty exc_name gives False."""
        assert should_retry(None, "") is False
        assert should_retry(None) is False
        assert should_retry(None, "OtherError") is False

    def test_different_rand_streams_produce_different_delays(self):
        """Test that two different rand streams produce different delays for the same attempt."""
        # Two different random streams
        def rand1():
            return 0.3
        def rand2():
            return 0.7

        delay1 = delays(5, base=0.5, cap=60.0, jitter=1.0, rand=rand1)
        delay2 = delays(5, base=0.5, cap=60.0, jitter=1.0, rand=rand2)

        assert delay1 != delay2, "Different rand streams should produce different delays"
        # Both should be within bounds
        assert 0.0 <= delay1 <= 60.0
        assert 0.0 <= delay2 <= 60.0