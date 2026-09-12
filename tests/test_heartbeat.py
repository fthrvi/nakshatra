import pytest
from heartbeat import node_state


class TestNodeState:
    def test_fresh_node_live(self):
        """Fresh node (just seen) should be live"""
        result = node_state(100.0, 100.0)
        assert result == ("live", "within miss_stale threshold")

    def test_exactly_at_stale_boundary_live(self):
        """Exactly at the stale boundary should still be live (inclusive)"""
        interval = 30.0
        miss_stale = 2
        now = 160.0
        last_seen = now - interval * miss_stale  # 160 - 60 = 100
        result = node_state(last_seen, now)
        assert result == ("live", "within miss_stale threshold")

    def test_one_second_past_stale_boundary_stale(self):
        """One second past stale boundary should be stale"""
        interval = 30.0
        miss_stale = 2
        now = 161.0
        last_seen = now - interval * miss_stale - 1  # 161 - 60 - 1 = 100
        result = node_state(last_seen, now)
        assert result == ("stale", "within miss_dead threshold")

    def test_exactly_at_dead_boundary_stale(self):
        """Exactly at the dead boundary should be stale"""
        interval = 30.0
        miss_dead = 10
        now = 430.0
        last_seen = now - interval * miss_dead  # 430 - 300 = 130
        result = node_state(last_seen, now)
        assert result == ("stale", "within miss_dead threshold")

    def test_past_dead_boundary_dead(self):
        """Past dead boundary should be dead"""
        interval = 30.0
        miss_dead = 10
        now = 431.0
        last_seen = now - interval * miss_dead - 1  # 431 - 300 - 1 = 130
        result = node_state(last_seen, now)
        assert result == ("dead", "exceeded miss_dead threshold")

    def test_timestamp_1s_in_future_live(self):
        """Timestamp 1s in the future (small skew) should be live"""
        result = node_state(101.0, 100.0)
        assert result == ("live", "small clock skew")

    def test_far_future_unknown_skew(self):
        """Far future timestamp should be unknown with clock skew reason"""
        result = node_state(200.0, 100.0)  # 100s in future, more than 30s interval
        assert result == ("unknown", "clock skew")

    def test_non_numeric_last_seen(self):
        """Non-numeric last_seen should return unknown"""
        result = node_state("abc", 100.0)
        assert result == ("unknown", "non-numeric last_seen")

    def test_non_numeric_now(self):
        """Non-numeric now should return unknown"""
        result = node_state(100.0, "abc")
        assert result == ("unknown", "non-numeric now")

    def test_boolean_last_seen(self):
        """Boolean last_seen should return unknown"""
        result = node_state(True, 100.0)
        assert result == ("unknown", "non-numeric last_seen")

    def test_boolean_now(self):
        """Boolean now should return unknown"""
        result = node_state(100.0, False)
        assert result == ("unknown", "non-numeric now")

    def test_custom_interval(self):
        """Custom interval should work correctly"""
        # With interval=10, miss_stale=2, miss_dead=5
        # live: age <= 20, stale: 20 < age <= 50, dead: age > 50
        result = node_state(70.0, 100.0, interval=10.0, miss_stale=2, miss_dead=5)
        assert result == ("stale", "within miss_dead threshold")

    def test_miss_stale_ge_miss_dead_clamping(self):
        """miss_stale >= miss_dead should clamp and mention in reason"""
        result = node_state(100.0, 160.0, miss_stale=5, miss_dead=3)
        assert result[0] == "live"
        assert "miss_stale >= miss_dead" in result[1]
        assert "clamped" in result[1]

    def test_miss_stale_ge_miss_dead_stale_case(self):
        """miss_stale >= miss_dead with age in stale range"""
        # With miss_stale=3, miss_dead=2, clamped to miss_dead=4
        # age = 100, interval=30, so age = 100, interval*miss_stale=90, interval*clamped=120
        result = node_state(60.0, 160.0, interval=30.0, miss_stale=3, miss_dead=2)
        assert result[0] == "stale"
        assert "miss_stale >= miss_dead" in result[1]
        assert "clamped" in result[1]

    def test_miss_stale_ge_miss_dead_dead_case(self):
        """miss_stale >= miss_dead with age in dead range"""
        # With miss_stale=2, miss_dead=2, clamped to miss_dead=3
        # age = 100, interval=30, so age = 100, interval*clamped=90
        result = node_state(50.0, 160.0, interval=30.0, miss_stale=2, miss_dead=2)
        assert result[0] == "dead"
        assert "miss_stale >= miss_dead" in result[1]
        assert "clamped" in result[1]