import pytest
from capability import serving_capacity


class TestServingCapacity:
    def test_holds_full_model(self):
        # Card with enough capacity for all layers
        result = serving_capacity(
            vram_bytes=32_000_000_000,
            model_bytes_per_layer=1_000_000_000,
            n_layers=10,
            ctx_tokens=1000,
            kv_bytes_per_token=128
        )
        assert result["max_layers"] == 10
        assert result["holds_full_model"] is True
        assert result["reason"] == "OK"

    def test_partial_split(self):
        # Card that holds partial layers
        result = serving_capacity(
            vram_bytes=15_000_000_000,
            model_bytes_per_layer=2_000_000_000,
            n_layers=10,
            ctx_tokens=1000,
            kv_bytes_per_token=128
        )
        usable = int(15_000_000_000 * 0.90)  # 13,500,000,000
        kv = 1000 * 128  # 128,000
        remaining = usable - kv  # 13,499,872,000
        expected_layers = min(remaining // 2_000_000_000, 10)  # 6
        assert result["max_layers"] == expected_layers
        assert result["holds_full_model"] is False
        assert result["reason"] == "OK"

    def test_kv_exceeds_usable_memory(self):
        # KV cache exceeds usable memory
        result = serving_capacity(
            vram_bytes=1_000_000_000,  # 1 GB
            model_bytes_per_layer=100_000_000,
            n_layers=5,
            ctx_tokens=1_000_000,  # 1M tokens
            kv_bytes_per_token=1024  # 1 KB per token
        )
        assert result["max_layers"] == 0
        assert result["holds_full_model"] is False
        assert "KV cache" in result["reason"]
        assert "needs more than this card has" in result["reason"]

    def test_max_layers_capped_at_n_layers(self):
        # Ensure max_layers never exceeds n_layers
        result = serving_capacity(
            vram_bytes=100_000_000_000,  # 100 GB
            model_bytes_per_layer=100_000_000,
            n_layers=8,
            ctx_tokens=1000,
            kv_bytes_per_token=128
        )
        assert result["max_layers"] <= 8
        assert result["max_layers"] == 8  # Should be exactly 8 since we have plenty of memory

    def test_headroom_reduces_capacity(self):
        # Compare with and without headroom
        result_high_headroom = serving_capacity(
            vram_bytes=10_000_000_000,
            model_bytes_per_layer=1_000_000_000,
            n_layers=10,
            ctx_tokens=1000,
            kv_bytes_per_token=128,
            headroom=0.50  # High headroom
        )
        result_low_headroom = serving_capacity(
            vram_bytes=10_000_000_000,
            model_bytes_per_layer=1_000_000_000,
            n_layers=10,
            ctx_tokens=1000,
            kv_bytes_per_token=128,
            headroom=0.05  # Low headroom
        )
        assert result_high_headroom["max_layers"] < result_low_headroom["max_layers"]

    def test_headroom_clamping_at_upper_bound(self):
        # Headroom > 0.9 should be clamped to 0.9
        result = serving_capacity(
            vram_bytes=10_000_000_000,
            model_bytes_per_layer=1_000_000_000,
            n_layers=10,
            ctx_tokens=1000,
            kv_bytes_per_token=128,
            headroom=1.5  # Should be clamped to 0.9
        )
        usable = int(10_000_000_000 * (1 - 0.9))  # 1,000,000,000
        assert result["usable_bytes"] == usable

    def test_headroom_clamping_at_lower_bound(self):
        # Headroom < 0.0 should be clamped to 0.0
        result = serving_capacity(
            vram_bytes=10_000_000_000,
            model_bytes_per_layer=1_000_000_000,
            n_layers=10,
            ctx_tokens=1000,
            kv_bytes_per_token=128,
            headroom=-0.5  # Should be clamped to 0.0
        )
        usable = int(10_000_000_000 * (1 - 0.0))  # 10,000,000_000
        assert result["usable_bytes"] == usable

    def test_zero_vram(self):
        # Zero vram_bytes should return error
        result = serving_capacity(
            vram_bytes=0,
            model_bytes_per_layer=1_000_000_000,
            n_layers=10,
            ctx_tokens=1000,
            kv_bytes_per_token=128
        )
        assert result["max_layers"] == 0
        assert "must be a positive integer" in result["reason"]

    def test_negative_vram(self):
        # Negative vram_bytes should return error
        result = serving_capacity(
            vram_bytes=-1000,
            model_bytes_per_layer=1_000_000_000,
            n_layers=10,
            ctx_tokens=1000,
            kv_bytes_per_token=128
        )
        assert result["max_layers"] == 0
        assert "must be a positive integer" in result["reason"]

    def test_non_int_argument(self):
        # Non-integer argument should return error
        result = serving_capacity(
            vram_bytes=10_000_000_000,
            model_bytes_per_layer=1.5,  # Non-int
            n_layers=10,
            ctx_tokens=1000,
            kv_bytes_per_token=128
        )
        assert result["max_layers"] == 0
        assert "must be a positive integer" in result["reason"]

    def test_realistic_case(self):
        # The realistic case from the task description
        result = serving_capacity(
            vram_bytes=25769803776,      # 24 GiB
            model_bytes_per_layer=419430400,  # 400 MiB
            n_layers=32,
            ctx_tokens=98304,
            kv_bytes_per_token=128,
            headroom=0.10
        )
        # Calculated values from the task
        usable = int(25769803776 * 0.90)  # 23192823398
        kv = 98304 * 128  # 12582912
        remaining = usable - kv  # 23180240486
        raw = remaining // 419430400  # 55
        expected_max_layers = min(raw, 32)  # 32
        
        assert result["usable_bytes"] == usable
        assert result["kv_bytes"] == kv
        assert result["max_layers"] == expected_max_layers
        assert result["holds_full_model"] is True
        assert result["reason"] == "OK"