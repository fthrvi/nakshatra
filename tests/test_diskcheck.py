import pytest
from diskcheck import room_for


def test_comfortable_fit():
    # 100 MiB free, 50 MiB needed, 5% reserve
    # usable = 100 * 0.95 = 95 MiB
    # headroom = 95 - 50 = 45 MiB
    ok, why = room_for(100 * 1024 * 1024, 50 * 1024 * 1024)
    assert ok is True
    assert "45 MiB headroom" in why


def test_exact_fit_at_reserve_boundary():
    # 100 MiB free, 95 MiB needed, 5% reserve
    # usable = 100 * 0.95 = 95 MiB
    # exactly fits
    ok, why = room_for(100 * 1024 * 1024, 95 * 1024 * 1024)
    assert ok is True
    assert "0 MiB headroom" in why


def test_one_byte_short():
    # 100 MiB free, 96 MiB needed, 5% reserve
    # usable = 100 * 0.95 = 95 MiB
    # shortfall = 96 - 95 = 1 MiB
    ok, why = room_for(100 * 1024 * 1024, 96 * 1024 * 1024)
    assert ok is False
    assert "need 1 MiB more" in why


def test_shortfall_reported_in_mib():
    # 100 MiB free, 100 MiB needed, 5% reserve
    # usable = 95 MiB
    # shortfall = 100 - 95 = 5 MiB
    ok, why = room_for(100 * 1024 * 1024, 100 * 1024 * 1024)
    assert ok is False
    assert "need 5 MiB more" in why


def test_reserve_reduces_usable_space():
    # 100 MiB free, 95 MiB needed
    # With 5% reserve: usable = 95 MiB, fits
    ok1, _ = room_for(100 * 1024 * 1024, 95 * 1024 * 1024, reserve=0.05)
    # With 10% reserve: usable = 90 MiB, doesn't fit
    ok2, why2 = room_for(100 * 1024 * 1024, 95 * 1024 * 1024, reserve=0.10)
    assert ok1 is True
    assert ok2 is False
    assert "need 5 MiB more" in why2


def test_reserve_clamping_low():
    # reserve < 0.0 should be clamped to 0.0
    ok, why = room_for(100 * 1024 * 1024, 100 * 1024 * 1024, reserve=-0.1)
    assert ok is True
    assert "0 MiB headroom" in why


def test_reserve_clamping_high():
    # reserve > 0.5 should be clamped to 0.5
    ok, why = room_for(200 * 1024 * 1024, 100 * 1024 * 1024, reserve=0.6)
    assert ok is True
    assert "0 MiB headroom" in why


def test_zero_need():
    ok, why = room_for(100 * 1024 * 1024, 0)
    assert ok is True
    assert "95 MiB headroom" in why


def test_negative_free_bytes():
    ok, why = room_for(-100, 50)
    assert ok is False
    assert "invalid free_bytes" in why


def test_negative_need_bytes():
    ok, why = room_for(100, -50)
    assert ok is False
    assert "invalid need_bytes" in why


def test_non_int_free_bytes():
    ok, why = room_for(100.5, 50)
    assert ok is False
    assert "invalid free_bytes" in why


def test_non_int_need_bytes():
    ok, why = room_for(100, 50.5)
    assert ok is False
    assert "invalid need_bytes" in why


def test_never_raises():
    # Test various edge cases that shouldn't raise
    room_for(0, 0)
    room_for(0, 1)
    room_for(1, 0)
    room_for(float('inf'), 100)
    room_for(100, float('inf'))
    room_for(100, 100, reserve=0.0)
    room_for(100, 100, reserve=0.5)
    room_for(100, 100, reserve=1.0)
    room_for(100, 100, reserve=-1.0)