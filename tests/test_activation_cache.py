"""Tests for the activation replay cache (v1.1 O(t) recovery, upstream half)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from recovery.activation_cache import ActivationReplayCache  # noqa: E402


def test_record_and_replay_in_order():
    c = ActivationReplayCache()
    for i in range(5):
        c.record("s1", i, bytes([i]) * 10)
    replay = c.get_replay("s1", 0)
    assert [k for k, _ in replay] == [0, 1, 2, 3, 4]
    assert replay[3][1] == bytes([3]) * 10


def test_replay_from_step():
    c = ActivationReplayCache()
    for i in range(5):
        c.record("s1", i, b"x")
    assert [k for k, _ in c.get_replay("s1", 2)] == [2, 3, 4]


def test_has_full_prefix_detects_hole():
    c = ActivationReplayCache()
    for i in [0, 1, 3]:          # missing step 2
        c.record("s1", i, b"x")
    assert c.has_full_prefix("s1", 1)
    assert not c.has_full_prefix("s1", 3)   # hole at 2 → unsafe to splice


def test_evicted_session_returns_empty():
    c = ActivationReplayCache()
    assert c.get_replay("nope", 0) == []
    assert not c.has_full_prefix("nope", 0)


def test_per_session_step_bound():
    c = ActivationReplayCache(max_steps_per_session=3)
    for i in range(6):
        c.record("s1", i, b"x")
    keys = [k for k, _ in c.get_replay("s1", 0)]
    assert keys == [3, 4, 5]                 # oldest fell off; bounded length


def test_lru_session_eviction():
    c = ActivationReplayCache(max_sessions=2)
    c.record("a", 0, b"x"); c.record("b", 0, b"x"); c.record("c", 0, b"x")
    # "a" was oldest, evicted when "c" arrived
    assert c.get_replay("a", 0) == []
    assert c.get_replay("b", 0) and c.get_replay("c", 0)


def test_byte_budget_evicts_oldest():
    c = ActivationReplayCache(byte_budget=100)
    c.record("a", 0, b"x" * 60)
    c.record("b", 0, b"x" * 60)              # total 120 > 100 → evict "a"
    assert c.get_replay("a", 0) == []
    assert len(c.get_replay("b", 0)) == 1


def test_idempotent_record_keeps_bytes_correct():
    c = ActivationReplayCache()
    c.record("s", 0, b"xxxx")
    c.record("s", 0, b"yy")                  # re-record same step (shorter)
    assert c.get_replay("s", 0) == [(0, b"yy")]
    assert c.stats()["bytes"] == 2


def test_drop_session_frees_bytes():
    c = ActivationReplayCache()
    c.record("s", 0, b"x" * 50)
    assert c.stats()["bytes"] == 50
    c.drop_session("s")
    assert c.stats()["bytes"] == 0 and c.get_replay("s", 0) == []


def test_thread_safety_smoke():
    import threading
    c = ActivationReplayCache(max_sessions=128)
    def worker(sid):
        for i in range(50):
            c.record(sid, i, b"x" * 16)
    ts = [threading.Thread(target=worker, args=(f"s{i}",)) for i in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
    for i in range(8):
        assert len(c.get_replay(f"s{i}", 0)) == 50
