"""Tests for Phase A of the worker hardening sprint (2026-05-20).

Covers the four cheap-wins commit:

  A1 — gRPC max_receive_message_length explicit cap
  A2 — Inference stream idle timeout (_iter_with_idle_timeout helper)
  A3 — _peer_streams LRU cap
  A4 — STHAMBHA_PILLAR_SPKI_SHA256 length validation
  A5 — STHAMBHA_REFUSE_UNSIGNED startup gate
  A6 — math.isfinite guard on recent_rpc_ms (safe_rpc_ms helper)
  A7 — STHAMBHA_REFUSE_UNVERIFIED_FETCH gate (default true in Mode C)
  A8 — concurrent slice cap + reduced subprocess timeout

The pure-function helpers (A4, A5, A6, A7) test directly. The stateful
pieces (A2 wrap, A3 LRU eviction) test the helper / inspect class
internals after constructed scenarios. A1 + A8 are constant-check tests
since the actual integration is `grpc.server(options=...)` and POST
handler — exercising those needs a running daemon.
"""
from __future__ import annotations

import math
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

# scripts/ isn't on sys.path by default in the Nakshatra test runner;
# match the convention from test_nakshatra_auth.py / test_nakshatra_sandbox.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import worker  # noqa: E402


# ── A1: gRPC message-size cap is set as an explicit constant ─────────


def test_a1_grpc_message_cap_constant_is_16mib():
    assert worker.WORKER_GRPC_MAX_MESSAGE_BYTES == 16 * 1024 * 1024


# ── A2: _iter_with_idle_timeout ──────────────────────────────────────


def test_a2_iter_with_idle_timeout_passes_through_fast_items():
    """When items arrive faster than the idle window, every item is yielded
    in order with no timeout."""
    def producer():
        yield 1
        yield 2
        yield 3

    out = list(
        worker._iter_with_idle_timeout(producer(), idle_seconds=1.0)
    )
    assert out == [1, 2, 3]


def test_a2_iter_with_idle_timeout_raises_on_silent_stream():
    """A producer that never yields raises TimeoutError after the deadline."""
    blocking_q: queue.Queue = queue.Queue()

    def producer():
        # Wait for an item that never comes
        while True:
            blocking_q.get()
            yield

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        list(worker._iter_with_idle_timeout(producer(), idle_seconds=0.2))
    elapsed = time.monotonic() - t0
    assert 0.15 < elapsed < 1.0  # rough — must timeout in the window


def test_a2_iter_with_idle_timeout_propagates_producer_error():
    """An exception raised inside the producer surfaces to the caller."""
    class Sentinel(RuntimeError): pass

    def producer():
        yield 1
        raise Sentinel("upstream broke")

    out: list = []
    with pytest.raises(Sentinel):
        for item in worker._iter_with_idle_timeout(producer(), idle_seconds=1.0):
            out.append(item)
    assert out == [1]


# ── A3: _peer_streams LRU cap ────────────────────────────────────────


def test_a3_max_peer_streams_constant():
    assert worker.MAX_PEER_STREAMS == 64
    assert worker.MAX_PEER_STREAMS > 0


def test_a3_peer_streams_uses_ordered_dict():
    """The cache type matters: OrderedDict supports move_to_end + popitem(last=False).
    A regression to plain dict would break the LRU semantics silently."""
    import collections as _c
    # Construct a minimal servicer-shaped object to inspect the type.
    # We do not need a real daemon; the dict init is independent.
    class _DaemonStub:
        def __init__(self):
            self.recent_rpc_ms = _c.deque(maxlen=20)
        def info(self):
            return dict(layer_start=0, layer_end=1, n_embd=4,
                        has_token_embd=True, has_lm_head=True, n_vocab=10)

    daemon = _DaemonStub()
    servicer = worker.WorkerServicer(
        daemon=daemon, mode="middle",
        layer_start=0, layer_end=1, model_id="test",
    )
    assert isinstance(servicer._peer_streams, _c.OrderedDict)
    assert servicer._peer_evictions == 0


# ── A4: validate_spki_hash_env ───────────────────────────────────────


def test_a4_spki_hash_empty_is_none():
    assert worker.validate_spki_hash_env(None) is None
    assert worker.validate_spki_hash_env("") is None
    assert worker.validate_spki_hash_env("   ") is None


def test_a4_spki_hash_valid_64_hex_is_canonicalised():
    h = "AB" * 32  # 64 chars uppercase hex
    assert worker.validate_spki_hash_env(h) == "ab" * 32
    # whitespace stripped
    assert worker.validate_spki_hash_env("  " + h + "\n") == "ab" * 32


def test_a4_spki_hash_short_value_raises():
    with pytest.raises(ValueError, match="64 hex chars"):
        worker.validate_spki_hash_env("abc")


def test_a4_spki_hash_non_hex_raises():
    bad = "z" * 64
    with pytest.raises(ValueError, match="hex"):
        worker.validate_spki_hash_env(bad)


def test_a4_spki_hash_long_value_raises():
    too_long = "a" * 65
    with pytest.raises(ValueError, match="64 hex chars"):
        worker.validate_spki_hash_env(too_long)


# ── A5: should_refuse_unsigned_startup ───────────────────────────────


def test_a5_refuse_unsigned_default_off():
    """When the env is unset (or empty), refuse is never triggered."""
    assert worker.should_refuse_unsigned_startup(
        None, auth_available=False, has_worker_key=False,
        pillar_url="http://pillar:7777",
    ) is False


def test_a5_refuse_unsigned_no_pillar_no_refuse():
    """No pillar URL = no requests to sign = no refuse regardless of env."""
    assert worker.should_refuse_unsigned_startup(
        "true", auth_available=False, has_worker_key=False,
        pillar_url="",
    ) is False


def test_a5_refuse_unsigned_triggers_when_no_auth_module():
    assert worker.should_refuse_unsigned_startup(
        "true", auth_available=False, has_worker_key=False,
        pillar_url="http://pillar:7777",
    ) is True


def test_a5_refuse_unsigned_triggers_when_no_key():
    assert worker.should_refuse_unsigned_startup(
        "true", auth_available=True, has_worker_key=False,
        pillar_url="http://pillar:7777",
    ) is True


def test_a5_refuse_unsigned_satisfied_when_signed():
    """Auth module present + key loaded + pillar set = refuse does not trigger."""
    assert worker.should_refuse_unsigned_startup(
        "true", auth_available=True, has_worker_key=True,
        pillar_url="http://pillar:7777",
    ) is False


def test_a5_refuse_unsigned_truthy_values():
    """1 / yes / true (case-insensitive) all opt in; other strings do not."""
    for truthy in ("true", "TRUE", "True", "1", "yes", "YES"):
        assert worker.should_refuse_unsigned_startup(
            truthy, False, False, "http://pillar:7777"
        ) is True
    for falsy in ("false", "0", "no", "maybe", ""):
        assert worker.should_refuse_unsigned_startup(
            falsy, False, False, "http://pillar:7777"
        ) is False


# ── A6: safe_rpc_ms ──────────────────────────────────────────────────


def test_a6_safe_rpc_ms_finite_passthrough():
    assert worker.safe_rpc_ms(0.0) == 0.0
    assert worker.safe_rpc_ms(12.5) == 12.5
    assert worker.safe_rpc_ms(1e9) == 1e9


def test_a6_safe_rpc_ms_nan_rejected():
    assert worker.safe_rpc_ms(math.nan) is None


def test_a6_safe_rpc_ms_inf_rejected():
    assert worker.safe_rpc_ms(math.inf) is None
    assert worker.safe_rpc_ms(-math.inf) is None


def test_a6_safe_rpc_ms_negative_rejected():
    """Negative ms indicates clock skew backward — drop it."""
    assert worker.safe_rpc_ms(-1.0) is None
    assert worker.safe_rpc_ms(-1e-9) is None


def test_a6_safe_rpc_ms_non_numeric_rejected():
    assert worker.safe_rpc_ms("not a number") is None
    assert worker.safe_rpc_ms(None) is None


# ── A7: should_refuse_unverified_fetch ───────────────────────────────


def test_a7_refuse_unverified_default_on():
    """Default env value is interpreted as 'true'; empty sha → refuse."""
    assert worker.should_refuse_unverified_fetch(None, "") is True
    assert worker.should_refuse_unverified_fetch(None, None) is True


def test_a7_refuse_unverified_allows_valid_sha():
    """With a sha present, never refuse regardless of env."""
    sha = "a" * 64
    assert worker.should_refuse_unverified_fetch(None, sha) is False
    assert worker.should_refuse_unverified_fetch("true", sha) is False
    assert worker.should_refuse_unverified_fetch("false", sha) is False


def test_a7_refuse_unverified_opt_out():
    """Operators can set the env to false to allow unverified fetch."""
    assert worker.should_refuse_unverified_fetch("false", "") is False
    assert worker.should_refuse_unverified_fetch("0", "") is False
    assert worker.should_refuse_unverified_fetch("no", "") is False


def test_a7_refuse_unverified_explicit_true():
    """Explicit true is the same as default."""
    assert worker.should_refuse_unverified_fetch("true", "") is True
    assert worker.should_refuse_unverified_fetch("1", "") is True
    assert worker.should_refuse_unverified_fetch("yes", "") is True


# ── A8: slice constants + running count ──────────────────────────────


def test_a8_max_concurrent_slices_is_one_by_default():
    assert worker.MAX_CONCURRENT_SLICES == 1


def test_a8_slice_subprocess_timeout_reduced():
    """Phase A8 reduced the slice subprocess timeout from 3600s to 1800s
    to bound the worst-case DoS surface per spawn."""
    assert worker.SLICE_SUBPROCESS_TIMEOUT_S == 1800


def test_a8_running_slice_count_initial_zero():
    # Sanity — _SLICE_TASKS starts empty in a fresh module load.
    # Test order independence: if previous tests added entries, clean.
    with worker._SLICE_LOCK:
        worker._SLICE_TASKS.clear()
    assert worker._running_slice_count() == 0


def test_a8_running_slice_count_only_counts_running():
    with worker._SLICE_LOCK:
        worker._SLICE_TASKS.clear()
        worker._SLICE_TASKS["t1"] = {"status": "running"}
        worker._SLICE_TASKS["t2"] = {"status": "completed"}
        worker._SLICE_TASKS["t3"] = {"status": "failed"}
        worker._SLICE_TASKS["t4"] = {"status": "running"}
    try:
        assert worker._running_slice_count() == 2
    finally:
        with worker._SLICE_LOCK:
            worker._SLICE_TASKS.clear()
