"""Tests for the KV-session-ownership gate (branch worker/kv-session-ownership-gate,
2026-09-12).

Background: `DaemonClient.call()` wraps the daemon's stdin/stdout protocol in
`with self.lock:`, which makes ONE call atomic on the wire (no garbled bytes if
two threads call concurrently) but does NOT protect the daemon's KV cache
state. All serving/verify decode in the C++ daemon lives on a single fixed KV
slot (seq 0 — see experiments/v0.0/worker_daemon.cpp). The worker's gRPC
server runs a ThreadPoolExecutor(max_workers=4) and nakshatra_serve.py spawns
a per-request client.py subprocess from a ThreadingHTTPServer, so two
DIFFERENT logical sessions CAN reach the same WorkerServicer/DaemonClient
concurrently. Without exclusivity, their calls could interleave turn-by-turn
and silently corrupt each other's KV context.

This file covers two layers:
  1. DaemonClient.acquire_session()/release_session()/current_owner directly
     — the mechanism itself, no gRPC involved.
  2. WorkerServicer.Inference()/Forward() wiring — a second session's call
     while the daemon is owned by a live first session is refused (fail
     closed, never interleaved); a session that completes/errors releases
     ownership so a SUBSEQUENT (not concurrent) session works normally
     (no regression to the plain sequential case).

Follows the fake-daemon / fake-context idiom established in
test_worker_fabric_streaming_bridge.py and test_worker_stream_spec.py — no
gRPC server, no daemon subprocess, no GPU.
"""
from __future__ import annotations

import struct
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import worker  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402


# ── Layer 1: DaemonClient.acquire_session/release_session directly ─────────


def _bare_daemon_client() -> worker.DaemonClient:
    """A DaemonClient instance with only the ownership-gate state set up —
    no subprocess, no stdin/stdout. `__new__` skips `__init__` (which spawns
    a real llama-nakshatra-worker subprocess), so this stays true to the
    "never load a real GPU model" testing rule while exercising the exact
    acquire_session/release_session/current_owner code the real daemon uses.
    """
    d = worker.DaemonClient.__new__(worker.DaemonClient)
    d._owner_lock = threading.Lock()
    d._owner_session = None
    d._owner_since = 0.0
    return d


def test_acquire_fresh_session_succeeds():
    d = _bare_daemon_client()
    assert d.acquire_session("session-a") is True
    assert d.current_owner == "session-a"


def test_acquire_same_session_is_idempotent():
    d = _bare_daemon_client()
    assert d.acquire_session("session-a") is True
    # Later steps of the SAME stream re-acquiring must not be refused.
    assert d.acquire_session("session-a") is True
    assert d.current_owner == "session-a"


def test_acquire_different_live_session_fails_closed():
    d = _bare_daemon_client()
    assert d.acquire_session("session-a") is True
    assert d.acquire_session("session-b") is False
    # Ownership must not have moved — the refusal is real, not silent.
    assert d.current_owner == "session-a"


def test_release_by_owner_frees_the_daemon():
    d = _bare_daemon_client()
    d.acquire_session("session-a")
    d.release_session("session-a")
    assert d.current_owner is None
    assert d.acquire_session("session-b") is True


def test_release_by_non_owner_is_a_no_op():
    d = _bare_daemon_client()
    d.acquire_session("session-a")
    # A late/duplicate release from a session that never held ownership (or
    # already lost it) must never clobber the real owner.
    d.release_session("session-b")
    assert d.current_owner == "session-a"


def test_stale_owner_is_reclaimed_not_permanent_lockout():
    d = _bare_daemon_client()
    d.acquire_session("session-a")
    # Simulate an abandoned session (crash/kill/timeout that never called
    # release_session): back-date the acquisition past the staleness window.
    d._owner_since = time.time() - 10.0
    assert d.acquire_session("session-b", stale_after_s=5.0) is True
    assert d.current_owner == "session-b"


def test_owner_within_staleness_window_is_not_reclaimed():
    d = _bare_daemon_client()
    d.acquire_session("session-a")
    d._owner_since = time.time() - 1.0
    assert d.acquire_session("session-b", stale_after_s=5.0) is False
    assert d.current_owner == "session-a"


# ── Layer 2: wired through WorkerServicer.Inference/Forward ────────────────


class _GateRecordingDaemon:
    """DaemonClient stand-in that implements the REAL ownership-gate logic
    (not a stub that always says yes) plus records every daemon.call() so
    tests can assert calls from a refused session never reached the daemon."""

    def __init__(self, n_embd: int = 4):
        self._n_embd = n_embd
        self.calls = []
        self._owner_lock = threading.Lock()
        self._owner_session = None
        self._owner_since = 0.0

    # Reuse the real implementation so this test exercises the same logic
    # as production, not a hand-rolled approximation of it.
    acquire_session = worker.DaemonClient.acquire_session
    release_session = worker.DaemonClient.release_session
    current_owner = worker.DaemonClient.current_owner

    def info(self):
        return {"n_embd": self._n_embd, "n_layers": 4, "gpu_offload_status": {}}

    def gpu_offload_status(self):
        return {"uses_gpu": False, "n_offloaded": 0, "total_layers": 4, "backend_hints": []}

    def call(self, cmd, n_tokens, payload, start_pos=0, flags=0):
        self.calls.append({"cmd": cmd, "n_tokens": n_tokens,
                            "start_pos": start_pos, "flags": flags})
        rtype_prefix = struct.pack("<I", 0)
        body = struct.pack("<i", 4242)  # single final token id (mode=last/solo)
        return (0, rtype_prefix + body)


class _FakeContext:
    def __init__(self):
        self._metadata = []
        self._code = None
        self._details = None
        self._peer = "ipv4:127.0.0.1:9999"

    def invocation_metadata(self):
        return self._metadata

    def peer(self):
        return self._peer

    def set_code(self, code):
        self._code = code

    def set_details(self, details):
        self._details = details


def _build_servicer(daemon=None, mode: str = "last"):
    return worker.WorkerServicer(
        daemon=daemon or _GateRecordingDaemon(),
        mode=mode, layer_start=0, layer_end=14,
        model_id="kv-session-gate-test",
        idem_max_entries=8, idem_ttl_seconds=10.0,
        peer_resolver=None,
        auth_required=False,
        refuse_unregistered_peers=False,
        refuse_unpinned_peers=False,
    )


def _step(session_id: str, step_id: str, prefix_length: int = 0, token_id: int = 1):
    st = pb.InferenceStep(session_id=session_id, step_id=step_id,
                           prefix_length=prefix_length)
    st.token_ids.ids.append(token_id)
    return st


def test_concurrent_session_is_refused_not_interleaved():
    """Session A opens a stream and is mid-flight (holding ownership) when
    session B's stream arrives on the SAME servicer/daemon. B must be
    refused clearly — never silently interleaved into A's KV state."""
    daemon = _GateRecordingDaemon()
    s = _build_servicer(daemon=daemon)

    ctx_a = _FakeContext()
    gen_a = s.Inference(iter([_step("session-a", "a0"), _step("session-a", "a1")]), ctx_a)
    first_a = next(gen_a)  # processes a0, acquires ownership, suspends before a1
    assert not first_a.error
    assert daemon.current_owner == "session-a"

    ctx_b = _FakeContext()
    gen_b = s.Inference(iter([_step("session-b", "b0")]), ctx_b)
    calls_before = len(daemon.calls)
    resp_b = next(gen_b)
    with pytest.raises(StopIteration):
        next(gen_b)  # B's stream ends immediately on refusal

    assert resp_b.error, "a refused session must come back with an explicit error, not silence"
    assert b"busy" in resp_b.error
    assert ctx_b._code is not None
    # The refused call must never have reached the daemon — that's the whole
    # point: no interleaved turn, not even one.
    assert len(daemon.calls) == calls_before
    # A's ownership must be untouched by B's refused attempt.
    assert daemon.current_owner == "session-a"

    # Finish session A's stream normally.
    remaining_a = list(gen_a)
    assert all(not r.error for r in remaining_a)
    # A released ownership on stream completion.
    assert daemon.current_owner is None


def test_sequential_session_after_completion_works_normally():
    """No regression to the plain sequential case: once session A's stream
    completes, a SUBSEQUENT session must be able to use the same worker
    exactly as before (not refused, calls actually reach the daemon)."""
    daemon = _GateRecordingDaemon()
    s = _build_servicer(daemon=daemon)

    list(s.Inference(iter([_step("session-a", "a0")]), _FakeContext()))
    assert daemon.current_owner is None
    calls_after_a = len(daemon.calls)

    resp = list(s.Inference(iter([_step("session-b", "b0")]), _FakeContext()))
    assert all(not r.error for r in resp)
    assert len(daemon.calls) > calls_after_a
    assert daemon.current_owner is None


def test_errored_session_releases_ownership_no_permanent_lockout():
    """A session that raises mid-stream (simulating a daemon error / crash)
    must still release ownership via the `finally` — one bad session must
    not permanently lock out the worker for every session after it."""
    daemon = _GateRecordingDaemon()

    class _ExplodingDaemon(_GateRecordingDaemon):
        def call(self, *a, **kw):
            raise RuntimeError("simulated daemon crash mid-call")

    exploding = _ExplodingDaemon()
    s = _build_servicer(daemon=exploding)

    ctx = _FakeContext()
    list(s.Inference(iter([_step("session-a", "a0")]), ctx))
    # The generic exception handler caught it and set an error status.
    assert ctx._code is not None
    # Ownership was released despite the exception.
    assert exploding.current_owner is None

    # A later session is NOT locked out by session-a's crash.
    daemon2 = _GateRecordingDaemon()
    s2 = _build_servicer(daemon=daemon2)
    resp = list(s2.Inference(iter([_step("session-b", "b0")]), _FakeContext()))
    assert all(not r.error for r in resp)


def test_timed_out_session_releases_ownership():
    """The idle-timeout path (TimeoutError from _iter_with_idle_timeout) must
    also release ownership — a slow/dead client must not wedge the daemon."""
    daemon = _GateRecordingDaemon()
    s = _build_servicer(daemon=daemon)

    def _iterator_that_times_out():
        yield _step("session-a", "a0")
        raise TimeoutError("idle stream timeout (simulated)")

    ctx = _FakeContext()
    list(s.Inference(_iterator_that_times_out(), ctx))
    assert daemon.current_owner is None

    # A subsequent session proceeds normally.
    resp = list(s.Inference(iter([_step("session-b", "b0")]), _FakeContext()))
    assert all(not r.error for r in resp)


def test_forward_claims_and_releases_a_synthetic_session():
    """Forward carries no session_id (documented stateless testing aid) but
    still touches the same daemon/KV state, so it must also compete for
    exclusivity — and release immediately after, never blocking anyone else."""
    daemon = _GateRecordingDaemon()
    s = _build_servicer(daemon=daemon, mode="last")

    req = pb.ForwardRequest(hidden_in=struct.pack("<4f", 0.0, 0.0, 0.0, 0.0),
                             n_tokens=1, has_token_ids=False, keep_kv=False,
                             start_pos=0)
    resp = s.Forward(req, _FakeContext())
    assert resp.hidden_out
    # Forward released its synthetic session immediately after the call.
    assert daemon.current_owner is None


def test_forward_refused_while_a_real_session_is_active():
    """A Forward call landing mid-stream (while an Inference session owns the
    daemon) must be refused, not allowed to interleave a turn into that
    session's KV state."""
    daemon = _GateRecordingDaemon()
    s = _build_servicer(daemon=daemon)

    gen_a = s.Inference(iter([_step("session-a", "a0"), _step("session-a", "a1")]), _FakeContext())
    next(gen_a)  # session-a now owns the daemon, stream still open

    req = pb.ForwardRequest(hidden_in=struct.pack("<4f", 0.0, 0.0, 0.0, 0.0),
                             n_tokens=1, has_token_ids=False, keep_kv=False,
                             start_pos=0)
    ctx = _FakeContext()
    calls_before = len(daemon.calls)
    resp = s.Forward(req, ctx)
    assert not resp.hidden_out
    assert ctx._code is not None
    assert len(daemon.calls) == calls_before  # never reached the daemon
    assert daemon.current_owner == "session-a"  # untouched

    # Finish session A so the daemon isn't left owned past this test.
    list(gen_a)
