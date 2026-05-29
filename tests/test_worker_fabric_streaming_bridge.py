"""Tests for the streaming Inference RPC fabric bridge (2026-05-29).

The fabric bridge for the unary Forward RPC lives in
test_fabric_backend.py (Phase F.1 / first_worker_round_trip). This
file covers the streaming-side bridge added to WorkerServicer.Inference
on the same OQ8 shape: for mode=first + transport=fabric, the local
decode's hidden_state ships via fabric and the chain's final token
comes back via the FEEDBACK wormhole, then yielded on the gRPC
stream as token_ids.

Run with ``pytest --noconftest`` per the established hardening test
pattern (though tests/conftest.py is import-tolerant since 2026-05-26
so plain pytest also works).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import worker  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402


# ── Stubs ───────────────────────────────────────────────────────────


class _StubDaemon:
    """Smallest viable DaemonClient stand-in. Returns a deterministic
    payload for any decode call so tests can assert on what the bridge
    actually saw."""
    def __init__(self, n_embd: int = 4):
        self._n_embd = n_embd

    def info(self):
        return {"n_embd": self._n_embd, "n_layers": 4,
                "gpu_offload_status": {}}

    def gpu_offload_status(self):
        return {"uses_gpu": False, "n_offloaded": 0,
                "total_layers": 4, "backend_hints": []}

    def call(self, cmd, n_tokens, payload, start_pos=0, flags=0):
        # Return rtype prefix + n_tokens * n_embd * 4 bytes of fp32.
        # Each token-slot of the response carries the cmd byte so
        # the test can verify CMD_TOKEN_DECODE vs CMD_EMBD_DECODE.
        rtype_prefix = struct.pack("<I", 0)
        body = bytes([cmd]) * (n_tokens * self._n_embd * 4)
        return (0, rtype_prefix + body)


class _FakeContext:
    """Minimal gRPC ServicerContext. Records auth metadata + any
    status/details the handler sets; otherwise does nothing."""
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


def _build_first_servicer():
    """Build a WorkerServicer in mode=first with auth disabled.
    auth_required=False means the bridge test doesn't need to
    fabricate signed envelopes — the Inference handler's
    _check_grpc_auth becomes a no-op."""
    return worker.WorkerServicer(
        daemon=_StubDaemon(n_embd=4),
        mode="first", layer_start=0, layer_end=14,
        model_id="bridge-test",
        idem_max_entries=8, idem_ttl_seconds=10.0,
        peer_resolver=None,
        auth_required=False,
        refuse_unregistered_peers=False,
        refuse_unpinned_peers=False,
    )


# ── 1. Bridge fires on the first worker when wired ──────────────────


def test_streaming_inference_first_worker_bridge_returns_token():
    """The OQ8 path on the streaming side: a single InferenceStep
    arrives with token_ids (a prompt); the first worker decodes
    locally; the bridge is invoked with the decoded hidden_state;
    the bridge returns a token; the gRPC stream yields an
    InferenceStep with token_ids carrying that token."""
    s = _build_first_servicer()

    bridge_calls = []

    def fake_bridge(payload, *, step_id, layer_idx):
        bridge_calls.append({
            "payload": payload, "step_id": step_id,
            "layer_idx": layer_idx,
        })
        # Return a deterministic token id (0xCAFE = 51966) as 4 bytes.
        return struct.pack("<i", 0xCAFE)

    s.fabric_first_worker_bridge = fake_bridge

    req = pb.InferenceStep(
        session_id="sess-1",
        step_id="step-0",
        prefix_length=0,
    )
    req.token_ids.ids.extend([1, 2, 3, 4])

    ctx = _FakeContext()
    responses = list(s.Inference(iter([req]), ctx))
    assert len(responses) == 1
    out = responses[0]
    assert out.session_id == "sess-1"
    assert out.step_id == "step-0"
    assert out.prefix_length == 4              # 0 + 4 tokens in prompt
    assert list(out.token_ids.ids) == [0xCAFE]
    # Bridge saw the local-decode output (n_embd=4, 4 tokens, fp32 →
    # 64 bytes per token-decode of CMD_TOKEN_DECODE=1).
    assert len(bridge_calls) == 1
    assert bridge_calls[0]["step_id"] == "step-0"
    assert bridge_calls[0]["layer_idx"] == 14   # servicer.layer_end


def test_streaming_inference_bridge_timeout_yields_error_step():
    """When the bridge returns None (chain stall / timeout), the
    Inference handler yields an error step rather than wedging the
    stream or returning a bogus token. Same shape as the size-
    mismatch error path elsewhere in Inference."""
    s = _build_first_servicer()
    s.fabric_first_worker_bridge = lambda payload, **kw: None

    req = pb.InferenceStep(
        session_id="sess-1", step_id="step-0", prefix_length=0,
    )
    req.token_ids.ids.extend([1])

    responses = list(s.Inference(iter([req]), _FakeContext()))
    assert len(responses) == 1
    out = responses[0]
    assert out.error == b"fabric chain timed out before FEEDBACK arrived"
    assert out.session_id == "sess-1"
    assert out.step_id == "step-0"


# ── 2. Bridge is bypassed when not applicable ───────────────────────


def test_streaming_inference_no_bridge_keeps_existing_behavior():
    """gRPC-mode first worker (bridge not wired) — Inference yields
    hidden_state as before. Bridge is purely opt-in via the
    fabric_first_worker_bridge attribute; absence == legacy path."""
    s = _build_first_servicer()
    # No s.fabric_first_worker_bridge assignment — stays None.

    req = pb.InferenceStep(
        session_id="sess-1", step_id="step-0", prefix_length=0,
    )
    req.token_ids.ids.extend([1, 2])

    responses = list(s.Inference(iter([req]), _FakeContext()))
    assert len(responses) == 1
    out = responses[0]
    # First worker without bridge → returns hidden_state in the
    # legacy chain-walk shape (client.py walks the chain itself).
    assert out.HasField("hidden_state")
    assert out.hidden_state.n_tokens == 2


def test_streaming_inference_middle_worker_does_not_bridge():
    """A middle worker with the bridge somehow wired (shouldn't
    happen — main() only wires for mode=first — but defense in
    depth) MUST NOT bridge. Middle workers receive hidden_state
    over fabric directly, not via gRPC, so the bridge would be the
    wrong thing here."""
    s = worker.WorkerServicer(
        daemon=_StubDaemon(n_embd=4),
        mode="middle", layer_start=14, layer_end=20,
        model_id="bridge-test",
        idem_max_entries=8, idem_ttl_seconds=10.0,
        peer_resolver=None,
        auth_required=False,
        refuse_unregistered_peers=False,
        refuse_unpinned_peers=False,
    )
    # Force-wire the bridge to a function that would fail the test
    # if called — middle workers must never invoke it on the gRPC
    # Inference path.
    s.fabric_first_worker_bridge = lambda *a, **kw: pytest.fail(
        "middle worker incorrectly invoked the first-worker bridge")

    # Build a hidden_state step (mid-chain input shape — 4 tokens × 4
    # n_embd × 4 bytes f32 = 64 bytes).
    req = pb.InferenceStep(
        session_id="sess-1", step_id="step-0", prefix_length=0,
    )
    req.hidden_state.raw = b"\x00" * 64
    req.hidden_state.n_tokens = 4
    req.hidden_state.batch = 1

    responses = list(s.Inference(iter([req]), _FakeContext()))
    assert len(responses) == 1
    out = responses[0]
    assert out.HasField("hidden_state")          # not token_ids


# ── 3. Idempotency cache integration ────────────────────────────────


def test_bridge_response_cached_for_replay():
    """The streaming Inference handler caches its yield for
    (session_id, step_id) so a re-delivered duplicate step returns
    the cached response without re-invoking the daemon OR the
    bridge. Important for the recovery_replay capability — without
    this, a fabric round-trip would re-run on every replay."""
    s = _build_first_servicer()
    call_count = [0]

    def counting_bridge(payload, *, step_id, layer_idx):
        call_count[0] += 1
        return struct.pack("<i", 0xBEEF)

    s.fabric_first_worker_bridge = counting_bridge

    req = pb.InferenceStep(
        session_id="sess-replay", step_id="step-7", prefix_length=0,
    )
    req.token_ids.ids.extend([1])

    # First Inference call — bridge invoked once.
    list(s.Inference(iter([req]), _FakeContext()))
    assert call_count[0] == 1

    # Second Inference call with the SAME (session_id, step_id) —
    # cache hit, bridge MUST NOT run again.
    responses = list(s.Inference(iter([req]), _FakeContext()))
    assert call_count[0] == 1
    assert list(responses[0].token_ids.ids) == [0xBEEF]
