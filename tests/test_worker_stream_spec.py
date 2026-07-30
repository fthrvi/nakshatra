"""Tests for speculative decode ON THE STREAM — the worker.py side (2026-07-29).

docs/findings/cuda-chain-51-tok-s.md measured spec/async pipelining riding the unary
Forward RPC (12.20 tok/s) losing badly to plain streaming (51-53 tok/s), because unary
pays per-call RPC setup on every one of K+1 verify positions. The fix: let the persistent
Inference stream carry a verify traversal too — InferenceStep.all_logits (proto tag 17,
the streaming twin of ForwardRequest.all_logits) plus the SAME prefix_length-as-start_pos
KV-rewind primitive the streaming handler already used for plain decode.

This file covers the WorkerServicer.Inference side:
  - Info() advertises the new "stream_spec" capability.
  - all_logits=False is BYTE-IDENTICAL to today (single token on mode=last).
  - all_logits=True on mode=last returns one argmax per input position.
  - a middle/non-last worker is unaffected by all_logits either way (still hidden_state).
  - the 0x2 flag bit reaches daemon.call() exactly when step.all_logits is set, ORed
    with the existing keep_kv (0x1) bit — never clobbering it.
  - prefix_length rewind: a later step's SMALLER prefix_length reaches the daemon as a
    smaller start_pos (the KV-rewind primitive the client-side accept() loop relies on).

Follows the fake-daemon / fake-context idiom established in
test_worker_fabric_streaming_bridge.py — no gRPC server, no daemon subprocess, no GPU.
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


class _RecordingDaemon:
    """Fake DaemonClient that records every call's (cmd, n_tokens, start_pos, flags)
    and returns a deterministic payload shaped by whether the all_logits bit (0x2)
    is set — mirrors the real daemon's contract (single final token normally, one
    argmax per position when all_logits is requested)."""

    def __init__(self, n_embd: int = 4):
        self._n_embd = n_embd
        self.calls = []   # list of dicts: cmd, n_tokens, start_pos, flags

    def info(self):
        return {"n_embd": self._n_embd, "n_layers": 4, "gpu_offload_status": {}}

    def gpu_offload_status(self):
        return {"uses_gpu": False, "n_offloaded": 0, "total_layers": 4, "backend_hints": []}

    def call(self, cmd, n_tokens, payload, start_pos=0, flags=0):
        self.calls.append({"cmd": cmd, "n_tokens": n_tokens,
                            "start_pos": start_pos, "flags": flags})
        rtype_prefix = struct.pack("<I", 0)
        if flags & 0x2:
            # all_logits: one int32 argmax per position, deterministic so tests can
            # assert on it (position i -> 1000 + i).
            body = struct.pack(f"<{n_tokens}i", *[1000 + i for i in range(n_tokens)])
        else:
            # legacy: a single final token id.
            body = struct.pack("<i", 4242)
        return (0, rtype_prefix + body)


class _FakeContext:
    """Minimal gRPC ServicerContext — same shape as
    test_worker_fabric_streaming_bridge.py's _FakeContext."""

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


def _build_servicer(mode: str, daemon=None, n_embd: int = 4):
    return worker.WorkerServicer(
        daemon=daemon or _RecordingDaemon(n_embd=n_embd),
        mode=mode, layer_start=0, layer_end=14,
        model_id="stream-spec-test",
        idem_max_entries=8, idem_ttl_seconds=10.0,
        peer_resolver=None,
        auth_required=False,
        refuse_unregistered_peers=False,
        refuse_unpinned_peers=False,
    )


# ── Capability advertisement ─────────────────────────────────────────


def test_info_advertises_stream_spec_capability():
    s = _build_servicer("last")
    resp = s.Info(pb.InfoRequest(), _FakeContext())
    assert "stream_spec" in resp.protocol_capabilities


# ── all_logits=False: byte-identical to today ────────────────────────


def test_all_logits_false_last_worker_returns_single_token():
    """Default (unset) all_logits — the existing behaviour, untouched."""
    s = _build_servicer("last")
    req = pb.InferenceStep(session_id="s1", step_id="step-0", prefix_length=0)
    req.token_ids.ids.extend([1, 2, 3])

    out = list(s.Inference(iter([req]), _FakeContext()))[0]
    assert list(out.token_ids.ids) == [4242]        # single legacy token, unchanged


def test_all_logits_default_is_false_when_field_unset():
    """A step built without touching all_logits at all reads as False (proto3
    default) — old clients that have never heard of the field get identical
    behaviour to before this change landed."""
    req = pb.InferenceStep(session_id="s1", step_id="step-0", prefix_length=0)
    assert req.all_logits is False


# ── all_logits=True: mode=last returns ALL per-position argmaxes ────


def test_all_logits_true_last_worker_returns_all_positions():
    s = _build_servicer("last")
    req = pb.InferenceStep(session_id="s1", step_id="step-0", prefix_length=0,
                            all_logits=True)
    req.token_ids.ids.extend([10, 11, 12, 13])   # cur + K=3 drafts, n_v=4

    out = list(s.Inference(iter([req]), _FakeContext()))[0]
    assert list(out.token_ids.ids) == [1000, 1001, 1002, 1003]   # one per position


def test_all_logits_true_single_token_round_trips():
    """K=0 edge case (n_v=1): all_logits still returns exactly one id — same
    shape as the K>0 case, just length 1."""
    s = _build_servicer("last")
    req = pb.InferenceStep(session_id="s1", step_id="step-0", prefix_length=0,
                            all_logits=True)
    req.token_ids.ids.extend([7])

    out = list(s.Inference(iter([req]), _FakeContext()))[0]
    assert list(out.token_ids.ids) == [1000]


# ── Non-last workers are unaffected by all_logits ────────────────────


def test_all_logits_true_middle_worker_still_returns_hidden_state():
    """A middle-mode worker must keep returning hidden_state regardless of
    all_logits — only the LAST worker's response shape changes."""
    daemon = _RecordingDaemon(n_embd=4)
    s = _build_servicer("middle", daemon=daemon)
    req = pb.InferenceStep(session_id="s1", step_id="step-0", prefix_length=0,
                            all_logits=True)
    req.hidden_state.raw = b"\x00" * (4 * 4 * 4)   # 4 tokens x n_embd=4 x f32
    req.hidden_state.n_tokens = 4
    req.hidden_state.batch = 1

    out = list(s.Inference(iter([req]), _FakeContext()))[0]
    assert out.HasField("hidden_state")
    assert not out.HasField("token_ids")
    # the flag still reaches the daemon (0x2 set) even though it doesn't change
    # this worker's OWN response shape — first_step=True here so keep_kv (0x1) unset.
    assert daemon.calls[0]["flags"] == 0x2


def test_all_logits_false_middle_worker_flags_zero():
    daemon = _RecordingDaemon(n_embd=4)
    s = _build_servicer("middle", daemon=daemon)
    req = pb.InferenceStep(session_id="s1", step_id="step-0", prefix_length=0)
    req.hidden_state.raw = b"\x00" * (4 * 4 * 4)
    req.hidden_state.n_tokens = 4
    req.hidden_state.batch = 1

    list(s.Inference(iter([req]), _FakeContext()))
    assert daemon.calls[0]["flags"] == 0x0


# ── flags bit composition: all_logits (0x2) ORs with keep_kv (0x1) ──


def test_all_logits_ors_with_keep_kv_on_non_first_step():
    """Second+ step on a stream sets keep_kv (0x1); all_logits must OR in 0x2,
    never replace it — the daemon needs BOTH bits together for a mid-stream
    verify round."""
    daemon = _RecordingDaemon(n_embd=4)
    s = _build_servicer("last", daemon=daemon)

    first = pb.InferenceStep(session_id="s1", step_id="step-0", prefix_length=0)
    first.token_ids.ids.extend([1])
    second = pb.InferenceStep(session_id="s1", step_id="step-1", prefix_length=1,
                              all_logits=True)
    second.token_ids.ids.extend([2, 3])

    list(s.Inference(iter([first, second]), _FakeContext()))
    assert daemon.calls[0]["flags"] == 0x0          # first step: cold, no all_logits
    assert daemon.calls[1]["flags"] == 0x3          # keep_kv | all_logits


# ── KV rewind: a smaller prefix_length reaches the daemon as start_pos ──


def test_prefix_length_rewind_reaches_daemon_as_smaller_start_pos():
    """This is the primitive the client-side accept() loop depends on: after a
    partial accept, the NEXT verify step's prefix_length is smaller than what a
    full-advance would have produced (kv_keep_after < start_pos + n_v). The
    worker must pass that smaller value straight through as start_pos — this
    IS the KV rewind (the daemon truncates its KV to start_pos before decoding,
    same code path _run_forward already exercises for the unary chain)."""
    daemon = _RecordingDaemon(n_embd=4)
    s = _build_servicer("last", daemon=daemon)

    # Cold prefill: 2 prompt tokens, prefix_length=0.
    prefill = pb.InferenceStep(session_id="s1", step_id="step-0", prefix_length=0)
    prefill.token_ids.ids.extend([1, 2])

    # A verify round: cur + K=2 drafts (n_v=3) at prefix_length=2 (the true
    # committed length after prefill).
    verify = pb.InferenceStep(session_id="s1", step_id="step-1", prefix_length=2,
                              all_logits=True)
    verify.token_ids.ids.extend([9, 10, 11])

    # Simulated REJECT at position 0: only `cur` was good, so the caller keeps
    # just prefix 2+1=3 (kv_keep_after(2, 0) == 3) instead of advancing by n_v=3
    # to 5 — this is the rewind: a SMALLER prefix_length than "no reject" would give.
    rewound = pb.InferenceStep(session_id="s1", step_id="step-2", prefix_length=3,
                               all_logits=True)
    rewound.token_ids.ids.extend([42, 43])

    list(s.Inference(iter([prefill, verify, rewound]), _FakeContext()))
    assert daemon.calls[0]["start_pos"] == 0
    assert daemon.calls[1]["start_pos"] == 2
    assert daemon.calls[2]["start_pos"] == 3          # rewound, not 2+3=5
    assert daemon.calls[2]["start_pos"] < daemon.calls[1]["start_pos"] + 3
