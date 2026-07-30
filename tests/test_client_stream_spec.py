"""Tests for speculative decode ON THE STREAM — the client.py side (2026-07-29).

Companion to tests/test_worker_stream_spec.py (the worker side). Covers:
  - stream_spec_disable_reasons: the pure capability/config gate — must refuse and
    fall back to plain streaming whenever any worker lacks "stream_spec", push mode
    is requested, plain --use-streaming isn't on, or no draft model is configured.
  - stream_spec_verify_fn + speculative.speculative_round wired together over FAKE
    per-worker streams (no gRPC, no daemon, no GPU): accept-all, accept-none, partial
    accept, and the KV-rewind wiring (the next round's request carries the REWOUND
    prefix_length, not a naive full advance).
  - call_inference_step's all_logits plumbing: default False, explicit True.

client.py optional-imports grpc/nakshatra_pb2, both present in this venv (needed to
build InferenceStep fakes), so importing it here is safe and matches
test_client_registry.py's convention.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import client as cli               # noqa: E402
import nakshatra_pb2 as pb         # noqa: E402
from speculative import speculative_round  # noqa: E402


# ── stream_spec_disable_reasons: the pure gate ───────────────────────


def _caps(*missing_from):
    """Two workers, both capable, unless listed in `missing_from` (worker ids)."""
    all_caps = ["streaming", "stream_spec"]
    reduced = ["streaming"]
    return [
        ("w0", reduced if "w0" in missing_from else all_caps),
        ("w1", reduced if "w1" in missing_from else all_caps),
    ]


def test_disable_reasons_empty_when_everything_ready():
    reasons = cli.stream_spec_disable_reasons(True, False, True, _caps())
    assert reasons == []


def test_disable_reasons_requires_streaming():
    reasons = cli.stream_spec_disable_reasons(False, False, True, _caps())
    assert any("use-streaming" in r for r in reasons)


def test_disable_reasons_rejects_push_mode():
    reasons = cli.stream_spec_disable_reasons(True, True, True, _caps())
    assert any("use-streaming" in r for r in reasons)


def test_disable_reasons_requires_draft_model():
    reasons = cli.stream_spec_disable_reasons(True, False, False, _caps())
    assert any("draft-model-path" in r for r in reasons)


def test_disable_reasons_refuses_when_one_worker_lacks_capability():
    """The house rule: protobuf silently ignores unknown fields, so an old
    worker missing 'stream_spec' must produce an explicit refusal — never a
    silent wrong answer."""
    reasons = cli.stream_spec_disable_reasons(True, False, True, _caps("w1"))
    assert len(reasons) == 1
    assert "stream_spec" in reasons[0]
    assert "w1" in reasons[0]
    assert "w0" not in reasons[0]


def test_disable_reasons_refuses_when_all_workers_lack_capability():
    reasons = cli.stream_spec_disable_reasons(True, False, True, _caps("w0", "w1"))
    assert len(reasons) == 1
    assert "w0" in reasons[0] and "w1" in reasons[0]


def test_disable_reasons_accumulates_multiple_reasons():
    """All three gates independently contribute — not short-circuited — so the
    fallback log line tells the operator everything wrong at once."""
    reasons = cli.stream_spec_disable_reasons(False, False, False, _caps("w0"))
    assert len(reasons) == 3


# ── call_inference_step: all_logits plumbing ─────────────────────────


class _CapturingStreamer:
    def __init__(self, worker_id, response):
        self.worker_id = worker_id
        self._response = response
        self.requests = []

    def step(self, request_step):
        self.requests.append(request_step)
        return self._response


def _hidden_response(n_tokens, n_embd):
    out = pb.InferenceStep(session_id="s", step_id="step-0", prefix_length=n_tokens)
    out.hidden_state.raw = b"\x00" * (n_tokens * n_embd * 4)
    out.hidden_state.batch = 1
    out.hidden_state.n_tokens = n_tokens
    return out


def test_call_inference_step_all_logits_defaults_false():
    streamer = _CapturingStreamer("w0", _hidden_response(2, 4))
    cli.call_inference_step(streamer, struct.pack("<2i", 1, 2), 2, True,
                            session_id="s", step_idx=0, prefix_length=0)
    assert streamer.requests[0].all_logits is False


def test_call_inference_step_all_logits_true_is_sent():
    streamer = _CapturingStreamer("w0", _hidden_response(2, 4))
    cli.call_inference_step(streamer, struct.pack("<2i", 1, 2), 2, True,
                            session_id="s", step_idx=0, prefix_length=0,
                            all_logits=True)
    assert streamer.requests[0].all_logits is True


# ── stream_spec_verify_fn + speculative_round, wired over fake streams ──


class _FakeDraft:
    """Scripted draft — proposes exactly the fixed token ids a test hands it,
    regardless of prefix. Lets each test control drafts and target argmaxes
    independently, same isolation test_speculative.py uses for accept()."""

    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.propose_calls = []

    def propose(self, prefix_tokens, k):
        self.propose_calls.append((list(prefix_tokens), k))
        assert k == len(self.proposals), f"test bug: k={k} != len(proposals)={len(self.proposals)}"
        return list(self.proposals)


class _FirstFakeStreamer:
    """Stand-in for the first worker's persistent Inference stream: tokens in,
    hidden_state out (n_tokens preserved so the fake last worker can size its
    own response). Records every request for assertion."""

    def __init__(self, n_embd):
        self.worker_id = "w0"
        self.n_embd = n_embd
        self.requests = []

    def step(self, request_step):
        self.requests.append(request_step)
        n_v = len(request_step.token_ids.ids)
        out = pb.InferenceStep(session_id=request_step.session_id,
                               step_id=request_step.step_id,
                               prefix_length=request_step.prefix_length + n_v)
        out.hidden_state.raw = b"\x00" * (n_v * self.n_embd * 4)
        out.hidden_state.batch = 1
        out.hidden_state.n_tokens = n_v
        return out


class _LastFakeStreamer:
    """Stand-in for the last worker: hidden_state in, token_ids out — a
    scripted queue of target-argmax lists, one per verify round, popped in
    call order so a test can drive multiple rounds (e.g. the rewind test)."""

    def __init__(self, target_argmax_rounds):
        self.worker_id = "w1"
        self._queue = [list(r) for r in target_argmax_rounds]
        self.requests = []

    def step(self, request_step):
        self.requests.append(request_step)
        n_v = request_step.hidden_state.n_tokens
        argmax = self._queue.pop(0)
        assert len(argmax) == n_v, f"canned round has {len(argmax)} argmaxes, request wants {n_v}"
        out = pb.InferenceStep(session_id=request_step.session_id,
                               step_id=request_step.step_id,
                               prefix_length=request_step.prefix_length + n_v)
        out.token_ids.ids.extend(argmax)
        return out


# sorted_stubs only needs the right LENGTH for a 2-worker chain — stream_spec_verify_fn
# walks sorted_stubs[1:-1] for middle workers (empty here) and reads len() for the last
# index; the (w, stub, info) tuple contents are never touched for a 2-worker chain.
_PLACEHOLDER_STUBS = [(None, None, None), (None, None, None)]


def test_accept_all_commits_k_plus_bonus():
    n_embd = 4
    first = _FirstFakeStreamer(n_embd)
    last = _LastFakeStreamer([[10, 11, 99]])   # t0=d0, t1=d1, t2=bonus
    verify_fn = cli.stream_spec_verify_fn(
        [first, last], _PLACEHOLDER_STUBS, n_embd,
        session_id="sess", step_idx=1, prefix_length=5)

    draft = _FakeDraft([10, 11])   # K=2
    res, drafts = speculative_round(draft, [1, 2, 3], 2, verify_fn)

    assert res.n_accepted == 2
    assert res.committed == [10, 11, 99]
    assert drafts == [10, 11]
    # every leg of the round set all_logits=True
    assert all(r.all_logits for r in first.requests)
    assert all(r.all_logits for r in last.requests)
    # prefix_length was held FIXED for the whole round (not advanced mid-round)
    assert first.requests[0].prefix_length == 5
    assert last.requests[0].prefix_length == 5


def test_accept_none_commits_one_correction():
    n_embd = 4
    first = _FirstFakeStreamer(n_embd)
    last = _LastFakeStreamer([[77, 11, 99]])   # t0 != d0 -> immediate reject
    verify_fn = cli.stream_spec_verify_fn(
        [first, last], _PLACEHOLDER_STUBS, n_embd,
        session_id="sess", step_idx=1, prefix_length=5)

    draft = _FakeDraft([10, 11])
    res, drafts = speculative_round(draft, [1, 2, 3], 2, verify_fn)

    assert res.n_accepted == 0
    assert res.committed == [77]                # correction only — progress guaranteed


def test_partial_accept_then_correction():
    n_embd = 4
    first = _FirstFakeStreamer(n_embd)
    last = _LastFakeStreamer([[10, 88, 99]])     # d0 matches, d1 doesn't
    verify_fn = cli.stream_spec_verify_fn(
        [first, last], _PLACEHOLDER_STUBS, n_embd,
        session_id="sess", step_idx=1, prefix_length=5)

    draft = _FakeDraft([10, 11])
    res, drafts = speculative_round(draft, [1, 2, 3], 2, verify_fn)

    assert res.n_accepted == 1
    assert res.committed == [10, 88]


def test_rewind_correctness_next_round_carries_kept_length_not_full_advance():
    """The KV-rewind wiring, client side: round 1 rejects at position 0 (n_accepted=0)
    at prefix_length=5 with n_v=3 — kv_keep_after(5, 0) == 6, NOT the naive
    full-advance 5+3=8. Round 2 must be driven at prefix_length=6; assert the actual
    request sent to both streamers carries 6, proving the caller wired the rewind
    through (this is the wiring test; worker.py honouring it as start_pos is proven
    separately in test_worker_stream_spec.py)."""
    from speculative import kv_keep_after

    n_embd = 4
    first = _FirstFakeStreamer(n_embd)
    last = _LastFakeStreamer([
        [77, 11, 99],    # round 1: immediate reject -> n_accepted=0
        [55, 66],        # round 2: K=1, accept-all -> bonus
    ])

    verify_fn_1 = cli.stream_spec_verify_fn(
        [first, last], _PLACEHOLDER_STUBS, n_embd,
        session_id="sess", step_idx=1, prefix_length=5)
    draft1 = _FakeDraft([10, 11])
    res1, _ = speculative_round(draft1, [1, 2, 3], 2, verify_fn_1)
    assert res1.n_accepted == 0

    rewound_prefix = kv_keep_after(5, res1.n_accepted)
    assert rewound_prefix == 6           # < the naive full-advance of 5+3=8

    verify_fn_2 = cli.stream_spec_verify_fn(
        [first, last], _PLACEHOLDER_STUBS, n_embd,
        session_id="sess", step_idx=2, prefix_length=rewound_prefix)
    draft2 = _FakeDraft([55])            # K=1
    res2, _ = speculative_round(draft2, [1, 2, 3, res1.committed[-1]], 1, verify_fn_2)
    assert res2.n_accepted == 1
    assert res2.committed == [55, 66]

    # The actual wire-level assertion: round 2's requests carry prefix_length=6,
    # not 8 — the rewind reached the transport layer, not just local bookkeeping.
    assert first.requests[-1].prefix_length == 6
    assert last.requests[-1].prefix_length == 6
