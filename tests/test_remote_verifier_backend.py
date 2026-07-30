"""Unit tests for scripts/remote_verifier_backend.py — the llama.cpp CPU-only whole-model
adapter that backs remote_proposals.VerifierSession's verify_fn on a live nakshatra_serve
(NKS_REMOTE_PROPOSALS=1). See docs/findings/remote-proposals.md "Remaining ask" #1/#2.

NO GPU, NO real GGUF, NO llama_cpp import in most of these tests: `FakeLlama` is a
deterministic stand-in shaped exactly like the llama_cpp.Llama surface `LlamaVerifier`
touches (n_tokens get/set, eval(), scores, _ctx.kv_cache_seq_rm, n_vocab, token_bos, close) —
same Markov-chain trick tests/test_remote_proposals.py and tests/test_speculative.py use, so
the fake needs no history beyond what a batch already carries. Proves: the adapter's
per-position argmax contract, the KV-rewind-on-mispredict primitive (mirrors
speculative.DraftModel's LCP rollback), a vocab-mismatch guardrail, that llama_cpp is never
imported unless a real model_path is used, and a loopback HTTP smoke round-trip through
remote_proposals.serve_verifier/http_submit — all with the fake model.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import remote_proposals as rp  # noqa: E402
import remote_verifier_backend as rvb  # noqa: E402


# ── FakeLlama: deterministic, no llama_cpp / no GGUF anywhere ─────────
class _FakeCtx:
    def __init__(self, owner: "FakeLlama"):
        self._owner = owner
        self.removed: list[int] = []   # p0 args kv_cache_seq_rm was called with

    def kv_cache_seq_rm(self, seq_id, p0, p1):
        self.removed.append(p0)
        self._owner._cache = self._owner._cache[:p0]
        for pos in [p for p in self._owner.scores if p >= p0]:
            del self._owner.scores[pos]


class FakeLlama:
    """Stand-in for llama_cpp.Llama exposing exactly the surface LlamaVerifier touches.
    `successor` is a token->token Markov map: eval()'s scores put successor[t] at the
    argmax for the position t was just written to — same trick the module docstrings of
    remote_proposals.py/speculative.py describe."""
    VOCAB = 64

    def __init__(self, successor: dict, bos: int = 1):
        self.successor = dict(successor)
        self.bos = bos
        self._cache: list = []
        self.scores: dict = {}
        self._ctx = _FakeCtx(self)
        self.eval_calls: list = []
        self.closed = False

    @property
    def n_tokens(self):
        return len(self._cache)

    @n_tokens.setter
    def n_tokens(self, v):
        self._cache = self._cache[:v]

    def eval(self, tokens):
        tokens = [int(t) for t in tokens]
        self.eval_calls.append(tokens)
        for t in tokens:
            pos = len(self._cache)
            self._cache.append(t)
            row = [0.0] * self.VOCAB
            row[self.successor.get(t, self.bos) % self.VOCAB] = 1.0
            self.scores[pos] = row

    def n_vocab(self):
        return self.VOCAB

    def token_bos(self):
        return self.bos

    def close(self):
        self.closed = True


# ── LlamaVerifier — the per-call argmax contract ──────────────────────
def test_verify_returns_correct_greedy_argmax_per_position():
    fake = FakeLlama(successor={1: 2, 2: 3, 3: 4, 4: 5})
    v = rvb.LlamaVerifier(llama=fake)
    out = v.verify([1, 2, 3], start_pos=0)
    assert out == [2, 3, 4]
    assert fake.n_tokens == 3
    assert fake.eval_calls == [[1, 2, 3]]


def test_verify_appends_without_rewind_when_start_pos_matches_cache():
    fake = FakeLlama(successor={1: 2, 2: 3, 3: 4, 4: 5})
    v = rvb.LlamaVerifier(llama=fake)
    v.verify([1, 2], start_pos=0)
    assert fake.n_tokens == 2
    # next call's start_pos == what's already cached: pure append, no rewind
    out = v.verify([3, 4], start_pos=2)
    assert out == [4, 5]
    assert fake._ctx.removed == []
    assert fake.n_tokens == 4


def test_verify_rewinds_kv_on_mispredict_start_pos_behind_cache():
    fake = FakeLlama(successor={1: 2, 2: 3, 3: 9, 9: 9})
    v = rvb.LlamaVerifier(llama=fake)
    v.verify([1, 2, 3], start_pos=0)              # caches positions 0,1,2
    assert fake.n_tokens == 3
    # a mispredict resets the caller's cursor back to position 1 — the adapter must
    # discard whatever it over-computed past that (the rewind primitive
    # VerifierSession.submit's cursor arithmetic assumes exists)
    out = v.verify([2, 9], start_pos=1)
    assert fake._ctx.removed == [1]
    assert fake.n_tokens == 3                     # 1 kept + 2 newly evaluated
    assert out == [3, 9]


def test_verify_raises_on_a_start_pos_gap_ahead_of_cache():
    fake = FakeLlama(successor={1: 2})
    v = rvb.LlamaVerifier(llama=fake)
    with pytest.raises(ValueError, match="ahead of this adapter's cached KV"):
        v.verify([1], start_pos=5)


def test_verify_rejects_empty_token_batch():
    fake = FakeLlama(successor={})
    v = rvb.LlamaVerifier(llama=fake)
    with pytest.raises(ValueError):
        v.verify([], start_pos=0)


# ── vocab guardrail ────────────────────────────────────────────────────
def test_expect_vocab_size_mismatch_raises_and_closes_the_handle():
    fake = FakeLlama(successor={})
    with pytest.raises(rvb.VerifierVocabError):
        rvb.LlamaVerifier(llama=fake, expect_vocab_size=999)
    assert fake.closed is True


def test_expect_vocab_size_match_constructs_cleanly():
    fake = FakeLlama(successor={1: 2})
    v = rvb.LlamaVerifier(llama=fake, expect_vocab_size=FakeLlama.VOCAB)
    assert v.bos_token() == fake.bos


# ── lazy import: llama_cpp is never touched when a fake is injected ──
def test_injected_fake_never_imports_llama_cpp():
    for name in ("llama_cpp", "remote_verifier_backend"):
        sys.modules.pop(name, None)
    import remote_verifier_backend as rvb2   # fresh import
    assert "llama_cpp" not in sys.modules
    fake = FakeLlama(successor={1: 2})
    v = rvb2.LlamaVerifier(llama=fake)
    assert "llama_cpp" not in sys.modules
    v.verify([1], start_pos=0)
    assert "llama_cpp" not in sys.modules


def test_missing_model_path_and_no_injected_llama_raises():
    with pytest.raises(ValueError, match="model_path"):
        rvb.LlamaVerifier()


# ── full protocol contract through the adapter (byte-identical oracle) ─
def _sequential_reference(succ, first, n, eos):
    out, cur = [], first
    for _ in range(n):
        cur = succ.get(cur, eos)
        out.append(cur)
        if cur == eos:
            break
    return out


def _make_draft(succ, eos, mode):
    def propose(context, k):
        out, last = [], context[-1]
        for i in range(k):
            wrong = mode == "adversarial" or (mode == "mixed" and i % 2 == 1)
            out.append(-1 if wrong else succ.get(last, eos))
            last = succ.get(last, eos)
        return out
    return propose


@pytest.mark.parametrize("mode", ["perfect", "adversarial", "mixed"])
def test_proposal_loop_byte_identical_through_llama_verifier(mode):
    """The same correctness oracle remote_proposals._selftest() proves for the bare
    Markov-oracle verify_fn, proven here through the REAL adapter class (LlamaVerifier
    wrapping a fake KV-cache-shaped model) — this is the "adapter satisfies
    VerifierSession's exact expectations" proof the task asked for, not just a
    call-shape check."""
    succ = {t: t + 1 for t in range(1, 20)}
    EOS = 99
    succ[20] = EOS
    succ[EOS] = EOS
    fake = FakeLlama(successor=succ, bos=1)
    verifier = rvb.LlamaVerifier(llama=fake)
    session = rp.VerifierSession(verifier, [1], max_window=8)
    out = rp.proposal_loop(
        draft_propose=_make_draft(succ, EOS, mode), submit=session.submit,
        prompt_tokens=[1], k=4, max_tokens=15, eos_ids={EOS},
    )
    ref = _sequential_reference(succ, 1, 15, EOS)
    assert out == ref


# ── make_verifier_session / start_proposals_server ────────────────────
def test_make_verifier_session_seeds_from_bos_when_no_prompt_given():
    fake = FakeLlama(successor={7: 8}, bos=7)
    verifier, session = rvb.make_verifier_session(llama=fake)
    assert session.prompt == [7]


def test_make_verifier_session_honors_explicit_prompt_tokens():
    fake = FakeLlama(successor={5: 6}, bos=1)
    verifier, session = rvb.make_verifier_session(llama=fake, prompt_tokens=[5])
    assert session.prompt == [5]


# ── loopback smoke: real thread + real HTTP + the FAKE model ─────────
def test_loopback_smoke_round_trip_via_http_submit_with_fake_model():
    succ = {1: 2, 2: 3, 3: 4, 4: 99, 99: 99}
    fake = FakeLlama(successor=succ, bos=1)
    httpd, thread, verifier, session = rvb.start_proposals_server(
        None, host="127.0.0.1", port=0, llama=fake, prompt_tokens=[1],
        log=lambda *_: None)
    assert isinstance(thread, threading.Thread)
    try:
        port = httpd.server_address[1]
        n_accepted, correction, cursor = rp.http_submit(f"127.0.0.1:{port}", [2, 3])
        assert n_accepted == 2
        assert correction == 4
        assert cursor == 3
        assert session.committed == [1, 2, 3, 4]
    finally:
        httpd.shutdown()
        thread.join(timeout=2)
        verifier.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
