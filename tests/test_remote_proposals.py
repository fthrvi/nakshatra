"""
Unit tests for scripts/remote_proposals.py — bounded external token proposals (WAN
speculative decoding without splitting the model; Mesh-LLM adoption #1).

No GPU, no model weights, no live services: `VerifierSession` is driven with a fake
deterministic Markov-oracle "verifier" (next token depends only on the last token, same
trick tests/test_speculative.py uses so the mock needs no history beyond what the module
already threads through `batch`). The correctness oracle asserted throughout: proposal_loop's
output must be BYTE-IDENTICAL to sequentially calling the verifier alone, for ANY draft
quality (perfect, adversarial, mixed) — a bad draft may only cost speed, never correctness.
"""
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from remote_proposals import (  # noqa: E402
    VerifierSession,
    http_submit,
    proposal_loop,
    serve_verifier,
)


# ---------------------------------------------------------------- shared fixtures / helpers

def _make_oracle(succ, eos_id):
    """A pure Markov oracle: next token depends only on the last token."""
    def oracle(tok):
        return succ.get(tok, eos_id)
    return oracle


def _make_verify_fn(oracle):
    def verify_fn(tokens, start_pos):
        return [oracle(t) for t in tokens]
    return verify_fn


def _sequential_reference(oracle, first_tok, n, eos_id=None):
    """Plain sequential reference: repeatedly ask the verifier alone for one token —
    what proposal_loop's output must match byte-for-byte regardless of draft quality."""
    out, cur = [], first_tok
    for _ in range(n):
        cur = oracle(cur)
        out.append(cur)
        if eos_id is not None and cur == eos_id:
            break
    return out


def _make_draft(oracle, mode):
    """mode: 'perfect' (always right), 'adversarial' (always wrong), 'mixed' (alternating).
    Only the FIRST wrong token in a proposal chunk matters for correctness (VerifierSession
    stops accepting at the first mismatch), so `last` just tracks the true oracle chain."""
    def propose(context, k):
        out, last = [], context[-1]
        for i in range(k):
            wrong = mode == "adversarial" or (mode == "mixed" and i % 2 == 1)
            out.append(-1 if wrong else oracle(last))
            last = oracle(last)
        return out
    return propose


@pytest.fixture
def chain():
    """A small Markov chain 100->101->...->115->EOS, deterministic and cheap to reason about."""
    succ = {t: t + 1 for t in range(100, 115)}
    eos_id = 999
    succ[115] = eos_id
    succ[eos_id] = eos_id
    return _make_oracle(succ, eos_id), eos_id


# ---------------------------------------------------------------- VerifierSession.submit()

def test_perfect_prefix_accepts_full_window(chain):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=8)
    n_accepted, correction, cursor = session.submit([101, 102, 103])
    assert n_accepted == 3
    assert correction == 104          # bonus token past the fully-accepted proposal
    assert cursor == 4                # start (0) + 1 (cur) + 3 (accepted) = 4
    assert session.committed == [100, 101, 102, 103, 104]


def test_immediate_mismatch_commits_only_correction(chain):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=8)
    n_accepted, correction, cursor = session.submit([-1, -1, -1])   # all wrong
    assert n_accepted == 0
    assert correction == 101          # the verifier's own true next token
    assert cursor == 1                # advanced by exactly 1, not 3
    assert session.committed == [100, 101]


def test_partial_accept_then_correction(chain):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=8)
    n_accepted, correction, cursor = session.submit([101, 102, -1, 999])
    assert n_accepted == 2
    assert correction == 103          # the true token where the draft first went wrong
    assert cursor == 3                # 0 + 1(cur) + 2(accepted)
    assert session.committed == [100, 101, 102, 103]


def test_cursor_rewind_after_mispredict_not_full_advance(chain):
    """The cursor after a mispredict must reflect only what was KEPT (n_accepted+1), never
    the full proposal length — this IS the KV-rewind analog: a real verifier truncates its
    KV to this smaller cursor on the *next* call, discarding whatever it may have
    speculatively computed past the mismatch."""
    oracle, eos_id = chain
    calls = []

    def spy_verify_fn(tokens, start_pos):
        calls.append((list(tokens), start_pos))
        return [oracle(t) for t in tokens]

    session = VerifierSession(spy_verify_fn, [100], max_window=8)
    n1, c1, cur1 = session.submit([101, -1, -1, -1, -1])   # accept 1, then miss
    assert n1 == 1 and cur1 == 2                             # NOT 0 + 1 + 5
    # next round must present the REWOUND cursor to verify_fn, not a naive full advance
    n2, c2, cur2 = session.submit([103])
    assert calls[-1] == ([c1, 103], 2)                       # start_pos == the rewound cursor
    assert n2 == 1 and cur2 == 4
    assert session.committed == [100, 101, 102, 103, 104]


def test_bounded_window_rejects_oversized_proposal(chain):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=2)
    before = (session.committed, session.cursor)
    n_accepted, correction, cursor = session.submit([101, 102, 103])   # window is 2
    assert (n_accepted, correction) == (0, None)
    assert cursor == before[1]
    assert session.committed == before[0]      # state untouched by a rejected proposal


def test_bounded_window_accepts_at_the_limit(chain):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=2)
    n_accepted, correction, cursor = session.submit([101, 102])        # exactly at the limit
    assert correction is not None
    assert n_accepted == 2


def test_empty_proposal_raises(chain):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=8)
    with pytest.raises(ValueError):
        session.submit([])


def test_verify_fn_contract_violation_raises(chain):
    oracle, eos_id = chain
    session = VerifierSession(lambda tokens, sp: [oracle(t) for t in tokens][:-1], [100], max_window=8)
    with pytest.raises(ValueError):
        session.submit([101, 102])


def test_construction_requires_nonempty_prompt_and_positive_window(chain):
    oracle, eos_id = chain
    with pytest.raises(ValueError):
        VerifierSession(_make_verify_fn(oracle), [], max_window=8)
    with pytest.raises(ValueError):
        VerifierSession(_make_verify_fn(oracle), [100], max_window=0)


# ---------------------------------------------------------------- proposal_loop byte-identity

@pytest.mark.parametrize("mode", ["perfect", "adversarial", "mixed"])
def test_byte_identical_to_sequential_verifier(chain, mode):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=8)
    out = proposal_loop(
        draft_propose=_make_draft(oracle, mode), submit=session.submit,
        prompt_tokens=[100], k=4, max_tokens=20, eos_ids={eos_id},
    )
    ref = _sequential_reference(oracle, 100, 20, eos_id=eos_id)
    assert out == ref
    if mode == "adversarial":
        # a maximally-wrong draft still makes progress: every round commits exactly the
        # correction (n_accepted==0 throughout), so len(out) must equal the reference length
        # exactly — no tokens skipped, none duplicated.
        assert len(out) == len(ref)


def test_perfect_draft_accepts_full_chunks_every_round(chain):
    """A perfect draft should never see a mispredict — n_accepted should equal k on every
    round while there's enough runway left (sanity check that acceptance is really firing,
    not just that the final byte-identity happens to still hold)."""
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=8)
    rounds = []
    real_submit = session.submit

    def spy_submit(proposal):
        result = real_submit(proposal)
        rounds.append((len(proposal), result[0]))
        return result

    proposal_loop(
        draft_propose=_make_draft(oracle, "perfect"), submit=spy_submit,
        prompt_tokens=[100], k=3, max_tokens=12, eos_ids={eos_id},
    )
    assert rounds, "expected at least one round"
    assert all(n_accepted == proposed for proposed, n_accepted in rounds)


def test_proposal_loop_respects_eos_mid_batch(chain):
    """EOS landing partway through a committed batch (accepted-prefix + correction) must
    stop generation immediately, not run past it."""
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [113], max_window=8)   # near the tail
    out = proposal_loop(
        draft_propose=_make_draft(oracle, "perfect"), submit=session.submit,
        prompt_tokens=[113], k=8, max_tokens=50, eos_ids={eos_id},
    )
    assert out[-1] == eos_id
    assert out.count(eos_id) == 1
    ref = _sequential_reference(oracle, 113, 50, eos_id=eos_id)
    assert out == ref


def test_proposal_loop_shrinks_k_on_bounded_window_rejection(chain):
    """If k exceeds the session's max_window, proposal_loop must auto-shrink and still
    converge to the byte-identical output, rather than looping forever or erroring."""
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=2)   # small window
    out = proposal_loop(
        draft_propose=_make_draft(oracle, "perfect"), submit=session.submit,
        prompt_tokens=[100], k=10, max_tokens=10, eos_ids={eos_id},   # k way over the window
    )
    ref = _sequential_reference(oracle, 100, 10, eos_id=eos_id)
    assert out == ref


def test_proposal_loop_rejects_when_min_k_still_too_big():
    """If even min_k exceeds max_window, the loop must fail loudly rather than spin."""
    succ = {1: 2, 2: 3}
    oracle = _make_oracle(succ, eos_id=99)
    session = VerifierSession(_make_verify_fn(oracle), [1], max_window=1)
    with pytest.raises(RuntimeError):
        proposal_loop(
            draft_propose=_make_draft(oracle, "perfect"), submit=session.submit,
            prompt_tokens=[1], k=4, max_tokens=5, min_k=2,   # min_k(2) > max_window(1)
        )


def test_proposal_loop_k_below_min_k_raises(chain):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=8)
    with pytest.raises(ValueError):
        proposal_loop(
            draft_propose=_make_draft(oracle, "perfect"), submit=session.submit,
            prompt_tokens=[100], k=1, max_tokens=5, min_k=3,
        )


def test_on_token_callback_fires_for_every_committed_token(chain):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=8)
    seen = []
    out = proposal_loop(
        draft_propose=_make_draft(oracle, "mixed"), submit=session.submit,
        prompt_tokens=[100], k=4, max_tokens=9, on_token=seen.append,
    )
    assert seen == out


# ---------------------------------------------------------------- loopback HTTP transport

@pytest.fixture
def live_verifier(chain):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=8)
    httpd = serve_verifier(session, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"127.0.0.1:{port}", session, oracle, eos_id
    httpd.shutdown()


def test_http_submit_round_trip_matches_in_process(live_verifier):
    peer, session, oracle, eos_id = live_verifier
    n_accepted, correction, cursor = http_submit(peer, [101, 102, 103])
    assert n_accepted == 3
    assert correction == 104
    assert cursor == 4
    # the server-side session state actually advanced (not a stateless echo)
    assert session.committed == [100, 101, 102, 103, 104]


def test_http_submit_bounded_window_rejection(chain):
    oracle, eos_id = chain
    session = VerifierSession(_make_verify_fn(oracle), [100], max_window=1)
    httpd = serve_verifier(session, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        n_accepted, correction, cursor = http_submit(f"127.0.0.1:{port}", [101, 102])
        assert (n_accepted, correction) == (0, None)
        assert session.committed == [100]      # untouched
    finally:
        httpd.shutdown()


def test_proposal_loop_over_http_byte_identical(live_verifier):
    peer, session, oracle, eos_id = live_verifier
    out = proposal_loop(
        draft_propose=_make_draft(oracle, "mixed"),
        submit=lambda p: http_submit(peer, p),
        prompt_tokens=[100], k=4, max_tokens=15, eos_ids={eos_id},
    )
    ref = _sequential_reference(oracle, 100, 15, eos_id=eos_id)
    assert out == ref


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
