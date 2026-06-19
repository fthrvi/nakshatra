"""
Unit tests for the daemon-independent core of speculative decoding (scripts/speculative.py).

These prove the greedy-acceptance algorithm and the KV bookkeeping are exactly correct —
the part that MUST be right before any C++ daemon / GPU work. No model or GPU is loaded:
`speculative_round` is driven with a fake target so the math is tested in isolation.

The correctness oracle (asserted in test_matches_plain_greedy): speculative decode must
produce the IDENTICAL token stream that plain greedy decode would, for any draft quality.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from speculative import accept, kv_keep_after, next_start_pos, speculative_round, AcceptResult  # noqa: E402


# ---------------------------------------------------------------- accept() core

def test_all_drafts_accepted_takes_bonus():
    # draft nailed all K; target agrees on every position and adds a bonus token.
    drafts = [10, 11, 12]
    target = [10, 11, 12, 13]          # t0..t2 match d0..d2, t3 = bonus
    r = accept(drafts, target)
    assert r.n_accepted == 3
    assert r.committed == [10, 11, 12, 13]   # 3 accepted + 1 bonus
    assert r.n_committed == 4


def test_immediate_reject_commits_one_correction():
    # draft wrong at the very first token → commit just the correction, make progress.
    drafts = [10, 11, 12]
    target = [99, 11, 12, 13]          # t0 != d0
    r = accept(drafts, target)
    assert r.n_accepted == 0
    assert r.committed == [99]         # exactly one token, like plain decode
    assert r.n_committed == 1


def test_partial_accept_then_correct():
    # first two drafts right, third wrong → commit 2 accepted + 1 correction.
    drafts = [10, 11, 12]
    target = [10, 11, 77, 13]          # t2 != d2
    r = accept(drafts, target)
    assert r.n_accepted == 2
    assert r.committed == [10, 11, 77]
    assert r.n_committed == 3          # always n_accepted + 1


def test_k1_accept_and_reject():
    assert accept([5], [5, 6]).committed == [5, 6]     # accept + bonus
    assert accept([5], [9, 6]).committed == [9]        # reject → correction only


def test_progress_is_always_at_least_one_token():
    # for any draft/target of matching length, at least one token is committed.
    for K in range(1, 6):
        drafts = list(range(K))
        target = [1000 + i for i in range(K + 1)]   # all mismatch
        r = accept(drafts, target)
        assert r.n_committed >= 1
        assert r.n_committed == r.n_accepted + 1


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accept([1, 2, 3], [1, 2, 3])        # need K+1 argmaxes, got K


def test_accepts_numpy_like_ints():
    # target_argmax may arrive as non-builtin ints (e.g. numpy); committed must be plain int.
    class FakeInt(int):
        pass
    r = accept([FakeInt(3)], [FakeInt(3), FakeInt(4)])
    assert r.committed == [3, 4]
    assert all(type(t) is int for t in r.committed)


# ---------------------------------------------------------------- KV bookkeeping

def test_kv_keep_after():
    # keep cur + accepted drafts; the committed correction/bonus has no KV yet.
    assert kv_keep_after(start_pos=100, n_accepted=0) == 101    # just cur
    assert kv_keep_after(start_pos=100, n_accepted=2) == 103    # cur + 2 accepted
    assert kv_keep_after(start_pos=100, n_accepted=3) == 104    # bonus case: all K kept


def test_next_start_pos_matches_committed_progress():
    # next start_pos advances by exactly the KV we kept (cur + accepted), and the number of
    # newly generated tokens equals n_accepted + 1 (accepted + correction/bonus).
    for n_acc in range(0, 4):
        sp = 50
        nsp = next_start_pos(sp, n_acc)
        assert nsp == sp + 1 + n_acc


# ------------------------------------------------------ end-to-end vs plain greedy

def _plain_greedy(prefix, target_oracle, n_tokens):
    """Reference: plain greedy decode using the oracle one token at a time."""
    seq = list(prefix)
    for _ in range(n_tokens):
        seq.append(target_oracle(seq))
    return seq[len(prefix):]


def _spec_greedy(prefix, target_oracle, draft_fn, k, n_tokens):
    """Speculative decode driven by a fake distributed target built from the same oracle.

    The fake verify_fn computes, for input [cur, d0..d_{K-1}], the target's greedy argmax at
    each position — i.e. target_oracle applied to the growing prefix — exactly what a correct
    daemon would return. This is the property the daemon patch must satisfy.
    """
    class _Draft:
        def propose(self, prefix_tokens, k_):
            return draft_fn(list(prefix_tokens), k_)

    seq = list(prefix)
    produced = []
    while len(produced) < n_tokens:
        cur = seq[-1]
        drafts = draft_fn(seq, k)

        def verify_fn(batch):
            # batch = [cur, d0..d_{K-1}]; argmax at position i = oracle(prefix so far up to i)
            outs = []
            ctx = list(seq[:-1])      # everything before cur
            for tok in batch:         # feed cur, then each draft
                ctx.append(tok)
                outs.append(target_oracle(ctx))
            return outs

        r, _ = speculative_round(_Draft(), seq, k, verify_fn)
        for t in r.committed:
            if len(produced) >= n_tokens:
                break
            produced.append(t)
            seq.append(t)
    return produced[:n_tokens]


def test_matches_plain_greedy_perfect_draft():
    # a deterministic oracle: next token = (last + 1) mod 100. A perfect draft predicts it.
    oracle = lambda s: (s[-1] + 1) % 100
    draft_fn = lambda s, k: [(s[-1] + 1 + i) % 100 for i in range(k)]
    prefix = [0]
    plain = _plain_greedy(prefix, oracle, 20)
    spec = _spec_greedy(prefix, oracle, draft_fn, k=4, n_tokens=20)
    assert spec == plain


def test_matches_plain_greedy_bad_draft():
    # a draft that is always WRONG must still produce the identical stream (just no speedup).
    oracle = lambda s: (s[-1] + 1) % 100
    draft_fn = lambda s, k: [777 for _ in range(k)]   # garbage proposals
    prefix = [0]
    plain = _plain_greedy(prefix, oracle, 17)
    spec = _spec_greedy(prefix, oracle, draft_fn, k=4, n_tokens=17)
    assert spec == plain


def test_matches_plain_greedy_mixed_draft():
    # a draft right ~half the time: identity must still hold under partial acceptance.
    oracle = lambda s: (s[-1] * 3 + 1) % 257
    def draft_fn(s, k):
        out, last = [], s[-1]
        for i in range(k):
            last = (last * 3 + 1) % 257 if i % 2 == 0 else (last + 5) % 257  # half wrong
            out.append(last)
        return out
    prefix = [1]
    plain = _plain_greedy(prefix, oracle, 23)
    spec = _spec_greedy(prefix, oracle, draft_fn, k=3, n_tokens=23)
    assert spec == plain


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
