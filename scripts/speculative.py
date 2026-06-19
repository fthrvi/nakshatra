"""
speculative.py — coordinator-side speculative decoding for the distributed chain.

Slice 1 of the speed stack (docs/2026-06-19-speed-stack-plan.md). This module is the
DAEMON-INDEPENDENT core: a local draft model proposes K tokens, and `accept()` decides —
given the distributed target's per-position greedy argmaxes — how many to commit. It is
pure and exhaustively unit-tested so the algorithm is correct before any C++/GPU work.

It is INERT: nothing imports it unless `NAKSHATRA_SPECULATIVE=1` wires it into the
client.py decode loop (a later step, gated on the daemon patch below).

── Why a daemon patch is still required (the real blocker, see plan §"Slice 1") ──
Greedy speculative verification feeds [cur, d0..d_{K-1}] through the chain in ONE traversal
and needs the target's argmax at EACH of the K+1 positions, then must DISCARD the KV of the
rejected tail. Today the patched `llama-nakshatra-worker` daemon:
  • computes argmax only at the last position  (worker_daemon.cpp:337/355/393), and
  • has no KV-truncation primitive            (KV only ever appends).
So this module's `accept()` is ready, but the live wiring waits on:
  (D1) last worker returns K+1 argmaxes (set batch.logits[i]=1 ∀i; loop argmax per position),
  (D2) a new daemon command KV_TRUNCATE(seq, n_keep) -> llama_kv_cache_seq_rm(...),
  (D3) worker.py: relax the len==4 response assumption; expose a TruncateKV RPC.
This file owns the math; those own the engine.

The contract, precisely
------------------------
Plain decode feeds [cur] at position `start_pos` and reads 1 argmax. Speculative decode feeds
[cur, d0, ..., d_{K-1}] (K+1 tokens) at [start_pos, start_pos+K+1) and reads K+1 argmaxes
t0..tK, where ti = the target's greedy next-token after seeing (cur, d0..d_{i-1}). Then:
  - ti is compared to di; on the first i where ti != di the draft is wrong → commit ti
    (the correction) and stop. All earlier drafts matched and are committed (= di).
  - if every di matched, commit the bonus tK as well.
Output is byte-identical to plain greedy decode — that identity is the correctness oracle.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AcceptResult:
    committed: List[int]   # new token ids to append to `generated` this step (len == n_accepted + 1)
    n_accepted: int        # how many of the K drafts the target agreed with (0..K)

    @property
    def n_committed(self) -> int:
        return len(self.committed)


def accept(drafts: Sequence[int], target_argmax: Sequence[int]) -> AcceptResult:
    """Greedy speculative acceptance.

    drafts        : the K token ids the draft model proposed (d0..d_{K-1}).
    target_argmax : the K+1 greedy argmaxes the distributed target produced for the
                    verify traversal of [cur, d0..d_{K-1}]  (t0..tK).

    Returns the tokens to commit and how many drafts were accepted. Always commits at
    least one token (the t0 correction even on an immediate reject), so decode always
    makes progress — exactly like plain decode commits one token per step.
    """
    K = len(drafts)
    if len(target_argmax) != K + 1:
        raise ValueError(f"target_argmax must have len(drafts)+1 = {K + 1}, got {len(target_argmax)}")
    committed: List[int] = []
    for i in range(K):
        committed.append(int(target_argmax[i]))   # commit the target's token (== di when they agree)
        if target_argmax[i] != drafts[i]:
            return AcceptResult(committed=committed, n_accepted=i)   # i matched, then 1 correction
    committed.append(int(target_argmax[K]))        # all K matched → take the bonus token
    return AcceptResult(committed=committed, n_accepted=K)


def kv_keep_after(start_pos: int, n_accepted: int) -> int:
    """KV positions to KEEP after a verify step (the rest must be truncated).

    The verify wrote K+1 positions at [start_pos, start_pos+K+1): `cur` plus K drafts.
    `cur` and the `n_accepted` matched drafts have valid KV; the first rejected draft (and
    everything after it) has invalid KV. The committed correction/bonus token is an OUTPUT —
    it has no KV yet and is fed next step as the new `cur`. So we keep cur + accepted drafts:
        keep = start_pos + 1 + n_accepted
    and the next step's start_pos becomes this value. (Bonus case n_accepted==K keeps all
    K+1 positions → no truncation needed, which this formula yields naturally.)
    """
    return start_pos + 1 + n_accepted


def next_start_pos(start_pos: int, n_accepted: int) -> int:
    """The decode loop's `prefix_length` for the following step (== kept KV length)."""
    return kv_keep_after(start_pos, n_accepted)


class DraftModel:
    """A small local llama.cpp model on the coordinator that proposes K greedy tokens.

    Lives alongside the existing `llama` handle in client.py's decode loop (client.py:107-110,
    used through the loop). Loaded with full weights (unlike the vocab_only tokenizer handle).
    Greedy (temp 0) so its proposals are deterministic and match the target's greedy regime.

    Kept deliberately thin and lazy-importing llama_cpp so importing this module never pulls
    a heavy dependency into paths that only need `accept()`.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1,
                 seed: int = 0, verbose: bool = False):
        from llama_cpp import Llama  # lazy: only when a real draft is constructed
        # logits_all=True is REQUIRED: in llama-cpp-python 0.3.x, eval() only writes per-token
        # logits into .scores when logits_all is set (the logits_all=False branch is a `pass`),
        # so greedy argmax off .scores returns garbage without it. n_ctx is kept modest to bound
        # the (n_ctx × n_vocab) scores allocation.
        self._llama = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                            logits_all=True, seed=seed, verbose=verbose)
        self.model_path = model_path

    def propose(self, prefix_tokens: Sequence[int], k: int) -> List[int]:
        """Greedily propose the next k tokens following `prefix_tokens`.

        Uses the draft's own KV via incremental eval. Greedy argmax each step. Returns up to
        k token ids (fewer only if the draft emits EOS, which the caller may treat as a hint).
        """
        import numpy as np
        llama = self._llama
        llama.reset()
        llama.eval(list(prefix_tokens))
        out: List[int] = []
        for _ in range(k):
            logits = llama.scores[llama.n_tokens - 1]   # last-token row (logits_all=True)
            tok = int(np.argmax(logits))
            out.append(tok)
            llama.eval([tok])
        return out

    def close(self) -> None:
        try:
            self._llama.close()
        except Exception:
            pass


def speculative_round(draft: DraftModel, prefix_tokens: Sequence[int], k: int,
                      verify_fn) -> Tuple[AcceptResult, List[int]]:
    """One speculative step, transport-agnostic (verify_fn injected for testing).

    draft        : the local draft model.
    prefix_tokens: all committed tokens so far (ending in `cur`).
    k            : draft depth K.
    verify_fn    : callable([cur, *drafts]) -> List[int] of K+1 target argmaxes. In production
                   this runs the distributed chain traversal; in tests it's a fake target.

    Returns (AcceptResult, drafts) — the caller appends result.committed to `generated`,
    truncates worker KV to kv_keep_after(start_pos, result.n_accepted), and continues.
    """
    cur = prefix_tokens[-1]
    drafts = draft.propose(prefix_tokens, k)
    target_argmax = verify_fn([cur, *drafts])
    return accept(drafts, target_argmax), drafts
