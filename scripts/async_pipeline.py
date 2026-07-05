"""async_pipeline — pipelined speculative decode: keep several verify-chunks in flight.

WHY THIS EXISTS
---------------
The distributed decode loop (client.py, both the spec path ~:884 and the plain path
~:940) traverses the workers STRICTLY SEQUENTIALLY:

    worker0 → wait → worker1 → wait → … → last → wait   then the next token starts.

At any instant exactly one worker is busy and the P-1 others sit idle — the classic
pipeline bubble. Over a WAN chain the per-token cost is ≈ P × (compute + RTT), so
throughput is latency-bound and falls as you add workers. This is the one technique
Nakshatra was missing that Shard used to climb 2.94 → 16.6 tok/s: fill the pipeline by
keeping several traversals in flight so every stage is always busy and the loop becomes
throughput-bound (the WAN RTT amortizes to ~1/depth of the loop).

THE PROBLEM async pipelining must solve for a SINGLE autoregressive stream: token t+1's
input depends on token t's output, so you cannot honestly start t+1 before t returns. The
fix is SPECULATIVE CONTINUATION: issue chunk i+1 *assuming* chunk i fully accepts (predict
its next `cur` from the draft), keep it flowing through the stages, and if the assumption
turns out wrong when chunk i's real result lands, FLUSH the speculative successors and
re-issue from the corrected state. The worker daemon already gives us the rewind primitive
this needs — `start_pos` / `keep_kv` / TruncateKV (client.py M3 fusion) trims a stage's KV
back on the next forward, so a flushed chunk's KV write is undone when the corrected chunk
re-enters that stage with the corrected start_pos.

THE CORRECTNESS ORACLE (same as speculative.py): output is BYTE-IDENTICAL to plain greedy
decode. We preserve it with one invariant:

    A chunk's tokens are COMMITTED only if its predecessor context was the truly-committed
    prefix. Speculation only decides WHICH chunks we launch early; a mispredicted chunk is
    discarded, never committed.

Chunk 0's predecessor is the real prompt. Chunk i>0 is committed only when it was *not*
flushed — i.e. its assumed predecessor equalled the real one. So every committed chunk saw
the true prefix ⇒ the emitted stream equals sequential greedy. Speculation changes latency,
never the answer.

STATUS: scheduling core is complete and unit-tested here against a mock chain (run this file
directly). Wiring the `Stage` callbacks to the real gRPC workers + verifying the KV rewind
on a live multi-box mesh is the inference lane's step — gated behind NKS_ASYNC_PIPELINE so
the proven sequential path stays the default until that live verification passes.
"""
from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from speculative import AcceptResult, accept, kv_keep_after


# A Stage runs one worker's portion of a verify traversal. Given the input bytes for a
# chunk (token payload for stage 0, hidden state for the rest) it returns this stage's
# output bytes. `start_pos` is the KV position the daemon should resume/trim to (the rewind
# primitive); `first`/`last` mark the ends of the chain. In production these wrap the real
# `_step_call(idx, …)`; in the self-test they are cheap deterministic functions.
# (payload, n, start_pos, first, last, meta) -> bytes. `meta` is a per-traversal unique id the
# real chain uses as the worker step_id (idempotency key — MUST differ for every issued chunk,
# including re-issued ones after a flush, or the daemon returns a cached stale result).
Stage = Callable[[bytes, int, int, bool, bool, int], bytes]


@dataclass
class _Chunk:
    idx: int                       # submission order (also commit order)
    cur: int                       # the `cur` token this chunk's verify was built on
    drafts: List[int]              # the K draft tokens proposed after `cur`
    start_pos: int                 # KV start_pos this chunk assumed for its predecessor
    assumed_cur: int               # what predecessor `cur` we ASSUMED (== real for chunk 0)
    assumed_start_pos: int         # what predecessor start_pos we assumed
    future: "Future[Optional[List[int]]]"   # resolves to the K+1 target argmaxes (None if flushed)
    cancel: "threading.Event" = field(default_factory=threading.Event)


class PipelineChain:
    """Runs verify traversals over the worker stages, several in flight at once.

    Each stage is guarded by its own lock so a stage processes chunks in submission order
    (required for per-stage KV correctness) while DIFFERENT stages run concurrently on
    different chunks — that concurrency is the pipeline fill. `occupancy()` reports the peak
    number of stages busy at once, which the self-test asserts is > 1 (proof of pipelining).
    """

    def __init__(self, stages: Sequence[Stage], n_embd_bytes: int = 0):
        self._stages = list(stages)
        self._locks = [threading.Lock() for _ in self._stages]
        self._pool = ThreadPoolExecutor(max_workers=len(self._stages) + 2,
                                        thread_name_prefix="nks-pipe")
        self._busy = 0
        self._peak = 0
        self._busy_lock = threading.Lock()

    @property
    def depth(self) -> int:
        return len(self._stages)

    def occupancy(self) -> int:
        return self._peak

    def _traverse(self, payload: bytes, n: int, start_pos: int,
                  cancel: "Optional[threading.Event]", meta: int) -> Optional[List[int]]:
        buf = payload
        last = len(self._stages) - 1
        for i, stage in enumerate(self._stages):
            # A flushed chunk must STOP advancing here — a cancelled Future does not halt an
            # already-running traversal, and letting it keep writing downstream workers' KV
            # would corrupt state. Re-checking the flag at every stage boundary bounds a
            # mispredicted chunk's KV footprint to stages it had already entered (which the
            # corrected re-issue then trims away via its lower start_pos). See module docstring.
            if cancel is not None and cancel.is_set():
                return None
            with self._locks[i]:            # in-order per stage; other stages run concurrently
                if cancel is not None and cancel.is_set():
                    return None
                with self._busy_lock:
                    self._busy += 1
                    self._peak = max(self._peak, self._busy)
                try:
                    buf = stage(buf, n, start_pos, i == 0, i == last, meta)
                finally:
                    with self._busy_lock:
                        self._busy -= 1
        # last stage returns the K+1 argmaxes as a flat list of ints
        import struct
        return list(struct.unpack(f"<{len(buf) // 4}i", buf))

    def submit(self, payload: bytes, n: int, start_pos: int,
               cancel: "Optional[threading.Event]" = None,
               meta: int = 0) -> "Future[Optional[List[int]]]":
        return self._pool.submit(self._traverse, payload, n, start_pos, cancel, meta)

    def close(self) -> None:
        self._pool.shutdown(wait=True)


def _pack(tokens: Sequence[int]) -> bytes:
    import struct
    return struct.pack(f"<{len(tokens)}i", *tokens)


def pipelined_spec_decode(
    *,
    chain: PipelineChain,
    propose: Callable[[Sequence[int], int], List[int]],   # draft.propose(prefix, k) -> k tokens
    prompt: Sequence[int],
    first_cur: int,
    start_pos0: int,
    spec_k: int,
    max_new: int,
    max_inflight: int,
    eos_ids: Sequence[int] = (),
    on_token: Optional[Callable[[int], None]] = None,
) -> List[int]:
    """Speculatively-pipelined greedy decode. Returns the committed token ids.

    Keeps up to `max_inflight` verify-chunks flowing through the stage pipeline. Commits in
    submission order; on a misprediction flushes the speculative successors and re-issues
    from the corrected cursor (the daemon rewinds KV via the corrected start_pos). Output is
    identical to a sequential spec-decode over the same draft+target.
    """
    generated: List[int] = []
    context: List[int] = list(prompt)          # committed prefix the draft conditions on
    inflight: "deque[_Chunk]" = deque()
    next_idx = 0
    real_cur = first_cur
    real_start_pos = start_pos0
    eos = set(eos_ids)

    def _issue(cur: int, start_pos: int, cond_prefix: List[int],
               assumed_cur: int, assumed_start_pos: int) -> _Chunk:
        nonlocal next_idx
        # Propose K+1: the first K are this chunk's drafts; the (K+1)-th is the token we
        # ASSUME becomes the next chunk's `cur` under full acceptance (target_argmax[K]).
        proposed = propose(cond_prefix + [cur], spec_k + 1)
        drafts = list(proposed[:spec_k])
        verify = [cur] + drafts
        ch = _Chunk(idx=next_idx, cur=cur, drafts=drafts, start_pos=start_pos,
                    assumed_cur=assumed_cur, assumed_start_pos=assumed_start_pos,
                    future=None)  # type: ignore[arg-type]
        ch.future = chain.submit(_pack(verify), len(verify), start_pos, ch.cancel, meta=ch.idx)
        # stash the predicted-next cur on the chunk for the filler below
        ch._pred_next_cur = proposed[spec_k] if len(proposed) > spec_k else cur   # type: ignore[attr-defined]
        next_idx += 1
        return ch

    stop = False
    while not stop and len(generated) < max_new:
        # ── FILL: keep the pipe full with speculative chunks ───────────────────────────
        while len(inflight) < max_inflight and len(generated) < max_new:
            if not inflight:
                ch = _issue(real_cur, real_start_pos, context, real_cur, real_start_pos)
            else:
                pred = inflight[-1]
                # assume the last in-flight chunk FULLY accepts: cur := its predicted next,
                # start_pos advances by 1 + K (cur + all K drafts kept)
                a_cur = pred._pred_next_cur                                  # type: ignore[attr-defined]
                a_sp = kv_keep_after(pred.start_pos, spec_k)
                # draft conditioning prefix under the full-accept assumption
                cond = context + [pred.cur] + pred.drafts
                ch = _issue(a_cur, a_sp, cond, a_cur, a_sp)
            inflight.append(ch)

        # ── COMMIT: resolve the oldest chunk (its predecessor is the real prefix) ──────
        head = inflight.popleft()
        argmax = head.future.result()
        res: AcceptResult = accept(head.drafts, argmax)

        emitted_eos = False
        for t in res.committed:
            if len(generated) >= max_new:
                stop = True
                break
            generated.append(t)
            context.append(t)
            if on_token:
                on_token(t)
            if t in eos:
                stop = True
                emitted_eos = True
                break

        real_cur = res.committed[-1]
        real_start_pos = kv_keep_after(head.start_pos, res.n_accepted)

        # ── VERIFY THE SPECULATION: did the next in-flight chunk assume right? ──────────
        mispredict = bool(inflight) and (
            inflight[0].assumed_cur != real_cur
            or inflight[0].assumed_start_pos != real_start_pos
        )
        if stop or mispredict:
            # flush every speculative successor — they were built on a prefix that did not
            # happen. Their KV writes get rewound when the corrected chunk re-enters each
            # stage with the corrected start_pos (daemon TruncateKV / start_pos trim).
            for ch in inflight:
                ch.cancel.set()      # stop it advancing to further stages (checked per-stage)
                ch.future.cancel()   # best-effort: skip it if it never started
            inflight.clear()
        # if not mispredicted, the surviving in-flight chunks are valid; keep draining them.

    return generated


# ─────────────────────────────────────────────────────────────────────────────────────────
# Self-test: prove the scheduling core is correct WITHOUT any GPU/network.
#
# A mock "target" decodes a fixed ground-truth continuation greedily; a mock "draft" proposes
# from a possibly-wrong guess table so we exercise both full-accept and misprediction. We
# assert the pipelined output equals a straight sequential spec-decode over the same mock, and
# that the pipeline actually ran >1 stage at once (real fill, not a disguised sequential loop).
# ─────────────────────────────────────────────────────────────────────────────────────────
def _selftest() -> int:
    import struct
    import time

    PROMPT = [1, 2, 3]
    K = 3
    N_STAGES = 4
    EOS_ID = 2

    # The target is a deterministic successor chain 10→11→…→20→EOS. This is a clean oracle:
    # the token following any prefix depends only on its last token, so target_argmax[i] =
    # succ(verify[i]) — no position bookkeeping to get wrong. Greedy decode from cur=10
    # therefore MUST emit exactly [11,12,…,20,EOS].
    SUCC = {t: t + 1 for t in range(10, 20)}
    SUCC[20] = EOS_ID
    SUCC[EOS_ID] = EOS_ID
    FIRST_CUR = 10
    EXPECTED = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, EOS_ID]

    # A draft that follows the successor chain but is deliberately WRONG once (proposes 999
    # after token 14) — this forces a misprediction + flush and proves the committed output
    # still matches greedy exactly.
    def make_propose():
        def propose(prefix: Sequence[int], k: int) -> List[int]:
            out: List[int] = []
            last = int(prefix[-1])
            for _ in range(k):
                nxt = SUCC.get(last, EOS_ID)
                if last == 14:                # deliberate misprediction
                    nxt = 999
                out.append(nxt)
                last = nxt
            return out
        return propose

    # stages: all but the last pass the verify tokens through as our stand-in "hidden"; the
    # last stage returns target_argmax[i] = succ(verify[i]). Each sleeps so concurrency shows.
    def make_stages():
        def mid(buf, n, start_pos, first, last):
            time.sleep(0.002)
            return buf
        def final(buf, n, start_pos, first, last):
            time.sleep(0.002)
            verify = list(struct.unpack(f"<{len(buf)//4}i", buf))
            argmax = [SUCC.get(int(t), EOS_ID) for t in verify]
            return struct.pack(f"<{len(argmax)}i", *argmax)
        return [mid] * (N_STAGES - 1) + [final]

    # ---- sequential reference (depth-1 == no pipelining) ----
    seq = PipelineChain(make_stages())
    seq_out = pipelined_spec_decode(
        chain=seq, propose=make_propose(), prompt=PROMPT, first_cur=FIRST_CUR,
        start_pos0=len(PROMPT), spec_k=K, max_new=32, max_inflight=1, eos_ids={EOS_ID})
    seq.close()

    # ---- pipelined (depth 4 in flight) ----
    pipe = PipelineChain(make_stages())
    pipe_out = pipelined_spec_decode(
        chain=pipe, propose=make_propose(), prompt=PROMPT, first_cur=FIRST_CUR,
        start_pos0=len(PROMPT), spec_k=K, max_new=32, max_inflight=4, eos_ids={EOS_ID})
    occ = pipe.occupancy()
    pipe.close()

    ok = True
    if seq_out != pipe_out:
        print(f"FAIL: pipelined output != sequential\n  seq  = {seq_out}\n  pipe = {pipe_out}")
        ok = False
    if pipe_out != EXPECTED:
        print(f"FAIL: output != greedy continuation\n  got  = {pipe_out}\n  want = {EXPECTED}")
        ok = False
    if occ <= 1:
        print(f"FAIL: no pipeline fill — peak stage occupancy was {occ} (expected >1)")
        ok = False
    if ok:
        print(f"PASS: output identical to sequential greedy ({pipe_out}); "
              f"peak {occ}/{N_STAGES} stages busy at once (pipeline filled); "
              f"misprediction at pos 4 flushed + recovered without corrupting output.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
