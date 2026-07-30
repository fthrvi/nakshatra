# The three-card matrix — the definitive measurement (2026-07-29 ~23:55)

Biswa: *"if we use all three cards can we test both async and speculative, with enough space
for the drafter"* — yes, and it changed every number. Prithvi was told first, consented
("yes do it now; I won't even notice"), and was fully paused (ears + gateway stopped, all
weights unloaded) so nothing could summon a model mid-measurement. He was woken with the
results immediately afterwards.

**Layout:** 7900 XT (20 GB) = draft ALONE · 9070 XT (16 GB) = chain L0-15 · RTX 3060 (12 GB) =
chain L15-48. Qwen3-30B-A3B Q3, 96 tokens, same prompt, dual-arch ROCm wheel
(`-DAMDGPU_TARGETS=gfx1100;gfx1201` — the gfx1201-only wheel segfaults on the 7900 XT).

| run | tok/s | worker RPC | earlier (draft sharing a card) |
|---|---|---|---|
| **plain streaming** | **47.88** | 1.99 s (~100 % of wall) | 45-53 |
| stream-spec | 32.76 | 1.57 s | 16.09 → **2.0×** |
| unary spec | 32.67 | 1.60 s | 12.20 → **2.7×** |
| async pipelined | 23.82 | 3.36 s | 6.60 → **3.6×** |

All three spec variants byte-identical (`acbf8f81c3`); plain differs by the documented
batch-shape GPU non-determinism. Draft latency alone on the idle 7900 XT: **21 ms per K=4
(5.3 ms/token)**.

## 1. The contention hypothesis was RIGHT — and worth 2-3.6×

Giving the drafter its own card doubled speculation and more than tripled pipelining. Every
earlier "speculation is hopeless here" number was partly an artefact of the draft fighting a
resident worker slice for the same GPU. That is now corrected in the record.

## 2. Plain streaming still wins — and the arithmetic finally closes

Per round with K=4 and ~2.2 tokens accepted: 21 ms draft + ~36 ms verify pass ≈ 57 ms for 2.2
tokens = **26 ms/token**. Plain streaming: ~20 ms/token. Speculation loses by exactly that
margin, and the books now balance — stream-spec wall 2.93 s = 1.57 s workers + 0.92 s drafting
+ ~0.44 s orchestration. Nothing unexplained remains.

## 3. The load-bearing reason: a K-token verify pass is NOT free on this model

Speculation assumes verifying K+1 tokens costs about the same as one token (memory-bandwidth
bound: the weights stream once either way). Measured here, per-stage time went **11 ms
(1 token) → ~26 ms (5 tokens)**. The batch is *not* free, because **Qwen3-30B-A3B activates
only ~3 B parameters per token** — there is little weight-streaming cost to amortise, so the
pass is compute-bound rather than bandwidth-bound and scales with tokens.

**Speculation is a trick for slow, heavy targets. This target is already fast.** That is a
property of the model, not of the transport (fixed), the draft device (fixed), or the code
(correct, byte-identity proven three times on two models).

## What this settles

- **Chain serving: plain streaming, 48 tok/s.** Final, on all evidence.
- `--stream-spec`, `--speculative`, `NKS_ASYNC_PIPELINE`: correct, merged, default OFF.
- The honest re-test that could flip this: a **dense** target big enough to be
  bandwidth-bound per token (the 70 B split, Sthambha Stage 2) — note the dense 32 B was
  *not* enough (17.19 plain vs 8.59 spec, `dense-target-spec-test.md`), so the bar is higher
  than "dense".
- **EAGLE-style drafting** (`eagle_hidden` capability + `--eagle-head/--eagle-base` already in
  this repo) remains the one structural idea untested: it removes the separate draft forward
  entirely rather than making it cheaper.
