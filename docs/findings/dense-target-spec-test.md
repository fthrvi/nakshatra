# The dense-target test — prediction FALSIFIED, and the real culprit found (2026-07-29 ~23:30)

Tonight's stream-spec finding predicted: *"speculation loses on Qwen3-30B-A3B because an MoE
with ~3 B active params is already draft-speed; it should pay on a DENSE target."*
We tested that immediately — no 70 B is on disk, but **Qwen3-32B Q4_K_M (dense, 32 B ACTIVE)**
was already sliced across both boxes (`~/.nakshatra/qwen-chain.yaml`, hub L0-38 / ijru L38-64),
which tests the same hypothesis with a 10× swing in the target/draft ratio.

## Result: the prediction is WRONG

| run | tok/s | worker RPC time | chain calls (64 tokens) | output_sha256 |
|---|---|---|---|---|
| plain streaming | **17.19** | 3.70 s (~100 % of wall) | 64 | `5be717e62f7c` |
| stream-spec (K=4) | **8.59** | 2.38 s (~32 % of wall) | **23** | `5be717e62f7c` |

Byte-identical output — the correctness spine held again on a completely different model.

**But look at what speculation actually achieved**: 64 tokens in **23 chain round-trips**
(2.8 tokens accepted per round, frequently `accept 4/4`) and **35 % less total worker time**
(3.70 s → 2.38 s). The distributed part of the system did *measurably less work*, exactly as
the technique promises. The run was still 2× slower overall because **~5 s of its 7.45 s wall
clock was the client's own draft model**.

## The real culprit: draft LATENCY, not the target/draft parameter ratio

Per-proposal the 0.6 B draft is costing **~50 ms** — about the same as one full pass through a
32 B model split across two GPUs (26 ms + 29 ms). That is absurd on its face for a 0.6 B model
and is the actual bug in the loop. Likely contributors, in order of suspicion:
1. **GPU contention** — the draft shares the 9070 XT with a 12.4 GB resident worker slice.
2. **Per-call overhead** in llama-cpp-python (Python round-trip per single-token proposal, no
   batching, context re-established each call).
3. Draft context sized 4096 while only ~100 tokens are ever used.

So the corrected conclusion, now measured on both an MoE and a dense target:
**speculation's *chain-side* economics work (fewer round trips, less worker time, identical
output); its *client-side* drafting is what loses.** Bigger targets will not fix a 50 ms draft —
at 32 B dense the target was already ~55 ms/token and it still lost.

## What would actually make speculation pay here

1. **EAGLE-style drafting** — this repo already has the hooks (`eagle_hidden` worker capability,
   `--eagle-head/--eagle-base/--eagle-config` client flags). A draft head that reuses the target's
   hidden states removes the separate-model forward entirely; that is the structural fix, not a
   faster GPU.
2. **A dedicated device for the draft** — measure it first: time `DraftModel.propose()` in
   isolation, on an idle card, before blaming contention. (Note: the current ROCm wheel is
   `gfx1201`-only, so the 7900 XT segfaults — rebuild with both targets to try this.)
3. **Batch/persist the draft context** rather than re-entering llama-cpp-python per token.

**Unchanged operational recommendation:** plain streaming for chain serving. The stream-spec
code stays (merged, flag-gated OFF, byte-identity proven twice on two different models) — it is
correct and its chain-side numbers are good; it is waiting on a drafting path worth its price.

## Reproduce
Same as `cuda-chain-51-tok-s.md` but with `--config ~/.nakshatra/qwen-chain.yaml`,
`--model-path ~/.nakshatra/models/Qwen_Qwen3-32B-Q4_K_M.gguf`, hub port 5560 / ijru 5570, and
**`--n-ctx 1024` on the hub worker** — at 4096 the 38-layer slice OOMs the 9070 XT during
compute-buffer reservation (`failed to allocate compute pp buffers`), which is the same
~476 MiB scratch-beyond-weights effect documented in `rebalance-q3-split.md`.
