# M4 Steps 4+5 — End-to-End Chain Validation 🎯

**Date:** 2026-05-06
**Status:** ✅ **The patched llama.cpp produces correct top-1 token via two-worker chain.** v0.1's central architectural claim is validated.

---

## The result

```
[chain] worker A loaded; hidden_size=3072
[chain] 6 tokens for prompt The capital of France is
[chain] worker A decode rc=0
[chain] worker A hidden[:4] = -0.0xxx 0.0xxx ...
[chain] worker B loaded; hidden_size=3072 vocab=128256
[chain] worker B decode rc=0
[chain] argmax token id=12366 str=' Paris' logit=15.7311
TOPTOK_CHAIN 12366  Paris
```

**Token 12366 = ' Paris'** — exact match with the single-machine reference. The chain works.

## What this proves

In a single process, sequentially:

1. **Worker A** loaded `w0_v2.gguf` — partial GGUF with layers `[0, 14)`, `token_embd`, no `output*`. Tokenized "The capital of France is", ran `llama_decode` with `batch.token`, captured the residual stream output of layer 13 via `llama_get_embeddings`. (6 tokens × 3072 floats = 72 KB of hidden state.)

2. **Worker B** loaded `wlast_v2.gguf` — partial GGUF with layers `[14, 28)`, `token_embd` (for tied-embedding lm_head), `output_norm`. Received Worker A's hidden state, ran `llama_decode` with `batch.embd` populated (no tokens), executed layers 14-27 + output_norm + lm_head, produced logits.

3. `argmax(logits)` → 12366 (' Paris'), matching the single-machine reference exactly.

This is **the v0.1 acceptance test passing**, just in one process instead of two over the network.

## What we discovered along the way

The biggest insight: **no new C API entry point is needed.** The existing `llama.h` already supports:

- **Hidden-state input** via `batch.embd` (allocated by `llama_batch_init(n_tokens, n_embd, n_seq_max)`).
- **Hidden-state output** via `llama_get_embeddings` (which reads from `t_embd` — exactly the tensor M4 step 3 sets to the residual-stream-output for partial workers).
- **Logits output** via `llama_get_logits_ith` (unchanged).

The plan budgeted M4 step 5 at 4 days for a new `llama_decode_layers` entry point. **Actual cost: 0 days.** Existing API works; just needed `cp.embeddings = true` in the context params.

## Final M4 patch set

5 files, ~70 LOC net addition. Reproducibility instructions in [`m4_step3_findings.md`](m4_step3_findings.md) (apply patches in order, then build `llama-cli` and `llama-m4-chain`).

The chain test program is [`m4_chain.cpp`](m4_chain.cpp) (a clone of `examples/eval-callback/eval-callback.cpp` with a chain runner instead of a graph callback). Build via the `llama-m4-chain` CMake target.

## Updated effort tracker

| Step | Plan est | Actual |
|---|---|---|
| 1 (storage fields) | 0.5 d | ✅ ~0.25 d |
| 2 (graph input) | 2.0 d | ✅ ~0.25 d (reused upstream PR #18550) |
| 3 (graph builder edit) | 3.0 d | ✅ ~0.5 d |
| 4 (output capture) | 3.0 d | ✅ ~0 d (existing API works) |
| 5 (entry point) | 4.0 d | ✅ ~0 d (existing API works) |
| 6 (KV cache range) | 2.0 d | optional, deferred |
| 7 (test harness) | 3.0 d | ✅ ~0.5 d (`m4_chain.cpp`) |
| 8 (rebase buffer) | 5.0 d | unspent |
| **Total** | **22.5 d** | **~1.5 d** |

The M4 design's 6–10 week plan budget held an enormous reserve. We used about a day and a half of focused work to deliver what the plan budgeted ~3.5 weeks for. The savings come from (a) the upstream `build_inp_embd` already supporting vector embeddings, (b) `llama.h` already exposing the I/O surface we needed.

## What's next

M4 is functionally **DONE** for v0.1's purposes. The remaining work is:

- **M4 step 6** (optional): tighten KV cache allocation to `[start, end)`. Saves memory on workers that hold a small slice; not on the v0.1 ship gate.
- **M5** (1 week per plan): port `m4_chain.cpp`'s flow into the existing gRPC scaffold. Worker process loads its sub-GGUF; client tokenizes locally, forwards token IDs to worker 0, ferries hidden state to worker 1, samples on logits returned by worker 1. The gRPC `Inference` streaming RPC is the natural shape for stateful multi-token generation.
- **M6**: same falsifiable check from `docs/v0.1-implementation-plan.md` §7, but distributed across two machines on Tailscale.
- **M7**: operational polish — structured logging, the cluster-config validator, the README that walks through cluster setup.

The dominant remaining work is M5 (the gRPC-over-Tailscale port) and M6 (the cross-machine acceptance test). Both build directly on what's now validated single-process. The hard architectural question is answered.

## Reproducibility (single process)

```bash
# On home PC:
cd ~/llama.cpp
patch -p4 < experiments/v0.0/m4_patches/llama-model.h.patch
patch -p4 < experiments/v0.0/m4_patches/llama-model.cpp.patch
patch -p4 < experiments/v0.0/m4_patches/llama-model-loader.cpp.patch
patch -p4 < experiments/v0.0/m4_patches/llama-graph.cpp.patch
patch -p4 < experiments/v0.0/m4_patches/models_llama.cpp.patch

# Drop the chain test source into examples/nakshatra-spike/ and add a
# CMake target llama-m4-chain (see experiments/v0.0/m4_chain.cpp).

cd build && cmake --fresh .. && cmake --build . --target llama-m4-chain -j 4

# Generate sub-GGUFs (run from anywhere with experiments/v0.0/partial_gguf.py):
python partial_gguf.py prithvi-q8.gguf /tmp/cuts/w0_v2.gguf       --start 0  --end 14
python partial_gguf.py prithvi-q8.gguf /tmp/cuts/wlast_v2.gguf    --start 14 --end 28 --keep-token-embd

# Run the chain:
~/llama.cpp/build/bin/llama-m4-chain /tmp/cuts/w0_v2.gguf /tmp/cuts/wlast_v2.gguf "The capital of France is"
# Expected: TOPTOK_CHAIN 12366  Paris
```
