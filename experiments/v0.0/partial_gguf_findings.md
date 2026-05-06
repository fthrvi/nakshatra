# Phase 0a — Partial-GGUF Load Test, Findings

**Date:** 2026-05-06
**Status:** RESOLVED. Outcome: hard load error → **Path B-prime confirmed**.

Operationalises [`docs/v0.0-validation-plan.md`](../../docs/v0.0-validation-plan.md) §"Phase 0a — Partial-GGUF load test", which itself derives from [`docs/path-a-vs-path-b-memo.md`](../../docs/path-a-vs-path-b-memo.md) §1.6.

---

## Setup

| | Value |
|---|---|
| Host | `MentoringInstitute@bishwa` (Tailscale `100.109.164.69`) |
| Hardware | Intel iMac, AMD Radeon Pro 5700 XT, 64 GB RAM, macOS 26.2 |
| llama.cpp | version 8142 (commit `8c2c0108d`), Vulkan/MoltenVK build |
| Source GGUF | `/Users/MentoringInstitute/models/llama-3.3-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf` (40 GB, 80 blocks, 724 tensors) |
| Script | [`partial_gguf.py`](partial_gguf.py) using `gguf-py` from llama.cpp source tree |
| Cut | `--keep 20` (retain blocks `blk.0`–`blk.19`, drop `blk.20`–`blk.79`) |

The memo's central claim under test (§1.5): llama.cpp's loader walks the layer count from metadata without partial-load support, and hard-fails when expected tensors are absent. The experiment falsifies or confirms that claim with a real GGUF.

---

## Run 1 — drop output head AND layers `[20, 80)`

This is the "Nakshatra worker" simulation: a worker that owns only the first 20 blocks would not have `output.weight` or `output_norm.weight` either.

```
$ python partial_gguf.py Llama-3.3-70B-Instruct-Q4_K_M.gguf partial.gguf --keep 20
[arch]     llama
[blocks]   src=80, keeping=[0, 20)
[tensors]  src=724  keeping=182  dropping=542
[kvs]      copied=35  skipped=0
[done]     wrote partial.gguf  (10 GB)
```

Then:

```
$ llama-cli -m partial.gguf -p "Hello" -n 1
...
llama_model_load: error loading model: missing tensor 'output_norm.weight'
llama_model_load_from_file_impl: failed to load model
common_init_from_params: failed to load model 'partial.gguf'
Failed to load the model
```

**Result: hard error at load**, before any inference attempt.

Full log: [`partial_load_attempt.log`](partial_load_attempt.log).

---

## Run 2 — drop ONLY layers `[20, 80)` (keep output head)

This isolates the "missing layer tensors" failure mode from the "missing output head" failure mode. Only the canonical memo §1.6 case (a `blk.N.*` tensor missing).

```
$ python partial_gguf.py Llama-3.3-70B-Instruct-Q4_K_M.gguf partial_layersonly.gguf --keep 20 --keep-output
[arch]     llama
[blocks]   src=80, keeping=[0, 20)
[tensors]  src=724  keeping=184  dropping=540
[kvs]      copied=35  skipped=0
[done]     wrote partial_layersonly.gguf  (11 GB)
```

Then:

```
$ llama-cli -m partial_layersonly.gguf -p "Hello" -n 1
...
llama_model_load: error loading model: missing tensor 'blk.20.attn_norm.weight'
llama_model_load_from_file_impl: failed to load model
Failed to load the model
```

**Result: hard error at load** on the first missing layer tensor (`blk.20.attn_norm.weight`).

Full log: [`partial_load_layersonly.log`](partial_load_layersonly.log).

The memo predicted `missing tensor: blk.20.attn_q.weight` specifically. The observed tensor is `blk.20.attn_norm.weight` — same block, different tensor within it. The loader walks per-block tensors alphabetically (`attn_norm` < `attn_q`), so this is the same prediction in spirit; the *exact* missing tensor depends on iteration order, not on architecture.

---

## Interpretation

Both runs land squarely in the **"Hard error at load"** branch of [`docs/v0.0-validation-plan.md`](../../docs/v0.0-validation-plan.md) §"Phase 0a — Falsifiable outcomes":

> Hard error (e.g. `missing tensor: blk.20.attn_q.weight`) at load → Path B-prime confirmed. Patched `llama_decode` is the right path.

The loader behaves exactly as the memo §1.5 inferred from reading `llama.h` and `src/llama.cpp`: it walks the architecture's expected tensor list (driven by `llama.block_count=80`) and hard-fails on the first missing one, with no recovery, no fallback, no "this slot is hosted elsewhere" semantics. There is no usable partial-model load path in upstream llama.cpp.

This means a Nakshatra worker that holds only a layer range CANNOT use vanilla `llama_model_load_from_file` — the loader will reject any GGUF that doesn't have the full tensor set declared in metadata. The memo's recommended fix (Path B-prime: patch `llama_decode` to accept hidden-state input/output, and bypass the model loader's full-set assumption) is therefore necessary, not optional.

---

## Decision

**Greenlight v0.1's Path B-prime C++ work.** Phase 0a's central question is resolved: the patched-`llama_decode` approach is on the v0.1 critical path because no cheaper alternative exists in upstream llama.cpp.

Phase 0b (the `cb_eval` orchestration spike) can proceed independently — its design does not depend on Phase 0a's outcome, and its purpose is to validate the orchestration protocol, not the loader.

---

## Adjacent observations (not load-bearing for the decision)

- **Loader fails fast.** No partial-progress; the failure is at GGUF metadata-vs-tensor reconciliation time, before any compute graph is built. This is operationally good — a Nakshatra worker that misconfigures its sub-GGUF will fail loudly at startup, not silently produce wrong outputs.
- **Tensor iteration order is alphabetical within a block.** `attn_norm` comes before `attn_q`, so the first complaint for a missing block is `attn_norm`. Useful to know when reading future loader errors.
- **35 KV metadata fields copied with 0 skipped.** `gguf-py`'s `add_*` typed methods cover everything in a Llama-3.3-70B GGUF without falling back to a generic path. The script is simple and reusable for v0.1's pre-split GGUF tooling.
- **Sub-GGUF size scales linearly with kept layers.** 20 of 80 layers + embedding + output = 11 GB out of a 40 GB source. Confirms the v0.1 plan to pre-split GGUFs into worker-sized shards is space-efficient.

---

## Reproducibility

The full reproduction recipe lives in [`partial_gguf.py`](partial_gguf.py) and [`docs/v0.0-validation-plan.md`](../../docs/v0.0-validation-plan.md). To rerun on `bishwa`:

```bash
ssh MentoringInstitute@bishwa
cd ~/nakshatra-v0 && source venv/bin/activate
python partial_gguf.py \
  /Users/MentoringInstitute/models/llama-3.3-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
  /tmp/partial_layersonly.gguf --keep 20 --keep-output
~/llama.cpp/build/bin/llama-cli -m /tmp/partial_layersonly.gguf -p "Hello" -n 1
```

Expected: `error loading model: missing tensor 'blk.20.<some attn or ffn tensor>'`. If the error message changes shape (different prefix, different failure point), llama.cpp upstream may have evolved its loader behavior — re-evaluate against this findings doc.
