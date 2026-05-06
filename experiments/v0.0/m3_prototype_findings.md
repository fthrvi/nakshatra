# M3 Loader Patch — Prototype Findings

**Date:** 2026-05-06
**Status:** Prototype validated. Loader patch shape works; downstream code requires M4's decode patch to complete the chain. Source has been reverted; no patches currently in tree.

Operationalises the M3 milestone from [`docs/v0.1-implementation-plan.md`](../../docs/v0.1-implementation-plan.md) §4. Real M3 deliverable is a clean, metadata-KV-driven patch covering all Llama-family architectures; this doc is the time-boxed *prototype* that validated the patch shape with a hardcoded range and only the LLAMA case.

---

## What the prototype did

A single edit to `~/llama.cpp/src/llama-model.cpp`, in the `LLM_ARCH_LLAMA` (and shared) case of `llama_model::load_tensors`. The patch:

1. Hardcoded `nks_layer_start = 0`, `nks_layer_end = 14` (real M3 reads from `nakshatra.layer_range` metadata KV).
2. Made `tok_embd` creation conditional on `nks_layer_start == 0`.
3. Made `output_norm` and `output` creation conditional on `nks_layer_end >= n_layer`.
4. Added `if (i < nks_layer_start || i >= nks_layer_end) continue;` at the top of the per-layer for-loop.
5. Added one `LLAMA_LOG_INFO` line announcing the partial-load mode.

Total: ~25 lines added, 0 deleted. Surgical.

The patched binary was built (`cmake --build . --target llama-cli`) and tested against a sub-GGUF produced by `experiments/v0.0/partial_gguf.py --keep 14` on the 28-layer `prithvi-q8.gguf`. The sub-GGUF contains:

- `token_embd.weight` ✓
- `rope_freqs.weight` ✓
- `blk.[0, 14).*` ✓ (all per-layer tensors)
- `output.weight` ✗ (dropped)
- `output_norm.weight` ✗ (dropped)
- `blk.[14, 28).*` ✗ (dropped)

## Result

Patched `llama-cli` on the partial GGUF: **exit code 139 (segfault)**.

Crucially, the segfault is **not** the previous "missing tensor 'blk.14.attn_norm.weight'" load error from Phase 0a. The output progressed past `Loading model... |-` (the spinner), meaning the loader's missing-tensor checks DID accept the partial file. The crash happens later — almost certainly in context init / graph build, which still assumes `output_norm`, `output`, and the missing layer structs are present and dereferenceable.

That is exactly the boundary the v0.1 plan draws between **M3 (loader patch)** and **M4 (decode patch)**:

- M3 makes the loader accept partial GGUFs. **Validated by this prototype.**
- M4 makes the rest of the inference pipeline (graph builder, context init, KV cache, lm_head) handle partial layer ranges gracefully. **Not in scope here.**

## Patch shape (for the real M3)

The clean version of this patch should:

1. **Add a metadata KV constant** `LLM_KV_NAKSHATRA_LAYER_RANGE` (or two scalars `*_LAYER_RANGE_START` and `*_LAYER_RANGE_END`). Editing:
   - `src/llama-arch.h` — declare the enum member
   - `src/llama-arch.cpp` — string-name mapping
2. **Read the range early in `load_tensors`** (before the `switch (arch)`). Default `[0, n_layer)` means "full model, normal behavior — no partial load."
3. **Apply the range conditionals to every architecture case** that follows the same pattern (LLAMA, MISTRAL3, REFACT, MINICPM, GRANITE family, etc.). Most Llama-family cases share the same shape; a small helper macro can keep the per-case diff to ~3 lines each.
4. **Reject partial GGUFs without the metadata KV** with a clear error: this prevents accidentally accepting a malformed file.

For v0.1 critical path, only Llama-family architectures need the patch — the chosen v0.1 model will be Llama-family per the plan §9 open question.

## Estimated effort for clean M3

- Metadata KV plumbing: 0.25 day
- Range-read logic in `load_tensors`: 0.5 day
- Per-architecture conditional rollouts (Llama family): 1 day
- Test harness using the sub-GGUF tool: 0.5 day
- Negative test (unpatched GGUF rejected, partial-without-KV rejected): 0.25 day

**Subtotal: ~2.5 days.** Sits well within the plan's 1-2 week M3 budget — leaves room for unexpected refactors in upstream llama.cpp during the implementation window.

## What's NOT in M3

The segfault we observed proves: the `output_norm`, `output`, and missing layer structs are dereferenced somewhere downstream. M3's responsibility is *only* to let the model file load without throwing. M4 handles:

- Graph builder skipping out-of-range layers (`src/llama-graph.cpp` and per-arch model files)
- Context init not allocating KV cache for layers we don't have (`src/llama-context.cpp`)
- New entry point `llama_decode_layers(ctx, hidden_in, layer_start, layer_end, hidden_out)` that bypasses the embedding lookup and lm_head when those tensors are absent

That's the 6-10 weeks of work in M4. The good news: the M3 prototype confirms M3 itself is small and contained, so the timeline buffer can be allocated to M4.

## Reproducibility

The prototype patch is NOT in the tree. To reproduce the experiment, apply the diff fragment described in §"What the prototype did" to a checkout of llama.cpp at commit `c46583b` (the home PC's current tip), then build `llama-cli` and run against a sub-GGUF produced by:

```bash
python experiments/v0.0/partial_gguf.py \
  /home/prithvi/prithvi/training/prithvi-merged/prithvi-q8.gguf \
  /tmp/prithvi_q8_partial.gguf --keep 14
```

Expected: clean unpatched llama.cpp throws `missing tensor 'blk.14.attn_norm.weight'`. Patched llama.cpp progresses past the loader and segfaults later (until M4 lands).
