# M3 Loader Patch — Findings

**Date:** 2026-05-06
**Status:** Loader patch complete and validated. Source is reverted on the home PC; the patch lives in this directory as a documentation artifact and as the input to a future `llama.cpp` fork branch.

Operationalises the **M3** milestone from [`docs/v0.1-implementation-plan.md`](../../docs/v0.1-implementation-plan.md) §4. Supersedes [`m3_prototype_findings.md`](m3_prototype_findings.md) (which captured an earlier hardcoded prototype).

---

## What M3 delivered

A clean, metadata-KV-driven patch to `llama.cpp` that lets `llama_model_loader::load_tensors` accept partial GGUFs declaring a Nakshatra layer range. Two source files touched:

| File | Lines added | Lines removed | Purpose |
|---|---|---|---|
| `src/llama-model.cpp` | ~30 | ~10 | Partial-load logic in the LLAMA-family case of `load_tensors` |
| `src/llama-model-loader.cpp` | 2 | 0 | Explicit template instantiations for `get_key<bool>` and `get_key<uint32_t>` over the string-keyed overload |

**Total patch size: ~32 lines added, ~10 removed across 2 files.** Within the v0.1 plan's M3 budget of 1–2 weeks, comfortably.

The patch is preserved in this directory at:

- [`m3_patches/llama-model.cpp.patch`](m3_patches/llama-model.cpp.patch)
- [`m3_patches/llama-model-loader.cpp.patch`](m3_patches/llama-model-loader.cpp.patch)

These are unified diffs against `~/llama.cpp` at commit `c46583b`. To re-apply on a fresh checkout: `patch -p4 < experiments/v0.0/m3_patches/llama-model.cpp.patch` (or use the diff manually).

The companion change is in [`partial_gguf.py`](partial_gguf.py): four `nakshatra.*` metadata KVs are now written to every sub-GGUF the script produces.

## Metadata KVs introduced

Written by `partial_gguf.py`, read by patched `load_tensors`:

| Key | Type | Meaning |
|---|---|---|
| `nakshatra.layer_range_start` | `uint32` | First layer index this sub-GGUF holds (inclusive) |
| `nakshatra.layer_range_end` | `uint32` | One-past-last layer index this sub-GGUF holds (exclusive) |
| `nakshatra.has_token_embd` | `bool` | Whether `token_embd.weight` is present (worker 0) |
| `nakshatra.has_lm_head` | `bool` | Whether `output_norm.weight` + `output.weight` are present (last worker) |

Defaults preserve upstream behavior on full GGUFs:

- If `layer_range_*` are absent, the patched loader assumes `[0, n_layer)` (the full model).
- If `has_token_embd` / `has_lm_head` are absent, they default to `(layer_start == 0)` / `(layer_end >= n_layer)` respectively.

A normal (non-Nakshatra) full GGUF therefore loads with **unchanged behavior** under the patched binary. Only sub-GGUFs that *declare themselves* Nakshatra-shaped opt into the partial-load path.

## Validation

Test setup: `prithvi-q8.gguf` (Llama-3.2-3B, 28 blocks, Q8_0). Sub-GGUF cut at `[0, 14)` with `output.weight` and `output_norm.weight` dropped (worker 0 shape).

Probes inserted at four points in the patched section confirmed the patch fires correctly:

```
### NAKSHATRA-PROBE: entering LLAMA case in load_tensors, n_layer=28 ###
### NKS-1: before KV read ###
### NKS-2: after KV read, partial=1 start=0 end=14 embd=1 lm=0 ###
### NKS-3: after embd, before output ###
### NKS-4: before per-layer loop ###
```

All four probes fire in order. KV reads return the correct values (`partial=1, start=0, end=14, embd=1 (worker 0), lm=0 (no lm_head)`). The for-loop's `if (i < nks_start || i >= nks_end) continue;` correctly skips creation of `blk.[14, 28).*` tensors. The patched binary then segfaults *downstream of `load_tensors`* — exactly the M3-vs-M4 boundary.

The same partial GGUF run against an **unpatched** `llama-cli` from earlier in the project produces:

```
llama_model_load: error loading model: missing tensor 'blk.14.attn_norm.weight'
```

Same as Phase 0a's original observation. So:

- **Without patch**: hard error from `create_tensor`'s missing-tensor check (the ~"hard error" Phase 0a outcome).
- **With patch**: load proceeds; downstream graph build / context init crashes because they assume `output_norm`, `output`, and `layers[14..27]` are populated.

The downstream crash is **M4's responsibility**. M3's contract is exactly what was delivered: make the loader accept partial GGUFs that declare themselves so.

## Architectures covered

The current patch targets the `LLM_ARCH_LLAMA` shared switch case, which also covers:

- `LLM_ARCH_LLAMA`
- `LLM_ARCH_REFACT`
- `LLM_ARCH_MINICPM`
- `LLM_ARCH_GRANITE`, `LLM_ARCH_GRANITE_MOE`
- `LLM_ARCH_MISTRAL3`
- `LLM_ARCH_LLAMA_EMBED`

For v0.1 (single-Llama-family target), this is enough. v0.5+ extends to other architectures (Qwen, Mistral, etc.) as needed; each follows the same pattern (~3 lines of conditional + range check) per case.

## What M3 deliberately does NOT cover

- **Graph builder.** `src/llama-graph.cpp`'s per-architecture graph construction still iterates `0..n_layer` and dereferences layer structs unconditionally. M4 patches the graph builder to respect the layer range OR introduces a new `llama_decode_layers` entry point that skips the embedding lookup, layer iteration outside range, output norm, and lm_head as appropriate.
- **Context init.** `src/llama-context.cpp` allocates KV cache for all `n_layer` layers. With the loader patch, KV slots for layers outside our range are wasted but harmless (no compute happens there). M4's clean version conditionally allocates only the layers we own.
- **`llama_decode` itself.** The standard `llama_decode(ctx, batch)` will crash on a partial-loaded model because it tries to run the full forward pass. M4 introduces the new entry point.
- **Architectures other than Llama-family.** Out of scope for v0.1.

## Next milestones

- **M4 (decode patch):** the dominant work, 6–10 weeks. Adds `llama_decode_layers` and graph-builder partial-range handling. After M4, the segfault we saw becomes "decode runs and emits hidden state."
- **M5 (two-worker integration):** Glues the M2 gRPC scaffold to the M3+M4 patched llama.cpp. Worker process loads a sub-GGUF, exposes a real `Inference` stream that calls `llama_decode_layers` per request.
- **M6 (acceptance test):** The end-to-end falsifiable check from `docs/v0.1-implementation-plan.md` §7.

## Effort recap (vs plan §4)

The plan estimated M3 at 1–2 weeks. Actual focused work for the patch + validation: a handful of hours, dominated by the linker-error iteration (template instantiation discovery) and the build feedback loop. The remaining M3 budget can be reallocated to M4's slack (which is the true critical path).

## Open question for v0.1 (still open)

The metadata KV name `nakshatra.layer_range_*` is a placeholder. If upstream `llama.cpp` later adds its own partial-load API, the namespacing should not collide. Watch the llama.cpp PR tracker through M4; rename if needed before tagging v0.1.
