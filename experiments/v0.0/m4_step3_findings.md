# M4 Step 3 — Per-Architecture Graph Builder Edit

**Date:** 2026-05-06
**Status:** Code complete and validated. Partial-load workers now run end-to-end through `llama_decode` without crashing. Middle-worker testing requires the gRPC chain (M5).

Operationalises step 3 from [`docs/m4-decode-patch-design.md`](../../docs/m4-decode-patch-design.md) §"Sequencing within M4". This is the milestone that closes the M3 segfault gap — the patched binary now decodes a partial-loaded GGUF cleanly.

---

## What this step delivered

Three hunks in `src/models/llama.cpp` (the per-arch graph builder for the Llama family):

1. **For-loop range** — iterates `[model.nks_layer_start, model.nks_layer_end)` instead of `[0, n_layer)`.
2. **`inp_out_ids` made conditional** — `model.nks_has_lm_head ? build_inp_out_ids() : nullptr`. The "select rows for output tokens" subselection at the end of the last layer now only fires when this worker has the lm_head.
3. **Output norm + lm_head wrapped in `if (model.nks_has_lm_head)`** — middle/first workers expose the residual stream output of their last in-range layer as the graph's terminal tensor; last workers proceed through norm + lm_head as before.

Plus the previous step's patches (still required):
- M4 step 1: storage fields on `llama_model` + KV-read refactor
- M4 step 2: `build_inp_embd` guard for `tok_embd == nullptr`

Total patch set after step 3: **5 files, ~70 LOC added net.**

[Patches in m4_patches/](m4_patches/):
- `llama-model.h.patch` (storage fields)
- `llama-model.cpp.patch` (loader + LLAMA-case range)
- `llama-model-loader.cpp.patch` (template instantiations)
- `llama-graph.cpp.patch` (build_inp_embd guard)
- `models_llama.cpp.patch` (graph builder edit) ← **NEW in step 3**

## Validation

Built clean. Tested four cases on the home PC:

| Case | Result |
|---|---|
| Baseline full GGUF | ✓ produces 'Paris' (no regression) |
| **w0** [0, 14) — first worker | ✓ runs to completion. Decode finishes, samples a token, exits cleanly. Output token is numerically meaningless because llama-cli samples from the residual stream as if it were logits — but **no crash, no assertion failure**. |
| **wmid** [10, 18) — middle worker | abort in `ggml_backend_tensor_set` from `llm_graph_input_embd::set_input` |
| **wlast** [14, 28) — last worker | ✓ runs to completion. Same caveat as w0 about output meaningfulness. |

The middle-worker abort is **expected and correct behavior**: a middle worker requires `ubatch.embd` to be populated with the upstream worker's hidden state (which is what M4 step 2's nullptr-tok_embd path uses). `llama-cli` always passes tokens, never pre-computed embeddings, so middle workers cannot be exercised via `llama-cli` — they can only be tested via the gRPC chain that ferries hidden state between processes (M5).

## What this means

The patched binary now produces well-defined output for first and last workers given an appropriate input shape. End-to-end correctness across a multi-worker chain still requires:

- **M4 step 4** (~3 days) — capturing the residual-stream output for first/middle workers into a caller-provided buffer, and accepting hidden-state input via the new entry point.
- **M4 step 5** (~3 days) — the `llama_decode_layers` C API entry point that wraps these mechanics for callers (the Nakshatra worker process).

After step 5, a single-machine three-process chain (worker 0 → middle → last) running with M5 gRPC plumbing can produce the same top-1 token as a single-machine `llama-cli` reference. That's the v0.1 acceptance test.

## Updated effort tracker

| Step | Design est | Actual | Notes |
|---|---|---|---|
| 1 (storage fields) | 0.5 d | ✅ ~0.25 d | refactored M3 |
| 2 (graph input) | 2.0 d | ✅ ~0.25 d | reused upstream PR #18550 |
| 3 (graph builder edit) | 3.0 d | ✅ ~0.5 d | three small hunks; the inp_out_ids fix was the only surprise |
| 4 (hidden-state output capture) | 3.0 d | — | next |
| 5 (entry point) | 4.0 d | — | dominant non-graph work |
| 6 (KV cache range) | 2.0 d | — | optional |
| 7 (test harness) | 3.0 d | — | reuses M2 scaffold |
| 8 (rebase buffer) | 5.0 d | — | unchanged |
| **Total** | 22.5 d | **~1.0 done** | tracking ahead of plan |

## Reproducibility

Apply five patches in order against `~/llama.cpp` at commit `c46583b`:

```bash
cd ~/llama.cpp
patch -p4 < experiments/v0.0/m4_patches/llama-model.h.patch
patch -p4 < experiments/v0.0/m4_patches/llama-model.cpp.patch
patch -p4 < experiments/v0.0/m4_patches/llama-model-loader.cpp.patch
patch -p4 < experiments/v0.0/m4_patches/llama-graph.cpp.patch
patch -p4 < experiments/v0.0/m4_patches/models_llama.cpp.patch
cd build && cmake --build . --target llama-cli -j 4

# Tests
./bin/llama-cli -m ~/prithvi/training/prithvi-merged/prithvi-q8.gguf \
                -p "The capital of France is" -n 1 -ngl 0 -t 4 -c 256
# Expected: produces ' Paris'

# Generate sub-GGUFs (uses experiments/v0.0/partial_gguf.py)
python experiments/v0.0/partial_gguf.py prithvi-q8.gguf /tmp/w0.gguf   --start 0  --end 14
python experiments/v0.0/partial_gguf.py prithvi-q8.gguf /tmp/wlast.gguf --start 14 --end 28 --keep-token-embd

./bin/llama-cli -m /tmp/w0.gguf   -p "Hi" -n 1 -ngl 0 -t 4 -c 256
./bin/llama-cli -m /tmp/wlast.gguf -p "Hi" -n 1 -ngl 0 -t 4 -c 256
# Expected: both run to completion (output tokens are numerically meaningless,
# but no crash). Middle workers can ONLY be tested via the gRPC chain (M5).
```

Source has been reverted on the home PC; the five patches in `m4_patches/` are documentation artifacts only.
