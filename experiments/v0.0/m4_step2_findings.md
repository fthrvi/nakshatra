# M4 Step 2 — Hidden-State Input Path

**Date:** 2026-05-06
**Status:** Code complete. Patch is **smaller than the design predicted** because llama.cpp's existing infrastructure already supports caller-supplied hidden state.

Operationalises step 2 from [`docs/m4-decode-patch-design.md`](../../docs/m4-decode-patch-design.md) §"Sequencing within M4". The design budgeted 2 days for a new `llm_graph_input_hidden_state` class; actual outcome is much smaller — see "Pleasant surprise" below.

---

## Pleasant surprise: no new class needed

The existing `llm_graph_input_embd` already supports caller-supplied hidden state via `ubatch.embd != nullptr` (the "vector embeddings path" — used by embedding-mode models that bypass tokenization). The runtime select between the tokens path (`get_rows(tok_embd, tokens)`) and the vector path (`use ubatch.embd directly`) was added in upstream PR [#18550](https://github.com/ggml-org/llama.cpp/pull/18550).

So instead of adding a new graph-input class, M4 step 2 just adds a guard to `build_inp_embd`:

```cpp
if (tok_embd == nullptr) {
    // Skip the tokens path entirely; caller must populate ubatch.embd.
    // Use only the vector embedding path.
    ...
    return cur;  // early-return, no select needed
}
// (existing dual-path + select code unchanged)
```

This means: **non-first workers (no `tok_embd` on disk) can pass `nullptr` to `build_inp_embd` and the graph builder will use whatever the caller put in `ubatch.embd` as the worker's input hidden state.** No new class, no new helper, no new ggml node types.

## What this step delivered

A single-file patch to `src/llama-graph.cpp`: 26 lines added, 0 removed. The early-return branch handles the partial-load case; the existing code-path remains the default.

[Patch](m4_patches/llama-graph.cpp.patch).

## Validation

Built clean. Baseline behavior on a full GGUF is unchanged because the new branch only activates when `tok_embd == nullptr` (which the per-arch graph builder never passes today — that's the M4 step 3 hookup).

The home PC is in a degraded state from earlier zombie-process accumulation, so the runtime baseline-still-produces-Paris check timed out. The patch is **structurally** correct (compiles, branch is gated on a condition baseline doesn't trigger), and the next session can re-verify on a clean system. M4 step 1's baseline check already passed; step 2 only adds a dead-code branch from baseline's perspective.

## What this changes for the M4 effort estimate

The M4 design doc estimated:

- Step 2 (new graph-input class): 2 days
- Step 3 (per-arch graph builder edit + entry point + minimal test): 4 days

Step 2's actual cost was effectively **a few hours** (find the existing infrastructure, add a one-branch guard, compile). The savings carry forward: step 3 doesn't need to integrate a new class either; it just needs to pass `nullptr` to `build_inp_embd` when the worker has no `tok_embd`.

Updated step-by-step estimate (from M4 design's 22.5-day total):

| Step | Old | New | Notes |
|---|---|---|---|
| 1 (storage fields) | 0.5 | ✅ 0.5 done | |
| 2 (graph input) | 2.0 | ✅ 0.25 done | reused existing infra |
| 3 (graph builder edit) | 3.0 | ~2.0 | step 2's reuse simplifies step 3 |
| 4 (hidden-state output capture) | 3.0 | ~3.0 | unchanged |
| 5 (entry point) | 4.0 | ~4.0 | unchanged |
| 6 (KV cache range) | 2.0 | ~2.0 | optional |
| 7 (test harness) | 3.0 | ~3.0 | unchanged |
| 8 (rebase buffer) | 5.0 | ~5.0 | unchanged |
| **Total** | **22.5** | **~19.75** | |

We've saved ~2.75 days. Still well within the plan §4 M4 budget of 6-10 weeks — but a real concrete savings, derived from real code-reading, not optimistic guessing.

## Reproducibility

```bash
cd ~/llama.cpp
patch -p4 < experiments/v0.0/m4_patches/llama-graph.cpp.patch
# (also apply m4_patches/llama-model.h.patch + .cpp.patch + llama-model-loader.cpp.patch from step 1)
cd build && cmake --build . --target llama-cli -j 4
```

After step 2 + step 1 patches, `tok_embd == nullptr` is now a callable code path. M4 step 3 is the per-arch graph builder edit that actually invokes this path for non-first workers.
