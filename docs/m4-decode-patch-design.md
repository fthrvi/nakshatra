# M4 — Decode Patch Design

**Date:** 2026-05-06
**Status:** Design — refines the v0.1 implementation plan §3.2.2 with concrete patch points after a code-survey of the Llama graph builder, graph inputs, and context init.

Sibling to:
- [`v0.1-implementation-plan.md`](v0.1-implementation-plan.md) — the parent plan
- [`path-a-vs-path-b-memo.md`](path-a-vs-path-b-memo.md) §5 — original effort estimate
- [`../experiments/v0.0/m3_findings.md`](../experiments/v0.0/m3_findings.md) — what M3 already delivered

---

## What M4 ships

A new C API entry point `llama_decode_layers` that runs forward through a worker's assigned layer range only, accepting either a token batch (worker 0) or a hidden-state buffer (middle/last workers) as input, and emitting either a hidden-state buffer (worker 0/middle) or filling the existing logits buffer (last worker) as output.

After M4 lands, the M3 segfault we observed becomes a successful partial-decode that produces well-defined hidden-state output bytes that downstream workers can consume.

## Patch points (concrete)

| File | What changes | Approx LOC |
|---|---|---|
| `src/llama-model.h` | Add 4 fields to `llama_model`: `nks_layer_start`, `nks_layer_end`, `nks_has_token_embd`, `nks_has_lm_head`. Default to `0`, `n_layer`, `true`, `true` (i.e., normal full-model behavior). | ~6 |
| `src/llama-model.cpp` | Populate those fields in `load_tensors` from the same KVs M3 introduced (refactor the M3 patch to write the model fields instead of locals; the for-loop conditional reads from the model fields). | ~10 |
| `src/llama-graph.h` | Declare a new `llm_graph_input_hidden_state` class (mirror of `llm_graph_input_embd`, but the input tensor is `[n_embd, n_tokens, batch]` from a caller-provided buffer rather than the embedding lookup). Declare `build_inp_hidden_state()` helper. | ~30 |
| `src/llama-graph.cpp` | Implement the new class + helper. Mostly mirrors the existing `build_inp_embd` path; difference is the source of bytes (caller buffer vs. `model.tok_embd` lookup). | ~50 |
| `src/models/llama.cpp` | The dominant per-arch change. The template `llm_build_llama<embed>::llm_build_llama` modifies its body: (a) replace the unconditional `inpL = build_inp_embd(model.tok_embd)` with `inpL = nks_has_token_embd ? build_inp_embd(model.tok_embd) : build_inp_hidden_state()`. (b) change the per-layer `for (int il = 0; il < n_layer; ++il)` to `for (int il = nks_layer_start; il < nks_layer_end; ++il)`. (c) wrap the `output_norm` + `lm_head` block in `if (nks_has_lm_head) { ... } else { res->t_embd = inpL; res->t_logits = nullptr; }` so middle workers expose `inpL` (the residual stream output) as the graph's terminal tensor. | ~30 |
| `src/llama-context.cpp` | Adjust KV cache allocation to only `[nks_layer_start, nks_layer_end)`. Leave layers outside the range unallocated (0-byte). Save substantial memory on workers that hold a small slice. | ~20 |
| `include/llama.h` | Declare `llama_decode_layers(ctx, batch, hidden_in, hidden_in_bytes, hidden_out, hidden_out_bytes)`. | ~15 |
| `src/llama.cpp` (or `src/llama-decode.cpp` depending on layout) | Implement `llama_decode_layers`. Routes the call: builds the appropriate graph input, calls into the same `llama_decode` core, copies graph output into caller's `hidden_out` buffer (or leaves logits in place for last worker). | ~80 |

**Total: ~240 LOC across 7 files.**

## New entry point — proposed signature

```c
// include/llama.h

enum llama_decode_layers_mode {
    LLAMA_DECODE_LAYERS_FIRST  = 0,  // tokens in, hidden out
    LLAMA_DECODE_LAYERS_MIDDLE = 1,  // hidden in, hidden out
    LLAMA_DECODE_LAYERS_LAST   = 2,  // hidden in, logits via existing API
};

LLAMA_API int32_t llama_decode_layers(
    struct llama_context * ctx,
    enum llama_decode_layers_mode mode,
    struct llama_batch batch,            // mode FIRST: tokens; otherwise: only batch.n_tokens used
    const float * hidden_in,             // mode MIDDLE/LAST: input bytes
    size_t        hidden_in_bytes,       // expected: n_tokens * n_embd * sizeof(float) (or whatever wire dtype)
    float       * hidden_out,            // mode FIRST/MIDDLE: output buffer
    size_t        hidden_out_bytes       // expected: n_tokens * n_embd * sizeof(float)
);
```

The `mode` parameter is redundant with the worker's loaded layer range, but explicit is easier to reason about in caller code. Internally the function asserts that mode matches `(nks_has_token_embd, nks_has_lm_head)` and rejects mismatches with a clear error.

For LAST mode, logits are read via the existing `llama_get_logits_ith()` after the call returns 0. Same behavior as `llama_decode` for downstream code.

## Effort breakdown

| Task | Days | Notes |
|---|---|---|
| Storage fields on `llama_model` + KV-read refactor | 0.5 | Mostly mechanical; M3's KV-read code moves into a struct |
| `llm_graph_input_hidden_state` class + `build_inp_hidden_state()` | 2 | The new input type; mirrors `llm_graph_input_embd` |
| Per-arch graph builder edit (Llama family) | 3 | Template body changes; 7 architectures share the case |
| KV cache range adjustment | 2 | Optional for v0.1 (works without it; just wastes memory) |
| `llama_decode_layers` entry point + glue | 4 | The dominant non-graph work |
| Test harness: distributed 2-process via gRPC, end-to-end token match | 3 | Reuses M2 scaffold |
| Edge cases (logits-all interaction, n_outputs, batch sizes) | 3 | Discovered while iterating |
| Upstream rebase + conflict resolution buffer | 5 | Plan §6's monthly rebase commitment; 5 days of buffer over the M4 window |

**Subtotal: 22.5 days = ~4.5 weeks at full-time.** The v0.1 plan estimated **6–10 weeks**. The buffer between my estimate and the plan's covers:

- Discovered architecture quirks (the Llama family has variants like Granite-MoE that use a different FFN path; the patch may need additional conditionals)
- Upstream churn during the M4 implementation window (conservative: 1–2 mid-cycle rebases)
- Test-suite work to make the M5 acceptance test reliable across both workers

I am keeping the plan's **6–10 weeks** estimate as authoritative; this design doc just shows that the lower bound is reachable if upstream is stable and no major architecture rework is required.

## Risks (M4-specific)

| Risk | Likelihood | Mitigation |
|---|---|---|
| `llm_graph_input_hidden_state` interacts badly with the scheduler's reserve pass (wrong tensor shapes during dry-run) | Medium | The reserve pass uses worst-case shapes; we'd populate them with `n_ubatch` × `n_embd` × `n_tokens_max` placeholders. Same pattern `llm_graph_input_embd` already uses. |
| The "tied embedding" path (output = tok_embd duplicate) breaks when worker has neither tok_embd nor output | Medium | Last worker has output, so it has tok_embd-as-output if needed via `TENSOR_DUPLICATED`. Worker 0 has tok_embd directly. Middle workers have neither and don't need either — they only run middle layers. The patch has to avoid the duplication path on middle workers. |
| KV cache allocation out-of-range for layers outside [start, end) | High if untreated | The `llama-context.cpp` change is the explicit fix; without it we waste memory but don't crash. M3 already loaded fine without this fix. |
| Multi-token-batch correctness | High if untreated | Worker 0's hidden_out shape and middle worker's hidden_in shape must agree on `[batch, n_tokens, n_embd]` layout. Define the wire format once in proto + C API + assert at runtime. |
| `flash_attention` or `fused_gated_delta_net` fusion-vs-partial interactions | Low–Medium | M4's new entry point doesn't use cb_eval, so the Phase 0b fusion-breaking artifact does NOT recur. But the graph-input change might disable some fusions we currently get for free. Profile post-M4. |

## What M4 still does NOT cover

- **gRPC integration in the worker.** The worker.py already has the gRPC scaffold from M1+M2 but currently uses llama-cpp-python's high-level `Llama` class. M5 swaps that out for ctypes-bound `llama_decode_layers` calls (or a thin C++ worker if Python ctypes proves limited).
- **Stateful streaming inference.** M4's `llama_decode_layers` can be called multiple times for token-by-token generation; KV cache per-session lives in `llama_context`. M5 wires this to the gRPC `Inference` streaming RPC.
- **Architectures other than Llama family.** Out of scope for v0.1.

## Pre-M4 prerequisites (resolve first)

- **Wire-dtype decision.** The plan §8.3 says fp16. Confirm on M4 start; the entry-point signature uses `float` but bytes-based shape allows fp16 just as easily.
- **Sub-GGUF generation for the chosen v0.1 model.** [`partial_gguf.py`](../experiments/v0.0/partial_gguf.py) supports arbitrary cuts after the recent generalization; settle on the v0.1 model (per `v0.1-implementation-plan.md` §9 open question) before producing test sub-GGUFs at scale.
- **Branch hygiene.** The M3 patch lives in `experiments/v0.0/m3_patches/` only as a documentation artifact. M4 starts a real branch in a llama.cpp fork. Pick the fork host (probably `fthrvi/llama.cpp` mirroring our nakshatra fork) and the branch name (`nakshatra-v0.1` is the obvious choice).

## Sequencing within M4

A safe order that derisks the dominant work first:

1. Storage fields + KV refactor (0.5 day)
2. New graph input type (2 days)
3. Per-arch graph builder edit + entry point + minimal test ("worker 0 only, hidden-state out") (4 days)
4. Add hidden-state input path; verify worker 0 → hypothetical middle-worker chain produces same hidden state as a full forward at the cut point (3 days)
5. Last-worker mode (last layers + norm + lm_head from received hidden state) (3 days)
6. KV cache range adjustment (2 days, optional but recommended)
7. Two-process distributed test, distributed top-1 token matches single-machine reference (3 days)

If steps 1–3 land cleanly in week 1, the project is on track for the lower-bound estimate. If they spill into week 2, the upper-bound estimate is the right plan.
