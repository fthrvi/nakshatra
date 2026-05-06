# Phase 0b prerequisite — cb_eval ergonomics, INTERIM FINDINGS

**Date:** 2026-05-06
**Status:** **BLOCKED on Python binding bug.** C++ path is known-working. Spike pivots to C++ next session.

This doc captures what we learned trying to validate `cb_eval` ergonomics in `llama-cpp-python` for the Phase 0b spike. The header claim from `docs/v0.0-validation-plan.md` ("Try `llama-cpp-python` first") doesn't pan out cleanly; we hit a Python-binding wall and switched plans.

## What we tested

[`cb_eval_probe.py`](cb_eval_probe.py) registers a `ggml_backend_sched_eval_callback` via the low-level `llama_cpp` ctypes API, runs a short decode, and counts how many times the callback fires.

Iterations tried:
1. Self-defined `CFUNCTYPE` for the callback type
2. Used the exposed `llama_cpp.ggml_backend_sched_eval_callback` type directly
3. Switched from `llama_batch_init` + manual fill to `llama_batch_get_one` (matching the C++ reference path)
4. Tested on macOS (`bishwa`, lab Mac with Metal+Vulkan) and Linux (home PC, CPU-only build)

## What we observed

| Setup | Callback invocations during decode |
|---|---|
| llama-cpp-python 0.3.22 (any of the iterations above) | **1 ask + 1 eval, both for `embd` only** |
| llama.cpp's own `examples/eval-callback` C++ tool, same model | **~1,600 invocations** (16,312 lines of per-node tensor output) |

The C++ tool produces a callback for every node in the 875-node compute graph: `embd`, `norm-0`, `attn_norm-0`, `Qcur-0`, `Kcur-0`, `Vcur-0`, ..., across all 28 layers. The Python binding fires the callback only for `embd` and stops.

The `graph splits = 1` value (visible in both runs) was initially suspected as the cause but is actually a red herring — the C++ tool also runs with `graph splits = 1` and gets per-node callbacks anyway.

## Diagnosis

The `cb_eval` API itself works correctly. The bug is in how `llama-cpp-python`'s ctypes wrapping passes the Python callback to the underlying scheduler. Possibilities (not narrowed down):

- Function-pointer lifetime / GC interaction
- Subtle CFUNCTYPE signature mismatch with the C-side struct field
- A llama-cpp-python wrapper layer that intercepts and short-circuits cb_eval after the first invocation
- The internal sched the wrapper hands to llama_init_from_model not honoring the cparams.cb_eval field

Each iteration to test a hypothesis costs ~5–10 min (rebuild + rerun + inspect output). Diminishing returns kicked in after ~1.5 hours.

## Decision

**Pivot Phase 0b's spike to a C++ implementation.** Rationale:

1. **C++ is known-working.** `examples/eval-callback/eval-callback.cpp` is a 50-line example that gives us per-node callbacks out of the box. We extend that, not write from scratch.
2. **Not throwaway work.** The C++ binary is closer to v0.1's destination than the Python hack would have been. v0.1 ultimately needs a custom C++ worker linking llama.cpp; the spike binary is a small step in that direction.
3. **Faster path to data.** Estimated 3–5 hours of focused C++ work vs uncertain hours debugging the Python binding.

The validation plan's fallback was always "if Python doesn't work, write a thin C++ binary." We're activating that fallback.

## What stays valid from this exploration

- The ctypes path to `cb_eval` exposure was found and documented (`llama_cpp.ggml_backend_sched_eval_callback`).
- The `ggml_tensor` struct layout (defined in `cb_eval_probe.py`) matches build 8142 and successfully decodes tensor names. Reusable if we ever come back to Python.
- The `MentoringInstitute@bishwa` setup (venv, llama-cpp-python 0.3.22, gguf-py from llama.cpp source tree) is reproducible and documented.
- `prithvi/training/prithvi-merged/prithvi-q8.gguf` (Llama-3.2-3B fine-tune, 28 blocks) on the home PC is a working test model.

## Next session — concrete plan

1. **Start from `~/llama.cpp/examples/eval-callback/eval-callback.cpp`** on the home PC.
2. **Add TCP socket I/O.** Worker mode A: capture hidden state at a target tensor name (e.g., `l_out-13` for a mid-layer cut on a 28-layer model), serialize, send over socket. Worker mode B: receive bytes, inject at the same tensor's place in the next decode.
3. **Use a single Linux machine, two processes** for the spike (per Option A from this session).
4. **Falsifiable test:** distributed decode produces same top-1 token as single-machine reference. Same shape as the test from `docs/v0.0-validation-plan.md` Phase 0b's success criterion, just differently implemented.

Estimated session length: 3–5 hours. If it works, v0.1 commitment is fully de-risked.

## Observations worth keeping

- **The `cb_eval` mechanism is per-node, not per-split** (we initially guessed wrong). This is good news for v0.1: future C++ workers can use cb_eval as a low-effort intercept point if a quick test or verification path is needed, separate from the patched-`llama_decode` production path.
- **macOS builds of llama-cpp-python auto-engage Metal** even when `-DGGML_METAL=OFF` is passed to `pip install`. This is a separate issue from the cb_eval binding bug; it would have made the macOS path harder anyway. Linux home PC is the right venue for the spike.
