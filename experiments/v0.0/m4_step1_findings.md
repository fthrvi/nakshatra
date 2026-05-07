# M4 Step 1 — Storage Fields + KV-Read Refactor

**Date:** 2026-05-06
**Status:** Code complete, validated, with two new findings that scope subsequent M4 steps.

Operationalises step 1 from [`docs/m4-decode-patch-design.md`](../../docs/m4-decode-patch-design.md) §"Sequencing within M4". Refactors M3's hardcoded-in-LLAMA-case approach into a clean model-field-based design that the graph builder (M4 step 3) and the new entry point (M4 step 5) will read from.

---

## What this step delivered

5 fields added to `struct llama_model` (in [`m4_patches/llama-model.h.patch`](m4_patches/llama-model.h.patch)):

```cpp
int  nks_layer_start    = 0;
int  nks_layer_end      = 0;     // 0 => "default to hparams.n_layer in load_tensors"
bool nks_has_token_embd = true;
bool nks_has_lm_head    = true;
bool nks_partial        = false;
```

KV reads hoisted to the top of `load_tensors` — happens once per model load, regardless of architecture (in [`m4_patches/llama-model.cpp.patch`](m4_patches/llama-model.cpp.patch)). On a normal full GGUF (no Nakshatra KVs), all defaults preserve upstream behavior. On a sub-GGUF, the four `nks_*` fields populate from the `nakshatra.layer_range_start/end/has_token_embd/has_lm_head` KVs and the LLAMA case reads from `this->nks_*` instead of computing locally.

Total patch size: **~50 LOC across 3 files** (loader instantiations carry over from M3; the marginal M4-step-1 cost is ~30 LOC).

## Validation

Built clean. Tested against four GGUFs:

| File | Layer range | Has tok_embd | Has lm_head | Result |
|---|---|---|---|---|
| `prithvi-q8.gguf` (full) | [0, 28) | yes | yes | ✓ produces 'Paris' (matches reference) |
| `w0.gguf` | [0, 14) | yes | no | passes loader, segfaults downstream (M3 boundary preserved) |
| `wmid.gguf` | [10, 18) | no | no | passes loader, segfaults downstream |
| `wlast2.gguf` (with `--keep-token-embd`) | [14, 28) | yes | yes (tied) | **NEW: fails at `done_getting_tensors: wrong number of tensors`** |

The full GGUF baseline confirms M4 step 1 doesn't regress any normal-load behavior. The first three rows behave as M3 did.

## Two findings that scope subsequent M4 steps

### Finding 1: top-level `rope_freqs.weight` is unclaimed when `start > 0`

The `wlast2.gguf` failure traces to the per-layer rope_freqs creation in `load_tensors`:

```cpp
layer.rope_freqs = create_tensor(
    tn(LLM_TENSOR_ROPE_FREQS, "weight", i),
    {n_rot/2},
    TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0)
);
```

For `i == 0` the call resolves to the global `rope_freqs.weight` tensor (model-wide, not per-layer); for `i > 0` it resolves to a per-layer name and uses `TENSOR_DUPLICATED` to share the i=0 buffer. When `nks_layer_start > 0`, the loop skips `i == 0` entirely, so the global `rope_freqs.weight` tensor in the file is **never claimed** by any `create_tensor` call. The loader's `done_getting_tensors` check then complains.

**Fix in M4 step 2 (graph input):** when building the worker's compute graph, the new `llm_graph_input_hidden_state` doesn't need `rope_freqs` at all (RoPE has already been applied at the previous worker). The simplest production fix is in `partial_gguf.py`: drop `rope_freqs.weight` from sub-GGUFs where `start > 0`. Update the script to do this; loader patch stays as-is.

**Alternate fix:** in the patched loader, when `nks_layer_start > 0`, explicitly claim `rope_freqs.weight` if present (call `create_tensor` for it with `TENSOR_NOT_REQUIRED` to get it counted without using it). About 3 LOC, contained.

### Finding 2: tied-embedding fallback wants `tok_embd` on the last worker

For models with tied input/output embeddings (Llama-3.2-3B is one), the loader's `output` creation has a fallback:

```cpp
output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), ..., TENSOR_NOT_REQUIRED);
if (output == NULL) {
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), ..., TENSOR_DUPLICATED);
}
```

If the model file has no `output.weight` (which is the tied-embedding case), the loader falls back to `token_embd.weight` and uses it as the output projection. This means **the last worker of a tied-embedding model needs `token_embd.weight` even though it's not the first worker**.

`partial_gguf.py` already supports `--keep-token-embd` for this purpose. M5's cluster setup needs to:

- Detect tied embeddings at sub-GGUF generation time (inspect the source GGUF for `output.weight` — absent ⇒ tied)
- Force `--keep-token-embd` for the last-worker sub-GGUF in the tied case

This is an operator-tooling concern, not a llama.cpp patch concern. Documented here so M5 doesn't get surprised.

## What's next in M4

Step 1 is mechanically done; the segfault we observed for w0 is the SAME M3 boundary segfault as before, just now driven by the cleaner model-field API. Subsequent M4 steps remove that segfault by patching the graph builder and adding the new entry point:

- **Step 2** (~2 days): `llm_graph_input_hidden_state` class — new graph input that takes raw bytes instead of an embedding lookup.
- **Step 3** (~3 days): per-arch graph builder modifications. The Llama template body changes to (a) substitute hidden-state input when `nks_has_token_embd == false`, (b) iterate only `[nks_layer_start, nks_layer_end)`, (c) skip `output_norm + lm_head` when `nks_has_lm_head == false`.
- **Step 4** (~3 days): hidden-state output capture — the graph terminal tensor for middle workers becomes the residual stream output of the last layer in range, written to a caller-provided buffer.
- **Step 5** (~3 days): `llama_decode_layers` C API entry point.

After step 3, the w0 segfault becomes a clean "produces hidden state" success.

## Reproducibility

Patches in [`m4_patches/`](m4_patches/) apply cleanly against `~/llama.cpp` at commit `c46583b`. Source has been reverted on the home PC; no patches in the working tree.

```bash
cd ~/llama.cpp
patch -p4 < experiments/v0.0/m4_patches/llama-model.h.patch
patch -p4 < experiments/v0.0/m4_patches/llama-model.cpp.patch
patch -p4 < experiments/v0.0/m4_patches/llama-model-loader.cpp.patch
cd build && cmake --build . --target llama-cli -j 4

# Tests:
./bin/llama-cli -m ~/prithvi/training/prithvi-merged/prithvi-q8.gguf \
                -p "The capital of France is" -n 1 -ngl 0 -t 4 -c 256
# Expected: produces ' Paris' (M4 step 1 doesn't regress baseline)

./bin/llama-cli -m /tmp/cuts/w0.gguf -p "Hi" -n 1 -ngl 0 -t 4 -c 256
# Expected: passes loader (no "missing tensor" error), segfaults downstream
# (M3 boundary preserved — M4 step 3 closes this gap)
```
