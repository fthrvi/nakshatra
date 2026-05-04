# v0.0 Spike — Step-by-step results

Test environment: CPU-only llama.cpp build (`/home/prithvi/llama.cpp/build/`),
Qwen3-0.6B fp16 GGUF at `/tmp/nakshatra-test/qwen3-0.6b-full.gguf`,
prompt `"The capital of France is"` (5 tokens, no chat template), 28 transformer
blocks, hidden dim 1024.

## Step 0 — Determinism reference (2026-05-04)

`llama-completion` with `--seed 42 --top-k 1 --temp 0` produced bit-identical
output across two runs, modulo timing fields. Decoded text after the chat-templated
prompt: `"<think>\nOkay, the"`. The chat template-vs-bare-prompt discrepancy
between `llama-completion` and our libllama-direct programs is resolved by
always regenerating the reference with our own program when comparing.

## Step 1 — cb_eval observation (2026-05-04)

`spike/v0.0/cb_eval_observe` ran one `llama_decode` with a callback that
returned `false` on every `ask=true`, just printing the tensor name + shape +
dtype on first encounter. Saw **763 unique tensors**.

Cut-point identification:
- Per-block outputs are named `l_out-N` for `N` in `[0, 27]`.
- `l_out-13` shape: `[1024, 5, 1, 1]`, dtype `f32` → 20480 bytes for this prompt.
- `l_out-27` shape: `[1024, 1, 1, 1]` — only the last position survives into
  the final block; llama.cpp drops earlier positions before the LM head since
  only the last token's logits are needed for sampling.

Conclusion: `l_out-13` is the canonical cut tensor. Wire payload at 5-token
prefill is 20 KB; per-decode-step thereafter is 4 KB.

## Step 2 — cb_eval-write load-bearing test (2026-05-04)

`spike/v0.0/cb_eval_write` ran two modes:

| Mode | Argmax token | Decoded | Bytes zeroed |
|---|---|---|---|
| `reference` | **12095** | ` Paris` | 0 |
| `perturb`   | **0**     | `!`     | 20480 |

`cb` call counts:
- `reference`: ask=1 post=0 zeroed=0 (callback returned false on ask=true so no post-compute notification fired).
- `perturb`:   ask=1 post=1 zeroed=1, 20480 bytes (exactly hidden×n_tokens×sizeof(f32)).

**Conclusion: cb_eval-write works.** Modifying `t->data` inside the callback at
`l_out-13` propagates through layers 14..27, output_norm, and lm_head, and changes
the predicted token. The 20480-byte zeroing exactly matches the expected
tensor size, with no out-of-bounds writes (process exited 0).

This unblocks Steps 3, 4, 5 of the spike plan. Moving forward with the original
two-worker activation-handoff design.

## Open observations worth registering

- The `l_out-N` naming convention appears to be Qwen3-specific (or `llama-graph.cpp`
  convention more broadly). Other architectures may emit different names. For v0.1
  this means the layer-range API patches need to use llama.cpp's internal
  per-architecture graph builder, not raw tensor-name-matching, for portability
  across model families.
- The dtype at the cut is `f32`, not the model's `f16` weight dtype. Activations
  are computed in higher precision than weights; this is normal but the wire
  format must accommodate either f32 or f16-after-cast (4 KB/step vs 2 KB/step).
- `l_out-27` shape `[1024, 1, ...]` confirms llama.cpp's last-position
  optimization. Workers in the v0.1 pipeline will need to be aware of which
  position(s) they are computing for; mid-pipeline workers carry all positions,
  the last worker only the last position.
