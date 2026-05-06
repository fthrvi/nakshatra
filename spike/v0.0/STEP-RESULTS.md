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

## Step 2 — cb_eval-write load-bearing test (2026-05-04, conclusion later corrected)

`spike/v0.0/cb_eval_write` ran two modes:

| Mode | Argmax token | Decoded | Bytes zeroed |
|---|---|---|---|
| `reference` | **12095** | ` Paris` | 0 |
| `perturb`   | **0**     | `!`     | 20480 |

`cb` call counts:
- `reference`: ask=1 post=0 zeroed=0 (callback returned false on ask=true so no post-compute notification fired).
- `perturb`:   ask=1 post=1 zeroed=1, 20480 bytes (exactly hidden×n_tokens×sizeof(f32)).

> **CORRECTION (2026-05-06).** The original conclusion below ("cb_eval-write
> works; modification propagates through layers 14..27") was **wrong**. It was
> a measurement error: I checked the *argmax token* but never checked the
> underlying logit values. When Step 3 forced me to instrument `logits[]`, it
> turned out that under `perturb`, the logits were all *exactly* 0.0 — not
> "small numbers from a noise-collapsed forward pass" but **literally zero,
> meaning the LM head never executed**. The `argmax=0` was an artefact of the
> uninitialised/zeroed output buffer, not a "the prediction changed because we
> scrambled layer 13" signal. Same observable, completely different cause.
>
> The corrected understanding (validated by Step 3, see below): **simply
> returning `true` on `ask=true` for `l_out-13` causes the rest of the
> compute graph to be skipped or read from a stale buffer**, regardless of
> whether the callback also performs any modification on `ask=false`. So
> Step 2 didn't validate cb_eval-write — it inadvertently triggered the same
> "downstream skipped" failure mode that Step 3 also hits. cb_eval is *not* a
> usable mechanism for capturing or injecting hidden states mid-graph.
>
> **What we still know is true from Step 2:** the byte-level mechanics fire
> (callback fires once, `ggml_nbytes` matches `hidden × n_tokens × 4`,
> `memset` writes the right number of bytes, no out-of-bounds). What we
> *don't* know from Step 2 alone is whether any of those writes propagated.
> They didn't.

**Original (incorrect) conclusion, kept for reference:** "cb_eval-write
works. Modifying `t->data` inside the callback at `l_out-13` propagates
through layers 14..27, output_norm, and lm_head, and changes the predicted
token... This unblocks Steps 3, 4, 5 of the spike plan." — None of this
held up under Step 3's instrumentation.

## Step 3 — handoff between two contexts (2026-05-06, original design failed)

`spike/v0.0/cb_eval_handoff` was built to test a `ctxA → ctxB` byte-level
activation handoff entirely in one process: ctxA's callback captures
`l_out-13`'s bytes (and aborts decode), then ctxB's callback overwrites
its own `l_out-13` post-compute with the captured bytes. Because A and B
run the same model on the same input, the override is mathematically a
no-op — captured bytes equal the bytes B would have computed itself — so
B's argmax should equal Step 2's reference (12095, ` Paris`).

It didn't. B's argmax was 0 (`!`), and instrumentation showed that **B's
logits were all exactly zero** (not noisy, not NaN — `min=max=mean=0.0000,
n_nan=0, n_inf=0`).

Three control variants were run to triangulate:

| Mode | Description | Argmax | Logit signature |
|---|---|---|---|
| `handoff`         | A captures + aborts; B restores | 0 | all 0.0000 |
| `handoff_no_abort` | A captures but doesn't abort; B restores | 0 | all 0.0000 |
| `ctxb_only`       | skip A; B's callback returns interest but doesn't modify | 0 | all 0.0000 |
| `passthrough`     | B's callback returns true on ask=true, returns false on ask=false WITHOUT touching `t->data` | 0 | all 0.0000 |

`passthrough` is the smoking gun: the callback **registers interest** in
`l_out-13` and that's it — no capture, no inject, no modification — and
downstream computation still produces all-zero logits.

**Conclusion: returning `true` on `cb_eval`'s `ask=true` for any
mid-graph tensor causes downstream ops to read from a stale or
uninitialised buffer.** The mechanism is consistent with `ggml-sched`
treating the marked tensor as a "graph output" — its data is staged for
host-side delivery — and downstream consumers reading from a different
buffer than where the callback writes.

This kills the cb_eval-based two-worker scheme as it was sketched in
both the spike plan and `path-a-vs-path-b-memo.md` §5.7. The cb_eval
hook is observation-only on tensors **whose downstream values you don't
care about**. It cannot be used to capture intermediate activations
without simultaneously breaking the rest of the forward pass.

## Pivot

The spike's purpose — de-risk activation transport, orchestration, and
determinism before committing to v0.1 C++ work — does not change. Only
the mechanism does. The pivot uses APIs llama.cpp already supports:

- **Capture (Worker A):** load a sub-GGUF containing `blk.0..13` +
  `token_embd` + `output_norm` + `output`, with `block_count = 14`.
  Run `llama_decode` with `llama_context_params.embeddings = true` and
  `pooling_type = LLAMA_POOLING_TYPE_NONE`. The hidden states extracted
  from a 14-block model's last block *are* `l_out-13` of the original
  28-block model.
- **Inject (Worker B):** load a sub-GGUF containing `blk.14..27`
  re-indexed to `blk.0..13` + `output_norm` + `output`, with
  `block_count = 14` and no `token_embd`. Set `llama_batch.embd` to
  Worker A's captured hidden states (instead of `llama_batch.token`).
  `llama_decode` produces logits naturally; argmax should match the
  single-process reference (12095, ` Paris`).

This is also closer to the v0.1 production architecture, so the spike's
de-risking value goes up, not down.

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
