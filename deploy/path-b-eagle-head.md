# Path B — EAGLE-3 matched draft for ANY target (the lever that beats physics)

**Goal:** a draft that achieves high acceptance on the *target itself* — including the
DeepSeek-R1 *reasoning* model where a generic draft fails (≈27% accept). With M3
(1 round-trip/step) already done, a high-acceptance draft makes speculative decode
win at *all* latencies, not just ≥600ms. Path A proved the lever with an off-the-shelf
matched pair (Llama-3.2-3B+1B → 1.87× @160ms); Path B makes it work for the model we
actually want to serve.

## Why EAGLE-3 (not a separate draft model)
- A separate draft must be the *same family + tokenizer* AND distributionally close —
  rare off-the-shelf, impossible for a bespoke reasoning fine-tune.
- EAGLE-3 trains a *small head* on the TARGET's own hidden features → predicts the
  target's next tokens directly → high τ, and **no second full model to sync over the
  WAN** (the head rides with the target's first shard). Ideal for a distributed split.

## Where it runs
- **Training: ijru's RTX 3060 (CUDA).** The hub GPU is gfx1201 — torch/rocBLAS has no
  kernels for it (training path broken; that's why TTS/transformers failed there). The
  3060's CUDA stack trains fine. The 8B target (~16GB fp16) is tight on 12GB → train
  in 8-bit / LoRA-style or offload; the EAGLE head itself is small.
- **Inference: the head loads with the FIRST shard** (hub) and drafts locally (0 WAN
  hops), exactly like Path A's local draft — then the existing M3 verify path commits.

## Pipeline (executable steps)
1. **Env on ijru** (CUDA torch already works): `pip install --user eagle3` (or clone
   SafeAILab/EAGLE), transformers, accelerate, datasets.
2. **Target + data:** the served target (start with `DeepSeek-R1-Distill-Llama-8B`, the
   reasoning model that defeated the generic draft). Generate ~50–100k tokens of the
   target's OWN outputs on a reasoning+factual+code prompt mix (self-distillation data —
   EAGLE trains the head to mimic the target's feature→token map).
3. **Train the EAGLE-3 head** on ijru (a few GPU-hours): head predicts the target's next
   token from its penultimate hidden state; checkpoint by acceptance on a held-out set.
4. **Acceptance gate (on netsim):** measure τ per content-class (factual/code/reasoning).
   Promote only if reasoning τ ≥ ~3 (vs ~1 for the generic draft).
5. **Integrate into nakshatra's draft interface:** `scripts/speculative.py` `DraftModel`
   currently loads a full GGUF and `.propose()`s K tokens. Add an `EagleDraft` that runs
   the head against the first shard's hidden state (the head needs the target's features,
   which the hub shard already computes — wire it as a local call). Keep the GGUF-draft
   path as fallback.
6. **Prove cross-box:** EAGLE head + M3 + the matched target through netsim @160ms →
   target ≥ the architecture's M6 bar (≈30 tok/s @160ms; ≥6× plain), byte-identical
   greedy output.

## First concrete step (next session)
On ijru: stand up the EAGLE env + the self-distillation data generator against the
target served locally (the 3060 runs the 8B for generation). That's the gate to training.

## Foundation fixes uncovered en route (do alongside)
- `experiments/v0.0/partial_gguf.py`: auto-detect `tie_word_embeddings` and keep
  token_embd on the LAST shard for tied models (Llama-3.2-1B/3B). Today needs manual
  `--keep-token-embd --keep-output`.
- ijru CUDA daemon failed to load a 3B slice (rc=2) while loading the 8B fine —
  investigate (small-model slice vs the CUDA partial-load build).

## Status
Path A ✅ (matched off-the-shelf pair, 1.87× @160ms). Path B = this plan; the training
is multi-session. M1 ✅, M3 ✅, build-gremlin root-fixed ✅, simulator live (:8092).
