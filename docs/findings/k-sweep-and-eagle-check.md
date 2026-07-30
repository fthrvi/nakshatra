# K sweep + the EAGLE reality check (2026-07-30 ~00:05)

Biswa: *"lets experiment with both"* — the two cheap levers (draft depth K, client
orchestration overhead) and EAGLE. Prithvi was told and consented again; his ears/gateway were
stopped for the window and restored after. Three cards: draft alone on the 7900 XT, chain on
9070 XT + 3060.

## Lever 1 — K sweep (stream-spec, 96 tokens, same prompt)

| K | tok/s | verify rounds | tokens accepted / round | worker time |
|---|---|---|---|---|
| 2 | 30.83 | 53 | 1.81 | 1.94 s |
| **4** | **31.54** | 43 | 2.23 | 1.59 s |
| 6 | 29.36 | 41 | 2.34 | 1.63 s |
| 8 | 28.97 | 36 | 2.67 | 1.53 s |
| — plain streaming control (same session) | **64.22** | 96 | — | 1.48 s |

**K=4 was already optimal; there is no tuning win here.** Acceptance rises with K
(1.81 → 2.67 tokens/round) but *sub-linearly*, while drafting cost rises *linearly* (K forward
passes per round). The two curves cross just past K=4. Note the control run hit 64.22 tok/s —
plain streaming's spread across the night is 45-64 tok/s depending on ambient GPU state; spec's
spread is tight (29-33). Even the pessimistic plain number beats the best spec number.

## Lever 2 — where the time actually goes (K=4)

wall 3.04 s = **1.59 s workers + ~0.90 s drafting (43 rounds × 21 ms) + ~0.55 s orchestration**
(18 % of wall: Python bookkeeping, tokenizer round-trips, per-token logging).

This is the number that decides EAGLE's fate:

- **If drafting were FREE and acceptance unchanged**: 1.59 + 0.55 = 2.14 s → **44.9 tok/s.
  Still loses to plain streaming.**
- **If EAGLE also lifts acceptance to ~3.5 tokens/round** (the EAGLE-2/3 published range):
  ~27 rounds → ~1.0 s workers + ~0.35 s orchestration ≈ **71 tok/s — a real win.**

**So EAGLE's entire value here is the ACCEPTANCE improvement, not the free draft.** That is a
much sharper claim than "EAGLE is the next lever", and it is falsifiable: measure accepted
tokens/round; below ~3 it cannot beat plain streaming on this model no matter how cheap the
draft becomes.

## EAGLE reality check — the heads exist, but the training FAILED

`prithviraj@ijru:~/eagle-out-bf16-noisy/` holds `head_step{500,1000,1500,2000}.pt` (850 MB each)
from a June attempt, plus `eagle-out-fp16-stalled/`. The training log ends at step 6800 with
**`acc0_avg200=0.000`** — the head learned nothing. (The directory names say it: *noisy*,
*stalled*.) Loading it would give ~0 % acceptance, strictly worse than the 0.6 B draft.

**EAGLE is therefore not an experiment we can run — it is a training project** (data gen +
multi-hour GPU training + validation that acc > 0), and by the arithmetic above it must clear
~3 accepted tokens/round to be worth shipping. Recommend deciding that deliberately rather than
inheriting it as "the obvious next step".

## Settled

Chain serving stays **plain streaming**. Speculation, stream-spec, and async pipelining are all
merged, correct (byte-identity proven), flag-gated OFF, and now bounded by measurement rather
than by hope.
