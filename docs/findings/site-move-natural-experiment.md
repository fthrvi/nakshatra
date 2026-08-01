# The doctrine, confirmed by accident: same split, same hardware, 10.5× slower (2026-08-01)

ijru moved to a different site overnight. Nothing about the model, the slices, the
build, or the prompt changed — only the link between the two stages. That makes
this the cleanest test of "route, don't split" the project has ever had, because
it is the *same split*, not a re-implementation.

| | link RTT | throughput |
|---|---|---|
| 2026-07-29, both stages on the home LAN | ~0.3 ms | **51.16 tok/s** |
| 2026-08-01, ijru behind WireGuard at another site | **~190 ms** | **4.85 tok/s** |

Same chain (`qwen3-30b-q3-chain-v2.yaml`, hub ROCm L0-15 + ijru CUDA L15-48), same
prompt ("Explain, step by step, why the sky appears blue…"), same 96-token budget.
96 tokens in 19.8 s.

**4.85 tok/s is within noise of the June-era 4.94 tok/s** that produced the
route-don't-split doctrine in the first place. The doctrine was inferred from a
slow configuration we then fixed; this is the controlled version of that
observation, arrived at by the world changing rather than by us changing it.

## What it actually shows

The 27× speedup reported for the CUDA rebuild was **never a property of the
hardware alone** — it was hardware *and* a sub-millisecond link. Per decode step
the chain pays one round trip; at 190 ms that round trip dominates everything the
GPUs do (7 ms + 11 ms of compute). Splitting buys capability, and it charges the
link for it, every single token.

## The gap this exposes

`elastic_unconscious` gates escalation on **free VRAM only**. With ijru remote it
still selects the split chain — 4.85 tok/s — when routing a whole 14B on the hub
would be several times faster. The selector cannot see the one variable that
just changed by three orders of magnitude.

`Node` already carries an `rtt_ms` field; `measure_live()` never populates it.
Populating it, and refusing a split whose per-step round trip exceeds its
per-stage compute, is the smallest honest fix — and it is the first empirical
step toward the placement-crossover model that the research brief lists as an
open question.

## Operational note

Every hardcoded LAN address for ijru broke at once, silently: the new site reuses
the same `10.0.0.0/24` range, so `10.0.0.233` did not fail loudly, it just went
nowhere, and the capacity gate reported "ijru 0.0 free" — which reads as *busy*
rather than *moved*. Address roaming nodes by overlay IP or ssh alias only. The
overlay address is identity; the LAN address is merely location.
