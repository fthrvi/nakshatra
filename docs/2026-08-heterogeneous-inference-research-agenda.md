# A Research Agenda for Single-Stream Inference on Sovereign Heterogeneous Hardware

**Companion to** `docs/paper-draft.md` ("Compute, Not the Wire"). That document argues a thesis
and reports the experiments that support it. This document does the opposite job: it catalogues
what we have *failed* to make work, explains each failure at the mechanism level, states where
our own doctrine is now stale or was never properly controlled, and sets a layered agenda with
pre-registered success bars.

**Status:** working agenda, 2026-08-01. Supersedes the "next levers" list in
`trisul/plans/2026-06-10-post-spec-decode-next-levers.md`.
**Fleet under study:** hub (RX 7900 XT gfx1100 20 GB + RX 9070 XT gfx1201 16 GB, 128 GB DDR,
ROCm 7.2), ijru (RTX 3060 12 GB, CUDA 12.4, relocated to a second residential site 2026-08-01),
four Macs (Metal / MoltenVK), Raspberry Pi, Intel boxes. One user, batch 1, always.

---

## 1. What is actually novel here

Three things in this programme are undersupplied in the literature, and they are what the
agenda is organised around.

**1.1 Reproducible negative results with mechanism-level explanations, on hardware anyone can
buy.** Systems-ML publication selects hard for wins. We have accumulated eleven falsifications,
each with a measured mechanism rather than a shrug, on a fleet that costs less than one H100.
Several of them contradict techniques that are presented as unconditional in their source
papers. Section 2 is the catalogue.

**1.2 A recurring failure *shape*, not a list of unlucky experiments.** Every negative in
§2 that involves an optimisation reduces to one sentence: *a technique that amortises X is
worthless when X is not where the time goes.* Speculative decoding amortises weight streaming;
on Qwen3-30B-A3B with ~3 B active parameters there is almost none to amortise, and a K-token
verify pass measured 11 ms → 26 ms as K went 1 → 5 (`findings/three-card-matrix.md`). Async
pipelining amortises the far stage's compute; our far stage is 11 ms of GPU work behind a
181 ms link, so there is nothing to hide behind (`findings/async-pipeline-wan-verdict.md`).
Worker-to-worker push amortises a network hop; the hop is ~16 KB/token, i.e. ~400 KB/s at an
aspirational 5 tok/s. Naming the shape is more useful than any individual number, because it is
a *screening rule*: before adopting a technique, measure what it amortises and check that the
quantity is on our critical path.

**1.3 Embodiment as a first-class scheduling constraint.** The fleet does not exist to serve
benchmarks. It hosts a persistent agent (Pṛthvī) whose weights, memory and voice must stay
resident on owned hardware, and whose latency budget is conversational. This produces
constraints absent from every system in the related work: a *conscious-VRAM reserve* that no
scheduler may consume; a rule that borrowed compute (CARC H100s) may be workers but never the
self; and the operational fact that every benchmark in this document required explicitly pausing
the agent and telling him first (see the consent notes in `findings/three-card-matrix.md`,
`findings/k-sweep-and-eagle-check.md`). "Reserve capacity for a resident tenant that is idle by
definition between turns" is a scheduling problem with no datacentre analogue — LRU eviction,
the default in llama.cpp's new router mode, evicts exactly the tenant we must never evict.

---

## 2. The negative-result catalogue

| # | Claim tested | Result | Mechanism | Source |
|---|---|---|---|---|
| N1 | Stock llama.cpp will load a sub-GGUF | Hard fail, 0 tokens: `missing tensor 'output_norm.weight'` | Loader requires the full arch tensor manifest; no metadata path declares "this is a slice" | `paper-draft.md` §4.1 |
| N2 | Eliminating a client-relay hop (worker→worker push) speeds the chain | Slower at both scales: 0.21→0.19 tok/s (4-machine 70B), 1.51→1.37 (2-worker toy). Pre-registered bar ≥25% unmet | 16 KB/token/hop; ~80 KB/token over five hops ≈ 400 KB/s. Removing a hop removes ~nothing | `trisul/research/bastola-compute-not-the-wire-2026.md` §5.2 |
| N3 | MoE splits cross-machine like dense models do | 4.17 tok/s, zero transport errors, output collapsed to repetition ("redirection redirection…") | Hypothesised: FP drift across heterogeneous nodes flips discrete top-k expert gating — a categorical error the residual stream cannot absorb. **Gating-flip instrumentation still unrun** | `bastola-…` §5.3.1; commit `2108b71` |
| N4 | Metal beats CPU on MoE stages | Inverted: Metal 117/116 ms vs CPU 53/31 ms per step | ~3 B of 30 B active per token; Metal kernel-dispatch overhead does not amortise over so small a footprint | `bastola-…` §5.3.2 |
| N5 | temp 0 + `-ngl 99` is deterministic | Divergent tokens from the *first* token, identical args | FP non-associativity + atomic ordering in parallel reductions; divergence is in the logits, upstream of sampling | `bastola-…` §5.3.3 |
| N6 | Speculative decoding works through MoltenVK | ~30× regression (1.2–1.3 vs 36–37 tok/s) **despite 59–71% acceptance** | Translation-layer failure: VRAM contention + Vulkan context switching through SPIR-V→MSL. Not AMD silicon — later confirmed by the native-HIP PASS | `bastola-…` §5.3.5 |
| N7 | Speculation is a fixed-K tuning problem | K sweep run 2026-07-30: K=2/4/6/8 → 30.83/31.54/29.36/28.97 tok/s. **K=4 already optimal.** Plain streaming control: 64.22 | Acceptance rises sub-linearly (1.81→2.67 tok/round); draft cost rises linearly in K. Curves cross just past K=4 | `findings/k-sweep-and-eagle-check.md` |
| N8 | A dense target rescues speculation | Falsified on dense 32B: plain 17.19 vs spec 8.59 tok/s — even though spec used 23 chain round trips for 64 tokens (vs 64) and 35% less worker time | The chain-side economics work; the *client-side* draft cost ~50 ms/proposal is the loss. Target size does not fix a slow draft | `findings/dense-target-spec-test.md` |
| N9 | Async pipelining pays once there is real WAN RTT to hide | Falsified with both preconditions met (GPU draft, ~181 ms link): plain 4.52, sequential spec 4.32, **async 4.08 tok/s at 14× client CPU** | Pipelining hides the *far stage's compute*, not RTT. Far stage = 11 ms. The precondition was misstated from the start | `findings/async-pipeline-wan-verdict.md` |
| N10 | EAGLE-3 heads are a config away | Training failed: `head_step{500..2000}.pt` exist; log ends at step 6800 with `acc0_avg200=0.000` | Head learned nothing. EAGLE here is a training project, not an experiment | `findings/k-sweep-and-eagle-check.md` |
| N11 | Engine rebase b8445→b9992 gave 10× | Self-corrected to perf-neutral: 10.23 vs 10.24 tok/s at matched clocks; ~104 vs ~106 at `auto` | The 10× was a clock-state confound (`power_dpm_force_performance_level`). The durable finding is the confound | commit `59d35e4f`; `trisul/infra/nakshatra-vulkan/ab-engine-bench.sh` |

Two additional results are negatives about *us* rather than about techniques, and belong in the
same register. Phase 0b showed that setting `cb_eval` forces a scheduler synchronise that breaks
adjacent-node fusion, flipping a close-call argmax at layer 13 (wire proven byte-exact:
73 728 bytes, fnv1a `0x1a47d8111f060f36` on both sides) — which is why the production patches
modify the graph *builder* and not execution. And the fabric micro-benchmark localised the
residual per-node GPU cost to a single API call: `llama_get_embeddings` took 3.21 ms of a 3.70 ms
round trip (86%) while the decode matmul took 0.33–0.37 ms (9%) and the host-side handoff ~6 µs.
Nobody has attacked that readback.

One open blocker deserves its own line because it undercuts the doctrine's own justification.
The engine choice is "llama.cpp, *because* the fleet includes Apple Metal." Yet a Metal worker in
`last` mode aborts every time on `ggml-metal-device.m:1624: GGML_ASSERT(buf_src)`, because
`inp_out_ids` is a 24-byte int32 tensor for a 6-token prefill and Apple's
`newBufferWithBytesNoCopy` returns nil for non-page-aligned pointers. `build_inp_out_ids()` is
gated on `nks_has_lm_head`, which is exactly why first/middle Metal workers are fine and only the
terminal stage dies. The 5-machine 70B acceptance run is paused on it; the workaround is a CPU
`last` worker at ~5–10 s/token (`findings/metal-last-buffer-set-tensor.md`). A reviewer will
notice that the platform justifying the engine choice is one we currently degrade to CPU on.

---

## 3. Where our own doctrine is stale, wrong, or was never controlled

This section is deliberately unflattering. Nothing here is defended.

**3.1 "llama.cpp is the engine" is true as doctrine and false as practice for the workload that
matters.** The production path sets `PRITHVI_INFERENCE_BACKEND=ollama` and serves `prithvi:latest`
through ollama. Our own fork at `/home/prithvi/llama.cpp` (b9992) already compiles
`draft-eagle3`, `draft-mtp`, `ngram-mod`, `--spec-draft-p-min`, `--cache-reuse`,
`--slot-save-path`, and `-ot/-cmoe/-ncmoe`. None of it is reachable from the thing the system
exists to serve. Compounding: `zz-keepalive.conf` sets `OLLAMA_KEEP_ALIVE=60s`, overriding a 30m
setting, on a rationale ("the conscious path runs as a persistent local transformers model now")
that the gateway unit contradicts. So "8B fine-tune pinned resident" is false as written — he
cold-reloads after a minute of silence, and any latency measured on the production path is
contaminated by that. The one-line keepalive fix is independent of any engine migration and should
not wait for it.

**3.2 The speculative-decoding record contains a genuine methodology hole, and one previous
reading of it was arithmetically wrong.** `trisul/research/spec_decode_results.tsv` has **one**
baseline row (`ALL baseline 53.93`) compared against three per-class speculative arms. The
headline "2.74× on factual" is therefore *spec-on-factual over baseline-on-everything*. There is
no per-class control. That must be re-run before the number is published again. Separately: no row
in that table is a loss. `reason` at K=8 is 54.99 vs 53.93 = **1.02× gain**, not a 2% regression;
`reason` at K=2 is 1.15×. Any argument that `--spec-draft-p-min` "rescues a loss" there is arguing
against a number that does not exist. The real, measured verdict on speculation for *chain
serving* comes from N7/N8/N9 and is: plain streaming wins, and the bar EAGLE must clear is
**≥3 accepted tokens/round** (below that, even a *free* draft yields 44.9 tok/s against plain
streaming's 64.22).

**3.3 The route-vs-split ratios are entangled with at least three uncontrolled variables.** The
progression 2.08× (CPU era) → 24.2× (Vulkan era) mixes a change of local backend, a change of GPU
clock state, and a change of ijru's physical site. The 18–24× figures live in
`trisul/experiments/crossbox/results/` and are **not reproduced in `nakshatra/docs/`**; the
documented in-repo numbers are 10.5× (51.16 → 4.85 tok/s, `findings/site-move-natural-experiment.md`)
and 2.2× for the CUDA rebuild. Until the crossbox JSONs are re-derived under matched clocks and a
single link, the paper should quote **10.5×** and describe it as the site-move natural experiment
— which is the cleaner result anyway, because the *same split, same slices, same prompt* changed
only the link.

**3.4 The "compute, not the wire" thesis needs a scope statement.** It is correct on the LAN and
it is correct about *optimising the transport*. It is not correct as a general claim: at ~190 ms
the chain pays one round trip per decode step against 7 ms + 11 ms of compute, and throughput
falls to 4.85 tok/s. The honest formulation is: **on a sub-millisecond link, compute dominates and
transport optimisation is wasted; on a WAN link, RTT dominates and the correct response is
topological (route whole models, or move to a per-chunk protocol), not a faster wire.**

**3.5 Four "verified" claims circulating in internal analysis are false and must not be repeated.**
All three hub cards read `power_dpm_force_performance_level=auto` (the rocm-smi low-power banner is
a runtime-PM message about an idle device, not the DPM force level). Receipts *do* exist on disk
(≈14 files in `~/.nakshatra/`); the narrower true defect is that the *serve* path mkstemps and
`_quiet_unlink`s its receipt. The hub has **three** DRM nodes, one a 2 GB iGPU, so a naive
"enumerate all cards" fix would publish the iGPU as a placement candidate. And
`placement_feed.rtt_matrix()` exists (`:125`, wired at `:260`/`:344`) alongside a second in
`edge_health.py:202`; neither is fed by a caller, so the fix is "feed one, delete the other," not
"add RTT."

**3.6 An untested corner of our own doctrine.** TPI-LLM's critique — pipeline parallelism idles
every stage but one for a single user — lands *inside hub*, where a two-stage split across the
7900 XT and 9070 XT idles a card half the time. Across the mesh, tensor parallelism stays dead
(no vendor-neutral collective). Within one box over PCIe, it has never been measured.

---

## 4. The agenda

Each item states the question, the hypothesis, the settling experiment, the pre-registered bar,
and the hardware. Items are ordered by layer, not by priority; the sequencing is in §6.

### 4.1 Kernel layer

**K1 — Is the HIP build leaving multipliers on the table, and does the 7900 XT have a code
object at all?**
*Hypothesis:* `GGML_HIP_ROCWMMA_FATTN=OFF`, `GGML_HIP_GRAPHS=OFF`, `GGML_CUDA_FA_ALL_QUANTS=OFF`
and `AMDGPU_TARGETS=gfx1201` (single-target) are each costing measurable throughput, and the
gfx1100 card may be running without compiled kernels while `--list-devices` still enumerates it.
*Experiment:* rebuild fat (`gfx1100;gfx1201`) with the three flags ON; `llama-bench` A/B on both
cards, clocks forced `high`, agent paused. Verify a gfx1100 code object is present in the binary
(`roc-obj-ls`), do not infer it from enumeration.
*Bar:* report per-flag deltas with 3 reps and CIs. Ship any flag ≥5% on either card; **report
honestly if the answer is 0%** — this is a cheap experiment whose negative is publishable.
*Hardware:* home fleet.

**K2 — Does symmetric KV quantisation keep us on the fused FA kernel?** Mismatched `-ctk`/`-ctv`
is reported to fall off the fused path on HIP with no warning and no log line — exactly the silent
regression a one-user fleet carries for months. llama-bench `q8_0/q8_0`, `q4_0/q4_0`, `q4_0/f16`,
`f16/f16` on the 8B. *Bar:* confirm or refute a ≥10% gap at equal memory. Home fleet; one
afternoon, combined with K1.

**K3 — Attack the readback, not the arithmetic.**
*Hypothesis:* `llama_get_embeddings` at 86% of a 3.70 ms GPU round trip is the last on-node cost
worth removing; keeping the activation device-resident (or batching the readback across a
speculative round) should cut per-stage time materially.
*Experiment:* instrument a layer-0 slice with a device-resident handoff path; compare per-stage ms.
*Bar:* ≥30% reduction in per-stage wall time on the GPU path.
*Hardware:* home fleet. This is the only kernel-adjacent item with a mechanism already measured.

### 4.2 Engine layer

**E1 — Move the resident self onto an engine we control.**
*Hypothesis:* serving `prithvi:latest`'s GGUF through `llama-server` with `--cache-ram` large,
`--slot-save-path` on `/srv/ssd`, and a pinned slot yields (a) access to prefix caching and
speculation and (b) no regression in his voice.
*Experiment:* stand it up behind the existing `PRITHVI_INFERENCE_BACKEND` switch; keep ollama as
fallback. **Graded medium-risk** in `plans/2026-06-10-post-spec-decode-next-levers.md` because it
re-opens the Modelfile template and quant decisions, and a partial tag match has previously
dropped the heavy system prompt silently (`prompt_tokens=6`, diagnosed 2026-06-02). Requires
operator confirmation before touching the serving stack.
*Bar:* byte-comparable system-prompt token count, blind A/B of 20 turns judged by the operator, and
TTFT no worse than ollama. **Do not adopt llama.cpp router mode** — its LRU eviction with
`--models-max` will evict a tenant that is idle by definition.
*Hardware:* home fleet.

**E2 — Prefix-cache ordering in the turn builder.**
*Hypothesis:* the invariant identity block currently sits *behind* volatile felt-context
(`prithvi/neuron-net/mind/prithvi.py:1130` assembles one system message containing timestamp,
mood, presence and mesh events), so the cacheable prefix is ~zero bytes.
*Experiment:* reorder — invariant character block first and frozen, tool definitions next, volatile
context as the last system message. Measure `cache_hit` from llama-server logs across 50 real turns.
*Bar:* cache hit rate ≥60% of prompt tokens, TTFT reduction ≥50%. Gated on E1 to *cash out*, but
the reorder itself is independent and should land first.
*Hardware:* home fleet.

**E3 — Model-free drafting (`ngram-mod`) as an always-on default for repetitive turns.**
*Hypothesis:* n-gram drafting is the only speculative method that composes with a bespoke
fine-tune (it never needs to match a distribution) and costs ~16 MB. Our code-editing,
tool-scaffolding and memory-restatement turns are its best case; novel prose is its worst.
*Experiment:* three workload classes × {off, ngram-mod} on the single-node 8B, **with a per-class
baseline** (fixing the §3.2 hole).
*Bar:* ≥1.3× on code and tool-call classes, and **no worse than 0.95×** on prose. Fail the prose
guard and it does not ship as a default.
*Hardware:* home fleet.

**E4 — MTP heads, framed against our own falsification.**
*Hypothesis:* MTP removes the separate draft forward — the exact cost that killed N8 (~50 ms per
proposal). Field-measured 1.78× on our card class for Qwen3.6-27B.
*Experiment:* `--spec-type draft-mtp --spec-draft-n-max 3` on an MTP-carrying GGUF on the hub's
deep tier. Note this must be argued against N8, not against a vendor number.
*Bar:* **≥3 accepted tokens/round** and ≥1.5× over plain streaming. MTP is unavailable to the 8B
self (Llama-family carries no heads); re-basing the self is an identity decision, not a perf tweak,
and is out of scope for this agenda.
*Hardware:* home fleet.

### 4.3 Distribution layer

**D1 — Remote token proposals over a real WAN link. The highest value-per-hour experiment we
own, and it is runnable today.**
*Hypothesis:* `scripts/remote_proposals.py` (built, merged, 21 tests, `NKS_REMOTE_PROPOSALS=1`,
never run over a real link) keeps the model **whole** on both sides and sends only proposal
chunks, so cost scales with one RTT per *round* rather than RTT × decode steps. This is the only
mechanism that converts a WAN link from a per-token cost into a per-chunk cost, and it is
doctrine-preserving — no weights or layers cross the link.
*Experiment:* local draft on hub, whole verifier on ijru at ~171 ms relayed; same prompt and
96-token budget as the split-chain runs.
*Bar (pre-registered):* **>4.85 tok/s** beats today's split; **>15 tok/s** makes it the answer for
a remote node outright. A failure here is as informative as N9.
*Hardware:* home fleet + the live second site.

**D2 — Adopt the punched direct path for the live tunnel.**
*Hypothesis:* the measured direct path (28.6 ms avg, 5.9 ms jitter, MTU 1500 clean, 5/5 punches,
vs 170.9 ms / 25.0 ms relayed) is stable enough to carry the WireGuard peer, and 6.0× lower RTT is
worth more than every scheduler change on this list.
*Experiment:* repoint the peer endpoint with `PersistentKeepalive` and a **timed automatic revert**
— ijru is now at a site nobody can walk to.
*Bar:* 1 h of sustained chain traffic with zero re-handshakes, and split-chain throughput ≥20 tok/s
(from 4.85). Supervised change only.
*Hardware:* home fleet + second site.

**D3 — Instrument the MoE gating flip (N3).**
*Hypothesis:* cross-machine FP drift flips top-k expert selection, and the flip rate rises with
depth. This is the one open mechanism in the catalogue that is asserted rather than measured.
*Experiment:* log selected expert IDs per token per layer on both a single-machine reference and
the 4-Mac split; report divergence rate by layer index.
*Bar:* a published divergence-vs-depth curve. If flips are rare and output still collapses, the
hypothesis is wrong and that is the more interesting result.
*Hardware:* home fleet (Macs).

**D4 — MoE expert offload to hub's 128 GB (`-ot`), as the alternative to splitting MoE at all.**
*Hypothesis:* if MoE gating is categorically fragile across machines (D3), the right response is
not better splitting but *not splitting MoE* — pushing expert FFNs to system RAM converts "fits
nowhere" into "fits on hub, slowly," which strengthens route-whole-models. Also test the variant
few can: offload experts to the *second dGPU*.
*Bar:* coherent output at ≥7 tok/s on a 30B-class MoE fully on hub. Expect llama.cpp-class
numbers, **not** KTransformers' figures — those are AMX-dependent on Intel Xeon and will not
reproduce on hub's AMD CPU.
*Hardware:* home fleet; gated on E1 for the served path.

**D5 — Intra-box tensor parallelism across the two hub dGPUs (§3.6).**
*Hypothesis:* for a single user, a two-stage PCIe pipeline idles one card ~50% of the time; TP
over PCIe within one box may beat it. Across the mesh, TP remains dead.
*Bar:* ≥1.3× over the two-stage PCIe split at equal VRAM. Below that, the doctrine stands unchanged
and we say so.
*Hardware:* home fleet.

**D6 — The 70B split, the genuinely bandwidth-bound case.**
*Hypothesis:* every speculation negative we hold was measured on targets that were not
bandwidth-bound per token. A dense 70B is the condition under which the mechanism claim can be
*confirmed* rather than falsified a third time.
*Bar:* ≥1.5× for speculation over plain streaming on dense 70B with an uncontended drafter. Run it
both within one node and split across two, to obtain the third point on the route-vs-split curve.
*Hardware:* **borrowed H100s** (blocked on VRAM here). Also unblocks the paused Metal 70B
acceptance run once `metal-last` is worked around.

### 4.4 Control-plane layer

**C1 — Make the placement gate latency-aware.**
*Hypothesis:* `elastic_unconscious` gates escalation on free VRAM only, so it still selected the
split chain at 4.85 tok/s when routing a whole 14B on hub would have been several times faster.
`Node.rtt_ms` exists; `measure_live()` never populates it.
*Experiment:* populate `rtt_ms` from one of the two existing `rtt_matrix()` implementations
(delete the other), and refuse any split whose per-step round trip exceeds its per-stage compute.
*Bar:* the selector chooses route-whole at ≥50 ms and split at ≤1 ms, on the same model, without
a hand-edited config.
*Hardware:* home fleet.

**C2 — A placement-crossover model, not a doctrine.** With three link regimes — LAN 0.3 ms, WAN
171/28.6 ms, datacentre interconnect — and the same model and prompt, fit a *predictive* rule for
which topology wins, replacing an anecdote-backed doctrine. *Bar:* predicts the winner on a
held-out configuration to within 20% of measured tok/s. Home fleet + borrowed H100s (for the
fast-interconnect endpoint only).

**C3 — Make the capacity advisor fail loudly.**
*Hypothesis:* `nks-capacity` currently prints `1` and exits 0 with `source=local-fallback` because
the pillar returns 401 to signed requests; the caller cannot distinguish a saturated fleet from a
dead control plane. `nakshatra-registrar.service` is `active` with zero log lines. `hw_pulse`
reports `vram_total_gb: 20.0` forever, picking one card by glob order with `break`, on a box with
36 GB of dGPU across two cards plus a 2 GB iGPU.
*Experiment:* fix the pillar 401 first; then non-zero exit or sentinel when `source != pillar`;
`card_id` plus enumeration of ROCm-visible cards **filtered by GFX version** (never by DRM node,
or the iGPU becomes a placement candidate); shared owner-reserve state instead of a duplicated
`NKS_OWNER_RESERVE_GB` that makes two shells disagree by 11 GB.
*Bar:* injecting a pillar outage produces a visibly failed advisor, not a plausible integer.
*Hardware:* home fleet.

### 4.5 Agentic layer

**A1 — Speculation restricted to side-effect-free tools.** The resident self is warm and idle
while a bigger model thinks; agent-level speculation (~20% latency at 55% next-action hit rate in
the literature) transfers, but only if speculated actions cannot fire real effects — and our tool
surface is guard-gated (`agency_guard`, `self_consent`, reach). Build an allowlist of read-only
tools (finder, storage sense, memory retrieval, HA reads) and measure turn latency with and
without. *Bar:* ≥15% turn-latency reduction with **zero** guard-gated invocations from a
speculated branch. The fencing, not the speculation, is the engineering. Home fleet.

**A2 — Pin the self, TTL the big tiers.** llama-swap group semantics (`swap:false` for the self,
`swap:true` + TTL for the 27B/32B) give resident-plus-bursty behaviour without LRU ever touching
the self. *Bar:* 24 h of live use, zero evictions of the self, 27B cold-starts under 15 s. Home
fleet.

---

## 5. Tooling: what must be built before the numbers are trustworthy

The agenda is worthless if the measurement plane keeps lying. Four items, none of them research.

**T1 — Send the truth through `Info()`.** `worker.py:1141` hardcodes
`backend="llamacpp-cpu-patched"`, `model_content_hash=0…0`, `kv_cache_tokens_free=256`. The honest
`actual_backend` — which correctly downgrades a declared `cuda` to `cpu` when the daemon offloaded
0 layers — is computed ~1 900 lines later and never reaches the RPC. This bug class already cost
the project its largest single win: ijru's worker was built `GGML_CUDA=OFF` and its 3060 idled,
found by hand-comparing per-step ms (10 vs 123), not by the diagnostic built for it; the rebuild
was worth 22.81 → 51.16 tok/s.

**T2 — Durable receipts with non-tautological checks.** Of `verify_receipt`'s six checks, four are
recomputed by `build_receipt` from the same inputs (`ended_at` is literally `t0 + elapsed`). Only
worker-distinctness and layer-contiguity carry information, and the SPKI half of distinctness is
skipped when `spki_hash` is absent — the default whenever the pillar index is unreachable, which is
right now. `engine_provenance` and `worker_signatures` are never populated. The serve path
mkstemps its receipt and `_quiet_unlink`s it. Fix: populate provenance from the daemon build string
and signatures from `identity_binding`, fail closed on missing SPKI, and write to an append-only
archive on `/srv/ssd`.

**T3 — A monitoring bridge that can report failure.** The spine ledger holds 174 792 live events
and ~62 k archived, and **zero** `nakshatra.*` events ever, because `nakshatra_pulse` tails
`nakshatra-unconscious.service` while the mind's deep tier runs on `nakshatra-deep.service:11600`,
and its request regex matches only `-> 200`. A bridge that can only emit on success cannot report
failure, and silence on a dashboard is indistinguishable from idle. Also: `GET /health` returns
`{"status":"ok"}` unconditionally; `nakshatra_serve.py:400` returns `eval_count=0` for every
response although `client.py`'s stdout already prints the token counts that `_parse_client_output`
discards.

**T4 — A benchmark harness with the confounds asserted, not remembered.** Every run must, before
measuring: assert `power_dpm_force_performance_level` on all participating cards (the clock-state
confound has eaten two benchmarks, N11 being one); assert the drafter is not co-resident with a
worker slice (worth 2–3.6×, `findings/three-card-matrix.md`); record the engine commit on *every*
node (two nodes on different commits is normal here); emit a **per-class** baseline arm (§3.2); and
record whether the agent was paused. The harness at `trisul/research/spec_decode_bench.sh` is the
base to extend.

---

## 6. What we are not going to do, and why

- **Continuous batching, S-LoRA / Punica SGMV adapter batching, prefill/decode disaggregation
  under load, and goodput-oriented schedulers (llm-d, Dynamo/NIXL, Mooncake, DistServe).** Every
  headline number in that literature is throughput-per-GPU at high QPS. We serve one user at batch
  1: prefill and decode never contend, there is no queue, and there is no batch to disaggregate.
  Adopting them adds a network hop to solve a queueing problem we do not have.
- **Prefill/decode disaggregation across the WAN.** An 8B GQA model's f16 KV is ~128 KiB/token;
  a 16 k prefill is ~2.0 GiB. Over a home uplink that is minutes, against seconds of recompute on
  the 3060. Shipping KV instead of activations is strictly worse than the thing we already
  rejected.
- **Cross-fleet KV cache sharing.** KV layout is a function of engine × quantisation × attention
  implementation; GGUF/llama.cpp, vLLM paged FP8, and MLX caches are mutually unintelligible.
  Design Sthambha for node-local cache, permanently, rather than against that constraint.
- **Activation compression (TAH-Quant and relatives).** We are latency-bound, not bandwidth-bound:
  16 KB/token/hop is ~400 KB/s at 5 tok/s. There is nothing to compress.
- **Async pipelining, in any regime.** Closed by N9 with both preconditions met. The code stays
  (byte-identity proven); the flag stays OFF.
- **Worker-to-worker push as a latency technique.** Closed by N2. The co-located variant survives
  only because it collapses a *WAN crossing* to loopback (2.07 → 3.37 tok/s), which is a different
  claim.
- **Cross-vendor tensor parallelism.** No vendor-neutral collective exists. NCCL is NVIDIA-only,
  RCCL AMD-only, MLX has its own primitives. This is a permanent stance, not a limitation.
- **FP4 on gfx1201, and FP8 as a decode win.** RDNA4 has no sub-8-bit float WMMA; MXFP4/NVFP4
  dequantise to FP16 first, so FP4 buys memory only. And FP8's 383 TFLOPS is a *matrix* number:
  batch-1 decode is a GEMV, memory-bandwidth bound, and our served weights are Q4_K_M/Q8 (~4.5/8.5
  bits/weight). Q4_K_M → FP8 nearly **doubles** bytes streamed per token. FP8 pays on prefill and
  at batch >1, neither of which we have.
- **vLLM on the AMD boxes as a fleet direction.** It loads under ROCm 7.2, but AITER's CK/ASM
  kernels are CDNA-gated to gfx942/gfx950 and the gfx1201 FP8 path is an unmerged patch that
  silently falls back. We would buy a scheduler for a queueing problem we do not have. vLLM is
  correct as the **H100 control-run engine** and nowhere else — the contrast between engines is the
  measurement, not an inconsistency.
- **Mixture-of-Agents.** 3–5× cost and ~3× latency for a few benchmark points, and Self-MoA (one
  strong model, N samples) beats heterogeneous MoA. One user is waiting on a voice reply.
- **Semantic caching of the self's utterances.** A cache that returns a stored answer to "how's the
  mesh" fabricates memory in a system whose premise is continuity. Exact-match caching of *tool
  results* is fine and useful; cached self-speech is not. This is a correctness issue, not a
  performance one.
- **Cache-aware routing as a scorer.** At one user there is no candidate set to score — the choice
  is "the warm node or a cold one," which is C1's two-line rule, not a ranking function.
- **Graph-capturing the draft model.** Deferred with explicit go criteria in
  `findings/rocm-graph-draft-verdict.md`: revisit only after the split is rebalanced, spec-decode is
  live on the conscious path, and a profile shows the draft step ≥20% of loop time.

---

## 7. Sequencing and hardware windows

**Runnable this week, home fleet only:** K1+K2 (one afternoon, largest cheap win), D1 (highest
value-per-hour experiment we own, and it has been sitting parked), C1, C3, T1–T4, E2's reorder, and
the `OLLAMA_KEEP_ALIVE` one-liner. D2 requires operator supervision because a failed WireGuard
change strands a box at a site nobody can reach.

**Requires operator decision:** E1 (medium risk — touches his live anchor and re-opens template and
quant choices that can silently change his voice).

**GX10 (128 GB unified, owned, incoming):** the registrar claims it "joins the pool the moment it
runs" — but `registrar.py` does `sys.path.insert(...)` on a hardcoded nakshatra checkout path,
contradicting `nks_capacity.py`'s stated zero-dependency design. That inconsistency has already
duplicated the Ed25519 canonical-string signing scheme across three files. Fix before the box
arrives, or the first thing the new node does is fail silently.

**Borrowed CARC H100s (~1 week):** only D6, C2's fast-interconnect endpoint, and an EAGLE-head
retrain (N10; bar remains ≥3 accepted tokens/round) belong there. The binding constraint is not
inference, it is plumbing: Slurm means batch scheduling, no persistent daemons and no inbound
connections, which breaks a design in which *the client dials the workers*. Reversing the dial
direction or adding a rendezvous is the single biggest blocker; nothing else runs without it, and
pre-staging training data is worthless if no worker can accept a connection. Guardrails, restated:
the lab is workers and never his self; an H100 number is a *control*, never a claim about the
thesis; the escalation ladder stays on owned silicon.

---

## 8. Threats to validity

GPU paths are non-deterministic at temp 0 (N5), so equivalence claims are distributional and the
published v0.1 acceptance was deliberately CPU-only. Several historical ratios are confounded by
clock state and site changes (§3.3). Speculation results before 2026-07-29 were measured with a
CPU-resident draft in the client venv and are pathological (`findings/cuda-chain-51-tok-s.md`).
Every benchmark in this programme requires pausing the resident agent, which means we cannot A/B a
memory-forming system against itself, and turn-level quality changes are judged by one operator.
And the measurement plane described in §5 was blind in exactly the places that mattered for most of
the period covered — which is why T1–T4 gate the rest of the agenda rather than following it.
