# ROCm graph-captured draft — research verdict (2026-07-29, infra ride)

**What it is.** Shard's final speed jump (16.6 → ~30 tok/s) came from CUDA-graph-capturing
the draft model's decode step: the small model's per-token kernel-launch overhead dominates
once everything else is pipelined, and graph replay amortizes it. The ROCm equivalent is
hipGraph; llama.cpp's CUDA-graph path largely compiles under HIP on recent builds.

**Their own caveat.** Shard's notes say graph capture "barely" pays outside their regime —
it matters only when the draft step is a first-order cost in the decode loop.

**Our measured reality (tonight's GPU-mesh proof, receipts in ~/.nakshatra/):**
- Sequential spec-decode on the Q3 cross-vendor chain: 4.58 tok/s. The loop is dominated by
  ijru's verify stage (123 ms/step — compute on 35 layers), not by the 0.6B draft.
- The place draft cost DID explode was the async-pipeline path (~7× client CPU from
  speculative continuation re-proposals) — but that path is now ruled WAN-regime/OFF at home,
  and its fix is proposal reuse, not graph capture.

**Verdict: NO-GO for now.** Graph-capturing the draft optimizes a cost that is not on our
critical path. Revisit ONLY after (in order):
1. the layer split is rebalanced (hub 10ms vs ijru 123ms — the real LAN lever), and
2. spec-decode goes live on the conscious 8B serve path, and
3. a profile then shows draft step ≥ ~20% of loop time.
If all three hold, the experiment is: llama.cpp HIP build with GGML_CUDA_GRAPHS on gfx1201,
llama-bench the 0.6B draft step with/without, on the 9070 XT at perflevel high (NOT the
soma-throttled default — see reference_gpu_perflevel_soma_throttle).

This closes the last row of the Shard adoption table: three techniques merged and proven,
one (async pipelining) proven-and-correctly-shelved for LAN, this one consciously deferred
with measurable go criteria.
