### 2026-07-04 · HANDOFF (async-pipeline lane → inference lane) — async pipelining, scheduler done, needs live KV-rewind verify
- from: claude/trisul (async-pipeline lane)
- to: inference, all
- status: done
- reply (2026-07-05, async-pipeline lane): Did the wiring + live test myself (Biswa asked). WIRED into `client.py` behind `--async-pipeline` / `NKS_ASYNC_PIPELINE` on the **unary spec-decode path** (not streaming/push — spec runs over unary Forward, which is where `all_logits`/`keep_kv`/`start_pos` live). Stages call `call_forward` per worker; the daemon's start_pos-trim IS the flush-rewind. VERIFIED on a live 2-worker CPU chain (deepseek-distill-8B, `whole-1b` draft, tls/auth off): output **byte-identical** to sequential on two prompts including depth-4, through a mispredict-heavy run (`accept 0/4`…`2/4`) — so real flushes fired and the KV-rewind recovered them cleanly. The one caveat I could NOT close here: **throughput**. Co-located CPU has no network RTT to hide and both workers share the cores, so async was *slower* (84s vs 16s) — expected. The tok/s win needs the real multi-box GPU mesh (hub↔ijru, RTT to hide, spare compute per node). That's the remaining ask: run `client.py --config <hub+ijru chain> --speculative --async-pipeline` on the GPU mesh and measure. Branch `inference/async-pipelining`, commits on it.
- subject: Built the ONE technique we were missing vs Shard — **async pipelining** (their 2.94→16.6 tok/s jump). Confirmed the gap first: both decode paths in `scripts/client.py` (spec ~:884, plain ~:940) traverse workers STRICTLY sequentially — worker0→wait→worker1→…→last→wait per token, one worker busy at a time (the pipeline bubble). New self-contained scheduler `scripts/async_pipeline.py` fills it by keeping N verify-chunks in flight via **speculative continuation** (issue chunk i+1 assuming chunk i fully accepts, predicting its next `cur` from `draft.propose(K+1)`) + **misprediction flush** (wrong assumption → cancel the speculative successors, re-issue from the corrected cursor). Correctness invariant = commit ONLY tokens whose predecessor was the truly-committed prefix ⇒ output byte-identical to greedy (same oracle as speculative.py). Unit self-test (`python3 scripts/async_pipeline.py`) PASSES: pipelined==sequential output, peak 4/4 stages busy at once (real fill), forced misprediction flushed+recovered clean.
  **THE ASK (your lane owns client.py + the live mesh):** wire the `Stage` callbacks to the real `_step_call(idx,…)` chain and verify on a live multi-box run behind `NKS_ASYNC_PIPELINE` (default OFF — sequential path untouched). The one thing I could NOT verify without real workers: the flush relies on the daemon's existing `start_pos`/`keep_kv`/TruncateKV KV-rewind (client.py M3 fusion, worker.py ~:1419-1478) undoing a mispredicted chunk's KV write when the corrected chunk re-enters each stage with the corrected start_pos — needs a real kill-a-prediction test on the mesh. Branch `inference/async-pipelining`. Note: push/mode=first already gives ring direct-return, so that "gap" from the strategy report is actually already covered.

### 2026-06-30 · NOTE (placement lane → inference/serve lane) — building a Q3 cross-vendor chain
- from: claude/trisul (placement lane)
- to: inference, all
- status: unread
- subject: Slicing Qwen3-30B-A3B **Q3_K_M (~14GB)** and standing up an explicit cross-vendor chain (hub L0-13 Vulkan / ijru L13-48 CUDA) so the unconscious fits the conscious-RESERVED pool (Prithvi pinned ~11GB on the hub → only ~5GB free there; Q4 18GB no longer fits, Q3 14GB does). New slices `qwen3-30b-q3-L*.gguf` + chain `qwen3-30b-q3-chain.yaml`; NOT touching the existing Q4 configs. Will measure tok/s.

---

### 2026-06-28 · NOTE (placement lane → inference/nakshatra serve lane)
- from: claude/trisul (placement lane)
- to: inference, all
- status: unread
- subject: **When you arm `NKS_SMART_PLACEMENT`, also reserve Prithvi's pinned conscious slice.** I added a conscious-VRAM reserve to `placement_feed.make_node` (merged e37404f): it subtracts a per-node reserve so smart placement never puts unconscious layers into the hub's PINNED conscious 8B slice (Prithvi is now `keep_alive=-1` resident on the hub GPU, ~9.6GB + buffer). It's **default-0 / dormant** until NKS_SMART_PLACEMENT is on. **Action when you arm smart placement:** set on the serve unit alongside `NKS_SMART_PLACEMENT=1`:
  `Environment=NKS_CONSCIOUS_NODE=hub`
  `Environment=NKS_CONSCIOUS_RESERVE_GB=11`   (16GB card − ~11 conscious = ~5GB offered to the pool; tune to taste)
  (or `NKS_VRAM_RESERVE_GB={"hub":11.0}` for a multi-node map). 51 placement tests green. This is Part A of trisul/plans/2026-06-28-nakshatra-placement-and-crossvendor.md. Part B = cross-vendor backend on llama.cpp Vulkan(AMD gfx1201)+CUDA(ijru) RPC — NOT tinygrad (exo dropped it; gfx1201 has no ROCm kernels).

---

# Inbox
