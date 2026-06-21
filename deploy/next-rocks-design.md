# Next rocks — design + status (2026-06-21)

Status of the four rocks Biswa asked for ("do all"):
- **#4 TTL placement directory → live: ✅ DONE + tested** (`slice_directory.py`
  per-node files + `slice_node.py` serve+heartbeat + `from_env` `NAKSHATRA_SLICE_DIR_PATH`).
- **#2 Big-model placement core: ✅ DONE + tested** (`placement.py`: route-whole,
  metro-clusters, contiguous-span VRAM-capped assignment, one-cluster preference).
  REMAINING (needs roster/daemon integration, not pure-python): (a) feed `plan()`
  output into the roster/`RosterWorkerController` so it actually launches the
  assigned spans; (b) **MoE expert-placement** — for frontier MoE models, place
  *experts* (only the firing ones cross the wire) instead of contiguous layers;
  (c) FLOPS+RTT water-filling for heterogeneous GPUs (PipeDream-style DP).
- **#1 Sleep-instead-of-reap: DESIGNED below** (C++/HIP — dedicated effort).
- **#3 Logit-subset surgery (head unlock): DESIGNED below** (cnets + GPU — dedicated).

---

## #1 — Sleep-instead-of-reap (the <1s wake)

**Goal:** today reaping kills the worker → next request pays the full ~7.5s cold
start (process spawn + ROCm init + disk read + VRAM upload). Instead, on idle,
keep the process ALIVE, free only VRAM, hold dequantized slices in **pinned host
RAM**; wake = async H2D DMA → likely <1s. (vLLM "sleep mode" pattern, ported to
ROCm via HIP; NVIDIA CRIU-snapshot/GPUDirect don't exist on gfx1201.)

**Where:** `experiments/v0.0/worker_daemon.cpp` (the llama.cpp worker). Steps:
1. Add a control op `cmd=SLEEP` and `cmd=WAKE` to the daemon's command loop.
2. SLEEP: `llama_free` the GPU model/context but keep the GGUF mapping + a
   `hipHostMalloc` pinned copy of the layer weights in host RAM; set a `sleeping`
   flag; respond ready-but-cold.
3. WAKE: re-create the context and `hipMemcpyAsync` weights host→device (pinned →
   fast DMA), re-warm KV; flip the flag. Measure wake latency.
4. Lifecycle: `serve_lifecycle.ChainController` grows `sleep()`/`wake()`; the
   reaper calls `sleep()` (not `stop()`) for OWNED GPUs, `stop()` only for
   borrowed ones (ownership-aware — already modeled). `warm()`/`begin()` call
   `wake()` instead of `start()` when the worker is asleep.
5. Gate: byte-identical output after a sleep/wake cycle vs no cycle.

**Risk:** HIP pinned-memory ceiling; llama.cpp context re-create cost; gfx1201
ROCm quirks. Build on the hub's gfx1201 worker (it serves prithvi-unconscious).

## #3 — Logit-subset surgery (unblock the head on 12GB)

**Problem (measured):** EAGLE-3 head training for an 8B target OOMs the 12GB 3060
for any batch >1, because `cnets.dataprepare` materializes the target's FULL
128256-vocab logits: `target = outs.logits` ≈ 128256×seq×4B ≈ **552MB/step** (the
exact failing allocation). Head training currently runs only at batch-size-1
(noisy), so it can't get *good*.

**Fix:** EAGLE only needs the **draft_vocab (32000)** columns of the target
distribution (the head predicts in draft space via `t2d`/`d2t`). Subset BEFORE
materializing the big tensor → 552MB → ~138MB, freeing ~400MB → grad-accum (real
batch) fits → the head can actually train down.

**Where:** `~/EAGLE/eagle/traineagle3/cnets.py`:
1. In `dataprepare`, after the target forward, **index logits to the draft vocab**:
   keep only columns where `t2d` is True (or gather `d2t`), so `target` is
   `[B, seq, 32000]` not `[B, seq, 128256]`. Verify the downstream `target_p`
   softmax + the `acces` argmax then compare in the SAME draft space (this likely
   ALSO fixes the near-zero training-time acc, a vocab-space mismatch).
2. Re-enable `ACCUM=4` (or 8) in `standalone_train.py` — should now fit.
3. Verify: GPU stays <12GB at ACCUM=4; `acc0_avg200` climbs past the bs=1 plateau.
4. Then run `eval_head.py` (update its draft cast to `.bfloat16()`) for the real
   held-out acceptance number.

**Risk:** getting the t2d indexing right without corrupting the distillation
target — it's the core of EAGLE-3's loss. Do on a copy, gate on a few hundred
steps showing acc climbing vs the current bs=1 baseline.

**Honest note:** even unblocked, a converged high-τ head is many GPU-hours; this
removes the *ceiling*, it doesn't make it instant.

---

## EAGLE → live: expose target hidden states from the C++ worker (the speedup payoff)

**Goal:** at inference the trained EAGLE-3 head needs the target's `hidden_states[0,1,2]`
(embedding + first 2 layers — all on the FIRST worker, which holds layers 0-16).
Today `worker_daemon.cpp` emits only the FINAL layer's hidden via `llama_get_embeddings`.
We must emit the 3 EAGLE-input hidden states so the head (Python `EagleDraft`) can draft.

**Why it's not a one-shot:** llama.cpp exposes no intermediate-layer API. Capture needs
a **ggml eval callback** (`llama_context_params.cb_eval` + `cb_eval_user_data`, set BEFORE
`llama_init_from_model`). The callback fires per tensor during graph eval; you match the
**layer-output tensors by name** and memcpy their data. **Tensor names are version-specific**
→ step 1 is discovery, not guessing.

**Executable steps (build-in-the-loop on the hub gfx1201 — local llama.cpp build, reliable):**
1. **Discovery probe: ✅ DONE 2026-06-21.** Built `llama-eval-callback` on the hub and
   dumped tensor names for this exact build. The 3 EAGLE inputs map to:
   `hidden_states[0]` = **`embd`** (GET_ROWS, embedding output);
   `hidden_states[1]` = **`l_out-0`** (ADD, after decoder layer 0);
   `hidden_states[2]` = **`l_out-1`** (ADD, after decoder layer 1).
   (Per-layer outputs are `l_out-N`; final = `result_norm`/`result_output`.) The cb_eval
   capture is now CONCRETE — match `t->name in {embd, l_out-0, l_out-1}`, no guessing.
2. **Capture: ✅ DONE 2026-06-21.** in the callback, when `t->name` matches the 3 target layers, copy
   `ggml_backend_tensor_get` into a `std::vector<float>` keyed by layer (size n_tokens×n_embd).
3. **Protocol cmd=5 EAGLE_HIDDEN: ✅ DONE+TESTED 2026-06-21.** like TOKEN_DECODE but the response payload is
   `float32 hidden3[n_tokens * 3 * n_embd]` (the 3 captured layers concatenated, matching
   cnets `torch.cat((h0,h1,h2),dim=-1)`). result_type=3.
4. **Client `EagleDraft`** (`scripts/speculative.py`): on the first shard, request cmd=5,
   assemble the 3 hidden states, run the trained head (`head_step*.pt` → the EAGLE-3
   forward: fc → midlayer → norm → lm_head → draft tokens via d2t), propose K. Keep the
   GGUF-draft path as fallback (the existing `DraftModel` interface).
5. **Gate:** byte-identical greedy output vs no-spec; measure τ + tok/s on netsim.

**Gating:** end-to-end value needs a TRAINED head (batch-4 run in progress). The C++
plumbing (steps 1-3) can be built + shape-verified independently of head quality.

**Honest note:** this is a multi-hour focused ggml build, best done with the hub GPU in
the loop (discovery → capture → verify shapes), NOT blind. Scoped + ready; flagged as the
next dedicated C++ session rather than rushed marathon-tail code.


**C++ STATUS 2026-06-21:** steps 1-3 DONE+TESTED on hub gfx1201. `worker_daemon.cpp` (backup `.pre-eagle`): EagleCapture + `eagle_cb_eval` (matches embd/l_out-0/l_out-1, host/device-safe via ggml_backend_tensor_get) + `cmd=5 EAGLE_HIDDEN` returns float32[n_tokens*3*n_embd] result_type=3. Verified: 4 toks→49152 real floats (4*3*4096); cmd=1 regression PASS (existing decode intact). REMAINING: step 4 `EagleDraft` in speculative.py (needs the trained head, ~0.60 + climbing) + step 5 gate/measure.

---

## EagleDraft (step 4) — FEASIBILITY PROVEN, build plan (2026-06-21)

**Key finding:** `~/EAGLE/eagle/model/cnets.py` is the EAGLE-3 INFERENCE twin of the
training head — `class Model` with `midlayer` (LlamaDecoderLayeremb), `fc(target_hidden×3)`,
`d2t`/`t2d`, `lm_head(draft_vocab)` — i.e. SAME architecture as my trained `head_step*.pt`.
So we REUSE EAGLE's own EAGLE-3 inference draft, not reinvent the recurrence (avoids the
subtle-wrong trap).

**Build (focused session, CPU for the head — tiny):**
1. `scripts/eagle_draft.py`: load `head_step*.pt` (trainable params + d2t/t2d) into
   `eagle.model.cnets.Model` (inference). Map our saved keys → its module names.
2. `propose(hidden3, prefix_last_token, k)`: feed the target's 3 hidden states (from the
   worker's `cmd=5 EAGLE_HIDDEN`) into the Model's draft generation; take the **top-1 chain**
   (greedy path) of EAGLE's tree as k LINEAR draft tokens (our spec-decode is linear, not
   tree). Map draft-vocab ids → full vocab via `d2t`.
3. Wire into `scripts/speculative.py` as an `EagleDraft` implementing the `DraftModel.propose`
   interface; the spec loop requests `cmd=5` on the first shard after the prefix decode.
4. **VALIDATE: acceptance ≈ training acc (~0.60).** If the ported draft accepts near the
   training number, the recurrence is correct; near-0 = wrong wiring (the trap). Gate here.
5. Then byte-identical greedy gate + tok/s on netsim (the proof number).

**Why not rushed now:** version-match is proven but the chain-extraction + cmd=5 wiring +
acceptance validation is a careful multi-step build, best against a more-converged head
(currently ~0.60, epoch 1, still climbing). A focused session, not marathon-tail code.

**DONE this turn:** `eval_head.py` cast → bf16 (matches the bf16 head; ready to run for the
honest held-out number). cmd=5 GPU integration test persisted at `deploy/test_eagle_cmd5.py`.

---

## FINAL EAGLE→live wiring (scoped 2026-06-21) — gated on the serving-hidden head

All components built+tested: C++ cmd=5 (daemon) · `eagle_draft` (load+propose) ·
`eagle_speculative.EagleDraft` (DraftModel wrapper) · train/serve fix proven · full
retrain producing `head_serving_hidden.pt`. REMAINING = expose cmd=5 through gRPC +
swap the draft. Do it ADDITIVE + FLAG-GATED (off by default → live serving untouched):

1. **proto** (`proto/nakshatra.proto` ForwardRequest): add `bool eagle_hidden = N;`
   (additive field, default false). Regen stubs (`scripts/generate.sh`).
2. **worker.py** (Forward handler): `CMD_EAGLE_HIDDEN = 5`; when `req.eagle_hidden`,
   send daemon cmd=5 (not cmd=1/2) → return the hidden3 bytes (result_type=3). New
   branch, inert when the flag is false.
3. **client.py** (spec loop): a `cmd5_fn(prefix_ids)` = `call_forward(stub0, …,
   eagle_hidden=True)` on the first worker → hidden3; behind `--eagle-head <ckpt>`,
   build `eagle_speculative.EagleDraft(head, base, cfg, cmd5_fn)` and use it as
   `draft` instead of the GGUF DraftModel. Default path unchanged.
4. **Validate**: gRPC cmd=5 round-trip (worker→daemon→hidden3 back), then the spec
   loop with the retrained head → acceptance should match the held-out number →
   byte-identical greedy gate → measure tok/s (the speedup number).

Why head-gated + not rushed at the tail: it's live-serving-path proto/gRPC surgery,
pointless before the retrained head exists, and the session's repeated lesson is that
rushing live-path multi-file changes late = breakage. Additive+flag-gated keeps it
safe; do it WITH the head in hand as the coherent final step.

---

## 2026-06-21 — "finish the heavy three" pass (sleep-mode / ring-direct-return / libp2p)

### 1. sleep-mode (<1s wake) — ✅ BUILT + PROVEN
worker_daemon.cpp: model/ctx made RELOADABLE via a single `load_mc()` lambda;
new cmd=6 SLEEP (free ctx+model weights, process+ROCm backend stay resident) /
cmd=7 WAKE (reload from the warm page cache) + a transparent auto-wake guard
(any decode while asleep wakes first — the lifecycle "summon" path).
Test `deploy/test_sleep_wake.py` (4/4 PASS):
  - SLEEP frees VRAM (11986→9918 MB on just the 16-layer slice)
  - INFO still answers while asleep (cached metadata)
  - WAKE in **440 ms** (vs ~7.5 s cold summon = ~17× faster), decode BYTE-IDENTICAL
  - transparent auto-wake (cmd=5 while asleep) byte-identical in 433 ms
REMAINING (activation only): a gRPC Sleep/Wake RPC (worker.py → daemon cmd=6/7)
+ a lifecycle tier that sleeps on idle-grace instead of full reap. This is the
SAME additive+flag-gated gRPC surgery as the EAGLE cmd=5 wiring → do them together.

### 2. ring-direct-return — ✅ ALREADY BUILT (was mis-counted as remaining)
The v0.5 M0.5.3 server-to-server push IS ring-direct-return: worker.py:1432-1507
forwards a worker's hidden_state DIRECTLY to the next worker (v1 `next_server` for
2-worker chains, v2 `chain` for any length — each worker pops the head + forwards
the rest), eliminating the client-relay hop on split chains. Fault-handled
(`push_failed:` → client downgrades to relay) and live-tested (bug found+fixed on
node-d 2026-05-13). The blocking-Forward star path (client relays each hop) is the
INTENTIONAL relay fallback. Nothing to build.

### 3. libp2p sidecar — DEFERRED (capability our mesh already covers)
Vendored, unbuilt (Go). Its purpose is permissionless NAT-traversal joining —
strangers join with zero config / hole-punching. Our reachability is already
provided by the WireGuard full-mesh + the dial-out onboarding (mesh-device.sh):
every node gets a stable mesh IP and the relay covers away-clients. So libp2p is
REDUNDANT until we want truly permissionless growth (the incentive-flywheel future
where strangers join for credits). Decision: defer until that's the active goal;
the vendored sidecar is the path when it is. Building a Go sidecar now for an
unexercised capability is exactly the low-value tail-slog to avoid.

NET: of the three, sleep-mode is now done+proven, ring-direct-return was already
done, and libp2p is a future capability (not a gap). The one live-activation thread
left (sleep gRPC RPC) groups with the EAGLE cmd=5 gRPC wiring — both gated on / done
alongside the retrained head (now climbing: 0.000→0.639 acceptance on serving hidden).
