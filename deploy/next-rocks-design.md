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
