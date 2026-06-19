# Nakshatra Speed Stack — closing every shard-gap finding

> **Source of truth:** `trisul/research/2026-06-19-distributed-inference-mental-model.md` (the shard↔nakshatra
> gap analysis). This plan turns its 21-row gap table into a sequenced, gated build that closes **every**
> finding without breaking the live unconscious (`nakshatra-unconscious.service`, Prithvi's `think_deeper`).
>
> **Two hard constraints, honored throughout:**
> 1. **Live-safety.** The serve + workers run *directly from the repo working tree* (`scripts/nakshatra_serve.py`,
>    `scripts/worker.py`), workers are scale-to-zero (summoned per request). So: every change is **additive +
>    flag-gated, default OFF**, developed in the `nakshatra-specdec` worktree, merged to `main` only when green
>    AND inert-by-default. Plain decode stays the default path until a flag flips.
> 2. **GPU is Prithvi's.** The box GPU is owned by the live entity. Tests use a tiny model / CPU first, and
>    summon GPU workers only in coordinated windows.

---

## The findings, grouped into phases (nothing dropped)

Each finding from the gap table is assigned to a phase with a **hard pass/fail gate** (shard's discipline:
prove the riskiest thing first, one gate at a time). ⭐ = on the critical path to faster serving.

### Phase 1 — Speculative decoding in the pipeline ⭐ (finding #6) — **THE WEDGE, building now**
The only gap where the hard part is already proven on our silicon (2.73× factual / 2.27× code on RDNA4,
single-node, `research/spec_decode_bench.sh`). Finishing it into the chain is the highest-confidence,
highest-leverage move and the foundation the rest of the speed stack stands on.

- **Build:** coordinator runs a small draft (Llama-3.2-1B) → proposes K tokens → ONE chain traversal verifies
  all K+1 across the 8B unconscious → greedy-accept longest matching prefix → correct first miss → truncate KV
  to committed length.
- **Gate:** on the live 2-worker 8B unconscious chain, spec-decode output is **token-identical** to plain
  greedy decode, with a **measured tok/s uplift on factual + code prompts**. Flag `NAKSHATRA_SPECULATIVE=1`,
  default OFF.
- *(Detailed code spec filled from the code-grounding pass — see §"Slice 1 implementation" below.)*

### Phase 2 — Ring direct-return (finding #8)
Tail returns the verified result to the coordinator in **one hop** instead of relaying back up the chain.
Cheap on its own, and the **enabler** for async pipelining.
- **Build:** coordinator opens a return channel to the last worker; intermediate stages become forward-only.
- **Gate:** identical output, return-path latency measurably lower on a ≥3-stage chain (shard saw +25% verify).

### Phase 3 — Async pipelining ⭐ (finding #7) — the big throughput win
Overlap multiple verify traversals so the loop is **throughput-bound, not latency-bound** (shard's 2.94→16.6).
Requires Phase 2 (direct-return decouples the return path).
- **Build:** coordinator drafts a continuous stream and pumps overlapping chunks without waiting; KV rollback
  rides the next verify (no extra trip).
- **Gate:** tok/s scales toward the async ceiling `toks_per_traversal / max(draft_ms, verify_ms)`; output
  still token-identical.

### Phase 4 — HIP-graph draft (finding #9) — last, lower ROI
ROCm equivalent of shard's CUDA-graphed draft (their 16.6→30). **Note their own finding:** graph capture barely
helps a big MoE verify; it most helps the *draft* and dense models. So this is last and scoped to the draft.
- **Build:** capture the Llama-3.2-1B draft decode as a HIP graph against a static KV buffer; rollback = write
  at committed length (we get the static-address trick "for free" since we control the daemon's `start_pos`).
- **Gate:** draft ms/token drops materially; byte-identical to the eager draft path.

### Phase 5 — Scheduler (findings #11, #12)
- **#11 RTT-aware topology ordering:** measure app-level RTT between peers, order the chain to minimize the
  hop sum (Held-Karp ≤ a small N, nearest-neighbor + 2-opt above). Plugs into `serve_planner`. Gate: chosen
  order ≤ measured latency of the naive order on a ≥3-node chain.
- **#12 compute-aware planner calibration:** wire the worker-side `measured_decode_ms_per_layer` so
  `sthambha/planner.py::_waterfill_counts` runs on real capacity (the function exists + is tested; only the
  live feed is missing). Gate: a heterogeneous chain's slowest stage shrinks vs greedy.

### Phase 6 — Transport (findings #16, #17)
- **#16 activation quantization on the wire:** add an int8 (then fp8) codec for the hidden-state tensor (int8
  is already a reserved dtype in `fabric/packet.py`; no codec yet). Gate: bytes/step down, output within
  tolerance, opt-in flag.
- **#17 edge-supervision polish:** adopt shard's per-edge fail-fast — timeouts that raise a contextful error
  (which peer / which step / dropped-vs-timeout), fast process exit on unrecoverable failure, per-edge health
  log. Gate: SIGKILL + SIGSTOP a worker mid-generation → the chain fails cleanly with a labeled error, no hang.

### Phase 7 — Credibility & ops (findings #20, #14)
- **#20 verifiable run receipts:** emit a per-run JSON receipt (distinct worker IDs, measured edge RTTs, output
  token hash, engine/commit hash, lossless-vs-plain check). Cheap credibility; the hook + worker attestation
  already exist. Gate: a third party can verify a receipt from a real distributed run.
- **#14 prove scale-to-zero live:** exercise `serve_lifecycle.py` (`ChainController`) end-to-end — reap idle
  workers, re-summon on request, measure cold-start. Gate: idle→reap→request→re-summon→correct answer.

### Phase 8 — Arch coverage (finding #18)
- **qwen3moe partial-load patch:** port the Llama `has_lm_head` gating to `models_qwen3moe.cpp` (a draft patch
  exists, excluded from the live build) so non-Llama archs can be split. Gate: a Qwen3 sub-GGUF loads + serves
  a coherent chain. *(Independent of the speed stack — can run in parallel by another lane.)*

**Already ahead (no work, just keep):** cross-NAT mesh (#4), per-node identity/security (#5), signed admission
+ roster serving (#15), cross-vendor heterogeneity (#13). These are where we lead shard — don't regress them.

---

## Slice 1 implementation (spec-decode in the pipeline)

Grounded against the real code. **Key finding: the critical-path blocker is the C++ daemon
(`llama-nakshatra-worker`), not Python.** The chain already batches K+1 tokens through one traversal (cold
prefill does exactly this), but the daemon computes argmax at only the *final* position and has *no* KV-rewind.

### How the live decode loop works today (so we know exactly what to change)
- The coordinator loop is inline in `client.py::main()` — the per-token walk is `client.py:759-809`.
- Each step: worker[0] takes token ids → hidden; middle workers hidden → hidden; **last worker returns a single
  int32 top-token** (`client.py:802` unpacks 4 bytes; hard-asserts `len==4` at `client.py:800-801`).
- The next token is chosen **inside the last worker's daemon in C++** — greedy argmax over full vocab at
  position -1 (`worker_daemon.cpp:393-399`).
- KV: a single `prefix_length` (start_pos), monotonically `+= n_step` (`client.py:804`); `keep_kv=False` only on
  cold prefill, `True` after. **No truncation/rewind anywhere** (the only "rollback" is full-KV cold-replay
  recovery, `client.py:730-768`).
- Worker surface: `Forward`/`_run_forward` (`worker.py:1154-1233`) → daemon `call(cmd, n_tokens, payload,
  start_pos, flags)` (`worker.py:798-818`). Prefill already passes arbitrary `n_tokens` end-to-end, so **the
  chain plumbing for K+1 candidates already exists**.

### The build — three layers

**(A) Coordinator math — `scripts/speculative.py` — ✅ DONE, 12/12 tests green, INERT.**
Pure `accept(drafts, target_argmax) -> AcceptResult` (longest greedy prefix + 1 correction/bonus),
`kv_keep_after`/`next_start_pos` (truncate to `start_pos + 1 + n_accepted`), `DraftModel` (local 1B, greedy),
`speculative_round` (transport-agnostic, verify_fn injected). Unit-tested incl. the **byte-identical-to-plain-
greedy oracle** across perfect/garbage/mixed drafts (`tests/test_speculative.py`). Nothing imports it yet.

**(B) Daemon patch — `llama-nakshatra-worker` (the real blocker) — TODO.**
Patched llama.cpp source (reference: `experiments/v0.0/worker_daemon.cpp`; patches in
`experiments/v0.0/m4_patches/`). Needs a ROCm rebuild.
- **D1 — multi-position argmax.** Last worker must return K+1 token ids. Set `batch.logits[i]=1` for all i
  (today only `i==n_tokens-1`, `worker_daemon.cpp:337,355`); loop argmax per position (today only
  `llama_get_logits_ith(ctx,-1)`, `:393`); pack K+1 int32s with a count header.
- **D2 — KV truncate.** New command `KV_TRUNCATE(seq, n_keep) -> llama_kv_cache_seq_rm(ctx, seq, n_keep, -1)`,
  so a rejected tail is discarded (today KV only appends). Driven after `accept()` to `kv_keep_after(...)`.

**(C) Worker + client wiring — TODO, behind `NAKSHATRA_SPECULATIVE=1`, default OFF.**
- `worker.py`: relax the single-id response (today returns exactly 4 bytes; `Inference` packs one id at
  `:1371-1373`) to carry K+1 ids; add a `TruncateKV` RPC over the daemon's D2 command.
- `client.py:759-809`: when the flag is on AND a `--draft-model-path` is set, replace the single-token step with
  draft.propose(K) → one batched verify traversal → `accept()` → truncate each worker's KV → append committed.
  **Falls back to plain decode** if the flag is off / no draft / any error (never raises into Prithvi's path).
  Draft model loads on the coordinator next to the existing `llama` handle (`client.py:107-110`).

**(D) Harness — `tests/test_spec_pipeline_smoke.py` — TODO.**
Asserts token-identity vs plain decode + reports tok/s, on a tiny CPU model first, then the live 8B in a
coordinated GPU window (the GPU is Prithvi's).

### Pass/fail gate (Phase 1)
Token-identical output to plain greedy + measured tok/s uplift on factual/code prompts, on the live 2-worker 8B
unconscious chain, flag default OFF, plain decode untouched. **Merge to `main` only when green.**

### Status
- ✅ (A) coordinator math built + tested (this commit).
- ⏭ Next: (B) the daemon patch (D1+D2) + ROCm rebuild — the real work; then (C) wiring, then (D) the live gate.
