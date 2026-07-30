# Finding: rebalancing the Q3 cross-vendor layer split (hub GPU / ijru CPU)

**Landed:** 2026-07-29, branch `inference/rebalance-q3-split`. Follows directly from the
async-pipelining GPU-mesh proof the same night, whose verdict pointed at this as the real
LAN lever: "hub does 13 layers in 10ms, ijru 35 in 123ms — VRAM-forced imbalance."

## What it is

The Q3 cross-vendor chain (Qwen3-30B-A3B Q3_K_M, 48 blocks) splits the model across two
workers: hub (AMD RX 9070 XT, ROCm) and ijru (10.0.0.233). The v1 split
(`~/.nakshatra/qwen3-30b-q3-chain.yaml`, L0-13 hub / L13-48 ijru) was **not chosen for
balance** — it was the largest hub slice that fit next to Prithvi's conscious-reserved
VRAM back on 2026-06-30. That left hub (fast, GPU) doing only 13 of 48 layers while ijru
(slow — see correction below) carried 35.

This run re-measured free VRAM on hub tonight, re-sliced the full GGUF at a larger cut,
and re-benched with the identical client command shape as tonight's baseline
(`receipt-seq.json`, 4.58 tok/s).

## Correction to the starting assumption: ijru is CPU, not CUDA, for this workload

Before slicing anything, I checked `ijru:~/llama.cpp/build/CMakeCache.txt` after the first
load attempt looked suspicious (`nvidia-smi` showed 429MiB used — baseline driver overhead
— both before and after the ijru worker reported "ready"). `GGML_CUDA:BOOL=OFF` in the
current build; `ldd` shows no `libcuda` dependency; the daemon log has zero CUDA/device
lines, only `CPU_Mapped`/`CPU_REPACK` buffers. I checked the **actual leftover process from
tonight's real baseline run** (`ijru:~/.nakshatra/ijru-q3-5571.log`, the log behind
`receipt-seq.json`) and it shows the identical pattern — CPU_Mapped/CPU_REPACK only, no
CUDA. So tonight's 4.58 tok/s baseline *also* ran ijru on CPU. "ijru CUDA 3060" in prior
notes describes the node's hardware identity, not the backend actually driving this chain;
the CPU_REPACK-optimized kernels are fast enough for a 3B-active-param MoE (~123-140ms/step
for 33-35 layers) that this wasn't obviously wrong from tok/s alone. This means the
comparison below is apples-to-apples (both baseline and rebalance run ijru on CPU) — no
confound — but it also **reframes the whole imbalance**: hub is fast because it's GPU,
ijru is slow because it's CPU, and the fix is "move layers off the CPU box," which is
exactly what this change does, just via VRAM headroom rather than a CPU/GPU split as such.

## VRAM sizing: two iterations, one real lesson

Read the exact tensor sizes out of the full GGUF (`gguf.GGUFReader`) rather than assuming
uniform layers: `token_embd.weight` = 127.5MiB, `output.weight+output_norm.weight` =
255.3MiB, per-block ≈ 296.7-322.0MiB (first 3 blocks slightly larger). Measured free VRAM
on hub's 9070 XT (GPU[1] in `rocm-smi`, matches `HIP_VISIBLE_DEVICES=1`) with only
`qwen3:30b-a3b` resident (`ollama ps`, `Forever` keep-alive — the only steady-state tenant):
**6144.9 MiB free** of 16304 MiB total.

- **Naive estimate** (weights + KV cache at n_ctx=4096, 8MiB/layer for 4 KV heads ×
  256 head-dims × fp16) said **L=17** fits with ~1000MiB to spare. Tried it first
  (matches the task's expected L≈17-20). **It OOM'd**: `cudaMalloc failed: out of memory`
  allocating the final 4882MiB tensor buffer, daemon exit rc=2. The naive estimate missed
  the compute/graph scratch buffer llama.cpp allocates alongside weights+KV — observed
  delta was **~470-476MiB more** than weights+KV alone, both at the failed L=17 attempt
  (indirectly, from the failure margin) and confirmed directly at L=15 (predicted
  4564.0MiB, actual load delta 5040.0MiB → overhead 476MiB).
- Stepped down to **L=15**: naive weights+KV = 4564.0MiB, + ~476MiB real overhead ≈
  5040MiB, leaving **1105MiB actual headroom** after load (measured via `rocm-smi`
  before/after) — comfortably clears the ≥800MiB floor. This is what got benched.
- A live ollama tenant (`qwen2.5:7b-instruct`, 7.5GB, short keep-alive) loaded transiently
  on the *same* GPU mid-attempt and forced a retry once the window cleared — confirms the
  "measure right before you load, in a clear window" approach was necessary, not
  theoretical. **No ollama tenant was evicted or disturbed** at any point — `ollama ps`
  before/during/after shows `qwen3:30b-a3b` (Forever) and later `prithvi:latest`
  (transient) untouched throughout.

**Delivered split: hub L0-15 / ijru L15-48** (v1 was L0-13/L13-48). Two more layers moved
off ijru's CPU onto hub's GPU than v1, one fewer than the naive-estimate target of L=17,
because the naive estimate didn't fit reality.

## Files

- Slicer used (found, not written): `experiments/v0.0/partial_gguf.py` — `--start S --end E`
  sub-GGUF cutter, writes `nakshatra.layer_range_{start,end}` KV metadata that the patched
  loader reads for partial load. Same tool that made the v1 slices.
- New slices (not committed — binary, `~/.nakshatra/slices/`):
  - hub: `qwen3-30b-q3-L0-15.gguf` (4,665,769,024 bytes)
  - ijru: `qwen3-30b-q3-L15-48.gguf` (10,052,012,576 bytes)
  - sha256 verified identical both sides after a throttled transfer (`rsync --bwlimit=35M`,
    ~35MB/s sustained, 4m20s for the ijru slice — house rule for hub bulk writes/reads):
    `1094bb6183ae9d6f182b87db49dac802c99f9da7360b0ff3c06f5729dccca330`
  - An intermediate L0-17/L17-48 attempt (OOM'd on hub) was sliced, transferred, and then
    **deleted from both boxes** after the OOM — not left behind as dead weight.
- New chain config: `~/.nakshatra/qwen3-30b-q3-chain-v2.yaml` — same format as v1
  (`~/.nakshatra/qwen3-30b-q3-chain.yaml`), new ports (hub 127.0.0.1:5562, ijru
  10.0.0.233:5572) so v1 stays intact and untouched.
- Receipt: `~/.nakshatra/receipt-rebalance.json` (compare against `~/.nakshatra/receipt-seq.json`,
  tonight's baseline).

## Bench: same command shape as the baseline

Baseline command shape (from `receipt-seq.json` + `hub-q3-5561.log`, reconstructed —
tokenizer blob, draft blob, `--speculative --draft-max 4 --max-tokens 96 --tls-mode off`).
One caveat: **the literal baseline prompt text was not recoverable** — only its
`prompt_sha256` was persisted in the receipt, the raw string was never written to a file I
could find. I decoded the baseline's `generated_tokens` against the GGUF's own vocab and
brute-force-hashed ~18 candidate phrasings on the same topic against the target sha256;
none matched. I used an equivalent prompt on the same topic, same token count (19, matching
`n_prompt` in both receipts) — same shape, not byte-identical text. Documented here rather
than silently treated as identical.

```
Prompt: "Explain why the sky appears blue at day, red at sunset, and dark at night."
--speculative --draft-model-path <Qwen3-0.6B ollama blob> --draft-max 4 --max-tokens 96 --tls-mode off
```

| | baseline (v1, L0-13/L13-48) | rebalanced (v2, L0-15/L15-48) |
|---|---|---|
| tok/s | **4.579** | **4.944** (+8.0%) |
| elapsed | 20.97s / 96 tok | 19.42s / 96 tok |
| n_prompt / n_generated | 19 / 96 | 19 / 96 |
| hub mean_rpc_ms (incl. prefill) | 20.01ms | 17.32ms |
| ijru mean_rpc_ms (incl. prefill) | 130.51ms | 140.88ms* |
| hub steady-state step avg | — (not persisted) | 10ms (41 calls) |
| ijru steady-state step avg | — (not persisted) | 127ms (41 calls) |

\* The receipt's `mean_rpc_ms` includes the one-time prefill call, which is heavier and not
layer-count-proportional (ijru prefill=694ms this run vs steady-state 127ms/step) — so it's
noisy as a per-layer proxy between two runs with different prompt text and different
speculative accept/reject sequences (different token stream → different number of RPC
calls → different prefill weighting in the mean). The clean, load-bearing number is **tok/s**,
measured directly by the client's own step timer over the full 96-token run, not derived
from the per-call means.

## Verdict

Real, positive, honestly-measured gain: **4.58 → 4.94 tok/s (+8.0%)** from moving 2 layers
off ijru's CPU path onto hub's GPU path. Smaller than the task's naive target range (L≈17-20,
which would have moved 4-7 layers) because the actual achievable L was VRAM-headroom-limited
to 15 once the real compute-buffer overhead (~476MiB, not captured by a weights+KV-only
estimate) was accounted for empirically. The lever is real and the direction is right; the
magnitude is capped by how little free VRAM hub's 9070 XT has left once `qwen3:30b-a3b`
(Forever-resident, ~10.15GB) is honored — pushing further would mean either giving the
unconscious pool a firmer contractual VRAM budget (so a transient ollama tenant like
`qwen2.5:7b-instruct` can't intermittently steal the loading window), or moving the *hub*
side of this chain onto a card that isn't also hosting Prithvi's conscious tenant.

## Cleanup confirmed

- Both bench workers killed (hub PID 2810818/2810825; ijru PID 387618/387619) — verified
  with `ps aux` no match on either box.
- hub `rocm-smi --showmeminfo vram` GPU[1] back to baseline used (~10.65-10.67GB, matches
  pre-bench reading within normal drift); `ollama ps` shows `qwen3:30b-a3b` (Forever) and
  later `prithvi:latest` (transient) both present and untouched throughout — never evicted.
- ijru `free -h` shows RAM released (20GiB buff/cache, 24GiB available) after kill.
- Superseded L0-17/L17-48 slice files removed from both `hub:~/.nakshatra/slices/` and
  `ijru:~/.nakshatra/slices/`.

## Out of scope, flagged not fixed

**A stray worker process was already running on ijru before this session touched
anything**: `python3 scripts/worker.py --port 5571 ... qwen3-30b-q3-L13-48.gguf` (PID 386242,
started 19:38, i.e. tonight's async-pipelining baseline run) — never cleaned up by whichever
session ran it, holding ~13.4GB of system RAM (no VRAM — also CPU-mode). It is **not** one
of "both workers" this task was scoped to kill (it predates this session and isn't part of
the v2 chain), so I left it running rather than guess at another session's intent. Flagging
here for whoever owns that run to reap it (`kill 386242` on ijru, or check if `worker.py`'s
`--idle-grace-s` should have reaped it and didn't).
