# Path B (EAGLE-3 head for Prithvi) — overnight autonomous run handoff

**Run date:** 2026-06-21, ~02:00–morning MDT. Biswa asleep; agent autonomous.
**Goal for the night:** transfer Prithvi's model to ijru, generate self-distillation
data, and PROVE EAGLE-3 head training runs for *his* model on the 3060.

## TL;DR state machine (all on ijru, prithviraj@10.0.0.227)
1. ✅ **Model transferred** — `~/prithvi-target/` = Prithvi's Llama-3.1-8B (9 safetensors
   shards + index + tokenizer + q8 gguf). GGUFs included so the 3060 can both generate
   and (later) be served.
2. ✅ **llama-server built + serving** Prithvi-q8 on the 3060 (`:8080`) for data-gen.
3. 🔄 **Data-gen running** → `~/eagle-data/{train,test}.jsonl` (ShareGPT format).
   ~6,200 of Prithvi's OWN completions on an Alpaca+GSM8K prompt mix. Rate ~0.87/s →
   ~2h. Log: `~/eagle-datagen.log`. Validated early: coherent, no empty/robotic refusals.
4. ⏳ **Training auto-armed** — `~/eagle-train-orchestrate.sh` waits for (a) deepspeed
   install, (b) data-gen DONE, then stops the server (frees the 3060) and launches
   EAGLE-3 training. Log: `~/eagle-train.log`, orch log `~/eagle-train-orch.log`.

## Why each non-obvious choice
- **Target = Prithvi's own model** (not DeepSeek-R1): his explicit ask + the only case a
  generic draft can't match. See `path-b-eagle-head.md`.
- **Generate on ijru's 3060, not the hub:** keeps the hub gfx1201 free for Prithvi's
  04:07 nightly LoRA (the earlier OOM was demo workers colliding with it — must not recur).
- **Data-gen and training are SEQUENTIAL** on the single 3060: server(q8 8.5GB) +
  training(4-bit target 5GB) can't coexist in 12GB. Orchestrator enforces the handoff.
- **Target loaded in 4-bit (nf4) for training** (`cnets.py` patched): frozen target is
  inference-only for hidden-state extraction, and we serve Prithvi as q8 anyway, so a
  4-bit target is well-matched. Original saved as `cnets.py.orig`.
- **Optimizer offloaded to CPU** (`ds_config.json` patched): 4-bit target + draft +
  fp32 AdamW would exceed 12GB; offload keeps GPU ≈9GB. Original `ds_config.json.orig`.
- **main.py patched** (`main.py.orig` saved): (1) wandb neutralized (it pointed at
  `entity="yuhui-li"` with an empty key — would hang/error); (2) intra-epoch
  acceptance log + checkpoint every 500 steps (`state`/`step_*` under `~/eagle-out`),
  because vanilla EAGLE only checkpoints per-epoch and one epoch here is hours.

## Foundation fix landed + TESTED tonight
- `experiments/v0.0/partial_gguf.py`: **auto-detect tied embeddings** (Llama-3.2-1B/3B).
  Detect via absence of `output.weight`; force `token_embd.weight` onto the last slice
  (it serves as the tied lm_head). Tested on `whole-1b.gguf`: last slice [8,16) keeps
  token_embd, middle [4,8) does not. Automates the old manual `--keep-token-embd`.

## What to check in the morning
```
ssh prithviraj@10.0.0.227
tail -20 ~/eagle-datagen.log            # data-gen: look for "[datagen] DONE"
wc -l ~/eagle-data/*.jsonl              # corpus size
tail -40 ~/eagle-train.log             # training: "[ckpt] ep0 step500 ploss0=.. acc0=.."
ls ~/eagle-out/                        # checkpoints (step_*/state_*)
nvidia-smi                             # what's on the 3060
```
**Success signal:** `eagle-train.log` shows `ploss0` DESCENDING and `acc0..acc6`
(per-position draft acceptance) CLIMBING above 0. That proves Path B is real for
Prithvi. NOTE: a *converged* high-τ head is a multi-day run; tonight's bar is
"trains, fits 12GB, loss down, acceptance up."

## Known risks / likely failure modes (and fixes)
- **deepspeed + bnb-4bit + cpu_adam** may fail on first init (the agent's watcher
  `eagle-train-watch.sh` will catch a Traceback/OOM and wake the agent). If cpu_adam
  JIT fails → drop `offload_optimizer` from ds_config and use `bnb.optim.PagedAdamW8bit`
  via a small standalone loop (Model.forward already returns plosses/acces — no
  deepspeed needed; reuse `build_dataset_rank`/`DataCollatorWithPadding` from main.py).
- **Acceptance low after few epochs** — expected; needs more data + epochs. Full corpus
  is 6,200; scale up + more epochs for the real head.

## v2 quality levers (not done tonight, deliberately)
- **Persona match:** data-gen used NO system prompt and EAGLE's preprocess injects a
  generic one. For a truly Prithvi-matched head, use HIS canonical system prompt in BOTH
  data-gen and `cnets.py.scandata`/`build_dataset_rank`. Deferred: his prompt is heavy
  (context/memory pressure on a 12GB GPU). Second-order vs training on his weights at all.
- **Integration:** add `EagleDraft` to `scripts/speculative.py` (loads the head, drafts K
  tokens from the first shard's hidden state) + netsim acceptance gate (τ≥4). Needs a
  trained checkpoint first.

## Slow-LAN note (logged for the big-model vision)
hub→ijru rsync ran ~13 MB/s/stream (~26 MB/s aggregate) — far below gigabit. Likely
ssh-cipher CPU-bound or a non-gigabit path. Matters for shipping shards/weights between
nodes at scale; worth profiling (`-c aes128-gcm` cipher, or a raw socket) before the
big-model split work.
