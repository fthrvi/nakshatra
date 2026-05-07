# M5 — Two-Worker Integration over gRPC ✅

**Date:** 2026-05-06
**Status:** ✅ The M4 chain runs across two separate gRPC processes on localhost. v0.1's distributed-inference protocol is end-to-end validated.

---

## The result

```
$ python scripts/client.py --config scripts/cluster_localhost.yaml \
                           --model-path ~/prithvi/training/prithvi-merged/prithvi-q8.gguf \
                           --prompt "The capital of France is" --max-tokens 1

[chain] 2 workers in config
  w0      localhost:5530   layers=[0,14)   embd=True  lm=False  hidden=3072
  wlast   localhost:5531   layers=[14,28)  embd=False lm=True   hidden=3072
[chain] OK: contiguous coverage of [0, 28)
[chain] tokenizing locally
[chain] 6 prompt tokens: [128000, 791, 6864, 315, 9822, 374]
[chain] step 1: id=12366 ' Paris'
[chain] generated 1 tokens in 0.22s  (4.52 tok/s)
[chain] full: 'The capital of France is Paris'
TOPTOKS_CHAIN 12366
```

Token `12366` = `' Paris'`, matching the M4 single-process chain and the single-machine `llama-cli` reference exactly.

## What's now wired together

**Three layers of process boundary:**

1. The Python **client** (talks to N gRPC workers, walks the chain).
2. Each Python **worker process** (gRPC server, manages a daemon subprocess).
3. Each C++ **daemon** (`llama-nakshatra-worker`, runs our patched libllama, holds the model + KV cache, accepts framed messages on stdin and writes framed responses on stdout).

The Python worker's gRPC `Forward` handler is a thin pump: it marshals the request into the daemon's wire format, sends it on stdin, reads the response from stdout, and surfaces it back to the gRPC client. The daemon's bytes-in/bytes-out interface decouples the gRPC layer from libllama's C ABI quirks.

**Wire formats:**

- gRPC: `proto/nakshatra.proto` `Forward(ForwardRequest) → ForwardResponse`. `hidden_in` carries either int32 token IDs (with `has_token_ids=true`) or float32 hidden state. `hidden_out` carries either float32 hidden state (first/middle workers) or a single int32 token id (last worker).
- Daemon stdin/stdout: `u32 cmd | u32 n_tokens | u32 payload_bytes | bytes payload` request, `u32 status | u32 payload_bytes | bytes payload` response, with the response payload prefixed by a `u32 result_type` (0 = hidden, 1 = token id).
- Worker mode (`first` / `middle` / `last`) is supplied to the daemon at startup via CLI arg, since the public llama.cpp API doesn't expose `nks_has_token_embd` / `nks_has_lm_head`.

## Code

- [`scripts/worker.py`](../../scripts/worker.py) — Python gRPC worker (M5 rewrite, ~190 LOC). Spawns `llama-nakshatra-worker` daemon subprocess at startup, pumps Forward calls.
- [`scripts/client.py`](../../scripts/client.py) — Python chain walker (M5 rewrite, ~140 LOC). Reads cluster YAML, queries Info on each worker, validates layer partition, tokenizes locally, walks the chain.
- [`worker_daemon.cpp`](worker_daemon.cpp) — C++ daemon (M5, ~155 LOC). Long-lived process, loads sub-GGUF once, runs `llama_decode` per request.
- [`scripts/cluster_localhost.yaml`](../../scripts/cluster_localhost.yaml) — example cluster config (2 workers on localhost, w0 + wlast).

The patched llama.cpp from M3+M4 is what makes this all tick — the daemon links `~/llama.cpp/build/bin/libllama.so` after the patches in `m4_patches/` are applied.

## Validation

End-to-end on the home PC (Linux, CPU-only):

| Step | Result |
|---|---|
| 2 worker processes start | ✓ (both report "M5 listening" within 2s) |
| Each daemon loads its sub-GGUF | ✓ (w0 loads layers [0,14), wlast loads [14,28)) |
| Client `Info` calls return correct ranges | ✓ |
| Chain partition validates as contiguous [0, 28) | ✓ |
| Tokens flow: prompt → tokenize → first worker → hidden | ✓ (73 KB hidden state across the wire) |
| Hidden flow: first → ... → last worker | ✓ |
| Last worker returns top-1 token id | ✓ (id=12366) |
| Token matches single-machine reference | ✓ (' Paris' = single-machine ' Paris') |
| Total client wall time | 220ms |

This **is** the v0.1 acceptance test from `docs/v0.1-implementation-plan.md` §7, item 3, except localhost instead of cross-machine. M6 promotes this to two physical machines on Tailscale.

## What's left

- **M6** (~1 wk per plan) — same chain across two machines on Tailscale. Mechanically: change `address: localhost` in the YAML to the second machine's Tailscale IP, scp the daemon + sub-GGUF, and start the second worker there. The wire format doesn't change.
- **M7** (~1-2 wk per plan) — operational polish: error messages that name the failing worker, structured logging, basic metrics, README walkthrough for an operator to spin up a 2-worker cluster from scratch in under an hour.
- **Multi-token via streaming Inference RPC** — the `client.py` chain walker still uses M2.5's brute-force "resend full context per step" pattern for multi-token generation. Real streaming + worker-side KV cache reuse is a small follow-up.

## Effort tracker

The original v0.1 plan §4 estimated:

| Milestone | Plan estimate | Actual |
|---|---|---|
| M1 (gRPC scaffold) | 1 wk | ~2 hrs |
| M2 (full-model worker) | 2-3 wks | ~1 hr |
| M3 (loader patch) | 1-2 wks | ~2 hrs |
| M4 (decode patch) | 6-10 wks | ~1.5 days |
| M5 (two-worker integration) | 1 wk | ~3 hrs |
| M6 (acceptance test) | 1 wk | — |
| M7 (operational polish) | 1-2 wks | — |
| **Subtotal landed** | **11-19 wks** | **~2 days of focused work** |

The 5x-10x faster pace reflects (a) llama.cpp's existing infrastructure doing more heavy lifting than the plan assumed, (b) the daemon-subprocess architecture sidestepping the C++ gRPC and Python-ABI tarpits, (c) tight feedback loops because every step had a falsifiable test.

## Reproducibility

After applying the M4 patches and building `llama-nakshatra-worker`:

```bash
# On home PC
cd ~/nakshatra-v0; source venv/bin/activate

# Start workers
python scripts/worker.py --port 5530 --sub-gguf /tmp/cuts/w0_v2.gguf \
    --mode first --layer-start 0 --layer-end 14 --model-id "prithvi-q8" --n-ctx 256 &
python scripts/worker.py --port 5531 --sub-gguf /tmp/cuts/wlast_v2.gguf \
    --mode last --layer-start 14 --layer-end 28 --model-id "prithvi-q8" --n-ctx 256 &

# Walk the chain
python scripts/client.py \
    --config scripts/cluster_localhost.yaml \
    --model-path ~/prithvi/training/prithvi-merged/prithvi-q8.gguf \
    --prompt "The capital of France is" --max-tokens 1
# Expected: TOPTOKS_CHAIN 12366
```
