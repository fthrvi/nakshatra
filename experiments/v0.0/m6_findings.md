# M6 — Cross-Machine Acceptance Test ✅

**Date:** 2026-05-06
**Status:** ✅ **The v0.1 acceptance test passes across two physical machines on Tailscale.** Distributed inference is functionally live.

---

## The result

```
$ python scripts/client.py --config scripts/cluster_crossmachine.yaml \
                           --model-path ~/prithvi/training/prithvi-merged/prithvi-q8.gguf \
                           --prompt "The capital of France is" --max-tokens 1

[chain] 2 workers in config
  home-pc   100.101.134.33:5530   layers=[0,14)   embd=True  lm=False  hidden=3072
  bishwa    100.109.164.69:5531   layers=[14,28)  embd=False lm=True   hidden=3072
[chain] OK: contiguous coverage of [0, 28)
[chain] step 1: id=12366 ' Paris'
[chain] generated 1 tokens in 0.51s  (1.96 tok/s)
TOPTOKS_CHAIN 12366
```

**Token 12366 (' Paris')** — matches the M5 localhost run, the M4 single-process run, and the single-machine `llama-cli` reference. All four runs converge.

## What this proves

The v0.1 plan (`docs/v0.1-implementation-plan.md` §7) defines acceptance items 1–4:

> 1. Pre-split GGUF tool produces sub-GGUFs that the patched loader accepts and the unpatched loader rejects.
> 2. Patched `llama_decode_layers` on a complete-model layer range reproduces `llama_decode`'s top-1 token.
> 3. Two-worker cluster ... produces the same top-1 next token as a single-machine `llama-cli` running the full model.
> 4. The single-machine reference is run with greedy decoding.

All four are satisfied. The architecture works end-to-end with:

- **Two physical machines on Tailscale** — home PC (Linux, AMD RX 9070 XT, ROCm available, ran CPU-only) at `100.101.134.33` + lab Mac `bishwa` (Intel Mac, Radeon Pro 5700 XT, Vulkan/MoltenVK available, ran CPU-only) at `100.109.164.69`.
- **Cross-vendor** at the OS layer (Linux + macOS) and at the CPU vendor layer (AMD + Intel) — the v0.1 plan's resolved-decision §9 marked cross-vendor as a v0.1 risk; outcome: works fine in CPU-only mode.
- **Patched `llama.cpp`** built independently on each machine from the M3+M4 patches in `experiments/v0.0/m4_patches/`. Source commits differ between the two boxes (`c46583b` on home PC, `8c2c0108d` on bishwa); patches applied cleanly to both.
- **Hidden state ferried over Tailscale** — 6 prompt tokens × 3072 floats × 4 bytes = 72 KB across one network hop.

Wall time: **510ms total** for one token. About 220ms is the actual compute on each worker; 290ms is Tailscale round-trip for the 72 KB hidden state. Both contributions are CPU-bound at this scale; GPU offload (when we wire it back up post-v0.1) would shrink the compute side dramatically.

## What changed for M6 vs M5

Almost nothing in code:

- New cluster YAML [`scripts/cluster_crossmachine.yaml`](../../scripts/cluster_crossmachine.yaml) — same shape as `cluster_localhost.yaml` with Tailscale IPs.
- The patches and the `llama-nakshatra-worker` binary were rebuilt on `bishwa` after applying the M4 patches there. The build picked up cleanly via macOS's `/Library/Developer/CommandLineTools` toolchain plus `/usr/local/bin/cmake`. Built CPU-only via `-DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF` to avoid kernel-divergence with the Linux worker.
- The 1.8 GB `wlast_v2.gguf` was scp'd over Tailscale from home PC to bishwa: 9 minutes wall time (~3.4 MB/s, gated by Tailscale's encrypted overhead on consumer hardware).

The wire format is unchanged from M5. The Python worker and the chain-walking client are unchanged. Cross-machine "just works" once the binary + sub-GGUF are present on the second host.

## Effort tracker

| Milestone | Plan estimate | Actual |
|---|---|---|
| M1 (gRPC scaffold) | 1 wk | ~2 hrs |
| M2 (full-model worker) | 2-3 wks | ~1 hr |
| M3 (loader patch) | 1-2 wks | ~2 hrs |
| M4 (decode patch) | 6-10 wks | ~1.5 days |
| M5 (two-worker integration) | 1 wk | ~3 hrs |
| **M6 (cross-machine acceptance)** | 1 wk | ~30 min real work + 9 min waiting for SCP |
| M7 (operational polish) | 1-2 wks | — |
| **Subtotal landed** | **12-20 wks** | **~2 days of focused work** |

The plan budgeted 12-20 weeks for the work that landed in ~2 days of focused effort. Reasons:

1. llama.cpp's existing infrastructure already supports vector embeddings as input (PR #18550) and embeddings output via `llama_get_embeddings`. M4 step 5 (the new C API entry point) wasn't needed.
2. The daemon-subprocess architecture sidesteps the C++ gRPC and Python-vs-libllama ABI tarpits that would otherwise be slow.
3. Tailscale handles all the cross-machine network plumbing; gRPC just speaks HTTP/2 over TCP on top.
4. Tight feedback loops — every milestone had a falsifiable test (Phase 0a → Phase 0b → M3 segfault test → M4 chain test → M5 localhost chain → M6 cross-machine chain), each catching one class of bug before the next layer was added.

## What's left for v0.1 ship

- Item 5 of §7's ship gate: "The cluster runs continuously for at least 10 single-token generations without crashing or producing NaN logits." → Multi-token test via the existing M2.5 brute-force loop. Trivial to verify; just bump `--max-tokens 10`.
- Item 6: "An external operator can stand up a 2-worker cluster from scratch following only the README + the cluster config schema, in under 1 hour." → That's M7's deliverable — a written runbook + structured logging.

## Reproducibility

```bash
# On HOST 1 (Linux, the model is available):
python experiments/v0.0/partial_gguf.py \
    /home/prithvi/prithvi/training/prithvi-merged/prithvi-q8.gguf \
    /tmp/cuts/w0_v2.gguf --start 0 --end 14
python experiments/v0.0/partial_gguf.py \
    /home/prithvi/prithvi/training/prithvi-merged/prithvi-q8.gguf \
    /tmp/cuts/wlast_v2.gguf --start 14 --end 28 --keep-token-embd

# Apply M4 patches and build llama-nakshatra-worker on each machine.
# Then on HOST 2:
scp /tmp/cuts/wlast_v2.gguf user@host2:/tmp/wlast_v2.gguf

# Edit scripts/cluster_crossmachine.yaml with the actual Tailscale IPs.

# Start workers (one per machine):
# Host 1:
python scripts/worker.py --port 5530 --sub-gguf /tmp/cuts/w0_v2.gguf \
    --mode first --layer-start 0 --layer-end 14 --model-id "prithvi-q8" --n-ctx 256 &
# Host 2:
python scripts/worker.py --port 5531 --sub-gguf /tmp/wlast_v2.gguf \
    --mode last --layer-start 14 --layer-end 28 --model-id "prithvi-q8" --n-ctx 256 \
    --daemon-bin /Users/.../llama.cpp/build/bin/llama-nakshatra-worker &

# Run the chain client (from anywhere with access to both Tailscale IPs):
python scripts/client.py --config scripts/cluster_crossmachine.yaml \
    --model-path /path/to/full/prithvi-q8.gguf \
    --prompt "The capital of France is" --max-tokens 1
# Expected: TOPTOKS_CHAIN 12366
```
