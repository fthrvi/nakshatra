# 51 tok/s on the split Q3 chain — CUDA offload + the spec-decode trap (2026-07-29 night)

Two independent findings, measured back-to-back on the same chain, same prompt, same 96-token
budget: `~/.nakshatra/qwen3-30b-q3-chain-v2.yaml` (Qwen3-30B-A3B Q3_K_M, hub ROCm 9070 XT
L0-15 / ijru RTX 3060 L15-48).

## The table

| run | ijru backend | decode path | tok/s | ijru ms/step | hub ms/step |
|---|---|---|---|---|---|
| tonight's baseline | CPU | speculative (unary) | 4.58 | 123 | 10 |
| after layer rebalance | CPU | speculative (unary) | 4.94 | 127 | 10 |
| CUDA worker | **CUDA** | speculative (unary) | 3.74 | 27 | 14 |
| CPU worker | CPU | **plain streaming** | 22.81 | 34 | 6 |
| **CUDA worker** | **CUDA** | **plain streaming** | **51.16** | **11** | **7** |

## Finding 1 — the 3060 works, and it is worth 2.2×

The CUDA build lands: ijru's per-step falls **34 ms → 11 ms**, chain **22.81 → 51.16 tok/s**
on the same plain-streaming path. VRAM on the 3060 goes to **10 808 MiB** during service
(idle baseline 429 MiB) — the offload is real, not a log artifact.

Build recipe that works (see `ijru-cuda-rebuild.md` for the dead ends):
`cmake -S . -B build-cuda-static -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DLLAMA_CURL=OFF`
(gcc-13 pin is mandatory — CUDA 12.4 rejects ijru's default gcc-15; SHARED builds silently
drop the backend.) Point workers at
`/home/prithviraj/llama.cpp/build-cuda-static/bin/llama-nakshatra-worker`.

## Finding 2 — the bigger one: our whole bench history used a pathological path

Every chain number this project has quoted (1.86 → 4.58 → 4.94 tok/s) came from the
**speculative + unary** path with a **CPU-resident draft model** (llama-cpp-python in the
client venv is a CPU build). On the same hardware the **plain streaming** path is **4.6×
faster** (4.94 → 22.81 with ijru still on CPU). With CUDA the gap widens to 13.7×
(3.74 spec vs 51.16 plain), because once the workers are fast the CPU draft is nearly all of
the wall clock: in the CUDA+spec run the two workers together used 2.2 s of a 25.7 s run —
the other ~23 s was the client drafting (`user` time 3m25s ≈ 7.5 cores busy).

**Consequences to take seriously:**
- Speculative decoding is only a win when the draft runs on a GPU. On this stack it is a
  large net LOSS. Either build llama-cpp-python with HIP for the client venv, or keep spec
  OFF for chain serving.
- Tonight's earlier conclusions stand but should be read in that light: the async-pipelining
  LAN-negative verdict and the +8 % rebalance were both measured on the spec path. The
  rebalance conclusion (hub underused) is *unchanged* — plain-streaming step times show the
  same shape (hub 7 ms vs ijru 11 ms, now nearly balanced).
- **Re-run DONE (same night, both stages on GPU): the async-pipelining verdict got WORSE,
  decisively.** Same chain, same prompt, spec path both sides: sequential **4.99 tok/s** vs
  pipelined **0.10 tok/s** — a 50× penalty (946 s for 96 tokens). Worker time in that run:
  hub 1.88 s + ijru 3.39 s out of **946 s** — i.e. >99 % of the wall clock was the client's
  CPU draft doing speculative-continuation re-proposals, and the far stage's per-step even
  degraded (26 ms → 38 ms) from interleaving contention. Conclusion, now measured in both
  regimes: **async pipelining is not a LAN technique on this stack at all** — its cost scales
  with draft work, and our draft is CPU-bound. It becomes worth re-testing only when BOTH
  (a) the draft runs on a GPU and (b) there is real WAN RTT to hide. `NKS_ASYNC_PIPELINE`
  stays default OFF; correctness remains proven (byte-identical), so the code stays.

## Where this puts the project

A 30B-class MoE **split across two heterogeneous boxes** (AMD ROCm + NVIDIA CUDA, different
llama.cpp builds) now serves at **51 tok/s** — vs 1.86 tok/s for the June split experiment
that produced the "route, don't split" doctrine. The doctrine still holds for *latency-bound
WAN* topologies, but on a LAN with both stages on GPUs, splitting is no longer a penalty:
it is how a 30B fits at all when one card cannot hold it.

## Reproduce

```
# ijru (CUDA worker)
cd ~/nakshatra && python3 scripts/worker.py --port 5572 \
  --sub-gguf ~/.nakshatra/slices/qwen3-30b-q3-L15-48.gguf --mode last \
  --layer-start 15 --layer-end 48 \
  --daemon-bin ~/llama.cpp/build-cuda-static/bin/llama-nakshatra-worker \
  --n-ctx 4096 --n-gpu-layers 99
# hub (ROCm worker, 9070 XT)
cd ~/nakshatra && HIP_VISIBLE_DEVICES=1 .venv/bin/python scripts/worker.py --port 5562 \
  --sub-gguf ~/.nakshatra/slices/qwen3-30b-q3-L0-15.gguf --mode first \
  --layer-start 0 --layer-end 15 \
  --daemon-bin ~/llama.cpp/build/bin/llama-nakshatra-worker --n-ctx 4096 --n-gpu-layers 99
# client
.venv/bin/python scripts/client.py --config ~/.nakshatra/qwen3-30b-q3-chain-v2.yaml \
  --model-path <full Q3 gguf or the ollama blob> --max-tokens 96 --tls-mode off --use-streaming \
  --prompt "Explain, step by step, why the sky appears blue at noon and red at sunset."
```
⚠️ Test hygiene learned the hard way: run the daemon **through `worker.py`**, not directly over
SSH — a direct run gets EOF on stdin, exits immediately, and frees its VRAM, which reads as
"no offload" if you measure a moment later. Verify with `nvidia-smi` *while the worker serves*.
