# Nakshatra

> Distributed LLM inference across heterogeneous workers (NVIDIA / AMD / Apple Silicon / CPU). Splits one model by layer ranges; patched llama.cpp + gRPC chain protocol. (Inspired by [Petals](README_PETALS.md); v0.1 design is independent.)

Status (2026-05-06): **v0.1 functionally alive.** Two-worker cluster on Tailscale produces the same top-1 token as a single-machine `llama-cli` reference. See [`experiments/v0.0/m6_findings.md`](experiments/v0.0/m6_findings.md) for the empirical result.

**Paper:** [Nakshatra: Vendor-Agnostic Distributed Inference on Heterogeneous Consumer Hardware](https://pnl.market/research/6a017d83b86a1bf1c69ea714) (Bastola, 2026).

---

## What this is

Nakshatra splits one transformer model across multiple machines. Each worker holds a contiguous range of the model's layers — say worker A holds layers 0–13 and worker B holds layers 14–27 of a 28-layer model. A request flows: tokenizer on the client → worker A computes its layers and emits a hidden-state vector → worker A ships the vector to worker B over the network → worker B finishes the layers, applies the language-model head, returns the next token.

Per-token network traffic is the size of one hidden state vector (12 KB for a 3B model, 16 KB for a 70B model) crossing one hop per worker boundary. Trivial bandwidth. Weights stay local on each worker — they are never streamed during inference.

The architectural commitments and roadmap live in [`docs/`](docs/). Start with [`petals-architecture.md`](docs/petals-architecture.md) for the design and [`v0.1-implementation-plan.md`](docs/v0.1-implementation-plan.md) for the milestones and acceptance test.

---

## Quickstart — stand up a 2-worker cluster

The walkthrough below takes a fresh pair of machines from "we have Python and a GGUF" to "we're seeing the right token come back." Targets the v0.1 §7 ship gate: under one hour for an external operator.

### Prerequisites

On **both** machines:

- Python 3.9+
- `git`, `cmake`, a C++17 compiler (gcc 13+ on Linux, Xcode CLT on macOS)
- About 4 GB of free disk per worker for a sub-GGUF of a 3B-class model (more for larger models)
- Tailscale or any other point-to-point IP transport between the two machines
- SSH between them (for shipping the sub-GGUF)

On the **first machine** only:

- The full GGUF of the model you want to run. Llama-family architectures only for v0.1. Tested with Llama-3.2-3B-class fine-tunes (28 layers, hidden_size=3072).

### 1. Clone and build patched llama.cpp on each machine

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
git checkout c46583b   # or close enough; recent llama.cpp commits work
```

Apply the M3+M4 patches (in [`experiments/v0.0/m4_patches/`](experiments/v0.0/m4_patches/)) on each machine:

```bash
cd ~/llama.cpp
patch -p4 < /path/to/nakshatra/experiments/v0.0/m4_patches/llama-model.h.patch
patch -p4 < /path/to/nakshatra/experiments/v0.0/m4_patches/llama-model.cpp.patch
patch -p4 < /path/to/nakshatra/experiments/v0.0/m4_patches/llama-model-loader.cpp.patch
patch -p4 < /path/to/nakshatra/experiments/v0.0/m4_patches/llama-graph.cpp.patch
patch -p4 < /path/to/nakshatra/experiments/v0.0/m4_patches/models_llama.cpp.patch
```

Set up the worker daemon as a CMake target by dropping the source into `examples/nakshatra-spike/`:

```bash
mkdir -p examples/nakshatra-spike
cp /path/to/nakshatra/experiments/v0.0/worker_daemon.cpp examples/nakshatra-spike/
cat > examples/nakshatra-spike/CMakeLists.txt <<EOF
set(TARGET llama-nakshatra-worker)
add_executable(\${TARGET} worker_daemon.cpp)
install(TARGETS \${TARGET} RUNTIME)
target_link_libraries(\${TARGET} PRIVATE common llama \${CMAKE_THREAD_LIBS_INIT})
target_compile_features(\${TARGET} PRIVATE cxx_std_17)
EOF

# Add nakshatra-spike to examples/CMakeLists.txt's add_subdirectory list, then:
mkdir -p build && cd build
cmake -DGGML_METAL=ON ..   # macOS Metal. For Linux: -DGGML_HIPBLAS=ON (ROCm) or -DGGML_CUDA=ON (NVIDIA).
                           # For deterministic regression tests only, build with ALL GPU backends OFF.
cmake --build . --target llama-nakshatra-worker -j
```

You should now have `~/llama.cpp/build/bin/llama-nakshatra-worker`.

### 2. Generate sub-GGUFs (on the machine that has the full model)

```bash
cd /path/to/nakshatra
python3 -m venv ~/nakshatra-venv && source ~/nakshatra-venv/bin/activate
pip install ~/llama.cpp/gguf-py grpcio grpcio-tools pyyaml

python experiments/v0.0/partial_gguf.py \
    /path/to/full/model.gguf \
    /tmp/cuts/w0.gguf --start 0 --end 14
python experiments/v0.0/partial_gguf.py \
    /path/to/full/model.gguf \
    /tmp/cuts/wlast.gguf --start 14 --end 28 --keep-token-embd
```

The `--keep-token-embd` on the last worker is required for **tied-embedding** models (Llama-3.2 family included) — the lm_head falls back to using token_embd as its output projection.

Cut points (`--start`, `--end`) must add up to a contiguous partition of the model's layer count. For an N-layer model with K workers, divide as evenly as you can.

### 3. Ship the back-half sub-GGUF to the second machine

```bash
scp /tmp/cuts/wlast.gguf user@host2:/tmp/wlast.gguf
```

### 4. Generate gRPC stubs and copy worker scripts

On **both** machines:

```bash
cd /path/to/nakshatra
bash scripts/generate.sh     # produces scripts/nakshatra_pb2*.py
```

### 5. Write the cluster YAML

```yaml
# scripts/cluster.yaml — adjust addresses, ports, paths to your setup
model:
  id: my-model-q4
  hidden_size: 3072       # match the model
  num_blocks: 28
  wire_dtype: f32

workers:
  - id: machine-a
    address: 100.X.Y.Z      # Tailscale IP of host 1
    port: 5530
    layer_range: [0, 14]
    sub_gguf_path: /tmp/cuts/w0.gguf
    mode: first
  - id: machine-b
    address: 100.A.B.C      # Tailscale IP of host 2
    port: 5531
    layer_range: [14, 28]
    sub_gguf_path: /tmp/wlast.gguf
    mode: last
```

The first worker must have `mode: first`, the last `mode: last`, intermediates `middle`. Layer ranges must form a contiguous `[0, num_blocks)` partition with no gaps.

### 6. Start the workers

On **host 1**:

```bash
python scripts/worker.py --port 5530 \
    --sub-gguf /tmp/cuts/w0.gguf --mode first \
    --layer-start 0 --layer-end 14 --model-id my-model-q4 --n-ctx 256
```

On **host 2**:

```bash
python scripts/worker.py --port 5531 \
    --sub-gguf /tmp/wlast.gguf --mode last \
    --layer-start 14 --layer-end 28 --model-id my-model-q4 --n-ctx 256 \
    --daemon-bin /Users/.../llama.cpp/build/bin/llama-nakshatra-worker
```

Each worker prints `M5 listening on :PORT` once its daemon has loaded the sub-GGUF (a few seconds for 3B-class models).

### 7. Run the chain

From any machine that can reach both Tailscale IPs:

```bash
python scripts/client.py --config scripts/cluster.yaml \
    --model-path /path/to/full/model.gguf \
    --prompt "The capital of France is" --max-tokens 10
```

Expected output:

```
[chain] OK: contiguous coverage of [0, 28)
[chain] step 1: id=12366 ' Paris'
[chain] step 2: id=13 '.'
...
[chain] generated 10 tokens in 3.7s  (2.7 tok/s)
[chain] full: 'The capital of France is Paris. The capital of France is Paris. The'
TOPTOKS_CHAIN 12366 13 578 6864 315 9822 374 12366 13 578
```

If the first generated token matches what `llama-cli` produces on the same prompt with greedy decoding, the cluster is operating correctly.

### Pre-flight check

Before starting the chain, validate your YAML and worker reachability:

```bash
python scripts/validate_cluster.py --config scripts/cluster.yaml
```

This connects to each worker, verifies `Info` returns the expected layer range, checks the partition is contiguous, and reports the first/last worker flags. It does **not** load any models — it's a fast network + config sanity check.

---

## Repository layout

```
docs/
├── petals-architecture.md          v0.1 design (the "what")
├── path-a-vs-path-b-memo.md        C++ feasibility memo (the "why this approach")
├── petals-deep-read.md             upstream-Petals source-reading notes
├── north-star.md                   L1-L4 vision (substrate / engine / OS / agents)
├── v0.0-validation-plan.md         Phase 0 gates (both RESOLVED)
├── v0.1-implementation-plan.md     7 milestones, ship gate, resolved decisions
└── m4-decode-patch-design.md       M4 patch points

experiments/v0.0/
├── partial_gguf.py                 sub-GGUF generator
├── partial_gguf_findings.md        Phase 0a evidence — loader rejects partials
├── spike.cpp + spike_findings.md   Phase 0b evidence — orchestration protocol
├── cb_eval_probe.py + findings     Python pivot story
├── m4_patches/                     the 5 patches that make llama.cpp do partial-load
├── m4_chain.cpp + findings         M4 single-process chain validation
├── worker_daemon.cpp               C++ daemon that runs patched libllama
├── m5_findings.md                  M5 — gRPC chain on localhost
└── m6_findings.md                  M6 — cross-machine acceptance test

proto/
└── nakshatra.proto                 v0.1 wire contract

scripts/
├── worker.py                       Python gRPC worker (spawns C++ daemon)
├── client.py                       chain-walking client
├── validate_cluster.py             pre-flight YAML + reachability check
├── generate.sh                     regenerates Python protobuf stubs
├── cluster_localhost.yaml          example: 2 workers on one host
└── cluster_crossmachine.yaml       example: 2 workers on Tailscale
```

## Architecture in one paragraph

Each worker is a Python gRPC process that spawns a long-lived C++ daemon (`llama-nakshatra-worker`). The daemon holds the model slice and KV cache, accepts framed binary messages over stdin/stdout, and runs `llama_decode` per request. The Python worker pumps gRPC requests through to the daemon and back. The client tokenizes the prompt locally, calls the first worker with token IDs, ferries the returned hidden state through any middle workers, and gets back a token id from the last worker. The v0.3 federation extends this with sub-GGUF auto-fetch (workers download missing slices from peers), latency-aware chain assembly via a pillar registry, and Metal / ROCm GPU offload across heterogeneous machines. **Outputs on GPU paths are reproducible *in distribution*, not byte-for-byte** — kernel-level non-determinism in current backends. Bit-identical reproducibility is available via CPU-only workers (`--gpu-backend cpu`), kept for regression tests. See `docs/v0.5-design-lock.md` for the full property statement.

## Status — what's shipped, what's coming

**Shipped (v0.1 → v0.3):**

- GPU acceleration — Metal on Macs, ROCm on Linux. Live on the 5-machine lab cluster.
- Streaming KV-cache reuse (M2.5) — 5× speedup on multi-token generation vs naive re-prefill.
- Sub-GGUF distribution: workers advertise their cached slices and auto-fetch missing ones from peers over HTTP byte-range (Phase 4 / 4a).
- Latency-aware chain assembly via a pillar registry (Phase H / I).
- `/healthz` endpoint per worker with rich state (identity, daemon liveness, recent RPC latency, GPU offload, GPU inventory via ioreg).

**In-progress (v0.5 protocol foundations, partial):**

- `Inference` streaming RPC (`--use-streaming`) — bidi streams replace per-step Forward calls (M0.5.1).
- Idempotency cache on workers (M0.5.2) — replayed `(session_id, step_id)` returns cached response without re-decoding.
- Server-to-server activation push (`--use-streaming-push`, M0.5.3 v1) — 2-worker chains working today; multi-hop is v2.
- Client-side recovery on stream failure (M0.5.4 v0) — replays history through fresh streams; continuation is plausible but not bit-identical (per the non-determinism property).

**Deferred:**

- Alternate-worker recovery from a registry of redundant peers (M0.5.4 v1).
- Latency-based silent-failure detection (M0.5.4 v2).
- Non-Llama model architectures — the partial-load patches today gate only `models_llama.cpp`; Qwen3 / Gemma / etc. need their own per-arch patch.
- DHT-based public-network peer discovery (v1.0+).
- Opt-in cryptographic verification (v1.0+).

See [`docs/v0.5-design-lock.md`](docs/v0.5-design-lock.md) for the v0.5 design contract and acceptance criteria.

## License & attribution

Forked from [Petals](README_PETALS.md). The Nakshatra additions (everything in `docs/`, `experiments/v0.0/`, `proto/`, `scripts/`, plus the patches in `experiments/v0.0/m4_patches/`) are MIT-licensed (per the upstream Petals repository's license).
