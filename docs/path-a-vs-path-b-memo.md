# Path A vs Path B Feasibility Memo — llama.cpp's RPC for Nakshatra

**Question.** Can Nakshatra's LlamaCppBackend reuse llama.cpp's existing RPC machinery (`tools/rpc/rpc-server.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp`), or must it bypass that layer and drive GGML directly?

**Short answer.** The naive Path A — "spawn `rpc-server` as a subprocess and forward Nakshatra's per-layer activation requests through it" — is **not feasible**. llama.cpp's existing RPC is a *remote GGML compute backend*, not a *distributed model server*. Its abstraction level is wrong for what Nakshatra needs. Recommendation is **Path B** with a specific shape that reuses llama.cpp's per-vendor compute (Vulkan, CUDA, ROCm, Metal) but bypasses the RPC layer.

Sources read in `~/llama.cpp/`:
- `ggml/include/ggml-rpc.h` (30 lines)
- `tools/rpc/rpc-server.cpp` (337 lines)
- `ggml/src/ggml-rpc/ggml-rpc.cpp` (2118 lines, key sections inspected)
- `include/llama.h` (relevant sections around model/context params and RPC)
- `src/llama.cpp` and `common/arg.cpp` (how the master process integrates RPC backends)

---

## 1. What llama.cpp's RPC actually exposes today

### 1.1 The shape of the public interface

The entire public surface of the RPC backend is six functions in `ggml-rpc.h:14-26`:

```c
ggml_backend_t          ggml_backend_rpc_init(const char * endpoint, uint32_t device);
bool                    ggml_backend_is_rpc(ggml_backend_t backend);
ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);
void                    ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);
void                    ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                      size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);
ggml_backend_reg_t      ggml_backend_rpc_reg(void);
ggml_backend_reg_t      ggml_backend_rpc_add_server(const char * endpoint);
```

These are the **GGML backend** API — initialise, query memory, start a server. There is no notion of "model," "layer," "token," or "inference session" anywhere in the header. The unit of abstraction is a *ggml backend*, the same abstraction used for CPU/CUDA/Metal/Vulkan locally.

### 1.2 The wire protocol

The protocol version is `RPC_PROTO_MAJOR_VERSION 3` / minor 6 (`ggml-rpc.h:9-11`). It is a **custom binary protocol over TCP**, not gRPC, not protobuf, not JSON. Messages are `#pragma pack(push, 1)` packed structs (`ggml-rpc.cpp:69-223`). TCP_NODELAY is set on every socket (`ggml-rpc.cpp:304-309`). MAX_CHUNK_SIZE is 1 GiB (`ggml-rpc.cpp:42`).

The complete command set is enumerated at `ggml-rpc.cpp:91-111`:

```c
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_COUNT,
};
```

**Note the granularity.** These are GGML primitives: allocate a buffer on the remote device, push raw tensor bytes into it, run an arbitrary ggml graph against allocated buffers, pull tensor bytes back. There is no `RPC_CMD_FORWARD_LAYER`, no `RPC_CMD_DECODE_TOKEN`, no `RPC_CMD_KV_CACHE_*`. The "model" does not exist in this protocol.

### 1.3 The unit of work

The unit dispatched by `RPC_CMD_GRAPH_COMPUTE` is **an arbitrary ggml computation graph**. The wire format is documented inline at `ggml-rpc.cpp:1478-1479`:

```
| device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
```

The master serialises an entire ggml graph (every node, every tensor metadata struct) and sends it; the server deserialises into a `ggml_cgraph`, calls `ggml_backend_graph_compute(backends[device], graph)` (`ggml-rpc.cpp:1539`), and confirms success. Tensor data lives in remote buffers that were previously allocated and populated.

**This means the master decides graph topology.** The remote `rpc-server` is, structurally, a dumb GPU-over-TCP. It executes whatever graph the master sends.

### 1.4 The cache_dir / FNV-hash mechanism

`SET_TENSOR_HASH` (`ggml-rpc.cpp:1274-1314`) is the only mechanism by which weights "stay local" on a worker. It works as follows:

1. When the master wants to push a tensor larger than `HASH_THRESHOLD = 10 MiB` (`ggml-rpc.cpp:116`), it first computes an FNV-1a hash of the data and sends only the hash via `RPC_CMD_SET_TENSOR_HASH` (`ggml-rpc.cpp:185-193`, `ggml-rpc.cpp:636-655`).
2. The server checks `cache_dir/<hash>` on disk (`ggml-rpc.cpp:1254-1272`); if present, it loads the cached file into the remote buffer and returns success.
3. If not cached, server returns `result=0` and the master falls back to `RPC_CMD_SET_TENSOR` with the full data, which the server writes to disk (`ggml-rpc.cpp:1240-1249`) for next time.

This is **opportunistic content-addressable caching of weight blobs**, not a distributed model store. Two implications:

- Weights are not "owned" by the worker. They are owned by the master, optionally cached by the worker on disk.
- The worker has no idea what the cached blobs *are* — they're hash-keyed byte ranges, not "layer 42's q_proj weight."

### 1.5 What the master actually does

`src/llama.cpp:923-936` confirms the architecture: when the master process inits the model, RPC-backed devices are enumerated alongside local CUDA/Metal/Vulkan devices and inserted into `model->devices`. They are inserted **at the front** of the device list (`src/llama.cpp:970`) "to minimize network transfers" — the master prefers to assign weights to RPC devices first, so distant slow weights live there and local fast weights live locally.

The model is loaded by the master via `llama_model_load_from_file` (`llama.h:451-453`). There is no API for partial loading. The closest things are `vocab_only` (`llama.h:310`) and `no_alloc` (`llama.h:317`, "only load metadata and simulate memory allocations"); neither produces a working partial-model worker. `tensor_split` (`llama.h:296`) controls splitting *across already-loaded devices*, not which layers exist.

---

## 2. Whether Nakshatra's protocol can sit on top

**No, not in the form the original §8.7 sketched.** The naive Path A — "Nakshatra worker spawns `rpc-server` as a subprocess and forwards per-layer activation requests through it" — does not type-check at the protocol boundary.

To see why concretely, walk through what would happen:

- Nakshatra worker receives a Nakshatra request: "for session S, run forward over layers [16, 32) on this hidden state tensor."
- The worker would need to translate that into the rpc-server's protocol. But the rpc-server protocol doesn't accept "run forward over layers [16, 32)." It accepts "allocate buffer," "set tensor bytes at address X," "compute this ggml graph against these buffers."
- The graph for "run forward over layers [16, 32) of a Llama model" is a complex ggml subgraph involving every weight tensor, every intermediate, attention masks, RoPE positions, KV cache, etc. The Nakshatra worker would need to **construct this graph itself** — i.e., it would need to know how to build the layer subgraph for whatever architecture is being served.

Constructing the layer subgraph is exactly what llama.cpp's `src/llama-model.cpp` and `src/llama-graph.cpp` do today. So "wrap rpc-server" reduces to "reimplement the model's forward graph in Nakshatra's worker code," which is more or less Path B with extra steps.

**The wire-format mismatch is not a missing field; it is a missing abstraction.** llama.cpp's RPC speaks GGML; Nakshatra's protocol speaks transformer layers. To bridge them you would need to take ownership of the graph construction, at which point you no longer need rpc-server at all.

---

## 3. Per-layer dispatch granularity

The question as originally posed in §8.7 — "does llama.cpp's RPC support per-layer dispatch?" — is malformed. llama.cpp's RPC is **finer-grained than per-layer** (it's per-graph-op) but in the wrong direction: the *master* assembles per-layer subgraphs and dispatches the resulting ggml graph. The remote does not own a layer; it owns a memory region and a compute device.

Concretely on Nakshatra's three needs:

- **"Load layers [N, M) from a partial GGUF on the worker."** Not supported. llama.cpp has no partial-model load. The master must load the full GGUF (or pretend to with `vocab_only`/`no_alloc`, which doesn't produce runnable inference). The closest thing — pre-splitting a GGUF into per-worker sub-GGUFs and loading each on its respective worker — is doable as a build-time script but is independent of llama.cpp's RPC code.
- **"Run forward over those layers."** Not directly. llama.cpp's `llama_decode` (the standard inference call) runs the *entire* forward pass: embed → layers → norm → lm_head → logits. To run only layers [N, M), you have to either modify the graph builder or operate at the GGML level.
- **"Return activations."** Not directly. `llama_decode` returns logits or embeddings, not raw mid-layer hidden states. There is `cb_eval` (`llama.h:350`, a per-eval callback) which lets you observe intermediate tensors, but observing is not the same as exporting them as the official output of a worker.

So: even ignoring RPC, the *single-process llama.cpp inference path* does not natively expose the operation Nakshatra needs ("hidden state in → run layers [N, M) → hidden state out"). This is the load-bearing missing piece, and it exists at the `llama_decode`/graph-builder layer, not the RPC layer.

---

## 4. Tensor handoff at the boundary (DLPack)

Less load-bearing once §1–§3 are settled, but for completeness:

- `rpc-server` produces output tensors as bytes in an opaque remote buffer. To extract them, the master sends `RPC_CMD_GET_TENSOR` with a `(tensor, offset, size)` request (`ggml-rpc.cpp:195-199`, server handler at `:1349`). Bytes come back over TCP.
- llama.cpp's GGML tensors are `ggml_tensor` structs with raw data pointers, dtype enums, and shape arrays. There is no built-in DLPack export. Adding DLPack would be a small standalone PR (build a `DLManagedTensor` wrapping a `ggml_tensor`) — call it 1–2 days of work — but it is not present today.
- For Nakshatra's wire format (per `docs/petals-architecture.md` §4.1), what matters is `(shape, dtype, raw bytes)`. The 3D activation tensor is small (16 KB/token at hidden=8192), so a copy at the boundary is cheap regardless.

This is not a Path A vs Path B decision driver. Either path needs a small boundary shim.

---

## 5. Recommendation

### 5.1 Reject naive Path A

**Do not** wrap llama.cpp's `rpc-server` as Nakshatra's worker. The abstraction level is wrong: rpc-server is a remote GGML backend, the Nakshatra worker is a remote transformer-layer-range executor, and bridging the two requires reimplementing the graph builder anyway.

### 5.2 Reject pure Path B

Going all the way down to GGML graph APIs — building each layer's subgraph from scratch in C++ in Nakshatra's codebase — is the maximally flexible option but throws away llama.cpp's existing per-architecture graph builders (Llama, Falcon, Mistral, Mixtral, Qwen, etc.). For v0.1 this is too much work, and it tightly couples Nakshatra to GGML internals which change frequently upstream.

### 5.3 Recommended path: **B-prime — bypass RPC, reuse llama.cpp's graph builder**

Each Nakshatra worker is a customized llama.cpp inference process. It links against llama.cpp as a library and uses llama.cpp's per-architecture graph construction code, but:

- **Loads only its layer range.** v0.1: pre-split the GGUF into per-worker sub-GGUFs as a build-time script, so each worker's `llama_model_load_from_file` sees only its assigned layers plus whatever pre/post-processing tensors that worker needs (embedding for worker 0, lm_head for worker N-1, RoPE freqs everywhere). Pre-splitting is operational work, not llama.cpp work.
- **Exposes a "hidden-state in → hidden-state out" entry point.** This is the C++ work. It requires either (a) a new `llama_decode_layers(ctx, hidden_in, layer_start, layer_end, hidden_out)` API contributed upstream, (b) a Nakshatra-side fork/patch of `llama_decode` that does the same, or (c) using `cb_eval` (`llama.h:350`) to inject and extract activations through a hacky callback path. Option (a) is the cleanest; option (b) is the most pragmatic for a v0.1 timeline.
- **Speaks Nakshatra's protocol on top.** gRPC over Tailscale for the inter-worker protocol (per `docs/petals-architecture.md` §6.2). The Nakshatra protocol layer is independent of llama.cpp.

### 5.4 Estimated effort

For the C++ wrapper described in §5.3, with one full-time-equivalent engineer comfortable with llama.cpp's codebase:

- **Pre-split GGUF script** (Python, using `gguf-py/`): ~1 week.
- **Customized inference entry point** (option b: patched `llama_decode` accepting hidden-state input/output): **6–10 weeks** depending on how much of `src/llama-graph.cpp` and `src/llama-context.cpp` needs to be touched. This is the dominant effort.
- **DLPack boundary shim**: ~1 week.
- **Integration with Nakshatra's gRPC service**: ~2 weeks.

**Total: 10–14 weeks for the C++ side.** This is in the lower half of §9's "8–16 weeks" budget for the LlamaCppBackend wrapper, and below the §8.7 "Path B" upper estimate of 16+. The savings come from reusing llama.cpp's per-architecture graph builders instead of reimplementing them on top of GGML.

### 5.5 Specific blockers and risks

- **No upstream API for partial-layer execution.** The recommended approach (option b in §5.3) is a fork-and-patch of `llama_decode`. Upstreaming a clean version of this (option a) is a 4–8 week additional effort and a separate political/coordination question with the llama.cpp maintainers. v0.1 should ship on the local patch; upstreaming can come later.
- **Pre-split GGUF format is unofficial.** v0.1 tooling will produce sub-GGUFs that the broader llama.cpp ecosystem doesn't recognise as valid. This is fine for our private clusters but is a gotcha for documentation and operator UX.
- **`cb_eval` is not a substitute for proper API.** Option (c) — using the eval callback to inject and extract hidden states — would technically avoid the patch but produces a fragile, inverted control flow that will rot fast. Reject it unless the patch path proves blocked.
- **llama.cpp is fast-moving.** The patched `llama_decode` will need rebasing roughly monthly. Budget ~0.5 engineer-day per upstream resync.

### 5.6 What llama.cpp's existing RPC is still useful for

Within a single Nakshatra worker that has multiple local GPUs (e.g. a future cloud-NVIDIA deployment with 4× H100), llama.cpp's RPC-as-internal-fabric remains valid. The Nakshatra worker can use llama.cpp's standard RPC mode internally to span its local GPUs while still presenting a single Nakshatra-protocol endpoint upstream. This is an internal optimisation, not a v0.1 concern, and does not change the architecture above.

---

## 6. Summary

| Question (from §8.7) | Answer |
|---|---|
| Does llama.cpp's RPC support per-layer dispatch? | No — it dispatches arbitrary ggml graphs, finer-grained than layers but in the wrong direction. |
| Can Nakshatra spawn rpc-server as a subprocess? | No — wire format mismatch is a missing abstraction, not a missing field. |
| Does it support partial-model loading on workers? | No — master loads the full model; worker is a dumb GPU-over-TCP. |
| Path A or Path B? | Neither as originally framed. **Path B-prime**: bypass RPC, reuse llama.cpp's graph builder, patch `llama_decode` for hidden-state I/O. |
| Estimated C++ effort | 10–14 weeks for v0.1, dominated by the patched `llama_decode` work. |
| Blocker for naive Path A? | Yes — llama.cpp's RPC operates at GGML primitive level, not transformer-layer level. Reusing it would require Nakshatra's worker to assemble per-layer ggml subgraphs itself, which is more work than just doing Path B-prime. |

**Decision.** Adopt Path B-prime. Update §8.7 of `petals-architecture.md` accordingly.
