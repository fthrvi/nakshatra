# Petals Architecture — Reading Notes for Nakshatra

A working map of how Petals splits a model across machines, what flows over the wire, and where it is welded to PyTorch/CUDA. The intent is to identify the surfaces Nakshatra will need to abstract to support heterogeneous (NVIDIA / AMD / Apple Silicon) workers, then to scope what Nakshatra actually builds in v0.1 versus what it inherits from Petals' design.

Sources read:
- `src/petals/server/block_functions.py`
- `src/petals/server/handler.py`
- `src/petals/client/inference_session.py`
- `src/petals/utils/auto_config.py`

References below use `path:line` against the upstream Petals tree as cloned into `nakshatra/`.

---

## Table of contents

0. [Project framing](#0-project-framing)
1. [How Petals splits a model across workers](#1-how-petals-splits-a-model-across-workers)
2. [Wire protocol](#2-wire-protocol)
3. [CUDA / PyTorch-specific assumptions](#3-cuda--pytorch-specific-assumptions)
4. [Vendor-agnostic abstractions](#4-where-vendor-agnostic-abstractions-could-replace-the-assumptions)
5. [Trust model and verification](#5-trust-model-and-verification)
6. [Things Nakshatra explicitly DEFERS from Petals](#6-things-nakshatra-explicitly-defers-from-petals)
7. [MVP slice on actual available hardware](#7-mvp-slice-on-actual-available-hardware)
8. [Open questions for v0.1 design](#8-open-questions-for-v01-design)
   - 8.1 Static config file format
   - 8.2 Failure handling mid-chain
   - 8.3 Quantization at the activation boundary
   - 8.4 Model file management
   - 8.5 KV cache invalidation on worker restart
   - 8.6 Worker capability handshake
   - 8.7 Reuse llama.cpp's existing RPC backend, or bypass it?
9. [Realistic scope and timeline](#9-realistic-scope-and-timeline)
- [A.1 Operational addendum: lab-cluster optimization surfaced during research](#a1-operational-addendum-lab-cluster-optimization-surfaced-during-research)
10. [Appendix: file map for future edits](#10-appendix-file-map-for-future-edits)

---

## 0. Project framing

**Nakshatra is a distributed inference engine.** It runs a single large model split across multiple heterogeneous workers (NVIDIA, AMD, Apple Silicon, CPU) and serves token generation. That is the entirety of its scope.

**Prithvi is a separate project** — a coordination and economic layer (peer marketplace, payments, reputation, scheduling). Prithvi is **out of scope for this document and out of scope for Nakshatra v0.1**.

The two projects are independent:

- Nakshatra runs perfectly well with no Prithvi at all. A single operator with a private cluster (a research lab, a self-hosted deployment, a closed pilot) uses Nakshatra directly via static configuration.
- Prithvi can integrate Nakshatra as one of several inference backends. It can equally integrate Petals, vLLM, or any other engine. Prithvi's value is the coordination layer; the inference engine is pluggable.

Concretely, this means: do not import Prithvi concerns (peer reputation, payment, adversarial workers, on-chain attestation) into Nakshatra's core APIs. Nakshatra's contract is "given this set of workers and this model, generate tokens." Trust assumptions for v0.1 are addressed in §5 below; they do not require Prithvi.

---

## 1. How Petals splits a model across workers

### 1.1 Models are registered, not hardcoded

Model support is plugin-style. Each supported architecture (Llama, Falcon, etc.) calls `register_model_classes(...)` at import time, populating a global `_CLASS_MAPPING` keyed by HuggingFace `config.model_type` (`auto_config.py:22`, `auto_config.py:25-29`).

The `AutoDistributedModelForCausalLM` family (`auto_config.py:90-99`) is just dispatch: it loads a HuggingFace `AutoConfig`, looks up the right Petals subclass for that `model_type`, and delegates `from_pretrained`. **Petals does not introspect the model itself** — it relies on HF Transformers to define the block structure, then attaches a distributed wrapper.

Implication: layer assignment is per-block, where a "block" = one transformer decoder layer in HF's class hierarchy. Llama-70B → 80 blocks; the unit of distribution is one block.

### 1.2 Block UIDs and spans

Each transformer block is identified by a `ModuleUID` string. UIDs within a single request are joined by `CHAIN_DELIMITER` (`.`) into composite strings like `model_prefix.42.model_prefix.43.model_prefix.44` (see usage in `handler.py:330`, `inference_session.py:253`).

Server-side, each running server holds a dictionary `module_backends: Dict[ModuleUID, TransformerBackend]` (`handler.py:58`). The server advertises its UIDs into a Hivemind DHT.

Client-side, contiguous ranges of blocks held by one peer are represented as a `RemoteSpanInfo` with `start`, `end`, `peer_id` (used throughout `inference_session.py:249-271`).

### 1.3 Routing — choosing which servers handle which blocks

The client never picks layer assignments globally. Instead, on every inference call (and on every retry), it calls:

```
self._sequence_manager.make_sequence(
    block_idx, update_end, mode="min_latency", cache_tokens_needed=self._max_length
)
```
(`inference_session.py:376-378`)

This is a graph-search over the DHT-advertised servers. The "min_latency" mode (other modes exist in routing/) optimises for shortest expected wall-clock path. The returned spans cover `[block_idx, update_end)`, possibly using more servers than strictly needed, possibly reaching past `update_end` (line 380 clips it).

Critically: **the client builds the chain dynamically per request**, and rebuilds segments of it on failure (`inference_session.py:325-358`). There is no static partition of layers to machines. A server can come and go; the next request will re-route around it.

### 1.4 Pipeline execution

For a single forward pass over a span owned by one server:
1. Client sends hidden_states + prompts + hypo_ids (`inference_session.py:128-164`).
2. Server iterates the requested backends in order, running each block's `forward_pool.submit_task` (`block_functions.py:63-79`). Output of block `i` becomes input to block `i+1` *within the same machine*.
3. Server returns the last block's output.
4. Client takes that output and passes it to the next server in its chain (`inference_session.py:322-344`).

Crucially the server processes `[start, end)` of its span **before** returning to the client. Inter-server hops happen between spans, not between blocks. This batches network roundtrips by however many contiguous layers a server holds.

### 1.5 Optional: server-to-server pipelining (`rpc_push`)

If `config.use_server_to_server` is set, the client tells server N "after you're done, push your output directly to server N+1 instead of waiting for me to relay it" (`inference_session.py:174-182`, `handler.py:320-350`). The `next_servers` metadata field carries the ordered list of `(peer_id, session_id, start, end)` tuples for downstream hops. This is opt-in; the default still goes via the client.

---

## 2. Wire protocol

### 2.1 Transport stack

- libp2p (via `hivemind.p2p`) provides the underlying P2P transport, NAT traversal, peer discovery.
- Each RPC is a gRPC-style method on `TransformerConnectionHandler`, registered as a libp2p stream handler.
- DHT (Kademlia, via Hivemind) advertises which peer holds which UIDs.

### 2.2 Protobuf message types

All messages are `hivemind.proto.runtime_pb2`:
- `ExpertRequest { uid: string, tensors: repeated Tensor, metadata: bytes }` — primary client→server payload.
- `ExpertResponse { tensors: repeated Tensor }` — server→client payload.
- `Tensor { ... }` — Hivemind's torch-tensor wire encoding (raw buffer + dtype + shape + compression code).
- `ExpertUID`, `ExpertInfo` — used by `rpc_info`.

`request.metadata` is a MSGPack-encoded dict, decoded on the server with `MSGPackSerializer.loads(...)` (`handler.py:148`, `handler.py:262`).

### 2.3 RPC methods exposed by a server

| Method | Shape | Purpose | Code |
|---|---|---|---|
| `rpc_forward` | unary→unary | Single forward pass over a span (no KV cache) | `handler.py:352-378` |
| `rpc_forward_stream` | stream→stream | Same as above, chunked for tensors > `DEFAULT_MAX_MSG_SIZE` | `handler.py:380-409` |
| `rpc_backward` | unary→unary | Single backward pass over a span | `handler.py:434-459` |
| `rpc_backward_stream` | stream→stream | Chunked backward | `handler.py:461-488` |
| `rpc_inference` | stream→stream | Stateful inference session with server-side KV cache | `handler.py:132-195` |
| `rpc_push` | unary→unary | Direct server-to-server activation push | `handler.py:310-318` |
| `rpc_info` | unary→unary | Capacity / version / KV-cache headroom probe | `handler.py:575-592` |

### 2.4 Inference session state machine

Inference is the interesting case because it is stateful (KV cache) and the protocol is bidirectional streaming.

Sequence per server hop:
1. Client opens `rpc_inference` stream with the server (`inference_session.py:71-76`).
2. Client sends first `ExpertRequest`. Server reads `metadata` for `max_length`, `points`, `session_id`, `alloc_timeout`, `args_structure` (`handler.py:148-167`).
3. Server allocates KV cache via `_allocate_cache(...)` (`handler.py:170-172`, `handler.py:532-547`). Cache lives for the duration of the stream.
4. Server enters `iterate_rpc_inference` loop (`block_functions.py:144-238`). For each step request:
   - Deserialize input tensors.
   - Optionally update prefix length from `start_from_position` metadata (cache-truncation case).
   - Submit to either a merged short-inference pool (≤128 tokens, ≤1 token for NF4) or sequential per-block submission (`block_functions.py:199-226`).
   - Serialize last-block output, optionally push to next server, yield response.
   - Increment `prefix_length`.
5. Stream closes when client sends an empty `ExpertRequest` (`inference_session.py:198-207`, recognized at `_read_inputs_from_queue`).

### 2.5 Per-step request metadata fields

Carried in MSGPack-encoded `request.metadata`:

| Key | Type | Purpose |
|---|---|---|
| `session_id` | str (uuid) | Identifies a stateful inference session on the server |
| `step_id` | str (uuid) | Idempotency key — prevents double-counting if a push arrives twice (`handler.py:263-281`) |
| `max_length` | int | Requested KV cache size in tokens |
| `points` | float | Quality-of-service "credits" used by the prioritizer |
| `start_from_position` | int | Truncate KV cache to this prefix length before stepping |
| `next_servers` | list of `[peer_id, session_id, start, end]` | Chain of downstream servers for `rpc_push` |
| `args_structure` | bytes | Packing info to reconstruct `*args, **kwargs` from flat tensor list (`utils/packaging.py`) |
| `output_compression` | list[int] | Per-tensor compression codes for the response |
| `active_adapter` | str | PEFT adapter name (LoRA-style fine-tune) |
| `pushed` | bool | Marks a request that arrived via `rpc_push` rather than direct client send |
| `alloc_timeout` | float | Max wait time for KV cache allocation |

### 2.6 Tensor payload conventions

- All tensors travel as Hivemind `Tensor` protos, produced by `serialize_torch_tensor(t, compression)` and consumed by `deserialize_torch_tensor(...)`.
- The first tensor in an inference request is `hidden_states`, shape `[batch, seq_len, hidden]`, dtype matching `requested_backends[0].dtype` (`block_functions.py:175-179`).
- Second tensor is `prompts` (deep-prompt tuning, optional, often a `DUMMY` sentinel).
- Third is `hypo_ids`, dtype-asserted to `torch.int64` (`block_functions.py:180`) — used for beam search hypothesis routing.
- Backend `dtype` (`block_functions.py:53`) is the wire dtype for that span. All clients must cast to it before sending.

### 2.7 Streaming for large tensors

`DEFAULT_MAX_MSG_SIZE` from libp2p caps single-message size. `rpc_forward_stream` and `rpc_backward_stream` use `split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)` (`handler.py:407-409`) to chunk a serialized tensor across multiple `ExpertResponse` messages. Inference uses a different mechanism — short steps stay within size limits, and KV cache is server-resident so payloads are small.

---

## 3. CUDA / PyTorch-specific assumptions

These are the welds that make the codebase NVIDIA/PyTorch-only. Each is a candidate for an abstraction in Nakshatra.

### 3.1 `torch.Tensor` is the universal currency
Every file imports `torch` directly and operates on `torch.Tensor`. There is no abstract `Tensor` interface. Every shape/dtype/device operation (`.to(dtype)`, `.cpu()`, `.ndim`, `torch.cat`, `torch.is_grad_enabled`) is a torch API call.
- `block_functions.py:55, 60, 65, 71, 98-104`
- `inference_session.py:115-117, 309-313`

### 3.2 Wire serialization is torch-shaped
`serialize_torch_tensor` / `deserialize_torch_tensor` (Hivemind) only know about torch tensors. The compression codes they emit (`NONE`, `FLOAT16`, `BFLOAT16`, `QUANTIZE_8BIT`, `QUANTIZE_BLOCKWISE_8BIT`) reference Hivemind's torch-coupled compression registry.
- `block_functions.py:9, 230, 232`
- `handler.py:19, 22, 355, 437`
- `inference_session.py:10, 158-160, 165`

### 3.3 `BatchTensorDescriptor` is torch-coupled
Schema metadata uses `hivemind.utils.tensor_descr.BatchTensorDescriptor`, which carries a `torch.dtype` and a Hivemind compression code. Anywhere a non-torch backend produces a tensor, it must be converted to torch before this layer sees it.
- `inference_session.py:14, 144-146`

### 3.4 Hardcoded torch dtype assertions
- `block_functions.py:180` — `hypo_ids.dtype == torch.int64` (typed assert)
- Multiple `.to(dtype)` calls assume torch dtype objects throughout `block_functions.py` and `handler.py`

### 3.5 bitsandbytes / NF4 in protocol-visible logic
The merge-pool decision branches on `quant_type == QuantType.NF4` (`block_functions.py:199`). NF4 is a bitsandbytes feature; bitsandbytes ships only CUDA kernels at the version pinned in `setup.cfg` (`bitsandbytes==0.41.1`). So the *server's wire-level batching policy* is conditioned on a CUDA-only feature.

### 3.6 `TransformerBackend` runs torch compute pools
`backend.forward_pool`, `backward_pool`, `inference_pool` are `PrioritizedTaskPool`s from Hivemind that run torch operators on whatever device the backend was initialised on (CUDA, in practice). The pool primitive itself is a torch-async-task abstraction; it cannot transparently dispatch to JAX/MLX/llama.cpp.
- `block_functions.py:67-78, 114-118, 130-134, 218-226`

### 3.7 KV cache is torch tensors in `memory_cache`
`backend.memory_cache.allocate_cache(...)` (`handler.py:546`) yields handles to torch tensors (CPU- or GPU-resident depending on offload config). Cache descriptors are torch-shaped (`backend.get_inference_cache_descriptors`). `cache_bytes_per_token` (`handler.py:582`) assumes a fixed per-token byte cost in torch tensor terms.

### 3.8 Client unconditionally moves to CPU before sending
```python
inputs = inputs.cpu()
prompts = prompts.cpu()
hypo_ids = hypo_ids.cpu()
```
(`inference_session.py:311-313`)

This is reasonable for serialization, but it presumes a torch CPU intermediary even on non-torch frontends.

### 3.9 `tensor_parallel==1.0.23` for intra-block sharding
Declared in `setup.cfg`. Used to shard a single block across multiple GPUs on one host. NCCL-centric, CUDA-centric.

### 3.10 PEFT (`peft==0.8.2`)
Adapter support is HuggingFace-PEFT, which is torch-only. Visible at the protocol level via the `active_adapter` metadata field (`handler.py:490-494`).

### 3.11 `cuda_graphs.py`, ROCm-not-supported notes
A grep on the codebase (during the read) shows `petals/utils/cuda_graphs.py` and ROCm references — not in the four files reviewed but adjacent. CUDA graphs are an NVIDIA-only optimisation.

### 3.12 Activation shape contract is OK
`hidden_states.ndim == 3` and `outputs[0].shape == inputs.shape` (`inference_session.py:166-168`) are math invariants of transformer forward passes, not vendor assumptions. These are safe to keep.

---

## 4. Where vendor-agnostic abstractions could replace the assumptions

This section sketches the layers a Nakshatra refactor would introduce. The protocol structure (UIDs, spans, RPC method shapes, session state machine) does *not* need to change. What changes is the tensor type, the backend, and — most importantly — the choice to skip PyTorch entirely for the v0.1 path.

### 4.1 Abstract `Tensor` via DLPack — and why the wire path is cheap

**Replaces 3.1, 3.2, 3.3, 3.8.**

Adopt DLPack as the framework-neutral tensor handoff format. PyTorch, JAX, MLX, TensorFlow, NumPy, TVM, CuPy all implement DLPack import/export. Define:

```
class NTensor(Protocol):
    shape: tuple[int, ...]
    dtype: WireDType   # nakshatra-defined enum: F32, F16, BF16, I32, I64, ...
    def to_dlpack(self) -> PyCapsule: ...
    def to_bytes(self) -> bytes: ...
    @classmethod
    def from_bytes(cls, b, shape, dtype) -> "NTensor": ...
```

Every backend (llama.cpp, MLX, torch, etc.) provides an adapter that can produce/consume `NTensor`. The wire path (`serialize_tensor` / `deserialize_tensor`) speaks `NTensor`, not `torch.Tensor`. The receiving backend hydrates into its native tensor type. SafeTensors is useful prior art for the on-disk format; the wire format can borrow its (shape, dtype, raw bytes) framing.

#### Why per-token activation transport is cheap — the load-bearing insight

This is the architectural reason Nakshatra works at all, and it deserves explicit math.

For Llama-70B at fp16, `hidden_size = 8192`. The per-token activation tensor crossing one network hop is:

```
1 (batch) × 1 (token) × 8192 (hidden) × 2 bytes (fp16) = 16 KB
```

Across a typical 5-server pipeline, every generated token traverses 5 hops of activation transport: ~80 KB of bytes per token, end to end. At 5 tokens/second generation that's **400 KB/s** of network traffic. Trivial on any modern home network, trivial on Tailscale, trivial on a coffee-shop LAN.

Compare this to llama.cpp's current `rpc-server` mode, which streams the **full model weights** from master to each worker on every cluster startup — for Llama-70B at 4-bit that's roughly 26 GB per worker per restart. That cost is paid once per restart, but it is the dominant operational cost of running llama.cpp's RPC mode at scale. It is also the reason llama.cpp RPC clusters are slow to spin up and brittle to restart.

Nakshatra's architectural commitment is **weights stay local on each worker; only activations travel**. Workers load their assigned layers once at startup (from local disk or a one-time fetch) and never ship them across the network during inference. The per-token wire cost stays in the kilobyte range regardless of model size, and adding a worker means moving 16 KB/token through it, not 26 GB at startup.

This is what makes Nakshatra architecturally faster than llama.cpp RPC for steady-state inference, and it is the property that makes pipeline-parallel-across-vendors economically viable at all. Every other design decision in this document inherits this assumption.

### 4.2 Abstract `Backend` interface — LlamaCppBackend is the primary v0.1 path

**Replaces 3.6, 3.7.**

```
class Backend(Protocol):
    compute_dtype: WireDType
    def forward_block(self, block_id, hidden: NTensor) -> NTensor: ...
    def allocate_kv_cache(self, batch, max_len) -> CacheHandle: ...
    def step_with_cache(self, block_id, hidden, cache_handle, position) -> NTensor: ...
    def cache_bytes_per_token(self) -> int: ...
```

#### LlamaCppBackend is primary, not one of many

The original draft of this section listed several backends (Torch CUDA, Torch ROCm, Torch MPS, MLX, llama.cpp, ggml-Vulkan) as roughly co-equal options. That framing is wrong for v0.1.

**LlamaCppBackend is the primary backend.** llama.cpp already supports CUDA + ROCm + Metal + Vulkan + MoltenVK + CPU through one C++ codebase. By building Nakshatra v0.1 on top of llama.cpp, we sidestep — entirely — most of the welds catalogued in §3:

- **No bitsandbytes.** GGUF quantization is built into llama.cpp and works on every supported backend. The CUDA-only NF4 problem (§3.5) does not exist for us.
- **No PyTorch MPS dtype quirks.** Apple Silicon support is via Metal, exposed through llama.cpp's existing Metal backend. We do not pay the cost of bf16 being half-broken on MPS, of the Apple Silicon MPS backend being a moving target, or of waiting for PyTorch to fix dtype gaps per release.
- **No ROCm-on-macOS impossibility.** Intel Macs with AMD GPUs use Vulkan via MoltenVK, which llama.cpp already supports. PyTorch has no ROCm story on macOS at all and will not have one; that branch of the universe is closed.
- **Most of §3 stops applying.** The "torch is the universal currency" weld (§3.1, §3.2, §3.3, §3.4, §3.7, §3.8) is something Nakshatra inherits *only if it commits to PyTorch*. By committing to llama.cpp instead, the weld is irrelevant: there is no torch on the worker's hot path.

This is a strategic decision, not a limitation. Avoiding PyTorch in v0.1 is the choice. We are not building a PyTorch distributed runtime; we are building a llama.cpp distributed runtime, with PyTorch as a possible later backend for users whose research workflow needs it.

#### Backend roadmap

- **v0.1: LlamaCppBackend only.** Covers ~90% of consumer hardware (CUDA, ROCm Linux, Metal, Vulkan including MoltenVK on older Intel Macs, plain CPU). This is the entire backend story for v0.1.
- **v0.2: Pure-Vulkan backend.** For hardware where llama.cpp's Vulkan path is weak or where we want more direct kernel control — Intel Arc GPUs are the obvious case. Same NTensor / Backend interface, different compute path.
- **v0.3+: TorchCUDABackend, MLXBackend, etc.** Added only when a specific user need (research workflows, training, MLX-only models) justifies the maintenance cost. PyTorch backends come back in scope when there's a concrete reason to inherit the welds in §3.

#### What the LlamaCppBackend wrapper actually has to do

This is the load-bearing engineering work of v0.1. It is not a thin shim. It needs to:

- Expose llama.cpp's per-block forward in a way the Nakshatra worker can call. llama.cpp's existing RPC backend (tools/rpc/rpc-server.cpp, ggml/src/ggml-rpc.cpp) already implements per-layer dispatch at the GGML graph level — the open question is whether Nakshatra's LlamaCppBackend can reuse that machinery or must bypass it. This question is load-bearing for v0.1 scope and is escalated to §8 as an open question (see §8.7).
- Hold a partial GGUF (only the assigned layer range) in memory or mmap.
- Allocate and manage a KV cache slice corresponding to that layer range.
- Convert between llama.cpp's internal tensor types and `NTensor` at the boundary (DLPack export from llama.cpp is doable; this is also non-trivial C++ work).

§9 below addresses the engineering scope honestly.

### 4.3 Wire-dtype canonicalisation
**Replaces 3.4.**

Pick a small set of wire dtypes — `F32`, `F16`, `BF16`, `I32`, `I64`, plus a `QUANT8_BLOCKWISE` and `QUANT4_NF4` (or a generic `QUANTIZED { scheme_id, payload }`). Every backend must be able to consume those. Internal compute precision is the backend's choice.

The `hypo_ids.dtype == torch.int64` assertion becomes `hypo_ids.dtype == WireDType.I64`.

### 4.4 Quantization plug-ins
**Replaces 3.5.**

`QuantType` becomes an open registry, not an enum tied to bitsandbytes:
- GGUF Q4_K_M / Q5_0 / Q8_0 / etc. (llama.cpp — primary in v0.1).
- bitsandbytes NF4, INT8 (CUDA only; declared in TorchCUDABackend's capabilities, not v0.1).
- MLX 4-bit / 8-bit (Apple Silicon, not v0.1).
- Backend declares its supported quant schemes via `Backend.supported_quants()`.

The merge-pool branching at `block_functions.py:199` becomes a backend capability query: `if backend.short_inference_max_tokens(quant) > batch * len: merge_pools()`.

Activation transport is **always dequantized** to a wire dtype (F16 or BF16). Quantization is a within-backend storage decision; activations between backends are full-precision. See §8.3 for the open question of whether v0.5+ should support int8 activation transport.

### 4.5 Schema descriptor abstraction
**Replaces 3.3.**

Replace `hivemind.BatchTensorDescriptor` with a `nakshatra.TensorDescriptor`:

```
@dataclass
class TensorDescriptor:
    shape: tuple[int | None, ...]   # None = batch dimension
    dtype: WireDType
    compression: int                # nakshatra compression code
```

`outputs_schema`, `args_schema`, `kwargs_schema` on the backend become this neutral type.

### 4.6 KV cache as opaque buffer + descriptor
**Replaces 3.7.**

`MemoryCache` allocates byte buffers with a `(shape, dtype, layout)` descriptor. Backends own the layout; the cache layer owns the lifecycle and admission control. `cache_bytes_per_token` is computed by the backend.

### 4.7 Adapter (PEFT) abstraction — defer
**3.10.** PEFT/LoRA support is fundamentally per-backend and torch-coupled. We will defer adapter support to v1.0+. The `active_adapter` metadata field can remain reserved in the protocol; v0.1 backends reject non-empty adapter names.

### 4.8 Tensor parallel — defer; constrain to single-vendor permanently
**3.9.** Cross-vendor tensor parallel (one block sharded across NVIDIA + Apple Silicon at once) is genuinely hard — collective ops have no vendor-neutral runtime. Nakshatra's permanent stance: **pipeline parallel across vendors, tensor parallel only within a single vendor**. A future `TorchCUDABackend` can use `tensor_parallel` internally; an `MLXBackend` cannot, and won't. v0.1 ships with no tensor parallelism at all.

### 4.9 What stays untouched (conceptually)

The **design ideas** in Petals' protocol are vendor-agnostic and should be retained:

- The notion of `ModuleUID` and contiguous layer spans.
- The shape of the RPC surface — forward, backward, stateful inference session, info probe.
- Step-id idempotency for any future server-to-server push.
- `prefix_length` accounting and `start_from_position` cache truncation.
- Protobuf message envelopes and MSGPack metadata.

What is **not** retained for v0.1 is the specific transport implementation — Hivemind DHT, libp2p, dynamic re-routing on failure. Those are addressed in §6.

---

## 5. Trust model and verification

**v0.1 assumes trusted workers.** This is an explicit, load-bearing assumption.

The design target for v0.1 is a private network of known operators: a research group's lab cluster, a small consortium of mutually-known parties, a self-hosted deployment by a single org. In that setting, workers do not need to be verified — they are operated by people who already trust each other, on a private network (Tailscale, VPN, LAN). A worker that returns garbage activations is debugged as an operator error, not modelled as an adversary.

This assumption simplifies a great deal. We do not need to verify computation, sign activations, or police peer behaviour in v0.1.

For later versions, when Nakshatra is integrated with Prithvi or otherwise exposed to mutually-untrusting workers, the relevant techniques and their realistic costs:

- **Redundant compute.** Run the same layer span on N workers; compare outputs bytewise (or within a numerical tolerance band). Cheap to implement, costs N× compute per verified token. Practical for spot-checking; expensive as a default. This works *because* transformer activations are deterministic for the same input weights and dtype settings — bitwise reproducibility is feasible if all replicas use the same kernel and dtype, and numerical reproducibility within ε is feasible in general. This determinism is the property that makes redundant-compute verification cheap relative to the alternatives below.
- **Spot-check verification.** Sample some fraction (e.g. 1%) of generation steps and re-run them on trusted infrastructure. Catches systematic cheating with low overhead; misses single-token attacks. Pairs well with reputation slashing.
- **Reputation-based slashing.** Workers stake collateral; bad outputs (caught by spot-check or redundant compute) trigger forfeiture. This is a Prithvi-layer concern, not a Nakshatra concern. When integrated with Prithvi, Nakshatra exposes the verification hooks and Prithvi handles the economics.
- **TEE-based attestation.** Run worker compute inside a trusted execution environment (NVIDIA Confidential Compute on H100, AMD SEV-SNP / MI300 attestation). Strong guarantee, but **only available on H100-class NVIDIA and MI300-class AMD** — not on consumer hardware, not on Apple Silicon, not on any of the AMD GPUs Nakshatra targets in v0.1-v0.3. So TEEs are not a viable verification primitive for the target hardware tier.
- **zkML.** Generate zero-knowledge proofs of correct inference. Currently 1000–10000× slowdown for transformer inference; not practical today. Watch the research; do not block on it.

The realistic v1.0+ verification stack is **redundant compute + spot-check + Prithvi reputation slashing**, in that order, with TEEs as an opt-in for operators with H100/MI300 hardware. v0.1 implements none of this.

---

## 6. Things Nakshatra explicitly DEFERS from Petals

Petals contains a great deal of machinery that Nakshatra does not need in v0.1 and will explicitly skip. Each item below is a feature we are deliberately not inheriting, with a brief note on when (if ever) it returns.

### 6.1 Hivemind DHT for peer discovery
**Defer.** v0.1 uses static configuration: a YAML file listing each worker's address, port, and the layer range it owns. See §8.1 for the schema question.

Returns in **v1.0+** if and when Nakshatra needs to support a public network of mutually-unknown workers. In a private-cluster setting the DHT buys nothing and costs significant operational complexity (bootstrap nodes, peer churn, NAT punching). Static config is correct for the v0.1 trust model (§5).

### 6.2 libp2p P2P transport
**Defer.** v0.1 uses plain gRPC over Tailscale or LAN. Tailscale handles NAT traversal and peer authentication out of band; the application protocol does not need to.

Returns in **v1.0+** alongside the DHT, if/when public-network operation becomes a goal. Until then, Tailscale is a strictly better operator experience than libp2p for a small private cluster.

### 6.3 Dynamic chain re-routing on worker failure
**Defer.** Petals' `_update_sequence` and history-replay logic (`inference_session.py:364-391`, `inference_session.py:113-117`) is sophisticated but premised on a public network where workers come and go constantly. v0.1 simply fails the request with a clear error if any worker in the chain dies mid-inference.

Returns in **v0.5** as an opt-in feature for operators willing to pay the complexity cost. The Petals approach (replay client-side history through a replacement worker to rebuild KV cache) is the right design; it's just not justified in a known-operator cluster.

Note that v0.1 still implements client-side full-request retry on failure (§8.2). What v0.5 adds is in-flight KV cache preservation, not the existence of any failure handling.

### 6.4 PEFT / LoRA adapter support
**Defer.** Per §4.7 — not on the v0.1 critical path; v0.1 backends reject non-empty `active_adapter` metadata.

Returns in **v1.0+** when there is a concrete user need and a backend that supports adapters. Likely arrives alongside TorchCUDABackend (§4.2).

### 6.5 Cross-vendor tensor parallelism
**Defer permanently.** Per §4.8 — pipeline-only across vendors, tensor-parallel only within a single vendor. This is not a v0.1 limitation; it is a permanent design constraint driven by the absence of a vendor-neutral collective-ops runtime.

Within-vendor tensor parallel returns in **v0.3+** if and when a backend (e.g. TorchCUDABackend with NCCL) wants to support it internally.

### 6.6 Server-to-server push (`rpc_push`)
**Defer.** Petals' direct server-to-server activation push (`handler.py:310-318`, `handler.py:320-350`, `inference_session.py:174-182`) is a useful latency optimization — it removes a round-trip through the client per pipeline hop. But it adds protocol surface area (the `next_servers` chain, the `pushed=true` dedup, the swallowed-failure semantics) that is not justified for v0.1's trust model and small chains.

Returns in **v0.5**, after dynamic re-routing (§6.3) lands. The two features compose naturally and should ship together.

---

## 7. MVP slice on actual available hardware

The original draft of this section assumed access to an Apple Silicon machine. We do not have one. The hardware actually available is:

- **4 × Intel iMac with AMD Radeon Pro 5700 XT** (lab cluster). Run llama.cpp with Vulkan via MoltenVK; this configuration is proven working.
- **1 × Intel iMac Pro with AMD Vega 56 8GB** (lab). Same Vulkan-via-MoltenVK path.
- **1 × Linux desktop with AMD Radeon RX 9070 XT** (home). ROCm available.
- **Possibly: rented cloud NVIDIA GPU** (1× H100 or A100 hour-leased) for cross-vendor validation.

The MVP gradient is structured so that each step changes exactly one variable from the previous step, so when something breaks we can attribute the breakage:

### v0.1 — Two iMacs, same OS, same vendor, same backend

Llama-2-7B (or smaller, depending on 5700 XT memory budget after KV cache) split across two of the lab iMacs. Both use LlamaCppBackend. Both run macOS. Both use the Vulkan-via-MoltenVK driver path. Single token end to end through a two-hop pipeline.

This proves the protocol design and the worker orchestration without crossing OS or vendor boundaries. If this works, the protocol and the LlamaCppBackend wrapper are both correct.

**v0.1 success criterion (falsifiable):** Given a fixed prompt ("The capital of France is") and a fixed sampling configuration (greedy decoding, no temperature, no top-k/top-p sampling), Nakshatra running in two-iMac mode must produce the same first generated token as a reference single-machine llama.cpp run with the same GGUF, the same sampling configuration, and the same seed. Bitwise-equal hidden states are not required (parallelization can introduce numerical differences within ε), but the top-1 next-token prediction MUST match. If the distributed run picks a different first token than the single-machine reference, the test fails and Nakshatra has a correctness bug.

This criterion replaces "single token end to end" as the v0.1 milestone. Subsequent versions add their own falsifiable criteria (v0.2: same as v0.1 but with a Linux ROCm worker in the chain; v0.3: same with a CUDA worker added).

### v0.2 — Add the home AMD 9070 XT

Add the Linux + ROCm machine as a third worker. Same LlamaCppBackend codebase, different driver path (ROCm instead of Vulkan/MoltenVK), different OS (Linux instead of macOS).

This proves the codebase is genuinely OS-portable and that the LlamaCppBackend abstracts the driver layer correctly. If this works without code changes (only build changes), the abstraction is sound.

### v0.3 — Add a rented NVIDIA cloud GPU

Add a rented H100 or A100 instance running CUDA. Still LlamaCppBackend (llama.cpp's CUDA path), but a fundamentally different vendor's hardware.

**This is the moment Nakshatra proves its thesis.** A single inference request flowing through Apple iMacs (Vulkan/MoltenVK) → Linux AMD (ROCm) → cloud NVIDIA (CUDA), with one token coming back, demonstrates that the architecture delivers vendor-agnostic distributed inference. Every other claim in this document is downstream of that demo working.

After v0.3, scale tests (more workers, larger model) and then the v0.5 work (server-to-server push, dynamic re-routing) become meaningful.

---

## 8. Open questions for v0.1 design

These are decisions that need to be made before code is written, that this document does not yet answer.

### 8.1 Static config file format

Without a DHT (§6.1), v0.1 needs a config format that lists workers and their assigned layer ranges. A YAML schema along these lines is the obvious starting point:

```yaml
model: llama-2-7b
hidden_size: 4096
num_blocks: 32
workers:
  - id: imac-lab-01
    address: 100.64.0.11
    port: 5555
    blocks: [0, 16]      # half-open interval [start, end)
    backend: llamacpp
    quant: Q4_K_M
  - id: imac-lab-02
    address: 100.64.0.12
    port: 5555
    blocks: [16, 32]
    backend: llamacpp
    quant: Q4_K_M
```

**v0.1 default: this YAML format, validated on client startup.** Open questions: how to express replication (multiple workers covering the same range), how to express worker health-check endpoints, whether to allow per-worker dtype/quant overrides. Future versions revisit when DHT-based discovery (§6.1) returns.

### 8.2 Failure handling mid-chain

If a worker dies mid-inference, do we attempt recovery (Petals-style, §6.3) or fail the request?

**v0.1 default: fail the in-flight request with a clear error message identifying the dead worker, but the client transparently retries the entire request (from token 0) up to N times (default N=3, configurable) before surfacing the error to the user.**

This is NOT in-flight recovery — KV cache on the dead worker is lost, the conversation prefix is reprocessed from scratch on the retry. But from the user's perspective transient single-worker failures are masked, and only persistent failures (worker stays down across all retry attempts) reach the application layer.

Implementing this in v0.1 requires nothing beyond what §6.3 already defers: client gets an error, client tries again. No protocol changes, no in-flight state recovery, no Petals-style sequence replay. The full Petals re-routing work (which preserves in-flight KV cache by replaying client-side history through a replacement worker) remains deferred to v0.5 per §6.3.

### 8.3 Quantization at the activation boundary

Internal weight quantization is a backend choice (Q4_K_M, Q5_0, etc.). The open question is whether **activations on the wire** are quantized.

**v0.1 default: activations transport in fp16, always.** This costs 16 KB/token at hidden_size=8192 (§4.1), which is trivial. Quantizing activations to int8 saves bandwidth at the cost of accuracy and protocol complexity, and is not justified at v0.1 traffic levels.

The door stays open for **v0.5+ int8 activation transport** if and when bandwidth becomes a real constraint — most likely for very-high-throughput batched inference, not single-stream generation.

### 8.4 Model file management

How does each worker get its slice of the model?

**v0.1 default: operators pre-stage model files manually.** Workers do not fetch model files from a central registry. The config file documents which GGUF file each worker needs and the operator copies it to the right machine. We document the procedure clearly and keep it boring.

A future version (v1.0+) revisits this with a content-addressed model registry if/when public-network operation arrives. In a private cluster, manual pre-staging is correct.

### 8.5 KV cache invalidation on worker restart

If worker B restarts mid-conversation, the in-progress KV cache on B is lost. The conversation prefix processed through B is no longer cached on B, even though the client thinks it is.

**v0.1 default: the client must restart the conversation. The failure mode is documented explicitly in the user-facing error.** We do not silently recover; we surface the loss.

A future version (v0.5+, paired with §6.3) revisits this. The Petals approach — replay client-side history through any replacement worker to rebuild the KV cache — is the right design. v0.1 does not implement it.

### 8.6 Worker capability handshake

When a client connects to a worker, the two need to verify they are compatible: same protocol version, compatible model file, matching wire dtype, the layer range each worker actually serves, current KV cache headroom. Petals' `rpc_info` (§2.3) is the prior art.

**v0.1 default: a single `info` RPC returning a structured capability blob (JSON or protobuf), called once per worker on session initialization.** The blob includes:

- Nakshatra protocol version
- Backend type (e.g. "llamacpp-vulkan", "llamacpp-rocm", "llamacpp-cuda")
- Wire dtypes the backend supports (subset of WireDType)
- Quant schemes the backend supports (e.g. "Q4_K_M", "Q5_0")
- Layer range owned [start, end)
- Model identifier and content hash (so client can verify all workers are running the same model build)
- Current KV cache headroom in tokens

Mismatches fail loudly at session start, with clear error messages. Sketch wire shape now; refine during implementation.

### 8.7 Reuse llama.cpp's existing RPC backend, or bypass it? — RESOLVED

**Resolution: neither Path A nor pure Path B. Adopt Path B-prime.** Full reasoning in [`path-a-vs-path-b-memo.md`](path-a-vs-path-b-memo.md).

The naive Path A is infeasible. llama.cpp's existing RPC (`tools/rpc/rpc-server.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp`) is a *remote GGML compute backend*, not a *distributed model server* — its commands are GGML primitives (`RPC_CMD_ALLOC_BUFFER`, `RPC_CMD_SET_TENSOR`, `RPC_CMD_GRAPH_COMPUTE`, etc.; `ggml-rpc.cpp:91-111`), the master holds the full model, and the worker is structurally a dumb GPU-over-TCP that runs whatever ggml graph the master sends. There is no partial-model load on the worker, no per-layer-forward command, no notion of an inference session. Wrapping `rpc-server` would require Nakshatra's worker to assemble per-layer ggml subgraphs in C++ itself, which is more work than just bypassing the RPC layer.

Pure Path B (drop all the way to GGML graph APIs) is rejected as too expensive — it throws away llama.cpp's per-architecture graph builders.

**Path B-prime**, recommended: each Nakshatra worker links llama.cpp as a library, uses its existing graph builders, but (a) loads only its assigned layer range from a pre-split sub-GGUF and (b) exposes a patched `llama_decode_layers(hidden_in, layer_start, layer_end) → hidden_out` entry point. The Nakshatra inter-worker protocol (gRPC over Tailscale, §6.2) sits above this. Estimated C++ effort: **10–14 weeks** with prior llama.cpp experience, **15–28 weeks** for an engineer new to llama.cpp internals (memo §5.4). Dominant risk is rebasing the patched `llama_decode` against fast-moving upstream — committed monthly rebase cadence, ~0.5 engineer-day per scheduled rebase plus 2–5 days for occasional conflicts (memo §5.5). Long-term play is to upstream a clean `llama_decode_layers` API; v0.1 ships on the local patch.

**Pre-implementation validation plan.** Before C++ work begins, two experiments establish whether Path B-prime is feasible:

1. **Partial-GGUF load test** (½ day): produce a sub-GGUF with `gguf-py/` and attempt to load it with `llama-cli`. Confirms or revises the "no partial load" claim that justifies Path B-prime. See memo §1.6.
2. **v0.0 cb_eval spike** (1–2 weekends): two workers each loading the full 7B model, activations flowing between them via `cb_eval` callbacks. Validates the protocol and orchestration without paying the v0.1 C++ cost upfront. See memo §5.7.

Both should be completed before the v0.1 timeline (§9) starts running.

---

## 9. Realistic scope and timeline

**Realistic time budget**, assuming one full-time-equivalent engineer:

- **4–8 weeks**: design finalization, protocol spec (gRPC `.proto` files, exact field semantics), service skeleton in Python or Go.
- **8–16 weeks**: LlamaCppBackend C++ wrapper. Biggest variance: §8.7's Path A vs Path B decision. Path A (reuse llama.cpp's existing RPC machinery) is closer to 8 weeks. Path B (bypass and reimplement) is 16+ weeks and may push toward 24.
- **4–8 weeks**: end-to-end integration, debugging the v0.1 → v0.2 → v0.3 gradient, operational tooling, observability.

**Total at full-time effort: 16–32 weeks.**

**At half-time effort, double everything: 32–64 weeks (8–16 months).**

**At quarter-time (evenings and weekends only), triple it: 48–96 weeks (12–24 months).** This is the most likely real scenario for a single developer with other commitments. The architecture doc is the cheap easy part; what determines whether Nakshatra ships is sustained implementation effort over a multi-quarter window.

**Without a committed engineer — the user themselves or a co-implementing collaborator — this project will join the long list of decentralized-AI projects that produced compelling architecture documents and no working code.** The architecture doc is the cheap part. The implementation is the work.

That said: the architectural commitments here are well-scoped. LlamaCppBackend as primary (§4.2) collapses most of the cross-vendor complexity into a problem llama.cpp has already solved. The activation-transport math (§4.1) confirms the design is bandwidth-feasible on home hardware. The trust model (§5) and deferred features (§6) make v0.1 tractable. The MVP gradient (§7) is incremental enough to debug. The remaining work is to do the work.

---

## A.1 Operational addendum: lab-cluster optimization surfaced during research

While reading llama.cpp's RPC source for the §8.7 memo, we discovered that `rpc-server` supports a `--cache` flag (`tools/rpc/rpc-server.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp:1240–1314`) that enables FNV-1a content-addressable caching of weight blobs ≥10 MiB on workers' local disks. The first cluster startup pays the full weight-streaming cost (the 9-minute number observed during the lab cluster's 70B commissioning). Subsequent startups with the same model send only hashes; cached blobs are loaded from disk.

This is **not a Nakshatra feature** — it's an upstream llama.cpp capability we didn't know we were missing. Adding `--cache /path/to/cache` to the lab cluster's rpc-server invocations should meaningfully reduce warm-cache cluster startup time without any Nakshatra code.

Relevance to Nakshatra: confirms that llama.cpp's RPC, despite being the wrong abstraction for Nakshatra, has more sophisticated machinery than the surface appears. Worth re-reading periodically; future llama.cpp versions may add capabilities (e.g. partial-model load) that change Nakshatra's design space.

---

## 10. Appendix: file map for future edits

| Concern | File | Anchor |
|---|---|---|
| Block-level forward over a span | `src/petals/server/block_functions.py` | `run_rpc_forward`, lines 32-81 |
| Streaming inference state machine | `src/petals/server/block_functions.py` | `iterate_rpc_inference`, lines 144-237 |
| Server RPC surface | `src/petals/server/handler.py` | `TransformerConnectionHandler`, lines 55-592 |
| Server-to-server push | `src/petals/server/handler.py` | `_push_outputs`, `rpc_push`, lines 310-350 |
| KV cache allocation | `src/petals/server/handler.py` | `_allocate_cache`, lines 532-547 |
| Client per-server session | `src/petals/client/inference_session.py` | `_ServerInferenceSession`, lines 26-217 |
| Client-side chain orchestration | `src/petals/client/inference_session.py` | `InferenceSession.step`, lines 284-362 |
| Failure-recovery re-routing | `src/petals/client/inference_session.py` | `_update_sequence`, lines 364-391 |
| Model registration plug-in | `src/petals/utils/auto_config.py` | `register_model_classes`, lines 22-29 |
| Auto-dispatch by `model_type` | `src/petals/utils/auto_config.py` | `_AutoDistributedBase`, lines 32-52 |
