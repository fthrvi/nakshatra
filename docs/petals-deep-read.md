# Petals Deep Read

A line-by-line dissection of four core files, scoped to what Nakshatra needs to know to abstract the framework off CUDA/PyTorch:

- `src/petals/server/block_functions.py`
- `src/petals/server/handler.py`
- `src/petals/client/inference_session.py`
- `src/petals/utils/auto_config.py`

Citations are `path:line` against the upstream tree under `nakshatra/src/petals/`. Anything that demonstrably lives outside these four files is flagged as out-of-scope rather than guessed at.

---

## 1. How Petals splits a model across workers

### 1.1 Where assignment is decided

**The four files do not contain the assignment algorithm.** They contain the *consumers* of an assignment that is computed elsewhere.

What the files actually show:

- `auto_config.py` is a model-class registry, not an assigner. `_CLASS_MAPPING = {}` is populated by per-architecture subpackages calling `register_model_classes(...)` (`auto_config.py:22`, `auto_config.py:25-29`). `AutoDistributedModelForCausalLM.from_pretrained` (`auto_config.py:90-91`, dispatched via `_AutoDistributedBase.from_pretrained` at `auto_config.py:36-52`) just looks up the right HF subclass for a given `config.model_type` and delegates. So Petals reuses the HF block decomposition; it never introspects the model itself.

- The unit of distribution is one `ModuleUID` (`handler.py:32`, imported from `petals.data_structures`). The server-side mapping is `module_backends: Dict[ModuleUID, TransformerBackend]` (`handler.py:58`). Each `TransformerBackend` instance corresponds to one block's worth of weights and pools. The handler treats this dict as ground truth — a request whose UID isn't in the dict is rejected at `handler.py:528-529` (`raise RuntimeError(f"Remote peer does not serve {uid}")`).

- The client side enters server sessions over **spans** (`RemoteSpanInfo`, imported at `inference_session.py:18`). A span is the `(peer_id, start, end)` triple the client uses to address a contiguous range of blocks on one server: `inference_session.py:253` joins UIDs with `CHAIN_DELIMITER`:
  ```python
  span_uids = CHAIN_DELIMITER.join(self._sequence_manager.block_uids[span.start : span.end])
  ```

- The client *requests* a path each time it needs one — for the initial path on the first step (`inference_session.py:329-330`) and after any failure (same lines, `attempt_no >= 1`) — by calling:
  ```python
  updated_spans = self._sequence_manager.make_sequence(
      block_idx, update_end, mode="min_latency", cache_tokens_needed=self._max_length
  )
  ```
  (`inference_session.py:376-378`). The mode `"min_latency"` is the only mode used in this file; other modes presumably exist in `RemoteSequenceManager` (out of scope here).

- Total block count comes from `len(self._sequence_manager)` (`inference_session.py:236`), so the client trusts the sequence manager — not the local model — for "how many blocks does this model have."

### 1.2 What the algorithm therefore is, from these files

From the four files alone you can reconstruct:

1. Servers register a contiguous slice of blocks they're willing to host as `module_backends` (mechanism for choosing that slice is out of scope).
2. Servers advertise those UIDs over a DHT (the DHT itself is referenced, e.g. `dht: DHT` at `handler.py:62` and the `dht_client_mode` flag returned by `rpc_info` at `handler.py:581`, but not exercised in these files).
3. The client's `RemoteSequenceManager` (referenced at `inference_session.py:17`, body not in these files) takes `(block_idx, update_end, mode="min_latency", cache_tokens_needed)` and returns a list of spans tiling that range. The semantics — greedy min-latency tiling, replication, load balancing — live in `petals.client.routing`, **not in any file read here**.

If Nakshatra wants to retarget the assignment algorithm, the lever is `RemoteSequenceManager.make_sequence` and the DHT advertising on the server side, neither of which is in these four files.

---

## 2. The complete wire protocol

### 2.1 Transport

All RPCs are hivemind P2P streams over libp2p. The protobuf types are `runtime_pb2.ExpertRequest` and `runtime_pb2.ExpertResponse` (imported at `handler.py:26`, `inference_session.py:13`, `block_functions.py:11`). Tensors travel inside as `runtime_pb2.Tensor` after `serialize_torch_tensor` / before `deserialize_torch_tensor` (hivemind's compression-aware codec, `block_functions.py:9`). Metadata is a `bytes` field carrying a `MSGPackSerializer.dumps(dict)` payload (`handler.py:123`, `handler.py:148`, `inference_session.py:161`).

`runtime_pb2.ExpertRequest` carries three fields the code touches: `uid` (str), `tensors` (repeated Tensor), and `metadata` (bytes). The `uid` field for multi-block requests is a `CHAIN_DELIMITER`-joined string of `ModuleUID`s (`handler.py:524`, `_check_uids` splits on `CHAIN_DELIMITER`).

Streaming chunk size is `DEFAULT_MAX_MSG_SIZE` from `hivemind.p2p.p2p_daemon` (`handler.py:25`); large tensors get split with `split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)` (`handler.py:408`, `handler.py:487`).

### 2.2 Client → worker RPCs (defined on `TransformerConnectionHandler`)

| Method | Pattern | Defined at | Purpose |
|---|---|---|---|
| `rpc_inference` | bidi stream | `handler.py:132-195` | Multi-step incremental inference with KV cache |
| `rpc_forward` | unary | `handler.py:352-378` | One-shot forward through a chain of blocks |
| `rpc_forward_stream` | server stream | `handler.py:380-409` | Same, output split for size |
| `rpc_backward` | unary | `handler.py:434-459` | Backward pass through a chain |
| `rpc_backward_stream` | server stream | `handler.py:461-488` | Same, output split for size |
| `rpc_push` | unary | `handler.py:310-318` | Worker-to-worker activation forwarding |
| `rpc_info` | unary | `handler.py:575-592` | Health/version/cache-availability probe |

#### rpc_inference (the central one)

Wire shape on the client side (`inference_session._ServerInferenceSession.create`, `inference_session.py:59-76`):

```python
stub = TransformerConnectionHandler.get_stub(p2p, span.peer_id)
inputs_queue = asyncio.Queue()
outputs_stream = await asyncio.wait_for(
    stub.rpc_inference(cls._read_inputs_from_queue(inputs_queue)),
    config.connect_timeout,
)
```

So the client opens a single bidirectional stream per `(peer, span)`. The session id is a UUID4 generated client-side (`inference_session.py:50`).

**First request from client** (`inference_session.py:155-162`):
- `uid`: `CHAIN_DELIMITER.join(...)` of the span's UIDs
- `tensors`: `[hidden_states, prompts, hypo_ids]` produced by `pack_args_kwargs(inputs, prompts, hypo_ids)` (`inference_session.py:129`), serialized with `serialize_torch_tensor(tensor.to(proto.dtype), proto.compression)` against the per-server `inference_schema`
- `metadata` keys (`inference_session.py:131-141`, decoded server-side at `handler.py:148-154`):
  - `session_id` (UUID4)
  - `step_id` (UUID4 per step, used for dedup of pushed-vs-direct)
  - `max_length` (int, KV cache budget — required, asserted at `handler.py:158-159`)
  - `points` (number, used for prioritization)
  - `alloc_timeout` (float)
  - `args_structure` (the structure produced by `pack_args_kwargs`)
  - `start_from_position` (int, optional, lets a session rewind — see `block_functions.py:163-168` and `inference_session.py:91-95`)
  - `next_servers` (optional list of `(peer_id_b58, session_id, start, end)` for server-to-server push, `inference_session.py:137-139`)
  - `active_adapter` (str, looked up by `_get_active_adapter` at `handler.py:490-494`)

**Subsequent requests on the same stream** carry the same shape but the metadata only re-sends the per-step bits (`session_id`, `step_id`, optional `start_from_position`/`next_servers`/`pushed`); the bulk session metadata is sent only when `not self.stepped` (`inference_session.py:132-133`).

**End of session**: client puts an empty `runtime_pb2.ExpertRequest()` into the queue (`inference_session.py:203`); server breaks out of `_read_inputs_from_queue` when it sees `not next_input_message.uid and not next_input_message.tensors` (`inference_session.py:83-84`) and out of `_iterate_inference_steps` when `request.tensors` is falsy (`handler.py:261`).

**Per-step server-side timeout**: every step must arrive within `step_timeout`; the very first request has its own `step_timeout` wait (`handler.py:140-143`); subsequent steps wait via `asyncio.wait([anext_task, get_push_task], timeout=self.step_timeout, ...)` (`handler.py:291-293`). The whole session is wrapped in `async with timeout(self.session_timeout)` at `handler.py:138`.

**Output schema**: each yielded `ExpertResponse` carries one tensor — the new hidden state — serialized with `requested_backends[-1].outputs_schema` (`block_functions.py:228-232`). Compression is taken from the schema unless the client overrides via metadata `output_compression` (`handler.py:421-427`).

#### rpc_forward / rpc_forward_stream

Unary or server-streamed; metadata contains `points`, `args_structure`, `active_adapter`, optional `output_compression`. Tensors deserialized at `handler.py:355` / via `_gather_inputs` at `handler.py:109-130` (which validates that all chunks share one `block_uid`, `handler.py:117-120`). Computation goes through `run_rpc_forward` (`block_functions.py:32-81`).

#### rpc_backward / rpc_backward_stream

Same shape as forward, but `flat_tensors` is `(inputs, grad_outputs, prompts, ...)` and the response is `[grad_inputs]` or `[grad_inputs, grad_prompts]` (`block_functions.py:140-141`). Schema for grads is built ad-hoc from `args_schema * len(grads)` at `handler.py:504-507` — there's an explicit `# TODO generalize` admitting this is duct tape.

#### rpc_info

Returns a msgpack-serialized dict with at least `version`, `dht_client_mode`, and `cache_tokens_available` (`handler.py:578-583`):

```python
result = {
    "version": petals.__version__,
    "dht_client_mode": self.dht.client_mode,
    CACHE_TOKENS_AVAILABLE: backend.memory_cache.bytes_left // max(backend.cache_bytes_per_token.values()),
}
```

If a specific UID is requested, the per-block `get_info()` is merged in (`handler.py:586-590`), with a hard guard against key collisions.

### 2.3 Worker → worker: `rpc_push`

The "next servers" handoff exists so a server can forward activations directly to the next server in the chain instead of round-tripping through the client.

Initiating side (`handler.py:320-350`, `_push_outputs`):

```python
next_peer_id, next_session_id, next_start, next_end = next_servers[0]
next_peer_id = PeerID.from_base58(next_peer_id)
next_uid = CHAIN_DELIMITER.join(f"{self.dht_prefix}{UID_DELIMITER}{i}" for i in range(next_start, next_end))

next_tensors = [serialized_outputs] + request.tensors[1:]
next_metadata = metadata.copy()
next_metadata.update(session_id=next_session_id, next_servers=next_servers[1:], pushed=True)

stub = self.get_stub(self._p2p, next_peer_id)
await stub.rpc_push(
    runtime_pb2.ExpertRequest(uid=next_uid, tensors=next_tensors, metadata=...),
    timeout=self.request_timeout,
)
```

Two things to note:
1. `next_servers[1:]` peels off the first hop — the chain is sent as a popped list so the receiver knows its own next hop.
2. The first tensor (the hidden state output of *this* server) is reused with its already-correct serialization to avoid a re-encode (`handler.py:332-333`).

Receiving side (`handler.py:310-318`):

```python
async def rpc_push(self, request, context):
    requested_uids = self._check_uids(request.uid)
    metadata = MSGPackSerializer.loads(request.metadata)
    session_id = metadata["session_id"]
    self._put_into_session_queue(session_id, request)
    return runtime_pb2.ExpertResponse()
```

The pushed activation is dropped into the receiving session's input queue; from inside `_iterate_inference_steps` it's indistinguishable from a client-sent step, except for the `pushed=True` metadata flag (`handler.py:265-268`). The receiving side uses `step_id` to dedup against any duplicate that arrives from the client (`handler.py:270-281`); if the push lost the race, it logs `arrived late X% of the time`.

The push path is gated by two conditions:
- Client side: `config.use_server_to_server` and a populated `next_session` chain (`inference_session.py:136-139`, `inference_session.py:174-182`).
- Server side: `can_push = not has_prompts` (`block_functions.py:233`) — deep prompts disable push, presumably because the next server doesn't have the prompt tensor.

### 2.4 Tensor packing / args_structure

The `args_structure` field is the one piece of "schema" the wire protocol carries explicitly:

- Client packs with `pack_args_kwargs(inputs, prompts, hypo_ids)` → `(input_tensors, args_structure)` (`inference_session.py:129`, `inference_session.py:21`).
- Server unpacks with `unpack_args_kwargs(flat_tensors, args_structure)` (`block_functions.py:50, 94, 173`).
- The server-side comments (`block_functions.py:49-50, 92-94, 172-173`) note `kwargs` is currently dropped (`# TODO: kwargs currently is unused, it can be used later for peft-like adaptation`).

---

## 3. CUDA-specific or PyTorch-specific assumptions

These four files do not contain literal `cuda` calls, but they are densely PyTorch-coupled. Anything outside of CPU PyTorch tensor semantics will need an abstraction layer.

### 3.1 Hard PyTorch dependencies

- `import torch` at `block_functions.py:8`, `handler.py:11`, `inference_session.py:9`. Type annotations (`*flat_tensors: torch.Tensor`, `block_functions.py:33`; `inputs: torch.Tensor`, `inference_session.py:99`) are load-bearing — the call sites assume tensor methods.
- `hivemind.compression.serialization.{deserialize,serialize}_torch_tensor` (`block_functions.py:9`, `handler.py:18,22`). This is the wire codec; it produces `torch.Tensor` and consumes `torch.Tensor`. Replacing PyTorch means replacing or wrapping this codec.
- `hivemind.utils.tensor_descr.BatchTensorDescriptor` (`inference_session.py:14`), used at `inference_session.py:146` (`BatchTensorDescriptor.from_tensor(arg, compression)`) — torch-specific descriptor extraction.

### 3.2 Tensor-shape and dtype assertions baked into the protocol

- `assert hidden_states.ndim == 3` — `block_functions.py:56`, `block_functions.py:79`, `block_functions.py:212`, `handler.py:418`.
- `assert hypo_ids.dtype == torch.int64` — `block_functions.py:180`, `inference_session.py:307`.
- `inputs.ndim == 3` for backward — `block_functions.py:110`.
- Output activation must match input shape exactly — `inference_session.py:166-168`:
  ```python
  assert outputs[0].shape == inputs.shape, ...
  ```

### 3.3 Tensor ops the server runs on every step

`block_functions.py`:
- `hidden_states.to(dtype)` — line 55, line 179. Cast to backend dtype.
- `prompts.to(...).split(1, dim=0)` and `.squeeze(0)` — line 60, line 104, line 187.
- `hidden_states[:, : prompt.shape[1]] += prompt` — line 65, line 112, line 123. Mutating tensor slice add.
- `result.to(proto.dtype)` before serialize — line 230, `handler.py:430`, `handler.py:518`.
- `torch.cat(grad_prompts_reversed[::-1], dim=0)` — line 140.
- `grad_outputs[:, : prompt.shape[1]].unsqueeze(0)` — line 138.

All of these assume PyTorch tensor semantics (autograd-aware, in-place ops, `dim` kwarg, dtype objects).

### 3.4 Device handling on the client

Client moves everything to CPU before serialization (`inference_session.py:309-313`):
```python
inputs_device = inputs.device
inputs_dtype = inputs.dtype
inputs = inputs.cpu()
prompts = prompts.cpu()
hypo_ids = hypo_ids.cpu()
```
and casts back at the end (`inference_session.py:361`):
```python
outputs = outputs.to(device=inputs_device, dtype=inputs_dtype)
```
This is generic-PyTorch (works for `cuda`, `mps`, `cpu`), but it presumes a `torch.device` API. There is no explicit `cuda` reference; an MPS or ROCm tensor on the client would round-trip cleanly through this code.

A grad-context check at `inference_session.py:291-292` warns if `torch.is_grad_enabled()` — autograd-only, but only as a warning.

### 3.5 Quantization / NF4 (the one place CUDA leaks in)

`block_functions.py:19` imports `QuantType` from `petals.utils.convert_block`. The only branch in these files keyed on it is `block_functions.py:199`:

```python
merge_max_tokens = MAX_NF4_SHORT_INFERENCE_TOKENS if quant_type == QuantType.NF4 else MAX_SHORT_INFERENCE_TOKENS
```

with constants `MAX_SHORT_INFERENCE_TOKENS = 128` and `MAX_NF4_SHORT_INFERENCE_TOKENS = 1` (`block_functions.py:26-27`). The comment at `block_functions.py:23-25` is explicit:

```
# We prioritize short inference requests and make them use a *merged* inference pool,
# so they are processed without interruptions and extra overheads
# TODO: Increase the NF4 threshold once bitsandbytes ships efficient NF4 kernel for parallel forward
```

`bitsandbytes` is a CUDA-only library (no ROCm/MPS support at time of this code). So while `block_functions.py` itself has no `cuda` token, it knows about NF4 quantization and therefore implicitly knows about CUDA. Anywhere `QuantType.NF4` is in play, the assumption is a CUDA worker.

### 3.6 Pool/backend coupling

- `backend.forward_pool`, `backend.backward_pool`, `backend.inference_pool` — used at `block_functions.py:71, 118, 134, 218, 224`. All assert `isinstance(backend.inference_pool, PrioritizedTaskPool)` (`block_functions.py:67, 114, 130`) — `# petals support only prioritized pools`.
- `backend.dtype` — line 53, 60, 98, 99, 104, 179, 187. Single dtype per backend; mixed precision across blocks is not modeled here.
- `backend.outputs_schema` — `block_functions.py:231`, `handler.py:419`. Output schema is per-backend, fixed at construction.
- `backend.memory_cache.allocate_cache(...)` — `handler.py:546`. The KV cache is owned by the backend, not the handler.

### 3.7 What is *not* CUDA-specific in these files

Worth calling out: `handler.py` has no torch-device code. It is a transport layer over hivemind's `ConnectionHandler` and only touches tensors via serialize/deserialize. The CUDA assumption is concentrated in (a) backend internals (out of scope here), (b) the NF4 quant branch, and (c) the tensor-op style of `block_functions.py`.

---

## 4. Where workers store model weights

**Direct answer from these files: the four files don't show weight loading — they show weights already in memory.**

What is visible:

- Per-block weights live inside a `TransformerBackend` instance (`handler.py:33`). The handler holds them as `module_backends: Dict[ModuleUID, TransformerBackend]` (`handler.py:58`) and passes them to compute functions as `requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)` (`handler.py:149`, `handler.py:359`, `handler.py:441`).
- The backend exposes `forward_pool`, `backward_pool`, `inference_pool` (each a `PrioritizedTaskPool`, `block_functions.py:67`); a fixed `dtype` (`block_functions.py:53`); an `outputs_schema` (`handler.py:419`); `args_schema` and `kwargs_schema` (`handler.py:506`); and a `memory_cache` for KV cache (`handler.py:546`). Whether weights are on GPU, CPU, mmapped from disk, or paged is **not visible in these four files** — that's `petals.server.backend.TransformerBackend` (out of scope) plus `petals.utils.convert_block`.
- KV cache (not weights, but worth disambiguating since both "live on the worker"):
  - Allocated lazily per `rpc_inference` session in an async context manager (`handler.py:532-547`):
    ```python
    descriptors = [backend.get_inference_cache_descriptors(batch_size, max_length) for backend in backends]
    async with backends[0].memory_cache.allocate_cache(*chain(*descriptors), timeout=timeout) as handles:
        yield nested_pack(handles, descriptors)
    ```
  - The single `memory_cache` from `backends[0]` is shared across all backends on that worker (line 546's `backends[0].memory_cache.allocate_cache` covers `*chain(*descriptors)` from every backend) — so KV cache memory is a process-wide pool, not per-block.
  - Cache budget is reported back to clients via `rpc_info`'s `cache_tokens_available = backend.memory_cache.bytes_left // max(backend.cache_bytes_per_token.values())` (`handler.py:582`). So the server thinks of cache headroom in tokens, computed from a bytes-left field plus a per-token cost estimate.
  - `alloc_timeout` lets a client wait for cache to free up rather than fail (`handler.py:153`, `handler.py:171`).

To answer the original question fully — disk vs. memory vs. streamed — Nakshatra needs to read `petals/server/backend.py` and `petals/utils/convert_block.py`. The four files in scope here treat weight residency as opaque.

---

## 5. How clients discover available workers

**Also not directly in these four files**, but the surface area through which they interact with discovery is fully visible.

What's in scope:

- Discovery is driven by `RemoteSequenceManager` (imported at `inference_session.py:17` from `petals.client.routing`). The client never talks to the DHT directly in `inference_session.py`; everything goes through this manager.
- Manager attributes the client uses:
  - `self._sequence_manager.block_uids` — full ordered list of block UIDs for the model (`inference_session.py:253`).
  - `self._sequence_manager.config` — a `ClientConfig` with `connect_timeout`, `request_timeout`, `max_retries`, `use_server_to_server` (`inference_session.py:75`, `188`, `136`, `349`).
  - `self._sequence_manager.state.p2p` — the libp2p instance used to open streams (`inference_session.py:258`).
  - `self._sequence_manager.rpc_info` — the cached schema returned by servers' `rpc_info` (`inference_session.py:261`, paired with `handler.py:575-592`). This contains `inference_schema` used to negotiate compression: `inference_session.py:144-146`:
    ```python
    server_side_inference_schema, kwargs_schema = self.rpc_info["inference_schema"]
    compression = server_side_inference_schema[0].compression
    ```
- Manager methods the client uses:
  - `make_sequence(start, end, mode="min_latency", cache_tokens_needed=...)` — returns a list of `RemoteSpanInfo` covering the requested block range (`inference_session.py:376-378`).
  - `get_request_metadata("rpc_inference", span_uids, peer_id=span.peer_id)` — builds the per-server metadata dict (used at `inference_session.py:254`); presumably attaches authentication, points balance, etc.
  - `on_request_success(peer_id)` / `on_request_failure(peer_id)` — feedback hooks the client uses at `inference_session.py:343, 346` to update routing state.
  - `get_retry_delay(attempt_no)` — backoff schedule (`inference_session.py:351`).

The DHT itself is constructed and consumed elsewhere; on the server side it is passed in at `handler.py:62` and only used to expose `dht.client_mode` to `rpc_info` (`handler.py:581`). UID advertisement to the DHT is not in `handler.py`.

So: from these four files alone the discovery story is "the client asks `RemoteSequenceManager` for a path, and `RemoteSequenceManager` knows how to find one." The actual mechanism — DHT key format, peer scoring, latency probing — is in `petals.client.routing` and `petals.server.reachability`/`block_selection`, none of which are read here.

---

## 6. Worker dropping mid-request

This is the one question fully answered by the files in scope. Petals' fault tolerance lives almost entirely in `inference_session.InferenceSession.step` and is built on three primitives: a retry loop, a path re-planner, and a replayable client-side history.

### 6.1 The retry loop

`inference_session.py:322-357`:

```python
server_idx = 0
block_idx = 0
while block_idx < self.num_blocks:
    for attempt_no in itertools.count():
        logger.debug(f"Inference: block {block_idx}, attempt {attempt_no}")
        server_session = None
        try:
            if not self._server_sessions or attempt_no >= 1:
                self._update_sequence(server_idx, block_idx, attempt_no)

            server_session = self._server_sessions[server_idx]
            assert server_session.position == self.position, ...
            inputs = server_session.step(
                inputs,
                prompts[server_session.span.start : server_session.span.end],
                hypo_ids,
                step_id=step_id,
            )

            server_idx += 1
            block_idx = server_session.span.end
            self._sequence_manager.on_request_success(server_session.span.peer_id)
            break
        except Exception as e:
            self._sequence_manager.on_request_failure(
                server_session.span.peer_id if server_session is not None else None
            )
            if attempt_no + 1 == self._sequence_manager.config.max_retries:
                raise
            delay = self._sequence_manager.get_retry_delay(attempt_no)
            logger.warning(...)
            maybe_log_traceback(e)
            time.sleep(delay)
```

Failure-handling shape:
- Per-block, not per-request: the `for attempt_no` loop is inside the `while block_idx < self.num_blocks` loop. If block 50's worker dies, only that span is retried; blocks 0-49's outputs are already in `inputs`.
- `on_request_failure(peer_id)` notifies the sequence manager so it stops routing to the dead peer.
- Bounded retries by `config.max_retries`.
- Backoff via `get_retry_delay(attempt_no)`, blocking with `time.sleep` (this is sync code wrapping the async coroutines via `RemoteExpertWorker.run_coroutine`).

### 6.2 Path re-planning

`inference_session.py:364-391`, `_update_sequence`:

```python
def _update_sequence(self, server_idx, block_idx, attempt_no):
    self._exit_server_sessions(self._server_sessions[server_idx : server_idx + 1])

    n_prev_spans = len(self._server_sessions)
    update_end = self._server_sessions[server_idx].span.end if server_idx < n_prev_spans else self.num_blocks
    if attempt_no >= 1:
        logger.debug(
            f"Due to a server failure, remote attention caches "
            f"from block {block_idx} to {update_end} will be regenerated"
        )

    updated_spans = self._sequence_manager.make_sequence(
        block_idx, update_end, mode="min_latency", cache_tokens_needed=self._max_length
    )
    updated_spans[-1].end = min(updated_spans[-1].end, update_end)
    updated_sessions = self._enter_server_sessions(updated_spans)

    if server_idx < n_prev_spans:
        updated_sessions[0].history = self._server_sessions[server_idx].history
    self._server_sessions[server_idx : server_idx + 1] = updated_sessions

    for i in range(max(server_idx - 1, 0), min(server_idx + len(updated_spans), len(self._server_sessions) - 1)):
        self._server_sessions[i].next_session = self._server_sessions[i + 1]
```

Behavior:
- Closes the dead session via `_exit_server_sessions` (which itself swallows close errors at `inference_session.py:273-278`).
- Asks the sequence manager for a *replacement path* covering only the affected range `[block_idx, update_end)`. It does not re-route already-completed spans.
- The replacement may be one server or several (`updated_spans`), and may extend past `update_end`; the line `updated_spans[-1].end = min(updated_spans[-1].end, update_end)` clamps the last span back.
- `updated_sessions[0].history = self._server_sessions[server_idx].history` (line 386) — the new server inherits the *input history* of the dead one. This is what makes recovery work without a global checkpoint: the new worker can rebuild the KV cache by replaying from the start of the span.
- `next_session` links are repaired around the replacement so the server-to-server push chain stays intact.

### 6.3 The replayable history

`inference_session.py:55-56`:

```python
self.history = None  # Used in case of server failures to regenerate attention caches on new servers
```

History is accumulated on every call (`inference_session.py:113-117`):

```python
n_input_tokens = inputs.shape[1]
if self.history is None:
    self.history = inputs
elif self.history.shape[1] == self._position:
    self.history = torch.cat([self.history, inputs[:, -n_input_tokens:]], dim=1)
```

And the first step on a new (or replacement) session sends the *full* prefix, not just the new tokens (`inference_session.py:123-126`):

```python
if not self.stepped:
    inputs = self.history  # Pass full inputs including prefix
else:
    inputs = inputs[:, -n_input_tokens:]  # No need to pass prefix further
```

So the client trades memory (storing full hidden-state history per block range) for fault tolerance (any span can be reconstructed by replay).

`position` setter at `inference_session.py:90-95` lets a higher layer rewind:

```python
@position.setter
def position(self, start_from_position):
    assert start_from_position <= self._position
    self._position = start_from_position
    if self.history is not None and self.history.shape[1] >= start_from_position:
        self.history = self.history[:, :start_from_position, :] if start_from_position > 0 else None
```

paired server-side with `start_from_position` handling at `block_functions.py:163-168`.

### 6.4 Server-side timeouts that cause "drops"

A worker doesn't have to crash to be considered dropped — the server itself will drop a slow client:

- `async with timeout(self.session_timeout)` wraps the entire `rpc_inference` (`handler.py:138`).
- First step gated by `await asyncio.wait_for(anext(requests), self.step_timeout)` (`handler.py:140`); timeout returns silently with a warning log.
- Subsequent steps gated by `asyncio.wait([anext_task, get_push_task], timeout=self.step_timeout, return_when=asyncio.FIRST_COMPLETED)` (`handler.py:291-293`); both pending tasks are cancelled and the iterator returns on timeout (`handler.py:301-305`).

From the client's perspective these manifest as exceptions out of `_step` (whose `await asyncio.wait_for(anext(self._outputs_stream), self.config.request_timeout)` at `inference_session.py:188` will raise) and feed straight into the retry loop above.

### 6.5 What is *not* handled

Worth flagging for Nakshatra's design discussion:
- No checkpoint of intermediate hidden states beyond the per-span `history`. If every server in the chain dies simultaneously, the client retries from `history[:position]` but cannot fall further back than the last-completed step.
- No correctness guarantee under non-determinism — a replacement server's recomputed KV cache must produce numerically equivalent outputs to the original. With deterministic FP/dtype settings this is fine; with non-deterministic kernels this is implicitly papered over.
- Worker-to-worker `rpc_push` failures are swallowed (`handler.py:346-350` — `except Exception: logger.debug(...)`). The client-driven path will still pick up the work because every step still flows through the client; the push is purely an optimization.
- Deep prompts (`has_prompts` at `block_functions.py:233`) disable the push optimization, so models using prompt-tuning fall back to client-routed forwarding for every step.

---

## Quick reference: file → responsibility

| File | What it owns |
|---|---|
| `auto_config.py` | Model class registry; `model_type` → Petals subclass dispatch |
| `server/handler.py` | gRPC-style RPC surface (`rpc_inference`, `rpc_forward[_stream]`, `rpc_backward[_stream]`, `rpc_push`, `rpc_info`); session/step timeouts; KV cache allocation context manager; cross-handler session routing within a worker process |
| `server/block_functions.py` | The actual chained forward/backward/inference compute over a span of `TransformerBackend`s; the NF4-aware merged-pool fast path |
| `client/inference_session.py` | Per-server bidi stream wrapper, multi-server fault-tolerant `step()` loop, path replanning, replayable history |

## Out-of-scope but referenced (not read here)

- `petals.server.backend.TransformerBackend` — owns weights and pools.
- `petals.utils.convert_block` — defines `QuantType`, presumably handles weight conversion / quantization.
- `petals.client.routing.RemoteSequenceManager` — discovery, path planning, peer scoring.
- `petals.data_structures` — `ModuleUID`, `RemoteSpanInfo`, `Handle`, `InferenceMetadata`, `CHAIN_DELIMITER`, `UID_DELIMITER`, `RPCInfo`.
- `petals.server.task_pool` — `PrioritizedTaskPool`.
- `petals.utils.packaging` — `pack_args_kwargs` / `unpack_args_kwargs` (the args_structure wire-format convention).
- The hivemind layer in general — DHT, P2P transport, compression codec.

These are the next places to read if Nakshatra wants to fill in the gaps in §1.1 (assignment), §4 (weight residency), and §5 (discovery).
