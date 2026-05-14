# Finding: Metal `last`-mode worker crashes on `newBufferWithBytesNoCopy`

**Discovered:** 2026-05-13 during the v0.5 §7 5-machine 70B acceptance run on bishwa.
**Status:** Open. §7 acceptance run paused on this finding; CPU-`last` workaround remains available (used by the 4-Mac M0.5.1 path recorded in `v0.5-design-lock.md` §10 earlier today).

## Symptom

When the patched `llama-nakshatra-worker` daemon runs in `last` mode (`has_lm_head=true`, receives a hidden state via `EMBD_DECODE`, returns a top-1 token id) **and** offloads layers to Metal (`--n-gpu-layers 99 --gpu-backend metal`), `llama_decode` aborts with:

```
init: embeddings required but some input tokens were not marked as outputs -> overriding
ggml/src/ggml-metal/ggml-metal-device.m:1624: GGML_ASSERT(buf_src) failed
```

Backtrace:

```
ggml_metal_buffer_set_tensor + 525
ggml_backend_sched_graph_compute_async + 428
llama_context::graph_compute
llama_context::process_ubatch
llama_context::decode
llama_decode
worker_daemon main + 2119
```

The assertion fires at the very first `EMBD_DECODE` call on a `last`-mode worker — the cold prefill of a 6-token prompt for Llama-3.3-70B layers [56, 80). Repros every time.

## Why CPU works and `first`/`middle` workers on Metal also work

| Mode    | Backend | Result |
|---------|---------|--------|
| first   | Metal   | ✅ works (prithvi-5530 with ROCm, mac3-2-5531 with Metal) |
| middle  | Metal   | ✅ works (mac4-5533 with Metal, mentorings-5532 with CPU) |
| last    | CPU     | ✅ works (the 4-Mac M0.5.1 chain; bishwa-test:5563 cpu mode `last`) |
| last    | Metal   | ❌ crashes (this finding) |

The graph emitted by the patched `models_llama.cpp` only calls `build_inp_out_ids()` when `model.nks_has_lm_head` is true (per `experiments/v0.0/m4_patches/models_llama.cpp.patch:8`). So the `inp_out_ids` tensor is present in `last`-mode graphs and absent in `first`/`middle`-mode graphs. CPU works because the CPU backend uses plain `memcpy` for input upload; Metal goes through `newBufferWithBytesNoCopy`, which has strict alignment preconditions.

## Root-cause hypothesis

The assertion site (bishwa's `libggml-metal.0.9.7.dylib`, source `ggml-metal-device.m:1614-1626`):

```objc
@autoreleasepool {
    void * data_ptr = (void *)(uintptr_t) data;
    id<MTLBuffer> buf_src = [buf->dev->mtl_device newBufferWithBytesNoCopy:data_ptr
                                                           length:size
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
    GGML_ASSERT(buf_src);
    ...
}
```

Apple's `newBufferWithBytesNoCopy:length:options:deallocator:` returns `nil` when:
- `data_ptr` is not page-aligned (page size = 4 KB on Apple Silicon, 16 KB on Intel macOS),
- or `length` is not a multiple of the system page size,
- or `length` is zero or too large.

`inp_out_ids` is a small `int32` tensor — one entry per requested output token. For `last`-mode with `cp.embeddings=true` plus only the final token marked `logits=1`, llama.cpp's batch validator hits the "embeddings required but some input tokens were not marked as outputs → overriding" path, which expands the requested outputs (so `inp_out_ids` data length grows from 4 bytes to `n_tokens * 4` bytes). For a 6-token prefill that's 24 bytes — still far short of 4 KB / 16 KB and not page-aligned. Metal returns `nil` and the assert fires.

Verified experimentally 2026-05-13: changing `cp.embeddings = true` to `cp.embeddings = !mode_last` (so the override path doesn't fire) **suppressed the override log line but did NOT change the assertion** — the same `GGML_ASSERT(buf_src)` still fires at the same site. Conclusion: the assertion is not specifically about the override; it is about Metal's inability to wrap small tensor data via `newBufferWithBytesNoCopy`. The override path just made the symptom more visible.

## Why this isn't fixable in our code

The failing call is inside upstream llama.cpp's `libggml-metal`, not in Nakshatra's patched files (`worker_daemon.cpp`, the `m4_patches/`). Specifically the `ggml_metal_buffer_set_tensor` function is the upstream backend's host-to-device transfer path. To fix it from our side we would need one of:

1. **Patch `libggml-metal` locally.** Edit `ggml-metal-device.m` so small/unaligned uploads use `[id<MTLBuffer> contents]` + `memcpy` (or Metal's `setBytes` for command-encoder inputs) instead of `newBufferWithBytesNoCopy`. ~10-30 LOC, surgical, but a new patch surface to maintain.
2. **Bump `libggml-metal` to a newer upstream commit.** Newer llama.cpp versions are known to handle small-tensor uploads via the `setBytes` path. Risky because the Nakshatra `m4_patches/` were validated against the May 2026 llama.cpp tree; the graph builder API may have moved.
3. **Avoid the input tensor entirely.** Re-shape the daemon-side `last`-mode batch so `inp_out_ids` either isn't needed or has a size that Metal accepts. We tried `cp.embeddings = !mode_last`; it didn't help. Marking all 6 tokens as outputs would yield 24 bytes — still below page size.

None of these are 5-minute fixes.

## Workaround (currently in use)

Run `last`-mode workers on **CPU**: `--gpu-backend cpu --n-gpu-layers 0`. This is the configuration the 4-Mac M0.5.1 chain has been using all day (`bishwa-test:5563`). It is slow on bishwa for 24 layers of 70B (~5-10 s per token expected), but it bypasses the Metal path entirely and the §7 acceptance gate can still complete.

## What this means for v0.5 §7

- The 4-Mac M0.5.1 acceptance ✅ already passed today using CPU-`last`. Not affected by this finding.
- The 5-machine 70B acceptance (with prithvi as ROCm-`first` and bishwa as Metal-`last`) is **blocked** on Metal-`last` until one of the fix paths above is applied.
- §9.6 closure already noted that the cross-vendor 5-machine run was pending prithvi cold-boot. This finding adds that it's *also* pending Metal-`last` (or a switch to CPU-`last`).

## Repro

1. Cold-boot prithvi (or any host without the wedged-daemon prior state).
2. Start workers per `scripts/cluster_l3370b_5machine.yaml`. Bishwa-5534 must be launched with `--gpu-backend metal --n-gpu-layers 99`.
3. From any client host with `~/models/llama-3.3-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf`:
   ```
   python3 scripts/client.py --config scripts/cluster_l3370b_5machine.yaml \
       --model-path ~/models/llama-3.3-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
       --prompt "The capital of France is" --max-tokens 1 --use-streaming
   ```
4. Observe `Inference stream to 'bishwa-5534' failed: INTERNAL — Inference stream error: short read from daemon (got 0 bytes)`.
5. On bishwa, `tail /tmp/bishwa-worker-5534.log` shows the GGML_ASSERT backtrace.

## Cluster state at time of writing

- Prithvi `:5530` worker still running, healthy (advertises full v0.5 capabilities — first machine to do so cluster-wide).
- Bishwa `:5534` worker killed; daemon binary on bishwa is reverted to upstream `cp.embeddings = true`.
- All 4 Mac `~/nakshatra-v0/` worker.py files are still on the intermediate (pre-§9.1/§9.5) revision. Updating them is independent of this finding.
