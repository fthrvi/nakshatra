# Finding: speculative decode ON THE STREAMING TRANSPORT

**Landed:** 2026-07-29/30, branch `inference/stream-spec`. Follows directly from
`docs/findings/cuda-chain-51-tok-s.md`'s closing line: "the genuinely promising direction
is not a better draft but speculative decode ON the streaming transport — i.e. teach the
streaming path to carry K-token verify batches."

## The gap, and what was actually missing

Tonight's baseline measured on the live hub(ROCm 9070 XT)/ijru(CUDA 3060) Q3 cross-vendor
chain: plain `--use-streaming` decode = 51-53 tok/s, `--speculative` (unary Forward, even
with a GPU draft) = 12.20 tok/s, `--async-pipeline` = 6.60 tok/s. Root cause: spec and async
both ride the unary `Forward` RPC — that's where `all_logits`/`keep_kv`/`start_pos` live —
which pays per-call setup on every one of K+1 verify positions, while plain decode rides the
persistent `Inference` stream at 5-11ms/stage/token. Speculation's premise (amortise one
expensive step over K tokens) dies when the *transport* per call costs more than the compute
it's amortising.

Scouting `proto/nakshatra.proto` and `scripts/worker.py` found the gap was narrow:
`ForwardRequest` already had `all_logits`; `InferenceStep` didn't. `worker.py`'s streaming
`Inference` handler already used `step.prefix_length` as `start_pos` — the KV-rewind
primitive — and already called the *same* `_run_forward`-adjacent daemon path Forward uses.
**The only missing piece was a way to request all-position verify on the stream.**

## What was built

1. **proto** (`proto/nakshatra.proto:137`): `InferenceStep.all_logits` (tag 17, additive
   — 16 was the highest tag in use). Regenerated stubs the repo's way
   (`python -m grpc_tools.protoc -I proto --python_out=scripts --grpc_python_out=scripts
   proto/nakshatra.proto`, matching `scripts/generate.sh`); diffed to confirm the change is
   purely additive (`scripts/nakshatra_pb2.py` — only the new field, no other churn;
   `nakshatra_pb2_grpc.py` unchanged, no new RPC).

2. **worker.py** (`scripts/worker.py:1178,1427,1490-1499`):
   - `Info()` advertises a new `"stream_spec"` capability.
   - The `Inference` handler ORs `step.all_logits` into the same `0x2` daemon flag bit
     `_run_forward` already sets for the unary path — one daemon primitive, two transports.
   - On `mode == "last"` with `all_logits=True`: unpacks the daemon's full
     `n_tokens`-int32 response into `token_ids.ids` (one argmax per input position) instead
     of a single id. Non-last workers are untouched either way — they keep returning
     `hidden_state`. `all_logits=False` (the default, proto3 semantics) is byte-identical to
     the pre-existing code path — verified by unit test, not just by inspection.

3. **client.py** (`scripts/client.py:210,290,313,628,1100`):
   - `call_inference_step` grew an `all_logits` kwarg (default `False`).
   - `stream_spec_disable_reasons(...)` — a **pure** gate function (no gRPC): refuses
     stream-spec unless `--use-streaming` (not push) is set, a draft model is configured,
     and **every** worker in the chain advertises `"stream_spec"`. Protobuf silently ignores
     fields it doesn't recognize, so an old worker missing the capability would otherwise
     return one token where K+1 were expected — corrupting `accept()` silently. The gate
     makes that failure mode structurally impossible: any miss prints
     `[stream-spec] requested but disabled: ...` and falls back to plain streaming.
   - `stream_spec_verify_fn(...)` — builds a `verify_fn` closure matching
     `speculative.speculative_round`'s contract, walking the **persistent per-worker
     Inference streams** (first → middle* → last, `all_logits=True` on every leg,
     `prefix_length` held fixed for the round) instead of unary `Forward`.
   - New CLI flag `--stream-spec` / env `NKS_STREAM_SPEC=1`, default OFF. The decode loop
     gained one branch (`stream_spec_active and step > already_done`) that calls
     `speculative_round(stream_draft, tokens + generated, spec_k, verify_fn)` — the
     **unchanged** `accept()`/`kv_keep_after()` from `speculative.py`, already proven
     byte-identical-to-greedy in `tests/test_speculative.py`. Only the transport is new;
     the acceptance math is reused verbatim. On failure, `stream_spec_active` is disabled
     for the rest of the session by the existing recovery loop, same as `spec_active`.

## Tests: 22 new, all with fakes (no gRPC, no daemon, no GPU)

`tests/test_worker_stream_spec.py` (9) — `WorkerServicer.Inference` against a recording
fake daemon:
- `Info()` advertises `stream_spec`.
- `all_logits=False` → single token, unchanged (byte-identical-to-today assertion).
- `all_logits=True` on `mode=last` → all `n_tokens` argmaxes returned, including the K=0
  edge case.
- a `mode=middle` worker is unaffected by `all_logits` (still returns `hidden_state`), but
  the `0x2` bit still reaches `daemon.call` (flags composition, not response shape, is what
  changes).
- the `0x2` bit ORs with the existing `0x1` keep_kv bit — never clobbers it.
- **KV-rewind, worker side**: a later step whose `prefix_length` is *smaller* than a naive
  full-advance reaches the daemon as the correspondingly smaller `start_pos`.

`tests/test_client_stream_spec.py` (13):
- `stream_spec_disable_reasons`: empty when ready; non-empty (with the right reason text)
  for not-streaming, push-mode, no-draft-model, one-worker-missing-capability,
  all-workers-missing, and reasons *accumulate* (not short-circuited) so the operator sees
  everything wrong at once.
- `call_inference_step` sends `all_logits=False` by default, `True` when asked.
- `stream_spec_verify_fn` + `speculative_round` wired over **fake per-worker streams**
  (2-worker chain, no gRPC): accept-all (commits K+1, bonus token), accept-none (immediate
  reject, one correction), partial accept.
- **KV-rewind, client side**: after a reject, the *next* round's request to both fake
  streamers carries `prefix_length == kv_keep_after(...)`, not a naive full advance —
  proving the rewind reaches the wire, not just local bookkeeping (complements the
  worker-side rewind test above).

```
$ .venv/bin/python -m pytest tests/test_worker_stream_spec.py tests/test_client_stream_spec.py -q
22 passed in 0.1x s
```

Full suite (excluding the pre-existing petals-derived `torch`/`hivemind`-gated files this
venv doesn't have, and 3 pre-existing failures confirmed identical on `forgejo/main` before
this branch touched anything — `test_client_tls.py` ×2, `test_worker_phase_a.py::test_a1_*`,
both about a stale grpc-message-cap constant, unrelated to this change):
**901 passed, 1 skipped, 3 pre-existing failures (same on main)**.

## Live proof — real 2-box GPU chain (hub ROCm 9070 XT L0-15 / ijru CUDA 3060 L15-48)

Both workers launched to survive the shell (`systemd-run --user --unit=streamspec-hub` on
hub with `HIP_VISIBLE_DEVICES=1`; `setsid nohup ... < /dev/null &` on ijru), same chain yaml
(`~/.nakshatra/qwen3-30b-q3-chain-v2.yaml`), same tokenizer/draft blobs, same prompt, same
`--max-tokens 96 --tls-mode off`, client run with `HIP_VISIBLE_DEVICES=1` so the draft used
the 9070 XT (ROCm `llama-cpp-python` 0.3.28, confirmed via `libggml-hip.so` in the venv).
Before starting: released `qwen3:30b-a3b` (the voice-test model — stateless, re-summonable),
never touched `prithvi:latest` (which wasn't even resident at the time). Re-pinned the voice
model afterward with the documented `keep_alive:-1` call.

VRAM offload confirmed live (not a log artifact — measured while serving, per the earlier
finding's hygiene note): hub GPU[1] 0.7GB → 6.0GB used after load; ijru `nvidia-smi`
429 MiB → 10,808 MiB.

### The three-way table (two samples each — see "on the numbers" below)

| run | tok/s (2 samples) | n_calls | worker-time / wall | output sha256 |
|---|---|---|---|---|
| plain streaming (control) | **24.21, 52.75** | 96 | 2.74s+1.20s / 3.96s (~99%); 0.70s+1.08s / 1.82s (~98%) | `d8522cc25569…` |
| unary `--speculative` (GPU draft) | 15.91, 16.09 | 44 | 0.50s+1.18s / 6.04s (~28%); 0.49s+1.16s / 5.97s (~28%) | `acbf8f81c377…` |
| **`--stream-spec`** (GPU draft) | 12.56, **16.05** | 44 | 0.54s+1.18s / 7.64s (~23%); 0.49s+1.14s / 5.98s (~27%) | `acbf8f81c377…` |

### Byte-identity — the correctness claim this task actually needs

**`--stream-spec` and unary `--speculative` produced the exact same `output_sha256`
(`acbf8f81c377…`) on every run** — same accept/reject sequence, same 44 verify rounds, same
generated tokens down to the byte. That's the load-bearing proof: the new streaming
transport reproduces, exactly, what the pre-existing (unmodified, already-tested) unary
verify traversal computes. It's not a coincidence of one run — both `unary-spec` and
`stream-spec` were run twice and matched each other both times.

**Plain streaming did *not* match the speculative paths** (`d8522cc25569…` vs
`acbf8f81c377…`), and this needs an honest explanation rather than a hand-wave: it is **not**
a stream-spec bug. It reproduces identically across two runs of plain streaming by itself
(`d8522cc25569…` both times) and across two runs of unary spec by itself — so within a given
*batch shape*, this stack is deterministic on this hardware tonight. The divergence tracks
**batch shape**, not transport: plain decode forwards one token at a time (batch=1);
speculative decode (either transport) forwards a K+1=5-token verify batch. Batched vs
single-token GPU matmuls use different reduction orders in rocBLAS/cuBLAS kernels, and
floating point isn't associative — this is exactly the substrate property this repo already
documented and revised its own acceptance criteria around on 2026-05-13
(`docs/v0.5-design-lock.md`: *"non-determinism is a substrate property, not a bug"*; "Metal
and, by extension, every other parallel inference backend... produce non-deterministic
floating-point outputs"). The task's stated oracle ("byte-identical to plain greedy decode")
already doesn't hold for the **pre-existing, unmodified** unary spec path on this GPU chain
tonight — stream-spec inherits that pre-existing, documented, out-of-scope property rather
than introducing a new one. The oracle that *is* satisfiable, and *was* satisfied twice: spec
output is stable within a batch shape, and stream-spec == unary-spec exactly.

### On the numbers, honestly

Two samples per config, taken minutes apart on the same shared hardware, show real variance
(plain: 24.21 → 52.75 tok/s) — `rocm-smi` warned `AMD GPU device(s) is/are in a low-power
state` throughout, and per-call latency for the *same* code path swung 3-4× between samples
(hub steps2-N avg 26ms → 7ms). This matches the prior finding's own caveat ("9070 XT stayed
soma-throttled, no sudo — absolute numbers conservative"). So: read the table as noisy in
absolute terms, but the **relative** structure is consistent across both samples and is the
actual finding:

- **stream-spec ties unary-spec, doesn't beat it here** (16.05 vs 16.09 tok/s on the more
  stable second sample; 12.56 vs 15.91 on the first, still same order of magnitude). Both
  ran the identical 44 verify rounds (byte-identical output forces byte-identical
  accept/reject decisions, hence identical round counts) — so this is an apples-to-apples
  comparison of the same computation over two transports, and the transport swap alone did
  not recover the gap to plain streaming.
- **Why not:** worker RPC time is only ~23-28% of wall clock for *both* spec variants, vs
  ~98-99% for plain streaming. That ratio is the tell — on this box, most of the speculative
  wall clock is neither the wire nor the worker compute; it's the **client's own draft
  compute**, which shares the same 9070 XT as the hub worker (`HIP_VISIBLE_DEVICES=1` for
  both). This matches the addendum in `cuda-chain-51-tok-s.md` almost exactly ("even with a
  GPU draft... the two workers together used 2.2s of a 7.9s run"). Per-call unary RPC setup
  is real (per the earlier 51-vs-12.20 finding) but on *this* co-located, low-RTT LAN
  topology it is not the dominant cost once the draft itself runs fast enough to be visible —
  the client's 4 draft forward passes per round, contending with the hub worker for the same
  GPU, are.

## Honest limits

- **Stream-spec does not, by itself, close the gap to plain streaming on this deployment.**
  It proves the mechanism (persistent-stream verify traversal, correct KV rewind, capability
  gate) end to end on real cross-vendor GPU hardware, and it is a strict improvement over
  unary spec in the specific sense that it removes the per-call unary setup cost — but that
  cost is not what's limiting throughput here tonight. A topology with more inter-worker
  network RTT to hide, or a draft that runs on hardware the workers don't share, would be a
  fairer test of the transport hypothesis in isolation; this box conflates "less unary
  overhead" with "draft competing for the same GPU," and only the live 2-box run could have
  surfaced that.
- **The byte-identity oracle only closes the loop stream-spec ↔ unary-spec, not either ↔
  plain.** That's a pre-existing, already-documented property of this stack (batch-shape
  floating point non-determinism), not something this branch introduces or could fix; noted
  rather than silently accepted.
- Only one prompt / one draft depth (K=4) was measured live; the unit tests cover K=0
  (single-token) and varied accept patterns, but a live sweep over K wasn't run.

## Cleanup confirmed

- `systemctl --user stop streamspec-hub` — unit stopped (transient, GC'd immediately after);
  no `worker.py` / `llama-nakshatra-worker` process left on hub (`pgrep` clean); hub GPU[1]
  back to pre-test-order-of-magnitude usage (ollama's own tenants, not the bench).
- ijru: `pkill` on both `worker.py --port 5572` and the CUDA daemon; `nvidia-smi` confirmed
  back to **429 MiB / 12288 MiB**, the documented idle baseline; port 5572 no longer
  listening.
- ijru's git state restored exactly as found: `git checkout main` (257ff02) +
  `git stash pop` restored the pre-existing uncommitted local patches (CUDA-rebuild-era
  worker.py/client.py/worker_daemon.cpp experiments) untouched — never committed over, never
  discarded. `eagle` conda env never invoked.
- Voice model (`qwen3:30b-a3b`) re-pinned via `keep_alive:-1`; `prithvi:latest` was never
  resident during this session and was never touched.

## Files

- `proto/nakshatra.proto` — `InferenceStep.all_logits` (tag 17).
- `scripts/nakshatra_pb2.py` — regenerated (additive-only diff verified).
- `scripts/worker.py` — `stream_spec` capability + `all_logits` honored on `Inference`.
- `scripts/client.py` — `--stream-spec`/`NKS_STREAM_SPEC`, `stream_spec_disable_reasons`,
  `stream_spec_verify_fn`, decode-loop branch, recovery-loop disable-on-failure.
- `tests/test_worker_stream_spec.py`, `tests/test_client_stream_spec.py` — 22 new tests.
- Receipts from the live run: `/tmp/claude-1000/wt-stream-spec/.receipts/*.json` (not
  committed — scratch, matches the repo's convention of not checking in run receipts).
