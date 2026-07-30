# Finding: bounded external token proposals — WAN spec-decode without splitting the model

**Landed:** 2026-07-29, branch `inference/remote-proposals` (Mesh-LLM adoption #1, one of
three parallel lanes staked in `INBOX.md` the same night as the async-pipelining GPU proof).
**Status:** Protocol core done + unit-proven (`scripts/remote_proposals.py`, no GPU/model/live
services). Mounted on `nakshatra_serve` behind `NKS_REMOTE_PROPOSALS=1` with a real
llama.cpp CPU-only verify_fn on branch `serving/live-seams` (2026-07-29) — see "What landed
since" below. Still NOT run over a real WAN link — that's the one item left; see "Remaining
ask" #3.

## What it is

A remote node runs the **whole** verifier model (never split — this repo's doctrine is route
whole models, don't split, unlike Shard's pipeline-parallel chain-splitting). A local node
runs a small same-tokenizer draft model. The wire between them carries only **proposal
chunks and accept/correct responses** — a handful of ints per round trip, never activations
or layers. This is WAN speculative decoding, adopted from Mesh-LLM's technique, adapted to
fit this repo's existing spec-decode + KV-rewind vocabulary (`speculative.py`,
`async_pipeline.py`) instead of a fresh protocol.

Two pieces, both in `scripts/remote_proposals.py`:

- **`VerifierSession`** (server-side, sits next to the whole model): wraps an abstract
  per-position greedy oracle `verify_fn(tokens, start_pos) -> argmaxes`. Maintains the
  committed prefix and a KV cursor itself. `submit(proposal)` returns
  `(n_accepted, correction_token, cursor)`. Oversized proposals are **rejected, not
  raised** — `(0, None, cursor)` with no state change — because over a real WAN link a
  misjudged window size is a routine protocol event, not a bug.
- **`proposal_loop`** (client-side, sits next to the draft): draft proposes K tokens →
  `submit` → commit what came back → repeat. On a bounded-window rejection it halves K and
  retries, never losing progress.

## The invariant

Same shape as the oracle `speculative.py` and `async_pipeline.py` already assert:

> The final output must be **byte-identical** to what the verifier model alone would
> generate greedily, for ANY draft quality (perfect, adversarial, mixed).

It holds because `VerifierSession.submit()` always re-derives its `cur` and `cursor` from
what it **just committed**, never from what the draft assumed — a bad draft costs speed
(more single-token rounds), never correctness. `n_accepted` is the same greedy longest-prefix
match speculative.py's `accept()` computes; `cursor` advancing by only `n_accepted + 1`
(never the full proposal length) is the WAN analog of `speculative.py`'s `kv_keep_after` /
`async_pipeline.py`'s `start_pos` trim — a real verifier truncates its KV to the returned
cursor before the next call, so a rejected tail's KV is discarded rather than leaking
forward. `tests/test_remote_proposals.py::test_cursor_rewind_after_mispredict_not_full_advance`
proves this with a spy on `verify_fn` that checks the exact `start_pos` presented next round.

21 tests (`tests/test_remote_proposals.py`): perfect/adversarial/mixed drafts vs a sequential
verifier-only reference, bounded-window rejection (both under and over the limit), cursor
rewind, EOS mid-batch, `proposal_loop`'s auto-shrink-K-on-rejection, and a loopback HTTP
round-trip proving the protocol survives JSON serialization before any real network is
involved.

## The WAN motivation

The night before this landed, the **async-pipelining GPU-mesh proof** (branch
`inference/async-pipelining`, see `BRANCHES.md`/`INBOX.md` 2026-07-29) measured chain-split
async pipelining on the real hub(ROCm)↔ijru(CUDA) LAN chain: **4.58 tok/s sequential vs 1.04
tok/s async — negative**. Correctness was proven (byte-identical receipts, real KV-rewind
under mispredict), but on sub-millisecond LAN RTT there was nothing worth hiding, and the
client-side draft-continuation scheduler burned ~7× CPU for no win. The verdict on that
finding: async pipelining (and by extension any chain-splitting technique) is a **WAN-regime**
technique — it pays for itself only when there's real RTT to amortize across a deep pipeline,
and even then it multiplies that RTT by pipeline depth if the model is split across the WAN
hop itself.

Bounded external token proposals are the WAN answer that avoids that multiplication
entirely: **the model stays whole on one remote node.** There is exactly one round trip per
speculative chunk (not one per pipeline stage), and each round trip is small — proposal
tokens out, accept/correct back — regardless of how large the verifier model is. This is the
same shape Mesh-LLM and Shard's own remote-verifier mode use to make WAN federation
economical without chain-splitting: hide the WAN RTT behind draft-generated tokens instead of
behind pipeline fill.

## What landed since (branch `serving/live-seams`, 2026-07-29) — items 1 & 2

- **`nakshatra_serve.py` mounts `VerifierSession`** behind
  `_maybe_start_proposals_server()`, gated on `NKS_REMOTE_PROPOSALS=1` (default
  unset/OFF ⇒ this function no-ops and `scripts/remote_verifier_backend.py` — the adapter
  module — is never even imported; the flag-off serve is byte-identical to before this
  seam existed, proven by a subprocess test that `llama_cpp`/`remote_verifier_backend`
  never land in `sys.modules` on a plain import). Still rides the SAME thin
  `serve_verifier`/`http_submit` stdlib loopback transport described above — that
  characterization is unchanged, just now actually mounted, on `NKS_PROPOSALS_PORT`
  (default 11601) in a daemon thread alongside the OpenAI/Ollama facade.
- **A real whole-model `verify_fn`**: `scripts/remote_verifier_backend.py`'s
  `LlamaVerifier` wraps a `llama_cpp.Llama` handle loaded `n_gpu_layers=0` (hardcoded,
  CPU-only — never touches conscious/voice VRAM), `logits_all=True` (same requirement as
  `speculative.py`'s `DraftModel`). It rewinds the KV cache to `start_pos` via
  `llama._ctx.kv_cache_seq_rm(...)` before each batch — the exact primitive item 2 asked
  for, modeled directly on `DraftModel.propose`'s LCP rollback — and returns one greedy
  argmax per input token via `.scores`. Proven against a deterministic `FakeLlama` (no
  llama_cpp, no GGUF) in `tests/test_remote_verifier_backend.py`: the per-position argmax
  contract, the rewind-on-mispredict path (asserting the exact `kv_cache_seq_rm` call), a
  vocab-mismatch guardrail (`NKS_VERIFIER_EXPECT_VOCAB`), and the full
  `proposal_loop`/`VerifierSession` byte-identical-to-sequential oracle running THROUGH
  the adapter for perfect/adversarial/mixed drafts — plus a loopback smoke that starts the
  verifier thread on port 0 with the fake model and round-trips one proposal via
  `http_submit`.

**Honesty note:** "vocab must match the draft family" is enforced only when the operator
sets `NKS_VERIFIER_EXPECT_VOCAB` — there is no draft model reference on the server side to
compare against automatically (the draft lives on the WAN client, which this branch does
not build). And `VerifierSession` still runs ONE continuous generation stream per process
(seeded from the verifier's BOS token by default) — a per-request session pool is not
built; that's still a production-transport question, same as before.

## Remaining ask (explicit, deferred) — item 3

1. ~~Mount `VerifierSession` on `nakshatra_serve` behind a real endpoint~~ — done above.
2. ~~Wire a real whole-model `verify_fn`~~ — done above.
3. **A live WAN test**, e.g. across the sovereign mesh's actual inter-site RTT (the UNM lab
   site or a VPS hop), measuring tok/s WITH vs WITHOUT proposals against a real remote
   verifier + local draft — the throughput half of the correctness-is-proven,
   throughput-is-not pattern this lane inherited from the async-pipelining finding. Needs
   a real GGUF on both ends (verifier + draft, matching tokenizer families) and live mesh
   nodes — out of scope for a no-GPU/no-live-services branch.
4. ~~Land behind `NKS_REMOTE_PROPOSALS`~~ — done; now actually read (`nakshatra_serve.py`'s
   `_maybe_start_proposals_server()`), still default OFF.
