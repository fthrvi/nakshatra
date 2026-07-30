# Finding: bounded external token proposals — WAN spec-decode without splitting the model

**Landed:** 2026-07-29, branch `inference/remote-proposals` (Mesh-LLM adoption #1, one of
three parallel lanes staked in `INBOX.md` the same night as the async-pipelining GPU proof).
**Status:** Protocol core done + unit-proven (`scripts/remote_proposals.py`, no GPU/model/live
services). NOT wired into `nakshatra_serve` or `client.py`. NOT run over a real WAN link or
against a real whole-model verifier — see "Remaining ask" below.

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

## Remaining ask (explicit, deferred)

This branch delivers the **pure protocol module only** — no wiring, no live model, no WAN
test, per the no-GPU/no-live-services scope of this lane. What's left, for whichever lane
picks it up next:

1. **Mount `VerifierSession` on `nakshatra_serve`** behind a real endpoint (the current
   `serve_verifier`/`http_submit` stdlib loopback pair is a deliberately thin seam — enough
   to prove the protocol survives serialization, not the production transport).
2. **Wire a real whole-model `verify_fn`** — a real forward pass over an actual llama.cpp
   (or equivalent) handle that truncates its KV cache to `start_pos` before evaluating each
   batch (the real KV-rewind primitive this module's cursor arithmetic assumes exists;
   `speculative.py`'s `DraftModel` and the worker daemon's `TruncateKV` are the closest
   existing analogs to model this on).
3. **A live WAN test**, e.g. across the sovereign mesh's actual inter-site RTT (the UNM lab
   site or a VPS hop), measuring tok/s WITH vs WITHOUT proposals against a real remote
   verifier + local draft — the throughput half of the correctness-is-proven,
   throughput-is-not pattern this lane inherited from the async-pipelining finding.
4. Land behind `NKS_REMOTE_PROPOSALS` (declared, default OFF, in `scripts/remote_proposals.py`
   already — not yet read anywhere, since nothing wires the module in yet).
