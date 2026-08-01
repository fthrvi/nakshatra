# Remote token proposals over a real WAN link — 5.87 tok/s, and it beats the split chain (2026-08-01)

`remote-proposals.md` closed with one open item: *"A live WAN test … Still NOT run
over a real WAN link — that's the one item left."* Built, merged, 21 tests, never
run against a real remote verifier. ijru moving sites supplied the link.

## Setup

- **local (hub):** Qwen3-0.6B draft on the 9070 XT, via the repo's incremental
  `speculative.DraftModel`.
- **remote (ijru, ~181 ms away):** whole **Qwen3-14B Q4_K_M** on the RTX 3060,
  8 926 MiB resident. The model is never split; the wire carries only proposal
  chunks and accept/correct responses — a handful of ints per round trip.
- Same prompt and 96-token budget as every split-chain run.

## Result

| arm | tok/s | round trips | tokens / round trip |
|---|---|---|---|
| k=1 — one round trip per token (naive remote) | 4.10 | 54 | 1.78 |
| **k=4 — bounded proposals** | **5.87** | **36** | **2.67** |
| *(reference)* 30B split chain over the same link | 4.85 | — | — |

**Pre-registered bar was >4.85 tok/s. Met: 5.87, a 1.21× win over splitting**, and
**1.43× over naive remote generation**. The second bar (>15 tok/s, "the answer for
a remote node outright") was **not** met.

**Output was byte-identical across all four runs** — k=1 and k=4, and both draft
implementations. The correctness invariant (a bad draft changes how many rounds it
takes, never what comes out) now holds over a real WAN link, not just loopback.

## Where the remaining time goes

16.35 s / 36 rounds = **454 ms per round**, of which RTT is 181 ms. So ~273 ms is
compute: the 14B evaluating a 5-token batch with `logits_all=True` on a 3060, plus
the draft. **The link is only 40% of each round** — which is the whole point of the
technique, and also why it stops short of 15 tok/s. On the punched direct path
(28.6 ms, see `direct-path-live-wan.md`) the same arithmetic predicts ~8.8 tok/s,
at which point verifier compute dominates entirely and the next lever is the
verifier, not the wire.

## Two bugs in the shipped code, found by running it

Both were invisible to the 21 tests and to loopback use.

1. **`make_verifier_session` never primed the KV.** `VerifierSession` sets
   `cursor = len(seed_tokens) - 1` — it assumes the verifier has already evaluated
   the prefix — but a fresh `LlamaVerifier` has `n_tokens == 0`. Every multi-token
   prompt therefore failed the first `submit()` with *"start_pos ahead of this
   adapter's cached KV"*. It survived until now only because the default seed is a
   **single BOS token**, where `cursor == 0 == n_cached` and the arithmetic is
   accidentally consistent. The first real prompt (19 tokens) broke it.

2. **The error path destroyed the error.** `serve_verifier` did
   `send_error(400, str(e))`, and our own exception messages contain em-dashes.
   The HTTP reason-phrase is latin-1 by spec, so the handler raised
   `UnicodeEncodeError` *while reporting the error*, dropped the connection, and
   the client saw `RemoteDisconnected` with no diagnostic at all. Reason phrase is
   now ASCII; the detail goes in the body.

## A measurement error worth recording

The first run of this experiment gave **2.94 tok/s** and appeared to fail the bar.
The cause was in the harness, not the protocol: the draft re-evaluated the entire
context every round instead of using `speculative.DraftModel`, whose docstring
names precisely this — *"Instead of re-evaluating the whole prefix every round
(O(prefix) → O(n²) over a generation)"*. Switching to the incremental draft doubled
throughput, 2.94 → 5.87. **A benchmark harness that bypasses the optimised path in
the repo it is benchmarking will produce a confident, wrong negative.**
