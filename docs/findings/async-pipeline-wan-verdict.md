# Async pipelining, tested in the regime it was reserved for — it still loses (2026-08-01)

`cuda-chain-51-tok-s.md` shelved async pipelining with an explicit condition:
*"It becomes worth re-testing only when BOTH (a) the draft runs on a GPU and
(b) there is real WAN RTT to hide."* ijru moving sites supplied (b); the ROCm
`llama-cpp-python` build supplied (a). Both held for the first time.

## The measurement — 96 tokens, same prompt, same chain, ~181 ms link

| decode path | wall | tok/s | client CPU |
|---|---|---|---|
| plain streaming | 21.2 s | **4.52** | 0.7 s |
| sequential speculative, GPU draft | 22.2 s | 4.32 | 3.5 s |
| **async pipelined, GPU draft** | 23.5 s | **4.08** | **9.6 s** |

All three produced byte-identical output. Pipelining is **10% slower than plain
streaming** while burning **14× its client CPU**, with every precondition met.

**Verdict: dead for this stack.** Not "off by default pending a better regime" —
the regime arrived and it lost.

## The precondition was wrong, and that is the actual finding

The original verdict assumed the technique needed *latency to hide*. It doesn't.
Async pipelining overlaps **draft work** with the **far stage's compute** — it
starts drafting round N+1 while the far worker is still computing round N. What
it hides is the far stage's *processing time*.

It cannot hide RTT, because the round trip is precisely what it is waiting on.
Pipelining does not remove round trips; it only starts the local draft earlier.
And the GPU draft is now ~21 ms while one round trip is ~181 ms — so there is
nothing left to overlap. Every millisecond it saves on drafting is invisible
behind a wait it cannot shorten, and the redundant speculative re-proposals it
pays for that overlap are pure loss.

So the condition should never have been "real WAN RTT". It should have been
**"a far stage whose compute time is large relative to the draft"** — a slow
*worker*, not a slow *link*. On this stack the far stage is 11 ms of GPU work.
There was never anything to hide behind, on the LAN or across the country.

## Why this matters beyond one flag

This is the second technique in this project whose textbook justification
survives contact with the code and dies on the arithmetic — after speculative
decoding, which assumes verifying k tokens costs about what verifying one costs
and fails on a 3B-active MoE. Both failures share a shape: **a technique that
amortises X is worthless when X is not where the time goes.** Speculation
amortises weight streaming; pipelining amortises far-stage compute. This chain
spends its time on neither — it spends it on the wire.

`NKS_ASYNC_PIPELINE` stays default OFF. Correctness remains proven (byte
identical), so the code stays; nothing here argues the implementation is wrong.
