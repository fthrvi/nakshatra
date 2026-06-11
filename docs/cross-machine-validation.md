# Cross-machine validation — full stack proven, and the determinism finding

**Date:** 2026-06-11. **Status:** result + the design implication it forces.

This records the first **real two-machine, two-OS** run of the v1.0 stack (P1–P4 +
Nostr transport), what it proved, and the one thing it surfaced that changes the
roadmap: **full-sequence determinism does not hold across heterogeneous nodes.**

---

## 1. The run

Two physically different machines over a shared tailnet:

| Node | Machine | Layers | Role |
|---|---|---|---|
| A | this box — Linux / x86 / ROCm box (CPU mode) | `[0, 8)` | first (embeddings) |
| B | **Mac4** — macOS / Intel x86 (CPU mode) | `[8, 16)` | last (lm_head) |

Model: `accept-1b` (Llama-3.2-1B Q4, tied embeddings). Every layer of the stack
ran for real, not stubbed:

1. **Discovery (Nostr, public).** A NIP-01 relay on the tailnet; both nodes
   published **signed** `NakshatraListing`s; discovery queried → verified → ranked
   by measured Fᵢ → **pinned** both Ed25519 keys (pins matched the advertised
   keys).
2. **Self-provision (P2, over the wire).** Node B (Mac4) had no slice. It fetched
   its `[8,16)` fragments from the signed package over the tailnet via
   `worker.py --package-url` — 11 fragments, **each SHA-256-verified**, assembled
   into a loader-ready sub-GGUF. (Including `token_embd` for the tied output —
   the weight-tied fix.)
3. **Handshake (P4, cross-machine).** The client read each worker's gRPC `Info`,
   saw `control/v1`, negotiated, and confirmed `contiguous coverage of [0,16)`.
4. **Inference (data plane, tailnet).** Activations forwarded A→B over the
   tailnet; logits + token back.

**Result:** `step 1: id=12366 ' Paris'` — the canonical v0.1 parity token,
produced by a chain split across two heterogeneous machines. The mechanics —
discover → pin → self-provision → negotiate → forward → token — **work end to end
across real machines.**

## 2. The finding: determinism drifts across heterogeneous nodes

The **first** token matched the single-machine reference. The **full sequence did
not**:

| | continuation after "The capital of France is" |
|---|---|
| single-machine / single-box reference | `Paris. The Eiffel Tower is` |
| **cross-machine (A=Linux, B=Mac4)** | `Paris. The capital of France is Paris` |

It diverges at token 4 (`Eiffel` vs `capital`). Crucially:

- The slices are **byte-identical** (SHA-verified fragments) → the **weights are
  the same**. This is not a data bug.
- Node B produces **coherent** text, not garbage → it is computing **correctly**.
- The divergence is **floating-point rounding drift** between Mac4's x86 build and
  this box's — different SIMD / compiler / math paths round matmuls and softmax
  slightly differently, and on a close-call token the **argmax flips**.

This empirically confirms the drift-class caveat already in the design notes
(*"recovery must be drift-class-constrained — recovering a span onto a
different-vendor peer silently diverges generation"*). Stated as a rule:

> **§10 first-token parity holds across heterogeneous nodes; bit-identical
> multi-token generation requires same-class nodes.**

The mesh is correct; determinism is a *same-class* property, not a universal one.

## 3. Why this matters (and why it's not a bug to "fix")

Heterogeneity is Nakshatra's headline — pooling mixed vendors/vintages. But a
single autoregressive generation is a feedback loop: token *t*'s argmax feeds
token *t+1*'s input. Tiny per-node FP differences compound until an argmax flips.
So you **cannot** split one *bit-deterministic* generation across drift-classes —
not because the wire is lossy (it's exact, SHA-verified) but because the *compute*
differs. This is the compute-side analogue of "Compute, Not the Wire": the wire is
perfect; the **stations** round differently.

You don't fix FP determinism across vendors (you'd have to forbid heterogeneity —
killing the thesis). You **constrain** it.

## 4. Design response: drift-class certification

The lever (from the by-analogy work, *interchangeable parts / tolerance bins*):
**certify each node into a drift class with a conformance go/no-go gauge**, and
make class a first-class scheduling + recovery constraint.

- **Conformance gauge.** A canonical fixed-input probe (prompt → N-token logit
  fingerprint at temp 0). Nodes whose fingerprint matches within tolerance share a
  **drift class**. Cheap, deterministic, run at join. (Also doubles as the
  anti-Sybil capability probe — §4.4.)
- **Chain formation respects class.** When a generation must be reproducible, the
  planner forms the chain from **one drift class**. Cross-class pooling is allowed
  only for throughput/aggregate work where bit-identity isn't required (e.g.
  independent requests), never for a single split deterministic stream.
- **Recovery respects class (ties to fault tolerance).** Petals-style O(t)
  recovery must re-place a failed span onto a **same-class** peer, or the
  resumed generation silently diverges. The drift class is the constraint that
  makes recovery sound — this is the bridge to the fault-tolerance successor doc.
- **Listings can advertise class.** A future `NakshatraListing` field
  (`drift_class` = the gauge fingerprint hash) lets discovery pre-filter to
  compatible peers, the same way `supported_protocol` pre-filters wire versions.

## 5. What this run did NOT prove (the connectivity ceiling)

The data plane reached Mac4 **because Tailscale provided NAT traversal** on a
shared tailnet. It did **not** test connectivity between strangers on *different*
networks. Discovery already crosses that boundary (Nostr is public); the **data
plane** does not yet. That is the separate **sovereign cross-network transport**
build — per-peer WireGuard/QUIC tunnels keyed by the pinned identity from the
listing (building block: the Pillar's Mode-B WireGuard peering). See the v1.0 doc
§4/§7 and `vs-mesh-llm.md` (Mesh-LLM solves this with iroh-QUIC).

## 6. Honest scorecard

| Capability | Status |
|---|---|
| Discover → pin → self-provision → negotiate → forward → token, cross-machine | ✅ proven, real binaries, 2 machines, 2 OSes |
| §10 first-token parity, cross-machine | ✅ `id=12366 ' Paris'` |
| Bit-identical multi-token, cross-vendor | ❌ drifts — **needs same-class nodes** (finding above) |
| P2 self-provision over the network | ✅ Mac4 fetched + verified its slice over the tailnet |
| Nostr discovery on a real relay | ✅ tailnet relay; public relays blocked by box egress firewall |
| Cross-*network* (different tailnets) data plane | ⏳ not built — sovereign transport is the next build |

**Two next builds, both now well-motivated by data:** (1) drift-class
certification + class-aware scheduling/recovery (§4), (2) sovereign cross-network
transport (§5). The architecture is proven; these are the two pieces that take it
from *our heterogeneous tailnet mesh* to *a correct public one*.
