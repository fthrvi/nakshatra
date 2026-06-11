# Nakshatra vs. Mesh-LLM — design & architecture

**What this is:** an honest side-by-side of Nakshatra and **Mesh-LLM**
([github.com/Mesh-LLM/mesh-llm](https://github.com/Mesh-LLM/mesh-llm) — Rust,
iroh-QUIC, vendored llama.cpp, Apache-2.0), the first credible independent
occupant of Nakshatra's slot. Basis: the 45-agent fact-checked deep-dive
(`trisul/research/2026-06-09-mesh-llm-distributed-inference-deepdive.md`, 38
claims verified, 18 corrected) and the v1.0 work it informed
([`v1.0-discovery-and-distribution.md`](v1.0-discovery-and-distribution.md)).

The short version: **Mesh-LLM is a networking achievement on stock llama.cpp with
an open trust model. Nakshatra is an engine-and-measurement achievement with a
closed, signed, compute-aware mesh.** Their weaknesses are our strengths and
vice-versa — which is exactly why the v1.0 plan was *borrow their mechanism, keep
our wall.*

---

## 1. The shared bet (where we are the same)

Both teams reached the same core architecture independently — which is itself
evidence it's right for the regime (WAN-tolerant single-stream large-model
inference):

- **Pipeline-split by contiguous layer ranges**, not tensor-parallel. Each node
  holds a block of adjacent transformer layers.
- **Weights stay local; only activations cross the wire** (~16 KB/token for a
  70B). The link is *not* the binding constraint.
- **Single-node-first**: if one machine can hold the whole model, it serves
  locally with no split traffic.
- **GGUF / llama.cpp** as the compute substrate.
- **OpenAI-compatible surface** (`/v1/chat/completions`) so existing clients work.
- **Discovery ≠ admission ≠ transport** as three separable layers — Nakshatra
  *adopted* this decoupling from Mesh-LLM (P1).
- **Content-addressed layer packages** — per-layer GGUF fragments + a SHA-256
  manifest. Nakshatra's P2 adopts Mesh-LLM's `model-package.json` shape nearly
  wholesale.

## 2. Where we diverge (the real differences)

| Dimension | Mesh-LLM | Nakshatra |
|---|---|---|
| **Headline strength** | *Networking* — Nostr discovery, iroh-QUIC NAT traversal, content-addressed packages, on **stock** llama.cpp | *Measurement + engine depth* — a ~70-LOC partial-load patch, the only reproducible RDNA4/ROCm numbers, a compute-aware planner |
| **Trust model** | **Open public mesh.** Relay/worker nodes see **plaintext prompts** (QUIC transport-only, no E2E); listings effectively unsigned; `api-key:"mesh"` open default | **Closed, signed, pinned.** Ed25519-signed RPC, SPKI/TLS-pinned, admission-gated. Listings **self-signed**; admission **pins** against the advertised key |
| **Peer ranking** | `score_mesh` ranks by **RTT** (latency) | Ranks by **measured compute Fᵢ** (decode-ms/layer) — peer selection inherits the planner's compute-awareness. The move Mesh-LLM structurally can't make |
| **Weight provenance** | Manifest = SHA-256 **integrity only** | Manifest = SHA-256 **+ Ed25519 signature** — provenance bound to the same identity that signs the mesh |
| **Throughput** | **No published numbers**; "certified parity" is internal self-validation | Measured + published ("Compute, Not the Wire", Zenodo DOI 10.5281/zenodo.20514966); §10 byte-identical-token proven on real binaries |
| **Confidentiality** | Unsolved (plaintext to relay) | A *feature* — the closed wall is deliberate, not a gap |
| **Mixture-of-Agents** | Ships it (`model:"mesh"` fan-out + reducer) | **Skips it** — spends compute (the scarce resource), re-opens prompt exposure; the job is one fast single-stream 70B |
| **Wire versioning** | Versioned protobuf control plane, ALPN-negotiated, legacy fallback | Adopted the *pattern*; added explicit control-version **negotiate-or-reject** at discovery and the live handshake (P4) |
| **Maturity gaps** | Has discovery, NAT traversal, FT, MoE, batching | Had none of those at the start; v1.0 added discovery + packages + routing + versioning |

## 3. The deepest Nakshatra-specific edge

The micro-benchmark localized **86% of a 3.7 ms GPU round-trip to a GPU→host DMA
readback** (`llama_get_embeddings`) on the RX 9070 XT / ROCm — the residual
on-node cost is **DMA, not matmul or wire**. Nobody else benchmarks this hardware.
This is the next structural lever (Toyota **SMED**: keep hidden-state on-device
between co-located stages / fuse the boundary copy), and it is *ours* to own —
unreachable by a networking-only approach. (Single-cluster, unreplicated; it
inverts in CPU mode — replicate before optimizing on it.)

## 4. How the difference shaped v1.0 — *borrow the mechanism, keep the wall*

Every P-item took a Mesh-LLM idea and re-cut it to Nakshatra's posture:

| | Mesh-LLM idea | What Nakshatra did differently |
|---|---|---|
| **P1** | Nostr discovery | Listings **Ed25519-signed**; admission **pins** the advertised key (closes the unsigned-listing / open-join gap). Rank by **compute Fᵢ**, not RTT. **Two keys, two jobs:** secp256k1 signs the relay event (anti-spam), Ed25519 signs the listing (admission) |
| **P2** | Layer packages | Manifest **signed** + content-revision; provenance bound to mesh identity. Same byte-range fetch path; superset format |
| **P3** | OpenAI entry-proxy | Signed forward **behind the wall**, not `api-key:"mesh"` open default |
| **P4** | Two-plane discipline | Added control-version **negotiation** (data plane already hard-rejects); pre-join filter + live handshake |
| **P5** | Mixture-of-Agents | **Declined** — orthogonal to single-stream 70B; spends compute; re-opens exposure |
| **P6** | `wanted` compute-market signal | **Watch** — park until proof-of-work receipts land |

What we **deliberately did not copy**: the open-join public mesh and plaintext-to-
relay. Mesh-LLM's confidentiality is unsolved; importing its discovery model
naively would import that weakness. So discovery is public gossip; *joining and
serving* stay signed, pinned, and closed (`THREAT_MODEL.md`).

## 5. Open questions on both sides

- **MoE across machines is unsolved on both sides** — Nakshatra's collapses
  (FP-drift flipping top-k gating); Mesh-LLM claims per-expert distribution with
  no reproducible cross-machine MoE throughput.
- **Fault tolerance** — Mesh-LLM has churn handling; Nakshatra's next design doc
  is Petals-style O(t) dual-cache recovery (a churning mesh of strangers
  *requires* per-token recovery, and recovery must be **drift-class-constrained**
  — recovering a span onto a different-vendor peer silently diverges generation).
- **Sybil / false advertising** on public discovery survives signed listings (a
  key can still lie about VRAM) — verify capacity with a probe, don't trust it.

---

*Sources: Mesh-LLM repo + docs (docs.anarchai.org), DESIGN.md, message_protocol.md,
SKIPPY_SPLITS.md, LAYER_PACKAGE_REPOS.md, MOA_GATEWAY.md; Nakshatra v1.0 design
doc; "Compute, Not the Wire" (Zenodo).*
