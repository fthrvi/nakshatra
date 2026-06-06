# Nakshatra Research Findings: Topology & Resilience

A running collection of papers analyzing distributed systems resilience patterns relevant to Nakshatra's worker mesh, peer discovery, and failure recovery.

---

## Paper 1: The Forgiving Tree (arXiv 0802.3267)

**Citation:** Tom Hayes, Navin Rustagi, Jared Saia, Amitabh Trehan (2008). "The Forgiving Tree: A Self-Healing Distributed Data Structure." [abs](https://arxiv.org/abs/0802.3267)

### Core Problem
Self-healing in peer-to-peer networks under repeated attack by an **omniscient adversary** (worst-case attacker with full system knowledge). Each round: adversary deletes one arbitrary node; network responds by adding new edges to stay healthy.

### Key Results
1. **Bounded diameter growth:** Network diameter never exceeds $O(\log \Delta)$ times its original diameter, where $\Delta$ = max node degree initially. For polylogarithmic $\Delta$ (typical P2P), this translates to $O(\log \log n)$ — minimal degradation.
2. **Bounded degree:** Any node's degree grows by **≤3** over its original degree, preventing hotspot overload.
3. **Efficient per-round cost:** $O(1)$ latency per round; each node sends/receives $O(1)$ messages/round.
4. **Setup:** $O(\text{diameter})$ latency; each node sends $O(\log n)$ messages per incident edge (acceptable one-time cost).

### Relevance to Nakshatra

**Direct:**
- **Worker mesh resilience:** Nakshatra's 5-machine cluster + future lab nodes form a mesh. When a worker fails (or a layer becomes unavailable), the forgiving-tree approach offers a **topology-independent** repair: add edges locally rather than re-registering globally with Sthambha.
- **Bounded resource growth:** Key constraint — workers have limited bandwidth/CPU. The $O(1)$ message/round + degree-of-3 growth ensures repairs don't become a bottleneck (compare to naive mesh healing, which might flood all nodes).
- **Orthogonal to admission control:** This complements Sthambha's layer-cache + planner. Registry can focus on *what* compute is available; topology layer ensures *connectivity is maintained* under continuous churn.

**Potential implementation hooks:**
- Worker heartbeat → detect node deletion candidate.
- Local repair: identify $O(1)$ new peers to connect to (via Sthambha's registry) rather than full re-mesh.
- Verify: degree stays bounded, latency to all workers stays $O(\log \log n)$-competitive.

### Status
**Verdict:** SOLVING (for Nakshatra mesh resilience v0.2+). Orthogonal to current v0.1 (which assumes stable 5-machine cluster); becomes critical as we scale to 10+ nodes or enable dynamic worker exit/reentry.

### Open Questions
- How does this interact with **sub-GGUF layer distribution**? When a layer dies on worker-B, do we re-fetch from peer directly (topologically repaired) or re-shard via coordinator?
- **Asynchronous / batched deletes:** Paper assumes sequential single-node deletes. Real failures are bursty. Can we adapt to $k$-joint-deletes per round?

---

## Meta: Adding Papers

To add a new paper, append a new `## Paper N: ...` section following the template above:
- **Citation** + link
- **Core Problem** — 2–3 sentences
- **Key Results** — numbered, quantitative
- **Relevance to Nakshatra** — specific hooks
- **Status** — SOLVED / SOLVING / BENEFICIAL (per /papers skill)
- **Open Questions** — bridging gaps for our stack

