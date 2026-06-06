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

## Paper 2: Peer-to-Peer Secure Multi-Party Numerical Computation (arXiv 0810.1624)

**Citation:** Danny Bickson, Genia Bezman, Danny Dolev, Benny Pinkas (2008). "Peer-to-Peer Secure Multi-Party Numerical Computation." [abs](https://arxiv.org/abs/0810.1624)

### Core Problem
Enable **privacy-preserving collaborative computation** in large-scale P2P networks. Multiple nodes want to jointly compute a function (e.g., recommendations, trust scores) over their *secret* inputs without revealing individual data. Classical secure multi-party computation (SMC) is theoretically sound but impractical at P2P scale (millions of nodes).

### Key Results
1. **Scalable SMC framework:** Identify a single approach (among several candidates) that is both **theoretically secure** and **practically scalable** to networks of millions of nodes and hundreds of millions of edges.
2. **Netflix-prize algorithm implemented:** Successfully compute neighborhood-based collaborative filtering (a state-of-the-art recommendation algorithm, Netflix Progress Prize 2007) in a P2P network with **zero loss of accuracy** while preserving input privacy.
3. **Large-scale validation:** Extensive simulations on real Internet topologies demonstrate applicability; claimed to be first large-scale SMC simulation at this scale.
4. **Practical secret-keeping:** Nodes collaborate without revealing their private state to peers or a central coordinator.

### Relevance to Nakshatra

**Direct:**
- **Distributed routing decisions:** Currently, Sthambha (L3) is the central registry deciding worker assignments. With SMC, workers could **collectively decide peer routing** (who should talk to whom for layer fetching, bandwidth balancing) without a central authority, while each worker keeps its internal state (load, available bandwidth, reliability metrics) secret.
- **Trust & reputation computation:** Instead of a central reputation score, workers collaboratively compute mutual trust (e.g., "Worker-B is reliable for Layer-X") without exposing individual request histories.
- **Collaborative layer caching:** Nodes could agree on shared cache eviction policies and layer placement without revealing cache contents or access patterns.

**Potential implementation hooks:**
- **Phase 1 (v0.2+):** Prototype SMC for a single decision (e.g., "which worker should fetch Layer-5?") among a small subset (3–5 workers).
- **Phase 2:** Extend to full routing layer, replacing central Sthambha decisions with Worker collective agreements.
- **Privacy guarantee:** No worker learns another's latency, available VRAM, or compute capacity unless explicitly revealed.

### Status
**Verdict:** BENEFICIAL (deferred, strategic architecture shift). Solves a real problem (central coordinator is a single point of failure + privacy leakage), but requires significant refactor of Sthambha's decision pipeline. Current v0.1 assumes trusted Sthambha; this enables v1.0+ migration to peer-driven governance.

### Open Questions
- **Latency overhead:** SMC adds communication rounds. What's the actual $O(?)$ cost per decision vs. single-round registry query? Is it acceptable for layer-fetch routing (low frequency) vs. per-token routing (high frequency)?
- **Byzantine workers:** SMC assumes **honest-but-curious** adversaries. What if a worker actively lies about its capacity or tries to game route assignments? Extend with Byzantine-resilient SMC (e.g., PBFT over SMC)?
- **Dynamic membership:** Paper assumes static node set. How do workers join/leave without breaking the SMC protocol state?

---

## Paper 3: M-Banking Security - Session & Request ID Framework (arXiv 1002.1174)

**Citation:** Geeta S. Navale, Swati S. Joshi, Aaradhana A. Deshmukh (2010). "M-Banking Security — A Futuristic Improved Security Approach." [abs](https://arxiv.org/abs/1002.1174)

### Core Problem
Prevent **replay attacks** and **unauthorized request execution** in distributed transaction systems. Even with encryption, an attacker can intercept and resend valid requests multiple times or impersonate a legitimate user across session boundaries. Need mechanisms to ensure each transaction executes exactly once and only authenticated entities can trigger requests.

### Key Results
1. **Session ID framework:** Unique session tokens issued per authenticated login/connection. Sessions expire or can be revoked, preventing reuse of stolen credentials.
2. **Request ID mechanism:** Per-transaction unique identifiers (combining timestamp, nonce, request sequence). Each Request ID is processed exactly once; duplicates are rejected.
3. **Double-layer idempotency:** Together, Session ID (session-level auth) + Request ID (transaction-level deduplication) prevent both impersonation and replay within a session.
4. **Steganography integration:** Optional data-hiding technique (not core to the ID framework, but combined for defense-in-depth).

### Relevance to Nakshatra

**Direct:**
- **Worker-to-coordinator authentication:** When a worker submits an inference task (e.g., "process tokens for layer-5 of llama-70b"), Sthambha pillar must verify:
  - Is this worker's session still valid? (Session ID check)
  - Have I already executed this exact task? (Request ID check → prevents double-execution)
- **Fault tolerance:** Worker crashes during token processing → retransmits same request. Without Request ID deduplication, the pillar would execute twice, wasting compute and producing inconsistent outputs.
- **Replay attack resistance:** If a worker's credentials are compromised and attacker gains access to old Session ID + Request ID pairs, they cannot replay — old Session ID is invalidated on disconnect, and Request IDs are already marked executed.

**Potential implementation hooks:**
- **Worker registration:** Each worker gets a `session_id` on startup (cryptographic token).
- **Task submission:** Each inference task includes:
  ```python
  {
    "session_id": "worker_node_b_session_xyz",
    "request_id": "inference_llama70b_token_5893_ts_1717689600_nonce_8f3a",
    "model": "llama-70b",
    "tokens": [...],
  }
  ```
- **Sthambha pillar logging:** Maintain `{request_id: execution_result}` map; return cached result on duplicate.
- **Session timeout:** After 1 hour (configurable), invalidate `session_id`, force re-authentication.

### Status
**Verdict:** SOLVING (critical for v0.1+ robustness). Session/Request ID are **industry standard** (HTTP cookies, JWT, AWS request signing all use this pattern). Nakshatra v0.1 should implement immediately for worker fault tolerance. Low implementation cost; high safety gain.

### Open Questions
- **Request ID TTL:** How long do we keep completed Request IDs in the dedup map? (Infinite = memory leak; too short = window for replay.) Suggest: keep for 24 hours or until worker reconnects.
- **Clock skew:** Nonce-based Request IDs assume loosely synchronized clocks. If a worker's clock drifts, it might generate duplicate nonces. Solution: include worker ID + sequence counter instead of timestamp.
- **Distributed pillar:** If Sthambha has multiple pillar replicas (v1.0+), how do they share the Request ID dedup map? Consensus (Raft/Paxos) or eventual consistency with conflict resolution?

---

## Meta: Adding Papers

To add a new paper, append a new `## Paper N: ...` section following the template above:
- **Citation** + link
- **Core Problem** — 2–3 sentences
- **Key Results** — numbered, quantitative
- **Relevance to Nakshatra** — specific hooks
- **Status** — SOLVED / SOLVING / BENEFICIAL (per /papers skill)
- **Open Questions** — bridging gaps for our stack

