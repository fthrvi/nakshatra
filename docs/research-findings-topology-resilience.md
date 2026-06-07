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

## Paper 4: Parallel Gaussian Process Regression with Low-Rank Approximations (arXiv 1305.5826)

**Citation:** Jie Chen, Nannan Cao, Kian Hsiang Low, Ruofei Ouyang, Colin Keng-Yan Tan, Patrick Jaillet (2013). "Parallel Gaussian Process Regression with Low-Rank Covariance Matrix Approximations." [abs](https://arxiv.org/abs/1305.5826)

### Core Problem
**Gaussian Processes (GP) cannot scale** to large datasets due to cubic time cost $O(n^3)$ in inverting the $n \times n$ covariance matrix. For real-time routing/prediction in distributed systems, we need a scalable approach that:
1. Reduces computation from $O(n^3)$ to practical time
2. Distributes the work across parallel machines
3. Maintains predictive accuracy close to "full" GP trained on all data

### Key Results
1. **Low-rank covariance approximation:** Replace full $n \times n$ matrix with $n \times m$ approximation using $m$ "landmark" points ($m \ll n$). Reduces local computation from $O(n^3)$ to $O(n \cdot m^2)$.
2. **Two distributable methods:**
   - **Fully Distributed GP (FD-GP):** Each machine trains locally on data chunks, coordinator aggregates predictions. Minimal communication, $O(n^3/p^3)$ per machine (where $p$ = number of machines).
   - **Iterative GP (IG-GP):** Machines collaboratively refine shared landmark points over multiple rounds. Better accuracy than FD-GP, higher communication cost.
3. **Theoretical guarantee:** Distributed GP predictions are **equivalent to centralized approximate GP**, ensuring no degradation from distribution.
4. **Empirical validation:** 20-node cluster demonstrates linear speedup (2x nodes ≈ 2x speedup) while maintaining FG-equivalent accuracy on real datasets.

### Relevance to Nakshatra

**Direct:**
- **Distributed performance prediction:** Each worker can train a local GP from its inference logs (latency, throughput, cache hits) without centralizing data. Coordinator aggregates worker GPs to predict "which worker should handle next token?" without seeing raw logs.
- **Privacy + scalability:** Workers never share raw telemetry, only compact low-rank GP models (~10s of KB each). Coordinator combines them for system-wide routing decisions.
- **Adaptive routing:** As workers' performance changes (e.g., one worker gets overloaded), its GP updates locally, changes propagate to coordinator within one aggregation cycle — no sync delays.

**Potential implementation hooks:**
```python
# Per worker: train low-rank GP locally
worker_gp = LowRankGP(
    X=worker_logs[["queue_depth", "token_pos", "model"]],
    y=worker_logs["latency_ms"],
    num_inducing_points=50  # m=50 << n (thousands of logs)
)
# Sthambha coordinator: aggregate worker GPs
def predict_latency(query):
    predictions = [worker_a_gp.predict(query),
                   worker_b_gp.predict(query),
                   worker_c_gp.predict(query)]
    return aggregate(predictions)  # mean or weighted by trust
```

- **Phase 1 (v0.2+):** Each worker maintains a lightweight low-rank GP of its own performance.
- **Phase 2:** Coordinator uses aggregated GPs to drive adaptive layer-fetch routing (predict which peer will respond fastest).
- **Phase 3:** Workers can use aggregated system GP to predict inter-worker latency for prefetching decisions.

### Status
**Verdict:** BENEFICIAL (strategic, not critical for v0.1). Solves a real problem (centralized telemetry collection) and gives strong theoretical guarantees. Current v0.1 can use simple heuristics (worker availability + round-robin); parallel GP enables v0.2+ adaptive routing with learned worker models. Implementation effort: medium (requires GP library + aggregation logic).

### Open Questions
- **Landmark selection:** How do we choose which $m$ points are "most informative"? Paper doesn't fully specify. Greedy maximin distance? K-means? Adaptive (sample high-uncertainty points)?
- **FD-GP vs IG-GP trade-off:** When is FD-GP sufficient (fast) vs. when do we need IG-GP iterations (slow but accurate)? SLO-driven heuristic needed.
- **Concept drift:** Workers' performance changes over time (new model version, hardware swap). Do we retrain worker GPs hourly? How do we forget old stale data?
- **Non-stationary latency:** Network jitter, thermal throttling, background processes — latency isn't stationary. Do we model with heteroscedastic GP (varying noise per region)?

---

## Paper 5: Dynamic Multi-Level Multi-Agent Simulations (arXiv 1311.5108)

**Citation:** Jean-Baptiste Soyez, Gildas Morvan, Daniel Dupont, Rochdi Merzouki (2013). "A Methodology to Engineer and Validate Dynamic Multi-Level Multi-Agent Based Simulations." [abs](https://arxiv.org/abs/1311.5108)

### Core Problem
Complex distributed systems (multi-scale, multi-domain) waste compute resources simulating **fine-grained detail that's irrelevant to current queries**. Need a methodology to **dynamically adjust granularity** — use lightweight representations during stable operation, zoom in to fine detail only during anomalies/crises — without losing information fidelity.

### Key Results
1. **IRM4MLS meta-model:** Generic agent-based framework supporting multi-level hierarchies. Agents can represent different domains or different scales of the same phenomenon.
2. **Two core mechanisms:**
   - **Activation/Deactivation:** Subsystems (agents) are turned on/off based on operational state. E.g., disable weather-simulation agents if conditions are stable; activate only when needed.
   - **Aggregation/Disaggregation:** Agents at the same level can be fused (aggregated) into coarser summaries during normal operation, then split (disaggregated) into fine-grained individuals during critical periods.
3. **Theoretical guarantee:** Detail level adjustment preserves information fidelity — predictions remain accurate while computation scales with operational need.
4. **No performance loss:** Switching between levels doesn't degrade system predictive quality, only changes resource footprint.

### Relevance to Nakshatra

**Direct:**
- **Adaptive simulation granularity:** Nakshatra's 5-worker cluster can operate in two modes:
  - **Aggregated (normal):** Workers pooled into one "WorkerPool" agent; predict throughput from pool-level statistics (minimal compute).
  - **Disaggregated (crisis):** Individual worker agents; per-token latency heuristics; routing optimization (full compute).

**Practical hooks:**
- **Normal operation:** Sthambha tracks aggregate `{throughput_avg, latency_p95, error_rate}` per pool. Coordinator uses simple round-robin routing. Cost: negligible.
- **Load spike / SLO breach:** Disaggregate → track per-worker `{queue_length, predicted_latency, reliability}`. Activate adaptive routing (Parallel GP from Paper 4). Activate Forgiving Tree monitoring (Paper 1). Cost: scales with urgency.
- **Node failure:** Immediately disaggregate affected cluster. Activate topology repair. Once healed, re-aggregate if remaining workers are healthy.

**Implementation:**
```python
class NakshatraSimulator:
    def __init__(self):
        self.detail_level = "aggregated"
        self.worker_pool = WorkerPool(num_workers=5)
    
    def should_disaggregate(self):
        """Check if we need fine-grained simulation."""
        return (
            self.slo_violation_risk() or
            any_worker_error_spike() or
            network_latency_spike() or
            queue_buildup()
        )
    
    def simulate_token_routing(self, token):
        if self.detail_level == "aggregated":
            # Fast: pool-level prediction
            worker = self.worker_pool.suggest_via_aggregate_model()
        else:
            # Detailed: per-worker heuristics
            workers_sorted = sort(
                [w.predict_latency(token) for w in self.worker_pool.workers]
            )
            worker = workers_sorted[0]  # pick fastest
        
        result = worker.process(token)
        
        # Adapt detail level
        if self.should_disaggregate() and self.detail_level == "aggregated":
            self.detail_level = "disaggregated"
            self.activate_adaptive_routing()
            self.activate_topology_monitoring()
        elif self.is_stable() and self.detail_level == "disaggregated":
            self.detail_level = "aggregated"
            self.deactivate_adaptive_routing()
            self.deactivate_topology_monitoring()
        
        return result
```

### Status
**Verdict:** BENEFICIAL (architectural pattern, medium priority). Solves resource scaling elegantly — Nakshatra stays lean during normal operation (critical for edge deployment), auto-scales observation granularity on demand. Complements Papers 1 (topology) + 4 (adaptive routing) + 3 (session/request dedup). Can be prototyped in v0.2; enables v0.3+ self-adaptive behavior.

### Open Questions
- **Disaggregation trigger SLO:** What latency/error threshold triggers the switch? Too aggressive = always detailed = no savings. Too lazy = misses problems. Heuristic or learned?
- **Activation state persistence:** When disaggregating, do we "cold-start" per-worker models (inaccurate) or warm-start from aggregated statistics (biased)? Hybrid initialization strategy?
- **Multi-granularity stacking:** Can we have 3+ levels (ultra-fine per-token microbenchmarks, medium per-layer, coarse per-model)? How does the framework compose?
- **Information fidelity metrics:** How do we formally verify that aggregation doesn't lose predictive power? Test suite of synthetic SLO violations?

---

## Meta: Adding Papers

To add a new paper, append a new `## Paper N: ...` section following the template above:
- **Citation** + link
- **Core Problem** — 2–3 sentences
- **Key Results** — numbered, quantitative
- **Relevance to Nakshatra** — specific hooks
- **Status** — SOLVED / SOLVING / BENEFICIAL (per /papers skill)
- **Open Questions** — bridging gaps for our stack

