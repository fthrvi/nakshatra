# Nakshatra North Star — The Four-Layer Vision

**Date:** 2026-05-06
**Status:** Vision document — not an implementation spec. Sibling to [`petals-architecture.md`](petals-architecture.md) (the v0.1 spec) and [`path-a-vs-path-b-memo.md`](path-a-vs-path-b-memo.md) (C++ feasibility). This doc deliberately lives outside `petals-architecture.md` to keep v0.1's scope tight; if a v0.1 design decision conflicts with anything in this doc, this doc yields, not v0.1.

---

## Why this doc exists

`petals-architecture.md` §0 is explicit and load-bearing: **Nakshatra is a distributed inference engine. That is the entirety of its scope.** That discipline is what makes v0.1 shippable.

But the working hypothesis behind investing in Nakshatra at all is bigger than "an inference engine." This doc captures the bigger hypothesis without contaminating the v0.1 scope. It says: *if* v0.1 ships, *if* the network grows past a private alpha, here is the shape we are building toward, and here are the v0.1 invariants that keep that shape possible.

If this doc and `petals-architecture.md` ever conflict, `petals-architecture.md` wins. This is the dream; that is the contract.

---

## The four-layer stack

```
[L4]  Agent applications              ← Prithvi + third-party agents
[L3]  NakshatraOS                     ← runtime: scheduler, registry, identity, IPC
[L2]  Nakshatra (inference engine)    ← v0.1 — what we are shipping first
[L1]  Heterogeneous compute swarm     ← anyone's hardware: AMD / NVIDIA / Apple / Intel-Mac / CPU
```

**L1 — the swarm.** Anyone with hardware joins as a worker. v0.1 targets the consumer hardware tier (RX 9070 XT, AMD Radeon Pro 5700 XT / Vega 56 on Intel Macs via Vulkan-MoltenVK, ROCm Linux). Future versions extend to anyone with a llama.cpp-supported backend.

**L2 — Nakshatra.** A distributed inference engine. One large model (or several) split across the swarm. Pipeline-parallel across vendors, tensor-parallel only within a vendor (`petals-architecture.md` §4.8). v0.1 ships this layer and stops here — no L3, no L4.

**L3 — NakshatraOS.** A runtime layered above Nakshatra. Provides:

- A **model registry** — many models live in the network simultaneously, each addressable by id and content hash
- An **agent scheduler** — long-running agents are placed on nodes, can migrate, can persist state
- **Content-addressable storage** — model weights, agent state, shared blobs (IPFS-shaped, but inside the swarm)
- **Identity and permissions** — agents act on behalf of users; permission grants are scoped
- **Inter-agent IPC** — message-passing between agents, possibly across users

L3 does not exist in v0.1 and does not exist in any current design draft. It is the next major architectural commitment after Nakshatra itself proves out.

**L4 — agent applications.** Prithvi is the first. Many more become possible once L3 is real (see ["What this opens up"](#what-this-opens-up--for-third-party-developers) below).

---

## What NakshatraOS is and isn't

**Is:** a *distributed runtime* in the Erlang/BEAM, Ray, and Plan 9 lineage. Lightweight processes (agents) live on whatever node the scheduler picks. State and memory are network-transparent but not network-pretend. Failure is explicit, not hidden. The user-facing experience can *feel* like one computer; the implementation does not pretend it is one.

**Isn't:** Linux for the network. Single-system-image — pretending shared memory and shared filesystem and microsecond IPC across the network — is the trap that buried Mosix, Plan 9 (in the mainstream), Inferno, K42, and a dozen others. The latency, partition, and consistency realities of a network *cannot* be hidden behind a kernel-style abstraction without breaking real workloads. NakshatraOS embraces distribution; it does not pretend to dissolve it.

The "supercomputer" framing is poetically powerful and operationally misleading. The right framing is **"a programmable distributed runtime where agents and models are first-class citizens."**

---

## Road and cars

The metaphor that makes this stack click: **Nakshatra is the road, agents are the cars, NakshatraOS is the traffic system.**

- The road (Nakshatra) does not know or care what cars exist. It just provides the surface.
- Cars (agents) follow rules they did not write but can use them to coordinate.
- The traffic system (NakshatraOS) enforces lanes (capabilities), signals (scheduling), and addresses (registry).

This separation is the architecture's most important property. It is *why* Prithvi and Nakshatra are separate projects. It is *why* a third-party developer can build a code agent or a research agent without rewriting Nakshatra. It is *why* a hardware contributor can join the swarm without endorsing any particular agent.

---

## What this opens up — for third-party developers

The L3 surface is what makes this useful to anyone other than us. Once NakshatraOS exists, here is what becomes buildable that is hard or impossible today.

### Long-horizon agents (the strongest case)

Today's hosted-AI economics punish agents that run for days. A code-review agent that watches a repo continuously, a research agent that crawls and summarizes a topic over a weekend, a trading or monitoring agent that observes streams 24/7 — these all break either on cost or on cold-start. NakshatraOS's scheduler can place a long-running agent on contributed compute and migrate it on failure; the agent does not care which physical node it is on.

This is the category with no good home today. It is the strongest reason to build this stack.

### Personal AI that survives the device

A user's agent — Prithvi-class or otherwise — lives in the network, not in an app. Reinstalling, switching laptops, switching phones does not lose it. State syncs through L3's content-addressable layer; identity comes from L3's identity primitive. This is the property that lets "your AI" actually be yours, not a tenant of someone else's product.

### Inference-as-a-service for indie apps

A small developer wants to ship a chatbot, a code helper, a research tool. Today: pay OpenAI/Anthropic, or self-host a model, or both. NakshatraOS adds a third option: call any model in the network's registry, pay in NRN if it is a public network, pay in nothing if it is your own private cluster. Llama-405B, Qwen-72B, future open frontier models become accessible from a $0/month app.

### Federated micro-clusters for small orgs

A startup's team has 8 laptops and 2 desktops. A research lab has a few iMacs. A maker collective has a few PCs. NakshatraOS lets them pool that hardware into a private inference cluster — same software stack as the public network, scoped to their group. No AWS bill, no SaaS dependency for the AI surface. v0.1's "trusted operators only" trust model (`petals-architecture.md` §5) is exactly what a private cluster wants.

### Agent-to-agent federation

The IPC primitive (NakshatraOS message-passing) makes "my agent talks to your agent" tractable. Cross-service agents — a tax agent talking to a bank agent talking to an accountant agent — stop needing a custom integration per pair. The substrate provides the wire format; agents implement protocols on top.

This is the primitive most missing from today's AI stack. Every agent product reinvents it, badly.

### Surface integrators

A developer wants to bring Prithvi (or another agent) to Cursor, to Telegram, to a shell, to a phone, to a VR headset. Today, each integration costs the agent team an integration. NakshatraOS makes the agent's identity, memory, and state available to any surface that authenticates against L3 — the integration cost drops from O(N×M) to O(N+M). N agents × M surfaces becomes "any surface implements the L3 client; any agent runs on L3."

### Model publishers

Fine-tuned models, LoRAs, specialty models (legal, medical, code, music, low-resource languages) become publishable to the network's registry. Like NPM for models. A model that does not make sense on a hyperscaler — too niche, too small a market — can find its audience here because hosting cost is shared across the swarm.

### Hardware contributors

Anyone with a GPU joins the swarm. Lab clusters opt in for student access. Home gamers contribute idle cycles. Enterprise idle compute (weekends, holidays) becomes monetizable. The contribution is a one-line installer and a YAML config — same shape as joining a BitTorrent swarm, with NRN-denominated earnings on a public network.

### Agent skill ecosystem

Skills as installable agents. A user's Prithvi composes its capabilities by installing skills from a registry: a research skill, a code skill, a music skill, a domain-specific skill. Each skill is itself an agent with its own memory and state. The substrate makes the composition work.

This is roughly Cursor's MCP ecosystem evolved one layer up: not tools-called-by-an-agent, but agents-installed-into-an-agent.

### Tools and infrastructure

The substrate creates demand for: observability dashboards (uptime, throughput, latency across the swarm), scheduling research (placement algorithms), verification (redundant compute, attestation, reputation), debugging tools (distributed-trace for a multi-hop inference call). These are real engineering products with real customers — anyone running on the network needs them.

---

## What this does NOT enable (honest limits)

If we let the vision sound bigger than it is, we lose credibility and we make worse design decisions. The honest limits:

- **This is not AGI infrastructure.** It runs whatever models exist. It does not improve them. The model is a separate project.
- **It is not "private by default."** A request flowing through the swarm passes through workers you do not control. Real privacy requires the private-cluster mode (one trusted operator's machines), or future zkML/TEE work that is years out.
- **It is not always cheaper than cloud.** OpenAI's bulk economics are formidable. The swarm wins when workload duration matters more than peak throughput (long-horizon agents) or when avoiding vendor lock-in matters (sovereignty). It loses on bursty single-shot inference at scale.
- **It does not replace frontier closed models.** GPT-5, Claude-5, Gemini-3 will keep being a generation ahead of the best open models for the foreseeable future. The network serves *open* models. Apps that need frontier closed models still call those APIs.
- **Latency sensitivity is real.** Pipeline-parallel hops add round-trips; the activation-transport math (`petals-architecture.md` §4.1) is generous on bandwidth but not on latency. Real-time voice, low-latency UI completion, and similar workloads may not be a fit.
- **Bootstrapping is the hardest part.** A network without nodes is useless. The first 100 nodes is the hardest milestone. This is not a footnote; it is the existential question.

A vision that admits these honestly is one we can build to. A vision that papers over them produces a marketing brochure and no working code.

---

## v0.1 invariants that protect this future

If we want NakshatraOS to be possible later without rewriting Nakshatra, v0.1 must hold these lines. Most are already correct in `petals-architecture.md`:

1. **Multi-model in the protocol from day one.** `ModuleUID` strings (`petals-architecture.md` §1.2) namespace cleanly across multiple models. Do not accidentally bake `model_id` as a single global value anywhere in the worker or client.
2. **Multi-tenant inference sessions stay multi-tenant.** `rpc_inference` is per-`session_id` (§2.4) and one server can host many sessions concurrently. Resist any "simplify for v0.1" pressure that flattens this.
3. **Capability handshake stays rich.** §8.6's design — content hash, layer range, supported quants, KV headroom — is what a future scheduler reads from each worker. Resist temptation to flatten it into a thin "is-it-up?" check.
4. **Static config is replaceable, not architectural.** §8.1's YAML is fine for v0.1's trust model, but the worker code must not assume "I am pinned to layers [16, 32) forever." A future scheduler will reassign at runtime.
5. **One worker, multiple models eventually.** v0.1 ships single-model for sanity, but the worker abstraction should not hardcode "the model" as a singleton field. Ground the worker on `ModuleUID` from day one.
6. **The protocol does not know about agents.** L3's existence must be invisible to L2. Nakshatra never imports an agent concept. This is `petals-architecture.md` §0 again, restated.

Architecture decisions during v0.1 that touch any of these should be reviewed against this list.

---

## Open questions before NakshatraOS-v0

These do not need answers now. They need answers before NakshatraOS-v0 starts implementation.

- **Identity layer.** Prithvi already has Ed25519 + mnemonic. Does NakshatraOS adopt that pattern, or define its own and let Prithvi be a client? (Probably the former — "user" is not a Nakshatra concept.)
- **Payment / accounting.** NRN on Substrate? Off-chain credits with periodic settlement? Punt entirely until a public network exists?
- **Permission model.** Capability-based (object-capability tokens), OAuth-shaped (scoped grants), or something else? This shape determines the third-party developer experience more than any other decision.
- **Model registry governance.** Who decides what models get ingested? Reputation? Curation? Open registry with content-hashing only?
- **Multi-tenant trust model.** v0.1 assumes trusted workers. NakshatraOS at public-network scale assumes the opposite. The architecture for that transition (verification, slashing, sandboxing) is a multi-year research direction (`petals-architecture.md` §5 surveys the options).
- **Sandboxing for third-party agents.** If anyone can publish an agent that runs on the network, what isolation guarantees does NakshatraOS provide? wasm? Process boundaries? Trusted enclaves where available?

---

## Roadmap markers (sequence, not schedule)

| Stage | What ships |
|---|---|
| Nakshatra v0.0 | Validation experiments. Decision gate for v0.1. |
| Nakshatra v0.1 | Two-iMac MVP. LlamaCppBackend Path B-prime. Single model. Trusted operators. (`petals-architecture.md` §7-9) |
| Nakshatra v0.5 | Dynamic re-routing, server-to-server push. Larger private clusters. |
| Nakshatra v1.0 | DHT, public-network mode, opt-in verification. |
| **NakshatraOS v0** | First L3: model registry on top of Nakshatra v1.0. Many models, one network. |
| NakshatraOS v0.5 | Agent runtime: spawn, migrate, persist. Long-horizon agents become real. |
| NakshatraOS v1.0 | Developer SDK. Third-party apps. Identity + permissions + IPC. |
| Prithvi (orthogonal) | Integrates whichever Nakshatra/NakshatraOS layer is current and stable. Today's BYOK Claude → tomorrow's NakshatraOS-hosted. |

The dotted line is between Nakshatra v1.0 and NakshatraOS v0. Anything before that line is the inference engine alone. Anything from NakshatraOS v0 onward is the L3 commitment. Crossing that line is the hardest architectural decision the project will face — make it deliberately.

---

## How this doc is used

- When designing a v0.1 feature, ask: *does this break any of the six invariants above?* If yes, redesign.
- When tempted to import an L3 concept (model registry, identity, agent, scheduler) into v0.1, stop. That belongs in NakshatraOS.
- When evaluating a v0.5 or v1.0 feature, ask: *does this move us toward NakshatraOS or sideways?*
- When pitching the project externally, lead with v0.1 (the inference engine) and reference this doc as the trajectory. Do not lead with NakshatraOS — that is vapor until v0.1 ships.

The architecture doc is the contract. This doc is the trajectory. Do not confuse them.
