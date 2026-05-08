# Three-Project Architecture: Sthambha + Nakshatra + Prithvi

**Date:** 2026-05-08
**Status:** Architectural decision. Sthambha repo not yet created (Phase 1 task). Nakshatra and Prithvi exist today but with current responsibility-tangling that this doc describes how to fix.
**Sibling docs:** [`north-star.md`](north-star.md) (the L1-L4 vision), [`v0.1-implementation-plan.md`](v0.1-implementation-plan.md) (Nakshatra's milestones), [`petals-architecture.md`](petals-architecture.md) (Nakshatra's design).

This doc is the source of truth for the project boundary decisions. If a future session is unsure where a piece of code belongs, this doc decides.

---

## TL;DR

Three projects, one clear responsibility each:

| Project | Layer | What it is | Where it lives |
|---|---|---|---|
| **Sthambha** | L3 — coordination | Registry, pillar (Pi), identity, model/layer cache, peer discovery. The phone book + traffic system of the network. | New repo, not yet created. Phase 1 below. |
| **Nakshatra** | L2 — inference engine | Patched llama.cpp + worker daemon + sub-GGUF tooling + gRPC chain protocol. Takes a model, splits it across workers, produces tokens. | This repo (`bbastola899/nakshatra`). Already exists. |
| **Prithvi** | L4 — agent / being | Consciousness, voice, memory, gateway, OpenAI-compatible API, platform adapters (Telegram/Discord/etc). | `bbastola899/prithvi`. Already exists. |

Plus: **neuron-chain (Substrate blockchain) is paused indefinitely.** Code preserved at `prithvi/neuron-chain/`. Returns in v1.0+ when going public-network with mutually-untrusted contributors. Not before.

---

## Why three projects (not one, not five)

### Why not one big repo

Today Prithvi accidentally absorbed L3 because nothing else existed when those primitives were being built (`pillar/`, `core/registry.py`, `core/model_cache.py`, `core/identity.py`, `core/dht.py`). That's an accident of history, not a design choice. Extracting before more code piles on top is much easier than extracting later.

A single mega-repo also means:
- Anyone using L3 inherits Prithvi's chain dependency
- Anyone using Nakshatra inherits Prithvi's consciousness layer
- New agents (not Prithvi-flavored) have to either fork Prithvi or duplicate L3
- CI for one piece runs CI for all pieces

### Why not five layers

L1 (raw hardware swarm) and L4-derivatives (specific surfaces like Telegram, voice, etc) are real concerns but don't need separate top-level projects. L1 is just "computers exist"; L4 surfaces are submodules of agents.

The three-project split matches the three irreducible concerns:
1. *Where is everyone and what do they have?* → Sthambha
2. *How do we run a model across many of them?* → Nakshatra
3. *What does a user-facing being feel like?* → Prithvi

### What each project owns / disowns

**Sthambha owns:**
- Peer registry (node lifecycle: JOINING → ONLINE → OFFLINE)
- Pillar daemon (lightweight, runs on Pi 5, holds state but doesn't compute)
- Cryptographic identity (Ed25519 keypair = node ID)
- Model + layer cache index (which nodes have which slices of which models)
- Heartbeat / failure detection
- Peer discovery (DHT or central rendezvous, depending on phase)
- Transport primitives (QUIC, NAT traversal)
- Operator CLI (`sthambha registry list`, `sthambha pillar deploy`)

**Sthambha does NOT own:**
- Actual model compute (that's Nakshatra)
- Model selection logic for tasks (that's Prithvi's inference router)
- Conversation state, agent memory, voice (that's Prithvi)
- Token/payment economics (paused — neuron-chain when it returns)

**Nakshatra owns:**
- Patched llama.cpp loader (accepts sub-GGUFs declaring layer ranges)
- Patched llama.cpp graph builder (iterates only owned layer range)
- Worker daemon (`llama-nakshatra-worker`, C++ binary)
- gRPC worker service + proto schema
- Sub-GGUF generation tool (`partial_gguf.py`)
- Chain orchestration client (`scripts/client.py`)
- Streaming KV reuse / per-token routing protocol
- Performance / latency / vendor-portability concerns

**Nakshatra does NOT own:**
- Where workers are or what they're called (that's Sthambha)
- What model to run for a given prompt (that's Prithvi's choice; Nakshatra just runs whatever is requested)
- Multi-model registry, model fetching, content-addressing (that's Sthambha)
- Identity, payments, reputation (Sthambha for identity; payments paused)

**Prithvi owns:**
- The "being" — consciousness primitives (Witness, PAD emotions, Om pulse, DMN, Shakti, Reflection)
- User-facing memory vaults (per-API-key relationship state)
- Voice (STT/TTS), vision, agent tool-calling
- OpenAI-compatible gateway (`/v1/chat/completions`, etc)
- Platform adapters: Telegram, Discord, Signal, WhatsApp
- Inference router (task classification → model choice)
- Soul-contracts and the Sanskrit-named cognition primitives

**Prithvi does NOT own:**
- Where workers are or how to find them (Sthambha)
- How to run a model split across them (Nakshatra)
- L3 plumbing of any kind — it's a *consumer* of L3, not a provider

**neuron-chain (Substrate) ownership: PAUSED.**
- Code preserved at `prithvi/neuron-chain/`
- No new pallets, no new chain features, no new dependencies on chain state
- When public-network mode arrives (v1.0+), this layer returns. Until then, ignore.

---

## How they connect

```
You speak to Prithvi (web, Telegram, voice, etc.)
   │
   │ "Run model X for this conversation"
   ▼
Prithvi router: pick model X based on task type
   │
   │ Query: "Where can I run X?"
   ▼
Sthambha (running on Pi 5):
   - looks up which peers have model X cached, layer-range by layer-range
   - returns a chain plan: e.g. [home-pc 0-13, bishwa 14-27]
   │
   │ "Use this chain"
   ▼
Prithvi → opens gRPC streams to those peers
   │
   ▼
Nakshatra workers (each holding a layer slice):
   - first worker takes tokens, produces hidden state
   - middles ferry hidden state through their layers
   - last produces logits, samples a token
   │
   ▼
Token comes back to Prithvi → user sees response
```

Sthambha doesn't know about consciousness. Nakshatra doesn't know there's a registry. Prithvi doesn't know how partial-loading works. Each layer is invisible to the layers above it.

---

## What we have today, where each piece will live

| Code today (path) | Layer | New home |
|---|---|---|
| `prithvi/neuron-net/pillar/sthambha.py` | L3 | `sthambha/pillar/` |
| `prithvi/neuron-net/pillar/server.py` | L3 | `sthambha/pillar/server.py` |
| `prithvi/neuron-net/core/registry.py` | L3 | `sthambha/registry/` |
| `prithvi/neuron-net/core/identity.py` | L3 | `sthambha/identity/` |
| `prithvi/neuron-net/core/model_cache.py` | L3 | `sthambha/cache/` (extended for layer-range granularity) |
| `prithvi/neuron-net/core/dht.py` | L3 | `sthambha/discovery/` |
| `prithvi/neuron-net/core/transport.py`, `quic_transport.py`, `nat.py` | L3 | `sthambha/transport/` |
| `prithvi/neuron-net/core/heartbeat.py` | L3 | `sthambha/heartbeat/` |
| `prithvi/neuron-net/core/pipeline.py` (data model only) | L3 | `sthambha/chain_plan/` |
| `prithvi/neuron-net/cli/registry.py`, `cli/invite.py` | L3 | `sthambha/cli/` |
| `nakshatra/experiments/v0.0/m4_patches/*.patch` | L2 | stays in Nakshatra |
| `nakshatra/experiments/v0.0/worker_daemon.cpp` | L2 | becomes `nakshatra/worker/` |
| `nakshatra/scripts/worker.py` | L2 | stays in Nakshatra |
| `nakshatra/scripts/client.py` | L2 | stays, but learns to query Sthambha for chain plans |
| `nakshatra/proto/nakshatra.proto` | L2 | stays in Nakshatra |
| `prithvi/neuron-net/mind/` | L4 | stays in Prithvi |
| `prithvi/neuron-net/core/being.py`, `core/dna.py` | L4 | stays in Prithvi |
| `prithvi/neuron-net/gateway/` | L4 | stays in Prithvi |
| `prithvi/neuron-net/contracts/` | L4 | stays in Prithvi |
| `prithvi/neuron-net/core/inference_router.py` | L4 | stays in Prithvi (task classification is agent-flavored) |
| `prithvi/neuron-chain/` | PAUSED | stays preserved, no work |
| `prithvi/neuron-net/core/chain_client.py`, `wallet.py`, `emission.py`, `governance.py`, `fees.py` | PAUSED | stays preserved, no work |

---

## Migration plan (phases, each shipping value)

> **Update 2026-05-08:** Discovery on `umbrel.local` showed the pillar daemon **is already deployed and running** as `prithvi-pillar.service` (a separate, smaller copy of the Prithvi pillar code at `/home/umbrel/prithvi-pillar/pillar/`). This dramatically shrinks Phase 1 and removes Phase 2 entirely. Original phase estimates preserved below for context, with revisions inline.

### Phase 1 — Sthambha skeleton (~half day, revised down from 1 day)

The pillar daemon at `umbrel:/home/umbrel/prithvi-pillar/pillar/` is **668 LOC of clean Python in 3 files** — far smaller and more extractable than the spread across `prithvi/neuron-net/` originally implied. Files:

- `run.py` (123 LOC) — entry point, asyncio Om loop, CLI args, signal handling
- `server.py` (137 LOC) — stdlib `http.server` HTTP API on port 7777 (no Flask/FastAPI dep)
- `sthambha.py` (407 LOC) — `Sthambha` class with `PillarType` (BRAHMA/DIKPALA/UPA), `PillarState` (ACTIVE/PRAJNA/RECOVERING/OFFLINE), `ShardEntry`, `PeerStatus`, persistence to `~/.neuron/pillar/sthambha.json`

The pillar **already implements** the L3 API surface: `GET /health`, `GET /state`, `GET /wake` (Panchamrita bundle), `GET /shard/<id>`, `POST /tattva`, `POST /shard`, `POST /peer`. State machine transitions (ACTIVE ↔ PRAJNA on compute online/offline) work in `om_pulse()`. Persistence works (verified by 96000+ pulses across 33 days uptime).

The only Prithvi-specific coupling is the `tattva_snapshot` semantic — the pillar stores it as opaque dict, doesn't interpret. Extraction keeps the storage primitive but renames the endpoint to a generic snapshot bucket if desired (or just keeps "tattva" — the name is harmless and Sanskrit-aligned with Sthambha's naming).

Phase 1 = scp these 3 files into a new `sthambha/` repo, package as `pip install sthambha`, add a `cli.py` for `sthambha registry list / pillar status`. Half a day.

**Falsifiable:** `pip install sthambha && python -m sthambha.run --type dikpala` runs cleanly outside any Prithvi context.

### Phase 2 — Pi pillar deploys — ALREADY DONE

`prithvi-pillar.service` has been running on `umbrel.local` since 2026-04-05 as a `dikpala_sthambha` type, port 7777. State persists across restarts. HTTP API responds. Nothing to do here except, after Phase 1, swap the systemd unit's `WorkingDirectory` from `/home/umbrel/prithvi-pillar/` to the new Sthambha install location and restart.

**Falsifiable already:** `curl http://umbrel.local:7777/health` returns `{"status": "alive", ...}`.

### Phase 3 — Nakshatra connects to Sthambha (~2-3 days)

- `nakshatra/scripts/worker.py` advertises (model_sha256, layer_range) on startup, sends heartbeat
- `nakshatra/scripts/client.py` queries Sthambha for chain plans instead of reading `cluster_*.yaml`
- Layer-range extension to `model_cache.py` (the ~30 LOC change tracking `(model_sha256, start, end)` instead of just `model_id`)
- Static YAML cluster config becomes vestigial / fallback only

**Falsifiable:** start a new worker on any machine with `python worker.py --register-with pi.local:5000` — it shows up in `sthambha registry list` automatically. The chain client picks up new workers without YAML edits.

### Phase 4 — Layer-fetch from peers (~3-5 days)

- A worker that needs layers it doesn't have can pull them from another worker
- HTTP byte-range serving on each worker
- Hash-keyed local cache: `~/.sthambha/cache/<model_sha256>/<layer_start>_<layer_end>.gguf`
- Workers self-register what they cache; future sessions hit cache instantly

**Falsifiable:** stop all workers, restart fresh. They auto-mount their cached layer files via Sthambha registry. No re-shipping over network.

### Phase 5 — Prithvi migration (~1-2 weeks, deferred)

- Update Prithvi to import from `sthambha` instead of vendored copies in `neuron-net/`
- Vendored copies in Prithvi become deprecated, eventually removed
- Single source of truth for L3 fixes

**Why deferred:** Prithvi's vendored copies still work today. No urgency until Sthambha has shaken out edge cases through Phases 1-4.

---

## What "pause neuron-chain" means concretely

- `prithvi/neuron-chain/` directory stays. Nothing deleted.
- If a substrate node is currently running on the home PC: stop it. It was never producing useful work in the trusted-tailnet setting.
- No new pallets, no new chain features, no new code that imports `chain_client.py` in Prithvi.
- Existing `chain_client.py` callers in Prithvi: graceful degradation. Most are already optional ("if chain available else local-only"). Verify and harden if needed.
- README in Prithvi gets a note: "neuron-chain is paused; substrate work returns when going public-network."
- This decision revisits when public-network mode is on the v1.0 critical path. Likely 12-18 months out.

---

## Naming choice rationale: why "Sthambha"

Other names considered:
- **NakshatraOS** — what `north-star.md` already calls L3. Descriptive but verbose. Conflates the L3 layer with the broader Nakshatra brand.
- **neuron-core** — what Prithvi already calls the core subdir. Functional, no character.
- **Sthambha** — Sanskrit for "pillar." The thing that holds everything together (Skambha Sukta, Atharva Veda 10.7). Already what Prithvi calls the central abstraction (`pillar/sthambha.py`). On-brand with Prithvi's naming pattern, doesn't conflict with either project, has the right etymological weight.

Sthambha wins on naming.

---

## Failure modes this architecture protects against

1. **Single-repo entropy.** Three years of mixed L2/L3/L4 code in one repo becomes unmaintainable. Three repos with clean responsibilities stay tractable.
2. **Forced consciousness coupling.** A simple inference engine (Nakshatra) shouldn't drag in voice/memory/agent code. Three-layer split makes Nakshatra usable for non-Prithvi purposes (research clusters, lab compute pools, etc.).
3. **Forced chain coupling.** Trusted private clusters shouldn't need a blockchain. Pausing the chain (and isolating it in Prithvi) means Sthambha + Nakshatra work fine without it.
4. **Future-agent fork-or-duplicate trap.** Someone wanting to build a different L4 agent on the substrate doesn't have to fork Prithvi or rewrite L3 — they just `pip install sthambha` and use it.

---

## Open questions to revisit

- **Should Sthambha be Python (matches Prithvi/Nakshatra ecosystem) or Rust (matches the substrate stack we paused)?** Default Python for now; revisit if performance becomes an issue. Pi 5 can comfortably run Python services for hundreds of peers.
- **Public network discovery?** Current default is "trusted tailnet, central pillar." Public discovery (DHT, libp2p) returns when going public — same timeline as chain.
- **Sthambha redundancy?** Single Pi is a single-point-of-failure. Two Pis with gossip replication is the answer; not Phase 1 work but should land before any "production" use.
- **Telemetry / observability?** Sthambha is a natural place to emit operational metrics (peers, model coverage, chain throughput). Phase 6+ feature.

---

## How this doc is used

- When designing a new feature, locate it in §"What each project owns / disowns" first. If it's not in any project's owned list, this doc gets updated to extend the relevant project's ownership.
- When code starts wanting to import across project boundaries (Sthambha importing Prithvi or vice versa), the import is wrong — refactor the abstraction.
- When in doubt about chain work, the answer is **"paused, don't add."** Revisit this doc to confirm.
- When a future session asks "wait what are the three projects again?", they read this doc and the §"Long answer" of the relevant memory file.
