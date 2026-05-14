# Nakshatra — Build Instructions

## What this project is

Nakshatra is the **L2 inference engine** in a four-project architecture. One layer in the stack — not the whole system. Read `docs/four-project-architecture.md` first if you have not already; it is the source of truth for project boundaries.

In short:
- **Neuron (L1)** — chain & economic substrate. `~/neuron/` (`fthrvi/neuron`, not yet pushed). PAUSED at the project level; extracted from Prithvi- on 2026-05-14.
- **Nakshatra (L2)** — *this repo*. Patched llama.cpp + worker daemon + sub-GGUF tooling + gRPC chain protocol.
- **Sthambha (L3)** — peer registry, pillar daemon, identity, layer cache, planner (`plan_split` shipped 2026-05-14), network fabric (designed). `~/sthambha/` (`fthrvi/sthambha`).
- **Prithvi (L4)** — the agent / being / consciousness. `fthrvi/Prithvi-` on home PC.

## What goes here, what does not

**Belongs in Nakshatra:**
- llama.cpp loader + graph builder patches
- Worker daemon (`experiments/v0.0/worker_daemon.cpp`)
- Python gRPC worker (`scripts/worker.py`)
- Sub-GGUF generation (`partial_gguf.py`)
- Chain orchestration client (`scripts/client.py`)
- Streaming KV reuse, per-token routing, latency / vendor-portability

**Does NOT belong in Nakshatra:**
- Peer registry, identity, pillar daemon, model/layer cache, fabric → Sthambha
- Consciousness, voice, gateway, OpenAI-compatible API → Prithvi
- Substrate chain / NRN tokenomics / wallet / receipts → **Neuron** at `~/neuron/`. Still PAUSED at the project level; returns alongside Sthambha fabric Mode C. Don't add code that imports chain modules.

If a feature wants to do registry / identity / cache work, it is Sthambha-shaped — flag it as such rather than absorbing it here. Nakshatra stays an inference engine.

## Key reference docs (read in this order)

1. `docs/four-project-architecture.md` — architectural decision (2026-05-08), project boundaries, migration plan.
2. `docs/north-star.md` — L1-L4 long vision.
3. `docs/petals-architecture.md` — v0.1 spec (the contract we shipped against).
4. `docs/v0.1-implementation-plan.md` — milestones M1-M7 (all green).
5. `docs/petals-deep-read.md` — Petals comparison + design rationale.

If a v0.1 design decision conflicts with `north-star.md`, v0.1 wins (per the doc's own §"Why this doc exists").

## Cluster

5 machines: home PC (Linux + RX 9070 XT) + 4 lab Macs (mac3-2, mentorings-imac-pro, mac4, bishwa). Plain `ssh <hostname>` works via `~/.ssh/config`. See `~/.claude/projects/-Users-bishwanathbastola-nakshatra/memory/nakshatra_cluster_hosts.md` for usernames + Tailscale IPs.

## Patterns

- Use `lsof -ti:PORT | xargs kill -9` to kill workers, NOT `pkill -f scripts/worker.py` (the pkill pattern matches its own command line and self-terminates).
- Linux home PC needs `loginctl enable-linger prithvi` so workers survive SSH disconnect.
- Streaming KV reuse: first step `keep_kv=false`, subsequent steps `keep_kv=true` with `start_pos=prefix_length`.
- Sub-GGUFs declare layer ranges via `nakshatra.layer_range_start/end` metadata; `partial_gguf.py` is the canonical generator.
