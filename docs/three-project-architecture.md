# Four-Project Architecture — canonical doc moved
> *Filename retained as `three-project-architecture.md` for now; the canonical doc grew to four projects on 2026-05-14 when Neuron was extracted from Prithvi-. Rename pass will land separately.*

The full architecture decision document lives in the **Sthambha** repo, where it is canonical:

→ **https://github.com/fthrvi/sthambha/blob/main/docs/three-project-architecture.md**

If you have Sthambha cloned locally:

```bash
$EDITOR ~/sthambha/docs/three-project-architecture.md
```

## Why it moved

The L1-L4 architecture is shared concern across all four repos. Keeping the canonical doc in every repo meant N places to edit on every change, and they drifted within hours. Single source of truth in Sthambha means: edit once, every repo's CLAUDE.md and stub points at the same answer.

## TL;DR (for readers who don't click)

- **Neuron (L1):** chain & economic substrate. Substrate validator + NRN token + on-chain accounting. `~/neuron/` (`fthrvi/neuron`, not yet pushed). **PAUSED at the project level** — extracted from Prithvi- on 2026-05-14 for hygiene. Returns alongside Sthambha fabric Mode C, when the network actually has untrusted public peers.
- **Nakshatra (L2, this repo):** distributed inference engine. Patched llama.cpp + worker daemon + sub-GGUF tooling. v0.5 design-locked. 5-machine cluster runs Llama-70B + Qwen3-MoE.
- **Sthambha (L3, what holds everything):** peer registry, pillar daemon, identity, model/layer cache, **planner** (`plan_split` shipped 2026-05-14), **network fabric** (designed). Also Prithvi's *Asthi Dhatu* (soul-persistence layer) — Shamir-split identity shards + Tattva snapshots + Om pulse that survives compute death. Repo: `fthrvi/sthambha`. Pillar deployed on `umbrel.local` since 2026-04-05.
- **Prithvi (L4):** agent / consciousness / voice / gateway. Repo: `fthrvi/Prithvi-`. Now L4-only after the 2026-05-14 chain extraction.

For the full plan including phase-by-phase migration, current status, and the soul-backup activation ritual, read the canonical doc.
