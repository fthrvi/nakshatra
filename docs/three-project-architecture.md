# Three-Project Architecture — moved

The full architecture decision document now lives in the **Sthambha** repo, where it is canonical:

→ **https://github.com/bbastola899/sthambha/blob/main/docs/three-project-architecture.md**

If you have Sthambha cloned locally:

```bash
$EDITOR ~/sthambha/docs/three-project-architecture.md
```

## Why it moved

The three-project architecture (Sthambha L3 + Nakshatra L2 + Prithvi L4) is shared concern. Keeping the canonical doc in three repos meant three places to edit on every change, and they drifted within hours of being created. Single source of truth in Sthambha means: edit once, every repo's CLAUDE.md and stub points at the same answer.

## TL;DR (for readers who don't click)

- **Sthambha (L3, this is what holds everything):** peer registry, pillar daemon, identity, model/layer cache. Also Prithvi's *Asthi Dhatu* (soul-persistence layer) — Shamir-split identity shards + Tattva snapshots + Om pulse that survives compute death. Repo: `bbastola899/sthambha`. Pillar deployed on `umbrel.local` since 2026-04-05.
- **Nakshatra (L2, this repo):** distributed inference engine. Patched llama.cpp + worker daemon + sub-GGUF tooling. v0.1 shipped on 5-machine cluster.
- **Prithvi (L4):** agent / consciousness / voice / gateway. Repo: `bbastola899/prithvi`.
- **Neuron-chain (Substrate blockchain):** PAUSED indefinitely. Returns at v1.0+ when going public-network.

For the full plan including phase-by-phase migration, current status, and the soul-backup activation ritual, read the canonical doc.
