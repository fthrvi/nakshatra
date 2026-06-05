# Nakshatra — Agent Guide

> For AI agents working ON this repo. **Nakshatra is L2** of the four-project stack:
> distributed inference — large models (70B-class reasoning) split across a GPU mesh.
> It is Prithvi's **"unconscious"** — the deep-thinking brain the conscious 8B escalates
> to via `think_deeper`.

## What this repo is

A Petals-derived distributed-inference engine. It serves an **OpenAI-compatible
`/v1/chat/completions` + `/v1/models`** surface — the contract Prithvi's `think_deeper`
client depends on. **Keep that contract stable** (request/response shapes); if the served
model name or auth changes, note it in the inbox so the Prithvi client updates.

## Layout

- `src/` — the engine · `proto/` — wire protos · `tests/` — pytest · `docs/` — design ·
  `benchmarks/` · `experiments/` · `spike/` — exploration · `README_PETALS.md` — upstream lineage.

## Run / test

```bash
pip install -e .          # pyproject.toml
pytest tests/ -q
```

## Coordination

Worked by multiple sessions/machines. This repo **does not yet carry a `BRANCHES.md`/`INBOX.md`** —
if you start parallel work, add them (copy the pattern from `prithvi`/`sthambha`: a branch registry
+ a directed-message inbox, `sector/short-task` branch names, never push over a branch another owner
marks `active` or a `main` someone's actively committing to). There is a `CLAUDE.md` here already.

## Conventions

- Match surrounding code; small, focused commits explaining the *why*.
- **Git author here is legacy `tankaifish`** (not `Bishwanath Bastola` as in the other repos) —
  check `git log` / the local config before committing so identity stays consistent.
- **Never type `git pull`/`push`** — a 5-minute reposync handles it.
