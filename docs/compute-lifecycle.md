# Compute lifecycle — "summon, don't squat"

The governing principle for this mesh: **a worker holds a GPU only while it is
serving an active session.** When a chain goes idle past an *ownership-aware*
grace window it is torn down and the (often borrowed) machine returns to its
owner. On the next request the chain is re-summoned (cold-start). The mesh
defaults to *dark* — nodes light up when work arrives, then go dark.

Why this is the right default for *this* network: it isn't a datacenter of
dedicated GPUs. It's a **sovereign mesh that borrows compute across
heterogeneous, partly-shared machines** (your box, lab Macs, future peers). The
defining constraint is that most of the fleet isn't yours — squatting on a
borrowed GPU 24/7 breaks the social contract that makes borrowing possible.
Scale-to-zero *is* that contract.

## Ownership-aware grace (the one rule that matters)
Grace scales with **who owns the node**:
- **Borrowed / shared** (lab Macs, peers) → release **aggressively** (~60–120s).
  Politeness is non-negotiable for someone else's machine.
- **Dedicated** (your own box) → a modest warm window (~5–10 min) absorbs
  follow-up questions in a reasoning session without re-squatting.

This single rule lets the mesh stay polite enough to grow to strangers while
staying snappy on hardware you own.

## L4 — serve-side scale-to-zero (BUILT, 2026-06-11 · `scripts/serve_lifecycle.py`)
The serve process (no GPU) stays up as the front door and owns the worker
lifecycle:
- **`ChainLifecycle`** — request-gated `begin()`/`end()` (cold-starts the chain
  and blocks until ready; counts in-flight requests so it never reaps
  mid-generation) + a background **reaper** that tears the chain down after the
  idle grace.
- **`ChainController`** (pluggable mechanism): `start()` / `stop()` / `is_ready()`.
  - `SystemdLocalController` — `systemctl --user start/stop` local worker units;
    readiness = the worker gRPC ports accept a connection.
- Env-gated on the serve (`NAKSHATRA_LIFECYCLE_{UNITS,PROBES,IDLE_GRACE_S,
  START_TIMEOUT_S}`); absent ⇒ legacy always-on.

**Live + proven:** the 8B unconscious now reaps after 600s idle (frees ~8.7 GB)
and re-summons in ~7.5 s on the next `think_deeper`. The conscious (ollama) already
idle-unloads, so the GPU goes nearly empty when Prithvi rests.

## L3 — Sthambha as a compute-lease manager (THE DESTINATION, next build)
The `ChainController` interface is the seam. The same `ChainLifecycle` policy
lifts to the mesh by swapping the controller and moving ownership to the pillar,
which alone has the global view (registered peers + active plans):

1. **`RemoteController`** — summons/reaps REMOTE workers (the lab Macs, peers) for
   a planned chain. Same surface as `SystemdLocalController`; `start()` brings up
   each peer's worker (today: the SSH bring-up sequence proven for the 70B fleet —
   see `[[project_l2_to_l4_connected]]` for the recipe + gotchas; tomorrow: a small
   per-node agent the pillar calls). `is_ready()` = all peers serving.
2. **Lease in the pillar** — `POST /lease {model, ttl}` returns a summoned chain;
   the pillar tracks last-activity and **reaps** the chain when its lease idles
   past the **per-node ownership-aware grace** (a node declares its grace +
   ownership class — `dedicated` | `borrowed` — at registration). Workers also
   self-heartbeat; a worker MAY self-reap if it loses contact (decentralised
   safety — the sovereign-mesh ethos: each node governs itself).
3. **Keep the *plan* hot, not the *weights*** — the pillar already holds the
   layer-split plan, so re-summon is "load weights on the assigned peers", not
   "re-plan". Cold-start is the only cost, and it's the right one to pay.

Then `client.py --registry` / `nakshatra_serve`'s `registry_url` lease a chain on
demand, and the whole mesh inherits "summon, don't squat" for free — borrowed
machines are only ever busy during an actual query.

**Status:** L4 done + live. L3 lease manager = the next focused build (touches the
shared Pi pillar — coordinate with its owner). The seam (`ChainController`) and the
policy (`ChainLifecycle`, ownership-aware grace) are already in place to receive it.
