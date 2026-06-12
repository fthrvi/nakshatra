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

## Mesh-wide — remote summon/reap (BUILT, 2026-06-11)
The `ChainController` seam now reaches across machines, so the SAME
`ChainLifecycle` governs a chain spread over the fleet:
- **`RemoteSshController`** — summons/reaps REMOTE workers over SSH. `start()`
  runs each node's (self-detaching) launch command, `stop()` `pkill`s it
  (returning the borrowed machine), `is_ready()` = all peers' gRPC ports up.
  **Proven live on a lab Mac:** summoned in ~6s → reaped after idle ("returning
  the borrowed machine") → re-summoned in ~4s.
- **`CompositeController`** — one chain across local + remote nodes (e.g. this box
  `[0,10)` local + 3 lab Macs remote); ready iff ALL are ready.
- **`from_env`** composes them: `NAKSHATRA_LIFECYCLE_UNITS` (local systemd) +
  `NAKSHATRA_LIFECYCLE_REMOTE_CONFIG` (a JSON of remote workers) → a
  CompositeController. Borrowed nodes get a SHORT grace (~90s).
- **`deploy/lifecycle.70b.example.json`** — the 70B-across-the-lab-Macs config
  (launch commands bake in the proven gotchas). Re-launch the 70B + point the serve
  at it ⇒ the lab Macs run ONLY during a deep query, then return to their owners.

So scale-to-zero is no longer just local — the **whole mesh** now obeys "summon,
don't squat," driven from the serve.

## L3 — Sthambha as the compute-lease OWNER (the remaining hop)
The serve-driven controllers above realise the *outcome* for a single consumer
(Prithvi's `think_deeper`). The final hop centralises *ownership* in the pillar so
MANY consumers share one arbiter with the global view:
1. **Lease in the pillar** — `POST /lease {model, ttl}` returns a summoned chain;
   the pillar tracks last-activity and **reaps** it when the lease idles past the
   **per-node ownership-aware grace** (each node declares grace + `dedicated` |
   `borrowed` at registration). The `RemoteSshController` becomes the pillar's
   summon mechanism (or a small per-node agent it signals). Workers also
   self-heartbeat; a worker MAY self-reap if it loses contact (decentralised
   safety — each node governs itself).
2. **Keep the *plan* hot, not the *weights*** — the pillar already holds the
   layer-split plan, so re-summon is "load weights on the assigned peers", not
   "re-plan".

Then `client.py --registry` / `registry_url` lease a chain on demand and the mesh
inherits the policy globally. **Status:** L4 + mesh-wide remote = done. The
pillar-OWNED lease is the remaining build (touches the shared Pi pillar —
coordinate with its owner); the seam + policy are already in place to receive it.
