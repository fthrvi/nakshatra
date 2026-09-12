# The courteous worker — a lease, not a poll

**Design. 2026-09-04. Nothing here is built. Prompted by adding an RTX 5070 in WSL2 on a
family member's gaming PC; written to hold for every consumer machine that follows.**

## The problem is general, and the current rule does not solve it

Every consumer machine that joins has an **owner whose use outranks ours**, and on every
platform the worker is **blind to some of what that owner is doing**:

| platform | what the worker cannot see |
|---|---|
| Windows host, worker in WSL2 | any Windows-side process. `nvidia-smi` in WSL shows memory and utilisation only |
| Linux desktop | that a game is about to launch; a Wayland compositor's intent |
| macOS | a Final Cut export or a Metal app spinning up on unified memory |
| headless server | nothing — no owner contention. **No sentinel needed.** |

The rule that exists today for blackwell — *poll every 10 min, take the card after six idle
samples* — is a **one-time decision followed by an open-ended hold.** It asks "is it free?"
once an hour and then holds 8 GB for as long as it likes on a stale answer. It was written for
a fire-once training job; it is wrong for a node that *holds memory to serve*.

⚠️ And the worst part is structural: **our own 8 GB is indistinguishable from the owner's
usage in the only signal WSL has.** Once we are loaded, the "is it idle?" question can no
longer be answered by looking at memory.

## The lease: silence means stop

Permission is a **short-lived grant that must be continuously renewed.** The worker holds the
GPU only while fresh evidence keeps arriving that it is still fine to.

    sentinel (on the host, sees the owner) ──heartbeat every 1–2 s──▶ worker (in WSL / the daemon)

- Each heartbeat carries `{at, clear: bool, reason}`. `clear=False` is an explicit "the owner
  needs it"; a *missing* heartbeat is treated identically.
- The worker keeps a lease with a TTL of a few heartbeat intervals. **Every action it takes
  is conditional on the lease being live.**
- When the lease lapses — game launched, owner flipped a switch, sentinel crashed, PC going to
  sleep, WSL shut down — the worker, within seconds:
  1. stops accepting new work,
  2. cancels in-flight steps,
  3. frees the KV cache,
  4. unloads the weights,
  5. tells the coordinator it has left the pipeline.

**The default state is "release." Staying loaded requires active, continuous proof.** With
polling, silence means keep going; with a lease, **silence means stop.** That single inversion
is what makes it safe on a machine we do not own: *the failure mode of the sentinel is
releasing the card, never holding it.*

## Why this generalises — one policy, many sentinels

The lease policy is **pure** and written **once**: given a heartbeat history and `now`, is the
lease live, stale, or lapsed? `heartbeat.py` (2026-09-04) already has exactly this shape —
`node_state(last_seen, now, interval, miss_stale, miss_dead)` → live / stale / dead — at a
30 s interval for mesh liveness. The lease is the same function at a 1–2 s interval, where
"stale" means *stop admitting* and "dead" means *release now*.

What is per-platform is **only the sentinel**, and it is small: read whatever signal that OS
offers, emit a heartbeat. It never decides anything.

| platform | sentinel signal | size |
|---|---|---|
| Windows host | foreground window is fullscreen/exclusive; a process from a known-game list; **an explicit owner switch** (a tray toggle: "I need the GPU") | a small tray app or scheduled task |
| Linux desktop | a fullscreen client on the compositor; `/proc` for known-game binaries; the same owner switch | a user service |
| macOS | `NSWorkspace` frontmost app; Metal device activity; owner switch | a menu-bar item |
| headless | none — the lease is permanently live | — |

⚠️ **The owner switch is the one signal every platform must have.** Heuristics about games
will miss things; a person saying "not now" must always win, instantly, without knowing
anything about how the worker functions.

So: **the class of problem is solved once.** Each new platform still needs its sentinel
written — the signal source is inherently OS-specific and cannot be abstracted away — but
nobody re-derives the *policy*, and a sentinel that is buggy on a new platform fails toward
release. That is the honest answer to "will we have this again": the *policy* work, no; the
*signal-reading* work, a small amount per platform, with a safe default if it is wrong.

## What the lease cannot do, and the rule that covers it

Detection is not prevention. If the owner's game issues its first allocation at the instant
we hold 8 GB, **that allocation can fail before any heartbeat arrives.** The lease bounds how
long we are in the way; it cannot make us not in the way.

So a second rule, independent of the sentinel: **never sit tight against the card.** On a
12 GB card that someone else games on, a worker that plans to hold 8 GB is the problem. The
capability report should treat *owner headroom* as a first-class reservation — on shared
hardware, offer the network what is left after the owner's typical peak, not after our own
comfort. Codex's estimate: 4 GB headroom is often insufficient. On blackwell, that may mean
this node serves a **smaller slice** than its VRAM suggests, or serves only when the owner is
demonstrably away (sentinel reports the screen locked for N minutes). That is a placement
input, not a courtesy afterthought.

## What the pipeline sees when a stage leases out

A stage vanishing mid-token is the failure Codex named, and it is the same failure as any node
dying — which the mesh already handles: the O(t) recovery path (activation-replay cache,
proven 2026-06-11) catches a departed stage up on a replacement, and `heartbeat.py`'s
stale/dead band keeps the node listed-but-not-routed-to during a short absence.

What the lease adds is **notice**. A polite departure is better than a crash: the worker
should, when the lease lapses, send one final message — *leaving, layers [a,b), last step N*
— before unloading, so the coordinator can start the replacement immediately rather than after
a timeout. Bounded timeouts, cancellation propagation and idempotent retries are already
required for the dying-node case; the lease does not introduce new requirements, it just
exercises the existing ones more often.

## Join-path integration

The join sequence should refuse to declare a shared machine `joined` without a sentinel:

- `capability` reports `shared_host: true` when the platform is a desktop OS with an active
  user session (or the operator says so).
- `selfcheck` treats `shared_host and not sentinel_heartbeat_seen` as **not ready** — with the
  reason "shared machine, no owner sentinel; would hold the GPU against its owner."
- The economics stay off, but when they turn on: a node whose sentinel reports high owner
  contention should be **placed less**, not paid less. Reliability is a placement input.

## Order of work

1. **The lease policy** — pure, reusing `heartbeat.py`'s shape. Small. Testable with a dict.
2. **The release action** — in the acting layer: cancel → free KV → unload → announce leaving.
   Testable against a local daemon.
3. **The Windows sentinel** for blackwell — the owner switch first (a tray toggle), fullscreen
   detection second. It must be installable without touching the owner's files or accounts:
   a scheduled task in the operator's own user, reading only window state.
4. **Headroom as a placement input** — `capability` learns `owner_reserve_bytes`.
5. Then, and only then, blackwell serves a slice.

## The invariant, stated once

**We use the compute. We never touch the owner's data, and we never keep the owner waiting.**
The lease enforces the second half. The first half is enforced by what the worker *is*: a
process that loads a model slice from a signed package into GPU memory and speaks gRPC. It
opens no user files, reads no home directory, and its only outbound connections are to peers
it has pinned. On blackwell the sentinel must respect the same line — it reads *window state*,
never *window contents*.
