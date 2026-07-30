# Declarative model lifecycle reconciliation (Mesh-LLM adoption #3)

**Status:** module + tests landed on `serving/lifecycle-reconcile`. Not armed on any live
serve — see "The remaining ask" below.

## Why

`scripts/serve_lifecycle.py` already does scale-to-zero: `ChainLifecycle.begin()` summons
the chain on the first request and blocks until ready (~7.5s cold start), and a background
reaper tears it down after an idle-grace window with no in-flight requests. That's all
*reactive* — it answers "is this one chain up right now, and should it come down" — but
there's no *declared* shape to converge toward across several models. Mesh-LLM's daemon
reconciliation is the missing piece: an operator writes down which models should be
`resident`, `on-demand`, or `absent`, and a small loop nudges reality toward that
declaration on a schedule, independent of whether a request happens to be in flight.

## Design

```
serve_models.desired.yaml  --load_desired-->  {model_id: DesiredEntry}
                                                     |
                              current = backend.list_running()
                                                     |
                                        plan(desired, current)      <- PURE, no I/O
                                                     |
                                  [Summon | Reap | Noop, ...] (each with a reason)
                                                     |
                               converge(plan, backend, dry_run=True)  <- DRY-RUN DEFAULT
```

**`plan()` is a pure function** — `dict[str, DesiredEntry] -> list[Action]`, no backend
calls, fully unit-testable without touching anything real. The three desired states:

- `resident` — must be up. Missing → `Summon`. Running → `Noop` (already matches). Never
  reaped by reconcile (that's what "resident" means — it's the operator opting a model
  *out* of scale-to-zero's idle-reap).
- `on-demand` — the existing `ChainLifecycle` (summon-on-request, reap-after-idle-grace)
  governs it. Reconcile reports it as a `Noop` every cycle whether it's up or down, but
  takes no action either way — forcing a summon here would defeat the point of scale-to-
  zero, and reaping it out from under an active session would be worse.
- `absent` — must not be running. Running → `Reap`. Down → `Noop`.

**Conscious-reserve spirit:** a model that's running but not named in the desired-state
yaml at all is reported — a `Noop` action with a reason string — never reaped. Reconcile
only ever acts on what the operator explicitly declared; the same instinct as
`placement_feed`'s conscious VRAM reserve (never subtract/evict what you don't have a
declared opinion about). Only an explicit `desired: absent` entry authorizes a reap of
that model id.

**`converge()` executes (or dry-runs) a plan** against a `Backend`
(`list_running()`/`summon(id)`/`reap(id)`). A backend exception is caught and logged, never
raised — `converge()` itself is a single non-crashing pass.

**`Reconciler` is the stateful loop driver.** `plan`/`converge` are pure/stateless per call;
`Reconciler` owns the backend plus cross-cycle failure/backoff bookkeeping so `tick()` is
safe to call forever: a failed summon/reap is logged and NOT retried again until its
backoff window elapses (exponential, base 30s, capped at 900s by default) — so a wedged
node doesn't get hammered every cycle, and a successful retry clears the backoff. `tick()`
itself never raises; `run_forever()` also catches a malformed desired-state yaml mid-loop
and just tries again next interval rather than dying.

**`LifecycleBackend` adapts the REAL `serve_lifecycle.ChainController` surface**
(`start()`/`stop()`/`is_ready()`) onto the `Backend` protocol, keyed by model id.
`serve_lifecycle.py` governs exactly one chain per controller today —
`nakshatra_serve.py` builds a single `ChainLifecycle` from env
(`serve_lifecycle.from_env()`), not a per-model registry — so multi-model desired-state is
expressed today by registering one controller per declared model id
(`LifecycleBackend({model_id: controller, ...})`, or `.from_chain_lifecycles({model_id:
chain_lifecycle, ...})` which unwraps each `ChainLifecycle.controller`). When a real
multi-model controller registry exists, only the *construction* of that dict changes —
the adapter and everything above it doesn't.

`FakeBackend` is the in-memory test double: an in-memory running set + call log, with
`fail_summon_next(model_id, times=N)` / `fail_reap_next(...)` to make the next N attempts
raise (then succeed) — that's what exercises the retry/backoff path in tests without
touching systemd/SSH/GPU.

## The dry-run-default safety stance

Every entry point defaults to dry-run:

- `converge(actions, backend, dry_run=True)` — the keyword defaults to `True`.
- `Reconciler.run_once(...)` / `run_forever(...)` — same.
- The CLI (`python3 scripts/lifecycle_reconcile.py --desired ... --once`) — dry-run unless
  `--apply` is passed, or `NKS_RECONCILE_APPLY=1` is set in the environment. It prints
  `[reconcile] DRY-RUN (default) — pass --apply or NKS_RECONCILE_APPLY=1 to actually
  summon/reap` on every dry-run invocation so it's never a silent no-op.

In dry-run, a `Summon`/`Reap` action is logged as `[reconcile] [dry-run] would summon/reap
<model> (<reason>)` and returned unexecuted — the backend is never called, so `FakeBackend`
tests can assert `backend.calls == []` and real-controller tests can assert
`controller.started == controller.stopped == 0`. This matters here specifically because a
wrong reap on the live `nakshatra-unconscious` serve tears down Prithvi's `think_deeper`
reasoner mid-window; a wrong summon on a borrowed node squats someone else's GPU. Both are
one bad yaml line away without a dry-run default, so the loop is built to require an
explicit, deliberate step (`--apply` / the env var) before it's allowed to touch anything.

## What's tested (28 tests, `tests/test_lifecycle_reconcile.py`)

- `plan()`: missing-resident → summon; declared-absent-and-running → reap; unknown-running
  → reported as `Noop`, never reaped; steady state (resident-up, absent-down) → all `Noop`;
  on-demand → always `Noop` regardless of running state; absent-and-already-down → `Noop`.
- `load_desired()`: parses the shipped example yaml; fails loud (raises
  `DesiredStateError`) on invalid YAML, a missing `models` key, an unknown `desired` value,
  a duplicate model id, and a missing model name.
- `converge()`: dry-run (default) executes nothing; `dry_run=False` calls through and
  updates the backend; `Noop` never touches the backend; a backend exception is caught and
  surfaced as `ok=False`, not raised.
- `Reconciler`: end-to-end dry-run touches nothing; `dry_run=False` converges a
  missing-resident + a stale-absent in one pass; never reaps an unknown-running model;
  retries a failed summon with exponential backoff (held on the next tick while backoff is
  active, retried and succeeds once the clock passes the window — using an injectable
  clock, no real sleeping); backoff state is never populated in dry-run; the loop survives
  a malformed desired-state yaml without raising.
- `LifecycleBackend`: matches `serve_lifecycle.ChainController`'s real
  `start`/`stop`/`is_ready` surface; summon/reap call through and `list_running()` reflects
  `is_ready()`; an unknown model id raises `KeyError`; an `is_ready()` exception is caught
  and logged, reported as not-running (never crashes `list_running()`); `.from_chain_lifecycles`
  correctly unwraps `ChainLifecycle.controller`.
- One end-to-end test wires a real `Reconciler` to a `LifecycleBackend` over stub
  controllers (duck-typing `ChainController`, no systemd/GPU) and confirms a dry-run plan
  against it summons/reaps nothing.

All CPU-only, no GPU model loaded, no live service touched — `FakeBackend` and duck-typed
stub controllers stand in for anything real. `python3 scripts/lifecycle_reconcile.py
--desired scripts/serve_models.desired.example.yaml --once` was run directly (no
`--model-lifecycle-id` and again with one set but no `NAKSHATRA_LIFECYCLE_*` env) and
confirmed to print the dry-run banner and exit 0 without touching anything.

## The remaining ask (operator's call)

Two things intentionally left undone here:

1. **A systemd timer unit** (`nakshatra-lifecycle-reconcile.timer` +
   `.service`, mirroring the existing `deploy/systemd/` pattern) that runs
   `lifecycle_reconcile.py --desired <path> --interval N` (or a periodic `--once` via the
   timer instead of the script's own loop — either shape works; the module doesn't care
   which). Not written here because it needs a real desired-state yaml path and a decision
   on cadence, both operator calls.
2. **Arming it against the live `nakshatra-unconscious` serve.** Today that serve builds
   its `ChainLifecycle` from `NAKSHATRA_LIFECYCLE_*` env inside `nakshatra_serve.py`'s
   `main()` — there's no long-lived handle an external process can reach to build a
   `LifecycleBackend` from. The CLI's `main()` here calls `serve_lifecycle.from_env()`
   itself (same env the live unit would export) to reconstruct an equivalent controller and
   wraps it under `--model-lifecycle-id`, which works for the CLI's own process but means
   reconcile and the serve are two independent processes each calling
   `SystemdLocalController`/`RosterWorkerController` methods against the same systemd
   units — safe (idempotent start/stop, `is_ready()` is a pure TCP probe) but worth the
   operator's eyes before it's scheduled unattended, since it's the first thing in this
   repo that acts on the live unconscious lifecycle from *outside* the request path.

Neither of these was done as part of this branch — pure module + tests only, per the
brief. `NKS_RECONCILE_APPLY` stays unset / no `--apply` anywhere in this repo until the
operator decides to wire the timer.
