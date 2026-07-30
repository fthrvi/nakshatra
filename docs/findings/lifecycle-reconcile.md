# Declarative model lifecycle reconciliation (Mesh-LLM adoption #3)

**Status:** module + tests landed on `serving/lifecycle-reconcile`. Controller-registry
handle + a staged (not enabled) systemd timer landed on `serving/live-seams` — see
"What landed since" below. Still not armed on any live serve.

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

## What landed since (branch `serving/live-seams`, 2026-07-29)

Both items below are now built. Neither is ARMED — no unit is installed/enabled, and
`NKS_RECONCILE_APPLY` still isn't set anywhere in this repo. That decision remains the
operator's, per the original ask.

1. **The controller-registry handle.** `nakshatra_serve.py` now carries a module-level
   `controllers: dict[str, ChainController]` registry (just above `build_server`) that
   `main()` populates via `_register_lifecycle_controller()` right after it arms
   scale-to-zero — keyed by `NAKSHATRA_LIFECYCLE_MODEL_ID` (new), falling back to the
   existing `NAKSHATRA_LIFECYCLE_ROSTER_MODEL`, then the literal `"default"`.
   `LifecycleBackend.from_serve_registry()` (new classmethod, `scripts/lifecycle_reconcile.py`)
   reads it directly — either an injected dict (tests) or, by default, imports
   `nakshatra_serve` and reads its live `controllers`.

   **Honesty note, unchanged from before:** this handle is real but only reaches an
   IN-PROCESS consumer — something importing `nakshatra_serve` in the SAME interpreter
   that ran its `main()` (a future embedded reconciler thread, a REPL attached to the
   live serve, a test). It does **not** bridge the process boundary the standalone CLI /
   systemd timer below uses — that's still a separate OS process, and its `main()` still
   calls `serve_lifecycle.from_env()` itself to reconstruct an equivalent controller from
   the same env, exactly as described in the paragraph this replaced. Two independent
   processes driving the same idempotent systemd-unit/TCP-probe primitives remains safe;
   `from_serve_registry()` is the seam a *future* in-process reconciler would use to skip
   that duplication, not something that closes the gap for today's separate-process CLI.

2. **A systemd timer unit** — `deploy/systemd/nakshatra-reconcile.service` +
   `deploy/systemd/nakshatra-reconcile.timer` (mirroring the existing `deploy/systemd/`
   pattern, `OnCalendar=*:0/10`). **Staged only: nothing in this repo runs
   `systemctl --user enable`/`start` on it.** The service runs
   `lifecycle_reconcile.py --desired <path> --model-lifecycle-id <id> --once` with no
   `--apply` and no `NKS_RECONCILE_APPLY` — every scheduled run is a dry-run pass that
   only logs what it would summon/reap. The unit file's own comments say, in the
   operator's words: arming real execution (adding `--apply`, or setting
   `NKS_RECONCILE_APPLY=1`) is the operator's explicit call, made after reading a run or
   two of the dry-run logs — this remains the first thing in the repo that would act on
   the live unconscious lifecycle from *outside* the request path, so it should not go
   live silently.

## What's still the operator's call

- Copying `scripts/serve_models.desired.example.yaml` to a real path (default
  `~/.nakshatra/serve_models.desired.yaml`, documented in the staged `.service` file) and
  declaring actual model ids/desired-states in it.
- Setting `MODEL_LIFECYCLE_ID` in the staged unit (or `NAKSHATRA_LIFECYCLE_MODEL_ID` on
  the live serve) so the two line up — today it's a label only (see the honesty note
  above), but it should still name the real chain for the plan/log output to make sense.
- Actually enabling the timer (`systemctl --user enable --now nakshatra-reconcile.timer`
  after copying both unit files into `~/.config/systemd/user/`) and, once its dry-run
  logs look right, arming `--apply`/`NKS_RECONCILE_APPLY=1`.
- A genuinely multi-model live registry: `controllers` today gets exactly the ONE
  controller `nakshatra_serve.py`'s single `ChainLifecycle` builds (unchanged
  architecture) — `from_serve_registry()` reads whatever's in that dict, so a real
  multi-model registry only needs `_register_lifecycle_controller()`'s construction to
  change, not the reconcile side.
