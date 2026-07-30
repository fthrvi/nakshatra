"""lifecycle_reconcile.py — declarative model lifecycle reconciliation (Mesh-LLM
adoption #3: daemon reconciliation, adapted onto our scale-to-zero).

serve_lifecycle.py already answers "is a chain up right now, and should it be
reaped after idling" — but that's all REACTIVE (summon-on-request, reap-on-idle).
What's missing is DESIRED STATE: an operator says, in one yaml, which models
should be `resident` (always up), `on-demand` (scale-to-zero governs it, we
don't touch it), or `absent` (must not be running) — and a small loop converges
reality toward that declaration:

    desired-state yaml  --load_desired-->  {model_id: DesiredEntry}
                                                 |
                          current = backend.list_running()
                                                 |
                                    plan(desired, current)   <- PURE, no I/O
                                                 |
                              [Summon | Reap | Noop, ...] (with reasons)
                                                 |
                           converge(plan, backend, dry_run=True)  <- DRY-RUN DEFAULT

`Reconciler` is the stateful loop driver: it owns the backend + cross-cycle
failure/backoff bookkeeping so a flaky summon/reap is logged and retried next
cycle, never crashes the loop.

Conscious-reserve spirit: reconcile NEVER reaps a model it did not itself see
declared `absent`. A model running-but-undeclared is reported (a Noop action
with a reason), not touched — same instinct as placement_feed's conscious VRAM
reserve: don't subtract/evict what you don't own an opinion about.

Safety: DRY-RUN IS THE DEFAULT everywhere (`converge(..., dry_run=True)`,
`Reconciler.run_once/run_forever(..., dry_run=True)`, the CLI's default).
Real execution needs `--apply` or `NKS_RECONCILE_APPLY=1`.

Backend surface (see `Backend`): `list_running() -> list[str]`,
`summon(model_id) -> None`, `reap(model_id) -> None`. `LifecycleBackend` adapts
this onto the REAL `serve_lifecycle.ChainController` surface (`start`/`stop`/
`is_ready`) — today's serve governs exactly ONE chain per controller, so
multi-model desired-state is expressed by registering one controller per
declared model id. `FakeBackend` is the in-memory test double.

CLI:
  python3 lifecycle_reconcile.py --desired serve_models.desired.yaml --once
  python3 lifecycle_reconcile.py --desired serve_models.desired.yaml --interval 60
  python3 lifecycle_reconcile.py --desired serve_models.desired.yaml --once --apply
"""
from __future__ import annotations

import enum
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── desired state ──────────────────────────────────────────────────────
class Desired(str, enum.Enum):
    RESIDENT = "resident"      # must be up; reconcile summons it if missing, never reaps it
    ON_DEMAND = "on-demand"    # scale-to-zero governs it; reconcile takes no action either way
    ABSENT = "absent"          # must not be running; reconcile reaps it if it's up


@dataclass
class DesiredEntry:
    model_id: str
    desired: Desired
    note: str = ""


class DesiredStateError(ValueError):
    """The desired-state yaml is malformed or invalid — fails loud (no silent
    degrade to an empty/partial desired set, which would make reconcile reap
    or ignore models the operator actually declared)."""


def load_desired(path: str) -> "dict[str, DesiredEntry]":
    """Parse a desired-state yaml (see serve_models.desired.example.yaml) into
    {model_id: DesiredEntry}. Raises DesiredStateError on anything malformed:
    bad YAML, missing 'models' list, missing id, unknown desired value,
    duplicate id."""
    try:
        import yaml
    except ImportError as e:      # pragma: no cover - verified present in .venv
        raise DesiredStateError(
            f"pyyaml is required to parse {path!r} ({e}); "
            f"pip install pyyaml") from e

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except OSError as e:
        raise DesiredStateError(f"{path}: cannot read: {e}") from e
    except yaml.YAMLError as e:
        raise DesiredStateError(f"{path}: invalid YAML: {e}") from e

    if data is None:
        data = {}
    if not isinstance(data, dict) or "models" not in data:
        raise DesiredStateError(f"{path}: missing top-level 'models' list")
    models = data["models"]
    if not isinstance(models, list):
        raise DesiredStateError(f"{path}: 'models' must be a list")

    out: "dict[str, DesiredEntry]" = {}
    for i, entry in enumerate(models):
        if not isinstance(entry, dict):
            raise DesiredStateError(f"{path}: models[{i}] must be a mapping")
        model_id = entry.get("name") or entry.get("id") or entry.get("model")
        if not model_id:
            raise DesiredStateError(f"{path}: models[{i}] missing 'name' (model id)")
        if model_id in out:
            raise DesiredStateError(f"{path}: duplicate model id {model_id!r}")
        raw = entry.get("desired")
        try:
            desired = Desired(raw)
        except ValueError:
            valid = ", ".join(d.value for d in Desired)
            raise DesiredStateError(
                f"{path}: models[{i}] ({model_id!r}) has invalid desired={raw!r}; "
                f"must be one of: {valid}") from None
        out[model_id] = DesiredEntry(model_id=model_id, desired=desired,
                                     note=str(entry.get("note", "") or ""))
    return out


# ── the plan: pure, no I/O ────────────────────────────────────────────
class ActionKind(str, enum.Enum):
    SUMMON = "summon"
    REAP = "reap"
    NOOP = "noop"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    model_id: str
    reason: str


def plan(desired: "dict[str, DesiredEntry]", current) -> "list[Action]":
    """Pure function: no backend I/O, no side effects. Compares declared
    desired-state against the CURRENT set of running model ids and returns the
    convergence plan (Summon/Reap/Noop, each with a reason).

    - resident, not running   -> Summon
    - resident, running       -> Noop (already matches)
    - on-demand, either state -> Noop (scale-to-zero's call, not reconcile's)
    - absent, running         -> Reap
    - absent, not running     -> Noop (already matches)
    - running but NOT declared at all -> Noop ("reporting only"): reconcile
      never reaps a model it wasn't told about — only an explicit
      `desired: absent` entry authorizes a reap.
    """
    current = set(current)
    actions: "list[Action]" = []
    for model_id, entry in desired.items():
        running = model_id in current
        if entry.desired is Desired.RESIDENT:
            if running:
                actions.append(Action(ActionKind.NOOP, model_id,
                                      "resident and already running"))
            else:
                actions.append(Action(ActionKind.SUMMON, model_id,
                                      "declared resident but not running"))
        elif entry.desired is Desired.ON_DEMAND:
            state = "running" if running else "down"
            actions.append(Action(ActionKind.NOOP, model_id,
                                  f"on-demand ({state}) — left to scale-to-zero, "
                                  f"reconcile takes no action"))
        elif entry.desired is Desired.ABSENT:
            if running:
                actions.append(Action(ActionKind.REAP, model_id,
                                      "declared absent but currently running"))
            else:
                actions.append(Action(ActionKind.NOOP, model_id,
                                      "absent and already down"))
        else:  # pragma: no cover - Desired is exhaustive; load_desired already validates
            raise DesiredStateError(f"unhandled desired state {entry.desired!r} for {model_id}")

    declared = set(desired)
    for model_id in sorted(current - declared):
        actions.append(Action(ActionKind.NOOP, model_id,
                              "running but NOT declared in desired-state — reporting only, "
                              "never reaping what reconcile did not itself declare"))
    return actions


# ── the backend interface + implementations ───────────────────────────
class Backend:
    """The interface Reconciler drives. `list_running()` returns the current
    set of resident model ids; `summon`/`reap` act on ONE model id and MAY
    raise — converge()/Reconciler catch it, log it, and hand it to the
    backoff bookkeeping rather than letting it crash the loop."""

    def list_running(self) -> "list[str]":
        raise NotImplementedError

    def summon(self, model_id: str) -> None:
        raise NotImplementedError

    def reap(self, model_id: str) -> None:
        raise NotImplementedError


class FakeBackend(Backend):
    """In-memory backend for tests: tracks a running set + a call log, and can
    be told to fail summon/reap for a given model N times (then succeed) so
    tests can exercise the retry/backoff path without touching anything real."""

    def __init__(self, running: "Optional[set]" = None):
        self.running: "set[str]" = set(running or ())
        self.calls: "list[tuple]" = []
        self._fail_summon: "dict[str, int]" = {}
        self._fail_reap: "dict[str, int]" = {}

    def fail_summon_next(self, model_id: str, times: int = 1) -> None:
        self._fail_summon[model_id] = self._fail_summon.get(model_id, 0) + times

    def fail_reap_next(self, model_id: str, times: int = 1) -> None:
        self._fail_reap[model_id] = self._fail_reap.get(model_id, 0) + times

    def list_running(self) -> "list[str]":
        return sorted(self.running)

    def summon(self, model_id: str) -> None:
        self.calls.append(("summon", model_id))
        if self._fail_summon.get(model_id, 0) > 0:
            self._fail_summon[model_id] -= 1
            raise RuntimeError(f"fake summon failure for {model_id}")
        self.running.add(model_id)

    def reap(self, model_id: str) -> None:
        self.calls.append(("reap", model_id))
        if self._fail_reap.get(model_id, 0) > 0:
            self._fail_reap[model_id] -= 1
            raise RuntimeError(f"fake reap failure for {model_id}")
        self.running.discard(model_id)


class LifecycleBackend(Backend):
    """Adapts serve_lifecycle.py's REAL ChainController surface
    (`start()`/`stop()`/`is_ready()`) onto the Reconciler's Backend protocol,
    keyed by model id. serve_lifecycle governs exactly ONE chain per
    controller today (nakshatra_serve.py builds a single ChainLifecycle from
    env) — multi-model desired-state is expressed by registering one
    controller per declared model id here; when a real multi-model registry
    exists this adapter doesn't need to change, only its construction does."""

    def __init__(self, controllers: "dict[str, object]",
                 log: Callable[[str], None] = print):
        self._controllers = dict(controllers)
        self._log = log

    @classmethod
    def from_chain_lifecycles(cls, lifecycles: "dict[str, object]",
                              log: Callable[[str], None] = print) -> "LifecycleBackend":
        """Build from {model_id: serve_lifecycle.ChainLifecycle} — unwraps
        each to its `.controller` (the actual start/stop/is_ready surface).
        The ChainLifecycle's own idle-grace/in-flight-request policy still
        governs day-to-day scale-to-zero; reconcile only nudges the
        underlying controller toward the declared shape (summon a `resident`
        that's down, reap an `absent` that's up)."""
        return cls({mid: lc.controller for mid, lc in lifecycles.items()}, log=log)

    def _require(self, model_id: str):
        ctrl = self._controllers.get(model_id)
        if ctrl is None:
            raise KeyError(f"lifecycle_reconcile: no controller registered "
                           f"for declared model {model_id!r}")
        return ctrl

    def list_running(self) -> "list[str]":
        out = []
        for model_id, ctrl in self._controllers.items():
            try:
                if ctrl.is_ready():
                    out.append(model_id)
            except Exception as e:
                self._log(f"[reconcile] is_ready probe failed for {model_id}: {e}")
        return out

    def summon(self, model_id: str) -> None:
        self._require(model_id).start()

    def reap(self, model_id: str) -> None:
        self._require(model_id).stop()


# ── converge: execute (or dry-run) a plan ─────────────────────────────
@dataclass
class ActionResult:
    action: Action
    executed: bool     # False for Noop, dry-run, and backoff-held actions
    ok: bool            # True unless a summon/reap call raised
    detail: str = ""


def converge(actions: "list[Action]", backend: "Backend", *, dry_run: bool = True,
            log: Callable[[str], None] = print) -> "list[ActionResult]":
    """Execute a plan against `backend`. DRY-RUN IS THE DEFAULT: with
    dry_run=True (the default), no backend.summon/reap call happens — every
    Summon/Reap is logged as '[dry-run] would …' and returned unexecuted.
    A backend exception is caught and logged here, never raised — this is a
    single non-crashing pass; retry/backoff bookkeeping across cycles is
    Reconciler's job, not converge()'s."""
    results: "list[ActionResult]" = []
    for a in actions:
        if a.kind is ActionKind.NOOP:
            results.append(ActionResult(a, executed=False, ok=True, detail=a.reason))
            continue
        verb = "summon" if a.kind is ActionKind.SUMMON else "reap"
        if dry_run:
            log(f"[reconcile] [dry-run] would {verb} {a.model_id} ({a.reason})")
            results.append(ActionResult(a, executed=False, ok=True, detail="dry-run"))
            continue
        try:
            log(f"[reconcile] {verb} {a.model_id} ({a.reason})")
            fn = backend.summon if a.kind is ActionKind.SUMMON else backend.reap
            fn(a.model_id)
            results.append(ActionResult(a, executed=True, ok=True, detail="ok"))
        except Exception as e:
            log(f"[reconcile] {verb} {a.model_id} FAILED: {e}")
            results.append(ActionResult(a, executed=True, ok=False, detail=str(e)))
    return results


# ── Reconciler: the stateful loop driver ──────────────────────────────
class Reconciler:
    """Owns the backend plus cross-cycle failure/backoff state. `plan()` and
    `converge()` above are pure/stateless per call; this is what makes a
    repeated `tick()` safe to run forever: a failed summon/reap is logged and
    NOT retried again until its backoff window elapses (exponential, capped),
    so a wedged node doesn't get hammered every cycle. `tick()` itself never
    raises — a bad desired-state load or a backend blip is logged and the
    loop just tries again next cycle."""

    def __init__(self, backend: "Backend", *, base_backoff_s: float = 30.0,
                max_backoff_s: float = 900.0, log: Callable[[str], None] = print,
                clock: Callable[[], float] = time.monotonic):
        self.backend = backend
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        self._log = log
        self._clock = clock
        self._fail_count: "dict[str, int]" = {}
        self._retry_after: "dict[str, float]" = {}

    def tick(self, desired: "dict[str, DesiredEntry]", *,
             dry_run: bool = True) -> "list[ActionResult]":
        now = self._clock()
        try:
            current = set(self.backend.list_running())
        except Exception as e:
            self._log(f"[reconcile] list_running failed: {e} — skipping this cycle")
            return []

        full_plan = plan(desired, current)
        runnable: "list[Action]" = []
        held: "list[ActionResult]" = []
        for a in full_plan:
            if a.kind is not ActionKind.NOOP:
                until = self._retry_after.get(a.model_id)
                if until is not None and now < until:
                    remaining = until - now
                    self._log(f"[reconcile] {a.model_id}: backoff active "
                             f"({remaining:.0f}s left) — skipping this cycle")
                    held.append(ActionResult(a, executed=False, ok=True,
                                             detail=f"backoff {remaining:.0f}s remaining"))
                    continue
            runnable.append(a)

        results = converge(runnable, self.backend, dry_run=dry_run, log=self._log)

        if not dry_run:
            for r in results:
                if r.action.kind is ActionKind.NOOP or not r.executed:
                    continue
                mid = r.action.model_id
                if r.ok:
                    if mid in self._fail_count:
                        self._log(f"[reconcile] {mid}: recovered, clearing backoff")
                    self._fail_count.pop(mid, None)
                    self._retry_after.pop(mid, None)
                else:
                    n = self._fail_count.get(mid, 0) + 1
                    self._fail_count[mid] = n
                    backoff = min(self.max_backoff_s, self.base_backoff_s * (2 ** (n - 1)))
                    self._retry_after[mid] = now + backoff
                    self._log(f"[reconcile] {mid}: failure #{n}, backing off {backoff:.0f}s")

        return results + held

    def run_once(self, desired: "dict[str, DesiredEntry]", *,
                 dry_run: bool = True) -> "list[ActionResult]":
        return self.tick(desired, dry_run=dry_run)

    def run_forever(self, desired_path: str, *, interval_s: float, dry_run: bool = True,
                    stop_event: "Optional[threading.Event]" = None) -> None:
        """Loop mode: reload the desired-state yaml each cycle (so an edit to
        it takes effect on the next tick, no restart needed) and tick(). Never
        raises out of the loop — a malformed yaml or backend blip is logged
        and retried next interval."""
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            try:
                desired = load_desired(desired_path)
                self.tick(desired, dry_run=dry_run)
            except DesiredStateError as e:
                self._log(f"[reconcile] desired-state error: {e} — skipping this cycle "
                         f"(fix the yaml)")
            except Exception as e:  # pragma: no cover - belt & suspenders, loop must never die
                self._log(f"[reconcile] unexpected error this cycle: {e}")
            stop_event.wait(interval_s)


# ── CLI ────────────────────────────────────────────────────────────────
def _default_log(msg: str) -> None:
    print(msg, flush=True)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Declarative model lifecycle reconciliation: converge the "
                    "live serve_lifecycle chain(s) toward an operator-declared "
                    "desired-state yaml (resident/on-demand/absent).")
    ap.add_argument("--desired", required=True,
                    help="path to the desired-state yaml "
                         "(see serve_models.desired.example.yaml)")
    ap.add_argument("--model-lifecycle-id",
                    default=os.environ.get("NAKSHATRA_LIFECYCLE_ROSTER_MODEL", ""),
                    help="model id the live serve_lifecycle.from_env() chain answers "
                         "for (today's serve governs ONE chain; multi-model wiring "
                         "is the remaining ask — see docs/findings/lifecycle-reconcile.md)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true",
                      help="run a single reconcile pass and exit (default)")
    mode.add_argument("--interval", type=float, default=None, metavar="SECONDS",
                      help="loop mode: reconcile every SECONDS until interrupted "
                           "(reloads --desired each cycle)")
    ap.add_argument("--apply", action="store_true",
                    help="actually summon/reap. Without this (or "
                         "NKS_RECONCILE_APPLY=1) every run is a dry-run: actions "
                         "are planned and logged, nothing is executed.")
    args = ap.parse_args(argv)

    dry_run = not (args.apply or os.environ.get("NKS_RECONCILE_APPLY") == "1")
    if dry_run:
        _default_log("[reconcile] DRY-RUN (default) — pass --apply or "
                     "NKS_RECONCILE_APPLY=1 to actually summon/reap")

    try:
        desired = load_desired(args.desired)
    except DesiredStateError as e:
        _default_log(f"[reconcile] FATAL: {e}")
        return 2

    if not args.model_lifecycle_id:
        _default_log("[reconcile] no --model-lifecycle-id / "
                     "NAKSHATRA_LIFECYCLE_ROSTER_MODEL set — nothing wired to a "
                     "real controller, nothing to reconcile against. (Build a "
                     "LifecycleBackend/Reconciler programmatically for anything "
                     "beyond the single live chain.)")
        return 0

    import serve_lifecycle
    lc = serve_lifecycle.from_env(log=_default_log)
    if lc is None:
        _default_log("[reconcile] serve_lifecycle.from_env() returned nothing "
                     "(no NAKSHATRA_LIFECYCLE_* env set) — nothing to reconcile "
                     "against.")
        return 0
    backend = LifecycleBackend({args.model_lifecycle_id: lc.controller}, log=_default_log)
    reconciler = Reconciler(backend, log=_default_log)

    if args.interval:
        reconciler.run_forever(args.desired, interval_s=args.interval, dry_run=dry_run)
    else:
        reconciler.run_once(desired, dry_run=dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
