"""lifecycle_reconcile.py — declarative model lifecycle reconciliation, tested without
GPU/network (FakeBackend + a stub ChainController-shaped object for the adapter).

Proves: plan() is pure (missing->summon, declared-absent-and-running->reap,
unknown-running->report-not-reap, steady state->noop, on-demand is always a noop
regardless of running state); load_desired() fails loud on malformed yaml; converge()
executes nothing in dry-run (the default); Reconciler retries a failed summon/reap with
backoff next cycle and never raises; LifecycleBackend adapts the REAL ChainController
surface (start/stop/is_ready)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import lifecycle_reconcile as rec  # noqa: E402


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _entry(model_id, desired, note=""):
    return rec.DesiredEntry(model_id=model_id, desired=rec.Desired(desired), note=note)


# ── plan() — pure ──────────────────────────────────────────────────────
def test_plan_missing_resident_summons():
    desired = {"m1": _entry("m1", "resident")}
    actions = rec.plan(desired, current=set())
    assert len(actions) == 1
    a = actions[0]
    assert a.kind is rec.ActionKind.SUMMON and a.model_id == "m1"
    assert "resident" in a.reason and "not running" in a.reason


def test_plan_declared_absent_but_running_reaps():
    desired = {"m1": _entry("m1", "absent")}
    actions = rec.plan(desired, current={"m1"})
    assert len(actions) == 1
    assert actions[0].kind is rec.ActionKind.REAP and actions[0].model_id == "m1"


def test_plan_unknown_running_is_reported_not_reaped():
    # m1 is running but NOT in desired at all — must be Noop, never Reap.
    desired = {"m2": _entry("m2", "resident")}
    actions = rec.plan(desired, current={"m1", "m2"})
    kinds = {a.model_id: a.kind for a in actions}
    assert kinds["m1"] is rec.ActionKind.NOOP
    assert "NOT declared" in [a for a in actions if a.model_id == "m1"][0].reason
    assert kinds["m2"] is rec.ActionKind.NOOP  # m2 resident + already running


def test_plan_steady_state_all_noop():
    desired = {
        "resident-up": _entry("resident-up", "resident"),
        "absent-down": _entry("absent-down", "absent"),
    }
    actions = rec.plan(desired, current={"resident-up"})
    assert all(a.kind is rec.ActionKind.NOOP for a in actions)
    assert len(actions) == 2


def test_plan_on_demand_always_noop_regardless_of_state():
    desired = {"m1": _entry("m1", "on-demand")}
    up = rec.plan(desired, current={"m1"})
    down = rec.plan(desired, current=set())
    assert up[0].kind is rec.ActionKind.NOOP
    assert down[0].kind is rec.ActionKind.NOOP
    assert "on-demand" in up[0].reason and "on-demand" in down[0].reason


def test_plan_absent_and_already_down_is_noop():
    desired = {"m1": _entry("m1", "absent")}
    actions = rec.plan(desired, current=set())
    assert actions[0].kind is rec.ActionKind.NOOP
    assert "already down" in actions[0].reason


# ── load_desired() — fails loud ───────────────────────────────────────
def test_load_desired_parses_the_shipped_example():
    path = SCRIPTS_DIR / "serve_models.desired.example.yaml"
    desired = rec.load_desired(str(path))
    assert set(desired) == {"nakshatra-unconscious-30b", "qwen3-coder-30b",
                            "llama-3.3-70b-experimental"}
    assert desired["nakshatra-unconscious-30b"].desired is rec.Desired.RESIDENT
    assert desired["qwen3-coder-30b"].desired is rec.Desired.ON_DEMAND
    assert desired["llama-3.3-70b-experimental"].desired is rec.Desired.ABSENT


def test_load_desired_bad_yaml_fails_loud(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [this is not: valid: yaml: at: all: [")
    try:
        rec.load_desired(str(bad))
        assert False, "expected DesiredStateError"
    except rec.DesiredStateError:
        pass


def test_load_desired_missing_models_key_fails_loud(tmp_path):
    p = tmp_path / "nomodels.yaml"
    p.write_text("not_models: []\n")
    try:
        rec.load_desired(str(p))
        assert False, "expected DesiredStateError"
    except rec.DesiredStateError as e:
        assert "models" in str(e)


def test_load_desired_bad_desired_value_fails_loud(tmp_path):
    p = tmp_path / "badval.yaml"
    p.write_text("models:\n  - name: m1\n    desired: sort-of-up\n")
    try:
        rec.load_desired(str(p))
        assert False, "expected DesiredStateError"
    except rec.DesiredStateError as e:
        assert "sort-of-up" in str(e)


def test_load_desired_duplicate_id_fails_loud(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text("models:\n  - name: m1\n    desired: resident\n"
                 "  - name: m1\n    desired: absent\n")
    try:
        rec.load_desired(str(p))
        assert False, "expected DesiredStateError"
    except rec.DesiredStateError as e:
        assert "duplicate" in str(e).lower()


def test_load_desired_missing_name_fails_loud(tmp_path):
    p = tmp_path / "noname.yaml"
    p.write_text("models:\n  - desired: resident\n")
    try:
        rec.load_desired(str(p))
        assert False, "expected DesiredStateError"
    except rec.DesiredStateError as e:
        assert "name" in str(e)


# ── converge() — dry-run is the default, executes nothing ─────────────
def test_converge_dry_run_default_executes_nothing():
    backend = rec.FakeBackend()
    actions = [rec.Action(rec.ActionKind.SUMMON, "m1", "test")]
    results = rec.converge(actions, backend, log=lambda *_: None)  # dry_run defaults True
    assert backend.calls == []
    assert results[0].executed is False and results[0].ok is True


def test_converge_apply_executes_and_updates_backend():
    backend = rec.FakeBackend()
    actions = [rec.Action(rec.ActionKind.SUMMON, "m1", "test")]
    results = rec.converge(actions, backend, dry_run=False, log=lambda *_: None)
    assert backend.calls == [("summon", "m1")]
    assert "m1" in backend.running
    assert results[0].executed is True and results[0].ok is True


def test_converge_noop_never_touches_backend():
    backend = rec.FakeBackend()
    actions = [rec.Action(rec.ActionKind.NOOP, "m1", "steady")]
    rec.converge(actions, backend, dry_run=False, log=lambda *_: None)
    assert backend.calls == []


def test_converge_backend_failure_is_caught_not_raised():
    backend = rec.FakeBackend()
    backend.fail_summon_next("m1")
    actions = [rec.Action(rec.ActionKind.SUMMON, "m1", "test")]
    results = rec.converge(actions, backend, dry_run=False, log=lambda *_: None)
    assert results[0].ok is False
    assert "fake summon failure" in results[0].detail
    assert "m1" not in backend.running  # summon() only adds on success


# ── Reconciler — end to end, retry/backoff, dry-run ────────────────────
def test_reconciler_dry_run_end_to_end_executes_nothing():
    backend = rec.FakeBackend()
    desired = {"m1": _entry("m1", "resident"), "m2": _entry("m2", "absent")}
    backend.running.add("m2")
    r = rec.Reconciler(backend, log=lambda *_: None)
    results = r.run_once(desired, dry_run=True)
    assert backend.calls == []
    assert backend.running == {"m2"}  # unchanged
    kinds = {res.action.model_id: res.action.kind for res in results}
    assert kinds["m1"] is rec.ActionKind.SUMMON
    assert kinds["m2"] is rec.ActionKind.REAP


def test_reconciler_apply_converges_missing_and_extra():
    backend = rec.FakeBackend()
    backend.running.add("stale")
    desired = {"m1": _entry("m1", "resident"), "stale": _entry("stale", "absent")}
    r = rec.Reconciler(backend, log=lambda *_: None)
    r.run_once(desired, dry_run=False)
    assert backend.running == {"m1"}
    assert ("summon", "m1") in backend.calls
    assert ("reap", "stale") in backend.calls


def test_reconciler_never_reaps_unknown_running():
    backend = rec.FakeBackend()
    backend.running.add("mystery")
    desired = {"m1": _entry("m1", "resident")}
    r = rec.Reconciler(backend, log=lambda *_: None)
    r.run_once(desired, dry_run=False)
    assert "mystery" in backend.running  # never touched
    assert all(call[1] != "mystery" for call in backend.calls)


def test_reconciler_retries_summon_failure_with_backoff():
    backend = rec.FakeBackend()
    backend.fail_summon_next("m1", times=1)
    desired = {"m1": _entry("m1", "resident")}
    clock = [1000.0]
    r = rec.Reconciler(backend, base_backoff_s=10.0, clock=lambda: clock[0],
                       log=lambda *_: None)

    results1 = r.run_once(desired, dry_run=False)
    assert results1[0].ok is False
    assert backend.calls == [("summon", "m1")]
    assert "m1" not in backend.running

    # same instant: still within backoff -> held, no second attempt
    results2 = r.run_once(desired, dry_run=False)
    assert backend.calls == [("summon", "m1")]
    assert results2[0].executed is False
    assert "backoff" in results2[0].detail

    # advance the clock past the backoff window -> retried, this time succeeds
    clock[0] += 11.0
    results3 = r.run_once(desired, dry_run=False)
    assert backend.calls == [("summon", "m1"), ("summon", "m1")]
    assert results3[0].ok is True
    assert "m1" in backend.running


def test_reconciler_backoff_never_applies_in_dry_run():
    # dry-run must never populate backoff state (nothing was actually attempted)
    backend = rec.FakeBackend()
    backend.fail_summon_next("m1", times=99)
    desired = {"m1": _entry("m1", "resident")}
    clock = [0.0]
    r = rec.Reconciler(backend, clock=lambda: clock[0], log=lambda *_: None)
    r.run_once(desired, dry_run=True)
    r.run_once(desired, dry_run=True)
    assert backend.calls == []
    assert r._retry_after == {}


def test_reconciler_loop_never_crashes_on_bad_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("not_models: true\n")
    backend = rec.FakeBackend()
    r = rec.Reconciler(backend, log=lambda *_: None)
    stop = __import__("threading").Event()
    stop.set()  # run the loop body exactly once then exit
    r.run_forever(str(p), interval_s=0.01, dry_run=True, stop_event=stop)
    # if we get here without an exception, the loop survived the bad yaml


# ── LifecycleBackend — adapts the REAL ChainController surface ────────
class _StubController:
    """Duck-types serve_lifecycle.ChainController without touching systemd/SSH/GPU."""
    def __init__(self, ready=False):
        self._ready = ready
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1
        self._ready = True

    def stop(self):
        self.stopped += 1
        self._ready = False

    def is_ready(self):
        return self._ready


def test_lifecycle_backend_uses_real_controller_shape():
    import serve_lifecycle as sl
    # _StubController duck-types serve_lifecycle.ChainController's real surface
    # (start/stop/is_ready) — assert the real base class exposes exactly that.
    assert {"start", "stop", "is_ready"} <= set(dir(sl.ChainController))
    up = _StubController(ready=True)
    down = _StubController(ready=False)
    backend = rec.LifecycleBackend({"up-model": up, "down-model": down}, log=lambda *_: None)
    assert backend.list_running() == ["up-model"]


def test_lifecycle_backend_summon_reap_call_through():
    ctrl = _StubController(ready=False)
    backend = rec.LifecycleBackend({"m1": ctrl}, log=lambda *_: None)
    backend.summon("m1")
    assert ctrl.started == 1
    assert backend.list_running() == ["m1"]
    backend.reap("m1")
    assert ctrl.stopped == 1
    assert backend.list_running() == []


def test_lifecycle_backend_unknown_model_raises_keyerror():
    backend = rec.LifecycleBackend({}, log=lambda *_: None)
    try:
        backend.summon("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_lifecycle_backend_is_ready_exception_reported_as_not_running():
    class _Flaky(_StubController):
        def is_ready(self):
            raise RuntimeError("probe blew up")
    logs = []
    backend = rec.LifecycleBackend({"m1": _Flaky()}, log=logs.append)
    assert backend.list_running() == []
    assert any("probe blew up" in m for m in logs)


def test_lifecycle_backend_from_chain_lifecycles_unwraps_controller():
    class _FakeChainLifecycle:
        def __init__(self, controller):
            self.controller = controller
    ctrl = _StubController(ready=True)
    backend = rec.LifecycleBackend.from_chain_lifecycles(
        {"m1": _FakeChainLifecycle(ctrl)}, log=lambda *_: None)
    assert backend.list_running() == ["m1"]


# ── end-to-end with the real reconciler + LifecycleBackend, dry-run only ──
def test_end_to_end_plan_against_lifecycle_backend_dry_run():
    resident_down = _StubController(ready=False)
    absent_up = _StubController(ready=True)
    backend = rec.LifecycleBackend(
        {"resident-model": resident_down, "absent-model": absent_up}, log=lambda *_: None)
    desired = {
        "resident-model": _entry("resident-model", "resident"),
        "absent-model": _entry("absent-model", "absent"),
    }
    r = rec.Reconciler(backend, log=lambda *_: None)
    results = r.run_once(desired, dry_run=True)
    # dry-run: nothing on the stub controllers actually changed
    assert resident_down.started == 0
    assert absent_up.stopped == 0
    kinds = {res.action.model_id: res.action.kind for res in results}
    assert kinds["resident-model"] is rec.ActionKind.SUMMON
    assert kinds["absent-model"] is rec.ActionKind.REAP


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
