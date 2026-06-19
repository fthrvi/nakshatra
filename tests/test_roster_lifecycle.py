"""RosterWorkerController — autonomous from_roster worker launch, tested without GPU/network.

Proves: it launches ONE local worker per planner slot (self-provisioning ranges), skips REMOTE
slots (those belong to the mesh/SSH controller), skips already-serving ports, tears down on stop,
and from_env assembles it when NAKSHATRA_LIFECYCLE_ROSTER_MODEL is set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import serve_lifecycle as sl


class _FakeProc:
    def __init__(self, w):
        self.w = w
        self.terminated = False
    def terminate(self):
        self.terminated = True
    def wait(self, timeout=None):
        return 0
    def kill(self):
        self.terminated = True


def _chain(workers):
    return {"model": {"id": "m", "hidden_size": 4096, "num_blocks": 32}, "workers": workers}


def _spec():
    return sl.RosterWorkerSpec(model_id="dsr1", hidden_size=4096, package_location="/pkg",
                              num_layers=32)


def _controller(workers, open_ports=()):
    launched = []

    def launch(w):
        p = _FakeProc(w)
        launched.append(p)
        return p

    c = sl.RosterWorkerController(_spec(), plan_fn=lambda: _chain(workers), launch_fn=launch)
    c._port_open = staticmethod(lambda host, port: port in open_ports)
    return c, launched


def test_launches_one_local_worker_per_slot():
    workers = [{"id": "a", "address": "127.0.0.1", "port": 5560, "layer_range": [0, 16], "mode": "first"},
               {"id": "b", "address": "127.0.0.1", "port": 5561, "layer_range": [16, 32], "mode": "last"}]
    c, launched = _controller(workers)
    c.start()
    assert len(launched) == 2
    assert {p.w["port"] for p in launched} == {5560, 5561}
    assert c._probes == [("127.0.0.1", 5560), ("127.0.0.1", 5561)]


def test_skips_remote_slots():
    # a remote slot is probed (part of the chain) but NOT launched locally
    workers = [{"id": "a", "address": "127.0.0.1", "port": 5560, "layer_range": [0, 16], "mode": "first"},
               {"id": "r", "address": "10.50.0.9", "port": 5560, "layer_range": [16, 32], "mode": "last"}]
    c, launched = _controller(workers)
    c.start()
    assert [p.w["id"] for p in launched] == ["a"], "only the local slot should launch"
    assert ("10.50.0.9", 5560) in c._probes


def test_skips_already_serving_ports():
    workers = [{"id": "a", "address": "127.0.0.1", "port": 5560, "layer_range": [0, 16], "mode": "first"},
               {"id": "b", "address": "127.0.0.1", "port": 5561, "layer_range": [16, 32], "mode": "last"}]
    c, launched = _controller(workers, open_ports=(5560,))   # a already up
    c.start()
    assert [p.w["port"] for p in launched] == [5561]


def test_stop_terminates_launched():
    workers = [{"id": "a", "address": "127.0.0.1", "port": 5560, "layer_range": [0, 32], "mode": "solo"}]
    c, launched = _controller(workers)
    c.start()
    c.stop()
    assert all(p.terminated for p in launched)
    assert c._procs == []


def test_adopts_and_reaps_prior_process_workers():
    """Cross-restart squatting fix: a port already served by a PRIOR serve process is
    recorded as adopted and reaped BY PORT on stop — not skipped-and-orphaned (which
    left ~6GB squatting until a manual kill)."""
    workers = [{"id": "a", "address": "127.0.0.1", "port": 5560, "layer_range": [0, 16], "mode": "first"},
               {"id": "b", "address": "127.0.0.1", "port": 5561, "layer_range": [16, 32], "mode": "last"}]
    c, launched = _controller(workers, open_ports=(5560,))   # a = a stale prior-process worker
    c.start()
    assert c._adopted_ports == [5560]                         # recorded for port-reaping
    assert [p.w["port"] for p in launched] == [5561]          # b launched fresh (we own it)
    reaped = []
    c._reap_listener = staticmethod(lambda port: reaped.append(port))
    c.stop()
    assert reaped == [5560]                                   # adopted worker reaped by port
    assert c._procs == [] and c._adopted_ports == []          # and our own launched proc torn down


def test_is_ready_requires_all_probes_open():
    workers = [{"id": "a", "address": "127.0.0.1", "port": 5560, "layer_range": [0, 16], "mode": "first"},
               {"id": "b", "address": "127.0.0.1", "port": 5561, "layer_range": [16, 32], "mode": "last"}]
    c, _ = _controller(workers, open_ports=(5560, 5561))
    c.start()
    assert c.is_ready() is True
    c._port_open = staticmethod(lambda host, port: port == 5560)  # b drops
    assert c.is_ready() is False


def _gated_controller(workers, placement):
    """A controller with the lifecycle gate ON, its placement decision stubbed (so we test the
    start() gating without an admission control plane / roster)."""
    launched = []

    def launch(w):
        p = _FakeProc(w)
        launched.append(p)
        return p

    spec = sl.RosterWorkerSpec(model_id="dsr1", hidden_size=4096, package_location="/pkg",
                               num_layers=32, lifecycle_gate=True)
    c = sl.RosterWorkerController(spec, plan_fn=lambda: _chain(workers), launch_fn=launch)
    c._port_open = staticmethod(lambda host, port: False)
    c._lifecycle_placement = lambda: placement
    return c, launched


def _placement(state):
    import unconscious_lifecycle as ul
    return ul.Placement(state, [], 0, state.value)


def test_gate_idle_does_not_summon():
    import unconscious_lifecycle as ul
    workers = [{"id": "a", "address": "127.0.0.1", "port": 5560, "layer_range": [0, 32], "mode": "solo"}]
    c, launched = _gated_controller(workers, _placement(ul.UnconsciousState.IDLE_NO_HARDWARE))
    c.start()
    assert launched == [], "no owned hardware → must not summon"
    assert c._procs == []


def test_gate_disabled_does_not_summon():
    import unconscious_lifecycle as ul
    workers = [{"id": "a", "address": "127.0.0.1", "port": 5560, "layer_range": [0, 32], "mode": "solo"}]
    c, launched = _gated_controller(workers, _placement(ul.UnconsciousState.DISABLED))
    c.start()
    assert launched == [], "operator disabled → must not summon"


def test_gate_ready_summons_as_normal():
    import unconscious_lifecycle as ul
    workers = [{"id": "a", "address": "127.0.0.1", "port": 5560, "layer_range": [0, 16], "mode": "first"},
               {"id": "b", "address": "127.0.0.1", "port": 5561, "layer_range": [16, 32], "mode": "last"}]
    c, launched = _gated_controller(workers, _placement(ul.UnconsciousState.READY))
    c.start()
    assert {p.w["port"] for p in launched} == {5560, 5561}, "READY → summon as usual"


def test_gate_off_by_default_never_calls_lifecycle():
    # default spec (lifecycle_gate=False) must behave EXACTLY as before — never consult the gate.
    workers = [{"id": "a", "address": "127.0.0.1", "port": 5560, "layer_range": [0, 32], "mode": "solo"}]
    c, launched = _controller(workers)
    called = []
    c._lifecycle_placement = lambda: called.append(True)
    c.start()
    assert called == [], "gate off → _lifecycle_placement must not be consulted"
    assert len(launched) == 1


def test_from_env_builds_roster_controller(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("NAKSHATRA_LIFECYCLE"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NAKSHATRA_LIFECYCLE_ROSTER_MODEL", "dsr1")
    monkeypatch.setenv("NAKSHATRA_LIFECYCLE_ROSTER_HIDDEN_SIZE", "4096")
    monkeypatch.setenv("NAKSHATRA_LIFECYCLE_ROSTER_NUM_LAYERS", "32")
    monkeypatch.setenv("NAKSHATRA_LIFECYCLE_ROSTER_PACKAGE", "/pkg")
    lc = sl.from_env(log=lambda *_: None)
    assert lc is not None
    assert isinstance(lc.controller, sl.RosterWorkerController)
    assert lc.controller.spec.model_id == "dsr1" and lc.controller.spec.hidden_size == 4096


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
