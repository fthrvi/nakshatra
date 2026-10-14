"""Pillar warm-pool advisories, with fake daemons, clocks, and HOME."""
from __future__ import annotations

import collections
import hashlib
import json
import struct
import threading
import time

import pytest

import worker


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


class FakeClock:
    def __init__(self, value=1_700_000_000.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeDaemon:
    def __init__(self, *, entered=None, release=None):
        self.closed = False
        self.recent_rpc_ms = collections.deque(maxlen=20)
        self.entered = entered
        self.release = release

    def info(self):
        return {
            "layer_start": 0, "layer_end": 1, "n_embd": 4,
            "has_token_embd": True, "has_lm_head": True, "n_vocab": 8,
        }

    def gpu_offload_status(self):
        return {"n_offloaded": 1, "total_layers": 1, "uses_gpu": True}

    def call(self, *_args, **_kwargs):
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=2)
        return 0, struct.pack("<Ii", 0, 7)

    def close(self):
        self.closed = True


def make_controller(tmp_path, clock, factory, *, idle=10, grace=5,
                    lease=False, enabled=True):
    model = tmp_path / "slice.gguf"
    model.write_bytes(b"verified model slice")
    return worker.IdleDaemonController(
        factory,
        str(model),
        hashlib.sha256(model.read_bytes()).hexdigest(),
        idle,
        release_grace_s=grace,
        owner_lease_enabled=lease,
        now=clock,
        wall_now=clock,
        model_id="model-a",
        warm_directives_enabled=enabled,
    )


def directive(clock, action, lifetime=60):
    return worker.WarmDirective("model-a", action, clock() + lifetime)


def test_keep_suppresses_release_grace_and_idle_until_expiry(tmp_path):
    clock = FakeClock()
    daemon = FakeDaemon()
    controller = make_controller(tmp_path, clock, lambda: daemon)

    assert controller.acquire_session("done") is True
    controller.release_session("done")
    controller.apply_warm_directive(directive(clock, "keep", 20))
    clock.advance(11)
    assert controller.unload_if_idle(force_window=True) is False
    assert daemon.closed is False

    clock.advance(10)
    assert controller.unload_if_idle() is True
    assert daemon.closed is True


def test_prewarm_loads_standby_only_while_owner_lease_is_live(tmp_path):
    clock = FakeClock()
    daemons = []

    def factory():
        daemons.append(FakeDaemon())
        return daemons[-1]

    controller = make_controller(tmp_path, clock, factory, lease=True)
    warm = directive(clock, "prewarm")
    controller.apply_warm_directive(warm)
    assert controller.residency == "away"
    assert len(daemons) == 1

    controller.update_owner_lease("live", "owner clear", resume=True)
    assert controller.residency == "standby"
    controller.apply_warm_directive(warm)
    assert controller.residency == "loaded"
    assert len(daemons) == 2


def test_prewarm_is_ignored_while_owner_lease_is_stale(tmp_path):
    clock = FakeClock()
    daemons = []

    def factory():
        daemons.append(FakeDaemon())
        return daemons[-1]

    controller = make_controller(tmp_path, clock, factory, lease=True)
    controller.update_owner_lease("live", "owner clear", resume=True)
    controller.update_owner_lease("stale", "owner heartbeat stale")
    controller.apply_warm_directive(directive(clock, "prewarm"))

    assert controller.residency == "standby"
    assert len(daemons) == 1


def test_release_unloads_immediately_when_idle_safe(tmp_path):
    clock = FakeClock()
    daemon = FakeDaemon()
    controller = make_controller(
        tmp_path, clock, lambda: daemon, idle=1_000, grace=-1,
    )

    controller.apply_warm_directive(directive(clock, "release"))

    assert daemon.closed is True
    assert controller.residency == "standby"


def test_release_waits_for_session_and_in_flight_guards(tmp_path):
    clock = FakeClock()
    entered = threading.Event()
    finish = threading.Event()
    daemon = FakeDaemon(entered=entered, release=finish)
    controller = make_controller(
        tmp_path, clock, lambda: daemon, idle=1_000, grace=-1,
    )

    assert controller.acquire_session("active") is True
    controller.apply_warm_directive(directive(clock, "release"))
    assert daemon.closed is False

    call = threading.Thread(
        target=lambda: controller.call(worker.CMD_TOKEN_DECODE, 1, b"data")
    )
    call.start()
    assert entered.wait(timeout=2)

    controller.release_session("active")
    assert daemon.closed is False
    finish.set()
    call.join(timeout=2)
    assert not call.is_alive()
    assert daemon.closed is True
    assert controller.residency == "standby"


def test_release_reclaims_stale_abandoned_session(tmp_path):
    clock = FakeClock()
    daemon = FakeDaemon()
    controller = make_controller(
        tmp_path, clock, lambda: daemon, idle=1_000, grace=-1,
    )

    assert controller.acquire_session("abandoned") is True
    clock.advance(worker.KV_SESSION_STALE_AFTER_S + 1)
    controller.apply_warm_directive(directive(clock, "release"))

    assert daemon.closed is True
    assert controller.current_owner is None
    assert controller.residency == "standby"


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def test_invalid_directive_from_pillar_is_ignored(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(worker, "_SANDBOX_AVAILABLE", False)
    monkeypatch.setattr(
        worker._wauth,
        "open_pillar_url",
        lambda *_args, **_kwargs: FakeResponse({
            "warm": {"model": "other", "action": "prewarm", "until": 1_700_000_100},
        }),
    )

    assert worker.register_with_pillar(
        "https://pillar.test", {}, model_id="model-a",
        warm_directive_handler=seen.append, now=lambda: 1_700_000_000,
    ) is True
    assert seen == []
    assert capsys.readouterr().out.count("ignoring invalid warm directive") == 1


def test_far_future_directive_is_clamped_and_absence_clears():
    now = lambda: 1_700_000_000.0
    parsed = worker.parse_warm_directive({
        "warm": {"model": "model-a", "action": "keep", "until": now() + 999},
    }, "model-a", now=now)
    assert parsed == worker.WarmDirective("model-a", "keep", now() + 300)
    assert worker.parse_warm_directive({}, "model-a", now=now) is None


def test_heartbeat_passes_warm_response_handler_to_registration(monkeypatch):
    stop = threading.Event()
    wake = threading.Event()
    wake.set()
    seen = []

    def fake_register(_url, _payload, **kwargs):
        assert kwargs["model_id"] == "model-a"
        kwargs["warm_directive_handler"](
            worker.WarmDirective("model-a", "keep", 1_700_000_100)
        )
        stop.set()
        return True

    monkeypatch.setattr(worker, "register_with_pillar", fake_register)
    worker.heartbeat_loop(
        "https://pillar.test", {}, stop_event=stop,
        residency_event=wake, warm_directive_handler=seen.append,
        model_id="model-a",
    )
    assert seen == [worker.WarmDirective("model-a", "keep", 1_700_000_100)]


def test_opt_out_ignores_directives(tmp_path):
    clock = FakeClock()
    daemon = FakeDaemon()
    controller = make_controller(
        tmp_path, clock, lambda: daemon, enabled=False,
    )
    controller.apply_warm_directive(directive(clock, "keep"))
    clock.advance(11)
    assert controller.warm_directive is None
    assert controller.unload_if_idle() is True


def test_owner_lease_lapse_overrides_keep(tmp_path):
    clock = FakeClock()
    daemons = []

    def factory():
        daemons.append(FakeDaemon())
        return daemons[-1]

    controller = make_controller(tmp_path, clock, factory, lease=True)
    controller.update_owner_lease("live", "owner clear", resume=True)
    controller.ensure_loaded()
    loaded = daemons[-1]
    controller.apply_warm_directive(directive(clock, "keep"))

    controller.update_owner_lease("lapsed", "owner lease expired")
    assert loaded.closed is True
    assert controller.residency == "away"


def test_prewarm_and_concurrent_request_spawn_exactly_once(tmp_path):
    clock = FakeClock()
    spawn_entered = threading.Event()
    finish_spawn = threading.Event()
    daemons = []

    def factory():
        if daemons:
            spawn_entered.set()
            assert finish_spawn.wait(timeout=2)
        daemons.append(FakeDaemon())
        return daemons[-1]

    controller = make_controller(tmp_path, clock, factory)
    clock.advance(11)
    assert controller.unload_if_idle() is True

    prewarm = threading.Thread(
        target=lambda: controller.apply_warm_directive(directive(clock, "prewarm"))
    )
    prewarm.start()
    assert spawn_entered.wait(timeout=2)
    request = threading.Thread(
        target=lambda: controller.call(worker.CMD_TOKEN_DECODE, 1, b"data")
    )
    request.start()
    time.sleep(0.03)
    finish_spawn.set()
    prewarm.join(timeout=2)
    request.join(timeout=2)

    assert not prewarm.is_alive() and not request.is_alive()
    assert len(daemons) == 2
    assert controller.residency == "loaded"


def test_repeated_directive_logs_only_on_change(tmp_path, capsys):
    clock = FakeClock()
    controller = make_controller(tmp_path, clock, FakeDaemon)
    warm = directive(clock, "keep")
    controller.apply_warm_directive(warm)
    controller.apply_warm_directive(warm)
    assert capsys.readouterr().out.count("warm directive:") == 1
