"""Idle daemon residency tests.  All process/GPU/clock effects are fakes."""
from __future__ import annotations

import collections
import hashlib
import struct
import threading
import time

import grpc
import pytest

import nakshatra_pb2 as pb
import worker


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeDaemon:
    def __init__(self, *, entered=None, release=None):
        self.closed = False
        self.calls = []
        self.recent_rpc_ms = collections.deque(maxlen=20)
        self.entered = entered
        self.release = release

    def info(self):
        return {
            "layer_start": 0, "layer_end": 1, "n_embd": 4,
            "has_token_embd": True, "has_lm_head": True, "n_vocab": 8,
        }

    def gpu_offload_status(self):
        return {
            "n_offloaded": 1, "total_layers": 1, "uses_gpu": True,
            "backend_hints": ["cuda"], "log_lines": [],
        }

    def call(self, cmd, n_tokens, payload, start_pos=0, flags=0):
        self.calls.append((cmd, n_tokens, payload, start_pos, flags))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=2)
        return 0, struct.pack("<Ii", 0, 7)

    def close(self):
        self.closed = True


def _controller(tmp_path, clock, factory, idle=10.0):
    model = tmp_path / "slice.gguf"
    model.write_bytes(b"verified model slice")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    return worker.IdleDaemonController(
        factory, str(model), digest, idle, now=clock,
    ), model


def test_unloads_after_idle_with_fake_clock_and_daemon(tmp_path):
    clock = FakeClock()
    daemons = []

    def factory():
        daemons.append(FakeDaemon())
        return daemons[-1]

    controller, _ = _controller(tmp_path, clock, factory)
    clock.advance(9.9)
    assert controller.unload_if_idle() is False
    clock.advance(0.2)
    assert controller.unload_if_idle() is True
    assert daemons[0].closed is True
    assert controller.residency == "standby"


def test_does_not_unload_with_open_session_or_in_flight_step(tmp_path):
    clock = FakeClock()
    entered = threading.Event()
    release = threading.Event()
    daemon = FakeDaemon(entered=entered, release=release)
    controller, _ = _controller(tmp_path, clock, lambda: daemon)

    assert controller.acquire_session("open-session") is True
    clock.advance(20)
    assert controller.unload_if_idle() is False
    controller.release_session("open-session")

    call_thread = threading.Thread(
        target=lambda: controller.call(worker.CMD_TOKEN_DECODE, 1, b"data")
    )
    call_thread.start()
    assert entered.wait(timeout=2)
    assert controller.in_flight == 1
    assert controller.unload_if_idle() is False
    release.set()
    call_thread.join(timeout=2)
    assert not call_thread.is_alive()
    assert controller.in_flight == 0


def test_first_concurrent_requests_perform_exactly_one_reload(tmp_path):
    clock = FakeClock()
    daemons = []

    def factory():
        # Widen the reload race without involving any real process.
        if daemons:
            time.sleep(0.03)
        daemons.append(FakeDaemon())
        return daemons[-1]

    controller, _ = _controller(tmp_path, clock, factory)
    clock.advance(11)
    assert controller.unload_if_idle() is True

    barrier = threading.Barrier(8)
    errors = []

    def request():
        try:
            barrier.wait(timeout=2)
            controller.call(worker.CMD_TOKEN_DECODE, 1, b"data")
        except Exception as e:  # pragma: no cover - assertion reports details
            errors.append(e)

    threads = [threading.Thread(target=request) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert errors == []
    assert len(daemons) == 2  # one startup + one single-flight reload
    assert controller.residency == "loaded"
    assert len(daemons[1].calls) == 8


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = ""

    def invocation_metadata(self):
        return []

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def test_digest_mismatch_refuses_reload_as_unavailable(tmp_path):
    clock = FakeClock()
    daemons = []

    def factory():
        daemons.append(FakeDaemon())
        return daemons[-1]

    controller, model = _controller(tmp_path, clock, factory)
    servicer = worker.WorkerServicer(
        controller, "first", 0, 1, "model", auth_required=False,
        refuse_unregistered_peers=False, refuse_unpinned_peers=False,
    )
    clock.advance(11)
    assert controller.unload_if_idle() is True
    model.write_bytes(b"tampered after startup")

    context = FakeContext()
    request = pb.ForwardRequest(
        hidden_in=struct.pack("<i", 1), n_tokens=1, has_token_ids=True,
    )
    response = servicer.Forward(request, context)

    assert response.hidden_out == b""
    assert context.code == grpc.StatusCode.UNAVAILABLE
    assert "SHA-256 mismatch" in context.details
    assert controller.residency == "standby"
    assert len(daemons) == 1


def test_spawn_failure_is_unavailable_and_stays_in_standby(tmp_path):
    clock = FakeClock()
    attempts = 0

    def factory():
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            raise RuntimeError("fake GPU load failed")
        return FakeDaemon()

    controller, _ = _controller(tmp_path, clock, factory)
    servicer = worker.WorkerServicer(
        controller, "first", 0, 1, "model", auth_required=False,
        refuse_unregistered_peers=False, refuse_unpinned_peers=False,
    )
    clock.advance(11)
    assert controller.unload_if_idle() is True

    context = FakeContext()
    servicer.Forward(pb.ForwardRequest(
        hidden_in=struct.pack("<i", 1), n_tokens=1, has_token_ids=True,
    ), context)

    assert context.code == grpc.StatusCode.UNAVAILABLE
    assert "fake GPU load failed" in context.details
    assert controller.residency == "standby"
    assert attempts == 2


def test_residency_transition_triggers_immediate_heartbeat(tmp_path, monkeypatch):
    clock = FakeClock()
    transition = threading.Event()
    daemons = []

    def factory():
        daemons.append(FakeDaemon())
        return daemons[-1]

    model = tmp_path / "slice.gguf"
    model.write_bytes(b"verified")
    controller = worker.IdleDaemonController(
        factory, str(model), hashlib.sha256(model.read_bytes()).hexdigest(),
        10, now=clock, residency_event=transition,
    )
    payload = {"node_id": "fake", "residency": controller.residency}
    seen = []
    heartbeat_seen = threading.Event()

    def fake_register(_url, sent, **_kwargs):
        seen.append(dict(sent))
        heartbeat_seen.set()
        return True

    monkeypatch.setattr(worker, "register_with_pillar", fake_register)
    stop = threading.Event()
    thread = threading.Thread(
        target=worker.heartbeat_loop,
        args=("https://pillar.invalid", payload, 3600, stop),
        kwargs={"daemon_for_timing": controller,
                "residency_event": transition},
    )
    thread.start()

    clock.advance(11)
    assert controller.unload_if_idle() is True
    assert heartbeat_seen.wait(timeout=2)
    assert seen[-1]["residency"] == "standby"

    heartbeat_seen.clear()
    controller.call(worker.CMD_TOKEN_DECODE, 1, b"data")
    assert heartbeat_seen.wait(timeout=2)
    assert seen[-1]["residency"] == "loaded"

    stop.set()
    transition.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_zero_disables_idle_unload(tmp_path):
    clock = FakeClock()
    daemon = FakeDaemon()
    controller, _ = _controller(tmp_path, clock, lambda: daemon, idle=0)
    clock.advance(10_000)
    assert controller.unload_if_idle() is False
    assert daemon.closed is False
    assert controller.residency == "loaded"
