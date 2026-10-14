"""Owner lease tests. No real daemon, GPU, systemd, or user state is touched."""
from __future__ import annotations

import collections
import hashlib
import json
import struct
import threading

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
        return {"layer_start": 0, "layer_end": 1, "n_embd": 4,
                "has_token_embd": True, "has_lm_head": True, "n_vocab": 8}

    def gpu_offload_status(self):
        return {}

    def call(self, cmd, n_tokens, payload, start_pos=0, flags=0):
        self.calls.append((cmd, n_tokens, payload, start_pos, flags))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait(timeout=2)
        if self.closed:
            raise RuntimeError("daemon closed by owner lease")
        return 0, struct.pack("<Ii", 0, 7)

    def close(self):
        self.closed = True
        if self.release is not None:
            self.release.set()


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


def make_lease(tmp_path, clock, *, ttl=6, resume=0, daemon_factory=None):
    model = tmp_path / "slice.gguf"
    model.write_bytes(b"verified model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    daemons = []

    def default_factory():
        daemons.append(FakeDaemon())
        return daemons[-1]

    factory = daemon_factory or default_factory
    controller = worker.IdleDaemonController(
        factory, str(model), digest, 600,
        release_grace_s=30, owner_lease_enabled=True, now=clock,
    )
    lease = worker.OwnerLease(
        worker.OwnerLeaseConfig(5571, "ab" * 32, ttl, resume),
        controller, now=clock,
    )
    servicer = worker.WorkerServicer(
        controller, "first", 0, 1, "model", auth_required=False,
        refuse_unregistered_peers=False, refuse_unpinned_peers=False,
    )
    return controller, lease, servicer, daemons


def forward(servicer):
    context = FakeContext()
    response = servicer.Forward(pb.ForwardRequest(
        hidden_in=struct.pack("<i", 1), n_tokens=1, has_token_ids=True,
    ), context)
    return response, context


def test_lease_is_default_deny_and_away_at_start(tmp_path):
    clock = FakeClock()
    controller, lease, servicer, daemons = make_lease(tmp_path, clock)

    assert lease.reconcile()["state"] == "lapsed"
    assert controller.residency == "away"
    assert daemons[0].closed is True
    _response, context = forward(servicer)
    assert context.code == grpc.StatusCode.UNAVAILABLE
    assert "clear heartbeat" in context.details


def test_clear_heartbeat_becomes_live_then_standby(tmp_path):
    clock = FakeClock()
    controller, lease, _servicer, _daemons = make_lease(
        tmp_path, clock, resume=2,
    )

    assert lease.heartbeat(at=1_700_000_000, clear=True, reason="owner idle")["state"] == "live"
    assert controller.residency == "away"
    clock.advance(2)
    assert lease.heartbeat(at=1_700_000_002, clear=True, reason="owner idle")["resume_ready"] is True
    assert controller.residency == "standby"


def test_silence_past_ttl_unloads_goes_away_and_refuses_new_rpc(tmp_path):
    clock = FakeClock()
    controller, lease, servicer, daemons = make_lease(tmp_path, clock)
    lease.heartbeat(at=0, clear=True, reason="clear")
    response, context = forward(servicer)
    assert context.code is None and response.hidden_out
    assert controller.residency == "loaded"

    clock.advance(6.1)
    snap = lease.reconcile()
    assert snap["state"] == "lapsed"
    assert controller.residency == "away"
    assert daemons[-1].closed is True
    _response, context = forward(servicer)
    assert context.code == grpc.StatusCode.UNAVAILABLE
    assert "silence means stop" in context.details


def test_stale_lease_keeps_loaded_work_but_refuses_new_sessions(tmp_path):
    clock = FakeClock()
    controller, lease, servicer, daemons = make_lease(tmp_path, clock)
    lease.heartbeat(at=0, clear=True, reason="clear")
    _response, context = forward(servicer)
    assert context.code is None and controller.residency == "loaded"

    clock.advance(3.1)
    assert lease.reconcile()["state"] == "stale"
    assert controller.residency == "loaded" and daemons[-1].closed is False
    _response, context = forward(servicer)
    assert context.code == grpc.StatusCode.UNAVAILABLE
    assert "stale" in context.details


def test_clear_false_lapses_immediately(tmp_path):
    clock = FakeClock()
    controller, lease, _servicer, _daemons = make_lease(tmp_path, clock)
    lease.heartbeat(at=0, clear=True, reason="clear")
    controller.call(worker.CMD_TOKEN_DECODE, 1, b"data")

    snap = lease.heartbeat(at=1, clear=False, reason="owner launched a game")
    assert snap["state"] == "lapsed"
    assert controller.residency == "away"
    assert "launched a game" in controller.lease_reason


def test_bad_or_missing_bearer_is_401_and_endpoint_is_loopback(tmp_path):
    clock = FakeClock()
    controller, _lease, _servicer, _daemons = make_lease(tmp_path, clock)
    lease = worker.OwnerLease(
        worker.OwnerLeaseConfig(0, "cd" * 32, 6, 0), controller, now=clock,
    )
    class FakeServer:
        def __init__(self, address, handler):
            self.server_address = address
            self.handler = handler

        def serve_forever(self):
            return

    httpd, thread = worker.start_owner_lease_server(lease, server_class=FakeServer)
    thread.join(timeout=2)
    assert httpd.server_address == ("127.0.0.1", 0)
    body = json.dumps({"at": 1, "clear": True, "reason": "clear"}).encode()
    for authorization in ("", "Bearer " + "ef" * 32):
        status, payload = worker.owner_lease_http_response(
            lease, "/lease", authorization, body,
        )
        assert status == 401 and payload == {"error": "unauthorized"}
    status, payload = worker.owner_lease_http_response(
        lease, "/lease", "Bearer " + "cd" * 32, body,
    )
    assert status == 200 and payload["state"] == "live"


def test_lapse_cancels_in_flight_daemon_call(tmp_path):
    clock = FakeClock()
    entered = threading.Event()
    release = threading.Event()
    daemons = []

    def factory():
        daemon = FakeDaemon(entered=entered if daemons else None,
                            release=release if daemons else None)
        daemons.append(daemon)
        return daemon

    controller, lease, _servicer, _ = make_lease(
        tmp_path, clock, daemon_factory=factory,
    )
    lease.heartbeat(at=0, clear=True, reason="clear")
    errors = []

    def invoke():
        try:
            controller.call(worker.CMD_TOKEN_DECODE, 1, b"data")
        except Exception as e:  # assertion below reports the exact type/reason
            errors.append(e)

    call_thread = threading.Thread(target=invoke)
    call_thread.start()
    assert entered.wait(timeout=2)
    lease.heartbeat(at=1, clear=False, reason="owner active")
    call_thread.join(timeout=2)
    assert not call_thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], worker.DaemonUnavailableError)
    assert "owner active" in str(errors[0])
    assert controller.in_flight == 0 and controller.current_owner is None


def test_resume_requires_continuous_live_heartbeats(tmp_path):
    clock = FakeClock()
    controller, lease, _servicer, _daemons = make_lease(
        tmp_path, clock, ttl=6, resume=3,
    )
    lease.heartbeat(at=0, clear=True, reason="clear")
    clock.advance(7)  # a LAPSE (silence beyond TTL) breaks continuity before RESUME_S completes
    assert lease.reconcile()["state"] == "lapsed"
    assert controller.residency == "away"

    lease.heartbeat(at=4, clear=True, reason="clear again")
    clock.advance(2)
    assert lease.heartbeat(at=6, clear=True, reason="still clear")["resume_ready"] is False
    assert controller.residency == "away"
    clock.advance(1)
    assert lease.heartbeat(at=7, clear=True, reason="still clear")["resume_ready"] is True
    assert controller.residency == "standby"


def test_non_ascii_bearer_is_a_clean_401_not_a_crash():
    class _Cfg:
        token = "a" * 64

    class _Lease:
        config = _Cfg()

    status, payload = worker.owner_lease_http_response(
        _Lease(), "/lease", "Bearer ééé", b'{"at": 1, "clear": true}')
    assert status == 401 and payload == {"error": "unauthorized"}


def test_a_late_heartbeat_does_not_restart_the_resume_window():
    """Stale is jitter (a heartbeat a little late), not a lapse: on blackwell a Windows sentinel
    with a 1 s counter sample landed in the stale band every beat and never accrued resume time."""
    clock = {"t": 0.0}
    cfg = worker.OwnerLeaseConfig(port=5571, token="t" * 64, ttl_s=6.0, resume_s=30.0) \
        if "resume_s" in worker.OwnerLeaseConfig._fields else None
    if cfg is None:
        import pytest
        pytest.skip("OwnerLeaseConfig shape changed")

    class _Daemon:
        residency = "away"
        def update_owner_lease(self, *a, **k):
            pass

    lease = worker.OwnerLease(cfg, _Daemon(), now=lambda: clock["t"])
    lease.heartbeat(at=0, clear=True, reason="owner clear")
    for _ in range(9):              # every 3.5 s: always past TTL/2 (stale band), never past TTL
        clock["t"] += 3.5
        snap = lease.heartbeat(at=0, clear=True, reason="owner clear")
    assert clock["t"] >= 30 and snap["resume_ready"] is True
