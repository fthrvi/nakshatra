"""Standby-hop warm-up tolerance; pure fakes, no sockets, nodes, or network."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
grpc = pytest.importorskip("grpc")
import client  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402


class _Clock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds

    def advance(self, seconds):
        self.value += seconds


class _RpcError(grpc.RpcError):
    def __init__(self, code, details="fake failure"):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class _ForwardStub:
    def __init__(self, outcomes, clock=None, elapsed_s=0.0):
        self.outcomes = list(outcomes)
        self.clock = clock
        self.elapsed_s = elapsed_s
        self.timeouts = []

    def Forward(self, _request, timeout=None, metadata=None):
        del metadata
        self.timeouts.append(timeout)
        if self.clock is not None:
            self.clock.advance(self.elapsed_s)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return pb.ForwardResponse(hidden_out=outcome)


def _call(stub, policy):
    return client.call_forward(
        stub, b"input", 1, True, worker_id=policy.worker_id,
        rpc_policy=policy,
    )


def test_standby_gets_extended_deadline_on_first_rpc_only():
    stub = _ForwardStub([b"first", b"later"])
    policy = client.HopRpcPolicy("hop-a", "standby", warmup_timeout_s=60,
                                 log=lambda _line: None)

    assert _call(stub, policy) == b"first"
    assert _call(stub, policy) == b"later"

    assert stub.timeouts == [client.STEP_TIMEOUT_S + 60, client.STEP_TIMEOUT_S]


def test_default_warmup_allowance_comes_from_environment(monkeypatch):
    monkeypatch.delenv("NAKSHATRA_WARMUP_TIMEOUT_S", raising=False)
    policy = client.HopRpcPolicy("hop-a", "standby", log=lambda _line: None)
    assert policy.warmup_timeout_s == 60.0


@pytest.mark.parametrize("residency", ["loaded", "unknown", None])
def test_loaded_or_unknown_keeps_normal_deadline(residency):
    stub = _ForwardStub([b"ok"])
    policy = client.HopRpcPolicy("hop-a", residency, warmup_timeout_s=60,
                                 log=lambda _line: None)

    assert _call(stub, policy) == b"ok"
    assert stub.timeouts == [client.STEP_TIMEOUT_S]


def test_standby_unavailable_retries_once_then_recovery_receives_failure():
    unavailable = _RpcError(grpc.StatusCode.UNAVAILABLE)
    stub = _ForwardStub([unavailable, unavailable])
    clock = _Clock()
    policy = client.HopRpcPolicy(
        "hop-a", "standby", warmup_timeout_s=60,
        now=clock.now, sleep=clock.sleep, log=lambda _line: None,
    )

    with pytest.raises(client.EdgeError) as error:
        _call(stub, policy)

    assert error.value.kind is client.EdgeFailureKind.UNAVAILABLE
    assert stub.timeouts == [client.STEP_TIMEOUT_S + 60] * 2
    assert clock.sleeps == [client.WARMUP_UNAVAILABLE_BACKOFF_S]


def test_warmup_log_is_one_line_and_contains_measured_time():
    clock = _Clock()
    lines = []
    stub = _ForwardStub([b"ok"], clock=clock, elapsed_s=2.5)
    policy = client.HopRpcPolicy(
        "hop-cold", "standby", warmup_timeout_s=60,
        now=clock.now, sleep=clock.sleep, log=lines.append,
    )

    assert _call(stub, policy) == b"ok"
    assert lines == ["[chain] warming hop-cold (standby) … 2.50s"]


def test_streaming_gateway_path_retries_and_restores_normal_step_deadline():
    """The gateway uses --use-streaming, so cover that transport explicitly."""
    unavailable = _RpcError(grpc.StatusCode.UNAVAILABLE)

    class _StreamingStub:
        def __init__(self):
            self.calls = 0

        def Inference(self, requests, metadata=None):
            del metadata
            self.calls += 1
            if self.calls == 1:
                def fail():
                    next(requests)
                    raise unavailable
                    yield  # pragma: no cover - makes this a generator
                return fail()

            def answer_two_steps():
                for _ in range(2):
                    request = next(requests)
                    yield pb.InferenceStep(
                        session_id=request.session_id, step_id=request.step_id,
                    )
            return answer_two_steps()

    class _RecordingStream(client.InferenceStream):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.deadlines = []

        def _wait_response(self, timeout_s):
            self.deadlines.append(timeout_s)
            return super()._wait_response(timeout_s)

    clock = _Clock()
    stub = _StreamingStub()
    policy = client.HopRpcPolicy(
        "hop-stream", "standby", warmup_timeout_s=60,
        now=clock.now, sleep=clock.sleep, log=lambda _line: None,
    )
    stream = _RecordingStream(stub, "hop-stream", policy)
    try:
        stream.step(pb.InferenceStep(session_id="s", step_id="prefill"))
        stream.step(pb.InferenceStep(session_id="s", step_id="decode-1"))
    finally:
        stream.close()

    assert stub.calls == 2
    assert stream.deadlines == [client.STEP_TIMEOUT_S + 60] * 2 + [client.STEP_TIMEOUT_S]
    assert clock.sleeps == [client.WARMUP_UNAVAILABLE_BACKOFF_S]


def test_pillar_chain_residency_is_projected_and_missing_is_unknown(monkeypatch):
    bodies = iter([
        b'{"chain": [{"node_id":"cold","address":"host:1","layer_start":0,"layer_end":1,"residency":"standby"}]}',
        b'{"chain": [{"node_id":"old","address":"host:1","layer_start":0,"layer_end":1}]}',
    ])

    class _Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self.body

    monkeypatch.setattr(client, "_registry_urlopen",
                        lambda *_args, **_kwargs: _Response(next(bodies)))

    assert client._try_pillar_chain("https://pillar.invalid", "m")[0]["residency"] == "standby"
    assert client._try_pillar_chain("https://pillar.invalid", "m")[0]["residency"] == "unknown"
