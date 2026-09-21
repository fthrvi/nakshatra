"""client.info_with_startup_wait: a worker that says "starting" is retried; every other failure is immediate."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
grpc = pytest.importorskip("grpc")
import client  # noqa: E402


class _Err(grpc.RpcError):
    def __init__(self, code, details):
        self._c, self._d = code, details

    def code(self):
        return self._c

    def details(self):
        return self._d


class _Stub:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def Info(self, request, timeout=None):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def _clock():
    t = [0.0]
    return (lambda s: t.__setitem__(0, t[0] + s)), (lambda: t[0])


def _starting():
    return _Err(grpc.StatusCode.UNAVAILABLE, "worker starting: registration with the pillar has not succeeded (retry)")


def test_a_starting_worker_is_retried_until_it_answers():
    sleep, now = _clock()
    stub = _Stub([_starting(), _starting(), "INFO"])
    assert client.info_with_startup_wait(stub, object(), wait_s=30, sleep=sleep, now=now) == "INFO"
    assert stub.calls == 3


def test_it_gives_up_after_the_wait_budget():
    sleep, now = _clock()
    stub = _Stub([_starting()] * 100)
    with pytest.raises(grpc.RpcError):
        client.info_with_startup_wait(stub, object(), wait_s=5, sleep=sleep, now=now)
    assert 5 <= stub.calls <= 7


@pytest.mark.parametrize("err", [
    _Err(grpc.StatusCode.UNAVAILABLE, "failed to connect to all addresses"),   # not up at all: not a startup transient
    _Err(grpc.StatusCode.UNAUTHENTICATED, "unknown keyid"),
    _Err(grpc.StatusCode.INTERNAL, "worker starting"),                          # wrong code: never retried
])
def test_other_failures_raise_immediately(err):
    sleep, now = _clock()
    stub = _Stub([err, "INFO"])
    with pytest.raises(grpc.RpcError):
        client.info_with_startup_wait(stub, object(), wait_s=30, sleep=sleep, now=now)
    assert stub.calls == 1
