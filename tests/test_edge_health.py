"""
Unit tests for edge supervision (scripts/edge_health.py). Pure — no gRPC connection opened.

Covers: failure classification across gRPC StatusCodes + socket errors; the transport-fault
vs deterministic-bug distinction (so recovery doesn't churn on protocol errors); the
history-based UNAVAILABLE→DROPPED disambiguation; EdgeError context formatting; and the
EdgeHealth latency/percentile/rtt_matrix math that feeds finding #11.
"""
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from edge_health import (  # noqa: E402
    EdgeFailureKind, classify, EdgeError, EdgeHealth,
)


class _FakeCode:
    def __init__(self, name):
        self.name = name


class FakeRpcError(Exception):
    """Duck-typed stand-in for grpc.RpcError (has .code() and .details())."""
    def __init__(self, code_name, details=""):
        self._code = _FakeCode(code_name)
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


# ---------------------------------------------------------------- classify: gRPC

def test_classify_grpc_status_codes():
    cases = {
        "DEADLINE_EXCEEDED": EdgeFailureKind.TIMEOUT,
        "UNAVAILABLE": EdgeFailureKind.UNAVAILABLE,
        "CANCELLED": EdgeFailureKind.CANCELLED,
        "UNAUTHENTICATED": EdgeFailureKind.AUTH,
        "PERMISSION_DENIED": EdgeFailureKind.AUTH,
        "RESOURCE_EXHAUSTED": EdgeFailureKind.RESOURCE,
        "INTERNAL": EdgeFailureKind.INTERNAL,
        "INVALID_ARGUMENT": EdgeFailureKind.PROTOCOL,
        "FAILED_PRECONDITION": EdgeFailureKind.PROTOCOL,
        "SOMETHING_NEW": EdgeFailureKind.OTHER,
    }
    for name, expect in cases.items():
        kind, reason = classify(FakeRpcError(name, "boom"))
        assert kind is expect, (name, kind)
        assert name in reason


def test_invalid_argument_is_not_a_transport_fault():
    # the key safety property: a protocol bug must NOT trigger worker-swap churn
    kind, _ = classify(FakeRpcError("INVALID_ARGUMENT"))
    assert kind is EdgeFailureKind.PROTOCOL
    assert not kind.is_transport_fault
    # while a real edge drop SHOULD be swap-worthy
    assert EdgeFailureKind.UNAVAILABLE.is_transport_fault
    assert EdgeFailureKind.TIMEOUT.is_transport_fault
    assert EdgeFailureKind.DROPPED.is_transport_fault
    assert not EdgeFailureKind.AUTH.is_transport_fault
    assert not EdgeFailureKind.INTERNAL.is_transport_fault


# ---------------------------------------------------------------- classify: socket

def test_classify_socket_errors():
    assert classify(socket.timeout())[0] is EdgeFailureKind.TIMEOUT
    assert classify(ConnectionRefusedError())[0] is EdgeFailureKind.UNAVAILABLE
    assert classify(BrokenPipeError())[0] is EdgeFailureKind.DROPPED
    assert classify(ConnectionResetError())[0] is EdgeFailureKind.DROPPED
    assert classify(EOFError())[0] is EdgeFailureKind.DROPPED
    assert classify(OSError("nope"))[0] is EdgeFailureKind.UNAVAILABLE
    assert classify(ValueError("weird"))[0] is EdgeFailureKind.OTHER


# ---------------------------------------------------------------- EdgeError

def test_edge_error_is_runtimeerror_and_formats_context():
    e = EdgeError.from_exc(FakeRpcError("DEADLINE_EXCEEDED", "slow"),
                           peer_id="w2", step=7, phase="forward")
    assert isinstance(e, RuntimeError)          # so client.py recovery catches it when wired
    assert e.kind is EdgeFailureKind.TIMEOUT
    s = str(e)
    assert "w2" in s and "step=7" in s and "timeout" in s and "forward" in s
    assert e.is_transport_fault


def test_edge_error_history_disambiguates_unavailable_to_dropped():
    h = EdgeHealth()
    # never-succeeded edge: UNAVAILABLE stays UNAVAILABLE (never came up / refused)
    e1 = EdgeError.from_exc(FakeRpcError("UNAVAILABLE"), peer_id="w1",
                            prev_peer="coord", health=h)
    assert e1.kind is EdgeFailureKind.UNAVAILABLE
    # after a success on coord→w1, a later UNAVAILABLE means it DROPPED mid-stream
    h.record_latency("coord", "w1", 12.0)
    e2 = EdgeError.from_exc(FakeRpcError("UNAVAILABLE"), peer_id="w1",
                            prev_peer="coord", health=h)
    assert e2.kind is EdgeFailureKind.DROPPED
    assert e2.is_transport_fault


# ---------------------------------------------------------------- EdgeHealth math

def test_health_latency_percentiles_and_counts():
    h = EdgeHealth(window=100)
    for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        h.record_latency("a", "b", ms)
    snap = h.snapshot("a", "b")
    assert snap["n"] == 10
    assert snap["ok"] == 10
    assert snap["fail"] == 0
    assert 40 <= snap["p50_ms"] <= 60          # median-ish
    assert snap["p99_ms"] >= 90                 # tail
    assert snap["edge"] == ("a", "b")


def test_health_window_evicts_old_samples():
    h = EdgeHealth(window=3)
    for ms in [1, 2, 3, 4, 5]:
        h.record_latency("a", "b", ms)
    snap = h.snapshot("a", "b")
    assert snap["n"] == 3                        # only last 3 kept
    assert snap["p50_ms"] == 4                   # median of [3,4,5]


def test_health_failure_counts_by_kind():
    h = EdgeHealth()
    h.record_failure("a", "b", EdgeFailureKind.TIMEOUT)
    h.record_failure("a", "b", EdgeFailureKind.TIMEOUT)
    h.record_failure("a", "b", EdgeFailureKind.DROPPED)
    snap = h.snapshot("a", "b")
    assert snap["fail"] == 3
    assert snap["fail_by_kind"] == {"timeout": 2, "dropped": 1}


def test_health_empty_edge_snapshot():
    h = EdgeHealth()
    snap = h.snapshot("x", "y")
    assert snap["n"] == 0 and snap["p50_ms"] == 0.0 and snap["fail"] == 0


def test_rtt_matrix_feeds_topology_order():
    # rtt_matrix() is directed one-way p50 — exactly order_chain()'s input shape.
    h = EdgeHealth()
    for ms in [10, 12, 14]:
        h.record_latency("coord", "w0", ms)
    for ms in [40, 44, 48]:
        h.record_latency("w0", "w1", ms)
    m = h.rtt_matrix()
    assert m[("coord", "w0")] == 12              # p50
    assert m[("w0", "w1")] == 44
    # only measured edges appear (missing edges absent → order_chain falls back)
    assert ("w1", "coord") not in m


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
