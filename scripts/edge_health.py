"""
edge_health.py — fail-fast edge error context + per-edge health for the distributed chain.

Speed-stack finding #17 (trisul research 2026-06-19). Shard's property: a chain edge that
fails does so with CONTEXT — which peer, which step, dropped vs timed-out — never an opaque
"broken pipe". Today nakshatra classifies gRPC failures inconsistently (sys.exit vs
RuntimeError vs PushFailure) with no failure-kind. This module is the pure core for the
**gRPC chain plane**:

  • classify(exc) → (EdgeFailureKind, reason)  — turn a gRPC/socket error into a typed kind.
  • EdgeError(RuntimeError)                     — a structured edge failure carrying peer/step/
                                                  phase/kind; subclasses RuntimeError so the
                                                  existing recovery loop catches it when wired.
  • EdgeHealth                                  — per-edge ring of recent latencies + failure
                                                  counters; snapshot() → p50/p99/fail counts;
                                                  rtt_matrix() → directed p50 the topology
                                                  optimiser (#11) consumes directly.

PLANE BOUNDARY: this is the gRPC chain plane. Its latency source is client.py's `timing`
dict (worker_id → per-call seconds). It is SEPARATE from the fabric/UDP plane's
link_stats.py (whose rtt_ns_p50/p99 slots are a different transport and emit zero today) —
do NOT fold gRPC RTT into FabricLinkSnapshot; that conflates two planes.

Pure: no gRPC/socket connection is opened here; classify() duck-types on `.code()`/`.details()`
so it works on a real grpc.RpcError or a fake. Wiring (logging at the failure sites; and the
real robustness win — converting call_inference_step's sys.exit at client.py:220 into a raise
so the existing recovery loop can swap workers mid-stream) is a separate reviewer-gated seam.
"""
from __future__ import annotations

import enum
import socket
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple


class EdgeFailureKind(enum.Enum):
    TIMEOUT = "timeout"          # peer too slow / hung — deadline exceeded
    UNAVAILABLE = "unavailable"  # never came up / connection refused
    DROPPED = "dropped"          # died mid-stream — reset / broken pipe / EOF
    CANCELLED = "cancelled"      # the call was cancelled
    AUTH = "auth"                # unauthenticated / permission denied
    RESOURCE = "resource"        # resource exhausted (OOM / quota)
    INTERNAL = "internal"        # peer-side internal error
    PROTOCOL = "protocol"        # INVALID_ARGUMENT — a deterministic client/protocol bug
    OTHER = "other"

    @property
    def is_transport_fault(self) -> bool:
        """True for faults a worker-swap / retry can actually fix. PROTOCOL/AUTH/INTERNAL/
        RESOURCE are deterministic or peer-side — swapping alternates on them just churns,
        so the recovery loop should NOT treat them as edge failures."""
        return self in (EdgeFailureKind.TIMEOUT, EdgeFailureKind.UNAVAILABLE,
                        EdgeFailureKind.DROPPED, EdgeFailureKind.CANCELLED)


# gRPC StatusCode.name → kind. UNAVAILABLE collapses refused/dropped at the gRPC layer (it
# can't distinguish "never up" from "died"); socket-level errors below DO distinguish them.
_GRPC_KIND = {
    "DEADLINE_EXCEEDED": EdgeFailureKind.TIMEOUT,
    "UNAVAILABLE": EdgeFailureKind.UNAVAILABLE,
    "CANCELLED": EdgeFailureKind.CANCELLED,
    "UNAUTHENTICATED": EdgeFailureKind.AUTH,
    "PERMISSION_DENIED": EdgeFailureKind.AUTH,
    "RESOURCE_EXHAUSTED": EdgeFailureKind.RESOURCE,
    "INTERNAL": EdgeFailureKind.INTERNAL,
    "ABORTED": EdgeFailureKind.INTERNAL,
    "DATA_LOSS": EdgeFailureKind.DROPPED,
    # INVALID_ARGUMENT / FAILED_PRECONDITION are deterministic protocol bugs, NOT edge
    # transport faults — classify them so recovery won't pointlessly churn alternates.
    "INVALID_ARGUMENT": EdgeFailureKind.PROTOCOL,
    "FAILED_PRECONDITION": EdgeFailureKind.PROTOCOL,
}


def classify(exc: BaseException) -> Tuple[EdgeFailureKind, str]:
    """Map a gRPC RpcError or a socket/OS error to a typed kind + a human reason.

    Duck-types on `.code()`/`.details()` so it accepts a real grpc.RpcError without importing
    grpc here (and a fake in tests). Falls through to socket/OS error inspection, then OTHER.
    """
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            sc = exc.code()
            name = getattr(sc, "name", str(sc))
        except Exception:
            name = "UNKNOWN"
        details_fn = getattr(exc, "details", None)
        details = ""
        if callable(details_fn):
            try:
                details = details_fn() or ""
            except Exception:
                details = ""
        return _GRPC_KIND.get(name, EdgeFailureKind.OTHER), f"grpc {name}: {details}".strip()

    if isinstance(exc, socket.timeout):
        return EdgeFailureKind.TIMEOUT, "socket timeout"
    if isinstance(exc, ConnectionRefusedError):
        return EdgeFailureKind.UNAVAILABLE, "connection refused"
    if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, EOFError)):
        return EdgeFailureKind.DROPPED, f"connection dropped: {type(exc).__name__}"
    if isinstance(exc, TimeoutError):
        return EdgeFailureKind.TIMEOUT, "timeout"
    if isinstance(exc, OSError):
        return EdgeFailureKind.UNAVAILABLE, f"os error: {exc}"
    return EdgeFailureKind.OTHER, repr(exc)


@dataclass
class EdgeError(RuntimeError):
    """A structured edge failure. Subclasses RuntimeError so the existing client.py recovery
    loop (`except (grpc.RpcError, RuntimeError, ...)`) catches it once wired — turning a
    process-killing sys.exit into a recoverable worker-swap."""
    peer_id: str
    kind: EdgeFailureKind
    reason: str
    step: Optional[int] = None
    phase: str = "forward"            # forward | return | connect
    underlying: Optional[BaseException] = None

    def __str__(self) -> str:
        step = "?" if self.step is None else self.step
        return (f"edge[{self.peer_id}] {self.phase} step={step} "
                f"{self.kind.value}: {self.reason}")

    @classmethod
    def from_exc(cls, exc: BaseException, *, peer_id: str, step: Optional[int] = None,
                 phase: str = "forward", health: "Optional[EdgeHealth]" = None,
                 prev_peer: Optional[str] = None) -> "EdgeError":
        kind, reason = classify(exc)
        # gRPC can't tell "never came up" from "died mid-stream" (both UNAVAILABLE). If we've
        # seen this edge succeed before, an UNAVAILABLE now means it DROPPED. The inbound edge
        # is prev_peer→peer_id (defaults to the entry edge if not given).
        if health is not None:
            kind = health.refine_kind(kind, prev_peer if prev_peer is not None else peer_id,
                                      peer_id)
            reason = f"{reason} [{kind.value}]" if kind.value not in reason else reason
        return cls(peer_id=peer_id, kind=kind, reason=reason, step=step, phase=phase,
                   underlying=exc)

    @property
    def is_transport_fault(self) -> bool:
        return self.kind.is_transport_fault


def _percentile(sorted_vals, q: float) -> float:
    """Nearest-rank percentile of an already-sorted list (q in [0,1])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return float(sorted_vals[idx])


@dataclass
class EdgeHealth:
    """Per-edge (a→b) rolling latency + failure health. The real RTT source finding #11 wants.

    Latencies are one-way send→reply ms for that directed edge. snapshot() yields p50/p99 +
    ok/fail counts; rtt_matrix() yields the directed p50 map order_chain() consumes directly.
    """
    window: int = 64
    _lat: Dict[Tuple[str, str], Deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque()))
    _fail: Dict[Tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    _ok: Counter = field(default_factory=Counter)

    def record_latency(self, a: str, b: str, ms: float) -> None:
        dq = self._lat[(a, b)]
        dq.append(float(ms))
        while len(dq) > self.window:
            dq.popleft()
        self._ok[(a, b)] += 1

    def record_failure(self, a: str, b: str, kind: EdgeFailureKind) -> None:
        self._fail[(a, b)][kind] += 1

    def refine_kind(self, kind: EdgeFailureKind, a: str, b: str) -> EdgeFailureKind:
        """Disambiguate UNAVAILABLE using edge history: a prior success on a→b means an
        UNAVAILABLE now is a mid-stream DROP, not a never-came-up refusal. gRPC's StatusCode
        alone can't tell these apart."""
        if kind is EdgeFailureKind.UNAVAILABLE and self._ok.get((a, b), 0) > 0:
            return EdgeFailureKind.DROPPED
        return kind

    def snapshot(self, a: str, b: str) -> Dict[str, object]:
        vals = sorted(self._lat.get((a, b), ()))
        fails = self._fail.get((a, b), Counter())
        return {
            "edge": (a, b),
            "n": len(vals),
            "p50_ms": _percentile(vals, 0.50),
            "p99_ms": _percentile(vals, 0.99),
            "ok": self._ok.get((a, b), 0),
            "fail": int(sum(fails.values())),
            "fail_by_kind": {k.value: v for k, v in fails.items()},
        }

    def rtt_matrix(self) -> Dict[Tuple[str, str], float]:
        """Directed p50 latency per measured edge — feed straight into topology_order.order_chain
        (it's already one-way, so pass it WITHOUT rtt_to_oneway)."""
        return {edge: _percentile(sorted(vals), 0.50)
                for edge, vals in self._lat.items() if vals}

    def edges(self):
        return list(self._lat.keys())
