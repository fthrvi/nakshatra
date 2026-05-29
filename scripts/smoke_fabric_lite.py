#!/usr/bin/env python3
"""Localhost smoke for the fabric_lite chain (sprint Phase F, part 2).

Composes 2 ``FabricBackend`` instances in-process — a "first" worker
and a "last" worker — wired together via real UDP loopback sockets in
the 2-worker chain shape. Drives a one-token round-trip through the
chain via the first worker's ``first_worker_round_trip`` and asserts
the last worker's stub-sampled token arrives back.

What this validates:
  - The 2-worker chain composition (first ↔ last over fabric)
  - first_worker_round_trip + serve loop integration end-to-end
  - The packet schema works across a real UDP loopback hop including
    AES-128-GCM auth + per-link seq monotonicity
  - The feedback wormhole returns to the first worker, not the chain
    head's gRPC reply (the OQ8 path)

What this doesn't validate (kept for the 2-machine cluster smoke):
  - setup_fabric (needs a live sthambha pillar for the /join handshake)
  - JoinClient against a real pillar
  - The worker.py main() boot path (daemon, gRPC server, TLS, auth)
  - Cross-network UDP (MTU, Tailscale, real latency)
  - Real daemon decode (we use stub forward_fns)

Run: ``source .venv/bin/activate && python scripts/smoke_fabric_lite.py``
Exit 0 on pass; non-zero on any check failure.
"""
from __future__ import annotations

import socket
import struct
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fabric import packet as fp
from fabric.backend import FabricBackend
from fabric.transport import FabricLink


# ── shape constants ────────────────────────────────────────────────


N_EMBD = 8                                   # tiny hidden size
ELEM_BYTES = N_EMBD * 4                      # f32 → 32 bytes per token
CHAIN_ID = 0xDEAD_BEEF_CAFE_F00D
KEY = b"smoke-fabric-key" * 1                # exactly 16 bytes


# ── stub forward_fns ───────────────────────────────────────────────


class _StubResult:
    """Stand-in for worker.ForwardResult — duck-typed so the backend
    never needs the real worker.py import."""
    def __init__(self, ok, payload, error="", client_error=False):
        self.ok = ok
        self.payload = payload
        self.error = error
        self.client_error = client_error


def first_worker_fn(payload, n_tokens, has_token_ids, keep_kv, start_pos):
    """First-worker stub: receives the client's prompt bytes via
    Forward, returns a hidden_state with a known fingerprint. In real
    operation this would be the layer-[0,k) decode; for the smoke
    it's just a deterministic byte transform."""
    # Mark every byte with 0x10 so the last worker can verify the
    # forward path actually carried our output.
    out = bytes((b ^ 0x10) for b in payload)
    return _StubResult(True, out)


def last_worker_fn(payload, n_tokens, has_token_ids, keep_kv, start_pos):
    """Last-worker stub: receives the mid-chain hidden state, returns
    a 4-byte int32 token id. The token's value encodes both
    n_tokens AND a fingerprint byte from the input so the smoke can
    verify the right activation arrived."""
    # Token id = 0xABCD0000 | first-input-byte. The fingerprint byte
    # should equal 0x10 (the XOR mark first_worker_fn applied to a
    # zero input).
    fingerprint = payload[0] if payload else 0
    token_id = 0xABCD0000 | fingerprint
    return _StubResult(True, struct.pack("<I", token_id))


# ── helpers ────────────────────────────────────────────────────────


def _udp_sock():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    return s


def _build_chain(first_sock, last_sock):
    """Wire two FabricBackends in the 2-worker chain shape:

        first ──FORWARD──> last
              <─FEEDBACK──

    Returns ``(first_backend, last_backend)``. The first backend has
    its forward + feedback both pointing at the last worker (2-worker
    shortcut from setup_fabric). The last backend has its inbound
    pointing at the first and its feedback link reusing inbound.
    """
    first_addr = first_sock.getsockname()
    last_addr = last_sock.getsockname()

    # First worker: forward + feedback are the same link (peer=last).
    first_to_last = FabricLink(first_sock, last_addr, KEY, CHAIN_ID)
    first = FabricBackend(first_worker_fn, "first", N_EMBD)
    first.set_links(forward=first_to_last, feedback=first_to_last)

    # Last worker: inbound + feedback are the same link (peer=first).
    last_to_first = FabricLink(last_sock, first_addr, KEY, CHAIN_ID)
    last = FabricBackend(last_worker_fn, "last", N_EMBD)
    last.set_links(inbound=last_to_first, feedback=last_to_first)

    return first, last


# ── checks ─────────────────────────────────────────────────────────


def check_round_trip_single_token(first, last) -> bool:
    """The headline check. First worker decodes a token's worth of
    prompt bytes; backend ships hidden via forward link; last worker
    decodes (stub: extracts fingerprint, samples token); ships
    FEEDBACK back; first worker's round_trip call returns the token
    bytes."""
    # Run last worker's serve loop in a daemon thread.
    serve_t = threading.Thread(
        target=last.serve, kwargs={"recv_timeout": 0.2}, daemon=True)
    serve_t.start()
    try:
        # First worker does its local decode (would normally happen
        # in WorkerServicer.Forward via _run_forward; we call the stub
        # directly here so the smoke proves the wire path in
        # isolation). The mid-chain payload going onto fabric is the
        # post-decode hidden state.
        result = first_worker_fn(b"\x00" * ELEM_BYTES, 1, False, False, 0)
        if not result.ok:
            print(f"  ✗ local first-worker decode failed: {result.error}")
            return False
        token_bytes = first.first_worker_round_trip(
            result.payload, step_id=0, layer_idx=14, timeout_s=2.0,
        )
        if token_bytes is None:
            print("  ✗ round-trip timed out without FEEDBACK")
            return False
        if len(token_bytes) != 4:
            print(f"  ✗ expected 4-byte token, got {len(token_bytes)}")
            return False
        token_id = struct.unpack("<I", token_bytes)[0]
        # Expected: 0xABCD0010 — high bits from last_worker_fn's
        # constant, low byte 0x10 from first_worker_fn's XOR mark.
        # If the round-trip is broken, either the high bits don't
        # match (last_worker_fn didn't run) or the low byte doesn't
        # match (the wrong bytes arrived at the last worker).
        if token_id != 0xABCD0010:
            print(f"  ✗ token round-trip corrupted: got {token_id:#010x}, "
                  f"expected 0xABCD0010")
            return False
        print(f"  ✓ token round-trip: {token_id:#010x} "
              f"(0xABCD high + 0x10 fingerprint preserved)")
        return True
    finally:
        last.stop()
        serve_t.join(timeout=2.0)


def check_counters_advanced(first, last) -> bool:
    """After a successful round-trip, both links' counters should show
    non-zero sent/recv on the schema §9 fields. Validates that the
    counter set Phase E ships is actually populated by real traffic
    (not just the unit-test fakes)."""
    f_link = first.forward_link
    l_link = last.inbound_link
    # First worker sent (FORWARD outbound) + recv'd (FEEDBACK inbound)
    # on the same link. Last worker recv'd (FORWARD) + sent (FEEDBACK).
    expectations = [
        ("first forward_link sent", f_link.counters["sent_packets"], 1),
        ("first forward_link recv", f_link.counters["recv_packets"], 1),
        ("last inbound_link sent", l_link.counters["sent_packets"], 1),
        ("last inbound_link recv", l_link.counters["recv_packets"], 1),
    ]
    ok = True
    for label, got, want in expectations:
        if got < want:
            print(f"  ✗ {label} = {got}; expected >= {want}")
            ok = False
    if ok:
        print(f"  ✓ counters populated: "
              f"sent/recv on both link sides ≥ 1")
    # Crucially: no auth fails, no alloc drops, no gaps.
    drops = (
        f_link.counters["recv_auth_fails"]
        + f_link.counters["recv_gaps"]
        + f_link.counters["recv_dropped_alloc"]
        + l_link.counters["recv_auth_fails"]
        + l_link.counters["recv_gaps"]
        + l_link.counters["recv_dropped_alloc"]
    )
    if drops > 0:
        print(f"  ✗ unexpected drops/gaps/auth-fails: {drops}")
        ok = False
    else:
        print(f"  ✓ no drops/gaps/auth-fails on either side")
    return ok


# ── main ───────────────────────────────────────────────────────────


def main() -> int:
    first_sock = _udp_sock()
    last_sock = _udp_sock()
    print(f"[smoke] first  @ 127.0.0.1:{first_sock.getsockname()[1]}")
    print(f"[smoke] last   @ 127.0.0.1:{last_sock.getsockname()[1]}")
    first, last = _build_chain(first_sock, last_sock)
    print(f"[smoke] chain wired: 2-worker fabric_lite, "
          f"chain_id={CHAIN_ID:#018x}")

    try:
        passed = 0
        failed = 0
        if check_round_trip_single_token(first, last):
            passed += 1
        else:
            failed += 1
        if check_counters_advanced(first, last):
            passed += 1
        else:
            failed += 1
        print(f"\n[smoke] {passed}/{passed + failed} checks passed")
        return 0 if failed == 0 else 1
    finally:
        first_sock.close()
        last_sock.close()


if __name__ == "__main__":
    sys.exit(main())
