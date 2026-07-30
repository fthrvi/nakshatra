"""Tests for the rendezvous-assisted UDP hole-punch (discovery/direct-path).

Two REAL UDP sockets bound to 127.0.0.1 exercise the full simultaneous-open
state machine end to end — no real network beyond localhost, no GPU, no live
meshd service touched. What real-NAT scenarios this canNOT prove (full-cone vs
restricted vs symmetric NAT behaviour) is documented honestly in
docs/findings/direct-path.md, not glossed over here.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from mesh.direct_path import (  # noqa: E402
    PathResult,
    answer_ping,
    maybe_direct,
    order_candidates,
    probe_rtt,
    punch,
)

TOKEN = b"shared-pair-token-for-tests"


def _udp_socket() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    return s


# ── candidate ordering (pure function, no sockets) ─────────────────────────

def test_order_candidates_lan_before_public():
    # Note: 100.64.0.0/10 (CGNAT shared space) and the 192.0.2.0/24 / 198.51.100.0/24
    # / 203.0.113.0/24 documentation ranges are all classified `is_private` by
    # Python's ipaddress module too (non-globally-routable) — use real globally
    # routable addresses for the "public" side so this test isn't fighting stdlib.
    endpoints = [
        ("8.8.8.8", 4001),         # public
        ("10.0.0.7", 4002),        # private (RFC1918)
        ("1.1.1.1", 4003),         # public
        ("192.168.1.42", 4004),    # private (RFC1918)
        ("172.20.5.9", 4005),      # private (RFC1918)
        ("9.9.9.9", 4006),         # public
    ]
    ordered = order_candidates(endpoints)
    private_set = {("10.0.0.7", 4002), ("192.168.1.42", 4004), ("172.20.5.9", 4005)}
    got_private = [e for e in ordered if e in private_set]
    got_public = [e for e in ordered if e not in private_set]
    # all private candidates precede all public ones
    assert ordered[:len(private_set)] == got_private
    assert ordered[len(private_set):] == got_public
    # stable within each group (relative order preserved)
    assert got_private == [("10.0.0.7", 4002), ("192.168.1.42", 4004), ("172.20.5.9", 4005)]
    assert got_public == [("8.8.8.8", 4001), ("1.1.1.1", 4003), ("9.9.9.9", 4006)]


def test_order_candidates_hostname_treated_as_public():
    # an unresolved hostname can't be classified private -> sorts after known-LAN IPs
    ordered = order_candidates([("worker.example.internal", 5000), ("10.0.0.1", 5001)])
    assert ordered == [("10.0.0.1", 5001), ("worker.example.internal", 5000)]


# ── both-sides punch success ────────────────────────────────────────────────

def test_punch_both_sides_success():
    sock_a = _udp_socket()
    sock_b = _udp_socket()
    try:
        addr_a = ("127.0.0.1", sock_a.getsockname()[1])
        addr_b = ("127.0.0.1", sock_b.getsockname()[1])
        results = {}

        def run(tag, sock, local_id, peer_id, peer_addr):
            results[tag] = punch(sock, local_id, peer_id, [peer_addr], TOKEN, timeout=2.0)

        ta = threading.Thread(target=run, args=("a", sock_a, "nks-a", "nks-b", addr_b))
        tb = threading.Thread(target=run, args=("b", sock_b, "nks-b", "nks-a", addr_a))
        ta.start(); tb.start()
        ta.join(3); tb.join(3)

        assert isinstance(results["a"], PathResult) and results["a"].direct
        assert isinstance(results["b"], PathResult) and results["b"].direct
        assert results["a"].endpoint == addr_b
        assert results["b"].endpoint == addr_a
        assert results["a"].rtt_ms is not None and results["a"].rtt_ms >= 0
        assert results["b"].rtt_ms is not None and results["b"].rtt_ms >= 0
        # loopback round trip should be fast, not the full timeout
        assert results["a"].rtt_ms < 500 and results["b"].rtt_ms < 500
    finally:
        sock_a.close(); sock_b.close()


def test_punch_lan_candidate_preferred_when_mixed_with_unreachable_public():
    """A candidate list with a real (LAN-classified, loopback counts as private)
    endpoint plus a decoy that nothing listens on: the punch still succeeds via
    the reachable candidate, proving order_candidates's LAN-first list doesn't
    block trying the rest — both are attempted, the live one wins."""
    sock_a = _udp_socket()
    sock_b = _udp_socket()
    try:
        addr_b = ("127.0.0.1", sock_b.getsockname()[1])
        decoy = ("127.0.0.1", 1)  # privileged port, nothing bound there — unreachable
        results = {}

        def run_a():
            results["a"] = punch(sock_a, "nks-a", "nks-b", [decoy, addr_b], TOKEN, timeout=2.0)

        def run_b():
            results["b"] = punch(sock_b, "nks-b", "nks-a", [("127.0.0.1", sock_a.getsockname()[1])],
                                 TOKEN, timeout=2.0)

        ta = threading.Thread(target=run_a)
        tb = threading.Thread(target=run_b)
        ta.start(); tb.start()
        ta.join(3); tb.join(3)
        assert results["a"].direct and results["a"].endpoint == addr_b
        assert results["b"].direct
    finally:
        sock_a.close(); sock_b.close()


# ── one-side-silent → timeout → relay fallback ──────────────────────────────

def test_punch_one_side_silent_falls_back():
    sock_a = _udp_socket()
    sock_b = _udp_socket()   # bound and listening at the OS level, but never
    try:                     # calls punch() — a peer that just never answers
        addr_b = ("127.0.0.1", sock_b.getsockname()[1])
        result = punch(sock_a, "nks-a", "nks-b", [addr_b], TOKEN, timeout=0.4)
        assert result.direct is False
        assert result.endpoint is None
        assert "timeout" in (result.reason or "").lower()
    finally:
        sock_a.close(); sock_b.close()


def test_punch_no_candidates_fails_immediately():
    sock_a = _udp_socket()
    try:
        result = punch(sock_a, "nks-a", "nks-b", [], TOKEN, timeout=1.0)
        assert result.direct is False
        assert "no candidate" in (result.reason or "").lower()
    finally:
        sock_a.close()


# ── bad-HMAC rejected (can't hijack a path) ─────────────────────────────────

def test_punch_rejects_bad_hmac_forged_hello_ack():
    """An attacker who intercepts A's HELLO and replies with a forged HELLO_ACK
    signed under the WRONG token must NOT be able to complete the handshake —
    A keeps waiting (and times out) rather than accepting the forged path."""
    sock_a = _udp_socket()
    attacker_sock = _udp_socket()
    try:
        attacker_addr = ("127.0.0.1", attacker_sock.getsockname()[1])
        stop = threading.Event()

        def attacker_loop():
            from mesh.direct_path import _Msg, _pack, _unpack, _T_HELLO, _T_HELLO_ACK
            attacker_sock.settimeout(0.05)
            wrong_token = b"not-the-shared-token"
            while not stop.is_set():
                try:
                    data, addr = attacker_sock.recvfrom(2048)
                except (socket.timeout, OSError):
                    continue
                # attacker can't verify with the real token, but can still SEE the
                # magic/type/nonce framing and forge a plausible-looking reply
                fake_nonce = os.urandom(8)
                forged = _pack(_Msg(_T_HELLO_ACK, "nks-b", fake_nonce, b"\x00" * 8), wrong_token)
                attacker_sock.sendto(forged, addr)

        t = threading.Thread(target=attacker_loop, daemon=True)
        t.start()
        result = punch(sock_a, "nks-a", "nks-b", [attacker_addr], TOKEN, timeout=0.5)
        stop.set()
        t.join(2)

        assert result.direct is False
        assert "timeout" in (result.reason or "").lower()
    finally:
        sock_a.close(); attacker_sock.close()


def test_pack_unpack_roundtrip_and_tamper_rejected():
    """Low-level wire check: a valid packet round-trips; flipping one byte (either
    the payload or the MAC) breaks verification."""
    from mesh.direct_path import _Msg, _pack, _unpack, _T_HELLO

    msg = _Msg(_T_HELLO, "nks-a", b"12345678", b"\x00" * 8)
    good = _pack(msg, TOKEN)
    parsed = _unpack(good, TOKEN)
    assert parsed is not None
    assert parsed.sender_id == "nks-a" and parsed.mtype == _T_HELLO

    # wrong token entirely
    assert _unpack(good, b"totally-different-token") is None

    # single-byte tamper in the body
    tampered = bytearray(good)
    tampered[-1] ^= 0xFF   # flip a bit in the MAC itself
    assert _unpack(bytes(tampered), TOKEN) is None

    body_tampered = bytearray(good)
    body_tampered[6] ^= 0xFF   # flip a bit inside the signed body (sender_id area)
    assert _unpack(bytes(body_tampered), TOKEN) is None


# ── RTT probe on an established path ────────────────────────────────────────

def test_probe_rtt_measures_round_trip():
    sock_a = _udp_socket()
    sock_b = _udp_socket()
    try:
        addr_b = ("127.0.0.1", sock_b.getsockname()[1])
        responder = threading.Thread(target=answer_ping, args=(sock_b, "nks-b", TOKEN, 2.0))
        responder.start()
        rtt = probe_rtt(sock_a, addr_b, TOKEN, local_id="nks-a", timeout=1.0, retries=3)
        responder.join(3)
        assert rtt is not None
        assert 0 <= rtt < 500
    finally:
        sock_a.close(); sock_b.close()


def test_probe_rtt_none_when_peer_silent():
    sock_a = _udp_socket()
    sock_b = _udp_socket()
    try:
        addr_b = ("127.0.0.1", sock_b.getsockname()[1])
        rtt = probe_rtt(sock_a, addr_b, TOKEN, local_id="nks-a", timeout=0.15, retries=2)
        assert rtt is None
    finally:
        sock_a.close(); sock_b.close()


# ── maybe_direct integration seam — default OFF, true no-op ────────────────

def test_maybe_direct_default_off_sends_nothing():
    sock_a = _udp_socket()
    sock_b = _udp_socket()
    try:
        os.environ.pop("NKS_DIRECT_PATH", None)   # ensure default (unset = OFF)
        addr_b = ("127.0.0.1", sock_b.getsockname()[1])
        result = maybe_direct(sock_a, "nks-a", "nks-b", [addr_b], TOKEN, timeout=0.2)
        assert result.direct is False
        assert "disabled" in (result.reason or "").lower()
        # confirm literally nothing arrived at B — a true no-op, not just a fast fail
        sock_b.settimeout(0.1)
        with pytest.raises(socket.timeout):
            sock_b.recvfrom(64)
    finally:
        sock_a.close(); sock_b.close()


def test_maybe_direct_enabled_runs_the_real_punch():
    sock_a = _udp_socket()
    sock_b = _udp_socket()
    try:
        os.environ["NKS_DIRECT_PATH"] = "1"
        addr_a = ("127.0.0.1", sock_a.getsockname()[1])
        addr_b = ("127.0.0.1", sock_b.getsockname()[1])
        results = {}

        def run(tag, sock, local_id, peer_id, peer_addr):
            results[tag] = maybe_direct(sock, local_id, peer_id, [peer_addr], TOKEN, timeout=1.0)

        ta = threading.Thread(target=run, args=("a", sock_a, "nks-a", "nks-b", addr_b))
        tb = threading.Thread(target=run, args=("b", sock_b, "nks-b", "nks-a", addr_a))
        ta.start(); tb.start()
        ta.join(2); tb.join(2)
        assert results["a"].direct and results["b"].direct
    finally:
        os.environ.pop("NKS_DIRECT_PATH", None)
        sock_a.close(); sock_b.close()
