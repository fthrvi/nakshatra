"""The standing relay's limits — the ones a spike relay could do without.

⚠️⚠️ THE UNBOUNDED WAIT WAS THE SHARP EDGE, not the missing allowlist. `_waiting` held a
socket per unpaired rendezvous_id with no expiry and no cap. Connect N times with N distinct
ids, never send a partner, and the relay holds N sockets forever — no secret needed, no id
guessed, just reachability. These tests demonstrate the attack and that it now fails.
"""
import socket
import struct
import time

import pytest

from transport.relay import MAGIC, RendezvousRelay, connect


def _half_pair(port, rid: bytes) -> socket.socket:
    """Register under `rid` and then do nothing — the attack, and also a normal waiter."""
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(MAGIC + struct.pack(">B", len(rid)) + rid)
    return s


@pytest.fixture
def relay():
    r = RendezvousRelay(host="127.0.0.1", port=0, max_waiting=4,
                        waiting_ttl_s=1.0, max_per_ip_per_min=1000)
    port = r.start()
    yield r, port
    r.stop()


def test_a_normal_pair_still_works(relay):
    r, port = relay
    a = connect("127.0.0.1", port, b"pair-1", timeout=5)
    b = connect("127.0.0.1", port, b"pair-1", timeout=5)
    a.sendall(b"hello")
    assert b.recv(5) == b"hello"
    b.sendall(b"world")
    assert a.recv(5) == b"world"
    a.close(); b.close()


def test_unpaired_waiters_are_capped(relay):
    """⚠️ The exhaustion: N distinct ids, no partners. Beyond the cap it refuses."""
    r, port = relay
    socks = [_half_pair(port, f"flood-{i}".encode()) for i in range(4)]
    time.sleep(0.3)
    assert len(r._waiting) <= r.max_waiting
    extra = _half_pair(port, b"flood-overflow")
    time.sleep(0.3)
    assert r.refused >= 1, "the 5th waiter should have been refused"
    assert len(r._waiting) <= r.max_waiting
    for s in socks + [extra]:
        s.close()


def test_the_cap_refuses_the_newcomer_not_an_existing_waiter(relay):
    """⚠️ A flood must not be able to evict half-pairs already in progress — which is
    exactly what LRU eviction would allow."""
    r, port = relay
    keeper = _half_pair(port, b"keeper")
    time.sleep(0.2)
    floods = [_half_pair(port, f"f{i}".encode()) for i in range(6)]
    time.sleep(0.3)
    assert b"keeper" in r._waiting, "an existing waiter was evicted by the flood"
    partner = connect("127.0.0.1", port, b"keeper", timeout=5)
    keeper.sendall(b"still-here")
    assert partner.recv(10) == b"still-here"
    keeper.close(); partner.close()
    for s in floods:
        s.close()


def test_stale_waiters_are_reaped(relay):
    """A partner that has not arrived in the TTL is not arriving."""
    r, port = relay
    s = _half_pair(port, b"abandoned")
    time.sleep(0.2)
    assert b"abandoned" in r._waiting
    time.sleep(1.2)                       # past waiting_ttl_s
    _half_pair(port, b"trigger-reap").close()
    time.sleep(0.3)
    assert b"abandoned" not in r._waiting, "stale waiter was never reaped"
    s.close()


def test_an_allowlist_refuses_an_unlisted_id():
    r = RendezvousRelay(host="127.0.0.1", port=0, allowlist={b"permitted"})
    port = r.start()
    try:
        ok_a = connect("127.0.0.1", port, b"permitted", timeout=5)
        ok_b = connect("127.0.0.1", port, b"permitted", timeout=5)
        ok_a.sendall(b"x")
        assert ok_b.recv(1) == b"x"
        ok_a.close(); ok_b.close()

        bad = _half_pair(port, b"not-permitted")
        time.sleep(0.3)
        assert r.refused >= 1
        assert b"not-permitted" not in r._waiting
        bad.close()
    finally:
        r.stop()


def test_no_allowlist_means_open_as_before():
    """⚠️ Default None ⇒ previous behaviour. Defaulting it ON with an empty set would deny
    every existing deployment on upgrade."""
    r = RendezvousRelay(host="127.0.0.1", port=0)
    port = r.start()
    try:
        assert r.allowlist is None
        a = connect("127.0.0.1", port, b"anything", timeout=5)
        b = connect("127.0.0.1", port, b"anything", timeout=5)
        a.sendall(b"z")
        assert b.recv(1) == b"z"
        a.close(); b.close()
    finally:
        r.stop()


def test_per_ip_rate_limit_refuses_before_reading_anything():
    r = RendezvousRelay(host="127.0.0.1", port=0, max_per_ip_per_min=3)
    port = r.start()
    try:
        opened = []
        for i in range(8):
            try:
                opened.append(_half_pair(port, f"r{i}".encode()))
            except OSError:
                pass
        time.sleep(0.4)
        assert r.refused >= 1, "the rate limit never fired"
        for s in opened:
            s.close()
    finally:
        r.stop()
