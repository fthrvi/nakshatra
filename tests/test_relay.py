"""Tests for the rendezvous relay (v1.1 §8.4) — pairs NAT'd peers + forwards an
identity-handshake-gated tunnel between them, untrusted."""
from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from nakshatra_auth import generate_keypair  # noqa: E402
from transport.relay import RendezvousRelay, connect  # noqa: E402
from transport.identity_handshake import mutual_handshake, HandshakeError  # noqa: E402


@pytest.fixture
def relay():
    r = RendezvousRelay(host="127.0.0.1", port=0)
    r.start()
    yield r
    r.stop()


def test_relay_pairs_and_pipes_bytes(relay):
    rid = b"pair-1"
    out = {}

    def side(tag, send, expect):
        s = connect("127.0.0.1", relay.port, rid, timeout=5)
        s.sendall(send)
        out[tag] = s.recv(64)
        s.close()

    ta = threading.Thread(target=side, args=("a", b"ping", b"pong"))
    tb = threading.Thread(target=side, args=("b", b"pong", b"ping"))
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    # each side received what the OTHER sent → the relay piped both directions
    assert out["a"] == b"pong" and out["b"] == b"ping"
    assert relay.paired == 1


def test_identity_handshake_through_relay(relay):
    """Two peers, each connecting OUT to the relay, run the mutual identity
    handshake END-TO-END through it — the relay forwards but cannot forge."""
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    rid = b"handshake-rdv"
    out = [None, None]

    def side(idx, priv, our_pub, peer_pin, is_init):
        try:
            s = connect("127.0.0.1", relay.port, rid, timeout=5)
            out[idx] = mutual_handshake(s, priv, our_pub, peer_pin, is_init,
                                        session_binding=b"relay-session")
            s.close()
        except Exception as e:  # noqa: BLE001
            out[idx] = e

    ta = threading.Thread(target=side, args=(0, priv_a, pub_a, pub_b, True))
    tb = threading.Thread(target=side, args=(1, priv_b, pub_b, pub_a, False))
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    assert getattr(out[0], "ok", False) and getattr(out[1], "ok", False)
    assert out[0].peer_pubkey_hex == pub_b and out[1].peer_pubkey_hex == pub_a


def test_relay_cannot_impersonate(relay):
    """If a peer pins the WRONG key (a MITM at the relay), the handshake through
    the relay still fails — the relay forwards bytes but can't forge the proof."""
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    _, attacker = generate_keypair()
    rid = b"mitm-rdv"
    out = [None, None]

    def side(idx, priv, our_pub, peer_pin, is_init):
        try:
            s = connect("127.0.0.1", relay.port, rid, timeout=5)
            out[idx] = mutual_handshake(s, priv, our_pub, peer_pin, is_init,
                                        session_binding=b"s")
            s.close()
        except Exception as e:  # noqa: BLE001
            out[idx] = e

    ta = threading.Thread(target=side, args=(0, priv_a, pub_a, attacker, True))  # A pins attacker
    tb = threading.Thread(target=side, args=(1, priv_b, pub_b, pub_a, False))
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    assert isinstance(out[0], HandshakeError)   # A rejects: B isn't the pinned key
