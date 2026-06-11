"""Tests for the identity-bound tunnel handshake (v1.1 §6)."""
from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from nakshatra_auth import generate_keypair  # noqa: E402
from transport.identity_handshake import (  # noqa: E402
    make_proof, verify_proof, new_nonce, mutual_handshake, HandshakeError)


# ── pure proof ────────────────────────────────────────────────────────

def test_proof_roundtrip():
    priv, pub = generate_keypair()
    a, b = new_nonce(), new_nonce()           # a = our nonce (challenge), b = peer's
    proof = make_proof(priv, "responder", my_nonce=b, their_nonce=a, session_binding=b"sess")
    assert verify_proof(pub, proof, "responder", our_nonce=a, session_binding=b"sess")


def test_wrong_key_fails():
    priv, _ = generate_keypair()
    _, other_pub = generate_keypair()
    a, b = new_nonce(), new_nonce()
    proof = make_proof(priv, "responder", b, a, b"sess")
    assert not verify_proof(other_pub, proof, "responder", our_nonce=a, session_binding=b"sess")


def test_session_binding_mismatch_fails():
    priv, pub = generate_keypair()
    a, b = new_nonce(), new_nonce()
    proof = make_proof(priv, "responder", b, a, b"session-X")
    # same proof, different underlying channel -> rejected (anti-relay-substitution)
    assert not verify_proof(pub, proof, "responder", our_nonce=a, session_binding=b"session-Y")


def test_role_confusion_fails():
    priv, pub = generate_keypair()
    a, b = new_nonce(), new_nonce()
    proof = make_proof(priv, "initiator", b, a, b"sess")
    assert not verify_proof(pub, proof, "responder", our_nonce=a, session_binding=b"sess")


def test_nonce_replay_into_other_challenge_fails():
    priv, pub = generate_keypair()
    a, b = new_nonce(), new_nonce()
    proof = make_proof(priv, "responder", b, a, b"sess")
    # verifier expecting a DIFFERENT challenge than the one signed -> fails
    assert not verify_proof(pub, proof, "responder", our_nonce=new_nonce(), session_binding=b"sess")


# ── mutual handshake over a real socket pair ──────────────────────────

def _run_side(sock, priv, our_pub, peer_pinned, is_init, binding, out, idx):
    try:
        out[idx] = mutual_handshake(sock, priv, our_pub, peer_pinned, is_init, binding)
    except Exception as e:  # noqa: BLE001
        out[idx] = e
    finally:
        sock.close()


def _handshake_pair(priv_a, pub_a, priv_b, pub_b, *, a_pins, b_pins, binding=b"chan-1"):
    sa, sb = socket.socketpair()
    out = [None, None]
    ta = threading.Thread(target=_run_side, args=(sa, priv_a, pub_a, a_pins, True, binding, out, 0))
    tb = threading.Thread(target=_run_side, args=(sb, priv_b, pub_b, b_pins, False, binding, out, 1))
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    return out


def test_mutual_handshake_success():
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    out = _handshake_pair(priv_a, pub_a, priv_b, pub_b, a_pins=pub_b, b_pins=pub_a)
    assert getattr(out[0], "ok", False) and getattr(out[1], "ok", False)
    assert out[0].peer_pubkey_hex == pub_b and out[1].peer_pubkey_hex == pub_a


def test_mutual_handshake_mitm_wrong_pin():
    # A pins the WRONG key for B (a MITM substituted itself) -> A must reject.
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    _, attacker_pub = generate_keypair()
    out = _handshake_pair(priv_a, pub_a, priv_b, pub_b, a_pins=attacker_pub, b_pins=pub_a)
    assert isinstance(out[0], HandshakeError)        # A rejects B (not the pinned key)


def test_mutual_handshake_binding_mismatch():
    # Two sides think they're on different channels -> both reject (anti-relay).
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    sa, sb = socket.socketpair()
    out = [None, None]
    ta = threading.Thread(target=_run_side, args=(sa, priv_a, pub_a, pub_b, True, b"chan-A", out, 0))
    tb = threading.Thread(target=_run_side, args=(sb, priv_b, pub_b, pub_a, False, b"chan-B", out, 1))
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    assert isinstance(out[0], HandshakeError) or isinstance(out[1], HandshakeError)
