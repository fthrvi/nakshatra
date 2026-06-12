"""Tests for coordination-free auto-tunnel pairing (v1.1 capstone)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import pytest  # noqa: E402
from mesh.pairing import pair_role, rendezvous_id  # noqa: E402

A = "aa" * 32   # smaller
B = "bb" * 32   # larger


def test_rendezvous_id_is_order_independent():
    assert rendezvous_id(A, B, "m") == rendezvous_id(B, A, "m")

def test_rendezvous_id_is_16_bytes():
    assert len(rendezvous_id(A, B, "m")) == 16

def test_mesh_id_domain_separates():
    assert rendezvous_id(A, B, "mesh1") != rendezvous_id(A, B, "mesh2")

def test_roles_are_opposite_and_consistent():
    # both serve (default) → pubkey tie-break
    ra = pair_role(A, B, "m")   # A is smaller
    rb = pair_role(B, A, "m")   # B is larger
    # same rendezvous id from both sides
    assert ra.rendezvous_id == rb.rendezvous_id
    # exactly one initiator, exactly one client
    assert ra.is_initiator != rb.is_initiator
    assert ra.is_client != rb.is_client
    # canonical rule: smaller pubkey is initiator + client
    assert ra.is_initiator and ra.is_client
    assert not rb.is_initiator and not rb.is_client


def test_server_role_follows_who_serves():
    # B serves a worker, A is consumer-only → A must be client, B server,
    # regardless of pubkey order (here A is the SMALLER key but still client,
    # which agrees with the tie-break; flip it to prove serving dominates).
    # larger key C serves; smaller key A consumes → A client even though smaller
    a = pair_role(A, B, "m", i_serve=False, peer_serves=True)   # A consumes B
    b = pair_role(B, A, "m", i_serve=True, peer_serves=False)   # B serves
    assert a.is_client and not b.is_client
    assert a.is_initiator != b.is_initiator     # X25519 ordering still by key

def test_serving_dominates_pubkey_tiebreak():
    # the LARGER pubkey B serves; smaller A is consumer-only.
    # tie-break alone would make A (smaller) the client — serving agrees here,
    # so flip: make the SMALLER key A the server and larger B the consumer.
    a = pair_role(A, B, "m", i_serve=True, peer_serves=False)   # A (smaller) serves
    b = pair_role(B, A, "m", i_serve=False, peer_serves=True)   # B (larger) consumes
    assert not a.is_client and b.is_client      # server-by-serving beats tie-break
    # (tie-break would have made A the client since it's smaller)

def test_role_is_restart_stable():
    assert pair_role(A, B, "m") == pair_role(A, B, "m")

def test_case_insensitive():
    assert rendezvous_id(A.upper(), B, "m") == rendezvous_id(A, B, "m")
    assert pair_role(A.upper(), B, "m").is_client == pair_role(A, B, "m").is_client

def test_self_pairing_rejected():
    with pytest.raises(ValueError):
        pair_role(A, A, "m")
