"""Coordination-free pairing for auto-tunnel bring-up (v1.1 capstone).

When two meshd nodes discover each other, they must agree — with NO extra
round-trip — on three things so a tunnel can form over the untrusted rendezvous
relay (transport/relay.py):

  1. a shared **rendezvous id** (the relay pairs the two outbound connections that
     present the same id);
  2. which side **dials as the X25519 initiator** vs responder
     (secure_channel.secure_handshake needs exactly one of each);
  3. which side runs the tunnel **client** (exposes a local port → peer's worker)
     vs **server** (accepts streams → its own local worker).

Both sides derive all three purely from the two Ed25519 mesh pubkeys + the mesh
id, so no negotiation is needed: each node runs `pair_role(my_pub, peer_pub,
mesh_id)` and gets a consistent, opposite role from the peer. The smaller pubkey
(lexicographic) is the canonical initiator/client; this is symmetric and stable
across restarts because the keys are persisted.

The rendezvous id is NOT a secret — the relay only uses it to pair sockets; the
Ed25519-pinned X25519 handshake is what authenticates and encrypts. So a
predictable id is fine (an attacker who guesses it still can't complete the
pinned handshake).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PairRole:
    rendezvous_id: bytes   # 16 bytes; what both present to the relay
    is_initiator: bool     # X25519 handshake initiator (secure_handshake is_init)
    is_client: bool        # True → expose a local port to peer's worker;
    #                        False → serve streams to our own local worker
    peer_pubkey_hex: str   # echoed for convenience


def rendezvous_id(a_pub_hex: str, b_pub_hex: str, mesh_id: str) -> bytes:
    """A deterministic 16-byte pairing id for the {a,b} pair in `mesh_id`.

    Order-independent (sorted) so both nodes compute the same value. Domain-
    separated by mesh_id so the same two keys in different meshes don't collide.
    """
    lo, hi = sorted((a_pub_hex.lower(), b_pub_hex.lower()))
    h = hashlib.sha256(f"nakshatra-rdv\x00{mesh_id}\x00{lo}\x00{hi}".encode())
    return h.digest()[:16]


def pair_role(my_pub_hex: str, peer_pub_hex: str, mesh_id: str,
              i_serve: bool = True, peer_serves: bool = True) -> PairRole:
    """My role in the auto-tunnel with `peer`. Opposite of what the peer derives.

    Two independent assignments, both coordination-free:

    • **is_initiator** (X25519 handshake ordering) — the lexicographically SMALLER
      pubkey initiates. Pure key order, so exactly one of each.

    • **is_client** (who consumes whose worker) — the node that SERVES a worker is
      the tunnel server; a consumer-only node is the client. `i_serve`/`peer_serves`
      come from the signed listing (`serving` non-empty), so both sides agree
      without a round-trip. When BOTH serve (or neither), fall back to the pubkey
      tie-break (smaller = client) so the choice is still deterministic.

    Both rules are restart-stable because the keys + the listing are persisted /
    signed."""
    if my_pub_hex.lower() == peer_pub_hex.lower():
        raise ValueError("cannot pair a node with itself (identical pubkeys)")
    i_am_smaller = my_pub_hex.lower() < peer_pub_hex.lower()
    if i_serve and not peer_serves:
        is_client = False                      # only I serve → I'm the server
    elif peer_serves and not i_serve:
        is_client = True                       # only peer serves → I consume it
    else:
        is_client = i_am_smaller               # both/neither serve → key tie-break
    return PairRole(
        rendezvous_id=rendezvous_id(my_pub_hex, peer_pub_hex, mesh_id),
        is_initiator=i_am_smaller,
        is_client=is_client,
        peer_pubkey_hex=peer_pub_hex,
    )
