"""Identity-bound tunnel handshake (v1.1 §6) — bind a transport session to the
pinned Ed25519 mesh identity.

The sovereign transport (v1.1) raises a per-peer tunnel (WireGuard / QUIC) between
two nodes that found each other via Nostr and pinned each other's Ed25519 keys.
WireGuard/QUIC give *a* secure channel — but to its OWN key, not the mesh
identity. This module is the small piece that closes that gap: a mutual
challenge-response proving each side holds the private key for the **pinned**
Ed25519 pubkey, and **binding that proof to the specific transport session** so a
man-in-the-middle who relays bytes can't substitute itself.

It is transport-agnostic on purpose — it authenticates the *session*, whether the
bytes ride WireGuard, QUIC, a relay, or (for the spike) a plain TCP socket over
the tailnet. Discovery is public; this is the line where admission becomes a
*proven* peer, not just a pinned hope.

Binding: each side signs `nakshatra-tunnel-v1 | role | their_nonce | my_nonce |
session_binding`, where `session_binding` is a value unique to the underlying
channel (e.g. a WireGuard public-key pair hash, a TLS exporter, or — for the
spike — a shared transcript). Replaying a signature into a different session
fails because the nonces + binding differ.

Reuses the repo's Ed25519 primitives (nakshatra_auth) so there is ONE signature
scheme across the data plane, the listings, and now the tunnel.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

HANDSHAKE_TAG = b"nakshatra-tunnel-v1"
NONCE_LEN = 32


class HandshakeError(Exception):
    """The peer failed to prove the pinned identity / the binding mismatched."""


def new_nonce() -> bytes:
    return os.urandom(NONCE_LEN)


def _signed_bytes(role: str, my_nonce: bytes, their_nonce: bytes,
                  session_binding: bytes) -> bytes:
    """The exact bytes a side signs. role ∈ {"initiator","responder"} so the two
    directions can never be confused/replayed across each other."""
    return b"\n".join([
        HANDSHAKE_TAG,
        role.encode("ascii"),
        their_nonce,        # the challenge we are answering
        my_nonce,           # our own nonce (so both sides commit)
        session_binding,    # ties the proof to THIS underlying channel
    ])


@dataclass(frozen=True)
class Proof:
    """One side's signed proof of the pinned identity for this session."""
    role: str
    nonce_b64: str
    signature_b64: str

    @property
    def nonce(self) -> bytes:
        return base64.b64decode(self.nonce_b64)


def make_proof(priv_bytes: bytes, role: str, my_nonce: bytes, their_nonce: bytes,
               session_binding: bytes) -> Proof:
    """Sign our identity proof for this session/direction."""
    priv = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
    sig = priv.sign(_signed_bytes(role, my_nonce, their_nonce, session_binding))
    return Proof(role=role,
                 nonce_b64=base64.b64encode(my_nonce).decode("ascii"),
                 signature_b64=base64.b64encode(sig).decode("ascii"))


def verify_proof(pinned_pubkey_hex: str, proof: Proof, expected_role: str,
                 our_nonce: bytes, session_binding: bytes) -> bool:
    """Verify the peer's proof binds the PINNED key to THIS session. Never raises.

    `our_nonce` is the challenge WE sent (which the peer signed as `their_nonce`);
    `proof.nonce` is the peer's own nonce. `expected_role` is the peer's role."""
    if proof.role != expected_role:
        return False
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pinned_pubkey_hex))
        sig = base64.b64decode(proof.signature_b64, validate=True)
        msg = _signed_bytes(expected_role, their_nonce=our_nonce,
                            my_nonce=proof.nonce, session_binding=session_binding)
        pub.verify(sig, msg)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


# ── one-call mutual handshake over a byte stream (spike + reuse) ──────────

@dataclass
class HandshakeResult:
    ok: bool
    peer_pubkey_hex: str
    session_binding: bytes


def _send(sock, data: bytes) -> None:
    sock.sendall(len(data).to_bytes(4, "big") + data)


def _recv(sock) -> bytes:
    hdr = _recv_exact(sock, 4)
    n = int.from_bytes(hdr, "big")
    if n > 1 << 20:
        raise HandshakeError("handshake frame too large")
    return _recv_exact(sock, n)


def _recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise HandshakeError("connection closed during handshake")
        buf += chunk
    return buf


def _proof_wire(p: Proof) -> bytes:
    return b"\n".join([p.role.encode(), p.nonce_b64.encode(), p.signature_b64.encode()])


def _proof_parse(b: bytes) -> Proof:
    role, nonce_b64, sig_b64 = b.split(b"\n", 2)
    return Proof(role.decode(), nonce_b64.decode(), sig_b64.decode())


def mutual_handshake(sock, priv_bytes: bytes, our_pubkey_hex: str,
                     pinned_peer_pubkey_hex: str, is_initiator: bool,
                     session_binding: bytes = b"") -> HandshakeResult:
    """Run the mutual identity handshake over an already-connected socket. Both
    sides must agree on `session_binding` (the underlying channel's binder; b"" is
    acceptable only for a same-process / trusted-transport spike). Returns a
    HandshakeResult(ok=True) iff the peer proved the pinned identity for this
    session; raises HandshakeError on protocol failure.

    Wire: exchange nonces, then exchange signed proofs. Initiator sends first."""
    my_role = "initiator" if is_initiator else "responder"
    peer_role = "responder" if is_initiator else "initiator"
    my_nonce = new_nonce()

    if is_initiator:
        _send(sock, my_nonce)
        their_nonce = _recv(sock)
    else:
        their_nonce = _recv(sock)
        _send(sock, my_nonce)

    my_proof = make_proof(priv_bytes, my_role, my_nonce, their_nonce, session_binding)
    if is_initiator:
        _send(sock, _proof_wire(my_proof))
        peer_proof = _proof_parse(_recv(sock))
    else:
        peer_proof = _proof_parse(_recv(sock))
        _send(sock, _proof_wire(my_proof))

    # The peer signed OUR nonce (my_nonce) as their `their_nonce`.
    if not verify_proof(pinned_peer_pubkey_hex, peer_proof, peer_role,
                        our_nonce=my_nonce, session_binding=session_binding):
        raise HandshakeError(
            f"peer failed to prove pinned identity {pinned_peer_pubkey_hex[:12]}… "
            f"for this session (MITM, wrong key, or binding mismatch)")
    return HandshakeResult(ok=True, peer_pubkey_hex=pinned_peer_pubkey_hex,
                           session_binding=session_binding)
