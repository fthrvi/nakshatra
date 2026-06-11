"""Encrypted channel over the relay (v1.1 §8.5b) — make the relay zero-knowledge.

The rendezvous relay forwards bytes; the identity handshake (§6) proves *who* the
peer is. This layer adds **confidentiality + forward secrecy** so the untrusted
relay sees only ciphertext — the last property separating "a relay that can't
impersonate" from "a relay that can't read."

Pattern = the Noise/WireGuard shape, built from vetted `cryptography` primitives:

  1. Each side makes an EPHEMERAL X25519 keypair (forward secrecy — keys vanish
     with the session).
  2. The ephemeral X25519 pubkey is AUTHENTICATED by the static **pinned Ed25519
     identity**: each side signs `tag|role|nonces|its_x25519_pub|session_binding`.
     Verifying against the pinned key makes the exchange MITM-proof (a relay that
     swaps in its own ephemeral key can't forge the signature) AND binds it to
     this session (anti-replay / anti-relay-substitution).
  3. X25519 ECDH → HKDF-SHA256 → one ChaCha20-Poly1305 key per direction.
  4. The byte stream is AEAD-sealed per record, monotonic nonce per direction
     (unique key per direction ⇒ counters start at 0, never reuse).

`SecureChannel` exposes `sendall`/`recv` so it's a **drop-in for the raw socket**
— `MuxTunnel(secure_channel)` runs unchanged, and the gRPC data plane is now
end-to-end encrypted across the relay.

Spike note: this composes audited primitives in the standard Noise_IK shape. A
production deployment should prefer WireGuard or a full Noise library for the
complete, formally-analysed protocol; this gives the same guarantees for the
relay spike without a kernel/root dependency.
"""
from __future__ import annotations

import base64
import os
import socket
import struct
from typing import Optional

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

HS_TAG = b"nakshatra-secure-v1"
NONCE_LEN = 32
MAX_RECORD = 64 * 1024
_LEN = struct.Struct(">I")


class SecureChannelError(Exception):
    """Handshake auth failed, or a record failed to decrypt (tamper/relay-MITM)."""


# ── handshake wire (length-prefixed frames over the raw socket) ──────────

def _send(sock, data: bytes) -> None:
    sock.sendall(_LEN.pack(len(data)) + data)


def _recv(sock) -> bytes:
    n = _LEN.unpack(_recv_exact(sock, 4))[0]
    if n > 1 << 20:
        raise SecureChannelError("handshake frame too large")
    return _recv_exact(sock, n)


def _recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise SecureChannelError("connection closed during handshake")
        buf += chunk
    return buf


def _signed(role: str, their_nonce: bytes, my_nonce: bytes,
            my_x_pub: bytes, session_binding: bytes) -> bytes:
    return b"\n".join([HS_TAG, role.encode(), their_nonce, my_nonce,
                       my_x_pub, session_binding])


def _x_pub_bytes(xpriv: x25519.X25519PrivateKey) -> bytes:
    return xpriv.public_key().public_bytes(serialization.Encoding.Raw,
                                           serialization.PublicFormat.Raw)


def secure_handshake(sock: socket.socket, priv_ed_bytes: bytes,
                     peer_pinned_ed_hex: str, is_initiator: bool,
                     session_binding: bytes = b"") -> "SecureChannel":
    """Mutually authenticate (pinned Ed25519) + agree a forward-secret key, and
    return a SecureChannel that encrypts everything after. Raises on any failure."""
    role = "initiator" if is_initiator else "responder"
    peer_role = "responder" if is_initiator else "initiator"
    ed_priv = ed25519.Ed25519PrivateKey.from_private_bytes(priv_ed_bytes)
    x_priv = x25519.X25519PrivateKey.generate()
    my_x_pub = _x_pub_bytes(x_priv)
    my_nonce = os.urandom(NONCE_LEN)

    # exchange nonces + ephemeral X25519 pubkeys (initiator first)
    my_hello = my_nonce + my_x_pub
    if is_initiator:
        _send(sock, my_hello)
        peer_hello = _recv(sock)
    else:
        peer_hello = _recv(sock)
        _send(sock, my_hello)
    their_nonce, their_x_pub = peer_hello[:NONCE_LEN], peer_hello[NONCE_LEN:NONCE_LEN + 32]
    if len(their_x_pub) != 32:
        raise SecureChannelError("malformed peer hello")

    # sign our (role, nonces, our ephemeral pub) under the static Ed25519 identity
    my_sig = ed_priv.sign(_signed(role, their_nonce, my_nonce, my_x_pub, session_binding))
    if is_initiator:
        _send(sock, my_sig)
        peer_sig = _recv(sock)
    else:
        peer_sig = _recv(sock)
        _send(sock, my_sig)

    # verify the peer's signature binds the PINNED identity to THIS exchange
    try:
        peer_ed = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(peer_pinned_ed_hex))
        peer_ed.verify(peer_sig, _signed(peer_role, my_nonce, their_nonce,
                                         their_x_pub, session_binding))
    except (InvalidSignature, ValueError, TypeError) as e:
        raise SecureChannelError(
            f"peer failed to authenticate the key exchange to pinned identity "
            f"{peer_pinned_ed_hex[:12]}… (MITM / wrong key / binding mismatch)") from e

    # ECDH → HKDF → per-direction ChaCha20-Poly1305 keys. Salt = sorted nonces so
    # both sides derive identically regardless of role.
    shared = x_priv.exchange(x25519.X25519PublicKey.from_public_bytes(their_x_pub))
    salt = b"".join(sorted([my_nonce, their_nonce]))
    keymat = HKDF(algorithm=hashes.SHA256(), length=64, salt=salt,
                  info=HS_TAG + b"|keys").derive(shared)
    k_i2r, k_r2i = keymat[:32], keymat[32:]
    if is_initiator:
        send_key, recv_key = k_i2r, k_r2i
    else:
        send_key, recv_key = k_r2i, k_i2r
    return SecureChannel(sock, send_key, recv_key,
                         peer_pubkey_hex=peer_pinned_ed_hex)


# ── the encrypted stream (drop-in for a socket: sendall / recv) ─────────

class SecureChannel:
    def __init__(self, sock: socket.socket, send_key: bytes, recv_key: bytes,
                 peer_pubkey_hex: str = ""):
        self._sock = sock
        self._send_aead = ChaCha20Poly1305(send_key)
        self._recv_aead = ChaCha20Poly1305(recv_key)
        self._send_ctr = 0
        self._recv_ctr = 0
        self._inbuf = b""
        self.peer_pubkey_hex = peer_pubkey_hex

    @staticmethod
    def _nonce(ctr: int) -> bytes:
        return b"\x00\x00\x00\x00" + ctr.to_bytes(8, "big")

    def sendall(self, data: bytes) -> None:
        mv = memoryview(data)
        for off in range(0, len(mv), MAX_RECORD):
            chunk = bytes(mv[off:off + MAX_RECORD])
            ct = self._send_aead.encrypt(self._nonce(self._send_ctr), chunk, None)
            self._send_ctr += 1
            self._sock.sendall(_LEN.pack(len(ct)) + ct)

    def _read_record(self) -> bool:
        """Decrypt one record into the inbuf. False on clean EOF; raises on tamper."""
        hdr = self._sock.recv(4)
        if not hdr:
            return False
        while len(hdr) < 4:
            more = self._sock.recv(4 - len(hdr))
            if not more:
                return False
            hdr += more
        n = _LEN.unpack(hdr)[0]
        ct = b""
        while len(ct) < n:
            more = self._sock.recv(n - len(ct))
            if not more:
                return False
            ct += more
        try:
            pt = self._recv_aead.decrypt(self._nonce(self._recv_ctr), ct, None)
        except (InvalidTag, InvalidSignature) as e:
            raise SecureChannelError("record failed to decrypt (tampered / relay-MITM)") from e
        self._recv_ctr += 1
        self._inbuf += pt
        return True

    def recv(self, n: int) -> bytes:
        while not self._inbuf:
            try:
                if not self._read_record():
                    return b""
            except OSError:
                return b""
        out, self._inbuf = self._inbuf[:n], self._inbuf[n:]
        return out

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
