"""NIP-01 Nostr event primitives (secp256k1 Schnorr) — the discovery wire.

This is the secp256k1 layer that the NostrRelay transport (relay.py) needs to
publish/query NakshatraListings over a public Nostr relay. It is intentionally
SEPARATE from the mesh identity: the Nostr key (secp256k1, here) authenticates an
event to the *relay* (anti-spam); the Ed25519 mesh key inside the listing content
is what *admission* pins against (relay.py / nakshatra_listing.py). Two keys, two
jobs — discovery is public gossip, the mesh stays Ed25519-pinned.

Requires `coincurve` (BIP340 Schnorr). Import errors are the caller's to handle.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from coincurve import PrivateKey, PublicKeyXOnly


def keygen() -> tuple[str, str]:
    """New Nostr identity. Returns (privkey_hex, xonly_pubkey_hex)."""
    pk = PrivateKey()
    return pk.secret.hex(), _xonly_hex(pk)


def pubkey_of(privkey_hex: str) -> str:
    return _xonly_hex(PrivateKey(bytes.fromhex(privkey_hex)))


def load_or_create_key(path) -> str:
    """Persisted Nostr identity (privkey hex), created on first use (0600).

    Replaceable-event semantics key on (kind, pubkey, d-tag): a fresh key per
    process would orphan every previous listing on the relay instead of
    replacing it, and the relay would accumulate dead listings forever. Any
    long-lived publisher (meshd) MUST load its event key from disk — same
    contract as nakshatra_auth.load_or_create_worker_key for the mesh key.
    """
    import os
    from pathlib import Path

    p = Path(path).expanduser()
    if p.exists():
        key = p.read_text().strip()
        pubkey_of(key)  # validates hex + curve point before anyone signs with it
        return key
    p.parent.mkdir(parents=True, exist_ok=True)
    key = keygen()[0]
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")
    return key


def _xonly_hex(pk: PrivateKey) -> str:
    # BIP340 x-only pubkey = the 32-byte x coordinate (drop the compressed prefix).
    return pk.public_key.format(compressed=True)[1:].hex()


def event_id(pubkey_hex: str, created_at: int, kind: int,
             tags: list, content: str) -> str:
    """NIP-01 event id: sha256 of the canonical [0,pubkey,created_at,kind,tags,
    content] serialization (no whitespace, UTF-8, non-ASCII preserved)."""
    payload = json.dumps([0, pubkey_hex, int(created_at), int(kind), tags, content],
                         separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_event(privkey_hex: str, kind: int, content: str, tags: list,
                created_at: int) -> dict:
    """Build + Schnorr-sign a NIP-01 event."""
    pk = PrivateKey(bytes.fromhex(privkey_hex))
    pub_hex = _xonly_hex(pk)
    eid = event_id(pub_hex, created_at, kind, tags, content)
    sig = pk.sign_schnorr(bytes.fromhex(eid)).hex()
    return {"id": eid, "pubkey": pub_hex, "created_at": int(created_at),
            "kind": int(kind), "tags": tags, "content": content, "sig": sig}


def verify_event(ev: dict) -> bool:
    """True iff the event id matches its fields AND the Schnorr signature is valid
    for the advertised pubkey. Never raises."""
    try:
        recomputed = event_id(ev["pubkey"], ev["created_at"], ev["kind"],
                              ev["tags"], ev["content"])
        if recomputed != ev["id"]:
            return False
        return PublicKeyXOnly(bytes.fromhex(ev["pubkey"])).verify(
            bytes.fromhex(ev["sig"]), bytes.fromhex(ev["id"]))
    except (KeyError, ValueError, TypeError, Exception):
        return False
