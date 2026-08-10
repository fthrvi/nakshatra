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
    long-lived publisher (meshd) MUST load its event key from disk.

    Full parity with nakshatra_auth.load_or_create_worker_key: a malformed /
    empty / truncated file REGENERATES (never raises — a half-written key from
    an interrupted create would otherwise brick a Restart=always daemon into a
    hot-loop); the write is atomic (tmp + os.replace, like FileRelay.publish);
    the parent dir is 0700; a loose-perm existing file is tightened to 0600 in
    place; a create race (two starts, O_EXCL loser) re-reads the winner's key.
    """
    import os
    import stat
    import sys
    from pathlib import Path

    p = Path(path).expanduser()

    def _atomic_write(key: str, *, exclusive: bool) -> str:
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, (key + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        if exclusive:
            # creating from absent: O_EXCL semantics via hardlink — refuse to
            # clobber a key a concurrent start just wrote.
            try:
                os.link(tmp, p)
            except FileExistsError:
                os.unlink(tmp)
                return load_or_create_key(p)   # a concurrent start won — adopt its key
            os.unlink(tmp)
        else:
            # overwriting a corrupt file: last-writer-wins is fine (both write
            # valid keys; a loser just re-reads the winner's on next start).
            os.replace(tmp, p)
        return key

    if p.exists():
        raw = p.read_text().strip()
        try:
            pubkey_of(raw)  # validates hex + curve point before anyone signs with it
        except Exception:
            print(f"[nostr] event key at {p} is malformed/empty; regenerating",
                  file=sys.stderr, flush=True)
            return _atomic_write(keygen()[0], exclusive=False)
        mode = stat.S_IMODE(p.stat().st_mode)
        if mode & 0o077:  # group/other bits set — tighten, don't trust a loose key file
            os.chmod(p, 0o600)
            print(f"[nostr] tightened {p} perms {oct(mode)} → 0600",
                  file=sys.stderr, flush=True)
        return raw

    return _atomic_write(keygen()[0], exclusive=True)


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
