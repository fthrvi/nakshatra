import base64
import json


def encode_join(coordinator: str, token: str, *, relay: str = "", expires_at: int = 0) -> str:
    obj = {
        "c": coordinator,
        "t": token,
    }
    if relay:
        obj["r"] = relay
    if expires_at:
        obj["x"] = expires_at
    payload = json.dumps(obj, separators=(",", ":"))
    # base64url encoding without padding
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    return encoded


def decode_join(code: str, *, now: int) -> dict:
    # Validate base64url
    try:
        # Add padding back if needed
        padding = 4 - (len(code) % 4)
        if padding != 4:
            code += "=" * padding
        decoded = base64.urlsafe_b64decode(code)
        payload = decoded.decode("utf-8")
    except Exception:
        raise ValueError("invalid base64url encoding")

    # Parse JSON
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise ValueError("invalid JSON")

    # Must be a JSON object
    if not isinstance(data, dict):
        raise ValueError("payload is not a JSON object")

    # Check required keys
    if "c" not in data:
        raise ValueError("coordinator missing")
    if "t" not in data:
        raise ValueError("token missing")

    coordinator = data["c"]
    token = data["t"]

    # Validate coordinator is a non-empty string
    if not isinstance(coordinator, str) or not coordinator:
        raise ValueError("coordinator missing or invalid")

    # Validate token is a non-empty string
    if not isinstance(token, str) or not token:
        raise ValueError("token missing or invalid")

    # Coordinator must start with http:// or https://
    if not (coordinator.startswith("http://") or coordinator.startswith("https://")):
        raise ValueError("coordinator must start with http:// or https://")

    # Check for unknown keys
    allowed_keys = {"c", "t", "r", "x"}
    for key in data:
        if key not in allowed_keys:
            raise ValueError(f"unknown key in code: {key}")

    # Handle relay (optional)
    relay = data.get("r", "")

    # Handle expiry
    expires_at = data.get("x", 0)
    if expires_at and now >= expires_at:
        raise ValueError("code expired")

    return {
        "coordinator": coordinator,
        "token": token,
        "relay": relay,
        "expires_at": expires_at,
    }

# ── signed network invites (2026-10-02, Nakshatra network hackathon slice) ──────────────────────
#
# The code above is the legacy coordinator join code: a bearer token, so whoever holds it can use
# it. A NETWORK INVITE is different: it is signed by the inviter's PERSON key, names the inviter's
# node and the peer addresses to dial first (so a new node needs no directory), carries a one-time
# nonce, and expires. Encoding: "nki1." + base64url(canonical JSON). Canonical JSON and the
# {alg, keyid, value} signature block match Sthambha's signer and trisul's cap_guard.

import hashlib
import secrets
import sqlite3
import threading
import time as _time

INVITE_PREFIX = "nki1."
_INVITE_KEYS = {"v", "inviter", "inviter_node", "peers", "nonce", "expires_at", "scope", "note", "ik", "release", "sig"}
_MAX_INVITE_LEN = 8192


def _canon(obj: dict) -> bytes:
    body = {k: v for k, v in obj.items() if k != "sig"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _unb64u(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def encode_invite(person_priv, inviter_node: str, peers: list, *, ttl_s: int = 86400,
                  scope: str = "member", note: str = "", now: int | None = None,
                  release: dict | None = None) -> str:
    """person_priv: a cryptography Ed25519PrivateKey (the inviter's person key).
    peers: [{"node": <node pub hex>, "addrs": ["host:port", ...]}, ...] to dial first."""
    now = int(now if now is not None else _time.time())
    inviter = person_priv.public_key().public_bytes_raw().hex()
    # ik: a ONE-TIME invite key (Ed25519 private, hex). Whoever holds the invite uses it to complete the
    # encrypted handshake with the inviter's node before the inviter knows their real key. It makes the
    # invite a secret: share it only with the person you are inviting. Redeeming it only ASKS to
    # connect; nothing is shared until the inviter accepts.
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed
    ik = _ed.Ed25519PrivateKey.generate().private_bytes_raw().hex()
    inv = {"v": 1, "inviter": inviter, "inviter_node": inviter_node, "peers": peers,
           "nonce": secrets.token_hex(16), "expires_at": now + int(ttl_s), "scope": scope, "ik": ik}
    if note:
        inv["note"] = note
    if release:
        # Where a NEW node gets Nakshatra and which release key it must pin. Signed with the rest of the
        # invite, so a newcomer trusts the release key because they trust the friend who invited them.
        inv["release"] = {k: str(release[k]) for k in ("url", "channel", "pubkey", "version")}
    inv["sig"] = {"alg": "Ed25519", "keyid": inviter[:16],
                  "value": base64.b64encode(person_priv.sign(_canon(inv))).decode("ascii")}
    return INVITE_PREFIX + _b64u(json.dumps(inv, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False).encode("utf-8"))


def decode_invite(code: str, *, now: int | None = None, trusted_inviters: set | None = None) -> dict:
    """Verify and return an invite. Raises ValueError with a plain reason on any problem.
    trusted_inviters: if given, the inviter's person key must be in it."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import ed25519
    now = int(now if now is not None else _time.time())
    if not isinstance(code, str) or not code.startswith(INVITE_PREFIX):
        raise ValueError("not a network invite")
    if len(code) > _MAX_INVITE_LEN:
        raise ValueError("invite too long")
    try:
        inv = json.loads(_unb64u(code[len(INVITE_PREFIX):]).decode("utf-8"))
    except Exception:
        raise ValueError("invite is not valid base64url JSON")
    if not isinstance(inv, dict) or set(inv) - _INVITE_KEYS:
        raise ValueError("invite has unknown fields")
    for k in ("inviter", "inviter_node", "nonce", "ik"):
        v = inv.get(k)
        if not isinstance(v, str) or not all(c in "0123456789abcdef" for c in v) or len(v) not in (32, 64):
            raise ValueError(f"invite field {k} is malformed")
    if inv.get("v") != 1:
        raise ValueError("unsupported invite version")
    peers = inv.get("peers")
    if not isinstance(peers, list) or not peers or len(peers) > 16:
        raise ValueError("invite must list 1-16 peers")
    for p in peers:
        if not isinstance(p, dict) or not isinstance(p.get("node"), str) or not isinstance(p.get("addrs"), list):
            raise ValueError("invite peer entry is malformed")
    rel = inv.get("release")
    if rel is not None:
        if not isinstance(rel, dict) or set(rel) != {"url", "channel", "pubkey", "version"} or \
                not all(isinstance(v, str) for v in rel.values()) or \
                not rel["url"].startswith(("http://", "https://")) or \
                len(rel["pubkey"]) != 64 or not all(c in "0123456789abcdef" for c in rel["pubkey"]):
            raise ValueError("invite release entry is malformed")
    sig = inv.get("sig") or {}
    try:
        if sig.get("alg") != "Ed25519":
            raise InvalidSignature
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(inv["inviter"])).verify(
            base64.b64decode(sig["value"]), _canon(inv))
    except (InvalidSignature, KeyError, ValueError, TypeError):
        raise ValueError("invite signature does not verify")
    if not isinstance(inv.get("expires_at"), int) or now >= inv["expires_at"]:
        raise ValueError("invite expired")
    if trusted_inviters is not None and inv["inviter"] not in trusted_inviters:
        raise ValueError("invite is from an inviter this node does not trust")
    return inv


class InviteBook:
    """Single-use enforcement on the inviter's side: consume() records the nonce in the same
    sqlite transaction that checks it, so two redemptions of one invite can never both succeed."""

    def __init__(self, path):
        self._db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS consumed (nonce TEXT PRIMARY KEY, invitee TEXT, ts INTEGER)")
        self._lock = threading.Lock()

    def consume(self, inv: dict, invitee_node: str, now: int | None = None) -> bool:
        now = int(now if now is not None else _time.time())
        with self._lock:
            try:
                self._db.execute("INSERT INTO consumed VALUES (?,?,?)", (inv["nonce"], invitee_node, now))
                return True
            except sqlite3.IntegrityError:
                return False


def invite_pub(inv: dict) -> str:
    """Public half of the invite's one-time key: what the inviter's node pins while it waits."""
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed
    return _ed.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(inv["ik"])).public_key().public_bytes_raw().hex()


def invite_rendezvous(inv: dict) -> bytes:
    """Where the inviter waits on the relay for this invite (16 bytes, derived from the nonce)."""
    return hashlib.sha256(b"nak-invite-v1|" + inv["nonce"].encode()).digest()[:16]
