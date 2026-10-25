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
import re
import secrets
import sqlite3
import threading
import time as _time
from urllib.parse import urlsplit

INVITE_PREFIX = "nki1."
ADDRESS_PREFIX = "nka1."
_INVITE_KEYS = {"v", "inviter", "inviter_node", "peers", "nonce", "expires_at", "scope", "note", "ik", "release",
                "from_name", "kind", "worker", "consumer", "sig"}
_ADDRESS_KEYS = {"v", "kind", "inviter", "inviter_node", "peers", "aid", "ik", "expires_at",
                 "from_name", "release", "sig"}
_MAX_INVITE_LEN = 8192
_MAX_ADDRESS_LEN = 8192
_MAX_ADDRESS_TTL_S = 365 * 86400
_WORKER_TIERS = {"stranger", "known", "trusted"}
_MAX_STARTER = 100_000_000


def _canon(obj: dict) -> bytes:
    body = {k: v for k, v in obj.items() if k != "sig"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _unb64u(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _valid_worker_redeem_url(value: object) -> bool:
    """Return whether *value* is an HTTP(S) URL with a usable host and optional port."""
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port  # Access validates numeric syntax and the 1-65535 upper bound.
    except ValueError:
        return False
    authority = parsed.netloc.rsplit("@", 1)[-1]
    return parsed.scheme in {"http", "https"} and bool(host) and \
        not authority.endswith(":") and (port is None or 1 <= port <= 65535)


def _validate_invite_kind(inv: dict) -> None:
    """Validate the U4 discriminator while treating pre-U4 invites as contacts."""
    kind = inv.get("kind", "contact")
    if kind not in {"contact", "worker", "consumer"}:
        raise ValueError("invite kind is unsupported")
    if kind == "contact":
        if "worker" in inv:
            raise ValueError("contact invite must not contain worker details")
        if "consumer" in inv:
            raise ValueError("contact invite must not contain consumer details")
        return
    detail_key = kind
    other_key = "consumer" if kind == "worker" else "worker"
    if other_key in inv:
        raise ValueError(f"{kind} invite must not contain {other_key} details")
    details = inv.get(detail_key)
    expected = {"redeem", "code", "tier"} | ({"starter"} if kind == "consumer" else set())
    if not isinstance(details, dict) or set(details) != expected:
        raise ValueError(f"{kind} invite details are malformed")
    redeem, code, tier = details.get("redeem"), details.get("code"), details.get("tier")
    if not _valid_worker_redeem_url(redeem):
        raise ValueError(f"{kind} redeem URL must be http(s)")
    if (not isinstance(code, str) or not 1 <= len(code) <= 64
            or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for ch in code)):
        raise ValueError(f"{kind} invite code is malformed")
    if tier not in _WORKER_TIERS:
        raise ValueError(f"{kind} invite tier is malformed")
    if kind == "consumer" and (type(details.get("starter")) is not int or
                               not 0 <= details["starter"] <= _MAX_STARTER):
        raise ValueError("consumer starter is malformed")


def encode_invite(person_priv, inviter_node: str, peers: list, *, ttl_s: int = 86400,
                  scope: str = "member", note: str = "", now: int | None = None,
                  release: dict | None = None, from_name: str = "", kind: str = "contact",
                  worker: dict | None = None, consumer: dict | None = None) -> str:
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
    if from_name:
        inv["from_name"] = str(from_name)[:40]        # what the inviter calls themselves (self-chosen)
    if release:
        # Where a NEW node gets Nakshatra and which release key it must pin. Signed with the rest of the
        # invite, so a newcomer trusts the release key because they trust the friend who invited them.
        inv["release"] = {k: str(release[k]) for k in ("url", "channel", "pubkey", "version")}
    if kind != "contact":
        inv["kind"] = kind
    if worker is not None:
        inv["worker"] = dict(worker)
    if consumer is not None:
        inv["consumer"] = dict(consumer)
    _validate_invite_kind(inv)
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
    _validate_invite_kind(inv)
    peers = inv.get("peers")
    if not isinstance(peers, list) or not peers or len(peers) > 16:
        raise ValueError("invite must list 1-16 peers")
    for p in peers:
        if not isinstance(p, dict) or not isinstance(p.get("node"), str) or not isinstance(p.get("addrs"), list):
            raise ValueError("invite peer entry is malformed")
    if "from_name" in inv and (not isinstance(inv["from_name"], str) or len(inv["from_name"]) > 40):
        raise ValueError("invite from_name is malformed")
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


def _validate_address_release(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"url", "channel", "pubkey", "version"}:
        raise ValueError("address release entry is malformed")
    if not all(isinstance(v, str) and v for v in value.values()):
        raise ValueError("address release entry is malformed")
    try:
        parsed = urlsplit(value["url"])
        port = parsed.port
    except ValueError as exc:
        raise ValueError("address release URL is malformed") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or value["url"].endswith("/")
            or (port is not None and not 1 <= port <= 65535)
            or not re.fullmatch(r"[A-Za-z0-9:/._~%\-\[\]]+", value["url"])):
        raise ValueError("address release URL is malformed")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", value["channel"]) or \
            not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", value["version"]):
        raise ValueError("address release channel or version is malformed")
    if len(value["pubkey"]) != 64 or any(c not in "0123456789abcdef" for c in value["pubkey"]):
        raise ValueError("address release key is malformed")


def _validate_address_peers(peers: object) -> None:
    if not isinstance(peers, list) or not peers or len(peers) > 16:
        raise ValueError("address must list 1-16 peers")
    for peer in peers:
        if not isinstance(peer, dict) or set(peer) != {"node", "addrs"}:
            raise ValueError("address peer entry is malformed")
        node, addrs = peer.get("node"), peer.get("addrs")
        if not isinstance(node, str) or len(node) != 64 or any(c not in "0123456789abcdef" for c in node):
            raise ValueError("address peer node is malformed")
        if not isinstance(addrs, list) or len(addrs) > 16 or not all(
                isinstance(addr, str) and 0 < len(addr) <= 512 for addr in addrs):
            raise ValueError("address peer dial hints are malformed")


def build_unsigned_address(person_pub: str, inviter_node: str, peers: list, *, days: int = 90,
                           now: int | None = None, from_name: str = "", release: dict) -> dict:
    """Build the fields of a public request address before its person signature.

    ``ik`` is deliberately published. It authenticates only the rendezvous endpoint; the signed
    contact.request envelope authenticates each stranger. Possession therefore permits an ASK,
    never messaging or contact access.
    """
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 365:
        raise ValueError("address lifetime must be 1-365 days")
    if not isinstance(inviter_node, str) or len(inviter_node) != 64 or \
            any(c not in "0123456789abcdef" for c in inviter_node):
        raise ValueError("address inviter_node is malformed")
    _validate_address_peers(peers)
    _validate_address_release(release)
    if not isinstance(from_name, str) or len(from_name) > 40:
        raise ValueError("address from_name is malformed")
    if not isinstance(person_pub, str) or len(person_pub) != 64 or \
            any(c not in "0123456789abcdef" for c in person_pub):
        raise ValueError("address inviter is malformed")
    now = int(now if now is not None else _time.time())
    address = {
        "v": 1, "kind": "address", "inviter": person_pub, "inviter_node": inviter_node,
        "peers": peers, "aid": secrets.token_hex(16),
        # Public seed by design: it pins the listener but conveys no contact authority.
        "ik": _ed.Ed25519PrivateKey.generate().private_bytes_raw().hex(),
        "expires_at": now + days * 86400, "release": dict(release),
    }
    if from_name:
        address["from_name"] = from_name
    return address


def attach_address_signature(address: dict, signature: bytes) -> str:
    """Encode an unsigned address with exactly the supplied raw Ed25519 signature."""
    if not isinstance(address, dict) or "sig" in address:
        raise ValueError("unsigned address is malformed")
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ValueError("address signer did not return a 64-byte signature")
    signed = dict(address)
    inviter = signed.get("inviter", "")
    signed["sig"] = {"alg": "Ed25519", "keyid": inviter[:16],
                     "value": base64.b64encode(signature).decode("ascii")}
    raw = json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return ADDRESS_PREFIX + _b64u(raw)


def encode_address(person_priv, inviter_node: str, peers: list, *, days: int = 90,
                   now: int | None = None, from_name: str = "", release: dict) -> str:
    """Mint a public, reusable request address signed by the person's key."""
    person_pub = person_priv.public_key().public_bytes_raw().hex()
    address = build_unsigned_address(
        person_pub, inviter_node, peers, days=days, now=now,
        from_name=from_name, release=release)
    return attach_address_signature(address, person_priv.sign(_canon(address)))


def decode_address(code: str, *, now: int | None = None,
                   trusted_inviters: set | None = None) -> dict:
    """Verify a public request address and return its signed fields."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import ed25519
    now = int(now if now is not None else _time.time())
    if not isinstance(code, str) or not code.startswith(ADDRESS_PREFIX):
        raise ValueError("not a Nakshatra request address")
    if len(code) > _MAX_ADDRESS_LEN:
        raise ValueError("address too long")
    encoded = code[len(ADDRESS_PREFIX):]
    if not encoded or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for c in encoded):
        raise ValueError("address is not valid base64url JSON")
    try:
        raw = _unb64u(encoded)
        address = json.loads(raw.decode("utf-8"))
    except Exception:
        raise ValueError("address is not valid base64url JSON")
    if not isinstance(address, dict) or set(address) != _ADDRESS_KEYS - {"from_name"} and \
            set(address) != _ADDRESS_KEYS:
        raise ValueError("address fields are missing or unknown")
    canonical = json.dumps(address, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if raw != canonical:
        raise ValueError("address encoding is not canonical")
    if address.get("v") != 1:
        raise ValueError("unsupported address version")
    if address.get("kind") != "address":
        raise ValueError("address kind is not address")
    for key, length in (("inviter", 64), ("inviter_node", 64), ("aid", 32), ("ik", 64)):
        value = address.get(key)
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"address field {key} is malformed")
    _validate_address_peers(address.get("peers"))
    _validate_address_release(address.get("release"))
    if "from_name" in address and (not isinstance(address["from_name"], str) or
                                    len(address["from_name"]) > 40):
        raise ValueError("address from_name is malformed")
    sig = address.get("sig") or {}
    try:
        if set(sig) != {"alg", "keyid", "value"} or sig.get("alg") != "Ed25519" or \
                sig.get("keyid") != address["inviter"][:16]:
            raise InvalidSignature
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(address["inviter"])).verify(
            base64.b64decode(sig["value"], validate=True), _canon(address))
    except (InvalidSignature, KeyError, ValueError, TypeError):
        raise ValueError("address signature does not verify")
    expires_at = address.get("expires_at")
    if isinstance(expires_at, bool) or not isinstance(expires_at, int) or now >= expires_at:
        raise ValueError("address expired")
    if expires_at > now + _MAX_ADDRESS_TTL_S:
        raise ValueError("address expiry is more than 365 days away")
    if trusted_inviters is not None and address["inviter"] not in trusted_inviters:
        raise ValueError("address is from a person this node does not trust")
    return address


def address_pub(address: dict) -> str:
    """Public half of the intentionally-public address rendezvous key."""
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed
    return _ed.Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(address["ik"])).public_key().public_bytes_raw().hex()


def address_rendezvous(address: dict) -> bytes:
    return hashlib.sha256(b"nak-address-v1|" + address["aid"].encode()).digest()[:16]


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
