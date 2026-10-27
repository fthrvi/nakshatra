"""Node-signed, contact-only statements about person-key custody.

The claim is deliberately self-attested: a known contact node reports how its local
person authority is configured.  It is not a person-key assertion and must never be
presented as one.
"""
from __future__ import annotations

import json
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


VERSION = 1
KIND = "nak.custody.v1"
MODES = {"file", "broker-biometric"}
MAX_TTL_S = 30 * 86400
REFRESH_BEFORE_S = 7 * 86400
MAX_FUTURE_SKEW_S = 300
_FIELDS = {"v", "kind", "person", "mode", "since", "issued_at", "expires_at", "sig"}


def canonical(statement: dict) -> bytes:
    """Canonical bytes signed by the reporting node (the ``sig`` field is excluded)."""
    body = {key: value for key, value in statement.items() if key != "sig"}
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def issue(person: str, mode: str, node_key: ed25519.Ed25519PrivateKey, *,
          since: int, issued_at: int | None = None, expires_at: int | None = None) -> dict:
    """Create one strict v1 statement signed only by the node key."""
    now = int(time.time() if issued_at is None else issued_at)
    expiry = now + MAX_TTL_S if expires_at is None else expires_at
    statement = {
        "v": VERSION,
        "kind": KIND,
        "person": person,
        "mode": mode,
        "since": since,
        "issued_at": now,
        "expires_at": expiry,
    }
    _validate_shape(statement, now=now, require_sig=False)
    statement["sig"] = node_key.sign(canonical(statement)).hex()
    return statement


def _hex_key(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def _validate_shape(statement: object, *, now: int, require_sig: bool = True,
                    allow_expired: bool = False) -> dict:
    if not isinstance(statement, dict):
        raise ValueError("custody statement is not an object")
    expected = _FIELDS if require_sig else _FIELDS - {"sig"}
    if set(statement) != expected:
        raise ValueError("custody statement fields are missing or unknown")
    if statement.get("v") != VERSION:
        raise ValueError("unsupported custody statement version")
    if statement.get("kind") != KIND:
        raise ValueError("unknown custody statement kind")
    if not _hex_key(statement.get("person")):
        raise ValueError("custody statement person is malformed")
    if statement.get("mode") not in MODES:
        raise ValueError("custody statement mode is unsupported")
    for field in ("since", "issued_at", "expires_at"):
        if type(statement.get(field)) is not int or statement[field] < 0:
            raise ValueError(f"custody statement {field} is malformed")
    if statement["since"] > statement["issued_at"]:
        raise ValueError("custody statement since is after issuance")
    if statement["issued_at"] > now + MAX_FUTURE_SKEW_S:
        raise ValueError("custody statement issuance is in the future")
    if statement["expires_at"] <= statement["issued_at"]:
        raise ValueError("custody statement expiry is not after issuance")
    if statement["expires_at"] - statement["issued_at"] > MAX_TTL_S:
        raise ValueError("custody statement lifetime exceeds 30 days")
    if not allow_expired and statement["expires_at"] <= now:
        raise ValueError("custody statement expired")
    if require_sig:
        sig = statement.get("sig")
        if not isinstance(sig, str) or len(sig) != 128 or any(
                char not in "0123456789abcdef" for char in sig):
            raise ValueError("custody statement signature is malformed")
    return statement


def verify(statement: object, *, expected_person: str, expected_node: str,
           now: int | None = None, allow_expired: bool = False) -> dict:
    """Verify a statement against the already-pinned contact person and node keys."""
    checked = _validate_shape(
        statement, now=int(time.time() if now is None else now), allow_expired=allow_expired)
    if checked["person"] != expected_person:
        raise ValueError("custody statement person is not this contact")
    if not _hex_key(expected_node):
        raise ValueError("contact node key is malformed")
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(expected_node)).verify(
            bytes.fromhex(checked["sig"]), canonical(checked)
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise ValueError("custody statement signature does not verify") from exc
    return dict(checked)


def display(statement: object, *, now: int | None = None) -> str:
    """Safe UI text. Any absent, expired, corrupt, or unverifiable value is unknown."""
    current = int(time.time() if now is None else now)
    try:
        checked = _validate_shape(statement, now=current)
        age = max(0, current - checked["issued_at"])
        if age < 3600:
            ago = "just now"
        elif age < 86400:
            hours = age // 3600
            ago = f"{hours}h ago"
        else:
            days = age // 86400
            ago = f"{days}d ago"
        mode = "phone approval" if checked["mode"] == "broker-biometric" else "file on disk"
        return f"{mode} (self-reported by their node, {ago})"
    except Exception:  # Stored/contact-controlled data must never break a listing.
        return "unknown"
