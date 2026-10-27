"""Strict, node-signed D8.1 introduction statements.

An introduction is contact-carried data, not authority.  The node signature lets both recipients
check that the same introducer authored the immutable statement; it never substitutes for the
third person's existing person-signed contact acceptance.
"""
from __future__ import annotations

import secrets
import json
import unicodedata

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

KIND = "nak.intro.v1"
VERSION = 1
DEFAULT_TTL_S = 7 * 86400
MAX_TTL_S = 30 * 86400
MAX_NOTE = 500
MAX_NAME = 40
MAX_CLOCK_SKEW_S = 300

_STATEMENT_KEYS = {
    "v", "kind", "introducer", "to", "about", "note", "issued_at", "expires_at", "intro_id",
}
_ABOUT_KEYS = {"person", "node", "display_name"}
_WRAPPER_KEYS = {"statement", "signature"}


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _hex(value, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(
            char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be {length} lowercase hex characters")
    return value


def strict_text(value, label: str, maximum: int, *, empty: bool = True) -> str:
    """Accept visible text without invisible controls or ambiguous whitespace."""
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value):
        raise ValueError(f"{label} must be {'1-' if not empty else '0-'}{maximum} characters")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in value):
        raise ValueError(f"{label} contains a control or invisible formatting character")
    if value and (value[0].isspace() or value[-1].isspace()):
        raise ValueError(f"{label} must not have leading or trailing whitespace")
    if any(left.isspace() and right.isspace() for left, right in zip(value, value[1:])):
        raise ValueError(f"{label} must not contain doubled whitespace")
    return value


def build_statement(*, introducer: str, to: str, about_person: str, about_node: str,
                    display_name: str, note: str = "", issued_at: int,
                    expires_at: int | None = None, intro_id: str | None = None) -> dict:
    statement = {
        "v": VERSION,
        "kind": KIND,
        "introducer": introducer,
        "to": to,
        "about": {
            "person": about_person,
            # The public node key is necessary for B to open the ordinary pairwise session on
            # which C's existing contact.accept is delivered.  It conveys no messaging authority.
            "node": about_node,
            "display_name": display_name,
        },
        "note": note,
        "issued_at": issued_at,
        "expires_at": issued_at + DEFAULT_TTL_S if expires_at is None else expires_at,
        "intro_id": intro_id or secrets.token_hex(16),
    }
    return validate_statement(statement, now=issued_at, allow_expired=True)


def validate_statement(value, *, now: int, allow_expired: bool = False) -> dict:
    if not isinstance(value, dict) or set(value) != _STATEMENT_KEYS:
        raise ValueError("introduction statement has unknown or missing fields")
    if type(value.get("v")) is not int or value["v"] != VERSION or value.get("kind") != KIND:
        raise ValueError("unknown introduction kind or version")
    introducer = _hex(value.get("introducer"), 64, "introducer")
    to = _hex(value.get("to"), 64, "introduction recipient")
    about = value.get("about")
    if not isinstance(about, dict) or set(about) != _ABOUT_KEYS:
        raise ValueError("introduction about field has unknown or missing fields")
    about_person = _hex(about.get("person"), 64, "introduced person")
    _hex(about.get("node"), 64, "introduced node")
    strict_text(about.get("display_name"), "display name", MAX_NAME, empty=False)
    strict_text(value.get("note"), "note", MAX_NOTE)
    _hex(value.get("intro_id"), 32, "intro_id")
    issued, expires = value.get("issued_at"), value.get("expires_at")
    if type(issued) is not int or type(expires) is not int:
        raise ValueError("introduction timestamps must be integers")
    if expires <= issued or expires - issued > MAX_TTL_S:
        raise ValueError("introduction lifetime must be positive and at most 30 days")
    if issued > int(now) + MAX_CLOCK_SKEW_S:
        raise ValueError("introduction was issued too far in the future")
    if not allow_expired and expires <= int(now):
        raise ValueError("introduction expired")
    if len({introducer, to, about_person}) != 3:
        raise ValueError("introducer, recipient, and introduced person must be different")
    return {
        **value,
        "about": dict(about),
    }


def sign(private_key: ed25519.Ed25519PrivateKey, statement: dict) -> dict:
    validated = validate_statement(statement, now=statement.get("issued_at", 0), allow_expired=True)
    if not isinstance(private_key, ed25519.Ed25519PrivateKey):
        raise TypeError("introduction signer must be the node's Ed25519 private key")
    return {"statement": validated, "signature": private_key.sign(_canonical(validated)).hex()}


def verify(value, node_public: str, *, now: int, allow_expired: bool = False) -> dict:
    if not isinstance(value, dict) or set(value) != _WRAPPER_KEYS:
        raise ValueError("signed introduction has unknown or missing fields")
    statement = validate_statement(value.get("statement"), now=now, allow_expired=allow_expired)
    signature = _hex(value.get("signature"), 128, "introduction signature")
    _hex(node_public, 64, "introducer node")
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(node_public)).verify(
            bytes.fromhex(signature), _canonical(statement))
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise ValueError("introduction node signature does not verify") from exc
    return statement


def same(left: dict, right: dict) -> bool:
    """Compare a signed wrapper byte-semantically, independent of dict insertion order."""
    try:
        return _canonical(left) == _canonical(right)
    except (TypeError, ValueError):
        return False
