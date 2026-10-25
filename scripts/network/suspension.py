"""Person-signed operator suspension state shared with authenticated contacts."""
from __future__ import annotations

import base64
import binascii
import json
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519
from sthambha.signer import canonical

from network.dlp import outbound_problem

SCHEMA = "nak.suspensions.v1"
PREFIX = "nks1."
MAX_NODES = 4096
# Keep a complete record comfortably below nakd's 1 MiB contact-session frame and its
# 384 KiB owner-socket request.  This is the encoded nks1 form; the JSON record sent in a
# ``susp`` frame is smaller still.
MAX_BLOB = 256 * 1024
MAX_SEQ = 2**53 - 1
_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_SIG_RE = re.compile(r"[0-9a-f]{128}\Z")
_FIELDS = {"schema", "person", "seq", "issued_at", "nodes", "reasons", "sig"}


def _encoded_size(record: dict) -> int:
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    # urlsafe_b64encode padding is stripped by encode_record.
    return len(PREFIX) + ((len(raw) * 4 + 2) // 3)


def suspension_statement(record: dict) -> bytes:
    """Return the canonical bytes covered by the operator person's signature."""
    if not isinstance(record, dict):
        raise ValueError("suspension record must be an object")
    return canonical(record)


def _validate(record: dict, person_hex: str) -> None:
    if not isinstance(record, dict):
        raise ValueError("suspension record must be an object")
    if set(record) != _FIELDS:
        raise ValueError("suspension record has missing or extra fields")
    if record.get("schema") != SCHEMA:
        raise ValueError("wrong suspension schema")
    if not isinstance(person_hex, str) or _HEX64_RE.fullmatch(person_hex) is None:
        raise ValueError("person key must be 64 lowercase hex characters")
    if record.get("person") != person_hex:
        raise ValueError("suspension record is for another person")
    if type(record.get("seq")) is not int or not 1 <= record["seq"] <= MAX_SEQ:
        raise ValueError(f"suspension seq must be an integer from 1 through {MAX_SEQ}")
    if type(record.get("issued_at")) is not int:
        raise ValueError("suspension issued_at must be an integer")
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("suspension nodes must be a list")
    if len(nodes) > MAX_NODES:
        raise ValueError(f"suspension record exceeds {MAX_NODES} nodes")
    if any(not isinstance(node, str) or _HEX64_RE.fullmatch(node) is None for node in nodes):
        raise ValueError("each suspended node must be 64 lowercase hex characters")
    if nodes != sorted(set(nodes)):
        raise ValueError("suspended nodes must be sorted and unique")
    reasons = record.get("reasons")
    if not isinstance(reasons, dict) or set(reasons) != set(nodes):
        raise ValueError("suspension reasons must contain exactly the suspended nodes")
    if any(not isinstance(reason, str) or len(reason) > 120 for reason in reasons.values()):
        raise ValueError("each suspension reason must be text of at most 120 characters")
    if any(outbound_problem(reason) for reason in reasons.values()):
        # The record goes to the onboard server and every contact: never carry a secret out.
        raise ValueError("a suspension reason contains a secret-shaped token or encoded blob")
    if not isinstance(record.get("sig"), str) or _SIG_RE.fullmatch(record["sig"]) is None:
        raise ValueError("suspension signature must be 128 lowercase hex characters")
    if _encoded_size(record) > MAX_BLOB:
        raise ValueError(f"suspension record exceeds the {MAX_BLOB // 1024} KiB encoded limit")


def build_unsigned_record(person_hex: str, seq: int, nodes, reasons, now: int) -> dict:
    """Build the normalized placeholder form whose canonical bytes are person-signed."""
    normalized = sorted(set(nodes)) if isinstance(nodes, (list, tuple, set, frozenset)) else nodes
    reason_map = dict(reasons) if isinstance(reasons, dict) else reasons
    record = {"schema": SCHEMA, "person": person_hex, "seq": seq, "issued_at": now,
              "nodes": normalized, "reasons": reason_map, "sig": "0" * 128}
    _validate(record, person_hex)
    return record


def attach_signature(record: dict, signature: bytes) -> dict:
    """Attach one raw Ed25519 signature without changing any signed record field."""
    if not isinstance(record, dict) or record.get("sig") != "0" * 128:
        raise ValueError("unsigned suspension record must carry the zero signature placeholder")
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ValueError("suspension signer did not return a 64-byte signature")
    signed = dict(record, nodes=list(record.get("nodes", [])), reasons=dict(record.get("reasons", {})))
    signed["sig"] = signature.hex()
    _validate(signed, signed.get("person", ""))
    return signed


def make_record(sign_fn, person_hex: str, seq: int, nodes, reasons, now: int) -> dict:
    """Build and sign one normalized full-state suspension record."""
    record = build_unsigned_record(person_hex, seq, nodes, reasons, now)
    try:
        signature = sign_fn(suspension_statement(record))
    except Exception as exc:
        raise ValueError(f"could not sign suspension record: {exc}") from exc
    return attach_signature(record, signature)


def verify_record(record: dict, person_hex: str) -> dict:
    """Strictly validate and authenticate a record, returning an independent copy."""
    _validate(record, person_hex)
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(person_hex)).verify(
            bytes.fromhex(record["sig"]), suspension_statement(record))
    except (ValueError, InvalidSignature) as exc:
        raise ValueError("suspension signature does not verify against the person key") from exc
    return dict(record, nodes=list(record["nodes"]), reasons=dict(record["reasons"]))


def encode_record(record: dict) -> str:
    verified = verify_record(record, record.get("person") if isinstance(record, dict) else "")
    raw = json.dumps(verified, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_record(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("suspension blob must be text")
    try:
        size = len(text.encode("ascii"))
    except UnicodeEncodeError as exc:
        raise ValueError("suspension blob must be ASCII") from exc
    if size > MAX_BLOB:
        raise ValueError(f"suspension blob exceeds the {MAX_BLOB // 1024} KiB encoded limit")
    if not text.startswith(PREFIX):
        raise ValueError("not an nks1 suspension blob")
    payload = text[len(PREFIX):]
    if not payload or "=" in payload:
        raise ValueError("suspension blob has invalid base64")
    try:
        raw = base64.b64decode(payload + "=" * (-len(payload) % 4), altchars=b"-_", validate=True)
        record = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("suspension blob is malformed") from exc
    if not isinstance(record, dict):
        raise ValueError("suspension blob must contain an object")
    verified = verify_record(record, record.get("person", ""))
    if encode_record(verified) != text:
        raise ValueError("suspension blob is not canonically encoded")
    return verified
