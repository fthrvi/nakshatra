"""Person-signed, cumulative agent-grant revocation records."""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import stat
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519
from sthambha.signer import canonical

SCHEMA = "nak.revocations.v1"
PREFIX = "nkr1."
MAX_GRANTS = 4096
MAX_BLOB = 256 * 1024
MAX_SEQ = 2**53 - 1
_PERSON_RE = re.compile(r"[0-9a-f]{64}\Z")
_GRANT_RE = re.compile(r"[0-9a-f]{16}\Z")
_SIG_RE = re.compile(r"[0-9a-f]{128}\Z")
_FIELDS = {"schema", "person", "seq", "issued_at", "grant_ids", "sig"}


def read_private_json(path: Path, max_bytes: int = MAX_BLOB) -> dict | None:
    """Read a small owner-only JSON object without following links.

    Missing files return ``None``. Every other filesystem or format defect is a
    ``ValueError`` so callers can choose whether it is fatal (the signing CLI) or
    merely worth logging (daemon startup).
    """
    try:
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) |
                     getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(str(exc)) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("must be a regular file")
        if info.st_uid != os.getuid():
            raise ValueError("must be owned by the current user")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("mode must be 0600")
        if info.st_size > max_bytes:
            raise ValueError(f"exceeds the {max_bytes}-byte size limit")
        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise ValueError(f"exceeds the {max_bytes}-byte size limit")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("must contain a JSON object")
        return value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(str(exc)) from exc
    finally:
        os.close(fd)


def revocation_statement(record: dict) -> bytes:
    """Return the exact canonical bytes covered by the person's signature."""
    if not isinstance(record, dict):
        raise ValueError("revocation record must be an object")
    return canonical(record)


def _validate_unsigned(record: dict, person_hex: str) -> None:
    if not isinstance(record, dict):
        raise ValueError("revocation record must be an object")
    if set(record) != _FIELDS:
        raise ValueError("revocation record has missing or extra fields")
    if record.get("schema") != SCHEMA:
        raise ValueError("wrong revocation schema")
    if not isinstance(person_hex, str) or _PERSON_RE.fullmatch(person_hex) is None:
        raise ValueError("person key must be 64 lowercase hex characters")
    if record.get("person") != person_hex:
        raise ValueError("revocation record is for another person")
    if type(record.get("seq")) is not int or not 1 <= record["seq"] <= MAX_SEQ:
        raise ValueError(f"revocation seq must be an integer from 1 through {MAX_SEQ}")
    if type(record.get("issued_at")) is not int:
        raise ValueError("revocation issued_at must be an integer")
    grant_ids = record.get("grant_ids")
    if not isinstance(grant_ids, list):
        raise ValueError("revocation grant_ids must be a list")
    if len(grant_ids) > MAX_GRANTS:
        raise ValueError(f"revocation record exceeds {MAX_GRANTS} grant ids")
    if any(not isinstance(gid, str) or _GRANT_RE.fullmatch(gid) is None for gid in grant_ids):
        raise ValueError("each revoked grant id must be 16 lowercase hex characters")
    if grant_ids != sorted(set(grant_ids)):
        raise ValueError("revoked grant ids must be sorted and unique")
    if not isinstance(record.get("sig"), str) or _SIG_RE.fullmatch(record["sig"]) is None:
        raise ValueError("revocation signature must be 128 lowercase hex characters")


def make_record(sign_fn, person_hex: str, seq: int, grant_ids, now: int) -> dict:
    """Build and sign one normalized cumulative record."""
    ids = sorted(set(grant_ids)) if isinstance(grant_ids, (list, tuple, set, frozenset)) else grant_ids
    record = {"schema": SCHEMA, "person": person_hex, "seq": seq,
              "issued_at": now, "grant_ids": ids, "sig": "0" * 128}
    _validate_unsigned(record, person_hex)
    unsigned = {key: value for key, value in record.items() if key != "sig"}
    try:
        signature = sign_fn(revocation_statement(unsigned))
    except Exception as exc:  # keep the public API consistently ValueError-based
        raise ValueError(f"could not sign revocation record: {exc}") from exc
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ValueError("revocation signer did not return a 64-byte signature")
    record["sig"] = signature.hex()
    return record


def verify_record(record: dict, person_hex: str) -> dict:
    """Strictly validate and authenticate a record, returning a shallow copy."""
    _validate_unsigned(record, person_hex)
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(person_hex)).verify(
            bytes.fromhex(record["sig"]), revocation_statement(record))
    except (ValueError, InvalidSignature) as exc:
        raise ValueError("revocation signature does not verify against the person key") from exc
    return dict(record, grant_ids=list(record["grant_ids"]))


def encode_record(record: dict) -> str:
    verified = verify_record(record, record.get("person") if isinstance(record, dict) else "")
    raw = json.dumps(verified, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_record(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("revocation blob must be text")
    try:
        size = len(text.encode("ascii"))
    except UnicodeEncodeError as exc:
        raise ValueError("revocation blob must be ASCII") from exc
    if size > MAX_BLOB:
        raise ValueError("revocation blob exceeds 256 KiB")
    if not text.startswith(PREFIX):
        raise ValueError("not an nkr1 revocation blob")
    payload = text[len(PREFIX):]
    if not payload or "=" in payload:
        raise ValueError("revocation blob has invalid base64")
    try:
        raw = base64.b64decode(payload + "=" * (-len(payload) % 4), altchars=b"-_", validate=True)
        record = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("revocation blob is malformed") from exc
    if not isinstance(record, dict):
        raise ValueError("revocation blob must contain an object")
    verified = verify_record(record, record.get("person", ""))
    if encode_record(verified) != text:
        raise ValueError("revocation blob is not canonically encoded")
    return verified
