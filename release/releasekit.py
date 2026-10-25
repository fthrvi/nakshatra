"""releasekit — shared by build.py (on the builder) and install.py (on every node).

Standard library only, plus `cryptography` WHEN AVAILABLE for Ed25519. install.py must run on a bare
machine (no venv yet), so verification falls back to a tiny pure-Python Ed25519 verify (RFC 8032)
when `cryptography` isn't importable. Signing (build side) always uses `cryptography`.

Manifest encoding = canonical JSON (sorted keys, compact, UTF-8, minus "sig") + {alg, keyid, value},
the same convention as Sthambha's signer, Nakshatra's invites and trisul's cap_guard.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import subprocess

MANIFEST_SCHEMA = "nak-release/1"
KEYROLL_SCHEMA = "nak-release-keyroll/1"
MAX_KEYROLL_LINKS = 8
_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")


def canonical(obj: dict, *, exclude=("sig",)) -> bytes:
    body = {k: v for k, v in obj.items() if k not in exclude}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_keyroll(obj: dict) -> bytes:
    return canonical(obj, exclude=("sig_old", "sig_new"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def public_from_private(priv_hex: str) -> str:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    priv = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
    return priv.public_key().public_bytes_raw().hex()


def _signature(pub_hex: str, raw: bytes) -> dict:
    return {"alg": "Ed25519", "keyid": pub_hex[:16],
            "value": base64.b64encode(raw).decode("ascii")}


def sign_message(message: bytes, priv_hex: str) -> tuple[str, dict]:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    priv = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
    pub_hex = priv.public_key().public_bytes_raw().hex()
    return pub_hex, _signature(pub_hex, priv.sign(message))


def external_sign_message(message: bytes, command: str, pub_hex: str) -> dict:
    """Run an approval broker and locally verify its detached Ed25519 signature."""
    if not isinstance(pub_hex, str) or not _PUBLIC_KEY.fullmatch(pub_hex):
        raise ValueError("external signer public key must be 64 lowercase hex characters")
    argv = shlex.split(command)
    if not argv:
        raise ValueError("external signer command is empty")
    result = subprocess.run(argv, input=message, capture_output=True)
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"external signer exited {result.returncode}" + (f": {detail}" if detail else ""))
    try:
        encoded = result.stdout.decode("ascii").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{128}", encoded):
            raise ValueError
        raw = bytes.fromhex(encoded)
    except (UnicodeDecodeError, ValueError) as e:
        raise RuntimeError("external signer did not return one hex Ed25519 signature") from e
    signature = _signature(pub_hex, raw)
    if not verify_signature(signature, pub_hex, message):
        raise RuntimeError("external signer returned a signature that does not verify")
    return signature


def sign(obj: dict, priv_hex: str) -> dict:
    pub_hex, signature = sign_message(canonical(obj), priv_hex)
    out = dict(obj)
    out["sig"] = signature
    return out


def sign_external(obj: dict, command: str, pub_hex: str) -> dict:
    out = dict(obj)
    out["sig"] = external_sign_message(canonical(obj), command, pub_hex)
    return out


def verify(obj: dict, pub_hex: str) -> bool:
    try:
        sig = obj["sig"]
        msg = canonical(obj)
    except (KeyError, TypeError):
        return False
    return verify_signature(sig, pub_hex, msg)


def verify_signature(sig: dict, pub_hex: str, msg: bytes) -> bool:
    try:
        if not isinstance(sig, dict) or set(sig) != {"alg", "keyid", "value"} \
                or sig.get("alg") != "Ed25519" or sig.get("keyid") != pub_hex[:16]:
            return False
        raw = base64.b64decode(sig["value"], validate=True)
        pub = bytes.fromhex(pub_hex)
        if len(raw) != 64 or len(pub) != 32:
            return False
    except (ValueError, TypeError):
        return False
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(pub).verify(raw, msg)
            return True
        except (InvalidSignature, ValueError):
            return False
    except ImportError:
        return _ed25519_verify_pure(pub, msg, raw)


def sign_keyroll(statement: dict, old_priv_hex: str, new_priv_hex: str) -> dict:
    """Sign one key-roll statement with both old authority and new proof-of-possession."""
    message = canonical_keyroll(statement)
    old_pub, sig_old = sign_message(message, old_priv_hex)
    new_pub, sig_new = sign_message(message, new_priv_hex)
    if statement.get("old_pubkey") != old_pub or statement.get("new_pubkey") != new_pub:
        raise ValueError("key-roll statement public keys do not match its private keys")
    return dict(statement, sig_old=sig_old, sig_new=sig_new)


def validate_keyroll_chain(chain: object) -> list[dict]:
    """Validate a complete, oldest-first key-roll chain and both signatures per link."""
    if not isinstance(chain, list) or not chain or len(chain) > MAX_KEYROLL_LINKS:
        raise ValueError(f"keyroll.json must contain 1..{MAX_KEYROLL_LINKS} links")
    previous = None
    seen = set()
    for index, link in enumerate(chain):
        if not isinstance(link, dict) or set(link) != {
                "schema", "old_pubkey", "new_pubkey", "issued_at", "reason",
                "sig_old", "sig_new"}:
            raise ValueError(f"key-roll link {index} has malformed fields")
        old, new = link["old_pubkey"], link["new_pubkey"]
        if link["schema"] != KEYROLL_SCHEMA:
            raise ValueError(f"key-roll link {index} has an unsupported schema")
        if not isinstance(old, str) or not _PUBLIC_KEY.fullmatch(old) \
                or not isinstance(new, str) or not _PUBLIC_KEY.fullmatch(new) or old == new:
            raise ValueError(f"key-roll link {index} has malformed public keys")
        if previous is not None and old != previous:
            raise ValueError(f"key-roll link {index} is not connected to the previous link")
        if index == 0:
            seen.add(old)
        if new in seen:
            raise ValueError("key-roll chain repeats an earlier key")
        if isinstance(link["issued_at"], bool) or not isinstance(link["issued_at"], int) \
                or link["issued_at"] < 0:
            raise ValueError(f"key-roll link {index} has a malformed issued_at")
        if not isinstance(link["reason"], str) or not link["reason"].strip() \
                or len(link["reason"]) > 1024:
            raise ValueError(f"key-roll link {index} has a malformed reason")
        message = canonical_keyroll(link)
        if not verify_signature(link["sig_old"], old, message):
            raise ValueError(f"key-roll link {index} has an invalid sig_old")
        if not verify_signature(link["sig_new"], new, message):
            raise ValueError(f"key-roll link {index} has an invalid sig_new")
        seen.add(new)
        previous = new
    return chain


def keyroll_tip(chain: object, pinned_pubkey: str, *, forbidden=()) -> str:
    """Return the tip reachable forward from the pin; never infer trust from older links."""
    links = validate_keyroll_chain(chain)
    try:
        start = next(i for i, link in enumerate(links) if link["old_pubkey"] == pinned_pubkey)
    except StopIteration as e:
        raise ValueError("key-roll chain has no link starting at the pinned key") from e
    forbidden = set(forbidden)
    # Connectivity was checked globally; this makes the trust direction explicit.
    expected = pinned_pubkey
    for link in links[start:]:
        if link["old_pubkey"] != expected:
            raise ValueError("key-roll chain does not continue from the pinned key")
        if link["new_pubkey"] in forbidden:
            raise ValueError("key-roll chain would roll back to a previously trusted key")
        expected = link["new_pubkey"]
    return expected


# ── pure-Python Ed25519 verification (RFC 8032 §5.1.7), used only on a bare machine ──────────────

_P = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _inv(x):
    return pow(x, _P - 2, _P)


def _add(a, b):
    (x1, y1, z1, t1), (x2, y2, z2, t2) = a, b
    A = (y1 - x1) * (y2 - x2) % _P
    B = (y1 + x1) * (y2 + x2) % _P
    C = 2 * t1 * t2 * _D % _P
    Dd = 2 * z1 * z2 % _P
    E, F, G, H = B - A, Dd - C, Dd + C, B + A
    return (E * F % _P, G * H % _P, F * G % _P, E * H % _P)


def _mul(s, pnt):
    q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            q = _add(q, pnt)
        pnt = _add(pnt, pnt)
        s >>= 1
    return q


def _eq(a, b):
    (x1, y1, z1, _), (x2, y2, z2, _) = a, b
    return (x1 * z2 - x2 * z1) % _P == 0 and (y1 * z2 - y2 * z1) % _P == 0


def _recover_x(y, sign_bit):
    if y >= _P:
        return None
    x2 = (y * y - 1) * _inv(_D * y * y + 1)
    if x2 == 0:
        return None if sign_bit else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _I % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign_bit:
        x = _P - x
    return x


def _decompress(s: bytes):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign_bit = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign_bit)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


_G = _decompress(bytes.fromhex("5866666666666666666666666666666666666666666666666666666666666666"))


def _ed25519_verify_pure(pub: bytes, msg: bytes, sig: bytes) -> bool:
    if len(pub) != 32 or len(sig) != 64:
        return False
    A = _decompress(pub)
    R = _decompress(sig[:32])
    if A is None or R is None:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= _L:
        return False
    h = int.from_bytes(hashlib.sha512(sig[:32] + pub + msg).digest(), "little") % _L
    return _eq(_mul(s, _G), _add(R, _mul(h, A)))
