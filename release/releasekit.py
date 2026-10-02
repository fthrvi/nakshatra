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

MANIFEST_SCHEMA = "nak-release/1"


def canonical(obj: dict) -> bytes:
    body = {k: v for k, v in obj.items() if k != "sig"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sign(obj: dict, priv_hex: str) -> dict:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    priv = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
    pub_hex = priv.public_key().public_bytes_raw().hex()
    out = dict(obj)
    out["sig"] = {"alg": "Ed25519", "keyid": pub_hex[:16],
                  "value": base64.b64encode(priv.sign(canonical(obj))).decode("ascii")}
    return out


def verify(obj: dict, pub_hex: str) -> bool:
    try:
        sig = obj["sig"]
        if sig.get("alg") != "Ed25519":
            return False
        raw = base64.b64decode(sig["value"])
        msg = canonical(obj)
        pub = bytes.fromhex(pub_hex)
    except (KeyError, ValueError, TypeError):
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
