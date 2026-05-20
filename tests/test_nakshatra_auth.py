"""Tests for scripts/nakshatra_auth.py — the worker-side Ed25519 +
TLS pinning helpers (Phase F of the worker hardening sprint, 2026-05-19).

This module is independent of the rest of Nakshatra (no torch / hivemind
dependencies pulled in) so it runs fast in any environment that has
cryptography installed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# scripts/ isn't on sys.path by default in the Nakshatra test runner;
# put it there so we can import nakshatra_auth.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nakshatra_auth as auth  # noqa: E402


# ── Keygen + persistence ─────────────────────────────────────────────

def test_generate_keypair_shape():
    priv, pub_hex = auth.generate_keypair()
    assert len(priv) == 32
    assert len(pub_hex) == 64
    bytes.fromhex(pub_hex)


def test_public_key_hex_from_private_matches_keygen():
    priv, pub_hex = auth.generate_keypair()
    assert auth.public_key_hex_from_private(priv) == pub_hex


def test_load_or_create_worker_key_idempotent(tmp_path):
    p = tmp_path / "worker.ed25519"
    priv1, pub1 = auth.load_or_create_worker_key(p)
    priv2, pub2 = auth.load_or_create_worker_key(p)
    assert priv1 == priv2
    assert pub1 == pub2


def test_load_or_create_worker_key_mode_600(tmp_path):
    p = tmp_path / "worker.ed25519"
    auth.load_or_create_worker_key(p)
    assert oct(p.stat().st_mode)[-3:] == "600"


def test_load_or_create_worker_key_regenerates_on_corruption(tmp_path):
    p = tmp_path / "worker.ed25519"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"too short")
    _, pub = auth.load_or_create_worker_key(p)
    assert len(pub) == 64  # fresh key regenerated


# ── Canonical string + sign + verify ─────────────────────────────────

def test_canonical_string_deterministic():
    a = auth.canonical_string("POST", "/peer", b'{"x":1}', 1)
    b = auth.canonical_string("POST", "/peer", b'{"x":1}', 1)
    assert a == b


def test_sign_verify_roundtrip():
    priv, pub_hex = auth.generate_keypair()
    sig = auth.sign_request(priv, "POST", "/peer", b'{"x":1}', 1700000000)
    assert auth.verify_request(pub_hex, "POST", "/peer", b'{"x":1}',
                                 1700000000, sig) is True


def test_verify_rejects_tampered_body():
    priv, pub_hex = auth.generate_keypair()
    sig = auth.sign_request(priv, "POST", "/peer", b'{"x":1}', 1)
    assert auth.verify_request(pub_hex, "POST", "/peer", b'{"x":2}', 1, sig) is False


def test_build_signed_envelope_produces_valid_header():
    priv, pub_hex = auth.generate_keypair()
    header, ts = auth.build_signed_envelope(priv, "worker-1", "POST", "/peer", b"")
    assert header.startswith(auth.AUTH_SCHEME)
    assert f'keyid="worker-1"' in header
    assert f'ts="{ts}"' in header


# ── SPKI pinning ─────────────────────────────────────────────────────

def _make_self_signed_cert_der():
    """Build a tiny self-signed cert in-memory so we don't need the
    Sthambha repo for a fixture."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import datetime as dt
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "test.example"),
    ])
    now = dt.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(issuer)
        .public_key(priv.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now).not_valid_after(now + dt.timedelta(days=1))
        .sign(private_key=priv, algorithm=hashes.SHA256())
    )
    return cert.public_bytes(encoding=serialization.Encoding.DER)


def test_spki_sha256_is_stable():
    der = _make_self_signed_cert_der()
    h1 = auth.compute_spki_sha256(der)
    h2 = auth.compute_spki_sha256(der)
    assert h1 == h2
    assert len(h1) == 64


def test_verify_pillar_cert_spki_accepts_match():
    der = _make_self_signed_cert_der()
    h = auth.compute_spki_sha256(der)
    auth.verify_pillar_cert_spki(der, h)  # no raise


def test_verify_pillar_cert_spki_rejects_mismatch():
    der = _make_self_signed_cert_der()
    with pytest.raises(auth.PillarSpkiMismatch):
        auth.verify_pillar_cert_spki(der, "0" * 64)


def test_verify_pillar_cert_spki_case_insensitive():
    der = _make_self_signed_cert_der()
    h = auth.compute_spki_sha256(der).upper()
    auth.verify_pillar_cert_spki(der, h)  # uppercase still works
