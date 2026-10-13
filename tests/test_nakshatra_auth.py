"""Tests for scripts/nakshatra_auth.py — the worker-side Ed25519 +
TLS pinning helpers (Phase F of the worker hardening sprint, 2026-05-19).

This module is independent of the rest of Nakshatra (no torch / hivemind
dependencies pulled in) so it runs fast in any environment that has
cryptography installed.
"""
from __future__ import annotations

import socket
import ssl
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

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

def _make_self_signed_cert_der(return_key=False):
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
    der = cert.public_bytes(encoding=serialization.Encoding.DER)
    if return_key:
        key_pem = priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return der, key_pem
    return der


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


def test_verify_pillar_response_spki_checks_connected_socket_before_body():
    der = _make_self_signed_cert_der()
    expected = auth.compute_spki_sha256(der)
    sock = types.SimpleNamespace(getpeercert=lambda binary_form=False: der if binary_form else {})
    response = types.SimpleNamespace(
        fp=types.SimpleNamespace(raw=types.SimpleNamespace(_sock=sock))
    )

    auth.verify_pillar_response_spki(response, expected)
    with pytest.raises(auth.PillarSpkiMismatch):
        auth.verify_pillar_response_spki(response, "0" * 64)


def test_verify_pillar_response_spki_refuses_when_socket_is_hidden():
    response = types.SimpleNamespace(fp=types.SimpleNamespace(raw=object()))
    with pytest.raises(auth.PillarSpkiMismatch, match="certificate is unavailable"):
        auth.verify_pillar_response_spki(response, "0" * 64)


def test_wrong_pin_sends_zero_http_request_bytes(tmp_path):
    """The SPKI gate runs after TLS but before request serialization."""
    cert_der, key_pem = _make_self_signed_cert_der(return_key=True)
    cert_pem = ssl.DER_cert_to_PEM_cert(cert_der).encode("ascii")
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    received = []
    ready = threading.Event()

    def serve_once():
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        ready.set()
        conn, _addr = listener.accept()
        try:
            with context.wrap_socket(conn, server_side=True) as tls_conn:
                tls_conn.settimeout(1.0)
                try:
                    received.append(tls_conn.recv(4096))
                except (socket.timeout, ssl.SSLError):
                    received.append(b"")
        finally:
            listener.close()

    thread = threading.Thread(target=serve_once, daemon=True)
    thread.start()
    assert ready.wait(2)
    port = listener.getsockname()[1]
    req = urlrequest.Request(
        f"https://127.0.0.1:{port}/secret?token=credential",
        data=b"sensitive-body",
        headers={"Authorization": "secret-header"},
        method="POST",
    )
    with pytest.raises(auth.PillarSpkiMismatch):
        auth.open_pillar_url(req, timeout=2, expected_spki_hash="0" * 64)
    thread.join(3)
    assert received == [b""]


def test_pillar_redirect_is_refused_without_contacting_location():
    target_hits = []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self):
            target_hits.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.1:{target.server_port}/stolen"
            )
            self.end_headers()

        def log_message(self, *_args):
            pass

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (target, origin)
    ]
    for thread in threads:
        thread.start()
    try:
        with pytest.raises(urlerror.HTTPError) as exc:
            auth.open_pillar_url(
                f"http://127.0.0.1:{origin.server_port}/peers", timeout=2
            )
        assert exc.value.code == 302
        time.sleep(0.05)
        assert target_hits == []
    finally:
        origin.shutdown()
        target.shutdown()


def test_redact_url_for_log_removes_all_credential_bearing_components():
    url = "https://alice:secret@[2001:db8::1]:9443/base?token=secret#private"
    assert auth.redact_url_for_log(url) == "https://[2001:db8::1]:9443/base"
    assert auth.redact_url_in_text(f"failed: {url}", url) == \
        "failed: https://[2001:db8::1]:9443/base"
