"""Worker-side Ed25519 helpers for talking to a Sthambha pillar.

Matches the canonical-string protocol from sthambha/auth.py byte-for-byte
so the worker's signed requests verify on the pillar's middleware. We
duplicate the helper here rather than import from sthambha because the
two repos release independently — the wire contract is in ADR 0006,
not a shared library.

Canonical string (signed bytes):

    method "\\n" path "\\n" sha256_hex(body) "\\n" timestamp_unix_seconds

Header on the wire:

    Authorization: Sthambha-Ed25519 keyid="<node_id>",sig="<b64>",ts="<unix>"

Phase F1 of the Nakshatra worker-side hardening sprint (2026-05-19).
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import os
import ssl
import stat
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit, urlunsplit

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


AUTH_SCHEME = "Sthambha-Ed25519"
DEFAULT_TIMESTAMP_WINDOW_S = 60.0
WORKER_KEY_PATH = Path.home() / ".nakshatra" / "keys" / "worker.ed25519"


# ── Keygen / load ─────────────────────────────────────────────────────

def generate_keypair() -> Tuple[bytes, str]:
    """Returns (private_key_bytes_32, public_key_hex_64)."""
    priv = ed25519.Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_hex = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    return priv_bytes, pub_hex


def public_key_hex_from_private(priv_bytes: bytes) -> str:
    priv = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
    return priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def load_worker_key(path: Optional[Path] = None) -> Tuple[bytes, str]:
    """Strictly load an existing node key without changing its custody.

    Long-running services must not silently create, rotate, or repair node identity.
    The private seed is accepted only when it is a regular 32-byte file owned by
    this user with no group or other permissions.  In particular, this function
    never chmods an operator's key.
    """
    key_path = Path(path) if path is not None else WORKER_KEY_PATH
    try:
        fd = os.open(key_path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"worker key is missing at {key_path}; create it through the join/setup flow"
        ) from exc
    try:
        key_stat = os.fstat(fd)
        if not stat.S_ISREG(key_stat.st_mode):
            raise ValueError(f"worker key at {key_path} is not a regular file; refusing")
        if key_stat.st_uid != os.getuid():
            raise PermissionError(
                f"worker key at {key_path} is owned by uid {key_stat.st_uid}, "
                f"not this service's uid {os.getuid()}; refusing"
            )
        mode = stat.S_IMODE(key_stat.st_mode)
        if mode & 0o077:
            raise PermissionError(
                f"worker key at {key_path} has mode {mode:04o}; refusing. "
                f"Run: chmod 600 {key_path}"
            )
        with os.fdopen(fd, "rb", closefd=False) as key_file:
            priv = key_file.read(33)
    finally:
        os.close(fd)
    if len(priv) != 32:
        raise ValueError(
            f"worker key at {key_path} is malformed: expected a 32-byte Ed25519 seed"
        )
    try:
        pub_hex = public_key_hex_from_private(priv)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"worker key at {key_path} is not a valid Ed25519 seed") from exc
    return priv, pub_hex


def load_or_create_worker_key(path: Path = WORKER_KEY_PATH) -> Tuple[bytes, str]:
    """Read the persisted worker key, or generate + persist a fresh one.
    File is mode 600. Returns (private_bytes, public_key_hex)."""
    if path.exists():
        priv = path.read_bytes()
        if len(priv) == 32:
            return priv, public_key_hex_from_private(priv)
        # Corrupted; regenerate (warn the caller via stderr).
        import sys
        print(f"[worker-auth] worker key at {path} is malformed; regenerating",
              file=sys.stderr, flush=True)

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    priv, pub_hex = generate_keypair()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, priv)
    finally:
        os.close(fd)
    return priv, pub_hex


# ── Canonical string + sign ──────────────────────────────────────────

def canonical_string(method: str, path: str, body: bytes,
                     timestamp_unix: int) -> bytes:
    body_hash = hashlib.sha256(body or b"").hexdigest()
    return (
        f"{method.upper()}\n{path}\n{body_hash}\n{int(timestamp_unix)}"
    ).encode("utf-8")


def sign_request(priv_bytes: bytes, method: str, path: str,
                 body: bytes, timestamp_unix: int) -> str:
    priv = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
    msg = canonical_string(method, path, body, timestamp_unix)
    return base64.b64encode(priv.sign(msg)).decode("ascii")


def build_auth_header(node_id: str, signature_b64: str,
                      timestamp_unix: int) -> str:
    return (
        f'{AUTH_SCHEME} keyid="{node_id}",'
        f'sig="{signature_b64}",ts="{int(timestamp_unix)}"'
    )


def build_signed_envelope(priv_bytes: bytes, node_id: str,
                          method: str, path: str, body: bytes,
                          timestamp_unix: Optional[int] = None
                          ) -> Tuple[str, int]:
    """One-call wrapper. Returns (header_value, timestamp_used)."""
    ts = timestamp_unix if timestamp_unix is not None else int(time.time())
    sig = sign_request(priv_bytes, method, path, body, ts)
    return build_auth_header(node_id, sig, ts), ts


def verify_request(public_key_hex: str, method: str, path: str,
                   body: bytes, timestamp_unix: int,
                   signature_b64: str) -> bool:
    """Used by inbound-request handlers (e.g. worker's POST /slice).
    Returns False on any failure — never raises."""
    try:
        pub_bytes = bytes.fromhex(public_key_hex)
        sig = base64.b64decode(signature_b64, validate=True)
    except (ValueError, TypeError):
        return False
    if len(pub_bytes) != 32 or len(sig) != 64:
        return False
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        pub.verify(sig, canonical_string(method, path, body, timestamp_unix))
        return True
    except (InvalidSignature, ValueError):
        return False


# ── TLS client with SPKI pinning (Phase F3) ───────────────────────────

class PillarSpkiMismatch(Exception):
    """Raised by verify_pillar_cert_spki when the cert's SPKI hash
    doesn't match the expected value. Worker should refuse to proceed."""


class PillarRedirectRefused(urlerror.HTTPError):
    """A pillar response attempted to redirect the authenticated request."""


def compute_spki_sha256(cert_pem_or_der: bytes) -> str:
    """SPKI SHA-256 hex of a cert (PEM or DER)."""
    if cert_pem_or_der.startswith(b"-----BEGIN"):
        cert = x509.load_pem_x509_certificate(cert_pem_or_der)
    else:
        cert = x509.load_der_x509_certificate(cert_pem_or_der)
    spki_der = cert.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(spki_der).hexdigest()


def verify_pillar_cert_spki(cert_der: bytes, expected_hash: str) -> None:
    """Raises PillarSpkiMismatch if the cert's SPKI doesn't match. Used
    on the TLS connection's server cert via ssock.getpeercert(binary_form=True)."""
    try:
        actual = compute_spki_sha256(cert_der)
    except (TypeError, ValueError) as e:
        raise PillarSpkiMismatch(
            "pillar SPKI verification failed: peer certificate is malformed"
        ) from e
    if actual != expected_hash.lower():
        raise PillarSpkiMismatch(
            f"pillar SPKI mismatch: expected {expected_hash}, got {actual}"
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that authenticates the peer before HTTP is written."""

    def __init__(self, *args, expected_spki_hash: str, **kwargs):
        self._expected_spki_hash = expected_spki_hash
        kwargs.pop("check_hostname", None)
        super().__init__(*args, **kwargs)

    def connect(self) -> None:
        super().connect()
        try:
            cert_der = self.sock.getpeercert(binary_form=True) if self.sock else None
            if not cert_der:
                raise PillarSpkiMismatch(
                    "pillar SPKI verification failed: TLS peer certificate is empty"
                )
            verify_pillar_cert_spki(cert_der, self._expected_spki_hash)
        except Exception:
            # A mismatched connection must never remain available for do_open()
            # to serialize its request line, headers, or body onto the socket.
            self.close()
            raise


class _PinnedHTTPSHandler(urlrequest.HTTPSHandler):
    def __init__(self, context: ssl.SSLContext, expected_spki_hash: str):
        super().__init__(context=context, check_hostname=False)
        self._expected_spki_hash = expected_spki_hash

    def https_open(self, req):
        expected = self._expected_spki_hash

        def connection(host, **kwargs):
            return _PinnedHTTPSConnection(
                host, expected_spki_hash=expected, **kwargs
            )

        return self.do_open(
            connection, req, context=self._context, check_hostname=False
        )


class _NoPillarRedirectHandler(urlrequest.HTTPRedirectHandler):
    """Never replay pillar credentials or accept authority via redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PillarRedirectRefused(
            req.full_url, code, "pillar redirect refused", headers, fp
        )


def open_pillar_url(
    req,
    *,
    timeout: float,
    expected_spki_hash: Optional[str] = None,
):
    """Open one pillar request without redirects and with pre-write pinning.

    For pinned HTTPS, :class:`_PinnedHTTPSConnection` checks the peer SPKI
    immediately after the TLS handshake.  ``urllib`` cannot send the HTTP
    request until ``connect()`` returns, so a mismatch exposes no path, query,
    headers, or body to the peer.  Plain overlay HTTP still uses the same
    no-redirect policy.
    """
    url = req.full_url if isinstance(req, urlrequest.Request) else str(req)
    scheme = urlsplit(url).scheme.lower()
    handlers = [_NoPillarRedirectHandler()]
    if scheme == "https":
        if not expected_spki_hash:
            raise PillarSpkiMismatch(
                "pillar SPKI verification failed: HTTPS pillar has no pin"
            )
        context = build_pillar_ssl_context(expected_spki_hash)
        handlers.append(_PinnedHTTPSHandler(context, expected_spki_hash))
    opener = urlrequest.build_opener(*handlers)
    return opener.open(req, timeout=timeout)


def redact_url_for_log(location: str) -> str:
    """Return a URL/path without userinfo, query parameters, or fragments."""
    try:
        parts = urlsplit(location)
        if parts.netloc:
            host = parts.hostname or ""
            if ":" in host:
                host = f"[{host}]"
            try:
                port = f":{parts.port}" if parts.port is not None else ""
            except ValueError:
                port = ""
            return urlunsplit((parts.scheme, host + port, parts.path, "", ""))
        return urlunsplit((parts.scheme, "", parts.path, "", ""))
    except (TypeError, ValueError):
        return "<invalid URL>"


def redact_url_in_text(value: object, location: str) -> str:
    """Redact a configured URL if an HTTP exception embeds it in text."""
    return str(value).replace(location, redact_url_for_log(location))


def verify_pillar_response_spki(response, expected_hash: Optional[str]) -> None:
    """Verify an opened urllib HTTPS response before its body is consumed.

    Legacy response-level verifier retained for callers outside the release
    worker. New pillar requests must use :func:`open_pillar_url`, which checks
    the pin before request bytes are sent.
    """
    if not expected_hash:
        return
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is None:
        connection = getattr(raw, "_connection", None)
        sock = getattr(connection, "sock", None)
    if sock is None or not hasattr(sock, "getpeercert"):
        raise PillarSpkiMismatch(
            "pillar SPKI verification failed: TLS peer certificate is unavailable"
        )
    cert_der = sock.getpeercert(binary_form=True)
    if not cert_der:
        raise PillarSpkiMismatch(
            "pillar SPKI verification failed: TLS peer certificate is empty"
        )
    verify_pillar_cert_spki(cert_der, expected_hash)


def build_pillar_ssl_context(
    expected_spki_hash: Optional[str] = None,
) -> ssl.SSLContext:
    """Build a TLS client context for pillar HTTPS connections.

    Phase I2 (2026-05-19): TLS 1.3 minimum by default. Workers running
    against legacy pillars can opt in to TLS 1.2 via
    STHAMBHA_TLS_ALLOW_1_2=true; the pillar must have the matching opt-
    in for the handshake to succeed.

    When ``expected_spki_hash`` is provided, the caller MUST use
    ``open_pillar_url`` so the pin is checked before HTTP is sent. This
    context turns off the default hostname-matching + CA verification because
    the pillar's cert is self-signed; we authenticate by SPKI hash instead,
    which survives cert rotations as long as the underlying keypair is stable.

    When ``expected_spki_hash`` is None, the context still allows
    self-signed certs through but provides no identity check — useful
    for development. The worker should WARN in this mode.
    """
    import os
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if os.environ.get("STHAMBHA_TLS_ALLOW_1_2", "").strip().lower() in (
            "true", "1", "yes"):
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    else:
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
