"""TLS support for the Nakshatra worker gRPC server (Phase 2 of the
2026-05-21 SPKI federation sprint).

Self-signed certificate generation + SPKI-pinning fingerprint helpers.
Mirror of sthambha/tls.py (sthambha Phase E1) — the contract is shared
across repos so that pillar-distributed `peer_spki_hash` values match
the bytes a worker actually serves on TLS handshake.

Workers verify each other's identity by pinning the SubjectPublicKeyInfo
SHA-256 hash, not the cert subject — RFC 7469 SPKI pin. The cert itself
can rotate; the SPKI hash stays stable as long as the underlying keypair
doesn't. Operators rotate the keypair by deleting the files at
``~/.nakshatra/tls/`` and restarting the worker; the new SPKI hash is
declared on the next ``/peer`` registration. Workers with stale resolver
caches refuse the rotated peer until their next refresh (default 60s).

stdlib + cryptography only — no extra deps. The ``cryptography``
runtime dep is already declared via ADR 0006 (shared with sthambha).
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import os
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

log = logging.getLogger(__name__)

# Default validity matches sthambha's: 10 years. Cert is pinned by SPKI
# hash so the calendar expiry doesn't matter to pinning workers; the
# rotation event is "operator deletes the files" not "cert expires."
DEFAULT_CERT_VALIDITY_DAYS = 365 * 10

# Canonical worker TLS dir + file names. Operators with their own CA
# can pre-populate these files; ``ensure_cert`` will leave them alone.
DEFAULT_TLS_DIR = Path.home() / ".nakshatra" / "tls"
CERT_FILENAME = "worker-cert.pem"
KEY_FILENAME = "worker-key.pem"


def generate_self_signed_cert(
    hostname: str = "nakshatra.local",
    output_dir: Path | None = None,
    *,
    validity_days: int = DEFAULT_CERT_VALIDITY_DAYS,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Generate an RSA-2048 self-signed cert + private key, written to
    ``output_dir/worker-cert.pem`` and ``output_dir/worker-key.pem``.

    RSA-2048 mirrors the sthambha pillar's choice (TLS 1.3 + Ed25519
    server certs need OpenSSL 3.x both ends; RSA-2048 keeps broader
    compatibility on older clients in the lab). The worker's
    *application-layer* Ed25519 keypair (``~/.nakshatra/keys/``) is
    unrelated — that's the signer for /peer registrations.

    Returns (cert_path, key_path). Refuses to overwrite an existing
    pair unless ``overwrite=True`` — accidental rotation breaks every
    other worker that's pinned the old SPKI hash.
    """
    if output_dir is None:
        output_dir = DEFAULT_TLS_DIR
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    cert_path = output_dir / CERT_FILENAME
    key_path = output_dir / KEY_FILENAME
    if (cert_path.exists() or key_path.exists()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing cert/key at {output_dir}; "
            f"pass overwrite=True to rotate"
        )

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Nakshatra"),
    ])
    # datetime.UTC was added in Python 3.11; use timezone.utc for back-
    # compat with the 3.9-era venvs on some cluster machines.
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(priv.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=validity_days))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName("localhost"),
            ]),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(private_key=priv, algorithm=hashes.SHA256())
    )

    cert_pem = cert.public_bytes(encoding=serialization.Encoding.PEM)
    key_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Atomic write for the cert; restrict private-key file mode to 600
    # via O_CREAT mode so the file is unreadable to other users from
    # the moment it exists.
    tmp_cert = cert_path.with_suffix(".pem.tmp")
    tmp_cert.write_bytes(cert_pem)
    os.replace(tmp_cert, cert_path)

    fd = os.open(str(key_path),
                  os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key_pem)
    finally:
        os.close(fd)

    log.info(f"nakshatra.tls: generated cert for {hostname} at {output_dir}")
    return cert_path, key_path


def compute_spki_hash(cert_path: Path | str) -> str:
    """SHA-256 of the cert's SubjectPublicKeyInfo (DER-encoded), hex-
    encoded. This is what workers declare in ``/peer.peer_spki_hash``
    and pin against on outbound gRPC handshakes (RFC 7469 model).

    The cert subject is irrelevant — it's just self-signed text. Only
    the pubkey matters; the SPKI hash is the canonical fingerprint of
    that pubkey.
    """
    with open(cert_path, "rb") as f:
        cert_pem = f.read()
    return compute_spki_hash_from_pem(cert_pem)


def compute_spki_hash_from_pem(cert_pem: bytes) -> str:
    """In-memory variant of :func:`compute_spki_hash`. Useful when the
    cert PEM is already loaded (e.g. for the gRPC server credentials
    construction) so we don't re-read the file."""
    cert = x509.load_pem_x509_certificate(cert_pem)
    spki_der = cert.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(spki_der).hexdigest()


def ensure_cert(
    output_dir: Path | None = None,
    *,
    hostname: str = "nakshatra.local",
) -> tuple[Path, Path, str]:
    """Idempotent worker-boot helper: read existing cert/key if both
    present at ``output_dir``; otherwise generate a fresh self-signed
    pair. Always returns ``(cert_path, key_path, spki_hex)``.

    The "operator pre-populated their own cert" path falls out for
    free: if the operator dropped a real CA-signed cert at
    ``~/.nakshatra/tls/worker-cert.pem`` (with matching key), this
    function leaves it alone and just computes the SPKI hash. The
    pillar distributes whatever hash the worker declares; operators
    can swap in real PKI without code changes.
    """
    if output_dir is None:
        output_dir = DEFAULT_TLS_DIR
    cert_path = output_dir / CERT_FILENAME
    key_path = output_dir / KEY_FILENAME
    if cert_path.exists() and key_path.exists():
        spki = compute_spki_hash(cert_path)
        log.info(f"nakshatra.tls: using existing cert at {cert_path} "
                 f"(spki_sha256={spki[:16]}…)")
    else:
        # One or both missing — refuse to half-rotate; either we have
        # both or we generate both. The half-state (cert without key,
        # key without cert) is operator error worth surfacing rather
        # than silently overwriting one side.
        if cert_path.exists() != key_path.exists():
            present = cert_path if cert_path.exists() else key_path
            missing = key_path if cert_path.exists() else cert_path
            raise FileExistsError(
                f"partial cert state at {output_dir}: {present.name} "
                f"present but {missing.name} missing. Remove {present.name} "
                f"to regenerate, or restore {missing.name}."
            )
        generate_self_signed_cert(hostname=hostname, output_dir=output_dir)
        spki = compute_spki_hash(cert_path)
        log.info(f"nakshatra.tls: generated self-signed cert at {cert_path} "
                 f"(spki_sha256={spki})")
    return cert_path, key_path, spki


def resolve_tls_required(env_value: str | None, pillar_url: str) -> bool:
    """Decide whether the worker should serve gRPC over TLS, given the
    env value and the configured pillar URL. Same truth-table shape as
    ``nakshatra_grpc_auth.resolve_auth_required``:

    ============ =========== ============================
    env_value    pillar_url  resolved
    ============ =========== ============================
    "true"       any         True
    "false"      any         False  (Mode A legacy / explicit opt-out)
    None/""      ""          False  (Mode A: no pillar, plaintext)
    None/""      set         True   (Mode B/C default: TLS on)
    ============ =========== ============================

    When a pillar is configured, TLS defaults ON because that's the
    Mode-C target. Operators with a legacy cluster set
    ``NAKSHATRA_TLS_REQUIRED=false`` explicitly and accept the boot
    WARN (parallel to Sthambha's STHAMBHA_AUTH_REQUIRED escape hatch).
    """
    val = (env_value or "").strip().lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return bool((pillar_url or "").strip())


def build_grpc_server_credentials(cert_path: Path | str,
                                   key_path: Path | str):
    """Build a ``grpc.ServerCredentials`` bound to the cert+key pair.
    The worker passes this to ``grpc.server.add_secure_port`` instead
    of ``add_insecure_port``.

    No mutual TLS at this layer (RT-S5 deferred per the sprint plan).
    The application-layer Ed25519 signature on the first frame (B-auth)
    is what proves the *caller's* identity; this credentials object
    proves the *server's* identity to the caller via SPKI pin.
    """
    import grpc  # imported lazily so test files that don't need gRPC
                 # can still import this module
    with open(cert_path, "rb") as f:
        cert_pem = f.read()
    with open(key_path, "rb") as f:
        key_pem = f.read()
    return grpc.ssl_server_credentials([(key_pem, cert_pem)])


# ── 2026-05-21 SPKI Phase 3 — outbound channel pinning ───────────────


PROBE_TIMEOUT_S = 5.0


class PinError(Exception):
    """Raised by :func:`open_pinned_channel` when SPKI pinning refuses
    to complete the channel.

    The ``reason`` field carries one of:
      - ``"unpinned_peer"`` — no expected SPKI available + refuse_unpinned
      - ``"probe_failed"`` — TLS probe to the peer's port failed
      - ``"spki_mismatch"`` — observed SPKI doesn't match expected

    ``details`` carries operator-readable specifics (address, expected
    hash, observed hash, underlying error) for the audit log.
    """

    def __init__(self, reason: str, **details):
        super().__init__(reason)
        self.reason = reason
        self.details = details

    def __str__(self) -> str:
        if self.reason == "spki_mismatch":
            return (f"spki_mismatch peer={self.details.get('address')} "
                    f"expected={self.details.get('expected', '')[:8]}… "
                    f"actual={self.details.get('actual', '')[:8]}…")
        if self.reason == "unpinned_peer":
            return f"unpinned_peer peer={self.details.get('address')}"
        if self.reason == "probe_failed":
            return (f"probe_failed peer={self.details.get('address')} "
                    f"error={self.details.get('error')}")
        return self.reason


def probe_peer_spki(
    address: str,
    *,
    timeout: float = PROBE_TIMEOUT_S,
) -> tuple[str, bytes]:
    """TLS-probe ``address`` (host:port), fetch the server's leaf
    certificate, and return ``(spki_hash_hex, cert_pem_bytes)``.

    Uses ``ssl.CERT_NONE`` because the peer's cert is self-signed —
    chain validation would always fail. The actual identity check is
    the SPKI hash comparison performed by the caller. The cert PEM
    we fetch here is then used as the trust anchor for the real gRPC
    handshake, so a man-in-the-middle that substitutes a different
    cert between the probe and the gRPC connection still fails the
    second handshake (gRPC sees a cert that doesn't match the root
    it was given).
    """
    import socket
    import ssl
    if ":" not in address:
        raise OSError(f"address {address!r} missing :port suffix")
    host, port_s = address.rsplit(":", 1)
    try:
        port = int(port_s)
    except ValueError as e:
        raise OSError(f"address {address!r}: port not int: {e}")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=timeout) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as ssock:
            cert_der = ssock.getpeercert(binary_form=True)
    if not cert_der:
        raise OSError(f"peer {address!r} did not present a certificate")
    cert_pem = ssl.DER_cert_to_PEM_cert(cert_der).encode()
    return compute_spki_hash_from_pem(cert_pem), cert_pem


def open_pinned_channel(
    address: str,
    expected_spki: str | None,
    *,
    refuse_unpinned: bool = True,
    probe_timeout_s: float = PROBE_TIMEOUT_S,
):
    """Open a gRPC channel to ``address`` with SPKI pinning.

    - ``expected_spki=None`` + ``refuse_unpinned=True`` raises
      :class:`PinError` ``unpinned_peer``. (Phase 3 default.)
    - ``expected_spki=None`` + ``refuse_unpinned=False`` opens an
      insecure channel — legacy Mode-A bringup.
    - ``expected_spki=<hex>``: TLS-probe the peer, compare the
      observed SPKI hash to the expected one. Mismatch raises
      ``spki_mismatch``. Probe failure raises ``probe_failed``.
      Match returns a ``grpc.Channel`` whose underlying TLS handshake
      uses the just-fetched cert as its trust anchor.

    The caller is responsible for closing the returned channel.
    """
    import grpc
    if expected_spki is None:
        if refuse_unpinned:
            raise PinError("unpinned_peer", address=address)
        return grpc.insecure_channel(address)
    try:
        actual_spki, cert_pem = probe_peer_spki(
            address, timeout=probe_timeout_s)
    except OSError as e:
        raise PinError("probe_failed", address=address, error=str(e))
    if actual_spki != expected_spki:
        raise PinError(
            "spki_mismatch",
            address=address,
            expected=expected_spki,
            actual=actual_spki,
        )
    # Self-signed peer certs are issued with CN=nakshatra.local (per
    # generate_self_signed_cert). The gRPC channel connects to an IP
    # (e.g. 203.0.113.11:5561) so the default hostname-verification
    # check would refuse the handshake. We're already pinning the SPKI
    # (RFC 7469 cert-pinning is stricter than hostname check + chain
    # validation combined), so override the expected server name to
    # the cert's documented CN. Mismatches in EITHER direction would
    # have failed the SPKI compare above.
    creds = grpc.ssl_channel_credentials(root_certificates=cert_pem)
    return grpc.secure_channel(
        address, creds,
        options=[("grpc.ssl_target_name_override", "nakshatra.local")],
    )
