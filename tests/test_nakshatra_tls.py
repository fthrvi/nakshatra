"""Tests for scripts/nakshatra_tls.py.

Phase 2.1 + 2.3 of the 2026-05-21 SPKI federation sprint: self-signed
cert generation + SPKI fingerprint computation. The wire-contract
property — that the hex string this module emits matches the byte
layout sthambha distributes in its /peers projection — is asserted by
the cross-repo wire test (Phase 1.8 + Phase 3.7).

These tests exercise the module on its own:
- generate_self_signed_cert writes both files with correct modes
- compute_spki_hash matches the canonical openssl recipe
- ensure_cert is idempotent (no rotation on second call)
- ensure_cert refuses the half-rotated state
- the hash is deterministic for the same key, distinct for fresh keys
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nakshatra_tls as nt  # noqa: E402


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# ── 2.1 — generate_self_signed_cert ─────────────────────────────────


def test_generate_writes_both_files(tmp_path):
    cert_path, key_path = nt.generate_self_signed_cert(
        hostname="t1.local", output_dir=tmp_path,
    )
    assert cert_path.exists()
    assert key_path.exists()
    assert cert_path.name == nt.CERT_FILENAME
    assert key_path.name == nt.KEY_FILENAME


def test_generate_key_file_mode_is_600(tmp_path):
    """The private key MUST be unreadable to other users. We set 0o600
    via O_CREAT mode bits, but check the post-write file mode in case
    umask interferes on some platforms."""
    _, key_path = nt.generate_self_signed_cert(
        hostname="t1.local", output_dir=tmp_path,
    )
    mode = key_path.stat().st_mode & 0o777
    assert mode == 0o600, f"key file mode {oct(mode)} is not 0o600"


def test_generate_refuses_overwrite_by_default(tmp_path):
    nt.generate_self_signed_cert(hostname="t1.local", output_dir=tmp_path)
    with pytest.raises(FileExistsError):
        nt.generate_self_signed_cert(hostname="t1.local", output_dir=tmp_path)


def test_generate_overwrites_when_forced(tmp_path):
    nt.generate_self_signed_cert(hostname="t1.local", output_dir=tmp_path)
    cert_path = tmp_path / nt.CERT_FILENAME
    spki_1 = nt.compute_spki_hash(cert_path)
    nt.generate_self_signed_cert(
        hostname="t1.local", output_dir=tmp_path, overwrite=True,
    )
    spki_2 = nt.compute_spki_hash(cert_path)
    # Fresh keypair → fresh SPKI hash.
    assert spki_1 != spki_2


def test_generate_creates_output_dir_if_missing(tmp_path):
    nested = tmp_path / "deeper" / "still" / "tls"
    assert not nested.exists()
    cert_path, _ = nt.generate_self_signed_cert(output_dir=nested)
    assert cert_path.exists()
    # Phase 2's worker-boot path expects this — it doesn't pre-create
    # ~/.nakshatra/tls/.


# ── 2.3 — compute_spki_hash ─────────────────────────────────────────


def test_compute_spki_hash_is_64_hex_chars(tmp_path):
    cert_path, _ = nt.generate_self_signed_cert(output_dir=tmp_path)
    h = nt.compute_spki_hash(cert_path)
    assert HEX64_RE.match(h), f"not 64 lowercase hex: {h!r}"


def test_compute_spki_hash_is_stable_across_calls(tmp_path):
    cert_path, _ = nt.generate_self_signed_cert(output_dir=tmp_path)
    h1 = nt.compute_spki_hash(cert_path)
    h2 = nt.compute_spki_hash(cert_path)
    assert h1 == h2


def test_compute_spki_hash_from_pem_matches_file_variant(tmp_path):
    cert_path, _ = nt.generate_self_signed_cert(output_dir=tmp_path)
    file_hash = nt.compute_spki_hash(cert_path)
    pem_bytes = cert_path.read_bytes()
    in_memory_hash = nt.compute_spki_hash_from_pem(pem_bytes)
    assert file_hash == in_memory_hash


def test_compute_spki_hash_matches_openssl_recipe(tmp_path):
    """Sanity check: the hash we emit is the same hash an operator
    gets from the canonical openssl one-liner. Skipped on systems
    without openssl in PATH (CI containers might not have it)."""
    cert_path, _ = nt.generate_self_signed_cert(output_dir=tmp_path)
    try:
        # openssl x509 -in cert.pem -pubkey -noout |
        #   openssl pkey -pubin -outform der |
        #   openssl dgst -sha256 -hex
        p1 = subprocess.run(
            ["openssl", "x509", "-in", str(cert_path), "-pubkey", "-noout"],
            capture_output=True, check=True,
        )
        p2 = subprocess.run(
            ["openssl", "pkey", "-pubin", "-outform", "der"],
            input=p1.stdout, capture_output=True, check=True,
        )
        # Hash the DER bytes ourselves rather than parsing dgst output.
        openssl_hash = hashlib.sha256(p2.stdout).hexdigest()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        pytest.skip(f"openssl unavailable: {e}")
    assert nt.compute_spki_hash(cert_path) == openssl_hash


def test_fresh_cert_yields_different_spki(tmp_path):
    """Independent keypairs MUST produce independent SPKI hashes —
    otherwise the pin gives a false sense of identity. (RSA-2048
    keygen has astronomically low collision probability; this test
    guards against an accidental constant return value.)"""
    cert_path_a, _ = nt.generate_self_signed_cert(output_dir=tmp_path / "a")
    cert_path_b, _ = nt.generate_self_signed_cert(output_dir=tmp_path / "b")
    assert nt.compute_spki_hash(cert_path_a) != nt.compute_spki_hash(cert_path_b)


# ── ensure_cert (the worker-boot entry point) ───────────────────────


def test_ensure_cert_generates_when_missing(tmp_path):
    cert_path, key_path, spki = nt.ensure_cert(output_dir=tmp_path)
    assert cert_path.exists()
    assert key_path.exists()
    assert HEX64_RE.match(spki)


def test_ensure_cert_is_idempotent(tmp_path):
    """Two ensure_cert calls back-to-back must not rotate the key — a
    worker restart triggers exactly one boot, but operator scripts
    that call ensure_cert from CLI MUST be safe to invoke repeatedly
    without invalidating workers that pinned the previous hash."""
    _, _, spki_1 = nt.ensure_cert(output_dir=tmp_path)
    _, _, spki_2 = nt.ensure_cert(output_dir=tmp_path)
    assert spki_1 == spki_2


def test_ensure_cert_refuses_half_state_cert_present(tmp_path):
    """If the cert is present but the key is missing, refuse to
    generate (an accidental ``rm worker-key.pem`` would silently
    rotate the SPKI and break every pinned peer). Operator must
    explicitly remove the partner file."""
    nt.ensure_cert(output_dir=tmp_path)
    (tmp_path / nt.KEY_FILENAME).unlink()
    with pytest.raises(FileExistsError, match="partial cert state"):
        nt.ensure_cert(output_dir=tmp_path)


def test_ensure_cert_refuses_half_state_key_present(tmp_path):
    nt.ensure_cert(output_dir=tmp_path)
    (tmp_path / nt.CERT_FILENAME).unlink()
    with pytest.raises(FileExistsError, match="partial cert state"):
        nt.ensure_cert(output_dir=tmp_path)


def test_ensure_cert_default_dir_is_under_home():
    """Sanity-only: the default writes into ~/.nakshatra/tls/. Most
    tests use a tmp_path so we never touch the real dir; this asserts
    the default constant is what we expect."""
    assert nt.DEFAULT_TLS_DIR == Path.home() / ".nakshatra" / "tls"


# ── build_grpc_server_credentials ───────────────────────────────────


def test_build_grpc_server_credentials_returns_server_credentials(tmp_path):
    """Belt-and-suspenders: confirm the helper returns a grpc
    ServerCredentials shape. Exercising the full handshake needs a
    live gRPC server and is covered by the cluster smoke; this is
    the shape check."""
    cert_path, key_path = nt.generate_self_signed_cert(output_dir=tmp_path)
    creds = nt.build_grpc_server_credentials(cert_path, key_path)
    import grpc
    assert isinstance(creds, grpc.ServerCredentials)


# ── resolve_tls_required (the boot-time env decoder) ─────────────────


def test_resolve_tls_required_explicit_true():
    assert nt.resolve_tls_required("true", "") is True
    assert nt.resolve_tls_required("1", "") is True
    assert nt.resolve_tls_required("yes", "") is True
    assert nt.resolve_tls_required("TRUE", "") is True  # case-insensitive


def test_resolve_tls_required_explicit_false_with_pillar():
    """Operator opt-out overrides the pillar-default-on. This is the
    Mode-A bringup path; the worker boot should emit a WARN when this
    happens (handled by the caller, not this function)."""
    assert nt.resolve_tls_required("false", "http://pillar:5530") is False
    assert nt.resolve_tls_required("0", "http://pillar:5530") is False
    assert nt.resolve_tls_required("no", "http://pillar:5530") is False


def test_resolve_tls_required_unset_no_pillar_is_false():
    """Mode A legacy bringup: no pillar, no TLS."""
    assert nt.resolve_tls_required(None, "") is False
    assert nt.resolve_tls_required("", "") is False
    assert nt.resolve_tls_required("   ", "  ") is False


def test_resolve_tls_required_unset_with_pillar_is_true():
    """Mode B/C default: pillar configured → TLS on."""
    assert nt.resolve_tls_required(None, "http://pillar:5530") is True
    assert nt.resolve_tls_required("", "http://pillar:5530") is True
    assert nt.resolve_tls_required("   ", "https://pillar:5531") is True
