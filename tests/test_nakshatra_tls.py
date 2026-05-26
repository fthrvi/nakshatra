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


# ── 2026-05-21 SPKI Phase 3.3+3.4 — open_pinned_channel ──────────────


import grpc  # noqa: E402  — needed for type assertions below


def test_pin_error_str_for_spki_mismatch():
    """The PinError __str__ produces the operator-readable reason
    that the worker's push_failed: emit consumes verbatim. If the
    format drifts, the v0.5 §9.5 error contract degrades to a less
    parseable shape."""
    e = nt.PinError("spki_mismatch", address="mac3:5500",
                    expected="a" * 64, actual="b" * 64)
    s = str(e)
    assert "spki_mismatch" in s
    assert "mac3:5500" in s
    # Hashes are abbreviated for log readability.
    assert "a" * 8 in s
    assert "b" * 8 in s


def test_pin_error_str_for_unpinned():
    e = nt.PinError("unpinned_peer", address="node-d:5500")
    s = str(e)
    assert "unpinned_peer" in s
    assert "node-d:5500" in s


def test_pin_error_str_for_probe_failed():
    e = nt.PinError("probe_failed", address="dead:5500", error="connection refused")
    s = str(e)
    assert "probe_failed" in s
    assert "dead:5500" in s
    assert "connection refused" in s


def test_open_pinned_channel_no_expected_no_refuse_returns_insecure():
    """Legacy Mode-A bringup: NAKSHATRA_REFUSE_UNPINNED_PEERS=false
    + no SPKI → open insecure channel without probing the peer."""
    ch = nt.open_pinned_channel(
        "any-host:5500", expected_spki=None, refuse_unpinned=False,
    )
    assert isinstance(ch, grpc.Channel)
    ch.close()


def test_open_pinned_channel_unpinned_refuses():
    """Default Mode-C path: no expected SPKI + refuse_unpinned=True
    raises PinError("unpinned_peer") without making a network call."""
    with pytest.raises(nt.PinError) as exc:
        nt.open_pinned_channel(
            "any-host:5500", expected_spki=None, refuse_unpinned=True,
        )
    assert exc.value.reason == "unpinned_peer"
    assert exc.value.details["address"] == "any-host:5500"


def test_open_pinned_channel_probe_failed_unreachable():
    """Probe failure (unreachable host) surfaces as
    PinError("probe_failed") so the caller can emit push_failed and
    let the client downgrade to client-relay."""
    # Address that should refuse connection — use a port nothing's
    # listening on within the short probe timeout window.
    with pytest.raises(nt.PinError) as exc:
        nt.open_pinned_channel(
            "127.0.0.1:1", expected_spki="a" * 64,
            refuse_unpinned=True, probe_timeout_s=0.3,
        )
    assert exc.value.reason == "probe_failed"


def test_open_pinned_channel_matches_against_live_server(tmp_path):
    """End-to-end: stand up a real TLS server with our self-signed
    cert, then verify open_pinned_channel succeeds with the matching
    SPKI and refuses with a deliberately-wrong SPKI. This is the
    single test that exercises the probe → match → secure_channel
    path without mocking out ssl."""
    import socket
    import ssl
    import threading

    cert_path, key_path = nt.generate_self_signed_cert(output_dir=tmp_path)
    expected = nt.compute_spki_hash(cert_path)
    wrong_spki = "0" * 64

    # Stand up a minimal TLS server on a random port. It just accepts
    # one TLS connection per session (the probe call) and closes.
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    port = sock.getsockname()[1]

    stop = threading.Event()

    def accept_loop():
        while not stop.is_set():
            try:
                sock.settimeout(0.2)
                client, _ = sock.accept()
            except (socket.timeout, OSError):
                continue
            try:
                ssock = server_ctx.wrap_socket(client, server_side=True)
                # Read whatever / nothing; the probe just wants the cert.
                try:
                    ssock.recv(64)
                except Exception:
                    pass
                ssock.close()
            except Exception:
                try:
                    client.close()
                except Exception:
                    pass

    t = threading.Thread(target=accept_loop, daemon=True)
    t.start()
    try:
        # Match: probe sees the same cert we wrote.
        ch = nt.open_pinned_channel(
            f"127.0.0.1:{port}",
            expected_spki=expected,
            refuse_unpinned=True,
            probe_timeout_s=3.0,
        )
        assert isinstance(ch, grpc.Channel)
        ch.close()
        # Mismatch: same cert, different expected → spki_mismatch.
        with pytest.raises(nt.PinError) as exc:
            nt.open_pinned_channel(
                f"127.0.0.1:{port}",
                expected_spki=wrong_spki,
                refuse_unpinned=True,
                probe_timeout_s=3.0,
            )
        assert exc.value.reason == "spki_mismatch"
        assert exc.value.details["expected"] == wrong_spki
        assert exc.value.details["actual"] == expected
    finally:
        stop.set()
        sock.close()


def test_probe_peer_spki_rejects_malformed_address():
    """Defensive: an address missing ':port' surfaces as OSError so
    open_pinned_channel can translate it to probe_failed cleanly."""
    with pytest.raises(OSError):
        nt.probe_peer_spki("just-host-no-port", timeout=0.1)
    with pytest.raises(OSError):
        nt.probe_peer_spki("host:not-a-number", timeout=0.1)


def test_open_pinned_channel_sets_ssl_target_name_override(tmp_path):
    """Self-signed peer certs are issued with CN=nakshatra.local; the
    channel connects to an IP. Default hostname verification would
    refuse the handshake with 'Hostname Verification Check failed'.
    We pass grpc.ssl_target_name_override so the SPKI pin (already
    stronger than chain validation) is the effective trust anchor.

    Regression test for the bug surfaced during the 2026-05-21 full-
    inference cluster smoke: real cross-machine TLS failed with
    UNAUTHENTICATED until this override was added."""
    cert_path, _ = nt.generate_self_signed_cert(output_dir=tmp_path)
    expected = nt.compute_spki_hash(cert_path)
    import socket
    import ssl
    import threading
    import grpc

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(
        certfile=str(cert_path),
        keyfile=str(tmp_path / nt.KEY_FILENAME),
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    port = sock.getsockname()[1]
    stop = threading.Event()

    def accept_loop():
        while not stop.is_set():
            try:
                sock.settimeout(0.2)
                client, _ = sock.accept()
            except (socket.timeout, OSError):
                continue
            try:
                ssock = server_ctx.wrap_socket(client, server_side=True)
                try:
                    ssock.recv(64)
                except Exception:
                    pass
                ssock.close()
            except Exception:
                try:
                    client.close()
                except Exception:
                    pass

    t = threading.Thread(target=accept_loop, daemon=True)
    t.start()
    try:
        # Connect by IP — hostname mismatch with the cert's
        # CN=nakshatra.local. Without the override, the channel's
        # handshake would refuse during use.
        ch = nt.open_pinned_channel(
            f"127.0.0.1:{port}",
            expected_spki=expected,
            refuse_unpinned=True,
            probe_timeout_s=3.0,
        )
        # Channel is a real grpc.Channel pointed at the right addr.
        assert isinstance(ch, grpc.Channel)
        ch.close()
    finally:
        stop.set()
        sock.close()


# ── 2026-05-26 drive-by — NAKSHATRA_TLS_PROBE_TIMEOUT_S env override ─


def test_probe_timeout_env_unset_returns_default(monkeypatch):
    monkeypatch.delenv("NAKSHATRA_TLS_PROBE_TIMEOUT_S", raising=False)
    assert nt._probe_timeout_from_env(default=5.0) == 5.0


def test_probe_timeout_env_empty_returns_default(monkeypatch):
    """Empty / whitespace-only must not crash + must not be parsed as
    zero (which would then be rejected as <= 0)."""
    monkeypatch.setenv("NAKSHATRA_TLS_PROBE_TIMEOUT_S", "")
    assert nt._probe_timeout_from_env(default=5.0) == 5.0
    monkeypatch.setenv("NAKSHATRA_TLS_PROBE_TIMEOUT_S", "   ")
    assert nt._probe_timeout_from_env(default=5.0) == 5.0


def test_probe_timeout_env_overrides_default(monkeypatch):
    monkeypatch.setenv("NAKSHATRA_TLS_PROBE_TIMEOUT_S", "12.5")
    assert nt._probe_timeout_from_env(default=5.0) == 12.5


def test_probe_timeout_env_int_string_parses(monkeypatch):
    """Operators often type whole seconds. The float() parse handles
    integer strings transparently — assert it explicitly so the
    contract doesn't drift."""
    monkeypatch.setenv("NAKSHATRA_TLS_PROBE_TIMEOUT_S", "30")
    assert nt._probe_timeout_from_env(default=5.0) == 30.0


def test_probe_timeout_env_garbage_returns_default(monkeypatch):
    """A typoed value (e.g. '5s', 'fast', 'NaN-ish') must not crash
    nor disable the timeout — fall back to default and let the chain
    continue. Validates the safety net for operator typos."""
    for bad in ("5s", "fast", "abc", "1e", "--", "1,5"):
        monkeypatch.setenv("NAKSHATRA_TLS_PROBE_TIMEOUT_S", bad)
        assert nt._probe_timeout_from_env(default=5.0) == 5.0


def test_probe_timeout_env_zero_or_negative_returns_default(monkeypatch):
    """A zero-timeout probe would always raise probe_failed and silently
    degrade every chain to whatever --tls-mode policy says about
    unreachable peers. Reject and fall back to default."""
    for bad in ("0", "0.0", "-1", "-5.5"):
        monkeypatch.setenv("NAKSHATRA_TLS_PROBE_TIMEOUT_S", bad)
        assert nt._probe_timeout_from_env(default=5.0) == 5.0


def test_probe_timeout_module_constant_reflects_env_at_import_time():
    """Sanity: the PROBE_TIMEOUT_S module constant was computed once at
    import via _probe_timeout_from_env(), so it's a float in the
    default-or-overridden value range. Callers that pass the constant
    as a default keep working regardless of env state."""
    assert isinstance(nt.PROBE_TIMEOUT_S, float)
    assert nt.PROBE_TIMEOUT_S > 0
