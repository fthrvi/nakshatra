"""Tests for worker.setup_tls — drive-by refactor from 2026-05-26.

setup_tls(args) factors the ~60 LOC TLS bringup block out of main() so
it can be unit-tested directly (and so main() reads top-to-bottom
instead of disappearing into the cert-init weeds). Behavior must be
byte-identical to the inline block it replaced; these tests guard the
contract.

Run with `pytest --noconftest tests/test_worker_setup_tls.py` per the
established hardening-test pattern.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import worker as w  # noqa: E402


def _args(pillar_url: str = "") -> types.SimpleNamespace:
    """Minimal args namespace that setup_tls touches. Real argparse
    namespace carries dozens more fields; setup_tls only reads one."""
    return types.SimpleNamespace(pillar_url=pillar_url)


# ── tls_required path ─────────────────────────────────────────────────


def test_setup_tls_required_path_returns_cert_paths_and_hash(monkeypatch):
    """The happy path: pillar configured + TLS required + ensure_cert
    returns successfully → returns TlsBoot(True, cert, key, hash) and
    emits the audit event for operator visibility."""
    monkeypatch.setattr(w, "_TLS_AVAILABLE", True)
    fake_wtls = MagicMock()
    fake_wtls.resolve_tls_required.return_value = True
    fake_wtls.ensure_cert.return_value = (
        Path("/etc/n/cert.pem"), Path("/etc/n/key.pem"), "a" * 64,
    )
    monkeypatch.setattr(w, "_wtls", fake_wtls)

    with patch.object(w, "_audit") as mock_audit:
        boot = w.setup_tls(_args(pillar_url="http://pillar"))

    assert boot.required is True
    assert boot.cert_path == Path("/etc/n/cert.pem")
    assert boot.key_path == Path("/etc/n/key.pem")
    assert boot.spki_hash == "a" * 64
    # Audit emission is part of the contract — operators expect to
    # see tls_cert_ready in the log after every successful boot.
    mock_audit.assert_any_call(
        "tls_cert_ready",
        cert_path="/etc/n/cert.pem",
        spki_sha256="a" * 64,
    )


def test_setup_tls_file_exists_error_exits(monkeypatch):
    """Half-rotated cert state (one of cert.pem / key.pem present,
    the other missing) → ensure_cert raises FileExistsError → we
    sys.exit so the operator notices BEFORE we rotate one side and
    silently break every pinned peer."""
    monkeypatch.setattr(w, "_TLS_AVAILABLE", True)
    fake_wtls = MagicMock()
    fake_wtls.resolve_tls_required.return_value = True
    fake_wtls.ensure_cert.side_effect = FileExistsError("half-rotated state")
    monkeypatch.setattr(w, "_wtls", fake_wtls)

    with pytest.raises(SystemExit) as ei:
        w.setup_tls(_args(pillar_url="http://pillar"))
    assert "TLS bringup refused" in str(ei.value)


def test_setup_tls_generic_failure_exits(monkeypatch):
    """Any other ensure_cert exception → sys.exit. Don't let the
    worker boot in plaintext when the operator asked for TLS — fail
    loud."""
    monkeypatch.setattr(w, "_TLS_AVAILABLE", True)
    fake_wtls = MagicMock()
    fake_wtls.resolve_tls_required.return_value = True
    fake_wtls.ensure_cert.side_effect = PermissionError("can't write")
    monkeypatch.setattr(w, "_wtls", fake_wtls)

    with pytest.raises(SystemExit) as ei:
        w.setup_tls(_args(pillar_url="http://pillar"))
    assert "TLS bringup failed" in str(ei.value)


# ── tls_disabled paths (each emits a different WARN) ─────────────────


def test_setup_tls_disabled_with_pillar_warns_and_audits(monkeypatch,
                                                            capsys):
    """Pillar configured but NAKSHATRA_TLS_REQUIRED explicitly false
    → Mode-A escape hatch. Print operator WARN + emit audit event +
    return TlsBoot(False, ...)."""
    monkeypatch.setattr(w, "_TLS_AVAILABLE", True)
    fake_wtls = MagicMock()
    fake_wtls.resolve_tls_required.return_value = False
    monkeypatch.setattr(w, "_wtls", fake_wtls)

    with patch.object(w, "_audit") as mock_audit:
        boot = w.setup_tls(_args(pillar_url="http://pillar"))

    assert boot == w.TlsBoot(False, None, None, "")
    out = capsys.readouterr().out
    assert "NAKSHATRA_TLS_REQUIRED is off" in out
    mock_audit.assert_any_call(
        "tls_disabled_with_pillar", pillar_url="http://pillar",
    )


def test_setup_tls_no_pillar_no_warn(monkeypatch, capsys):
    """Standalone worker (no pillar, no TLS required) → silent return.
    Don't spam operators running localhost smokes with WARNs about a
    pillar they never configured."""
    monkeypatch.setattr(w, "_TLS_AVAILABLE", True)
    fake_wtls = MagicMock()
    fake_wtls.resolve_tls_required.return_value = False
    monkeypatch.setattr(w, "_wtls", fake_wtls)

    boot = w.setup_tls(_args(pillar_url=""))

    assert boot == w.TlsBoot(False, None, None, "")
    assert "NAKSHATRA_TLS_REQUIRED" not in capsys.readouterr().out


def test_setup_tls_module_missing_warns(monkeypatch, capsys):
    """Legacy worker missing the nakshatra_tls module on PYTHONPATH →
    WARN at boot, return TlsBoot(False, ...). Boot continues in
    plaintext (the legacy pre-Phase-2 bringup path)."""
    monkeypatch.setattr(w, "_TLS_AVAILABLE", False)
    # _TLS_IMPORT_ERR only exists when the real import failed; on the
    # dev machine the import succeeds, so we have to add the attribute
    # rather than replace it.
    monkeypatch.setattr(w, "_TLS_IMPORT_ERR",
                        ImportError("No module named 'nakshatra_tls'"),
                        raising=False)

    boot = w.setup_tls(_args(pillar_url=""))

    assert boot == w.TlsBoot(False, None, None, "")
    assert "nakshatra_tls import failed" in capsys.readouterr().out


# ── return-shape guarantees ──────────────────────────────────────────


def test_tlsboot_unpacks_in_field_order():
    """The downstream call site does `tls.required`, `tls.cert_path`
    etc. by attribute, but the NamedTuple also unpacks positionally.
    Verify the field order so a future refactor that adds a field
    doesn't silently shift positions."""
    boot = w.TlsBoot(True, Path("/c"), Path("/k"), "h")
    assert boot.required is True
    assert boot.cert_path == Path("/c")
    assert boot.key_path == Path("/k")
    assert boot.spki_hash == "h"
    # Positional unpacking still works (some callers might).
    req, cert, key, h = boot
    assert (req, cert, key, h) == (True, Path("/c"), Path("/k"), "h")
