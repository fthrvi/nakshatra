"""Tests for scripts/nakshatra_cli.py.

Phase 2.8 of the 2026-05-21 SPKI federation sprint: minimal operator
CLI scaffold with a single `tls fingerprint` subcommand. Broader CLI
expansion (auth keygen / operator install / sign — mirror sthambha-cli)
is the dedicated operator-UX work item from the 2026-05-20 retro.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nakshatra_cli as cli  # noqa: E402
import nakshatra_tls as nt  # noqa: E402


def test_tls_fingerprint_prints_hash_for_explicit_cert(tmp_path):
    """Operator passes --cert; CLI prints the same hash compute_spki_hash
    would return, no extra formatting."""
    cert_path, _ = nt.generate_self_signed_cert(output_dir=tmp_path)
    expected = nt.compute_spki_hash(cert_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["tls", "fingerprint", "--cert", str(cert_path)])
    assert rc == 0
    assert buf.getvalue().strip() == expected


def test_tls_fingerprint_uses_default_cert_path_when_omitted(
        tmp_path, monkeypatch):
    """No --cert → fall back to ~/.nakshatra/tls/worker-cert.pem. We
    monkey-patch the module constant to a tmp_path so the test never
    touches the real user dir."""
    monkeypatch.setattr(nt, "DEFAULT_TLS_DIR", tmp_path)
    nt.ensure_cert(output_dir=tmp_path)
    expected = nt.compute_spki_hash(tmp_path / nt.CERT_FILENAME)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["tls", "fingerprint"])
    assert rc == 0
    assert buf.getvalue().strip() == expected


def test_tls_fingerprint_missing_cert_exits_2(tmp_path, monkeypatch):
    """Default path with no cert present — exit 2, error on stderr."""
    monkeypatch.setattr(nt, "DEFAULT_TLS_DIR", tmp_path)
    err = io.StringIO()
    out = io.StringIO()
    with redirect_stderr(err), redirect_stdout(out):
        rc = cli.main(["tls", "fingerprint"])
    assert rc == 2
    assert "cert not found" in err.getvalue()
    # Stdout stays empty so wrapping scripts (`spki=$(nakshatra-cli ...)`)
    # don't get noise on failure.
    assert out.getvalue() == ""


def test_tls_fingerprint_help_lists_subcommand():
    """Help output mentions the subcommand — sanity that the parser
    actually wires it."""
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "tls" in help_text


def test_no_args_errors():
    """Bare `nakshatra-cli` with no subcommand must error out, not
    silently no-op."""
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    # argparse calls sys.exit(2) on missing required subparser.
    assert exc.value.code == 2
