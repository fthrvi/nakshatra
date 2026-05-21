"""Tests for scripts/nakshatra_cli.py.

Phase 2.8 of the 2026-05-21 SPKI federation sprint shipped the single
`tls fingerprint` subcommand. The 2026-05-21 nakshatra-cli expansion
work item from the same-day retro brought the full operator surface:

  auth keygen / show-pubkey / sign
  operator install
  tls keygen
"""
from __future__ import annotations

import io
import json
import os
import stat
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nakshatra_auth as auth  # noqa: E402
import nakshatra_cli as cli  # noqa: E402
import nakshatra_tls as nt  # noqa: E402


# Most tests monkey-patch the CLI's KEYS_DIR + OPERATOR_PUBKEY_PATH +
# the TLS module's DEFAULT_TLS_DIR onto tmp_path so the real user dir
# is never touched.


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    keys_dir = tmp_path / "keys"
    tls_dir = tmp_path / "tls"
    monkeypatch.setattr(cli, "KEYS_DIR", keys_dir)
    monkeypatch.setattr(cli, "OPERATOR_PUBKEY_PATH",
                         keys_dir / "operator.pub.hex")
    monkeypatch.setattr(nt, "DEFAULT_TLS_DIR", tls_dir)
    return keys_dir, tls_dir


# ── tls fingerprint (Phase 2.8 — already shipped) ────────────────────


def test_tls_fingerprint_prints_hash_for_explicit_cert(tmp_path):
    cert_path, _ = nt.generate_self_signed_cert(output_dir=tmp_path)
    expected = nt.compute_spki_hash(cert_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["tls", "fingerprint", "--cert", str(cert_path)])
    assert rc == 0
    assert buf.getvalue().strip() == expected


def test_tls_fingerprint_uses_default_cert_path_when_omitted(
        tmp_path, monkeypatch):
    monkeypatch.setattr(nt, "DEFAULT_TLS_DIR", tmp_path)
    nt.ensure_cert(output_dir=tmp_path)
    expected = nt.compute_spki_hash(tmp_path / nt.CERT_FILENAME)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["tls", "fingerprint"])
    assert rc == 0
    assert buf.getvalue().strip() == expected


def test_tls_fingerprint_missing_cert_exits_2(tmp_path, monkeypatch):
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


def test_help_lists_all_subcommands():
    """Help output names every subcommand — sanity that the parser
    actually wires them."""
    parser = cli.build_parser()
    help_text = parser.format_help()
    for cmd in ("auth", "operator", "tls"):
        assert cmd in help_text


def test_no_args_errors():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


# ── auth keygen ──────────────────────────────────────────────────────


def test_auth_keygen_default_name_is_worker(isolated_dirs):
    """Default name 'worker' matches the convention worker.py expects
    at boot (~/.nakshatra/keys/worker.ed25519)."""
    keys_dir, _ = isolated_dirs
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["auth", "keygen"])
    assert rc == 0
    assert (keys_dir / "worker.ed25519").exists()
    assert (keys_dir / "worker.pub.hex").exists()
    assert "worker" in buf.getvalue()


def test_auth_keygen_private_key_file_mode_is_600(isolated_dirs):
    """The private key MUST be unreadable to other users from the
    moment it exists. The CLI uses O_CREAT with 0o600 baked in."""
    keys_dir, _ = isolated_dirs
    cli.main(["auth", "keygen", "--name", "k1"])
    priv_mode = (keys_dir / "k1.ed25519").stat().st_mode & 0o777
    assert priv_mode == 0o600, f"got {oct(priv_mode)}"


def test_auth_keygen_pubkey_file_matches_pub_from_priv(isolated_dirs):
    """The .pub.hex file content matches what
    public_key_hex_from_private(priv) emits — the wire-contract source."""
    keys_dir, _ = isolated_dirs
    cli.main(["auth", "keygen", "--name", "k2"])
    priv = (keys_dir / "k2.ed25519").read_bytes()
    stored_pub = (keys_dir / "k2.pub.hex").read_text().strip()
    assert stored_pub == auth.public_key_hex_from_private(priv)


def test_auth_keygen_refuses_overwrite_without_force(isolated_dirs):
    """An accidental second `auth keygen --name X` MUST refuse —
    rotating a key breaks the pillar's TOFU-locked public_key_hex
    for that worker. Operators must pass --force consciously."""
    cli.main(["auth", "keygen", "--name", "k3"])
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cli.main(["auth", "keygen", "--name", "k3"])
    assert rc == 2
    assert "already exists" in err.getvalue()
    assert "--force" in err.getvalue()


def test_auth_keygen_force_rotates(isolated_dirs):
    keys_dir, _ = isolated_dirs
    cli.main(["auth", "keygen", "--name", "k4"])
    pub_1 = (keys_dir / "k4.pub.hex").read_text().strip()
    rc = cli.main(["auth", "keygen", "--name", "k4", "--force"])
    assert rc == 0
    pub_2 = (keys_dir / "k4.pub.hex").read_text().strip()
    assert pub_1 != pub_2  # fresh keypair


def test_auth_keygen_json_output(isolated_dirs):
    keys_dir, _ = isolated_dirs
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["--json", "auth", "keygen", "--name", "k5"])
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert data["name"] == "k5"
    assert data["public_key_hex"] == (
        keys_dir / "k5.pub.hex").read_text().strip()
    assert "private_key_path" in data


# ── auth show-pubkey ─────────────────────────────────────────────────


def test_auth_show_pubkey_prints_only_hex(isolated_dirs):
    """Output is JUST the 64-char hex — wrapping scripts can
    `pub=$(nakshatra-cli auth show-pubkey)` cleanly."""
    keys_dir, _ = isolated_dirs
    cli.main(["auth", "keygen", "--name", "showme"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["auth", "show-pubkey", "--name", "showme"])
    assert rc == 0
    out = buf.getvalue().strip()
    assert len(out) == 64
    int(out, 16)  # valid hex
    assert out == (keys_dir / "showme.pub.hex").read_text().strip()


def test_auth_show_pubkey_missing_key_exits_2(isolated_dirs):
    err = io.StringIO()
    out = io.StringIO()
    with redirect_stderr(err), redirect_stdout(out):
        rc = cli.main(["auth", "show-pubkey", "--name", "ghost"])
    assert rc == 2
    assert "key not found" in err.getvalue()
    assert "auth keygen" in err.getvalue()  # remediation hint
    assert out.getvalue() == ""


# ── auth sign ────────────────────────────────────────────────────────


def test_auth_sign_produces_valid_authorization_header(isolated_dirs):
    """The header CLI emits parses back through nakshatra_auth's
    verify_request. Cross-contract test against the wire format."""
    cli.main(["auth", "keygen", "--name", "signer"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main([
            "auth", "sign", "--name", "signer",
            "--method", "POST", "--path", "/peer",
            "--body", '{"node_id":"w1"}',
        ])
    assert rc == 0
    header = buf.getvalue().strip()
    # Format: Sthambha-Ed25519 keyid="...",sig="...",ts="..."
    assert header.startswith('Sthambha-Ed25519 keyid="signer",sig="')
    assert ',ts="' in header


def test_auth_sign_verifies_against_pubkey(isolated_dirs):
    """Sig the CLI generates verifies with verify_request using the
    same canonical-string inputs. Wire-contract belt-and-suspenders."""
    keys_dir, _ = isolated_dirs
    cli.main(["auth", "keygen", "--name", "signer2"])
    pub_hex = (keys_dir / "signer2.pub.hex").read_text().strip()
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main([
            "--json", "auth", "sign", "--name", "signer2",
            "--method", "POST", "--path", "/peer",
            "--body", '{"x":1}',
        ])
    payload = json.loads(buf.getvalue())
    # Parse the header
    import re
    m = re.match(
        r'Sthambha-Ed25519 keyid="([^"]*)",sig="([^"]*)",ts="(\d+)"',
        payload["header"],
    )
    assert m, f"unexpected header shape: {payload['header']!r}"
    _kid, sig_b64, ts_str = m.groups()
    ok = auth.verify_request(
        pub_hex, "POST", "/peer", b'{"x":1}',
        int(ts_str), sig_b64,
    )
    assert ok is True


def test_auth_sign_uses_keyid_override(isolated_dirs):
    """Operators sometimes need the keyid in the header to differ from
    the local key filename (e.g. signing a request that the pillar
    expects under a different node_id than the local key name)."""
    cli.main(["auth", "keygen", "--name", "signer3"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main([
            "auth", "sign", "--name", "signer3", "--keyid", "fancy-node",
            "--method", "GET", "--path", "/peers",
        ])
    assert 'keyid="fancy-node"' in buf.getvalue()


def test_auth_sign_grpc_method_path_supported(isolated_dirs):
    """The contract is method/path-agnostic — gRPC paths
    (/nakshatra.Nakshatra/Forward with method tokens POST/STREAM)
    should sign cleanly. Test the documented use case."""
    cli.main(["auth", "keygen", "--name", "signer4"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main([
            "auth", "sign", "--name", "signer4",
            "--method", "STREAM",
            "--path", "/nakshatra.Nakshatra/Inference",
        ])
    assert rc == 0
    assert "Sthambha-Ed25519" in buf.getvalue()


# ── operator install ────────────────────────────────────────────────


def test_operator_install_with_pubkey_writes_file(isolated_dirs):
    keys_dir, _ = isolated_dirs
    pub = "a" * 64
    rc = cli.main(["operator", "install", "--pubkey", pub])
    assert rc == 0
    op_path = keys_dir / "operator.pub.hex"
    assert op_path.read_text().strip() == pub


def test_operator_install_pubkey_file_mode_is_600(isolated_dirs):
    keys_dir, _ = isolated_dirs
    cli.main(["operator", "install", "--pubkey", "b" * 64])
    mode = (keys_dir / "operator.pub.hex").stat().st_mode & 0o777
    assert mode == 0o600


def test_operator_install_with_from_key(isolated_dirs):
    """--from-key reads pubkey from a local key file (convenient when
    the operator generated the key on this machine)."""
    keys_dir, _ = isolated_dirs
    cli.main(["auth", "keygen", "--name", "opkey"])
    pub_hex = (keys_dir / "opkey.pub.hex").read_text().strip()
    rc = cli.main(["operator", "install", "--from-key", "opkey"])
    assert rc == 0
    assert (keys_dir / "operator.pub.hex").read_text().strip() == pub_hex


def test_operator_install_rejects_no_input(isolated_dirs):
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cli.main(["operator", "install"])
    assert rc == 2
    assert "--pubkey" in err.getvalue() or "--from-key" in err.getvalue()


def test_operator_install_rejects_both_inputs(isolated_dirs):
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cli.main(["operator", "install",
                       "--pubkey", "a" * 64,
                       "--from-key", "x"])
    assert rc == 2
    assert "mutually exclusive" in err.getvalue()


def test_operator_install_rejects_wrong_length_pubkey(isolated_dirs):
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cli.main(["operator", "install", "--pubkey", "abc"])
    assert rc == 2
    assert "64 hex chars" in err.getvalue()


def test_operator_install_rejects_non_hex_pubkey(isolated_dirs):
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cli.main(["operator", "install", "--pubkey", "z" * 64])
    assert rc == 2
    assert "valid hex" in err.getvalue()


def test_operator_install_replaces_existing_pubkey(isolated_dirs):
    """Operator key rotation: second `install` replaces the first.
    Unlike the worker key (TOFU-locked at the pillar), the operator
    key has no remote dependency — rotation is a pure filesystem op."""
    keys_dir, _ = isolated_dirs
    cli.main(["operator", "install", "--pubkey", "1" * 64])
    cli.main(["operator", "install", "--pubkey", "2" * 64])
    assert (keys_dir / "operator.pub.hex").read_text().strip() == "2" * 64


# ── tls keygen ──────────────────────────────────────────────────────


def test_tls_keygen_writes_cert_and_key(isolated_dirs):
    _, tls_dir = isolated_dirs
    rc = cli.main(["tls", "keygen"])
    assert rc == 0
    assert (tls_dir / nt.CERT_FILENAME).exists()
    assert (tls_dir / nt.KEY_FILENAME).exists()


def test_tls_keygen_refuses_overwrite_without_force(isolated_dirs):
    cli.main(["tls", "keygen"])
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cli.main(["tls", "keygen"])
    assert rc == 2
    assert "refusing to overwrite" in err.getvalue()
    assert "--force" in err.getvalue()


def test_tls_keygen_force_rotates(isolated_dirs):
    _, tls_dir = isolated_dirs
    cli.main(["tls", "keygen"])
    spki_1 = nt.compute_spki_hash(tls_dir / nt.CERT_FILENAME)
    rc = cli.main(["tls", "keygen", "--force"])
    assert rc == 0
    spki_2 = nt.compute_spki_hash(tls_dir / nt.CERT_FILENAME)
    assert spki_1 != spki_2  # fresh keypair → fresh SPKI


def test_tls_keygen_json_output(isolated_dirs):
    _, tls_dir = isolated_dirs
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["--json", "tls", "keygen"])
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert data["cert_path"].endswith(nt.CERT_FILENAME)
    assert data["key_path"].endswith(nt.KEY_FILENAME)
    assert len(data["spki_sha256_hex"]) == 64
    int(data["spki_sha256_hex"], 16)  # valid hex


def test_tls_keygen_fingerprint_roundtrip(isolated_dirs):
    """Keygen then fingerprint return the same hash. The user-facing
    flow: `nakshatra-cli tls keygen` → `nakshatra-cli tls fingerprint`
    is the operator's manual-cert install path."""
    _, tls_dir = isolated_dirs
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main(["--json", "tls", "keygen"])
    keygen_spki = json.loads(buf.getvalue())["spki_sha256_hex"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main(["tls", "fingerprint"])
    fingerprint_spki = buf.getvalue().strip()
    assert keygen_spki == fingerprint_spki
