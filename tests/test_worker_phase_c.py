"""Tests for Phase C of the worker hardening sprint (2026-05-20).

Covers the HTTP tier model + path sanitization + operator-key handling.

Unit-tested:
  C1 — operator pubkey load (file present / missing / malformed)
  C1 — verify_http_request across tiers (anonymous / authenticated / operator)
  C4 — validate_slice_path: happy + traversal + unicode + symlink + non-file

The route-level integration (FileServerHandler do_GET / do_POST tier
gating) is verified at cluster smoke depth — these unit tests cover the
helper contracts the handler depends on.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nakshatra_auth as auth  # noqa: E402
import worker as w  # noqa: E402


# ── C1: _load_operator_pubkey ────────────────────────────────────────


def test_c1_operator_pubkey_missing_file_returns_none(tmp_path):
    path = tmp_path / "missing.hex"
    assert w._load_operator_pubkey(path) is None


def test_c1_operator_pubkey_valid_hex_returned_lowercase(tmp_path):
    path = tmp_path / "op.hex"
    path.write_text("AB" * 32 + "\n")
    assert w._load_operator_pubkey(path) == "ab" * 32


def test_c1_operator_pubkey_malformed_length_returns_none(tmp_path):
    path = tmp_path / "op.hex"
    path.write_text("abc")
    assert w._load_operator_pubkey(path) is None


def test_c1_operator_pubkey_non_hex_returns_none(tmp_path):
    path = tmp_path / "op.hex"
    path.write_text("z" * 64)
    assert w._load_operator_pubkey(path) is None


def test_c1_operator_pubkey_strips_whitespace(tmp_path):
    path = tmp_path / "op.hex"
    path.write_text("\n   " + "ab" * 32 + "  \n\n")
    assert w._load_operator_pubkey(path) == "ab" * 32


# ── C1: verify_http_request — anonymous tier no-op ───────────────────


def test_c1_anonymous_tier_no_auth_check():
    """No auth required for anonymous tier; verifier returns None."""
    result = w.verify_http_request(
        auth_header="", method="GET", path="/healthz", body=b"",
        operator_pubkey=None, peer_resolver=None,
        tier=w.TIER_ANONYMOUS,
    )
    assert result is None


# ── C1: verify_http_request — operator tier ──────────────────────────


def test_c1_operator_tier_verifies_against_operator_key():
    priv, pub = auth.generate_keypair()
    ts = int(time.time())
    sig = auth.sign_request(priv, "POST", "/slice", b'{"k":"v"}', ts)
    header = auth.build_auth_header("anyone", sig, ts)
    keyid = w.verify_http_request(
        auth_header=header, method="POST", path="/slice", body=b'{"k":"v"}',
        operator_pubkey=pub, peer_resolver=None,
        tier=w.TIER_OPERATOR,
    )
    assert keyid == "anyone"  # operator tier ignores who signed; pubkey is the gate


def test_c1_operator_tier_rejects_wrong_key():
    priv_a, _pub_a = auth.generate_keypair()
    _priv_b, pub_b = auth.generate_keypair()
    ts = int(time.time())
    sig = auth.sign_request(priv_a, "POST", "/slice", b"", ts)
    header = auth.build_auth_header("alpha", sig, ts)
    with pytest.raises(ValueError, match="operator signature mismatch"):
        w.verify_http_request(
            auth_header=header, method="POST", path="/slice", body=b"",
            operator_pubkey=pub_b, peer_resolver=None,
            tier=w.TIER_OPERATOR,
        )


def test_c1_operator_tier_refuses_without_pubkey_installed():
    priv, _pub = auth.generate_keypair()
    ts = int(time.time())
    sig = auth.sign_request(priv, "POST", "/slice", b"", ts)
    header = auth.build_auth_header("alpha", sig, ts)
    with pytest.raises(ValueError, match="no operator pubkey"):
        w.verify_http_request(
            auth_header=header, method="POST", path="/slice", body=b"",
            operator_pubkey=None, peer_resolver=None,
            tier=w.TIER_OPERATOR,
        )


# ── C1: verify_http_request — authenticated tier ─────────────────────


class _StubResolver:
    """Minimal resolver shape — matches PillarPeerKeyResolver.resolve()."""
    def __init__(self, keys: dict):
        self._keys = keys

    def resolve(self, keyid):
        return self._keys.get(keyid)


def test_c1_authenticated_tier_verifies_via_resolver():
    priv, pub = auth.generate_keypair()
    resolver = _StubResolver({"alpha": pub})
    ts = int(time.time())
    sig = auth.sign_request(priv, "GET", "/file/w0.gguf", b"", ts)
    header = auth.build_auth_header("alpha", sig, ts)
    keyid = w.verify_http_request(
        auth_header=header, method="GET", path="/file/w0.gguf", body=b"",
        operator_pubkey=None, peer_resolver=resolver,
        tier=w.TIER_AUTHENTICATED,
    )
    assert keyid == "alpha"


def test_c1_authenticated_tier_rejects_unknown_keyid():
    priv, _pub = auth.generate_keypair()
    resolver = _StubResolver({})  # empty cache
    ts = int(time.time())
    sig = auth.sign_request(priv, "GET", "/file/x", b"", ts)
    header = auth.build_auth_header("alpha", sig, ts)
    with pytest.raises(ValueError, match="unknown keyid"):
        w.verify_http_request(
            auth_header=header, method="GET", path="/file/x", body=b"",
            operator_pubkey=None, peer_resolver=resolver,
            tier=w.TIER_AUTHENTICATED,
        )


def test_c1_authenticated_tier_rejects_stale_timestamp():
    priv, pub = auth.generate_keypair()
    resolver = _StubResolver({"alpha": pub})
    stale_ts = int(time.time()) - 120
    sig = auth.sign_request(priv, "GET", "/file/x", b"", stale_ts)
    header = auth.build_auth_header("alpha", sig, stale_ts)
    with pytest.raises(ValueError, match="out of window"):
        w.verify_http_request(
            auth_header=header, method="GET", path="/file/x", body=b"",
            operator_pubkey=None, peer_resolver=resolver,
            tier=w.TIER_AUTHENTICATED,
        )


def test_c1_authenticated_tier_rejects_missing_header():
    resolver = _StubResolver({})
    with pytest.raises(ValueError, match="missing authorization"):
        w.verify_http_request(
            auth_header="", method="GET", path="/file/x", body=b"",
            operator_pubkey=None, peer_resolver=resolver,
            tier=w.TIER_AUTHENTICATED,
        )


def test_c1_authenticated_tier_refuses_when_no_resolver():
    priv, _pub = auth.generate_keypair()
    ts = int(time.time())
    sig = auth.sign_request(priv, "GET", "/file/x", b"", ts)
    header = auth.build_auth_header("alpha", sig, ts)
    with pytest.raises(ValueError, match="no peer resolver"):
        w.verify_http_request(
            auth_header=header, method="GET", path="/file/x", body=b"",
            operator_pubkey=None, peer_resolver=None,
            tier=w.TIER_AUTHENTICATED,
        )


def test_c1_signature_covers_body_for_post():
    """The signature must cover the body bytes — replay with different
    body must fail."""
    priv, pub = auth.generate_keypair()
    ts = int(time.time())
    sig = auth.sign_request(priv, "POST", "/slice", b'{"a":1}', ts)
    header = auth.build_auth_header("alpha", sig, ts)
    with pytest.raises(ValueError, match="operator signature mismatch"):
        w.verify_http_request(
            auth_header=header, method="POST", path="/slice", body=b'{"a":2}',
            operator_pubkey=pub, peer_resolver=None,
            tier=w.TIER_OPERATOR,
        )


# ── C4: validate_slice_path — happy path ─────────────────────────────


def test_c4_validate_slice_path_accepts_file_under_root(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    target = root / "llama70b.gguf"
    target.write_text("fake gguf")
    resolved = w.validate_slice_path(str(target), root)
    assert resolved == target.resolve()


def test_c4_validate_slice_path_accepts_nested_file(tmp_path):
    root = tmp_path / "models"
    nested = root / "subdir"
    nested.mkdir(parents=True)
    target = nested / "model.gguf"
    target.write_text("data")
    resolved = w.validate_slice_path(str(target), root)
    assert resolved.is_file()


# ── C4: validate_slice_path — attack paths ───────────────────────────


def test_c4_validate_slice_path_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        w.validate_slice_path("", Path("/tmp"))


def test_c4_validate_slice_path_rejects_nul_byte(tmp_path):
    with pytest.raises(ValueError, match="dangerous Unicode"):
        w.validate_slice_path(str(tmp_path / "ev\x00il.gguf"), tmp_path)


def test_c4_validate_slice_path_rejects_zero_width(tmp_path):
    """Zero-width joiner in the path — abuse signal, refuse."""
    with pytest.raises(ValueError, match="dangerous Unicode"):
        w.validate_slice_path(
            str(tmp_path / "good​bad.gguf"), tmp_path,
        )


def test_c4_validate_slice_path_rejects_bidi_override(tmp_path):
    """RTL override (U+202E) — disguises filename appearance."""
    with pytest.raises(ValueError, match="dangerous Unicode"):
        w.validate_slice_path(
            str(tmp_path / "fooo‮yzx.gguf"), tmp_path,
        )


def test_c4_validate_slice_path_rejects_math_alpha(tmp_path):
    """Mathematical Alphanumeric Symbols (U+1D400-U+1D7FF) —
    Sthambha M3 class-of-bug (e.g. bold-sans-serif 'm' visually
    identical to ASCII 'm')."""
    with pytest.raises(ValueError, match="dangerous Unicode"):
        w.validate_slice_path(
            str(tmp_path / "\U0001D43Eodel.gguf"), tmp_path,
        )


def test_c4_validate_slice_path_rejects_outside_root(tmp_path):
    """A real file that lives outside the declared root must be refused
    (this is the /etc/passwd probe path)."""
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside.gguf"
    outside.write_text("data")
    with pytest.raises(ValueError, match="must resolve under"):
        w.validate_slice_path(str(outside), root)


def test_c4_validate_slice_path_rejects_symlink_escape(tmp_path):
    """A symlink inside root that points outside root must be refused.
    `Path.resolve()` follows symlinks, so the relative_to() check
    catches this."""
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "secret.gguf"
    outside.write_text("data")
    sneaky = root / "innocent.gguf"
    sneaky.symlink_to(outside)
    with pytest.raises(ValueError, match="must resolve under"):
        w.validate_slice_path(str(sneaky), root)


def test_c4_validate_slice_path_rejects_nonexistent(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        w.validate_slice_path(str(root / "ghost.gguf"), root)


def test_c4_validate_slice_path_rejects_directory(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    subdir = root / "subdir"
    subdir.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        w.validate_slice_path(str(subdir), root)


# ── C: tier constants are stable strings ─────────────────────────────


def test_c_tier_constants():
    """Locking the tier strings as a regression guard — a silent
    rename to TIER_ADMIN or similar would break operator-tier check
    semantics."""
    assert w.TIER_ANONYMOUS == "anonymous"
    assert w.TIER_AUTHENTICATED == "authenticated"
    assert w.TIER_OPERATOR == "operator"
    # All three are distinct
    assert len({w.TIER_ANONYMOUS, w.TIER_AUTHENTICATED, w.TIER_OPERATOR}) == 3


def test_c_operator_pubkey_default_path():
    """Path is documented for operators; lock it down."""
    assert w.OPERATOR_PUBKEY_PATH == (
        Path.home() / ".nakshatra" / "keys" / "operator.pub.hex"
    )


def test_c_dangerous_unicode_includes_known_ranges():
    """Regression guard — the ranges must include at least the codepoints
    we explicitly test. Adding new ranges is fine; removing is the bug."""
    assert any(lo <= 0x202E <= hi
               for lo, hi in w._DANGEROUS_UNICODE_RANGES)  # RTL override
    assert any(lo <= 0x200B <= hi
               for lo, hi in w._DANGEROUS_UNICODE_RANGES)  # zero-width
    assert any(lo <= 0x1D400 <= hi
               for lo, hi in w._DANGEROUS_UNICODE_RANGES)  # math alpha
    assert any(lo <= 0xFEFF <= hi
               for lo, hi in w._DANGEROUS_UNICODE_RANGES)  # BOM
