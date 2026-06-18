"""Tests for provenance.py — the build-provenance fingerprint contract.
Pure: no GPU, no real daemon (a tiny stub script stands in for --version).
Run: python -m pytest scripts/discovery/test_provenance.py -q
"""
import os
import sys
import stat
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as P


# ── fingerprint stability + sensitivity ─────────────────────────────────────────
def test_fingerprint_is_stable():
    a = P.fingerprint_from_facts("abc123", "deadbeef", "buildhost", "Jun 10 2026 19:30:51")
    b = P.fingerprint_from_facts("abc123", "deadbeef", "buildhost", "Jun 10 2026 19:30:51")
    assert a.fingerprint == b.fingerprint
    assert a.matches(b)


def test_each_fact_changes_the_fingerprint():
    base = P.fingerprint_from_facts("engineX", "codeY", "hostZ", "tsW")
    variants = [
        P.fingerprint_from_facts("ENGINE-different", "codeY", "hostZ", "tsW"),
        P.fingerprint_from_facts("engineX", "code-different", "hostZ", "tsW"),
        P.fingerprint_from_facts("engineX", "codeY", "host-different", "tsW"),
        P.fingerprint_from_facts("engineX", "codeY", "hostZ", "ts-different"),
    ]
    for v in variants:
        assert v.fingerprint != base.fingerprint
        assert not base.matches(v)
        assert P.provenance_changed(base, v)


def test_prov_version_isolates():
    a = P.fingerprint_from_facts("e", "c", "h", "t", prov_version=1)
    b = P.fingerprint_from_facts("e", "c", "h", "t", prov_version=2)
    # different version → never comparable even though facts match
    assert not a.matches(b)


def test_unknown_stamps_are_well_defined():
    # ad-hoc build: only the binary hash is real, stamps unknown — still stable.
    a = P.fingerprint_from_facts("enginehash", None, None, None)
    b = P.fingerprint_from_facts("enginehash", P.UNKNOWN, P.UNKNOWN, P.UNKNOWN)
    assert a.fingerprint == b.fingerprint
    assert a.code_sha == P.UNKNOWN and a.build_host == P.UNKNOWN and a.built_at == P.UNKNOWN


def test_binary_hash_dominates_identity():
    # two ad-hoc builds with identical stamps but different binaries are NOT the same build
    a = P.fingerprint_from_facts("binary-A", P.UNKNOWN, P.UNKNOWN, P.UNKNOWN)
    b = P.fingerprint_from_facts("binary-B", P.UNKNOWN, P.UNKNOWN, P.UNKNOWN)
    assert not a.matches(b)


# ── --version parsing (the real daemon format) ───────────────────────────────────
def test_parse_real_version_format():
    out = "nakshatra-fabric-worker\n  sha        a1b2c3d\n  built_on   ijru-pc\n  built_at   Jun 10 2026 19:30:51\n"
    s = P.parse_daemon_version(out)
    assert s == {"code_sha": "a1b2c3d", "build_host": "ijru-pc", "built_at": "Jun 10 2026 19:30:51"}


def test_parse_version_with_unknown_stamps():
    out = "nakshatra-fabric-worker\n  sha        unknown\n  built_on   unknown\n  built_at   Jun 10 2026 19:30:51\n"
    s = P.parse_daemon_version(out)
    assert s["code_sha"] == "unknown" and s["build_host"] == "unknown"
    assert s["built_at"] == "Jun 10 2026 19:30:51"


def test_parse_version_missing_lines_default_unknown():
    s = P.parse_daemon_version("garbage\nno useful lines here\n")
    assert s == {"code_sha": P.UNKNOWN, "build_host": P.UNKNOWN, "built_at": P.UNKNOWN}


# ── file hashing + end-to-end via a stub daemon ──────────────────────────────────
def test_sha256_file_matches_known():
    import hashlib
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"hello nakshatra")
        path = f.name
    try:
        assert P.sha256_file(path) == hashlib.sha256(b"hello nakshatra").hexdigest()
    finally:
        os.unlink(path)


def _write_stub_daemon(dirpath: str, version_body: str) -> str:
    """A tiny executable that prints the given --version body, like the real daemon.
    The body lives in a sibling file the stub cats — avoids any heredoc/indent games."""
    vfile = Path(dirpath) / "version.txt"
    vfile.write_text(version_body + "\n")
    p = Path(dirpath) / "stub-daemon"
    p.write_text("#!/usr/bin/env bash\n"
                 f'if [ "$1" = "--version" ]; then cat "{vfile}"; fi\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


def test_provenance_from_daemon_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        body = "nakshatra-fabric-worker\n  sha        feedface\n  built_on   testhost\n  built_at   Jan 01 2026 00:00:00"
        bin_ = _write_stub_daemon(d, body)
        prov = P.provenance_from_daemon(bin_)
        assert prov.code_sha == "feedface"
        assert prov.build_host == "testhost"
        assert prov.built_at == "Jan 01 2026 00:00:00"
        assert prov.engine_sha256 == P.sha256_file(bin_)  # bedrock = the binary bytes
        # editing the binary (even 1 byte) changes provenance → tamper-detectable
        Path(bin_).write_text(Path(bin_).read_text() + "\n# tweak\n")
        prov2 = P.provenance_from_daemon(bin_)
        assert P.provenance_changed(prov, prov2)


def test_provenance_from_daemon_failsoft_on_bad_version():
    # a binary that isn't executable-as-expected: stamps fall back to unknown,
    # but the binary hash still anchors a valid provenance.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "not-really-a-daemon"
        p.write_bytes(b"\x00\x01\x02 not an elf")
        prov = P.provenance_from_daemon(str(p))
        assert prov.engine_sha256 == P.sha256_file(str(p))
        assert prov.code_sha == P.UNKNOWN  # couldn't read --version


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
