from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "release"))

import release_trust  # noqa: E402
import releasekit  # noqa: E402


def _installed_release(tmp_path: Path) -> tuple[Path, str]:
    prefix = tmp_path / "node"
    current = prefix / "current"
    current.mkdir(parents=True)
    release_key = ed25519.Ed25519PrivateKey.generate()
    private_hex = release_key.private_bytes_raw().hex()
    public_hex = release_key.public_key().public_bytes_raw().hex()
    package_key = "24" * 32
    manifest = releasekit.sign({
        "schema": "nak-release/1",
        "version": "1.2.3",
        "package_signers": {
            "schema": "nak-package-signers/1", "keys": [package_key],
        },
    }, private_hex)
    (current / "manifest.json").write_text(json.dumps(manifest))
    (prefix / "config.json").write_text(json.dumps({
        "current": "1.2.3", "pubkey": public_hex,
    }))
    return prefix, package_key


def test_package_signers_come_from_verified_release_manifest(tmp_path):
    prefix, package_key = _installed_release(tmp_path)
    assert release_trust.load_release_package_signers(prefix) == frozenset({package_key})


def test_tampered_release_package_signers_are_refused(tmp_path):
    prefix, _package_key = _installed_release(tmp_path)
    manifest_path = prefix / "current" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["package_signers"]["keys"] = ["99" * 32]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(release_trust.ReleaseTrustError, match="cannot establish package trust"):
        release_trust.load_release_package_signers(prefix)
