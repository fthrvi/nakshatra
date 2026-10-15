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


def _write_signed_release(prefix: Path, private_hex: str, version: str,
                          package_key: str) -> None:
    release_dir = prefix / "releases" / version
    release_dir.mkdir(parents=True)
    manifest = releasekit.sign({
        "schema": "nak-release/1",
        "version": version,
        "package_signers": {
            "schema": "nak-package-signers/1", "keys": [package_key],
        },
    }, private_hex)
    (release_dir / "manifest.json").write_text(json.dumps(manifest))


def _installed_release(tmp_path: Path) -> tuple[Path, str, str]:
    prefix = tmp_path / "node"
    release_key = ed25519.Ed25519PrivateKey.generate()
    private_hex = release_key.private_bytes_raw().hex()
    public_hex = release_key.public_key().public_bytes_raw().hex()
    package_key = "24" * 32
    _write_signed_release(prefix, private_hex, "1.2.3", package_key)
    (prefix / "current").symlink_to("releases/1.2.3")
    (prefix / "config.json").write_text(json.dumps({
        "current": "1.2.3", "pubkey": public_hex,
    }))
    return prefix, package_key, private_hex


def test_package_signers_come_from_verified_release_manifest(tmp_path):
    prefix, package_key, _private_hex = _installed_release(tmp_path)
    assert release_trust.load_release_package_signers(prefix) == frozenset({package_key})


def test_tampered_release_package_signers_are_refused(tmp_path):
    prefix, _package_key, _private_hex = _installed_release(tmp_path)
    manifest_path = prefix / "current" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["package_signers"]["keys"] = ["99" * 32]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(release_trust.ReleaseTrustError, match="cannot establish package trust"):
        release_trust.load_release_package_signers(prefix)


def test_package_trust_follows_atomic_current_switch_before_config_commit(tmp_path):
    prefix, package_key, private_hex = _installed_release(tmp_path)
    _write_signed_release(prefix, private_hex, "1.2.4", package_key)
    (prefix / "current").unlink()
    (prefix / "current").symlink_to("releases/1.2.4")

    # install.py restarts services after switching the symlink but before it
    # commits the completed version to config.json.
    config_path = prefix / "config.json"
    config = json.loads(config_path.read_text())
    config["current"] = "1.2.3"
    config_path.write_text(json.dumps(config))

    assert release_trust.load_release_package_signers(prefix) == frozenset({package_key})


def test_package_trust_refuses_manifest_not_matching_current_symlink(tmp_path):
    prefix, _package_key, _private_hex = _installed_release(tmp_path)
    manifest = (prefix / "current" / "manifest.json").read_bytes()
    release_dir = prefix / "releases" / "1.2.4"
    release_dir.mkdir(parents=True)
    (release_dir / "manifest.json").write_bytes(manifest)
    (prefix / "current").unlink()
    (prefix / "current").symlink_to("releases/1.2.4")

    with pytest.raises(
            release_trust.ReleaseTrustError,
            match="manifest does not name the installed current version"):
        release_trust.load_release_package_signers(prefix)


@pytest.mark.parametrize("target", [
    "/tmp/outside/releases/1.2.3",
    "../outside/releases/1.2.3",
    "releases/../outside/1.2.3",
])
def test_package_trust_refuses_current_links_outside_prefix(tmp_path, target):
    prefix, _package_key, _private_hex = _installed_release(tmp_path)
    (prefix / "current").unlink()
    (prefix / "current").symlink_to(target)

    with pytest.raises(release_trust.ReleaseTrustError,
                       match="exactly the relative link"):
        release_trust.load_release_package_signers(prefix)


def test_package_trust_refuses_old_signed_manifest_rollback(tmp_path):
    prefix, _package_key, private_hex = _installed_release(tmp_path)
    old_package_key = "35" * 32
    _write_signed_release(prefix, private_hex, "1.2.2", old_package_key)
    (prefix / "current").unlink()
    (prefix / "current").symlink_to("releases/1.2.2")

    with pytest.raises(release_trust.ReleaseTrustError,
                       match="older than installer state 1.2.3"):
        release_trust.load_release_package_signers(prefix)


def test_package_trust_current_swap_cannot_redirect_manifest(tmp_path, monkeypatch):
    prefix, package_key, private_hex = _installed_release(tmp_path)
    replacement_key = "46" * 32
    _write_signed_release(prefix, private_hex, "1.2.4", replacement_key)
    real_readlink = release_trust.os.readlink

    def readlink_then_swap(path, *, dir_fd=None):
        target = real_readlink(path, dir_fd=dir_fd)
        (prefix / "current").unlink()
        (prefix / "current").symlink_to("releases/1.2.4")
        return target

    monkeypatch.setattr(release_trust.os, "readlink", readlink_then_swap)

    # The raw link selected 1.2.3. Swapping `current` immediately afterward
    # must not redirect the manifest open to 1.2.4.
    assert release_trust.load_release_package_signers(prefix) == frozenset({package_key})
    assert real_readlink(prefix / "current") == "releases/1.2.4"


def test_package_trust_refuses_symlinked_manifest(tmp_path):
    prefix, _package_key, _private_hex = _installed_release(tmp_path)
    manifest = prefix / "releases" / "1.2.3" / "manifest.json"
    outside = tmp_path / "manifest.json"
    outside.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(outside)

    with pytest.raises(release_trust.ReleaseTrustError,
                       match="cannot establish package trust"):
        release_trust.load_release_package_signers(prefix)
