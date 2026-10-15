from __future__ import annotations

import contextlib
import fcntl
import json
import stat
import sys
import threading
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "release"))

import release_trust  # noqa: E402
import releasekit  # noqa: E402
import install as release_install  # noqa: E402


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


def _write_transition(prefix: Path, pubkey: str, from_version: str | None,
                      to_version: str) -> Path:
    state = prefix / "state"
    state.mkdir(exist_ok=True)
    path = state / release_trust.TRANSITION_FILE
    path.write_text(json.dumps({
        "schema": release_trust.TRANSITION_SCHEMA,
        "pubkey": pubkey,
        "from": from_version,
        "to": to_version,
    }))
    path.chmod(0o600)
    return path


@contextlib.contextmanager
def _held_installer_lock(prefix: Path):
    lock = prefix / ".install.lock"
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


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
    config = json.loads((prefix / "config.json").read_text())
    transition = _write_transition(prefix, config["pubkey"], "1.2.3", "1.2.4")
    (prefix / "current").unlink()
    (prefix / "current").symlink_to("releases/1.2.4")

    with _held_installer_lock(prefix):
        assert release_trust.load_release_package_signers(prefix) == frozenset({package_key})
    assert stat.S_IMODE(transition.stat().st_mode) == 0o600


def test_package_trust_accepts_first_install_transition_without_config(tmp_path):
    prefix = tmp_path / "node"
    release_key = ed25519.Ed25519PrivateKey.generate()
    private_hex = release_key.private_bytes_raw().hex()
    public_hex = release_key.public_key().public_bytes_raw().hex()
    package_key = "24" * 32
    _write_signed_release(prefix, private_hex, "1.2.3", package_key)
    _write_transition(prefix, public_hex, None, "1.2.3")
    (prefix / "current").symlink_to("releases/1.2.3")

    with _held_installer_lock(prefix):
        assert release_trust.load_release_package_signers(prefix) == frozenset({package_key})


def test_package_trust_refuses_manifest_not_matching_current_symlink(tmp_path):
    prefix, _package_key, _private_hex = _installed_release(tmp_path)
    config = json.loads((prefix / "config.json").read_text())
    manifest = (prefix / "current" / "manifest.json").read_bytes()
    release_dir = prefix / "releases" / "1.2.4"
    release_dir.mkdir(parents=True)
    (release_dir / "manifest.json").write_bytes(manifest)
    _write_transition(prefix, config["pubkey"], "1.2.3", "1.2.4")
    (prefix / "current").unlink()
    (prefix / "current").symlink_to("releases/1.2.4")

    with _held_installer_lock(prefix):
        with pytest.raises(
                release_trust.ReleaseTrustError,
                match="manifest does not name the installed current version"):
            release_trust.load_release_package_signers(prefix)


def test_package_trust_refuses_transition_after_installer_lock_disappears(tmp_path):
    prefix, _package_key, private_hex = _installed_release(tmp_path)
    old_package_key = "35" * 32
    _write_signed_release(prefix, private_hex, "1.2.2", old_package_key)
    config = json.loads((prefix / "config.json").read_text())
    _write_transition(prefix, config["pubkey"], "1.2.3", "1.2.2")
    (prefix / "current").unlink()
    (prefix / "current").symlink_to("releases/1.2.2")

    with pytest.raises(release_trust.ReleaseTrustError, match="transition is stale"):
        release_trust.load_release_package_signers(prefix)

    with _held_installer_lock(prefix):
        assert release_trust.load_release_package_signers(prefix) == frozenset({old_package_key})


def test_overlapping_readers_cannot_impersonate_live_installer(tmp_path, monkeypatch):
    prefix, _package_key, private_hex = _installed_release(tmp_path)
    old_package_key = "35" * 32
    _write_signed_release(prefix, private_hex, "1.2.2", old_package_key)
    config = json.loads((prefix / "config.json").read_text())
    _write_transition(prefix, config["pubkey"], "1.2.3", "1.2.2")
    (prefix / "current").unlink()
    (prefix / "current").symlink_to("releases/1.2.2")
    (prefix / ".install.lock").touch(mode=0o600)

    real_flock = release_trust.fcntl.flock
    reader_a_inside_probe = threading.Event()
    release_reader_a = threading.Event()
    reader_a = None

    def pause_reader_a(fd, operation):
        real_flock(fd, operation)
        if threading.current_thread() is reader_a and operation & fcntl.LOCK_NB:
            reader_a_inside_probe.set()
            assert release_reader_a.wait(5.0), "reader A was not released"

    monkeypatch.setattr(release_trust.fcntl, "flock", pause_reader_a)
    reader_a_errors = []

    def read_stale_transition():
        try:
            release_trust.load_release_package_signers(prefix)
        except Exception as exc:
            reader_a_errors.append(exc)

    reader_a = threading.Thread(target=read_stale_transition, name="reader-a")
    reader_a.start()
    assert reader_a_inside_probe.wait(5.0), "reader A did not enter its lock probe"
    try:
        with pytest.raises(release_trust.ReleaseTrustError, match="transition is stale"):
            release_trust.load_release_package_signers(prefix)
    finally:
        release_reader_a.set()
        reader_a.join(timeout=5.0)

    assert not reader_a.is_alive()
    assert len(reader_a_errors) == 1
    assert isinstance(reader_a_errors[0], release_trust.ReleaseTrustError)
    assert "transition is stale" in str(reader_a_errors[0])


def test_real_installer_exclusive_lock_is_visible_to_trust_reader(tmp_path):
    prefix, _package_key, private_hex = _installed_release(tmp_path)
    old_package_key = "35" * 32
    _write_signed_release(prefix, private_hex, "1.2.2", old_package_key)
    config = json.loads((prefix / "config.json").read_text())
    _write_transition(prefix, config["pubkey"], "1.2.3", "1.2.2")
    (prefix / "current").unlink()
    (prefix / "current").symlink_to("releases/1.2.2")
    installer = release_install.Installer(prefix, systemd=False, make_venv=False)

    with installer._install_lock():
        assert release_trust.load_release_package_signers(prefix) == frozenset({old_package_key})

    with pytest.raises(release_trust.ReleaseTrustError, match="transition is stale"):
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
                       match="not the committed installer version"):
        release_trust.load_release_package_signers(prefix)


def test_package_trust_refuses_unrecorded_forward_switch(tmp_path):
    prefix, _package_key, private_hex = _installed_release(tmp_path)
    _write_signed_release(prefix, private_hex, "1.2.4", "35" * 32)
    next_current = prefix / "current.next"
    next_current.symlink_to("releases/1.2.4")
    next_current.replace(prefix / "current")

    with pytest.raises(release_trust.ReleaseTrustError,
                       match="not the committed installer version"):
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


def test_package_trust_refuses_symlinked_releases_directory(tmp_path):
    prefix, _package_key, _private_hex = _installed_release(tmp_path)
    real_releases = prefix / "real-releases"
    (prefix / "releases").rename(real_releases)
    (prefix / "releases").symlink_to(real_releases)

    with pytest.raises(release_trust.ReleaseTrustError,
                       match="cannot establish package trust"):
        release_trust.load_release_package_signers(prefix)


def test_package_trust_refuses_symlinked_version_directory(tmp_path):
    prefix, _package_key, _private_hex = _installed_release(tmp_path)
    release = prefix / "releases" / "1.2.3"
    real_release = prefix / "real-release"
    release.rename(real_release)
    release.symlink_to(real_release)

    with pytest.raises(release_trust.ReleaseTrustError,
                       match="cannot establish package trust"):
        release_trust.load_release_package_signers(prefix)
