"""Trust material carried by the currently installed signed node release."""
from __future__ import annotations

import base64
import errno
import fcntl
import json
import os
import re
import stat
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


RELEASE_MANIFEST_SCHEMA = "nak-release/1"
PACKAGE_SIGNER_SCHEMA = "nak-package-signers/1"
TRANSITION_SCHEMA = "nak-release-transition/1"
TRANSITION_FILE = "release-transition.json"
_ED25519_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")


class ReleaseTrustError(RuntimeError):
    """The installed release cannot establish package-signing trust."""


def _open_dir(name: str, *, dir_fd: int) -> int:
    return os.open(
        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=dir_fd)


def _load_json_at(name: str, *, dir_fd: int, private: bool = False) -> object:
    fd = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{name} is not a regular file")
        if private and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError(f"{name} is not private to its owner")
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            return json.load(stream)
    finally:
        if fd >= 0:
            os.close(fd)


def _version_tuple(version: object) -> tuple[int, ...]:
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError(f"release version is malformed: {version!r}")
    return tuple(int(part) for part in version.split("."))


def _installer_lock_is_held(*, root_fd: int) -> bool:
    """Return whether the prefix's exclusive installer flock is held."""
    try:
        fd = os.open(
            ".install.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("installer lock is not a regular file")
        try:
            # Observers share this probe lock.  Only a real installer takes LOCK_EX, so
            # overlapping readers cannot impersonate an installer to one another.
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return True
            raise
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
    finally:
        os.close(fd)


def _load_release_package_policy(
        prefix: str | os.PathLike[str] | None = None) -> dict:
    """Verify the installed release manifest and return its package policy.

    ``config.json`` is the installer's committed release state. ``current`` must
    be exactly the relative link ``releases/<version>``. The selected directory
    and manifest are opened beneath the prefix with directory file descriptors
    and ``O_NOFOLLOW`` so a link swap cannot redirect trust outside the prefix.
    A private installer transition journal supplies the pinned key on first
    install and authorizes exactly one target during update/rollback only while
    the prefix's installer flock is demonstrably held. Same-uid integrity is out
    of scope: filesystem permissions and flock do not constrain that attacker.
    """
    root = Path(prefix) if prefix is not None else Path(os.path.expanduser(
        os.environ.get("NAK_NODE_PREFIX", "~/.nakshatra-node")))
    try:
        root_fd = os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            target = os.readlink("current", dir_fd=root_fd)
            parts = target.split("/")
            if (len(parts) != 2 or parts[0] != "releases"
                    or not parts[1] or parts[1] in (".", "..")):
                raise ValueError(
                    "current must be exactly the relative link releases/<version>")
            installed_version = parts[1]
            _version_tuple(installed_version)
            releases_fd = _open_dir("releases", dir_fd=root_fd)
            try:
                release_fd = _open_dir(installed_version, dir_fd=releases_fd)
                try:
                    manifest = _load_json_at("manifest.json", dir_fd=release_fd)
                    if not isinstance(manifest, dict):
                        raise ValueError("manifest.json must contain an object")
                finally:
                    os.close(release_fd)
            finally:
                os.close(releases_fd)
            try:
                config = _load_json_at("config.json", dir_fd=root_fd)
            except FileNotFoundError:
                config = {}
            if not isinstance(config, dict):
                raise ValueError("config.json must contain an object")
            committed = config.get("current")
            if committed is not None:
                _version_tuple(committed)
            if installed_version == committed:
                release_key = config.get("pubkey")
            else:
                try:
                    state_fd = _open_dir("state", dir_fd=root_fd)
                    try:
                        transition = _load_json_at(
                            TRANSITION_FILE, dir_fd=state_fd, private=True)
                    finally:
                        os.close(state_fd)
                except FileNotFoundError:
                    # Commit writes config before removing the record. If those two reads raced,
                    # a fresh config snapshot establishes the now-committed target.
                    config = _load_json_at("config.json", dir_fd=root_fd)
                    if not isinstance(config, dict) or config.get("current") != installed_version:
                        raise ValueError(
                            f"current release {installed_version} is not the committed installer version")
                    release_key = config.get("pubkey")
                else:
                    if not isinstance(transition, dict) or transition.get("schema") != TRANSITION_SCHEMA \
                            or set(transition) != {"schema", "pubkey", "from", "to"}:
                        raise ValueError("release transition record is malformed")
                    transition_from, transition_to = transition["from"], transition["to"]
                    if transition_from is not None:
                        _version_tuple(transition_from)
                    _version_tuple(transition_to)
                    if transition_from == transition_to:
                        raise ValueError("release transition does not change versions")
                    if transition_from != committed or transition_to != installed_version:
                        raise ValueError(
                            f"current release {installed_version} is not the exact installer transition target")
                    if not _installer_lock_is_held(root_fd=root_fd):
                        raise ValueError(
                            "release transition is stale because no installer holds the prefix lock")
                    release_key = transition.get("pubkey")
                    configured_key = config.get("pubkey")
                    if configured_key is not None and configured_key != release_key:
                        raise ValueError("release transition key conflicts with config.json")
        finally:
            os.close(root_fd)
        if not isinstance(release_key, str) or not _ED25519_PUBLIC_KEY.fullmatch(release_key):
            raise ValueError("pinned release public key is malformed")
        if manifest.get("schema") != RELEASE_MANIFEST_SCHEMA:
            raise ValueError("release manifest schema is unsupported")
        if manifest.get("version") != installed_version:
            raise ValueError("release manifest does not name the installed current version")
        signature = manifest["sig"]
        if not isinstance(signature, dict) or signature.get("alg") != "Ed25519":
            raise ValueError("release manifest signature is malformed")
        canonical = json.dumps(
            {key: value for key, value in manifest.items() if key != "sig"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        raw_signature = base64.b64decode(signature["value"], validate=True)
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(release_key)).verify(
            raw_signature, canonical)
        policy = manifest["package_signers"]
        if not isinstance(policy, dict) or policy.get("schema") != PACKAGE_SIGNER_SCHEMA:
            raise ValueError("release package signer policy is missing or unsupported")
        keys = policy.get("keys")
        if not isinstance(keys, list) or not keys or any(
                not isinstance(key, str) or not _ED25519_PUBLIC_KEY.fullmatch(key)
                for key in keys):
            raise ValueError("release package signer allowlist is malformed")
        if len(keys) != len(set(keys)):
            raise ValueError("release package signer allowlist contains duplicates")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            InvalidSignature) as e:
        raise ReleaseTrustError(
            f"installed release at {root} cannot establish package trust: {e}") from e
    return policy


def load_release_package_signers(
        prefix: str | os.PathLike[str] | None = None) -> frozenset[str]:
    """Verify the installed release and return its package signer allowlist."""
    return frozenset(_load_release_package_policy(prefix)["keys"])


def load_release_package_catalog(
        prefix: str | os.PathLike[str] | None = None) -> dict[str, dict]:
    """Return the hardware catalog carried by the verified signed release."""
    policy = _load_release_package_policy(prefix)
    packages = policy.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ReleaseTrustError("installed signed release has no worker package catalog")
    catalog = {}
    required_names = {"gpu_vram_bytes", "cpu_ram_bytes", "disk_free_bytes"}
    for package_id, definition in packages.items():
        if not isinstance(package_id, str) or not _PACKAGE_ID.fullmatch(package_id):
            raise ReleaseTrustError("installed signed release has an invalid worker package id")
        if not isinstance(definition, dict) or not isinstance(definition.get("requirements"), dict):
            raise ReleaseTrustError(
                f"installed signed release has invalid requirements for package {package_id}")
        requirements = definition["requirements"]
        if set(requirements) != required_names or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in requirements.values()):
            raise ReleaseTrustError(
                f"installed signed release has invalid requirements for package {package_id}")
        catalog[package_id] = {
            "layers": definition.get("layers"),
            "requirements": dict(requirements),
            "requirements_basis": definition.get("requirements_basis"),
        }
    return catalog
