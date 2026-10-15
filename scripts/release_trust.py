"""Trust material carried by the currently installed signed node release."""
from __future__ import annotations

import base64
import json
import os
import re
import stat
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


RELEASE_MANIFEST_SCHEMA = "nak-release/1"
PACKAGE_SIGNER_SCHEMA = "nak-package-signers/1"
_ED25519_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")


class ReleaseTrustError(RuntimeError):
    """The installed release cannot establish package-signing trust."""


def _open_dir(name: str, *, dir_fd: int) -> int:
    return os.open(
        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=dir_fd)


def _load_json_at(name: str, *, dir_fd: int) -> object:
    fd = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{name} is not a regular file")
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


def load_release_package_signers(
        prefix: str | os.PathLike[str] | None = None) -> frozenset[str]:
    """Verify the installed release manifest and return its package signer allowlist.

    ``config.json`` is the installer's pinned release-key root. ``current`` must
    be exactly the relative link ``releases/<version>``. The selected directory
    and manifest are opened beneath the prefix with directory file descriptors
    and ``O_NOFOLLOW`` so a link swap cannot redirect trust outside the prefix.
    During an update the link may name a version newer than config's committed
    ``current`` value, but it may never name an older signed release.
    """
    root = Path(prefix) if prefix is not None else Path(os.path.expanduser(
        os.environ.get("NAK_NODE_PREFIX", "~/.nakshatra-node")))
    try:
        root_fd = os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            config = _load_json_at("config.json", dir_fd=root_fd)
            if not isinstance(config, dict):
                raise ValueError("config.json must contain an object")
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
        finally:
            os.close(root_fd)
        release_key = config["pubkey"]
        if not isinstance(release_key, str) or not _ED25519_PUBLIC_KEY.fullmatch(release_key):
            raise ValueError("pinned release public key is malformed")
        if manifest.get("schema") != RELEASE_MANIFEST_SCHEMA:
            raise ValueError("release manifest schema is unsupported")
        if manifest.get("version") != installed_version:
            raise ValueError("release manifest does not name the installed current version")
        committed_versions = [
            config.get(name) for name in ("current", "pending", "pending_version")
            if config.get(name) is not None
        ]
        installed_order = _version_tuple(installed_version)
        for committed in committed_versions:
            if installed_order < _version_tuple(committed):
                raise ValueError(
                    f"current release {installed_version} is older than installer state "
                    f"{committed}")
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
    return frozenset(keys)
