"""Verified tokenizer-only provisioning for consumer gateways.

A consumer-only node has no worker slice from which to load tokenizer metadata.
This module fetches the package's metadata GGUF without pulling any weights and
keeps the package trust boundary out of the gateway discovery loop.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from packaging.fetch_package import fetch_verified_metadata, verify_cached_metadata
from packaging.nakshatra_package import PackageError
from release_trust import ReleaseTrustError, load_release_package_signers


def _cache_stem(model_id: str) -> str:
    """Map every model id to one collision-free, path-safe cache key."""
    if not isinstance(model_id, str) or not model_id:
        raise PackageError("model_id must be a non-empty string")
    return hashlib.sha256(model_id.encode("utf-8")).hexdigest()


def _cache_root(root: str | os.PathLike[str] | None) -> Path:
    return Path(root) if root is not None else Path(os.path.expanduser(
        os.environ.get("NAKSHATRA_TOKENIZER_DIR", "~/.nakshatra/gateway/models")))


def tokenizer_path(model_id: str, root: str | os.PathLike[str] | None = None) -> Path:
    """Return the stable gateway path for a model's metadata-only GGUF."""
    return _cache_root(root) / f"{_cache_stem(model_id)}.metadata.gguf"


def tokenizer_manifest_path(
        model_id: str, root: str | os.PathLike[str] | None = None) -> Path:
    """Return the signed-manifest provenance path beside cached metadata."""
    return _cache_root(root) / f"{_cache_stem(model_id)}.package.json"


def _trusted_signers(trusted_pubkey: str | None,
                     release_pubkeys: set[str] | frozenset[str] | None) -> set[str]:
    if release_pubkeys is None:
        try:
            release_pubkeys = load_release_package_signers()
        except ReleaseTrustError as e:
            raise PackageError(str(e)) from e
    trusted = {key.strip().lower() for key in release_pubkeys
               if isinstance(key, str) and key.strip()}
    if not trusted or any(not re.fullmatch(r"[0-9a-f]{64}", key) for key in trusted):
        raise PackageError("signed release contains no valid package signer keys")
    narrow = (trusted_pubkey if trusted_pubkey is not None else
              os.environ.get("PACKAGE_TRUSTED_PUBKEY", "")).strip().lower()
    if narrow:
        if not re.fullmatch(r"[0-9a-f]{64}", narrow):
            raise PackageError("PACKAGE_TRUSTED_PUBKEY must be 64 hexadecimal characters")
        if narrow not in trusted:
            raise PackageError(
                "PACKAGE_TRUSTED_PUBKEY may only narrow to a signer trusted by the signed release")
        return {narrow}
    return trusted


def fetch_model_tokenizer(model_id: str, *, package_url: str | None = None,
                          trusted_pubkey: str | None = None,
                          release_pubkeys: set[str] | frozenset[str] | None = None,
                          root: str | os.PathLike[str] | None = None) -> str:
    """Fetch and atomically install one verified metadata-only tokenizer."""
    location = (package_url if package_url is not None else
                os.environ.get("PACKAGE_URL", "")).strip()
    if not location:
        raise PackageError("tokenizer fetch needs PACKAGE_URL")
    trusted = _trusted_signers(trusted_pubkey, release_pubkeys)
    dest = tokenizer_path(model_id, root)
    return fetch_verified_metadata(
        location, model_id, str(dest), trusted_pubkeys=trusted,
        cache_manifest_path=str(tokenizer_manifest_path(model_id, root)))


def verify_model_tokenizer(model_id: str, *, trusted_pubkey: str | None = None,
                           release_pubkeys: set[str] | frozenset[str] | None = None,
                           root: str | os.PathLike[str] | None = None) -> str:
    """Return a cached tokenizer only after fresh signature, size, and hash checks."""
    trusted = _trusted_signers(trusted_pubkey, release_pubkeys)
    return verify_cached_metadata(
        tokenizer_manifest_path(model_id, root), model_id, tokenizer_path(model_id, root),
        trusted_pubkeys=trusted)
