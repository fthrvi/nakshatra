"""Verified tokenizer-only provisioning for consumer gateways.

A consumer-only node has no worker slice from which to load tokenizer metadata.
This module fetches the package's metadata GGUF without pulling any weights and
keeps the package trust boundary out of the gateway discovery loop.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from packaging.fetch_package import fetch_verified_metadata
from packaging.nakshatra_package import PackageError


def tokenizer_path(model_id: str, root: str | os.PathLike[str] | None = None) -> Path:
    """Return the stable gateway path for a model's metadata-only GGUF."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._")
    if not safe:
        raise PackageError("model_id has no safe filename characters")
    base = Path(root) if root is not None else Path(os.path.expanduser(
        os.environ.get("NAKSHATRA_TOKENIZER_DIR", "~/.nakshatra/gateway/models")))
    return base / f"{safe}.metadata.gguf"


def fetch_model_tokenizer(model_id: str, *, package_url: str | None = None,
                          trusted_pubkey: str | None = None,
                          root: str | os.PathLike[str] | None = None) -> str:
    """Fetch and atomically install one verified metadata-only tokenizer."""
    location = (package_url if package_url is not None else
                os.environ.get("PACKAGE_URL", "")).strip()
    pubkey = (trusted_pubkey if trusted_pubkey is not None else
              os.environ.get("PACKAGE_TRUSTED_PUBKEY", "")).strip().lower()
    if not location or not pubkey:
        raise PackageError(
            "tokenizer fetch needs PACKAGE_URL and PACKAGE_TRUSTED_PUBKEY")
    if not re.fullmatch(r"[0-9a-f]{64}", pubkey):
        raise PackageError("PACKAGE_TRUSTED_PUBKEY must be 64 hexadecimal characters")
    dest = tokenizer_path(model_id, root)
    return fetch_verified_metadata(
        location, model_id, str(dest), trusted_pubkeys={pubkey})
