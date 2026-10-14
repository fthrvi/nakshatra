"""Trust material carried by the currently installed signed node release."""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


RELEASE_MANIFEST_SCHEMA = "nak-release/1"
PACKAGE_SIGNER_SCHEMA = "nak-package-signers/1"
_ED25519_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")


class ReleaseTrustError(RuntimeError):
    """The installed release cannot establish package-signing trust."""


def load_release_package_signers(
        prefix: str | os.PathLike[str] | None = None) -> frozenset[str]:
    """Verify the installed release manifest and return its package signer allowlist.

    ``config.json`` is the installer's pinned release-key root. The manifest is
    accepted only when that key verifies it and it names the installed current
    version. Package signer keys are read only from the verified manifest.
    """
    root = Path(prefix) if prefix is not None else Path(os.path.expanduser(
        os.environ.get("NAK_NODE_PREFIX", "~/.nakshatra-node")))
    try:
        config = json.loads((root / "config.json").read_text())
        manifest = json.loads((root / "current" / "manifest.json").read_text())
        release_key = config["pubkey"]
        if not isinstance(release_key, str) or not _ED25519_PUBLIC_KEY.fullmatch(release_key):
            raise ValueError("pinned release public key is malformed")
        if manifest.get("schema") != RELEASE_MANIFEST_SCHEMA:
            raise ValueError("release manifest schema is unsupported")
        if manifest.get("version") != config.get("current"):
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
    return frozenset(keys)
