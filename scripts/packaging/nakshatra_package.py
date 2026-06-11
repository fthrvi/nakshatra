"""NakshatraPackage — content-addressed, manifest-verified layer packages.

This is the v1.0 §5 "layer-package weight distribution" schema (docs/
v1.0-discovery-and-distribution.md). It adapts Mesh-LLM's `model-package.json`
(per-artifact SHA-256 + immutable revision pins, `shared/` + per-layer
`layers/layer-*.gguf` fragments) to Nakshatra, and goes one step further: the
manifest may carry an **Ed25519 signature** over its canonical bytes, binding
weight provenance to the same mesh identity that signs the data plane. So a
consumer gets both content-addressing (SHA-256) *and* identity (who packaged
this), where Mesh-LLM's manifest has only the former.

Design rules (from the doc):
  • relative-only artifact paths — never absolute, never `..` (poisoned-path guard)
  • per-artifact sha256 + byte size; fail-closed on any mismatch downstream
  • `revision` is content-derived (a hash over the sorted artifact set), so it is
    immutable by construction — change any fragment and the revision changes.

This module is pure schema + crypto. The packager (package_gguf.py) emits these;
the fetcher (fetch_package.py) consumes + verifies them.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "package.json"

KIND_SHARED = "shared"   # embeddings / output head / metadata GGUFs
KIND_LAYER = "layer"     # a single transformer block, layers/layer-NNN.gguf

# Roles for shared artifacts — drive *position-aware* fetch so a worker pulls
# exactly what partial_gguf.py would keep for its slice:
ROLE_METADATA = "metadata"      # KV-only GGUF; every worker needs it
ROLE_EMBEDDINGS = "embeddings"  # token_embd (+ rope_freqs); first worker only (start==0)
ROLE_HEAD = "head"              # output.weight + output_norm.weight; last worker only (end==n_layers)
_SHARED_ROLES = (ROLE_METADATA, ROLE_EMBEDDINGS, ROLE_HEAD)


class PackageError(Exception):
    """Manifest is malformed, unsafe, or fails verification."""


# ── artifact ──────────────────────────────────────────────────────────

@dataclass
class Artifact:
    """One fetchable fragment in a package."""
    path: str                       # relative, e.g. "layers/layer-007.gguf"
    sha256: str                     # 64 hex chars
    size: int                       # bytes
    kind: str                       # KIND_SHARED | KIND_LAYER
    layer_idx: Optional[int] = None  # set iff kind == KIND_LAYER
    role: Optional[str] = None       # set iff kind == KIND_SHARED (_SHARED_ROLES)

    def validate(self) -> None:
        _check_relative_path(self.path)
        if not (isinstance(self.sha256, str) and len(self.sha256) == 64):
            raise PackageError(f"artifact {self.path!r}: sha256 must be 64 hex chars")
        try:
            int(self.sha256, 16)
        except ValueError:
            raise PackageError(f"artifact {self.path!r}: sha256 is not hex")
        if not (isinstance(self.size, int) and self.size >= 0):
            raise PackageError(f"artifact {self.path!r}: size must be a non-negative int")
        if self.kind not in (KIND_SHARED, KIND_LAYER):
            raise PackageError(f"artifact {self.path!r}: unknown kind {self.kind!r}")
        if self.kind == KIND_LAYER:
            if not isinstance(self.layer_idx, int):
                raise PackageError(f"artifact {self.path!r}: layer kind needs an int layer_idx")
            if self.role is not None:
                raise PackageError(f"artifact {self.path!r}: layer kind must not carry a role")
        if self.kind == KIND_SHARED:
            if self.layer_idx is not None:
                raise PackageError(f"artifact {self.path!r}: shared kind must not carry layer_idx")
            if self.role not in _SHARED_ROLES:
                raise PackageError(f"artifact {self.path!r}: shared kind needs a role in {_SHARED_ROLES}")


def _check_relative_path(p: str) -> None:
    """Reject absolute paths, parent escapes, NUL, and backslashes — the
    artifact path is joined under a fetch root, so it must stay there."""
    if not isinstance(p, str) or not p:
        raise PackageError("artifact path must be a non-empty string")
    if p.startswith("/") or p.startswith("\\"):
        raise PackageError(f"artifact path must be relative: {p!r}")
    if "\\" in p or "\x00" in p:
        raise PackageError(f"artifact path has illegal characters: {p!r}")
    parts = p.split("/")
    if any(seg in ("", ".", "..") for seg in parts):
        raise PackageError(f"artifact path has empty/.. segment: {p!r}")


# ── package ───────────────────────────────────────────────────────────

@dataclass
class NakshatraPackage:
    model_id: str
    arch: str
    n_layers: int                       # total block_count of the source model
    artifacts: list[Artifact]
    schema_version: int = SCHEMA_VERSION
    revision: str = ""                  # content-derived; filled by recompute_revision()
    created_unix: int = 0
    # provenance (optional — present once signed)
    signer_pubkey_hex: Optional[str] = None
    signature_b64: Optional[str] = None

    # ---- lookups ----
    def shared_artifacts(self) -> list[Artifact]:
        return [a for a in self.artifacts if a.kind == KIND_SHARED]

    def shared_by_role(self, role: str) -> Optional[Artifact]:
        for a in self.artifacts:
            if a.kind == KIND_SHARED and a.role == role:
                return a
        return None

    def layer_artifact(self, idx: int) -> Optional[Artifact]:
        for a in self.artifacts:
            if a.kind == KIND_LAYER and a.layer_idx == idx:
                return a
        return None

    def artifacts_for_range(self, start: int, end: int) -> list[Artifact]:
        """The fragments a worker assigned layers [start, end) must fetch —
        *position-aware*, matching partial_gguf.py's tensor selection:
          • metadata  — always
          • embeddings — only the first worker (start == 0)
          • head       — only the last worker (end == n_layers)
          • layers     — the per-layer fragments in [start, end)
        Raises if a required fragment is missing from the manifest."""
        if not (0 <= start < end <= self.n_layers):
            raise PackageError(
                f"range [{start},{end}) invalid for {self.n_layers}-layer model")
        out: list[Artifact] = []
        meta = self.shared_by_role(ROLE_METADATA)
        if meta is None:
            raise PackageError("manifest is missing the metadata fragment")
        out.append(meta)
        if start == 0:
            emb = self.shared_by_role(ROLE_EMBEDDINGS)
            if emb is None:
                raise PackageError("manifest is missing the embeddings fragment (needed for start==0)")
            out.append(emb)
        if end == self.n_layers:
            head = self.shared_by_role(ROLE_HEAD)
            if head is None:
                raise PackageError("manifest is missing the head fragment (needed for end==n_layers)")
            out.append(head)
        for idx in range(start, end):
            a = self.layer_artifact(idx)
            if a is None:
                raise PackageError(f"manifest is missing layer {idx}")
            out.append(a)
        return out

    # ---- canonical bytes (deterministic; excludes the signature fields) ----
    def _canonical_obj(self) -> dict:
        arts = sorted(
            (
                {k: v for k, v in asdict(a).items()
                 if not (k in ("layer_idx", "role") and v is None)}
                for a in self.artifacts
            ),
            key=lambda d: d["path"],
        )
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "arch": self.arch,
            "n_layers": self.n_layers,
            "revision": self.revision,
            "created_unix": self.created_unix,
            "artifacts": arts,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(self._canonical_obj(), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")

    def recompute_revision(self) -> str:
        """Content-derived immutable pin: sha256 over the sorted (path,sha256)
        artifact set + identity. Any fragment change ⇒ new revision."""
        h = hashlib.sha256()
        h.update(f"{self.schema_version}\n{self.model_id}\n{self.arch}\n{self.n_layers}\n".encode())
        for a in sorted(self.artifacts, key=lambda a: a.path):
            h.update(f"{a.path}\t{a.sha256}\t{a.size}\n".encode())
        self.revision = h.hexdigest()
        return self.revision

    # ---- signing (the "go further") ----
    def sign(self, priv_bytes: bytes) -> None:
        """Sign the canonical bytes with an Ed25519 mesh key (32 raw bytes).
        Recomputes the revision first so the signature covers final content."""
        self.recompute_revision()
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
        self.signer_pubkey_hex = priv.public_key().public_bytes_raw().hex()
        self.signature_b64 = base64.b64encode(
            priv.sign(self.canonical_bytes())).decode("ascii")

    def verify_signature(self) -> bool:
        """True iff a signature is present and valid for the canonical bytes.
        Never raises. (Absence of a signature is False — callers decide policy.)"""
        if not (self.signer_pubkey_hex and self.signature_b64):
            return False
        try:
            pub = ed25519.Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(self.signer_pubkey_hex))
            pub.verify(base64.b64decode(self.signature_b64, validate=True),
                       self.canonical_bytes())
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False

    # ---- validation ----
    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise PackageError(
                f"unsupported schema_version {self.schema_version} (want {SCHEMA_VERSION})")
        if not self.model_id:
            raise PackageError("model_id is required")
        if not (isinstance(self.n_layers, int) and self.n_layers > 0):
            raise PackageError("n_layers must be a positive int")
        seen_paths: set[str] = set()
        seen_layers: set[int] = set()
        for a in self.artifacts:
            a.validate()
            if a.path in seen_paths:
                raise PackageError(f"duplicate artifact path {a.path!r}")
            seen_paths.add(a.path)
            if a.kind == KIND_LAYER:
                if not (0 <= a.layer_idx < self.n_layers):
                    raise PackageError(
                        f"layer_idx {a.layer_idx} out of range for {self.n_layers} layers")
                if a.layer_idx in seen_layers:
                    raise PackageError(f"duplicate layer fragment {a.layer_idx}")
                seen_layers.add(a.layer_idx)
        # revision must match content (immutability invariant)
        claimed = self.revision
        if claimed:
            recomputed = NakshatraPackage(
                model_id=self.model_id, arch=self.arch, n_layers=self.n_layers,
                artifacts=self.artifacts, schema_version=self.schema_version,
            ).recompute_revision()
            if claimed != recomputed:
                raise PackageError(
                    f"revision mismatch: manifest claims {claimed[:12]} but content "
                    f"hashes to {recomputed[:12]} (tampered or stale)")

    # ---- (de)serialization ----
    def to_json(self) -> str:
        obj = self._canonical_obj()
        if self.signer_pubkey_hex:
            obj["signer_pubkey_hex"] = self.signer_pubkey_hex
        if self.signature_b64:
            obj["signature_b64"] = self.signature_b64
        return json.dumps(obj, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "NakshatraPackage":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise PackageError(f"manifest is not valid JSON: {e}")
        if not isinstance(obj, dict):
            raise PackageError("manifest must be a JSON object")
        try:
            arts = [
                Artifact(
                    path=a["path"], sha256=a["sha256"], size=int(a["size"]),
                    kind=a["kind"], layer_idx=a.get("layer_idx"), role=a.get("role"),
                )
                for a in obj.get("artifacts", [])
            ]
        except (KeyError, TypeError, ValueError) as e:
            raise PackageError(f"malformed artifact entry: {e}")
        pkg = cls(
            model_id=obj.get("model_id", ""),
            arch=obj.get("arch", ""),
            n_layers=int(obj.get("n_layers", 0)),
            artifacts=arts,
            schema_version=int(obj.get("schema_version", 0)),
            revision=obj.get("revision", ""),
            created_unix=int(obj.get("created_unix", 0)),
            signer_pubkey_hex=obj.get("signer_pubkey_hex"),
            signature_b64=obj.get("signature_b64"),
        )
        return pkg


def new_package(model_id: str, arch: str, n_layers: int,
                artifacts: list[Artifact]) -> NakshatraPackage:
    """Build a package, stamp the content revision, and validate it.
    `created_unix` is left 0 here (Date.now-free); stamp it at call sites that
    have a real clock if provenance time matters."""
    pkg = NakshatraPackage(model_id=model_id, arch=arch, n_layers=n_layers,
                           artifacts=list(artifacts))
    pkg.recompute_revision()
    pkg.validate()
    return pkg
