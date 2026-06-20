#!/usr/bin/env python3
"""
package_tool.py — operator ergonomics for layer-packages: VALIDATE before you publish, CERTIFY
(sign) when it's clean. Sourced from mesh-llm's validate-package / preflight / certify toolchain,
re-cut to our signed/pinned posture (Ed25519 over the canonical manifest, not an open registry).

We already have the packager (`package_gguf.py`) and the verifying fetcher (`fetch_package.py`,
fail-closed on any sha/size mismatch). What was missing is the PRE-publish operator side:

  validate <manifest> [--root DIR]    structural + completeness + (with --root) on-disk sha256/size
                                      match. Catches a package that drifted from its manifest BEFORE
                                      a consumer's fetch refuses it.
  certify  <manifest> --key FILE      validate fully, then sign the manifest with the operator key so
           [--root DIR] [--out FILE]  consumers can pin provenance. Refuses to sign an invalid package.

Pure: no network, no GPU. Fail-closed — any problem is reported and (for certify) blocks signing.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # so `import nakshatra_package` works standalone

from nakshatra_package import (  # noqa: E402
    NakshatraPackage, PackageError, ROLE_METADATA, ROLE_EMBEDDINGS, ROLE_HEAD,
)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_package(pkg: NakshatraPackage, root: Optional[Path] = None) -> List[str]:
    """Return a list of problems (empty == valid). Checks, in order:
       1. per-artifact structural validity (path/sha/size/kind)
       2. no duplicate artifact paths
       3. completeness: every layer [0, n_layers) present; required shared roles present
       4. (if root) every artifact's on-disk bytes match its declared sha256 + size
    """
    problems: List[str] = []

    # 1. structural (delegates to each artifact's own validate)
    for a in pkg.artifacts:
        try:
            a.validate()
        except PackageError as e:
            problems.append(f"artifact: {e}")

    # 2. duplicate paths
    paths = [a.path for a in pkg.artifacts]
    dupes = sorted({p for p in paths if paths.count(p) > 1})
    for p in dupes:
        problems.append(f"duplicate artifact path: {p!r}")

    # 3. completeness — all layers + the shared roles a chain needs
    if pkg.n_layers <= 0:
        problems.append(f"n_layers must be > 0 (got {pkg.n_layers})")
    else:
        missing = [i for i in range(pkg.n_layers) if pkg.layer_artifact(i) is None]
        if missing:
            head = ", ".join(map(str, missing[:8])) + (" …" if len(missing) > 8 else "")
            problems.append(f"missing {len(missing)} layer artifact(s): [{head}] of {pkg.n_layers}")
    for role in (ROLE_METADATA, ROLE_EMBEDDINGS, ROLE_HEAD):
        if pkg.shared_by_role(role) is None:
            problems.append(f"missing required shared artifact role: {role!r}")

    # 4. on-disk integrity (drift check) — what the consumer's fetch would later enforce, run NOW
    if root is not None:
        for a in pkg.artifacts:
            fp = root / a.path
            if not fp.is_file():
                problems.append(f"file not found for artifact: {a.path}")
                continue
            actual_size = fp.stat().st_size
            if actual_size != a.size:
                problems.append(f"size mismatch {a.path}: on-disk {actual_size} != manifest {a.size}")
            actual_sha = _sha256_file(fp)
            if actual_sha != a.sha256:
                problems.append(f"sha256 mismatch {a.path}: on-disk {actual_sha[:12]}… != "
                                f"manifest {a.sha256[:12]}…")

    return problems


def _load(manifest: str) -> NakshatraPackage:
    return NakshatraPackage.from_json(Path(manifest).read_text())


def _load_priv(key_file: str) -> bytes:
    """Operator key: a file of raw-hex Ed25519 private key (same format as ~/.neuron/pillar/operator_key
    and the admission/identity keys). Returns raw 32 bytes for pkg.sign()."""
    raw = Path(key_file).read_text().strip()
    try:
        b = bytes.fromhex(raw)
    except ValueError:
        raise PackageError(f"key file {key_file} is not raw-hex Ed25519 private key")
    if len(b) != 32:
        raise PackageError(f"key file {key_file}: expected 32-byte (64-hex) private key, got {len(b)} bytes")
    return b


def cmd_validate(args) -> int:
    pkg = _load(args.manifest)
    root = Path(args.root) if args.root else None
    problems = validate_package(pkg, root)
    sig = "unsigned"
    if pkg.signature_b64:
        sig = ("signed+VALID by " + (pkg.signer_pubkey_hex or "?")[:16] + "…") if pkg.verify_signature() \
              else "signed but signature INVALID"
        if "INVALID" in sig:
            problems.append("manifest signature does not verify")
    if args.require_signature and not pkg.signature_b64:
        problems.append("manifest is unsigned but --require-signature was given")
    label = f"{pkg.model_id}@{(pkg.revision or '?')[:12]} ({pkg.n_layers} layers, {len(pkg.artifacts)} artifacts; {sig})"
    if problems:
        print(f"[pkg] INVALID — {label}")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print(f"[pkg] VALID — {label}" + ("" if root else "  (structural only; pass --root to verify bytes)"))
    return 0


def cmd_certify(args) -> int:
    pkg = _load(args.manifest)
    root = Path(args.root) if args.root else None
    problems = validate_package(pkg, root)
    if problems:
        print("[pkg] refusing to certify — package is INVALID:")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    pkg.sign(_load_priv(args.key))
    out = args.out or args.manifest
    Path(out).write_text(pkg.to_json())
    ok = pkg.verify_signature()
    print(f"[pkg] CERTIFIED {pkg.model_id}@{(pkg.revision or '?')[:12]} → {out}")
    print(f"       signer {pkg.signer_pubkey_hex[:16]}…  signature-verifies={ok}")
    return 0 if ok else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="validate / certify a nakshatra layer-package manifest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="structural + completeness + (with --root) on-disk byte check")
    v.add_argument("manifest"); v.add_argument("--root", default=None, help="package dir to byte-verify against")
    v.add_argument("--require-signature", action="store_true")
    v.set_defaults(fn=cmd_validate)
    c = sub.add_parser("certify", help="validate then Ed25519-sign the manifest (operator provenance)")
    c.add_argument("manifest"); c.add_argument("--key", required=True, help="raw-hex Ed25519 operator key file")
    c.add_argument("--root", default=None); c.add_argument("--out", default=None, help="write to (default: in place)")
    c.set_defaults(fn=cmd_certify)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (PackageError, FileNotFoundError) as e:
        print(f"[pkg] error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
