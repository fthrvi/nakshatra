#!/usr/bin/env python3
"""fetch_package.py — verified fetch + assemble of a Nakshatra layer package.

Given a package manifest and an assigned layer range [start, end), this:
  1. loads + validates the manifest (and verifies its Ed25519 signature under
     the caller's policy — require/trusted-keys),
  2. selects exactly the fragments this position needs (metadata + embeddings
     iff start==0 + head iff end==n + layers[start,end)),
  3. fetches each fragment from the package root (local dir or http(s) base),
     streaming SHA-256 and **failing closed** on any mismatch,
  4. reassembles them into a single loader-ready sub-GGUF that carries the same
     `nakshatra.layer_range_*` metadata a partial_gguf.py cut would — so the
     patched loader, scan_cache_dir, and the existing peer-fetch path all keep
     working unchanged.

The result is byte-for-byte interchangeable (tensor set + KVs) with the v0.x
monolithic sub-GGUF, but provisioned from content-addressed per-layer fragments.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib import request as urlrequest

import numpy as np
from gguf import GGUFReader, GGUFWriter

if __package__:
    from ._gguf_kv import copy_kvs, read_arch_and_block_count
    from .nakshatra_package import NakshatraPackage, PackageError, MANIFEST_FILENAME
else:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from packaging._gguf_kv import copy_kvs, read_arch_and_block_count
    from packaging.nakshatra_package import NakshatraPackage, PackageError, MANIFEST_FILENAME

_CHUNK = 8 * 1024 * 1024


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _read_manifest(location: str) -> tuple[NakshatraPackage, str]:
    """Load a manifest from a dir, a package.json path, or an http(s) URL.
    Returns (package, root) where root is the base to resolve artifacts against."""
    if _is_url(location):
        url = location if location.endswith(".json") else location.rstrip("/") + "/" + MANIFEST_FILENAME
        with urlrequest.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        root = url.rsplit("/", 1)[0] + "/"
    else:
        p = Path(location)
        if p.is_dir():
            p = p / MANIFEST_FILENAME
        text = p.read_text()
        root = str(p.parent)
    pkg = NakshatraPackage.from_json(text)
    pkg.validate()
    return pkg, root


def _fetch_one(root: str, rel_path: str, dest: Path) -> str:
    """Fetch a single artifact to dest, returning its streamed SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    if _is_url(root):
        url = root.rstrip("/") + "/" + rel_path
        with urlrequest.urlopen(url, timeout=120) as resp, open(dest, "wb") as out:
            for chunk in iter(lambda: resp.read(_CHUNK), b""):
                out.write(chunk)
                h.update(chunk)
    else:
        src = Path(root) / rel_path
        with open(src, "rb") as f, open(dest, "wb") as out:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                out.write(chunk)
                h.update(chunk)
    return h.hexdigest()


def _check_signature(pkg: NakshatraPackage, require_signature: bool,
                     trusted_pubkeys: set[str] | None) -> None:
    has_sig = bool(pkg.signature_b64)
    if require_signature and not has_sig:
        raise PackageError("manifest is unsigned but a signature is required")
    if has_sig:
        if not pkg.verify_signature():
            raise PackageError("manifest signature is INVALID — refusing (poisoned or tampered)")
        if trusted_pubkeys is not None and pkg.signer_pubkey_hex not in trusted_pubkeys:
            raise PackageError(
                f"manifest signed by untrusted key {pkg.signer_pubkey_hex[:16]}… "
                f"(not in the {len(trusted_pubkeys)} trusted keys)")


def fetch_and_assemble(location: str, start: int, end: int, dest_sub_gguf: str,
                       *, require_signature: bool = False,
                       trusted_pubkeys: set[str] | None = None,
                       staging_dir: str | None = None) -> str:
    """Fetch fragments for [start,end), verify each, assemble into dest_sub_gguf.
    Returns dest path on success; raises PackageError/RuntimeError on any failure.
    Fail-closed: a single SHA-256 mismatch aborts the whole assembly."""
    pkg, root = _read_manifest(location)
    _check_signature(pkg, require_signature, trusted_pubkeys)

    needed = pkg.artifacts_for_range(start, end)
    has_token_embd = (start == 0)
    has_lm_head = (end == pkg.n_layers)
    print(f"[fetch-pkg] {pkg.model_id}@{pkg.revision[:12]} range=[{start},{end}) "
          f"→ {len(needed)} fragments (embd={has_token_embd} head={has_lm_head})", flush=True)

    staging = Path(staging_dir) if staging_dir else Path(tempfile.mkdtemp(prefix="nks-pkg-"))
    staging.mkdir(parents=True, exist_ok=True)
    local: dict[str, Path] = {}
    try:
        # 1. fetch + verify every fragment (fail closed)
        for a in needed:
            lp = staging / a.path
            actual = _fetch_one(root, a.path, lp)
            if actual != a.sha256:
                raise PackageError(
                    f"SHA-256 mismatch for {a.path}: got {actual[:12]}, "
                    f"manifest says {a.sha256[:12]} — refusing (fail closed)")
            actual_size = lp.stat().st_size
            if actual_size != a.size:
                raise PackageError(
                    f"size mismatch for {a.path}: got {actual_size}, manifest says {a.size}")
            local[a.path] = lp
            print(f"[fetch-pkg]   ok {a.path} ({a.size/1e6:.1f} MB, sha={a.sha256[:12]})", flush=True)

        # 2. assemble into one loader-ready sub-GGUF
        _assemble(pkg, local, start, end, has_token_embd, has_lm_head, dest_sub_gguf)
        print(f"[fetch-pkg] assembled {dest_sub_gguf}", flush=True)
        return dest_sub_gguf
    finally:
        if staging_dir is None:
            shutil.rmtree(staging, ignore_errors=True)


def _tensors_from(path: Path) -> list:
    return list(GGUFReader(str(path)).tensors)


def _assemble(pkg: NakshatraPackage, local: dict[str, Path], start: int, end: int,
              has_token_embd: bool, has_lm_head: bool, dest: str) -> None:
    """Recombine fetched fragments into a sub-GGUF equivalent to partial_gguf.py's
    output for [start,end): KVs from metadata fragment + the position's tensors +
    the nakshatra.layer_range_* metadata the patched loader reads."""
    meta_path = local[_role_path(pkg, "metadata")]
    r_meta = GGUFReader(str(meta_path))
    arch, _ = read_arch_and_block_count(r_meta)

    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    tmp = dest + ".tmp"
    w = GGUFWriter(tmp, arch=arch)
    copy_kvs(r_meta, w)

    # Nakshatra slice metadata — read by the patched llama.cpp loader (M3).
    w.add_uint32("nakshatra.layer_range_start", start)
    w.add_uint32("nakshatra.layer_range_end", end)
    w.add_bool("nakshatra.has_token_embd", has_token_embd)
    w.add_bool("nakshatra.has_lm_head", has_lm_head)

    # Tensors in canonical order: prelude (first worker) → layers → head (last).
    # Dedup by name so a tied model (token_embd needed by both first AND last on
    # the same node) never writes a tensor twice.
    added: set[str] = set()

    def _add(tensors):
        for t in tensors:
            if t.name in added:
                continue
            w.add_tensor(t.name, np.array(t.data), raw_dtype=t.tensor_type)
            added.add(t.name)

    if has_token_embd:
        _add(_tensors_from(local[_role_path(pkg, "embeddings")]))
    for idx in range(start, end):
        _add(_tensors_from(local[pkg.layer_artifact(idx).path]))
    if has_lm_head:
        _add(_tensors_from(local[_role_path(pkg, "head")]))
        # Tied output projection lives in token_embd; the last worker pulls it
        # from the embeddings fragment (fetched for tied models, see
        # artifacts_for_range) when it wasn't already added as the first worker.
        if pkg.tied_embeddings and "token_embd.weight" not in added:
            emb = pkg.shared_by_role("embeddings")
            if emb and emb.path in local:
                _add([t for t in _tensors_from(local[emb.path])
                      if t.name == "token_embd.weight"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(tmp, dest)


def _role_path(pkg: NakshatraPackage, role: str) -> str:
    a = pkg.shared_by_role(role)
    if a is None:
        raise PackageError(f"manifest missing the {role} fragment")
    return a.path


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + assemble a Nakshatra layer-package slice.")
    ap.add_argument("location", help="package dir, package.json path, or http(s) base URL")
    ap.add_argument("start", type=int, help="first layer (inclusive)")
    ap.add_argument("end", type=int, help="one-past-last layer")
    ap.add_argument("dest", help="output sub-GGUF path")
    ap.add_argument("--require-signature", action="store_true",
                    help="refuse unsigned manifests")
    ap.add_argument("--trusted-pubkey", action="append", default=None,
                    help="only accept manifests signed by this Ed25519 hex pubkey (repeatable)")
    args = ap.parse_args()
    trusted = set(args.trusted_pubkey) if args.trusted_pubkey else None
    fetch_and_assemble(args.location, args.start, args.end, args.dest,
                       require_signature=args.require_signature, trusted_pubkeys=trusted)


if __name__ == "__main__":
    main()
