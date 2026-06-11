#!/usr/bin/env python3
"""package_gguf.py — fragment a full GGUF into a content-addressed layer package.

This is the v1.0 §5 packager. It splits a source GGUF into:

    package.json            (the NakshatraPackage manifest, optionally signed)
    shared/metadata.gguf    (all model KVs, no tensors — every worker needs it)
    shared/embeddings.gguf  (token_embd + rope_freqs etc.; first worker only)
    shared/output.gguf      (output.weight + output_norm.weight; last worker only)
    layers/layer-000.gguf   (blk.0.* tensors)
    layers/layer-001.gguf   ...

Each fragment's SHA-256 + byte-size is recorded in the manifest; the revision is
content-derived (immutable). A worker assigned layers [s,e) later fetches exactly
metadata + (embeddings iff s==0) + (head iff e==n) + layers[s,e), verifies each
hash fail-closed, and reassembles a loader-ready sub-GGUF (see fetch_package.py).

Tensor → fragment classification mirrors partial_gguf.py:
    blk.N.*                       → layers/layer-N
    output.weight, output_norm.*  → shared/output  (head)
    everything else top-level     → shared/embeddings (token_embd, rope_freqs, …)

Usage:
    python -m packaging.package_gguf SRC.gguf OUT_DIR [--model-id ID]
                                     [--sign-key NAME | --sign-worker-key]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter

# Support both `python -m packaging.package_gguf` and direct execution.
if __package__:
    from ._gguf_kv import copy_kvs, read_arch_and_block_count
    from .nakshatra_package import (Artifact, new_package, KIND_SHARED, KIND_LAYER,
                                    ROLE_METADATA, ROLE_EMBEDDINGS, ROLE_HEAD)
else:  # pragma: no cover - direct-run convenience
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from packaging._gguf_kv import copy_kvs, read_arch_and_block_count
    from packaging.nakshatra_package import (Artifact, new_package, KIND_SHARED, KIND_LAYER,
                                            ROLE_METADATA, ROLE_EMBEDDINGS, ROLE_HEAD)

HEAD_TENSORS = {"output.weight", "output_norm.weight"}


def _layer_index(name: str) -> int | None:
    if not name.startswith("blk."):
        return None
    try:
        return int(name.split(".")[1])
    except (IndexError, ValueError):
        return None


def _write_fragment(path: Path, arch: str, tensors: list, *, with_kvs_from=None) -> tuple[str, int]:
    """Write a GGUF fragment with the given tensors (and, if a reader is passed,
    all of its model KVs). Returns (sha256_hex, size_bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(path), arch=arch)
    if with_kvs_from is not None:
        copy_kvs(with_kvs_from, w)
    for t in tensors:
        w.add_tensor(t.name, np.array(t.data), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def package_gguf(src: Path, out_dir: Path, model_id: str,
                 sign_priv: bytes | None = None) -> Path:
    print(f"[package] reading {src}", flush=True)
    r = GGUFReader(str(src))
    arch, block_count = read_arch_and_block_count(r)
    if block_count is None:
        sys.exit("could not read block_count from source GGUF metadata")
    print(f"[package] arch={arch} n_layers={block_count} tensors={len(r.tensors)}", flush=True)

    # Classify tensors into prelude / head / per-layer buckets.
    prelude, head = [], []
    layers: dict[int, list] = {i: [] for i in range(block_count)}
    for t in r.tensors:
        li = _layer_index(t.name)
        if li is not None:
            if 0 <= li < block_count:
                layers[li].append(t)
            else:
                print(f"[package] WARN tensor {t.name} has out-of-range layer {li}; skipping", flush=True)
        elif t.name in HEAD_TENSORS:
            head.append(t)
        else:
            prelude.append(t)

    missing = [i for i in range(block_count) if not layers[i]]
    if missing:
        sys.exit(f"source GGUF is missing tensors for layers {missing[:8]} — not a full model")

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Artifact] = []

    # shared/metadata.gguf — all KVs, no tensors
    print("[package] writing shared/metadata.gguf (KVs)", flush=True)
    sha, size = _write_fragment(out_dir / "shared" / "metadata.gguf", arch, [], with_kvs_from=r)
    artifacts.append(Artifact("shared/metadata.gguf", sha, size, KIND_SHARED, role=ROLE_METADATA))

    # shared/embeddings.gguf — prelude tensors (token_embd, rope_freqs, …)
    print(f"[package] writing shared/embeddings.gguf ({len(prelude)} tensors)", flush=True)
    sha, size = _write_fragment(out_dir / "shared" / "embeddings.gguf", arch, prelude)
    artifacts.append(Artifact("shared/embeddings.gguf", sha, size, KIND_SHARED, role=ROLE_EMBEDDINGS))

    # shared/output.gguf — head tensors
    print(f"[package] writing shared/output.gguf ({len(head)} tensors)", flush=True)
    sha, size = _write_fragment(out_dir / "shared" / "output.gguf", arch, head)
    artifacts.append(Artifact("shared/output.gguf", sha, size, KIND_SHARED, role=ROLE_HEAD))

    # layers/layer-NNN.gguf — one per block
    for i in range(block_count):
        rel = f"layers/layer-{i:03d}.gguf"
        sha, size = _write_fragment(out_dir / rel, arch, layers[i])
        artifacts.append(Artifact(rel, sha, size, KIND_LAYER, layer_idx=i))
        if i < 2 or i == block_count - 1:
            print(f"[package] wrote {rel} ({len(layers[i])} tensors, {size/1e6:.1f} MB, sha={sha[:12]})", flush=True)

    # Weight tying: a model with no separate output.weight ties its output
    # projection to token_embd. The last worker then needs the embeddings
    # fragment too (the daemon falls back to token_embd for the tied output).
    tied = not any(t.name == "output.weight" for t in r.tensors)
    if tied:
        print("[package] weight-tied model (no output.weight) — last worker will "
              "also fetch the embeddings fragment for the tied output projection", flush=True)
    pkg = new_package(model_id, arch, block_count, artifacts, tied_embeddings=tied)
    pkg.created_unix = int(time.time())
    if sign_priv is not None:
        pkg.sign(sign_priv)
        print(f"[package] signed by {pkg.signer_pubkey_hex[:16]}…", flush=True)
    pkg.recompute_revision() if sign_priv is None else None
    pkg.validate()

    manifest_path = out_dir / "package.json"
    manifest_path.write_text(pkg.to_json())
    print(f"[package] revision {pkg.revision[:16]} → {manifest_path}", flush=True)
    print(f"[package] {len(artifacts)} artifacts, "
          f"{sum(a.size for a in artifacts)/1e9:.2f} GB total", flush=True)
    return manifest_path


def _load_sign_key(args) -> bytes | None:
    if args.sign_worker_key:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from nakshatra_auth import load_or_create_worker_key
        priv, pub = load_or_create_worker_key()
        print(f"[package] signing with worker key {pub[:16]}…", flush=True)
        return priv
    if args.sign_key:
        key_path = Path.home() / ".nakshatra" / "keys" / f"{args.sign_key}.ed25519"
        if not key_path.exists():
            sys.exit(f"sign key not found: {key_path}")
        priv = key_path.read_bytes()
        if len(priv) != 32:
            sys.exit(f"sign key malformed (expected 32 bytes): {key_path}")
        return priv
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Fragment a GGUF into a Nakshatra layer package.")
    ap.add_argument("src", type=Path, help="source full-model GGUF")
    ap.add_argument("out_dir", type=Path, help="output package directory")
    ap.add_argument("--model-id", default=None, help="model id (default: source filename stem)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sign-key", default=None, help="sign manifest with ~/.nakshatra/keys/NAME.ed25519")
    g.add_argument("--sign-worker-key", action="store_true", help="sign with the worker's Ed25519 key")
    args = ap.parse_args()

    model_id = args.model_id or args.src.stem
    sign_priv = _load_sign_key(args)
    package_gguf(args.src, args.out_dir, model_id, sign_priv)


if __name__ == "__main__":
    main()
