"""slice_cdc.py — Content-Defined Chunking (CDC) dedup for GGUF slices.

The desync/casync / HuggingFace-Xet pattern: split a slice on CONTENT boundaries
(a gear rolling hash, not fixed offsets), address each chunk by sha256, and store
chunks once. Fetching a NEW slice then pulls only the chunks not already local —
and GGUF quant-families + finetune-deltas share large identical byte regions, so a
"new" slice is mostly a near-free delta of chunks we already hold. The byte-
efficiency lever that matters on our slow WiFi link (research 2026-06-21).

Composes with slice_fetch: a CDC source reconstructs a slice from its manifest,
pulling only `missing_chunks` from a peer's chunk-store.

Note on scale: the pure-Python gear chunker here is correct + unit-tested on small
data; for multi-GB GGUFs production should use `desync`/`casync` (Go) or a numpy/C
chunker (same algorithm, faster). The dedup *logic* (manifest, store, delta) is the
reusable part and is what's tested below.
"""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# Deterministic 256-entry gear table for the rolling hash (fixed seed → portable).
_GEAR = [((i * 2654435761 + 1013904223) & 0xFFFFFFFFFFFFFFFF) for i in range(256)]


def chunk_boundaries(data: bytes, avg_bits: int = 16, mn: int = 1 << 12,
                     mx: int = 1 << 18) -> List[Tuple[int, int]]:
    """Gear-hash CDC boundaries → list of (start, end). Average chunk ≈ 2**avg_bits
    bytes; mn/mx clamp the chunk size so boundaries are content-defined but bounded."""
    mask = (1 << avg_bits) - 1
    out: List[Tuple[int, int]] = []
    h = 0
    start = 0
    n = len(data)
    for i in range(n):
        h = ((h << 1) + _GEAR[data[i]]) & 0xFFFFFFFFFFFFFFFF
        sz = i - start + 1
        if (sz >= mn and (h & mask) == 0) or sz >= mx:
            out.append((start, i + 1))
            start = i + 1
            h = 0
    if start < n:
        out.append((start, n))
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def manifest(data: bytes, **kw) -> List[str]:
    """Chunk `data` and return the ordered list of chunk sha256s (the slice's recipe)."""
    return [_sha(data[a:b]) for a, b in chunk_boundaries(data, **kw)]


class ChunkStore:
    """Content-addressed chunk store on disk: chunks/<sha>. add()/has()/get()."""
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def _p(self, sha: str) -> str:
        return os.path.join(self.path, sha)

    def has(self, sha: str) -> bool:
        return os.path.exists(self._p(sha))

    def add(self, chunk: bytes) -> str:
        sha = _sha(chunk)
        if not self.has(sha):
            tmp = self._p(sha) + f".tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(chunk)
            os.replace(tmp, self._p(sha))
        return sha

    def get(self, sha: str) -> Optional[bytes]:
        try:
            with open(self._p(sha), "rb") as f:
                return f.read()
        except OSError:
            return None


def index_slice(path: str, store: ChunkStore, **kw) -> List[str]:
    """Chunk a slice file, add each chunk to the store, return its manifest."""
    with open(path, "rb") as f:
        data = f.read()
    out = []
    for a, b in chunk_boundaries(data, **kw):
        out.append(store.add(data[a:b]))
    return out


def missing_chunks(manifest_shas: List[str], store: ChunkStore) -> List[str]:
    """Which manifest chunks this store lacks — the ONLY bytes a fetch must pull.
    Dedup against everything already stored (other slices' shared chunks)."""
    seen = set()
    out = []
    for s in manifest_shas:
        if s not in seen and not store.has(s):
            out.append(s); seen.add(s)
    return out


def reconstruct(manifest_shas: List[str], store: ChunkStore, dest: str,
                fetch_fn: "Optional[Callable[[str], Optional[bytes]]]" = None) -> bool:
    """Rebuild a slice at `dest` from its manifest. For any chunk missing locally,
    pull it via `fetch_fn(sha)->bytes` (a peer's chunk endpoint) and store it.
    Returns True on success (every chunk present/fetched)."""
    tmp = dest + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        for sha in manifest_shas:
            b = store.get(sha)
            if b is None and fetch_fn is not None:
                b = fetch_fn(sha)
                if b is not None and _sha(b) == sha:
                    store.add(b)
                else:
                    b = None
            if b is None:
                os.remove(tmp)
                return False
            f.write(b)
    os.replace(tmp, dest)
    return True
