"""slice_fetch.py — ensure a model's GGUF layer-slice is PRESENT on this node
before warm/load, fetched content-addressed + verified, cheapest source first.

This is the layer beneath slice_warm: warm() speeds the load of a slice already
on disk; this gets the slice onto the node in the first place. Answers the open
design question — "what does warm() do when the slice isn't in cache?":

  1. LOCAL cache  — already present + verified → instant (the common case;
                    download-once means we almost always stop here).
  2. MESH PEER    — a roster/directory node that holds this exact slice, pulled
                    over the WireGuard mesh. The RIGHT source on WiFi/away: a
                    peer on the local mesh is far cheaper than the WAN/HF origin.
  3. ORIGIN       — HuggingFace/ollama, LAST resort, only for a slice no peer
                    has yet (then we slice + cache + seed it to the mesh).

Content-addressed: the slice filename encodes model@<hash>-L<a>-<b>, and we
verify GGUF magic (+ an optional per-slice sha256) on arrival before atomically
placing it. A verified slice is cached forever and never re-fetched.

Sources are pluggable `(name, fetch_fn)` pairs where `fetch_fn(ref, dest_tmp)
-> bool` writes the bytes to dest_tmp. Real mesh/HF fetchers drop in here; the
resolution/verify/atomic-place logic is source-agnostic and unit-tested.
"""
from __future__ import annotations
import hashlib
import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

GGUF_MAGIC = b"GGUF"
FetchFn = Callable[["SliceRef", str], bool]
Source = Tuple[str, FetchFn]


@dataclass(frozen=True)
class SliceRef:
    """Identity of one layer-slice. `model_hash` is the model's content hash;
    `sha256` (optional) is the per-slice integrity hash for end-to-end verify."""
    model: str
    model_hash: str
    layer_start: int
    layer_end: int
    sha256: Optional[str] = None

    @property
    def filename(self) -> str:
        return f"{self.model}@{self.model_hash}-L{self.layer_start}-{self.layer_end}.gguf"


def cache_path(cache_dir: str, ref: SliceRef) -> str:
    return os.path.join(cache_dir, ref.filename)


def _sha256(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(path: str, ref: SliceRef) -> bool:
    """A slice is valid iff it's a real GGUF and (if a sha is known) matches it.
    Without a sha we accept any well-formed non-empty GGUF (best we can do until
    the directory carries per-slice hashes)."""
    try:
        if os.path.getsize(path) <= 4:
            return False
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return False
    except OSError:
        return False
    if ref.sha256:
        return _sha256(path) == ref.sha256
    return True


def ensure_present(ref: SliceRef, cache_dir: str, sources: List[Source],
                   log: Callable[[str], None] = print) -> str:
    """Return a local path to the verified slice, fetching from the first source
    that yields valid bytes. Raises FileNotFoundError if no source can provide it.
    Idempotent: a cache hit returns instantly without touching the network."""
    dest = cache_path(cache_dir, ref)
    if os.path.exists(dest) and verify(dest, ref):
        log(f"[slice-fetch] cache hit: {ref.filename}")
        return dest
    os.makedirs(cache_dir, exist_ok=True)
    last_err: Optional[str] = None
    for name, fetch in sources:
        tmp = f"{dest}.tmp.{os.getpid()}"
        try:
            ok = bool(fetch(ref, tmp))
        except Exception as e:               # a source failing must not abort the chain
            last_err = f"{name}: {e}"
            log(f"[slice-fetch] source {name} errored: {e}")
            ok = False
        if ok and verify(tmp, ref):
            os.replace(tmp, dest)            # atomic publish into the cache
            log(f"[slice-fetch] fetched {ref.filename} from {name} "
                f"({os.path.getsize(dest)/1e9:.2f}GB)")
            return dest
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        if ok:
            last_err = f"{name}: bad bytes (hash/format mismatch)"
            log(f"[slice-fetch] {name} returned invalid bytes — trying next source")
    raise FileNotFoundError(
        f"slice {ref.filename} unavailable from any source ({last_err or 'no sources'})")


# ── pluggable source builders ────────────────────────────────────────
def local_dir_source(dirpath: str, name: str = "local-dir") -> Source:
    """Copy from another local directory (a shared mount, a staging dir, or a
    sibling cache). Cheapest non-cache source; also the test substrate."""
    def fetch(ref: SliceRef, dest_tmp: str) -> bool:
        src = os.path.join(dirpath, ref.filename)
        if not os.path.exists(src):
            return False
        shutil.copyfile(src, dest_tmp)
        return True
    return (name, fetch)


def mesh_peer_source(holders_fn: Callable[[SliceRef], List[str]],
                     pull_fn: Callable[[str, SliceRef, str], bool],
                     name: str = "mesh-peer") -> Source:
    """Resolve peers that hold the slice (from the roster/TTL directory) and pull
    from the first that succeeds. `holders_fn(ref)` -> list of peer URLs/hosts
    (nearest-first); `pull_fn(peer, ref, dest_tmp)` -> bool does the transfer
    (HTTP GET over the mesh, scp, aria2c, …). The RIGHT source on WiFi: a mesh
    peer is local-network, unlike the WAN origin."""
    def fetch(ref: SliceRef, dest_tmp: str) -> bool:
        for peer in holders_fn(ref):
            try:
                if pull_fn(peer, ref, dest_tmp):
                    return True
            except Exception:
                continue
        return False
    return (name, fetch)


def origin_source(pull_fn: Callable[["SliceRef", str], bool],
                  name: str = "origin") -> Source:
    """Last-resort origin (HuggingFace/ollama): fetch the base + slice it, or pull
    a pre-sliced artifact. Wrapped so its failure just falls through the chain."""
    def fetch(ref: SliceRef, dest_tmp: str) -> bool:
        return bool(pull_fn(ref, dest_tmp))
    return (name, fetch)


# ── real mesh transport (HTTP, pairs with slice_server.py) ───────────
def http_pull(peer: str, ref: SliceRef, dest_tmp: str,
              timeout: float = 30.0, chunk: int = 8 << 20) -> bool:
    """Stream a slice from a peer running slice_server. `peer` is host:port (or a
    full base URL). Returns True iff the file landed with the expected size."""
    import urllib.request
    base = peer if peer.startswith("http") else f"http://{peer}"
    url = f"{base.rstrip('/')}/slice/{ref.filename}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        expect = int(r.headers.get("Content-Length") or 0)
        got = 0
        with open(dest_tmp, "wb") as f:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
                got += len(b)
    return got > 0 and (expect == 0 or got == expect)


def static_holders(peers: List[str]) -> Callable[["SliceRef"], List[str]]:
    """Simplest holders_fn: try a fixed peer list in order (a 404 on /slice just
    falls through to the next). No directory query needed — robust + good enough
    until the TTL directory lands."""
    def holders(ref: SliceRef) -> List[str]:
        return list(peers)
    return holders


def make_ensure_fn(refs: List[SliceRef], cache_dir: str,
                   peers: "Optional[List[str]]" = None,
                   holders_fn: "Optional[Callable[[SliceRef], List[str]]]" = None,
                   origin_pull: "Optional[Callable[[SliceRef, str], bool]]" = None,
                   log: Callable[[str], None] = print) -> Callable[[], List[str]]:
    """Build the standard fetch-if-absent chain (cache → mesh peers → origin) for
    a model's slices and return an ensure_fn() that the ChainLifecycle calls on
    summon. Returns the local paths of all `refs`, fetching any that are missing.
    Peer resolution: pass `holders_fn` (e.g. SliceDirectory.to_holders_fn() — the
    live TTL directory) for dynamic who-holds-what, OR `peers` for a static list.
    `origin_pull` is the last-resort HF/ollama fetcher (omit until wired)."""
    sources: List[Source] = []
    if holders_fn is not None:
        sources.append(mesh_peer_source(holders_fn, http_pull))
    elif peers:
        sources.append(mesh_peer_source(static_holders(peers), http_pull))
    if origin_pull is not None:
        sources.append(origin_source(origin_pull))

    def ensure() -> List[str]:
        return [ensure_present(r, cache_dir, sources, log=log) for r in refs]
    return ensure
