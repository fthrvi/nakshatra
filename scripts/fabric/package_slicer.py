"""
package_slicer — the PLANNER↔PACKAGER bridge: turn a planner layer assignment into a REAL, loader-
ready sub-GGUF, assembled on demand from a content-addressed layer package.

Slice 1's planner emitted `sub_gguf_path` strings pointing at PRE-CUT files — so the assignment was
only ever the boundary someone had already sliced. This removes that constraint: given a model's
content-addressed package (the v1.0 §5 packager output: per-layer fragments + shared
metadata/embeddings/head, each SHA-pinned), `PackageSlicer.slice_for(worker, start, end, model)`
fetches exactly the fragments that range needs and reassembles a loader-ready sub-GGUF — for ANY
contiguous range, with NO peer holding that exact pre-cut slice. So the planner can assign any split
(injected partition_fn, compute-aware later) and the slices materialise to match.

Composes, doesn't rebuild: the heavy lifting is `packaging/fetch_package.fetch_and_assemble`
(streaming SHA verify, fail-closed, signature policy). This adds:
  • content-addressed CACHING keyed by the package REVISION (immutable, content-derived), so a
    model update invalidates stale slices and a re-plan of the same range is instant;
  • the `slice_for(worker, start, end, model_id)` signature serve_planner.plan_chain expects;
  • signature policy passthrough (require_signature / trusted_pubkeys) — a stranger's package is
    refused unless its manifest is signed by a trusted key.

assemble_fn / manifest_reader are injectable so this tests without real GGUFs.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable, Optional

_PKG = str(Path(__file__).resolve().parents[1])  # scripts/
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


def _default_manifest_reader(location: str):
    """(model_id, revision, n_layers) for the package at `location` — a dir, package.json, or URL."""
    from packaging.fetch_package import _read_manifest
    pkg, _root = _read_manifest(location)
    return pkg.model_id, pkg.revision, pkg.n_layers


def _default_assemble_fn(location, start, end, dest, *, require_signature, trusted_pubkeys):
    from packaging.fetch_package import fetch_and_assemble
    return fetch_and_assemble(location, start, end, dest,
                              require_signature=require_signature, trusted_pubkeys=trusted_pubkeys)


class PackageSlicer:
    """Assemble (and cache) a model's layer slices from a content-addressed package.

    location: package dir, package.json path, or http(s) base.
    cache_dir: where assembled sub-GGUFs are kept (content-addressed by revision+range).
    require_signature / trusted_pubkeys: the package signature policy (fail-closed).
    """

    def __init__(self, location: str, *, cache_dir: Optional[str] = None,
                 require_signature: bool = False, trusted_pubkeys: Optional[set] = None,
                 assemble_fn: Optional[Callable] = None,
                 manifest_reader: Optional[Callable] = None):
        self.location = location
        self.cache_dir = Path(cache_dir or (Path.home() / ".nakshatra" / "slices"))
        self.require_signature = require_signature
        self.trusted_pubkeys = trusted_pubkeys
        self._assemble = assemble_fn or _default_assemble_fn
        self._read_manifest = manifest_reader or _default_manifest_reader
        self._model_id = None
        self._revision = None
        self.n_layers = None

    def _ensure_manifest(self):
        if self._revision is None:
            self._model_id, self._revision, self.n_layers = self._read_manifest(self.location)
        return self._revision

    def dest_for(self, model_id: str, start: int, end: int) -> Path:
        """Content-addressed slice path: a re-plan of the same (revision, range) reuses it; a model
        update (new revision) writes a fresh file rather than silently serving stale weights."""
        rev = self._ensure_manifest()
        return self.cache_dir / f"{model_id}@{rev[:12]}-L{start}-{end}.gguf"

    def slice_for(self, worker, start: int, end: int, model_id: str) -> str:
        """serve_planner.plan_chain hook: return a loader-ready sub-GGUF for [start,end), assembling
        it from the package on first use and reusing the cached copy thereafter."""
        dest = self.dest_for(model_id, start, end)
        if dest.exists() and dest.stat().st_size > 0:
            return str(dest)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self._assemble(self.location, start, end, str(dest),
                              require_signature=self.require_signature,
                              trusted_pubkeys=self.trusted_pubkeys)


# ── minimal model→package registry (for serve-time wiring in slice 3) ───────────────────────────────
def resolve_package_location(model_id: str, *, registry_path: Optional[str] = None) -> Optional[str]:
    """Look up a model's package location from ~/.nakshatra/packages.yaml (model_id -> dir/URL).
    Returns None if unregistered — the caller decides whether that's fatal (default-deny upstream)."""
    import yaml
    path = Path(registry_path or (Path.home() / ".nakshatra" / "packages.yaml"))
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    entries = data.get("packages", data) if isinstance(data, dict) else {}
    if isinstance(entries, dict):
        return entries.get(model_id)
    for e in entries or []:
        if e.get("model") == model_id or e.get("name") == model_id:
            return e.get("location") or e.get("package")
    return None
