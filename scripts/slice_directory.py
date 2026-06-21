"""slice_directory.py — TTL directory of which mesh node holds which GGUF slice.

Replaces the hand-written roster's "who-holds-what" with a self-publishing,
expiring directory (the Petals/hivemind pattern, simplified). Each node
periodically publishes the slices it serves (its slice_server's /slices) with a
timestamp; lookups return the live (non-expired) holders of a slice, freshest
first — which plugs straight into slice_fetch.mesh_peer_source as the holders_fn.
Dead nodes self-evict when their record ages past the TTL: no manual cleanup.

Backed by an optional JSON file so the directory can live on the shared mesh fs
(or be synced); in-memory otherwise. Uses wall-clock time so records compare
across processes/nodes; now_fn is injectable for deterministic TTL tests.
"""
from __future__ import annotations
import json
import os
import time
from typing import Callable, Dict, List, Optional


class SliceDirectory:
    def __init__(self, ttl_s: float = 120.0, path: "Optional[str]" = None,
                 now_fn: Callable[[], float] = time.time):
        self.ttl_s = ttl_s
        self.path = path
        self._now = now_fn
        self._records: Dict[str, dict] = {}   # node -> {ts, slices:[...], meta:{}}
        if path and os.path.exists(path):
            self._load()

    # publish / lookup ------------------------------------------------
    def publish(self, node: str, slices: List[str], meta: "Optional[dict]" = None) -> None:
        """Record that `node` (host:port of its slice_server) holds `slices` now."""
        self._records[node] = {"ts": self._now(), "slices": list(slices),
                               "meta": meta or {}}
        if self.path:
            self._save()

    def _live(self) -> Dict[str, dict]:
        now = self._now()
        return {n: r for n, r in self._records.items()
                if now - r.get("ts", 0) <= self.ttl_s}

    def holders(self, filename: str) -> List[str]:
        """Live nodes holding `filename`, freshest record first (best liveness)."""
        live = self._live()
        held = [(n, r["ts"]) for n, r in live.items()
                if filename in r.get("slices", ())]
        held.sort(key=lambda x: -x[1])
        return [n for n, _ in held]

    def all_live_nodes(self) -> List[str]:
        return list(self._live().keys())

    def prune(self) -> None:
        """Drop expired records (and persist the pruned view)."""
        self._records = self._live()
        if self.path:
            self._save()

    def to_holders_fn(self) -> Callable[[object], List[str]]:
        """Adapter for slice_fetch.mesh_peer_source(holders_fn=...): maps a
        SliceRef to its live holders by filename."""
        def fn(ref) -> List[str]:
            return self.holders(ref.filename)
        return fn

    # persistence -----------------------------------------------------
    def _save(self) -> None:
        try:
            tmp = f"{self.path}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(self._records, f)
            os.replace(tmp, self.path)        # atomic
        except OSError:
            pass

    def _load(self) -> None:
        try:
            with open(self.path) as f:
                self._records = json.load(f)
        except (OSError, ValueError):
            self._records = {}


def publish_self(directory: SliceDirectory, node: str, slices_dir: str,
                 meta: "Optional[dict]" = None) -> int:
    """Convenience: publish all *.gguf this node serves from `slices_dir`.
    Returns the count published. Call periodically (a heartbeat) so the record
    stays fresh; stop calling on shutdown and the TTL evicts it automatically."""
    try:
        files = [n for n in os.listdir(slices_dir) if n.endswith(".gguf")]
    except OSError:
        files = []
    directory.publish(node, files, meta=meta)
    return len(files)
