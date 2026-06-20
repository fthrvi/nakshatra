"""
wanted_tracker.py — the DEMAND side of the discovery signal (the `wanted` field).

`NakshatraListing.wanted` (signed + scored: a node offering a model the mesh wants gets a ranking
bump) already exists, but nothing populates it from reality. Sourced from mesh-llm's `wanted`
advertisement: a node broadcasts the models it needs but can't serve, so a GPU owner can SEE unmet
demand and plug in — the demand side of the incentive flywheel.

This is the source: a small in-memory TTL set of recently-unmet model demands. The serve/gate path
calls `note(model)` whenever a request can't be satisfied (no eligible worker, insufficient VRAM,
capacity-denied); the publisher calls `wanted()` to fold the live demand into its next listing. Demands
age out after `ttl` so the signal reflects CURRENT scarcity, not history.

Pure + in-process (no I/O); deterministic with an injected clock for tests.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

DEFAULT_TTL = 600.0          # a demand stays advertised for 10 min after it was last seen
DEFAULT_MAX = 64             # cap distinct models tracked (bounds a hostile/noisy caller)


class WantedTracker:
    def __init__(self, ttl: float = DEFAULT_TTL, max_models: int = DEFAULT_MAX,
                 clock: Optional[Callable[[], float]] = None):
        self.ttl = float(ttl)
        self.max_models = int(max_models)
        self._clock = clock or time.time
        self._seen: dict[str, float] = {}     # model_id -> last-demanded unix ts
        self._lock = threading.Lock()

    def note(self, model_id: str) -> None:
        """Record an unmet demand for a model (refreshes its TTL). No-op for empty ids."""
        if not model_id:
            return
        now = self._clock()
        with self._lock:
            self._seen[model_id] = now
            # opportunistic prune + cap: if over capacity, drop the oldest
            if len(self._seen) > self.max_models:
                self._prune(now)
                while len(self._seen) > self.max_models:
                    oldest = min(self._seen, key=self._seen.get)
                    del self._seen[oldest]

    def _prune(self, now: float) -> None:
        dead = [m for m, t in self._seen.items() if now - t > self.ttl]
        for m in dead:
            del self._seen[m]

    def wanted(self, now: Optional[float] = None) -> List[str]:
        """The current demand set: models noted within the TTL window, deduped + sorted (stable wire)."""
        now = self._clock() if now is None else now
        with self._lock:
            self._prune(now)
            return sorted(self._seen.keys())

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()
