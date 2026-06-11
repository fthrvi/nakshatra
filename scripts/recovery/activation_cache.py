"""Activation replay cache (v1.1 hardening) — the upstream half of the O(t) dual-cache.

The current recovery (client.py) resets EVERY worker's KV and cold-replays the
whole prefix on a fresh chain — O(chain × T) per failure. Petals' O(t) recovery
keeps the *surviving* workers' KV and rebuilds only the failed span. In
Nakshatra's worker-push model (A forwards activations straight to B, no client in
the per-hop loop), the data needed to rebuild B' is **what A forwarded to B for
each step** — so the *upstream* worker must cache its forwarded activations. This
is that cache.

On a next-hop failure at step T, the upstream replays `get_replay(session, 0)`
(its cached forwarded activations for steps 0..T) to the same-drift-class
replacement B' in **catch-up mode** (build KV, don't re-forward downstream — C's
KV is already intact), then resumes at T+1. Cost: O(T) on the ONE replaced link,
not O(chain × T). The surviving workers never reset.

This is the dual-cache's *upstream* half (the other half is the surviving workers'
own KV, which they simply keep). It is the data structure + memory policy; the
worker forward-path hooks + the catch-up replay RPC are the staged integration
(needs the cluster — see docs/v1.0-fault-tolerance.md).

Memory policy (the "trade memory for fault tolerance" knob):
  • per session, retain steps 0..T (KV rebuild needs the full prefix → no
    intra-session truncation; bounded by `max_steps_per_session`, ~= n_ctx);
  • across sessions, LRU-evict whole sessions past `max_sessions`;
  • `byte_budget` caps total retained bytes — oldest sessions evicted first.
Thread-safe: workers cache from concurrent per-link forward threads.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _Session:
    steps: "OrderedDict[int, bytes]" = field(default_factory=OrderedDict)
    nbytes: int = 0


class ActivationReplayCache:
    def __init__(self, max_steps_per_session: int = 512, max_sessions: int = 64,
                 byte_budget: int = 256 * 1024 * 1024):
        self.max_steps_per_session = max_steps_per_session
        self.max_sessions = max_sessions
        self.byte_budget = byte_budget
        self._sessions: "OrderedDict[str, _Session]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    # ── record what we forwarded ──
    def record(self, session_id: str, step_id: int, payload: bytes) -> None:
        """Cache the activation forwarded to the next hop at `step_id`."""
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                s = _Session()
                self._sessions[session_id] = s
            self._sessions.move_to_end(session_id)   # LRU touch
            if step_id in s.steps:                   # idempotent re-record
                self._bytes -= len(s.steps[step_id]); s.nbytes -= len(s.steps[step_id])
            s.steps[step_id] = payload
            s.nbytes += len(payload); self._bytes += len(payload)
            # bound the session length (oldest steps fall off — they would only
            # matter for recovery, and a span older than n_ctx can't be in KV anyway)
            while len(s.steps) > self.max_steps_per_session:
                _, old = s.steps.popitem(last=False)
                s.nbytes -= len(old); self._bytes -= len(old)
            self._evict_locked()

    # ── replay for recovery ──
    def get_replay(self, session_id: str, from_step: int = 0) -> list[tuple[int, bytes]]:
        """The cached forwarded activations for steps >= from_step, in order —
        replayed to a same-drift-class replacement to rebuild its KV. Empty if the
        session was evicted (→ caller falls back to full from-token-0 restart)."""
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return []
            self._sessions.move_to_end(session_id)
            return [(k, v) for k, v in s.steps.items() if k >= from_step]

    def has_full_prefix(self, session_id: str, up_to_step: int) -> bool:
        """True iff steps 0..up_to_step are ALL cached (a sound KV rebuild is
        possible). If a hole exists (eviction/loss), O(t) recovery is unsafe →
        the caller must do a clean full restart instead of a partial splice."""
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return False
            return all(i in s.steps for i in range(0, up_to_step + 1))

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            s = self._sessions.pop(session_id, None)
            if s:
                self._bytes -= s.nbytes

    def stats(self) -> dict:
        with self._lock:
            return {"sessions": len(self._sessions), "bytes": self._bytes,
                    "byte_budget": self.byte_budget}

    # ── eviction (LRU whole sessions; caller holds the lock) ──
    def _evict_locked(self) -> None:
        while len(self._sessions) > self.max_sessions or self._bytes > self.byte_budget:
            if not self._sessions:
                break
            _, s = self._sessions.popitem(last=False)   # oldest session
            self._bytes -= s.nbytes
