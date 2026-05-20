"""Append-only forensic audit log for worker events.

Phase D of the worker hardening sprint (2026-05-20). Mirrors Sthambha's
H10 (``audit.jsonl``) + I1 (size-bounded rotation) + L1 ((ip, reason)
dedup window).

The audit log lives at ``~/.nakshatra/audit.jsonl``. Each entry is one
JSON line: ``{"ts": <unix>, "event": "<event_name>", ...payload}``.
On overflow (>``MAX_AUDIT_BYTES``, default 256 MiB) the file is moved
to ``audit.jsonl.1`` and a fresh file starts.

Events emitted by the worker (non-exhaustive — others may be added in
future phases):

  worker_started, worker_stopped
  slice_spawned, slice_completed, slice_failed
  register_success, register_failed
  attestation_observed_false
  fetch_started, fetch_completed, fetch_failed
  auth_failure_grpc, auth_failure_http

``auth_failure_*`` events are de-duplicated by ``(event, ip, reason)``
within a 60-second window to keep the log usable under a recon storm.
Repeat failures within the window are silently dropped from the file
(the in-memory counter still ticks). Cap on the dedup map is 8192
entries; oldest is evicted via OrderedDict popitem.
"""
from __future__ import annotations

import collections
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional


AUDIT_DEFAULT_PATH = Path.home() / ".nakshatra" / "audit.jsonl"
MAX_AUDIT_BYTES = 256 * 1024 * 1024
AUDIT_DEDUP_WINDOW_S = 60.0
AUDIT_DEDUP_CAP = 8192


class AuditLogger:
    """Append-only JSONL logger with size-bounded rotation + (event,
    ip, reason) dedup for auth-failure storms.

    Thread-safe. ``log()`` is the only entry point you need; other
    methods are for testing or operational introspection.
    """

    def __init__(
        self,
        path: Path = AUDIT_DEFAULT_PATH,
        *,
        max_bytes: int = MAX_AUDIT_BYTES,
        dedup_window_s: float = AUDIT_DEDUP_WINDOW_S,
        dedup_cap: int = AUDIT_DEDUP_CAP,
    ):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._dedup: "collections.OrderedDict[tuple, float]" = (
            collections.OrderedDict()
        )
        self._dedup_window = dedup_window_s
        self._dedup_cap = dedup_cap
        self._dedup_suppressed = 0
        self._writes = 0
        self._rotations = 0

    # ── Public entry point ──────────────────────────────────────────

    def log(self, event: str, **payload: Any) -> bool:
        """Append an audit event. Returns ``True`` if written;
        ``False`` if deduplicated or the write failed.

        Auth-failure events go through the dedup window; other events
        are always written.
        """
        if event.startswith("auth_failure"):
            ip = str(payload.get("ip") or "")
            reason = str(payload.get("reason") or "")
            key = (event, ip, reason)
            now = time.time()
            with self._lock:
                if key in self._dedup:
                    last_ts = self._dedup[key]
                    if now - last_ts < self._dedup_window:
                        # Suppressed — refresh ts so frequent repeats
                        # stay alive in the dedup map (LRU eviction
                        # then favors actually-rare offenders).
                        self._dedup[key] = now
                        self._dedup.move_to_end(key)
                        self._dedup_suppressed += 1
                        return False
                self._dedup[key] = now
                self._dedup.move_to_end(key)
                while len(self._dedup) > self._dedup_cap:
                    self._dedup.popitem(last=False)

        record = {"ts": int(time.time()), "event": event, **payload}
        line = json.dumps(record, separators=(",", ":")) + "\n"
        try:
            with self._lock:
                self._maybe_rotate_locked()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(line)
                self._writes += 1
            return True
        except OSError as e:
            sys.stderr.write(f"[audit] write failed: {e}\n")
            return False

    # ── Introspection ──────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "path": str(self.path),
                "writes": self._writes,
                "rotations": self._rotations,
                "dedup_entries": len(self._dedup),
                "dedup_suppressed": self._dedup_suppressed,
                "size_bytes": (
                    self.path.stat().st_size if self.path.is_file() else 0
                ),
            }

    # ── Internal ───────────────────────────────────────────────────

    def _maybe_rotate_locked(self) -> None:
        """Caller holds ``_lock``. Best-effort rotation; rotation
        failures degrade to letting the file grow."""
        try:
            if not self.path.is_file():
                return
            if self.path.stat().st_size <= self.max_bytes:
                return
            rotated = self.path.parent / (self.path.name + ".1")
            if rotated.exists():
                rotated.unlink()
            self.path.rename(rotated)
            self._rotations += 1
        except OSError as e:
            sys.stderr.write(f"[audit] rotation failed: {e}\n")


# ── Module-level singleton helpers ───────────────────────────────────


_AUDIT: Optional[AuditLogger] = None


def init_audit(path: Optional[Path] = None) -> AuditLogger:
    """Initialise the module-level audit singleton. Idempotent —
    subsequent calls replace the singleton (useful in tests)."""
    global _AUDIT
    _AUDIT = AuditLogger(path or AUDIT_DEFAULT_PATH)
    return _AUDIT


def audit(event: str, **payload: Any) -> bool:
    """Emit an audit event via the module singleton. No-op if
    ``init_audit`` has not been called — keeps the worker bringup
    failure-soft when the audit path is unwritable.
    """
    if _AUDIT is None:
        return False
    return _AUDIT.log(event, **payload)


def get_audit_logger() -> Optional[AuditLogger]:
    """Returns the current singleton or None. For tests + ops UIs."""
    return _AUDIT
