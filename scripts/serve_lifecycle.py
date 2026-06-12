"""serve_lifecycle.py — scale-to-zero for the worker chain behind nakshatra_serve.

The principle: **compute is summoned, not squatted.** A worker holds a GPU only
while it's serving an active session. When the chain goes idle past an
ownership-aware grace window, it's torn down and the (often borrowed) machine
returns to its owner. On the next request the chain is re-summoned (cold-start).

This is the L4/serve-side bridge to the eventual L3 (Sthambha) compute-lease
manager: the *policy* (idle grace, ownership-aware) and the *mechanism* (the
`ChainController` actions) live behind a small interface so the destination — the
pillar leasing + reaping chains across the whole mesh — can swap the controller
without touching the serve. Today the controller is local systemd units; tomorrow
it's the pillar summoning remote workers.

Wiring (env on the serve; absent ⇒ feature OFF, legacy always-on behaviour):
  NAKSHATRA_LIFECYCLE_UNITS        space-sep systemd --user units to start/stop
                                   (e.g. "nakshatra-unconscious-worker@a nakshatra-unconscious-worker@b")
  NAKSHATRA_LIFECYCLE_PROBES       space-sep host:port readiness probes (the worker
                                   gRPC ports; LISTEN ⇒ daemon loaded + ready)
  NAKSHATRA_LIFECYCLE_IDLE_GRACE_S idle seconds before reaping (default 600 on a
                                   dedicated box; set SHORT — ~90 — for borrowed nodes)
  NAKSHATRA_LIFECYCLE_START_TIMEOUT_S  max wait for cold-start readiness (default 90)
"""
from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from typing import Callable, Optional


# ── pluggable mechanism: start/stop/probe the chain ──────────────────
class ChainController:
    """The actions the lifecycle drives. Local-systemd impl here; a future
    Sthambha impl summons/reaps REMOTE workers via the pillar with the same
    surface."""

    def start(self) -> None:        # bring the chain up
        raise NotImplementedError

    def stop(self) -> None:         # tear the chain down (free the GPU)
        raise NotImplementedError

    def is_ready(self) -> bool:     # are all workers serving?
        raise NotImplementedError


class SystemdLocalController(ChainController):
    """Start/stop local `systemctl --user` units; readiness = the worker gRPC
    ports accept a TCP connection (worker.py listens only after its daemon has
    loaded the model)."""

    def __init__(self, units: list[str], probes: list[tuple[str, int]],
                 log: Callable[[str], None] = print):
        self.units = units
        self.probes = probes
        self._log = log

    def _systemctl(self, *args: str) -> int:
        return subprocess.run(["systemctl", "--user", *args],
                              capture_output=True).returncode

    def start(self) -> None:
        for u in self.units:
            self._systemctl("start", u)

    def stop(self) -> None:
        for u in self.units:
            self._systemctl("stop", u)

    def is_ready(self) -> bool:
        for host, port in self.probes:
            try:
                with socket.create_connection((host, port), timeout=2):
                    pass
            except OSError:
                return False
        return True


# ── the policy: summon-on-demand + idle reap, request-aware ──────────
class ChainLifecycle:
    """Gates requests through the chain: summon-if-down (block until ready),
    count in-flight requests, and a background reaper that tears the chain down
    once it's been idle (no in-flight requests) past the grace window.

    Never reaps mid-request (the active counter guards it), so a long streaming
    generation can outlast the grace window safely."""

    def __init__(self, controller: ChainController, idle_grace_s: float = 600.0,
                 start_timeout_s: float = 90.0, poll_s: float = 1.0,
                 log: Callable[[str], None] = print):
        self.controller = controller
        self.idle_grace_s = idle_grace_s
        self.start_timeout_s = start_timeout_s
        self.poll_s = poll_s
        self._log = log
        self._lock = threading.Lock()
        self._active = 0                 # in-flight requests
        self._last_active = 0.0          # last begin/end (monotonic)
        self._up = False                 # our belief about chain state
        self._stop_evt = threading.Event()
        self._reaper: Optional[threading.Thread] = None

    # request gate ----------------------------------------------------
    def begin(self) -> None:
        """Ensure the chain is up (cold-start + block until ready), then mark a
        request in-flight. Raises TimeoutError if it can't summon in time."""
        with self._lock:
            self._active += 1
            self._last_active = time.monotonic()
            if self._up and self.controller.is_ready():
                return
            # cold start
            self._log("[lifecycle] chain idle/down — summoning workers…")
            self.controller.start()
        # wait for readiness OUTSIDE the lock (cold-start can take ~10-20s)
        deadline = time.monotonic() + self.start_timeout_s
        while time.monotonic() < deadline:
            if self.controller.is_ready():
                with self._lock:
                    self._up = True
                self._log("[lifecycle] chain ready (summoned)")
                return
            self._stop_evt.wait(self.poll_s)
        with self._lock:
            self._active = max(0, self._active - 1)
        raise TimeoutError(
            f"chain not ready within {self.start_timeout_s}s of summon")

    def end(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
            self._last_active = time.monotonic()

    # reaper ----------------------------------------------------------
    def start_reaper(self) -> None:
        if self._reaper is not None:
            return
        self._last_active = time.monotonic()
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()
        self._log(f"[lifecycle] reaper armed: idle-grace={self.idle_grace_s:.0f}s "
                  f"units={getattr(self.controller, 'units', '?')}")

    def _reap_loop(self) -> None:
        while not self._stop_evt.is_set():
            self._stop_evt.wait(min(30.0, self.idle_grace_s / 2 or 30.0))
            with self._lock:
                idle = time.monotonic() - self._last_active
                should = (self._up and self._active == 0
                          and idle >= self.idle_grace_s)
            if should:
                self._log(f"[lifecycle] idle {idle:.0f}s ≥ grace — reaping chain "
                          f"(freeing GPU; re-summoned on next request)")
                try:
                    self.controller.stop()
                finally:
                    with self._lock:
                        self._up = False

    def stop_reaper(self) -> None:
        self._stop_evt.set()


# ── build from env (returns None when not configured) ────────────────
def from_env(log: Callable[[str], None] = print) -> Optional[ChainLifecycle]:
    units = (os.environ.get("NAKSHATRA_LIFECYCLE_UNITS") or "").split()
    if not units:
        return None
    probes_raw = (os.environ.get("NAKSHATRA_LIFECYCLE_PROBES") or "").split()
    probes: list[tuple[str, int]] = []
    for hp in probes_raw:
        host, _, port = hp.rpartition(":")
        probes.append((host or "127.0.0.1", int(port)))
    grace = float(os.environ.get("NAKSHATRA_LIFECYCLE_IDLE_GRACE_S", "600"))
    start_to = float(os.environ.get("NAKSHATRA_LIFECYCLE_START_TIMEOUT_S", "90"))
    ctrl = SystemdLocalController(units, probes, log=log)
    return ChainLifecycle(ctrl, idle_grace_s=grace, start_timeout_s=start_to, log=log)
