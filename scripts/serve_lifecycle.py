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


from dataclasses import dataclass, field  # noqa: E402


@dataclass
class RemoteWorker:
    """One worker on a REMOTE node, summoned/reaped over SSH. `launch` is the full
    detached command to start it (it should `nohup … &`); `probe` is the
    host:port to TCP-check for readiness; `stop_match` is the `pkill -f` pattern
    that tears it down. This is the borrowed-machine member of a chain — it runs
    ONLY while a session is active, then the node returns to its owner."""
    name: str
    ssh: str                      # user@host (Tailscale ok)
    launch: str                   # remote shell command (must self-detach)
    probe: tuple                  # (host, port) reachable from the orchestrator
    stop_match: str               # pkill -f pattern


class RemoteSshController(ChainController):
    """Summon/reap a set of REMOTE workers over SSH — the multi-machine arm of
    scale-to-zero. Same surface as the local controller, so the same
    `ChainLifecycle` policy governs it. This is the seam the L3 (Sthambha) lease
    manager replaces with pillar-summoned workers; until then the serve drives it
    directly. Ownership-aware grace is set on the ChainLifecycle, not here."""

    def __init__(self, workers: "list[RemoteWorker]",
                 ssh_opts: str = "-o BatchMode=yes -o StrictHostKeyChecking=accept-new "
                                 "-o ConnectTimeout=8",
                 log: Callable[[str], None] = print):
        self.workers = workers
        self.ssh_opts = ssh_opts
        self._log = log

    def _ssh(self, uh: str, remote_cmd: str, timeout: float = 30.0) -> int:
        return subprocess.run(
            ["ssh", *self.ssh_opts.split(), uh, remote_cmd],
            capture_output=True, timeout=timeout).returncode

    def start(self) -> None:
        for w in self.workers:
            self._log(f"[lifecycle] summon remote {w.name} on {w.ssh}")
            try:
                self._ssh(w.ssh, w.launch)
            except Exception as e:        # pragma: no cover - network
                self._log(f"[lifecycle] summon {w.name} failed: {e}")

    def stop(self) -> None:
        for w in self.workers:
            self._log(f"[lifecycle] reap remote {w.name} on {w.ssh} "
                      f"(returning the borrowed machine)")
            try:
                self._ssh(w.ssh, f"pkill -f '{w.stop_match}' 2>/dev/null; "
                                 f"pkill -f llama-nakshatra-worker 2>/dev/null; true")
            except Exception as e:        # pragma: no cover - network
                self._log(f"[lifecycle] reap {w.name} failed: {e}")

    def is_ready(self) -> bool:
        for w in self.workers:
            host, port = w.probe
            try:
                with socket.create_connection((host, port), timeout=3):
                    pass
            except OSError:
                return False
        return True


class CompositeController(ChainController):
    """A chain spread across local + remote nodes: start/stop every member,
    ready iff ALL are ready. Lets one ChainLifecycle govern a mixed-ownership
    chain (e.g. this box `[0,10)` local + 3 lab Macs remote)."""

    def __init__(self, controllers: "list[ChainController]"):
        self.controllers = controllers

    def start(self) -> None:
        for c in self.controllers:
            c.start()

    def stop(self) -> None:
        for c in self.controllers:
            c.stop()

    def is_ready(self) -> bool:
        return all(c.is_ready() for c in self.controllers)


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
        # Adopt an already-running chain (e.g. workers up from a previous boot)
        # so the reaper will reap it after grace even if no request arrives.
        try:
            if self.controller.is_ready():
                self._up = True
                self._log("[lifecycle] adopted already-running chain")
        except Exception:
            pass
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


def _remote_workers_from_json(path: str) -> "list[RemoteWorker]":
    """Load remote worker specs from a JSON file:
      {"remote_workers": [
        {"name","ssh","launch","probe":"host:port","stop_match"}, ...]}"""
    import json
    data = json.loads(open(path).read())
    out = []
    for w in data.get("remote_workers", []):
        host, _, port = str(w["probe"]).rpartition(":")
        out.append(RemoteWorker(
            name=w["name"], ssh=w["ssh"], launch=w["launch"],
            probe=(host or "127.0.0.1", int(port)), stop_match=w["stop_match"]))
    return out


# ── build from env (returns None when not configured) ────────────────
def from_env(log: Callable[[str], None] = print) -> Optional[ChainLifecycle]:
    """Assemble the lifecycle from env. Local units and/or a remote-worker config
    compose into one chain controller (a CompositeController when both present).
    Returns None when nothing is configured (legacy always-on)."""
    units = (os.environ.get("NAKSHATRA_LIFECYCLE_UNITS") or "").split()
    remote_cfg = os.environ.get("NAKSHATRA_LIFECYCLE_REMOTE_CONFIG") or ""
    if not units and not remote_cfg:
        return None

    controllers: "list[ChainController]" = []
    if units:
        probes: list[tuple[str, int]] = []
        for hp in (os.environ.get("NAKSHATRA_LIFECYCLE_PROBES") or "").split():
            host, _, port = hp.rpartition(":")
            probes.append((host or "127.0.0.1", int(port)))
        controllers.append(SystemdLocalController(units, probes, log=log))
    if remote_cfg:
        controllers.append(RemoteSshController(
            _remote_workers_from_json(remote_cfg), log=log))

    ctrl = controllers[0] if len(controllers) == 1 else CompositeController(controllers)
    grace = float(os.environ.get("NAKSHATRA_LIFECYCLE_IDLE_GRACE_S", "600"))
    start_to = float(os.environ.get("NAKSHATRA_LIFECYCLE_START_TIMEOUT_S", "90"))
    return ChainLifecycle(ctrl, idle_grace_s=grace, start_timeout_s=start_to, log=log)
