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


@dataclass
class RosterWorkerSpec:
    """How to summon the LOCAL workers for a from_roster model — the autonomous arm of slice 4.
    Instead of fixed pre-cut systemd units, the controller asks the planner (serve_chain) for the
    current firewall-eligible assignment and launches one `worker.py` per LOCAL slot, each
    SELF-PROVISIONING its layer slice from the content-addressed package (`--package-url`). Remote
    slots are left to the mesh/SSH controllers — this one only owns this box's workers."""
    model_id: str
    hidden_size: int
    package_location: str
    num_layers: Optional[int] = None
    daemon_bin: str = str(__import__("pathlib").Path.home() / "llama.cpp" / "build" / "bin" / "llama-nakshatra-worker")
    python_bin: str = ""                   # which python runs worker.py (default: this interpreter)
    scripts_dir: str = ""                  # where worker.py lives (default: this file's dir)
    slice_dir: str = ""                    # self-provision dest dir (default ~/.nakshatra/slices)
    n_ctx: int = 2048
    n_gpu_layers: int = 99
    worker_env: dict = field(default_factory=dict)


class RosterWorkerController(ChainController):
    """Summon/reap THIS box's workers for a from_roster model at the planner's assigned ranges,
    self-provisioning each slice from the package. Same ChainController surface as the others, so the
    same ChainLifecycle (scale-to-zero, reaper, optional pillar lease) governs it unchanged.

    `plan_fn` is injectable (returns the chain dict {model, workers:[{id,address,port,layer_range,
    mode}]}) so this tests without a control plane; `launch_fn(spec_dict)->Popen` is injectable so it
    tests without a GPU."""

    def __init__(self, spec: RosterWorkerSpec, *, plan_fn: Optional[Callable] = None,
                 launch_fn: Optional[Callable] = None, log: Callable[[str], None] = print):
        self.spec = spec
        self._plan_fn = plan_fn
        self._launch_fn = launch_fn
        self._log = log
        self._procs: list = []
        self._probes: "list[tuple[str, int]]" = []
        self.chain_yaml: Optional[str] = None

    def _plan(self):
        if self._plan_fn is not None:
            return self._plan_fn()
        import sys as _sys
        from pathlib import Path as _P
        fab = str(_P(self.spec.scripts_dir or _P(__file__).resolve().parent) / "fabric")
        if fab not in _sys.path:
            _sys.path.insert(0, fab)
        from serve_chain import build_chain_from_roster
        import yaml as _yaml
        self.chain_yaml = build_chain_from_roster(
            self.spec.model_id, hidden_size=self.spec.hidden_size,
            num_layers=self.spec.num_layers, package_location=self.spec.package_location)
        return _yaml.safe_load(open(self.chain_yaml))

    def _default_launch(self, w: dict):
        import sys as _sys
        from pathlib import Path as _P
        scripts = self.spec.scripts_dir or str(_P(__file__).resolve().parent)
        py = self.spec.python_bin or _sys.executable
        slice_dir = _P(self.spec.slice_dir or (_P.home() / ".nakshatra" / "slices"))
        slice_dir.mkdir(parents=True, exist_ok=True)
        s, e = w["layer_range"]
        dest = slice_dir / f"{self.spec.model_id}-selfprov-L{s}-{e}.gguf"
        env = {**os.environ, **(self.spec.worker_env or {})}
        cmd = [py, str(_P(scripts) / "worker.py"), "--port", str(w["port"]),
               "--sub-gguf", str(dest), "--package-url", self.spec.package_location,
               "--mode", w["mode"], "--layer-start", str(s), "--layer-end", str(e),
               "--model-id", self.spec.model_id, "--daemon-bin", self.spec.daemon_bin,
               "--n-ctx", str(self.spec.n_ctx), "--n-gpu-layers", str(self.spec.n_gpu_layers),
               "--node-id", f"roster-{w['id']}"]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

    def start(self) -> None:
        chain = self._plan()
        self._probes = []
        launch = self._launch_fn or self._default_launch
        for w in chain["workers"]:
            host, port = w["address"], int(w["port"])
            self._probes.append((host, port))
            if host not in ("127.0.0.1", "localhost", "0.0.0.0"):
                self._log(f"[lifecycle] {w['id']} is REMOTE ({host}:{port}) — left to the "
                          f"mesh/SSH controller, not launched locally")
                continue
            if self._port_open(host, port):
                continue                                   # already serving — leave it
            self._log(f"[lifecycle] summon roster worker {w['id']} "
                      f"layers={w['layer_range']} (self-provision from package)")
            self._procs.append(launch(w))

    def stop(self) -> None:
        for p in self._procs:
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        self._procs = []

    @staticmethod
    def _port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            return False

    def is_ready(self) -> bool:
        return bool(self._probes) and all(self._port_open(h, p) for h, p in self._probes)


# ── L3: the pillar (Sthambha) as the lease authority ─────────────────
class PillarLeaseClient:
    """Consumer side of the pillar compute-lease. The pillar OWNS the lease (the
    global, multi-consumer view + the ownership-aware grace + the expiry clock);
    this client just (1) leases a model's chain, (2) RENEWS per request via the
    pure renew endpoint (no chain re-resolution → robust to transient peer-state
    blips), and (3) reads the lease state — an `expired` read is the pillar telling
    the serve to reap. So the serve reaps only when ALL consumers have gone idle
    (the politeness the serve can't see alone). Signs as a registered peer."""

    def __init__(self, pillar_url: str, model_id: str, keyid: str,
                 priv_key: bytes, log: Callable[[str], None] = print):
        self.base = pillar_url.rstrip("/")
        self.model_id = model_id
        self.keyid = keyid
        self.priv = priv_key
        self._log = log
        self.lease_id: Optional[str] = None

    def _call(self, method: str, path: str, body: Optional[dict] = None,
              timeout: float = 10.0):
        import json
        import urllib.request
        import urllib.error
        import nakshatra_auth as _na
        raw = json.dumps(body).encode() if body is not None else b""
        hdr, _ = _na.build_signed_envelope(self.priv, self.keyid, method, path, raw)
        req = urllib.request.Request(
            self.base + path, data=(raw if body is not None else None),
            method=method,
            headers={"Content-Type": "application/json", "Authorization": hdr})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read())
            except Exception:
                return e.code, {}
        except Exception as e:                       # network — fail soft
            return 0, {"error": str(e)}

    def lease(self, retries: int = 3) -> Optional[dict]:
        """Summon-or-renew the shared lease; returns the lease dict (incl.
        idle_grace_s, chain) or None. Retries a transient 409."""
        for _ in range(max(1, retries)):
            st, b = self._call("POST", "/lease", {"model_id": self.model_id})
            if st == 200:
                self.lease_id = b.get("lease_id")
                return b
            if st != 409:
                self._log(f"[lease] POST /lease -> {st}: {b.get('error')}")
                return None
        self._log("[lease] POST /lease kept returning 409 (chain not ready)")
        return None

    def renew(self) -> Optional[dict]:
        if not self.lease_id:
            return self.lease()
        st, b = self._call("POST", f"/lease/{self.lease_id}/renew", {})
        if st == 200:
            return b
        if st == 404:                                # lease gone → re-lease
            self.lease_id = None
            return self.lease()
        return None

    def is_expired(self) -> bool:
        """True iff the pillar says the lease has expired (all consumers idle) —
        i.e. it's safe/right to reap. Fail-SAFE: on a network error we say NOT
        expired (don't reap on uncertainty)."""
        if not self.lease_id:
            return False
        st, b = self._call("GET", f"/lease/{self.lease_id}")
        if st == 200:
            return b.get("state") == "expired"
        if st == 404:
            return True
        return False                                 # network blip → keep alive


# ── the policy: summon-on-demand + idle reap, request-aware ──────────
class ChainLifecycle:
    """Gates requests through the chain: summon-if-down (block until ready),
    count in-flight requests, and a background reaper that tears the chain down
    once it's been idle (no in-flight requests) past the grace window.

    Never reaps mid-request (the active counter guards it), so a long streaming
    generation can outlast the grace window safely."""

    def __init__(self, controller: ChainController, idle_grace_s: float = 600.0,
                 start_timeout_s: float = 90.0, poll_s: float = 1.0,
                 lease_client: "Optional[PillarLeaseClient]" = None,
                 log: Callable[[str], None] = print):
        self.controller = controller
        self.idle_grace_s = idle_grace_s
        self.start_timeout_s = start_timeout_s
        self.poll_s = poll_s
        # Optional L3 authority: when set, the pillar's lease grace overrides the
        # local one and the pillar's expiry decides the reap (so we never reap
        # while ANOTHER consumer holds the lease — the global politeness view).
        self.lease_client = lease_client
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
        # L3: register this consumer's activity with the pillar (renew the shared
        # lease) + adopt the pillar's ownership-aware grace. Best-effort.
        if self.lease_client is not None:
            try:
                info = self.lease_client.renew()
                if info and info.get("idle_grace_s"):
                    self.idle_grace_s = float(info["idle_grace_s"])
            except Exception as e:
                self._log(f"[lease] renew failed (continuing): {e}")
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
                local_due = (self._up and self._active == 0
                             and idle >= self.idle_grace_s)
            if not local_due:
                continue
            # L3: when leased, the PILLAR decides (it sees all consumers). Only
            # reap if the pillar's lease has expired — never while another
            # consumer holds it. Fail-safe: a network blip keeps the chain up.
            if self.lease_client is not None:
                try:
                    if not self.lease_client.is_expired():
                        continue
                except Exception:
                    continue
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
    roster_model = os.environ.get("NAKSHATRA_LIFECYCLE_ROSTER_MODEL") or ""
    if not units and not remote_cfg and not roster_model:
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
    if roster_model:
        # Autonomous from_roster launch: summon THIS box's workers at the planner's assigned ranges,
        # each self-provisioning its slice from the package. Hidden size is required (the chain header).
        spec = RosterWorkerSpec(
            model_id=roster_model,
            hidden_size=int(os.environ["NAKSHATRA_LIFECYCLE_ROSTER_HIDDEN_SIZE"]),
            package_location=os.environ.get("NAKSHATRA_LIFECYCLE_ROSTER_PACKAGE", ""),
            num_layers=int(os.environ["NAKSHATRA_LIFECYCLE_ROSTER_NUM_LAYERS"])
            if os.environ.get("NAKSHATRA_LIFECYCLE_ROSTER_NUM_LAYERS") else None,
            daemon_bin=os.environ.get("NAKSHATRA_LIFECYCLE_DAEMON_BIN")
            or RosterWorkerSpec.daemon_bin,
            python_bin=os.environ.get("NAKSHATRA_LIFECYCLE_PYTHON_BIN", ""),
            n_gpu_layers=int(os.environ.get("NAKSHATRA_LIFECYCLE_N_GPU_LAYERS", "99")),
            n_ctx=int(os.environ.get("NAKSHATRA_LIFECYCLE_N_CTX", "2048")))
        controllers.append(RosterWorkerController(spec, log=log))

    ctrl = controllers[0] if len(controllers) == 1 else CompositeController(controllers)
    grace = float(os.environ.get("NAKSHATRA_LIFECYCLE_IDLE_GRACE_S", "600"))
    start_to = float(os.environ.get("NAKSHATRA_LIFECYCLE_START_TIMEOUT_S", "90"))

    # L3: optional pillar lease authority. When a pillar URL + lease model are
    # set, the serve becomes a lease consumer — the pillar's ownership-aware grace
    # + global expiry govern the reap (so we never reap a chain another consumer
    # is using). Signs as a registered peer (the worker key on this box).
    lease_client = None
    pillar = os.environ.get("NAKSHATRA_LIFECYCLE_PILLAR_URL") or ""
    lease_model = os.environ.get("NAKSHATRA_LIFECYCLE_LEASE_MODEL") or ""
    if pillar and lease_model:
        try:
            import nakshatra_auth as _na
            priv, _pub = _na.load_or_create_worker_key()
            keyid = os.environ.get("NAKSHATRA_LIFECYCLE_LEASE_KEYID") or "unconscious-a"
            lease_client = PillarLeaseClient(pillar, lease_model, keyid, priv, log=log)
            log.info("L3 lease authority: pillar=%s model=%s as=%s",
                     pillar, lease_model, keyid) if hasattr(log, "info") else \
                log(f"[lease] pillar={pillar} model={lease_model} as={keyid}")
        except Exception as e:
            log(f"[lease] client unavailable: {e}")
    return ChainLifecycle(ctrl, idle_grace_s=grace, start_timeout_s=start_to,
                          lease_client=lease_client, log=log)
