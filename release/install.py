#!/usr/bin/env python3
"""install.py — install or update a Nakshatra node from a signed release. Standard library only.

First install (pins the release key and the source):
    python3 install.py install --from http://10.42.0.1:8960/nakshatra-node --channel canary \
        --pubkey <release-pubkey-hex>
Later (run by the hourly nak-update timer, or by hand):
    python3 ~/.nakshatra-node/bin/install.py update
    python3 ~/.nakshatra-node/bin/install.py rollback
    python3 ~/.nakshatra-node/bin/install.py status

What it guarantees:
  * nothing is unpacked or run unless manifest.json verifies against the PINNED release key and every
    file matches the sha256 the manifest lists (the check covers everything that ships);
  * releases install side by side (releases/<version>/); `current` is switched atomically;
  * after switching it runs the release's health checks, then drains and restarts changed services;
    pre-restart failures roll back, while a service that fails after restart is recorded for retry;
  * update never installs an older version (an equal version is touched only to retry pending/failed services),
    and never installs a different major version.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import releasekit as rk  # noqa: E402

DEFAULT_PREFIX = Path(os.environ.get("NAK_NODE_PREFIX", Path.home() / ".nakshatra-node"))
UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
_THIS_FILE = Path(__file__).resolve()      # resolved at LOAD time, before any `current` switch
UNIT_MARKER = "# Managed by nakshatra-node install.py"
SERVICE_ACTIVE_TIMEOUT_S = 20.0
SERVICE_STABILITY_S = 5.0


class InstallError(Exception):
    pass


def _vt(v: str) -> tuple:
    return tuple(int(x) for x in v.split("."))


def _fetch(source: str, rel: str, dest: Path) -> None:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(f"{source.rstrip('/')}/{rel}", timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        shutil.copyfile(Path(source) / rel, dest)


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _save_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
    os.replace(tmp, p)


def _load_config(prefix: Path) -> dict:
    p = prefix / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_config(prefix: Path, cfg: dict) -> None:
    tmp = prefix / "config.json.tmp"
    tmp.write_text(json.dumps(cfg, indent=1, sort_keys=True))
    os.replace(tmp, prefix / "config.json")


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    for m in tf.getmembers():
        target = (dest / m.name).resolve()
        if not str(target).startswith(str(dest.resolve()) + os.sep) and target != dest.resolve():
            raise InstallError(f"archive member escapes the install dir: {m.name}")
        if m.issym() or m.islnk():
            raise InstallError(f"archive contains a link, refusing: {m.name}")
    if hasattr(tarfile, "data_filter"):
        tf.extractall(dest, filter="data")   # the stdlib's own guard, on top of the checks above
    else:
        tf.extractall(dest)


def _render(arg: str, ctx: dict) -> str:
    for k, v in ctx.items():
        arg = arg.replace("{" + k + "}", str(v))
    return arg


def _switch(prefix: Path, version: str) -> None:
    tmp = prefix / ".current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(Path("releases") / version)
    os.replace(tmp, prefix / "current")


def _releases_in_use(rel: Path) -> set:
    """Release dir names that any of this user's processes has mapped or runs from (/proc maps + cmdline)."""
    import re
    pat = re.compile(re.escape(str(rel.resolve())) + r"/([^/\s]+)/")
    used, uid = set(), os.getuid()
    for pid in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            with open(f"/proc/{pid}/maps", errors="replace") as f:
                used.update(pat.findall(f.read()))
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                used.update(pat.findall(f.read().replace(b"\0", b" ").decode(errors="replace")))
        except OSError:
            continue
    return used


def _current(prefix: Path):
    c = prefix / "current"
    return os.readlink(c).split("/")[-1] if c.is_symlink() else None


class Installer:
    def __init__(self, prefix: Path = DEFAULT_PREFIX, *, systemd: bool = True, make_venv: bool = True,
                 unit_dir=None):
        self.prefix = Path(prefix)
        self.systemd = systemd
        self.make_venv = make_venv
        # Decided ONCE, here. An installer that is not driving systemd (tests, dev) writes its units
        # inside its own prefix and can NEVER touch the real ~/.config/systemd/user — on 2026-10-03 a
        # test that undid a monkeypatch rewrote the hub's real nak-update.service to a pytest tmp dir.
        self.unit_dir = Path(unit_dir) if unit_dir else (UNIT_DIR if systemd else self.prefix / "units")

    @property
    def _service_state_path(self) -> Path:
        return self.prefix / "state" / "services.json"

    def _record_service_issue(self, name: str, status: str, detail: str) -> None:
        state = _load_json(self._service_state_path)
        services = state.setdefault("services", {})
        services[name] = {"status": status, "detail": detail, "updated": int(time.time())}
        _save_json(self._service_state_path, state)

    def _clear_service_issue(self, name: str) -> None:
        state = _load_json(self._service_state_path)
        services = state.get("services", {})
        if name in services:
            del services[name]
            _save_json(self._service_state_path, state)

    def _service_issue_summary(self) -> str:
        services = _load_json(self._service_state_path).get("services", {})
        if not services:
            return ""
        grouped = {}
        for name, issue in services.items():
            grouped.setdefault(issue.get("status", "failed"), []).append(name)
        return "; " + "; ".join(f"{status}: {', '.join(sorted(names))}" for status, names in sorted(grouped.items()))

    # ── fetch + verify ──
    def fetch_verified(self, source: str, channel: str, pubkey: str, version=None) -> tuple:
        work = Path(tempfile.mkdtemp(prefix="nak-release-"))
        if version is None:
            _fetch(source, f"{channel}/latest.json", work / "latest.json")
            latest = json.loads((work / "latest.json").read_text())
            if not rk.verify(latest, pubkey) or latest.get("channel") != channel:
                raise InstallError("latest.json is not signed by the pinned release key")
            # A host could replay an OLD signed latest.json forever (audit 2026-10-03, finding 1):
            # a signed pointer carries an expiry, and a stale one is refused.
            exp = latest.get("expires_at")
            if isinstance(exp, int) and time.time() > exp:
                raise InstallError("the release host is serving an expired latest.json (stale or replayed)")
            version = latest["version"]
        _fetch(source, f"{channel}/{version}/manifest.json", work / "manifest.json")
        man = json.loads((work / "manifest.json").read_text())
        if man.get("schema") != rk.MANIFEST_SCHEMA or not rk.verify(man, pubkey):
            raise InstallError("manifest.json is not signed by the pinned release key")
        if man["version"] != version or man["channel"] != channel:
            raise InstallError("manifest names a different version or channel than requested")
        files = [(c["file"], c["sha256"]) for c in man["components"]]
        files += [(man["lock"]["file"], man["lock"]["sha256"]), (man["uv"]["file"], man["uv"]["sha256"])]
        files += [(p["lock"]["file"], p["lock"]["sha256"]) for p in (man.get("profiles") or {}).values()]
        for name, want in files:
            if "/" in name or name.startswith("."):
                raise InstallError(f"bad file name in manifest: {name}")
            _fetch(source, f"{channel}/{version}/{name}", work / name)
            got = rk.sha256_file(work / name)
            if got != want:
                raise InstallError(f"{name} does not match the signed manifest (sha256 {got[:12]} != {want[:12]})")
        return man, work

    # ── unpack + environment ──
    def unpack(self, man: dict, work: Path) -> Path:
        rel = self.prefix / "releases" / man["version"]
        if rel.exists():
            shutil.rmtree(rel)
        rel.mkdir(parents=True)
        for c in man["components"]:
            d = rel / c["name"]
            d.mkdir()
            with tarfile.open(work / c["file"]) as tf:
                _safe_extract(tf, d)
        component_names = {c["name"] for c in man["components"]}
        for artifact in man.get("artifacts", []):
            parts = Path(artifact["path"]).parts
            if artifact.get("component") not in component_names or not parts or Path(artifact["path"]).is_absolute() \
                    or ".." in parts:
                raise InstallError(f"bad artifact path in manifest: {artifact.get('path')}")
            dst = rel / artifact["component"] / artifact["path"]
            if not dst.is_file() or rk.sha256_file(dst) != artifact["sha256"]:
                raise InstallError(f"installed artifact {artifact['name']} does not match its signed sha256")
            os.chmod(dst, int(artifact.get("mode", "0644"), 8))
        shutil.copy2(work / man["lock"]["file"], rel / "requirements.lock")
        lock_to_install = rel / "requirements.lock"
        for pname in self.active_profiles(man):
            # The profile lock is the WHOLE environment (base + profile, resolved together): install it
            # instead of the base lock. More than one active profile would need a combined lock; refuse.
            if lock_to_install != rel / "requirements.lock":
                raise InstallError("more than one release profile is active on this node; not supported yet")
            pf = man["profiles"][pname]["lock"]["file"]
            shutil.copy2(work / pf, rel / pf)
            lock_to_install = rel / pf
        shutil.copy2(work / "uv", rel / "uv")
        os.chmod(rel / "uv", 0o755)
        (rel / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
        if self.make_venv:
            uv = str(rel / "uv")
            env = dict(os.environ, UV_CACHE_DIR=str(self.prefix / "uv-cache"))
            subprocess.run([uv, "venv", str(rel / "venv"), "--python", man["python"], "--quiet"], check=True, env=env)
            subprocess.run([uv, "pip", "install", "--quiet", "--require-hashes", "-r", str(lock_to_install),
                            "--python", str(rel / "venv" / "bin" / "python")], check=True, env=env)
        return rel

    def active_profiles(self, man: dict) -> list:
        """Release profiles this node opted into: each profile names a file (e.g. ~/.nakshatra/inference.env)
        whose presence turns it on. Nodes that opt in to nothing get the lean base environment."""
        ctx = {"home": Path.home(), "prefix": self.prefix}
        return [n for n, p in sorted((man.get("profiles") or {}).items())
                if Path(_render(p["requires"], ctx)).exists()]

    def _ctx(self, man: dict) -> dict:
        cur = self.prefix / "current"
        return {"python": cur / "venv" / "bin" / "python", "home": Path.home(), "prefix": self.prefix}

    def _pythonpath(self, man: dict) -> str:
        cur = self.prefix / "current"
        return ":".join(str(cur / c["name"] / c["pythonpath"]) for c in man["components"])

    # ── services + health ──
    def _write_units_by_new_installer(self, man: dict) -> list:
        """Units are written by the installer that SHIPS IN the release being installed, never by
        whichever (older) installer happens to run the update — an older one does not know newer
        unit fields (2026-10-03: 0.7.2's installer wrote 0.8.0's meshd unit without its env file, so
        meshd crash-looped on empty args). Falls back to this installer when the release ships none or
        it is this same file."""
        shipped = man.get("installer")
        new = (self.prefix / "current" / shipped) if shipped else None
        # Compare against where THIS installer lived when it was LOADED (_THIS_FILE), not Path(__file__)
        # now: updates run us via current/..., and once `current` has switched, resolving __file__ lands
        # in the NEW release — we then mistook the new installer for ourselves and wrote units with old
        # code (2026-10-03: that wrote + restarted the live gateway unit; 44 s outage).
        if new is None or not new.exists() or new.resolve() == _THIS_FILE:
            return self.write_units(man)
        args = [sys.executable, str(new), "--prefix", str(self.prefix), "write-units"]
        if not self.systemd:
            args.append("--no-systemd")
        args += ["--unit-dir", str(self.unit_dir)]      # always explicit: the child must write exactly here
        out = subprocess.run(args, capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            raise InstallError(f"the new release's installer could not write units: {out.stderr.strip()[-300:]}")
        return json.loads(out.stdout.strip().splitlines()[-1])

    def _unit_text(self, svc: dict, ctx: dict, man: dict) -> str:
        """One systemd unit from a spec entry. Optional keys (all from the SIGNED manifest):
        requires (opt-in per node: ConditionPathExists), after, env_file, environment {K: V},
        restart (default on-failure), hardening (the meshd/relay sandbox: read-only home except
        ~/.nakshatra). ${VARS} pass through to systemd, which expands them from the env file."""
        unit = [f"Description={svc['description']} (managed by nakshatra-node install.py)"]
        if svc.get("requires"):
            unit.append(f"ConditionPathExists={_render(svc['requires'], ctx)}")
        if svc.get("after"):
            unit.append("After=" + " ".join(svc["after"]))
        service = [f"Environment=PYTHONPATH={self._pythonpath(man)}"]
        if svc.get("env_file"):
            service.append(f"EnvironmentFile=-{_render(svc['env_file'], ctx)}")
        for k, v in (svc.get("environment") or {}).items():
            service.append(f"Environment={k}={_render(str(v), ctx)}")
        for command in svc.get("exec_start_pre", []):
            service.append("ExecStartPre=" + " ".join(_render(a, ctx) for a in command))
        service.append("ExecStart=" + " ".join(_render(a, ctx) for a in svc["exec"]))
        service.append(f"Restart={svc.get('restart', 'on-failure')}")
        service.append("RestartSec=5")
        if svc.get("hardening"):
            service += ["NoNewPrivileges=true", "PrivateTmp=true", "ProtectSystem=strict",
                        "ProtectHome=read-only"]
            paths = svc.get("read_write_paths") or ["{home}/.nakshatra"]
            service += [f"ReadWritePaths={_render(p, ctx)}" for p in paths]
        return (UNIT_MARKER + "\n[Unit]\n" + "\n".join(unit) + "\n\n[Service]\n" + "\n".join(service) +
                "\n\n[Install]\nWantedBy=default.target\n")

    @staticmethod
    def _managed_unit(path: Path) -> bool:
        try:
            head = path.read_text()[:1000]
        except OSError:
            return False
        # The Description marker recognizes units written by pre-marker installers during upgrade.
        return UNIT_MARKER in head or "(managed by nakshatra-node install.py)" in head

    def write_units(self, man: dict) -> list:
        ctx, names = self._ctx(man), []
        self.unit_dir.mkdir(parents=True, exist_ok=True)
        for name, svc in man.get("services", {}).items():
            # A node that has NOT opted into a service gets no unit file for it at all: writing one
            # anyway would overwrite whatever (possibly hand-written, possibly live) unit carries that name.
            if svc.get("requires") and not Path(_render(svc["requires"], ctx)).exists():
                unit_path = self.unit_dir / f"{name}.service"
                if unit_path.exists() and self._managed_unit(unit_path):
                    if self.systemd:
                        stopped = subprocess.run(
                            ["systemctl", "--user", "disable", "--now", f"{name}.service"], check=False)
                        if stopped.returncode != 0:
                            self._record_service_issue(
                                name, "removal-failed", "systemctl disable --now failed; managed unit retained")
                            continue
                    unit_path.unlink()
                    self._clear_service_issue(name)
                continue
            (self.unit_dir / f"{name}.service").write_text(self._unit_text(svc, ctx, man))
            names.append(name)
        bin_dir = self.prefix / "bin"
        bin_dir.mkdir(exist_ok=True)
        # Short commands (`nak`, `nak-mcp`): wrappers that always run the CURRENT release, so an update
        # or rollback moves them too. Linked into ~/.local/bin only on a real (systemd) install, and never
        # over a file that is not already our link.
        for cmd, module in man.get("commands", {}).items():
            w = bin_dir / cmd
            w.write_text(f"#!/bin/sh\nPYTHONPATH={self._pythonpath(man)} exec {ctx['python']} -m {module} \"$@\"\n")
            w.chmod(0o755)
            if self.systemd:
                link = Path.home() / ".local" / "bin" / cmd
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() or not link.exists():
                    if link.is_symlink():
                        link.unlink()
                    link.symlink_to(w)
        for f in ("install.py", "releasekit.py"):
            src, dst = HERE / f, bin_dir / f
            if not dst.exists() or src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        # Updates run the installer that ships INSIDE the signed release, so the installer itself is
        # updated through the same verified path; bin/ is only the bootstrap fallback.
        shipped = man.get("installer")
        cur_rel = self.prefix / "releases" / man["version"]
        updater = (self.prefix / "current" / shipped) if shipped and (cur_rel / shipped).exists() else bin_dir / "install.py"
        (self.unit_dir / "nak-update.service").write_text(
            "[Unit]\nDescription=Nakshatra node self-update (signed releases only)\n\n"
            f"[Service]\nType=oneshot\nExecStart={sys.executable} {updater} --prefix {self.prefix} update\n")
        (self.unit_dir / "nak-update.timer").write_text(
            "[Unit]\nDescription=Hourly Nakshatra node update check\n\n"
            "[Timer]\nOnCalendar=hourly\nRandomizedDelaySec=600\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n")
        if self.systemd:
            if subprocess.run(["systemctl", "--user", "daemon-reload"], check=False).returncode != 0:
                # Restarting now would run systemd's cached OLD unit against the new release.
                raise InstallError("systemctl --user daemon-reload failed; not restarting anything")
            subprocess.run(["systemctl", "--user", "enable", "--now", "nak-update.timer"], check=False)
        return names

    def restart(self, names: list, man: dict | None = None, *, force: bool = False) -> list:
        """Restart services only when needed, and never in the middle of a request (2026-10-03):
        - a service whose FINGERPRINT (its unit text + the commits of the components it runs + the env lock)
          is unchanged and that is already running is left alone (e.g. a Nakshatra-only release never
          restarts the Sthambha-only signer);
        - a service with `drain: {port, max_wait_s}` waits until nothing is connected to that port (up to
          max_wait_s), THEN restarts — a live Prithvi turn is not cut off. Code under current/ has already
          switched, so the wait is bounded: mixed versions must not linger.
        force=True (rollback) restarts every supplied name and keeps rollback's strict failure behavior.
        During a normal update, a post-restart process failure is recorded, never raised: rolling an
        otherwise valid release back would restart unrelated live services.
        Returns the names for which a restart was actually requested."""
        if not (self.systemd and names):
            return []
        fps = _load_json(self.prefix / "state" / "fingerprints.json")
        services = (man or {}).get("services", {})
        todo = []
        for n in names:
            fp = self._fingerprint(n, man) if man else None
            if not force and fp and fps.get(n) == fp and self._active(n):
                self._clear_service_issue(n)
                continue
            todo.append((n, fp))
        if todo:
            subprocess.run(["systemctl", "--user", "enable", *[f"{n}.service" for n, _ in todo]], check=False)
        restarted = []
        for n, fp in todo:
            drain = services.get(n, {}).get("drain") if not force else None
            if drain:
                if not self._wait_idle(int(drain["port"]), float(drain.get("max_wait_s", 900))):
                    self._record_service_issue(
                        n, "pending", f"drain timed out on port {drain['port']}; restart deferred")
                    continue
            result = subprocess.run(["systemctl", "--user", "restart", f"{n}.service"], check=False)
            restarted.append(n)
            if result.returncode == 0:
                healthy, detail = self._wait_stable(n, SERVICE_ACTIVE_TIMEOUT_S, SERVICE_STABILITY_S)
            else:
                healthy, detail = False, f"systemctl restart failed with exit status {result.returncode}"
            if healthy:
                self._clear_service_issue(n)
            else:
                self._record_service_issue(n, "failed", detail)
                if force:
                    raise InstallError(f"{n}.service {detail}")
            if healthy and fp:
                fps[n] = fp
        _save_json(self.prefix / "state" / "fingerprints.json", fps)
        return restarted

    def _fingerprint(self, name: str, man: dict) -> str:
        import hashlib
        svc = man.get("services", {}).get(name, {})
        wanted = svc.get("components") or [c["name"] for c in man.get("components", [])]
        commits = sorted(f"{c['name']}={c['commit']}" for c in man.get("components", []) if c["name"] in wanted)
        # The env lock by CONTENT (comments stripped): older locks carry build-path comments that differ
        # every build and made identical releases look changed (2026-10-03).
        rel = self.prefix / "releases" / man.get("version", "")
        lock_files = [man.get("lock", {}).get("file", "requirements.lock")] + [
            man["profiles"][p]["lock"]["file"] for p in self.active_profiles(man)]
        locks = []
        for lf in lock_files:
            try:
                body = "\n".join(l for l in (rel / lf).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#"))
            except OSError:
                body = lf
            locks.append(hashlib.sha256(body.encode()).hexdigest())
        unit = self._unit_text(svc, self._ctx(man), man) if svc else ""
        return hashlib.sha256("\n".join([unit, *commits, *locks]).encode()).hexdigest()

    def _active(self, name: str) -> bool:
        return subprocess.run(["systemctl", "--user", "is-active", "--quiet", f"{name}.service"]).returncode == 0

    def _wait_active(self, name: str, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            if self._active(name):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.5)

    def _service_runtime(self, name: str):
        r = subprocess.run(
            ["systemctl", "--user", "show", f"{name}.service", "--property=ActiveState,MainPID,NRestarts"],
            capture_output=True, text=True)
        if r.returncode != 0:
            return None
        props = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
        try:
            return props.get("ActiveState"), int(props.get("MainPID", "0")), int(props.get("NRestarts", "0"))
        except ValueError:
            return None

    def _wait_stable(self, name: str, timeout_s: float, stability_s: float) -> tuple[bool, str]:
        deadline = time.monotonic() + timeout_s
        first = None
        while True:
            snap = self._service_runtime(name)
            if snap and snap[0] == "active" and snap[1] > 0:
                first = snap
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
        if first is None:
            return False, "did not become active after restart"
        time.sleep(stability_s)
        second = self._service_runtime(name)
        if second is None or second[0] != "active":
            return False, "did not remain active during the stability window"
        if second[1] != first[1] or second[2] != first[2]:
            return False, (f"restarted during the stability window "
                           f"(MainPID {first[1]}->{second[1]}, NRestarts {first[2]}->{second[2]})")
        return True, ""

    def _wait_idle(self, port: int, max_wait_s: float) -> bool:
        """True once nothing is connected to `port` (checked twice, 2 s apart), False at the time limit."""
        end, quiet = time.time() + max_wait_s, 0
        while time.time() < end:
            r = subprocess.run(["ss", "-Htn", "state", "established", f"( sport = :{port} )"],
                               capture_output=True, text=True)
            # Fail CLOSED: if ss itself fails we cannot tell, so the port counts as busy.
            quiet = quiet + 1 if (r.returncode == 0 and not (r.stdout or "").strip()) else 0
            if quiet >= 2:
                return True
            time.sleep(2)
        return False

    def health(self, man: dict) -> None:
        ctx = self._ctx(man)
        env = dict(os.environ, PYTHONPATH=self._pythonpath(man))
        for cmd in man.get("health", []):
            r = subprocess.run([_render(a, ctx) for a in cmd], env=env, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise InstallError(f"health check failed: {' '.join(cmd)}\n{(r.stderr or r.stdout)[-800:]}")

    # ── the operations ──
    def install(self, source: str, channel: str, pubkey: str, version=None, *, allow_downgrade=False,
                min_version: str = "") -> str:
        self.prefix.mkdir(parents=True, exist_ok=True)
        cfg = _load_config(self.prefix)
        pinned = cfg.get("pubkey")
        if pinned and pubkey and pinned != pubkey:
            raise InstallError("this node already trusts a different release key; refusing to switch keys here")
        pubkey = pinned or pubkey
        if not pubkey:
            raise InstallError("first install needs --pubkey (the release key to trust)")
        man, work = self.fetch_verified(source, channel, pubkey, version)
        if min_version and _vt(man["version"]) < _vt(min_version):
            shutil.rmtree(work, ignore_errors=True)
            raise InstallError(f"the release host offers {man['version']}, older than {min_version} which your "
                               f"friend's invite names; refusing (stale or malicious host)")
        # Installed = a COMPLETED install recorded in config.json, not merely a `current` link: a
        # first install that died half-way must be retried, not reported as "already at X".
        prev = _current(self.prefix) if cfg.get("current") else None
        if prev:
            if man["version"] == prev and not allow_downgrade:
                issues = _load_json(self._service_state_path).get("services", {})
                retry = {name for name, issue in issues.items()
                         if issue.get("status") in ("failed", "pending")}
                shutil.rmtree(work, ignore_errors=True)
                if issues:
                    names = self._write_units_by_new_installer(man)
                    self.restart([name for name in names if name in retry], man)
                    return f"retried services for {prev}" + self._service_issue_summary()
                return f"already at {prev}; {man['version']} is not newer"
            if _vt(man["version"]) < _vt(prev) and not allow_downgrade:
                shutil.rmtree(work, ignore_errors=True)
                return f"already at {prev}; {man['version']} is not newer"
            if _vt(man["version"])[0] != _vt(prev)[0]:
                raise InstallError(f"{man['version']} is a different major version than {prev}; install it explicitly")
        new_dir = self.prefix / "releases" / man["version"]
        try:
            self.unpack(man, work)
        except Exception:
            if man["version"] != prev:
                shutil.rmtree(new_dir, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(work, ignore_errors=True)
        _switch(self.prefix, man["version"])
        try:
            names = self._write_units_by_new_installer(man)
            self.health(man)
            self.restart(names, man)
        except Exception as e:
            if prev:
                _switch(self.prefix, prev)
                prev_man = json.loads((self.prefix / "releases" / prev / "manifest.json").read_text())
                prev_names = self.write_units(prev_man)
                self.restart(prev_names, prev_man, force=True)
                raise InstallError(f"{man['version']} failed after switching and was rolled back to {prev}: {e}")
            # First install failed: leave NO half-installed node behind (no `current`, no release dir),
            # so the next attempt starts clean instead of believing it is already installed.
            try:
                (self.prefix / "current").unlink()
            except OSError:
                pass
            shutil.rmtree(new_dir, ignore_errors=True)
            raise InstallError(f"first install of {man['version']} failed and was removed: {e}")
        cfg.update({"source": source, "channel": channel, "pubkey": pubkey, "current": man["version"],
                    "previous": prev, "updated": int(time.time())})
        _save_config(self.prefix, cfg)
        self.prune(keep=(man["version"], prev))
        return (f"installed {man['version']} ({channel})" + (f", previous {prev}" if prev else "") +
                self._service_issue_summary())

    KEEP_RELEASES = 3

    def prune(self, keep=()) -> list:
        """Delete old release dirs: always keep `current`, `previous` and the newest KEEP_RELEASES.
        (2026-10-04: the hub had 28 releases, 1.9 GB.) Never fails an install."""
        rel = self.prefix / "releases"
        try:
            dirs = sorted((d for d in rel.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
        except OSError:
            return []
        protect = {k for k in keep if k} | {d.name for d in dirs[:self.KEEP_RELEASES]}
        cur = _current(self.prefix)
        if cur:
            protect.add(cur)
        # A service that was NOT restarted (its inputs did not change) still runs from an older release dir;
        # deleting it would break that process's next lazy import (the hub's signer ran from 0.10.8 under 0.11.0).
        protect |= _releases_in_use(rel)
        removed = []
        for d in dirs:
            if d.name not in protect:
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d.name)
        return removed

    def update(self) -> str:
        cfg = _load_config(self.prefix)
        if not cfg:
            raise InstallError("not installed yet; run install first")
        return self.install(cfg["source"], cfg["channel"], cfg["pubkey"])

    def rollback(self) -> str:
        cfg = _load_config(self.prefix)
        prev = cfg.get("previous")
        if not prev or not (self.prefix / "releases" / prev).exists():
            raise InstallError("no previous release to roll back to")
        cur = _current(self.prefix)
        _switch(self.prefix, prev)
        man = json.loads((self.prefix / "releases" / prev / "manifest.json").read_text())
        self.restart(self.write_units(man), man, force=True)
        cfg.update({"current": prev, "previous": cur, "updated": int(time.time())})
        _save_config(self.prefix, cfg)
        return f"rolled back to {prev}" + self._service_issue_summary()

    def status(self) -> dict:
        cfg = _load_config(self.prefix)
        return {"current": _current(self.prefix), "previous": cfg.get("previous"), "channel": cfg.get("channel"),
                "source": cfg.get("source"), "release_key": (cfg.get("pubkey") or "")[:16],
                "service_issues": _load_json(self._service_state_path).get("services", {})}


# ── join: the one command a newcomer pastes ─────────────────────────────────────────────────────
#
#   python3 install.py join '<nki1… invite>' [--name "what your friend sees"]
#
# 1. checks the invite is signed by the friend who made it and has not expired
# 2. installs the node from the release the invite names, pinning the release key the invite names
# 3. runs the release's own setup: keys (TEST custody), signer, a delegated agent, services
# 4. asks the friend to connect (they still have to accept)

INVITE_PREFIX = "nki1."


def parse_invite(code: str, now=None) -> dict:
    """Verify an invite with no third-party crypto (releasekit's pure Ed25519). Raises InstallError."""
    import base64
    now = int(now if now is not None else time.time())
    code = (code or "").strip()
    if not code.startswith(INVITE_PREFIX) or len(code) > 8192:
        raise InstallError("that is not a Nakshatra invite (it should start with nki1.)")
    try:
        body = code[len(INVITE_PREFIX):]
        inv = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise InstallError("the invite is damaged (copy the whole line again)")
    if not isinstance(inv, dict) or not isinstance(inv.get("inviter"), str) or len(inv["inviter"]) != 64:
        raise InstallError("the invite is damaged (copy the whole line again)")
    if not rk.verify(inv, inv["inviter"]):
        raise InstallError("the invite's signature does not verify: it was changed or is not genuine")
    if not isinstance(inv.get("expires_at"), int) or now >= inv["expires_at"]:
        raise InstallError("the invite has expired: ask your friend for a new one")
    rel = inv.get("release")
    if not isinstance(rel, dict) or not all(isinstance(rel.get(k), str) for k in ("url", "channel", "pubkey")):
        raise InstallError("the invite does not say where to get Nakshatra (your friend's node is too old)")
    return inv


def _systemd_user_ok() -> bool:
    try:
        return subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True,
                              timeout=10).returncode in (0, 1)
    except (OSError, subprocess.TimeoutExpired):
        return False


def join(prefix: Path, code: str, name: str) -> int:
    inv = parse_invite(code)
    rel = inv["release"]
    # The signature proves the invite is intact and was made with the key it names — NOT who that is.
    # Trust comes from the person who sent you the line (and the installer fingerprint inside it).
    print(f"invite is intact and signed by key {inv['inviter'][:16]}… (check with your friend that "
          f"this is theirs); installing Nakshatra from {rel['url']} ({rel['channel']})", flush=True)
    if not _systemd_user_ok():
        raise InstallError("this machine has no systemd user session. On WSL put [boot] systemd=true in "
                           "/etc/wsl.conf and restart WSL, then paste the line again.")
    inst = Installer(prefix)
    if _current(prefix) and _load_config(prefix).get("current"):
        print(f"a node is already installed ({_current(prefix)}); keeping it and joining with it", flush=True)
        cfg = _load_config(prefix)
        if cfg.get("pubkey") and cfg["pubkey"] != rel["pubkey"]:
            raise InstallError("this node trusts a different release key than the invite names; refusing to mix")
    else:
        print(inst.install(rel["url"], rel["channel"], rel["pubkey"], min_version=rel.get("version", "")), flush=True)
    man = json.loads((prefix / "current" / "manifest.json").read_text())
    env = dict(os.environ, PYTHONPATH=inst._pythonpath(man))
    py = str(prefix / "current" / "venv" / "bin" / "python")
    return subprocess.run([py, "-m", "network.join_setup", "--invite", code, "--name", name], env=env).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Install or update a Nakshatra node from a signed release.")
    ap.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("install")
    p.add_argument("--from", dest="source", required=True)
    p.add_argument("--channel", default="canary")
    p.add_argument("--pubkey", default="")
    p.add_argument("--version", default=None)
    sub.add_parser("update"); sub.add_parser("rollback"); sub.add_parser("status")
    p = sub.add_parser("write-units", help="(internal) write this release's systemd units; prints their names")
    p.add_argument("--no-systemd", action="store_true")
    p.add_argument("--unit-dir", default=None)
    p = sub.add_parser("join", help="join Nakshatra with an invite a friend sent you")
    p.add_argument("invite")
    p.add_argument("--name", default="", help="what your friend will see you as")
    a = ap.parse_args(argv)
    inst = Installer(a.prefix)
    try:
        if a.cmd == "write-units":
            w = Installer(a.prefix, systemd=not a.no_systemd, unit_dir=a.unit_dir)
            man = json.loads((a.prefix / "current" / "manifest.json").read_text())
            print(json.dumps(w.write_units(man)))
            return 0
        if a.cmd == "join":
            name = a.name or (input("Your name (what your friend will see): ").strip() if sys.stdin.isatty() else "")
            return join(a.prefix, a.invite, name or "friend")
        if a.cmd == "install":
            print(inst.install(a.source, a.channel, a.pubkey, a.version))
        elif a.cmd == "update":
            print(inst.update())
        elif a.cmd == "rollback":
            print(inst.rollback())
        else:
            print(json.dumps(inst.status(), indent=1))
        return 0
    except (InstallError, OSError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
