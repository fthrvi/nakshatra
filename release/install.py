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
  * after switching it runs the release's health checks and restarts its services; if anything fails
    it switches back to the previous release and says so;
  * update never installs an older or equal version, and never a different major version.
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


def _current(prefix: Path):
    c = prefix / "current"
    return os.readlink(c).split("/")[-1] if c.is_symlink() else None


class Installer:
    def __init__(self, prefix: Path = DEFAULT_PREFIX, *, systemd: bool = True, make_venv: bool = True):
        self.prefix = Path(prefix)
        self.systemd = systemd
        self.make_venv = make_venv

    # ── fetch + verify ──
    def fetch_verified(self, source: str, channel: str, pubkey: str, version=None) -> tuple:
        work = Path(tempfile.mkdtemp(prefix="nak-release-"))
        if version is None:
            _fetch(source, f"{channel}/latest.json", work / "latest.json")
            latest = json.loads((work / "latest.json").read_text())
            if not rk.verify(latest, pubkey) or latest.get("channel") != channel:
                raise InstallError("latest.json is not signed by the pinned release key")
            version = latest["version"]
        _fetch(source, f"{channel}/{version}/manifest.json", work / "manifest.json")
        man = json.loads((work / "manifest.json").read_text())
        if man.get("schema") != rk.MANIFEST_SCHEMA or not rk.verify(man, pubkey):
            raise InstallError("manifest.json is not signed by the pinned release key")
        if man["version"] != version or man["channel"] != channel:
            raise InstallError("manifest names a different version or channel than requested")
        files = [(c["file"], c["sha256"]) for c in man["components"]]
        files += [(man["lock"]["file"], man["lock"]["sha256"]), (man["uv"]["file"], man["uv"]["sha256"])]
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
        shutil.copy2(work / man["lock"]["file"], rel / "requirements.lock")
        shutil.copy2(work / "uv", rel / "uv")
        os.chmod(rel / "uv", 0o755)
        (rel / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
        if self.make_venv:
            uv = str(rel / "uv")
            env = dict(os.environ, UV_CACHE_DIR=str(self.prefix / "uv-cache"))
            subprocess.run([uv, "venv", str(rel / "venv"), "--python", man["python"], "--quiet"], check=True, env=env)
            subprocess.run([uv, "pip", "install", "--quiet", "--require-hashes", "-r", str(rel / "requirements.lock"),
                            "--python", str(rel / "venv" / "bin" / "python")], check=True, env=env)
        return rel

    def _ctx(self, man: dict) -> dict:
        cur = self.prefix / "current"
        return {"python": cur / "venv" / "bin" / "python", "home": Path.home(), "prefix": self.prefix}

    def _pythonpath(self, man: dict) -> str:
        cur = self.prefix / "current"
        return ":".join(str(cur / c["name"] / c["pythonpath"]) for c in man["components"])

    # ── services + health ──
    def write_units(self, man: dict) -> list:
        ctx, names = self._ctx(man), []
        UNIT_DIR.mkdir(parents=True, exist_ok=True)
        for name, svc in man.get("services", {}).items():
            cond = f"ConditionPathExists={_render(svc['requires'], ctx)}\n" if svc.get("requires") else ""
            unit = (f"[Unit]\nDescription={svc['description']} (managed by nakshatra-node install.py)\n{cond}\n"
                    f"[Service]\nEnvironment=PYTHONPATH={self._pythonpath(man)}\n"
                    f"ExecStart={' '.join(_render(a, ctx) for a in svc['exec'])}\nRestart=on-failure\nRestartSec=5\n\n"
                    f"[Install]\nWantedBy=default.target\n")
            (UNIT_DIR / f"{name}.service").write_text(unit)
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
        (UNIT_DIR / "nak-update.service").write_text(
            "[Unit]\nDescription=Nakshatra node self-update (signed releases only)\n\n"
            f"[Service]\nType=oneshot\nExecStart={sys.executable} {updater} --prefix {self.prefix} update\n")
        (UNIT_DIR / "nak-update.timer").write_text(
            "[Unit]\nDescription=Hourly Nakshatra node update check\n\n"
            "[Timer]\nOnCalendar=hourly\nRandomizedDelaySec=600\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n")
        if self.systemd:
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            subprocess.run(["systemctl", "--user", "enable", "--now", "nak-update.timer"], check=False)
        return names

    def restart(self, names: list) -> None:
        if self.systemd and names:
            subprocess.run(["systemctl", "--user", "enable", *[f"{n}.service" for n in names]], check=False)
            subprocess.run(["systemctl", "--user", "restart", *[f"{n}.service" for n in names]], check=False)

    def health(self, man: dict) -> None:
        ctx = self._ctx(man)
        env = dict(os.environ, PYTHONPATH=self._pythonpath(man))
        for cmd in man.get("health", []):
            r = subprocess.run([_render(a, ctx) for a in cmd], env=env, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise InstallError(f"health check failed: {' '.join(cmd)}\n{(r.stderr or r.stdout)[-800:]}")

    # ── the operations ──
    def install(self, source: str, channel: str, pubkey: str, version=None, *, allow_downgrade=False) -> str:
        self.prefix.mkdir(parents=True, exist_ok=True)
        cfg = _load_config(self.prefix)
        pinned = cfg.get("pubkey")
        if pinned and pubkey and pinned != pubkey:
            raise InstallError("this node already trusts a different release key; refusing to switch keys here")
        pubkey = pinned or pubkey
        if not pubkey:
            raise InstallError("first install needs --pubkey (the release key to trust)")
        man, work = self.fetch_verified(source, channel, pubkey, version)
        prev = _current(self.prefix)
        if prev:
            if _vt(man["version"]) <= _vt(prev) and not allow_downgrade:
                shutil.rmtree(work, ignore_errors=True)
                return f"already at {prev}; {man['version']} is not newer"
            if _vt(man["version"])[0] != _vt(prev)[0]:
                raise InstallError(f"{man['version']} is a different major version than {prev}; install it explicitly")
        try:
            self.unpack(man, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        _switch(self.prefix, man["version"])
        try:
            names = self.write_units(man)
            self.health(man)
            self.restart(names)
        except Exception as e:
            if prev:
                _switch(self.prefix, prev)
                prev_man = json.loads((self.prefix / "releases" / prev / "manifest.json").read_text())
                self.write_units(prev_man)
                self.restart(list(prev_man.get("services", {})))
                raise InstallError(f"{man['version']} failed after switching and was rolled back to {prev}: {e}")
            raise
        cfg.update({"source": source, "channel": channel, "pubkey": pubkey, "current": man["version"],
                    "previous": prev, "updated": int(time.time())})
        _save_config(self.prefix, cfg)
        return f"installed {man['version']} ({channel})" + (f", previous {prev}" if prev else "")

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
        self.restart(self.write_units(man))
        cfg.update({"current": prev, "previous": cur, "updated": int(time.time())})
        _save_config(self.prefix, cfg)
        return f"rolled back to {prev}"

    def status(self) -> dict:
        cfg = _load_config(self.prefix)
        return {"current": _current(self.prefix), "previous": cfg.get("previous"), "channel": cfg.get("channel"),
                "source": cfg.get("source"), "release_key": (cfg.get("pubkey") or "")[:16]}


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
    a = ap.parse_args(argv)
    inst = Installer(a.prefix)
    try:
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
