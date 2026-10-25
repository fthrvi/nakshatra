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
import base64
import errno
import fcntl
import functools
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
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
SIGNER_READY_TIMEOUT_S = 20.0
INSTALL_LOCK_TIMEOUT_S = 600.0
TRANSITION_SCHEMA = "nak-release-transition/1"
TRANSITION_FILE = "release-transition.json"


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


def _harden_release_url() -> None:
    """Normalize the optional public release-host override without following a symlink."""
    path = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "release-url"
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise InstallError(f"release-url is not a regular file: {path}")
    os.chmod(path, 0o600, follow_symlinks=False)


def _state_dir_fd(prefix: Path, *, create: bool = False) -> int:
    state = prefix / "state"
    if create:
        try:
            state.mkdir(mode=0o700)
        except FileExistsError:
            pass
    fd = os.open(
        state, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        # Older installers created state through generic helpers, so an existing directory may be
        # group/world-readable. Normalize it whenever the installer opens it.
        os.fchmod(fd, 0o700)
    except Exception:
        os.close(fd)
        raise
    return fd


def _save_private_state(prefix: Path, name: str, value: dict) -> None:
    """Atomically persist a private installer recovery journal.

    The permissions prevent disclosure to other OS users. Integrity against another process running
    as this same uid is explicitly out of scope; that requires privilege separation.
    """
    dir_fd = _state_dir_fd(prefix, create=True)
    tmp = f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    fd = -1
    try:
        fd = os.open(
            tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600, dir_fd=dir_fd)
        payload = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode("utf-8")
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        os.chmod(name, 0o600, dir_fd=dir_fd, follow_symlinks=False)
        os.fsync(dir_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        os.close(dir_fd)


def _load_private_state(prefix: Path, name: str) -> dict | None:
    try:
        dir_fd = _state_dir_fd(prefix)
    except FileNotFoundError:
        return None
    fd = -1
    try:
        try:
            fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
        except FileNotFoundError:
            return None
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise InstallError(f"installer state {name} is not a private regular file")
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            value = json.load(stream)
        if not isinstance(value, dict):
            raise InstallError(f"installer state {name} must contain an object")
        return value
    except (OSError, ValueError) as e:
        raise InstallError(f"cannot read installer state {name}: {e}") from e
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(dir_fd)


def _remove_private_state(prefix: Path, name: str) -> None:
    try:
        dir_fd = _state_dir_fd(prefix)
    except FileNotFoundError:
        return
    try:
        try:
            os.unlink(name, dir_fd=dir_fd)
        except FileNotFoundError:
            return
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


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


def _install_lock_is_held(prefix: Path) -> bool:
    """Non-blockingly probe for an exclusive installer operation."""
    try:
        fd = os.open(prefix / ".install.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return False
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise InstallError("installer lock is not a regular file")
        try:
            # Probes are mutually compatible; only the installer's LOCK_EX makes this
            # fail.  Otherwise one observer could look like an installer to another.
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return True
            raise
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
    finally:
        os.close(fd)


def _locked_operation(fn):
    """Serialize a complete public installer operation on this prefix."""
    @functools.wraps(fn)
    def locked(self, *args, **kwargs):
        with self._install_lock():
            if self._lock_depth == 1:
                self._recover_transition()
            return fn(self, *args, **kwargs)
    return locked


def _releases_in_use(rel: Path) -> set:
    """Release dirs referenced by this uid's processes.

    Resolve argv entries while ``current`` still names their release, and inspect cwd/fd links as well as
    maps.  Pure-Python programs need not retain a mapped file after startup, so maps alone is insufficient.
    """
    import re
    root = str(rel.resolve())
    pat = re.compile(re.escape(root) + r"/([^/\s]+)(?:/|$)")
    used, uid = set(), os.getuid()

    def remember(value) -> None:
        if not value:
            return
        value = os.path.realpath(os.fsdecode(value)).removesuffix(" (deleted)")
        used.update(pat.findall(value))

    for pid in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not pid.isdigit():
            continue
        proc = Path("/proc") / pid
        try:
            if proc.stat().st_uid != uid:
                continue
            cwd = os.path.realpath(proc / "cwd")
            remember(cwd)
            with open(proc / "maps", errors="replace") as f:
                used.update(pat.findall(f.read()))
            with open(proc / "cmdline", "rb") as f:
                argv = f.read().split(b"\0")
            for arg in argv:
                text = os.fsdecode(arg)
                # Keep direct release paths even if a file has since disappeared, and resolve paths
                # containing `current` before an install retargets that symlink.
                used.update(pat.findall(text))
                if text and (text.startswith("/") or "/" in text):
                    remember(text if text.startswith("/") else os.path.join(cwd, text))
            for fd in (proc / "fd").iterdir():
                remember(fd)
        except OSError:
            continue
    return used


def _uv_env(prefix: Path) -> dict:
    """Environment for every uv call. A locked sdist (llama-cpp-python) compiles C++ at one job per
    thread by default, and on a 23 GB, 32-thread WSL box g++ died with an internal compiler error
    (bro, 2026-10-13). Cap the build at ~3 GB per job unless the operator already chose a level."""
    env = dict(os.environ, UV_CACHE_DIR=str(prefix / "uv-cache"))
    if "CMAKE_BUILD_PARALLEL_LEVEL" not in env:
        jobs = os.cpu_count() or 1
        try:
            with open("/proc/meminfo") as f:
                avail_kb = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable:"))
            jobs = min(jobs, avail_kb // (3 * 1024 * 1024))
        except (OSError, StopIteration, ValueError):
            pass
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(max(1, jobs))
    return env


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
        self._lock_depth = 0
        self._lock_owner = None

    class _Lock:
        def __init__(self, installer):
            self.installer = installer
            self.file = None

        def __enter__(self):
            inst = self.installer
            owner = threading.get_ident()
            if inst._lock_depth and inst._lock_owner == owner:
                inst._lock_depth += 1
                return self
            inst.prefix.mkdir(parents=True, exist_ok=True)
            lock = inst.prefix / ".install.lock"
            fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
            self.file = os.fdopen(fd, "r+")
            deadline = time.monotonic() + INSTALL_LOCK_TIMEOUT_S
            while True:
                try:
                    fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as e:
                    if e.errno not in (errno.EACCES, errno.EAGAIN):
                        self.file.close()
                        raise
                    if time.monotonic() >= deadline:
                        self.file.close()
                        raise InstallError(
                            f"another installer operation still holds {lock} after "
                            f"{INSTALL_LOCK_TIMEOUT_S:g}s")
                    time.sleep(0.1)
            inst._lock_depth = 1
            inst._lock_owner = owner
            return self

        def __exit__(self, exc_type, exc, tb):
            inst = self.installer
            inst._lock_depth -= 1
            if self.file is not None:
                try:
                    fcntl.flock(self.file, fcntl.LOCK_UN)
                finally:
                    self.file.close()
                    inst._lock_owner = None
            return False

    def _install_lock(self):
        return self._Lock(self)

    @property
    def _transition_path(self) -> Path:
        return self.prefix / "state" / TRANSITION_FILE

    @staticmethod
    def _validated_transition(value: dict) -> tuple[str, str, str | None, str]:
        fields = set(value) if isinstance(value, dict) else set()
        if value.get("schema") != TRANSITION_SCHEMA or fields not in (
                {"schema", "pubkey", "from", "to"},
                {"schema", "pubkey", "previous_pubkey", "from", "to"}):
            raise InstallError("release transition record is malformed")
        pubkey = value["pubkey"]
        previous_pubkey = value.get("previous_pubkey", pubkey)
        from_version, to_version = value["from"], value["to"]
        try:
            for key in (pubkey, previous_pubkey):
                if not isinstance(key, str) or len(key) != 64 \
                        or any(char not in "0123456789abcdef" for char in key):
                    raise ValueError
            if from_version is not None:
                if not isinstance(from_version, str):
                    raise ValueError
                _vt(from_version)
            if not isinstance(to_version, str):
                raise ValueError
            _vt(to_version)
        except (TypeError, ValueError):
            raise InstallError("release transition record is malformed")
        if from_version == to_version:
            raise InstallError("release transition must change versions")
        return previous_pubkey, pubkey, from_version, to_version

    def _begin_transition(self, pubkey: str, from_version: str | None, to_version: str,
                          *, new_pubkey: str | None = None) -> None:
        target_pubkey = new_pubkey or pubkey
        transition = {
            "schema": TRANSITION_SCHEMA,
            "pubkey": target_pubkey,
            "from": from_version,
            "to": to_version,
        }
        if target_pubkey != pubkey:
            transition["previous_pubkey"] = pubkey
        self._validated_transition(transition)
        _save_private_state(self.prefix, TRANSITION_FILE, transition)

    def _clear_transition(self) -> None:
        _remove_private_state(self.prefix, TRANSITION_FILE)

    def _restore_release(self, version: str | None, *, restart: bool) -> None:
        if version is None:
            (self.prefix / "current").unlink(missing_ok=True)
        else:
            _switch(self.prefix, version)
        if version is not None and restart:
            try:
                man = json.loads((self.prefix / "releases" / version / "manifest.json").read_text())
                self.restart(self.write_units(man), man, force=True)
            except Exception:
                # A handled restoration failure must not leave authority to expose the failed target.
                self._clear_transition()
                raise
        # A real process death during restoration leaves the record durable so the next invocation
        # retries the conservative restore. Handled success/failure always removes that authority.
        self._clear_transition()

    def _recover_transition(self) -> dict | None:
        """Resolve a transaction left by a killed installer before doing any new mutation."""
        transition = _load_private_state(self.prefix, TRANSITION_FILE)
        if transition is None:
            return None
        previous_pubkey, pubkey, from_version, to_version = self._validated_transition(transition)
        recovered = {"state": "stale", "from": from_version, "to": to_version}
        cfg = _load_config(self.prefix)
        committed = cfg.get("current")
        if committed != from_version and committed != to_version:
            raise InstallError(
                f"release transition {from_version!r}->{to_version!r} conflicts with "
                f"committed version {committed!r}")
        expected_key = pubkey if committed == to_version else previous_pubkey
        if cfg.get("pubkey") not in (None, expected_key):
            raise InstallError("release transition key conflicts with the pinned release key")
        current = _current(self.prefix)
        if committed == to_version and current == to_version:
            # Config was committed and the process died before removing the record.
            self._clear_transition()
            recovered["recovery"] = "cleared-committed"
            return recovered
        if committed != from_version:
            raise InstallError("release transition has an inconsistent committed version")
        if current == from_version:
            # Either the process died before switching, or while restoring after a failed switch.
            # Conservatively re-apply the committed units/services; both states resolve identically.
            self._restore_release(from_version, restart=from_version is not None)
            recovered["recovery"] = "reapplied-committed"
            return recovered
        if current != to_version:
            raise InstallError(
                f"release transition target is {to_version!r}, but current is {current!r}")
        # A killed process cannot prove that health/restarts completed. Safely restore the last
        # committed release; a first install restores the pre-install state of no current release.
        self._restore_release(from_version, restart=from_version is not None)
        recovered["recovery"] = "restored-committed"
        return recovered

    def _bootstrap_pre_transition_handoff(self, man: dict) -> None:
        """Journal an old installer's already-exposed switch before it runs health/restarts.

        Installers predating ``nak-release-transition/1`` switch ``current`` and then invoke the
        installer shipped by the target release to write units. Their parent process still holds the
        prefix flock, which is the only bounded compatibility seam where the new installer may create
        the exact committed-version -> current-link journal on their behalf.
        """
        cfg = _load_config(self.prefix)
        committed = cfg.get("current")
        current = _current(self.prefix)
        if current == committed:
            return
        if not _install_lock_is_held(self.prefix):
            raise InstallError(
                "current differs from config.json but no installer holds the prefix lock")
        transition = _load_private_state(self.prefix, TRANSITION_FILE)
        if transition is not None:
            recorded_previous_key, recorded_key, recorded_from, recorded_to = \
                self._validated_transition(transition)
            if (recorded_from, recorded_to) != (committed, current):
                raise InstallError("release transition does not match the locked installer handoff")
            configured_key = cfg.get("pubkey")
            if configured_key is not None and configured_key != recorded_previous_key:
                raise InstallError("release transition key conflicts with the pinned release key")
            pubkey = recorded_key
        else:
            pubkey = cfg.get("pubkey")
        if not isinstance(pubkey, str) or len(pubkey) != 64 \
                or any(char not in "0123456789abcdef" for char in pubkey):
            raise InstallError("config.json has no valid pinned release key")
        if (committed is not None and not isinstance(committed, str)) or not isinstance(current, str):
            raise InstallError("installer handoff requires valid committed and current releases")
        try:
            if committed is not None:
                _vt(committed)
            _vt(current)
        except (TypeError, ValueError) as e:
            raise InstallError("pre-transition handoff has malformed release versions") from e
        if os.readlink(self.prefix / "current") != f"releases/{current}":
            raise InstallError("current must be exactly the relative link releases/<version>")
        if man.get("schema") != rk.MANIFEST_SCHEMA or man.get("version") != current \
                or not rk.verify(man, pubkey):
            raise InstallError(
                "current release manifest is not the exact target signed by the pinned release key")
        if transition is None:
            if committed is None:
                raise InstallError("pre-transition handoff requires an existing committed release")
            self._begin_transition(pubkey, committed, current)

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
    def _keyroll_tip(self, source: str, channel: str, work: Path, pubkey: str,
                     forbidden: tuple[str, ...]) -> str:
        path = work / "keyroll.json"
        try:
            if not path.exists():
                _fetch(source, f"{channel}/keyroll.json", path)
            chain = json.loads(path.read_text())
            return rk.keyroll_tip(chain, pubkey, forbidden=forbidden)
        except Exception as e:
            raise InstallError(
                "release metadata is not signed by the pinned release key and keyroll.json does not provide "
                f"a valid roll from the pinned key: {e}") from e

    def _release_signature_key(self, obj: dict, source: str, channel: str, work: Path,
                               pubkey: str, forbidden: tuple[str, ...]) -> str:
        if rk.verify(obj, pubkey):
            return pubkey
        keyid = ((obj.get("sig") or {}).get("keyid")
                 if isinstance(obj, dict) else None)
        shown = keyid if isinstance(keyid, str) else "malformed"
        try:
            tip = self._keyroll_tip(source, channel, work, pubkey, forbidden)
        except InstallError as e:
            raise InstallError(f"release signature keyid {shown!r} is not trusted: {e}") from e
        if not rk.verify(obj, tip):
            raise InstallError(
                f"release signature keyid {shown!r} is not the validated key-roll tip")
        return tip

    def fetch_verified(self, source: str, channel: str, pubkey: str, version=None,
                       *, forbidden_pubkeys=()) -> tuple:
        work = Path(tempfile.mkdtemp(prefix="nak-release-"))
        latest_key = None
        if version is None:
            _fetch(source, f"{channel}/latest.json", work / "latest.json")
            latest = json.loads((work / "latest.json").read_text())
            latest_key = self._release_signature_key(
                latest, source, channel, work, pubkey, tuple(forbidden_pubkeys))
            if latest.get("channel") != channel:
                raise InstallError("latest.json names a different release channel")
            # A host could replay an OLD signed latest.json forever (audit 2026-10-03, finding 1):
            # a signed pointer carries an expiry, and a stale one is refused.
            exp = latest.get("expires_at")
            if isinstance(exp, int) and time.time() > exp:
                raise InstallError("the release host is serving an expired latest.json (stale or replayed)")
            version = latest["version"]
        _fetch(source, f"{channel}/{version}/manifest.json", work / "manifest.json")
        man = json.loads((work / "manifest.json").read_text())
        manifest_key = self._release_signature_key(
            man, source, channel, work, pubkey, tuple(forbidden_pubkeys))
        if man.get("schema") != rk.MANIFEST_SCHEMA:
            raise InstallError("manifest.json has an unsupported schema")
        if latest_key is not None and manifest_key != latest_key:
            raise InstallError("latest.json and manifest.json are signed by different release keys")
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
        self._verified_release_key = manifest_key
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
        # Keep every signed profile lock beside the release even when it was not active during the
        # first install. An opt-in service can be enabled later (notably `nak worker on`) and must be
        # able to hydrate its declared profile without trusting or fetching a new, unsigned lock.
        for profile in (man.get("profiles") or {}).values():
            pf = profile["lock"]["file"]
            shutil.copy2(work / pf, rel / pf)
        selected_profile = self._selected_profile(man)
        if selected_profile:
            # A profile lock is the WHOLE environment (base + profile, resolved together). A declared
            # superset may stand in for every active subset; unrelated active profiles remain an error.
            pf = man["profiles"][selected_profile]["lock"]["file"]
            lock_to_install = rel / pf
        shutil.copy2(work / "uv", rel / "uv")
        os.chmod(rel / "uv", 0o755)
        (rel / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
        if self.make_venv:
            uv = str(rel / "uv")
            env = _uv_env(self.prefix)
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

    def _selected_profile(self, man: dict) -> str | None:
        """Choose the one active lock that covers all active release profiles."""
        active = self.active_profiles(man)
        if not active:
            return None
        if len(active) == 1:
            return active[0]
        profiles = man.get("profiles") or {}
        candidates = [
            candidate for candidate in active
            if all(
                name == candidate
                or candidate in profiles[name].get("satisfied_by", [])
                for name in active
            )
        ]
        if len(candidates) == 1:
            return candidates[0]
        raise InstallError(
            "incompatible release profiles are active on this node: " + ", ".join(active)
        )

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
        requires (opt-in per node: ConditionPathExists), after, env_file, optional_env_files,
        environment {K: V},
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
        for path in svc.get("optional_env_files", []):
            service.append(f"EnvironmentFile=-{_render(path, ctx)}")
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
        if svc.get("runtime_directory"):
            # ProtectHome=read-only also makes /run/user/<uid> read-only; systemd creates this one writable,
            # private (0700) and owned by the user (2026-10-04: bro's sidecar could not create its socket dir).
            service += [f"RuntimeDirectory={svc['runtime_directory']}", "RuntimeDirectoryMode=0700"]
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
            unit_path = self.unit_dir / f"{name}.service"
            # A node that has NOT opted into a service gets no unit file for it at all: writing one
            # anyway would overwrite whatever (possibly hand-written, possibly live) unit carries that name.
            if svc.get("requires") and not Path(_render(svc["requires"], ctx)).exists():
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
            if unit_path.exists() and not self._managed_unit(unit_path):
                raise InstallError(f"refusing to replace operator-owned unit {unit_path}")
            unit_path.write_text(self._unit_text(svc, ctx, man))
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
            UNIT_MARKER + "\n[Unit]\nDescription=Nakshatra node self-update (signed releases only)\n\n"
            f"[Service]\nType=oneshot\nExecStart={sys.executable} {updater} --prefix {self.prefix} update\n")
        (self.unit_dir / "nak-update.timer").write_text(
            UNIT_MARKER + "\n[Unit]\nDescription=Hourly Nakshatra node update check\n\n"
            "[Timer]\nOnCalendar=hourly\nRandomizedDelaySec=600\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n")
        if self.systemd:
            if subprocess.run(["systemctl", "--user", "daemon-reload"], check=False).returncode != 0:
                # Restarting now would run systemd's cached OLD unit against the new release.
                raise InstallError("systemctl --user daemon-reload failed; not restarting anything")
            subprocess.run(["systemctl", "--user", "enable", "--now", "nak-update.timer"], check=False)
        return names

    def _current_service(self, name: str) -> tuple[dict, dict, Path]:
        """Return the current signed-release manifest entry and its rendered opt-in path."""
        version = _current(self.prefix)
        if not version:
            raise InstallError("no current installed release")
        manifest_path = self.prefix / "current" / "manifest.json"
        try:
            man = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as e:
            raise InstallError(f"cannot read the current release manifest: {e}")
        if man.get("version") != version:
            raise InstallError("current release manifest does not match the current symlink")
        svc = (man.get("services") or {}).get(name)
        if not isinstance(svc, dict):
            raise InstallError(f"service {name!r} is not in the current release manifest")
        if not svc.get("requires"):
            raise InstallError(f"service {name!r} is not an opt-in service")
        requires = Path(_render(svc["requires"], self._ctx(man)))
        return man, svc, requires

    def _ensure_service_profile(self, svc: dict, man: dict) -> None:
        """Hydrate the hash-locked release profile an opt-in service declares.

        Services are commonly enabled after the node was installed. Keeping the profile lock in the
        signed release and installing it here makes that transition immediate while preserving the
        same --require-hashes boundary as initial installation.
        """
        name = svc.get("profile")
        if not name:
            return
        profile = (man.get("profiles") or {}).get(name)
        if not isinstance(profile, dict):
            raise InstallError(f"service requires unknown release profile {name!r}")
        marker = Path(_render(profile["requires"], self._ctx(man)))
        active = self.active_profiles(man)
        satisfying = [candidate for candidate in profile.get("satisfied_by", []) if candidate in active]
        if name not in active and not satisfying:
            raise InstallError(f"service profile {name!r} is not opted in: required file does not exist: {marker}")
        selected_profile = self._selected_profile(man)
        if selected_profile != name and selected_profile not in profile.get("satisfied_by", []):
            raise InstallError(
                f"active release profile {selected_profile!r} does not satisfy service profile {name!r}"
            )
        if not self.make_venv:
            return
        rel = self.prefix / "current"
        lock = rel / man["profiles"][selected_profile]["lock"]["file"]
        uv = rel / "uv"
        python = rel / "venv" / "bin" / "python"
        if not lock.is_file() or not uv.is_file() or not python.is_file():
            raise InstallError(
                f"current release cannot hydrate profile {selected_profile!r}; its lock, uv, or venv is missing"
            )
        env = _uv_env(self.prefix)
        try:
            subprocess.run([str(uv), "pip", "install", "--quiet", "--require-hashes", "-r", str(lock),
                            "--python", str(python)], check=True, env=env)
        except subprocess.CalledProcessError as e:
            raise InstallError(
                f"could not install release profile {selected_profile!r}: uv exited {e.returncode}"
            ) from e

    @_locked_operation
    def enable(self, name: str) -> str:
        """Write and start exactly one current-release opt-in service."""
        man, svc, requires = self._current_service(name)
        if not requires.exists():
            raise InstallError(f"{name} is not opted in: required file does not exist: {requires}")
        self.unit_dir.mkdir(parents=True, exist_ok=True)
        unit = self.unit_dir / f"{name}.service"
        if unit.exists() and not self._managed_unit(unit):
            raise InstallError(f"refusing to replace operator-owned unit {unit}")
        self._ensure_service_profile(svc, man)
        unit.write_text(self._unit_text(svc, self._ctx(man), man))
        if subprocess.run(["systemctl", "--user", "daemon-reload"], check=False).returncode != 0:
            raise InstallError("systemctl --user daemon-reload failed; service was not enabled")
        started = subprocess.run(
            ["systemctl", "--user", "enable", "--now", f"{name}.service"], check=False)
        if started.returncode != 0:
            raise InstallError(f"systemctl enable --now {name}.service failed with exit status {started.returncode}")
        healthy, detail = self._wait_stable(name, SERVICE_ACTIVE_TIMEOUT_S, SERVICE_STABILITY_S)
        if not healthy:
            self._record_service_issue(name, "failed", detail)
            raise InstallError(f"{name}.service {detail}")
        self._clear_service_issue(name)
        return f"enabled {name}.service"

    @_locked_operation
    def disable(self, name: str) -> str:
        """Stop and remove exactly one managed opt-in service, then remove its opt-in file."""
        _man, _svc, requires = self._current_service(name)
        unit = self.unit_dir / f"{name}.service"
        failure = None
        removed = False
        if unit.exists() and self._managed_unit(unit):
            stopped = subprocess.run(
                ["systemctl", "--user", "disable", "--now", f"{name}.service"], check=False)
            if stopped.returncode == 0:
                unit.unlink()
                removed = True
            else:
                failure = f"systemctl disable --now failed with exit status {stopped.returncode}"
                self._record_service_issue(name, "removal-failed", failure + "; managed unit retained")
        # If systemd could not stop the service, its opt-in is still authoritative state: retain it
        # so a later disable retries instead of leaving a live unit with its recovery handle erased.
        if failure:
            raise InstallError(f"could not fully disable {name}.service: {failure}")
        requires.unlink(missing_ok=True)
        if removed and subprocess.run(["systemctl", "--user", "daemon-reload"], check=False).returncode != 0:
            failure = "systemctl --user daemon-reload failed after removing the unit"
            self._record_service_issue(name, "removal-failed", failure)
        if failure:
            raise InstallError(f"could not fully disable {name}.service: {failure}")
        self._clear_service_issue(name)
        return f"disabled {name}.service"

    @_locked_operation
    def restart_service(self, name: str) -> str:
        """Rewrite and restart exactly one already-enabled current-release service.

        This is the narrow mutation used when an operator command changes an
        EnvironmentFile consumed by a service.  Keeping it in the verified
        installer preserves the signed manifest/unit boundary instead of
        teaching callers to invoke systemctl on release-owned units directly.
        """
        man, svc, requires = self._current_service(name)
        if not requires.exists():
            raise InstallError(f"{name} is not opted in: required file does not exist: {requires}")
        unit = self.unit_dir / f"{name}.service"
        if not unit.exists() or not self._managed_unit(unit):
            raise InstallError(f"{name}.service is not an enabled release-managed service")
        enabled = subprocess.run(
            ["systemctl", "--user", "is-enabled", "--quiet", f"{name}.service"], check=False)
        if enabled.returncode != 0:
            raise InstallError(f"{name}.service is not enabled")
        self._ensure_service_profile(svc, man)
        unit.write_text(self._unit_text(svc, self._ctx(man), man))
        if subprocess.run(["systemctl", "--user", "daemon-reload"], check=False).returncode != 0:
            raise InstallError("systemctl --user daemon-reload failed; service was not restarted")
        restarted = subprocess.run(
            ["systemctl", "--user", "restart", f"{name}.service"], check=False)
        if restarted.returncode != 0:
            raise InstallError(f"systemctl restart {name}.service failed with exit status {restarted.returncode}")
        healthy, detail = self._wait_stable(name, SERVICE_ACTIVE_TIMEOUT_S, SERVICE_STABILITY_S)
        if not healthy:
            self._record_service_issue(name, "failed", detail)
            raise InstallError(f"{name}.service {detail}")
        self._clear_service_issue(name)
        return f"restarted {name}.service"

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
        # Manifest JSON is canonicalized with sorted keys, which puts nak-net before nak-signer.
        # That ordering is unsafe across signer protocol changes: net can start against the old
        # signer and fail once before systemd retries it. Preserve relative order otherwise.
        todo.sort(key=lambda item: 0 if item[0] == "nak-signer" else 2 if item[0] == "nak-net" else 1)
        if todo:
            subprocess.run(["systemctl", "--user", "enable", *[f"{n}.service" for n, _ in todo]], check=False)
        restarted = []
        for n, fp in todo:
            if n == "nak-net" and not self._wait_signer_ready(SIGNER_READY_TIMEOUT_S):
                detail = "signer socket did not answer data_key; restart deferred"
                self._record_service_issue(n, "pending", detail)
                if force:
                    raise InstallError(f"{n}.service {detail}")
                continue
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

    @staticmethod
    def _signer_answers() -> bool:
        """Probe the operation nak-net needs, not merely whether the socket path exists."""
        path = Path(os.environ.get(
            "NAK_SIGNER_SOCK", Path.home() / ".sthambha" / "signer" / "signer.sock"))
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect(str(path))
            with sock.makefile("rwb") as stream:
                stream.write(b'{"op":"data_key"}\n')
                stream.flush()
                line = stream.readline(65537)
            reply = json.loads(line) if line and len(line) <= 65536 else {}
            encoded = reply.get("data_key")
            if reply.get("ok") is not True or not isinstance(encoded, str):
                return False
            return len(base64.b64decode(encoded, validate=True)) == 32
        except (OSError, ValueError, TypeError):
            return False
        finally:
            sock.close()

    def _wait_signer_ready(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            if self._signer_answers():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.2)

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
    @_locked_operation
    def install(self, source: str, channel: str, pubkey: str, version=None, *, allow_downgrade=False,
                min_version: str = "") -> str:
        self.prefix.mkdir(parents=True, exist_ok=True)
        _harden_release_url()
        cfg = _load_config(self.prefix)
        pinned = cfg.get("pubkey")
        if pinned and pubkey and pinned != pubkey:
            raise InstallError("this node already trusts a different release key; refusing to switch keys here")
        pubkey = pinned or pubkey
        if not pubkey:
            raise InstallError("first install needs --pubkey (the release key to trust)")
        history = cfg.get("pubkey_history", [])
        if not isinstance(history, list) or any(not isinstance(key, str) for key in history):
            raise InstallError("config.json has a malformed release-key history")
        man, work = self.fetch_verified(
            source, channel, pubkey, version, forbidden_pubkeys=tuple(history))
        release_pubkey = self._verified_release_key
        if min_version and _vt(man["version"]) < _vt(min_version):
            shutil.rmtree(work, ignore_errors=True)
            raise InstallError(f"the release host offers {man['version']}, older than {min_version} which your "
                               f"friend's invite names; refusing (stale or malicious host)")
        # Installed = a COMPLETED install recorded in config.json, not merely a `current` link: a
        # first install that died half-way must be retried, not reported as "already at X".
        prev = _current(self.prefix) if cfg.get("current") else None
        if prev:
            if man["version"] == prev:
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
        # Resolve `current/...` argv while it still points to the old release. A pure-Python service
        # can have no old-release mmap/fd left by the time current is switched.
        pre_switch_used = _releases_in_use(self.prefix / "releases")
        self._begin_transition(pubkey, prev, man["version"], new_pubkey=release_pubkey)
        try:
            _switch(self.prefix, man["version"])
            names = self._write_units_by_new_installer(man)
            self.health(man)
            self.restart(names, man)
        except Exception as e:
            if prev:
                self._restore_release(prev, restart=True)
                raise InstallError(f"{man['version']} failed after switching and was rolled back to {prev}: {e}")
            # First install failed: leave NO half-installed node behind (no `current`, no release dir),
            # so the next attempt starts clean instead of believing it is already installed.
            self._restore_release(None, restart=False)
            shutil.rmtree(new_dir, ignore_errors=True)
            raise InstallError(f"first install of {man['version']} failed and was removed: {e}")
        if release_pubkey != pubkey:
            history = [*history, pubkey]
            # Preserve order while avoiding duplication from hand-edited legacy configs.
            cfg["pubkey_history"] = list(dict.fromkeys(history))
        cfg.update({"source": source, "channel": channel, "pubkey": release_pubkey, "current": man["version"],
                    "previous": prev, "updated": int(time.time())})
        try:
            _save_config(self.prefix, cfg)
        except Exception as e:
            self._restore_release(prev, restart=prev is not None)
            if prev is None:
                shutil.rmtree(new_dir, ignore_errors=True)
            raise InstallError(f"could not commit release {man['version']}; restored {prev!r}: {e}") from e
        self._clear_transition()
        self.prune(keep=(man["version"], prev, *pre_switch_used))
        return (f"installed {man['version']} ({channel})" + (f", previous {prev}" if prev else "") +
                self._service_issue_summary())

    KEEP_RELEASES = 3

    @_locked_operation
    def prune(self, keep=()) -> list:
        """Delete old release dirs: always keep `current`, `previous` and the newest KEEP_RELEASES.
        (2026-10-04: the hub had 28 releases, 1.9 GB.) Never fails an install."""
        rel = self.prefix / "releases"
        try:
            dirs = sorted((d for d in rel.iterdir() if d.is_dir() and not d.name.startswith(".prune-")),
                          key=lambda d: d.stat().st_mtime, reverse=True)
        except OSError:
            return []
        cfg = _load_config(self.prefix)
        protect = ({k for k in keep if k} | {cfg.get("current"), cfg.get("previous")} |
                   {d.name for d in dirs[:self.KEEP_RELEASES]})
        protect.discard(None)
        cur = _current(self.prefix)
        if cur:
            protect.add(cur)
        # A service that was NOT restarted (its inputs did not change) still runs from an older release dir;
        # deleting it would break that process's next lazy import (the hub's signer ran from 0.10.8 under 0.11.0).
        protect |= _releases_in_use(rel)
        removed = []
        for d in dirs:
            if d.name in protect or not d.exists():
                continue
            # Rename first: after this atomic step no new process can enter via the release's old
            # pathname. Re-scan while quarantined; cwd/fd/maps expose the quarantine name and argv
            # still exposes the original one. Restore instead of deleting if either is observed.
            quarantine = rel / f".prune-{d.name}-{os.getpid()}-{time.time_ns()}"
            try:
                os.replace(d, quarantine)
                live = _releases_in_use(rel)
                if d.name in live or quarantine.name in live:
                    os.replace(quarantine, d)
                    protect.add(d.name)
                    continue
                shutil.rmtree(quarantine)
                removed.append(d.name)
            except OSError:
                if quarantine.exists() and not d.exists():
                    try:
                        os.replace(quarantine, d)
                    except OSError:
                        pass
        return removed

    @_locked_operation
    def update(self) -> str:
        cfg = _load_config(self.prefix)
        if not cfg:
            raise InstallError("not installed yet; run install first")
        return self.install(cfg["source"], cfg["channel"], cfg["pubkey"])

    @_locked_operation
    def rollback(self) -> str:
        cfg = _load_config(self.prefix)
        prev = cfg.get("previous")
        if not prev or not (self.prefix / "releases" / prev).exists():
            raise InstallError("no previous release to roll back to")
        cur = _current(self.prefix)
        man = json.loads((self.prefix / "releases" / prev / "manifest.json").read_text())
        if man.get("schema") != rk.MANIFEST_SCHEMA or not rk.verify(man, cfg["pubkey"]):
            raise InstallError(
                "previous release is not signed by the currently pinned release key; "
                "refusing a release-key rollback")
        self._begin_transition(cfg["pubkey"], cur, prev)
        try:
            _switch(self.prefix, prev)
            self.restart(self.write_units(man), man, force=True)
        except Exception as e:
            self._restore_release(cur, restart=cur is not None)
            raise InstallError(f"rollback to {prev} failed; restored {cur}: {e}") from e
        cfg.update({"current": prev, "previous": cur, "updated": int(time.time())})
        try:
            _save_config(self.prefix, cfg)
        except Exception as e:
            self._restore_release(cur, restart=cur is not None)
            raise InstallError(f"could not commit rollback to {prev}; restored {cur}: {e}") from e
        self._clear_transition()
        return f"rolled back to {prev}" + self._service_issue_summary()

    def status(self) -> dict:
        transition = _load_private_state(self.prefix, TRANSITION_FILE)
        transition_status = None
        if transition is not None:
            _previous_pubkey, _pubkey, from_version, to_version = self._validated_transition(transition)
            if _install_lock_is_held(self.prefix):
                transition_status = {"state": "pending", "from": from_version, "to": to_version}
            else:
                # A journal without the lock is never trust authority. Recover it under the lock and
                # report what status repaired so an operator can diagnose the interrupted install.
                with self._install_lock():
                    transition_status = self._recover_transition()
        cfg = _load_config(self.prefix)
        return {"current": _current(self.prefix), "previous": cfg.get("previous"), "channel": cfg.get("channel"),
                "source": cfg.get("source"), "release_key": (cfg.get("pubkey") or "")[:16],
                "release_transition": transition_status,
                "service_issues": _load_json(self._service_state_path).get("services", {})}


# ── join: the one command a newcomer pastes ─────────────────────────────────────────────────────
#
#   python3 install.py join '<nki1… invite | nka1… address>' [--name "what they see"]
#
# 1. checks the invite is signed by the friend who made it and has not expired
# 2. installs the node from the release the invite names, pinning the release key the invite names
# 3. runs the release's own setup: keys (TEST custody), signer, a delegated agent, services
# 4. asks the friend to connect (they still have to accept)

INVITE_PREFIX = "nki1."
ADDRESS_PREFIX = "nka1."


def parse_invite(code: str, now=None) -> dict:
    """Verify an invite with no third-party crypto (releasekit's pure Ed25519). Raises InstallError."""
    import base64
    from urllib.parse import urlsplit
    now = int(now if now is not None else time.time())
    code = (code or "").strip()
    prefix = ADDRESS_PREFIX if code.startswith(ADDRESS_PREFIX) else \
        INVITE_PREFIX if code.startswith(INVITE_PREFIX) else ""
    if not prefix or len(code) > 8192:
        raise InstallError("that is not a Nakshatra invite or request address (expected nki1. or nka1.)")
    try:
        body = code[len(prefix):]
        raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        inv = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise InstallError("the invite is damaged (copy the whole line again)")
    if not isinstance(inv, dict) or not isinstance(inv.get("inviter"), str) or len(inv["inviter"]) != 64:
        raise InstallError("the invite is damaged (copy the whole line again)")
    if not rk.verify(inv, inv["inviter"]):
        raise InstallError("the invite's signature does not verify: it was changed or is not genuine")
    if not isinstance(inv.get("expires_at"), int) or now >= inv["expires_at"]:
        raise InstallError("the invite has expired: ask your friend for a new one")
    is_address = prefix == ADDRESS_PREFIX
    if is_address:
        canonical_address = json.dumps(
            inv, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if raw != canonical_address:
            raise InstallError("the request address encoding is not canonical")
        expected = {"v", "kind", "inviter", "inviter_node", "peers", "aid", "ik", "expires_at",
                    "release", "sig"}
        if "from_name" in inv:
            expected.add("from_name")
        if set(inv) != expected or inv.get("v") != 1 or inv.get("kind") != "address":
            raise InstallError("the request address fields are malformed")
        for key, length in (("inviter", 64), ("inviter_node", 64), ("aid", 32), ("ik", 64)):
            value = inv.get(key)
            if not isinstance(value, str) or len(value) != length or \
                    any(ch not in "0123456789abcdef" for ch in value):
                raise InstallError(f"the request address {key} is malformed")
        peers = inv.get("peers")
        if not isinstance(peers, list) or not 1 <= len(peers) <= 16:
            raise InstallError("the request address peer hints are malformed")
        for peer in peers:
            if not isinstance(peer, dict) or set(peer) != {"node", "addrs"} or \
                    not isinstance(peer.get("node"), str) or len(peer["node"]) != 64 or \
                    any(ch not in "0123456789abcdef" for ch in peer["node"]) or \
                    not isinstance(peer.get("addrs"), list) or len(peer["addrs"]) > 16 or \
                    not all(isinstance(addr, str) and 0 < len(addr) <= 512 for addr in peer["addrs"]):
                raise InstallError("the request address peer hints are malformed")
        if "from_name" in inv and (not isinstance(inv["from_name"], str) or len(inv["from_name"]) > 40):
            raise InstallError("the request address name is malformed")
        if inv["expires_at"] > now + 365 * 86400:
            raise InstallError("the request address expiry is more than 365 days away")
    elif inv.get("kind") == "address":
        raise InstallError("an address payload must use the nka1. prefix")
    kind = inv.get("kind", "contact")
    worker = inv.get("worker")
    consumer = inv.get("consumer")
    if not is_address and kind not in {"contact", "worker", "consumer"}:
        raise InstallError("the invite kind is not supported by this installer")
    if not is_address and kind == "contact" and "worker" in inv:
        raise InstallError("the contact invite contains unexpected worker details")
    if not is_address and kind == "contact" and "consumer" in inv:
        raise InstallError("the contact invite contains unexpected consumer details")
    if not is_address and kind in {"worker", "consumer"}:
        details = worker if kind == "worker" else consumer
        expected = {"redeem", "code", "tier"} | ({"starter"} if kind == "consumer" else set())
        if not isinstance(details, dict) or set(details) != expected or \
                (kind == "worker" and "consumer" in inv) or (kind == "consumer" and "worker" in inv):
            raise InstallError(f"the {kind} invite details are malformed")
        try:
            redeem = urlsplit(details.get("redeem"))
            host, port = redeem.hostname, redeem.port
            authority = redeem.netloc.rsplit("@", 1)[-1]
        except (TypeError, ValueError):
            redeem = host = port = authority = None
        if redeem is None or redeem.scheme not in {"http", "https"} or not host or \
                authority.endswith(":") or (port is not None and not 1 <= port <= 65535) or \
                not isinstance(details.get("code"), str) or not details["code"].strip() or \
                len(details["code"]) > 64 or details.get("tier") not in {"stranger", "known", "trusted"} or \
                (kind == "consumer" and (type(details.get("starter")) is not int or
                                         not 0 <= details["starter"] <= 10**12)):
            raise InstallError(f"the {kind} invite details are malformed")
    rel = inv.get("release")
    if not isinstance(rel, dict) or not all(isinstance(rel.get(k), str) for k in ("url", "channel", "pubkey")):
        raise InstallError("the invite does not say where to get Nakshatra (your friend's node is too old)")
    if is_address and (set(rel) != {"url", "channel", "pubkey", "version"} or
                       not all(isinstance(value, str) and value for value in rel.values()) or
                       not rel["url"].startswith(("http://", "https://")) or
                       len(rel["pubkey"]) != 64 or
                       any(ch not in "0123456789abcdef" for ch in rel["pubkey"])):
        raise InstallError("the request address release entry is malformed")
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
    artifact = "request address" if inv.get("kind") == "address" else "invite"
    print(f"{artifact} is intact and signed by key {inv['inviter'][:16]}… (check where you found it "
          f"that this is theirs); installing Nakshatra from {rel['url']} ({rel['channel']})", flush=True)
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
    env = dict(os.environ, PYTHONPATH=inst._pythonpath(man), NAK_NODE_PREFIX=str(prefix))
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
    p.add_argument("--include-updater", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--unit-dir", default=None)
    p = sub.add_parser("enable", help="write and start one current-release opt-in service")
    p.add_argument("service")
    p = sub.add_parser("disable", help="stop and remove one current-release opt-in service")
    p.add_argument("service")
    p = sub.add_parser("restart", help="rewrite and restart one enabled current-release opt-in service")
    p.add_argument("service")
    p = sub.add_parser("join", help="join Nakshatra with an nki1 invite or nka1 request address")
    p.add_argument("invite")
    p.add_argument("--name", default="", help="what your friend will see you as")
    a = ap.parse_args(argv)
    inst = Installer(a.prefix)
    try:
        if a.cmd == "write-units":
            w = Installer(a.prefix, systemd=not a.no_systemd, unit_dir=a.unit_dir)
            man = json.loads((a.prefix / "current" / "manifest.json").read_text())
            w._bootstrap_pre_transition_handoff(man)
            names = w.write_units(man)
            if a.include_updater:
                names.append("nak-update")
            print(json.dumps(names))
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
        elif a.cmd == "enable":
            print(inst.enable(a.service))
        elif a.cmd == "disable":
            print(inst.disable(a.service))
        elif a.cmd == "restart":
            print(inst.restart_service(a.service))
        else:
            print(json.dumps(inst.status(), indent=1))
        return 0
    except (InstallError, OSError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
