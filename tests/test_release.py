"""Tests for the signed release + installer (release/)."""
from __future__ import annotations

import base64
import importlib.util
import fcntl
import io
import json
import time
import os
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest

REL = Path(__file__).resolve().parent.parent / "release"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(REL))
sys.path.insert(0, str(SCRIPTS))

import releasekit as rk  # noqa: E402
import build as B  # noqa: E402
import install as I  # noqa: E402
import release_trust  # noqa: E402

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402


def _key(tmp: Path):
    k = ed25519.Ed25519PrivateKey.generate()
    p = tmp / "release-TEST.key"
    p.write_text(k.private_bytes_raw().hex())
    return p, k.public_key().public_bytes_raw().hex()


def _repo(tmp: Path, name: str, module_body: str) -> Path:
    r = tmp / name
    (r / "pkg").mkdir(parents=True)
    (r / "pkg" / f"{name}_mod.py").write_text(module_body)
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"], check=True)
    return r


def _commit(r: Path, name: str, body: str):
    (r / "pkg" / f"{name}_mod.py").write_text(body)
    subprocess.run(["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "c2"], check=True)


@pytest.fixture
def env(tmp_path, monkeypatch):
    key, pub = _key(tmp_path)
    a = _repo(tmp_path, "alpha", "OK = True\n")
    b = _repo(tmp_path, "beta", "OK = True\n")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\n"
                       "if [ \"$1\" = \"--version\" ]; then echo 'uv 0.0.0'; exit 0; fi\n"
                       "while [ $# -gt 0 ]; do if [ \"$1\" = \"-o\" ]; then echo 'cryptography==46.0.6 --hash=sha256:00' > \"$2\"; fi; shift; done\n")
    fake_uv.chmod(0o755)
    monkeypatch.setattr(B, "SPEC", {
        "name": "nakshatra-node", "python": "3.12", "requirements": ["cryptography==46.0.6"],
        "components": [{"name": "alpha", "repo": str(a), "paths": ["pkg"], "pythonpath": "pkg"},
                       {"name": "beta", "repo": str(b), "paths": ["pkg"], "pythonpath": "pkg"}],
        "services": {},
        "health": [[sys.executable, "-c", "import alpha_mod, beta_mod; assert alpha_mod.OK and beta_mod.OK"]]})
    monkeypatch.setattr(I, "UNIT_DIR", tmp_path / "units")   # belt and braces; the installer below is told explicitly
    dist = tmp_path / "dist"
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    return {"key": key, "pub": pub, "a": a, "b": b, "uv": fake_uv, "dist": dist, "inst": inst, "tmp": tmp_path}


def _build(e, version, **kw):
    return B.build(version, "canary", {}, e["key"], e["dist"], e["uv"], **kw)


def _set_package_signer(e, key: str) -> None:
    policy = e["a"] / "pkg" / "package-signers.json"
    policy.write_text(json.dumps({
        "schema": "nak-package-signers/1",
        "keys": [key],
    }))
    subprocess.run(["git", "-C", str(e["a"]), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(e["a"]), "-c", "user.email=t@t", "-c", "user.name=t",
        "commit", "-qm", "package trust",
    ], check=True)
    B.SPEC["package_signers"] = {
        "component": "alpha", "path": "pkg/package-signers.json",
    }


def _load_c9f0fed_installer(tmp_path: Path):
    old = tmp_path / "c9f0fed-installer"
    old.mkdir()
    for name in ("install.py", "releasekit.py"):
        body = subprocess.check_output(
            ["git", "show", f"c9f0fed:release/{name}"], cwd=REL.parent)
        (old / name).write_bytes(body)
    spec = importlib.util.spec_from_file_location("install_c9f0fed", old / "install.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    original_path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


def _branch_release_repo(tmp_path: Path) -> Path:
    """Make a commit containing the release code under test for a real signed build."""
    repo = tmp_path / "branch-release-source"
    (repo / "release").mkdir(parents=True)
    (repo / "scripts").mkdir()
    for name in ("install.py", "releasekit.py", "package-signers.json"):
        shutil.copy2(REL / name, repo / "release" / name)
    shutil.copy2(SCRIPTS / "release_trust.py", repo / "scripts" / "release_trust.py")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
        "commit", "-qm", "branch release under test",
    ], check=True)
    return repo


def test_pure_python_verify_agrees_with_cryptography():
    k = ed25519.Ed25519PrivateKey.generate()
    pub = k.public_key().public_bytes_raw().hex()
    signed = rk.sign({"a": 1, "b": [2, 3]}, k.private_bytes_raw().hex())
    raw = __import__("base64").b64decode(signed["sig"]["value"])
    assert rk._ed25519_verify_pure(bytes.fromhex(pub), rk.canonical(signed), raw)
    assert not rk._ed25519_verify_pure(bytes.fromhex(pub), rk.canonical(dict(signed, a=2)), raw)
    assert rk.verify(signed, pub) and not rk.verify(dict(signed, a=2), pub)


def test_build_install_happy_path(env):
    _build(env, "0.1.0")
    msg = env["inst"].install(str(env["dist"]), "canary", env["pub"])
    assert "installed 0.1.0" in msg
    node = env["tmp"] / "node"
    assert os.readlink(node / "current") == "releases/0.1.0"
    assert (node / "bin" / "install.py").exists() and (env["tmp"] / "units" / "nak-update.timer").exists()
    assert json.loads((node / "config.json").read_text())["pubkey"] == env["pub"]


def test_first_install_health_can_load_release_package_signers(env, monkeypatch):
    package_key = "12" * 32
    _set_package_signer(env, package_key)
    _build(env, "0.1.0")
    inst = env["inst"]
    real_health = inst.health
    observed = []

    def health(man):
        transition = inst._transition_path
        observed.append({
            "signers": release_trust.load_release_package_signers(inst.prefix),
            "transition": json.loads(transition.read_text()),
            "mode": transition.stat().st_mode & 0o777,
            "config_exists": (inst.prefix / "config.json").exists(),
        })
        return real_health(man)

    monkeypatch.setattr(inst, "health", health)
    inst.install(str(env["dist"]), "canary", env["pub"])

    assert observed == [{
        "signers": frozenset({package_key}),
        "transition": {
            "schema": I.TRANSITION_SCHEMA,
            "pubkey": env["pub"],
            "from": None,
            "to": "0.1.0",
        },
        "mode": 0o600,
        "config_exists": False,
    }]
    assert not inst._transition_path.exists()


def test_build_pins_package_signers_from_exact_component_commit(env, monkeypatch):
    policy_path = env["a"] / "pkg" / "package-signers.json"
    trusted = "12" * 32
    packages = {
        "llama32-3b": {
            "layers": 28,
            "requirements": {
                "gpu_vram_bytes": 4 * 1024 ** 3,
                "cpu_ram_bytes": 8 * 1024 ** 3,
                "disk_free_bytes": 12 * 1024 ** 3,
            },
            "requirements_basis": "test fixture",
        },
    }
    policy_path.write_text(json.dumps({
        "schema": "nak-package-signers/1", "keys": [trusted], "packages": packages,
    }))
    subprocess.run(["git", "-C", str(env["a"]), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(env["a"]), "-c", "user.email=t@t", "-c", "user.name=t",
        "commit", "-qm", "package trust",
    ], check=True)
    monkeypatch.setitem(B.SPEC, "package_signers", {
        "component": "alpha", "path": "pkg/package-signers.json",
    })
    # Dirty working-tree policy changes must not enter a release of the recorded commit.
    policy_path.write_text(json.dumps({
        "schema": "nak-package-signers/1", "keys": ["34" * 32],
    }))

    out = _build(env, "0.1.0")
    manifest = json.loads((out / "manifest.json").read_text())

    assert rk.verify(manifest, env["pub"])
    assert manifest["package_signers"] == {
        "schema": "nak-package-signers/1", "keys": [trusted], "packages": packages,
    }


def test_tampered_artifact_refused_and_nothing_switched(env):
    out = _build(env, "0.1.0")
    tgz = next(out.glob("alpha-*.tar.gz"))
    tgz.write_bytes(tgz.read_bytes() + b"x")
    with pytest.raises(I.InstallError, match="does not match the signed manifest"):
        env["inst"].install(str(env["dist"]), "canary", env["pub"])
    assert not (env["tmp"] / "node" / "current").exists()


def test_wrong_or_missing_key_refused(env):
    _build(env, "0.1.0")
    (env["tmp"] / "o").mkdir()
    _, other = _key(env["tmp"] / "o")
    with pytest.raises(I.InstallError, match="not signed by the pinned release key"):
        env["inst"].install(str(env["dist"]), "canary", other)
    with pytest.raises(I.InstallError, match="needs --pubkey"):
        env["inst"].install(str(env["dist"]), "canary", "")


def test_pinned_key_cannot_be_swapped(env):
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    with pytest.raises(I.InstallError, match="different release key"):
        env["inst"].install(str(env["dist"]), "canary", "ab" * 32)


def test_update_installs_newer_only_and_records_previous(env):
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    assert "not newer" in env["inst"].update()
    _commit(env["a"], "alpha", "OK = True\nV = 2\n")
    _build(env, "0.1.1")
    assert "installed 0.1.1" in env["inst"].update()
    cfg = json.loads((env["tmp"] / "node" / "config.json").read_text())
    assert cfg["current"] == "0.1.1" and cfg["previous"] == "0.1.0"


def test_c9f0fed_updater_bootstraps_transition_and_timer_crossing(env, monkeypatch):
    package_key = json.loads((REL / "package-signers.json").read_text())["keys"][0]
    source = _branch_release_repo(env["tmp"])
    monkeypatch.setattr(B, "SPEC", {
        "name": "nakshatra-node",
        "python": "3.12",
        "installer": "nakshatra/release/install.py",
        "package_signers": {"component": "nakshatra", "path": "release/package-signers.json"},
        "components": [{
            "name": "nakshatra", "repo": str(source),
            "paths": ["release", "scripts"], "pythonpath": "scripts",
        }],
        "requirements": [],
        "services": {},
        "health": [],
    })
    _build(env, "0.15.13")

    prefix = env["tmp"] / "node"
    old_release = prefix / "releases" / "0.15.12"
    old_release.mkdir(parents=True)
    (old_release / "manifest.json").write_text(json.dumps({
        "version": "0.15.12", "components": [], "services": {}, "profiles": {},
    }))
    (prefix / "current").symlink_to("releases/0.15.12")
    I._save_config(prefix, {
        "source": str(env["dist"]), "channel": "canary", "pubkey": env["pub"],
        "current": "0.15.12", "previous": None,
    })
    # 0.15.12 and older created this directory through generic helpers.  Exercise
    # the real crossing with the permissive mode observed on deployed nodes.
    state = prefix / "state"
    state.mkdir(mode=0o775)
    state.chmod(0o775)

    old = _load_c9f0fed_installer(env["tmp"])
    updater = old.Installer(
        prefix, systemd=False, make_venv=False, unit_dir=env["tmp"] / "units")
    observed = []

    def observe(stage):
        observed.append((stage, release_trust.load_release_package_signers(prefix)))

    updater.health = lambda _man: observe("health")
    updater.restart = lambda _names, _man=None, force=False: observe("restart") or []

    assert "installed 0.15.13" in updater.update()
    assert observed == [
        ("health", frozenset({package_key})),
        ("restart", frozenset({package_key})),
    ]
    assert os.readlink(prefix / "current") == "releases/0.15.13"
    assert json.loads((prefix / "config.json").read_text())["current"] == "0.15.13"
    assert json.loads((prefix / "state" / I.TRANSITION_FILE).read_text())["from"] == "0.15.12"
    assert stat.S_IMODE(state.stat().st_mode) == 0o700

    # The crossing rewrites the timer's oneshot to the installer in the signed new release. Execute
    # that exact path as the next timer tick: it must recover the completed compatibility journal.
    timer_unit = (env["tmp"] / "units" / "nak-update.service").read_text()
    new_installer = prefix / "current" / "nakshatra" / "release" / "install.py"
    assert str(new_installer) in timer_unit and " update" in timer_unit
    timer_home = env["tmp"] / "timer-home"
    timer_home.mkdir()
    timer = subprocess.run(
        [sys.executable, str(new_installer), "--prefix", str(prefix), "update"],
        env=dict(os.environ, HOME=str(timer_home)), capture_output=True, text=True)
    assert timer.returncode == 0, timer.stderr
    assert "already at 0.15.13" in timer.stdout
    assert not (prefix / "state" / I.TRANSITION_FILE).exists()


def test_pre_transition_bootstrap_verifies_target_with_pinned_key(tmp_path):
    prefix = tmp_path / "node"
    release = prefix / "releases" / "0.15.13"
    release.mkdir(parents=True)
    (tmp_path / "pinned").mkdir()
    (tmp_path / "other").mkdir()
    _pinned_key, pinned_pub = _key(tmp_path / "pinned")
    other_key, _other_pub = _key(tmp_path / "other")
    manifest = rk.sign({
        "schema": rk.MANIFEST_SCHEMA, "version": "0.15.13",
    }, other_key.read_text().strip())
    (release / "manifest.json").write_text(json.dumps(manifest))
    (prefix / "current").symlink_to("releases/0.15.13")
    I._save_config(prefix, {"current": "0.15.12", "pubkey": pinned_pub})
    inst = I.Installer(prefix, systemd=False, make_venv=False, unit_dir=tmp_path / "units")

    with inst._install_lock():
        with pytest.raises(I.InstallError, match="not the exact target signed by the pinned"):
            inst._bootstrap_pre_transition_handoff(manifest)

    assert not inst._transition_path.exists()


def test_failed_health_check_rolls_back(env, monkeypatch):
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    _commit(env["a"], "alpha", "OK = False\n")          # the new release is broken
    _build(env, "0.1.1")
    with pytest.raises(I.InstallError, match="rolled back to 0.1.0"):
        env["inst"].update()
    assert os.readlink(env["tmp"] / "node" / "current") == "releases/0.1.0"
    assert not env["inst"]._transition_path.exists()


def test_manual_rollback(env):
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    _commit(env["a"], "alpha", "OK = True\nV = 3\n")
    _build(env, "0.1.1")
    env["inst"].update()
    assert "rolled back to 0.1.0" in env["inst"].rollback()
    assert os.readlink(env["tmp"] / "node" / "current") == "releases/0.1.0"


def test_installer_rollback_restart_can_load_release_package_signers(env, monkeypatch):
    old_package_key = "12" * 32
    new_package_key = "34" * 32
    _set_package_signer(env, old_package_key)
    _build(env, "0.1.0")
    inst = env["inst"]
    inst.install(str(env["dist"]), "canary", env["pub"])
    (env["a"] / "pkg" / "package-signers.json").write_text(json.dumps({
        "schema": "nak-package-signers/1", "keys": [new_package_key],
    }))
    _commit(env["a"], "alpha", "OK = True\nV = 3\n")
    _build(env, "0.1.1")
    inst.update()
    real_restart = inst.restart
    observed = []

    def restart(names, man=None, *, force=False):
        observed.append(release_trust.load_release_package_signers(inst.prefix))
        return real_restart(names, man, force=force)

    monkeypatch.setattr(inst, "restart", restart)
    assert "rolled back to 0.1.0" in inst.rollback()

    assert observed == [frozenset({old_package_key})]
    assert not inst._transition_path.exists()


def test_failed_installer_rollback_restores_committed_release(env, monkeypatch):
    _build(env, "0.1.0")
    inst = env["inst"]
    inst.install(str(env["dist"]), "canary", env["pub"])
    _commit(env["a"], "alpha", "OK = True\nV = 3\n")
    _build(env, "0.1.1")
    inst.update()
    real_restart = inst.restart
    calls = 0

    def fail_target_restart(names, man=None, *, force=False):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise I.InstallError("target restart failed")
        return real_restart(names, man, force=force)

    monkeypatch.setattr(inst, "restart", fail_target_restart)
    with pytest.raises(I.InstallError, match="rollback to 0.1.0 failed; restored 0.1.1"):
        inst.rollback()

    assert calls == 2
    assert os.readlink(inst.prefix / "current") == "releases/0.1.1"
    assert json.loads((inst.prefix / "config.json").read_text())["current"] == "0.1.1"
    assert not inst._transition_path.exists()


def test_next_operation_restores_a_release_left_mid_transition(env, monkeypatch):
    class SimulatedProcessDeath(BaseException):
        pass

    _build(env, "0.1.0")
    inst = env["inst"]
    inst.install(str(env["dist"]), "canary", env["pub"])
    _commit(env["a"], "alpha", "OK = True\nV = 4\n")
    _build(env, "0.1.1")

    def die_after_switch(_man):
        raise SimulatedProcessDeath

    monkeypatch.setattr(inst, "health", die_after_switch)
    with pytest.raises(SimulatedProcessDeath):
        inst.update()

    assert os.readlink(inst.prefix / "current") == "releases/0.1.1"
    assert json.loads((inst.prefix / "config.json").read_text())["current"] == "0.1.0"
    assert json.loads(inst._transition_path.read_text()) == {
        "schema": I.TRANSITION_SCHEMA,
        "pubkey": env["pub"],
        "from": "0.1.0",
        "to": "0.1.1",
    }
    with pytest.raises(release_trust.ReleaseTrustError, match="transition is stale"):
        release_trust.load_release_package_signers(inst.prefix)

    status = inst.status()
    assert os.readlink(inst.prefix / "current") == "releases/0.1.0"
    assert not inst._transition_path.exists()
    assert status["release_transition"] == {
        "state": "stale", "from": "0.1.0", "to": "0.1.1",
        "recovery": "restored-committed",
    }


def test_status_reports_transition_pending_while_installer_lock_is_live(env):
    _build(env, "0.1.0")
    inst = env["inst"]
    inst.install(str(env["dist"]), "canary", env["pub"])
    _commit(env["a"], "alpha", "OK = True\nV = 4\n")
    _build(env, "0.1.1")
    man, work = inst.fetch_verified(str(env["dist"]), "canary", env["pub"])
    try:
        inst.unpack(man, work)
    finally:
        shutil.rmtree(work)

    with inst._install_lock():
        inst._begin_transition(env["pub"], "0.1.0", "0.1.1")
        I._switch(inst.prefix, "0.1.1")
        assert inst.status()["release_transition"] == {
            "state": "pending", "from": "0.1.0", "to": "0.1.1",
        }

    recovered = inst.status()
    assert recovered["release_transition"]["state"] == "stale"
    assert recovered["release_transition"]["recovery"] == "restored-committed"
    assert os.readlink(inst.prefix / "current") == "releases/0.1.0"


def test_existing_state_directory_permissions_are_normalized(tmp_path):
    prefix = tmp_path / "node"
    state = prefix / "state"
    state.mkdir(parents=True, mode=0o775)
    state.chmod(0o775)

    I._save_private_state(prefix, "test.json", {"ok": True})

    assert state.stat().st_mode & 0o777 == 0o700
    assert (state / "test.json").stat().st_mode & 0o777 == 0o600


def test_rollback_restarts_only_units_returned_by_write_units(tmp_path, monkeypatch):
    prefix = tmp_path / "node"
    old = prefix / "releases" / "0.1.0"
    new = prefix / "releases" / "0.1.1"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    old_man = {"version": "0.1.0", "services": {"kept": {}, "opted-out": {}}}
    (old / "manifest.json").write_text(json.dumps(old_man))
    (prefix / "current").symlink_to("releases/0.1.1")
    (prefix / "config.json").write_text(json.dumps({
        "current": "0.1.1", "previous": "0.1.0", "pubkey": "12" * 32,
    }))
    inst = I.Installer(prefix, systemd=True, make_venv=False, unit_dir=tmp_path / "units")
    restarted = []
    monkeypatch.setattr(inst, "write_units", lambda man: ["kept"])
    monkeypatch.setattr(inst, "restart", lambda names, man, force=False: restarted.append((names, force)) or names)

    inst.rollback()

    assert restarted == [(["kept"], True)]


def test_archive_escaping_install_dir_refused(env):
    out = _build(env, "0.1.0")
    evil = io.BytesIO()
    with tarfile.open(fileobj=evil, mode="w:gz") as tf:
        data = b"pwned"
        ti = tarfile.TarInfo("../../escape.txt"); ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    tgz = next(out.glob("alpha-*.tar.gz"))
    tgz.write_bytes(evil.getvalue())
    man = json.loads((out / "manifest.json").read_text())
    for c in man["components"]:
        if c["name"] == "alpha":
            c["sha256"] = rk.sha256_file(tgz)
    signed = rk.sign({k: v for k, v in man.items() if k != "sig"}, env["key"].read_text().strip())
    (out / "manifest.json").write_text(json.dumps(signed))
    latest = json.loads((env["dist"] / "canary" / "latest.json").read_text())
    latest = rk.sign({k: v for k, v in latest.items() if k != "sig"}, env["key"].read_text().strip())
    (env["dist"] / "canary" / "latest.json").write_text(json.dumps(latest))
    with pytest.raises(I.InstallError, match="escapes the install dir"):
        env["inst"].install(str(env["dist"]), "canary", env["pub"])
    assert not (env["tmp"] / "escape.txt").exists()


def test_versions_are_immutable(env):
    _build(env, "0.1.0")
    with pytest.raises(SystemExit, match="immutable"):
        _build(env, "0.1.0")


def test_update_run_from_the_installed_copy_works(env, monkeypatch):
    # Regression (found on blackwell 2026-10-02): the update service runs the installer from the
    # node's own copy; copying it "onto itself" crashed after `current` had already switched.
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    monkeypatch.setattr(I, "HERE", env["tmp"] / "node" / "bin")
    _commit(env["a"], "alpha", "OK = True\nV = 4\n")
    _build(env, "0.1.1")
    assert "installed 0.1.1" in env["inst"].update()


def test_any_failure_after_switch_rolls_back(env, monkeypatch):
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    _commit(env["a"], "alpha", "OK = True\nV = 5\n")
    _build(env, "0.1.1")
    real = I.Installer.write_units
    calls = {"n": 0}

    def flaky(self, man):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full while writing units")
        return real(self, man)

    monkeypatch.setattr(I.Installer, "write_units", flaky)
    with pytest.raises(I.InstallError, match="rolled back to 0.1.0"):
        env["inst"].update()
    assert os.readlink(env["tmp"] / "node" / "current") == "releases/0.1.0"


def test_update_service_runs_the_installer_shipped_in_the_release(env, monkeypatch):
    (env["a"] / "release").mkdir()
    real = Path(__file__).resolve().parent.parent / "release"
    for f in ("install.py", "releasekit.py"):        # ship the REAL installer: it writes the units now
        (env["a"] / "release" / f).write_text((real / f).read_text())
    subprocess.run(["git", "-C", str(env["a"]), "add", "."], check=True)
    subprocess.run(["git", "-C", str(env["a"]), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "r"], check=True)
    spec = dict(B.SPEC, installer="alpha/release/install.py")
    spec["components"] = [dict(spec["components"][0], paths=["pkg", "release"]), spec["components"][1]]
    monkeypatch.setattr(B, "SPEC", spec)
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    unit = (env["tmp"] / "units" / "nak-update.service").read_text()
    assert "current/alpha/release/install.py" in unit and "update" in unit


def test_command_wrappers_follow_current_release(env, monkeypatch):
    monkeypatch.setitem(B.SPEC, "commands", {"nak": "alpha_mod"})
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    w = env["tmp"] / "node" / "bin" / "nak"
    body = w.read_text()
    assert os.access(w, os.X_OK) and "current/venv/bin/python -m alpha_mod" in body and "releases/" not in body
    # systemd=False (tests, dev) never touches the real ~/.local/bin
    assert not (Path.home() / ".local" / "bin" / "nak").resolve().is_relative_to(env["tmp"])


def test_failed_first_install_leaves_nothing_half_installed(env, monkeypatch):
    _build(env, "0.1.0")
    inst = env["inst"]
    # NEVER monkeypatch.undo() here: it also undoes the fixture's patches (that is how this test once
    # rewrote the hub's real systemd units). Scope the failure with a context instead.
    with monkeypatch.context() as m:
        m.setattr(I.Installer, "health", lambda self, man: (_ for _ in ()).throw(I.InstallError("boom")))
        with pytest.raises(I.InstallError, match="first install of 0.1.0 failed and was removed"):
            inst.install(str(env["dist"]), "canary", env["pub"])
    node = env["tmp"] / "node"
    assert not (node / "current").exists() and not (node / "releases" / "0.1.0").exists()
    assert not inst._transition_path.exists()
    assert "installed 0.1.0" in inst.install(str(env["dist"]), "canary", env["pub"])     # retry works


def test_a_stale_current_link_without_a_recorded_install_is_retried(env):
    _build(env, "0.1.0")
    node = env["tmp"] / "node"
    (node / "releases" / "0.1.0").mkdir(parents=True)
    (node / "current").symlink_to("releases/0.1.0")      # the half state older installers could leave
    assert "installed 0.1.0" in env["inst"].install(str(env["dist"]), "canary", env["pub"])


def test_expired_latest_pointer_is_refused(env):
    _build(env, "0.1.0", now=int(time.time()) - 61 * 86400)      # signed 61 days ago, TTL 60 days
    with pytest.raises(I.InstallError, match="expired latest.json"):
        env["inst"].install(str(env["dist"]), "canary", env["pub"])


def test_join_refuses_a_release_older_than_the_invite_names(env):
    _build(env, "0.1.0")
    with pytest.raises(I.InstallError, match="older than 0.2.0"):
        env["inst"].install(str(env["dist"]), "canary", env["pub"], min_version="0.2.0")


def test_a_non_systemd_installer_can_never_write_real_units(tmp_path):
    real = Path.home() / ".config" / "systemd" / "user"
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False)
    assert inst.unit_dir == tmp_path / "node" / "units" and inst.unit_dir != real


def test_hardened_opt_in_service_unit_renders_like_the_handwritten_one(tmp_path):
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    svc = {"description": "meshd", "requires": "{home}/.nakshatra/meshd.env", "env_file": "{home}/.nakshatra/meshd.env",
           "after": ["network-online.target"], "environment": {"RENDEZVOUS": "45.63.109.137:51820"},
           "exec": ["{python}", "{prefix}/current/nakshatra/scripts/mesh/meshd.py", "--mesh-id=${MESH_ID}"],
           "restart": "always", "hardening": True}
    man = {"components": [{"name": "nakshatra", "pythonpath": "scripts"}]}
    text = inst._unit_text(svc, inst._ctx(man), man)
    home = str(Path.home())
    for line in (f"ConditionPathExists={home}/.nakshatra/meshd.env", f"EnvironmentFile=-{home}/.nakshatra/meshd.env",
                 "Environment=RENDEZVOUS=45.63.109.137:51820", "--mesh-id=${MESH_ID}", "Restart=always",
                 "ProtectSystem=strict", f"ReadWritePaths={home}/.nakshatra", "After=network-online.target"):
        assert line in text, line


def test_meshd_and_relay_ship_in_the_release_but_only_run_where_opted_in():
    spec = json.loads((Path(__file__).resolve().parent.parent / "release" / "spec.json").read_text())
    for name in ("nakshatra-meshd", "nakshatra-relay"):
        svc = spec["services"][name]
        assert svc["requires"].endswith(".env") and svc["hardening"] is True and svc["restart"] == "always"


def test_registrar_is_a_hardened_opt_in_release_service(tmp_path):
    spec = json.loads((Path(__file__).resolve().parent.parent / "release" / "spec.json").read_text())
    svc = spec["services"]["nakshatra-registrar"]
    assert svc["requires"].endswith("/.nakshatra/registrar.env")
    assert svc["env_file"] == svc["requires"]
    assert svc["after"] == ["network-online.target"]
    assert svc["restart"] == "always" and svc["hardening"] is True
    assert svc["components"] == ["nakshatra"]
    assert "profiles" not in svc and "drain" not in svc
    assert svc["exec"][-1] == "{prefix}/current/nakshatra/scripts/registrar.py"

    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    man = {"components": [{"name": "nakshatra", "pythonpath": "scripts"}]}
    text = inst._unit_text(svc, inst._ctx(man), man)
    home = str(Path.home())
    for line in (f"ConditionPathExists={home}/.nakshatra/registrar.env",
                 f"EnvironmentFile=-{home}/.nakshatra/registrar.env",
                 "After=network-online.target", "Restart=always", "ProtectSystem=strict",
                 f"ReadWritePaths={home}/.nakshatra"):
        assert line in text


def test_registrar_fingerprint_tracks_only_the_nakshatra_component(tmp_path):
    spec = json.loads((Path(__file__).resolve().parent.parent / "release" / "spec.json").read_text())
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False, unit_dir=tmp_path / "units")

    def manifest(nak, sth):
        return {"version": "1", "lock": {"file": "requirements.lock"}, "profiles": {},
                "components": [{"name": "nakshatra", "commit": nak, "pythonpath": "scripts"},
                               {"name": "sthambha", "commit": sth, "pythonpath": "."}],
                "services": {"nakshatra-registrar": spec["services"]["nakshatra-registrar"]}}

    baseline = inst._fingerprint("nakshatra-registrar", manifest("n1", "s1"))
    assert inst._fingerprint("nakshatra-registrar", manifest("n1", "s2")) == baseline
    assert inst._fingerprint("nakshatra-registrar", manifest("n2", "s1")) != baseline


def test_registrar_enable_disable_is_isolated_from_other_services(tmp_path, monkeypatch):
    spec = json.loads((Path(__file__).resolve().parent.parent / "release" / "spec.json").read_text())
    prefix, units = tmp_path / "node", tmp_path / "units"
    release = prefix / "releases" / "1.2.3"
    release.mkdir(parents=True)
    units.mkdir()
    opt_in = tmp_path / "home" / ".nakshatra" / "registrar.env"
    svc = dict(spec["services"]["nakshatra-registrar"], requires=str(opt_in), env_file=str(opt_in))
    man = {"version": "1.2.3", "components": [{"name": "nakshatra", "pythonpath": "scripts"}],
           "services": {"nakshatra-registrar": svc}}
    (release / "manifest.json").write_text(json.dumps(man))
    (prefix / "current").symlink_to("releases/1.2.3")
    opt_in.parent.mkdir(parents=True)
    opt_in.write_text("NKS_NODE_ID=test\n")
    (units / "operator.service").write_text("operator owned\n")
    inst = I.Installer(prefix, systemd=True, make_venv=False, unit_dir=units)
    calls = []

    def systemctl(args, **kwargs):
        calls.append(args)
        stdout = "ActiveState=active\nMainPID=42\nNRestarts=0\n" if "show" in args else ""
        return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(I.subprocess, "run", systemctl)
    monkeypatch.setattr(I, "SERVICE_STABILITY_S", 0)
    assert inst.enable("nakshatra-registrar") == "enabled nakshatra-registrar.service"
    assert I.UNIT_MARKER in (units / "nakshatra-registrar.service").read_text()
    assert (units / "operator.service").read_text() == "operator owned\n"
    assert all("operator" not in " ".join(call) for call in calls)

    calls.clear()
    assert inst.disable("nakshatra-registrar") == "disabled nakshatra-registrar.service"
    assert not opt_in.exists() and not (units / "nakshatra-registrar.service").exists()
    assert (units / "operator.service").read_text() == "operator owned\n"
    assert calls == [
        ["systemctl", "--user", "disable", "--now", "nakshatra-registrar.service"],
        ["systemctl", "--user", "daemon-reload"],
    ]


def test_p2p_sidecar_service_is_opt_in_and_uses_the_release_binary(tmp_path):
    spec = json.loads((Path(__file__).resolve().parent.parent / "release" / "spec.json").read_text())
    svc = spec["services"]["nakshatra-p2p"]
    assert svc["requires"].endswith("/.nakshatra/p2p.env")
    assert svc["exec"][0] == "{prefix}/current/nakshatra/bin/nakshatra-sidecar"
    assert "-quic" in svc["exec"]
    assert svc["environment"]["NAKSHATRA_P2P_NAKD_INBOUND"] == \
        "-nakd-inbound=127.0.0.1:51830"
    assert "$NAKSHATRA_P2P_NAKD_INBOUND" in svc["exec"]
    assert "$NAKSHATRA_P2P_CONSUMER_ARGS" in svc["exec"]
    assert "$NAKSHATRA_P2P_GATEWAY_ARGS" not in svc["exec"]
    assert not any(arg.startswith("-inbound") for arg in svc["exec"])
    assert "-dial-listen=${XDG_RUNTIME_DIR}/nakshatra/p2p.sock" in svc["exec"] and "-relays=${P2P_RELAYS}" in svc["exec"]
    assert svc["optional_env_files"] == [
        "{home}/.nakshatra/worker.env",
        "{home}/.nakshatra/p2p-consumer.env",
    ]
    assert svc["exec"][-1] == "$NAKSHATRA_P2P_WORKER_ARGS"
    pre = svc["exec_start_pre"][0]
    assert "scripts/sidecar_key.py" in " ".join(pre) and "--if-missing" in pre
    assert spec["binaries"] == [{"name": "nakshatra-sidecar", "file": "nakshatra-sidecar",
                                 "install_path": "bin/nakshatra-sidecar",
                                 "component": "nakshatra", "source": "third_party/shard-libp2p-sidecar"}]
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False)
    text = inst._unit_text(svc, inst._ctx({"components": [{"name": "nakshatra", "pythonpath": "scripts"}]}),
                           {"components": [{"name": "nakshatra", "pythonpath": "scripts"}]})
    assert "ExecStartPre=" in text and "sidecar_key.py" in text
    assert f"ReadWritePaths={Path.home()}/.config" in text
    assert f"EnvironmentFile=-{Path.home()}/.nakshatra/worker.env" in text
    worker_env = f"EnvironmentFile=-{Path.home()}/.nakshatra/worker.env"
    consumer_env = f"EnvironmentFile=-{Path.home()}/.nakshatra/p2p-consumer.env"
    assert consumer_env in text
    assert text.index(worker_env) < text.index(consumer_env)
    assert "Environment=NAKSHATRA_P2P_NAKD_INBOUND=-nakd-inbound=127.0.0.1:51830" in text
    assert " $NAKSHATRA_P2P_NAKD_INBOUND " in text
    assert " $NAKSHATRA_P2P_CONSUMER_ARGS " in text
    assert "NAKSHATRA_P2P_GATEWAY_ARGS" not in text
    assert " $NAKSHATRA_P2P_WORKER_ARGS" in text


def test_worker_service_is_hardened_profiled_and_release_owned(tmp_path):
    spec = json.loads((Path(__file__).resolve().parent.parent / "release" / "spec.json").read_text())
    svc = spec["services"]["nakshatra-worker"]
    assert svc["requires"].endswith("/.nakshatra/worker.env")
    assert svc["env_file"] == svc["requires"] and svc["profile"] == "worker"
    assert svc["optional_env_files"] == ["{home}/.nakshatra/owner-lease.env"]
    worker_profile = spec["profiles"]["worker"]
    assert worker_profile["requires"] == svc["requires"]
    assert worker_profile["satisfied_by"] == ["inference"]
    assert worker_profile["requirements"] == [
        "grpcio==1.81.1", "protobuf==6.33.6", "numpy==1.26.4", "gguf==0.19.0"
    ]
    assert "llama_cpp_python==0.3.28" not in worker_profile["requirements"]
    assert svc["hardening"] is True
    assert svc["read_write_paths"] == ["{home}/.nakshatra/worker", "{home}/.nakshatra/tls"]
    assert svc["environment"]["NAKSHATRA_AUDIT_PATH"] == \
        "{home}/.nakshatra/worker/audit.jsonl"
    command = " ".join(svc["exec"])
    assert "current/nakshatra/scripts/worker_service.py" in command
    assert "current/nakshatra/scripts/worker.py" in command
    assert "--build-json=" in command and "--engine-base=" in command and "--node-key=" in command
    assert "--p2p-opt-in=" in command
    assert "--package-url" not in command and "--pillar-url" not in command
    assert "${PACKAGE_URL}" not in command and "${PILLAR_URL}" not in command
    assert "--skip-sha256" not in command

    inst = I.Installer(Path("/nonexistent-prefix"), systemd=False, make_venv=False,
                       unit_dir=tmp_path / "units")
    man = {"version": "0.1.0", "components": [{"name": "nakshatra", "pythonpath": "scripts"}],
           "services": {"nakshatra-worker": svc}, "profiles": {}}
    text = inst._unit_text(svc, inst._ctx(man), man)
    assert "ProtectSystem=strict" in text
    assert f"ReadWritePaths={Path.home()}/.nakshatra/worker" in text
    assert f"ReadWritePaths={Path.home()}/.nakshatra/tls" in text
    assert f"EnvironmentFile=-{Path.home()}/.nakshatra/owner-lease.env" in text
    assert f"Environment=NAKSHATRA_AUDIT_PATH={Path.home()}/.nakshatra/worker/audit.jsonl" in text


def test_enabling_a_profiled_service_installs_its_signed_lock(tmp_path, monkeypatch):
    prefix = tmp_path / "node"
    current = prefix / "releases" / "1.2.3"
    (current / "venv" / "bin").mkdir(parents=True)
    for path in (current / "uv", current / "venv" / "bin" / "python"):
        path.write_text("")
    (current / "requirements-inference.lock").write_text("numpy==1.26.4 --hash=sha256:00\n")
    (prefix / "current").symlink_to("releases/1.2.3")
    marker = tmp_path / "inference.env"
    marker.write_text("")
    man = {"profiles": {"inference": {"requires": str(marker),
                                        "lock": {"file": "requirements-inference.lock"}}}}
    calls = []
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kw: calls.append((args, kw)) or
                        type("R", (), {"returncode": 0})())
    inst = I.Installer(prefix, systemd=False, make_venv=True, unit_dir=tmp_path / "units")
    inst._ensure_service_profile({"profile": "inference"}, man)
    args, kwargs = calls[0]
    assert args[:3] == [str(prefix / "current" / "uv"), "pip", "install"]
    assert "--require-hashes" in args and str(prefix / "current" / "requirements-inference.lock") in args
    assert kwargs["check"] is True


def _profile_unpack_fixture(tmp_path, monkeypatch, *, inference: bool):
    prefix, work = tmp_path / "node", tmp_path / "work"
    work.mkdir()
    worker_marker = tmp_path / "worker.env"
    inference_marker = tmp_path / "inference.env"
    worker_marker.write_text("")
    if inference:
        inference_marker.write_text("")
    for name in ("requirements.lock", "requirements-worker.lock", "requirements-inference.lock", "uv"):
        (work / name).write_text(name + "\n")
    man = {
        "version": "1.2.3",
        "python": "3.12",
        "components": [],
        "lock": {"file": "requirements.lock"},
        "profiles": {
            "worker": {
                "requires": str(worker_marker),
                "lock": {"file": "requirements-worker.lock"},
                "satisfied_by": ["inference"],
            },
            "inference": {
                "requires": str(inference_marker),
                "lock": {"file": "requirements-inference.lock"},
                "satisfied_by": [],
            },
        },
    }
    calls = []
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kw: calls.append((args, kw)) or
                        type("R", (), {"returncode": 0})())
    inst = I.Installer(prefix, systemd=False, make_venv=True, unit_dir=tmp_path / "units")
    rel = inst.unpack(man, work)
    return inst, man, rel, calls


def test_worker_only_node_installs_worker_lock(tmp_path, monkeypatch):
    _inst, _man, _rel, calls = _profile_unpack_fixture(tmp_path, monkeypatch, inference=False)
    pip = next(args for args, _kwargs in calls if "pip" in args)
    assert str(tmp_path / "node" / "releases" / "1.2.3" / "requirements-worker.lock") in pip
    assert not any("requirements-inference.lock" in arg for arg in pip)


def test_inference_superset_installs_once_and_satisfies_worker_service(tmp_path, monkeypatch):
    inst, man, rel, calls = _profile_unpack_fixture(tmp_path, monkeypatch, inference=True)
    pip_calls = [args for args, _kwargs in calls if "pip" in args]
    assert len(pip_calls) == 1
    assert str(rel / "requirements-inference.lock") in pip_calls[0]

    (rel / "venv" / "bin").mkdir(parents=True)
    (rel / "venv" / "bin" / "python").write_text("")
    (inst.prefix / "current").symlink_to("releases/1.2.3")
    inst._ensure_service_profile({"profile": "worker"}, man)
    assert str(inst.prefix / "current" / "requirements-inference.lock") in calls[-1][0]


def test_active_superset_satisfies_service_without_subset_marker(tmp_path, monkeypatch):
    inst, man, rel, calls = _profile_unpack_fixture(tmp_path, monkeypatch, inference=True)
    Path(man["profiles"]["worker"]["requires"]).unlink()
    (rel / "venv" / "bin").mkdir(parents=True)
    (rel / "venv" / "bin" / "python").write_text("")
    (inst.prefix / "current").symlink_to("releases/1.2.3")

    inst._ensure_service_profile({"profile": "worker"}, man)

    assert str(inst.prefix / "current" / "requirements-inference.lock") in calls[-1][0]


def test_incompatible_active_profiles_are_still_refused(tmp_path, monkeypatch):
    prefix, work = tmp_path / "node", tmp_path / "work"
    work.mkdir()
    profiles = {}
    for name in ("worker", "analytics"):
        marker = tmp_path / f"{name}.env"
        marker.write_text("")
        lock = f"requirements-{name}.lock"
        (work / lock).write_text(lock + "\n")
        profiles[name] = {"requires": str(marker), "lock": {"file": lock}}
    (work / "requirements.lock").write_text("base\n")
    (work / "uv").write_text("uv\n")
    man = {"version": "1.2.3", "python": "3.12", "components": [],
           "lock": {"file": "requirements.lock"}, "profiles": profiles}
    inst = I.Installer(prefix, systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    with pytest.raises(I.InstallError, match="incompatible release profiles.*analytics, worker"):
        inst.unpack(man, work)


def test_go_sidecar_is_hash_covered_verified_and_installed(env, monkeypatch):
    source = env["a"] / "sidecar"
    source.mkdir()
    (source / "go.mod").write_text("module test/sidecar\n\ngo 1.22\n")
    (source / "main.go").write_text("package main\nfunc main() {}\n")
    subprocess.run(["git", "-C", str(env["a"]), "add", "."], check=True)
    subprocess.run(["git", "-C", str(env["a"]), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "sidecar"], check=True)
    monkeypatch.setitem(B.SPEC, "binaries", [{"name": "nakshatra-sidecar", "file": "nakshatra-sidecar",
                                               "component": "alpha", "source": "sidecar"}])
    fake_go = env["tmp"] / "go"
    fake_go.write_text("#!/bin/sh\nif [ \"$1\" = version ]; then echo 'go version go1.25.7 linux/amd64'; exit 0; fi\n"
                       "while [ $# -gt 0 ]; do if [ \"$1\" = \"-o\" ]; then "
                       "printf sidecar-binary > \"$2\"; exit 0; fi; shift; done\nexit 2\n")
    fake_go.chmod(0o755)
    out = _build(env, "0.1.0", go_bin=fake_go)
    man = json.loads((out / "manifest.json").read_text())
    artifact = man["artifacts"][0]
    assert man["toolchains"]["go"]["version"] == "go1.25.7"
    assert artifact["path"] == "bin/nakshatra-sidecar"
    with tarfile.open(next(out.glob("alpha-*.tar.gz"))) as tf:
        assert "bin/nakshatra-sidecar" in tf.getnames()  # current/older installers already verify this archive
    assert not (out / "nakshatra-sidecar").exists()     # no new top-level file an older updater would skip
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    installed = env["tmp"] / "node" / "current" / "alpha" / "bin" / "nakshatra-sidecar"
    assert artifact["sha256"] == rk.sha256_file(installed)
    assert installed.read_bytes() == b"sidecar-binary" and os.access(installed, os.X_OK)


def test_tampered_go_sidecar_is_refused(env, monkeypatch):
    source = env["a"] / "sidecar"
    source.mkdir()
    (source / "main.go").write_text("package main\nfunc main() {}\n")
    subprocess.run(["git", "-C", str(env["a"]), "add", "."], check=True)
    subprocess.run(["git", "-C", str(env["a"]), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "sidecar"], check=True)
    monkeypatch.setitem(B.SPEC, "binaries", [{"name": "nakshatra-sidecar", "file": "nakshatra-sidecar",
                                               "component": "alpha", "source": "sidecar"}])
    fake_go = env["tmp"] / "go"
    fake_go.write_text("#!/bin/sh\nif [ \"$1\" = version ]; then echo 'go version go1.25.7 linux/amd64'; exit 0; fi\n"
                       "while [ $# -gt 0 ]; do if [ \"$1\" = \"-o\" ]; then "
                       "printf sidecar > \"$2\"; exit 0; fi; shift; done\nexit 2\n")
    fake_go.chmod(0o755)
    out = _build(env, "0.1.0", go_bin=fake_go)
    archive = next(out.glob("alpha-*.tar.gz"))
    unpacked = env["tmp"] / "tampered-component"
    with tarfile.open(archive) as tf:
        tf.extractall(unpacked, filter="data")
    (unpacked / "bin" / "nakshatra-sidecar").write_bytes(b"tampered")
    with tarfile.open(archive, "w:gz") as tf:
        for path in sorted(unpacked.rglob("*")):
            tf.add(path, arcname=path.relative_to(unpacked))
    man = json.loads((out / "manifest.json").read_text())
    for component in man["components"]:
        if component["name"] == "alpha":
            component["sha256"] = rk.sha256_file(archive)  # outer archive is valid; inner hash is not
    man = rk.sign({k: v for k, v in man.items() if k != "sig"}, env["key"].read_text().strip())
    (out / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(I.InstallError, match="installed artifact nakshatra-sidecar does not match"):
        env["inst"].install(str(env["dist"]), "canary", env["pub"])


def test_go_build_uses_only_pinned_hermetic_environment(env, monkeypatch):
    source = env["a"] / "sidecar"
    source.mkdir()
    (source / "go.mod").write_text("module test/sidecar\n\ngo 1.25\n")
    (source / "main.go").write_text("package main\nfunc main() {}\n")
    subprocess.run(["git", "-C", str(env["a"]), "add", "."], check=True)
    subprocess.run(["git", "-C", str(env["a"]), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "sidecar"], check=True)
    commit = subprocess.check_output(["git", "-C", str(env["a"]), "rev-parse", "HEAD"], text=True).strip()
    capture, capture_args, fake_go = env["tmp"] / "go.env", env["tmp"] / "go.args", env["tmp"] / "go"
    fake_go.write_text("#!/bin/sh\n"
                       "if [ \"$1\" = version ]; then echo 'go version go1.25.7 linux/amd64'; exit 0; fi\n"
                       f"/usr/bin/env | /usr/bin/sort > {capture}\n"
                       f"printf '%s\\n' \"$@\" > {capture_args}\n"
                       "while [ $# -gt 0 ]; do if [ \"$1\" = -o ]; then printf binary > \"$2\"; exit 0; fi; shift; done\n"
                       "exit 2\n")
    fake_go.chmod(0o755)
    for key, value in {"GOFLAGS": "-modfile=evil.mod", "GOENV": "/tmp/evil", "GOAMD64": "v4",
                       "GOTOOLCHAIN": "auto", "SECRET_BUILD_INPUT": "leak"}.items():
        monkeypatch.setenv(key, value)
    out = env["tmp"] / "sidecar-bin"
    assert B._build_go_binary(env["a"], commit, "sidecar", out, fake_go) == "go1.25.7"
    got = dict(line.split("=", 1) for line in capture.read_text().splitlines() if "=" in line)
    assert got["GOENV"] == "off" and got["GOFLAGS"] == "-mod=readonly"
    assert got["GOAMD64"] == "v1" and got["CGO_ENABLED"] == "0" and got["GOTOOLCHAIN"] == "local"
    assert got["GOOS"] == "linux" and got["GOARCH"] == "amd64"
    assert "nak-go-build-" in got["GOCACHE"] and "nak-go-build-" in got["GOMODCACHE"]
    assert "nak-go-build-" in got["GOPATH"] and "SECRET_BUILD_INPUT" not in got
    args = capture_args.read_text().splitlines()
    assert args[0] == "build" and not any(arg == "-tags" or arg.startswith("-tags=") for arg in args)


def test_go_build_refuses_unpinned_toolchain(env):
    source = env["a"] / "sidecar"
    source.mkdir()
    (source / "main.go").write_text("package main\nfunc main() {}\n")
    subprocess.run(["git", "-C", str(env["a"]), "add", "."], check=True)
    subprocess.run(["git", "-C", str(env["a"]), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "sidecar"], check=True)
    commit = subprocess.check_output(["git", "-C", str(env["a"]), "rev-parse", "HEAD"], text=True).strip()
    fake_go = env["tmp"] / "wrong-go"
    fake_go.write_text("#!/bin/sh\necho 'go version go1.25.6 linux/amd64'\n")
    fake_go.chmod(0o755)
    with pytest.raises(RuntimeError, match="require go1.25.7"):
        B._build_go_binary(env["a"], commit, "sidecar", env["tmp"] / "out", fake_go)


def test_units_are_written_by_the_installer_shipped_in_the_new_release(tmp_path):
    prefix = tmp_path / "node"
    (prefix / "current" / "pkg").mkdir(parents=True)
    (prefix / "current" / "manifest.json").write_text(json.dumps({"services": {}, "installer": "pkg/install.py"}))
    marker = tmp_path / "called"
    (prefix / "current" / "pkg" / "install.py").write_text(
        "import json,sys\n"
        f"open({str(marker)!r},'w').write(' '.join(sys.argv[1:]))\n"
        "print(json.dumps(['from-the-new-installer']))\n")
    inst = I.Installer(prefix, systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    man = json.loads((prefix / "current" / "manifest.json").read_text())
    assert inst._write_units_by_new_installer(man) == ["from-the-new-installer"]
    called = marker.read_text()
    assert "write-units" in called and "--no-systemd" in called and str(tmp_path / "units") in called


def test_profile_locks_are_shipped_hashed_and_include_subsumption(env, monkeypatch, tmp_path):
    monkeypatch.setitem(B.SPEC, "profiles", {
        "worker": {"requirements": ["numpy==1.26.4"],
                   "requires": str(tmp_path / "worker.env"), "satisfied_by": ["inference"]},
        "inference": {"requirements": ["numpy==1.26.4"],
                      "requires": str(tmp_path / "inference.env")},
    })
    _build(env, "0.1.0")
    release = env["dist"] / "canary" / "0.1.0"
    man = json.loads((release / "manifest.json").read_text())
    assert man["profiles"]["worker"]["lock"]["file"] == "requirements-worker.lock"
    assert man["profiles"]["worker"]["lock"]["sha256"] == \
        rk.sha256_file(release / "requirements-worker.lock")
    assert man["profiles"]["worker"]["satisfied_by"] == ["inference"]
    assert man["profiles"]["inference"]["lock"]["file"] == "requirements-inference.lock"
    inst = env["inst"]
    assert inst.active_profiles(man) == []                       # not opted in → lean base env
    (tmp_path / "inference.env").write_text("")
    assert inst.active_profiles(man) == ["inference"]
    msg = inst.install(str(env["dist"]), "canary", env["pub"])
    assert "installed 0.1.0" in msg
    assert (env["tmp"] / "node" / "releases" / "0.1.0" / "requirements-inference.lock").exists()


def test_build_refuses_false_profile_superset_claim(env, monkeypatch, tmp_path):
    monkeypatch.setitem(B.SPEC, "profiles", {
        "worker": {"requirements": ["numpy==1.26.4"], "requires": str(tmp_path / "worker.env"),
                   "satisfied_by": ["inference"]},
        "inference": {"requirements": ["numpy==2.0.0"], "requires": str(tmp_path / "inference.env")},
    })
    with pytest.raises(RuntimeError, match="numpy.*1.26.4.*2.0.0"):
        _build(env, "0.1.0")
    assert not (env["dist"] / "canary" / "0.1.0").exists()


def test_units_are_written_only_for_services_the_node_opted_into(tmp_path):
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    (tmp_path / "node").mkdir()
    opted = tmp_path / "meshd.env"
    man = {"version": "0.1.0", "components": [{"name": "nakshatra", "pythonpath": "scripts"}], "services": {
        "svc-on": {"description": "x", "requires": str(opted), "exec": ["{python}"]},
        "svc-off": {"description": "y", "requires": str(tmp_path / "absent.env"), "exec": ["{python}"]},
        "svc-always": {"description": "z", "exec": ["{python}"]}}}
    opted.write_text("")
    (tmp_path / "units").mkdir()
    (tmp_path / "units" / "svc-off.service").write_text("HAND-WRITTEN, LIVE")
    names = inst.write_units(man)
    assert "svc-on" in names and "svc-always" in names and "svc-off" not in names
    assert (tmp_path / "units" / "svc-off.service").read_text() == "HAND-WRITTEN, LIVE"   # untouched


def test_write_units_refuses_eligible_operator_owned_unit(tmp_path):
    units = tmp_path / "units"
    units.mkdir()
    opted = tmp_path / "identity"
    opted.write_text("ready\n")
    operator_unit = units / "svc.service"
    operator_unit.write_text("[Service]\nExecStart=/operator/owned\n")
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False, unit_dir=units)
    man = {
        "version": "0.1.0", "components": [],
        "services": {"svc": {"description": "managed", "requires": str(opted),
                              "exec": ["{python}"]}},
    }

    with pytest.raises(I.InstallError, match="refusing to replace operator-owned unit"):
        inst.write_units(man)

    assert operator_unit.read_text() == "[Service]\nExecStart=/operator/owned\n"


def test_removed_opt_in_stops_disables_and_removes_only_managed_unit(tmp_path, monkeypatch):
    units = tmp_path / "units"
    units.mkdir()
    (tmp_path / "node").mkdir()
    inst = I.Installer(tmp_path / "node", systemd=True, make_venv=False, unit_dir=units)
    man = {"version": "0.1.0", "components": [{"name": "nakshatra", "pythonpath": "scripts"}],
           "services": {
               "managed": {"description": "managed", "requires": str(tmp_path / "gone.env"),
                           "exec": ["{python}"]},
               "operator": {"description": "operator", "requires": str(tmp_path / "also-gone.env"),
                            "exec": ["{python}"]}}}
    (units / "managed.service").write_text(inst._unit_text(man["services"]["managed"], inst._ctx(man), man))
    (units / "operator.service").write_text("[Service]\nExecStart=/operator/owned\n")
    calls = []
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kw: calls.append(args) or
                        type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    inst.write_units(man)
    assert not (units / "managed.service").exists()
    assert (units / "operator.service").read_text() == "[Service]\nExecStart=/operator/owned\n"
    assert ["systemctl", "--user", "disable", "--now", "managed.service"] in calls
    assert ["systemctl", "--user", "disable", "--now", "operator.service"] not in calls


def test_removed_opt_in_retains_managed_unit_when_disable_fails(tmp_path, monkeypatch):
    units = tmp_path / "units"
    units.mkdir()
    (tmp_path / "node").mkdir()
    inst = I.Installer(tmp_path / "node", systemd=True, make_venv=False, unit_dir=units)
    svc = {"description": "managed", "requires": str(tmp_path / "gone.env"), "exec": ["{python}"]}
    man = {"version": "0.1.0", "components": [], "services": {"managed": svc}}
    unit = units / "managed.service"
    unit.write_text(inst._unit_text(svc, inst._ctx(man), man))
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kw:
                        type("R", (), {"returncode": 1 if "disable" in args else 0,
                                       "stdout": "", "stderr": "bus error"})())

    inst.write_units(man)

    assert unit.exists()
    issue = inst.status()["service_issues"]["managed"]
    assert issue["status"] == "removal-failed"


def test_the_gateway_service_needs_its_own_explicit_opt_in():
    spec = json.loads((Path(__file__).resolve().parent.parent / "release" / "spec.json").read_text())
    gw = spec["services"]["nakshatra-unconscious"]
    assert gw["requires"].endswith("unconscious.release")      # NOT unconscious.env: no accidental cutover
    assert "--bind" in gw["exec"] and "${BIND}" in gw["exec"]


def test_the_running_installer_recognises_the_new_one_even_after_current_switched(tmp_path, monkeypatch):
    # Updates run via <prefix>/current/... ; after the switch `current` points at the NEW release. The
    # running (old) installer must still see the new installer as a DIFFERENT file and hand over to it.
    prefix = tmp_path / "node"
    old_rel, new_rel = prefix / "releases" / "0.9.0" / "pkg", prefix / "releases" / "0.10.0" / "pkg"
    for d in (old_rel, new_rel):
        d.mkdir(parents=True)
    (old_rel / "install.py").write_text("# old\n")
    marker = tmp_path / "called"
    (new_rel / "install.py").write_text(
        "import json,sys\n" f"open({str(marker)!r},'w').write('new')\n" "print(json.dumps(['by-new']))\n")
    (prefix / "current").symlink_to("releases/0.10.0")                 # already switched
    monkeypatch.setattr(I, "_THIS_FILE", (old_rel / "install.py").resolve())   # we were loaded from 0.9.0
    inst = I.Installer(prefix, systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    assert inst._write_units_by_new_installer({"installer": "pkg/install.py"}) == ["by-new"]
    assert marker.read_text() == "new"


def _svc_man(commits, lock="L1"):
    return {"version": "0.1.0", "lock": {"sha256": lock}, "profiles": {},
            "components": [{"name": "nakshatra", "commit": commits[0], "pythonpath": "scripts"},
                           {"name": "sthambha", "commit": commits[1], "pythonpath": "."}],
            "services": {"sig": {"description": "s", "exec": ["{python}"], "components": ["sthambha"]},
                         "gw": {"description": "g", "exec": ["{python}"], "components": ["nakshatra"],
                                "drain": {"port": 11599, "max_wait_s": 30}}}}


def test_restart_only_services_whose_inputs_changed_and_drain_busy_ones(tmp_path, monkeypatch):
    inst = I.Installer(tmp_path / "node", systemd=True, make_venv=False, unit_dir=tmp_path / "units")
    calls, waited = [], []
    def systemctl(args, **kw):
        calls.append(args)
        stdout = "ActiveState=active\nMainPID=42\nNRestarts=0\n" if "show" in args else ""
        return type("R", (), {"returncode": 0, "stdout": stdout})()
    monkeypatch.setattr(I.subprocess, "run", systemctl)
    monkeypatch.setattr(I, "SERVICE_STABILITY_S", 0)
    monkeypatch.setattr(I.Installer, "_wait_idle", lambda self, port, mx: waited.append(port) or True)
    restarted = lambda: [a[-1] for a in calls if a[:3] == ["systemctl", "--user", "restart"]]
    assert sorted(inst.restart(["sig", "gw"], _svc_man(["n1", "s1"]))) == ["gw", "sig"]   # first time: both
    assert waited == [11599]                                                              # gw drained first
    calls.clear(); waited.clear()
    assert inst.restart(["sig", "gw"], _svc_man(["n2", "s1"])) == ["gw"]    # nakshatra-only release: signer left alone
    calls.clear()
    assert inst.restart(["sig", "gw"], _svc_man(["n2", "s1"])) == []        # nothing changed: nothing restarted
    assert inst.restart(["sig", "gw"], _svc_man(["n2", "s1"]), force=True) == ["sig", "gw"]   # rollback: all


def test_update_restarts_signer_then_waits_for_data_key_before_net(tmp_path, monkeypatch):
    inst = I.Installer(tmp_path / "node", systemd=True, make_venv=False, unit_dir=tmp_path / "units")
    calls = []

    def systemctl(args, **kwargs):
        calls.append(args)
        stdout = "ActiveState=active\nMainPID=42\nNRestarts=0\n" if "show" in args else ""
        return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(I.subprocess, "run", systemctl)
    monkeypatch.setattr(I, "SERVICE_STABILITY_S", 0)
    probes = []
    monkeypatch.setattr(I.Installer, "_wait_signer_ready",
                        lambda self, timeout: probes.append(timeout) or True)
    man = {"version": "1.0.0", "components": [], "profiles": {}, "services": {
        "nak-net": {"description": "net", "exec": ["net"]},
        "nak-signer": {"description": "signer", "exec": ["signer"]},
    }}

    assert inst.restart(["nak-net", "nak-signer"], man) == ["nak-signer", "nak-net"]
    restarts = [call[-1] for call in calls if call[:3] == ["systemctl", "--user", "restart"]]
    assert restarts == ["nak-signer.service", "nak-net.service"]
    assert probes == [I.SIGNER_READY_TIMEOUT_S]


def test_signer_readiness_probe_requires_data_key_answer(tmp_path, monkeypatch):
    path = tmp_path / "signer.sock"
    ready = threading.Event()

    def signer():
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)
        ready.set()
        conn, _ = server.accept()
        with conn:
            assert conn.makefile("rb").readline() == b'{"op":"data_key"}\n'
            encoded = base64.b64encode(bytes(range(32))).decode()
            conn.sendall(json.dumps({"ok": True, "data_key": encoded}).encode() + b"\n")
        server.close()

    thread = threading.Thread(target=signer, daemon=True)
    thread.start()
    assert ready.wait(1)
    monkeypatch.setenv("NAK_SIGNER_SOCK", str(path))
    assert I.Installer._signer_answers()
    thread.join(timeout=1)
    assert not thread.is_alive()


@pytest.mark.parametrize("bad_value", ["test-value", base64.b64encode(b"short").decode()])
def test_bad_signer_data_key_keeps_nak_net_running_old_code(tmp_path, monkeypatch, bad_value):
    path = tmp_path / "signer.sock"
    ready = threading.Event()

    def signer():
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)
        ready.set()
        conn, _ = server.accept()
        with conn:
            assert conn.makefile("rb").readline() == b'{"op":"data_key"}\n'
            conn.sendall(json.dumps({"ok": True, "data_key": bad_value}).encode() + b"\n")
        server.close()

    thread = threading.Thread(target=signer, daemon=True)
    thread.start()
    assert ready.wait(1)
    monkeypatch.setenv("NAK_SIGNER_SOCK", str(path))
    monkeypatch.setattr(I, "SIGNER_READY_TIMEOUT_S", 0)
    calls = []
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kwargs: calls.append(args) or
                        type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    inst = I.Installer(tmp_path / "node", systemd=True, make_venv=False, unit_dir=tmp_path / "units")
    man = {"version": "1", "components": [], "profiles": {},
           "services": {"nak-net": {"description": "net", "exec": ["net"]}}}

    assert inst.restart(["nak-net"], man) == []
    assert not any(call[:3] == ["systemctl", "--user", "restart"] for call in calls)
    issue = json.loads(inst._service_state_path.read_text())["services"]["nak-net"]
    assert issue["status"] == "pending" and "data_key" in issue["detail"]
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_net_restart_is_deferred_when_new_signer_protocol_does_not_answer(tmp_path, monkeypatch):
    inst = I.Installer(tmp_path / "node", systemd=True, make_venv=False, unit_dir=tmp_path / "units")
    calls = []
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kwargs: calls.append(args) or
                        type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(I.Installer, "_wait_signer_ready", lambda self, timeout: False)
    assert inst.restart(["nak-net"], {"version": "1", "components": [], "profiles": {},
                                      "services": {"nak-net": {"description": "net", "exec": ["net"]}}}) == []
    assert not any(call[:3] == ["systemctl", "--user", "restart"] for call in calls)
    assert json.loads(inst._service_state_path.read_text())["services"]["nak-net"]["status"] == "pending"


def test_install_hardens_release_url_to_0600(env, monkeypatch):
    net = env["tmp"] / "net"
    net.mkdir()
    release_url = net / "release-url"
    release_url.write_text("https://releases.example\n")
    release_url.chmod(0o644)
    monkeypatch.setenv("NAK_NET_DIR", str(net))
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    assert stat.S_IMODE(release_url.stat().st_mode) == 0o600


def test_unstable_restarted_service_is_reported_without_release_rollback(env, monkeypatch):
    monkeypatch.setitem(B.SPEC, "services", {
        "svc": {"description": "test service", "exec": ["{python}"], "components": ["alpha"]}})
    _build(env, "0.1.0")
    _commit(env["a"], "alpha", "OK = True\nV = 9\n")
    _build(env, "0.1.1")
    inst = env["inst"]
    inst.systemd = True
    monkeypatch.setattr(I, "SERVICE_ACTIVE_TIMEOUT_S", 0)
    monkeypatch.setattr(I, "SERVICE_STABILITY_S", 0)
    calls = []

    def systemctl(args, **kwargs):
        calls.append(args)
        rc = 0
        stdout = ""
        if "show" in args:
            if os.readlink(env["tmp"] / "node" / "current").endswith("0.1.0"):
                stdout = "ActiveState=active\nMainPID=41\nNRestarts=0\n"
            else:
                rc = 3
        return type("R", (), {"returncode": rc, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(I.subprocess, "run", systemctl)
    inst.install(str(env["dist"]), "canary", env["pub"], version="0.1.0")
    old_fp = json.loads((env["tmp"] / "node" / "state" / "fingerprints.json").read_text())["svc"]
    msg = inst.update()
    assert "failed: svc" in msg
    assert os.readlink(env["tmp"] / "node" / "current") == "releases/0.1.1"
    assert inst.status()["service_issues"]["svc"]["status"] == "failed"
    assert json.loads((env["tmp"] / "node" / "state" / "fingerprints.json").read_text())["svc"] == old_fp
    restarts = [a[-1] for a in calls if a[:3] == ["systemctl", "--user", "restart"]]
    assert restarts.count("svc.service") == 2       # initial install + failed update; no rollback restart


def test_drain_timeout_defers_restart_and_fingerprint(tmp_path, monkeypatch):
    inst = I.Installer(tmp_path / "node", systemd=True, make_venv=False, unit_dir=tmp_path / "units")
    calls = []
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kw: calls.append(args) or
                        type("R", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.setattr(I.Installer, "_wait_idle", lambda self, port, mx: False)

    assert inst.restart(["gw"], _svc_man(["n1", "s1"])) == []

    assert not any(a[:3] == ["systemctl", "--user", "restart"] for a in calls)
    assert inst.status()["service_issues"]["gw"]["status"] == "pending"
    assert "gw" not in json.loads((tmp_path / "node" / "state" / "fingerprints.json").read_text())


def test_service_health_requires_stable_pid_and_restart_count(tmp_path, monkeypatch):
    inst = I.Installer(tmp_path / "node", systemd=True, make_venv=False, unit_dir=tmp_path / "units")
    snapshots = iter([
        "ActiveState=active\nMainPID=41\nNRestarts=0\n",
        "ActiveState=active\nMainPID=42\nNRestarts=1\n",
    ])
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kw:
                        type("R", (), {"returncode": 0, "stdout": next(snapshots)})())

    healthy, detail = inst._wait_stable("svc", 0.1, 0)

    assert not healthy and "MainPID 41->42" in detail and "NRestarts 0->1" in detail


def test_the_gateway_drains_before_restart_in_the_shipped_spec():
    spec = json.loads((Path(__file__).resolve().parent.parent / "release" / "spec.json").read_text())
    assert spec["services"]["nakshatra-unconscious"]["drain"]["port"] == 11599
    assert spec["services"]["nak-signer"]["components"] == ["sthambha"]


def test_identical_releases_fingerprint_the_same_despite_lock_comments(tmp_path):
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    for v, path in (("0.1.0", "/dist/0.1.0"), ("0.1.1", "/dist/0.1.1")):
        rel = tmp_path / "node" / "releases" / v
        rel.mkdir(parents=True)
        (rel / "requirements.lock").write_text(f"# uv pip compile {path}/requirements.in\ncffi==2.1.1 \\\n"
                                               f"    --hash=sha256:aa\n    # via -r {path}/requirements.in\n")
    man = lambda v: dict(_svc_man(["n1", "s1"]), version=v, lock={"file": "requirements.lock", "sha256": v})
    assert inst._fingerprint("gw", man("0.1.0")) == inst._fingerprint("gw", man("0.1.1"))


def test_drain_fails_closed_when_ss_fails(tmp_path, monkeypatch):
    inst = I.Installer(tmp_path / "node", systemd=True, make_venv=False, unit_dir=tmp_path / "units")
    monkeypatch.setattr(I.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 1, "stdout": ""})())
    monkeypatch.setattr(I.time, "sleep", lambda s: None)
    assert inst._wait_idle(11599, 0.05) is False


def test_prune_reads_current_previous_and_keeps_newest_from_config(tmp_path):
    import os as _os
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    rel = tmp_path / "node" / "releases"
    for i, v in enumerate(["0.1.0", "0.2.0", "0.3.0", "0.4.0", "0.5.0", "0.6.0"]):
        (rel / v).mkdir(parents=True)
        _os.utime(rel / v, (1000 + i, 1000 + i))
    (tmp_path / "node" / "current").symlink_to("releases/0.6.0")
    I._save_config(tmp_path / "node", {"current": "0.6.0", "previous": "0.1.0"})
    removed = inst.prune()
    assert sorted(removed) == ["0.2.0", "0.3.0"]
    assert sorted(p.name for p in rel.iterdir()) == ["0.1.0", "0.4.0", "0.5.0", "0.6.0"]


def test_prune_never_deletes_a_release_a_live_process_runs_from(tmp_path, monkeypatch):
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    rel = tmp_path / "node" / "releases"
    import os as _os
    for i, v in enumerate(["0.1.0", "0.2.0", "0.3.0", "0.4.0", "0.5.0"]):
        (rel / v).mkdir(parents=True)
        _os.utime(rel / v, (1000 + i, 1000 + i))
    monkeypatch.setattr(I, "_releases_in_use", lambda r: {"0.1.0"})     # e.g. the unrestarted signer
    inst.KEEP_RELEASES = 1
    inst.prune(keep=("0.5.0",))
    assert sorted(p.name for p in rel.iterdir()) == ["0.1.0", "0.5.0"]


def test_releases_in_use_finds_this_process(tmp_path):
    rel = tmp_path / "releases"
    (rel / "9.9.9").mkdir(parents=True)
    lib = rel / "9.9.9" / "x.bin"
    lib.write_bytes(b"\0" * 4096)
    import mmap
    with open(lib, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            assert "9.9.9" in I._releases_in_use(rel)
        finally:
            m.close()


def test_releases_in_use_resolves_pure_python_argv_through_current(tmp_path):
    import subprocess as _sp
    rel = tmp_path / "node" / "releases"
    app = rel / "9.9.9" / "app"
    app.mkdir(parents=True)
    script = app / "stay.py"
    script.write_text("import time\ntime.sleep(30)\n")
    (tmp_path / "node" / "current").symlink_to("releases/9.9.9")
    proc = _sp.Popen([sys.executable, str(tmp_path / "node" / "current" / "app" / "stay.py")])
    try:
        for _ in range(100):
            if proc.poll() is not None or "9.9.9" in I._releases_in_use(rel):
                break
            time.sleep(0.01)
        assert proc.poll() is None
        assert "9.9.9" in I._releases_in_use(rel)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_prune_restores_release_that_becomes_live_during_quarantine(tmp_path, monkeypatch):
    inst = I.Installer(tmp_path / "node", systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    inst.KEEP_RELEASES = 1
    rel = tmp_path / "node" / "releases"
    for stamp, v in enumerate(("0.1.0", "0.2.0"), start=1):
        (rel / v).mkdir(parents=True)
        os.utime(rel / v, (stamp, stamp))

    def live_after_rename(root):
        quarantined = [p.name for p in root.iterdir() if p.name.startswith(".prune-0.1.0-")]
        return set(quarantined)

    monkeypatch.setattr(I, "_releases_in_use", live_after_rename)
    assert inst.prune() == []
    assert (rel / "0.1.0").is_dir()
    assert not any(p.name.startswith(".prune-") for p in rel.iterdir())


def test_installer_lock_times_out_a_second_caller(tmp_path, monkeypatch):
    first = I.Installer(tmp_path / "node", systemd=False, make_venv=False)
    second = I.Installer(tmp_path / "node", systemd=False, make_venv=False)
    monkeypatch.setattr(I, "INSTALL_LOCK_TIMEOUT_S", 0)
    with first._install_lock():
        with pytest.raises(I.InstallError, match="another installer operation"):
            second.prune()


def test_installer_exclusive_lock_waits_for_shared_probe_without_deadlock(
        tmp_path, monkeypatch):
    prefix = tmp_path / "node"
    prefix.mkdir()
    installer = I.Installer(prefix, systemd=False, make_venv=False)
    lock_path = prefix / ".install.lock"
    attempted = threading.Event()
    acquired = threading.Event()
    failures = []
    real_flock = I.fcntl.flock

    with lock_path.open("a+") as reader:
        real_flock(reader, fcntl.LOCK_SH)
        installer_thread = None

        def observe_attempt(fd, operation):
            if (threading.current_thread() is installer_thread
                    and operation == (fcntl.LOCK_EX | fcntl.LOCK_NB)):
                attempted.set()
            return real_flock(fd, operation)

        monkeypatch.setattr(I.fcntl, "flock", observe_attempt)

        def acquire_installer_lock():
            try:
                with installer._install_lock():
                    acquired.set()
            except BaseException as exc:
                failures.append(exc)

        installer_thread = threading.Thread(
            target=acquire_installer_lock, name="installer-lock-waiter")
        installer_thread.start()
        assert attempted.wait(1.0), "installer never attempted its exclusive lock"
        assert not acquired.is_set()
        real_flock(reader, fcntl.LOCK_UN)
        installer_thread.join(timeout=1.0)

    assert not installer_thread.is_alive()
    assert acquired.is_set()
    assert failures == []


def _one_service_install(tmp_path, *, managed=True):
    prefix, units = tmp_path / "node", tmp_path / "units"
    release = prefix / "releases" / "1.2.3"
    release.mkdir(parents=True)
    opt_in = tmp_path / "home" / ".nakshatra" / "p2p.env"
    svc = {"description": "p2p", "requires": str(opt_in), "exec": ["{prefix}/current/bin/p2p"]}
    man = {"version": "1.2.3", "components": [], "services": {
        "nakshatra-p2p": svc,
        "unrelated": {"description": "other", "requires": str(tmp_path / "other.env"),
                      "exec": ["/do/not/touch"]},
    }}
    (release / "manifest.json").write_text(json.dumps(man))
    (prefix / "current").symlink_to("releases/1.2.3")
    units.mkdir()
    inst = I.Installer(prefix, systemd=True, make_venv=False, unit_dir=units)
    if managed:
        (units / "nakshatra-p2p.service").write_text(inst._unit_text(svc, inst._ctx(man), man))
    return inst, man, opt_in, units


def test_enable_writes_starts_and_checks_only_requested_opt_in_service(tmp_path, monkeypatch):
    inst, _man, opt_in, units = _one_service_install(tmp_path, managed=False)
    opt_in.parent.mkdir(parents=True)
    opt_in.write_text("P2P_RELAYS=test\n")
    (units / "unrelated.service").write_text("operator owned\n")
    calls = []

    def systemctl(args, **kwargs):
        calls.append(args)
        stdout = "ActiveState=active\nMainPID=42\nNRestarts=0\n" if "show" in args else ""
        return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(I.subprocess, "run", systemctl)
    monkeypatch.setattr(I, "SERVICE_STABILITY_S", 0)
    assert inst.enable("nakshatra-p2p") == "enabled nakshatra-p2p.service"
    assert I.UNIT_MARKER in (units / "nakshatra-p2p.service").read_text()
    assert (units / "unrelated.service").read_text() == "operator owned\n"
    assert calls[0] == ["systemctl", "--user", "daemon-reload"]
    assert calls[1] == ["systemctl", "--user", "enable", "--now", "nakshatra-p2p.service"]
    assert all("unrelated" not in " ".join(c) for c in calls)


def test_enable_refuses_missing_opt_in_and_daemon_reload_failure(tmp_path, monkeypatch):
    inst, _man, opt_in, units = _one_service_install(tmp_path, managed=False)
    with pytest.raises(I.InstallError, match="not opted in"):
        inst.enable("nakshatra-p2p")
    opt_in.parent.mkdir(parents=True)
    opt_in.write_text("")
    calls = []
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kw: calls.append(args) or
                        type("R", (), {"returncode": 1, "stdout": "", "stderr": ""})())
    with pytest.raises(I.InstallError, match="daemon-reload failed"):
        inst.enable("nakshatra-p2p")
    assert calls == [["systemctl", "--user", "daemon-reload"]]
    assert (units / "nakshatra-p2p.service").exists()


def test_restart_service_rewrites_and_restarts_only_enabled_managed_unit(tmp_path, monkeypatch):
    inst, _man, opt_in, units = _one_service_install(tmp_path)
    opt_in.parent.mkdir(parents=True)
    opt_in.write_text("P2P_RELAYS=test\n")
    (units / "unrelated.service").write_text("operator owned\n")
    calls = []

    def systemctl(args, **kwargs):
        calls.append(args)
        stdout = "ActiveState=active\nMainPID=42\nNRestarts=0\n" if "show" in args else ""
        return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(I.subprocess, "run", systemctl)
    monkeypatch.setattr(I, "SERVICE_STABILITY_S", 0)
    assert inst.restart_service("nakshatra-p2p") == "restarted nakshatra-p2p.service"
    assert calls[:3] == [
        ["systemctl", "--user", "is-enabled", "--quiet", "nakshatra-p2p.service"],
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "restart", "nakshatra-p2p.service"],
    ]
    assert all("unrelated" not in " ".join(call) for call in calls)
    assert (units / "unrelated.service").read_text() == "operator owned\n"


def test_disable_stops_removes_only_managed_unit_and_opt_in(tmp_path, monkeypatch):
    inst, _man, opt_in, units = _one_service_install(tmp_path)
    opt_in.parent.mkdir(parents=True)
    opt_in.write_text("")
    (units / "unrelated.service").write_text("operator owned\n")
    calls = []
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kw: calls.append(args) or
                        type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    assert inst.disable("nakshatra-p2p") == "disabled nakshatra-p2p.service"
    assert not opt_in.exists()
    assert not (units / "nakshatra-p2p.service").exists()
    assert (units / "unrelated.service").read_text() == "operator owned\n"
    assert calls == [
        ["systemctl", "--user", "disable", "--now", "nakshatra-p2p.service"],
        ["systemctl", "--user", "daemon-reload"],
    ]


def test_disable_failure_retains_managed_unit_and_opt_in_and_cli_exits_nonzero(
        tmp_path, monkeypatch, capsys):
    inst, _man, opt_in, units = _one_service_install(tmp_path)
    opt_in.parent.mkdir(parents=True)
    opt_in.write_text("")
    monkeypatch.setattr(I.subprocess, "run", lambda args, **kw:
                        type("R", (), {"returncode": 1, "stdout": "", "stderr": ""})())
    with pytest.raises(I.InstallError, match="could not fully disable"):
        inst.disable("nakshatra-p2p")
    assert opt_in.exists()
    assert (units / "nakshatra-p2p.service").exists()

    monkeypatch.setattr(I, "UNIT_DIR", units)
    assert I.main(["--prefix", str(inst.prefix), "disable", "nakshatra-p2p"]) == 1
    assert "systemctl disable --now failed" in capsys.readouterr().err
    assert opt_in.exists()


def test_p2p_unit_gets_a_writable_private_runtime_dir():
    spec = json.loads((Path(__file__).resolve().parent.parent / "release" / "spec.json").read_text())
    svc = spec["services"]["nakshatra-p2p"]
    assert svc["runtime_directory"] == "nakshatra" and svc.get("hardening")
    inst = I.Installer(Path("/nonexistent-prefix"), systemd=False, make_venv=False, unit_dir=Path("/nonexistent-units"))
    man = {"version": "0.1.0", "components": [], "services": {"nakshatra-p2p": svc}, "profiles": {}}
    text = inst._unit_text(svc, inst._ctx(man), man)
    assert "RuntimeDirectory=nakshatra" in text and "RuntimeDirectoryMode=0700" in text
    assert "ProtectHome=read-only" in text
