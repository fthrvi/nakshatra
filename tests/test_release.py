"""Tests for the signed release + installer (release/)."""
from __future__ import annotations

import io
import json
import time
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REL = Path(__file__).resolve().parent.parent / "release"
sys.path.insert(0, str(REL))

import releasekit as rk  # noqa: E402
import build as B  # noqa: E402
import install as I  # noqa: E402

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


def test_failed_health_check_rolls_back(env, monkeypatch):
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    _commit(env["a"], "alpha", "OK = False\n")          # the new release is broken
    _build(env, "0.1.1")
    with pytest.raises(I.InstallError, match="rolled back to 0.1.0"):
        env["inst"].update()
    assert os.readlink(env["tmp"] / "node" / "current") == "releases/0.1.0"


def test_manual_rollback(env):
    _build(env, "0.1.0")
    env["inst"].install(str(env["dist"]), "canary", env["pub"])
    _commit(env["a"], "alpha", "OK = True\nV = 3\n")
    _build(env, "0.1.1")
    env["inst"].update()
    assert "rolled back to 0.1.0" in env["inst"].rollback()
    assert os.readlink(env["tmp"] / "node" / "current") == "releases/0.1.0"


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
