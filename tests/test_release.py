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


def test_rollback_restarts_only_units_returned_by_write_units(tmp_path, monkeypatch):
    prefix = tmp_path / "node"
    old = prefix / "releases" / "0.1.0"
    new = prefix / "releases" / "0.1.1"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    old_man = {"version": "0.1.0", "services": {"kept": {}, "opted-out": {}}}
    (old / "manifest.json").write_text(json.dumps(old_man))
    (prefix / "current").symlink_to("releases/0.1.1")
    (prefix / "config.json").write_text(json.dumps({"current": "0.1.1", "previous": "0.1.0"}))
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


def test_p2p_sidecar_service_is_opt_in_and_uses_the_release_binary(tmp_path):
    spec = json.loads((Path(__file__).resolve().parent.parent / "release" / "spec.json").read_text())
    svc = spec["services"]["nakshatra-p2p"]
    assert svc["requires"].endswith("/.nakshatra/p2p.env")
    assert svc["exec"][0] == "{prefix}/current/nakshatra/bin/nakshatra-sidecar"
    assert "-quic" in svc["exec"] and "-nakd-inbound=127.0.0.1:${NAK_DIRECT_PORT}" in svc["exec"]
    assert "-dial-listen=${XDG_RUNTIME_DIR}/nakshatra/p2p.sock" in svc["exec"] and "-relays=${P2P_RELAYS}" in svc["exec"]
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
    capture, fake_go = env["tmp"] / "go.env", env["tmp"] / "go"
    fake_go.write_text("#!/bin/sh\n"
                       "if [ \"$1\" = version ]; then echo 'go version go1.25.7 linux/amd64'; exit 0; fi\n"
                       f"/usr/bin/env | /usr/bin/sort > {capture}\n"
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


def test_a_profile_lock_is_shipped_verified_and_installed_only_where_opted_in(env, monkeypatch, tmp_path):
    monkeypatch.setitem(B.SPEC, "profiles", {"inference": {"requirements": ["numpy==1.26.4"],
                                                           "requires": str(tmp_path / "inference.env")}})
    _build(env, "0.1.0")
    man = json.loads((env["dist"] / "canary" / "0.1.0" / "manifest.json").read_text())
    assert man["profiles"]["inference"]["lock"]["file"] == "requirements-inference.lock"
    inst = env["inst"]
    assert inst.active_profiles(man) == []                       # not opted in → lean base env
    (tmp_path / "inference.env").write_text("")
    assert inst.active_profiles(man) == ["inference"]
    msg = inst.install(str(env["dist"]), "canary", env["pub"])
    assert "installed 0.1.0" in msg
    assert (env["tmp"] / "node" / "releases" / "0.1.0" / "requirements-inference.lock").exists()


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
