"""U4 B1: a worker built and run entirely from the installed signed release."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from network import nak
import worker
import worker_service


TREE = "6d5b567d598666d20281db6548bbefd582d26f05"


def _fake_release(prefix: Path) -> Path:
    current = prefix / "current"
    installer = current / "nakshatra" / "release" / "install.py"
    installer.parent.mkdir(parents=True)
    installer.write_text("# signed test installer\n")
    (current / "nakshatra" / "engine").mkdir(parents=True)
    (current / "nakshatra" / "engine" / "BASE").write_text(f"COMMIT={'a' * 40}\nTREE={TREE}\n")
    provision = current / "nakshatra" / "deploy" / "provision-worker.sh"
    provision.parent.mkdir(parents=True)
    provision.write_text("#!/bin/sh\n")
    key = ed25519.Ed25519PrivateKey.generate()
    pub = key.public_key().public_bytes_raw().hex()
    man = {"version": "1.2.3", "installer": "nakshatra/release/install.py",
           "bootstrap": {"install.py": hashlib.sha256(installer.read_bytes()).hexdigest()}}
    canonical = json.dumps(man, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    man["sig"] = {"alg": "Ed25519", "keyid": pub[:16],
                  "value": base64.b64encode(key.sign(canonical)).decode()}
    (current / "manifest.json").write_text(json.dumps(man))
    prefix.mkdir(exist_ok=True)
    (prefix / "config.json").write_text(json.dumps({"current": "1.2.3", "pubkey": pub}))
    return provision


def test_nak_worker_build_records_release_tree_backend_and_daemon(tmp_path, monkeypatch):
    home, prefix = tmp_path / "home", tmp_path / "node"
    provision = _fake_release(prefix)
    daemon = home / ".nakshatra" / "worker" / "engine" / "llama" / "build" / "bin" / "llama-nakshatra-worker"
    daemon.parent.mkdir(parents=True)
    daemon.write_text("binary")
    daemon.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    calls = []

    class FakePopen:
        def __init__(self, args, **kwargs):
            calls.append((args, kwargs))
            self.stdout = iter([f"NAK_BUILD_DAEMON={daemon}\n", "NAK_BUILD_BACKEND=cuda\n",
                                "NAK_BUILD_N_GPU_LAYERS=99\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    assert nak.main(["worker", "build"]) == 0
    args, kwargs = calls[0]
    assert args == ["bash", str(provision)]
    assert kwargs["env"]["WORKER_BUILD_ONLY"] == "1"
    assert kwargs["env"]["WORKER_DIR"] == str(home / ".nakshatra" / "worker" / "engine")
    record_path = home / ".nakshatra" / "worker" / "build.json"
    record = json.loads(record_path.read_text())
    assert record["daemon_bin"] == str(daemon.resolve())
    assert record["backend"] == "cuda" and record["n_gpu_layers"] == 99
    assert record["engine_tree"] == TREE and record["release_version"] == "1.2.3"
    assert record_path.stat().st_mode & 0o777 == 0o600


def test_nak_worker_on_writes_env_once_and_uses_verified_installer(tmp_path, monkeypatch):
    home, prefix = tmp_path / "home", tmp_path / "node"
    _fake_release(prefix)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return type("R", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(subprocess, "run", run)
    env_path = home / ".nakshatra" / "worker.env"
    with pytest.raises(SystemExit, match="--pillar-url"):
        nak.main(["worker", "on", "--model-id", "qwen",
                  "--package-url", "https://models.test/qwen", "--package-pubkey", "cd" * 32,
                  "--layers", "0-16", "--port", "5530"])
    assert not env_path.exists()

    argv = ["worker", "on", "--model-id", "qwen", "--package-url", "https://models.test/qwen", "--package-pubkey", "cd" * 32,
            "--pillar-url", "https://pillar.test", "--pillar-spki", "ab" * 32,
            "--listen-address", "0.0.0.0",
            "--layers", "0-16", "--port", "5530"]
    assert nak.main(argv) == 0
    first = env_path.read_text()
    assert 'MODEL_ID="qwen"' in first and 'PACKAGE_URL="https://models.test/qwen"' in first
    assert 'PILLAR_URL="https://pillar.test"' in first
    assert f'PILLAR_SPKI_SHA256="{"ab" * 32}"' in first
    assert 'TRUSTED_PILLAR_NETS="10.42.0.0/24"' in first
    assert 'LISTEN="0.0.0.0"' in first
    assert "MODE=first" in first and "LAYER_START=0" in first and "LAYER_END=16" in first
    assert "SUB_GGUF=" in first and "skip-sha256" not in first
    assert (home / ".nakshatra" / "inference.env").exists()
    assert (home / ".nakshatra" / "tls").is_dir()
    installer = prefix / "current" / "nakshatra" / "release" / "install.py"
    assert calls[-1] == [sys.executable, str(installer), "--prefix", str(prefix),
                         "enable", "nakshatra-worker"]

    # Existing configuration is authoritative: a bare `on` is enough and does not rewrite it.
    assert nak.main(["worker", "on"]) == 0
    assert env_path.read_text() == first


def test_worker_launcher_refuses_stale_tree_and_execs_worker_without_skip(tmp_path, monkeypatch, capsys):
    root = tmp_path / "worker"
    daemon = root / "engine" / "llama" / "build" / "bin" / "llama-nakshatra-worker"
    daemon.parent.mkdir(parents=True)
    daemon.write_text("binary")
    daemon.chmod(0o755)
    build = root / "build.json"
    build.write_text(json.dumps({"daemon_bin": str(daemon), "backend": "rocm", "n_gpu_layers": 99,
                                 "engine_tree": TREE}))
    base = tmp_path / "BASE"
    base.write_text(f"TREE={TREE}\n")
    worker = tmp_path / "worker.py"
    worker.write_text("# release worker\n")
    private = ed25519.Ed25519PrivateKey.generate()
    key = tmp_path / "worker.ed25519"
    key.write_bytes(private.private_bytes_raw())
    key.chmod(0o600)
    sub = root / "slice.gguf"
    argv = ["--build-json", str(build), "--engine-base", str(base), "--worker-root", str(root),
            "--worker-script", str(worker), "--node-key", str(key), "--port", "5530",
            "--sub-gguf", str(sub), "--mode", "first",
            "--layer-start", "0", "--layer-end", "16", "--model-id", "qwen"]
    monkeypatch.setenv("PILLAR_URL", "https://pillar.test:8443")
    monkeypatch.setenv("PILLAR_SPKI_SHA256", "ab" * 32)
    monkeypatch.setenv("PACKAGE_URL", "https://models.test/qwen?token=package-secret")
    monkeypatch.setenv("PACKAGE_TRUSTED_PUBKEY", "cd" * 32)
    monkeypatch.setenv("NAKSHATRA_TLS_REQUIRED", "false")
    monkeypatch.setenv("NAKSHATRA_AUTH_REQUIRED", "false")
    executed = []
    assert worker_service.main(argv, execve=lambda path, command, env: executed.append((path, command, env))) == 0
    path, command, env = executed[0]
    pub = private.public_key().public_bytes_raw().hex()
    assert path == sys.executable and command[:2] == [sys.executable, str(worker)]
    assert command[command.index("--daemon-bin") + 1] == str(daemon.resolve())
    assert command[command.index("--gpu-backend") + 1] == "rocm"
    assert command[command.index("--node-id") + 1] == pub and env["NAKSHATRA_NODE_ID"] == pub
    assert "--skip-sha256" not in command
    assert command[command.index("--package-trusted-pubkey") + 1] == "cd" * 32
    assert "package-secret" not in " ".join(command)
    assert env["NAKSHATRA_PILLAR_URL"] == "https://pillar.test:8443"
    assert env["STHAMBHA_PILLAR_SPKI_SHA256"] == "ab" * 32
    assert env["NAKSHATRA_PACKAGE_URL"].endswith("package-secret")
    assert env["NAKSHATRA_LISTEN_ADDRESS"] == "[::]"
    assert env["NAKSHATRA_TLS_REQUIRED"] == "true"
    assert env["NAKSHATRA_AUTH_REQUIRED"] == "true"
    assert env["NAKSHATRA_RELEASE_SERVICE"] == "true"
    assert env["NAKSHATRA_AUDIT_PATH"] == str(root / "audit.jsonl")

    monkeypatch.setenv("LISTEN", "0.0.0.0")
    executed.clear()
    assert worker_service.main(argv, execve=lambda path, command, env: executed.append((path, command, env))) == 0
    assert executed[0][2]["NAKSHATRA_LISTEN_ADDRESS"] == "0.0.0.0"

    build.write_text(json.dumps({"daemon_bin": str(daemon), "backend": "rocm", "n_gpu_layers": 99,
                                 "engine_tree": "0" * 40}))
    assert worker_service.main(argv, execve=lambda *_: None) == 1
    error = capsys.readouterr().err
    assert "REFUSING TO START" in error and "nak worker build" in error
    monkeypatch.delenv("PACKAGE_TRUSTED_PUBKEY")
    refused = []
    assert worker_service.main(argv, execve=lambda *a: refused.append(a)) != 0 and refused == []


def test_worker_launcher_refuses_missing_pillar_and_permissive_node_key(tmp_path, monkeypatch, capsys):
    root = tmp_path / "worker"
    daemon = root / "daemon"
    daemon.parent.mkdir(parents=True)
    daemon.write_text("binary")
    daemon.chmod(0o755)
    build = root / "build.json"
    build.write_text(json.dumps({"daemon_bin": str(daemon), "backend": "cpu", "n_gpu_layers": 0,
                                 "engine_tree": TREE}))
    base = tmp_path / "BASE"
    base.write_text(f"TREE={TREE}\n")
    worker_script = tmp_path / "worker.py"
    worker_script.write_text("# release worker\n")
    key = tmp_path / "worker.ed25519"
    key.write_bytes(ed25519.Ed25519PrivateKey.generate().private_bytes_raw())
    key.chmod(0o600)
    argv = ["--build-json", str(build), "--engine-base", str(base), "--worker-root", str(root),
            "--worker-script", str(worker_script), "--node-key", str(key), "--port", "5530",
            "--sub-gguf", str(root / "slice.gguf"), "--mode", "first",
            "--layer-start", "0", "--layer-end", "16", "--model-id", "qwen"]
    monkeypatch.setenv("PACKAGE_URL", "https://models.test/qwen")
    monkeypatch.setenv("PACKAGE_TRUSTED_PUBKEY", "cd" * 32)
    monkeypatch.delenv("PILLAR_URL", raising=False)

    assert worker_service.main(argv, execve=lambda *_: None) == 1
    assert "worker.env must set PILLAR_URL" in capsys.readouterr().err

    monkeypatch.setenv("PILLAR_URL", "https://pillar.test")
    monkeypatch.setenv("PILLAR_SPKI_SHA256", "ab" * 32)
    key.chmod(0o640)
    assert worker_service.main(argv, execve=lambda *_: None) == 1
    error = capsys.readouterr().err
    assert "not owner-only" in error
    assert f"chmod 600 {key}" in error


def test_worker_cli_reads_sensitive_urls_and_bind_from_env_but_flags_override(tmp_path, monkeypatch):
    sub = tmp_path / "slice.gguf"
    sub.write_bytes(b"test")
    monkeypatch.setattr(worker, "DaemonClient", lambda *_args, **_kwargs: object())
    captured = []

    class Parsed(Exception):
        pass

    def capture(args):
        captured.append(args)
        raise Parsed

    monkeypatch.setattr(worker, "setup_tls", capture)
    monkeypatch.setenv("NAKSHATRA_PILLAR_URL", "https://pillar.env")
    monkeypatch.setenv("NAKSHATRA_PACKAGE_URL", "https://package.env")
    monkeypatch.delenv("NAKSHATRA_LISTEN_ADDRESS", raising=False)
    required = ["--sub-gguf", str(sub), "--mode", "first", "--layer-start", "0", "--layer-end", "1"]

    monkeypatch.setattr(sys, "argv", ["worker.py", *required])
    with pytest.raises(Parsed):
        worker.main()
    assert (captured[-1].pillar_url, captured[-1].package_url, captured[-1].listen_address) == (
        "https://pillar.env", "https://package.env", "[::]")

    monkeypatch.setattr(sys, "argv", ["worker.py", *required,
                                      "--pillar-url", "https://pillar.flag",
                                      "--package-url", "https://package.flag",
                                      "--listen-address", "0.0.0.0"])
    with pytest.raises(Parsed):
        worker.main()
    assert (captured[-1].pillar_url, captured[-1].package_url, captured[-1].listen_address) == (
        "https://pillar.flag", "https://package.flag", "0.0.0.0")


@pytest.mark.parametrize(("address", "expected"), [
    ("::", "[::]:5530"),
    ("::1", "[::1]:5530"),
    ("0.0.0.0", "0.0.0.0:5530"),
    ("127.0.0.1", "127.0.0.1:5530"),
    ("worker.example", "worker.example:5530"),
])
def test_grpc_listen_endpoint_formats_ipv6(address, expected):
    assert worker.grpc_listen_endpoint(address, 5530) == expected


def test_package_url_audit_omits_credentials_query_and_fragment(monkeypatch):
    url = "https://alice:secret@models.test:8443/qwen/package.json?token=secret#private"
    safe = "https://models.test:8443/qwen/package.json"
    audited = []
    monkeypatch.setattr(worker, "_audit",
                        lambda event, **payload: audited.append((event, payload)))
    monkeypatch.setitem(
        sys.modules,
        "packaging.fetch_package",
        type("Fetch", (), {
            "fetch_and_assemble": staticmethod(lambda *_args, **_kwargs: None),
        }),
    )

    worker.provision_from_package(url, 0, 1, "/tmp/test.gguf")

    assert audited[0][1]["package_url"] == safe
    assert "alice" not in repr(audited)
    assert "secret" not in repr(audited)
    assert "token" not in repr(audited)
    assert "private" not in repr(audited)


def test_package_url_console_output_omits_credentials_query_and_fragment(
        tmp_path, monkeypatch, capsys):
    url = "https://alice:secret@models.test:8443/qwen?token=secret#private"
    safe = "https://models.test:8443/qwen"
    sub = tmp_path / "missing.gguf"
    monkeypatch.setenv("NAKSHATRA_PACKAGE_URL", url)
    monkeypatch.setenv("NAKSHATRA_AUDIT_PATH", str(tmp_path / "audit.jsonl"))

    def fail_with_url(*_args, **_kwargs):
        raise RuntimeError(f"fetch failed for {url}")

    monkeypatch.setattr(worker, "provision_from_package", fail_with_url)
    monkeypatch.setattr(sys, "argv", [
        "worker.py", "--sub-gguf", str(sub), "--mode", "first",
        "--layer-start", "0", "--layer-end", "1",
    ])

    with pytest.raises(SystemExit) as exc:
        worker.main()

    logged = capsys.readouterr().out + str(exc.value)
    assert safe in logged
    assert "alice" not in logged
    assert "secret" not in logged
    assert "token=" not in logged
    assert "#private" not in logged


@pytest.mark.parametrize(("location", "expected"), [
    ("https://u:p@example.com/a/b?q=s#f", "https://example.com/a/b"),
    ("https://u:p@[2001:db8::1]:9443/a?q=s", "https://[2001:db8::1]:9443/a"),
    ("/models/qwen?token=secret#fragment", "/models/qwen"),
])
def test_package_location_for_log(location, expected):
    assert worker.package_location_for_log(location) == expected


def test_nak_worker_on_defaults_to_wildcard_listen(tmp_path, monkeypatch):
    home, prefix = tmp_path / "home", tmp_path / "node"
    _fake_release(prefix)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs:
                        type("R", (), {"returncode": 0, "stdout": ""})())

    argv = ["worker", "on", "--model-id", "qwen",
            "--package-url", "https://models.test/qwen", "--package-pubkey", "cd" * 32,
            "--pillar-url", "https://pillar.test", "--pillar-spki", "ab" * 32,
            "--layers", "0-16", "--port", "5530"]
    assert nak.main(argv) == 0
    assert 'LISTEN="[::]"' in (home / ".nakshatra" / "worker.env").read_text()


@pytest.mark.parametrize(("url", "pin", "trusted"), [
    ("https://pillar.test:7777", "AB" * 32, "10.42.0.0/24"),
    ("http://127.0.0.1:7777", "", ""),
    ("http://[::1]:7777", "", ""),
    ("http://10.42.0.3:7777", "", "10.42.0.0/24"),
    ("http://192.168.77.4:7777", "", "192.168.77.0/24"),
])
def test_worker_launcher_accepts_pinned_https_or_trusted_http(url, pin, trusted):
    assert worker_service.validate_pillar_transport(url, pin, trusted) == (
        url, pin.lower()
    )


@pytest.mark.parametrize(("url", "pin", "trusted", "message"), [
    ("https://pillar.test", "", "10.42.0.0/24", "requires PILLAR_SPKI_SHA256"),
    ("https://pillar.test", "abc", "10.42.0.0/24", "64 hexadecimal"),
    ("http://203.0.113.9:7777", "", "10.42.0.0/24", "outside loopback"),
    ("http://pillar.test:7777", "", "10.42.0.0/24", "IP literal"),
    ("ftp://10.42.0.3/file", "", "10.42.0.0/24", "http:// or https://"),
])
def test_worker_launcher_refuses_untrusted_pillar_transport(url, pin, trusted, message):
    with pytest.raises(worker_service.WorkerServiceError, match=message):
        worker_service.validate_pillar_transport(url, pin, trusted)


def test_nak_worker_on_requires_https_pin_before_writing_env(tmp_path, monkeypatch):
    home, prefix = tmp_path / "home", tmp_path / "node"
    _fake_release(prefix)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    argv = ["worker", "on", "--model-id", "qwen",
            "--package-url", "https://models.test/qwen", "--package-pubkey", "cd" * 32,
            "--pillar-url", "https://pillar.test", "--layers", "0-16", "--port", "5530"]

    with pytest.raises(SystemExit, match="--pillar-spki"):
        nak.main(argv)
    assert not (home / ".nakshatra" / "worker.env").exists()


def test_pillar_url_redaction_omits_credentials_query_and_fragment(monkeypatch, capsys):
    url = "https://alice:secret@pillar.test:8443/base?token=secret#private"
    safe = "https://pillar.test:8443/base"
    audited = []
    monkeypatch.setattr(worker, "_audit",
                        lambda event, **payload: audited.append((event, payload)))

    def fail(*_args, **_kwargs):
        raise worker.urlerror.URLError(f"failed at {url}")

    monkeypatch.setattr(worker._wauth, "open_pillar_url", fail)
    assert worker.register_with_pillar(url, {}, spki_hash="ab" * 32) is False

    logged = capsys.readouterr().out + repr(audited)
    assert safe in logged
    assert "alice" not in logged
    assert "secret" not in logged
    assert "token=" not in logged
    assert "#private" not in logged


def test_worker_status_reports_build_and_unit_state(tmp_path, monkeypatch, capsys):
    home, prefix = tmp_path / "home", tmp_path / "node"
    _fake_release(prefix)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    record = home / ".nakshatra" / "worker" / "build.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"engine_tree": TREE, "backend": "cpu"}))
    (home / ".nakshatra" / "worker.env").write_text("PORT=5530\n")

    def run(args, **kwargs):
        value = "active\n" if "is-active" in args else "enabled\n"
        return type("R", (), {"returncode": 0, "stdout": value})()

    monkeypatch.setattr(subprocess, "run", run)
    assert nak.main(["worker", "status"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["configured"] is True and out["build_state"] == "ready"
    assert out["unit"] == {"active": "active", "enabled": "enabled"}


@pytest.mark.parametrize("url", ["https://pillar.test/base", "https://pillar.test/?token=x",
                                 "https://user:pw@pillar.test", "https://pillar.test/#frag"])
def test_pillar_url_must_be_a_bare_origin(url):
    with pytest.raises(worker_service.WorkerServiceError, match="bare origin"):
        worker_service.validate_pillar_transport(url, "ab" * 32, "10.42.0.0/24")
