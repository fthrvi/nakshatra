"""U4 B1: a worker built and run entirely from the installed signed release."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from network import nak
import worker
import worker_service


TREE = "6d5b567d598666d20281db6548bbefd582d26f05"


def _worker_on_args(**overrides):
    values = {
        "model_id": "qwen", "package_url": "https://models.test/qwen",
        "package_pubkey": "cd" * 32, "pillar_url": "http://localhost",
        "pillar_spki": None, "trusted_pillar_nets": "10.42.0.0/24",
        "listen_address": "0.0.0.0", "layers": "0-16", "port": 5530,
        "idle_unload_s": 600.0, "release_grace_s": 30.0,
        "warm_directives": "on", "mode": "first",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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
    assert "IDLE_UNLOAD_S=600" in first
    assert "RELEASE_GRACE_S=30" in first
    assert "WARM_DIRECTIVES=1" in first
    assert ('NAKSHATRA_P2P_WORKER_ARGS="-inbound=127.0.0.1:5530 '
            '-require-direct-activation"') in first
    assert "SUB_GGUF=" in first and "skip-sha256" not in first
    inference_env = home / ".nakshatra" / "inference.env"
    assert not inference_env.exists()
    assert (home / ".nakshatra" / "tls").is_dir()
    installer = prefix / "current" / "nakshatra" / "release" / "install.py"
    assert calls[-1] == [sys.executable, str(installer), "--prefix", str(prefix),
                         "enable", "nakshatra-worker"]

    # Existing worker and inference configuration is authoritative: a bare `on` does not rewrite
    # either file, and worker activation never creates or removes the gateway's profile marker.
    inference_env.write_text("EXISTING=leave-me-alone\n")
    assert nak.main(["worker", "on"]) == 0
    assert env_path.read_text() == first
    assert inference_env.read_text() == "EXISTING=leave-me-alone\n"


def test_nak_worker_on_and_off_restart_opted_in_sidecar_through_installer(
        tmp_path, monkeypatch):
    home, prefix = tmp_path / "home", tmp_path / "node"
    _fake_release(prefix)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    p2p_env = home / ".nakshatra" / "p2p.env"
    p2p_env.parent.mkdir(parents=True)
    p2p_env.write_text("P2P_RELAYS=test\n")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return type("R", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(subprocess, "run", run)
    argv = ["worker", "on", "--model-id", "qwen",
            "--package-url", "https://models.test/qwen", "--package-pubkey", "cd" * 32,
            "--pillar-url", "https://pillar.test", "--pillar-spki", "ab" * 32,
            "--layers", "0-16", "--port", "5530"]
    assert nak.main(argv) == 0
    installer = prefix / "current" / "nakshatra" / "release" / "install.py"
    assert calls[-2:] == [
        [sys.executable, str(installer), "--prefix", str(prefix), "enable", "nakshatra-worker"],
        [sys.executable, str(installer), "--prefix", str(prefix), "restart", "nakshatra-p2p"],
    ]

    calls.clear()
    assert nak.main(["worker", "off"]) == 0
    assert "NAKSHATRA_P2P_WORKER_ARGS" not in (
        home / ".nakshatra" / "worker.env").read_text()
    installer_calls = [call for call in calls if call and call[0] == sys.executable]
    assert installer_calls == [
        [sys.executable, str(installer), "--prefix", str(prefix), "disable", "nakshatra-worker"],
        [sys.executable, str(installer), "--prefix", str(prefix), "restart", "nakshatra-p2p"],
    ]


def test_worker_off_with_live_gateway_writes_consumer_override_before_p2p_restart(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    (state / "p2p.env").write_text("P2P_RELAYS=test\n")
    (state / "worker.env").write_text(
        'PORT=5530\nNAKSHATRA_P2P_WORKER_ARGS="-inbound=127.0.0.1:5530"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_worker_installer", lambda *_args: None)
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: True)
    restarts = []

    def p2p_installer(_sp, action):
        assert (state / "p2p-consumer.env").exists()
        restarts.append(action)

    monkeypatch.setattr(nak, "_p2p_installer", p2p_installer)

    nak._worker_disable(object())

    assert restarts == ["restart"]
    assert "NAKSHATRA_P2P_WORKER_ARGS" not in (state / "worker.env").read_text()
    assert (state / "p2p-consumer.env").read_text() == (
        'NAKSHATRA_P2P_NAKD_INBOUND=""\nNAKSHATRA_P2P_WORKER_ARGS=""\n')


def test_worker_enable_failed_stop_retains_new_config_and_worker_inbound(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    (state / "p2p.env").write_text("P2P_RELAYS=test\n")
    (state / "p2p-consumer.env").write_text(
        'NAKSHATRA_P2P_NAKD_INBOUND=""\nNAKSHATRA_P2P_WORKER_ARGS=""\n')
    monkeypatch.setenv("HOME", str(home))
    actions = []

    def worker_installer(_sp, action):
        actions.append(action)
        raise SystemExit(f"worker {action} failed")

    monkeypatch.setattr(nak, "_worker_installer", worker_installer)
    monkeypatch.setattr(nak, "_p2p_installer", lambda *_args: None)

    with pytest.raises(SystemExit, match=r"enable failed; rollback could not disable"):
        nak._worker_enable(object(), _worker_on_args())

    worker_env = state / "worker.env"
    assert actions == ["enable", "disable"]
    assert worker_env.exists()
    assert '-inbound=127.0.0.1:5530' in worker_env.read_text()
    assert not (state / "p2p-consumer.env").exists()


def test_worker_enable_confirmed_stop_restores_prior_consumer_state(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    (state / "p2p.env").write_text("P2P_RELAYS=test\n")
    consumer = state / "p2p-consumer.env"
    original_consumer = (
        'NAKSHATRA_P2P_NAKD_INBOUND=""\nNAKSHATRA_P2P_WORKER_ARGS=""\n')
    consumer.write_text(original_consumer)
    monkeypatch.setenv("HOME", str(home))
    actions = []

    def worker_installer(_sp, action):
        actions.append(action)
        if action == "enable":
            raise SystemExit("worker enable failed")

    monkeypatch.setattr(nak, "_worker_installer", worker_installer)
    monkeypatch.setattr(nak, "_p2p_installer", lambda *_args: None)

    with pytest.raises(SystemExit, match="worker enable failed"):
        nak._worker_enable(object(), _worker_on_args())

    assert actions == ["enable", "disable"]
    assert not (state / "worker.env").exists()
    assert consumer.read_text() == original_consumer


def test_nak_worker_relay_toggle_is_atomic_restarts_p2p_and_shows_in_status(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    worker_env = state / "worker.env"
    worker_env.write_text(
        'PORT=5530\nNAKSHATRA_P2P_WORKER_ARGS="-inbound=127.0.0.1:5530 '
        '-require-direct-activation"\n'
    )
    (state / "p2p.env").write_text("P2P_RELAYS=test\n")
    monkeypatch.setenv("HOME", str(home))
    calls = []
    monkeypatch.setattr(nak, "_p2p_installer", lambda _sp, action: calls.append(action))

    assert nak.main(["worker", "relay", "on"]) == 0
    assert worker_env.read_text().count("-allow-relayed-activation") == 1
    assert calls == ["restart"]

    class StatusSP:
        @staticmethod
        def run(args, **_kwargs):
            return type("R", (), {"returncode": 0, "stdout": "inactive\n"})()

    assert nak._worker_status(StatusSP())["allow_relayed_activation"] is True
    assert nak.main(["worker", "relay", "off"]) == 0
    assert "-allow-relayed-activation" not in worker_env.read_text()
    assert calls == ["restart", "restart"]
    assert nak._worker_status(StatusSP())["allow_relayed_activation"] is False


def test_nak_worker_relay_restart_failure_restores_exact_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    worker_env = state / "worker.env"
    original = b'PORT=5530\nEXISTING=keep\n'
    worker_env.write_bytes(original)
    (state / "p2p.env").write_text("P2P_RELAYS=test\n")
    monkeypatch.setenv("HOME", str(home))

    calls = []

    def fail_once(*_args):
        calls.append("restart")
        if len(calls) == 1:
            raise SystemExit("restart failed")

    monkeypatch.setattr(nak, "_p2p_installer", fail_once)
    with pytest.raises(SystemExit, match="restart failed"):
        nak.main(["worker", "relay", "on"])
    assert worker_env.read_bytes() == original
    assert calls == ["restart", "restart"]


def test_nak_worker_relay_surfaces_restart_and_rollback_failures(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    worker_env = state / "worker.env"
    original = b"PORT=5530\nEXISTING=keep\n"
    worker_env.write_bytes(original)
    (state / "p2p.env").write_text("P2P_RELAYS=test\n")
    monkeypatch.setenv("HOME", str(home))
    calls = []

    def fail_both(*_args):
        calls.append("restart")
        raise SystemExit(f"failure-{len(calls)}")

    monkeypatch.setattr(nak, "_p2p_installer", fail_both)
    with pytest.raises(SystemExit, match=r"relay restart failed: failure-1; rollback restart failed: failure-2"):
        nak.main(["worker", "relay", "on"])
    assert worker_env.read_bytes() == original
    assert calls == ["restart", "restart"]


def test_nak_worker_on_without_p2p_reports_literal_only(tmp_path, monkeypatch, capsys):
    home, prefix = tmp_path / "home", tmp_path / "node"
    _fake_release(prefix)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs:
                        type("R", (), {"returncode": 0, "stdout": ""})())
    assert nak.main([
        "worker", "on", "--model-id", "qwen",
        "--package-url", "https://models.test/qwen", "--package-pubkey", "cd" * 32,
        "--pillar-url", "https://pillar.test", "--pillar-spki", "ab" * 32,
        "--layers", "0-16", "--port", "5530",
    ]) == 0
    assert "p2p is not enabled; this worker is reachable only by its literal address" in capsys.readouterr().out


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
    p2p_opt_in = tmp_path / "p2p.env"
    p2p_opt_in.write_text("P2P_RELAYS=test\n")
    argv = ["--build-json", str(build), "--engine-base", str(base), "--worker-root", str(root),
            "--worker-script", str(worker), "--node-key", str(key),
            "--p2p-opt-in", str(p2p_opt_in), "--port", "5530",
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
    assert env["NAKSHATRA_WORKER_IDLE_UNLOAD_S"] == "600"
    assert env["NAKSHATRA_TLS_REQUIRED"] == "true"
    assert env["NAKSHATRA_AUTH_REQUIRED"] == "true"
    assert env["NAKSHATRA_RELEASE_SERVICE"] == "true"
    assert env["NAKSHATRA_AUDIT_PATH"] == str(root / "audit.jsonl")
    from sidecar_key import peer_id_from_node_pub
    assert env["NAKSHATRA_P2P_PEER_ID"] == peer_id_from_node_pub(pub)

    monkeypatch.setenv("LISTEN", "0.0.0.0")
    monkeypatch.setenv("NAKSHATRA_P2P_PEER_ID", "attacker-controlled")
    p2p_opt_in.unlink()
    executed.clear()
    assert worker_service.main(argv, execve=lambda path, command, env: executed.append((path, command, env))) == 0
    assert executed[0][2]["NAKSHATRA_LISTEN_ADDRESS"] == "0.0.0.0"
    assert "NAKSHATRA_P2P_PEER_ID" not in executed[0][2]

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


def test_nak_worker_on_writes_idle_unload_override(tmp_path, monkeypatch):
    home, prefix = tmp_path / "home", tmp_path / "node"
    _fake_release(prefix)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs:
                        type("R", (), {"returncode": 0, "stdout": ""})())

    argv = ["worker", "on", "--model-id", "qwen",
            "--package-url", "https://models.test/qwen", "--package-pubkey", "cd" * 32,
            "--pillar-url", "https://pillar.test", "--pillar-spki", "ab" * 32,
            "--layers", "0-16", "--port", "5530", "--idle-unload-s", "0",
            "--release-grace-s", "-1", "--warm-directives", "off"]
    assert nak.main(argv) == 0
    body = (home / ".nakshatra" / "worker.env").read_text()
    assert "IDLE_UNLOAD_S=0" in body
    assert "RELEASE_GRACE_S=-1" in body
    assert "WARM_DIRECTIVES=0" in body


def test_worker_launcher_passes_idle_unload_and_defaults_old_env_to_600(
        tmp_path, monkeypatch):
    root = tmp_path / "worker"
    daemon = root / "daemon"
    daemon.parent.mkdir(parents=True)
    daemon.write_text("binary")
    daemon.chmod(0o755)
    build = root / "build.json"
    build.write_text(json.dumps({"daemon_bin": str(daemon), "backend": "cpu",
                                 "n_gpu_layers": 0, "engine_tree": TREE}))
    base = tmp_path / "BASE"
    base.write_text(f"TREE={TREE}\n")
    worker_script = tmp_path / "worker.py"
    worker_script.write_text("# release worker\n")
    key = tmp_path / "worker.ed25519"
    key.write_bytes(ed25519.Ed25519PrivateKey.generate().private_bytes_raw())
    key.chmod(0o600)
    args = type("Args", (), {
        "build_json": build, "engine_base": base, "worker_root": root,
        "worker_script": worker_script, "node_key": key, "p2p_opt_in": None,
        "port": 5530, "sub_gguf": root / "slice.gguf", "mode": "first",
        "layer_start": 0, "layer_end": 1, "model_id": "qwen", "n_ctx": 2048,
    })()
    monkeypatch.setenv("PILLAR_URL", "https://pillar.test")
    monkeypatch.setenv("PILLAR_SPKI_SHA256", "ab" * 32)
    monkeypatch.setenv("PACKAGE_URL", "https://models.test/qwen")
    monkeypatch.setenv("PACKAGE_TRUSTED_PUBKEY", "cd" * 32)
    monkeypatch.delenv("IDLE_UNLOAD_S", raising=False)

    assert worker_service.worker_command(args)[1]["NAKSHATRA_WORKER_IDLE_UNLOAD_S"] == "600"
    assert worker_service.worker_command(args)[1]["NAKSHATRA_WORKER_WARM_DIRECTIVES"] == "1"
    monkeypatch.setenv("IDLE_UNLOAD_S", "12.5")
    assert worker_service.worker_command(args)[1]["NAKSHATRA_WORKER_IDLE_UNLOAD_S"] == "12.5"
    monkeypatch.setenv("WARM_DIRECTIVES", "0")
    assert worker_service.worker_command(args)[1]["NAKSHATRA_WORKER_WARM_DIRECTIVES"] == "0"


def test_worker_launcher_validates_and_passes_owner_lease(tmp_path, monkeypatch):
    root = tmp_path / "worker"
    daemon = root / "daemon"
    daemon.parent.mkdir(parents=True)
    daemon.write_text("binary")
    daemon.chmod(0o755)
    build = root / "build.json"
    build.write_text(json.dumps({"daemon_bin": str(daemon), "backend": "cpu",
                                 "n_gpu_layers": 0, "engine_tree": TREE}))
    base = tmp_path / "BASE"
    base.write_text(f"TREE={TREE}\n")
    worker_script = tmp_path / "worker.py"
    worker_script.write_text("# release worker\n")
    key = tmp_path / "worker.ed25519"
    key.write_bytes(ed25519.Ed25519PrivateKey.generate().private_bytes_raw())
    key.chmod(0o600)
    args = type("Args", (), {
        "build_json": build, "engine_base": base, "worker_root": root,
        "worker_script": worker_script, "node_key": key, "p2p_opt_in": None,
        "port": 5530, "sub_gguf": root / "slice.gguf", "mode": "first",
        "layer_start": 0, "layer_end": 1, "model_id": "qwen", "n_ctx": 2048,
    })()
    monkeypatch.setenv("PILLAR_URL", "https://pillar.test")
    monkeypatch.setenv("PILLAR_SPKI_SHA256", "ab" * 32)
    monkeypatch.setenv("PACKAGE_URL", "https://models.test/qwen")
    monkeypatch.setenv("PACKAGE_TRUSTED_PUBKEY", "cd" * 32)
    monkeypatch.setenv("OWNER_LEASE_PORT", "5580")
    monkeypatch.setenv("OWNER_LEASE_TOKEN", "ef" * 32)
    monkeypatch.setenv("OWNER_LEASE_TTL_S", "8")
    monkeypatch.setenv("OWNER_LEASE_RESUME_S", "12")

    env = worker_service.worker_command(args)[1]
    assert env["NAKSHATRA_OWNER_LEASE_ENABLED"] == "true"
    assert env["NAKSHATRA_OWNER_LEASE_PORT"] == "5580"
    assert env["NAKSHATRA_OWNER_LEASE_TOKEN"] == "ef" * 32
    assert env["NAKSHATRA_OWNER_LEASE_TTL_S"] == "8"
    assert env["NAKSHATRA_OWNER_LEASE_RESUME_S"] == "12"
    assert env["NAKSHATRA_WORKER_RELEASE_GRACE_S"] == "30"

    monkeypatch.setenv("OWNER_LEASE_TOKEN", "too-short")
    with pytest.raises(worker_service.WorkerServiceError, match=r"32\+ random bytes"):
        worker_service.worker_command(args)


def test_owner_lease_cli_never_prints_token_and_restarts_enabled_worker(
        tmp_path, monkeypatch, capsys):
    home, prefix = tmp_path / "home", tmp_path / "node"
    _fake_release(prefix)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    token = "12" * 32
    monkeypatch.setattr(nak.secrets, "token_hex", lambda _n: token)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["systemctl", "--user", "is-enabled"]:
            return type("R", (), {"returncode": 0, "stdout": "enabled\n"})()
        return type("R", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(subprocess, "run", run)
    assert nak.main(["worker", "owner-lease", "on", "--port", "5580"]) == 0
    output = capsys.readouterr().out
    path = home / ".nakshatra" / "owner-lease.env"
    assert "127.0.0.1:5580" in output and str(path) in output
    assert token not in output
    assert token in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    installer = prefix / "current" / "nakshatra" / "release" / "install.py"
    assert calls[-1] == [sys.executable, str(installer), "--prefix", str(prefix),
                         "restart", "nakshatra-worker"]

    assert nak.main(["worker", "owner-lease", "status"]) == 0
    status_output = capsys.readouterr().out
    assert token not in status_output
    assert json.loads(status_output)["port"] == 5580

    assert nak.main(["worker", "owner-lease", "off"]) == 0
    assert not path.exists()
    assert token not in capsys.readouterr().out


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
