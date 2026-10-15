from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
CONTAINER = ROOT / "deploy" / "container"
NAKC = CONTAINER / "nakc"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


supervisor = _load("nak_container_supervisor", CONTAINER / "supervisor.py")
bootstrap = _load("nak_container_bootstrap", CONTAINER / "bootstrap.py")
sys.path.insert(0, str(ROOT / "release"))
release_installer = _load("nak_container_release_installer", ROOT / "release" / "install.py")
sys.path.insert(0, str(ROOT / "scripts"))
from network import nak  # noqa: E402


def _effective_p2p_argv(tmp_path: Path, home: Path) -> list[str]:
    spec = json.loads((ROOT / "release" / "spec.json").read_text())
    service = spec["services"]["nakshatra-p2p"]
    manifest = {"version": "test", "components": [{"name": "nakshatra", "pythonpath": "scripts"}]}
    prefix = tmp_path / "prefix"
    installer = release_installer.Installer(
        prefix, systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    unit_path = tmp_path / "p2p.service"
    unit_path.write_text(installer._unit_text(service, installer._ctx(manifest), manifest))
    return supervisor.parse_unit(
        unit_path, {"HOME": str(home), "XDG_RUNTIME_DIR": str(tmp_path / "run")}
    ).argv


def _legacy_p2p_argv(
        tmp_path: Path, home: Path, relays: str, worker_args: list[str] | None = None) -> list[str]:
    return [
        str(tmp_path / "prefix" / "current" / "nakshatra" / "bin" / "nakshatra-sidecar"),
        "-key", str(home / ".config" / "nakshatra-sidecar" / "node.key"),
        "-listen", "/ip4/0.0.0.0/tcp/0", "-quic",
        "-nakd-inbound=127.0.0.1:51830",
        f"-dial-listen={tmp_path / 'run' / 'nakshatra' / 'p2p.sock'}",
        f"-relays={relays}",
        *(worker_args or []),
    ]


def _nakc(tmp_path: Path, *args: str, **extra_env: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, NAKC_DRY_RUN="1", NAKC_ENGINE="docker",
               NAKC_CONFIG_DIR=str(tmp_path / "config"), **extra_env)
    return subprocess.run([str(NAKC), *args], env=env, text=True, capture_output=True, check=False)


def _dry_run_argv(output: str, occurrence: int = 0) -> list[str]:
    lines = [line for line in output.splitlines() if line.startswith("+ docker run ")]
    return shlex.split(lines[occurrence][2:])


def test_containerfile_is_nonroot_pinned_and_does_not_copy_secrets():
    body = (CONTAINER / "Containerfile").read_text()
    assert re.search(r"^ARG BASE_IMAGE=(?![^\n]*(?:latest|edge)\b)[^\n]+:[^\n]+$", body, re.MULTILINE)
    assert re.search(r"^USER 11600:11600$", body, re.MULTILINE)
    assert not re.search(r"^ADD\s+https?://", body, re.MULTILINE)
    assert not re.search(r"COPY[^\n]*(?:\.env|\.key|id_rsa|credentials|\.nakshatra|\.sthambha)", body, re.I)
    assert "sudo" not in body
    assert 'ENTRYPOINT ["/usr/bin/tini"' in body


def test_nakc_default_run_is_hardened_and_mounts_only_the_state_volume(tmp_path):
    result = _nakc(tmp_path, "start")
    assert result.returncode == 0, result.stderr
    argv = _dry_run_argv(result.stdout)
    assert ["--cap-drop", "ALL"] == argv[argv.index("--cap-drop"):argv.index("--cap-drop") + 2]
    assert ["--security-opt", "no-new-privileges"] == argv[
        argv.index("--security-opt"):argv.index("--security-opt") + 2]
    assert "--read-only" in argv and "--network" in argv and argv[argv.index("--network") + 1] == "bridge"
    assert "--network=host" not in argv and "--privileged" not in argv
    assert "127.0.0.1:11600:11599/tcp" in argv
    assert not any(item.startswith(("NAK_CONTAINER=", "NAK_GATEWAY_BIND=", "NAK_GATEWAY_PORT="))
                   for item in argv)
    assert not any(item.endswith(":/share") or "dst=/share" in item for item in argv)
    mounts = [argv[index + 1] for index, item in enumerate(argv) if item == "--volume"]
    assert mounts == ["nakshatra-node-state:/state"]
    assert not any(str(Path.home()) in item for item in argv)
    assert not any(item.startswith("--gpus") or item in {"/dev/kfd", "/dev/dri"} for item in argv)


def test_share_is_read_only_by_default_and_rw_only_when_explicit(tmp_path):
    shared = tmp_path / "friend-share"
    shared.mkdir()
    assert _nakc(tmp_path, "share", str(shared)).returncode == 0
    argv = _dry_run_argv(_nakc(tmp_path, "start").stdout)
    share = argv[argv.index("--mount") + 1]
    assert share == f"type=bind,src={shared},dst=/share,readonly"

    assert _nakc(tmp_path, "share", "--share-rw", str(shared)).returncode == 0
    argv = _dry_run_argv(_nakc(tmp_path, "start").stdout)
    share = argv[argv.index("--mount") + 1]
    assert share == f"type=bind,src={shared},dst=/share"


def test_only_fixed_p2p_port_and_gpu_devices_are_added_while_worker_is_on(tmp_path):
    worker = _nakc(tmp_path, "worker", "on", "--port", "50051", NAKC_VARIANT="nvidia")
    assert worker.returncode == 0, worker.stderr
    argv = _dry_run_argv(worker.stdout)
    assert "50051:50051/tcp" not in argv
    assert "0.0.0.0:4001:4001/tcp" in argv and "0.0.0.0:4001:4001/udp" in argv
    assert "NAK_CONTAINER_WORKER=1" in argv and "NAK_CONTAINER_P2P_PORT=4001" in argv
    assert "--gpus" in argv and argv[argv.index("--gpus") + 1] == "all"
    assert "deliberate public port" in worker.stderr
    assert "+ docker exec -i nakshatra-node nak p2p on" in worker.stdout
    clean = _nakc(tmp_path / "clean", "start", NAKC_VARIANT="cpu")
    clean_argv = _dry_run_argv(clean.stdout)
    assert not any(item.endswith(":4001/tcp") or item.endswith(":4001/udp") for item in clean_argv)


def test_worker_p2p_port_is_configurable_and_persisted(tmp_path):
    worker = _nakc(tmp_path, "worker", "on", "--port", "50051", NAKC_P2P_PORT="24001")
    assert worker.returncode == 0, worker.stderr
    argv = _dry_run_argv(worker.stdout)
    assert "0.0.0.0:24001:24001/tcp" in argv and "0.0.0.0:24001:24001/udp" in argv
    restarted = _nakc(tmp_path, "start")
    argv = _dry_run_argv(restarted.stdout)
    assert "0.0.0.0:24001:24001/tcp" in argv and "NAK_CONTAINER_P2P_PORT=24001" in argv


def test_worker_dry_run_golden_is_byte_identical(tmp_path):
    result = _nakc(tmp_path, "worker", "on", "--port", "50051")
    assert result.returncode == 0
    assert result.stderr == (
        "nakc: deliberate public port: libp2p sidecar 0.0.0.0:4001 (tcp+udp); "
        "worker gRPC stays on container loopback\n"
    )
    assert result.stdout == (
        "+ docker run -d --name nakshatra-node --hostname nakshatra-node "
        "--label io.nakshatra.role=node --restart unless-stopped --user 11600:11600 "
        "--cap-drop ALL --security-opt no-new-privileges --read-only "
        "--tmpfs /tmp:rw\\,nosuid\\,nodev\\,size=2g\\,mode=1777 --pids-limit 512 "
        "--memory 16g --network bridge --stop-timeout 20 --publish "
        "127.0.0.1:11600:11599/tcp --env NAK_UPDATE_INTERVAL_S=3600 --volume "
        "nakshatra-node-state:/state --env NAK_CONTAINER_WORKER=1 --env "
        "NAK_CONTAINER_P2P_PORT=4001 --publish 0.0.0.0:4001:4001/tcp --publish "
        "0.0.0.0:4001:4001/udp nakshatra-node:cpu \n"
        "+ docker exec -i nakshatra-node nak p2p on \n"
        "+ docker exec -i nakshatra-node nak worker on --port 50051 \n"
    )


def test_join_dry_run_never_prints_the_invite(tmp_path):
    secret = "nki1.SUPER_SECRET_TOKEN"
    result = _nakc(tmp_path, "join", f"fake bootstrap {secret}")
    assert result.returncode == 0
    assert secret not in result.stdout + result.stderr
    assert "<invite redacted>" in result.stdout


def test_mcp_is_a_direct_interactive_exec(tmp_path):
    result = _nakc(tmp_path, "mcp")
    assert result.returncode == 0
    assert shlex.split(result.stdout.removeprefix("+ ")) == [
        "docker", "exec", "-i", "nakshatra-node", "nak-mcp"
    ]


def test_gateway_on_uses_fixed_internal_endpoint_and_explicit_lan_ack(tmp_path):
    result = _nakc(
        tmp_path, "gateway", "on", "--pillar-url", "https://pillar.example",
        "--pillar-spki", "17" * 32, "--ledger-url", "https://pillar.example/ledger",
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    run_index = next(index for index, line in enumerate(lines) if line.startswith("+ docker run "))
    exec_index = next(index for index, line in enumerate(lines) if line.startswith("+ docker exec "))
    assert run_index < exec_index
    argv = _dry_run_argv(result.stdout)
    assert "NAK_CONTAINER_CONSUMER_P2P_PORT=4001" in argv
    assert "4001:4001/tcp" in argv and "4001:4001/udp" in argv
    assert not any(item.startswith("0.0.0.0:4001:") for item in argv)
    assert result.stderr == (
        "nakc: deliberate public port: libp2p sidecar 0.0.0.0:4001 (tcp+udp) "
        "for direct paths; forwards nothing inbound\n"
    )
    assert not any(" nak p2p on " in f" {line} " for line in lines)
    command = lines[exec_index]
    assert shlex.split(command.removeprefix("+ ")) == [
        "docker", "exec", "-i", "nakshatra-node", "nak", "gateway", "on",
        "--bind", "0.0.0.0", "--port", "11599", "--allow-lan",
        "--pillar-url", "https://pillar.example", "--pillar-spki", "17" * 32,
        "--ledger-url", "https://pillar.example/ledger",
    ]


def test_gateway_p2p_port_is_configurable_and_persisted(tmp_path):
    enabled = _nakc(tmp_path, "gateway", "on", NAKC_P2P_PORT="24001")
    assert enabled.returncode == 0, enabled.stderr
    enabled_argv = _dry_run_argv(enabled.stdout)
    assert "NAK_CONTAINER_CONSUMER_P2P_PORT=24001" in enabled_argv
    assert "24001:24001/tcp" in enabled_argv and "24001:24001/udp" in enabled_argv

    restarted_argv = _dry_run_argv(_nakc(tmp_path, "start").stdout)
    assert "NAK_CONTAINER_CONSUMER_P2P_PORT=24001" in restarted_argv
    assert "24001:24001/tcp" in restarted_argv and "24001:24001/udp" in restarted_argv


@pytest.mark.parametrize(
    ("p2p_port", "gateway_port", "message"),
    [
        ("11599", None, "container gateway port 11599"),
        ("11600", None, "host gateway port 11600"),
        ("24001", "24001", "host gateway port 24001"),
    ],
)
def test_gateway_rejects_p2p_gateway_port_collisions_before_persisting(
        tmp_path, p2p_port, gateway_port, message):
    extra_env = {"NAKC_P2P_PORT": p2p_port}
    if gateway_port is not None:
        extra_env["NAKC_GATEWAY_PORT"] = gateway_port
    result = _nakc(tmp_path, "gateway", "on", **extra_env)
    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "config" / "gateway-on").exists()
    assert not (tmp_path / "config" / "p2p-port").exists()


@pytest.mark.parametrize("port", ["04001", "+4001", "-4001", "0", "65536"])
def test_gateway_rejects_noncanonical_p2p_ports_before_persisting(tmp_path, port):
    result = _nakc(tmp_path, "gateway", "on", NAKC_P2P_PORT=port)
    assert result.returncode == 2
    assert "canonical port between 1 and 65535" in result.stderr
    assert not (tmp_path / "config" / "gateway-on").exists()
    assert not (tmp_path / "config" / "p2p-port").exists()


@pytest.mark.parametrize("port", ["050051", "+50051", "-50051", "0", "65536"])
def test_worker_rejects_noncanonical_ports_before_persisting(tmp_path, port):
    result = _nakc(tmp_path, "worker", "on", "--port", port)
    assert result.returncode == 2
    assert "canonical port between 1 and 65535" in result.stderr
    assert not (tmp_path / "config" / "worker-port").exists()
    assert not (tmp_path / "config" / "p2p-port").exists()


def test_start_rejects_a_noncanonical_host_gateway_port(tmp_path):
    result = _nakc(tmp_path, "start", NAKC_GATEWAY_PORT="011600")
    assert result.returncode == 2
    assert "NAKC_GATEWAY_PORT must be a canonical port" in result.stderr


@pytest.mark.parametrize(
    ("command", "old_values"),
    [
        (("gateway", "on"), {"gateway-on": "old", "p2p-port": "4002"}),
        (("worker", "on", "--port", "50051"),
         {"worker-port": "50052", "p2p-port": "4002"}),
    ],
)
def test_role_enable_rolls_back_config_when_container_start_fails(
        tmp_path, command, old_values):
    config = tmp_path / "config"
    config.mkdir()
    for name, value in old_values.items():
        (config / name).write_text(value + "\n")
    engine = tmp_path / "failing-container-engine"
    engine.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then exit 0; fi\n"
        "if [ \"$1\" = container ] && [ \"$2\" = inspect ]; then exit 1; fi\n"
        "if [ \"$1\" = run ]; then printf '%s\\n' \"$*\" >>\"$NAKC_TEST_LOG\"; exit 23; fi\n"
        "exit 0\n"
    )
    engine.chmod(0o755)
    engine_log = tmp_path / "engine.log"
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        NAKC_ENGINE=str(engine),
        NAKC_CONFIG_DIR=str(config),
        NAKC_P2P_PORT="4001",
        NAKC_TEST_LOG=str(engine_log),
    )
    result = subprocess.run(
        [str(NAKC), *command], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert {name: (config / name).read_text().strip() for name in old_values} == old_values
    attempted, restored = engine_log.read_text().splitlines()
    assert "4001:4001/tcp" in attempted
    assert "4002:4002/tcp" in restored


def test_gateway_off_removes_consumer_publish_on_next_start(tmp_path):
    assert _nakc(tmp_path, "gateway", "on").returncode == 0
    disabled = _nakc(tmp_path, "gateway", "off")
    assert disabled.returncode == 0, disabled.stderr
    assert "+ docker exec -i nakshatra-node nak gateway off" in disabled.stdout

    restarted_argv = _dry_run_argv(_nakc(tmp_path, "start").stdout)
    assert not any(item.startswith("NAK_CONTAINER_CONSUMER_P2P_PORT=") for item in restarted_argv)
    assert "4001:4001/tcp" not in restarted_argv
    assert "4001:4001/udp" not in restarted_argv


def test_gateway_off_keeps_worker_publish(tmp_path):
    assert _nakc(tmp_path, "gateway", "on").returncode == 0
    assert _nakc(tmp_path, "worker", "on", "--port", "50051").returncode == 0
    assert _nakc(tmp_path, "gateway", "off").returncode == 0

    restarted_argv = _dry_run_argv(_nakc(tmp_path, "start").stdout)
    assert "NAK_CONTAINER_WORKER=1" in restarted_argv
    assert "NAK_CONTAINER_P2P_PORT=4001" in restarted_argv
    assert "0.0.0.0:4001:4001/tcp" in restarted_argv
    assert "0.0.0.0:4001:4001/udp" in restarted_argv
    assert not any(item.startswith("NAK_CONTAINER_CONSUMER_P2P_PORT=") for item in restarted_argv)


def test_gateway_on_rejects_endpoint_overrides(tmp_path):
    result = _nakc(tmp_path, "gateway", "on", "--port", "12345")
    assert result.returncode == 2
    assert "fixes the gateway bind and port" in result.stderr


@pytest.mark.parametrize("option", ["--bi", "--p", "--p=11600", "--po", "--allow"])
def test_gateway_on_rejects_abbreviated_endpoint_overrides_before_persisting(tmp_path, option):
    args = (option,) if option in {"--allow", "--p=11600"} else (option, "11600")
    result = _nakc(tmp_path, "gateway", "on", *args)
    assert result.returncode == 2
    assert "fixes the gateway bind and port" in result.stderr
    assert not (tmp_path / "config" / "gateway-on").exists()
    assert not (tmp_path / "config" / "p2p-port").exists()


def test_bootstrap_parser_accepts_only_the_fingerprinted_join_form():
    install_sha, kit_sha = "a" * 64, "b" * 64
    line = (
        "mkdir -p ~/nak-join && cd ~/nak-join && "
        "curl -fsSO https://releases.example/canary/1.2.3/install.py "
        "-O https://releases.example/canary/1.2.3/releasekit.py && "
        "printf '%s  install.py\\n%s  releasekit.py\\n' "
        f"{install_sha} {kit_sha} | sha256sum -c --quiet && "
        "python3 install.py join 'nki1.abc_DEF-123'"
    )
    parsed = bootstrap.parse_invite_line(line)
    assert parsed == (
        "https://releases.example/canary/1.2.3/install.py", install_sha,
        "https://releases.example/canary/1.2.3/releasekit.py", kit_sha,
        "nki1.abc_DEF-123",
    )
    for bad in (line + " && touch /tmp/pwned", line.replace("a" * 64, "x" * 64),
                line.replace("releasekit.py", "https://evil.example/releasekit.py", 1)):
        try:
            bootstrap.parse_invite_line(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe bootstrap line was accepted")


def test_generated_unit_parser_expands_env_files_without_copying_commands(tmp_path):
    env_file = tmp_path / "worker.env"
    env_file.write_text('PORT=50051\nEXTRA="--one two"\n')
    unit_file = tmp_path / "fake.service"
    unit_file.write_text(
        supervisor.UNIT_MARKER + "\n"
        "[Service]\n"
        f"EnvironmentFile=-{env_file}\n"
        "Environment=BASE=ready\n"
        f"ExecStartPre={sys.executable} -c 'raise SystemExit(0)'\n"
        f"ExecStart={sys.executable} -c 'import time; time.sleep(1)' --port=${{PORT}} $EXTRA\n"
        "Restart=always\n"
    )
    unit = supervisor.parse_unit(unit_file, {"HOME": str(tmp_path)})
    assert unit.argv[-3:] == ["--port=50051", "--one", "two"]
    assert unit.pre[0][0] == sys.executable and unit.restart == "always"


def test_signed_p2p_unit_keeps_legacy_worker_and_hub_argv(tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_dir = home / ".nakshatra"
    env_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    # Exact release-6ea54bf assignment shapes: the p2p opt-in supplied the relay and the
    # worker env supplied only its worker inbound arguments. The signed unit supplied nakd.
    (env_dir / "p2p.env").write_text("P2P_RELAYS=legacy-relay\n")
    worker_args = ["-inbound=127.0.0.1:5530", "-require-direct-activation"]
    (env_dir / "worker.env").write_text(
        'NAKSHATRA_P2P_WORKER_ARGS="-inbound=127.0.0.1:5530 '
        '-require-direct-activation"\n'
    )
    assert _effective_p2p_argv(tmp_path, home) == _legacy_p2p_argv(
        tmp_path, home, "legacy-relay", worker_args
    )

    # A hub that never had worker.env still gets precisely the old signed unit's nakd target.
    (env_dir / "worker.env").unlink()
    (env_dir / "p2p.env").write_text("P2P_RELAYS=hub-relay\n")
    assert _effective_p2p_argv(tmp_path, home) == _legacy_p2p_argv(
        tmp_path, home, "hub-relay"
    )


def test_gateway_created_consumer_effective_p2p_argv_has_no_inbound(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_p2p_installer", lambda *_args: None)

    nak._enable_p2p_service(object(), owner_token="ab" * 32)

    values = supervisor._env_file(home / ".nakshatra" / "p2p.env")
    assert values["NAKSHATRA_P2P_NAKD_INBOUND"] == ""
    effective = _effective_p2p_argv(tmp_path, home)
    assert "-listen" in effective and "/ip4/0.0.0.0/tcp/0" in effective
    assert "-quic" in effective
    assert not any(arg.startswith(("-nakd-inbound", "-inbound")) for arg in effective)


def test_explicit_p2p_on_promotes_gateway_consumer_to_legacy_nakd_inbound(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    token = "ab" * 32
    monkeypatch.setenv("HOME", str(home))
    calls = []
    monkeypatch.setattr(
        nak, "_p2p_installer", lambda _sp, action: calls.append(action))

    nak._enable_p2p_service(object(), owner_token=token)
    p2p_env = home / ".nakshatra" / "p2p.env"
    before = p2p_env.read_bytes()
    expected = before.replace(b'NAKSHATRA_P2P_NAKD_INBOUND=""\n', b"")
    owner = home / ".nakshatra" / "gateway" / "p2p-profile-created"
    owner.parent.mkdir(parents=True)
    owner.write_text(f"{token}\n")
    net_dir = home / ".nakshatra" / "net"
    net_dir.mkdir()
    monkeypatch.setenv("NAK_NET_DIR", str(net_dir))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs:
        type("Result", (), {"returncode": 0})())

    calls.clear()
    assert nak.main(["p2p", "on"]) == 0

    assert p2p_env.read_bytes() == expected
    assert not owner.exists()
    assert calls == ["enable", "restart"]
    effective = _effective_p2p_argv(tmp_path, home)
    assert "-nakd-inbound=127.0.0.1:51830" in effective


def test_host_worker_on_promotes_gateway_consumer_and_gateway_off_preserves_p2p(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    token = "cd" * 32
    monkeypatch.setenv("HOME", str(home))
    p2p_calls = []
    monkeypatch.setattr(
        nak, "_p2p_installer", lambda _sp, action: p2p_calls.append(action))
    monkeypatch.setattr(nak, "_worker_installer", lambda *_args: None)
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)

    nak._enable_p2p_service(object(), owner_token=token)
    owner = home / ".nakshatra" / "gateway" / "p2p-profile-created"
    owner.parent.mkdir(parents=True)
    owner.write_text(f"{token}\n")
    assert nak.main([
        "worker", "on", "--model-id", "qwen",
        "--package-url", "https://models.test/qwen", "--package-pubkey", "cd" * 32,
        "--pillar-url", "http://localhost", "--listen-address", "0.0.0.0",
        "--layers", "0-16", "--port", "5530",
    ]) == 0

    p2p_env = home / ".nakshatra" / "p2p.env"
    assert "NAKSHATRA_P2P_NAKD_INBOUND=" not in p2p_env.read_text()
    assert not owner.exists()
    effective = _effective_p2p_argv(tmp_path, home)
    assert "-nakd-inbound=127.0.0.1:51830" in effective
    assert "-inbound=127.0.0.1:5530" in effective
    assert p2p_calls == ["enable", "restart"]

    nak._gateway_off(object())

    assert p2p_env.exists()
    assert p2p_calls == ["enable", "restart"]


def test_fresh_worker_effective_p2p_argv_matches_legacy_worker(tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_dir = home / ".nakshatra"
    env_dir.mkdir(parents=True)
    (env_dir / "p2p.env").write_text("P2P_RELAYS=legacy-relay\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_worker_installer", lambda *_args: None)
    monkeypatch.setattr(nak, "_p2p_installer", lambda *_args: None)
    args = SimpleNamespace(
        model_id="qwen", package_url="https://models.test/qwen", package_pubkey="cd" * 32,
        pillar_url="http://localhost", pillar_spki=None, trusted_pillar_nets="10.42.0.0/24",
        listen_address="0.0.0.0", layers="0-16", port=5530, idle_unload_s=600.0,
        release_grace_s=30.0, warm_directives="on", mode="first",
    )

    nak._worker_enable(object(), args)

    worker_args = ["-inbound=127.0.0.1:5530", "-require-direct-activation"]
    assert _effective_p2p_argv(tmp_path, home) == _legacy_p2p_argv(
        tmp_path, home, "legacy-relay", worker_args
    )


def test_container_env_keeps_worker_private_without_rewriting_gateway_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_dir = home / ".nakshatra"
    env_dir.mkdir(parents=True)
    (env_dir / "p2p.env").write_text("P2P_RELAYS=test\n")
    (env_dir / "worker.env").write_text("PORT=50051\nLISTEN=0.0.0.0\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_CONTAINER_WORKER", "1")
    monkeypatch.setenv("NAK_CONTAINER_P2P_PORT", "4001")
    instance = supervisor.Supervisor(tmp_path / "prefix", tmp_path / "units", tmp_path / "sock",
                                     refresh=False, control=False)
    instance.reconcile_container_env()
    assert not (env_dir / "unconscious.env").exists()
    worker_env = supervisor._env_file(env_dir / "worker.env")
    assert worker_env["LISTEN"] == "127.0.0.1"
    assert worker_env["NAKSHATRA_P2P_WORKER_ARGS"] == (
        "-listen /ip4/0.0.0.0/tcp/4001 -inbound=127.0.0.1:50051 "
        "-require-direct-activation"
    )


def test_container_consumer_effective_p2p_command_listens_but_forwards_nothing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_dir = home / ".nakshatra"
    env_dir.mkdir(parents=True)
    p2p_env = env_dir / "p2p.env"
    p2p_env.write_text(
        'P2P_RELAYS=test\nNAKSHATRA_P2P_WORKER_ARGS="-listen /ip4/0.0.0.0/tcp/4001"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NAK_CONTAINER_WORKER", raising=False)
    instance = supervisor.Supervisor(tmp_path / "prefix", tmp_path / "units", tmp_path / "sock",
                                     refresh=False, control=False)

    instance.reconcile_container_env()

    assert supervisor._env_file(p2p_env) == {
        "P2P_RELAYS": "test", "NAKSHATRA_P2P_NAKD_INBOUND": ""
    }
    assert "0.0.0.0" not in p2p_env.read_text()
    effective = _effective_p2p_argv(tmp_path, home)
    assert "-listen" in effective and "/ip4/0.0.0.0/tcp/0" in effective
    assert "-quic" in effective
    assert not any(arg.startswith(("-nakd-inbound", "-inbound")) for arg in effective)


def test_published_container_consumer_uses_fixed_tcp_and_quic_port_without_inbound(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_dir = home / ".nakshatra"
    env_dir.mkdir(parents=True)
    p2p_env = env_dir / "p2p.env"
    p2p_env.write_text("P2P_RELAYS=test\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NAK_CONTAINER_WORKER", raising=False)
    monkeypatch.setenv("NAK_CONTAINER_CONSUMER_P2P_PORT", "24001")
    instance = supervisor.Supervisor(tmp_path / "prefix", tmp_path / "units", tmp_path / "sock",
                                     refresh=False, control=False)

    instance.reconcile_container_env()

    values = supervisor._env_file(p2p_env)
    assert values == {
        "P2P_RELAYS": "test",
        "NAKSHATRA_P2P_NAKD_INBOUND": "",
        "NAKSHATRA_P2P_WORKER_ARGS": "-listen /ip4/0.0.0.0/tcp/24001",
    }
    effective = _effective_p2p_argv(tmp_path, home)
    assert effective[-2:] == ["-listen", "/ip4/0.0.0.0/tcp/24001"]
    assert "-quic" in effective  # derives /ip4/0.0.0.0/udp/24001/quic-v1
    assert not any(arg.startswith(("-nakd-inbound", "-inbound")) for arg in effective)


def test_supervisor_reload_refreshes_environment_file_without_starting_service(tmp_path):
    units = tmp_path / "units"
    units.mkdir()
    env_file = tmp_path / "worker.env"
    env_file.write_text("PORT=50051\n")
    unit_file = units / "worker.service"
    unit_file.write_text(
        supervisor.UNIT_MARKER + "\n"
        "[Service]\n"
        f"EnvironmentFile=-{env_file}\n"
        f"ExecStart={sys.executable} -c 'raise SystemExit(0)' --port=${{PORT}}\n"
        "Restart=on-failure\n"
    )
    instance = supervisor.Supervisor(tmp_path / "prefix", units, tmp_path / "sock", refresh=False, control=False)
    instance.load_units(["worker"])
    assert instance.services["worker"].unit.argv[-1] == "--port=50051"
    assert instance.services["worker"].process is None
    env_file.write_text("PORT=50052\n")
    instance.load_units(["worker"])
    assert instance.services["worker"].unit.argv[-1] == "--port=50052"
    assert instance.services["worker"].process is None


def test_supervisor_restarts_with_backoff_and_stops_child_on_sigterm(tmp_path):
    counter = tmp_path / "counter"
    timeline = tmp_path / "timeline"
    terminated = tmp_path / "terminated"
    fake = tmp_path / "fake_service.py"
    fake.write_text(
        "import os, signal, time\n"
        "from pathlib import Path\n"
        "counter=Path(os.environ['COUNTER']); timeline=Path(os.environ['TIMELINE'])\n"
        "count=int(counter.read_text())+1 if counter.exists() else 1\n"
        "counter.write_text(str(count))\n"
        "with timeline.open('a') as out: out.write(str(time.monotonic())+'\\n')\n"
        "if count < 3: raise SystemExit(7)\n"
        "def stop(*_): Path(os.environ['TERMINATED']).write_text('yes'); raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "while True: time.sleep(0.05)\n"
    )
    units = tmp_path / "units"
    units.mkdir()
    command = shlex.join([sys.executable, str(fake)])
    (units / "fake.service").write_text(
        supervisor.UNIT_MARKER + "\n"
        "[Service]\n"
        f"Environment=COUNTER={counter}\n"
        f"Environment=TIMELINE={timeline}\n"
        f"Environment=TERMINATED={terminated}\n"
        f"ExecStart={command}\n"
        "Restart=on-failure\n"
    )
    sock = tmp_path / "supervisor.sock"
    prefix = tmp_path / "prefix"
    current = prefix / "current"
    current.mkdir(parents=True)
    installer = current / "install.py"
    installer.write_text("import json\nprint(json.dumps(['fake']))\n")
    (current / "manifest.json").write_text('{"installer":"install.py"}')
    env = dict(os.environ, NAK_SUPERVISOR_BACKOFF_BASE_S="0.05", NAK_SUPERVISOR_BACKOFF_CAP_S="0.1",
               XDG_RUNTIME_DIR=str(tmp_path / "runtime"))
    process = subprocess.Popen(
        [sys.executable, str(CONTAINER / "supervisor.py"),
         "--no-control", "--update-interval", "0", "--unit-dir", str(units), "--socket", str(sock),
         "--prefix", str(prefix)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if counter.exists() and counter.read_text() == "3":
                break
            time.sleep(0.03)
        assert counter.read_text() == "3"
        stamps = [float(value) for value in timeline.read_text().splitlines()]
        assert stamps[1] - stamps[0] >= 0.035
        assert stamps[2] - stamps[1] >= 0.07
        process.terminate()
        assert process.wait(timeout=5) == 0
        assert terminated.read_text() == "yes"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_supervisor_uses_only_installer_names_with_managed_marker(tmp_path, capsys):
    units = tmp_path / "units"
    units.mkdir()
    command = shlex.join([sys.executable, "-c", "raise SystemExit(0)"])
    managed = supervisor.UNIT_MARKER + f"\n[Service]\nExecStart={command}\n"
    (units / "good.service").write_text(managed)
    (units / "listed-but-unmarked.service").write_text(f"[Service]\nExecStart={command}\n")
    (units / "injected.service").write_text(managed)
    (units / "nak-update.service").write_text(managed)
    instance = supervisor.Supervisor(tmp_path / "prefix", units, tmp_path / "sock",
                                     refresh=False, control=False)
    instance.load_units(["good", "listed-but-unmarked"])
    assert set(instance.services) == {"good"}
    output = capsys.readouterr().out
    assert "ignoring unit not returned by signed installer: injected.service" in output
    assert "ignoring unit not returned by signed installer: nak-update.service" in output
    assert "ignoring installer-listed unit without managed marker: listed-but-unmarked.service" in output
    instance.begin_update()
    assert instance.updating is False

    (units / "good.service").unlink()
    instance.reload(generate=False)
    assert instance.services == {}


def test_daemon_reload_regenerates_the_installer_allowlist(tmp_path, monkeypatch):
    units = tmp_path / "units"
    units.mkdir()
    command = shlex.join([sys.executable, "-c", "raise SystemExit(0)"])
    for name in ("old", "new"):
        (units / f"{name}.service").write_text(
            supervisor.UNIT_MARKER + f"\n[Service]\nExecStart={command}\n"
        )
    instance = supervisor.Supervisor(tmp_path / "prefix", units, tmp_path / "sock",
                                     refresh=True, control=False)
    instance.load_units(["old"])
    monkeypatch.setattr(instance, "generate_units", lambda: ["new"])
    assert instance.handle({"action": "reload"}) == {"ok": True}
    assert set(instance.services) == {"new"}


def test_cpu_image_smoke_when_docker_is_usable_without_sudo(tmp_path):
    if not shutil.which("docker"):
        pytest.skip("docker is not installed")
    available = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    if available.returncode:
        pytest.skip("docker info is not usable by this user")
    suffix = f"{os.getpid()}-{tmp_path.name}".lower().replace("_", "-")
    image = f"nakshatra-node-smoke:{suffix}"
    container = f"nakshatra-node-smoke-{suffix}"
    volume = f"nakshatra-node-smoke-{suffix}"
    try:
        subprocess.run(
            ["docker", "build", "-t", image, "-f", str(CONTAINER / "Containerfile"), str(CONTAINER)],
            check=True,
        )
        subprocess.run(
            ["docker", "run", "-d", "--name", container, "--network", "none", "--read-only",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
             "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777", "--volume", f"{volume}:/state", image],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(["docker", "exec", container, "nak", "--help"], check=True)
        subprocess.run(["docker", "exec", container, "nak-container-control", "status", "--quiet"], check=True)
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)
        subprocess.run(["docker", "volume", "rm", "-f", volume], capture_output=True, check=False)
        subprocess.run(["docker", "image", "rm", "-f", image], capture_output=True, check=False)
