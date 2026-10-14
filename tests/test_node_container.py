from __future__ import annotations

import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

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
    assert not any(" nak p2p on " in f" {line} " for line in lines)
    command = lines[exec_index]
    assert shlex.split(command.removeprefix("+ ")) == [
        "docker", "exec", "-i", "nakshatra-node", "nak", "gateway", "on",
        "--bind", "0.0.0.0", "--port", "11599", "--allow-lan",
        "--pillar-url", "https://pillar.example", "--pillar-spki", "17" * 32,
        "--ledger-url", "https://pillar.example/ledger",
    ]


def test_gateway_on_rejects_endpoint_overrides(tmp_path):
    result = _nakc(tmp_path, "gateway", "on", "--port", "12345")
    assert result.returncode == 2
    assert "fixes the gateway bind and port" in result.stderr


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
        "-listen /ip4/0.0.0.0/tcp/4001 -inbound=127.0.0.1:50051 -require-direct-activation"
    )


def test_container_consumer_p2p_env_has_no_listen_args(tmp_path, monkeypatch):
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

    assert supervisor._env_file(p2p_env) == {"P2P_RELAYS": "test"}
    assert "0.0.0.0" not in p2p_env.read_text()

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
