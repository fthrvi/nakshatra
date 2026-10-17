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
FINGERPRINT_INPUTS = {
    "Containerfile", "supervisor.py", "systemctl.py", "bootstrap.py", "control.py", "dispatch.sh",
}


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


def _systemd_env_file(path: Path) -> dict[str, str]:
    """Small test oracle for the EnvironmentFile forms in the review matrix."""
    try:
        physical = path.read_text().splitlines(keepends=True)
    except OSError:
        return {}
    logical = []
    index = 0
    while index < len(physical):
        line = physical[index].rstrip("\n")
        while line.endswith("\\") and index + 1 < len(physical):
            index += 1
            line = line[:-1] + physical[index].rstrip("\n")
        stripped = line.lstrip()
        if "=" in stripped:
            _name, value = stripped.split("=", 1)
            quote = value[:1] if value[:1] in {"'", '"'} else ""
            while quote and not value.endswith(quote) and index + 1 < len(physical):
                index += 1
                value += "\n" + physical[index].rstrip("\n")
            line = stripped.split("=", 1)[0] + "=" + value
        logical.append(line)
        index += 1
    result = {}
    for raw in logical:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("export ") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.rstrip(" \t\r")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        result[name] = value
    return result


def _systemd_effective_p2p_argv(tmp_path: Path, home: Path) -> list[str]:
    spec = json.loads((ROOT / "release" / "spec.json").read_text())
    service = spec["services"]["nakshatra-p2p"]
    manifest = {"version": "test", "components": [{"name": "nakshatra", "pythonpath": "scripts"}]}
    installer = release_installer.Installer(
        tmp_path / "prefix", systemd=False, make_venv=False, unit_dir=tmp_path / "units")
    unit_path = tmp_path / "p2p-systemd.service"
    unit_path.write_text(installer._unit_text(service, installer._ctx(manifest), manifest))
    environment = {"HOME": str(home), "XDG_RUNTIME_DIR": str(tmp_path / "run")}
    exec_start = ""
    lines = unit_path.read_text().splitlines()
    # Like systemd, EnvironmentFile values override Environment= regardless of directive position;
    # EnvironmentFile entries themselves apply strictly in their listed order.
    for line in lines:
        if line.startswith("Environment="):
            environment.update(supervisor._assignments(line.removeprefix("Environment=")))
    for line in lines:
        if line.startswith("EnvironmentFile="):
            environment.update(_systemd_env_file(Path(line.split("=", 1)[1].removeprefix("-"))))
        elif line.startswith("ExecStart="):
            exec_start = line.removeprefix("ExecStart=")
    return supervisor._command(exec_start, environment)


def _review_assignment(form: str, name: str, value: str) -> str:
    if form == "plain":
        return f"{name}={value}\n"
    if form == "leading-whitespace":
        return f" \t{name}={value}\n"
    if form == "export":
        return f" export {name}={value}\n"
    if form == "single-quoted":
        return f"{name}='{value}'\n"
    if form == "double-quoted":
        return f'{name}="{value}"\n'
    split = max(1, len(value) // 2)
    if form == "unquoted-continuation":
        return f"{name}={value[:split]}\\\n{value[split:]}\n"
    if form == "double-quoted-continuation":
        return f'{name}="{value[:split]}\\\n{value[split:]}"\n'
    if form == "multiline-single-quoted":
        return f"{name}='{value[:split]}\n{value[split:]}'\n"
    if form == "space-before-equals":
        return f"{name} ={value}\n"
    if form == "tab-before-equals":
        return f"{name}\t={value}\n"
    raise AssertionError(form)


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


def _dry_build_argv(script: Path = NAKC, variant: str = "cpu") -> list[str]:
    result = subprocess.run(
        [str(script), "start"],
        env=dict(
            os.environ,
            NAKC_DRY_RUN="1",
            NAKC_ENGINE="true",
            NAKC_VARIANT=variant,
            NAKC_CONFIG_DIR=str(script.parent / ".test-config"),
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("+ true build "))
    return shlex.split(line.removeprefix("+ "))


def _image_fingerprint(script: Path = NAKC, variant: str = "cpu") -> str:
    argv = _dry_build_argv(script, variant)
    label = argv[argv.index("--label") + 1]
    return label.removeprefix("io.nakshatra.image-inputs=")


def _freshness_engine(
        tmp_path: Path, *, image_id: str | None = "sha256:old", label: str | None = None,
        container_id: str | None = None, next_id: str = "sha256:new",
        fail_build: bool = False, fail_runs: int = 0) -> tuple[Path, Path, dict[str, str]]:
    state = tmp_path / "engine-state"
    state.mkdir()
    if image_id is not None:
        (state / "image_id").write_text(image_id + "\n")
    if label is not None:
        (state / "image_label").write_text(label + "\n")
    if container_id is not None:
        (state / "container_id").write_text(container_id + "\n")
        (state / "running").touch()
    (state / "next_id").write_text(next_id + "\n")
    (state / "fail_build").write_text("1\n" if fail_build else "0\n")
    (state / "fail_runs").write_text(f"{fail_runs}\n")
    log = tmp_path / "engine.log"
    engine = tmp_path / "fake-image-engine"
    engine.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "s=$NAKC_TEST_STATE\n"
        "printf '%s\\n' \"$*\" >>\"$NAKC_TEST_LOG\"\n"
        "case \"$1 $2\" in\n"
        "  'image inspect')\n"
        "    if [ \"${3:-}\" = -f ]; then fmt=$4; ref=$5; else fmt=; ref=$3; fi\n"
        "    suffix=\n"
        "    case \"$ref\" in *-prev) suffix=_prev;; esac\n"
        "    [ -f \"$s/image_id$suffix\" ] || exit 1\n"
        "    case \"$fmt\" in\n"
        "      *Labels*) [ -f \"$s/image_label$suffix\" ] && cat \"$s/image_label$suffix\" || true;;\n"
        "      *Id*) cat \"$s/image_id$suffix\";;\n"
        "    esac\n"
        "    exit 0;;\n"
        "  'container inspect') [ -f \"$s/running\" ]; exit $?;;\n"
        "esac\n"
        "case \"$1\" in\n"
        "  inspect)\n"
        "    case \"$3\" in\n"
        "      *State.Running*) [ -f \"$s/running\" ] && printf 'true\\n' || printf 'false\\n';;\n"
        "      *Image*) [ -f \"$s/container_id\" ] && cat \"$s/container_id\" || exit 1;;\n"
        "    esac;;\n"
        "  tag)\n"
        "    case \"$2\" in\n"
        "      *-prev) src=_prev;;\n"
        "      *) src=;;\n"
        "    esac\n"
        "    case \"$3\" in\n"
        "      *-prev) dst=_prev;;\n"
        "      *) dst=;;\n"
        "    esac\n"
        "    cp \"$s/image_id$src\" \"$s/image_id$dst\"\n"
        "    if [ -f \"$s/image_label$src\" ]; then\n"
        "      cp \"$s/image_label$src\" \"$s/image_label$dst\"\n"
        "    else\n"
        "      rm -f \"$s/image_label$dst\"\n"
        "    fi;;\n"
        "  build)\n"
        "    [ \"$(cat \"$s/fail_build\")\" = 0 ] || exit 41\n"
        "    label=\n"
        "    while [ \"$#\" -gt 0 ]; do\n"
        "      if [ \"$1\" = --label ]; then label=$2; break; fi\n"
        "      shift\n"
        "    done\n"
        "    cp \"$s/next_id\" \"$s/image_id\"\n"
        "    printf '%s\\n' \"${label#io.nakshatra.image-inputs=}\" >\"$s/image_label\";;\n"
        "  rm) rm -f \"$s/running\" \"$s/container_id\";;\n"
        "  run)\n"
        "    failures=$(cat \"$s/fail_runs\")\n"
        "    if [ \"$failures\" -gt 0 ]; then\n"
        "      printf '%s\\n' $((failures - 1)) >\"$s/fail_runs\"\n"
        "      exit 42\n"
        "    fi\n"
        "    cp \"$s/image_id\" \"$s/container_id\"\n"
        "    : >\"$s/running\";;\n"
        "  exec|ps|stop) :;;\n"
        "esac\n"
    )
    engine.chmod(0o755)
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        NAKC_ENGINE=str(engine),
        NAKC_CONFIG_DIR=str(tmp_path / "config"),
        NAKC_TEST_STATE=str(state),
        NAKC_TEST_LOG=str(log),
    )
    return state, log, env


def _logged_run_argv(path: Path) -> list[str]:
    runs = [shlex.split(line) for line in path.read_text().splitlines()
            if line.startswith("run ")]
    assert runs
    return runs[-1]


def _reconcile_logged_container(monkeypatch, instance, argv: list[str]) -> None:
    for name in ("NAK_CONTAINER_WORKER", "NAK_CONTAINER_P2P_PORT",
                 "NAK_CONTAINER_CONSUMER_P2P_PORT"):
        monkeypatch.delenv(name, raising=False)
    for index, item in enumerate(argv):
        if item == "--env":
            name, value = argv[index + 1].split("=", 1)
            if name.startswith("NAK_CONTAINER_"):
                monkeypatch.setenv(name, value)
    instance.reconcile_container_env()


def test_containerfile_is_nonroot_pinned_and_does_not_copy_secrets():
    body = (CONTAINER / "Containerfile").read_text()
    assert re.search(r"^ARG BASE_IMAGE=(?![^\n]*(?:latest|edge)\b)[^\n]+:[^\n]+$", body, re.MULTILINE)
    assert re.search(r"^USER 11600:11600$", body, re.MULTILINE)
    assert not re.search(r"^ADD\s+https?://", body, re.MULTILINE)
    assert not re.search(r"COPY[^\n]*(?:\.env|\.key|id_rsa|credentials|\.nakshatra|\.sthambha)", body, re.I)
    assert "sudo" not in body
    assert 'ENTRYPOINT ["/usr/bin/tini"' in body


def test_containerfile_copy_sources_are_all_fingerprint_inputs():
    body = (CONTAINER / "Containerfile").read_text().replace("\\\n", " ")
    copied = set()
    for line in body.splitlines():
        if not re.match(r"^\s*COPY\s", line, re.I):
            continue
        payload = line.split(None, 1)[1].strip()
        while payload.startswith("--"):
            _option, payload = payload.split(None, 1)
        if payload.startswith("["):
            operands = json.loads(payload)
        else:
            operands = shlex.split(payload)
        copied.update(operands[:-1])
    assert copied <= FINGERPRINT_INPUTS


@pytest.mark.parametrize("name", sorted(FINGERPRINT_INPUTS))
def test_image_fingerprint_changes_when_any_input_byte_changes(tmp_path, name):
    copied = tmp_path / "container"
    shutil.copytree(CONTAINER, copied)
    baseline = _image_fingerprint(copied / "nakc")
    path = copied / name
    path.write_bytes(path.read_bytes() + b"\n# fingerprint mutation\n")
    assert _image_fingerprint(copied / "nakc") != baseline


def test_image_fingerprint_changes_with_variant_and_base_image(tmp_path):
    assert _image_fingerprint(variant="nvidia") != _image_fingerprint(variant="cpu")
    copied = tmp_path / "container"
    shutil.copytree(CONTAINER, copied)
    script = copied / "nakc"
    baseline = _image_fingerprint(script)
    script.write_text(script.read_text().replace(
        "cpu) printf '%s\\n' 'debian:12.12-slim'",
        "cpu) printf '%s\\n' 'debian:12.13-slim'",
    ))
    assert _image_fingerprint(script) != baseline


def test_missing_image_builds_with_fingerprint_label(tmp_path):
    state, log, env = _freshness_engine(tmp_path, image_id=None)
    result = subprocess.run(
        [str(NAKC), "start"], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    lines = log.read_text().splitlines()
    build = next(shlex.split(line) for line in lines if line.startswith("build "))
    assert build[build.index("--label") + 1] == (
        f"io.nakshatra.image-inputs={_image_fingerprint()}"
    )
    assert not any(line.startswith("tag ") for line in lines)
    assert (state / "image_label").read_text().strip() == _image_fingerprint()


def test_dry_run_prints_labeled_build_without_executing_engine(tmp_path):
    engine_log = tmp_path / "engine-was-called"
    engine = tmp_path / "must-not-run-engine"
    engine.write_text(
        "#!/bin/sh\n"
        "printf called >\"$NAKC_TEST_LOG\"\n"
        "exit 99\n"
    )
    engine.chmod(0o755)
    result = subprocess.run(
        [str(NAKC), "start"],
        env=dict(
            os.environ,
            HOME=str(tmp_path / "home"),
            NAKC_DRY_RUN="1",
            NAKC_ENGINE=str(engine),
            NAKC_CONFIG_DIR=str(tmp_path / "config"),
            NAKC_TEST_LOG=str(engine_log),
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not engine_log.exists()
    build = shlex.split(result.stdout.splitlines()[0].removeprefix("+ "))
    assert build[0:2] == [str(engine), "build"]
    assert build[build.index("--label") + 1] == (
        f"io.nakshatra.image-inputs={_image_fingerprint()}"
    )


def test_current_image_label_does_not_build(tmp_path):
    _state, log, env = _freshness_engine(tmp_path, label=_image_fingerprint())
    result = subprocess.run(
        [str(NAKC), "start"], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    lines = log.read_text().splitlines()
    assert not any(line.startswith(("build ", "tag ")) for line in lines)


def test_stale_image_is_preserved_as_prev_before_build(tmp_path):
    state, log, env = _freshness_engine(tmp_path, label="stale")
    result = subprocess.run(
        [str(NAKC), "start"], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    actions = [line.split()[0] for line in log.read_text().splitlines()]
    assert actions.index("tag") < actions.index("build")
    assert (state / "image_id_prev").read_text().strip() == "sha256:old"
    assert (state / "image_id").read_text().strip() == "sha256:new"


def test_failed_image_build_keeps_old_tag_and_running_container(tmp_path):
    state, log, env = _freshness_engine(
        tmp_path, label="stale", container_id="sha256:old", fail_build=True)
    result = subprocess.run(
        [str(NAKC), "start"], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "image build failed" in result.stderr
    assert (state / "image_id").read_text().strip() == "sha256:old"
    assert (state / "image_id_prev").read_text().strip() == "sha256:old"
    assert (state / "container_id").read_text().strip() == "sha256:old"
    lines = log.read_text().splitlines()
    assert not any(line.startswith(("rm ", "run ")) for line in lines)


def test_update_rebuilds_stale_image_recreates_with_stored_args_then_updates(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "worker-port").write_text("50051\n")
    (config / "p2p-port").write_text("24001\n")
    _state, log, env = _freshness_engine(
        tmp_path, label="stale", container_id="sha256:old")
    result = subprocess.run(
        [str(NAKC), "update"], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    lines = log.read_text().splitlines()
    build_index = next(i for i, line in enumerate(lines) if line.startswith("build "))
    rm_index = next(i for i, line in enumerate(lines) if line.startswith("rm "))
    run_index = next(i for i, line in enumerate(lines) if line.startswith("run "))
    update_index = next(i for i, line in enumerate(lines) if line.endswith(
        "exec nakshatra-node nak-container-control update"))
    assert build_index < rm_index < run_index < update_index
    update_run = shlex.split(lines[run_index])
    assert "NAK_CONTAINER_WORKER=1" in update_run
    assert "0.0.0.0:24001:24001/tcp" in update_run
    assert "image rebuilt sha256:old->sha256:new, container recreated" in result.stdout

    log.write_text("")
    started = subprocess.run(
        [str(NAKC), "start"], env=env, text=True, capture_output=True, check=False)
    assert started.returncode == 0, started.stderr
    start_run = next(shlex.split(line) for line in log.read_text().splitlines()
                     if line.startswith("run "))
    assert update_run == start_run


def test_update_recreates_old_container_without_build(tmp_path):
    _state, log, env = _freshness_engine(
        tmp_path, image_id="sha256:new", label=_image_fingerprint(),
        container_id="sha256:old")
    result = subprocess.run(
        [str(NAKC), "update"], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    lines = log.read_text().splitlines()
    assert not any(line.startswith("build ") for line in lines)
    assert any(line.startswith("rm ") for line in lines)
    assert any(line.startswith("run ") for line in lines)
    assert result.stdout == "image current, container recreated\n"


def test_update_with_current_image_and_container_only_runs_control_update(tmp_path):
    _state, log, env = _freshness_engine(
        tmp_path, image_id="sha256:new", label=_image_fingerprint(),
        container_id="sha256:new")
    result = subprocess.run(
        [str(NAKC), "update"], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    lines = log.read_text().splitlines()
    assert not any(line.startswith(("build ", "rm ", "run ")) for line in lines)
    assert lines[-1].endswith("exec nakshatra-node nak-container-control update")
    assert result.stdout == "image current\n"


def test_update_recreate_failure_restores_prev_and_restarts_then_fails(tmp_path):
    state, log, env = _freshness_engine(
        tmp_path, label="stale", container_id="sha256:old", fail_runs=1)
    result = subprocess.run(
        [str(NAKC), "update"], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "attempting image rollback" in result.stderr
    assert "restored previous image generation" in result.stderr
    assert (state / "image_id").read_text().strip() == "sha256:old"
    assert (state / "container_id").read_text().strip() == "sha256:old"
    lines = log.read_text().splitlines()
    assert sum(line.startswith("run ") for line in lines) == 2
    assert any(line == "tag nakshatra-node:cpu-prev nakshatra-node:cpu" for line in lines)
    assert not any(line.endswith("nak-container-control update") for line in lines)


def test_custom_image_without_manage_opt_in_is_never_rebuilt(tmp_path):
    _state, log, env = _freshness_engine(tmp_path, label=None)
    env["NAKC_IMAGE"] = "registry.example/nakshatra-node:official"
    result = subprocess.run(
        [str(NAKC), "start"], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert not any(line.startswith(("build ", "tag "))
                   for line in log.read_text().splitlines())


def test_status_reports_freshness_and_running_image_generation(tmp_path):
    _state, _log, env = _freshness_engine(
        tmp_path, image_id="sha256:new", label=_image_fingerprint(),
        container_id="sha256:old")
    result = subprocess.run(
        [str(NAKC), "status"], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "image current; running container does not use current image ID\n"
    )


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
    lines = result.stdout.splitlines(keepends=True)
    build = shlex.split(lines[0].removeprefix("+ "))
    assert build[:2] == ["docker", "build"]
    assert build[build.index("--label") + 1] == (
        f"io.nakshatra.image-inputs={_image_fingerprint()}"
    )
    assert "--pull" in build
    assert "docker" not in build[1:]
    assert "".join(lines[1:]) == (
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


def test_nakc_supervisor_promotes_worker_then_demotes_to_gateway_consumer(
        tmp_path, monkeypatch):
    """Container role changes regenerate and check effective argv at both boundaries."""
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    (state / "p2p.env").write_text("P2P_RELAYS=test\n")
    engine_log = tmp_path / "engine.log"
    engine = tmp_path / "role-engine"
    engine.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >>\"$NAKC_TEST_LOG\"\n"
        "if [ \"$1 $2\" = 'image inspect' ]; then exit 0; fi\n"
        "if [ \"$1 $2\" = 'container inspect' ]; then exit 1; fi\n"
        "if [ \"$1\" = inspect ]; then printf 'false\\n'; exit 0; fi\n"
        "case \" $* \" in\n"
        "  *' nak worker on '*)\n"
        "    mkdir -p \"$NAKC_TEST_HOME/.nakshatra\"\n"
        "    printf '%s\\n' 'PORT=5530' "
        "\"NAKSHATRA_P2P_WORKER_ARGS=\\\"-inbound=127.0.0.1:5530 "
        "-require-direct-activation\\\"\" >\"$NAKC_TEST_HOME/.nakshatra/worker.env\";;\n"
        "  *' nak worker off '*)\n"
        "    rm -f \"$NAKC_TEST_HOME/.nakshatra/worker.env\";;\n"
        "esac\n"
        "exit 0\n"
    )
    engine.chmod(0o755)
    env = dict(
        os.environ, HOME=str(home), NAKC_ENGINE=str(engine),
        NAKC_CONFIG_DIR=str(tmp_path / "config"), NAKC_TEST_LOG=str(engine_log),
        NAKC_TEST_HOME=str(home),
    )
    instance = supervisor.Supervisor(
        tmp_path / "prefix", tmp_path / "units", tmp_path / "sock",
        refresh=False, control=False,
    )
    monkeypatch.setenv("HOME", str(home))

    gateway = subprocess.run(
        [str(NAKC), "gateway", "on"], env=env, text=True, capture_output=True, check=False)
    assert gateway.returncode == 0, gateway.stderr
    _reconcile_logged_container(monkeypatch, instance, _logged_run_argv(engine_log))
    effective = _effective_p2p_argv(tmp_path, home)
    assert not any(arg.startswith(("-nakd-inbound", "-inbound=")) for arg in effective)

    worker = subprocess.run(
        [str(NAKC), "worker", "on", "--port", "5530"],
        env=env, text=True, capture_output=True, check=False)
    assert worker.returncode == 0, worker.stderr
    _reconcile_logged_container(monkeypatch, instance, _logged_run_argv(engine_log))
    effective = _effective_p2p_argv(tmp_path, home)
    assert [arg for arg in effective if arg.startswith("-inbound=")] == [
        "-inbound=127.0.0.1:5530"]

    disabled = subprocess.run(
        [str(NAKC), "worker", "off"], env=env, text=True, capture_output=True, check=False)
    assert disabled.returncode == 0, disabled.stderr
    _reconcile_logged_container(monkeypatch, instance, _logged_run_argv(engine_log))
    effective = _effective_p2p_argv(tmp_path, home)
    assert not any(arg.startswith(("-nakd-inbound", "-inbound=")) for arg in effective)


def test_nakc_failed_worker_stop_retains_worker_container_and_supervisor_inbound(
        tmp_path, monkeypatch):
    """The inner durable recovery opt-in prevents outer consumer demotion."""
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    (state / "p2p.env").write_text("P2P_RELAYS=test\n")
    engine_log = tmp_path / "engine.log"
    engine = tmp_path / "retained-worker-engine"
    engine.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >>\"$NAKC_TEST_LOG\"\n"
        "if [ \"$1 $2\" = 'image inspect' ]; then exit 0; fi\n"
        "if [ \"$1 $2\" = 'container inspect' ]; then exit 1; fi\n"
        "if [ \"$1\" = inspect ]; then printf 'false\\n'; exit 0; fi\n"
        "case \" $* \" in\n"
        "  *' nak worker on '*)\n"
        "    mkdir -p \"$NAKC_TEST_HOME/.nakshatra\"\n"
        "    printf '%s\\n' 'PORT=5530' "
        "\"NAKSHATRA_P2P_WORKER_ARGS=\\\"-inbound=127.0.0.1:5530 "
        "-require-direct-activation\\\"\" >\"$NAKC_TEST_HOME/.nakshatra/worker.env\"\n"
        "    exit 23;;\n"
        "  *' nak worker status '*)\n"
        "    printf '%s\\n' '{\"configured\":true,\"unit\":{\"active\":\"failed\",\"enabled\":\"disabled\"}}'\n"
        "    exit 0;;\n"
        "esac\n"
        "exit 0\n"
    )
    engine.chmod(0o755)
    env = dict(
        os.environ, HOME=str(home), NAKC_ENGINE=str(engine),
        NAKC_CONFIG_DIR=str(tmp_path / "config"), NAKC_TEST_LOG=str(engine_log),
        NAKC_TEST_HOME=str(home),
    )

    result = subprocess.run(
        [str(NAKC), "worker", "on", "--port", "5530"],
        env=env, text=True, capture_output=True, check=False)

    assert result.returncode != 0
    assert "retained worker container for inner recovery state" in result.stderr
    assert (tmp_path / "config" / "worker-port").read_text() == "5530\n"
    assert len([line for line in engine_log.read_text().splitlines()
                if line.startswith("run ")]) == 1

    monkeypatch.setenv("HOME", str(home))
    instance = supervisor.Supervisor(
        tmp_path / "prefix", tmp_path / "units", tmp_path / "sock",
        refresh=False, control=False,
    )
    _reconcile_logged_container(monkeypatch, instance, _logged_run_argv(engine_log))
    effective = _effective_p2p_argv(tmp_path, home)
    assert [arg for arg in effective if arg.startswith("-inbound=")] == [
        "-inbound=127.0.0.1:5530"]


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


@pytest.mark.parametrize(("args", "expected"), [
    (("gateway", "paid", "on"), ["nak", "gateway", "paid", "on"]),
    (("gateway", "paid", "off"), ["nak", "gateway", "paid", "off"]),
    (("worker", "paid", "on", "--ledger-url", "https://ledger.test",
      "--ledger-spki", "ab" * 32),
     ["nak", "worker", "paid", "on", "--ledger-url", "https://ledger.test",
      "--ledger-spki", "ab" * 32]),
    (("worker", "paid", "off"), ["nak", "worker", "paid", "off"]),
])
def test_nakc_paid_commands_pass_through_without_role_reconfiguration(
        tmp_path, args, expected):
    result = _nakc(tmp_path, *args)
    assert result.returncode == 0, result.stderr
    exec_lines = [line for line in result.stdout.splitlines()
                  if line.startswith("+ docker exec ")]
    assert shlex.split(exec_lines[-1].removeprefix("+ ")) == [
        "docker", "exec", "-i", "nakshatra-node", *expected]
    assert not (tmp_path / "config" / "gateway-on").exists()
    assert not (tmp_path / "config" / "worker-port").exists()


def test_nakc_paid_passthrough_quotes_shell_looking_unknown_args_without_outer_effects(
        tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "gateway-on").write_text("1\n")
    (config / "p2p-port").write_text("24001\n")
    marker = tmp_path / "shell-argument-ran"
    before = {path.name: path.read_bytes() for path in config.iterdir()}
    baseline_run = _dry_run_argv(_nakc(tmp_path, "start").stdout)
    hostile = (
        "--unknown", f"$(touch {marker})", "--privileged",
        "--volume=/tmp:/host", "--name=outer-state-changed", f"; touch {marker}",
    )

    result = _nakc(tmp_path, "gateway", "paid", "on", *hostile)

    assert result.returncode == 0, result.stderr
    assert _dry_run_argv(result.stdout) == baseline_run
    exec_lines = [line for line in result.stdout.splitlines()
                  if line.startswith("+ docker exec ")]
    assert shlex.split(exec_lines[-1].removeprefix("+ ")) == [
        "docker", "exec", "-i", "nakshatra-node",
        "nak", "gateway", "paid", "on", *hostile,
    ]
    assert not marker.exists()
    assert {path.name: path.read_bytes() for path in config.iterdir()} == before


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


@pytest.mark.parametrize(
    "worker_args",
    [
        ("--port", "50051", "--port", "4001"),
        ("--port=50051", "--port=4001"),
        ("--port", "50051", "--port", "050052"),
    ],
)
def test_worker_rejects_repeated_ports_before_persisting(tmp_path, worker_args):
    result = _nakc(tmp_path, "worker", "on", *worker_args, NAKC_P2P_PORT="4001")
    assert result.returncode == 2
    assert "accepts only one --port" in result.stderr
    assert not (tmp_path / "config" / "worker-port").exists()
    assert not (tmp_path / "config" / "p2p-port").exists()


@pytest.mark.parametrize(
    ("old_values", "worker_args", "extra_env"),
    [
        ({}, ("--port", "50051"), {"NAKC_P2P_PORT": "50051"}),
        ({"worker-port": "50051", "p2p-port": "4001"}, (),
         {"NAKC_P2P_PORT": "50051"}),
        ({"p2p-port": "50051"}, ("--port", "50051"), {}),
    ],
)
def test_worker_rejects_p2p_port_collision_without_persisting(
        tmp_path, old_values, worker_args, extra_env):
    config = tmp_path / "config"
    config.mkdir()
    for name, value in old_values.items():
        (config / name).write_text(value + "\n")
    before = {path.name: path.read_bytes() for path in config.iterdir()}

    result = _nakc(tmp_path, "worker", "on", *worker_args, **extra_env)

    assert result.returncode == 2
    assert "NAKC_P2P_PORT must not collide with worker port 50051" in result.stderr
    assert {path.name: path.read_bytes() for path in config.iterdir()} == before


def test_gateway_rejects_worker_p2p_port_collision_without_persisting(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "worker-port").write_text("50051\n")
    (config / "p2p-port").write_text("4001\n")
    before = {path.name: path.read_bytes() for path in config.iterdir()}

    result = _nakc(tmp_path, "gateway", "on", NAKC_P2P_PORT="50051")

    assert result.returncode == 2
    assert "NAKC_P2P_PORT must not collide with worker port 50051" in result.stderr
    assert {path.name: path.read_bytes() for path in config.iterdir()} == before


def test_start_rejects_a_noncanonical_host_gateway_port(tmp_path):
    result = _nakc(tmp_path, "start", NAKC_GATEWAY_PORT="011600")
    assert result.returncode == 2
    assert "NAKC_GATEWAY_PORT must be a canonical port" in result.stderr


def test_start_validates_complete_run_argv_before_touching_existing_container(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "worker-port").write_text("50051\n")
    engine_log = tmp_path / "engine.log"
    engine = tmp_path / "logging-container-engine"
    engine.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >>\"$NAKC_TEST_LOG\"\n"
        "exit 0\n"
    )
    engine.chmod(0o755)
    result = subprocess.run(
        [str(NAKC), "start"],
        env=dict(
            os.environ,
            HOME=str(tmp_path / "home"),
            NAKC_ENGINE=str(engine),
            NAKC_CONFIG_DIR=str(config),
            NAKC_P2P_PORT="50051",
            NAKC_TEST_LOG=str(engine_log),
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "NAKC_P2P_PORT must not collide with worker port 50051" in result.stderr
    assert not engine_log.exists()


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
        'PORT=5530\nNAKSHATRA_P2P_WORKER_ARGS="-inbound=127.0.0.1:5530 '
        '-require-direct-activation"\n'
    )
    legacy_worker = _legacy_p2p_argv(
        tmp_path, home, "legacy-relay", worker_args
    )
    assert _effective_p2p_argv(tmp_path, home) == legacy_worker

    monkeypatch.setattr(nak, "_p2p_installer", lambda *_args: None)
    nak._worker_relay(object(), True)
    assert _effective_p2p_argv(tmp_path, home) == [
        *legacy_worker, "-allow-relayed-activation"
    ]

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
    assert "-relay-activation-fallback" in effective
    assert not any(arg.startswith(("-nakd-inbound", "-inbound")) for arg in effective)


@pytest.mark.parametrize("assignment_prefix", [" ", " export "])
def test_explicit_p2p_on_keeps_live_gateway_protected_until_gateway_off(
        tmp_path, monkeypatch, assignment_prefix):
    home = tmp_path / "home"
    token = "ab" * 32
    monkeypatch.setenv("HOME", str(home))
    calls = []
    monkeypatch.setattr(
        nak, "_p2p_installer", lambda _sp, action: calls.append(action))
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: True)
    monkeypatch.setattr(nak, "_worker_unit_enabled", lambda _sp: False)
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)

    nak._enable_p2p_service(object(), owner_token=token)
    p2p_env = home / ".nakshatra" / "p2p.env"
    (home / ".nakshatra" / "p2p-consumer.env").write_text(
        'NAKSHATRA_P2P_NAKD_INBOUND=""\nNAKSHATRA_P2P_WORKER_ARGS=""\n')
    p2p_env.write_text(p2p_env.read_text().replace(
        'NAKSHATRA_P2P_NAKD_INBOUND=""',
        f'{assignment_prefix}NAKSHATRA_P2P_NAKD_INBOUND=""',
    ))
    before = p2p_env.read_bytes()
    expected = before.replace(
        f'{assignment_prefix}NAKSHATRA_P2P_NAKD_INBOUND=""\n'.encode(), b""
    )
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
    assert (home / ".nakshatra" / "p2p-consumer.env").exists()
    assert calls == ["enable", "restart"]
    effective = _effective_p2p_argv(tmp_path, home)
    assert not any(arg.startswith(("-nakd-inbound", "-inbound")) for arg in effective)
    assert "-relay-activation-fallback" not in effective

    nak._gateway_off(object())

    assert not (home / ".nakshatra" / "p2p-consumer.env").exists()
    assert calls == ["enable", "restart", "restart"]
    assert "-nakd-inbound=127.0.0.1:51830" in _effective_p2p_argv(tmp_path, home)


@pytest.mark.parametrize(
    ("nakd_prefix", "worker_prefix"),
    [(" ", " export "), (" export ", " ")],
)
def test_host_worker_on_promotes_gateway_consumer_and_gateway_off_preserves_p2p(
        tmp_path, monkeypatch, nakd_prefix, worker_prefix):
    home = tmp_path / "home"
    token = "cd" * 32
    monkeypatch.setenv("HOME", str(home))
    p2p_calls = []
    monkeypatch.setattr(
        nak, "_p2p_installer", lambda _sp, action: p2p_calls.append(action))
    monkeypatch.setattr(nak, "_worker_installer", lambda *_args: None)
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)

    nak._enable_p2p_service(object(), owner_token=token)
    p2p_env = home / ".nakshatra" / "p2p.env"
    (home / ".nakshatra" / "p2p-consumer.env").write_text(
        'NAKSHATRA_P2P_NAKD_INBOUND=""\nNAKSHATRA_P2P_WORKER_ARGS=""\n')
    p2p_env.write_text(p2p_env.read_text().replace(
        'NAKSHATRA_P2P_NAKD_INBOUND=""',
        f'{nakd_prefix}NAKSHATRA_P2P_NAKD_INBOUND=""',
    ))
    (home / ".nakshatra" / "worker.env").write_text(
        "PORT=5530\n"
        f"{worker_prefix}NAKSHATRA_P2P_WORKER_ARGS=-inbound=127.0.0.1:59998\n"
    )
    owner = home / ".nakshatra" / "gateway" / "p2p-profile-created"
    owner.parent.mkdir(parents=True)
    owner.write_text(f"{token}\n")
    assert nak.main([
        "worker", "on", "--model-id", "qwen",
        "--package-url", "https://models.test/qwen", "--package-pubkey", "cd" * 32,
        "--pillar-url", "http://localhost", "--listen-address", "0.0.0.0",
        "--layers", "0-16", "--port", "5530",
    ]) == 0

    assert "NAKSHATRA_P2P_NAKD_INBOUND=" not in p2p_env.read_text()
    assert not (home / ".nakshatra" / "p2p-consumer.env").exists()
    assert not owner.exists()
    effective = _effective_p2p_argv(tmp_path, home)
    assert "-nakd-inbound=127.0.0.1:51830" in effective
    assert "-inbound=127.0.0.1:5530" in effective
    worker_body = (home / ".nakshatra" / "worker.env").read_text()
    assert "59998" not in worker_body
    assert worker_body.count("NAKSHATRA_P2P_WORKER_ARGS=") == 1
    assert p2p_calls == ["enable", "restart"]

    nak._gateway_off(object())

    assert p2p_env.exists()
    assert p2p_calls == ["enable", "restart"]
    assert _effective_p2p_argv(tmp_path, home) == effective


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


@pytest.mark.parametrize(("transition", "faults"), [
    ("gateway-on", ()),
    ("gateway-on", ("gateway:enable:1",)),
    ("gateway-on", ("gateway:enable:1", "gateway:disable:1")),
    ("gateway-on", ("p2p:restart:1",)),
    ("gateway-restart", ("gateway:restart:1",)),
    ("gateway-restart", ("gateway:restart:1", "gateway:restart:2")),
    ("gateway-off", ()),
    ("gateway-off", ("gateway:disable:1",)),
    ("gateway-off", ("p2p:disable:1",)),
    ("gateway-off", ("p2p:disable:1", "p2p:restart:1")),
    ("gateway-off-operator", ("p2p:restart:1",)),
    ("worker-on", ()),
    ("worker-on", ("worker:enable:1",)),
    ("worker-on", ("worker:enable:1", "worker:disable:1")),
    ("worker-on", ("p2p:restart:1",)),
    ("worker-on", ("p2p:restart:1", "worker:disable:1")),
    ("worker-off", ()),
    ("worker-off", ("worker:disable:1",)),
    ("worker-off", ("p2p:restart:1",)),
    ("p2p-on-takeover", ()),
    ("p2p-on-takeover", ("p2p:enable:1",)),
    ("p2p-on-takeover", ("p2p:enable:1", "p2p:restart:1")),
    ("p2p-on-takeover", ("p2p:restart:1",)),
    ("p2p-on-takeover", ("p2p:restart:1", "p2p:restart:2")),
    ("p2p-off", ()),
    ("p2p-off", ("p2p:disable:1",)),
    ("worker-relay-on", ()),
    ("worker-relay-on", ("p2p:restart:1",)),
    ("gateway-worker-gateway-off", ()),
])
def test_host_transition_matrix_preserves_effective_p2p_invariant(
        tmp_path, monkeypatch, transition, faults):
    """Every host role transition leaves effective sidecar argv safe, including rollback paths."""
    home = tmp_path / "home"
    state = home / ".nakshatra"
    net = state / "net"
    net.mkdir(parents=True)
    (state / "p2p.env").write_text(
        'P2P_RELAYS=test\nNAKSHATRA_P2P_GATEWAY_ARGS="-relay-activation-fallback"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NET_DIR", str(net))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs:
        type("Result", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.setattr(
        nak, "load_release_package_signers", lambda: frozenset({"23" * 32}))

    live = {"gateway": False, "worker": False, "p2p": True}
    if transition in {"gateway-restart", "gateway-off", "gateway-off-operator", "worker-on",
                      "worker-off", "p2p-on-takeover"}:
        live["gateway"] = True
        (state / "gateway.env").write_text("PORT=11600\nOLD=1\n")
        nak._write_p2p_consumer_env()
    if transition in {"worker-off", "p2p-off", "worker-relay-on"}:
        live["worker"] = True
        (state / "worker.env").write_text(
            'PORT=5530\nNAKSHATRA_P2P_WORKER_ARGS="-inbound=127.0.0.1:5530 '
            '-require-direct-activation"\n')
        (state / "p2p-consumer.env").unlink(missing_ok=True)
    if transition == "gateway-on":
        # The reproduction: configured worker, disabled unit. Configuration must not suppress
        # consumer protection.
        (state / "worker.env").write_text("PORT=5530\n")
    if transition in {"gateway-off", "p2p-on-takeover"}:
        token = "ab" * 32
        (state / "p2p.env").write_text(
            f'P2P_RELAYS=test\nNAKSHATRA_P2P_NAKD_INBOUND=""\n'
            f'# nak-gateway-owned: {token}\n')
        owner = state / "gateway" / "p2p-profile-created"
        owner.parent.mkdir(parents=True)
        owner.write_text(token + "\n")

    calls = []
    counts = {}
    fault_set = set(faults)

    def assert_invariant(boundary):
        if not live["p2p"] or not (state / "p2p.env").exists():
            return
        effective = _effective_p2p_argv(tmp_path, home)
        inbound = [arg for arg in effective if arg.startswith("-inbound=")]
        nakd_inbound = [arg for arg in effective if arg.startswith("-nakd-inbound=")]
        fallback = [arg for arg in effective if arg == "-relay-activation-fallback"]
        relay_acceptance = [arg for arg in effective if arg == "-allow-relayed-activation"]
        loaded_envs = [state / "p2p.env", state / "worker.env",
                       state / "p2p-consumer.env"]
        for path in loaded_envs:
            if path.exists():
                body = path.read_text()
                assert "NAKSHATRA_P2P_GATEWAY_ARGS" not in body, (
                    transition, faults, boundary, path, body)
                if path.name != "p2p-consumer.env":
                    assert "-relay-activation-fallback" not in body, (
                        transition, faults, boundary, path, body)
        if live["worker"]:
            assert inbound == ["-inbound=127.0.0.1:5530"], (
                transition, faults, boundary, calls, effective)
        elif live["gateway"] or (state / "gateway.env").exists():
            assert inbound == [] and nakd_inbound == [], (
                transition, faults, boundary, calls, effective)
        else:
            assert effective == _legacy_p2p_argv(tmp_path, home, "test"), (
                transition, faults, boundary, calls, effective)

        # I4: dialer fallback belongs to the active consumer role; worker relay acceptance is an
        # independent, explicit worker opt-in. Both are checked at every installer boundary.
        consumer_values = supervisor._env_file(state / "p2p-consumer.env")
        consumer_role = (
            not live["worker"]
            and (live["gateway"] or (state / "gateway.env").exists())
            and consumer_values.get("NAKSHATRA_P2P_CONSUMER_ARGS")
            == "-relay-activation-fallback"
        )
        assert fallback == (["-relay-activation-fallback"] if consumer_role else []), (
            transition, faults, boundary, calls, effective)
        worker_relay_on = live["worker"] and nak._worker_relay_enabled(state / "worker.env")
        assert relay_acceptance == (["-allow-relayed-activation"] if worker_relay_on else []), (
            transition, faults, boundary, calls, effective)

    def operate(service, action):
        key = (service, action)
        counts[key] = counts.get(key, 0) + 1
        calls.append(f"{service}:{action}:{counts[key]}")
        if action in {"enable", "restart"}:
            # enable may start before its stability check fails; restart may leave either
            # generation live. Model the conservative live outcome.
            live[service] = True
        assert_invariant(calls[-1])
        if calls[-1] in fault_set:
            raise SystemExit(f"injected {calls[-1]} failure")
        if action == "disable":
            live[service] = False
            {"gateway": state / "gateway.env", "worker": state / "worker.env",
             "p2p": state / "p2p.env"}[service].unlink(missing_ok=True)

    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: live["gateway"])
    monkeypatch.setattr(nak, "_worker_unit_enabled", lambda _sp: live["worker"])
    monkeypatch.setattr(nak, "_p2p_unit_enabled", lambda _sp: live["p2p"])
    monkeypatch.setattr(nak, "_gateway_installer", lambda _sp, action: operate("gateway", action))
    monkeypatch.setattr(nak, "_worker_installer", lambda _sp, action: operate("worker", action))
    monkeypatch.setattr(nak, "_p2p_installer", lambda _sp, action: operate("p2p", action))

    gateway_args = SimpleNamespace(
        port=11600, bind="127.0.0.1", allow_lan=False,
        pillar_url="https://pillar.example:7778", pillar_spki="17" * 32,
        ledger_url="https://pillar.example/ledger", models="auto",
        package_url="https://models.example/qwen", package_pubkey="23" * 32,
    )
    worker_args = SimpleNamespace(
        model_id="qwen", package_url="https://models.example/qwen",
        package_pubkey="cd" * 32, pillar_url="http://localhost", pillar_spki=None,
        trusted_pillar_nets="10.42.0.0/24", listen_address="0.0.0.0",
        layers="0-16", port=5530, idle_unload_s=600.0, release_grace_s=30.0,
        warm_directives="on", mode="first",
    )

    def drive():
        if transition in {"gateway-on", "gateway-restart"}:
            nak._gateway_on(object(), gateway_args)
        elif transition in {"gateway-off", "gateway-off-operator"}:
            nak._gateway_off(object())
        elif transition == "worker-on":
            nak._worker_enable(object(), worker_args)
        elif transition == "worker-off":
            nak._worker_disable(object())
        elif transition == "p2p-on-takeover":
            nak.main(["p2p", "on"])
        elif transition == "worker-relay-on":
            nak._worker_relay(object(), True)
        elif transition == "gateway-worker-gateway-off":
            nak._gateway_on(object(), gateway_args)
            nak._worker_enable(object(), worker_args)
            nak._gateway_off(object())
        else:
            nak.main(["p2p", "off"])

    if faults:
        with pytest.raises(SystemExit, match="injected"):
            drive()
    else:
        drive()

    assert_invariant("exit")


def test_container_env_keeps_worker_private_without_rewriting_gateway_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_dir = home / ".nakshatra"
    env_dir.mkdir(parents=True)
    (env_dir / "p2p.env").write_text("P2P_RELAYS=test\n")
    (env_dir / "worker.env").write_text("PORT=50051\nLISTEN=0.0.0.0\n")
    (env_dir / "p2p-consumer.env").write_text(
        'NAKSHATRA_P2P_NAKD_INBOUND=""\nNAKSHATRA_P2P_WORKER_ARGS=""\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_CONTAINER_WORKER", "1")
    monkeypatch.setenv("NAK_CONTAINER_P2P_PORT", "4001")
    instance = supervisor.Supervisor(tmp_path / "prefix", tmp_path / "units", tmp_path / "sock",
                                     refresh=False, control=False)
    instance.reconcile_container_env()
    assert not (env_dir / "unconscious.env").exists()
    assert not (env_dir / "p2p-consumer.env").exists()
    worker_env = supervisor._env_file(env_dir / "worker.env")
    assert worker_env["LISTEN"] == "127.0.0.1"
    assert worker_env["NAKSHATRA_P2P_WORKER_ARGS"] == (
        "-listen /ip4/0.0.0.0/tcp/4001 -inbound=127.0.0.1:50051 "
        "-require-direct-activation"
    )


def test_container_env_rejects_equal_worker_and_p2p_ports_without_rewriting(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_dir = home / ".nakshatra"
    env_dir.mkdir(parents=True)
    p2p_env = env_dir / "p2p.env"
    worker_env = env_dir / "worker.env"
    consumer_env = env_dir / "p2p-consumer.env"
    p2p_env.write_text("P2P_RELAYS=test\nNAKSHATRA_P2P_NAKD_INBOUND=127.0.0.1:51830\n")
    worker_env.write_text("PORT=50051\nLISTEN=0.0.0.0\n")
    consumer_env.write_text(
        'NAKSHATRA_P2P_NAKD_INBOUND=""\nNAKSHATRA_P2P_WORKER_ARGS=""\n')
    before = {path: path.read_bytes() for path in (p2p_env, worker_env, consumer_env)}
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_CONTAINER_WORKER", "1")
    monkeypatch.setenv("NAK_CONTAINER_P2P_PORT", "50051")
    instance = supervisor.Supervisor(tmp_path / "prefix", tmp_path / "units", tmp_path / "sock",
                                     refresh=False, control=False)

    with pytest.raises(ValueError, match="container p2p port must not equal worker PORT 50051"):
        instance.reconcile_container_env()

    assert {path: path.read_bytes() for path in (p2p_env, worker_env, consumer_env)} == before


def test_container_consumer_rejects_noncanonical_port_without_rewriting_any_env(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_dir = home / ".nakshatra"
    env_dir.mkdir(parents=True)
    p2p_env = env_dir / "p2p.env"
    worker_env = env_dir / "worker.env"
    p2p_env.write_text(
        "P2P_RELAYS=test\nNAKSHATRA_P2P_NAKD_INBOUND=-nakd-inbound=127.0.0.1:51830\n"
    )
    worker_env.write_text(
        "PORT=50051\nNAKSHATRA_P2P_WORKER_ARGS=-inbound=127.0.0.1:50051\n"
    )
    before = {path: path.read_bytes() for path in (p2p_env, worker_env)}
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NAK_CONTAINER_WORKER", raising=False)
    monkeypatch.setenv("NAK_CONTAINER_CONSUMER_P2P_PORT", "04001")
    instance = supervisor.Supervisor(tmp_path / "prefix", tmp_path / "units", tmp_path / "sock",
                                     refresh=False, control=False)

    with pytest.raises(ValueError, match="consumer p2p port must be between 1 and 65535"):
        instance.reconcile_container_env()

    assert {path: path.read_bytes() for path in (p2p_env, worker_env)} == before


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
    assert "-relay-activation-fallback" in effective
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

    assert supervisor._env_file(p2p_env) == {
        "P2P_RELAYS": "test", "NAKSHATRA_P2P_NAKD_INBOUND": "",
    }
    assert supervisor._env_file(env_dir / "p2p-consumer.env") == {
        "NAKSHATRA_P2P_NAKD_INBOUND": "",
        "NAKSHATRA_P2P_CONSUMER_ARGS": "-relay-activation-fallback",
        "NAKSHATRA_P2P_WORKER_ARGS": "-listen /ip4/0.0.0.0/tcp/24001",
    }
    effective = _effective_p2p_argv(tmp_path, home)
    assert effective[-2:] == ["-listen", "/ip4/0.0.0.0/tcp/24001"]
    assert "-quic" in effective  # derives /ip4/0.0.0.0/udp/24001/quic-v1
    assert "-relay-activation-fallback" in effective
    assert not any(arg.startswith(("-nakd-inbound", "-inbound")) for arg in effective)


def test_container_consumer_clears_stale_inbound_from_every_loaded_env_file(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_dir = home / ".nakshatra"
    env_dir.mkdir(parents=True)
    (env_dir / "p2p.env").write_text(
        "P2P_RELAYS=test\n"
        " NAKSHATRA_P2P_NAKD_INBOUND=-nakd-inbound=127.0.0.1:51830\n"
        " export NAKSHATRA_P2P_WORKER_ARGS=-inbound=127.0.0.1:51831\n"
        'NAKSHATRA_P2P_GATEWAY_ARGS="-relay-activation-fallback"\n'
    )
    (env_dir / "worker.env").write_text(
        "PORT=50051\n"
        " export NAKSHATRA_P2P_NAKD_INBOUND=-nakd-inbound=127.0.0.1:59999\n"
        " NAKSHATRA_P2P_WORKER_ARGS=-inbound=127.0.0.1:59998\n"
        'NAKSHATRA_P2P_CONSUMER_ARGS="-relay-activation-fallback"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NAK_CONTAINER_WORKER", raising=False)
    monkeypatch.setenv("NAK_CONTAINER_CONSUMER_P2P_PORT", "24001")
    instance = supervisor.Supervisor(tmp_path / "prefix", tmp_path / "units", tmp_path / "sock",
                                     refresh=False, control=False)

    instance.reconcile_container_env()

    assert "51830" not in (env_dir / "p2p.env").read_text()
    assert "51831" not in (env_dir / "p2p.env").read_text()
    assert "59999" not in (env_dir / "worker.env").read_text()
    assert "59998" not in (env_dir / "worker.env").read_text()
    assert "NAKSHATRA_P2P_GATEWAY_ARGS" not in (env_dir / "p2p.env").read_text()
    assert "NAKSHATRA_P2P_CONSUMER_ARGS" not in (env_dir / "worker.env").read_text()
    assert supervisor._env_file(env_dir / "p2p-consumer.env") == {
        "NAKSHATRA_P2P_NAKD_INBOUND": "",
        "NAKSHATRA_P2P_CONSUMER_ARGS": "-relay-activation-fallback",
        "NAKSHATRA_P2P_WORKER_ARGS": "-listen /ip4/0.0.0.0/tcp/24001",
    }
    effective = _effective_p2p_argv(tmp_path, home)
    assert effective[-2:] == ["-listen", "/ip4/0.0.0.0/tcp/24001"]
    assert "-relay-activation-fallback" in effective
    assert not any(arg.startswith(("-nakd-inbound", "-inbound")) for arg in effective)


@pytest.mark.parametrize("env_name", ["p2p.env", "worker.env"])
@pytest.mark.parametrize("form", [
    "plain",
    "leading-whitespace",
    "export",
    "single-quoted",
    "double-quoted",
    "unquoted-continuation",
    "double-quoted-continuation",
    "multiline-single-quoted",
    "space-before-equals",
    "tab-before-equals",
])
def test_consumer_override_wins_over_review_parser_matrix(
        tmp_path, monkeypatch, env_name, form):
    home = tmp_path / "home"
    env_dir = home / ".nakshatra"
    env_dir.mkdir(parents=True)
    (env_dir / "p2p.env").write_text("P2P_RELAYS=test\n")
    (env_dir / "worker.env").write_text("PORT=50051\n")
    target = env_dir / env_name
    target.write_text(
        target.read_text()
        + _review_assignment(
            form, "NAKSHATRA_P2P_NAKD_INBOUND", "-nakd-inbound=127.0.0.1:59999")
        + _review_assignment(
            form, "NAKSHATRA_P2P_WORKER_ARGS",
            "-listen /ip4/0.0.0.0/tcp/59997 -inbound=127.0.0.1:59998",
        )
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NAK_CONTAINER_WORKER", raising=False)
    monkeypatch.setenv("NAK_CONTAINER_CONSUMER_P2P_PORT", "24001")
    instance = supervisor.Supervisor(
        tmp_path / "prefix", tmp_path / "units", tmp_path / "sock",
        refresh=False, control=False,
    )

    instance.reconcile_container_env()

    for effective in (
            _systemd_effective_p2p_argv(tmp_path, home),
            _effective_p2p_argv(tmp_path, home)):
        assert effective[-2:] == ["-listen", "/ip4/0.0.0.0/tcp/24001"]
        assert not any(arg.startswith(("-nakd-inbound", "-inbound")) for arg in effective)
        assert not any("5999" in arg for arg in effective)


@pytest.mark.parametrize("raw", [
    "NAME=value",
    " NAME=value",
    " export NAME=value",
    "NAME =value",
    "NAME\t=value",
    "NAME\r=value",
])
def test_env_assignment_normalizers_have_parser_parity(raw):
    assert supervisor._env_assignment(raw) == nak._env_assignment(raw)
    assert supervisor._env_assignment(raw) == ("NAME", "value")


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
    if os.environ.get("NAKC_TEST_REAL_DOCKER") != "1":
        pytest.skip("real Docker smoke test requires NAKC_TEST_REAL_DOCKER=1")
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
