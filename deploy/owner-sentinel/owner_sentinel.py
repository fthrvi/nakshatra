#!/usr/bin/env python3
"""Owner-side lease sentinel for Linux desktops.

This process deliberately has no fail-open path.  If observing local owner state or
sending a heartbeat fails, that heartbeat is omitted and the worker's lease expires.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


DEFAULT_INTERVAL_SECONDS = 2.0
DEFAULT_GPU_MEMORY_THRESHOLD_MIB = 1024
DEFAULT_AMD_BUSY_THRESHOLD_PERCENT = 30
DEFAULT_HTTP_TIMEOUT_SECONDS = 1.0
WORKER_MARKERS = (
    "llama-nakshatra-worker",
    "nakshatra-worker",
    "scripts/worker.py",
    "/worker.py",
)


@dataclass(frozen=True)
class Decision:
    clear: bool
    reason: str


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    name: str
    command: str = ""


@dataclass(frozen=True)
class Settings:
    port: int
    token: str
    pause_file: Path
    process_list: Path
    gpu_memory_threshold_mib: int = DEFAULT_GPU_MEMORY_THRESHOLD_MIB
    amd_busy_threshold_percent: int = DEFAULT_AMD_BUSY_THRESHOLD_PERCENT
    proc_root: Path = Path("/proc")
    drm_sysfs_root: Path = Path("/sys/class/drm")
    http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS

    @property
    def lease_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/lease"


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ValueError("invalid owner lease environment line")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _bounded_int(value: str, name: str, minimum: int, maximum: int) -> int:
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} is outside its allowed range")
    return parsed


def load_settings(
    env_file: Path,
    *,
    pause_file: Path | None = None,
    process_list: Path | None = None,
    require_credentials: bool = True,
) -> Settings:
    env = read_env_file(env_file)
    token = env.get("OWNER_LEASE_TOKEN", "")
    port_text = env.get("OWNER_LEASE_PORT", "")
    if require_credentials and (not token or not port_text):
        raise ValueError("OWNER_LEASE_PORT and OWNER_LEASE_TOKEN are required")
    port = _bounded_int(port_text or "5571", "OWNER_LEASE_PORT", 1, 65535)
    memory_threshold = _bounded_int(
        env.get("OWNER_GPU_MEMORY_THRESHOLD_MIB", str(DEFAULT_GPU_MEMORY_THRESHOLD_MIB)),
        "OWNER_GPU_MEMORY_THRESHOLD_MIB",
        0,
        1_048_576,
    )
    amd_threshold = _bounded_int(
        env.get("OWNER_AMD_BUSY_THRESHOLD_PERCENT", str(DEFAULT_AMD_BUSY_THRESHOLD_PERCENT)),
        "OWNER_AMD_BUSY_THRESHOLD_PERCENT",
        0,
        100,
    )
    return Settings(
        port=port,
        token=token,
        pause_file=pause_file or Path.home() / ".config" / "nakshatra" / "pause",
        process_list=process_list or env_file.with_name("games.txt"),
        gpu_memory_threshold_mib=memory_threshold,
        amd_busy_threshold_percent=amd_threshold,
    )


def _normalise_process_name(value: str) -> str:
    name = Path(value.replace("\\", "/")).name.lower()
    return name[:-4] if name.endswith(".exe") else name


def read_process_names(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    names = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            names.add(_normalise_process_name(line))
    return frozenset(names)


def snapshot_processes(proc_root: Path = Path("/proc")) -> tuple[ProcessInfo, ...]:
    processes = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            name = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
            raw_command = (entry / "cmdline").read_bytes()
            command = raw_command.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            # Processes routinely exit or become inaccessible during a /proc scan.
            continue
        processes.append(ProcessInfo(int(entry.name), name, command))
    return tuple(processes)


def is_worker_process(process: ProcessInfo) -> bool:
    searchable = f"{process.name} {process.command}".lower()
    name = _normalise_process_name(process.name)
    return name in {"vmmem", "vmmemwsl"} or name.startswith("wsl") or any(
        marker in searchable for marker in WORKER_MARKERS
    )


def configured_process_is_running(processes: Iterable[ProcessInfo], configured: frozenset[str]) -> bool:
    return any(_normalise_process_name(process.name) in configured for process in processes)


def _parse_used_memory(value: str) -> int:
    match = re.match(r"\s*(\d+)\s*(?:MiB)?\s*$", value, flags=re.IGNORECASE)
    if not match:
        raise ValueError("unexpected nvidia-smi memory value")
    return int(match.group(1))


def nvidia_owner_is_busy(
    processes: Mapping[int, ProcessInfo],
    threshold_mib: int,
    *,
    which: Callable[[str], str | None] | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> bool:
    which = which or shutil.which
    run = run or subprocess.run
    executable = which("nvidia-smi")
    if executable is None:
        return False
    completed = run(
        [
            executable,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    for row in csv.reader(completed.stdout.splitlines()):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("unexpected nvidia-smi output")
        pid = int(row[0].strip())
        reported = ProcessInfo(pid, row[1].strip())
        process = processes.get(pid, reported)
        if _parse_used_memory(row[2]) > threshold_mib and not is_worker_process(process):
            return True
    return False


def _process_uses_drm(process: ProcessInfo, proc_root: Path) -> bool:
    try:
        descriptors = (proc_root / str(process.pid) / "fd").iterdir()
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if target.startswith("/dev/dri/"):
                return True
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return False


def amdgpu_owner_is_busy(
    processes: Iterable[ProcessInfo],
    threshold_percent: int,
    *,
    drm_sysfs_root: Path = Path("/sys/class/drm"),
    proc_root: Path = Path("/proc"),
) -> bool:
    busy_files = sorted(drm_sysfs_root.glob("card*/device/gpu_busy_percent"))
    if not busy_files:
        return False
    if max(int(path.read_text(encoding="ascii").strip()) for path in busy_files) <= threshold_percent:
        return False

    drm_users = [process for process in processes if _process_uses_drm(process, proc_root)]
    # sysfs exposes aggregate utilisation, not per-process attribution.  A busy
    # card is owner-busy unless every observable DRM client is our own worker.
    return not drm_users or any(not is_worker_process(process) for process in drm_users)


def decide(settings: Settings) -> Decision:
    if settings.pause_file.exists():
        return Decision(False, "owner paused")

    processes = snapshot_processes(settings.proc_root)
    configured = read_process_names(settings.process_list)
    if configured_process_is_running(processes, configured):
        return Decision(False, "configured process running")

    by_pid = {process.pid: process for process in processes}
    if nvidia_owner_is_busy(by_pid, settings.gpu_memory_threshold_mib):
        return Decision(False, "owner GPU process")
    if amdgpu_owner_is_busy(
        processes,
        settings.amd_busy_threshold_percent,
        drm_sysfs_root=settings.drm_sysfs_root,
        proc_root=settings.proc_root,
    ):
        return Decision(False, "owner GPU busy")
    return Decision(True, "owner clear")


def send_heartbeat(settings: Settings, decision: Decision, *, at: float | None = None) -> None:
    body = json.dumps(
        {"at": time.time() if at is None else at, "clear": decision.clear, "reason": decision.reason},
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        settings.lease_url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=settings.http_timeout_seconds) as response:
        response.read()


def heartbeat_iteration(
    settings: Settings,
    *,
    decision_fn: Callable[[Settings], Decision] = decide,
    send_fn: Callable[[Settings, Decision], None] = send_heartbeat,
) -> Decision | None:
    try:
        decision = decision_fn(settings)
        send_fn(settings, decision)
        return decision
    except Exception:
        # Silence is the safety signal.  In particular, do not print the token or
        # send a guessed state after a local probe failed.
        return None


def _parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Renew the local Nakshatra owner lease")
    parser.add_argument("--env-file", type=Path, default=here / "owner-lease.env")
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--process-list", type=Path)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.interval <= 0:
        return 2
    try:
        settings = load_settings(
            args.env_file,
            pause_file=args.pause_file,
            process_list=args.process_list,
            require_credentials=not args.dry_run,
        )
    except Exception:
        return 1

    if args.dry_run:
        try:
            decision = decide(settings)
        except Exception:
            return 1
        print(json.dumps({"clear": decision.clear, "reason": decision.reason}, separators=(",", ":")))
        return 0

    while True:
        started = time.monotonic()
        result = heartbeat_iteration(settings)
        if args.once:
            return 0 if result is not None else 1
        # Keep the cadence at --interval regardless of how long the probes took.
        time.sleep(max(0.2, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    sys.exit(main())
