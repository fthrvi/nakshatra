"""Read-only worker hardware/runtime preflight.

Every operating-system boundary is injectable.  Besides making the probes
best-effort, this keeps the test suite away from the host's real GPUs and node
state.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Mapping


SCHEMA = "nak-worker-preflight/1"
GIB = 1024 ** 3
PROBE_TIMEOUT_S = 2.0


def _run(run, argv: list[str]):
    try:
        result = run(
            argv, capture_output=True, text=True, check=False,
            timeout=PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if result is None or getattr(result, "returncode", 1) != 0:
        return None
    return getattr(result, "stdout", "") or ""


def _integer(value) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _nvidia(run, which) -> tuple[list[dict], bool]:
    if not which("nvidia-smi"):
        return [], False
    output = _run(run, [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in (output or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        total_mib, free_mib = _integer(parts[1]), _integer(parts[2])
        if total_mib is None:
            continue
        rows.append({
            "vendor": "nvidia", "name": parts[0] or "NVIDIA GPU",
            "vram_bytes": total_mib * 1024 ** 2,
            "vram_free_bytes": free_mib * 1024 ** 2 if free_mib is not None else None,
            "source": "nvidia-smi", "status": "detected",
        })
    if not rows:
        rows.append({
            "vendor": "nvidia", "name": None, "vram_bytes": None,
            "vram_free_bytes": None, "source": "nvidia-smi", "status": "unknown",
        })
    return rows, True


def _rocm(run, which) -> tuple[list[dict], bool]:
    if not which("rocm-smi"):
        return [], False
    output = _run(run, ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    rows = []
    try:
        body = json.loads(output or "")
    except (TypeError, ValueError):
        body = {}
    if isinstance(body, dict):
        for card, values in body.items():
            if not isinstance(values, dict):
                continue
            normalized = {str(key).lower(): value for key, value in values.items()}
            total = next((_integer(value) for key, value in normalized.items()
                          if "vram total memory" in key), None)
            used = next((_integer(value) for key, value in normalized.items()
                         if "vram total used memory" in key), None)
            if total is None:
                continue
            name = next((str(value).strip() for key, value in normalized.items()
                         if key in {"card series", "card model"} and str(value).strip()), None)
            rows.append({
                "vendor": "amd", "name": name or str(card), "vram_bytes": total,
                "vram_free_bytes": max(0, total - used) if used is not None else None,
                "source": "rocm-smi", "status": "detected",
            })
    return rows, True


def _amd_sysfs(glob_paths, read_text) -> tuple[list[dict], bool]:
    rows = []
    try:
        paths = list(glob_paths("/sys/class/drm/card*/device/mem_info_vram_total"))
    except OSError:
        return rows, False
    for raw_path in paths:
        path = Path(raw_path)
        try:
            if read_text(path.with_name("vendor")).strip().lower() != "0x1002":
                continue
            total = _integer(read_text(path))
        except OSError:
            total = None
        if total is None:
            continue
        used = None
        try:
            used = _integer(read_text(path.with_name("mem_info_vram_used")))
        except OSError:
            pass
        card = path.parents[1].name if len(path.parents) >= 2 else "AMD GPU"
        rows.append({
            "vendor": "amd", "name": card, "vram_bytes": total,
            "vram_free_bytes": max(0, total - used) if used is not None else None,
            "source": "sysfs", "status": "detected",
        })
    return rows, bool(paths)


def _memory(read_text) -> dict:
    try:
        lines = read_text(Path("/proc/meminfo")).splitlines()
        values = {}
        for line in lines:
            name, value = line.split(":", 1)
            fields = value.split()
            if fields:
                values[name] = int(fields[0]) * 1024
        return {"total_bytes": values.get("MemTotal"),
                "available_bytes": values.get("MemAvailable"), "source": "procfs"}
    except (OSError, ValueError):
        return {"total_bytes": None, "available_bytes": None, "source": "unknown"}


def _disk(state_dir: Path, disk_usage) -> dict:
    target = state_dir
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        usage = disk_usage(target)
        return {"state_dir": str(state_dir), "total_bytes": int(usage.total),
                "free_bytes": int(usage.free), "source": "filesystem"}
    except (OSError, ValueError):
        return {"state_dir": str(state_dir), "total_bytes": None,
                "free_bytes": None, "source": "unknown"}


def _runtime(which, glob_paths, find_spec) -> dict:
    present = {name: bool(which(name)) for name in (
        "bash", "git", "python3", "cc", "cmake", "curl", "systemctl",
        "nvcc", "hipconfig", "rocminfo", "glslc",
    )}
    try:
        vulkan_icd = bool(list(glob_paths("/usr/share/vulkan/icd.d/*.json")))
    except OSError:
        vulkan_icd = False
    if not present["nvcc"]:
        try:
            present["nvcc"] = bool(list(glob_paths("/usr/local/cuda*/bin/nvcc")))
        except OSError:
            pass
    python_modules = {}
    for module in ("grpc", "google.protobuf", "numpy", "gguf"):
        try:
            python_modules[module] = find_spec(module) is not None
        except (ImportError, AttributeError, ValueError):
            python_modules[module] = False
    required_missing = [name for name in ("bash", "git", "python3", "cc", "systemctl")
                        if not present[name]]
    if not (present["cmake"] or present["curl"]):
        required_missing.append("cmake (or curl for the pinned download)")
    return {
        "commands": present,
        "worker_python_modules": python_modules,
        "vulkan_icd": vulkan_icd,
        "required_missing": required_missing,
        "build_ready": not required_missing,
        "backends": {
            "cuda": present["nvcc"],
            "rocm": present["hipconfig"] or present["rocminfo"],
            "vulkan": present["glslc"] and vulkan_icd,
            "cpu": True,
        },
    }


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / GIB:.1f} GiB"


def _package_verdict(package_id: str, definition: Mapping, accelerators: list[dict],
                     memory: dict, disk: dict, runtime: dict) -> dict:
    requirements = definition.get("requirements", {})
    gpu_need = _integer(requirements.get("gpu_vram_bytes"))
    ram_need = _integer(requirements.get("cpu_ram_bytes"))
    disk_need = _integer(requirements.get("disk_free_bytes"))
    public_requirements = {
        "gpu_vram_bytes": gpu_need, "cpu_ram_bytes": ram_need,
        "disk_free_bytes": disk_need,
    }

    def result(code: str, verdict: str, reason: str, mode: str | None = None) -> dict:
        return {"id": package_id, "verdict_code": code, "verdict": verdict,
                "reason": reason, "mode": mode, "requirements": public_requirements}

    if None in (gpu_need, ram_need, disk_need):
        return result("cannot_serve", "cannot serve", "signed catalog requirements are invalid")
    if not runtime["build_ready"]:
        missing = ", ".join(runtime["required_missing"])
        return result("cannot_serve", f"cannot serve: needs runtime {missing}, you have missing",
                      f"missing required worker runtime: {missing}")
    free_disk = disk.get("free_bytes")
    if free_disk is None or free_disk < disk_need:
        verdict = f"cannot serve: needs {_fmt_bytes(disk_need)} disk, you have {_fmt_bytes(free_disk)}"
        return result("cannot_serve", verdict, "not enough known free space under the worker state directory")

    best_gpu = None
    for accelerator in accelerators:
        capacity = accelerator.get("vram_free_bytes")
        if capacity is None:
            capacity = accelerator.get("vram_bytes")
        if capacity is not None and (best_gpu is None or capacity > best_gpu[0]):
            best_gpu = (capacity, accelerator)
    if best_gpu and best_gpu[0] >= gpu_need:
        accelerator = best_gpu[1]
        backend = "cuda" if accelerator.get("vendor") == "nvidia" else (
            "rocm" if runtime["backends"]["rocm"] else "vulkan")
        if runtime["backends"].get(backend):
            return result("can_serve", "can serve", f"{backend} runtime and enough VRAM detected", backend)

    available_ram = memory.get("available_bytes")
    if available_ram is not None and available_ram >= ram_need:
        reason = "enough available RAM for the worker's CPU fallback"
        if best_gpu and best_gpu[0] >= gpu_need:
            reason += "; accelerator build runtime is missing"
        return result("can_serve_cpu", "can serve on CPU (slow)", reason, "cpu")

    have_gpu = _fmt_bytes(best_gpu[0] if best_gpu else None)
    have_ram = _fmt_bytes(available_ram)
    verdict = (f"cannot serve: needs {_fmt_bytes(gpu_need)} VRAM or {_fmt_bytes(ram_need)} RAM, "
               f"you have {have_gpu} VRAM and {have_ram} RAM")
    return result("cannot_serve", verdict, "no accelerator or CPU fallback has enough known memory")


def collect_preflight(
    package_catalog: Mapping[str, Mapping], *, catalog_error: str | None = None,
    state_dir: Path | None = None, run: Callable = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    glob_paths: Callable[[str], list] = lambda pattern: list(Path("/").glob(pattern.lstrip("/"))),
    read_text: Callable[[Path], str] = lambda path: path.read_text(),
    disk_usage: Callable = shutil.disk_usage,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> dict:
    """Collect a complete preflight report; probe failures become unknown values."""
    state_dir = state_dir or (Path.home() / ".nakshatra" / "worker")
    accelerators, nvidia_seen = _nvidia(run, which)
    rocm_rows, rocm_seen = _rocm(run, which)
    accelerators.extend(rocm_rows)
    sysfs_seen = False
    if not rocm_rows:
        sysfs_rows, sysfs_seen = _amd_sysfs(glob_paths, read_text)
        accelerators.extend(sysfs_rows)
    if (rocm_seen or sysfs_seen) and not any(
            row.get("vendor") == "amd" for row in accelerators):
        accelerators.append({
            "vendor": "amd", "name": None, "vram_bytes": None,
            "vram_free_bytes": None,
            "source": "rocm-smi" if rocm_seen else "sysfs", "status": "unknown",
        })
    memory = _memory(read_text)
    disk = _disk(state_dir, disk_usage)
    runtime = _runtime(which, glob_paths, find_spec)
    packages = [
        _package_verdict(package_id, definition, accelerators, memory, disk, runtime)
        for package_id, definition in sorted(package_catalog.items())
        if isinstance(package_id, str) and isinstance(definition, Mapping)
    ]
    ready = any(item["verdict_code"] in {"can_serve", "can_serve_cpu"} for item in packages)
    return {
        "schema": SCHEMA,
        "catalog": {"signed_release": catalog_error is None, "error": catalog_error},
        "accelerators": accelerators,
        "accelerator_state": "unknown" if (nvidia_seen or rocm_seen or sysfs_seen) and not any(
            row.get("status") == "detected" for row in accelerators) else (
                "cpu_only" if not accelerators else "detected"),
        "memory": memory,
        "disk": disk,
        "runtime": runtime,
        "packages": packages,
        "next_command": "nak worker build" if ready else "nak worker preflight",
    }


def render_plain(report: Mapping) -> str:
    memory = report.get("memory", {})
    disk = report.get("disk", {})
    accelerators = report.get("accelerators", [])
    accelerator_text = report.get("accelerator_state", "unknown")
    detected = []
    for accelerator in accelerators:
        if accelerator.get("status") != "detected":
            continue
        capacity = accelerator.get("vram_free_bytes")
        qualifier = "free" if capacity is not None else "total"
        if capacity is None:
            capacity = accelerator.get("vram_bytes")
        detected.append(
            f"{accelerator.get('name') or accelerator.get('vendor', 'GPU')} "
            f"({_fmt_bytes(capacity)} {qualifier})")
    if detected:
        accelerator_text = ", ".join(detected)
    lines = [
        f"Accelerators: {accelerator_text}",
        f"RAM available: {_fmt_bytes(memory.get('available_bytes'))}",
        f"Disk free for worker state: {_fmt_bytes(disk.get('free_bytes'))}",
    ]
    missing = report.get("runtime", {}).get("required_missing", [])
    lines.append("Runtime: ready" if not missing else f"Runtime missing: {', '.join(missing)}")
    catalog_error = report.get("catalog", {}).get("error")
    if catalog_error:
        lines.append(f"Signed package catalog: unknown ({catalog_error})")
    for package in report.get("packages", []):
        lines.append(f"{package['id']}: {package['verdict']}")
    lines.append(f"Next command: {report.get('next_command', 'nak worker preflight')}")
    return "\n".join(lines)
