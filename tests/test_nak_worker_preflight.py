from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from network import nak  # noqa: E402
import worker_preflight as wp  # noqa: E402


GIB = 1024 ** 3
CATALOG = {
    "llama32-3b": {
        "layers": 28,
        "requirements": {
            "gpu_vram_bytes": 4 * GIB,
            "cpu_ram_bytes": 8 * GIB,
            "disk_free_bytes": 12 * GIB,
        },
    },
}


def _collect(tmp_path, *, tools=(), outputs=None, files=None, globs=None, disk_free=40 * GIB):
    outputs = outputs or {}
    files = files or {
        "/proc/meminfo": "MemTotal:       16777216 kB\nMemAvailable:   12582912 kB\n",
    }
    globs = globs or {}
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        value = outputs.get(tuple(argv))
        if isinstance(value, Exception):
            raise value
        if value is None:
            pytest.fail(f"unexpected subprocess: {argv}")
        return SimpleNamespace(returncode=0, stdout=value)

    def read_text(path):
        value = files.get(str(path))
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise FileNotFoundError(path)
        return value

    def glob_paths(pattern):
        return [Path(item) for item in globs.get(pattern, [])]

    def disk_usage(_path):
        if isinstance(disk_free, Exception):
            raise disk_free
        return SimpleNamespace(total=100 * GIB, used=100 * GIB - disk_free, free=disk_free)

    report = wp.collect_preflight(
        CATALOG, state_dir=tmp_path / "state", run=run,
        which=lambda name: f"/bin/{name}" if name in tools else None,
        glob_paths=glob_paths, read_text=read_text, disk_usage=disk_usage,
        find_spec=lambda name: object() if name in {"grpc", "google.protobuf", "numpy", "gguf"} else None,
    )
    return report, calls


BASE_TOOLS = {"bash", "git", "python3", "cc", "cmake", "systemctl"}


def test_nvidia_probe_and_gpu_verdict_never_use_unmocked_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(wp.subprocess, "run", lambda *_a, **_k: pytest.fail("real subprocess escaped"))
    command = ("nvidia-smi", "--query-gpu=name,memory.total,memory.free",
               "--format=csv,noheader,nounits")
    report, calls = _collect(
        tmp_path, tools=BASE_TOOLS | {"nvidia-smi", "nvcc"},
        outputs={command: "NVIDIA RTX 5070, 12288, 10000\n"},
    )

    assert report["accelerators"][0]["name"] == "NVIDIA RTX 5070"
    assert report["packages"][0]["verdict"] == "can serve"
    assert report["packages"][0]["mode"] == "cuda"
    assert report["next_command"] == "nak worker build"
    assert len(calls) == 1 and calls[0][1]["timeout"] == wp.PROBE_TIMEOUT_S


def test_rocm_smi_probe_and_verdict(tmp_path):
    command = ("rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json")
    body = json.dumps({"card0": {
        "Card series": "AMD Radeon RX 9070 XT",
        "VRAM Total Memory (B)": str(16 * GIB),
        "VRAM Total Used Memory (B)": str(2 * GIB),
    }})
    report, _calls = _collect(
        tmp_path, tools=BASE_TOOLS | {"rocm-smi", "hipconfig"}, outputs={command: body})

    assert report["accelerators"][0]["source"] == "rocm-smi"
    assert report["accelerators"][0]["vram_free_bytes"] == 14 * GIB
    assert report["packages"][0]["mode"] == "rocm"


def test_amd_sysfs_fallback_uses_vulkan_runtime(tmp_path):
    total = "/sys/class/drm/card1/device/mem_info_vram_total"
    used = "/sys/class/drm/card1/device/mem_info_vram_used"
    vendor = "/sys/class/drm/card1/device/vendor"
    report, calls = _collect(
        tmp_path, tools=BASE_TOOLS | {"glslc"},
        files={
            "/proc/meminfo": "MemTotal: 16777216 kB\nMemAvailable: 12582912 kB\n",
            total: str(8 * GIB), used: str(1 * GIB), vendor: "0x1002\n",
        },
        globs={
            "/sys/class/drm/card*/device/mem_info_vram_total": [total],
            "/usr/share/vulkan/icd.d/*.json": ["/usr/share/vulkan/icd.d/radeon.json"],
        },
    )

    assert calls == []
    assert report["accelerators"][0]["name"] == "card1"
    assert report["packages"][0]["verdict"] == "can serve"
    assert report["packages"][0]["mode"] == "vulkan"


def test_no_accelerator_reports_cpu_slow(tmp_path):
    report, calls = _collect(tmp_path, tools=BASE_TOOLS)

    assert calls == []
    assert report["accelerator_state"] == "cpu_only"
    assert report["packages"][0]["verdict"] == "can serve on CPU (slow)"
    assert report["packages"][0]["mode"] == "cpu"


def test_garbage_timeout_and_unknown_resources_never_fail_hard(tmp_path):
    command = ("nvidia-smi", "--query-gpu=name,memory.total,memory.free",
               "--format=csv,noheader,nounits")
    report, _calls = _collect(
        tmp_path, tools=BASE_TOOLS | {"nvidia-smi", "nvcc"},
        outputs={command: subprocess.TimeoutExpired(command, 2)},
        files={"/proc/meminfo": "not meminfo"}, disk_free=OSError("no stat"),
    )

    assert report["accelerator_state"] == "unknown"
    assert report["accelerators"][0]["vram_bytes"] is None
    assert report["memory"]["available_bytes"] is None
    assert report["disk"]["free_bytes"] is None
    assert report["packages"][0]["verdict"].startswith("cannot serve: needs")
    assert "you have unknown" in report["packages"][0]["verdict"]
    assert report["next_command"] == "nak worker preflight"


def test_cli_json_shape_and_plain_verdict(monkeypatch, capsys):
    report = {
        "schema": wp.SCHEMA,
        "catalog": {"signed_release": True, "error": None},
        "accelerators": [], "accelerator_state": "cpu_only",
        "memory": {"available_bytes": 10 * GIB},
        "disk": {"free_bytes": 20 * GIB},
        "runtime": {"required_missing": []},
        "packages": [{"id": "llama32-3b", "verdict": "can serve on CPU (slow)"}],
        "next_command": "nak worker build",
    }
    monkeypatch.setattr(nak, "_worker_preflight", lambda _sp: report)

    assert nak.main(["worker", "preflight", "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["schema"] == wp.SCHEMA
    assert parsed["packages"][0]["id"] == "llama32-3b"
    assert parsed["next_command"] == "nak worker build"

    assert nak.main(["worker", "preflight"]) == 0
    plain = capsys.readouterr().out
    assert "llama32-3b: can serve on CPU (slow)" in plain
    assert "Next command: nak worker build" in plain
