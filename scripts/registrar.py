#!/usr/bin/env python3
"""Advertise this node's idle compute capacity to a Sthambha pillar.

The registrar is deliberately separate from a model worker: placement needs to know
about idle capacity before it can decide what worker to summon.  It signs each
``POST /peer`` with the node's existing Ed25519 identity and never creates or rotates
that identity.

Configuration is read from the environment by :func:`load_config`:

  NKS_NODE_ID              stable node id (defaults to hostname)
  NKS_PILLAR_URL           pillar base URL (default http://10.42.0.3:7777)
  NKS_NODE_ADDRESS         address advertised for this node
  NKS_OWNER_RESERVE_GB     free VRAM kept for the owner (default 0)
  NKS_GPU_BACKEND          cuda | rocm (auto-detected when unset)
  NKS_GPU_MODEL            model label used by rocm-smi (default AMD-GPU)
  NKS_OLLAMA_URL           local Ollama API; unset disables model discovery
  NKS_OLLAMA_MODELS        optional comma-separated allowlist
  NKS_OLLAMA_TIMEOUT_S     local Ollama metadata timeout (default 5)
  NKS_REGISTRAR_INTERVAL   heartbeat seconds (default 30)
  NKS_REGISTRAR            0 disables heartbeats without stopping the service
  MESH_COMPUTE_STATE       mesh-compute state directory (default ~/.mesh-compute)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import nakshatra_auth as na


@dataclass(frozen=True)
class RegistrarConfig:
    pillar_url: str
    node_id: str
    address: str
    reserve_gb: float
    interval_s: float
    gpu_backend: str
    gpu_model: str
    ollama_url: str
    ollama_models: frozenset[str]
    ollama_timeout_s: float
    mesh_compute_state: Path
    enabled: bool


def _nonnegative_float(env: Mapping[str, str], name: str, default: str) -> float:
    try:
        value = float(env.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def load_config(
    env: Optional[Mapping[str, str]] = None,
    *,
    home: Optional[Path] = None,
    hostname: Optional[str] = None,
) -> RegistrarConfig:
    """Load and validate one immutable heartbeat configuration."""
    source = os.environ if env is None else env
    base_home = Path.home() if home is None else Path(home)
    interval = _nonnegative_float(source, "NKS_REGISTRAR_INTERVAL", "30")
    if interval == 0:
        raise ValueError("NKS_REGISTRAR_INTERVAL must be greater than zero")
    backend = source.get("NKS_GPU_BACKEND", "").strip().lower()
    if backend not in ("", "cuda", "rocm", "cpu"):
        raise ValueError("NKS_GPU_BACKEND must be cuda, rocm, cpu, or unset")
    models = frozenset(
        name.strip() for name in source.get("NKS_OLLAMA_MODELS", "").split(",")
        if name.strip()
    )
    return RegistrarConfig(
        pillar_url=source.get("NKS_PILLAR_URL", "http://10.42.0.3:7777").rstrip("/"),
        node_id=source.get("NKS_NODE_ID", "").strip() or hostname or socket.gethostname(),
        address=source.get("NKS_NODE_ADDRESS", "").strip(),
        reserve_gb=_nonnegative_float(source, "NKS_OWNER_RESERVE_GB", "0"),
        interval_s=interval,
        gpu_backend=backend,
        gpu_model=source.get("NKS_GPU_MODEL", "AMD-GPU").strip() or "AMD-GPU",
        ollama_url=source.get("NKS_OLLAMA_URL", "").rstrip("/"),
        ollama_models=models,
        ollama_timeout_s=_nonnegative_float(source, "NKS_OLLAMA_TIMEOUT_S", "5"),
        mesh_compute_state=Path(source.get("MESH_COMPUTE_STATE", base_home / ".mesh-compute")),
        enabled=source.get("NKS_REGISTRAR", "1").strip() != "0",
    )


def _gpu(config: RegistrarConfig):
    """Return ``(free_gb, total_gb, model, backend)`` best-effort."""
    backend = config.gpu_backend
    if not backend:
        backend = "cuda" if shutil.which("nvidia-smi") else (
            "rocm" if shutil.which("rocm-smi") else ""
        )
    try:
        if backend == "cuda":
            lines = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.free,name",
                 "--format=csv,noheader,nounits"], timeout=6
            ).decode().splitlines()
            best = max((line.split(",") for line in lines), key=lambda parts: int(parts[0]))
            total, free = int(best[0]) / 1024, int(best[1]) / 1024
            return free, total, best[2].strip(), "cuda"
        if backend == "rocm":
            output = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram"], timeout=6
            ).decode()
            cards = {}
            for line in output.splitlines():
                if "Total Memory" in line or "Used Memory" in line:
                    gpu = line.split(":")[0].strip()
                    value = int(line.split(":")[-1].strip())
                    cards.setdefault(gpu, {})["used" if "Used" in line else "total"] = value
            best = max(cards.values(), key=lambda card: card.get("total", 0), default={})
            total_bytes = best.get("total", 0)
            return ((total_bytes - best.get("used", 0)) / (1024**3),
                    total_bytes / (1024**3), config.gpu_model, "rocm")
    except Exception:
        pass
    return 0.0, 0.0, "unknown", backend or "cpu"


def _ollama_json(config: RegistrarConfig, path: str, body=None):
    raw = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if raw is not None else {}
    request = urllib.request.Request(config.ollama_url + path, data=raw, headers=headers)
    with urllib.request.urlopen(request, timeout=config.ollama_timeout_s) as response:
        return json.load(response)


def _ollama_model_id(name: str) -> str:
    return name[:-len(":latest")] if name.endswith(":latest") else name


def _ollama_block_count(model_info, family: str = "") -> int:
    if not isinstance(model_info, dict):
        return 0
    preferred = model_info.get(f"{family}.block_count") if family else None
    try:
        preferred = int(preferred)
    except (TypeError, ValueError):
        preferred = 0
    if preferred > 0:
        return preferred
    candidates = set()
    for key, value in model_info.items():
        if not isinstance(key, str) or not key.endswith(".block_count"):
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            candidates.add(value)
    return candidates.pop() if len(candidates) == 1 else 0


def _ollama_inventory(config: RegistrarConfig):
    """Return whole-model Sthambha offerings/files; discovery is best-effort."""
    if not config.ollama_url:
        return [], []
    try:
        models = _ollama_json(config, "/api/tags").get("models", [])
    except Exception as exc:
        print(f"  ollama discovery failed: {type(exc).__name__}: {exc}", flush=True)
        return [], []
    offerings, cached_files, seen = [], [], set()
    for item in models if isinstance(models, list) else []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if not isinstance(name, str) or not name:
            continue
        model_id = _ollama_model_id(name)
        if config.ollama_models and name not in config.ollama_models and model_id not in config.ollama_models:
            continue
        capabilities = item.get("capabilities")
        if isinstance(capabilities, list) and capabilities and "completion" not in capabilities:
            continue
        if model_id in seen:
            continue
        try:
            shown = _ollama_json(config, "/api/show", {"model": name})
            details = shown.get("details") or item.get("details") or {}
            family = details.get("family", "") if isinstance(details, dict) else ""
            block_count = _ollama_block_count(shown.get("model_info"), family)
        except Exception as exc:
            print(f"  ollama model {name!r} skipped: {type(exc).__name__}: {exc}", flush=True)
            continue
        if block_count <= 0:
            print(f"  ollama model {name!r} skipped: no unambiguous block_count", flush=True)
            continue
        common = {"model_id": model_id, "model_sha256": "",
                  "layer_start": 0, "layer_end": block_count}
        try:
            size_bytes = max(0, int(item.get("size", 0) or 0))
        except (TypeError, ValueError):
            size_bytes = 0
        offerings.append(dict(common))
        cached_files.append({**common, "size_bytes": size_bytes, "file_path": ""})
        seen.add(model_id)
    return offerings, cached_files


def _read_small_json(path: Path, limit: int = 64 * 1024):
    try:
        if path.stat().st_size > limit:
            return None
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _iso_to_unix(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (OverflowError, ValueError):
        return None


def _health_block(health: dict) -> dict:
    """Return only well-typed health fields, warning once about bad input."""
    result = {}
    invalid = []

    state = health.get("state", "unknown")
    if isinstance(state, str):
        result["state"] = state
    else:
        invalid.append("state")

    if "reasons" in health:
        reasons = health["reasons"]
        if isinstance(reasons, list) and all(isinstance(reason, str) for reason in reasons):
            result["reasons"] = [reason[:200] for reason in reasons[:10]]
        else:
            invalid.append("reasons")

    if "last_crash" in health:
        last_crash = health["last_crash"]
        if not isinstance(last_crash, dict):
            invalid.append("last_crash")
        elif "ts" in last_crash:
            last_crash_ts = _iso_to_unix(last_crash["ts"])
            if last_crash_ts is None:
                invalid.append("last_crash.ts")
            else:
                result["last_crash_ts"] = last_crash_ts

    if "checked_ts" in health:
        checked_ts = _iso_to_unix(health["checked_ts"])
        if checked_ts is None:
            invalid.append("checked_ts")
        else:
            result["checked_ts"] = checked_ts

    if invalid:
        print(f"  warning: ignored invalid health.json fields: {', '.join(invalid)}", flush=True)
    return result


def _mesh_compute_blocks(config: RegistrarConfig) -> dict:
    offer = _read_small_json(config.mesh_compute_state / "offer.json")
    result = {"compute": offer if offer is not None else {"offered": False}}
    health = _read_small_json(config.mesh_compute_state / "health.json")
    if health is not None:
        result["health"] = _health_block(health)
    return result


def build_payload(config: RegistrarConfig, public_key_hex: str) -> dict:
    """Probe local capacity and build the exact signed registration body."""
    free, total, model, backend = _gpu(config)
    offered = max(0.0, free - config.reserve_gb)
    layer_offerings, cached_files = _ollama_inventory(config)
    return {
        "node_id": config.node_id,
        "node_type": "compute",
        "address": config.address,
        "public_key_hex": public_key_hex,
        "budget": {"vram_offered_gb": round(offered, 2)},
        "hardware": {
            "platform": platform.system().lower(),
            "arch": platform.machine(),
            "gpus": [{"model": model, "vram_total_gb": round(total, 1),
                      "vram_free_gb": round(free, 1), "backend": backend}],
        },
        "gpu_info": {"name": model, "vram_gb": round(total, 1)},
        "layer_offerings": layer_offerings,
        "cached_files": cached_files,
        **_mesh_compute_blocks(config),
    }


def heartbeat_once(config: RegistrarConfig, private_key: bytes, public_key_hex: str):
    """Build, sign, and POST one heartbeat.  Returns status/offered/model."""
    payload = build_payload(config, public_key_hex)
    raw = json.dumps(payload).encode()
    auth_header, _ = na.build_signed_envelope(
        private_key, config.node_id, "POST", "/peer", raw
    )
    request = urllib.request.Request(
        config.pillar_url + "/peer", data=raw, method="POST",
        headers={"Authorization": auth_header, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        gpu = payload["hardware"]["gpus"][0]
        return (response.status, round(payload["budget"]["vram_offered_gb"], 1), gpu["model"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--once", action="store_true", help="send one heartbeat and exit")
    args = parser.parse_args(argv)
    try:
        config = load_config()
        private_key, public_key_hex = na.load_worker_key()
    except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
        print(f"registrar: {exc}", flush=True)
        return 1
    print(f"registrar: {config.node_id} → {config.pillar_url} every "
          f"{config.interval_s:.0f}s (reserve {config.reserve_gb:g}GB)", flush=True)
    if args.once:
        print(heartbeat_once(config, private_key, public_key_hex), flush=True)
        return 0
    while True:
        if config.enabled:
            try:
                status, offered, model = heartbeat_once(config, private_key, public_key_hex)
                print(f"  [{time.strftime('%H:%M:%S')}] {status} · {model} offering {offered}GB",
                      flush=True)
            except Exception as exc:
                print(f"  heartbeat failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(config.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
