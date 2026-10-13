#!/usr/bin/env python3
"""Fail-closed launcher for the signed-release Nakshatra worker service.

The GPU daemon is machine-local (it was compiled for this node), while this launcher and worker.py
come from the current signed release.  Before execing worker.py, bind those halves together by
requiring build.json to name the exact engine TREE shipped by this release.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from urllib.parse import urlsplit


class WorkerServiceError(RuntimeError):
    pass


DEFAULT_TRUSTED_PILLAR_NETS = "10.42.0.0/24"
_SPKI_RE = re.compile(r"[0-9a-fA-F]{64}")


def validate_pillar_transport(
    pillar_url: str,
    pillar_spki: str = "",
    trusted_pillar_nets: str = DEFAULT_TRUSTED_PILLAR_NETS,
) -> tuple[str, str]:
    """Validate the signed-release worker's pillar trust boundary.

    HTTPS is authenticated by a required SHA-256 SPKI pin. Plain HTTP is
    accepted only for loopback or an IP literal inside an operator-trusted
    encrypted-overlay CIDR.
    """
    try:
        parsed = urlsplit(pillar_url)
        host = parsed.hostname
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError("port out of range")
    except ValueError as e:
        raise WorkerServiceError(f"PILLAR_URL is invalid: {e}") from e
    if parsed.scheme not in {"http", "https"} or not host:
        raise WorkerServiceError("PILLAR_URL must be an http:// or https:// URL with a host")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment or parsed.username or parsed.password:
        # Requests are signed over their absolute path (/peer, /peers, ...): a base path would make the signed
        # path and the sent path differ and every call would fail authentication. The pillar is an origin.
        raise WorkerServiceError("PILLAR_URL must be a bare origin like https://host:port (no path, query or credentials)")

    pin = (pillar_spki or "").strip().lower()
    if pin and not _SPKI_RE.fullmatch(pin):
        raise WorkerServiceError("PILLAR_SPKI_SHA256 must be exactly 64 hexadecimal characters")
    if parsed.scheme == "https":
        if not pin:
            raise WorkerServiceError(
                "HTTPS PILLAR_URL requires PILLAR_SPKI_SHA256 (use `nak worker on --pillar-spki`)"
            )
        return pillar_url, pin

    try:
        networks = [
            ipaddress.ip_network(item.strip(), strict=False)
            for item in (trusted_pillar_nets or "").split(",")
            if item.strip()
        ]
    except ValueError as e:
        raise WorkerServiceError(f"TRUSTED_PILLAR_NETS contains an invalid CIDR: {e}") from e
    if host.lower() == "localhost":
        return pillar_url, ""
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError as e:
        raise WorkerServiceError(
            "HTTP PILLAR_URL host must be localhost or an IP literal inside TRUSTED_PILLAR_NETS"
        ) from e
    if address.is_loopback or any(
        address.version == network.version and address in network for network in networks
    ):
        return pillar_url, ""
    raise WorkerServiceError(
        f"HTTP PILLAR_URL host {host!r} is outside loopback/TRUSTED_PILLAR_NETS; "
        "use pinned HTTPS or add its encrypted-overlay CIDR explicitly"
    )


def engine_tree(base: Path) -> str:
    try:
        matches = re.findall(r"^TREE=([0-9a-f]{40})$", base.read_text(), re.MULTILINE)
    except OSError as e:
        raise WorkerServiceError(f"cannot read release engine identity {base}: {e}") from e
    if len(matches) != 1:
        raise WorkerServiceError(f"release engine identity {base} has no single valid TREE")
    return matches[0]


def _load_build(path: Path) -> dict:
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise WorkerServiceError(f"{path} is missing; run `nak worker build` first") from e
    except (OSError, ValueError) as e:
        raise WorkerServiceError(f"cannot read valid worker build record {path}: {e}") from e
    if not isinstance(record, dict):
        raise WorkerServiceError(f"worker build record {path} is not a JSON object")
    return record


def _load_node_key(path: Path) -> bytes:
    """Read a regular, owner-only node key without a stat/read race."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise WorkerServiceError(f"cannot read node identity {path}: {e}") from e
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkerServiceError(f"node identity {path} is not a regular file")
        if metadata.st_mode & 0o077:
            raise WorkerServiceError(
                f"node identity {path} is not owner-only; run `chmod 600 {path}`"
            )
        with os.fdopen(fd, "rb", closefd=False) as handle:
            private = handle.read()
    finally:
        os.close(fd)
    if len(private) != 32:
        raise WorkerServiceError(f"node identity {path} is malformed (expected 32 raw bytes)")
    return private


def worker_command(args) -> tuple[list[str], dict[str, str]]:
    record = _load_build(args.build_json)
    wanted = engine_tree(args.engine_base)
    got = record.get("engine_tree")
    if got != wanted:
        raise WorkerServiceError(
            f"worker daemon engine TREE {got or '(missing)'} does not match this release's {wanted}; "
            "run `nak worker build` for the current release"
        )

    worker_root = args.worker_root.resolve()
    try:
        daemon = Path(record["daemon_bin"]).resolve(strict=True)
    except (KeyError, OSError) as e:
        raise WorkerServiceError("worker build record names no existing daemon; run `nak worker build`") from e
    if not daemon.is_relative_to(worker_root) or not daemon.is_file() or not os.access(daemon, os.X_OK):
        raise WorkerServiceError(f"worker daemon {daemon} is not an executable inside {worker_root}")

    backend = record.get("backend")
    if backend not in {"cpu", "cuda", "rocm", "vulkan", "metal"}:
        raise WorkerServiceError(f"worker build record has invalid backend {backend!r}")
    try:
        n_gpu_layers = int(record["n_gpu_layers"])
    except (KeyError, TypeError, ValueError) as e:
        raise WorkerServiceError("worker build record has no valid n_gpu_layers") from e
    if n_gpu_layers < 0:
        raise WorkerServiceError("worker build record has a negative n_gpu_layers")

    private = _load_node_key(args.node_key)
    from nakshatra_auth import public_key_hex_from_private
    node_id = public_key_hex_from_private(private)

    pillar_url = os.environ.get("PILLAR_URL", "").strip()
    if not pillar_url:
        raise WorkerServiceError("worker.env must set PILLAR_URL")
    pillar_spki = os.environ.get("PILLAR_SPKI_SHA256", "").strip()
    trusted_pillar_nets = os.environ.get(
        "TRUSTED_PILLAR_NETS", DEFAULT_TRUSTED_PILLAR_NETS
    ).strip()
    pillar_url, pillar_spki = validate_pillar_transport(
        pillar_url, pillar_spki, trusted_pillar_nets
    )
    package_url = os.environ.get("PACKAGE_URL", "").strip()
    if not package_url:
        raise WorkerServiceError("worker.env must set PACKAGE_URL")
    listen_address = os.environ.get("LISTEN", "").strip() or "[::]"
    # B3: remote packages must be signed by an operator-pinned signer; without the pin every bootstrap fails.
    package_pubkey = os.environ.get("PACKAGE_TRUSTED_PUBKEY", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", package_pubkey):
        raise WorkerServiceError(
            "worker.env must set PACKAGE_TRUSTED_PUBKEY (64 hex; use `nak worker on --package-pubkey`)")
    idle_unload_raw = os.environ.get("IDLE_UNLOAD_S", "600").strip() or "600"
    try:
        idle_unload_s = float(idle_unload_raw)
    except ValueError as e:
        raise WorkerServiceError("IDLE_UNLOAD_S must be a finite number >= 0") from e
    if not math.isfinite(idle_unload_s) or idle_unload_s < 0:
        raise WorkerServiceError("IDLE_UNLOAD_S must be a finite number >= 0")
    release_grace_raw = os.environ.get("RELEASE_GRACE_S", "30").strip() or "30"
    try:
        release_grace_s = float(release_grace_raw)
    except ValueError as e:
        raise WorkerServiceError("RELEASE_GRACE_S must be finite (negative disables it)") from e
    if not math.isfinite(release_grace_s):
        raise WorkerServiceError("RELEASE_GRACE_S must be finite (negative disables it)")

    owner_names = (
        "OWNER_LEASE_PORT", "OWNER_LEASE_TOKEN", "OWNER_LEASE_TTL_S",
        "OWNER_LEASE_RESUME_S",
    )
    owner_enabled = any(os.environ.get(name, "").strip() for name in owner_names)
    owner_values = {}
    if owner_enabled:
        token = os.environ.get("OWNER_LEASE_TOKEN", "").strip().lower()
        if len(token) < 64 or len(token) % 2 or not re.fullmatch(r"[0-9a-f]+", token):
            raise WorkerServiceError(
                "owner-lease.env OWNER_LEASE_TOKEN must be 32+ random bytes encoded as hex"
            )
        try:
            owner_port = int(os.environ.get("OWNER_LEASE_PORT", "5571"))
            owner_ttl = float(os.environ.get("OWNER_LEASE_TTL_S", "6"))
            owner_resume = float(os.environ.get("OWNER_LEASE_RESUME_S", "30"))
        except ValueError as e:
            raise WorkerServiceError(f"owner-lease.env has an invalid value: {e}") from e
        if not 1 <= owner_port <= 65535:
            raise WorkerServiceError("OWNER_LEASE_PORT must be between 1 and 65535")
        if not math.isfinite(owner_ttl) or owner_ttl <= 0:
            raise WorkerServiceError("OWNER_LEASE_TTL_S must be finite and > 0")
        if not math.isfinite(owner_resume) or owner_resume < 0:
            raise WorkerServiceError("OWNER_LEASE_RESUME_S must be finite and >= 0")
        owner_values = {
            "NAKSHATRA_OWNER_LEASE_ENABLED": "true",
            "NAKSHATRA_OWNER_LEASE_PORT": str(owner_port),
            "NAKSHATRA_OWNER_LEASE_TOKEN": token,
            "NAKSHATRA_OWNER_LEASE_TTL_S": f"{owner_ttl:g}",
            "NAKSHATRA_OWNER_LEASE_RESUME_S": f"{owner_resume:g}",
        }

    if not args.worker_script.is_file():
        raise WorkerServiceError(f"release worker script is missing: {args.worker_script}")
    command = [
        sys.executable, str(args.worker_script),
        "--port", str(args.port),
        "--sub-gguf", str(args.sub_gguf),
        "--mode", args.mode,
        "--layer-start", str(args.layer_start),
        "--layer-end", str(args.layer_end),
        "--model-id", args.model_id,
        "--daemon-bin", str(daemon),
        "--n-ctx", str(args.n_ctx),
        "--gpu-backend", backend,
        "--n-gpu-layers", str(n_gpu_layers),
        "--node-id", node_id,
        "--no-file-server",
        "--package-trusted-pubkey", package_pubkey,
    ]
    env = dict(
        os.environ,
        NAKSHATRA_NODE_ID=node_id,
        NAKSHATRA_PILLAR_URL=pillar_url,
        STHAMBHA_PILLAR_SPKI_SHA256=pillar_spki,
        NAKSHATRA_PACKAGE_URL=package_url,
        NAKSHATRA_LISTEN_ADDRESS=listen_address,
        NAKSHATRA_WORKER_IDLE_UNLOAD_S=f"{idle_unload_s:g}",
        NAKSHATRA_WORKER_RELEASE_GRACE_S=f"{release_grace_s:g}",
        # A signed-release worker never inherits Mode-A downgrade switches.
        NAKSHATRA_RELEASE_SERVICE="true",
        NAKSHATRA_TLS_REQUIRED="true",
        NAKSHATRA_AUTH_REQUIRED="true",
        NAKSHATRA_AUDIT_PATH=str(worker_root / "audit.jsonl"),
    )
    for name in (
        "NAKSHATRA_OWNER_LEASE_ENABLED", "NAKSHATRA_OWNER_LEASE_PORT",
        "NAKSHATRA_OWNER_LEASE_TOKEN", "NAKSHATRA_OWNER_LEASE_TTL_S",
        "NAKSHATRA_OWNER_LEASE_RESUME_S",
    ):
        env.pop(name, None)
    env.update(owner_values)
    # Never trust an ambient PeerId.  The endpoint is advertised only while
    # this node has opted into the release-managed sidecar, and is derived
    # from the same node key that signs the worker's pillar registration.
    env.pop("NAKSHATRA_P2P_PEER_ID", None)
    if args.p2p_opt_in is not None and args.p2p_opt_in.is_file():
        from sidecar_key import peer_id_from_node_pub
        env["NAKSHATRA_P2P_PEER_ID"] = peer_id_from_node_pub(node_id)
    return command, env


def main(argv=None, *, execve=os.execve) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-json", type=Path, required=True)
    ap.add_argument("--engine-base", type=Path, required=True)
    ap.add_argument("--worker-root", type=Path, required=True)
    ap.add_argument("--worker-script", type=Path, required=True)
    ap.add_argument("--node-key", type=Path, required=True)
    ap.add_argument("--p2p-opt-in", type=Path)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--sub-gguf", type=Path, required=True)
    ap.add_argument("--mode", choices=("first", "middle", "last", "solo"), required=True)
    ap.add_argument("--layer-start", type=int, required=True)
    ap.add_argument("--layer-end", type=int, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--n-ctx", type=int, default=2048)
    args = ap.parse_args(argv)
    try:
        command, env = worker_command(args)
    except WorkerServiceError as e:
        print(f"[worker-service] REFUSING TO START: {e}", file=sys.stderr, flush=True)
        return 1
    execve(command[0], command, env)
    return 0  # pragma: no cover - os.execve does not return


if __name__ == "__main__":
    raise SystemExit(main())
