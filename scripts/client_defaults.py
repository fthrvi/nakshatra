"""Release-local defaults shared by Nakshatra clients and gateways.

These helpers only inspect this uid's release state.  They never contact the
pillar or the activation sidecar, which keeps default resolution safe to use
from tests and before a request is dispatched.
"""
from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path
from typing import Callable, Mapping


REGISTRY_KEYID_SOURCE_ENV = "NAKSHATRA_REGISTRY_KEYID_SOURCE"


def default_p2p_socket_path(
    env: Mapping[str, str] | None = None,
    *,
    uid: int | None = None,
) -> Path:
    """Return the release sidecar's conventional control-socket path."""
    source = os.environ if env is None else env
    runtime = source.get("XDG_RUNTIME_DIR", "").strip()
    if not runtime:
        effective_uid = os.getuid() if uid is None else uid
        runtime = f"/run/user/{effective_uid}"
    return Path(runtime) / "nakshatra" / "p2p.sock"


def resolve_p2p_socket(env: Mapping[str, str] | None = None) -> str:
    """Resolve an explicit socket or a safe, live release-sidecar default.

    Presence in the environment is authoritative, including an explicitly
    blank value.  The conventional path is selected only when it is a Unix
    socket owned by this uid; a stale file, directory, or another uid's socket
    is ignored.
    """
    source = os.environ if env is None else env
    if "NAKSHATRA_P2P_SOCKET" in source:
        return source.get("NAKSHATRA_P2P_SOCKET", "").strip()

    candidate = default_p2p_socket_path(source)
    try:
        info = candidate.lstat()
    except OSError:
        return ""
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        return ""
    return str(candidate)


def _environment_file_value(path: Path, name: str) -> str:
    """Read one systemd EnvironmentFile-style scalar without executing it."""
    try:
        if path.stat().st_size > 64 * 1024:
            return ""
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return ""

    result = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw = stripped.split("=", 1)
        if key.strip() != name:
            continue
        try:
            parts = shlex.split(raw, comments=True, posix=True)
        except ValueError:
            continue
        result = " ".join(parts).strip()
    return result


def resolve_registry_keyid(
    env: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
    key_loader: Callable[[Path], tuple[bytes, str]] | None = None,
) -> tuple[str, str]:
    """Return ``(keyid, provenance)`` for pillar and worker signatures.

    The release registrar is the first constructed identity because it
    registers ``NKS_NODE_ID -> this node key``.  Release workers instead use
    the same public-key hex as both their node id and keyid, so that is the
    fallback when this node has no registrar configuration.
    """
    source = os.environ if env is None else env
    if "NKS_REGISTRY_KEYID" in source:
        keyid = source.get("NKS_REGISTRY_KEYID", "").strip()
        inherited_source = source.get(REGISTRY_KEYID_SOURCE_ENV, "").strip()
        return keyid, inherited_source or "NKS_REGISTRY_KEYID environment variable"

    base_home = Path.home() if home is None else Path(home)
    registrar_env = base_home / ".nakshatra" / "registrar.env"
    registrar_id = _environment_file_value(registrar_env, "NKS_NODE_ID")
    if registrar_id:
        return registrar_id, f"NKS_NODE_ID in {registrar_env}"

    if key_loader is None:
        import nakshatra_auth
        key_loader = nakshatra_auth.load_or_create_worker_key
    key_path = base_home / ".nakshatra" / "keys" / "worker.ed25519"
    _private_key, public_key_hex = key_loader(key_path)
    return public_key_hex, f"worker public key at {key_path}"


def apply_client_defaults(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy an environment and materialize both release-client defaults."""
    original = os.environ if env is None else env
    resolved = dict(original)

    p2p_socket = resolve_p2p_socket(original)
    if p2p_socket:
        resolved["NAKSHATRA_P2P_SOCKET"] = p2p_socket

    keyid, keyid_source = resolve_registry_keyid(original)
    if keyid:
        resolved["NKS_REGISTRY_KEYID"] = keyid
        if "NKS_REGISTRY_KEYID" not in original:
            resolved[REGISTRY_KEYID_SOURCE_ENV] = keyid_source
    return resolved
