"""Single-source payment network configuration and mainnet safety gates."""
from __future__ import annotations

import json
import os
import re
import secrets
import stat
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEVNET_GENESIS_HASH = "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG"
MAINNET_GENESIS_HASH = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"
DEVNET_PROGRAM = "GLWqQsSqUsDkPvkc8rxcoQUyro7ZQd5denht21NtdTTo"
DEVNET_MINT = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"
MAINNET_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {char: index for index, char in enumerate(_B58)}
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_NETWORK_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_EXPLORER_CLUSTER = re.compile(r"[A-Za-z0-9_-]{0,64}")
_ENTRY_FIELDS = frozenset({"rpc", "genesis", "program", "mint"})


class NetworkConfigError(ValueError):
    """The payment network file or one of its safety gates is invalid."""


@dataclass(frozen=True)
class PayNetwork:
    name: str
    rpc: str
    genesis: str
    program: str
    mint: str
    explorer: str | None


@dataclass(frozen=True)
class NetworkDocument:
    path: Path
    active: str
    networks: Mapping[str, PayNetwork]
    raw: dict


def network_path(home: Path | None = None) -> Path:
    return (Path.home() if home is None else Path(home)) / ".nakshatra" / "pay" / "network.json"


def arm_path(home: Path | None = None) -> Path:
    return network_path(home).with_name("mainnet.armed")


def limits_path(home: Path | None = None) -> Path:
    return network_path(home).with_name("limits.json")


def _decode_base58(value: str, field: str) -> bytes:
    if not value:
        raise NetworkConfigError(f"{field} must be a non-empty base58 value")
    number = 0
    try:
        for char in value:
            number = number * 58 + _B58_INDEX[char]
    except KeyError as exc:
        raise NetworkConfigError(f"{field} contains an invalid base58 character") from exc
    zeroes = len(value) - len(value.lstrip("1"))
    body = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    return b"\0" * zeroes + body


def _pubkey(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise NetworkConfigError(f"{field} must be a string")
    if allow_empty and not value:
        return ""
    raw = _decode_base58(value, field)
    if len(raw) != 32:
        raise NetworkConfigError(f"{field} must contain exactly 32 bytes")
    return value


def validate_rpc_url(value: object, field: str = "payment RPC", *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise NetworkConfigError(f"{field} must be a string")
    if allow_empty and not value:
        return ""
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise NetworkConfigError(f"{field} is not a valid URL: {exc}") from exc
    host = (parsed.hostname or "").lower()
    if (parsed.scheme not in {"http", "https"} or not host
            or parsed.username is not None or parsed.password is not None
            or parsed.fragment or (port is not None and not 1 <= port <= 65535)):
        raise NetworkConfigError(
            f"{field} must be an http(s) URL without credentials or a fragment")
    if parsed.scheme != "https" and host not in _LOOPBACK_HOSTS:
        raise NetworkConfigError(f"{field} must use https unless its host is loopback")
    return value


def _read_private_regular(path: Path, description: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise NetworkConfigError(f"cannot inspect {description} at {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise NetworkConfigError(f"refusing {path}: {description} must be a regular file, not a symlink")
    if info.st_uid != os.geteuid():
        raise NetworkConfigError(f"refusing {path}: {description} must be owned by uid {os.geteuid()}")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise NetworkConfigError(f"refusing {path}: {description} must have mode 0600")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise NetworkConfigError(f"cannot read {description} at {path}: {exc}") from exc


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise NetworkConfigError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _load_json(raw: bytes, description: str):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object)
    except NetworkConfigError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise NetworkConfigError(f"invalid JSON in {description}: {exc}") from exc


def _parse_entry(name: str, value: object) -> PayNetwork:
    if not isinstance(value, dict):
        raise NetworkConfigError(f"network {name!r} must be a JSON object")
    fields = set(value)
    if not _ENTRY_FIELDS <= fields or fields - (_ENTRY_FIELDS | {"explorer_cluster"}):
        raise NetworkConfigError(
            f"network {name!r} must contain only rpc, genesis, program, mint, "
            "and optional explorer_cluster")
    rpc = validate_rpc_url(value["rpc"], f"network {name!r} rpc", allow_empty=True)
    genesis = _pubkey(value["genesis"], f"network {name!r} genesis")
    if name == "devnet" and genesis != DEVNET_GENESIS_HASH:
        raise NetworkConfigError("network 'devnet' must use the Solana devnet genesis")
    if name == "mainnet" and genesis != MAINNET_GENESIS_HASH:
        raise NetworkConfigError("network 'mainnet' must use the Solana mainnet genesis")
    # The real clusters may only appear under their own names, so the mainnet
    # gates (arm file, limits) can never be skipped by renaming the entry.
    if genesis == MAINNET_GENESIS_HASH and name != "mainnet":
        raise NetworkConfigError(
            f"network {name!r} uses the Solana mainnet genesis; only 'mainnet' may")
    if genesis == DEVNET_GENESIS_HASH and name != "devnet":
        raise NetworkConfigError(
            f"network {name!r} uses the Solana devnet genesis; only 'devnet' may")
    if name not in {"devnet", "mainnet"} and rpc:
        host = (urllib.parse.urlsplit(rpc).hostname or "").lower()
        if host not in _LOOPBACK_HOSTS:
            raise NetworkConfigError(
                f"network {name!r} is a test network and must use a loopback rpc")
    program = _pubkey(value["program"], f"network {name!r} program", allow_empty=True)
    mint = _pubkey(value["mint"], f"network {name!r} mint", allow_empty=True)
    explorer = value.get("explorer_cluster")
    if explorer is None:
        explorer = "devnet" if name == "devnet" else "" if name == "mainnet" else None
    if explorer is not None and (not isinstance(explorer, str)
                                 or _EXPLORER_CLUSTER.fullmatch(explorer) is None):
        raise NetworkConfigError(
            f"network {name!r} explorer_cluster must be a short URL-safe string")
    return PayNetwork(name, rpc, genesis, program, mint, explorer)


def _check_environment(network: PayNetwork, environ: Mapping[str, str], path: Path) -> None:
    checks = {
        "NAK_PAY_RPC": network.rpc,
        "NAKSHATRA_PAY_RPC_URL": network.rpc,
        "NAK_PAY_PROGRAM": network.program,
        "NAK_PAY_MINT": network.mint,
    }
    for name, expected in checks.items():
        supplied = environ.get(name, "").strip()
        if supplied and supplied != expected:
            raise NetworkConfigError(
                f"{name} disagrees with {path}; network.json is the only payment network source")


def load_document(*, home: Path | None = None,
                  environ: Mapping[str, str] | None = None) -> NetworkDocument | None:
    """Load and strictly validate network.json, returning ``None`` only when absent."""
    path = network_path(home)
    if not os.path.lexists(path):
        return None
    raw_bytes = _read_private_regular(path, "payment network config")
    try:
        value = _load_json(raw_bytes, f"payment network config {path}")
    except NetworkConfigError:
        raise
    if not isinstance(value, dict) or set(value) != {"active", "networks"}:
        raise NetworkConfigError("payment network config must contain exactly active and networks")
    active = value["active"]
    entries = value["networks"]
    if not isinstance(active, str) or _NETWORK_NAME.fullmatch(active) is None:
        raise NetworkConfigError("payment network config active must be a valid network name")
    if not isinstance(entries, dict) or not entries:
        raise NetworkConfigError("payment network config networks must be a non-empty object")
    networks: dict[str, PayNetwork] = {}
    for name, entry in entries.items():
        if not isinstance(name, str) or _NETWORK_NAME.fullmatch(name) is None:
            raise NetworkConfigError("payment network names must be lowercase URL-safe names")
        networks[name] = _parse_entry(name, entry)
    if active not in networks:
        raise NetworkConfigError(f"active payment network {active!r} is not configured")
    document = NetworkDocument(path, active, networks, value)
    _check_environment(networks[active], os.environ if environ is None else environ, path)
    return document


def _require_limits(home: Path | None) -> tuple[bool, str]:
    path = limits_path(home)
    if not os.path.lexists(path):
        return False, "mainnet limits.json is required (no default spend limits on real money)"
    try:
        _read_private_regular(path, "mainnet limits.json")
    except NetworkConfigError as exc:
        return False, str(exc)
    return True, "ready"


def _arm_status(network: PayNetwork, home: Path | None) -> tuple[bool, str]:
    path = arm_path(home)
    if not os.path.lexists(path):
        return False, "mainnet arm file is missing; run `nak pay network arm mainnet`"
    try:
        raw = _read_private_regular(path, "mainnet arm file")
        value = _load_json(raw, "mainnet arm file")
    except NetworkConfigError as exc:
        return False, str(exc)
    required = {"network", "genesis", "program", "mint", "armed_at"}
    if not isinstance(value, dict) or set(value) != required:
        return False, "mainnet arm file has the wrong shape; run `nak pay network arm mainnet`"
    expected = {
        "network": "mainnet", "genesis": network.genesis,
        "program": network.program, "mint": network.mint,
    }
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        return False, "mainnet arm file does not match network.json; run `nak pay network arm mainnet`"
    armed_at = value.get("armed_at")
    if isinstance(armed_at, bool) or not isinstance(armed_at, int) or armed_at <= 0:
        return False, "mainnet arm file has an invalid armed_at; run `nak pay network arm mainnet`"
    return True, "ready"


def mainnet_gates(network: PayNetwork, *, home: Path | None = None) -> dict[str, tuple[bool, str]]:
    configured = bool(network.rpc and network.program and network.mint)
    configured_reason = ("ready" if configured else
                         "mainnet program not configured" if not network.program else
                         "mainnet rpc not configured" if not network.rpc else
                         "mainnet mint not configured")
    return {
        "configuration": (configured, configured_reason),
        "arm": _arm_status(network, home),
        "limits": _require_limits(home),
    }


def require_mainnet_ready(network: PayNetwork, *, home: Path | None = None) -> None:
    for _name, (ready, reason) in mainnet_gates(network, home=home).items():
        if not ready:
            raise NetworkConfigError(reason)


def active_network(*, home: Path | None = None,
                   environ: Mapping[str, str] | None = None,
                   enforce_mainnet: bool = True) -> PayNetwork | None:
    document = load_document(home=home, environ=environ)
    if document is None:
        return None
    network = document.networks[document.active]
    if not network.rpc:
        raise NetworkConfigError(f"{network.name} rpc not configured")
    if not network.program:
        message = "mainnet program not configured" if network.name == "mainnet" else f"{network.name} program not configured"
        raise NetworkConfigError(message)
    if not network.mint:
        raise NetworkConfigError(f"{network.name} mint not configured")
    if network.name == "mainnet" and enforce_mainnet:
        require_mainnet_ready(network, home=home)
    return network


def require_devnet_flow(flow: str, *, home: Path | None = None,
                        environ: Mapping[str, str] | None = None) -> None:
    document = load_document(home=home, environ=environ)
    if document is not None and document.active == "mainnet":
        raise NetworkConfigError(f"{flow} is DEVNET-ONLY and is refused on mainnet")


def legacy_rpc_url(environ: Mapping[str, str]) -> str:
    """Return the pre-network.json RPC setting without changing its precedence."""
    return (environ.get("NAK_PAY_RPC", "").strip()
            or environ.get("NAKSHATRA_PAY_RPC_URL", "").strip())


def legacy_pubkey_value(environ: Mapping[str, str], name: str) -> str:
    if name not in {"NAK_PAY_PROGRAM", "NAK_PAY_MINT"}:
        raise NetworkConfigError(f"unsupported legacy payment setting {name}")
    return environ.get(name, "").strip()


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    body = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")
    try:
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(body)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        temp.unlink(missing_ok=True)


def initialize_from_environment(*, home: Path | None = None,
                                environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    path = network_path(home)
    if os.path.lexists(path):
        raise NetworkConfigError(f"refusing to overwrite existing payment network config {path}")
    rpc = legacy_rpc_url(environ)
    program = legacy_pubkey_value(environ, "NAK_PAY_PROGRAM")
    mint = legacy_pubkey_value(environ, "NAK_PAY_MINT")
    if not rpc:
        raise NetworkConfigError("NAK_PAY_RPC is required to initialize network.json")
    if not program:
        raise NetworkConfigError("NAK_PAY_PROGRAM is required to initialize network.json")
    if not mint:
        raise NetworkConfigError("NAK_PAY_MINT is required to initialize network.json")
    validate_rpc_url(rpc, "NAK_PAY_RPC")
    _pubkey(program, "NAK_PAY_PROGRAM")
    _pubkey(mint, "NAK_PAY_MINT")
    value = {
        "active": "devnet",
        "networks": {
            "devnet": {
                "rpc": rpc, "genesis": DEVNET_GENESIS_HASH,
                "program": program, "mint": mint, "explorer_cluster": "devnet",
            },
            "mainnet": {
                "rpc": "", "genesis": MAINNET_GENESIS_HASH,
                "program": "", "mint": MAINNET_MINT, "explorer_cluster": "",
            },
        },
    }
    _atomic_write_json(path, value)
    return path


def use_network(name: str, *, home: Path | None = None,
                environ: Mapping[str, str] | None = None) -> tuple[Path, Path]:
    document = load_document(home=home, environ=environ)
    if document is None:
        raise NetworkConfigError("payment network config is absent; run `nak pay network init`")
    if name not in document.networks:
        raise NetworkConfigError(f"unknown payment network {name!r}")
    target = document.networks[name]
    target_environ = os.environ if environ is None else environ
    _check_environment(target, target_environ, document.path)
    if not target.rpc:
        raise NetworkConfigError(f"{name} rpc not configured")
    if not target.program:
        raise NetworkConfigError("mainnet program not configured" if name == "mainnet" else f"{name} program not configured")
    if not target.mint:
        raise NetworkConfigError(f"{name} mint not configured")
    if name == "mainnet":
        require_mainnet_ready(target, home=home)
    original = _read_private_regular(document.path, "payment network config")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = document.path.with_name(f"{document.path.name}.{stamp}.{time.time_ns()}.bak")
    fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(original)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    updated = dict(document.raw)
    updated["active"] = name
    _atomic_write_json(document.path, updated)
    return document.path, backup


def arm_mainnet(*, home: Path | None = None,
                environ: Mapping[str, str] | None = None,
                armed_at: int | None = None) -> Path:
    document = load_document(home=home, environ=environ)
    if document is None or "mainnet" not in document.networks:
        raise NetworkConfigError("mainnet is not configured in network.json")
    network = document.networks["mainnet"]
    if not network.program:
        raise NetworkConfigError("mainnet program not configured")
    if not network.rpc:
        raise NetworkConfigError("mainnet rpc not configured")
    if not network.mint:
        raise NetworkConfigError("mainnet mint not configured")
    value = {
        "network": "mainnet", "genesis": network.genesis,
        "program": network.program, "mint": network.mint,
        "armed_at": int(time.time() if armed_at is None else armed_at),
    }
    path = arm_path(home)
    _atomic_write_json(path, value)
    return path


def disarm_mainnet(*, home: Path | None = None) -> bool:
    path = arm_path(home)
    existed = os.path.lexists(path)
    if existed:
        path.unlink()
    return existed


def installed_user_services(*, home: Path | None = None) -> tuple[str, ...]:
    home_path = Path.home() if home is None else Path(home)
    roots = (home_path / ".config" / "systemd" / "user", Path("/etc/systemd/user"))
    names = ("nak-net", "nakshatra-worker", "nakshatra-gateway")
    return tuple(name for name in names if any((root / f"{name}.service").is_file() for root in roots))
