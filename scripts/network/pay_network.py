"""Single-source payment network configuration and mainnet safety gates."""
from __future__ import annotations

import fcntl
import base64
import binascii
import hashlib
import json
import os
import re
import secrets
import stat
import time
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping


DEVNET_GENESIS_HASH = "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG"
MAINNET_GENESIS_HASH = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"
DEVNET_PROGRAM = "GLWqQsSqUsDkPvkc8rxcoQUyro7ZQd5denht21NtdTTo"
DEVNET_MINT = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"
MAINNET_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
MAINNET_ARM_SCHEMA = "nakshatra.mainnet-arm.v1"

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {char: index for index, char in enumerate(_B58)}
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_NETWORK_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_EXPLORER_CLUSTER = re.compile(r"[A-Za-z0-9_-]{0,64}")
_ENTRY_FIELDS = frozenset({"rpc", "genesis", "program", "mint"})
_ARM_FIELDS = frozenset({
    "schema", "node_pubkey", "network", "genesis", "rpc", "program", "mint",
    "limits_sha256", "issued_at",
})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_LIMIT_FIELDS = frozenset({"cap_total", "days", "per_tx", "per_day"})


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
    source: bytes = b""


def network_path(home: Path | None = None) -> Path:
    return (Path.home() if home is None else Path(home)) / ".nakshatra" / "pay" / "network.json"


def arm_path(home: Path | None = None) -> Path:
    return network_path(home).with_name("mainnet.armed")


def limits_path(home: Path | None = None) -> Path:
    return network_path(home).with_name("limits.json")


def mainnet_limits_path(home: Path | None = None) -> Path:
    return network_path(home).with_name("limits.mainnet.json")


def active_limits_path(home: Path | None = None) -> Path:
    document = load_document(home=home)
    return (mainnet_limits_path(home) if document is not None and document.active == "mainnet"
            else limits_path(home))


def validate_limits(value: object) -> dict[str, int]:
    """Validate the one limits schema shared by CLI and spend paths."""
    if not isinstance(value, dict) or set(value) != _LIMIT_FIELDS:
        raise NetworkConfigError(
            "pay limits must contain cap_total, days, per_tx, and per_day")
    parsed: dict[str, int] = {}
    for name in ("cap_total", "per_tx", "per_day"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or not 0 < item < 2**64:
            raise NetworkConfigError(f"pay limit {name} must be a positive u64")
        parsed[name] = item
    days = value["days"]
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 3650:
        raise NetworkConfigError("pay limit days must be between 1 and 3650")
    parsed["days"] = days
    if parsed["per_tx"] > parsed["per_day"]:
        raise NetworkConfigError("pay limit per-tx cannot exceed per-day")
    if parsed["per_day"] > parsed["cap_total"]:
        raise NetworkConfigError("pay limit per-day cannot exceed the on-chain cap")
    return parsed


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


def _home_path(home: Path | None) -> Path:
    return Path.home() if home is None else Path(home)


def _directory_flags() -> int:
    return (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))


def _file_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_child_directory(parent_fd: int, name: str, path: Path, *,
                          create: bool) -> int | None:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            return None
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise NetworkConfigError(f"cannot inspect payment directory {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise NetworkConfigError(f"refusing symlinked payment directory {path}")
    if not stat.S_ISDIR(before.st_mode):
        raise NetworkConfigError(f"refusing {path}: payment path component is not a directory")
    if before.st_uid != os.geteuid():
        raise NetworkConfigError(
            f"refusing {path}: payment directory must be owned by uid {os.geteuid()}")
    try:
        fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise NetworkConfigError(f"cannot securely open payment directory {path}: {exc}") from exc
    opened = os.fstat(fd)
    if ((opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISDIR(opened.st_mode)):
        os.close(fd)
        raise NetworkConfigError(f"refusing {path}: payment directory changed while opening")
    return fd


@contextmanager
def _pay_directory(home: Path | None, *, create: bool) -> Iterator[int | None]:
    home_path = _home_path(home)
    try:
        home_fd = os.open(home_path, _directory_flags())
    except OSError as exc:
        raise NetworkConfigError(f"cannot securely open home directory {home_path}: {exc}") from exc
    nak_fd = None
    pay_fd = None
    try:
        nak_path = home_path / ".nakshatra"
        nak_fd = _open_child_directory(home_fd, ".nakshatra", nak_path, create=create)
        if nak_fd is None:
            yield None
            return
        pay_path = nak_path / "pay"
        pay_fd = _open_child_directory(nak_fd, "pay", pay_path, create=create)
        yield pay_fd
    finally:
        if pay_fd is not None:
            os.close(pay_fd)
        if nak_fd is not None:
            os.close(nak_fd)
        os.close(home_fd)


def _read_private_at(pay_fd: int, filename: str, path: Path, description: str,
                     *, missing_ok: bool = False) -> bytes | None:
    try:
        before = os.stat(filename, dir_fd=pay_fd, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise NetworkConfigError(f"cannot read {description} at {path}: file is missing")
    except OSError as exc:
        raise NetworkConfigError(f"cannot inspect {description} at {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise NetworkConfigError(
            f"refusing {path}: {description} must be a regular file, not a symlink")
    try:
        fd = os.open(filename, _file_flags(), dir_fd=pay_fd)
    except OSError as exc:
        raise NetworkConfigError(f"cannot securely open {description} at {path}: {exc}") from exc
    try:
        opened = os.fstat(fd)
        if ((opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or not stat.S_ISREG(opened.st_mode)):
            raise NetworkConfigError(f"refusing {path}: {description} changed while opening")
        if opened.st_uid != os.geteuid():
            raise NetworkConfigError(
                f"refusing {path}: {description} must be owned by uid {os.geteuid()}")
        if stat.S_IMODE(opened.st_mode) != 0o600:
            raise NetworkConfigError(f"refusing {path}: {description} must have mode 0600")
        chunks = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        after_path = os.stat(filename, dir_fd=pay_fd, follow_symlinks=False)
    except OSError as exc:
        raise NetworkConfigError(f"refusing {path}: {description} changed while reading") from exc
    identity = (opened.st_dev, opened.st_ino)
    if ((after_path.st_dev, after_path.st_ino) != identity
            or (after_fd.st_size, after_fd.st_mtime_ns)
            != (opened.st_size, opened.st_mtime_ns)):
        raise NetworkConfigError(f"refusing {path}: {description} changed while reading")
    return b"".join(chunks)


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


def _load_document_at(pay_fd: int, *, home: Path | None,
                      environ: Mapping[str, str] | None) -> NetworkDocument | None:
    path = network_path(home)
    raw_bytes = _read_private_at(
        pay_fd, path.name, path, "payment network config", missing_ok=True)
    if raw_bytes is None:
        return None
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
    document = NetworkDocument(path, active, networks, value, raw_bytes)
    _check_environment(networks[active], os.environ if environ is None else environ, path)
    return document


def load_document(*, home: Path | None = None,
                  environ: Mapping[str, str] | None = None) -> NetworkDocument | None:
    """Load and strictly validate network.json, returning ``None`` only when absent."""
    # Preserve the legacy path exactly when network.json does not exist.  In
    # particular, old installations may have symlinked or foreign-owned parent
    # directories; those become a refusal only once they opt into this file.
    if not os.path.lexists(network_path(home)):
        return None
    with _pay_directory(home, create=False) as pay_fd:
        if pay_fd is None:
            return None
        return _load_document_at(pay_fd, home=home, environ=environ)


def _require_limits_at(pay_fd: int | None, home: Path | None) -> tuple[bool, str]:
    path = mainnet_limits_path(home)
    if pay_fd is None:
        return False, "mainnet limits.mainnet.json is required (no default spend limits on real money)"
    try:
        raw = _read_private_at(
            pay_fd, path.name, path, "mainnet limits.mainnet.json", missing_ok=True)
    except NetworkConfigError as exc:
        return False, str(exc)
    if raw is None:
        return False, "mainnet limits.mainnet.json is required (no default spend limits on real money)"
    try:
        validate_limits(_load_json(raw, f"mainnet limits {path}"))
    except NetworkConfigError as exc:
        return False, str(exc)
    return True, "ready"


def _require_limits(home: Path | None) -> tuple[bool, str]:
    with _pay_directory(home, create=False) as pay_fd:
        return _require_limits_at(pay_fd, home)


def _canonical(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _node_public_key(home: Path | None) -> str:
    from nakshatra_auth import load_worker_key

    key_path = _home_path(home) / ".nakshatra" / "keys" / "worker.ed25519"
    try:
        _seed, public = load_worker_key(key_path)
    except (OSError, ValueError, PermissionError) as exc:
        raise NetworkConfigError(f"cannot load this node's signing key: {exc}") from exc
    if _HEX64.fullmatch(public) is None:  # defensive: load_worker_key already guarantees this
        raise NetworkConfigError("this node's signing public key is malformed")
    return public


def _installed_release_public(home: Path | None) -> str:
    default = _home_path(home) / ".nakshatra-node"
    root = Path(os.path.expanduser(os.environ.get("NAK_NODE_PREFIX", str(default))))
    root_fd = None
    fd = None
    try:
        root_fd = os.open(root, _directory_flags())
        fd = os.open("config.json", _file_flags(), dir_fd=root_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise NetworkConfigError(
                "installed release config must be an owner-owned regular file")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, 1_048_577 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 1_048_576:
                raise NetworkConfigError("installed release config is too large")
        value = _load_json(b"".join(chunks), f"installed release config {root / 'config.json'}")
    except NetworkConfigError:
        raise
    except OSError as exc:
        raise NetworkConfigError(
            f"cannot read installed release config {root / 'config.json'}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)
    public = value.get("pubkey") if isinstance(value, dict) else None
    if not isinstance(public, str) or _HEX64.fullmatch(public) is None:
        raise NetworkConfigError("installed release config has no valid pinned release public key")
    return public


def _decode_signed_arm(value: object) -> tuple[dict, bytes, dict]:
    if not isinstance(value, dict) or set(value) != {"statement_b64", "signature"}:
        raise NetworkConfigError("mainnet arm file has the wrong signed-file shape")
    encoded, signature = value["statement_b64"], value["signature"]
    if not isinstance(encoded, str):
        raise NetworkConfigError("mainnet arm statement_b64 must be a string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise NetworkConfigError("mainnet arm statement_b64 is invalid") from exc
    if len(raw) > 4096:
        raise NetworkConfigError("mainnet arm statement is too large")
    statement = _load_json(raw, "mainnet arm statement")
    if (not isinstance(statement, dict) or set(statement) != _ARM_FIELDS
            or statement.get("schema") != MAINNET_ARM_SCHEMA):
        raise NetworkConfigError("mainnet arm statement has malformed fields")
    if _canonical(statement) != raw:
        raise NetworkConfigError("mainnet arm statement is not canonical JSON")
    if not isinstance(signature, dict) or set(signature) != {"alg", "keyid", "value"}:
        raise NetworkConfigError("mainnet arm signature has malformed fields")
    return statement, raw, signature


def _verify_release_signature(signature: dict, public_hex: str, raw: bytes) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import ed25519

    if signature.get("alg") != "Ed25519" or signature.get("keyid") != public_hex[:16]:
        raise NetworkConfigError(
            "mainnet arm signature does not name the installed pinned release key")
    try:
        encoded = signature.get("value")
        if not isinstance(encoded, str):
            raise ValueError("signature value is not a string")
        raw_signature = base64.b64decode(encoded, validate=True)
        if len(raw_signature) != 64:
            raise ValueError("signature is not 64 bytes")
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex)).verify(
            raw_signature, raw)
    except (ValueError, binascii.Error, InvalidSignature) as exc:
        raise NetworkConfigError(
            "mainnet arm signature does not verify under the installed release key") from exc


def _verify_arm_document(value: object, *, network: PayNetwork, home: Path | None,
                         pay_fd: int, limits_raw: bytes | None = None) -> dict:
    statement, raw, signature = _decode_signed_arm(value)
    public_hex = _installed_release_public(home)
    _verify_release_signature(signature, public_hex, raw)
    node_public = _node_public_key(home)
    expected = {
        "node_pubkey": node_public,
        "network": "mainnet",
        "genesis": network.genesis,
        "rpc": network.rpc,
        "program": network.program,
        "mint": network.mint,
    }
    if statement.get("node_pubkey") != node_public:
        raise NetworkConfigError("mainnet arm statement is for a different node signing key")
    if any(statement.get(key) != wanted for key, wanted in expected.items()):
        raise NetworkConfigError(
            "mainnet arm file does not match the active mainnet entry")
    if (not isinstance(statement.get("rpc"), str) or len(statement["rpc"]) > 512
            or not statement["rpc"].startswith("https://")):
        raise NetworkConfigError("mainnet arm statement rpc must be an HTTPS URL")
    issued_at = statement.get("issued_at")
    if isinstance(issued_at, bool) or not isinstance(issued_at, int) or issued_at <= 0:
        raise NetworkConfigError("mainnet arm statement issued_at must be a positive integer")
    limits = mainnet_limits_path(home)
    if limits_raw is None:
        limits_raw = _read_private_at(
            pay_fd, limits.name, limits, "mainnet limits.mainnet.json", missing_ok=True)
    if limits_raw is None:
        raise NetworkConfigError(
            "mainnet limits.mainnet.json is required (no default spend limits on real money)")
    validate_limits(_load_json(limits_raw, f"mainnet limits {limits}"))
    expected_hash = hashlib.sha256(limits_raw).hexdigest()
    if statement.get("limits_sha256") != expected_hash:
        raise NetworkConfigError("mainnet arm statement does not match limits.mainnet.json")
    return statement


def verify_mainnet_limits_bytes(limits_raw: bytes, *, home: Path | None = None) -> None:
    """Refuse unless these exact limits bytes are the ones the signed arm names.

    The spend path calls this with the bytes it is about to parse, so swapping
    limits.mainnet.json after the gate check can never take effect.
    """
    document = load_document(home=home)
    if document is None or document.active != "mainnet":
        raise NetworkConfigError("mainnet limits are only valid while mainnet is active")
    network = document.networks["mainnet"]
    path = arm_path(home)
    with _pay_directory(home, create=False) as pay_fd:
        if pay_fd is None:
            raise NetworkConfigError("mainnet arm file is missing; run `nak pay network arm mainnet`")
        raw = _read_private_at(pay_fd, path.name, path, "mainnet arm file", missing_ok=True)
        if raw is None:
            raise NetworkConfigError("mainnet arm file is missing; run `nak pay network arm mainnet`")
        _verify_arm_document(_load_json(raw, "mainnet arm file"), network=network,
                             home=home, pay_fd=pay_fd, limits_raw=limits_raw)


def _read_source_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise NetworkConfigError(f"cannot open signed mainnet arm file {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise NetworkConfigError("signed mainnet arm input must be a regular file")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, 16385 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 16384:
                raise NetworkConfigError("signed mainnet arm input is too large")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _arm_status_at(network: PayNetwork, home: Path | None,
                   pay_fd: int | None) -> tuple[bool, str]:
    path = arm_path(home)
    if pay_fd is None:
        return False, "mainnet arm file is missing; run `nak pay network arm mainnet`"
    try:
        raw = _read_private_at(
            pay_fd, path.name, path, "mainnet arm file", missing_ok=True)
        if raw is None:
            return False, "mainnet arm file is missing; run `nak pay network arm mainnet`"
        value = _load_json(raw, "mainnet arm file")
    except NetworkConfigError as exc:
        return False, str(exc)
    try:
        _verify_arm_document(value, network=network, home=home, pay_fd=pay_fd)
    except NetworkConfigError as exc:
        return False, str(exc)
    return True, "ready"


def _arm_status(network: PayNetwork, home: Path | None) -> tuple[bool, str]:
    with _pay_directory(home, create=False) as pay_fd:
        return _arm_status_at(network, home, pay_fd)


def mainnet_gates(network: PayNetwork, *, home: Path | None = None) -> dict[str, tuple[bool, str]]:
    configured = bool(network.rpc and network.program and network.mint)
    configured_reason = ("ready" if configured else
                         "mainnet program not configured" if not network.program else
                         "mainnet rpc not configured" if not network.rpc else
                         "mainnet mint not configured")
    with _pay_directory(home, create=False) as pay_fd:
        return {
            "configuration": (configured, configured_reason),
            "arm": _arm_status_at(network, home, pay_fd),
            "limits": _require_limits_at(pay_fd, home),
        }


def _mainnet_gates_at(network: PayNetwork, *, home: Path | None,
                      pay_fd: int) -> dict[str, tuple[bool, str]]:
    configured = bool(network.rpc and network.program and network.mint)
    configured_reason = ("ready" if configured else
                         "mainnet program not configured" if not network.program else
                         "mainnet rpc not configured" if not network.rpc else
                         "mainnet mint not configured")
    return {
        "configuration": (configured, configured_reason),
        "arm": _arm_status_at(network, home, pay_fd),
        "limits": _require_limits_at(pay_fd, home),
    }


def require_mainnet_ready(network: PayNetwork, *, home: Path | None = None) -> None:
    for _name, (ready, reason) in mainnet_gates(network, home=home).items():
        if not ready:
            raise NetworkConfigError(reason)


def _require_mainnet_ready_at(network: PayNetwork, *, home: Path | None,
                              pay_fd: int) -> None:
    for _name, (ready, reason) in _mainnet_gates_at(
            network, home=home, pay_fd=pay_fd).items():
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


def _write_all(fd: int, body: bytes) -> None:
    view = memoryview(body)
    while view:
        view = view[os.write(fd, view):]


def _write_exclusive_at(pay_fd: int, filename: str, body: bytes) -> None:
    flags = (os.O_CREAT | os.O_EXCL | os.O_WRONLY
             | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    fd = os.open(filename, flags, 0o600, dir_fd=pay_fd)
    try:
        os.fchmod(fd, 0o600)
        _write_all(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json_at(pay_fd: int, path: Path, value: dict) -> None:
    temp = f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    body = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")
    try:
        _write_exclusive_at(pay_fd, temp, body)
        os.replace(temp, path.name, src_dir_fd=pay_fd, dst_dir_fd=pay_fd)
        os.fsync(pay_fd)
    finally:
        try:
            os.unlink(temp, dir_fd=pay_fd)
        except FileNotFoundError:
            pass
        else:
            os.fsync(pay_fd)


def _atomic_write_bytes_at(pay_fd: int, path: Path, body: bytes) -> None:
    temp = f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    try:
        _write_exclusive_at(pay_fd, temp, body)
        os.replace(temp, path.name, src_dir_fd=pay_fd, dst_dir_fd=pay_fd)
        os.fsync(pay_fd)
    finally:
        try:
            os.unlink(temp, dir_fd=pay_fd)
        except FileNotFoundError:
            pass
        else:
            os.fsync(pay_fd)


@contextmanager
def _mutation_lock(home: Path | None) -> Iterator[int]:
    with _pay_directory(home, create=True) as pay_fd:
        assert pay_fd is not None
        name = ".network.lock"
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            lock_fd = os.open(name, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=pay_fd)
            os.fchmod(lock_fd, 0o600)
        except FileExistsError:
            try:
                lock_fd = os.open(name, flags, dir_fd=pay_fd)
            except OSError as exc:
                raise NetworkConfigError(f"cannot securely open payment network lock: {exc}") from exc
        try:
            info = os.fstat(lock_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                raise NetworkConfigError("payment network lock must be a regular owner file")
            if stat.S_IMODE(info.st_mode) != 0o600:
                raise NetworkConfigError("payment network lock must have mode 0600")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            current = os.stat(name, dir_fd=pay_fd, follow_symlinks=False)
            if ((current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)
                    or stat.S_ISLNK(current.st_mode)):
                raise NetworkConfigError("payment network lock changed while locking")
            yield pay_fd
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


def initialize_from_environment(*, home: Path | None = None,
                                environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    path = network_path(home)
    with _mutation_lock(home) as pay_fd:
        try:
            os.stat(path.name, dir_fd=pay_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise NetworkConfigError(
                f"refusing to overwrite existing payment network config {path}")
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
        _atomic_write_json_at(pay_fd, path, value)
    return path


def use_network(name: str, *, home: Path | None = None,
                environ: Mapping[str, str] | None = None) -> tuple[Path, Path]:
    with _mutation_lock(home) as pay_fd:
        document = _load_document_at(pay_fd, home=home, environ=environ)
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
            raise NetworkConfigError(
                "mainnet program not configured" if name == "mainnet"
                else f"{name} program not configured")
        if not target.mint:
            raise NetworkConfigError(f"{name} mint not configured")
        if name == "mainnet":
            _require_mainnet_ready_at(target, home=home, pay_fd=pay_fd)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup = document.path.with_name(
            f"{document.path.name}.{stamp}.{time.time_ns()}.bak")
        _write_exclusive_at(pay_fd, backup.name, document.source)
        os.fsync(pay_fd)
        updated = dict(document.raw)
        updated["active"] = name
        _atomic_write_json_at(pay_fd, document.path, updated)
        return document.path, backup


def arm_request_info(*, home: Path | None = None,
                     environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the exact values the release-key arm requester must bind."""
    with _mutation_lock(home) as pay_fd:
        document = _load_document_at(pay_fd, home=home, environ=environ)
        if document is None or "mainnet" not in document.networks:
            raise NetworkConfigError("mainnet is not configured in network.json")
        network = document.networks["mainnet"]
        if not network.program:
            raise NetworkConfigError("mainnet program not configured")
        if not network.rpc:
            raise NetworkConfigError("mainnet rpc not configured")
        if not network.mint:
            raise NetworkConfigError("mainnet mint not configured")
        parsed = urllib.parse.urlsplit(network.rpc)
        if parsed.scheme != "https":
            raise NetworkConfigError("mainnet rpc must use HTTPS before an arm can be requested")
        limits = mainnet_limits_path(home)
        raw = _read_private_at(
            pay_fd, limits.name, limits, "mainnet limits.mainnet.json", missing_ok=False)
        assert raw is not None
        validate_limits(_load_json(raw, f"mainnet limits {limits}"))
        return {
            "node_pubkey": _node_public_key(home),
            "rpc": network.rpc,
            "program": network.program,
            "limits": str(limits),
            "limits_sha256": hashlib.sha256(raw).hexdigest(),
        }


def arm_mainnet(source: Path, *, home: Path | None = None,
                environ: Mapping[str, str] | None = None) -> Path:
    """Verify and install a release-key-signed mainnet arm file."""
    raw = _read_source_file(Path(source))
    value = _load_json(raw, f"signed mainnet arm file {source}")
    with _mutation_lock(home) as pay_fd:
        document = _load_document_at(pay_fd, home=home, environ=environ)
        if document is None or "mainnet" not in document.networks:
            raise NetworkConfigError("mainnet is not configured in network.json")
        network = document.networks["mainnet"]
        if not network.rpc or not network.program or not network.mint:
            raise NetworkConfigError("mainnet rpc, program, and mint must be configured")
        _verify_arm_document(value, network=network, home=home, pay_fd=pay_fd)
        path = arm_path(home)
        _atomic_write_bytes_at(pay_fd, path, raw)
        return path


def disarm_mainnet(*, home: Path | None = None) -> bool:
    path = arm_path(home)
    with _mutation_lock(home) as pay_fd:
        try:
            os.stat(path.name, dir_fd=pay_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        os.unlink(path.name, dir_fd=pay_fd)
        os.fsync(pay_fd)
        return True


def installed_user_services(*, home: Path | None = None) -> tuple[str, ...]:
    home_path = Path.home() if home is None else Path(home)
    roots = (home_path / ".config" / "systemd" / "user", Path("/etc/systemd/user"))
    names = ("nak-net", "nakshatra-worker", "nakshatra-gateway")
    return tuple(name for name in names if any((root / f"{name}.service").is_file() for root in roots))
