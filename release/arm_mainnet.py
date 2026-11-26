#!/usr/bin/env python3
"""Request a release-key-signed, node-bound Nakshatra mainnet arm statement."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import time
import urllib.parse
from pathlib import Path

import broker_signer
import releasekit as rk


SCHEMA = "nakshatra.mainnet-arm.v1"
GENESIS = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"
MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _program(value: str) -> str:
    number = 0
    try:
        for char in value:
            number = number * 58 + _B58.index(char)
    except ValueError as exc:
        raise ValueError("--program must be a base58 32-byte public key") from exc
    zeroes = len(value) - len(value.lstrip("1"))
    raw = b"\0" * zeroes + (number.to_bytes((number.bit_length() + 7) // 8, "big")
                             if number else b"")
    if len(raw) != 32:
        raise ValueError("--program must be a base58 32-byte public key")
    return value


def _rpc(value: str) -> str:
    if len(value) > 512:
        raise ValueError("--rpc must be at most 512 characters")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--rpc is not a valid URL: {exc}") from exc
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.fragment
            or (port is not None and not 1 <= port <= 65535)):
        raise ValueError("--rpc must be an HTTPS URL without credentials or a fragment")
    return value


def _hash_regular(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("--limits must name a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


def build_statement(*, node_pubkey: str, rpc: str, program: str, limits: Path,
                    issued_at: int | None = None) -> dict:
    if _HEX64.fullmatch(node_pubkey) is None:
        raise ValueError("--node-pubkey must be 64 lowercase hex characters")
    when = int(time.time()) if issued_at is None else issued_at
    if isinstance(when, bool) or not isinstance(when, int) or when <= 0:
        raise ValueError("issued_at must be a positive integer Unix timestamp")
    return {
        "schema": SCHEMA,
        "node_pubkey": node_pubkey,
        "network": "mainnet",
        "genesis": GENESIS,
        "rpc": _rpc(rpc),
        "program": _program(program),
        "mint": MINT,
        "limits_sha256": _hash_regular(Path(limits)),
        "issued_at": when,
    }


def _write_private(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(temp, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(body)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp, path)
        os.chmod(path, 0o600, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-pubkey", required=True)
    parser.add_argument("--rpc", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--limits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--release-broker", type=Path,
                        help="directory containing release-broker.json and release.pub")
    args = parser.parse_args(argv)
    try:
        statement = build_statement(
            node_pubkey=args.node_pubkey, rpc=args.rpc, program=args.program,
            limits=args.limits)
        raw = rk.canonical(statement)
        signature = broker_signer.sign_mainnet_arm(statement, args.release_broker)
        public_hex = broker_signer.load_release_public(args.release_broker)
        if not rk.verify_signature(signature, public_hex, raw):
            raise broker_signer.BrokerSigningError(
                "release broker returned a mainnet arm signature that does not verify")
        _write_private(args.out, {
            "statement_b64": base64.b64encode(raw).decode("ascii"),
            "signature": signature,
        })
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"wrote signed mainnet arm statement to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
