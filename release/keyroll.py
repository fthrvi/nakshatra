#!/usr/bin/env python3
"""Create and append dual-signed Nakshatra release-key roll chains."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import releasekit as rk


def _private_signer(path: Path, expected_pubkey: str | None = None):
    private_hex = path.read_text().strip()
    public_hex = rk.public_from_private(private_hex)
    if expected_pubkey is not None and expected_pubkey != public_hex:
        raise ValueError(f"{path} does not match the supplied public key")

    def sign(message: bytes) -> dict:
        _pub, signature = rk.sign_message(message, private_hex)
        return signature

    return public_hex, sign


def _external_signer(command: str, public_hex: str | None):
    if not public_hex:
        raise ValueError("an external key-roll signer requires its --*-pubkey")
    return public_hex, lambda message: rk.external_sign_message(message, command, public_hex)


def _signer(args, side: str):
    key = getattr(args, f"{side}_key")
    command = getattr(args, f"{side}_signer")
    public = getattr(args, f"{side}_pubkey")
    return _private_signer(key, public) if key is not None else _external_signer(command, public)


def make_link(*, old_pubkey: str, old_sign, new_pubkey: str, new_sign,
              issued_at: int, reason: str) -> dict:
    statement = {
        "schema": rk.KEYROLL_SCHEMA,
        "old_pubkey": old_pubkey,
        "new_pubkey": new_pubkey,
        "issued_at": issued_at,
        "reason": reason,
    }
    message = rk.canonical_keyroll(statement)
    link = dict(statement, sig_old=old_sign(message), sig_new=new_sign(message))
    rk.validate_keyroll_chain([link])
    return link


def _write_chain(path: Path, chain: list[dict]) -> None:
    rk.validate_keyroll_chain(chain)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(chain, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _add_signer_args(parser: argparse.ArgumentParser, side: str) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(f"--{side}-key", type=Path, help=f"file containing the {side} private key")
    group.add_argument(f"--{side}-signer", help=f"external command signing canonical bytes on stdin")
    parser.add_argument(f"--{side}-pubkey", help=f"lowercase hex {side} public key (required for signer)")


def _link_from_args(args) -> dict:
    old_pub, old_sign = _signer(args, "old")
    new_pub, new_sign = _signer(args, "new")
    return make_link(old_pubkey=old_pub, old_sign=old_sign,
                     new_pubkey=new_pub, new_sign=new_sign,
                     issued_at=args.issued_at, reason=args.reason)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="create a one-link keyroll.json chain")
    append = commands.add_parser("append", help="append one link to an existing chain")
    for sub in (make, append):
        _add_signer_args(sub, "old")
        _add_signer_args(sub, "new")
        sub.add_argument("--issued-at", type=int, default=None)
        sub.add_argument("--reason", required=True)
    make.add_argument("--out", type=Path, required=True)
    append.add_argument("--chain", type=Path, required=True)
    append.add_argument("--out", type=Path,
                        help="output path (default: atomically replace --chain)")
    args = parser.parse_args(argv)
    args.issued_at = int(time.time()) if args.issued_at is None else args.issued_at
    try:
        link = _link_from_args(args)
        if args.command == "make":
            chain, output = [link], args.out
        else:
            chain = json.loads(args.chain.read_text())
            rk.validate_keyroll_chain(chain)
            if chain[-1]["new_pubkey"] != link["old_pubkey"]:
                raise ValueError("new link's old key is not the existing chain tip")
            chain = [*chain, link]
            output = args.out or args.chain
        _write_chain(output, chain)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as e:
        parser.error(str(e))
    print(f"wrote {len(chain)} key-roll link(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
