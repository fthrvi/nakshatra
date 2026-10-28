#!/usr/bin/env python3
"""Refresh latest.json for an existing release through the approval broker."""
from __future__ import annotations

import argparse
from pathlib import Path

import build


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--channel", default="canary", choices=["canary", "stable"])
    parser.add_argument("--release-broker", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path.home() / "nakshatra-dist")
    args = parser.parse_args(argv)
    try:
        path = build.refresh_latest_pointer(
            args.version, args.channel, args.out, args.release_broker)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"refreshed {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
