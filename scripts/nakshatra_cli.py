"""nakshatra-cli — operator-side scaffolding for the worker.

Today this is a minimal subcommand surface; the broader operator-UX
sprint named in the 2026-05-20 retro will flesh out auth/operator
keygen / install / sign subcommands modelled on sthambha-cli. The
single subcommand shipped here is the one Phase 2 of the SPKI sprint
named as a small enough win to fold in:

  nakshatra-cli tls fingerprint [PATH]

Print the SHA-256(SPKI) of an existing cert without booting the
worker. Useful when an operator needs to verify what hash a worker
is about to declare, or to compare against the pillar's /peers
projection.

Run via:

    python -m scripts.nakshatra_cli tls fingerprint
    python scripts/nakshatra_cli.py tls fingerprint /path/to/cert.pem
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Same script-path pattern other scripts/ modules use so the file
# works whether invoked as `python scripts/nakshatra_cli.py` or
# `python -m scripts.nakshatra_cli`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nakshatra_tls as _nt  # noqa: E402


def cmd_tls_fingerprint(args: argparse.Namespace) -> int:
    """Print the SPKI SHA-256 hex of an existing cert. Defaults to
    ~/.nakshatra/tls/worker-cert.pem. Exits 2 if the cert is missing."""
    cert_path = (Path(args.cert).expanduser() if args.cert
                  else _nt.DEFAULT_TLS_DIR / _nt.CERT_FILENAME)
    if not cert_path.exists():
        print(f"  ✗ cert not found: {cert_path}", file=sys.stderr)
        print(f"    Use --cert PATH or start the worker with TLS "
              f"enabled to generate one.", file=sys.stderr)
        return 2
    try:
        fp = _nt.compute_spki_hash(cert_path)
    except Exception as e:
        print(f"  ✗ failed to read cert: {e}", file=sys.stderr)
        return 2
    print(fp)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="nakshatra-cli",
        description="Nakshatra worker operator tooling",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_tls = sub.add_parser("tls", help="TLS cert tooling")
    sp_tls_sub = sp_tls.add_subparsers(dest="subcmd", required=True)

    sp_fp = sp_tls_sub.add_parser(
        "fingerprint",
        help="Print SHA-256(SPKI) of a cert file (default: worker cert)",
    )
    sp_fp.add_argument(
        "--cert", "-c",
        help="Path to the cert PEM (default: ~/.nakshatra/tls/worker-cert.pem)",
    )
    sp_fp.set_defaults(func=cmd_tls_fingerprint)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
