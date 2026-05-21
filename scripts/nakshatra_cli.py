"""nakshatra-cli — operator-side tooling for the worker.

Subcommands (mirrors `sthambha-cli` shape where it makes sense):

  nakshatra-cli auth keygen [--name worker]
  nakshatra-cli auth show-pubkey [--name worker]
  nakshatra-cli auth sign --method M --path P [--body B] [--name worker]
  nakshatra-cli operator install (--pubkey HEX | --from-key NAME)
  nakshatra-cli tls keygen [--hostname H] [--output-dir D] [--force]
  nakshatra-cli tls fingerprint [--cert PATH]

Run via:

    python scripts/nakshatra_cli.py auth keygen
    python -m scripts.nakshatra_cli tls fingerprint
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Same script-path pattern other scripts/ modules use so the file
# works whether invoked as `python scripts/nakshatra_cli.py` or
# `python -m scripts.nakshatra_cli`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nakshatra_auth as _auth  # noqa: E402
import nakshatra_tls as _nt  # noqa: E402


# Worker-side conventions (Phase B + Phase C3):
# - Worker's Ed25519 keypair lives at ~/.nakshatra/keys/worker.{ed25519,pub.hex}
# - Operator pubkey (for POST /slice gating) at ~/.nakshatra/keys/operator.pub.hex
# - TLS cert at ~/.nakshatra/tls/worker-cert.pem (set by nakshatra_tls)
KEYS_DIR = Path.home() / ".nakshatra" / "keys"
DEFAULT_KEY_NAME = "worker"
OPERATOR_PUBKEY_PATH = KEYS_DIR / "operator.pub.hex"


def _keys_dir() -> Path:
    KEYS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    return KEYS_DIR


# ── auth ────────────────────────────────────────────────────────────


def cmd_auth_keygen(args: argparse.Namespace) -> int:
    """Generate an Ed25519 keypair into ~/.nakshatra/keys/<name>.*."""
    keys_dir = _keys_dir()
    name = args.name
    priv_path = keys_dir / f"{name}.ed25519"
    pub_path = keys_dir / f"{name}.pub.hex"
    if priv_path.exists() and not args.force:
        print(f"  ✗ key already exists: {priv_path}", file=sys.stderr)
        print(f"    pass --force to rotate (BREAKS any pillar-pinned "
              f"public_key_hex for this node)", file=sys.stderr)
        return 2
    priv, pub_hex = _auth.generate_keypair()
    # O_CREAT|O_TRUNC because --force may overwrite. Mode 0o600 baked in
    # so the private key isn't world-readable from the instant it exists.
    fd = os.open(str(priv_path),
                  os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, priv)
    finally:
        os.close(fd)
    pub_path.write_text(pub_hex + "\n")
    out = {
        "name": name,
        "private_key_path": str(priv_path),
        "public_key_hex_path": str(pub_path),
        "public_key_hex": pub_hex,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"  ✓ generated {name}")
        print(f"    private: {priv_path}  (mode 600)")
        print(f"    public:  {pub_path}")
        print(f"    pubkey:  {pub_hex}")
    return 0


def cmd_auth_show_pubkey(args: argparse.Namespace) -> int:
    """Print the public-key hex for an existing key. Output goes to
    stdout with no decoration so wrapping scripts can capture it
    directly (`pub=$(nakshatra-cli auth show-pubkey)`)."""
    priv_path = _keys_dir() / f"{args.name}.ed25519"
    if not priv_path.exists():
        print(f"  ✗ key not found: {priv_path}", file=sys.stderr)
        print(f"    generate one with: nakshatra-cli auth keygen "
              f"--name {args.name}", file=sys.stderr)
        return 2
    priv = priv_path.read_bytes()
    print(_auth.public_key_hex_from_private(priv))
    return 0


def cmd_auth_sign(args: argparse.Namespace) -> int:
    """Sign a request and print the Authorization header value. Operator
    parity tool — lets you craft a signed curl invocation by hand:

        $ HEADER=$(nakshatra-cli auth sign --method POST --path /peer \\
              --body '{"node_id":"w1"}')
        $ curl -H "Authorization: $HEADER" ...
    """
    priv_path = _keys_dir() / f"{args.name}.ed25519"
    if not priv_path.exists():
        print(f"  ✗ key not found: {priv_path}", file=sys.stderr)
        return 2
    priv = priv_path.read_bytes()
    body = args.body.encode() if args.body else b""
    keyid = args.keyid or args.name
    header, ts = _auth.build_signed_envelope(
        priv, keyid, args.method.upper(), args.path, body,
    )
    if args.json:
        print(json.dumps({
            "header": header,
            "ts": ts,
            "method": args.method.upper(),
            "path": args.path,
            "body": args.body,
            "keyid": keyid,
        }, indent=2))
    else:
        print(header)
    return 0


# ── operator ────────────────────────────────────────────────────────


def cmd_operator_install(args: argparse.Namespace) -> int:
    """Install an operator pubkey for this worker. Writes
    ~/.nakshatra/keys/operator.pub.hex — the path POST /slice's tier
    check (Phase C3) reads at startup.

    Two ways to source the pubkey:
      --pubkey HEX     : pass the 64-char hex directly
      --from-key NAME  : read pubkey from ~/.nakshatra/keys/NAME.ed25519
                         (convenient when the operator generated the key
                          on this machine with `auth keygen`)
    """
    if not args.pubkey and not args.from_key:
        print("  ✗ supply either --pubkey HEX or --from-key NAME",
              file=sys.stderr)
        return 2
    if args.pubkey and args.from_key:
        print("  ✗ --pubkey and --from-key are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.from_key:
        priv_path = _keys_dir() / f"{args.from_key}.ed25519"
        if not priv_path.exists():
            print(f"  ✗ key not found: {priv_path}", file=sys.stderr)
            return 2
        priv = priv_path.read_bytes()
        pub_hex = _auth.public_key_hex_from_private(priv)
    else:
        pub_hex = args.pubkey.strip().lower()
        if len(pub_hex) != 64:
            print(f"  ✗ pubkey must be 64 hex chars (got {len(pub_hex)})",
                  file=sys.stderr)
            return 2
        try:
            bytes.fromhex(pub_hex)
        except ValueError:
            print(f"  ✗ pubkey is not valid hex", file=sys.stderr)
            return 2
    # Atomic write at 0o600 so a partial write doesn't leave a
    # readable-by-others bad pubkey behind.
    KEYS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = OPERATOR_PUBKEY_PATH.with_suffix(".hex.tmp")
    fd = os.open(str(tmp),
                  os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (pub_hex + "\n").encode())
    finally:
        os.close(fd)
    os.replace(tmp, OPERATOR_PUBKEY_PATH)
    if args.json:
        print(json.dumps({
            "operator_pubkey_path": str(OPERATOR_PUBKEY_PATH),
            "public_key_hex": pub_hex,
        }, indent=2))
    else:
        print(f"  ✓ operator pubkey installed")
        print(f"    path:    {OPERATOR_PUBKEY_PATH}  (mode 600)")
        print(f"    pubkey:  {pub_hex}")
        print(f"    Worker reads this on startup (Phase C3). Restart any "
              f"running worker for the change to take effect.")
    return 0


# ── tls ─────────────────────────────────────────────────────────────


def cmd_tls_keygen(args: argparse.Namespace) -> int:
    """Generate a self-signed TLS cert + key for the worker gRPC server.
    Default output is ~/.nakshatra/tls/ — the path ensure_cert reads on
    worker boot. Refuses to overwrite an existing pair unless --force
    (rotating breaks every peer that's pinned the old SPKI hash)."""
    out_dir = (Path(args.output_dir).expanduser()
               if args.output_dir else None)
    try:
        cert_path, key_path = _nt.generate_self_signed_cert(
            hostname=args.hostname,
            output_dir=out_dir,
            overwrite=args.force,
        )
    except FileExistsError as e:
        print(f"  ✗ {e}", file=sys.stderr)
        print(f"    pass --force to rotate (WILL break peers that "
              f"pinned the prior SPKI)", file=sys.stderr)
        return 2
    spki = _nt.compute_spki_hash(cert_path)
    if args.json:
        print(json.dumps({
            "cert_path": str(cert_path),
            "key_path": str(key_path),
            "spki_sha256_hex": spki,
        }, indent=2))
    else:
        print(f"  ✓ TLS cert generated")
        print(f"    cert: {cert_path}")
        print(f"    key:  {key_path}  (mode 600)")
        print(f"    SPKI SHA-256: {spki}")
        print()
        print(f"  The worker will pick this cert up on next boot via "
              f"ensure_cert. To pre-announce this SPKI to the pillar, "
              f"include it in your registration body's peer_spki_hash "
              f"field (handled automatically by worker.py when "
              f"--pillar-url is set).")
    return 0


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


# ── parser ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="nakshatra-cli",
        description="Nakshatra worker operator tooling",
    )
    ap.add_argument("--json", action="store_true",
                    help="machine-readable JSON output (where applicable)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # auth
    sp_auth = sub.add_parser("auth", help="Ed25519 keypair tooling")
    sp_auth_sub = sp_auth.add_subparsers(dest="subcmd", required=True)

    sp_auth_kg = sp_auth_sub.add_parser(
        "keygen", help="generate an Ed25519 keypair into ~/.nakshatra/keys/",
    )
    sp_auth_kg.add_argument("--name", default=DEFAULT_KEY_NAME,
                              help=f"key name (filename stem); default "
                                   f"'{DEFAULT_KEY_NAME}' (matches the "
                                   f"convention worker.py expects)")
    sp_auth_kg.add_argument("--force", action="store_true",
                              help="overwrite existing key (BREAKS pillar "
                                   "TOFU-locked public_key_hex for this node)")
    sp_auth_kg.set_defaults(func=cmd_auth_keygen)

    sp_auth_sp = sp_auth_sub.add_parser(
        "show-pubkey", help="print the public-key hex for an installed key",
    )
    sp_auth_sp.add_argument("--name", default=DEFAULT_KEY_NAME)
    sp_auth_sp.set_defaults(func=cmd_auth_show_pubkey)

    sp_auth_sg = sp_auth_sub.add_parser(
        "sign", help="sign a request and print the Authorization header value",
    )
    sp_auth_sg.add_argument("--name", default=DEFAULT_KEY_NAME)
    sp_auth_sg.add_argument("--keyid", default="",
                              help="keyid to embed in the header; defaults to --name")
    sp_auth_sg.add_argument("--method", required=True,
                              help="HTTP method or gRPC method-token "
                                   "(POST / GET / STREAM)")
    sp_auth_sg.add_argument("--path", required=True,
                              help="request path or gRPC method path "
                                   "(e.g. /peer or /nakshatra.Nakshatra/Forward)")
    sp_auth_sg.add_argument("--body", default="",
                              help="request body as a string (default empty)")
    sp_auth_sg.set_defaults(func=cmd_auth_sign)

    # operator
    sp_op = sub.add_parser("operator", help="operator-key installation")
    sp_op_sub = sp_op.add_subparsers(dest="subcmd", required=True)

    sp_op_inst = sp_op_sub.add_parser(
        "install",
        help="install an operator pubkey at ~/.nakshatra/keys/operator.pub.hex",
    )
    sp_op_inst.add_argument("--pubkey", default="",
                              help="operator public key hex (64 chars)")
    sp_op_inst.add_argument("--from-key", default="",
                              help="read pubkey from "
                                   "~/.nakshatra/keys/<name>.ed25519")
    sp_op_inst.set_defaults(func=cmd_operator_install)

    # tls
    sp_tls = sub.add_parser("tls", help="TLS cert tooling")
    sp_tls_sub = sp_tls.add_subparsers(dest="subcmd", required=True)

    sp_tls_kg = sp_tls_sub.add_parser(
        "keygen", help="generate a self-signed cert in ~/.nakshatra/tls/",
    )
    sp_tls_kg.add_argument("--hostname", default="nakshatra.local",
                              help="CN + SAN DNS entry (default nakshatra.local)")
    sp_tls_kg.add_argument("--output-dir", default="",
                              help="output directory (default ~/.nakshatra/tls)")
    sp_tls_kg.add_argument("--force", action="store_true",
                              help="overwrite existing cert (BREAKS pinned peers)")
    sp_tls_kg.set_defaults(func=cmd_tls_keygen)

    sp_tls_fp = sp_tls_sub.add_parser(
        "fingerprint",
        help="Print SHA-256(SPKI) of a cert file (default: worker cert)",
    )
    sp_tls_fp.add_argument(
        "--cert", "-c",
        help="Path to the cert PEM (default: ~/.nakshatra/tls/worker-cert.pem)",
    )
    sp_tls_fp.set_defaults(func=cmd_tls_fingerprint)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
