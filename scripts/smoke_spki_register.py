"""Cluster smoke harness — register a peer with TLS cert + SPKI declaration.

Stripped-down "worker" for the 2026-05-21 SPKI federation cluster smoke.
Generates an Ed25519 worker key + a self-signed TLS cert, registers with
the pillar declaring peer_spki_hash, and starts a minimal TLS listener
so peers can TLS-probe this host.

Does NOT run inference — the goal is to exercise the security mechanics
in isolation. Pair with smoke_spki_probe.py on a different machine.

Usage:
    python scripts/smoke_spki_register.py \\
        --pillar-url http://home-pc:7778 \\
        --port 5550 --node-id smoke-node-d

The TLS listener accepts connections, performs the handshake, then
closes. That's all probe_peer_spki needs (a successful TLS handshake
that surfaces the server cert).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import sys
import threading
import time
from pathlib import Path
from urllib import request as urlrequest

# Allow standalone invocation: assume the script lives at scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nakshatra_auth as auth
import nakshatra_tls as nt


def _accept_loop(server_socket: socket.socket, ssl_ctx: ssl.SSLContext,
                  stop: threading.Event, label: str) -> None:
    """Accept TLS connections, complete handshake, close. The probe
    client only needs the cert; doing the handshake then dropping is
    sufficient."""
    server_socket.settimeout(0.5)
    while not stop.is_set():
        try:
            client, addr = server_socket.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            ssock = ssl_ctx.wrap_socket(client, server_side=True)
            try:
                # Read a few bytes (or just drain) so the client's
                # handshake completes before we close.
                ssock.settimeout(1.0)
                try:
                    ssock.recv(64)
                except (socket.timeout, ssl.SSLError, OSError):
                    pass
            finally:
                ssock.close()
            print(f"[{label}] handshake from {addr[0]}:{addr[1]}", flush=True)
        except (ssl.SSLError, OSError) as e:
            print(f"[{label}] handshake failed from {addr[0]}: {e}",
                  flush=True)
            try:
                client.close()
            except Exception:
                pass


def _register_with_pillar(pillar_url: str, node_id: str, address: str,
                           public_key_hex: str, peer_spki_hash: str,
                           priv_bytes: bytes) -> None:
    body = json.dumps({
        "node_id": node_id,
        "node_type": "compute",
        "address": address,
        "public_key_hex": public_key_hex,
        "peer_spki_hash": peer_spki_hash,
    }).encode("utf-8")
    ts = int(time.time())
    sig = auth.sign_request(priv_bytes, "POST", "/peer", body, ts)
    header = f'Sthambha-Ed25519 keyid="{node_id}",sig="{sig}",ts="{ts}"'
    req = urlrequest.Request(
        f"{pillar_url.rstrip('/')}/peer", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": header},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=10) as resp:
        body_resp = resp.read().decode("utf-8")
    print(f"[register] pillar said: {body_resp}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pillar-url", required=True)
    ap.add_argument("--port", type=int, required=True,
                    help="TLS listener port (this host)")
    ap.add_argument("--bind", default="0.0.0.0",
                    help="Bind address for TLS listener")
    ap.add_argument("--node-id", required=True)
    ap.add_argument("--public-address", required=True,
                    help="Tailscale IP + port the pillar will tell "
                         "probes to connect to (e.g. 203.0.113.14:5550)")
    ap.add_argument("--tls-dir", default=str(nt.DEFAULT_TLS_DIR),
                    help="Cert/key directory (default ~/.nakshatra/tls)")
    ap.add_argument("--keys-dir", default=str(
        Path.home() / ".nakshatra" / "keys"))
    ap.add_argument("--rotate-cert", action="store_true",
                    help="Force-regenerate cert before starting "
                         "(simulates operator cert rotation)")
    ap.add_argument("--skip-register", action="store_true",
                    help="Generate/load cert and start the TLS listener, "
                         "but do NOT register with the pillar. Used to "
                         "simulate the operator-rotated-cert-without-"
                         "re-registering attack: pillar still has the "
                         "prior SPKI; peer now serves a new one.")
    args = ap.parse_args()

    tls_dir = Path(args.tls_dir).expanduser()
    keys_dir = Path(args.keys_dir).expanduser()

    if args.rotate_cert:
        for fn in (nt.CERT_FILENAME, nt.KEY_FILENAME):
            try:
                (tls_dir / fn).unlink()
            except FileNotFoundError:
                pass
        print("[smoke] rotated TLS cert/key", flush=True)

    # Generate / load TLS cert
    cert_path, key_path, spki_hash = nt.ensure_cert(output_dir=tls_dir)
    print(f"[smoke] cert: {cert_path}", flush=True)
    print(f"[smoke] spki_sha256: {spki_hash}", flush=True)

    # Generate / load worker Ed25519 key
    keys_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    priv_path = keys_dir / "worker.ed25519"
    pub_path = keys_dir / "worker.pub.hex"
    if not priv_path.exists():
        priv_bytes, pub_hex = auth.generate_keypair()
        priv_path.write_bytes(priv_bytes)
        priv_path.chmod(0o600)
        pub_path.write_text(pub_hex)
        print(f"[smoke] generated worker keypair at {priv_path}", flush=True)
    else:
        priv_bytes = priv_path.read_bytes()
        pub_hex = pub_path.read_text().strip()
        print(f"[smoke] using existing worker keypair", flush=True)
    print(f"[smoke] public_key_hex: {pub_hex}", flush=True)

    # Bind TLS server
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind, args.port))
    sock.listen(16)
    print(f"[smoke] TLS listening on {args.bind}:{args.port}", flush=True)

    stop = threading.Event()
    t = threading.Thread(
        target=_accept_loop,
        args=(sock, ssl_ctx, stop, args.node_id),
        daemon=True,
    )
    t.start()

    # Register with pillar (unless skipped to simulate the
    # rotate-without-re-register attack model).
    if args.skip_register:
        print(f"[smoke] SKIP register — peer serves spki={spki_hash[:8]}… "
              f"but pillar roster will retain the prior value",
              flush=True)
    else:
        _register_with_pillar(
            args.pillar_url, args.node_id, args.public_address,
            pub_hex, spki_hash, priv_bytes,
        )
        print(f"[smoke] registered with {args.pillar_url} as {args.node_id}",
              flush=True)

    # Idle so peers can probe
    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("[smoke] shutting down", flush=True)
    finally:
        stop.set()
        try:
            sock.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
