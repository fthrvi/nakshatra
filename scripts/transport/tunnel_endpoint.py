"""tunnel_endpoint.py — one node's end of a sovereign tunnel (v1.1 §8.4).

Composes the three pieces into a runnable endpoint:
    relay.connect → identity_handshake.mutual_handshake → mux_tunnel.MuxTunnel

Two modes:
  • client — listen locally; forward each local TCP conn as a mux stream to the
    peer. Point a chain's worker address at the printed local port.
  • server — for each mux stream, dial a local target (e.g. the gRPC worker).

Both connect OUT to a reachable relay (NAT-friendly) and authenticate to the
pinned Ed25519 peer key before any bytes flow. This is the spike substrate that
carries the real gRPC data plane through the relay; production swaps the relayed
pipe for a WireGuard/QUIC tunnel (same auth, same mux-or-native-streams).

Usage:
  server: tunnel_endpoint.py server RELAY_HOST RELAY_PORT RDV MY_PRIV PEER_PUB INIT TARGET_HOST TARGET_PORT
  client: tunnel_endpoint.py client RELAY_HOST RELAY_PORT RDV MY_PRIV PEER_PUB INIT LISTEN_PORT
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transport.relay import connect  # noqa: E402
from transport.secure_channel import secure_handshake  # noqa: E402
from transport.mux_tunnel import MuxTunnel  # noqa: E402


def main() -> int:
    mode = sys.argv[1]
    relay_host, relay_port, rdv = sys.argv[2], int(sys.argv[3]), sys.argv[4].encode()
    my_priv = bytes.fromhex(sys.argv[5])
    peer_pub = sys.argv[6]
    is_init = sys.argv[7] == "1"

    print(f"[tunnel] connecting OUT to relay {relay_host}:{relay_port} (rdv={sys.argv[4]})",
          flush=True)
    sock = connect(relay_host, relay_port, rdv, timeout=30)
    print("[tunnel] paired; running ENCRYPTED handshake (X25519 + pinned Ed25519)…", flush=True)
    chan = secure_handshake(sock, my_priv, peer_pub, is_init,
                            session_binding=b"tunnel:" + rdv)
    print(f"[tunnel] authenticated + encrypted to peer {chan.peer_pubkey_hex[:16]} — "
          f"tunnel UP (relay sees only ciphertext)", flush=True)

    mux = MuxTunnel(chan)
    if mode == "server":
        target_host, target_port = sys.argv[8], int(sys.argv[9])
        print(f"[tunnel] server: mux streams → {target_host}:{target_port}", flush=True)
        mux.run_server(target_host, target_port)   # blocks
    else:
        listen_port = int(sys.argv[8])
        bound = mux.run_client("127.0.0.1", listen_port)
        print(f"[tunnel] client: listening on 127.0.0.1:{bound} → peer over relay", flush=True)
        import threading
        threading.Event().wait()   # block forever
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
