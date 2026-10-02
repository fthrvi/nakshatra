"""connect.py — THE way to reach a peer (unification U3, 2026-10-03).

Before this there were three copies of "dial the rendezvous relay, then run the pinned handshake":
meshd's tunnels, nakd's contact sessions and tunnel_endpoint. They differed only in timeouts and in the
session binding. Every caller now goes through `open_channel`, so there is exactly ONE place where a
better path plugs in: direct TCP via `mesh/direct_tunnel.py`, the UDP punch in `mesh/direct_path.py`,
the libp2p sidecar (unification U3b). Whatever the path, the security is the same, because
`secure_handshake` runs over whichever socket was opened:
- the peer is pinned by its Ed25519 node key;
- the key exchange is X25519 and the channel is ChaCha20;
- a `binding` ties the session to its purpose, so a relayed session can't be swapped.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transport.relay import connect as relay_connect  # noqa: E402
from transport.secure_channel import SecureChannel, secure_handshake  # noqa: E402


def open_channel(*, relay: Tuple[str, int], rendezvous_id: bytes, my_key: bytes, peer_pub_hex: str,
                 initiator: bool, binding: bytes, connect_timeout: float = 15.0,
                 wait_timeout: Optional[float] = None) -> Tuple[socket.socket, SecureChannel]:
    """Reach `peer_pub_hex` and return (raw socket, authenticated encrypted channel).

    relay          the rendezvous relay both sides dial OUT to (NAT-friendly; the fallback path)
    rendezvous_id  16 bytes both sides derive (mesh/pairing.py, or an invite's rendezvous)
    initiator      exactly one side is True (pairing decides; the smaller node key initiates)
    binding        what this session is FOR (b"tunnel:"+rid for meshd, the messaging domain for nakd,
                   b"nak-invite-v1|"+nonce for an invite); a channel opened for one purpose can't be
                   replayed into another
    wait_timeout   how long to wait for the partner and the handshake (None = block)

    Raises OSError or SecureChannelError; on failure the socket is closed.
    """
    sock = relay_connect(relay[0], int(relay[1]), rendezvous_id, timeout=connect_timeout)
    sock.settimeout(wait_timeout)
    try:
        chan = secure_handshake(sock, my_key, peer_pub_hex, initiator, binding)
    except BaseException:
        try:
            sock.close()
        except OSError:
            pass
        raise
    return sock, chan
