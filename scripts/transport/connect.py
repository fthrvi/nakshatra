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

import ipaddress
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


# ── direct paths (U3b, 2026-10-03) ─────────────────────────────────────────────────────────────────
# A direct path is the same pinned handshake over a different pipe: a TCP connection straight to the
# peer (LAN, public IPv6) instead of through the relay. The dialer first sends DIRECT_MAGIC + its node
# key so the listener knows WHICH pin to apply; the listener accepts only peers it has opted in for
# (that decision is the caller's `allow`). The session binding names both keys, so a direct channel can
# never be confused with a relayed one or replayed between pairs. The relay stays the fallback.

DIRECT_MAGIC = b"NKD1"
DIRECT_HELLO_TIMEOUT_S = 5.0


def direct_binding(purpose: bytes, a_pub_hex: str, b_pub_hex: str) -> bytes:
    lo, hi = sorted((a_pub_hex.lower(), b_pub_hex.lower()))
    return b"direct|" + purpose + b"|" + lo.encode() + b"|" + hi.encode()


def _pub_of(key: bytes) -> str:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    return ed25519.Ed25519PrivateKey.from_private_bytes(key).public_key().public_bytes_raw().hex()


def _handshake_direct_socket(sock: socket.socket, *, my_key: bytes, peer_pub_hex: str, purpose: bytes,
                             timeout: float) -> Tuple[socket.socket, SecureChannel]:
    """Finish the one direct protocol after TCP or the sidecar has supplied its socket."""
    try:
        my_pub = _pub_of(my_key)
        sock.settimeout(timeout)
        sock.sendall(DIRECT_MAGIC + bytes.fromhex(my_pub))
        chan = secure_handshake(sock, my_key, peer_pub_hex, True,
                                direct_binding(purpose, my_pub, peer_pub_hex))
        sock.settimeout(None)
        return sock, chan
    except BaseException:
        try:
            sock.close()
        except OSError:
            pass
        raise


def open_direct(hint: str, *, my_key: bytes, peer_pub_hex: str, purpose: bytes,
                timeout: float = 10.0) -> Tuple[socket.socket, SecureChannel]:
    """Dial the peer directly using its endpoint hint (filtered: no loopback, link-local, multicast or
    DNS names; see mesh/direct_tunnel.py) and run the pinned handshake as initiator. Raises on failure."""
    from mesh.direct_tunnel import dial_direct  # noqa: PLC0415
    sock, why = dial_direct(hint)
    if sock is None:
        raise OSError(f"no direct path: {why}")
    return _handshake_direct_socket(sock, my_key=my_key, peer_pub_hex=peer_pub_hex,
                                    purpose=purpose, timeout=timeout)


def open_via_sidecar(dial_addr: str, peer_node_pub_hex: str, *, my_key: bytes, purpose: bytes,
                     timeout: float) -> Tuple[socket.socket, SecureChannel]:
    """Ask the local libp2p sidecar for a hole-punched/direct stream, then run the exact same pinned
    Nakshatra handshake as :func:`open_direct`. Relay circuits may rendezvous DCUtR but the sidecar
    returns ``OK direct`` only after it has opened a non-relayed stream."""
    from mesh.direct_tunnel import parse_hint  # noqa: PLC0415
    from sidecar_key import peer_id_from_node_pub  # noqa: PLC0415

    candidates = parse_hint(dial_addr)
    if len(candidates) != 1:
        raise OSError("p2p dial address must be one loopback host:port")
    host, port = candidates[0]
    try:
        if not ipaddress.ip_address(host).is_loopback:
            raise OSError("p2p dial address must be loopback")
    except ValueError as e:
        raise OSError("p2p dial address must use a literal loopback IP") from e
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        sock.settimeout(timeout)
        sock.sendall(f"DIAL {peer_id_from_node_pub(peer_node_pub_hex)}\n".encode("ascii"))
        reply = bytearray()
        while b"\n" not in reply:
            chunk = sock.recv(256)
            if not chunk:
                raise OSError("p2p sidecar closed before replying")
            reply += chunk
            if len(reply) > 512:
                raise OSError("p2p sidecar reply is too long")
        line, extra = bytes(reply).split(b"\n", 1)
        if extra:
            raise OSError("p2p sidecar sent data before the direct stream was ready")
        if line != b"OK direct":
            reason = line.decode("utf-8", "replace")
            raise OSError(f"p2p sidecar refused direct stream: {reason}")
    except BaseException:
        try:
            sock.close()
        except OSError:
            pass
        raise
    return _handshake_direct_socket(sock, my_key=my_key, peer_pub_hex=peer_node_pub_hex,
                                    purpose=purpose, timeout=timeout)


def accept_direct(conn: socket.socket, *, my_key: bytes, purpose: bytes, allow) -> Tuple[str, SecureChannel]:
    """Listener side: read who is calling, ask `allow(peer_pub_hex) -> bool` (contacts who opted in to
    direct), then run the pinned handshake as responder. Returns (peer_pub_hex, channel). Raises
    PermissionError for a caller that is not allowed; the connection is closed on any failure."""
    try:
        conn.settimeout(DIRECT_HELLO_TIMEOUT_S)
        hello = b""
        while len(hello) < 36:
            more = conn.recv(36 - len(hello))
            if not more:
                raise OSError("direct caller hung up before saying who it is")
            hello += more
        if hello[:4] != DIRECT_MAGIC:
            raise OSError("not a Nakshatra direct connection")
        peer = hello[4:].hex()
        if not allow(peer):
            raise PermissionError("caller is not a contact with direct enabled")
        conn.settimeout(10.0)
        chan = secure_handshake(conn, my_key, peer, False, direct_binding(purpose, _pub_of(my_key), peer))
        conn.settimeout(None)
        return peer, chan
    except BaseException:
        try:
            conn.close()
        except OSError:
            pass
        raise


def local_endpoints(port: int) -> str:
    """This machine's candidate direct endpoints as an endpoint hint: LAN IPv4 + global IPv6. Read from
    `ip -o addr` (Linux); loopback / link-local / container bridges are left out."""
    import ipaddress  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    try:
        out = subprocess.run(["ip", "-o", "addr", "show", "up"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    hosts = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[1].startswith(("lo", "docker", "br-", "virbr", "veth")) or "wg" in parts[1]:
            continue
        if " temporary" in line or " deprecated" in line:    # privacy/expiring v6 addresses churn
            continue
        try:
            ip = ipaddress.ip_interface(parts[3]).ip
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local or ip.is_multicast:
            continue
        if ip.version == 6 and not ip.is_global:
            continue
        hosts.append(f"[{ip}]:{port}" if ip.version == 6 else f"{ip}:{port}")
    v4 = [h for h in dict.fromkeys(hosts) if not h.startswith("[")]
    v6 = [h for h in dict.fromkeys(hosts) if h.startswith("[")]
    return ",".join(v4[:3] + v6[:2])                       # LAN first (direct_tunnel keeps order), few
