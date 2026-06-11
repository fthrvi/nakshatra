"""Rendezvous relay (v1.1 §8.4) — the NAT-traversal substrate.

The connectivity recon (v1.1 §8.2) showed the common case is **both peers behind
NAT on different networks**, where direct WireGuard can't connect — so a relay is
mandatory (the same conclusion Bitcoin reaches with public "listening nodes" and
WebRTC reaches with TURN). This is that relay: a stable, reachable host that two
NAT'd peers each connect *out* to, which then **pipes bytes between them**.

Trust model — the relay is UNTRUSTED. It pairs two sockets by a shared
`rendezvous_id` and forwards raw bytes; it never interprets them. The peers run
the identity handshake (`identity_handshake.py`) **end-to-end over the relayed
pipe**, so:
  • the relay cannot impersonate a peer (it can't forge the pinned-key proof), and
  • with an encryption layer on top (WireGuard/Noise — the production transport),
    the relay sees only ciphertext (TURN-style). This module is the *rendezvous +
    forward* substrate; confidentiality is the transport layer above it.

Deployment: run `serve()` on a reachable host (a public-IPv4 VPS for true
cross-NAT, or the Pillar over tailnet/IPv6 for now). A peer calls `connect()` with
the rendezvous_id agreed out-of-band (e.g. derived from both pinned pubkeys), gets
back a socket already paired to its peer, and speaks the handshake + data plane
over it.

Pure stdlib, no root (binds a high port), no deps.
"""
from __future__ import annotations

import socket
import struct
import threading
from typing import Optional

MAGIC = b"NKSR1"            # rendezvous protocol marker
MAX_ID = 128
DEFAULT_PORT = 9777


# ── framing ───────────────────────────────────────────────────────────

def _send_id(sock: socket.socket, rendezvous_id: bytes) -> None:
    if not (0 < len(rendezvous_id) <= MAX_ID):
        raise ValueError("rendezvous_id must be 1..128 bytes")
    sock.sendall(MAGIC + struct.pack(">B", len(rendezvous_id)) + rendezvous_id)


def _recv_id(sock: socket.socket) -> Optional[bytes]:
    hdr = _recv_exact(sock, len(MAGIC) + 1)
    if hdr is None or hdr[:len(MAGIC)] != MAGIC:
        return None
    n = hdr[len(MAGIC)]
    if not (0 < n <= MAX_ID):
        return None
    return _recv_exact(sock, n)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


# ── client ────────────────────────────────────────────────────────────

def connect(relay_host: str, relay_port: int, rendezvous_id: bytes,
            timeout: float = 30.0) -> socket.socket:
    """Connect OUT to the relay and register under `rendezvous_id`. Returns a
    socket that is piped to whichever peer presents the same id (the relay blocks
    until the pair forms). The caller then runs the identity handshake + data
    plane over this socket. Works through NAT — it is an outbound connection."""
    # AF-agnostic: resolve v4/v6 and try each (the Pillar is v6, a VPS is v4).
    last = None
    for fam, _t, _p, _c, sa in socket.getaddrinfo(relay_host, relay_port,
                                                   0, socket.SOCK_STREAM):
        try:
            s = socket.socket(fam, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(sa)
            _send_id(s, rendezvous_id)
            s.settimeout(None)
            return s
        except OSError as e:
            last = e
            continue
    raise OSError(f"could not reach relay {relay_host}:{relay_port}: {last}")


# ── server ────────────────────────────────────────────────────────────

def _pipe(a: socket.socket, b: socket.socket) -> None:
    """Forward a→b until a closes. One direction; run two for full duplex."""
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class RendezvousRelay:
    """Pairs sockets by rendezvous_id and pipes between them. Untrusted forwarder."""

    def __init__(self, host: str = "::", port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self._waiting: dict[bytes, socket.socket] = {}
        self._lock = threading.Lock()
        self._srv: Optional[socket.socket] = None
        self._stop = threading.Event()
        self.paired = 0  # counter for observability/tests

    def _serve_socket(self) -> socket.socket:
        # Dual-stack where possible (bind :: accepts v4-mapped + v6).
        fam = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        s = socket.socket(fam, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if fam == socket.AF_INET6:
            try:
                s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
        s.bind((self.host, self.port))
        s.listen(64)
        return s

    def start(self) -> int:
        self._srv = self._serve_socket()
        self.port = self._srv.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True).start()
        return self.port

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        rid = _recv_id(conn)
        if rid is None:
            conn.close()
            return
        with self._lock:
            peer = self._waiting.pop(rid, None)
            if peer is None:
                self._waiting[rid] = conn
                return  # wait for the partner; the partner's _handle pairs us
        # we are the second arrival → pair the two sockets, pipe both directions
        self.paired += 1
        threading.Thread(target=_pipe, args=(peer, conn), daemon=True).start()
        threading.Thread(target=_pipe, args=(conn, peer), daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self._srv:
            try:
                self._srv.close()
            except OSError:
                pass


def serve(host: str = "::", port: int = DEFAULT_PORT) -> None:
    """Run the relay forever (CLI entrypoint). `::` = all v6+v4 (dual stack)."""
    relay = RendezvousRelay(host, port)
    p = relay.start()
    print(f"[relay] rendezvous relay listening on [{host}]:{p} (untrusted byte-forwarder)",
          flush=True)
    try:
        relay._stop.wait()
    except KeyboardInterrupt:
        relay.stop()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Nakshatra rendezvous relay (v1.1 §8.4)")
    ap.add_argument("--host", default="::")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args()
    serve(a.host, a.port)
