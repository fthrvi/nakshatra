"""Tests for the multiplexing tunnel (v1.1 §8.4) — many streams over one pipe."""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from transport.mux_tunnel import MuxTunnel  # noqa: E402


def _echo_server():
    """A local target that echoes each connection's bytes (stands in for worker B)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)

    def loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                break
            threading.Thread(target=_echo_one, args=(c,), daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()
    return srv, srv.getsockname()[1]


def _echo_one(c):
    try:
        while True:
            d = c.recv(4096)
            if not d:
                break
            c.sendall(d)
    except OSError:
        pass
    finally:
        c.close()


@pytest.fixture
def tunnel():
    echo_srv, echo_port = _echo_server()
    a, b = socket.socketpair()             # stands in for the relay pipe
    client = MuxTunnel(a)
    server = MuxTunnel(b)
    threading.Thread(target=server.run_server, args=("127.0.0.1", echo_port), daemon=True).start()
    local_port = client.run_client("127.0.0.1", 0)
    time.sleep(0.1)
    yield local_port
    client.close(); server.close(); echo_srv.close()


def _roundtrip(port, payload):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(payload)
    out = b""
    s.settimeout(5)
    while len(out) < len(payload):
        chunk = s.recv(4096)
        if not chunk:
            break
        out += chunk
    s.close()
    return out


def test_single_stream_roundtrip(tunnel):
    assert _roundtrip(tunnel, b"hello-through-the-mux") == b"hello-through-the-mux"


def test_many_concurrent_streams(tunnel):
    """The whole point: several independent streams over the ONE pipe at once."""
    results = {}

    def one(i):
        payload = f"stream-{i}-".encode() * 50
        results[i] = _roundtrip(tunnel, payload) == payload

    threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join(10)
    assert len(results) == 8 and all(results.values())


def test_large_payload(tunnel):
    payload = bytes(range(256)) * 4096   # 1 MB through the mux
    assert _roundtrip(tunnel, payload) == payload


# ── hardening (2026-10-02 design council): frame cap, stream cap, no head-of-line blocking ──
import struct as _struct
from transport import mux_tunnel as _mt  # noqa: E402


def test_oversized_frame_closes_tunnel_without_reading_body():
    a, b = socket.socketpair()
    server = MuxTunnel(b)
    t = threading.Thread(target=server.run_server, args=("127.0.0.1", 9), daemon=True)
    t.start()
    a.sendall(_struct.pack(">IBI", 1, _mt.DATA, 0xFFFFFFFF))   # header only, no body
    assert server._closed.wait(2), "tunnel should close on a frame larger than MAX_FRAME"


def test_stream_cap_refuses_extra_opens():
    echo_srv, echo_port = _echo_server()
    a, b = socket.socketpair()
    server = MuxTunnel(b)
    threading.Thread(target=server.run_server, args=("127.0.0.1", echo_port), daemon=True).start()
    for sid in range(1, 2 * (_mt.MAX_STREAMS + 10), 2):
        a.sendall(_struct.pack(">IBI", sid, _mt.OPEN, 0))
    time.sleep(1.5)
    with server._slock:
        n = len(server._streams)
    assert n <= _mt.MAX_STREAMS
    server.close(); echo_srv.close()


def test_slow_stream_does_not_block_other_streams():
    # Target: the FIRST connection never reads (a stalled consumer); later ones echo.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0)); srv.listen(16)
    held = []

    def loop():
        first = True
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                break
            if first:
                first = False
                c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                held.append(c)          # never read from it
            else:
                threading.Thread(target=_echo_one, args=(c,), daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()
    a, b = socket.socketpair()
    client, server = MuxTunnel(a), MuxTunnel(b)
    threading.Thread(target=server.run_server, args=("127.0.0.1", srv.getsockname()[1]), daemon=True).start()
    port = client.run_client("127.0.0.1", 0)
    time.sleep(0.1)

    slow = socket.create_connection(("127.0.0.1", port), timeout=5)

    def flood():
        try:
            blob = b"z" * 65536
            for _ in range(512):           # 32 MiB toward a consumer that never reads
                slow.sendall(blob)
        except OSError:
            pass

    threading.Thread(target=flood, daemon=True).start()
    time.sleep(1.0)
    start = time.time()
    assert _roundtrip(port, b"fast-stream-still-works") == b"fast-stream-still-works"
    assert time.time() - start < 5
    client.close(); server.close(); srv.close()
