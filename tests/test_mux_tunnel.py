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


def test_slow_stream_does_not_block_other_streams(monkeypatch):
    # A consumer that NEVER reads is stuck, not slow: after STALL_S with no progress it is closed
    # and the tunnel moves again (shortened here; production waits 30 s).
    import transport.mux_tunnel as M
    monkeypatch.setattr(M, "STALL_S", 0.5)
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


# ── review 2026-10-03: data-then-close must deliver every byte; a slow consumer is slowed, not killed ──

def _blaster(nbytes):
    """A target that writes nbytes as fast as it can, then closes (the request/response shape)."""
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
            try:
                c.sendall(b"x" * nbytes)
            finally:
                c.close()
    threading.Thread(target=loop, daemon=True).start()
    return srv, srv.getsockname()[1]


def _tunnel_to(port):
    a, b = socket.socketpair()
    client, server = MuxTunnel(a), MuxTunnel(b)
    threading.Thread(target=server.run_server, args=("127.0.0.1", port), daemon=True).start()
    return client, server, client.run_client("127.0.0.1", 0)


def _read_all(port, delay=0.0):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.settimeout(10)
    got = 0
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        got += len(chunk)
        if delay:
            time.sleep(delay)
    s.close()
    return got


def test_data_then_close_delivers_every_byte():
    srv, port = _blaster(200_000)
    client, server, lp = _tunnel_to(port)
    try:
        assert [_read_all(lp) for _ in range(10)] == [200_000] * 10
    finally:
        client.close(); server.close(); srv.close()


def test_slow_consumer_beyond_the_queue_cap_is_slowed_not_killed(monkeypatch):
    import transport.mux_tunnel as M
    monkeypatch.setattr(M, "MAX_QUEUED", 64 << 10)          # tiny cap so a slow reader exceeds it
    srv, port = _blaster(1_000_000)
    client, server, lp = _tunnel_to(port)
    try:
        assert _read_all(lp, delay=0.002) == 1_000_000
    finally:
        client.close(); server.close(); srv.close()
