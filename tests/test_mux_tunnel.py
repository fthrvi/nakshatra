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
