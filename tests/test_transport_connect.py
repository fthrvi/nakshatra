"""U3: one way to reach a peer — open_channel over a real local relay, both roles, binding enforced."""
import os
import socket
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

from transport.connect import accept_direct, open_channel, open_via_sidecar  # noqa: E402
from transport.relay import RendezvousRelay  # noqa: E402
from transport.secure_channel import SecureChannelError  # noqa: E402


def _key():
    k = ed25519.Ed25519PrivateKey.generate()
    return k.private_bytes_raw(), k.public_key().public_bytes_raw().hex()


@pytest.fixture
def relay():
    r = RendezvousRelay("127.0.0.1", 0, max_per_ip_per_min=100000)
    port = r.start()
    yield ("127.0.0.1", port)
    r.stop()


def _both(relay, a, b, rid, bind_a, bind_b):
    out = {}

    def side(name, me, peer, init, binding):
        try:
            out[name] = open_channel(relay=relay, rendezvous_id=rid, my_key=me[0], peer_pub_hex=peer[1],
                                     initiator=init, binding=binding, wait_timeout=10)
        except Exception as e:  # noqa: BLE001
            out[name] = e
    ts = [threading.Thread(target=side, args=("a", a, b, True, bind_a)),
          threading.Thread(target=side, args=("b", b, a, False, bind_b))]
    [t.start() for t in ts]
    [t.join(15) for t in ts]
    return out


def test_two_peers_get_one_encrypted_channel(relay):
    a, b = _key(), _key()
    out = _both(relay, a, b, os.urandom(16), b"tunnel:x", b"tunnel:x")
    (_, ca), (_, cb) = out["a"], out["b"]
    ca.sendall(b"hello")
    assert cb.recv(5) == b"hello"
    assert ca.peer_pubkey_hex == b[1] and cb.peer_pubkey_hex == a[1]


def test_a_channel_opened_for_another_purpose_fails(relay):
    a, b = _key(), _key()
    out = _both(relay, a, b, os.urandom(16), b"tunnel:x", b"nak-msg-v1")
    assert isinstance(out["a"], (SecureChannelError, OSError)) or isinstance(out["b"], (SecureChannelError, OSError))


def test_wrong_pinned_key_fails(relay):
    a, b, mallory = _key(), _key(), _key()
    out = {}
    rid = os.urandom(16)

    def side(name, me, pin, init):
        try:
            out[name] = open_channel(relay=relay, rendezvous_id=rid, my_key=me[0], peer_pub_hex=pin,
                                     initiator=init, binding=b"t", wait_timeout=10)
        except Exception as e:  # noqa: BLE001
            out[name] = e
    ts = [threading.Thread(target=side, args=("a", a, b[1], True)),
          threading.Thread(target=side, args=("m", mallory, a[1], False))]   # mallory poses as b
    [t.start() for t in ts]
    [t.join(15) for t in ts]
    assert isinstance(out["a"], (SecureChannelError, OSError))


def test_sidecar_socket_uses_the_same_direct_pinned_handshake(tmp_path):
    a, b = _key(), _key()
    path = tmp_path / "p2p.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    out = {}

    def server():
        conn, _ = listener.accept()
        line = b""
        while not line.endswith(b"\n"):
            line += conn.recv(1)
        out["dial"] = line
        conn.sendall(b"OK direct\n")
        peer, ch = accept_direct(conn, my_key=b[0], purpose=b"nak-msg-v1", allow=lambda p: p == a[1])
        out["peer"] = peer
        out["body"] = ch.recv(5)
        ch.sendall(b"world")

    t = threading.Thread(target=server)
    t.start()
    sock, ch = open_via_sidecar(str(path), b[1], my_key=a[0],
                                purpose=b"nak-msg-v1", timeout=5)
    ch.sendall(b"hello")
    assert ch.recv(5) == b"world"
    sock.close()
    t.join(5)
    listener.close()
    from sidecar_key import peer_id_from_node_pub
    assert out == {"dial": f"DIAL {peer_id_from_node_pub(b[1])}\n".encode(), "peer": a[1], "body": b"hello"}


def test_sidecar_dial_address_must_be_an_absolute_unix_socket():
    a, b = _key(), _key()
    with pytest.raises(OSError, match="absolute UNIX"):
        open_via_sidecar("relative.sock", b[1], my_key=a[0], purpose=b"x", timeout=0.1)


def test_sidecar_must_explicitly_confirm_a_direct_stream(tmp_path):
    a, b = _key(), _key()
    path = tmp_path / "p2p.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)

    def server():
        conn, _ = listener.accept()
        while not conn.recv(1).endswith(b"\n"):
            pass
        conn.sendall(b"ERR no direct connection before timeout\n")
        conn.close()

    t = threading.Thread(target=server)
    t.start()
    with pytest.raises(OSError, match="ERR no direct connection"):
        open_via_sidecar(str(path), b[1], my_key=a[0],
                         purpose=b"x", timeout=2)
    t.join(2)
    listener.close()
