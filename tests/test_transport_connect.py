"""U3: one way to reach a peer — open_channel over a real local relay, both roles, binding enforced."""
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

from transport.connect import open_channel  # noqa: E402
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
