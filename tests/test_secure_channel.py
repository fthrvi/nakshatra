"""Tests for the encrypted channel over the relay (v1.1 §8.5b)."""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from nakshatra_auth import generate_keypair  # noqa: E402
from transport.secure_channel import secure_handshake, SecureChannel, SecureChannelError  # noqa: E402
from transport.mux_tunnel import MuxTunnel  # noqa: E402


def _recording_pipe():
    """A 'relay' between two app sockets that records everything it forwards.
    Returns (a_app, b_app, record_list)."""
    a_app, a_net = socket.socketpair()
    b_app, b_net = socket.socketpair()
    record: list[bytes] = []

    def pump(src, dst):
        try:
            while True:
                d = src.recv(4096)
                if not d:
                    break
                record.append(d)
                dst.sendall(d)
        except OSError:
            pass

    threading.Thread(target=pump, args=(a_net, b_net), daemon=True).start()
    threading.Thread(target=pump, args=(b_net, a_net), daemon=True).start()
    return a_app, b_app, record


def _hs(sock, priv, peer_pin, is_init, out, idx, binding=b"s"):
    try:
        out[idx] = secure_handshake(sock, priv, peer_pin, is_init, binding)
    except Exception as e:  # noqa: BLE001
        out[idx] = e


def test_handshake_and_encrypted_roundtrip():
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    a_app, b_app, _ = _recording_pipe()
    out = [None, None]
    ta = threading.Thread(target=_hs, args=(a_app, priv_a, pub_b, True, out, 0))
    tb = threading.Thread(target=_hs, args=(b_app, priv_b, pub_a, False, out, 1))
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    ca, cb = out
    assert isinstance(ca, SecureChannel) and isinstance(cb, SecureChannel)
    assert ca.peer_pubkey_hex == pub_b and cb.peer_pubkey_hex == pub_a
    ca.sendall(b"hello over the encrypted channel")
    assert cb.recv(4096) == b"hello over the encrypted channel"
    cb.sendall(b"reply")
    assert ca.recv(4096) == b"reply"


def test_relay_sees_only_ciphertext():
    """The recording middle (= the untrusted relay) must NEVER see the plaintext."""
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    a_app, b_app, record = _recording_pipe()
    out = [None, None]
    ta = threading.Thread(target=_hs, args=(a_app, priv_a, pub_b, True, out, 0))
    tb = threading.Thread(target=_hs, args=(b_app, priv_b, pub_a, False, out, 1))
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    ca, cb = out
    record.clear()                                  # ignore handshake bytes
    marker = b"TOP-SECRET-ACTIVATION-PARIS-12366"
    ca.sendall(marker)
    assert cb.recv(4096) == marker
    time.sleep(0.1)
    wire = b"".join(record)
    assert marker not in wire and len(wire) > 0     # relay forwarded only ciphertext


def test_mitm_wrong_pin_rejected():
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    _, attacker = generate_keypair()
    a_app, b_app, _ = _recording_pipe()
    out = [None, None]
    ta = threading.Thread(target=_hs, args=(a_app, priv_a, attacker, True, out, 0))  # A pins attacker
    tb = threading.Thread(target=_hs, args=(b_app, priv_b, pub_a, False, out, 1))
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    assert isinstance(out[0], SecureChannelError)


def test_tampered_record_fails():
    # two channels with the SAME keys; tamper a ciphertext byte → decrypt raises
    key_s, key_r = b"\x01" * 32, b"\x02" * 32
    a, b = socket.socketpair()
    sender = SecureChannel(a, key_s, key_r)
    receiver = SecureChannel(b, key_s, key_r)   # send_key==recv_key so it can decrypt sender's records
    sender.sendall(b"important")
    # corrupt the bytes in flight
    raw = b.recv(4096)
    bad = bytearray(raw); bad[-1] ^= 0xFF
    a2, b2 = socket.socketpair()
    a2.sendall(bytes(bad)); a2.close()
    victim = SecureChannel(b2, key_s, key_r)
    with pytest.raises(SecureChannelError):
        victim.recv(4096)


def test_mux_runs_over_secure_channel():
    """MuxTunnel is a drop-in over the SecureChannel — the gRPC data plane path."""
    # local echo target
    esrv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    esrv.bind(("127.0.0.1", 0)); esrv.listen(8)
    eport = esrv.getsockname()[1]

    def echo_loop():
        while True:
            try:
                c, _ = esrv.accept()
            except OSError:
                break
            threading.Thread(target=lambda c=c: [c.sendall(d) for d in iter(lambda: c.recv(4096), b"")], daemon=True).start()
    threading.Thread(target=echo_loop, daemon=True).start()

    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    a_app, b_app, _ = _recording_pipe()
    out = [None, None]
    ta = threading.Thread(target=_hs, args=(a_app, priv_a, pub_b, True, out, 0))
    tb = threading.Thread(target=_hs, args=(b_app, priv_b, pub_a, False, out, 1))
    ta.start(); tb.start(); ta.join(5); tb.join(5)
    ca, cb = out
    server = MuxTunnel(cb)
    threading.Thread(target=server.run_server, args=("127.0.0.1", eport), daemon=True).start()
    client = MuxTunnel(ca)
    lport = client.run_client("127.0.0.1", 0)
    time.sleep(0.1)
    s = socket.create_connection(("127.0.0.1", lport), timeout=5)
    s.sendall(b"mux-over-encrypted")
    got = b""
    s.settimeout(5)
    while len(got) < len(b"mux-over-encrypted"):
        d = s.recv(4096)
        if not d:
            break
        got += d
    assert got == b"mux-over-encrypted"
    esrv.close()
