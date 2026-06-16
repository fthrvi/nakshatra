"""Admission-gated junction: admitted peers pair + forward bytes verbatim (blind); unadmitted denied."""
import sys, time, socket
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import junction as J


def _is_closed(sock) -> bool:
    """A denied connection is closed by the junction — surfaces as clean EOF OR a RST, depending
    on timing/OS. Both mean 'no channel'."""
    try:
        sock.settimeout(2)
        return sock.recv(16) == b""
    except (ConnectionResetError, ConnectionError, OSError):
        return True


def _admit_if_ok(req):
    return {"admitted": bool(req and req.get("ok")), "tier": "trusted", "tenant": "t",
            "reason": "ok" if (req and req.get("ok")) else "not on roster"}


def _start():
    j = J.AdmissionGatedJunction(host="127.0.0.1", port=0, admit_fn=_admit_if_ok)
    port = j.start()
    return j, port


def test_admitted_peers_pair_and_forward_verbatim():
    j, port = _start()
    try:
        rid = b"rendezvous-AB"
        a = J.connect_admitted("127.0.0.1", port, rid, {"ok": True})
        b = J.connect_admitted("127.0.0.1", port, rid, {"ok": True})
        time.sleep(0.3)
        assert j.admitted == 2 and j.paired == 1 and j.denied == 0
        # blind forward both directions — bytes arrive verbatim (the relay never interprets them)
        payload = bytes(range(256)) * 4          # binary "ciphertext"
        a.sendall(payload)
        got = b""
        b.settimeout(2)
        while len(got) < len(payload):
            chunk = b.recv(4096)
            if not chunk:
                break
            got += chunk
        assert got == payload, "junction altered the bytes (not a blind forwarder)"
    finally:
        j.stop()


def test_unadmitted_is_denied_no_pairing():
    j, port = _start()
    try:
        rid = b"rendezvous-X"
        bad = J.connect_admitted("127.0.0.1", port, rid, {"ok": False})   # not admitted
        time.sleep(0.3)
        assert j.denied == 1 and j.paired == 0 and j.admitted == 0
        # the junction closed the socket → the peer reads EOF
        assert _is_closed(bad), "denied peer's socket was not closed"
    finally:
        j.stop()


def test_no_preamble_is_denied():
    j, port = _start()
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(b"\x00\x00\x00\x00")           # zero-length preamble → invalid
        time.sleep(0.2)
        assert j.denied == 1 and j.paired == 0
        assert _is_closed(s)
    finally:
        j.stop()


def test_admitted_peer_not_paired_until_partner():
    j, port = _start()
    try:
        a = J.connect_admitted("127.0.0.1", port, b"solo", {"ok": True})
        time.sleep(0.2)
        assert j.admitted == 1 and j.paired == 0   # admitted but waiting for its partner
        a.close()
    finally:
        j.stop()


if __name__ == "__main__":
    test_admitted_peers_pair_and_forward_verbatim()
    test_unadmitted_is_denied_no_pairing()
    test_no_preamble_is_denied()
    test_admitted_peer_not_paired_until_partner()
    print("all junction tests PASS")
