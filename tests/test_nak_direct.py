"""U3b: direct paths — opt-in per contact, addresses swapped only inside the encrypted session, one
dialer (the pairing initiator), the same pinned handshake, the relay as the fallback."""
from __future__ import annotations

import socket
import subprocess
import time

import pytest

from test_nakd import _connect, net, wait  # noqa: F401  (net is a fixture)
from network import nakd
from transport import connect as C


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def loopback_ok(monkeypatch):
    # Tests run on loopback; production filters it out (SSRF guard in mesh/direct_tunnel).
    import mesh.direct_tunnel as DT
    monkeypatch.setattr(DT, "usable_endpoints", lambda eps, allow_lan=True: (list(eps), []))
    monkeypatch.setattr(nakd, "local_endpoints", lambda port: f"127.0.0.1:{port}")
    monkeypatch.setattr(nakd, "DIRECT_WAIT_S", 3.0)


def _pair_direct(net):
    a = net("a", direct_port=_free_port())
    b = net("b", direct_port=_free_port())
    _connect(net, a, b)                                  # first session goes through the relay
    assert wait(lambda: a.node._paths.get(b.person_pub) == "relay" and b.node._paths.get(a.person_pub) == "relay")
    return a, b


def _drop(node, person):
    with node._lock:
        sess = node._sessions.get(person)
    if sess:
        nakd._hard_close(sess[0])


def test_both_opt_in_then_the_next_session_is_direct(net, loopback_ok):
    a, b = _pair_direct(net)
    a.node.set_direct("b", True)
    b.node.set_direct(a.person_pub, True)
    assert wait(lambda: (a.node.store.contact(b.person_pub) or {}).get("direct_hint")
                and (b.node.store.contact(a.person_pub) or {}).get("direct_hint"))
    paired = net.relay.paired
    _drop(a.node, b.person_pub)
    assert wait(lambda: a.node._paths.get(b.person_pub) == "direct" and b.node._paths.get(a.person_pub) == "direct",
                timeout=40)
    assert net.relay.paired == paired                    # the relay was not used for the new session
    mid = b.node.send(a.person_pub, "over the direct path")["queued"]
    assert wait(lambda: b.node.delivered(mid))
    assert a.node.store.inbox()[0]["text"] == "over the direct path"


def test_one_sided_opt_in_never_reveals_or_dials(net, loopback_ok):
    a, b = _pair_direct(net)
    a.node.set_direct("b", True)                         # b never opts in
    time.sleep(1.0)
    assert not (b.node.store.contact(a.person_pub) or {}).get("direct_hint")   # b did not keep a's addresses
    assert not (a.node.store.contact(b.person_pub) or {}).get("direct_hint")   # b never sent its own
    _drop(a.node, b.person_pub)
    assert wait(lambda: a.node._paths.get(b.person_pub) == "relay", timeout=40)


def test_one_sided_opt_in_sends_only_capability_without_endpoints(net, loopback_ok, monkeypatch):
    a, b = _pair_direct(net)
    sent = []
    real_send = nakd.send_frame

    def capture(ch, frame, lock=None):
        if frame.get("t") in ("addr?", "addr"):
            sent.append(dict(frame))
        return real_send(ch, frame, lock)

    monkeypatch.setattr(nakd, "send_frame", capture)
    a.node.set_direct("b", True)
    time.sleep(0.5)
    assert {"t": "addr?", "direct": True} in sent
    assert not any(f.get("t") == "addr" and f.get("endpoints") for f in sent)


def test_turning_direct_off_forgets_addresses_and_falls_back_to_the_relay(net, loopback_ok):
    a, b = _pair_direct(net)
    a.node.set_direct("b", True)
    b.node.set_direct(a.person_pub, True)
    assert wait(lambda: (b.node.store.contact(a.person_pub) or {}).get("direct_hint"))
    _drop(a.node, b.person_pub)
    assert wait(lambda: a.node._paths.get(b.person_pub) == "direct" and b.node._paths.get(a.person_pub) == "direct",
                timeout=40)
    b.node.set_direct(a.person_pub, False)
    assert not b.node.store.contact(a.person_pub).get("direct_hint")
    assert wait(lambda: b.node._paths.get(a.person_pub) == "relay", timeout=40)
    assert wait(lambda: not (a.node.store.contact(b.person_pub) or {}).get("direct_hint"))


def test_listener_refuses_a_stranger_and_a_wrong_key(net, loopback_ok):
    a, b = _pair_direct(net)
    a.node.set_direct("b", True)
    from cryptography.hazmat.primitives.asymmetric import ed25519
    stranger = ed25519.Ed25519PrivateKey.generate().private_bytes_raw()
    with pytest.raises(Exception):
        C.open_direct(f"127.0.0.1:{a.node.direct_port}", my_key=stranger, peer_pub_hex=a.node.node,
                      purpose=nakd.MSG_DOMAIN.encode(), timeout=5)
    assert b.person_pub not in a.node._sessions or a.node._paths.get(b.person_pub) == "relay"


def test_both_p2p_advertisements_try_sidecar_after_endpoint_then_before_relay(net, loopback_ok, monkeypatch):
    monkeypatch.setattr(nakd.Node, "_p2p_available", lambda self: bool(self.p2p))
    a = net("a", direct_port=_free_port(), p2p={"dial": "/tmp/test-a-p2p.sock"})
    b = net("b", direct_port=_free_port(), p2p={"dial": "/tmp/test-b-p2p.sock"})
    _connect(net, a, b)
    assert wait(lambda: a.node._paths.get(b.person_pub) == "relay" and b.node._paths.get(a.person_pub) == "relay")
    a.node.set_direct("b", True)
    b.node.set_direct(a.person_pub, True)
    assert wait(lambda: (a.node.store.contact(b.person_pub) or {}).get("p2p")
                and (b.node.store.contact(a.person_pub) or {}).get("p2p"))

    real_open = C.open_direct
    targets = {a.node.node: a.node.direct_port, b.node.node: b.node.direct_port}
    calls = []
    monkeypatch.setattr(nakd, "open_direct", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("blocked")))

    def via(dial, peer_node_pub_hex, **kwargs):
        calls.append((dial, peer_node_pub_hex))
        return real_open(f"127.0.0.1:{targets[peer_node_pub_hex]}", peer_pub_hex=peer_node_pub_hex, **kwargs)

    monkeypatch.setattr(nakd, "open_via_sidecar", via)
    paired = net.relay.paired
    _drop(a.node, b.person_pub)
    assert wait(lambda: a.node._paths.get(b.person_pub) == "p2p" and b.node._paths.get(a.person_pub) == "p2p",
                timeout=40)
    assert calls and net.relay.paired == paired
    b.node.set_direct(a.person_pub, False)
    assert wait(lambda: a.node._paths.get(b.person_pub) == "relay" and b.node._paths.get(a.person_pub) == "relay",
                timeout=40)
    assert not (a.node.store.contact(b.person_pub) or {}).get("p2p")


def test_failed_p2p_attempt_clears_cached_capability_on_both_peers(net, loopback_ok, monkeypatch):
    monkeypatch.setattr(nakd.Node, "_p2p_available", lambda self: bool(self.p2p))
    a = net("a", direct_port=_free_port(), p2p={"dial": "/tmp/test-a-p2p.sock"})
    b = net("b", direct_port=_free_port(), p2p={"dial": "/tmp/test-b-p2p.sock"})
    _connect(net, a, b)
    assert wait(lambda: a.node._paths.get(b.person_pub) == "relay" and b.node._paths.get(a.person_pub) == "relay")
    a.node.set_direct("b", True)
    b.node.set_direct(a.person_pub, True)
    assert wait(lambda: (a.node.store.contact(b.person_pub) or {}).get("p2p")
                and (b.node.store.contact(a.person_pub) or {}).get("p2p"))
    monkeypatch.setattr(nakd, "open_direct", lambda *a, **kw: (_ for _ in ()).throw(OSError("direct failed")))
    monkeypatch.setattr(nakd, "open_via_sidecar",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("peer unavailable")))
    paired = net.relay.paired
    _drop(a.node, b.person_pub)
    assert wait(lambda: net.relay.paired > paired and a.node._paths.get(b.person_pub) == "relay"
                and b.node._paths.get(a.person_pub) == "relay",
                timeout=40)
    assert wait(lambda: all(not (node.store.contact(person) or {}).get("direct_hint")
                            and not (node.store.contact(person) or {}).get("p2p")
                            for node, person in ((a.node, b.person_pub), (b.node, a.person_pub))))


def test_nak_p2p_toggle_uses_only_the_test_state_dir(tmp_path, monkeypatch):
    from network import nak
    calls = []
    monkeypatch.setenv("NAK_NET_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    assert nak.main(["p2p", "on"]) == 0
    assert (tmp_path / "p2p-dial").read_text() == f"{tmp_path}/run/nakshatra/p2p.sock\n"
    assert calls[-1] == ["systemctl", "--user", "restart", "nak-net.service"]
    assert nak.main(["p2p", "off"]) == 0
    assert not (tmp_path / "p2p-dial").exists()
