"""U3b: direct paths — opt-in per contact, addresses swapped only inside the encrypted session, one
dialer (the pairing initiator), the same pinned handshake, the relay as the fallback."""
from __future__ import annotations

import socket
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


def test_turning_direct_off_forgets_addresses_and_falls_back_to_the_relay(net, loopback_ok):
    a, b = _pair_direct(net)
    a.node.set_direct("b", True)
    b.node.set_direct(a.person_pub, True)
    assert wait(lambda: (b.node.store.contact(a.person_pub) or {}).get("direct_hint"))
    b.node.set_direct(a.person_pub, False)
    assert not b.node.store.contact(a.person_pub).get("direct_hint")
    _drop(b.node, a.person_pub)
    assert wait(lambda: b.node._paths.get(a.person_pub) == "relay", timeout=40)


def test_listener_refuses_a_stranger_and_a_wrong_key(net, loopback_ok):
    a, b = _pair_direct(net)
    a.node.set_direct("b", True)
    from cryptography.hazmat.primitives.asymmetric import ed25519
    stranger = ed25519.Ed25519PrivateKey.generate().private_bytes_raw()
    with pytest.raises(Exception):
        C.open_direct(f"127.0.0.1:{a.node.direct_port}", my_key=stranger, peer_pub_hex=a.node.node,
                      purpose=nakd.MSG_DOMAIN.encode(), timeout=5)
    assert b.person_pub not in a.node._sessions or a.node._paths.get(b.person_pub) == "relay"
