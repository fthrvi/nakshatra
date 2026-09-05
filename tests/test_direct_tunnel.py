"""Skipping the relay when the peer is reachable — and refusing to be aimed anywhere."""
import socket
import threading

import pytest

from mesh.direct_tunnel import dial_direct, open_pipe, parse_hint


@pytest.fixture
def listener():
    """A real TCP listener on loopback, so `direct` can actually succeed in a test."""
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1)
    port = s.getsockname()[1]
    accepted = []
    def accept():
        try:
            c, _ = s.accept(); accepted.append(c)
        except OSError:
            pass
    threading.Thread(target=accept, daemon=True).start()
    yield port, accepted
    s.close()


@pytest.mark.parametrize("hint,expect", [
    ("192.168.1.5:51820", [("192.168.1.5", 51820)]),
    ("[2001:db8::1]:51820", [("2001:db8::1", 51820)]),
    ("10.0.0.2:5531, 203.0.113.9:5531", [("10.0.0.2", 5531), ("203.0.113.9", 5531)]),
    ("", []), ("nonsense", []), ("host:notaport", []), (None, []), (42, []),
])
def test_hint_parsing_never_raises(hint, expect):
    assert parse_hint(hint) == expect


@pytest.mark.parametrize("hint,why_contains", [
    ("169.254.169.254:80", "link-local"),      # ⚠️ cloud metadata — the SSRF case
    ("127.0.0.1:22", "loopback"),
    ("0.0.0.0:1", "unspecified"),
    ("[::1]:22", "loopback"),
    ("[ff02::1]:1", "multicast"),
])
def test_an_attacker_controlled_hint_cannot_aim_us(hint, why_contains):
    """⚠️⚠️ `endpoint_hint` arrives in a listing and ANYONE can publish one to a public
    relay. Unfiltered, this is an SSRF primitive: point a node at its own cloud metadata, or
    sweep an internal subnet one listing at a time."""
    sock, why = dial_direct(hint)
    assert sock is None
    assert why_contains in why, why


def test_a_private_hint_is_allowed_on_lan_and_refused_off_it():
    """Two boxes on one LAN genuinely reach each other at 192.168 — that is the ~129 ms
    dogleg this exists to remove. A peer across the internet advertising it is not."""
    _, why_on = dial_direct("192.168.99.99:9", allow_lan=True, timeout=0.2)
    assert "no candidate answered" in why_on          # tried it
    sock, why_off = dial_direct("192.168.99.99:9", allow_lan=False)
    assert sock is None and "private" in why_off      # never tried it


def _lan_ip():
    """A non-loopback address of THIS machine, so the filter permits the dial."""
    import ipaddress, subprocess, re
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
    except Exception:
        return None
    for m in re.finditer(r"inet (\d+\.\d+\.\d+\.\d+)/", out):
        ip = ipaddress.ip_address(m.group(1))
        if ip.is_private and not ip.is_loopback and not ip.is_link_local:
            return str(ip)
    return None


def test_direct_wins_when_the_peer_answers():
    """⚠️ A REAL successful dial. The previous version of this test never dialled anything —
    it asserted that loopback is refused and stopped, so the success path had NO test and
    open_pipe's `was_direct=True` branch was never executed. Found by an external review.
    Binds on this machine's own LAN address, which the filter permits with allow_lan=True."""
    ip = _lan_ip()
    if ip is None:
        pytest.skip("no non-loopback IPv4 on this machine")
    srv = socket.socket(); srv.bind((ip, 0)); srv.listen(1)
    port = srv.getsockname()[1]
    accepted = []
    threading.Thread(target=lambda: accepted.append(srv.accept()[0]), daemon=True).start()
    try:
        relay_called = []
        sock, why, direct = open_pipe(f"{ip}:{port}",
                                      lambda: relay_called.append(1) or socket.socketpair()[0],
                                      allow_lan=True, timeout=3.0)
        assert direct is True, why
        assert sock is not None and "direct to" in why
        assert not relay_called, "relay was consulted although the direct dial succeeded"
        sock.sendall(b"ping")
        import time; time.sleep(0.2)
        assert accepted and accepted[0].recv(4) == b"ping"
        sock.close()
    finally:
        srv.close()


def test_open_pipe_falls_back_to_the_relay(listener):
    """⚠️ The relay is the FLOOR, not a nicety: both peers behind NAT on different networks
    is the COMMON case. A failed direct attempt must cost a fallback, never a failure."""
    fake = socket.socketpair()[0]
    sock, why, was_direct = open_pipe("169.254.169.254:80", lambda: fake)
    assert sock is fake and was_direct is False
    assert "relay" in why and "direct unavailable" in why
    fake.close()


def test_open_pipe_reports_which_path_it_took():
    fake = socket.socketpair()[0]
    _, why, direct = open_pipe("", lambda: fake)
    assert direct is False and "relay" in why
    fake.close()


def test_an_empty_hint_goes_straight_to_the_relay_without_dialling():
    """A peer that published no endpoint is not unreachable — it is relay-only."""
    fake = socket.socketpair()[0]
    sock, why, direct = open_pipe(None, lambda: fake)
    assert sock is fake and not direct
    fake.close()
