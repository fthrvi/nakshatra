"""Path selection: every uncertainty resolves toward the slower, safer answer."""
import pytest

from pathchoice import choose_path

G1, G2 = "2001:db8::1", "2001:db8::2"
R = "relay.example.com"


def side(**kw):
    return {"ipv6": kw.get("ipv6", ""), "nat": kw.get("nat", "unknown"),
            "relay": kw.get("relay", "")}


def test_both_global_ipv6_wins_even_with_symmetric_nat():
    """⚠️ The ordering that is easy to get backwards: two routable v6 hosts do not care
    about NAT type, because there is no traversal problem to solve."""
    p, why = choose_path(side(ipv6=G1, nat="symmetric", relay=R),
                         side(ipv6=G2, nat="full-cone", relay=R))
    assert p == "ipv6", why


def test_link_local_is_not_a_path():
    p, _ = choose_path(side(ipv6="fe80::1", nat="full-cone", relay=R),
                       side(ipv6=G2, nat="full-cone", relay=R))
    assert p == "direct"


@pytest.mark.parametrize("bad", ["::1", "fd00::1", "::ffff:192.0.2.1", "ff02::1", "nonsense"])
def test_unroutable_v6_falls_through(bad):
    p, _ = choose_path(side(ipv6=bad, nat="symmetric", relay=R),
                       side(ipv6=G2, nat="symmetric", relay=R))
    assert p == "relay"


def test_punchable_pair_goes_direct():
    for a in ("none", "full-cone", "restricted", "port-restricted"):
        p, _ = choose_path(side(nat=a, relay=R), side(nat="full-cone", relay=R))
        assert p == "direct", a


def test_symmetric_never_punches():
    p, why = choose_path(side(nat="symmetric", relay=R), side(nat="full-cone", relay=R))
    assert p == "relay" and "symmetric" in why


def test_unknown_is_never_optimistically_punchable():
    """An unmeasured NAT is not an open NAT."""
    p, _ = choose_path(side(nat="unknown", relay=R), side(nat="full-cone", relay=R))
    assert p == "relay"


def test_a_missing_nat_key_is_unknown():
    p, _ = choose_path({"relay": R}, side(nat="full-cone", relay=R))
    assert p == "relay"


def test_an_unrecognised_nat_string_is_unknown():
    p, _ = choose_path(side(nat="carrier-grade-mystery", relay=R), side(nat="none", relay=R))
    assert p == "relay"


def test_no_punch_and_no_relay_says_none_rather_than_lying():
    p, why = choose_path(side(nat="symmetric"), side(nat="symmetric"))
    assert p == "none" and "no path" in why


@pytest.mark.parametrize("bad", [None, 42, "side", []])
def test_non_dict_sides_never_raise(bad):
    p, _ = choose_path(bad, bad)
    assert p == "none"
