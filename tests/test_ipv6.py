"""Tests for ipv6.select_global_ipv6 function."""

import pytest
from ipv6 import select_global_ipv6


def test_select_global_ipv6_basic():
    """A global address selected."""
    result = select_global_ipv6(["2001:db8::1"])
    assert result == "2001:db8::1"


def test_select_global_ipv6_link_local_rejected():
    """fe80::1 rejected."""
    result = select_global_ipv6(["fe80::1"])
    assert result is None


def test_select_global_ipv6_loopback_rejected():
    """::1 rejected."""
    result = select_global_ipv6(["::1"])
    assert result is None


def test_select_global_ipv6_ula_rejected():
    """fd00::1 (ULA) rejected."""
    result = select_global_ipv6(["fd00::1"])
    assert result is None


def test_select_global_ipv6_ipv4_mapped_rejected():
    """::ffff:192.0.2.1 rejected."""
    result = select_global_ipv6(["::ffff:192.0.2.1"])
    assert result is None


def test_select_global_ipv6_multicast_rejected():
    """ff02::1 rejected."""
    result = select_global_ipv6(["ff02::1"])
    assert result is None


def test_select_global_ipv6_garbage_rejected():
    """A garbage string rejected."""
    result = select_global_ipv6(["not-an-address"])
    assert result is None


def test_select_global_ipv6_non_string_rejected():
    """A non-string entry rejected."""
    result = select_global_ipv6([123, "2001:db8::1"])
    assert result == "2001:db8::1"


def test_select_global_ipv6_empty_list():
    """An empty list giving None."""
    result = select_global_ipv6([])
    assert result is None


def test_select_global_ipv6_only_rejects():
    """A list of only rejects giving None."""
    result = select_global_ipv6(["fe80::1", "::1", "fd00::1"])
    assert result is None


def test_select_global_ipv6_eui64_preferred():
    """An EUI-64 address preferred over a random-looking one."""
    # EUI-64 address: 2001:db8::1234:56ff:fe78:9abc (contains fffe in interface identifier)
    # Random-looking address: 2001:db8::abcd:ef01:2345:6789
    result = select_global_ipv6([
        "2001:db8::abcd:ef01:2345:6789",
        "2001:db8::1234:56ff:fe78:9abc"
    ])
    assert result == "2001:db8::1234:56ff:fe78:9abc"


def test_select_global_ipv6_determinism():
    """The same list in two different orders giving the same answer."""
    addresses = [
        "2001:db8::1",
        "2001:db8::2",
        "2001:db8::3"
    ]
    result1 = select_global_ipv6(addresses)
    result2 = select_global_ipv6(list(reversed(addresses)))
    assert result1 == result2

# ── regression: real addresses from a real machine ────────────────────────────────────────
# ⚠️ These are the three global v6 addresses a real dual-stack Linux box presented on
# 2026-09-04. The selector picked the PRIVACY address over the stable one, because the
# stability heuristic counted the CHARACTER '0' rather than zero GROUPS: the compressed
# `2601:8c0:681:f790::388c` has two '0' characters, the rotating
# `2601:8c0:681:f790:4dd4:c670:418c:7404` has four.
#
# Twelve synthetic tests missed it because not one of them used a COMPRESSED address. Real
# data found it in a single call. That is the whole argument for running against a live box
# before believing a heuristic.
REAL_BOX = [
    "2601:8c0:681:f790::388c",                        # stable, delegated /128
    "2601:8c0:681:f790:4dd4:c670:418c:7404",          # temporary / privacy
    "2601:8c0:681:f790:903f:e789:1f8d:9296",          # dynamic mngtmpaddr
]


def test_prefers_the_stable_address_over_the_privacy_ones():
    assert select_global_ipv6(REAL_BOX) == "2601:8c0:681:f790::388c"


def test_that_preference_does_not_depend_on_list_order():
    import itertools
    for perm in itertools.permutations(REAL_BOX):
        assert select_global_ipv6(list(perm)) == "2601:8c0:681:f790::388c"


def test_eui64_still_outranks_a_low_iid():
    """The EUI-64 signal is stronger evidence of stability than a mostly-zero IID."""
    eui = "2001:db8::201:2ff:fffe:3344"
    assert select_global_ipv6([eui, "2001:db8::1"]) == eui


def test_a_privacy_only_box_still_gets_an_answer():
    """⚠️ A box with ONLY privacy addresses is still reachable — the heuristic picks a worse
    address, it does not refuse. A wrong guess costs a re-announce, not a failure."""
    only_privacy = REAL_BOX[1:]
    assert select_global_ipv6(only_privacy) in only_privacy
