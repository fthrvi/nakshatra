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