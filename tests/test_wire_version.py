"""Tests for v1.0 §7 control-plane wire-version negotiation (P4).

Proves ALPN-style "highest common version or clean reject", legacy "0.1.0"
mapping, and that routing filters out version-incompatible peers BEFORE pinning
(the pre-join clean reject — never a silent attempt against an incompatible wire).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from nakshatra_auth import generate_keypair  # noqa: E402
from wire.version import (  # noqa: E402
    negotiate, negotiate_or_raise, is_compatible, parse_version,
    normalize_supported, VersionNegotiationError, LEGACY_VERSION_STRING,
    SUPPORTED_CONTROL_VERSIONS)
from discovery.nakshatra_listing import NakshatraListing  # noqa: E402
from discovery.relay import InMemoryRelay  # noqa: E402
from routing.model_router import route_or_local, Decision  # noqa: E402


# ── version parsing ───────────────────────────────────────────────────

def test_parse_legacy_and_numeric():
    assert parse_version(LEGACY_VERSION_STRING) == 1
    assert parse_version(2) == 2
    assert parse_version("3") == 3
    assert parse_version("1.4.0") == 1   # dotted → major
    with pytest.raises(ValueError):
        parse_version("garbage")
    with pytest.raises(ValueError):
        parse_version(True)              # bool is not a version


def test_normalize_empty_is_legacy_v1():
    assert normalize_supported([]) == [1]
    assert normalize_supported(None) == [1]
    assert normalize_supported(["0.1.0"]) == [1]
    assert normalize_supported([2, 1, "garbage", 1]) == [1, 2]


# ── negotiation ───────────────────────────────────────────────────────

def test_negotiate_picks_highest_common():
    res = negotiate([1, 2, 3], [2, 3, 4])
    assert res.compatible and res.agreed == 3

def test_negotiate_legacy_peer_compatible():
    # a pre-§7 peer advertises nothing → legacy v1; we support v1 → agree on 1
    assert negotiate(SUPPORTED_CONTROL_VERSIONS, []).agreed == 1

def test_no_overlap_is_clean_reject():
    res = negotiate([1, 2], [5, 6])
    assert not res.compatible and res.agreed is None
    with pytest.raises(VersionNegotiationError) as ei:
        negotiate_or_raise([1, 2], [5, 6])
    assert ei.value.local == [1, 2] and ei.value.remote == [5, 6]  # debuggable

def test_is_compatible_shortcut():
    assert is_compatible([1], [1, 2])
    assert not is_compatible([9], [1, 2])


# ── routing filters incompatible peers (§7 pre-join reject) ───────────

def _peer(node_id, serving, ms, endpoint, supported):
    priv, pub = generate_keypair()
    l = NakshatraListing(mesh_id="m1", node_id=node_id, ed25519_pubkey_hex=pub,
                         serving=serving, measured_decode_ms_per_layer=ms,
                         endpoint_hint=endpoint, supported_protocol=supported)
    l.sign(priv)
    return l


def test_incompatible_peer_is_not_routed_even_if_faster():
    relay = InMemoryRelay()
    # the fastest peer speaks a future protocol we can't talk to → must be skipped
    relay.publish(_peer("future", ["m"], ms=0.5, endpoint="http://future:8080",
                        supported=[99]))
    relay.publish(_peer("compatible", ["m"], ms=4.0, endpoint="http://ok:8080",
                        supported=list(SUPPORTED_CONTROL_VERSIONS)))
    t = route_or_local("m", ["local"], relay, mesh_id="m1")
    assert t.decision is Decision.ROUTE
    assert t.peer.node_id == "compatible"   # not "future", despite it being faster


def test_legacy_peer_still_routable():
    relay = InMemoryRelay()
    # a pre-§7 peer advertises no supported_protocol → treated as legacy v1 → ok
    relay.publish(_peer("legacy", ["m"], ms=3.0, endpoint="http://legacy:8080",
                        supported=[]))
    t = route_or_local("m", ["local"], relay, mesh_id="m1")
    assert t.decision is Decision.ROUTE and t.peer.node_id == "legacy"


def test_all_incompatible_is_not_found():
    relay = InMemoryRelay()
    relay.publish(_peer("a", ["m"], ms=1.0, endpoint="http://a:8080", supported=[42]))
    t = route_or_local("m", ["local"], relay, mesh_id="m1")
    assert t.decision is Decision.NOT_FOUND   # clean reject, not a silent attempt
