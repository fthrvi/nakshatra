"""Tests for v1.0 §7 live handshake glue (wire/handshake.py).

Proves the Info-handshake maps to negotiation: a worker advertises control/vN
caps, a client recovers a peer's versions (caps or legacy protocol_version),
negotiates the agreed version, and refuses an incompatible peer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from wire.version import SUPPORTED_CONTROL_VERSIONS, VersionNegotiationError  # noqa: E402
from wire.handshake import (  # noqa: E402
    advertise_capabilities, peer_supported_versions, negotiate_handshake,
    CAP_CONTROL_PREFIX)


def test_advertise_includes_control_tokens():
    caps = advertise_capabilities()
    assert caps == [f"{CAP_CONTROL_PREFIX}{v}" for v in SUPPORTED_CONTROL_VERSIONS]


def test_peer_versions_from_caps():
    assert peer_supported_versions("0.1.0", ["streaming", "control/v1", "control/v2"]) == [1, 2]


def test_peer_versions_legacy_fallback():
    # a pre-§7 worker: feature caps but no control/vN → fall back to protocol_version
    assert peer_supported_versions("0.1.0", ["streaming", "rpc_push"]) == [1]
    assert peer_supported_versions("0.1.0", []) == [1]


def test_peer_versions_garbage_protocol_is_v1():
    assert peer_supported_versions("garbage", []) == [1]


def test_negotiate_handshake_compatible():
    # our own advertised caps must always negotiate against us
    agreed = negotiate_handshake("0.1.0", advertise_capabilities())
    assert agreed in SUPPORTED_CONTROL_VERSIONS


def test_negotiate_handshake_legacy_worker_ok():
    # legacy worker (no control caps, "0.1.0") → v1, which we support
    assert negotiate_handshake("0.1.0", ["streaming", "rpc_push"]) == 1


def test_negotiate_handshake_incompatible_raises():
    with pytest.raises(VersionNegotiationError):
        negotiate_handshake("0.1.0", ["control/v99"], local=[1, 2])
