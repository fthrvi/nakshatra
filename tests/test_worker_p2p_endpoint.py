"""U4 C3 worker registration endpoint tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import worker  # noqa: E402


def test_worker_omits_endpoint_without_release_peer_id():
    assert worker.activation_endpoint_from_env({}) is None


def test_worker_registers_activation_endpoint_from_release_peer_id():
    peer_id = "12D3KooReleaseDerivedPeer"
    assert worker.activation_endpoint_from_env(
        {"NAKSHATRA_P2P_PEER_ID": peer_id}) == {
            "kind": "libp2p-activation-v1",
            "peer_id": peer_id,
        }


def test_worker_refuses_line_injection_in_peer_id():
    with pytest.raises(ValueError, match="invalid"):
        worker.activation_endpoint_from_env(
            {"NAKSHATRA_P2P_PEER_ID": "peer\nFORWARD attacker"})


def test_worker_refuses_non_ascii_peer_id():
    with pytest.raises(ValueError, match="invalid"):
        worker.activation_endpoint_from_env(
            {"NAKSHATRA_P2P_PEER_ID": "12D3KooUnicodePeeré"})
