"""U4 C3: roster-gated libp2p activation endpoints in the hub client."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import client as cli  # noqa: E402
from fabric import worker_join as wj  # noqa: E402


PEER_ID = "12D3KooWorkerActivationPeer"
PUBKEY = "ab" * 32
SPKI = "cd" * 32


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _pillar_open(peers: list[dict], chain: list[dict]):
    def open_url(url, timeout=10):
        payload = ({"chain": chain, "warnings": []}
                   if "/chain?" in url else {"peers": peers})
        return _Response(json.dumps(payload).encode())
    return open_url


def _endpoint():
    return {"kind": cli.ACTIVATION_ENDPOINT_KIND, "peer_id": PEER_ID}


def _peer(**updates):
    peer = {
        "node_id": "worker-a",
        "node_type": "compute",
        "address": "",
        "is_online": True,
        "public_key_hex": PUBKEY,
        "peer_spki_hash": SPKI,
        "endpoint": _endpoint(),
        # This is attacker-controlled registry data and must never be used as
        # the operator trust decision.
        "tier": "self",
        "layer_offerings": [{
            "model_id": "public-model", "model_sha256": "",
            "layer_start": 0, "layer_end": 8,
        }],
        "hardware": {"gpus": []},
    }
    peer.update(updates)
    return peer


def _chain_entry():
    return {
        "node_id": "worker-a", "address": "",
        "layer_start": 0, "layer_end": 8,
        "peer_spki_hash": SPKI, "endpoint": _endpoint(),
    }


def _eligible_at(min_tier: str):
    rank = {"stranger": 0, "known": 1, "trusted": 2, "self": 3}

    def eligible(model, standings):
        return wj.eligible_workers(
            model, standings, min_tier_fn=lambda _model: min_tier, rank=rank)
    return eligible


def test_pillar_activation_chain_is_admitted_by_operator_roster():
    roster = {PUBKEY: {"tier": "stranger", "tenant": "outside"}}
    with patch.object(cli, "_registry_urlopen",
                      side_effect=_pillar_open([_peer()], [_chain_entry()])):
        chain = cli.build_chain_from_registry(
            "http://pillar", "public-model", roster_loader=lambda: roster,
            eligible_fn=_eligible_at("stranger"))

    assert chain[0]["node_id"] == "worker-a"
    assert chain[0]["endpoint"] == _endpoint()
    assert chain[0]["peer_spki_hash"] == SPKI


def test_unrostered_activation_worker_is_rejected_before_forward():
    with patch.object(cli, "_registry_urlopen",
                      side_effect=_pillar_open([_peer()], [_chain_entry()])):
        with pytest.raises(PermissionError, match="identity firewall"):
            cli.build_chain_from_registry(
                "http://pillar", "public-model", roster_loader=lambda: {},
                eligible_fn=_eligible_at("stranger"))


def test_worker_claimed_tier_cannot_override_operator_roster():
    # Pillar says tier=self, but the operator roster says stranger.  A trusted
    # model must reject it.
    roster = {PUBKEY: {"tier": "stranger", "tenant": "outside"}}
    with patch.object(cli, "_registry_urlopen", side_effect=_pillar_open(
            [_peer(tier="self")], [_chain_entry()])):
        with pytest.raises(PermissionError, match="tier stranger < trusted"):
            cli.build_chain_from_registry(
                "http://pillar", "sensitive-model", roster_loader=lambda: roster,
                eligible_fn=_eligible_at("trusted"))


def test_chain_spki_must_match_full_node_record():
    roster = {PUBKEY: {"tier": "stranger", "tenant": "outside"}}
    chain_entry = _chain_entry()
    chain_entry["peer_spki_hash"] = "ef" * 32
    with patch.object(cli, "_registry_urlopen", side_effect=_pillar_open(
            [_peer()], [chain_entry])):
        with pytest.raises(PermissionError, match="SPKI mismatch"):
            cli.build_chain_from_registry(
                "http://pillar", "public-model", roster_loader=lambda: roster,
                eligible_fn=_eligible_at("stranger"))


def test_sidecar_forward_socket_and_spki_preservation():
    class ControlSocket:
        def __init__(self, response):
            self.response = response
            self.sent = b""

        def settimeout(self, timeout):
            self.timeout = timeout

        def connect(self, path):
            self.path = path

        def sendall(self, data):
            self.sent += data

        def recv(self, size):
            response, self.response = self.response, b""
            return response

        def close(self):
            pass

    control = ControlSocket(b"OK 127.0.0.1:43123\n")
    with patch.object(cli.socket, "socket", return_value=control):
        spec = {"endpoint": _endpoint(), "peer_spki_hash": SPKI,
                "address": "", "port": 0}
        resolved = cli._resolve_activation_endpoint(
            spec, socket_path="/run/user/1000/nakshatra/p2p.sock")

    assert control.sent == f"FORWARD {PEER_ID}\n".encode("ascii")
    assert control.path == "/run/user/1000/nakshatra/p2p.sock"
    assert (resolved["address"], resolved["port"]) == ("127.0.0.1", 43123)
    assert resolved["peer_spki_hash"] == SPKI
    assert spec["address"] == ""  # resolution does not rewrite registry identity


def test_sidecar_forward_rejects_non_loopback_reply():
    class BadControlSocket:
        def settimeout(self, timeout): pass
        def connect(self, path): pass
        def sendall(self, data): pass
        def recv(self, size): return b"OK 10.0.0.5:43123\n"
        def close(self): pass

    with patch.object(cli.socket, "socket", return_value=BadControlSocket()):
        with pytest.raises(RuntimeError, match="loopback"):
            cli._sidecar_forward(PEER_ID, "/run/user/1000/nakshatra/p2p.sock")


def test_sidecar_forward_wraps_socket_creation_failure():
    with patch.object(cli.socket, "socket", side_effect=OSError("socket denied")):
        with pytest.raises(RuntimeError, match="sidecar FORWARD failed: socket denied"):
            cli._sidecar_forward(PEER_ID, "/run/user/1000/nakshatra/p2p.sock")


def test_activation_peer_id_must_be_ascii():
    with pytest.raises(RuntimeError, match="invalid peer_id"):
        cli._parse_activation_endpoint({
            "kind": cli.ACTIVATION_ENDPOINT_KIND,
            "peer_id": "12D3KooUnicodePeeré",
        })


def test_activation_endpoint_refuses_missing_spki_before_sidecar_call():
    spec = {"endpoint": _endpoint(), "address": "", "port": 0}
    with patch.object(cli, "_sidecar_forward") as forward:
        with pytest.raises(RuntimeError, match="registered worker SPKI"):
            cli._resolve_activation_endpoint(spec)
    forward.assert_not_called()


def test_activation_endpoint_refuses_tls_off_before_sidecar_call():
    spec = {"endpoint": _endpoint(), "peer_spki_hash": SPKI,
            "address": "", "port": 0}
    with patch.object(cli, "_sidecar_forward") as forward:
        with pytest.raises(RuntimeError, match="require TLS"):
            cli._resolve_activation_endpoint(spec, tls_mode="off")
    forward.assert_not_called()


def test_spki_index_supports_node_and_peer_identity_keys():
    with patch.object(cli, "_registry_urlopen",
                      side_effect=_pillar_open([_peer()], [])):
        index = cli._fetch_spki_index("http://pillar")
    assert index["worker-a"] == SPKI
    assert index[PEER_ID] == SPKI
