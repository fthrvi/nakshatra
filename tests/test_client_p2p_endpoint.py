"""U4 C3: roster-gated libp2p activation endpoints in the hub client."""
from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import client as cli  # noqa: E402
from fabric import worker_join as wj  # noqa: E402
from sidecar_key import peer_id_from_node_pub  # noqa: E402


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


def _install_local_worker(tmp_path, monkeypatch, *, listen="[::]", port=5530):
    home = tmp_path / "home"
    key_path = home / ".nakshatra" / "keys" / "worker.ed25519"
    key_path.parent.mkdir(parents=True)
    private = ed25519.Ed25519PrivateKey.generate()
    key_path.write_bytes(private.private_bytes_raw())
    key_path.chmod(0o600)
    worker_env = home / ".nakshatra" / "worker.env"
    worker_env.write_text(f'PORT={port}\nLISTEN="{listen}"\n')
    monkeypatch.setenv("HOME", str(home))
    public = private.public_key().public_bytes_raw().hex()
    return home, public, peer_id_from_node_pub(public)


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


def test_consumer_roster_identity_is_never_admitted_as_activation_worker():
    roster = {PUBKEY: {"tier": "trusted", "tenant": "outside", "role": "consumer"}}
    with patch.object(cli, "_registry_urlopen",
                      side_effect=_pillar_open([_peer()], [_chain_entry()])):
        with pytest.raises(PermissionError, match="not admitted"):
            cli.build_chain_from_registry(
                "http://pillar", "public-model", roster_loader=lambda: roster,
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
    assert control.timeout == cli.SIDECAR_FORWARD_CONTROL_TIMEOUT_S == 20.0
    assert control.timeout > cli.SIDECAR_DIRECT_WAIT_S
    assert (resolved["address"], resolved["port"]) == ("127.0.0.1", 43123)
    assert resolved[cli._CLIENT_DIAL_ADDRESS] == "127.0.0.1:43123"
    assert resolved[cli._PUSH_DIAL_ADDRESS] is None
    assert resolved["peer_spki_hash"] == SPKI
    assert spec["address"] == ""  # resolution does not rewrite registry identity


def test_self_hop_by_node_public_key_uses_configured_ipv6_loopback(
        tmp_path, monkeypatch):
    home, public, _peer_id = _install_local_worker(tmp_path, monkeypatch)
    spec = {
        "id": "friendly-name", "node_id": public,
        "endpoint": _endpoint(), "peer_spki_hash": SPKI,
        "address": "", "port": 0,
    }

    with patch.object(cli, "_sidecar_forward") as forward:
        resolved = cli._resolve_activation_endpoint(spec, home=home)

    forward.assert_not_called()
    assert resolved[cli._CLIENT_DIAL_ADDRESS] == "[::1]:5530"
    assert (resolved["address"], resolved["port"]) == ("::1", 5530)
    assert resolved["peer_spki_hash"] == SPKI


def test_self_hop_by_sidecar_peer_id_and_remote_hop_form_mixed_chain(
        tmp_path, monkeypatch):
    home, _public, local_peer_id = _install_local_worker(
        tmp_path, monkeypatch, listen="0.0.0.0", port=5544)
    local = {
        "id": "bro", "node_id": "bro",
        "endpoint": {"kind": cli.ACTIVATION_ENDPOINT_KIND,
                     "peer_id": local_peer_id},
        "peer_spki_hash": "11" * 32, "address": "", "port": 0,
    }
    remote = {
        "id": "remote", "node_id": "remote",
        "endpoint": _endpoint(), "peer_spki_hash": SPKI,
        "address": "", "port": 0,
    }

    with patch.object(cli, "_sidecar_forward",
                      return_value=("127.0.0.1", 43123)) as forward:
        resolved = [cli._resolve_activation_endpoint(hop, home=home)
                    for hop in (local, remote)]

    assert [hop[cli._CLIENT_DIAL_ADDRESS] for hop in resolved] == [
        "127.0.0.1:5544", "127.0.0.1:43123",
    ]
    forward.assert_called_once_with(PEER_ID, socket_path=None)
    assert [hop["peer_spki_hash"] for hop in resolved] == ["11" * 32, SPKI]


def test_self_hop_never_guesses_a_missing_worker_port(tmp_path, monkeypatch):
    home, public, local_peer_id = _install_local_worker(tmp_path, monkeypatch)
    (home / ".nakshatra" / "worker.env").write_text('LISTEN="[::]"\n')
    spec = {
        "id": public, "node_id": public,
        "endpoint": {"kind": cli.ACTIVATION_ENDPOINT_KIND,
                     "peer_id": local_peer_id},
        "peer_spki_hash": SPKI,
    }

    with patch.object(cli, "_sidecar_forward") as forward:
        with pytest.raises(RuntimeError, match="valid PORT"):
            cli._resolve_activation_endpoint(spec, home=home)
    forward.assert_not_called()


def test_legacy_hop_resolver_supplies_client_and_colocated_push_addresses():
    spec = {
        "id": "legacy-a", "address": "198.51.100.7", "port": 5562,
        "internal_address": "127.0.0.1",
    }

    resolved = cli._resolve_activation_endpoint(spec)

    assert resolved[cli._CLIENT_DIAL_ADDRESS] == "198.51.100.7:5562"
    assert cli._push_dial_address(resolved) == "127.0.0.1:5562"
    assert resolved is not spec


def test_push_rejects_legacy_hop_that_bypassed_the_resolver():
    with pytest.raises(RuntimeError, match="not passed through"):
        cli._push_dial_address({
            "id": "legacy-a", "address": "198.51.100.7", "port": 5562,
        })


def test_typed_hop_can_never_fall_back_to_legacy_address_for_push():
    spec = {
        "id": "typed-a", "endpoint": _endpoint(),
        "peer_spki_hash": SPKI,
        # A populated legacy address must not evade direct-only policy.
        "address": "203.0.113.99", "port": 5562,
    }
    with patch.object(cli, "_sidecar_forward",
                      return_value=("127.0.0.1", 43123)):
        resolved = cli._resolve_activation_endpoint(spec)

    assert resolved[cli._CLIENT_DIAL_ADDRESS] == "127.0.0.1:43123"
    with pytest.raises(RuntimeError, match="client-relayed streaming"):
        cli._push_dial_address(resolved)


def test_push_gate_identifies_only_typed_downstream_hops():
    first = cli._resolve_activation_endpoint({
        "id": "first", "address": "192.0.2.1", "port": 5561,
    })
    last_spec = {
        "id": "last", "endpoint": _endpoint(), "peer_spki_hash": SPKI,
        "address": "192.0.2.2", "port": 5562,
    }
    with patch.object(cli, "_sidecar_forward",
                      return_value=("127.0.0.1", 43123)):
        last = cli._resolve_activation_endpoint(last_spec)

    assert cli._typed_push_downstream_workers([
        (first, object(), object()), (last, object(), object()),
    ]) == ["last"]


def test_push_gate_allows_typed_first_hop_when_downstream_is_legacy():
    first_spec = {
        "id": "first", "endpoint": _endpoint(), "peer_spki_hash": SPKI,
        "address": "192.0.2.1", "port": 5561,
    }
    with patch.object(cli, "_sidecar_forward",
                      return_value=("127.0.0.1", 43123)):
        first = cli._resolve_activation_endpoint(first_spec)
    last = cli._resolve_activation_endpoint({
        "id": "last", "address": "192.0.2.2", "port": 5562,
    })

    assert cli._typed_push_downstream_workers([
        (first, object(), object()), (last, object(), object()),
    ]) == []
    assert cli._push_dial_address(last) == "192.0.2.2:5562"


def test_recovery_rebuild_downgrades_push_when_alternate_is_typed(capsys):
    """The one replacement site must resolve and gate the replacement again."""
    class Info:
        protocol_capabilities = ["rpc_push"]

    first = cli._resolve_activation_endpoint({
        "id": "first", "address": "192.0.2.1", "port": 5561,
    })
    legacy_last = cli._resolve_activation_endpoint({
        "id": "last", "address": "192.0.2.2", "port": 5562,
    })
    initial = [(first, object(), Info()), (legacy_last, object(), Info())]
    assert cli._apply_streaming_push_gate(True, initial) is True

    typed_alternate = {
        "id": "last", "endpoint": _endpoint(), "peer_spki_hash": SPKI,
        # Recovery must not reuse this stale literal address.
        "address": "203.0.113.99", "port": 5562,
    }
    with patch.object(cli, "_sidecar_forward",
                      return_value=("127.0.0.1", 43123)) as forward:
        resolved_alternate = cli._resolve_activation_endpoint(typed_alternate)
    rebuilt = [(first, object(), Info()),
               (resolved_alternate, object(), Info())]

    assert cli._apply_streaming_push_gate(
        True, rebuilt, context="recovery") is False
    assert resolved_alternate[cli._CLIENT_DIAL_ADDRESS] == "127.0.0.1:43123"
    assert resolved_alternate[cli._PUSH_DIAL_ADDRESS] is None
    forward.assert_called_once_with(PEER_ID, socket_path=None)
    assert "[recovery] WARNING" in capsys.readouterr().err


def test_client_recovery_applies_typed_gate_to_rebuilt_chain(
        tmp_path, monkeypatch, capsys):
    """A real recovery rebuild downgrades before constructing its next push."""
    config = tmp_path / "chain.yaml"
    config.write_text(
        """\
model:
  id: public-model
workers:
  - id: first
    address: first.example
    port: 5561
    layer_range: [0, 4]
  - id: last
    address: last.example
    port: 5562
    layer_range: [4, 8]
    alternates:
      - id: last-alt
        address: stale.example
        port: 5562
        layer_range: [4, 8]
        peer_spki_hash: """ + SPKI + """
        endpoint:
          kind: libp2p-activation-v1
          peer_id: 12D3KooWorkerActivationPeer
""",
        encoding="utf-8",
    )

    class Tokenizer:
        def detokenize(self, _tokens):
            return b"generated"

    class Stream:
        def __init__(self, _stub, worker_id):
            self.worker_id = worker_id

        def close(self):
            pass

    def info_for(address):
        first = address == "first.example:5561"
        return SimpleNamespace(
            protocol_capabilities=["rpc_push"], protocol_version="0.1.0",
            layer_start=0 if first else 4, layer_end=4 if first else 8,
            has_token_embd=first, has_lm_head=not first, hidden_size=2,
            backend="test",
        )

    calls = []

    def inference_step(streamer, _payload, n_tokens, has_token_ids, **kwargs):
        calls.append((streamer.worker_id, bool(kwargs.get("chain"))))
        if kwargs.get("chain"):
            return struct.pack("<i", 201)
        if has_token_ids:
            return b"\x00" * (n_tokens * 2 * 4)
        return struct.pack("<i", 202)

    monkeypatch.setattr(sys, "argv", [
        "client.py", "--config", str(config), "--model-path", "model.gguf",
        "--use-streaming-push", "--simulate-fail-step", "1",
        "--max-tokens", "2",
    ])
    with (
        patch.object(cli, "_admit_activation_workers"),
        patch.object(cli, "_sidecar_forward",
                     return_value=("127.0.0.1", 43123)) as forward,
        patch.object(cli, "_open_chain_channel", side_effect=lambda addr, *_: addr),
        patch.object(cli.pb_grpc, "NakshatraStub", side_effect=lambda channel: channel),
        patch.object(cli, "info_with_startup_wait",
                     side_effect=lambda stub, *_args, **_kwargs: info_for(stub)),
        patch.object(cli, "tokenize_local", return_value=([101], Tokenizer())),
        patch.object(cli, "InferenceStream", Stream),
        patch.object(cli, "call_inference_step", side_effect=inference_step),
    ):
        cli.main()

    # First attempt used one server-to-server push.  After the typed alternate
    # replaced the downstream hop, recovery relayed through both client streams.
    assert calls == [("first", True), ("first", False), ("last-alt", False)]
    forward.assert_called_once_with(PEER_ID, socket_path=None)
    err = capsys.readouterr().err
    assert "[recovery] WARNING: push requested with typed activation" in err
    assert "attempts exhausted" not in err


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
