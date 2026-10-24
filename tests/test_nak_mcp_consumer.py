from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import nakshatra_auth as auth  # noqa: E402
import ledger_client  # noqa: E402
from network import nak  # noqa: E402
from network import nak_mcp  # noqa: E402
from network.nak_mcp import Server  # noqa: E402


class _UnusedClient:
    pass


class _Response:
    def __init__(self, value):
        self.body = io.BytesIO(json.dumps(value).encode())

    def read(self):
        return self.body.read()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _write_gateway_env(home: Path, gateway_port: int, ledger_url: str = ""):
    path = home / ".nakshatra" / "gateway.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"PORT={gateway_port}\nBIND=127.0.0.1\n"
        + (f'NAKSHATRA_LEDGER_URL="{ledger_url}"\n' if ledger_url else "")
        + "NAKSHATRA_LEDGER_MODE=enforce\n"
    )


def test_tools_list_contains_consumer_tools():
    names = {row["name"] for row in Server(_UnusedClient()).list_tools()}
    assert {"nak_models", "nak_ask", "nak_balance", "nak_wallet", "nak_worker_status"} <= names
    assert not {"nak_wallet_bind", "nak_withdraw"} & names


def test_nak_ask_uses_only_local_gateway_and_reports_charge(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    _write_gateway_env(home, 11600)
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/v1/models"):
            return _Response({"object": "list", "data": [{"id": "model-a"}]})
        payload = json.loads(request.data)
        assert request.full_url == "http://127.0.0.1:11600/v1/chat/completions"
        assert payload["model"] == "model-a"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        return _Response({
            "choices": [{"message": {"role": "assistant", "content": "world"}}],
            "nakshatra": {"credits_charged": 12},
        })

    monkeypatch.setattr(nak_mcp.urlrequest, "urlopen", fake_urlopen)
    server = Server(_UnusedClient())
    assert server.call("nak_models", {}) == "model-a"
    answer = server.call("nak_ask", {"prompt": "hello", "max_tokens": 9})
    assert answer == "world\nCredits charged: 12"
    assert all(request.full_url.startswith("http://127.0.0.1:11600/v1/") for request in requests)


def test_nak_ask_refuses_cleanly_when_gateway_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with pytest.raises(Exception, match="run: nak gateway on"):
        Server(_UnusedClient()).call("nak_ask", {"prompt": "hello"})


def test_nak_balance_is_signed_for_signer_with_no_account_parameter(tmp_path, monkeypatch):
    home = tmp_path / "home"
    key_path = home / ".nakshatra" / "keys" / "worker.ed25519"
    key_path.parent.mkdir(parents=True)
    private, public = auth.generate_keypair()
    key_path.write_bytes(private)
    key_path.chmod(0o600)
    observed = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(auth, "WORKER_KEY_PATH", key_path)
    _write_gateway_env(home, 11600, "http://ledger.test")

    def fake_urlopen(request, timeout):
        observed.append((request.full_url, request.get_header("Authorization")))
        return _Response({"balance": 77, "tier": "known", "keyid": "must-not-leak"})

    monkeypatch.setattr(ledger_client.urllib.request, "urlopen", fake_urlopen)
    output = Server(_UnusedClient()).call("nak_balance", {})

    assert json.loads(output) == {"balance": 77, "tier": "known"}
    assert observed[0][0] == "http://ledger.test/balance" and "account" not in observed[0][0]
    parts = dict(re.findall(r'(keyid|sig|ts)="([^"]+)"', observed[0][1]))
    assert parts["keyid"] == public
    assert auth.verify_request(public, "GET", "/balance", b"", int(parts["ts"]), parts["sig"])
    assert private.hex() not in output and public not in output


def test_nak_wallet_is_read_only_signed_and_redacts_key_material(tmp_path, monkeypatch):
    home = tmp_path / "home"
    key_path = home / ".nakshatra" / "keys" / "worker.ed25519"
    key_path.parent.mkdir(parents=True)
    private, public = auth.generate_keypair()
    key_path.write_bytes(private)
    key_path.chmod(0o600)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(auth, "WORKER_KEY_PATH", key_path)
    _write_gateway_env(home, 11600, "http://ledger.test")
    observed = []

    def fake_urlopen(request, timeout):
        observed.append(request)
        if request.full_url.endswith("/wallet/status"):
            return _Response({
                "account": "alice", "tier": "known", "balance": 200_000_000,
                "withdrawable": 100_000_000, "balance_usdc": "2.00000000",
                "withdrawable_usdc": "1.00000000",
                "private_key": private.hex(), "key_path": str(key_path),
                "pending_withdrawals": [{"id": "w1", "authorization": "secret"}],
            })
        return _Response({
            "deposit_address": "11111111111111111111111111111111",
            "keyid": public,
        })

    monkeypatch.setattr(ledger_client.urllib.request, "urlopen", fake_urlopen)
    output = Server(_UnusedClient()).call("nak_wallet", {})
    value = json.loads(output)

    assert set(value) == {"deposit_address", "status"}
    assert value["deposit_address"] == "1" * 32
    assert value["status"]["balance"] == 200_000_000
    assert value["status"]["pending_withdrawals"] == [{"id": "w1"}]
    assert private.hex() not in output and public not in output
    assert str(key_path) not in output and "secret" not in output
    assert [request.full_url for request in observed] == [
        "http://ledger.test/wallet/status", "http://ledger.test/wallet",
    ]
    for request, path in zip(observed, ("/wallet/status", "/wallet")):
        parts = dict(re.findall(r'(keyid|sig|ts)="([^"]+)"',
                                request.get_header("Authorization")))
        assert auth.verify_request(
            public, "GET", path, b"", int(parts["ts"]), parts["sig"]
        )


def test_worker_status_redacts_secret_shaped_fields_and_key_paths(monkeypatch):
    monkeypatch.setattr(nak, "_worker_status", lambda _sp: {
        "configured": True,
        "unit": {"active": "active"},
        "preflight": {
            "schema": "nak-worker-preflight/1",
            "packages": [{"id": "llama32-3b", "verdict": "can serve"}],
        },
        "token": "secret",
        "build": {"daemon_bin": "/home/u/.nakshatra/keys/worker.ed25519"},
    })
    output = Server(_UnusedClient()).call("nak_worker_status", {})
    assert "secret" not in output and "worker.ed25519" not in output
    parsed = json.loads(output)
    assert parsed["configured"] is True
    assert parsed["preflight"]["packages"][0]["verdict"] == "can serve"
