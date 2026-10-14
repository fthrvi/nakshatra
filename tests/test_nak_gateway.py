from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from network import nak  # noqa: E402
import client  # noqa: E402
import nakshatra_serve as serve  # noqa: E402


PIN = "17" * 32


def _args(**overrides):
    values = {
        "port": 11600,
        "bind": "127.0.0.1",
        "allow_lan": False,
        "pillar_url": "https://pillar.example:7778",
        "pillar_spki": PIN,
        "ledger_url": "https://pillar.example/ledger",
        "models": "auto",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gateway_on_writes_loopback_enforced_opt_in(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    calls = []
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: False)
    monkeypatch.setattr(nak, "_gateway_installer", lambda _sp, action: calls.append(action))

    nak._gateway_on(object(), _args())

    body = (home / ".nakshatra" / "gateway.env").read_text()
    assert "BIND=\"127.0.0.1\"" in body
    assert "PORT=11600" in body
    assert "MODELS=\"auto\"" in body
    assert "NAKSHATRA_LEDGER_MODE=enforce" in body
    assert "NAKSHATRA_CREDITS=1" in body
    assert f"STHAMBHA_PILLAR_SPKI_SHA256={PIN}" in body
    assert "NAKSHATRA_CHAIN_TLS_MODE=required" in body
    assert "NAKSHATRA_LEDGER_REQUESTER" not in body
    assert (home / ".nakshatra" / "inference.env").is_file()
    assert calls == ["enable"]


def test_gateway_refuses_non_loopback_without_allow_lan(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: pytest.fail("must not enable"))

    with pytest.raises(SystemExit, match="non-loopback"):
        nak._gateway_on(object(), _args(bind="0.0.0.0"))
    assert not (home / ".nakshatra" / "gateway.env").exists()


def test_gateway_allows_lan_only_with_explicit_flag(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    calls = []
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: False)
    monkeypatch.setattr(nak, "_gateway_installer", lambda _sp, action: calls.append(action))

    nak._gateway_on(object(), _args(bind="192.0.2.4", allow_lan=True))

    assert 'BIND="192.0.2.4"' in (home / ".nakshatra" / "gateway.env").read_text()
    assert calls == ["enable"]


def test_gateway_on_rejects_engine_url_registry_before_enable(tmp_path, monkeypatch):
    home = tmp_path / "home"
    models = tmp_path / "models.yaml"
    models.write_text("models:\n  - name: local\n    engine_url: http://127.0.0.1:11434\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: pytest.fail("must not enable"))

    with pytest.raises(SystemExit, match="refuses unmetered proxy models"):
        nak._gateway_on(object(), _args(models=str(models)))

    assert not (home / ".nakshatra" / "gateway.env").exists()


def test_gateway_off_removes_opt_in_without_starting_units(tmp_path, monkeypatch):
    home = tmp_path / "home"
    env = home / ".nakshatra" / "gateway.env"
    env.parent.mkdir(parents=True)
    env.write_text("NAKSHATRA_LEDGER_MODE=enforce\n")
    calls = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda _sp, action: calls.append(action))

    nak._gateway_off(object())

    assert calls == ["disable"]
    assert not env.exists()


def test_release_has_separate_gateway_service():
    spec = json.loads((ROOT / "release" / "spec.json").read_text())
    service = spec["services"]["nakshatra-gateway"]
    assert service["requires"].endswith("/.nakshatra/gateway.env")
    assert service["profile"] == "inference"
    assert service["environment"] == {"PORT": "11600", "BIND": "127.0.0.1", "MODELS": "auto"}
    assert "nakshatra-unconscious" in spec["services"]


def test_auto_models_uses_signed_peers_and_only_complete_local_models(tmp_path, monkeypatch):
    home = tmp_path / "home"
    model = home / ".nakshatra" / "worker" / "slice.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fake tokenizer-bearing slice")
    (home / ".nakshatra" / "worker.env").write_text(
        f'MODEL_ID="model-a"\nSUB_GGUF={model}\n')
    peers = [
        {"is_online": True, "layer_offerings": [
            {"model_id": "model-a", "layer_start": 0, "layer_end": 8},
            {"model_id": "model-gap", "layer_start": 0, "layer_end": 4},
        ]},
        {"is_online": True, "layer_offerings": [
            {"model_id": "model-a", "layer_start": 8, "layer_end": 16},
            {"model_id": "model-gap", "layer_start": 8, "layer_end": 12},
        ]},
    ]
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(client, "_fetch_registry_peers", lambda url: peers)

    models = serve._discover_auto_models("https://pillar.example:7778")

    assert list(models) == ["model-a"]
    assert models["model-a"].tokenizer_gguf == str(model)
    assert models["model-a"].num_layers == 16


@pytest.mark.parametrize(
    ("reason", "status"),
    [("insufficient credits (balance 1, cost 2)", 402),
     ("ledger unreachable (timeout)", 503)],
)
def test_gateway_maps_credit_refusals_to_clear_http_status(reason, status):
    backend = serve.ChainChatBackend()
    backend._ledger_hook_cached = SimpleNamespace(
        enabled=True, wants_receipt=True, fail_closed=True,
        gate=lambda _cost: (False, reason), settle=lambda _path: None)
    entry = serve.ModelEntry(
        name="model-a", tokenizer_gguf="/unused.gguf",
        registry_url="https://pillar.example", num_layers=2)

    with pytest.raises(serve.CreditGateError) as caught:
        backend.generate(entry, "hello", 1, {})

    assert caught.value.status == status
    assert reason in str(caught.value)


def test_gateway_gate_exception_denies_in_enforce_mode():
    backend = serve.ChainChatBackend()
    backend._ledger_hook_cached = SimpleNamespace(
        enabled=True, wants_receipt=True, fail_closed=True,
        gate=lambda _cost: (_ for _ in ()).throw(RuntimeError("ledger exploded")),
        settle=lambda _path: None)
    entry = serve.ModelEntry(
        name="model-a", tokenizer_gguf="/unused.gguf",
        registry_url="https://pillar.example", num_layers=2)

    with pytest.raises(serve.CreditGateError) as caught:
        backend.generate(entry, "hello", 1, {})

    assert caught.value.status == 503
    assert "ledger unavailable: ledger exploded" in str(caught.value)


def test_gateway_chain_client_requires_worker_tls(monkeypatch):
    monkeypatch.setenv("NAKSHATRA_CHAIN_TLS_MODE", "required")
    entry = serve.ModelEntry(
        name="model-a", tokenizer_gguf="/unused.gguf",
        registry_url="https://pillar.example", num_layers=2)

    cmd = serve.ChainChatBackend()._cmd(entry, "hello", 1)

    assert cmd[cmd.index("--tls-mode") + 1] == "required"
