from __future__ import annotations

import json
import sys
import threading
import time
from urllib.error import HTTPError
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from network import nak  # noqa: E402
import client  # noqa: E402
import nakshatra_serve as serve  # noqa: E402


PIN = "17" * 32
PACKAGE_SIGNER = "23" * 32


@pytest.fixture(autouse=True)
def _signed_release_package_signer(monkeypatch):
    monkeypatch.setattr(
        nak, "load_release_package_signers", lambda: frozenset({PACKAGE_SIGNER}))


def _wait_for_model_names(models, expected, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if list(models) == expected:
            return True
        time.sleep(0.005)
    return list(models) == expected


def _completed_chain_with_receipt(cmd, *, prompt=7, generated=2, text="settled"):
    receipt_path = Path(cmd[cmd.index("--receipt-out") + 1])
    receipt_path.write_text(json.dumps({
        "n_prompt": prompt,
        "n_generated": generated,
        "chain": [{"node_id": "own-node"}],
    }))
    return SimpleNamespace(
        returncode=0, stderr="", stdout=f"[chain] gen:  '{text}'\n")


def _args(**overrides):
    values = {
        "port": 11600,
        "bind": "127.0.0.1",
        "allow_lan": False,
        "pillar_url": "https://pillar.example:7778",
        "pillar_spki": PIN,
        "ledger_url": "https://pillar.example/ledger",
        "models": "auto",
        "package_url": "https://models.example/llama32-3b",
        "package_pubkey": PACKAGE_SIGNER,
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
    assert 'PACKAGE_URL="https://models.example/llama32-3b"' in body
    assert f"PACKAGE_TRUSTED_PUBKEY={PACKAGE_SIGNER}" in body
    assert "NAKSHATRA_LEDGER_MODE=enforce" in body
    assert "NAKSHATRA_CREDITS=1" in body
    assert f"STHAMBHA_PILLAR_SPKI_SHA256={PIN}" in body
    assert "NAKSHATRA_CHAIN_TLS_MODE=required" in body
    assert "NAKSHATRA_LEDGER_REQUESTER" not in body
    assert (home / ".nakshatra" / "inference.env").is_file()
    assert calls == ["enable"]


def test_gateway_auto_requires_package_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: pytest.fail("must not enable"))

    with pytest.raises(SystemExit, match="package-url"):
        nak._gateway_on(object(), _args(package_url=None, package_pubkey=None))


def test_gateway_auto_uses_join_url_but_ignores_join_key(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra" / "consumer-standing.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({
        "role": "consumer", "tier": "known", "operator": "op-1",
        "package_url": "https://models.example/from-join",
        "package_pubkey": "45" * 32,  # unsigned onboarding data: never a trust root
    }))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: False)
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)

    nak._gateway_on(object(), _args(package_url=None, package_pubkey=None))

    body = (home / ".nakshatra" / "gateway.env").read_text()
    assert 'PACKAGE_URL="https://models.example/from-join"' in body
    assert "PACKAGE_TRUSTED_PUBKEY" not in body


def test_gateway_package_pubkey_may_only_narrow_release_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: pytest.fail("must not enable"))

    with pytest.raises(SystemExit, match="may only narrow"):
        nak._gateway_on(object(), _args(package_pubkey="45" * 32))


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
    signer_policy = json.loads((ROOT / spec["package_signers"]["path"]).read_text())
    assert signer_policy == {
        "schema": "nak-package-signers/1",
        "keys": ["2a4d231e7a8a5a97fa89c429e7e3e8900a57f1d5cf780ca67ed053376c68dc77"],
    }
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


def test_auto_models_retry_409_then_appear_without_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    appeared = threading.Event()
    calls = []
    entry = serve.ModelEntry(name="llama32-3b", tokenizer_gguf="/fake.gguf",
                             registry_url="https://pillar.example", num_layers=28)

    def discover(_url):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise HTTPError("https://pillar.example/peers", 409,
                            "Conflict", {}, None)
        appeared.set()
        return {entry.name: entry}

    models = serve.AutoModelRegistry(
        "https://pillar.example", refresh_s=60,
        retry_delays=(0.01,), discover_fn=discover)
    try:
        models.start()
        assert appeared.wait(1.0), "409 retry did not run"
        assert _wait_for_model_names(models, ["llama32-3b"])
        assert calls[:2] == [1, 2]
    finally:
        models.close()


def test_auto_models_periodically_refresh_added_and_removed_models(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    refreshed = threading.Event()
    first = serve.ModelEntry(name="model-a", tokenizer_gguf="/a.gguf")
    second = serve.ModelEntry(name="model-b", tokenizer_gguf="/b.gguf")
    calls = 0

    def discover(_url):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {first.name: first}
        refreshed.set()
        return {second.name: second}

    models = serve.AutoModelRegistry(
        "https://pillar.example", refresh_s=0.01,
        retry_delays=(0.01,), discover_fn=discover)
    try:
        models.start()
        assert refreshed.wait(1.0), "periodic refresh did not run"
        assert _wait_for_model_names(models, ["model-b"])
    finally:
        models.close()


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


@pytest.mark.parametrize(
    "settlement",
    [None, {}, [], {"charged": 1}, {"receipt_id": "r"},
     {"receipt_id": "r", "charged": -1},
     {"receipt_id": "r", "charged": True}],
    ids=["null", "empty-dict", "list", "missing-receipt-id",
         "missing-charged", "negative-charged", "bool-charged"],
)
def test_gateway_enforce_requires_affirmative_settlement(
        monkeypatch, settlement):
    backend = serve.ChainChatBackend()
    backend._ledger_hook_cached = SimpleNamespace(
        enabled=True, wants_receipt=True, fail_closed=True,
        gate=lambda _cost: (True, ""), settle=lambda _path: settlement)
    monkeypatch.setattr(backend, "_client_env", lambda: {})
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **_kwargs: _completed_chain_with_receipt(
            cmd, text="must not escape"))
    entry = serve.ModelEntry(
        name="model-a", tokenizer_gguf="/unused.gguf",
        registry_url="https://pillar.example", num_layers=2)

    with pytest.raises(serve.CreditGateError, match="ledger settlement failed") as caught:
        backend.generate(entry, "hello", 1, {})

    assert caught.value.status == 503


def test_gateway_enforce_accepts_affirmative_settlement(monkeypatch):
    backend = serve.ChainChatBackend()
    backend._ledger_hook_cached = SimpleNamespace(
        enabled=True, wants_receipt=True, fail_closed=True,
        gate=lambda _cost: (True, ""),
        settle=lambda _path: {"receipt_id": "r", "charged": 0})
    monkeypatch.setattr(backend, "_client_env", lambda: {})
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **_kwargs: _completed_chain_with_receipt(cmd))
    entry = serve.ModelEntry(
        name="model-a", tokenizer_gguf="/unused.gguf",
        registry_url="https://pillar.example", num_layers=2)

    result = backend.generate(entry, "hello", 1, {})

    assert result.text == "settled" and result.credits_charged == 0
    assert result.prompt_eval_count == 7 and result.eval_count == 2


def test_default_mode_does_not_require_affirmative_settlement(monkeypatch):
    backend = serve.ChainChatBackend()
    backend._ledger_hook_cached = SimpleNamespace(
        enabled=True, wants_receipt=True, fail_closed=False,
        gate=lambda _cost: (True, ""), settle=lambda _path: None)
    monkeypatch.setattr(backend, "_client_env", lambda: {})
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stderr="", stdout="[chain] gen:  'legacy reply'\n"))
    entry = serve.ModelEntry(
        name="model-a", tokenizer_gguf="/unused.gguf",
        registry_url="https://pillar.example", num_layers=2)

    result = backend.generate(entry, "hello", 1, {})

    assert result.text == "legacy reply" and result.credits_charged is None


def test_gateway_chain_client_requires_worker_tls(monkeypatch):
    monkeypatch.setenv("NAKSHATRA_CHAIN_TLS_MODE", "required")
    entry = serve.ModelEntry(
        name="model-a", tokenizer_gguf="/unused.gguf",
        registry_url="https://pillar.example", num_layers=2)

    cmd = serve.ChainChatBackend()._cmd(entry, "hello", 1)

    assert cmd[cmd.index("--tls-mode") + 1] == "required"
