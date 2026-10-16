from __future__ import annotations

import json
import re
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
from release_trust import ReleaseTrustError  # noqa: E402


PIN = "17" * 32
PACKAGE_SIGNER = "23" * 32


@pytest.fixture(autouse=True)
def _signed_release_package_signer(monkeypatch):
    monkeypatch.setattr(
        nak, "load_release_package_signers", lambda: frozenset({PACKAGE_SIGNER}))
    monkeypatch.setattr(nak, "_p2p_installer", lambda *_args: None)


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
    profile = home / ".nakshatra" / "inference.env"
    owner = home / ".nakshatra" / "gateway" / "inference-profile-created"
    token = owner.read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{64}", token)
    assert f"# nak-gateway-owned: {token}\n" in profile.read_text()
    p2p_env = home / ".nakshatra" / "p2p.env"
    p2p_owner = home / ".nakshatra" / "gateway" / "p2p-profile-created"
    p2p_token = p2p_owner.read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{64}", p2p_token)
    assert 'NAKSHATRA_P2P_NAKD_INBOUND=""\n' in p2p_env.read_text()
    assert (home / ".nakshatra" / "p2p-consumer.env").read_text() == (
        'NAKSHATRA_P2P_NAKD_INBOUND=""\n'
        'NAKSHATRA_P2P_CONSUMER_ARGS="-relay-activation-fallback"\n'
        'NAKSHATRA_P2P_WORKER_ARGS=""\n'
    )
    assert "NAKSHATRA_P2P_GATEWAY_ARGS" not in p2p_env.read_text()
    assert "relay-activation-fallback" not in p2p_env.read_text()
    assert f"# nak-gateway-owned: {p2p_token}\n" in p2p_env.read_text()
    assert (home / ".nakshatra" / "p2p.lock").stat().st_mode & 0o777 == 0o600
    assert calls == ["enable"]


def test_gateway_signed_unit_argv_defaults_paid_gate_only_from_exact_env(
        tmp_path, monkeypatch):
    models = {"model": object()}
    observed = []

    class Server:
        server_address = ("127.0.0.1", 11600)

        def serve_forever(self):
            return None

        def shutdown(self):
            return None

        def server_close(self):
            return None

    def build_server(_bind, _port, _models, backend, _lifecycle):
        observed.append(backend._chain._paid_admission)
        return Server()

    monkeypatch.setattr(serve, "_load_models", lambda _path: models)
    monkeypatch.setattr(serve, "build_server", build_server)
    monkeypatch.setattr(serve, "_maybe_start_proposals_server", lambda: None)
    monkeypatch.setenv("NAKSHATRA_LEDGER_MODE", "enforce")
    monkeypatch.delenv("NAKSHATRA_GATEWAY_PAID", raising=False)
    argv = ["--models", str(tmp_path / "models.yaml")]

    assert serve.main(argv) == 0
    monkeypatch.setenv("NAKSHATRA_GATEWAY_PAID", "1")
    assert serve.main(argv) == 0
    monkeypatch.delenv("NAKSHATRA_GATEWAY_PAID")
    assert serve.main(argv) == 0

    assert observed == [False, True, False]


def test_gateway_paid_preflight_failure_leaves_env_untouched(tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_path = home / ".nakshatra" / "gateway.env"
    env_path.parent.mkdir(parents=True)
    original = (
        b"NAKSHATRA_LEDGER_MODE=enforce\n"
        b"NAKSHATRA_LEDGER_URL=https://ledger.test\n"
        + f"NAKSHATRA_LEDGER_SPKI_SHA256={'ab' * 32}\n".encode()
    )
    env_path.write_bytes(original)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        nak, "_paid_probe", lambda *_args: (_ for _ in ()).throw(SystemExit("probe failed")))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: pytest.fail("must not restart"))

    with pytest.raises(SystemExit, match="probe failed"):
        nak._gateway_paid(object(), "on")

    assert env_path.read_bytes() == original


def test_gateway_paid_failed_readiness_rolls_back_exact_env_and_restarts(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_path = home / ".nakshatra" / "gateway.env"
    env_path.parent.mkdir(parents=True)
    original = (
        b"PORT=11600\nNAKSHATRA_LEDGER_MODE=enforce\n"
        b"NAKSHATRA_LEDGER_URL=https://ledger.test\n"
        + f"NAKSHATRA_LEDGER_SPKI_SHA256={'ab' * 32}\n".encode()
        + b"ODD='leave me'\n"
    )
    env_path.write_bytes(original)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_paid_probe", lambda *_args: {"ready": True})
    monkeypatch.setattr(nak, "_wait_gateway_ready", lambda *_args: False)
    actions = []
    monkeypatch.setattr(nak, "_gateway_installer", lambda _sp, action: actions.append(action))

    with pytest.raises(SystemExit, match="did not become ready"):
        nak._gateway_paid(object(), "on")

    assert actions == ["restart", "restart"]
    assert env_path.read_bytes() == original


def test_gateway_paid_off_removes_only_gate(tmp_path, monkeypatch):
    home = tmp_path / "home"
    env_path = home / ".nakshatra" / "gateway.env"
    env_path.parent.mkdir(parents=True)
    env_path.write_bytes(b"PORT=11600\nNAKSHATRA_GATEWAY_PAID=1\nKEEP=exact\n")
    monkeypatch.setenv("HOME", str(home))
    actions = []
    monkeypatch.setattr(nak, "_gateway_installer", lambda _sp, action: actions.append(action))

    nak._gateway_paid(object(), "off")

    assert env_path.read_bytes() == b"PORT=11600\nKEEP=exact\n"
    assert actions == ["restart"]


def test_gateway_on_enables_p2p_before_gateway_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    calls = []
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: False)
    monkeypatch.setattr(nak, "_p2p_installer", lambda _sp, action: calls.append(("p2p", action)))
    monkeypatch.setattr(
        nak, "_gateway_installer", lambda _sp, action: calls.append(("gateway", action)))

    nak._gateway_on(object(), _args())

    assert calls == [("p2p", "enable"), ("gateway", "enable")]


def test_gateway_on_adds_final_consumer_override_without_rewriting_operator_p2p(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    p2p_env = home / ".nakshatra" / "p2p.env"
    p2p_env.parent.mkdir(parents=True)
    original = "P2P_RELAYS=/dns4/operator.example/tcp/443/p2p/operator\n"
    p2p_env.write_text(
        original + 'NAKSHATRA_P2P_GATEWAY_ARGS="-relay-activation-fallback"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: False)
    p2p_calls = []
    monkeypatch.setattr(nak, "_p2p_installer", lambda _sp, action: p2p_calls.append(action))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)

    nak._gateway_on(object(), _args())

    assert p2p_env.read_text() == original
    assert (home / ".nakshatra" / "p2p-consumer.env").read_text() == (
        'NAKSHATRA_P2P_NAKD_INBOUND=""\n'
        'NAKSHATRA_P2P_CONSUMER_ARGS="-relay-activation-fallback"\n'
        'NAKSHATRA_P2P_WORKER_ARGS=""\n'
    )
    assert p2p_calls == ["restart"]
    assert not (home / ".nakshatra" / "gateway" / "p2p-profile-created").exists()


def test_gateway_on_protects_consumer_when_worker_is_configured_but_disabled(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    (state / "p2p.env").write_text("P2P_RELAYS=operator\n")
    (state / "worker.env").write_text("PORT=5530\n")
    calls = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_worker_unit_enabled", lambda _sp: False)
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: False)
    monkeypatch.setattr(nak, "_p2p_installer", lambda _sp, action: calls.append(("p2p", action)))
    monkeypatch.setattr(
        nak, "_gateway_installer", lambda _sp, action: calls.append(("gateway", action)))

    nak._gateway_on(object(), _args())

    assert (state / "p2p-consumer.env").read_text() == (
        'NAKSHATRA_P2P_NAKD_INBOUND=""\n'
        'NAKSHATRA_P2P_CONSUMER_ARGS="-relay-activation-fallback"\n'
        'NAKSHATRA_P2P_WORKER_ARGS=""\n'
    )
    assert calls == [("p2p", "restart"), ("gateway", "enable")]


def test_gateway_off_removes_final_consumer_override_under_p2p_ownership(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    (state / "p2p.env").write_text("P2P_RELAYS=operator\n")
    (state / "p2p-consumer.env").write_text(
        'NAKSHATRA_P2P_NAKD_INBOUND=""\nNAKSHATRA_P2P_WORKER_ARGS=""\n')
    calls = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)
    monkeypatch.setattr(nak, "_p2p_installer", lambda _sp, action: calls.append(action))

    nak._gateway_off(object())

    assert not (state / "p2p-consumer.env").exists()
    assert calls == ["restart"]


def test_explicit_p2p_on_retires_gateway_ownership(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    gateway = state / "gateway"
    gateway.mkdir(parents=True)
    token = "ab" * 32
    p2p_env = state / "p2p.env"
    body = f"P2P_RELAYS=test\n# nak-gateway-owned: {token}\n"
    p2p_env.write_text(body)
    owner = gateway / "p2p-profile-created"
    owner.write_text(f"{token}\n")
    calls = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_p2p_installer", lambda _sp, action: calls.append(action))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)

    nak._enable_p2p_service(object())
    nak._gateway_off(object())

    assert calls == ["enable"]
    assert p2p_env.read_text() == body
    assert not owner.exists()


def test_gateway_auto_requires_package_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: pytest.fail("must not enable"))

    with pytest.raises(SystemExit, match="package-url"):
        nak._gateway_on(object(), _args(package_url=None, package_pubkey=None))


def test_gateway_enable_failure_rolls_back_created_profile_ownership(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: False)

    def fail_enable(_sp, action):
        if action == "enable":
            raise SystemExit("enable failed")

    monkeypatch.setattr(nak, "_gateway_installer", fail_enable)
    with pytest.raises(SystemExit, match="enable failed"):
        nak._gateway_on(object(), _args())

    assert not (home / ".nakshatra" / "gateway.env").exists()
    assert not (home / ".nakshatra" / "inference.env").exists()
    assert not (home / ".nakshatra" / "gateway" / "inference-profile-created").exists()
    assert not (home / ".nakshatra" / "p2p.env").exists()
    assert not (home / ".nakshatra" / "gateway" / "p2p-profile-created").exists()


def test_gateway_enable_failed_stop_retains_opt_in_and_consumer_protection(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: False)
    actions = []

    def gateway_installer(_sp, action):
        actions.append(action)
        raise SystemExit(f"gateway {action} failed")

    monkeypatch.setattr(nak, "_gateway_installer", gateway_installer)
    with pytest.raises(SystemExit, match=r"enable failed; rollback could not disable"):
        nak._gateway_on(object(), _args())

    state = home / ".nakshatra"
    assert actions == ["enable", "disable"]
    assert (state / "gateway.env").exists()
    assert (state / "inference.env").exists()
    assert (state / "p2p.env").exists()
    assert (state / "p2p-consumer.env").exists()


def test_gateway_restart_rollback_keeps_consumer_override_throughout(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    gateway_env = state / "gateway.env"
    gateway_env.write_text("PORT=11599\nOLD=1\n")
    (state / "p2p.env").write_text("P2P_RELAYS=operator\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: True)
    monkeypatch.setattr(nak, "_worker_unit_enabled", lambda _sp: False)
    monkeypatch.setattr(nak, "_p2p_installer", lambda *_args: None)
    actions = []

    def gateway_installer(_sp, action):
        actions.append(action)
        assert (state / "p2p-consumer.env").exists()
        if len(actions) == 1:
            raise SystemExit("new restart failed")

    monkeypatch.setattr(nak, "_gateway_installer", gateway_installer)
    with pytest.raises(SystemExit, match="new restart failed"):
        nak._gateway_on(object(), _args(port=11601))

    assert actions == ["restart", "restart"]
    assert gateway_env.read_text() == "PORT=11599\nOLD=1\n"
    assert (state / "p2p-consumer.env").exists()


def test_gateway_enable_failed_stop_promotes_retained_p2p_to_legacy_operator(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_unit_enabled", lambda _sp: False)
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)
    actions = []

    def fail_enable_and_stop(_sp, action):
        actions.append(action)
        raise SystemExit(f"{action} failed")

    monkeypatch.setattr(nak, "_p2p_installer", fail_enable_and_stop)
    with pytest.raises(SystemExit):
        nak._gateway_on(object(), _args())

    p2p_env = home / ".nakshatra" / "p2p.env"
    owner = home / ".nakshatra" / "gateway" / "p2p-profile-created"
    assert p2p_env.exists() and not owner.exists()
    assert "NAKSHATRA_P2P_NAKD_INBOUND" not in p2p_env.read_text()
    assert actions == ["enable", "disable", "disable", "restart"]


def test_gateway_off_failed_owned_p2p_stop_promotes_and_verifies_legacy_operator(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    gateway = state / "gateway"
    gateway.mkdir(parents=True)
    token = "cd" * 32
    (state / "gateway.env").write_text("PORT=11600\n")
    p2p_env = state / "p2p.env"
    p2p_env.write_text(
        f'P2P_RELAYS=test\nNAKSHATRA_P2P_NAKD_INBOUND=""\n'
        f'# nak-gateway-owned: {token}\n')
    (state / "p2p-consumer.env").write_text(
        'NAKSHATRA_P2P_NAKD_INBOUND=""\nNAKSHATRA_P2P_WORKER_ARGS=""\n')
    owner = gateway / "p2p-profile-created"
    owner.write_text(token + "\n")
    monkeypatch.setenv("HOME", str(home))
    actions = []

    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)
    monkeypatch.setattr(nak, "_worker_unit_enabled", lambda _sp: False)

    def p2p_installer(_sp, action):
        actions.append(action)
        if action == "disable":
            raise SystemExit("injected p2p disable failure")

    monkeypatch.setattr(nak, "_p2p_installer", p2p_installer)

    with pytest.raises(SystemExit, match="promoted to operator-owned legacy"):
        nak._gateway_off(object())

    assert actions == ["disable", "restart"]
    assert p2p_env.exists() and not owner.exists()
    assert not (state / "p2p-consumer.env").exists()
    assert "NAKSHATRA_P2P_NAKD_INBOUND" not in p2p_env.read_text()


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


def test_gateway_container_environment_cannot_bypass_allow_lan(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NAK_CONTAINER", "1")
    monkeypatch.setenv("NAK_GATEWAY_BIND", "0.0.0.0")
    monkeypatch.setenv("NAK_GATEWAY_PORT", "11599")

    with pytest.raises(SystemExit, match="non-loopback"):
        nak._gateway_on(object(), _args(bind="0.0.0.0", port=11599))


def test_gateway_cli_defaults_ignore_container_environment(monkeypatch):
    seen = []
    monkeypatch.setenv("NAK_CONTAINER", "1")
    monkeypatch.setenv("NAK_GATEWAY_BIND", "0.0.0.0")
    monkeypatch.setenv("NAK_GATEWAY_PORT", "11599")
    monkeypatch.setattr(nak, "_gateway_on", lambda _sp, args: seen.append((args.bind, args.port)))
    required = ["--pillar-url", "https://pillar.example", "--pillar-spki", PIN,
                "--ledger-url", "https://pillar.example/ledger"]
    assert nak.main(["gateway", "on", *required]) == 0
    assert seen.pop() == ("127.0.0.1", 11600)


@pytest.mark.parametrize("option", ["--bi", "--po", "--allow"])
def test_gateway_cli_rejects_long_option_abbreviations(monkeypatch, option):
    required = ["--pillar-url", "https://pillar.example", "--pillar-spki", PIN,
                "--ledger-url", "https://pillar.example/ledger"]
    monkeypatch.setattr(
        nak, "_gateway_on", lambda *_args: pytest.fail("abbreviation reached gateway mutation"))
    abbreviated = [option] if option == "--allow" else [option, "11600"]
    with pytest.raises(SystemExit) as exc:
        nak.main(["gateway", "on", *required, *abbreviated])
    assert exc.value.code == 2


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
    profile = home / ".nakshatra" / "inference.env"
    token = "ab" * 32
    profile.write_text(f"# gateway created\n# nak-gateway-owned: {token}\n")
    owner = home / ".nakshatra" / "gateway" / "inference-profile-created"
    owner.parent.mkdir()
    owner.write_text(f"{token}\n")
    calls = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda _sp, action: calls.append(action))

    nak._gateway_off(object())

    assert calls == ["disable"]
    assert not env.exists()
    assert not profile.exists()
    assert not owner.exists()


def test_gateway_off_disables_only_gateway_owned_p2p(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    gateway = state / "gateway"
    gateway.mkdir(parents=True)
    token = "cd" * 32
    p2p_env = state / "p2p.env"
    p2p_env.write_text(f"P2P_RELAYS=test\n# nak-gateway-owned: {token}\n")
    owner = gateway / "p2p-profile-created"
    owner.write_text(f"{token}\n")
    calls = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda _sp, action: calls.append(("gateway", action)))
    monkeypatch.setattr(nak, "_p2p_installer", lambda _sp, action: calls.append(("p2p", action)))

    nak._gateway_off(object())

    assert calls == [("gateway", "disable"), ("p2p", "disable")]
    assert not p2p_env.exists()
    assert not owner.exists()


def test_gateway_off_preserves_gateway_owned_p2p_when_worker_is_configured(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    gateway = state / "gateway"
    gateway.mkdir(parents=True)
    token = "cd" * 32
    p2p_env = state / "p2p.env"
    body = f"P2P_RELAYS=test\n# nak-gateway-owned: {token}\n"
    p2p_env.write_text(body)
    (state / "worker.env").write_text("PORT=50051\n")
    owner = gateway / "p2p-profile-created"
    owner.write_text(f"{token}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_worker_unit_enabled", lambda _sp: True)
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)
    monkeypatch.setattr(
        nak, "_p2p_installer", lambda *_args: pytest.fail("worker still owns p2p"))

    nak._gateway_off(object())

    assert p2p_env.read_text() == body
    assert not owner.exists()


def test_gateway_off_rechecks_worker_ownership_between_check_and_disable(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    gateway = state / "gateway"
    gateway.mkdir(parents=True)
    token = "cd" * 32
    p2p_env = state / "p2p.env"
    body = f"P2P_RELAYS=test\n# nak-gateway-owned: {token}\n"
    p2p_env.write_text(body)
    owner = gateway / "p2p-profile-created"
    owner.write_text(f"{token}\n")
    calls = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        nak, "_gateway_installer", lambda _sp, action: calls.append(("gateway", action)))
    monkeypatch.setattr(
        nak, "_p2p_installer", lambda _sp, action: calls.append(("p2p", action)))

    worker_enabled = False

    def worker_takes_ownership():
        nonlocal worker_enabled
        (state / "worker.env").write_text("PORT=50051\n")
        worker_enabled = True

    monkeypatch.setattr(nak, "_gateway_p2p_before_disable", worker_takes_ownership)
    monkeypatch.setattr(nak, "_worker_unit_enabled", lambda _sp: worker_enabled)
    nak._gateway_off(object())

    assert calls == [("gateway", "disable")]
    assert p2p_env.read_text() == body
    assert (state / "worker.env").is_file()
    assert not owner.exists()


def test_gateway_off_preserves_operator_p2p_when_marker_does_not_match(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    gateway = state / "gateway"
    gateway.mkdir(parents=True)
    p2p_env = state / "p2p.env"
    body = f"P2P_RELAYS=test\n# nak-gateway-owned: {'ab' * 32}\n"
    p2p_env.write_text(body)
    owner = gateway / "p2p-profile-created"
    owner.write_text("cd" * 32 + "\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)
    monkeypatch.setattr(
        nak, "_p2p_installer", lambda *_args: pytest.fail("mismatched p2p must be preserved"))

    nak._gateway_off(object())

    assert p2p_env.read_text() == body
    assert not owner.exists()


def test_gateway_off_preserves_preexisting_or_still_needed_inference_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    state.mkdir(parents=True)
    profile = state / "inference.env"
    profile.write_text("EXISTING=operator-owned\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)

    nak._gateway_off(object())
    assert profile.read_text() == "EXISTING=operator-owned\n"

    owner = state / "gateway" / "inference-profile-created"
    owner.parent.mkdir(exist_ok=True)
    token = "ab" * 32
    profile.write_text(f"# nak-gateway-owned: {token}\n")
    owner.write_text(f"{token}\n")
    (state / "unconscious.release").write_text("enabled\n")
    nak._gateway_off(object())
    assert profile.exists() and not owner.exists()


@pytest.mark.parametrize("replacement", [
    "EXISTING=operator-owned\n",
    "# nak-gateway-owned: " + "cd" * 32 + "\n",
])
def test_gateway_off_preserves_operator_replaced_profile(tmp_path, monkeypatch, replacement):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    gateway = state / "gateway"
    gateway.mkdir(parents=True)
    profile = state / "inference.env"
    profile.write_text(replacement)
    owner = gateway / "inference-profile-created"
    owner.write_text("ab" * 32 + "\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)

    nak._gateway_off(object())

    assert profile.read_text() == replacement
    assert not owner.exists()


def test_gateway_off_preserves_profile_with_stale_marker(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    gateway = state / "gateway"
    gateway.mkdir(parents=True)
    profile = state / "inference.env"
    profile.write_text("EXISTING=operator-owned\n")
    owner = gateway / "inference-profile-created"
    owner.write_text("legacy unbound marker\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)

    nak._gateway_off(object())

    assert profile.read_text() == "EXISTING=operator-owned\n"
    assert not owner.exists()


def test_gateway_off_worker_only_falls_back_to_worker_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state = home / ".nakshatra"
    gateway = state / "gateway"
    gateway.mkdir(parents=True)
    token = "ab" * 32
    (state / "inference.env").write_text(f"# nak-gateway-owned: {token}\n")
    (state / "worker.env").write_text("PORT=50051\n")
    (gateway / "inference-profile-created").write_text(f"{token}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nak, "_gateway_installer", lambda *_args: None)

    nak._gateway_off(object())

    assert not (state / "inference.env").exists()
    assert (state / "worker.env").exists()
    spec = json.loads((ROOT / "release" / "spec.json").read_text())
    assert spec["profiles"]["worker"]["satisfied_by"] == ["inference"]


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


def test_auto_models_retry_transient_release_trust_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    appeared = threading.Event()
    calls = 0
    entry = serve.ModelEntry(name="llama32-3b", tokenizer_gguf="/fake.gguf",
                             registry_url="https://pillar.example", num_layers=28)

    def discover(_url):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ReleaseTrustError("signed release transition is briefly unavailable")
        appeared.set()
        return {entry.name: entry}

    models = serve.AutoModelRegistry(
        "https://pillar.example", refresh_s=60,
        retry_delays=(0.01,), discover_fn=discover)
    try:
        models.start()
        assert appeared.wait(1.0), "release-trust retry did not run"
        assert _wait_for_model_names(models, ["llama32-3b"])
        assert calls == 2
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
