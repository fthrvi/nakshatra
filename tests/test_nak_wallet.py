from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from network import nak  # noqa: E402


class _FakeWallet:
    def __init__(self):
        self.calls = []

    def wallet(self):
        self.calls.append(("address", None))
        return {"status": "pending", "deposit_address": None}

    def wallet_status(self):
        self.calls.append(("status", None))
        return {"balance": 200_000_000, "withdrawable": 100_000_000}

    def balance(self):
        self.calls.append(("balance", None))
        return {"balance": 200_000_000, "tier": "known"}

    def bind(self, address):
        self.calls.append(("bind", address))
        return {"address": address, "changed": False}

    def withdraw(self, amount):
        self.calls.append(("withdraw", amount))
        return {"id": "withdrawal-1", "amount": amount, "state": "pending"}


def test_nak_wallet_commands_use_signed_client_and_convert_usdc(monkeypatch, capsys):
    hook = _FakeWallet()
    monkeypatch.setattr(nak, "_wallet_hook", lambda: hook)
    address = "1" * 32

    assert nak.main(["wallet", "address"]) == 0
    assert capsys.readouterr().out.strip() == "pending: deposit address creation queued"

    assert nak.main(["wallet", "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["withdrawable"] == 100_000_000
    assert status["balance_display"] == "2 USDC (200000000 base)"

    assert nak.main(["balance"]) == 0
    balance = json.loads(capsys.readouterr().out)
    assert balance["balance"] == 200_000_000
    assert balance["balance_usdc"] == "2 USDC (200000000 base)"

    assert nak.main(["wallet", "bind", address]) == 0
    assert json.loads(capsys.readouterr().out)["address"] == address

    assert nak.main(["wallet", "withdraw", "1.250001"]) == 0
    assert json.loads(capsys.readouterr().out)["amount"] == 125_000_100
    assert hook.calls == [
        ("address", None), ("status", None), ("balance", None), ("bind", address),
        ("withdraw", 125_000_100),
    ]


@pytest.mark.parametrize("amount", ["0", "-1", "nan", "1.0000001", "not-money"])
def test_nak_wallet_withdraw_rejects_invalid_usdc_before_call(
    amount, monkeypatch
):
    hook = _FakeWallet()
    monkeypatch.setattr(nak, "_wallet_hook", lambda: hook)
    with pytest.raises(SystemExit, match="withdraw amount"):
        nak.main(["wallet", "withdraw", amount])
    assert hook.calls == []


def test_wallet_hook_uses_gateway_ledger_configuration(tmp_path, monkeypatch):
    home = tmp_path / "home"
    env = home / ".nakshatra" / "gateway.env"
    env.parent.mkdir(parents=True)
    env.write_text(
        'NAKSHATRA_LEDGER_URL="https://front.example/ledger"\n'
        'NAKSHATRA_LEDGER_SPKI_SHA256="' + "ab" * 32 + '"\n'
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NAKSHATRA_LEDGER_URL", raising=False)
    monkeypatch.delenv("NAKSHATRA_LEDGER_SPKI_SHA256", raising=False)
    observed = {}

    import accounting

    def fake_wallet(**kwargs):
        observed.update(kwargs)
        return _FakeWallet()

    monkeypatch.setattr(accounting, "wallet", fake_wallet)
    assert isinstance(nak._wallet_hook(), _FakeWallet)
    assert observed == {
        "url": "https://front.example/ledger",
        "spki_hash": "ab" * 32,
    }
