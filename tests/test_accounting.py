"""U5: one accounting front door — metering keeps the gateway's safety contract; escrow is chosen once."""
import sys
from pathlib import Path
from urllib.error import URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import accounting  # noqa: E402
import ledger_client  # noqa: E402
import nakshatra_auth  # noqa: E402


def test_metering_is_the_ledger_hook_and_off_by_default(monkeypatch):
    monkeypatch.delenv("NAKSHATRA_CREDITS", raising=False)
    m = accounting.metering()
    assert type(m).__name__ == "LedgerHook" and m.enabled is False
    assert m.gate(10**9) == (True, "") and m.settle("/nonexistent") is None


def test_metering_fails_open_when_the_backend_is_down(monkeypatch, tmp_path):
    key_path = tmp_path / "home" / ".nakshatra" / "keys" / "worker.ed25519"
    key_path.parent.mkdir(parents=True)
    private_key, _ = nakshatra_auth.generate_keypair()
    key_path.write_bytes(private_key)
    key_path.chmod(0o600)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(nakshatra_auth, "WORKER_KEY_PATH", key_path)
    monkeypatch.setenv("NAKSHATRA_CREDITS", "1")
    monkeypatch.setattr(
        ledger_client.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(URLError("ledger down")),
    )
    allow, why = accounting.metering().gate(5)
    assert allow and "fail-open" in why


def test_escrow_backend_is_chosen_once_and_unknown_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("NAKSHATRA_ESCROW", raising=False)
    assert type(accounting.escrow(tmp_path)).__name__ == "LedgerAdapter"
    monkeypatch.setenv("NAKSHATRA_ESCROW", "solana-mainnet")
    with pytest.raises(ValueError, match="not a known escrow backend"):
        accounting.escrow(tmp_path)


def test_callers_go_through_the_front_door():
    root = Path(__file__).resolve().parent.parent / "scripts"
    assert "from accounting import metering" in (root / "nakshatra_serve.py").read_text()
    nakd = (root / "network" / "nakd.py").read_text()
    assert "_accounting.escrow(" in nakd and "LedgerAdapter(" not in nakd
