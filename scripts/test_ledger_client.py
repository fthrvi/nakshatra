"""
Tests for the serve-side credit hook (ledger_client.LedgerHook). The safety contract that
guards Prithvi's live brain: DEFAULT-OFF is a no-op, ledger errors FAIL OPEN (never block),
and denial happens ONLY when enabled AND enforced AND the ledger says no.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ledger_client import LedgerHook  # noqa: E402


def _hook(monkeypatch, **env):
    for k in [k for k in os.environ if k.startswith(("NAKSHATRA_CREDITS", "NAKSHATRA_LEDGER"))]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return LedgerHook()


def test_disabled_is_total_noop(monkeypatch):
    h = _hook(monkeypatch)                      # NAKSHATRA_CREDITS unset
    assert h.enabled is False
    assert h.wants_receipt is False
    assert h.gate(10_000) == (True, "")         # never gates
    assert h.settle("/does/not/exist") is None  # never settles


def test_enabled_allows_when_ledger_ok(monkeypatch):
    h = _hook(monkeypatch, NAKSHATRA_CREDITS="1")
    monkeypatch.setattr(h, "_get", lambda p: {"ok": True, "balance": 500})
    assert h.wants_receipt is True
    assert h.gate(100) == (True, "")


def test_enforce_denies_when_out_of_credits(monkeypatch):
    h = _hook(monkeypatch, NAKSHATRA_CREDITS="1", NAKSHATRA_CREDITS_ENFORCE="1")
    monkeypatch.setattr(h, "_get", lambda p: {"ok": False, "balance": -50})
    allow, reason = h.gate(100)
    assert allow is False and "insufficient" in reason


def test_advisory_when_enabled_but_not_enforced(monkeypatch):
    h = _hook(monkeypatch, NAKSHATRA_CREDITS="1")   # enabled, enforce off
    monkeypatch.setattr(h, "_get", lambda p: {"ok": False, "balance": -50})
    allow, reason = h.gate(100)
    assert allow is True and "advisory" in reason   # observes, doesn't block


def test_fail_open_on_ledger_error_even_when_enforced(monkeypatch):
    h = _hook(monkeypatch, NAKSHATRA_CREDITS="1", NAKSHATRA_CREDITS_ENFORCE="1")
    def boom(p):
        raise ConnectionError("ledger down")
    monkeypatch.setattr(h, "_get", boom)
    allow, reason = h.gate(100)
    assert allow is True and "fail-open" in reason   # the crux: NEVER block on error


def test_settle_posts_the_receipt(monkeypatch, tmp_path):
    h = _hook(monkeypatch, NAKSHATRA_CREDITS="1", NAKSHATRA_LEDGER_REQUESTER="opX")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"run_id": "r1", "n_generated": 2}))
    sent = {}
    monkeypatch.setattr(h, "_post", lambda path, obj: sent.update(path=path, **obj) or {"delta": {}})
    h.settle(str(rp))
    assert sent["path"] == "/settle"
    assert sent["receipt"]["run_id"] == "r1" and sent["requester"] == "opX"


def test_settle_fail_open(monkeypatch, tmp_path):
    h = _hook(monkeypatch, NAKSHATRA_CREDITS="1")
    monkeypatch.setattr(h, "_post", lambda path, obj: (_ for _ in ()).throw(ConnectionError()))
    rp = tmp_path / "r.json"
    rp.write_text("{}")
    assert h.settle(str(rp)) is None             # error swallowed; reply unaffected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
