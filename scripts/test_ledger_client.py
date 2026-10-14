"""Tests for the signed, fail-open serve-side Neuron ledger hook."""
from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import nakshatra_auth as auth  # noqa: E402
import ledger_client  # noqa: E402
from ledger_client import LedgerHook  # noqa: E402


class _Response:
    status = 200

    def __init__(self, body=b'{"ok":true,"balance":500}'):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _install_key(monkeypatch, tmp_path):
    home = tmp_path / "home"
    key_path = home / ".nakshatra" / "keys" / "worker.ed25519"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(auth, "WORKER_KEY_PATH", key_path)
    if key_path.exists():
        _private_key, public_key_hex = auth.load_worker_key()
        return key_path, public_key_hex
    key_path.parent.mkdir(parents=True)
    private_key, public_key_hex = auth.generate_keypair()
    key_path.write_bytes(private_key)
    key_path.chmod(0o600)
    return key_path, public_key_hex


def _missing_key(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    key_path = home / ".nakshatra" / "keys" / "worker.ed25519"
    monkeypatch.setattr(auth, "WORKER_KEY_PATH", key_path)
    return key_path


def _hook(monkeypatch, tmp_path, *, install_key=True, **env):
    for key in [
        key for key in os.environ
        if key.startswith(("NAKSHATRA_CREDITS", "NAKSHATRA_LEDGER"))
    ]:
        monkeypatch.delenv(key, raising=False)
    if install_key:
        _install_key(monkeypatch, tmp_path)
    else:
        _missing_key(monkeypatch, tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return LedgerHook()


def _http_error(request, status, body=None):
    payload = json.dumps(body or {}).encode()
    return urllib.error.HTTPError(
        request.full_url, status, "fake ledger response", {}, io.BytesIO(payload)
    )


def _auth_parts(request):
    header = request.get_header("Authorization")
    assert header.startswith(auth.AUTH_SCHEME + " ")
    return dict(re.findall(r'(keyid|sig|ts)="([^"]+)"', header))


def _assert_signed(request, public_key_hex, path, body):
    parts = _auth_parts(request)
    assert auth.verify_request(
        public_key_hex,
        request.get_method(),
        path,
        body,
        int(parts["ts"]),
        parts["sig"],
    )
    return parts


def test_disabled_is_total_noop(monkeypatch, tmp_path):
    h = _hook(monkeypatch, tmp_path)
    assert h.enabled is False
    assert h.url == "http://127.0.0.1:8097"
    assert h.wants_receipt is False
    assert h.gate(10_000) == (True, "")
    assert h.settle("/does/not/exist") is None


def test_enabled_allows_when_ledger_ok(monkeypatch, tmp_path):
    h = _hook(monkeypatch, tmp_path, NAKSHATRA_CREDITS="1")
    monkeypatch.setattr(h, "_get", lambda path: {"ok": True, "balance": 500})
    assert h.wants_receipt is True
    assert h.gate(100) == (True, "")


def test_enforce_denies_legacy_200_when_out_of_credits(monkeypatch, tmp_path):
    h = _hook(
        monkeypatch, tmp_path,
        NAKSHATRA_CREDITS="1", NAKSHATRA_CREDITS_ENFORCE="1",
    )
    monkeypatch.setattr(h, "_get", lambda path: {"ok": False, "balance": -50})
    allow, reason = h.gate(100)
    assert allow is False and "insufficient" in reason


def test_advisory_for_legacy_200_when_not_enforced(monkeypatch, tmp_path):
    h = _hook(monkeypatch, tmp_path, NAKSHATRA_CREDITS="1")
    monkeypatch.setattr(h, "_get", lambda path: {"ok": False, "balance": -50})
    allow, reason = h.gate(100)
    assert allow is True and "advisory" in reason


def test_fail_open_on_network_error_even_when_enforced(monkeypatch, tmp_path):
    h = _hook(
        monkeypatch, tmp_path,
        NAKSHATRA_CREDITS="1", NAKSHATRA_CREDITS_ENFORCE="1",
    )

    def boom(_path):
        raise ConnectionError("ledger down")

    monkeypatch.setattr(h, "_get", boom)
    allow, reason = h.gate(100)
    assert allow is True and "fail-open" in reason


def test_signed_post_verifies_against_exact_wire_body(monkeypatch, tmp_path):
    _key_path, public_key_hex = _install_key(monkeypatch, tmp_path)
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return _Response(b'{"delta":{"worker":3}}')

    monkeypatch.setattr(ledger_client.urllib.request, "urlopen", fake_urlopen)
    h = _hook(
        monkeypatch, tmp_path,
        NAKSHATRA_CREDITS="1", NAKSHATRA_LEDGER_KEYID="roster-node",
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"run_id": "r1", "n_generated": 2}))

    assert h.settle(str(receipt_path)) == {"delta": {"worker": 3}}
    assert len(requests) == 1
    request = requests[0]
    parts = _assert_signed(request, public_key_hex, "/settle", request.data)
    assert parts["keyid"] == "roster-node"
    body = json.loads(request.data)
    assert body["receipt"]["run_id"] == "r1"
    assert body["requester"] == "self"


@pytest.mark.parametrize("path", ["/balance", "/history?limit=2"])
def test_get_calls_are_signed_with_empty_body(monkeypatch, tmp_path, path):
    _key_path, public_key_hex = _install_key(monkeypatch, tmp_path)
    requests = []
    monkeypatch.setattr(
        ledger_client.urllib.request,
        "urlopen",
        lambda request, timeout: requests.append(request) or _Response(),
    )
    h = _hook(monkeypatch, tmp_path, NAKSHATRA_CREDITS="1")

    h._get(path)

    assert len(requests) == 1
    assert requests[0].data is None
    parts = _assert_signed(requests[0], public_key_hex, path, b"")
    assert parts["keyid"] == public_key_hex


def test_gate_can_consume_get_is_signed(monkeypatch, tmp_path):
    _key_path, public_key_hex = _install_key(monkeypatch, tmp_path)
    requests = []
    monkeypatch.setattr(
        ledger_client.urllib.request,
        "urlopen",
        lambda request, timeout: requests.append(request) or _Response(),
    )
    h = _hook(monkeypatch, tmp_path, NAKSHATRA_CREDITS="1")

    assert h.gate(7) == (True, "")
    path = "/can_consume?account=self&cost=7"
    _assert_signed(requests[0], public_key_hex, path, b"")


def test_no_key_makes_no_call_fails_open_and_is_never_created(
    monkeypatch, tmp_path, caplog,
):
    calls = []
    monkeypatch.setattr(
        ledger_client.urllib.request,
        "urlopen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    key_path = _missing_key(monkeypatch, tmp_path)
    with caplog.at_level(logging.WARNING, logger="ledger_client"):
        h = _hook(monkeypatch, tmp_path, install_key=False, NAKSHATRA_CREDITS="1")
        assert h.gate(5)[0] is True
        assert h.settle(str(tmp_path / "missing-receipt.json")) is None

    assert calls == []
    assert not key_path.exists()
    assert caplog.text.count("ledger signing key unavailable") == 1


def test_401_fails_open_and_logs_keyid_at_most_once_per_minute(
    monkeypatch, tmp_path, caplog,
):
    clock = iter((10.0, 20.0, 71.0))
    monkeypatch.setattr(ledger_client.time, "monotonic", lambda: next(clock))

    def refuse(request, timeout):
        raise _http_error(request, 401)

    monkeypatch.setattr(ledger_client.urllib.request, "urlopen", refuse)
    h = _hook(
        monkeypatch, tmp_path,
        NAKSHATRA_CREDITS="1", NAKSHATRA_LEDGER_KEYID="node-alias",
    )
    with caplog.at_level(logging.WARNING, logger="ledger_client"):
        assert h.gate(1)[0] is True
        assert h.gate(1)[0] is True
        assert h.gate(1)[0] is True

    assert caplog.text.count("ledger refused signature (keyid node-alias)") == 2


def test_402_denies_gate(monkeypatch, tmp_path):
    def over_limit(request, timeout):
        raise _http_error(request, 402, {"balance": -4})

    monkeypatch.setattr(ledger_client.urllib.request, "urlopen", over_limit)
    h = _hook(monkeypatch, tmp_path, NAKSHATRA_CREDITS="1")

    allow, reason = h.gate(5)
    assert allow is False
    assert "balance -4" in reason


@pytest.mark.parametrize(
    ("status", "message"),
    [(409, "disputed receipt"), (422, "forged receipt")],
)
def test_rejected_receipt_is_logged_and_never_retried(
    monkeypatch, tmp_path, caplog, status, message,
):
    calls = []

    def reject(request, timeout):
        calls.append(request)
        raise _http_error(request, status)

    monkeypatch.setattr(ledger_client.urllib.request, "urlopen", reject)
    h = _hook(monkeypatch, tmp_path, NAKSHATRA_CREDITS="1")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"run_id": "rejected"}))

    with caplog.at_level(logging.WARNING, logger="ledger_client"):
        assert h.settle(str(receipt_path)) is None
        assert h.settle(str(receipt_path)) is None

    assert len(calls) == 1
    assert message in caplog.text


def test_settle_network_error_fails_open(monkeypatch, tmp_path):
    h = _hook(monkeypatch, tmp_path, NAKSHATRA_CREDITS="1")
    monkeypatch.setattr(
        h, "_post", lambda path, obj: (_ for _ in ()).throw(ConnectionError())
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}")
    assert h.settle(str(receipt_path)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
