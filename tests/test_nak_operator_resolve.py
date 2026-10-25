from __future__ import annotations

import json
import re
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import nakshatra_auth as auth  # noqa: E402
from network import nak  # noqa: E402
from network.nak_mcp import Server  # noqa: E402


_AUTH = re.compile(
    r'^Sthambha-Ed25519 keyid="([0-9a-f]{64})",sig="([A-Za-z0-9+/=]+)",ts="?([0-9]+)"?$'
)


class _UnusedClient:
    pass


class _FakeLedger:
    def __init__(self, home: Path):
        private, self.keyid = auth.generate_keypair()
        key = home / ".nakshatra" / "keys" / "worker.ed25519"
        key.parent.mkdir(parents=True)
        key.write_bytes(private)
        key.chmod(0o600)
        self.hold = {
            "v": 1,
            "hold_id": "hold-disputed-1",
            "task_hash": "a" * 64,
            "payer_account": "payer-account-0123456789",
            "worker_keyid": "b" * 64,
            "amount": 125_000_000,
            "deadline": 2_000_000_000,
            "state": "disputed",
            "server_time": 1_900_000_000,
            "submitted_at": 1_800_000_000,
            "result_hash": "c" * 64,
            "received_at": 1_800_000_001,
            "received_hash": "c" * 64,
            "refund_after_submit": False,
            "dispute_reason": "quality",
            "locked": True,
        }
        self.requests: list[dict] = []
        self.post_error: int | None = None
        self.post_error_terminal_state: str | None = None
        self.error_detail = "operator refusal"
        self.replayed_reply = False

        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _reply(self, status: int, value: dict):
                raw = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _record(self, body: bytes):
                target = self.path
                match = _AUTH.fullmatch(self.headers.get("Authorization", ""))
                verified = bool(match and auth.verify_request(
                    match.group(1), self.command, target, body,
                    int(match.group(3)), match.group(2)))
                fixture.requests.append({
                    "method": self.command, "path": target, "body": body,
                    "verified": verified,
                })
                if not verified or match.group(1) != fixture.keyid:
                    self._reply(401, {"error": "bad signature"})
                    return False
                return True

            def do_GET(self):  # noqa: N802
                if not self._record(b""):
                    return
                parsed = urllib.parse.urlsplit(self.path)
                hold_id = urllib.parse.parse_qs(parsed.query).get("hold_id", [""])[0]
                if parsed.path != "/spend/task/status" or hold_id != fixture.hold["hold_id"]:
                    self._reply(404, {"error": "spend hold not found"})
                    return
                self._reply(200, fixture.hold)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                if not self._record(body):
                    return
                if self.path != "/spend/task/resolve":
                    self._reply(404, {"error": "not found"})
                    return
                if fixture.post_error is not None:
                    if fixture.post_error_terminal_state is not None:
                        fixture.hold["state"] = fixture.post_error_terminal_state
                    self._reply(fixture.post_error, {"error": fixture.error_detail})
                    return
                request = json.loads(body)
                fixture.hold["state"] = "paid" if request["outcome"] == "pay" else "refunded"
                result = {key: fixture.hold[key] for key in (
                    "v", "hold_id", "task_hash", "worker_keyid", "amount", "deadline", "state")}
                result["replayed"] = fixture.replayed_reply
                self._reply(200, result)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    fake = _FakeLedger(home)
    try:
        yield fake
    finally:
        fake.close()


def _tty_yes(monkeypatch, answer="yes", prompts=None):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input", lambda prompt: (prompts.append(prompt) if prompts is not None else None) or answer)


def test_operator_resolve_refuses_non_tty_before_ledger(ledger, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="interactive-only"):
        nak.main(["operator", "resolve", ledger.hold["hold_id"], "pay",
                  "--reason", "evidence checked", "--ledger", ledger.url])
    assert ledger.requests == []


def test_operator_resolve_no_confirmation_sends_no_resolution(ledger, monkeypatch, capsys):
    prompts = []
    _tty_yes(monkeypatch, answer="n", prompts=prompts)
    assert nak.main(["operator", "resolve", ledger.hold["hold_id"], "refund",
                     "--reason", "output invalid", "--ledger", ledger.url]) == 1
    assert [row["method"] for row in ledger.requests] == ["GET"]
    assert "1.25 USDC (125000000 base)" in prompts[0]
    assert "payer payer-account-01…" in prompts[0]
    assert "worker bbbbbbbbbbbbbbbb" in prompts[0]
    assert "outcome REFUND" in prompts[0]
    assert "unchanged" in capsys.readouterr().out


def test_secret_reason_is_refused_before_signing_or_http(ledger, monkeypatch):
    _tty_yes(monkeypatch)
    monkeypatch.setattr(
        auth, "build_signed_envelope", lambda *_args, **_kwargs: pytest.fail("must not sign"))
    with pytest.raises(SystemExit, match="secret or encoded blob"):
        nak.main(["operator", "resolve", ledger.hold["hold_id"], "pay",
                  "--reason", "leaked sk-proj-" + "A" * 24, "--ledger", ledger.url])
    assert ledger.requests == []


def test_signed_status_and_resolution_headers_verify(ledger, monkeypatch, capsys):
    _tty_yes(monkeypatch)
    assert nak.main(["operator", "resolve", ledger.hold["hold_id"], "pay",
                     "--reason", "evidence checked", "--ledger", ledger.url]) == 0
    assert [(row["method"], row["path"], row["verified"]) for row in ledger.requests] == [
        ("GET", "/spend/task/status?hold_id=hold-disputed-1", True),
        ("POST", "/spend/task/resolve", True),
    ]
    assert json.loads(ledger.requests[-1]["body"]) == {
        "hold_id": ledger.hold["hold_id"], "outcome": "pay", "reason": "evidence checked"}
    assert "is paid" in capsys.readouterr().out


def test_replayed_resolution_is_success(ledger, monkeypatch, capsys):
    ledger.replayed_reply = True
    _tty_yes(monkeypatch)
    assert nak.main(["operator", "resolve", ledger.hold["hold_id"], "refund",
                     "--reason", "review confirmed", "--ledger", ledger.url]) == 0
    assert "is refunded (replayed)" in capsys.readouterr().out


def test_409_after_committed_resolution_is_adopted_as_replay(ledger, monkeypatch, capsys):
    ledger.post_error = 409
    ledger.post_error_terminal_state = "paid"
    _tty_yes(monkeypatch)
    assert nak.main(["operator", "resolve", ledger.hold["hold_id"], "pay",
                     "--reason", "review confirmed", "--ledger", ledger.url]) == 0
    assert [row["method"] for row in ledger.requests] == ["GET", "POST", "GET"]
    assert "is paid (replayed)" in capsys.readouterr().out


@pytest.mark.parametrize("status", [403, 404, 409])
def test_ledger_resolution_errors_are_surfaced_escaped(ledger, monkeypatch, status):
    ledger.post_error = status
    ledger.error_detail = "denied\nFORGED\x1b[31m"
    _tty_yes(monkeypatch)
    with pytest.raises(SystemExit) as raised:
        nak.main(["operator", "resolve", ledger.hold["hold_id"], "pay",
                  "--reason", "review complete", "--ledger", ledger.url])
    message = str(raised.value)
    assert f"HTTP {status}" in message
    assert "\n" not in message and "\x1b" not in message
    assert r"\nFORGED\x1b[31m" in message


def test_operator_disputes_reads_named_holds(ledger, capsys):
    assert nak.main(["operator", "disputes", "--hold", ledger.hold["hold_id"],
                     "--ledger", ledger.url]) == 0
    output = capsys.readouterr().out
    assert "hold-disputed-1  disputed  1.25 USDC (125000000 base)" in output
    assert "reason quality" in output
    assert len(ledger.requests) == 1 and ledger.requests[0]["verified"]


def test_operator_disputes_accepts_local_task_ids(ledger, monkeypatch):
    seen = []

    def fake_call(_sock, request):
        seen.append(request)
        return {"hold_id": ledger.hold["hold_id"]}

    monkeypatch.setattr(nak, "call", fake_call)
    assert nak.main(["operator", "disputes", "a1b2c3d4",
                     "--ledger", ledger.url]) == 0
    assert seen == [{"op": "escrow_status", "task": "a1b2c3d4"}]


def test_plain_http_ledger_must_be_loopback(ledger, monkeypatch):
    _tty_yes(monkeypatch)
    with pytest.raises(SystemExit, match="https unless its host is loopback"):
        nak.main(["operator", "resolve", ledger.hold["hold_id"], "pay",
                  "--reason", "review complete", "--ledger", "http://ledger.example:18097"])
    assert ledger.requests == []


def test_mcp_never_exposes_dispute_resolution():
    for server in (Server(_UnusedClient()), Server(_UnusedClient(), allow_connect=True)):
        names = {row["name"] for row in server.list_tools()}
        assert "nak_operator_resolve" not in names
        assert not {name for name in names if "resolve" in name}
