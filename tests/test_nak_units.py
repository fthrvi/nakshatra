from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from network import nak, nak_mcp, nakd  # noqa: E402
from network.units import AmountError, MAX_SPEND_CAP, format_amount, parse_amount  # noqa: E402


@pytest.mark.parametrize(("text", "expected", "legacy"), [
    ("0.25", 25_000_000, False),
    ("0.25 USDC", 25_000_000, False),
    ("0.25usdc", 25_000_000, False),
    ("$0.25", 25_000_000, False),
    ("25000000 base", 25_000_000, False),
    ("1.00000001 USDC", 100_000_001, False),
    ("500", 500, True),
    (500, 500, True),
])
def test_parse_amount_table(text, expected, legacy):
    assert parse_amount(text) == (expected, legacy)


@pytest.mark.parametrize("text", [
    "-0.01", "NaN", "Infinity", "0.000000001", "1e-2", "USDC 1", "$", True,
    f"{MAX_SPEND_CAP + 1} base", "1000000.00000001 USDC", 0.25,
    "²", "9" * 5000, " 1", "1 ", "+1", "1e5", "0x10",
])
def test_parse_amount_rejects_unsafe_values(text):
    with pytest.raises(AmountError):
        parse_amount(text)


@pytest.mark.parametrize(("base", "shown"), [
    (0, "0 USDC (0 base)"),
    (25_000_000, "0.25 USDC (25000000 base)"),
    (100_000_001, "1.00000001 USDC (100000001 base)"),
    (200_000_000, "2 USDC (200000000 base)"),
])
def test_format_amount(base, shown):
    assert format_amount(base) == shown


def test_bare_cli_integer_stays_base_units_with_notice(capsys):
    assert nak._cli_amount("500") == 500
    assert "read bare integer '500' as base units" in capsys.readouterr().err


def test_grant_status_shows_human_and_base_caps(capsys):
    nak._print_grant_status({
        "agent": "agent", "caps": ["nak.pay"], "expires_at": 1_800_000_000,
        "spend_caps": {"per_tx_cap": 25_000_000, "day_cap": 100_000_000},
    })
    shown = capsys.readouterr().out
    assert "0.25 USDC (25000000 base) per payment" in shown
    assert "1 USDC (100000000 base) per day" in shown


class _TaskClient:
    def __init__(self):
        self.reward = None

    def task_post(self, _to, _title, _instructions, _acceptance, reward, _deadline):
        self.reward = reward
        return {"task_hash": "ab" * 32, "posted_to": ["friend"]}


def test_mcp_task_post_accepts_human_reward_and_returns_both_units():
    client = _TaskClient()
    shown = nak_mcp.Server(client).call("nak_task_post", {
        "to": "friend", "title": "work", "instructions": "Do it",
        "acceptance": [{"max_words": 2}], "reward": "0.05 USDC",
    })
    assert client.reward == 5_000_000
    assert "0.05 USDC (5000000 base)" in shown


@pytest.mark.parametrize("reward", [999999.99999999999, 1000000.00000000001, 0.0])
def test_mcp_task_post_refuses_json_float_rewards(reward):
    client = _TaskClient()
    result = nak_mcp.Server(client).handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "nak_task_post", "arguments": {
            "to": "friend", "title": "work", "instructions": "Do it",
            "acceptance": [{"max_words": 2}], "reward": reward,
        }},
    })["result"]
    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        'Not done: give reward as a string like "0.05 USDC" or an integer number of base units'
    )
    assert client.reward is None


def test_mcp_stdio_server_keeps_serving_after_bad_tool_argument(monkeypatch):
    bad = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "nak_task_post", "arguments": {
            "to": "friend", "title": "work", "instructions": "Do it",
            "acceptance": "not json", "reward": "1 USDC",
        }},
    }
    ping = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    stdin = io.StringIO(json.dumps(bad) + "\n" + json.dumps(ping) + "\n")
    stdout = io.StringIO()
    monkeypatch.setattr(nak_mcp, "NakClient", lambda _sock: _TaskClient())
    monkeypatch.setattr(nak_mcp.sys, "stdin", stdin)
    monkeypatch.setattr(nak_mcp.sys, "stdout", stdout)

    assert nak_mcp.main([]) == 0
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert replies[0]["result"]["isError"] is True
    assert replies[0]["result"]["content"][0]["text"].startswith("Not done: ")
    assert replies[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}


class _IntegerClient(_TaskClient):
    def inbox(self, *, limit):
        raise AssertionError(f"invalid limit reached client: {limit!r}")

    def agent_send(self, *_args, **kwargs):
        raise AssertionError(f"invalid expiry reached client: {kwargs!r}")


_TOOL_INTEGER_CASES = [
    ("nak_inbox", "limit", {}),
    ("nak_agent_send", "expires", {"to": "friend", "text": "hello"}),
    ("nak_task_post", "reward", {
        "to": "friend", "title": "work", "instructions": "Do it",
        "acceptance": [{"max_words": 2}],
    }),
    ("nak_task_post", "deadline_s", {
        "to": "friend", "title": "work", "instructions": "Do it",
        "acceptance": [{"max_words": 2}],
    }),
    ("nak_ask", "max_tokens", {"prompt": "hello"}),
]


@pytest.mark.parametrize(("tool", "field", "arguments"), _TOOL_INTEGER_CASES)
@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 10**100, True, 1.5])
def test_mcp_rejects_every_non_strict_or_unbounded_integer_argument(
        monkeypatch, tool, field, arguments, bad):
    monkeypatch.setattr(nak_mcp, "_models", lambda: (_ for _ in ()).throw(
        AssertionError("invalid max_tokens reached model discovery")))
    client = _IntegerClient()
    supplied = dict(arguments, **{field: bad})
    server = nak_mcp.Server(client)

    refused = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": supplied},
    })

    result = refused["result"]
    assert result["isError"] is True
    assert result["content"][0]["text"].startswith("Not done: ")
    assert server.handle({"jsonrpc": "2.0", "id": 2, "method": "ping"}) == {
        "jsonrpc": "2.0", "id": 2, "result": {},
    }


@pytest.mark.parametrize("field", ["reward", "deadline_s"])
@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 10**100, True, 1.5, -1])
def test_nakd_task_socket_rejects_non_strict_or_unbounded_integers(field, bad):
    class NeverPost:
        def post_task(self, *_args, **_kwargs):
            raise AssertionError("invalid task integer reached Node.post_task")

    request = {
        "op": "task_post", "to": "friend", "title": "work", "instructions": "Do it",
        "acceptance": [{"max_words": 2}], field: bad,
    }
    result = nakd.handle(NeverPost(), request)

    assert result["ok"] is False
    assert f"{field} must be an integer between" in result["error"]


@pytest.mark.parametrize(("where", "value", "message"), [
    ("params", [], "params must be an object"),
    ("params", "bad", "params must be an object"),
    ("arguments", [], "arguments must be an object"),
    ("arguments", "bad", "arguments must be an object"),
])
def test_mcp_rejects_non_object_params_and_arguments_and_keeps_serving(where, value, message):
    params = value if where == "params" else {"name": "nak_status", "arguments": value}
    server = nak_mcp.Server(_TaskClient())

    refused = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params,
    })

    assert refused == {
        "jsonrpc": "2.0", "id": 1,
        "error": {"code": -32602, "message": message},
    }
    assert server.handle({"jsonrpc": "2.0", "id": 2, "method": "ping"}) == {
        "jsonrpc": "2.0", "id": 2, "result": {},
    }


def test_mcp_catch_all_hides_exception_text_and_keeps_serving():
    class CrashingClient:
        def status(self):
            raise RuntimeError("secret internal path")

    server = nak_mcp.Server(CrashingClient())
    refused = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "nak_status", "arguments": {}},
    })

    assert refused == {
        "jsonrpc": "2.0", "id": 1,
        "error": {"code": -32603, "message": "internal error"},
    }
    assert "secret" not in json.dumps(refused)
    assert server.handle({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"] == {}


def test_mcp_stdio_survives_infinite_deadline_and_answers_next_request(monkeypatch):
    bad = ('{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
           '{"name":"nak_task_post","arguments":{"to":"friend","title":"work",'
           '"instructions":"Do it","acceptance":[{"max_words":2}],"deadline_s":1e309}}}')
    ping = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    stdin = io.StringIO(bad + "\n" + ping + "\n")
    stdout = io.StringIO()
    monkeypatch.setattr(nak_mcp, "NakClient", lambda _sock: _TaskClient())
    monkeypatch.setattr(nak_mcp.sys, "stdin", stdin)
    monkeypatch.setattr(nak_mcp.sys, "stdout", stdout)

    assert nak_mcp.main([]) == 0
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert replies[0]["result"]["isError"] is True
    assert replies[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_mcp_amount_fields_are_additive():
    row = nak_mcp._amount_fields({"balance": 25_000_000, "tier": "known"}, "balance")
    assert row["balance"] == 25_000_000
    assert row["balance_usdc"] == "0.25 USDC (25000000 base)"
    assert row["balance_display"] == "0.25 USDC (25000000 base)"
    assert json.loads(json.dumps(row))["tier"] == "known"
