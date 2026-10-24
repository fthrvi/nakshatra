"""M2.1 paid-answer policy: pure table plus human-only CLI/grant surfaces."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from network import nak, nak_mcp, nakd, verdict  # noqa: E402
from network.client import render_tasks  # noqa: E402
from test_nak_grant import _temporary_identity  # noqa: E402


@pytest.mark.parametrize(
    "mode,reward,auto_max,ask_strangers,stranger,rules_passed,escrowed",
    itertools.product(
        ("auto", "ask", "threshold"), (0, 9, 10, 11), (10,),
        (False, True), (False, True), (False, True), (False, True),
    ),
)
def test_decision_table_is_exhaustive(mode, reward, auto_max, ask_strangers, stranger,
                                      rules_passed, escrowed):
    if not rules_passed:
        expected = verdict.REJECT
    elif reward == 0 or not escrowed:
        expected = verdict.ACCEPT
    elif mode == "ask":
        expected = verdict.ASK
    elif ask_strangers and stranger:
        expected = verdict.ASK
    elif mode == "auto":
        expected = verdict.ACCEPT
    else:
        expected = verdict.ACCEPT if reward <= auto_max else verdict.ASK
    assert verdict.decide(mode=mode, reward=reward, auto_max=auto_max,
                          ask_strangers=ask_strangers, stranger=stranger,
                          rules_passed=rules_passed, escrowed=escrowed) == expected


def test_policy_defaults_and_malformed_values_fail_to_ask():
    assert verdict.policy({}) == {"mode": "threshold", "auto_max": 10_000_000,
                                   "ask_strangers": True}
    assert verdict.policy({"verdict_mode": "bogus", "verdict_auto_max": True,
                           "verdict_ask_strangers": "no"}) == {
                               "mode": "ask", "auto_max": 0, "ask_strangers": True}
    assert verdict.policy({"verdict_auto_max": verdict.MAX_AUTO_MAX + 1})["auto_max"] == 0


def test_reprompt_schedule_is_six_hourly_final_at_two_hours_and_capped_at_eight():
    deadline = 100_000

    def task(count, last):
        return {"spec": {"deadline": deadline},
                "verdict": {"approval": {"prompt_count": count, "last_prompted_at": last}}}

    assert nakd.Node._approval_due(task(0, 0), 1)
    assert not nakd.Node._approval_due(task(1, 10_000), 10_000 + nakd.APPROVAL_REPEAT_S - 1)
    assert nakd.Node._approval_due(task(1, 10_000), 10_000 + nakd.APPROVAL_REPEAT_S)
    final_at = deadline - nakd.APPROVAL_FINAL_S
    assert nakd.Node._approval_due(task(3, final_at - 1), final_at)
    assert not nakd.Node._approval_due(task(3, final_at), final_at + 1)
    assert not nakd.Node._approval_due(task(8, 0), 1)
    assert not nakd.Node._approval_due(task(1, 1), deadline)


def test_grant_renew_changes_only_named_verdict_fields_and_prints_policy(
        tmp_path, monkeypatch, capsys):
    _state, _key, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before = json.loads(grant_path.read_text())
    real_person_key = nak.PersonKey
    approvals = []

    class CapturingPersonKey:
        def __init__(self, kind, summary, **kwargs):
            approvals.append((kind, summary))
            self._key = real_person_key(kind, summary, **kwargs)

        def sign(self, statement):
            return self._key.sign(statement)

        def public_key(self):
            return self._key.public_key()

    monkeypatch.setattr(nak, "PersonKey", CapturingPersonKey)
    assert nak.main(["grant", "renew", "--yes", "--verdict", "auto",
                     "--verdict-auto-max", "123", "--verdict-ask-strangers", "no"]) == 0
    changed = json.loads(grant_path.read_text())
    assert changed["constraints"] == {"verdict_mode": "auto", "verdict_auto_max": 123,
                                      "verdict_ask_strangers": False}
    assert "verdict policy: automatic; strangers follow the same rule" in capsys.readouterr().out
    assert approvals[0][0] == "grant-renew"
    assert "ttl 365 days" in approvals[0][1]
    assert ("verdict policy verdict_mode -> auto, verdict_auto_max -> 123, "
            "verdict_ask_strangers -> false") in approvals[0][1]

    assert nak.main(["grant", "renew", "--yes", "--verdict-auto-max", "456"]) == 0
    preserved = json.loads(grant_path.read_text())["constraints"]
    assert preserved == {"verdict_mode": "auto", "verdict_auto_max": 456,
                         "verdict_ask_strangers": False}
    assert before.get("constraints") is None


def test_grant_confirmation_names_effective_verdict_policy(tmp_path, monkeypatch):
    _temporary_identity(tmp_path, monkeypatch)

    class Terminal:
        def isatty(self):
            return True

    prompts = []
    monkeypatch.setattr(nak.sys, "stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "n")
    assert nak.main(["grant", "renew", "--verdict", "ask"]) == 1
    assert "answers: always ask; ask for strangers" in prompts[0]


@pytest.mark.parametrize("value", ("-1", str(nak.MAX_SPEND_CAP + 1)))
def test_grant_refuses_bad_verdict_threshold_without_mutation(tmp_path, monkeypatch, value):
    _state, _key, grant_path = _temporary_identity(tmp_path, monkeypatch)
    before = grant_path.read_bytes()
    with pytest.raises(SystemExit, match="between"):
        nak.main(["grant", "renew", "--yes", "--verdict-auto-max", value])
    assert grant_path.read_bytes() == before


def test_task_decide_is_cli_only_and_requires_a_terminal(monkeypatch, tmp_path):
    class NotTerminal:
        def isatty(self):
            return False

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NAK_NET_DIR", raising=False)
    monkeypatch.setattr(nak.sys, "stdin", NotTerminal())
    monkeypatch.setattr(nak, "call", lambda *_a, **_k: pytest.fail("daemon must not be called"))
    with pytest.raises(SystemExit, match="interactive-only"):
        nak.main(["task", "decide", "12345678", "accept"])
    with pytest.raises(SystemExit):
        nak.main(["task", "decide", "12345678", "accept", "--yes"])
    names = {tool["name"] for tool in nak_mcp.Server(object(), allow_connect=True).list_tools()}
    assert "nak_task_decide" not in names
    assert "nak_task_dispute" not in names and "nak_task_resolve" not in names


def test_waiting_answer_is_visible_in_cli_status_tasks_and_mcp(monkeypatch, capsys):
    grant = {"agent": "agent", "caps": ["nak.task.post"], "expires_at": 2_000_000_000,
             "status": "grant expires later", "warning": "",
             "verdict_policy": {"mode": "threshold", "auto_max": 10_000_000,
                                "ask_strangers": True}}
    status = {"person": "a" * 64, "node": "b" * 64, "custody": "test", "relay": "local",
              "contacts": [], "pending_requests": 0, "open_invites": 0, "direct_port": None,
              "p2p": None, "outbox_refused": 0, "newest_refusal": None,
              "awaiting_approval": 1, "grant": grant}
    monkeypatch.setattr(nak, "call", lambda *_a, **_k: status)
    assert nak.main(["status"]) == 0
    shown = capsys.readouterr().out
    assert "1 paid answer waiting for your decision" in shown
    assert "paid-task verdicts: automatic through 10000000 credits; ask above; ask for strangers" in shown

    row = {"task_hash": "1" * 64, "role": "poster", "state": "awaiting_approval",
           "peer": "", "assignee": "c" * 64, "peer_name": "worker", "name_source": "yours",
           "funding_state": "funded", "spec": {"title": "drill", "instructions": "say ok",
                                                      "acceptance": [{"max_words": 2}],
                                                      "reward": {"amount": 5, "unit": "credits"}},
           "verdict": {"passed": True, "rules_passed": True, "reasons": [], "approval": {}}}
    assert "HUMAN DECISION REQUIRED" in render_tasks([row])

    class Client:
        def status(self):
            return status

    assert "1 paid answer waiting for your decision" in nak_mcp.Server(Client()).call("nak_status", {})


def test_disputed_task_shows_operator_wait_and_backstop_in_cli_and_mcp():
    row = {"task_hash": "2" * 64, "role": "poster", "state": "rejected",
           "peer": "", "assignee": "c" * 64, "peer_name": "worker", "name_source": "yours",
           "funding_state": "disputed",
           "spec": {"title": "drill", "instructions": "say ok", "deadline": 2_000_000_000,
                    "acceptance": [{"max_words": 2}],
                    "reward": {"amount": 5, "unit": "credits"}},
           "verdict": {"passed": False, "reasons": ["rejected by person"]}}
    expected = "disputed — waiting for the operator (backstop 2033-06-01)"
    assert expected in render_tasks([row])

    class Client:
        def tasks(self, _role):
            return [row]

    assert expected in nak_mcp.Server(Client()).call("nak_tasks", {})


def test_task_decide_loads_matching_person_key_and_sends_signed_proof(
        tmp_path, monkeypatch, capsys):
    state, _key, _grant = _temporary_identity(tmp_path, monkeypatch)
    task_hash = "1" * 64

    class Terminal:
        def isatty(self):
            return True

    calls = []
    def fake_call(_sock, req):
        calls.append(req)
        if req["op"] == "tasks":
            return {"tasks": [{"task_hash": task_hash}]}
        return {"task_hash": task_hash, "state": "accepted", "idempotent": False}

    monkeypatch.setattr(nak.sys, "stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "call", fake_call)
    assert nak.main(["task", "decide", "11111111", "accept"]) == 0
    decision = calls[1]
    assert decision["op"] == "task_decide" and decision["task"] == task_hash
    assert decision["decision"] == "accept" and decision["reason"] == ""
    from sthambha import signer as signer_mod
    _priv, person_pub = signer_mod.load_key(tmp_path / ".sthambha" / "person-TEST.key")
    from cryptography.hazmat.primitives.asymmetric import ed25519
    ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(person_pub)).verify(
        bytes.fromhex(decision["signature"]),
        verdict.decision_statement(task_hash, "accept", "", decision["ts"]))
    assert (state / "person.pub").read_text().strip() == person_pub
    assert '"state": "accepted"' in capsys.readouterr().out


def test_task_decide_help_says_posting_policy_is_frozen(capsys):
    with pytest.raises(SystemExit) as exc:
        nak.main(["task", "decide", "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "verdict policy" in help_text and "posted under" in help_text


def test_task_decide_refuses_a_person_key_that_does_not_match_the_signer(
        tmp_path, monkeypatch):
    _temporary_identity(tmp_path, monkeypatch)
    from sthambha import signer as signer_mod
    wrong_private, _wrong_public = signer_mod.new_key()
    signer_mod.save_key(tmp_path / ".sthambha" / "person-TEST.key", wrong_private)

    class Terminal:
        def isatty(self):
            return True

    monkeypatch.setattr(nak.sys, "stdin", Terminal())
    monkeypatch.setattr(nak, "call", lambda _sock, req: {
        "tasks": [{"task_hash": "1" * 64}]} if req["op"] == "tasks" else {})
    with pytest.raises(SystemExit, match="does not match"):
        nak.main(["task", "decide", "11111111", "accept"])
