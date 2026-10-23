"""Starter-credit obligations: signed invite -> accept -> capped, idempotent Neuron payment."""
from __future__ import annotations

import base64
import io
import json
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from sthambha import signer as signer_mod

import joincode
from network import nak, nak_mcp, nakd
from network.client import NakClient
from network.settle import NeuronEscrowAdapter, SettlementError
from network.store import Store
from test_nakd import net, wait  # noqa: F401


PAY = {"per_tx_cap": 100_000_000, "day_cap": 200_000_000}


class FakeStarterLedger:
    name = "neuron"
    open_on_assign = True

    def __init__(self):
        self.calls: list[dict] = []
        self.paid: dict[str, dict] = {}
        self.failures: list[tuple[int, str]] = []

    def starter(self, invite_id, to_keyid, amount):
        self.calls.append({"invite_id": invite_id, "to_keyid": to_keyid, "amount": amount})
        if self.failures:
            status, message = self.failures.pop(0)
            error = SettlementError(f"Neuron ledger /spend/starter refused HTTP {status}")
            error.status, error.ledger_error = status, message
            raise error
        replayed = invite_id in self.paid
        row = self.paid.setdefault(invite_id, {
            "v": 1, "invite_id": invite_id, "from_account": "inviter", "to_account": "consumer",
            "to_keyid": to_keyid, "amount": amount, "ts": int(time.time()), "replayed": False,
        })
        return dict(row, replayed=replayed)


def _consumer_invite(inviter, *, amount=12345, invite_id="consumer_code"):
    code = joincode.encode_invite(
        inviter.person_priv, inviter.node.node,
        [{"node": inviter.node.node, "addrs": ["127.0.0.1"]}],
        kind="consumer", consumer={"redeem": "https://onboard.test", "code": invite_id,
                                   "tier": "known", "starter": amount})
    inviter.node.register_invite(code)
    return code


def _accept_consumer(inviter, consumer, code):
    consumer.node.redeem(code, nickname="consumer", petname="inviter", wait_s=15)
    assert wait(lambda: len(inviter.node.store.requests()) == 1)
    request = inviter.node.store.requests()[0]
    assert request["ledger_key"] == consumer.node.node
    _human_accept(inviter, request["id"], "consumer")
    assert wait(lambda: (consumer.node.store.contact(inviter.person_pub) or {}).get("state") == "active")


def _human_accept(inviter, request_id, petname=""):
    ts = int(time.time())
    signature = inviter.person_priv.sign(nakd.accept_statement(request_id, petname, ts)).hex()
    return inviter.node.accept(request_id, petname=petname, ts=ts, signature=signature)


def test_two_node_accept_pays_once_and_restart_does_not_repeat(net, monkeypatch):
    inviter = net("inviter", caps=("nak.msg", "nak.pay"), constraints=PAY)
    consumer = net("consumer")
    ledger = FakeStarterLedger()
    inviter.logs = []
    inviter.node.log = inviter.logs.append
    inviter.node.settle = ledger
    monkeypatch.setenv("NAKSHATRA_ESCROW", "neuron")

    _accept_consumer(inviter, consumer, _consumer_invite(inviter))
    assert wait(lambda: inviter.node.starter_status()["paid_count"] == 1)
    assert inviter.node.store.authenticated_contact_node(consumer.person_pub) == consumer.node.node
    assert ledger.calls == [{"invite_id": "consumer_code", "to_keyid": consumer.node.node,
                             "amount": 12345}]
    approvals = [r for r in inviter.signer_requests if r.get("op") == "approve_spend"]
    assert len(approvals) == 1 and approvals[0]["touch"] is False
    assert any("starter: paid 12345 credits to consumer" in line for line in inviter.logs)

    old = inviter.node
    old.stop()
    inviter.node = nakd.Node(
        inviter.dir / "net", inviter.node_key, old._signer, agent="pa", relay=old.relay,
        log=inviter.logs.append, settlement=ledger, roster_loader=old._roster_loader).start()
    assert wait(lambda: inviter.node.starter_status()["paid_count"] == 1)
    time.sleep(0.1)
    assert len(ledger.calls) == 1


def test_replay_after_ledger_200_before_local_write_is_adopted(net, monkeypatch):
    inviter = net("inviter", caps=("nak.msg", "nak.pay"), constraints=PAY)
    consumer = net("consumer")
    ledger = FakeStarterLedger()
    inviter.node.settle = ledger
    _accept_consumer(inviter, consumer, _consumer_invite(inviter, invite_id="crash_case"))
    assert wait(lambda: "NAKSHATRA_ESCROW" in
                inviter.node.starter_status()["starters"][0]["reason"])
    monkeypatch.setenv("NAKSHATRA_ESCROW", "neuron")
    inviter.node.store.starter_update("crash_case", next_attempt=0)

    original = inviter.node.store.starter_update
    crashed = False

    def crash_before_paid(invite_id, **changes):
        nonlocal crashed
        if changes.get("state") == "paid" and not crashed:
            crashed = True
            raise RuntimeError("simulated crash before local paid write")
        return original(invite_id, **changes)

    monkeypatch.setattr(inviter.node.store, "starter_update", crash_before_paid)
    with pytest.raises(RuntimeError, match="simulated crash"):
        inviter.node._attempt_starter("crash_case")
    monkeypatch.setattr(inviter.node.store, "starter_update", original)
    assert len(ledger.paid) == 1
    assert inviter.node._attempt_starter("crash_case") is True
    assert len(ledger.calls) == 2 and len(ledger.paid) == 1
    assert inviter.node.starter_status()["paid_count"] == 1


def test_roster_404_retries_then_pays(net, monkeypatch):
    inviter = net("inviter", caps=("nak.msg", "nak.pay"), constraints=PAY)
    consumer = net("consumer")
    ledger = FakeStarterLedger()
    ledger.failures.append((404, "recipient not in roster"))
    inviter.node.settle = ledger
    monkeypatch.setenv("NAKSHATRA_ESCROW", "neuron")
    _accept_consumer(inviter, consumer, _consumer_invite(inviter, invite_id="late_roster"))
    assert wait(lambda: inviter.node.starter_status()["starters"][0]["reason"] == "recipient not in roster")
    pending = inviter.node.starter_status()["starters"][0]
    assert pending["state"] == "pending" and pending["attempts"] >= 1
    inviter.node.store.starter_update("late_roster", next_attempt=0)
    assert inviter.node.reconcile_starters() == 1
    assert inviter.node.starter_status()["paid_count"] == 1


@pytest.mark.parametrize("status", [409, 422])
def test_ledger_conflict_or_invalid_request_is_terminal(net, monkeypatch, status):
    inviter = net("inviter", caps=("nak.msg", "nak.pay"), constraints=PAY)
    consumer = net("consumer")
    ledger = FakeStarterLedger()
    message = "recipient already has a starter grant" if status == 409 else "recipient is reserved"
    ledger.failures.append((status, message))
    inviter.node.settle = ledger
    monkeypatch.setenv("NAKSHATRA_ESCROW", "neuron")
    _accept_consumer(inviter, consumer, _consumer_invite(inviter, invite_id="conflict"))
    assert wait(lambda: inviter.node.starter_status()["starters"][0]["state"] == "failed")
    failed = inviter.node.starter_status()["starters"][0]
    assert failed["reason"] == message
    inviter.node.reconcile_starters(now=int(time.time()) + nakd.STARTER_RETRY_WINDOW_S + 1)
    assert len(ledger.calls) == 1


def test_lapsed_grant_and_disabled_backend_stay_pending(net, monkeypatch):
    inviter = net("inviter", caps=("nak.msg", "nak.pay"), constraints=PAY)
    consumer = net("consumer")
    ledger = FakeStarterLedger()
    inviter.node.settle = ledger
    _accept_consumer(inviter, consumer, _consumer_invite(inviter, invite_id="lapsed"))
    assert wait(lambda: "NAKSHATRA_ESCROW" in
                inviter.node.starter_status()["starters"][0]["reason"])
    assert not ledger.calls
    row = inviter.node.starter_status()["starters"][0]
    assert row["state"] == "pending" and "NAKSHATRA_ESCROW" in row["reason"]

    monkeypatch.setenv("NAKSHATRA_ESCROW", "neuron")
    monkeypatch.setattr(inviter.node, "_grant_status", lambda *_a, **_k: {
        "valid": False, "caps": ["nak.msg", "nak.pay"],
        "problem": "lapsed: unused since 2026-09-01 00:00:00 UTC",
    })
    inviter.node.store.starter_update("lapsed", next_attempt=0)
    assert inviter.node.reconcile_starters() == 0 and not ledger.calls
    row = inviter.node.starter_status()["starters"][0]
    assert row["state"] == "pending" and "lapsed" in row["reason"]


def test_signature_tamper_is_failed_without_calling_ledger(net, monkeypatch):
    inviter = net("inviter", caps=("nak.msg", "nak.pay"), constraints=PAY)
    consumer = net("consumer")
    ledger = FakeStarterLedger()
    inviter.node.settle = ledger
    _accept_consumer(inviter, consumer, _consumer_invite(inviter, invite_id="tamper"))
    assert wait(lambda: "NAKSHATRA_ESCROW" in
                inviter.node.starter_status()["starters"][0]["reason"])
    row = inviter.node.store.starter_rows()[0]
    body = json.loads(joincode._unb64u(row["signed_invite"][5:]))
    body["consumer"]["starter"] += 1
    tampered = "nki1." + joincode._b64u(json.dumps(body, separators=(",", ":")).encode())
    pk = inviter.node.store._pk(("tamper",))
    sealed = inviter.node.store._enc("starter_obligations", "signed_invite", pk, tampered)
    inviter.node.store._q("UPDATE starter_obligations SET signed_invite=?,next_attempt=0 WHERE invite_id=?",
                          (sealed, "tamper"))
    monkeypatch.setenv("NAKSHATRA_ESCROW", "neuron")
    inviter.node.reconcile_starters()
    final = inviter.node.starter_status()["starters"][0]
    assert final["state"] == "failed" and "signature" in final["reason"] and not ledger.calls


def test_old_contact_request_without_ledger_key_connects_but_stays_pending(net):
    inviter = net("inviter", caps=("nak.msg", "nak.pay"), constraints=PAY)
    consumer = net("consumer")
    code = _consumer_invite(inviter, invite_id="old_node")
    inv = joincode.decode_invite(code)
    body = {"person": consumer.person_pub, "node": consumer.node.node, "nickname": "old",
            "invite": inv["nonce"], "mid": "a" * 24}
    env = consumer.node._sign("contact.request", inviter.person_pub, body)
    assert inviter.node._take_request(
        {"t": "env", "env": env, "body": body}, "invite", inv["nonce"])["t"] == "ack"
    _human_accept(inviter, inviter.node.store.requests()[0]["id"], "old")
    starter = inviter.node.starter_status()["starters"][0]
    assert starter["state"] == "pending" and starter["to_keyid"] is None
    assert starter["reason"] == "recipient ledger key unknown"


def test_claimed_rostered_key_waits_for_that_node_but_connected_contact_pays(net, monkeypatch):
    inviter = net("inviter", caps=("nak.msg", "nak.pay"), constraints=PAY)
    consumer = net("consumer")
    other = net("other")
    ledger = FakeStarterLedger()
    inviter.node.settle = ledger
    monkeypatch.setenv("NAKSHATRA_ESCROW", "neuron")
    code = _consumer_invite(inviter, invite_id="wrong_recipient")
    inv = joincode.decode_invite(code)
    # Build a valid person/agent-signed request whose grant merely names another rostered node key.
    # No one in this request possesses that node key, so only the later pinned handshake can reject it.
    agent_priv, agent_pub = signer_mod.new_key()
    grant = signer_mod.issue_grant(consumer.person_priv, "liar", agent_pub, ("nak.msg",),
                                   node=other.node.node)
    body = {"person": consumer.person_pub, "node": other.node.node, "ledger_key": other.node.node,
            "nickname": "consumer", "invite": inv["nonce"], "mid": "a" * 24}
    envelope = signer_mod._sign(agent_priv, {  # noqa: SLF001 - construct the hostile wire claim
        "schema": signer_mod.ENVELOPE_SCHEMA, "kind": "contact.request",
        "from_node": other.node.node, "author": f"agent:liar@{other.node.node[:16]}",
        "to": inviter.person_pub, "ts": int(time.time()), "nonce": "b" * 24,
        "body_sha256": nakd.body_hash(body), "grant": grant,
    })
    assert inviter.node._take_request(
        {"t": "env", "env": envelope, "body": body}, "invite", inv["nonce"])["t"] == "ack"
    request = inviter.node.store.requests()[0]
    _human_accept(inviter, request["id"], "consumer")
    assert wait(lambda: "waiting for consumer's node to connect" in
                inviter.node.starter_status()["starters"][0]["reason"])
    assert ledger.calls == []


def _control_server(node, tmp_path):
    stop = threading.Event()
    sock = tmp_path / f"{node.node[:8]}.sock"
    threading.Thread(target=nakd.serve_control, args=(node, sock, stop), daemon=True).start()
    assert wait(sock.exists)
    return sock, stop


def _mcp_tool(server, name, arguments):
    return server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}})["result"]


def test_mcp_cannot_accept_starter_request_but_can_accept_zero_starter(net, tmp_path):
    inviter = net("inviter", caps=("nak.msg", "nak.pay"), constraints=PAY)
    paid_consumer = net("paid-consumer")
    free_consumer = net("free-consumer")
    paid_consumer.node.redeem(_consumer_invite(inviter, invite_id="person_only"), wait_s=15)
    assert wait(lambda: len(inviter.node.store.requests()) == 1)
    paid_request = inviter.node.store.requests()[0]
    sock, stop = _control_server(inviter.node, tmp_path)
    server = nak_mcp.Server(NakClient(sock), allow_connect=True)

    refused = _mcp_tool(server, "nak_accept", {"id": paid_request["id"], "name": "paid"})
    assert refused["isError"]
    assert f"a person must accept this one: nak accept {paid_request['id']}" in refused["content"][0]["text"]
    assert inviter.node.store.request(paid_request["id"]) is not None
    assert inviter.node.store.starter_rows() == []

    free_consumer.node.redeem(
        _consumer_invite(inviter, amount=0, invite_id="zero_starter"), wait_s=15)
    assert wait(lambda: len(inviter.node.store.requests()) == 2)
    free_request = next(row for row in inviter.node.store.requests()
                        if row["person"] == free_consumer.person_pub)
    accepted = _mcp_tool(server, "nak_accept", {"id": free_request["id"], "name": "free"})
    assert not accepted["isError"]
    assert inviter.node.store.request(free_request["id"]) is None
    stop.set()


def test_cli_accept_builds_tty_and_person_signature_for_starter(monkeypatch):
    person = ed25519.Ed25519PrivateKey.generate()
    request_id = "abcd1234"
    calls = []

    class Terminal:
        def isatty(self):
            return True

    def fake_call(_sock, req):
        calls.append(req)
        if req["op"] == "requests":
            return {"requests": [{"id": request_id, "requires_person": True}]}
        return {"person": "1" * 64, "petname": "friend"}

    monkeypatch.setattr(nak.sys, "stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "call", fake_call)
    monkeypatch.setattr(nak, "_load_matching_person_key", lambda: person)
    assert nak.main(["accept", request_id, "--as", "friend"]) == 0
    proof = calls[1]
    person.public_key().verify(bytes.fromhex(proof["signature"]),
                               nakd.accept_statement(request_id, "friend", proof["ts"]))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_adapter_consumes_captured_starter_fixture_verbatim():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "starter_replies.json").read_text())
    replies = iter((fixture["grant"], fixture["replay"], fixture["status"]))
    requests = []

    def opener(request, **_kwargs):
        requests.append(request)
        return _Response(json.dumps(next(replies)).encode())

    key = ed25519.Ed25519PrivateKey.generate()
    keyid = key.public_key().public_bytes_raw().hex()
    adapter = NeuronEscrowAdapter("https://ledger.test", "ab" * 32, key.private_bytes_raw(), keyid,
                                  opener=opener)
    expected = fixture["grant"]
    assert adapter.starter(expected["invite_id"], expected["to_keyid"], expected["amount"]) == expected
    assert adapter.starter(expected["invite_id"], expected["to_keyid"], expected["amount"]) == fixture["replay"]
    assert adapter.starter_grant_status(expected["invite_id"]) == fixture["status"]
    assert json.loads(requests[0].data) == {
        "invite_id": expected["invite_id"], "to_keyid": expected["to_keyid"], "amount": expected["amount"]}
    assert urllib.parse.urlsplit(requests[2].full_url).path == "/spend/starter/status"


def test_old_store_schema_opens_and_agents_have_no_starter_tool(tmp_path):
    state = tmp_path / "net"
    state.mkdir()
    db = sqlite3.connect(state / "net.sqlite")
    db.executescript("""
        CREATE TABLE contacts (person TEXT PRIMARY KEY,node TEXT NOT NULL,petname TEXT,nickname TEXT,
          state TEXT NOT NULL,added INTEGER NOT NULL);
        CREATE TABLE requests (id TEXT PRIMARY KEY,person TEXT NOT NULL,node TEXT NOT NULL,nickname TEXT,
          invite_nonce TEXT,received INTEGER NOT NULL,state TEXT NOT NULL);
        CREATE TABLE invites (nonce TEXT PRIMARY KEY,ik TEXT NOT NULL,expires_at INTEGER NOT NULL,
          created INTEGER NOT NULL,note TEXT,consumed_by TEXT);
    """)
    db.close()
    store = Store(state, b"k" * 32)
    assert store.starter_rows() == []
    assert all("starter" not in name for name in (*nak_mcp.READ_TOOLS, *nak_mcp.CONNECT_TOOLS))
    assert "starter_status" in nakd.OPS


def test_fresh_store_keeps_old_positional_request_and_invite_inserts_rollback_safe(tmp_path):
    state = tmp_path / "net"
    current = Store(state, b"k" * 32)
    current._db.close()
    old_source = subprocess.run(
        ["git", "show", "8143916:scripts/network/store.py"], cwd=Path(__file__).parent.parent,
        check=True, capture_output=True, text=True).stdout
    old_module = tmp_path / "old_store.py"
    old_module.write_text(old_source)
    probe = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from old_store import Store
s = Store(Path(sys.argv[2]), b'k' * 32)
s.add_invite('old-invite', '11' * 32, 4102444800, 'old note')
s.add_request('22' * 32, '33' * 32, 'old nickname', 'old-invite')
"""
    result = subprocess.run([sys.executable, "-c", probe, str(tmp_path), str(state)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_store_adopts_side_data_from_earlier_starter_branch_schema(tmp_path):
    state = tmp_path / "net"
    old_source = subprocess.run(
        ["git", "show", "0c30a48:scripts/network/store.py"], cwd=Path(__file__).parent.parent,
        check=True, capture_output=True, text=True).stdout
    old_module = tmp_path / "old_starter_store.py"
    old_module.write_text(old_source)
    probe = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from old_starter_store import Store
s = Store(Path(sys.argv[2]), b'k' * 32)
s.add_invite('migrated-invite', '11' * 32, 4102444800, 'note', 'signed-code')
rid = s.add_request('22' * 32, '33' * 32, 'nickname', 'migrated-invite', '33' * 32)
Path(sys.argv[3]).write_text(rid)
s._db.close()
"""
    rid_file = tmp_path / "rid"
    result = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path), str(state), str(rid_file)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    current = Store(state, b"k" * 32)
    assert current.invite("migrated-invite")["code"] == "signed-code"
    assert current.request(rid_file.read_text())["ledger_key"] == "33" * 32
    assert {row[1] for row in current._q("PRAGMA table_info(invites)")} >= {"code"}
    assert {row[1] for row in current._q("PRAGMA table_info(requests)")} >= {"ledger_key"}


def test_handshake_marks_only_the_node_it_pinned(net):
    inviter = net("inviter", caps=("nak.msg", "nak.pay"), constraints=PAY)
    consumer = net("consumer")
    ledger = FakeStarterLedger()
    inviter.node.settle = ledger
    _accept_consumer(inviter, consumer, _consumer_invite(inviter))
    store = inviter.node.store
    assert wait(lambda: store.authenticated_contact_node(consumer.person_pub) == consumer.node.node)

    # A handshake that pinned some other node (the contact's node changed underneath it) is
    # not evidence for the contact's current node, and is never recorded.
    store._q("DELETE FROM authenticated_contacts")
    inviter.node._note_authenticated(consumer.person_pub, "ab" * 32)
    assert store.authenticated_contact_node(consumer.person_pub) == ""
    inviter.node._note_authenticated(consumer.person_pub, "")
    assert store.authenticated_contact_node(consumer.person_pub) == ""
    inviter.node._note_authenticated(consumer.person_pub, consumer.node.node)
    assert store.authenticated_contact_node(consumer.person_pub) == consumer.node.node
