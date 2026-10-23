"""D8.2 public request addresses: strangers may ask; only the person may accept."""
from __future__ import annotations

import threading
import time

import pytest

import joincode
from test_nakd import net, wait  # noqa: F401 - also resolves the sibling sthambha checkout
from network import nak, nak_mcp, nakd
from network.client import NakClient


RELEASE = {"url": "https://releases.example", "channel": "canary",
           "pubkey": "ef" * 32, "version": "1.2.3"}


def _address(p, *, days=90, now=None, name="Biswa"):
    code = joincode.encode_address(
        p.person_priv, p.node.node, [{"node": p.node.node, "addrs": ["127.0.0.1"]}],
        days=days, now=now, from_name=name, release=RELEASE)
    p.node.register_address(code, "install-line " + code)
    return code, joincode.decode_address(code, now=now + 1 if now is not None else None)


def _frame(inviter, stranger, address, **changes):
    body = {"person": stranger.person_pub, "node": stranger.node.node, "nickname": "Stranger",
            "address": address["aid"], "mid": "ab" * 12}
    body.update(changes)
    env = stranger.node._sign("contact.request", inviter.person_pub, body)
    return {"t": "env", "env": env, "body": body}


def _person_accept(party, request_id, petname="stranger"):
    ts = int(time.time())
    sig = party.person_priv.sign(nakd.accept_statement(request_id, petname, ts)).hex()
    return party.node.accept(request_id, petname, ts=ts, signature=sig)


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


def test_address_rotation_off_wrong_aid_and_expiry_are_immediate(net):
    inviter, stranger = net("inviter"), net("stranger")
    _old_code, old = _address(inviter)
    _new_code, new = _address(inviter)
    assert old["aid"] != new["aid"]
    refused = inviter.node._take_request(_frame(inviter, stranger, old), "")
    assert refused == {"t": "nack", "why": "request address is off, expired, or no longer current"}
    assert inviter.node.store.requests() == []
    inviter.node.address_off()
    assert inviter.node._take_request(_frame(inviter, stranger, new), "")["t"] == "nack"
    _code, current = _address(inviter)
    inviter.node.store._q("UPDATE address_state SET expires_at=? WHERE slot=1", (int(time.time()) - 1,))
    assert "expired" in inviter.node._take_request(_frame(inviter, stranger, current), "")["why"]


@pytest.mark.parametrize("ledger", [False, True], ids=["without-ledger", "with-ledger"])
def test_both_address_and_invite_are_rejected_without_consuming(net, ledger):
    inviter, stranger = net("inviter"), net("stranger")
    invite_code = inviter.invite()
    invite = joincode.decode_invite(invite_code)
    _code, address = _address(inviter)
    changes = {"invite": invite["nonce"]}
    if ledger:
        changes["ledger_key"] = stranger.node.node
    reply = inviter.node._take_request(_frame(inviter, stranger, address, **changes), invite["nonce"])
    assert reply == {"t": "nack", "why": "request cannot carry both an invite and an address"}
    assert any(row["nonce"] == invite["nonce"] for row in inviter.node.store.open_invites())
    assert inviter.node.store.requests() == []


def test_address_request_gates_self_block_contact_and_per_person(net):
    inviter, stranger = net("inviter"), net("stranger")
    _code, address = _address(inviter)

    self_body = {"person": inviter.person_pub, "node": inviter.node.node, "nickname": "me",
                 "address": address["aid"], "mid": "cd" * 12}
    self_env = inviter.node._sign("contact.request", inviter.person_pub, self_body)
    assert inviter.node._take_request({"t": "env", "env": self_env, "body": self_body}, "")["why"] == \
        "cannot connect to yourself"

    inviter.node.store.block(stranger.person_pub, int(time.time()) + 60, "test")
    assert inviter.node._take_request(_frame(inviter, stranger, address), "")["why"] == \
        "not accepting requests from you"
    inviter.node.store.unblock(stranger.person_pub)

    assert inviter.node._take_request(_frame(inviter, stranger, address), "")["t"] == "ack"
    assert "already have" in inviter.node._take_request(_frame(inviter, stranger, address), "")["why"]
    rid = inviter.node.store.requests()[0]["id"]
    inviter.node.store.decide_request(rid, False)
    inviter.node.store.upsert_contact(stranger.person_pub, stranger.node.node, "active", "known")
    assert inviter.node._take_request(_frame(inviter, stranger, address), "")["why"] == \
        "you are already a contact"


def test_address_pending_total_and_rolling_daily_caps(net):
    inviter, stranger = net("inviter"), net("stranger")
    _code, address = _address(inviter)
    for index in range(nakd.ADDRESS_PENDING_TOTAL_MAX):
        inviter.node.store.add_request(
            f"{index + 1:064x}", f"{index + 101:064x}", "load", "", source="address",
            source_id=address["aid"])
    assert inviter.node._take_request(_frame(inviter, stranger, address), "")["why"] == "busy, try later"

    for row in inviter.node.store.requests():
        inviter.node.store.decide_request(row["id"], True)
    existing = nakd.ADDRESS_PENDING_TOTAL_MAX
    for index in range(nakd.ADDRESS_DAILY_MAX - existing):
        rid = inviter.node.store.add_request(
            f"{index + 1000:064x}", f"{index + 2000:064x}", "daily", "", source="address",
            source_id=address["aid"])
        inviter.node.store.decide_request(rid, True)
    assert inviter.node._take_request(_frame(inviter, stranger, address), "")["why"] == \
        "daily request limit reached, try later"
    inviter.node.store._q(
        "UPDATE requests SET received=? WHERE id IN (SELECT request_id FROM request_sources)",
        (int(time.time()) - 86401,))
    assert inviter.node._take_request(_frame(inviter, stranger, address), "")["t"] == "ack"


def _control_server(node, tmp_path):
    stop = threading.Event()
    sock = tmp_path / "nakd.sock"
    threading.Thread(target=nakd.serve_control, args=(node, sock, stop), daemon=True).start()
    assert wait(sock.exists)
    return sock, stop


def test_address_accept_is_one_guard_for_mcp_socket_cli_and_person(net, tmp_path, monkeypatch):
    inviter, stranger = net("inviter"), net("stranger")
    _code, address = _address(inviter)
    assert inviter.node._take_request(_frame(inviter, stranger, address), "")["t"] == "ack"
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
    assert inviter.node._take_request(frame, "")["t"] == "ack"
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
    inviter.node._take_request(_frame(inviter, stranger, address), "")
    rid = inviter.node.store.requests()[0]["id"]
    before = int(time.time())
    result = inviter.node.decline(rid)
    assert before + 30 * 86400 <= result["blocked_until"] <= int(time.time()) + 30 * 86400
    assert inviter.node.blocked()[0]["person"] == stranger.person_pub
    assert inviter.node._take_request(_frame(inviter, stranger, address), "")["why"] == \
        "not accepting requests from you"
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
