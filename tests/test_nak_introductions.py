"""D8.1 introductions across three real temp nodes and the local relay."""
from __future__ import annotations

import time

import pytest

from test_nakd import net, wait  # noqa: F401 - first resolves the sibling sthambha checkout
from network import introductions as intro_mod
from network import nak_mcp, nakd
from network.client import NakClient


def _connect(inviter, peer, *, inviter_name="a", peer_name="peer"):
    assert peer.node.redeem(
        inviter.invite(), nickname=peer_name, petname=inviter_name, wait_s=15)["state"] == "request sent"
    request = next(row for row in inviter.node.store.requests() if row["person"] == peer.person_pub)
    inviter.node.accept(request["id"], petname=peer_name)
    assert wait(lambda: (peer.node.store.contact(inviter.person_pub) or {}).get("state") == "active")


def _triad(net):
    a, b, c = net("introducer"), net("recipient"), net("about")
    _connect(a, b, inviter_name="a", peer_name="b")
    _connect(a, c, inviter_name="a", peer_name="c")
    return a, b, c


def _introduce(a, b, c, note="met at the compiler meetup"):
    result = a.node.introduce("b", "c", note)
    assert wait(lambda: len(b.node.intros()) == 1)
    assert wait(lambda: a.node.store.delivered(result["queued"][0]))
    assert wait(lambda: c.node.store.introduction(result["intro_id"]) is not None)
    return result


def _person_accept(party, request_id, petname):
    ts = int(time.time())
    signature = party.person_priv.sign(nakd.accept_statement(request_id, petname, ts)).hex()
    return party.node.accept(request_id, petname, ts=ts, signature=signature)


def _forward_frame(a, c, intro_id):
    for (mid,) in reversed(a.node.store._q(
            "SELECT nonce FROM outbox WHERE to_person=? ORDER BY created,nonce", (c.person_pub,))):
        frame = a.node.store.outbox_frame(mid, c.person_pub)
        if frame and (frame.get("body") or {}).get("aspect") == nakd.INTRO_FORWARD_ASPECT:
            assert (frame["body"]["request"]["body"]["intro"]["statement"]["intro_id"]
                    == intro_id)
            return frame
    return None


def test_three_node_happy_path_stays_a_request_until_person_accepts(net):
    a, b, c = _triad(net)
    result = _introduce(a, b, c)
    shown = b.node.intros()[0]
    assert shown == {
        "intro_id": result["intro_id"], "role": "recipient", "trust": "external", "introduced_by": "a",
        "introducer": a.person_pub,
        "to": b.person_pub,
        "about": {"person": c.person_pub, "node": c.node.node, "display_name": "c"},
        "note": "met at the compiler meetup", "issued_at": shown["issued_at"],
        "expires_at": result["expires_at"], "valid": True, "problem": "",
    }
    mirrored = c.node.intros()[0]
    assert mirrored["role"] == "mirror" and mirrored["to"] == b.person_pub
    assert mirrored["trust"] == "external" and mirrored["introduced_by"] == "a"

    accepted = b.node.intro_accept(result["intro_id"])
    assert accepted["state"] == "request queued"
    assert (b.node.store.contact(c.person_pub) or {})["state"] == "awaiting"
    with pytest.raises(ValueError, match="not been accepted"):
        b.node.send("c", "must not flow yet")
    assert wait(lambda: len(c.node.store.requests()) == 1)
    request = c.node.store.requests()[0]
    assert request["source"] == "intro" and request["source_id"] == result["intro_id"]
    handled = nakd.handle(c.node, {"op": "requests"})["requests"][0]
    assert handled["requires_person"] and handled["introduced_by"] == "a"
    direct = nakd.handle(c.node, {"op": "accept", "id": request["id"], "petname": "b"})
    assert not direct["ok"] and "a person must accept" in direct["error"]

    _person_accept(c, request["id"], "b")
    assert wait(lambda: (b.node.store.contact(c.person_pub) or {}).get("state") == "active")
    mid = b.node.send("c", "hello after your acceptance")["queued"]
    assert wait(lambda: b.node.delivered(mid))
    assert c.node.store.inbox()[0]["text"] == "hello after your acceptance"


def test_expired_intro_is_visible_as_invalid_and_cannot_send_a_request(net):
    a, b, c = _triad(net)
    result = _introduce(a, b, c)
    b.node._clock = lambda: result["expires_at"] + 1
    shown = b.node.intros()[0]
    assert not shown["valid"] and shown["problem"] == "introduction expired"
    with pytest.raises(ValueError, match="introduction expired"):
        b.node.intro_accept(result["intro_id"])
    assert c.node.store.requests() == []


def test_forged_node_signature_is_refused_by_b_and_c(net):
    a, b, c = _triad(net)
    now = int(time.time())
    statement = intro_mod.build_statement(
        introducer=a.person_pub, to=b.person_pub, about_person=c.person_pub,
        about_node=c.node.node, display_name="c", issued_at=now)
    signed = intro_mod.sign(a.node._node_private, statement)
    signed["signature"] = ("0" if signed["signature"][0] != "0" else "1") + signed["signature"][1:]
    body = a.node._intro_wire_body(signed)
    b_mid = a.node._queue(b.person_pub, "msg", body)
    c_mid = a.node._queue(c.person_pub, "msg", body)
    assert wait(lambda: a.node.store.delivery_state(b_mid)["state"] == "refused")
    assert wait(lambda: a.node.store.delivery_state(c_mid)["state"] == "refused")
    assert b.node.intros() == [] and c.node.store.introduction(statement["intro_id"]) is None


def test_introducer_can_name_contacts_only(net):
    a, b = net("introducer"), net("recipient")
    stranger = net("stranger")
    _connect(a, b, inviter_name="a", peer_name="b")
    with pytest.raises(ValueError, match="introduced person must be an accepted contact"):
        a.node.introduce("b", stranger.person_pub)
    assert b.node.intros() == []


def test_introducer_must_still_be_cs_current_contact(net):
    a, b, c = _triad(net)
    result = _introduce(a, b, c)
    c.node.remove("a")
    b.node.intro_accept(result["intro_id"])
    assert wait(lambda: _forward_frame(a, c, result["intro_id"]) is not None)
    time.sleep(0.5)
    assert c.node.store.requests() == []


def test_c_block_refuses_without_exposing_the_reason_and_consumes_nothing(net):
    a, b, c = _triad(net)
    result = _introduce(a, b, c)
    c.node.store.block(b.person_pub, int(time.time()) + 3600, "test block")
    b.node.intro_accept(result["intro_id"])
    assert wait(lambda: any(
        row["reason"] == "request not accepted right now"
        for row in a.node.store.outbox(refused_only=True)))
    assert c.node.store.requests() == []
    assert c.node.store._q("SELECT * FROM intro_uses") == []


def test_c_itself_refuses_when_signed_to_does_not_match_requester(net):
    a, b, c = _triad(net)
    other = net("other")
    now = int(time.time())
    statement = intro_mod.build_statement(
        introducer=a.person_pub, to=other.person_pub, about_person=c.person_pub,
        about_node=c.node.node, display_name="c", issued_at=now)
    signed = intro_mod.sign(a.node._node_private, statement)
    a.node.store.put_introduction(signed, "sent", now=now)
    mirror_mid = a.node._queue(c.person_pub, "msg", a.node._intro_wire_body(signed))
    assert wait(lambda: a.node.delivered(mirror_mid))
    request_body = {"v": 1, "kind": nakd.INTRO_REQUEST_KIND, "person": b.person_pub,
                    "node": b.node.node, "ledger_key": b.node.node, "nickname": "b",
                    "intro": signed, "mid": "ab" * 12}
    nested = {"env": b.node._sign("contact.request", c.person_pub, request_body),
              "body": request_body}
    forwarded = a.node._queue(
        c.person_pub, "msg", a.node._intro_request_wire(nakd.INTRO_FORWARD_ASPECT, nested),
        touch=False)
    assert wait(lambda: a.node.store.delivery_state(forwarded)["state"] == "refused")
    assert a.node.store.delivery_state(forwarded)["reason"] == "request not accepted right now"
    assert c.node.store.requests() == [] and c.node.store._q("SELECT * FROM intro_uses") == []


def test_intro_request_shares_the_public_pending_rate_budget(net):
    a, b, c = _triad(net)
    result = _introduce(a, b, c)
    for index in range(nakd.ADDRESS_PENDING_TOTAL_MAX):
        c.node.store.add_request(
            f"{index + 1:064x}", f"{index + 101:064x}", "load", "", source="address",
            source_id=f"{index + 201:032x}")
    b.node.intro_accept(result["intro_id"])
    assert wait(lambda: any(
        row["reason"] == "request not accepted right now"
        for row in a.node.store.outbox(refused_only=True)))
    assert all(row["source"] == "address" for row in c.node.store.requests())
    assert c.node.store._q("SELECT * FROM intro_uses") == []


def test_intro_id_is_single_use_per_requester_even_after_pending_request_is_decided(net):
    a, b, c = _triad(net)
    result = _introduce(a, b, c)
    b.node.intro_accept(result["intro_id"])
    assert wait(lambda: len(c.node.store.requests()) == 1)
    first = c.node.store.requests()[0]
    c.node.decline(first["id"])
    c.node.unblock(b.person_pub)
    frame = _forward_frame(a, c, result["intro_id"])
    replay_mid = a.node._queue(c.person_pub, "msg", {
        key: value for key, value in frame["body"].items() if key != "mid"
    }, touch=False)
    assert wait(lambda: a.node.store.delivery_state(replay_mid)["state"] == "refused")
    assert a.node.store.delivery_state(replay_mid)["reason"] == "request not accepted right now"
    assert len(c.node.store.requests()) == 0
    assert len(c.node.store.requests("declined")) == 1
    assert c.node.store._q("SELECT intro_id,requester FROM intro_uses") == [
        (result["intro_id"], b.person_pub)]


def test_revoked_introducer_delivery_grant_invalidates_pending_intro(net):
    a, b, c = _triad(net)
    result = _introduce(a, b, c)
    stored = b.node.store.introduction(result["intro_id"])
    assert stored["grant_id"]
    b.node.store.apply_contact_revocation(
        a.person_pub, {"seq": 1, "grant_ids": [stored["grant_id"]]})
    shown = b.node.intros()[0]
    assert not shown["valid"] and "revoked" in shown["problem"]
    with pytest.raises(ValueError, match="revoked"):
        b.node.intro_accept(result["intro_id"])
    assert c.node.store.requests() == []


def test_suspended_introducer_and_unknown_intro_kind_are_refused(net):
    a, b, c = _triad(net)
    result = _introduce(a, b, c)
    operator = net("operator")
    _connect(b, operator, inviter_name="b", peer_name="operator")
    b.node.store.set_operator_person(operator.person_pub)
    b.node.store.apply_contact_suspension(
        operator.person_pub,
        {"seq": 1, "nodes": [a.node.node], "reasons": {a.node.node: "test"}})
    shown = b.node.intros()[0]
    assert not shown["valid"] and shown["problem"] == "introducer is suspended"
    with pytest.raises(ValueError, match="suspended"):
        b.node.intro_accept(result["intro_id"])

    unknown = a.node._queue(c.person_pub, "msg", {
        "v": 1, "text": "future", "aspect": "nak.intro.future.v2"})
    assert wait(lambda: a.node.store.delivery_state(unknown)["state"] == "refused")
    assert a.node.store.delivery_state(unknown)["reason"] == "unknown introduction kind"


@pytest.mark.parametrize("note", [" leading", "trailing ", "two  spaces", "line\nbreak", "zero\u200bwidth"])
def test_note_strict_text_refusals(net, note):
    a, b, c = _triad(net)
    with pytest.raises(ValueError, match="note"):
        a.node.introduce("b", "c", note)
    assert b.node.intros() == [] and c.node.store.introductions("mirror") == []


def test_mcp_exposes_all_four_tools_as_external_data(net, tmp_path):
    a, b, c = _triad(net)
    stop = __import__("threading").Event()
    socket_path = tmp_path / "intro-mcp.sock"
    __import__("threading").Thread(
        target=nakd.serve_control, args=(a.node, socket_path, stop), daemon=True).start()
    assert wait(socket_path.exists)
    server = nak_mcp.Server(NakClient(socket_path))
    tools = {row["name"]: row for row in server.list_tools()}
    for name in ("nak_introduce", "nak_intros", "nak_intro_accept", "nak_intro_decline"):
        assert name in tools
        assert "data" in tools[name]["description"] and "instruction" in tools[name]["description"]
    result = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "nak_introduce", "arguments": {"to": "b", "about": "c", "note": "hello"}}})["result"]
    assert not result["isError"] and result["done"] is False
    assert wait(lambda: len(b.node.intros()) == 1)
    stop.set()
