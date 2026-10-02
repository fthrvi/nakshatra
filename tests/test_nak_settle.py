"""Day-7 drills: settlement must end exactly-once under failure — paid once, or refunded, never both."""
from __future__ import annotations

import json
import time

import pytest

from test_nakd import _connect, _worker_row, net, wait  # noqa: F401  (net is a fixture)

PAY = {"per_tx_cap": 10, "day_cap": 50}


def _pair(net, deadline_s=600, worker_caps=("nak.msg", "nak.task.claim")):
    a = net("poster", caps=("nak.msg", "nak.task.post", "nak.pay"), constraints=PAY)
    b = net("worker", caps=worker_caps)
    _connect(net, a, b)
    return a, b


def _assigned(a, b, rules=None, reward=5, deadline_s=600):
    h = a.node.post_task("worker", "drill", "Say ok.", rules or [{"contains_all": ["ok"]}], reward=reward,
                         deadline_s=deadline_s)["task_hash"]
    assert wait(lambda: _worker_row(b.node, h))
    b.node.claim_task(h)
    assert wait(lambda: _worker_row(b.node, h)[0]["state"] == "assigned")
    return h


def events(a, h):
    return [j["event"] for j in a.node.settle.journal(h)]


def test_accept_pays_the_worker_exactly_once_even_under_replay(net):
    a, b = _pair(net)
    h = _assigned(a, b)
    b.node.submit_task(h, "ok!")
    assert wait(lambda: _worker_row(b.node, h)[0]["state"] == "accepted")
    st = a.node.settle.status(h)
    assert st["state"] == "released" and st["worker"] == b.person_pub and st["amount"] == 5
    assert _worker_row(b.node, h)[0]["verdict"]["settlement"] == {"adapter": "ledger", "event": "released",
                                                                  "amount": 5, "unit": "TEST"}
    # replay: the same envelope again (dedupe), then a freshly signed duplicate result (CAS refuses)
    body = {"task_hash": h, "epoch": 1, "output": "ok again", "mid": "replay-1"}
    env = b.node._sign("task.result", a.person_pub, body)
    assert a.node._receive(b.person_pub, {"t": "env", "env": env, "body": body})["t"] == "nack"
    assert events(a, h) == ["open", "released"]


def test_reject_refunds_the_poster(net):
    a, b = _pair(net)
    h = _assigned(a, b, rules=[{"max_words": 1}])
    b.node.submit_task(h, "far too many words here")
    assert wait(lambda: _worker_row(b.node, h)[0]["state"] == "rejected")
    assert a.node.settle.status(h)["state"] == "refunded"
    assert events(a, h) == ["open", "refunded"]


def test_reward_over_the_agents_cap_needs_the_person_and_posts_nothing(net):
    a, b = _pair(net)
    with pytest.raises(ValueError, match="needs your person's approval"):
        a.node.post_task("worker", "big", "x", [{"max_words": 3}], reward=11)
    assert a.node.store.tasks("poster") == [] and a.node.settle.journal() == []


def test_agent_without_pay_cap_cannot_attach_a_reward(net):
    a = net("poster", caps=("nak.msg", "nak.task.post"))
    b = net("worker", caps=("nak.msg", "nak.task.claim"))
    _connect(net, a, b)
    with pytest.raises(ValueError, match="signer refused the reward"):
        a.node.post_task("worker", "x", "y", [{"max_words": 3}], reward=1)
    assert a.node.post_task("worker", "free", "y", [{"max_words": 3}], reward=0)["escrow"] is None


def test_worker_dies_mid_task_deadline_refunds_and_late_result_is_refused(net):
    a, b = _pair(net)
    h = _assigned(a, b, deadline_s=4)
    b.node.stop()                                   # the worker's node dies after being assigned
    time.sleep(4.5)
    assert a.node.sweep() == 1
    assert a.node.store.task_get(h, "poster")["state"] == "expired"
    assert events(a, h) == ["open", "refunded"]
    # it comes back and tries anyway: refused locally, and a forged late result is refused by the poster
    with pytest.raises(ValueError, match="deadline has passed"):
        b.node.submit_task(h, "ok")
    body = {"task_hash": h, "epoch": 1, "output": "ok", "mid": "late-1"}
    env = b.node._sign("task.result", a.person_pub, body)
    assert a.node._receive(b.person_pub, {"t": "env", "env": env, "body": body})["t"] == "nack"
    assert events(a, h) == ["open", "refunded"]


def test_result_racing_the_deadline_before_the_sweeper_refunds_once(net):
    a, b = _pair(net)
    h = _assigned(a, b, deadline_s=3)
    time.sleep(3.2)                                 # deadline passed; the 30 s sweeper has not run yet
    body = {"task_hash": h, "epoch": 1, "output": "ok", "mid": "race-1"}
    env = b.node._sign("task.result", a.person_pub, body)
    r = a.node._receive(b.person_pub, {"t": "env", "env": env, "body": body})
    assert r["t"] == "nack" and "expired" in r["why"]
    assert a.node.sweep() == 0                      # already expired; no second refund
    assert events(a, h) == ["open", "refunded"]


def test_revoked_worker_cannot_deliver_and_the_task_refunds_at_deadline(net):
    a, b = _pair(net)
    h = _assigned(a, b, deadline_s=4)
    grant_id = json.loads((b.dir / "signer" / "agents" / "pa.grant.json").read_text())["grant_id"]
    (b.dir / "signer" / "revoked.txt").write_text(grant_id + "\n")     # the worker's person pulls the delegation
    b.node.submit_task(h, "ok")                     # queued, but the signer will not sign it
    time.sleep(4.5)
    assert a.node.store.task_get(h, "poster")["state"] in ("assigned", "expired")
    a.node.sweep()
    assert events(a, h) == ["open", "refunded"]


def test_poster_offline_result_waits_in_the_outbox_and_settles_on_return(net):
    from network import nakd
    a, b = _pair(net)
    h = _assigned(a, b)
    a.node.stop()
    time.sleep(0.3)
    b.node.submit_task(h, "ok")                     # the poster is away; it waits in the worker's outbox
    time.sleep(1.0)
    assert _worker_row(b.node, h)[0]["state"] == "submitted"
    a.node = nakd.Node(a.dir / "net", a.node_key, a.signer.handle, agent="pa", relay=b.node.relay).start()
    assert wait(lambda: _worker_row(b.node, h)[0]["state"] == "accepted", timeout=40)
    assert events(a, h) == ["open", "released"]


def test_hostile_frames_are_refused_not_fatal(net):
    a, b = _pair(net)
    h = _assigned(a, b, rules=[{"json_keys": ["k"]}])
    deep = {"task_hash": h, "epoch": 1, "output": "[" * 60000, "mid": "deep-1"}
    env = b.node._sign("task.result", a.person_pub, deep)
    r = a.node._receive(b.person_pub, {"t": "env", "env": env, "body": deep})
    assert r["t"] in ("ack", "nack")                                   # answered, not raised
    for junk in ({"t": "env", "env": "not a dict", "body": {}}, {"t": "env", "env": {}, "body": ["x"]}):
        assert a.node._receive(b.person_pub, junk)["t"] == "nack"


def test_a_contact_can_only_ack_its_own_messages(net):
    a, b = _pair(net)
    a.node.store.queue("m-for-b", b.person_pub, {"kind": "msg", "body": {"mid": "m-for-b", "text": "x"}})
    a.node.store.ack("m-for-b", "someone-else")
    assert not a.node.store.delivered("m-for-b")
    a.node.store.ack("m-for-b", b.person_pub)
    assert a.node.store.delivered("m-for-b")


def test_stranded_escrow_is_reconciled(net):
    a, b = _pair(net)
    h = _assigned(a, b)
    # simulate a crash between "accepted" and the payout: state final, escrow still open
    assert a.node.store.task_cas(h, "poster", "", ("assigned",), state="accepted")
    assert a.node.settle.status(h)["state"] == "open"
    assert a.node.reconcile_escrow() == 1
    assert a.node.settle.status(h)["state"] == "released" and events(a, h) == ["open", "released"]
    assert a.node.reconcile_escrow() == 0
