"""M2: fixed-price task holds on a shared, contract-shaped fake Neuron ledger."""
from __future__ import annotations

import copy
import io
import json
import time
import urllib.parse

import pytest

from network.settle import NeuronEscrowAdapter, SettlementError
from test_nakd import _connect, _worker_row, net, wait  # noqa: F401 - net is a fixture


PAY = {"per_tx_cap": 10, "day_cap": 50}


class FakeTaskLedger:
    """Injected ledger client with the M2 idempotency, CAS, and conservation rules."""

    def __init__(self):
        self.balances = {}
        self.holds = {}
        self.nonces = {}
        self.initial = 0
        self.transitions = []
        self.fail_pay = False
        self.status_override = None

    def fund(self, keyid, amount):
        self.balances[keyid] = self.balances.get(keyid, 0) + amount
        self.initial += amount
        self.check()

    def check(self):
        held = sum(row["amount"] for row in self.holds.values() if row["state"] == "open")
        assert sum(self.balances.values()) + held == self.initial

    def authorize(self, payer, task_hash, worker, amount, deadline):
        terms = (task_hash, worker, amount, deadline)
        prior = self.nonces.get((payer, task_hash))
        if prior:
            hold = self.holds[prior]
            if (hold["task_hash"], hold["worker_keyid"], hold["amount"], hold["deadline"]) != terms:
                raise SettlementError("idempotency conflict")
            return copy.deepcopy(hold)
        if self.balances.get(payer, 0) < amount:
            raise SettlementError("insufficient credits")
        hold_id = f"hold-{len(self.holds) + 1}"
        self.balances[payer] -= amount
        row = {"v": 1, "hold_id": hold_id, "task_hash": task_hash,
               "payer_account": payer, "worker_keyid": worker, "amount": amount,
               "deadline": deadline, "state": "open", "server_time": int(time.time())}
        self.holds[hold_id] = row
        self.nonces[(payer, task_hash)] = hold_id
        self.transitions.append((hold_id, "open"))
        self.check()
        return copy.deepcopy(row)

    def status(self, caller, hold_id):
        row = copy.deepcopy(self.holds[hold_id])
        if caller not in {row["payer_account"], row["worker_keyid"]}:
            raise SettlementError("task hold not found")
        if self.status_override and caller == row["worker_keyid"]:
            row.update(self.status_override)
        return row

    def pay(self, payer, hold_id):
        if self.fail_pay:
            raise SettlementError("ledger down")
        row = self.holds[hold_id]
        if row["payer_account"] != payer:
            raise SettlementError("task hold not found")
        if row["state"] == "refunded":
            raise SettlementError("task hold is refunded")
        if row["state"] == "open":
            row["state"] = "paid"
            self.balances[row["worker_keyid"]] = self.balances.get(row["worker_keyid"], 0) + row["amount"]
            self.transitions.append((hold_id, "paid"))
        self.check()
        return copy.deepcopy(row)

    def refund(self, payer, hold_id, _reason):
        row = self.holds[hold_id]
        if row["payer_account"] != payer:
            raise SettlementError("task hold not found")
        if row["state"] == "paid":
            raise SettlementError("task hold is paid")
        if row["state"] == "open":
            row["state"] = "refunded"
            self.balances[payer] = self.balances.get(payer, 0) + row["amount"]
            self.transitions.append((hold_id, "refunded"))
        self.check()
        return copy.deepcopy(row)


class FakeNeuronAdapter:
    name = "neuron"
    open_on_assign = True

    def __init__(self, ledger, keyid):
        self.ledger, self.keyid = ledger, keyid

    @staticmethod
    def _receipt(row, event):
        return {"adapter": "neuron", "event": event, "task_hash": row["task_hash"],
                "hold_id": row["hold_id"], "amount": row["amount"], "unit": "credits"}

    def open(self, task_hash, _poster, amount, unit, deadline, authorisation, *, worker_keyid=""):
        assert unit == "credits"
        assert authorisation["task_hash"] == task_hash and authorisation["amount"] == amount
        return self._receipt(
            self.ledger.authorize(self.keyid, task_hash, worker_keyid, amount, deadline), "open")

    def status(self, task_hash, *, hold_id=""):
        row = self.ledger.status(self.keyid, hold_id)
        row["ledger_state"] = row["state"]
        row["state"] = {"open": "open", "paid": "released", "refunded": "refunded"}[row["state"]]
        row["unit"] = "credits"
        return row

    def release(self, task_hash, _worker, *, hold_id=""):
        row = self.ledger.pay(self.keyid, hold_id)
        assert row["task_hash"] == task_hash
        return self._receipt(row, "released")

    def refund(self, task_hash, reason, *, hold_id=""):
        row = self.ledger.refund(self.keyid, hold_id, reason)
        assert row["task_hash"] == task_hash
        return self._receipt(row, "refunded")

    def journal(self, _task_hash=None):
        return []


def _pair(net, *, paid_card=True, suffix=""):
    poster = net("poster" + suffix, caps=("nak.msg", "nak.task.post", "nak.pay"), constraints=PAY)
    worker = net("worker" + suffix, caps=("nak.msg", "nak.task.claim"))
    ledger = FakeTaskLedger()
    ledger.fund(poster.node.node, 100)
    poster.node.settle = FakeNeuronAdapter(ledger, poster.node.node)
    worker.node.settle = FakeNeuronAdapter(ledger, worker.node.node)
    _connect(net, poster, worker)
    if paid_card:
        worker.node.card_set("Worker", "paid tasks", paid=True)
        assert wait(lambda: (poster.node.store.agent_card(worker.person_pub) or {}).get("paid") is True)
        assert poster.node.store.agent_card(worker.person_pub)["ledger_keyid"] == worker.node.node
    return poster, worker, ledger


def _assign(poster, worker, reward=5, deadline_s=600):
    result = poster.node.post_task(
        worker.dir.name, "drill", "Say ok.", [{"contains_all": ["ok"]}],
        reward=reward, deadline_s=deadline_s)
    task_hash = result["task_hash"]
    assert result["escrow"] is None and not poster.node.settle.ledger.holds
    assert wait(lambda: _worker_row(worker.node, task_hash))
    worker.node.claim_task(task_hash)
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["state"] == "assigned")
    return task_hash


def test_paid_card_requires_neuron_and_uses_the_nodes_own_key(net):
    legacy = net("legacy-card")
    with pytest.raises(ValueError, match="NAKSHATRA_ESCROW=neuron"):
        legacy.node.card_set("Legacy", "tasks", paid=True)
    ledger = FakeTaskLedger()
    legacy.node.settle = FakeNeuronAdapter(ledger, legacy.node.node)
    card = legacy.node.card_set("Paid", "tasks", paid=True)["card"]
    assert card["paid"] is True and card["ledger_keyid"] == legacy.node.node


def test_two_node_accept_pays_exactly_once_and_worker_sees_funded(net):
    poster, worker, ledger = _pair(net)
    task_hash = _assign(poster, worker)
    hold = next(iter(ledger.holds.values()))
    assert hold["state"] == "open" and hold["worker_keyid"] == worker.node.node
    assert _worker_row(worker.node, task_hash)[0]["funding_state"] == "funded"
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["funding_state"] == "paid")
    assert ledger.balances[worker.node.node] == 5
    assert poster.node.reconcile_escrow() == 0
    assert [event for _hold, event in ledger.transitions] == ["open", "paid"]


def test_reject_and_expiry_refund_once(net):
    poster, worker, ledger = _pair(net)
    task_hash = _assign(poster, worker)
    worker.node.submit_task(task_hash, "not acceptable")
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["funding_state"] == "refunded")
    assert ledger.balances[poster.node.node] == 100
    assert [event for _hold, event in ledger.transitions] == ["open", "refunded"]

    poster2, worker2, ledger2 = _pair(net, suffix="-expiry")
    expiring = _assign(poster2, worker2, deadline_s=2)
    time.sleep(2.2)
    assert poster2.node.sweep() == 1
    assert wait(lambda: _worker_row(worker2.node, expiring)[0]["funding_state"] == "refunded")
    assert [event for _hold, event in ledger2.transitions] == ["open", "refunded"]


def test_crash_window_and_ledger_down_reconcile_exactly_once(net):
    poster, worker, ledger = _pair(net)
    task_hash = _assign(poster, worker)
    ledger.fail_pay = True
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["funding_state"] == "pending")
    assert ledger.holds[next(iter(ledger.holds))]["state"] == "open"
    ledger.fail_pay = False
    assert poster.node.reconcile_escrow() == 1
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["funding_state"] == "paid")
    assert poster.node.reconcile_escrow() == 0
    assert [event for _hold, event in ledger.transitions] == ["open", "paid"]


@pytest.mark.parametrize("override,reason", [
    ({"amount": 6}, "amount"),
    ({"worker_keyid": "ef" * 32}, "another worker"),
    ({"task_hash": "ab" * 32}, "another task"),
    ({"state": "refunded"}, "refunded"),
])
def test_worker_refuses_mismatched_or_refunded_hold(net, override, reason):
    poster, worker, ledger = _pair(net)
    ledger.status_override = override
    task_hash = _assign(poster, worker)
    row = _worker_row(worker.node, task_hash)[0]
    assert row["funding_state"] == "unfunded" and reason in row["funding_reason"]
    with pytest.raises(ValueError, match="unfunded"):
        worker.node.submit_task(task_hash, "ok")


def test_unpaid_contact_cannot_be_assigned_and_over_cap_opens_no_hold(net):
    poster, worker, ledger = _pair(net, paid_card=False)
    result = poster.node.post_task("worker", "paid", "Do it.", [{"max_words": 2}], reward=5)
    task_hash = result["task_hash"]
    assert wait(lambda: _worker_row(worker.node, task_hash))
    worker.node.claim_task(task_hash)
    assert wait(lambda: any(row["refused"] for row in worker.node.store.outbox()))
    assert poster.node.store.task_get(task_hash, "poster")["state"] == "posted"
    assert ledger.holds == {}
    with pytest.raises(ValueError, match="needs your person's approval"):
        poster.node.post_task("worker", "too much", "Do it.", [{"max_words": 2}], reward=11)
    assert ledger.holds == {}


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_real_neuron_adapter_uses_contract_routes_and_signed_query_path():
    task_hash, worker = "12" * 32, "34" * 32
    hold = {"v": 1, "hold_id": "h-1", "task_hash": task_hash, "worker_keyid": worker,
            "amount": 7, "deadline": 2_000_000_000, "state": "open"}
    calls = []

    def opener(request, **kwargs):
        calls.append((request.method, request.full_url, request.data, request.headers, kwargs))
        if request.method == "POST" and request.full_url.endswith("/spend/task/pay"):
            value = dict(hold, state="paid")
        elif request.method == "GET":
            assert urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query) == {"hold_id": ["h-1"]}
            value = dict(hold, payer_account="payer", server_time=1)
        else:
            value = hold
        return _Response(json.dumps(value).encode())

    adapter = NeuronEscrowAdapter(
        "https://ledger.test", "ab" * 32, b"k" * 32, "56" * 32, opener=opener)
    auth = {"task_hash": task_hash, "amount": 7, "mint": "credits"}
    opened = adapter.open(task_hash, "poster", 7, "credits", 2_000_000_000, auth,
                          worker_keyid=worker)
    assert opened["hold_id"] == "h-1" and adapter.status(task_hash)["state"] == "open"
    assert adapter.release(task_hash, "worker")["event"] == "released"
    assert [call[0] for call in calls] == ["POST", "GET", "POST"]
    assert all(call[3]["Authorization"].startswith("Sthambha-Ed25519 ") for call in calls)
    assert all(call[4]["expected_spki_hash"] == "ab" * 32 for call in calls)


def test_real_neuron_adapter_refuses_non_https_unpinned_or_fail_open():
    args = ("http://ledger.test", "ab" * 32, b"k" * 32, "56" * 32)
    with pytest.raises(SettlementError, match="HTTPS"):
        NeuronEscrowAdapter(*args)
    with pytest.raises(SettlementError, match="SPKI"):
        NeuronEscrowAdapter("https://ledger.test", "", b"k" * 32, "56" * 32)
    with pytest.raises(SettlementError, match="enforce"):
        NeuronEscrowAdapter("https://ledger.test", "ab" * 32, b"k" * 32, "56" * 32,
                            mode="fail-open")
