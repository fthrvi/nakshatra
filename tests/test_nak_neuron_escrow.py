"""M2: fixed-price task holds on a shared, contract-shaped fake Neuron ledger."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import threading
import time
import urllib.error
import urllib.parse

import pytest

import nakshatra_auth
from network import settle as settle_mod
from network.settle import (HoldConflict, HoldNotFound, InsufficientFunds, NeuronEscrowAdapter,
                            SettlementError)
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
        self.fail_authorize = False
        self.refuse_authorize = 0        # HTTP status of a definitive refusal
        self.drop_response = False       # commit the hold, then lose the answer
        self.after_commit = None         # hook run after the hold commits (before the answer)
        self.lie_open_once = False       # one stale status read that still says open
        self.sweep_on_pay = False        # the ledger's own sweep refunds the hold as pay arrives
        self.status_override = None

    def fund(self, keyid, amount):
        self.balances[keyid] = self.balances.get(keyid, 0) + amount
        self.initial += amount
        self.check()

    def check(self):
        held = sum(row["amount"] for row in self.holds.values() if row["state"] == "open")
        assert sum(self.balances.values()) + held == self.initial

    def authorize(self, payer, task_hash, worker, amount, deadline):
        if self.fail_authorize:
            raise SettlementError("ledger down")
        if self.refuse_authorize:
            error = SettlementError("refused")
            error.status = self.refuse_authorize
            raise error
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
        if self.after_commit:
            self.after_commit()
        if self.drop_response:
            self.drop_response = False
            raise SettlementError("Neuron ledger /spend/task/authorize unavailable (TimeoutError)")
        return copy.deepcopy(row)

    def status(self, caller, hold_id):
        row = copy.deepcopy(self.holds[hold_id])
        if caller not in {row["payer_account"], row["worker_keyid"]}:
            raise SettlementError("task hold not found")
        if self.lie_open_once and caller == row["payer_account"]:
            self.lie_open_once = False
            row["state"] = "open"
        if self.status_override and caller == row["worker_keyid"]:
            row.update(self.status_override)
        return row

    def pay(self, payer, hold_id):
        if self.fail_pay:
            raise SettlementError("ledger down")
        row = self.holds[hold_id]
        if row["payer_account"] != payer:
            raise SettlementError("task hold not found")
        if self.sweep_on_pay and row["state"] == "open":
            row["state"] = "refunded"
            self.balances[payer] += row["amount"]
            self.transitions.append((hold_id, "refunded"))
        if row["state"] == "refunded":
            error = SettlementError("task hold is refunded")
            error.status = 409
            raise error
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


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class ReplayLedgerHTTP:
    """HTTP face of FakeTaskLedger that behaves like the real ledger_auth: it verifies the Ed25519
    signature and rejects a signature it has seen before (sha256(signature) is the nonce) with 401."""

    def __init__(self, ledger):
        self.ledger = ledger
        self.seen = set()
        self.timestamps = []
        self.lock = threading.Lock()

    def _fail(self, request, code):
        return urllib.error.HTTPError(request.full_url, code, "refused", {}, io.BytesIO(b"{}"))

    def __call__(self, request, timeout=None, expected_spki_hash=None):
        match = re.fullmatch(r'\S+ keyid="([0-9a-f]{64})",sig="([^"]+)",ts="(\d+)"',
                             request.headers["Authorization"])
        parts = urllib.parse.urlsplit(request.full_url)
        path = parts.path + ("?" + parts.query if parts.query else "")
        body = request.data or b""
        keyid, sig, ts = match.group(1), match.group(2), int(match.group(3))
        if not nakshatra_auth.verify_request(keyid, request.method, path, body, ts, sig):
            raise self._fail(request, 401)
        if abs(time.time() - ts) > 60:                       # the ledger's +/-60 s window
            raise self._fail(request, 401)
        nonce = hashlib.sha256(sig.encode()).hexdigest()
        with self.lock:
            if nonce in self.seen:
                raise self._fail(request, 401)
            self.seen.add(nonce)
            self.timestamps.append(ts)
        ledger = self.ledger
        try:
            if parts.path == "/spend/task/authorize":
                b = json.loads(body)
                row = ledger.authorize(keyid, b["task_hash"], b["worker_keyid"], b["amount"], b["deadline"])
            elif parts.path == "/spend/task/pay":
                row = ledger.pay(keyid, json.loads(body)["hold_id"])
            elif parts.path == "/spend/task/refund":
                b = json.loads(body)
                row = ledger.refund(keyid, b["hold_id"], b["reason"])
            else:
                hold_id = urllib.parse.parse_qs(parts.query)["hold_id"][0]
                return _Response(json.dumps(ledger.status(keyid, hold_id)).encode())
        except SettlementError as e:
            text = str(e)
            raise self._fail(request, 402 if "insufficient" in text else 404 if "not found" in text
                             else 409)
        row = {k: row[k] for k in ("v", "hold_id", "task_hash", "worker_keyid", "amount",
                                   "deadline", "state")}
        return _Response(json.dumps(row).encode())


def _real_adapter(party, http):
    return NeuronEscrowAdapter("https://ledger.test", "ab" * 32, party.node_key, party.node.node,
                               opener=http)


def _pair(net, *, paid_card=True, suffix="", real=False):
    poster = net("poster" + suffix, caps=("nak.msg", "nak.task.post", "nak.pay"), constraints=PAY)
    worker = net("worker" + suffix, caps=("nak.msg", "nak.task.claim"))
    ledger = FakeTaskLedger()
    poster.ledger = ledger
    ledger.fund(poster.node.node, 100)
    if real:
        http = ReplayLedgerHTTP(ledger)
        ledger.http = http
        poster.node.settle = _real_adapter(poster, http)
        worker.node.settle = _real_adapter(worker, http)
    else:
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
    assert result["escrow"] is None and not poster.ledger.holds
    assert wait(lambda: _worker_row(worker.node, task_hash))
    worker.node.claim_task(task_hash)
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["state"] == "assigned")
    # the worker verifies the hold right after it records the assignment; wait for that verdict
    assert wait(lambda: (_worker_row(worker.node, task_hash)[0]["funding_state"] == "funded"
                         or _worker_row(worker.node, task_hash)[0]["funding_reason"]))
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


def _paid_forgery(poster, worker, keyid, *, paid=True):
    """Store a card for `worker` on the poster directly (as if it had been accepted)."""
    previous = poster.node.store.agent_card(worker.person_pub) or {}
    card = {"display_name": "Worker", "offers": "paid tasks", "intents": [], "paid": paid,
            "price_hint": "", "models": [], "supports": [], "ledger_keyid": keyid,
            "updated_at": int(previous.get("updated_at", 0)) + 100}
    assert poster.node.store.set_agent_card(worker.person_pub, card)


def _post_and_claim(poster, worker, reward=5):
    task_hash = poster.node.post_task(
        worker.dir.name, "drill", "Say ok.", [{"contains_all": ["ok"]}], reward=reward)["task_hash"]
    assert wait(lambda: _worker_row(worker.node, task_hash))
    worker.node.claim_task(task_hash)
    return task_hash


def test_adapter_never_signs_the_same_second_twice():
    ledger = FakeTaskLedger()
    http = ReplayLedgerHTTP(ledger)
    # the fake ledger verifies signatures, so sign with a real key
    from cryptography.hazmat.primitives.asymmetric import ed25519
    key = ed25519.Ed25519PrivateKey.generate()
    keyid = key.public_key().public_bytes_raw().hex()
    ledger.fund(keyid, 100)
    adapter = NeuronEscrowAdapter("https://ledger.test", "ab" * 32, key.private_bytes_raw(), keyid,
                                  opener=http)
    th, worker = "12" * 32, "34" * 32
    auth = {"task_hash": th, "amount": 7, "mint": "credits"}
    adapter.open(th, "p", 7, "credits", int(time.time()) + 600, auth, worker_keyid=worker)
    for _ in range(6):                                   # rapid identical status calls
        assert adapter.status(th)["state"] == "open"
    assert adapter.release(th, "w")["event"] == "released"
    assert adapter.status(th)["state"] == "released"
    assert http.timestamps == sorted(set(http.timestamps)) and len(http.timestamps) == 9


def test_two_adapters_same_key_same_second_retry_once_on_401(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    monkeypatch.setattr(settle_mod.time, "time", lambda: 1_800_000_000.0)
    ledger = FakeTaskLedger()
    key = ed25519.Ed25519PrivateKey.generate()
    keyid = key.public_key().public_bytes_raw().hex()
    ledger.fund(keyid, 100)
    http = ReplayLedgerHTTP(ledger)
    th = "12" * 32
    auth = {"task_hash": th, "amount": 7, "mint": "credits"}
    first = NeuronEscrowAdapter("https://ledger.test", "ab" * 32, key.private_bytes_raw(), keyid, opener=http)
    first.open(th, "p", 7, "credits", 1_800_000_900, auth, worker_keyid="34" * 32)
    other = NeuronEscrowAdapter("https://ledger.test", "ab" * 32, key.private_bytes_raw(), keyid, opener=http)
    hold = next(iter(ledger.holds))
    assert first.status(th)["state"] == "open"           # first adapter: ts 1_800_000_001
    assert other.status(th, hold_id=hold)["state"] == "open"   # same ts as the first's first call -> 401 -> retry
    assert len(http.timestamps) == 3


def test_adapter_gives_up_after_one_401_retry():
    def always_401(request, **_kwargs):
        raise urllib.error.HTTPError(request.full_url, 401, "no", {}, io.BytesIO(b"{}"))
    calls = []
    adapter = NeuronEscrowAdapter("https://ledger.test", "ab" * 32, b"k" * 32, "56" * 32,
                                  opener=lambda r, **k: (calls.append(1), always_401(r))[1])
    with pytest.raises(SettlementError, match="HTTP 401"):
        adapter.status("12" * 32, hold_id="h")
    assert len(calls) == 2


@pytest.mark.parametrize("code,exc", [(402, InsufficientFunds), (404, HoldNotFound), (409, HoldConflict)])
def test_adapter_maps_ledger_statuses_to_distinct_errors(code, exc):
    def refuse(request, **_kwargs):
        raise urllib.error.HTTPError(request.full_url, code, "no", {}, io.BytesIO(b"{}"))
    adapter = NeuronEscrowAdapter("https://ledger.test", "ab" * 32, b"k" * 32, "56" * 32, opener=refuse)
    with pytest.raises(exc) as caught:
        adapter.refund("12" * 32, "why", hold_id="h")
    assert caught.value.status == code and isinstance(caught.value, SettlementError)


def test_verify_at_assign_then_immediate_submit_and_status_against_replay_ledger(net):
    poster, worker, ledger = _pair(net, real=True)
    task_hash = _assign(poster, worker)
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["funding_state"] == "funded")
    worker.node.submit_task(task_hash, "ok")             # immediately: its own status call, same second
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["funding_state"] == "paid")
    assert poster.node.reconcile_escrow() == 0
    assert poster.node.escrow_status(task_hash)["funding"] == "paid"
    assert poster.node.escrow_status(task_hash)["funding"] == "paid"
    assert [event for _hold, event in ledger.transitions] == ["open", "paid"]
    assert ledger.http.timestamps


def test_forged_ledger_keyid_card_is_refused_and_stores_nothing(net):
    poster, worker, ledger = _pair(net, paid_card=False)
    worker.node.card_set("Worker", "paid tasks", paid=True)
    assert wait(lambda: poster.node.store.agent_card(worker.person_pub) is not None)
    honest = poster.node.store.agent_card(worker.person_pub)
    assert honest["ledger_keyid"] == worker.node.node
    # now the worker's agent publishes a newer card naming a different ledger key
    forged = dict(worker.node.store.agent_card(worker.node.person), ledger_keyid="ee" * 32)
    forged["updated_at"] += 50
    worker.node.store.set_agent_card(worker.node.person, forged)
    worker.node._send_card_if_set(poster.person_pub)
    assert wait(lambda: any(row["refused"] for row in worker.node.store.outbox()))
    assert poster.node.store.agent_card(worker.person_pub)["ledger_keyid"] == worker.node.node


def test_assign_refuses_a_card_keyid_that_is_not_the_contacts_node_and_opens_no_hold(net):
    poster, worker, ledger = _pair(net)
    _paid_forgery(poster, worker, "ee" * 32)
    task_hash = _post_and_claim(poster, worker)
    assert wait(lambda: any(row["refused"] for row in worker.node.store.outbox()))
    assert poster.node.store.task_get(task_hash, "poster")["state"] == "posted"
    assert ledger.holds == {}


def test_assign_requires_the_card_to_say_paid_even_with_a_ledger_key(net):
    poster, worker, ledger = _pair(net)
    _paid_forgery(poster, worker, worker.node.node, paid=False)
    task_hash = _post_and_claim(poster, worker)
    assert wait(lambda: any(row["refused"] for row in worker.node.store.outbox()))
    assert poster.node.store.task_get(task_hash, "poster")["state"] == "posted"
    assert ledger.holds == {}


def test_loser_of_a_claim_race_gets_the_clean_reject_and_no_hold(net):
    poster = net("poster", caps=("nak.msg", "nak.task.post", "nak.pay"), constraints=PAY)
    w1 = net("w1", caps=("nak.msg", "nak.task.claim"))
    w2 = net("w2", caps=("nak.msg", "nak.task.claim"))
    ledger = FakeTaskLedger()
    ledger.fund(poster.node.node, 100)
    for party in (poster, w1, w2):
        party.node.settle = FakeNeuronAdapter(ledger, party.node.node)
    for worker in (w1, w2):
        _connect(net, poster, worker)
        worker.node.card_set("W", "paid tasks", paid=True)
        assert wait(lambda w=worker: (poster.node.store.agent_card(w.person_pub) or {}).get("paid") is True)
    task_hash = poster.node.post_task(["w1", "w2"], "drill", "Say ok.", [{"contains_all": ["ok"]}],
                                      reward=5)["task_hash"]
    assert wait(lambda: _worker_row(w1.node, task_hash) and _worker_row(w2.node, task_hash))
    w1.node.claim_task(task_hash)
    assert wait(lambda: _worker_row(w1.node, task_hash)[0]["state"] == "assigned")
    w2.node.claim_task(task_hash)
    assert wait(lambda: _worker_row(w2.node, task_hash)[0]["state"] != "claimed")
    assert not any(row["refused"] for row in w2.node.store.outbox())     # a reject, not a nack
    assert len(ledger.holds) == 1 and next(iter(ledger.holds.values()))["worker_keyid"] == w1.node.node
    assert poster.node.store.task_get(task_hash, "poster")["assignee"] == w1.person_pub


def test_definitive_refusal_puts_the_task_back_and_the_worker_can_claim_again(net):
    poster, worker, ledger = _pair(net)
    ledger.refuse_authorize = 402
    task_hash = _post_and_claim(poster, worker)
    assert wait(lambda: any(row["refused"] for row in worker.node.store.outbox()))
    row = poster.node.store.task_get(task_hash, "poster")
    assert row["state"] == "posted" and not row["assignee"] and not row["hold_id"]
    assert ledger.holds == {}
    # the nack carried the reopen marker: the worker's row went back to `offered` by itself
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["state"] == "offered")
    ledger.refuse_authorize = 0
    worker.node.claim_task(task_hash)
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["state"] == "assigned")
    row = poster.node.store.task_get(task_hash, "poster")
    assert row["hold_id"] and row["funding_state"] == "funded" and len(ledger.holds) == 1


def test_ambiguous_authorize_leaves_the_task_pending_and_reconcile_adopts_the_same_hold(net):
    poster, worker, ledger = _pair(net)
    ledger.drop_response = True                          # hold commits, the answer is lost
    task_hash = _post_and_claim(poster, worker)
    assert wait(lambda: len(ledger.holds) == 1)
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["funding_reason"].startswith("escrow:"))
    row = poster.node.store.task_get(task_hash, "poster")
    assert row["state"] == "assigned" and row["funding_state"] == "pending" and not row["hold_id"]
    poster.node.reconcile_escrow()                       # re-runs the idempotent authorize
    row = poster.node.store.task_get(task_hash, "poster")
    assert row["hold_id"] == next(iter(ledger.holds)) and row["funding_state"] == "funded"
    assert len(ledger.holds) == 1
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["state"] == "assigned")
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["funding_state"] == "funded")


def test_ambiguous_authorize_then_expiry_refunds_the_unrecorded_hold(net):
    poster, worker, ledger = _pair(net)
    ledger.drop_response = True
    task_hash = poster.node.post_task(worker.dir.name, "drill", "Say ok.", [{"contains_all": ["ok"]}],
                                      reward=5, deadline_s=2)["task_hash"]
    assert wait(lambda: _worker_row(worker.node, task_hash))
    worker.node.claim_task(task_hash)
    assert wait(lambda: len(ledger.holds) == 1)
    time.sleep(2.2)
    poster.node.sweep()
    poster.node.sweep()
    assert poster.node.store.task_get(task_hash, "poster")["state"] == "expired"
    assert [event for _hold, event in ledger.transitions] == ["open", "refunded"]
    assert ledger.balances[poster.node.node] == 100


def test_hold_opened_for_a_task_that_expired_meanwhile_is_refunded(net):
    poster, worker, ledger = _pair(net)

    def expire_now():
        poster.node.store.task_cas(task_hash, "poster", "", ("assigned",), state="expired")
    task_hash = poster.node.post_task(worker.dir.name, "drill", "Say ok.", [{"contains_all": ["ok"]}],
                                      reward=5)["task_hash"]
    assert wait(lambda: _worker_row(worker.node, task_hash))
    ledger.after_commit = expire_now
    worker.node.claim_task(task_hash)
    assert wait(lambda: [e for _h, e in ledger.transitions] == ["open", "refunded"])
    assert poster.node.store.task_get(task_hash, "poster")["hold_id"] is None
    assert ledger.balances[poster.node.node] == 100


def test_reconcile_adopts_the_ledgers_state_when_release_is_refused(net):
    poster, worker, ledger = _pair(net)
    task_hash = _assign(poster, worker)
    ledger.sweep_on_pay = True                           # the ledger's sweep refunds it as pay arrives
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["state"] == "accepted")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["funding_state"] == "pending")
    ledger.lie_open_once = True       # reconcile reads a stale `open`, tries release, gets 409
    assert poster.node.reconcile_escrow() == 1
    assert poster.node.store.task_get(task_hash, "poster")["funding_state"] == "refunded"
    assert [e for _h, e in ledger.transitions] == ["open", "refunded"]
    assert poster.node.reconcile_escrow() == 0


def _window_adapter(monkeypatch, now=1_800_000_000.0):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    clock = [now]
    monkeypatch.setattr(settle_mod.time, "time", lambda: clock[0])
    key = ed25519.Ed25519PrivateKey.generate()
    keyid = key.public_key().public_bytes_raw().hex()
    ledger = FakeTaskLedger()
    ledger.fund(keyid, 100)
    http = ReplayLedgerHTTP(ledger)
    adapter = NeuronEscrowAdapter("https://ledger.test", "ab" * 32, key.private_bytes_raw(), keyid,
                                  opener=http)
    th = "12" * 32
    adapter.open(th, "p", 7, "credits", int(now) + 600,
                 {"task_hash": th, "amount": 7, "mint": "credits"}, worker_keyid="34" * 32)
    return adapter, http, clock, th


def test_signing_backlog_is_refused_locally_and_never_locks_the_ledger_out(monkeypatch):
    adapter, http, clock, th = _window_adapter(monkeypatch)
    sent_before = len(http.timestamps)
    ok = busy = 0
    for _ in range(150):                                 # 150 calls inside one second
        try:
            adapter.status(th)
            ok += 1
        except settle_mod.LedgerBusy:
            busy += 1
    assert ok == settle_mod.MAX_TS_LEAD_S and busy == 150 - ok   # open() took the first slot
    assert len(http.timestamps) - sent_before == ok                      # busy calls sent nothing
    assert max(http.timestamps) <= clock[0] + settle_mod.MAX_TS_LEAD_S   # never ran ahead of the window
    started = time.perf_counter()
    with pytest.raises(settle_mod.LedgerBusy):
        adapter.status(th)
    assert time.perf_counter() - started < 0.05
    clock[0] += 40                                       # the clock catches up: calls work again
    assert adapter.status(th)["state"] == "open"
    assert adapter.release(th, "w")["event"] == "released"


def test_recovery_loop_over_many_stranded_tasks_defers_instead_of_locking_out(monkeypatch):
    adapter, http, clock, th = _window_adapter(monkeypatch)
    served = deferred = 0
    for _ in range(40):                                  # reconcile: status (+ settle) per task
        for _call in range(3):
            try:
                adapter.status(th)
                served += 1
            except settle_mod.LedgerBusy:
                deferred += 1
    assert served > 0 and deferred > 0
    clock[0] += 61                                       # next sweep: everything works, no 401 residue
    for _ in range(10):
        assert adapter.status(th)["state"] == "open"


def test_a_401_is_not_retried_when_the_timestamp_is_already_ahead_of_the_clock(monkeypatch):
    adapter, http, clock, th = _window_adapter(monkeypatch)
    for _ in range(10):
        adapter.status(th)                               # push _last_ts ahead of the clock
    sent = []
    adapter._opener = lambda request, **k: (sent.append(1), (_ for _ in ()).throw(
        urllib.error.HTTPError(request.full_url, 401, "no", {}, io.BytesIO(b"{}"))))[1]
    with pytest.raises(SettlementError, match="HTTP 401"):
        adapter.status(th)
    assert len(sent) == 1


def test_worker_refuses_a_hold_that_expires_before_the_task_deadline(net):
    poster, worker, ledger = _pair(net)
    ledger.status_override = {"deadline": int(time.time()) + 30}
    task_hash = _assign(poster, worker)
    row = _worker_row(worker.node, task_hash)[0]
    assert row["funding_state"] == "unfunded" and "deadline" in row["funding_reason"]


def test_reconcile_adopts_the_ledgers_terminal_state_on_409(net):
    poster, worker, ledger = _pair(net)
    task_hash = _assign(poster, worker)
    hold = next(iter(ledger.holds.values()))
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["funding_state"] == "paid")
    # strand it: the ledger swept the hold to refunded behind the poster's back, poster still pending
    poster.node.store.task_cas(task_hash, "poster", "", ("accepted",), funding_state="pending",
                               funding_reason="x")
    hold["state"] = "refunded"
    ledger.balances[poster.node.node] += hold["amount"]
    ledger.balances[worker.node.node] -= hold["amount"]
    assert poster.node.reconcile_escrow() == 1
    assert poster.node.store.task_get(task_hash, "poster")["funding_state"] == "refunded"
    assert poster.node.reconcile_escrow() == 0
