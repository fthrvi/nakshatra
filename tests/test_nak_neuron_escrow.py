"""M2: fixed-price task holds on a shared, contract-shaped fake Neuron ledger."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.parse

import pytest

import nakshatra_auth
from cryptography.hazmat.primitives.asymmetric import ed25519
from network import nakd, settle as settle_mod, tasks as task_mod, verdict as verdict_mod
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
        self.fail_submit = False
        self.submit_calls = 0
        self.fail_receive = False
        self.receive_calls = 0
        self.fail_authorize = False
        self.refuse_authorize = 0        # HTTP status of a definitive refusal
        self.drop_response = False       # commit the hold, then lose the answer
        self.after_commit = None         # hook run after the hold commits (before the answer)
        self.authorize_calls = 0
        self.loose_replay = False        # replay returns the stored hold even if the terms differ
        self.lie_open_once = False       # one stale status read that still says open
        self.sweep_on_pay = False        # the ledger's own sweep refunds the hold as pay arrives
        self.status_override = None
        self.submit_hash_override = ""

    def fund(self, keyid, amount):
        self.balances[keyid] = self.balances.get(keyid, 0) + amount
        self.initial += amount
        self.check()

    def check(self):
        held = sum(row["amount"] for row in self.holds.values() if row["state"] == "open")
        assert sum(self.balances.values()) + held == self.initial

    def authorize(self, payer, task_hash, worker, amount, deadline):
        self.authorize_calls += 1
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
                if not self.loose_replay:
                    error = SettlementError("idempotency conflict")
                    error.status = 409
                    raise error
            return copy.deepcopy(hold)
        if self.balances.get(payer, 0) < amount:
            raise SettlementError("insufficient credits")
        hold_id = f"hold-{len(self.holds) + 1}"
        self.balances[payer] -= amount
        row = {"v": 1, "hold_id": hold_id, "task_hash": task_hash,
               "payer_account": payer, "worker_keyid": worker, "amount": amount,
               "deadline": deadline, "state": "open", "server_time": int(time.time()),
               "submitted_at": 0, "result_hash": "", "received_at": 0, "received_hash": ""}
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

    def submit(self, worker, hold_id, result_hash):
        self.submit_calls += 1
        if self.fail_submit:
            raise SettlementError("ledger down")
        row = self.holds[hold_id]
        if row["worker_keyid"] != worker:
            raise SettlementError("task hold not found")
        if row["state"] != "open":
            error = SettlementError(f"task hold is {row['state']}")
            error.status = 409
            raise error
        if row["result_hash"] and row["result_hash"] != result_hash:
            error = SettlementError("different task result")
            error.status = 409
            raise error
        if not row["result_hash"]:
            row["submitted_at"] = int(time.time()) or 1
            row["result_hash"] = self.submit_hash_override or result_hash
        return copy.deepcopy(row)

    def received(self, payer, hold_id, result_hash):
        self.receive_calls += 1
        if self.fail_receive:
            raise SettlementError("ledger down")
        row = self.holds[hold_id]
        if row["payer_account"] != payer:
            raise SettlementError("task hold not found")
        if row["state"] != "open":
            error = SettlementError(f"task hold is {row['state']}")
            error.status = 409
            raise error
        if not row["result_hash"] or row["submitted_at"] > row["deadline"]:
            error = SettlementError("task result was not submitted on time")
            error.status = 409
            raise error
        if row["received_hash"] and row["received_hash"] != result_hash:
            error = SettlementError("different received task result")
            error.status = 409
            raise error
        if not row["received_hash"]:
            row["received_at"] = int(time.time()) or 1
            row["received_hash"] = result_hash
        return copy.deepcopy(row)

    def sweep(self, now):
        for hold_id, row in self.holds.items():
            if row["state"] != "open" or now < row["deadline"] + 24 * 3600:
                continue
            if (row["submitted_at"] and row["submitted_at"] <= row["deadline"]
                    and row["received_at"] and row["result_hash"] == row["received_hash"]):
                row["state"] = "paid"
                self.balances[row["worker_keyid"]] = self.balances.get(row["worker_keyid"], 0) + row["amount"]
                self.transitions.append((hold_id, "paid"))
            else:
                row["state"] = "refunded"
                self.balances[row["payer_account"]] = self.balances.get(row["payer_account"], 0) + row["amount"]
                self.transitions.append((hold_id, "refunded"))
        self.check()

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
        row = self.ledger.authorize(self.keyid, task_hash, worker_keyid, amount, deadline)
        if row["state"] != "open":                       # what the real adapter's _hold() raises
            raise settle_mod.HoldNotOpen(f"Neuron task hold is {row['state']}, not open",
                                         state=row["state"], hold=row)
        return self._receipt(row, "open")

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

    def submit(self, task_hash, result_hash, *, hold_id=""):
        row = self.ledger.submit(self.keyid, hold_id, result_hash)
        assert row["task_hash"] == task_hash
        receipt = self._receipt(row, "submitted")
        receipt.update(result_hash=result_hash, submitted_at=row["submitted_at"])
        return receipt

    def received(self, task_hash, result_hash, *, hold_id=""):
        row = self.ledger.received(self.keyid, hold_id, result_hash)
        assert row["task_hash"] == task_hash
        receipt = self._receipt(row, "received")
        receipt.update(result_hash=result_hash, received_at=row["received_at"])
        return receipt

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
            elif parts.path == "/spend/task/submit":
                b = json.loads(body)
                row = ledger.submit(keyid, b["hold_id"], b["result_hash"])
            elif parts.path == "/spend/task/received":
                b = json.loads(body)
                row = ledger.received(keyid, b["hold_id"], b["result_hash"])
            else:
                hold_id = urllib.parse.parse_qs(parts.query)["hold_id"][0]
                return _Response(json.dumps(ledger.status(keyid, hold_id)).encode())
        except SettlementError as e:
            text = str(e)
            raise self._fail(request, 402 if "insufficient" in text else 404 if "not found" in text
                             else 409)
        keys = ("v", "hold_id", "task_hash", "worker_keyid", "amount", "deadline", "state")
        if parts.path == "/spend/task/submit":
            keys += ("submitted_at", "result_hash")
        elif parts.path == "/spend/task/received":
            keys += ("submitted_at", "result_hash", "received_at", "received_hash")
        row = {k: row[k] for k in keys}
        return _Response(json.dumps(row).encode())


def _real_adapter(party, http):
    return NeuronEscrowAdapter("https://ledger.test", "ab" * 32, party.node_key, party.node.node,
                               opener=http)


def _pair(net, *, paid_card=True, suffix="", real=False, constraints=None, worker_tier="known"):
    poster = net("poster" + suffix, caps=("nak.msg", "nak.task.post", "nak.pay"),
                 constraints=PAY if constraints is None else constraints)
    worker = net("worker" + suffix, caps=("nak.msg", "nak.task.claim"), tier=worker_tier)
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


def _approval_script(tmp_path, code=1, *, capture=None, sleep=0):
    path = tmp_path / f"approval-{code}-{sleep}.sh"
    lines = ["#!/bin/sh"]
    if capture is not None:
        lines.append(f"printf '%s' \"$1\" > {shlex.quote(str(capture))}")
    if sleep:
        lines.append(f"sleep {sleep}")
    lines.append(f"exit {code}")
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o700)
    return path


def _ask_policy(*, mode="threshold", auto_max=4, strangers=False):
    return {**PAY, "verdict_mode": mode, "verdict_auto_max": auto_max,
            "verdict_ask_strangers": strangers}


def _submit_waiting(poster, worker, *, reward=5, output="ok"):
    task_hash = _assign(poster, worker, reward=reward)
    worker.node.submit_task(task_hash, output)
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["state"] == "awaiting_approval")
    return task_hash


def _person_decide(poster, task_hash, decision, reason="", *, ts=None, private=None):
    ts = int(time.time()) if ts is None else ts
    private = private or poster.person_priv
    signature = private.sign(
        verdict_mod.decision_statement(task_hash, decision, reason, ts)).hex()
    return poster.node.decide_task(task_hash, decision, reason, ts=ts, signature=signature)


def test_threshold_auto_pays_at_or_below_the_limit(net):
    poster, worker, ledger = _pair(net, constraints=_ask_policy(auto_max=5))
    task_hash = _assign(poster, worker, reward=5)
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["state"] == "accepted")
    assert next(iter(ledger.holds.values()))["state"] == "paid"


def test_over_threshold_hook_accepts_with_fixed_text_and_human_touch(net, tmp_path, monkeypatch):
    capture = tmp_path / "description.txt"
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 0, capture=capture)))
    poster, worker, ledger = _pair(net, constraints=_ask_policy())
    poster.signer_requests.clear()
    task_hash = _assign(poster, worker)
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["state"] == "accepted")
    assert capture.read_text() == (
        f"Pay worker ({worker.person_pub[:8]}) 0.00000005 USDC for task 'drill' ({task_hash[:12]})? "
        "The answer passed the task's rules.")
    assert next(iter(ledger.holds.values()))["state"] == "paid"
    assert wait(lambda: any(r.get("kind") == "task.accept" and r.get("touch") is True
                            for r in poster.signer_requests))


def test_hook_reject_refunds_and_unanswered_or_timeout_keeps_waiting(net, tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 10)))
    poster, worker, ledger = _pair(net, suffix="-reject", constraints=_ask_policy())
    task_hash = _assign(poster, worker)
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["state"] == "rejected")
    assert next(iter(ledger.holds.values()))["state"] == "refunded"

    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 1)))
    poster2, worker2, ledger2 = _pair(net, suffix="-unanswered", constraints=_ask_policy())
    task2 = _submit_waiting(poster2, worker2)
    assert wait(lambda: poster2.node.store.task_get(task2, "poster")["verdict"]["approval"]["prompt_count"] == 1)
    assert next(iter(ledger2.holds.values()))["state"] == "open"

    monkeypatch.setattr("network.nakd.APPROVAL_TIMEOUT_S", 0.05)
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 0, sleep=1)))
    poster3, worker3, ledger3 = _pair(net, suffix="-timeout", constraints=_ask_policy())
    task3 = _submit_waiting(poster3, worker3)
    assert wait(lambda: poster3.node.store.task_get(task3, "poster")["verdict"]["approval"]["prompt_count"] == 1)
    time.sleep(0.1)
    assert poster3.node.store.task_get(task3, "poster")["state"] == "awaiting_approval"
    assert next(iter(ledger3.holds.values()))["state"] == "open"


def test_deadline_silence_pays_and_stranger_always_asks(net, tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 1)))
    poster, worker, ledger = _pair(net, suffix="-silence", constraints=_ask_policy())
    task_hash = _submit_waiting(poster, worker)
    deadline = poster.node.store.task_get(task_hash, "poster")["spec"]["deadline"]
    assert poster.node.sweep(now=deadline + 1) == 1
    row = poster.node.store.task_get(task_hash, "poster")
    assert row["state"] == "accepted" and row["verdict"]["decision"] == "silence"
    assert next(iter(ledger.holds.values()))["state"] == "paid"

    poster2, worker2, ledger2 = _pair(
        net, suffix="-stranger", worker_tier="stranger",
        constraints=_ask_policy(mode="auto", auto_max=0, strangers=True))
    task2 = _submit_waiting(poster2, worker2)
    assert poster2.node.store.task_get(task2, "poster")["state"] == "awaiting_approval"
    assert next(iter(ledger2.holds.values()))["state"] == "open"


def test_rule_failure_auto_rejects_without_prompt_and_automatic_signing_is_no_touch(
        net, tmp_path, monkeypatch):
    capture = tmp_path / "must-not-run"
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 0, capture=capture)))
    poster, worker, ledger = _pair(net, constraints=_ask_policy(mode="ask"))
    poster.signer_requests.clear()
    task_hash = _assign(poster, worker)
    worker.node.submit_task(task_hash, "nope")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["state"] == "rejected")
    assert not capture.exists() and next(iter(ledger.holds.values()))["state"] == "refunded"
    calls = [r for r in poster.signer_requests
             if r.get("op") == "sign_envelope" and r.get("kind") == "task.reject"]
    assert calls and all(r.get("touch") is False for r in calls)


def test_human_decide_is_idempotent_and_restart_rearms_due_hook(net, tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 1)))
    poster, worker, ledger = _pair(net, suffix="-human", constraints=_ask_policy())
    task_hash = _submit_waiting(poster, worker)
    poster.signer_requests.clear()
    first = _person_decide(poster, task_hash, "accept")
    assert not first["idempotent"]
    with pytest.raises(ValueError, match="already used"):
        _person_decide(poster, task_hash, "accept")
    assert [event for _hold, event in ledger.transitions].count("paid") == 1
    assert any(r.get("kind") == "task.accept" and r.get("touch") is True
               for r in poster.signer_requests)

    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 1)))
    poster2, worker2, ledger2 = _pair(net, suffix="-restart", constraints=_ask_policy())
    task2 = _submit_waiting(poster2, worker2)
    assert wait(lambda: poster2.node.store.task_get(task2, "poster")["verdict"]["approval"]["prompt_count"] == 1)
    row = poster2.node.store.task_get(task2, "poster")
    verdict_data = dict(row["verdict"])
    verdict_data["approval"] = dict(verdict_data["approval"],
                                     last_prompted_at=int(time.time()) - settle_mod.MAX_TS_LEAD_S - 22_000)
    poster2.node.store.task_cas(task2, "poster", "", ("awaiting_approval",),
                                verdict=json.dumps(verdict_data))
    relay, roster_loader = poster2.node.relay, poster2.node._roster_loader
    poster2.node.stop()
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 0)))

    def signer_call(req):
        poster2.signer_requests.append(dict(req))
        return poster2.signer.handle(req)

    from network import nakd
    poster2.node = nakd.Node(poster2.dir / "net", poster2.node_key, signer_call, agent="pa",
                             relay=relay, settlement=FakeNeuronAdapter(ledger2, poster2.node.node),
                             roster_loader=roster_loader).start()
    assert wait(lambda: poster2.node.store.task_get(task2, "poster")["state"] == "accepted")
    assert next(iter(ledger2.holds.values()))["state"] == "paid"


def test_socket_decide_requires_fresh_matching_person_signature_and_refuses_replay(
        net, tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 1)))
    poster, worker, _ledger = _pair(net, constraints=_ask_policy())
    task_hash = _submit_waiting(poster, worker)
    base = {"op": "task_decide", "task": task_hash, "decision": "accept", "reason": ""}
    assert nakd.handle(poster.node, base)["ok"] is False

    old_ts = int(time.time()) - verdict_mod.DECISION_MAX_SKEW_S - 1
    old = dict(base, ts=old_ts, signature=poster.person_priv.sign(
        verdict_mod.decision_statement(task_hash, "accept", "", old_ts)).hex())
    assert "stale" in nakd.handle(poster.node, old)["error"]

    wrong = ed25519.Ed25519PrivateKey.generate()
    ts = int(time.time())
    bad = dict(base, ts=ts, signature=wrong.sign(
        verdict_mod.decision_statement(task_hash, "accept", "", ts)).hex())
    assert "invalid" in nakd.handle(poster.node, bad)["error"]

    good = dict(base, ts=ts, signature=poster.person_priv.sign(
        verdict_mod.decision_statement(task_hash, "accept", "", ts)).hex())
    assert nakd.handle(poster.node, good)["state"] == "accepted"
    assert "already used" in nakd.handle(poster.node, good)["error"]


def test_hook_sanitizes_hostile_local_petname_and_title_without_consuming_failed_spawn(
        net, tmp_path, monkeypatch):
    capture = tmp_path / "description.txt"
    script = _approval_script(tmp_path, 0, capture=capture)
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(script))
    poster, worker, _ledger = _pair(net, constraints=_ask_policy())
    hostile_name = "local\x00\x1b\u202ename\n" + "z" * 5000
    poster.node.store.upsert_contact(worker.person_pub, worker.node.node, "active", hostile_name,
                                     name_src="yours")
    hostile_title = "ti\x00\x1b\u2066tle\n" + "q" * 100
    task_hash = poster.node.post_task(
        worker.person_pub, hostile_title, "Say ok.", [{"contains_all": ["ok"]}],
        reward=5, deadline_s=600)["task_hash"]
    assert wait(lambda: _worker_row(worker.node, task_hash))
    worker.node.claim_task(task_hash)
    assert wait(lambda: _worker_row(worker.node, task_hash)[0]["state"] == "assigned")
    worker.node.submit_task(task_hash, "ok")
    assert wait(capture.exists)
    shown = capture.read_text()
    assert "\x00" not in shown and "\x1b" not in shown and "\u202e" not in shown
    assert "\u2066" not in shown and "\n" not in shown
    assert worker.person_pub[:8] in shown and len(shown) < 220

    huge = poster.node.store.task_get(task_hash, "poster")
    huge["spec"] = dict(huge["spec"], title="x\x00\x1b\u202e\n" + "q" * 5000)
    huge_description = poster.node._approval_description(huge)
    subprocess.run([str(script), huge_description], check=True)
    assert capture.read_text() == huge_description and len(huge_description) < 220
    poster.node.store.upsert_contact(worker.person_pub, worker.node.node, "active",
                                     "worker-chosen-name", name_src="self-chosen")
    no_self_chosen_name = poster.node._approval_description(huge)
    assert "worker-chosen-name" not in no_self_chosen_name
    assert no_self_chosen_name.startswith(f"Pay {worker.person_pub[:8]} ")

    poster2, worker2, _ledger2 = _pair(net, suffix="-spawn", constraints=_ask_policy())
    monkeypatch.setattr("network.nakd.subprocess.run", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad")))
    task2 = _submit_waiting(poster2, worker2)
    time.sleep(0.1)
    row = poster2.node.store.task_get(task2, "poster")
    assert row["verdict"]["approval"]["prompt_count"] == 0


@pytest.mark.parametrize("mode", ["world_writable", "wrong_owner", "not_regular"])
def test_approval_hook_refuses_unsafe_command_file(net, tmp_path, monkeypatch, mode):
    script = _approval_script(tmp_path, 0)
    if mode == "world_writable":
        script.chmod(0o722)
    elif mode == "wrong_owner":
        monkeypatch.setattr("network.nakd.os.getuid", lambda: script.lstat().st_uid + 1)
    else:
        script.unlink()
        script.mkdir()
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(script))
    poster, worker, _ledger = _pair(net, suffix="-" + mode, constraints=_ask_policy())
    task_hash = _submit_waiting(poster, worker)
    time.sleep(0.1)
    assert poster.node.store.task_get(task_hash, "poster")["verdict"]["approval"]["prompt_count"] == 0


def test_worker_submit_mark_failure_never_blocks_delivery_and_reconcile_retries(
        net, tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_APPROVAL_CMD", str(_approval_script(tmp_path, 1)))
    poster, worker, ledger = _pair(net, constraints=_ask_policy())
    task_hash = _assign(poster, worker)
    ledger.fail_submit = True
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["state"] == "awaiting_approval")
    local = worker.node.store.task_get(task_hash, "worker", poster.person_pub)
    assert local["state"] == "submitted" and local["submit_state"] == "pending"
    ledger.fail_submit = False
    worker.node.reconcile_escrow()
    local = worker.node.store.task_get(task_hash, "worker", poster.person_pub)
    assert local["submit_state"] == "marked" and ledger.submit_calls >= 2


def test_two_node_funded_delivery_records_payer_receipt_with_canonical_hash_and_no_touch(net):
    poster, worker, ledger = _pair(net, constraints=_ask_policy(auto_max=10))
    poster.signer_requests.clear()
    task_hash = _assign(poster, worker)
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["state"] == "accepted")
    hold = next(iter(ledger.holds.values()))
    expected = hashlib.sha256(b"ok").hexdigest()
    assert hold["result_hash"] == hold["received_hash"] == expected
    assert hold["received_at"] > 0 and ledger.receive_calls == 1
    calls = [r for r in poster.signer_requests
             if r.get("op") == "sign_envelope" and r.get("kind") == "task.accept"]
    assert calls and all(r.get("touch") is False for r in calls)


def test_poster_receive_failure_never_blocks_handling_and_reconcile_retries(net):
    poster, worker, ledger = _pair(net, constraints=_ask_policy(auto_max=10))
    task_hash = _assign(poster, worker)
    ledger.fail_receive = True
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["receive_state"] == "pending")
    local = poster.node.store.task_get(task_hash, "poster")
    assert local["state"] == "assigned" and local["verdict"]["pending_action"] == "accept"
    assert local["output"] == "ok"
    ledger.fail_receive = False
    poster.node.reconcile_escrow()
    assert poster.node.store.task_get(task_hash, "poster")["state"] == "accepted"
    hold = next(iter(ledger.holds.values()))
    assert hold["received_hash"] == hashlib.sha256(b"ok").hexdigest()
    assert ledger.receive_calls >= 2


def test_ledger_sweep_pays_on_time_delivery_while_poster_stays_offline(net):
    poster, worker, ledger = _pair(net, constraints=_ask_policy(mode="ask"))
    task_hash = _assign(poster, worker)
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["state"] == "awaiting_approval")
    hold = next(iter(ledger.holds.values()))
    assert hold["submitted_at"] <= hold["deadline"] and hold["received_at"] > 0
    poster.node.stop()
    ledger.sweep(hold["deadline"] + 24 * 3600)
    assert hold["state"] == "paid"
    assert ledger.balances[worker.node.node] == hold["amount"]


def test_ledger_sweep_refunds_worker_mark_that_was_never_delivered(net):
    poster, worker, ledger = _pair(net, constraints=_ask_policy(mode="ask"))
    task_hash = _assign(poster, worker)
    poster.node.stop()
    worker.node.submit_task(task_hash, "ok")
    hold = next(iter(ledger.holds.values()))
    assert hold["result_hash"] == hashlib.sha256(b"ok").hexdigest()
    assert hold["received_at"] == 0 and not hold["received_hash"]
    ledger.sweep(hold["deadline"] + 24 * 3600)
    assert hold["state"] == "refunded"
    assert ledger.balances[poster.node.node] == 100


def test_poster_receive_retries_stop_on_terminal_hold_and_are_bounded(net):
    poster, worker, ledger = _pair(net, suffix="-receive-terminal",
                                   constraints=_ask_policy(mode="ask"))
    task_hash = _assign(poster, worker)
    ledger.fail_receive = True
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["receive_state"] == "pending")
    hold = next(iter(ledger.holds.values()))
    ledger.fail_receive = False
    hold["state"] = "refunded"
    ledger.balances[hold["payer_account"]] += hold["amount"]
    before = ledger.receive_calls
    poster.node.reconcile_escrow()
    terminal = poster.node.store.task_get(task_hash, "poster")
    assert terminal["receive_state"] == "terminal"
    poster.node.reconcile_escrow()
    assert ledger.receive_calls == before + 1

    poster2, worker2, ledger2 = _pair(net, suffix="-receive-bounded",
                                      constraints=_ask_policy(mode="ask"))
    task2 = _assign(poster2, worker2)
    ledger2.fail_receive = True
    worker2.node.submit_task(task2, "ok")
    assert wait(lambda: poster2.node.store.task_get(task2, "poster")["receive_state"] == "pending")
    for _ in range(nakd.POSTER_RECEIVE_MAX_ATTEMPTS + 3):
        poster2.node.reconcile_escrow()
    exhausted = poster2.node.store.task_get(task2, "poster")
    assert exhausted["receive_state"] == "exhausted"
    calls = ledger2.receive_calls
    poster2.node.reconcile_escrow()
    assert ledger2.receive_calls == calls == nakd.POSTER_RECEIVE_MAX_ATTEMPTS


def test_result_hash_mismatch_rejects_and_refunds_before_accept(net):
    poster, worker, ledger = _pair(net, constraints=_ask_policy(auto_max=10))
    ledger.submit_hash_override = "f" * 64
    task_hash = _assign(poster, worker)
    worker.node.submit_task(task_hash, "ok")
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["state"] == "rejected")
    row = poster.node.store.task_get(task_hash, "poster")
    assert row["verdict"]["reasons"] == ["result does not match the ledger mark"]
    assert next(iter(ledger.holds.values()))["state"] == "refunded"


def test_worker_submit_retries_stop_on_terminal_hold_and_are_bounded(net):
    poster, worker, ledger = _pair(net, suffix="-terminal", constraints=_ask_policy())
    task_hash = _assign(poster, worker)
    ledger.fail_submit = True
    worker.node.submit_task(task_hash, "ok")
    hold = next(iter(ledger.holds.values()))
    ledger.fail_submit = False
    hold["state"] = "refunded"
    ledger.balances[hold["payer_account"]] += hold["amount"]
    before = ledger.submit_calls
    worker.node.reconcile_escrow()
    terminal = worker.node.store.task_get(task_hash, "worker", poster.person_pub)
    assert terminal["submit_state"] == "terminal"
    worker.node.reconcile_escrow()
    assert ledger.submit_calls == before + 1

    poster2, worker2, ledger2 = _pair(net, suffix="-bounded", constraints=_ask_policy())
    task2 = _assign(poster2, worker2)
    ledger2.fail_submit = True
    worker2.node.submit_task(task2, "ok")
    for _ in range(nakd.WORKER_SUBMIT_MAX_ATTEMPTS + 3):
        worker2.node.reconcile_escrow()
    exhausted = worker2.node.store.task_get(task2, "worker", poster2.person_pub)
    assert exhausted["submit_state"] == "exhausted"
    calls = ledger2.submit_calls
    worker2.node.reconcile_escrow()
    assert ledger2.submit_calls == calls == nakd.WORKER_SUBMIT_MAX_ATTEMPTS


def test_sweep_isolates_one_failed_task_from_later_expiry(net, monkeypatch):
    poster, worker, _ledger = _pair(net)
    first = task_mod.make_spec(poster.person_pub, "first", "x", [{"max_words": 1}], deadline_s=1)
    second = task_mod.make_spec(poster.person_pub, "second", "x", [{"max_words": 1}], deadline_s=1)
    first_hash, second_hash = task_mod.task_hash(first), task_mod.task_hash(second)
    poster.node.store.task_insert(first_hash, "poster", "", first, "awaiting_approval",
                                  targets=[worker.person_pub])
    poster.node.store.task_insert(second_hash, "poster", "", second, "posted",
                                  targets=[worker.person_pub])
    original = poster.node._apply_task_decision

    def fail_first(task_hash, *args, **kwargs):
        if task_hash == first_hash:
            raise RuntimeError("signer down")
        return original(task_hash, *args, **kwargs)

    monkeypatch.setattr(poster.node, "_apply_task_decision", fail_first)
    assert poster.node.sweep(now=max(first["deadline"], second["deadline"]) + 1) == 1
    assert poster.node.store.task_get(first_hash, "poster")["state"] == "awaiting_approval"
    assert poster.node.store.task_get(second_hash, "poster")["state"] == "expired"


def test_stranger_roster_reads_pinned_tsv_directly(net, tmp_path):
    poster, worker = net("roster-poster"), net("roster-worker")
    roster = tmp_path / "peers.tsv"
    roster.write_text(f"{worker.node.node}\tworker\top\ttrusted\ttenant\n")
    poster.node._roster_loader = None
    poster.node._roster_path = roster
    contact = {"node": worker.node.node}
    assert poster.node._worker_is_stranger(contact) is False
    roster.write_text(f"{worker.node.node}\tworker\top\tKnown\ttenant\n")
    assert poster.node._worker_is_stranger(contact) is True
    poster.node._roster_path = tmp_path / "missing.tsv"
    assert poster.node._worker_is_stranger(contact) is True


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
            value = dict(hold, payer_account="payer", server_time=1, submitted_at=0, result_hash="",
                         received_at=0, received_hash="")
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


def test_unrecorded_hold_refunded_by_the_ledger_reaches_posted_then_expired(net):
    poster, worker, ledger = _pair(net)
    ledger.drop_response = True
    task_hash = poster.node.post_task(worker.dir.name, "drill", "Say ok.", [{"contains_all": ["ok"]}],
                                      reward=5, deadline_s=2)["task_hash"]
    assert wait(lambda: _worker_row(worker.node, task_hash))
    worker.node.claim_task(task_hash)
    assert wait(lambda: len(ledger.holds) == 1)
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["funding_reason"].startswith("escrow:"))
    hold = next(iter(ledger.holds.values()))
    ledger.refund(poster.node.node, hold["hold_id"], "ledger sweep")     # behind the poster's back
    time.sleep(2.2)
    poster.node.sweep()                                   # reconcile sees "refunded": definitive, revert
    poster.node.sweep()                                   # now plain posted: expires
    row = poster.node.store.task_get(task_hash, "poster")
    assert row["state"] == "expired" and not row["hold_id"] and not row["assignee"]
    worker.node.sweep()                                   # the worker side expires its own stale claim
    assert _worker_row(worker.node, task_hash)[0]["state"] == "expired"
    calls = ledger.authorize_calls
    poster.node.sweep()
    poster.node.reconcile_escrow()
    assert ledger.authorize_calls == calls               # nothing left to retry
    assert ledger.balances[poster.node.node] == 100


def test_ledger_paid_hold_is_adopted_only_for_this_tasks_worker(net):
    poster, worker, ledger = _pair(net)
    ledger.drop_response = True
    task_hash = _post_and_claim(poster, worker)
    assert wait(lambda: len(ledger.holds) == 1)
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["funding_reason"].startswith("escrow:"))
    hold = next(iter(ledger.holds.values()))
    ledger.pay(poster.node.node, hold["hold_id"])
    poster.node.reconcile_escrow()
    row = poster.node.store.task_get(task_hash, "poster")
    assert row["hold_id"] == hold["hold_id"] and row["funding_state"] == "paid"
    # a paid hold naming someone else is never adopted: the task is put back instead
    poster2, worker2, ledger2 = _pair(net, suffix="-x")
    ledger2.drop_response = True
    task2 = _post_and_claim(poster2, worker2)
    assert wait(lambda: len(ledger2.holds) == 1)
    assert wait(lambda: poster2.node.store.task_get(task2, "poster")["funding_reason"].startswith("escrow:"))
    hold2 = next(iter(ledger2.holds.values()))
    ledger2.pay(poster2.node.node, hold2["hold_id"])
    hold2["worker_keyid"] = "ee" * 32
    ledger2.loose_replay = True
    poster2.node.reconcile_escrow()
    row2 = poster2.node.store.task_get(task2, "poster")
    assert row2["state"] == "posted" and not row2["hold_id"]


def test_sweep_does_not_expire_an_assigned_paid_task_whose_hold_is_unrecorded(net):
    poster, worker, ledger = _pair(net)
    ledger.drop_response = True
    task_hash = poster.node.post_task(worker.dir.name, "drill", "Say ok.", [{"contains_all": ["ok"]}],
                                      reward=5, deadline_s=2)["task_hash"]
    assert wait(lambda: _worker_row(worker.node, task_hash))
    worker.node.claim_task(task_hash)
    assert wait(lambda: len(ledger.holds) == 1)
    assert wait(lambda: poster.node.store.task_get(task_hash, "poster")["funding_reason"].startswith("escrow:"))
    time.sleep(2.2)
    ledger.fail_authorize = True                          # recovery cannot run: the hold stays unknown
    assert poster.node.sweep() == 0
    assert poster.node.store.task_get(task_hash, "poster")["state"] == "assigned"


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
