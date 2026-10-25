"""Operator suspension state, report signing, and contact-session propagation."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
for candidate in (os.environ.get("NAK_STHAMBHA_PATH", ""), "/home/prithvi/sthambha"):
    if candidate and (Path(candidate) / "sthambha" / "signer.py").exists():
        sys.path.insert(0, candidate)
        break

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402
from sthambha.signer import canonical  # noqa: E402

from network import nak, nak_mcp, nakd  # noqa: E402
from network.client import NakError  # noqa: E402
from network.store import Store  # noqa: E402
from network.suspension import (MAX_BLOB, decode_record, encode_record, make_record,  # noqa: E402
                                suspension_statement, verify_record)
from test_nakd import Party, wait  # noqa: E402
from transport.relay import RendezvousRelay  # noqa: E402


def _key():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, private.public_key().public_bytes_raw().hex()


def _record(private=None, person=None, seq=1, nodes=("11" * 32,), reasons=None, now=123):
    private = private or ed25519.Ed25519PrivateKey.generate()
    person = person or private.public_key().public_bytes_raw().hex()
    reasons = reasons if reasons is not None else {node: "abuse" for node in nodes}
    return make_record(private.sign, person, seq, nodes, reasons, now)


def _resign(record, private):
    out = dict(record)
    out["sig"] = private.sign(suspension_statement(out)).hex()
    return out


def _proof(private, kind, payload, ts=None):
    ts = int(time.time()) if ts is None else ts
    return {"ts": ts, "signature": private.sign(
        nakd.person_action_statement(kind, payload, ts)).hex()}


def _install_proof(private, record, ts=None):
    return _proof(private, "install-suspension",
                  nakd.suspension_record_proof_payload(record), ts)


def test_record_round_trip_and_codec_defects():
    private, person = _key()
    record = make_record(private.sign, person, 7, ["22" * 32, "11" * 32, "11" * 32],
                         {"11" * 32: "spam", "22" * 32: "fraud"}, 456)
    assert record["nodes"] == ["11" * 32, "22" * 32]
    assert verify_record(record, person) == record
    assert decode_record(encode_record(record)) == record
    with pytest.raises(ValueError, match="invalid base64"):
        decode_record(encode_record(record) + "=")
    with pytest.raises(ValueError, match="256 KiB"):
        decode_record("nks1." + "a" * MAX_BLOB)


def test_largest_full_reason_record_fits_session_frame_and_next_is_refused():
    private, person = _key()
    nodes = [f"{index:064x}" for index in range(763)]
    record = make_record(private.sign, person, 1, nodes,
                         {node: "x" * 120 for node in nodes}, 123)
    blob = encode_record(record)
    assert len(blob.encode()) <= MAX_BLOB
    left, right = socket.socketpair()
    try:
        nakd.send_frame(left, {"t": "susp", "rec": record})
        assert nakd.recv_frame(right) == {"t": "susp", "rec": record}
    finally:
        left.close()
        right.close()
    extra = f"{len(nodes):064x}"
    with pytest.raises(ValueError, match="encoded limit"):
        make_record(private.sign, person, 2, [*nodes, extra],
                    {**record["reasons"], extra: "x" * 120}, 124)


@pytest.mark.parametrize("mutate", [
    lambda r: dict(r, schema="nak.suspensions.v0"),
    lambda r: {key: value for key, value in r.items() if key != "issued_at"},
    lambda r: dict(r, extra=True),
    lambda r: dict(r, seq=True),
    lambda r: dict(r, nodes=["AA" * 32]),
    lambda r: dict(r, nodes=["22" * 32, "11" * 32],
                   reasons={"11" * 32: "a", "22" * 32: "b"}),
    lambda r: dict(r, reasons={}),
    lambda r: dict(r, reasons={"11" * 32: "x" * 121}),
    lambda r: dict(r, sig="0" * 127),
])
def test_record_rejects_schema_defects(mutate):
    private, person = _key()
    record = mutate(_record(private, person))
    if isinstance(record, dict) and "sig" in record and isinstance(record["sig"], str) and \
            len(record["sig"]) == 128:
        record = _resign(record, private)
    with pytest.raises(ValueError):
        verify_record(record, person)


def test_store_replaces_state_and_only_trusted_operator_enforces(tmp_path):
    store = Store(tmp_path / "store", b"s" * 32)
    private, person = _key()
    operator_node = "aa" * 32
    store.upsert_contact(person, operator_node, "active", "operator")
    first = _record(private, person, 1, ("11" * 32, "22" * 32))
    second = _record(private, person, 2, ("22" * 32,))
    empty = _record(private, person, 3, (), {})
    assert store.apply_contact_suspension(person, first)
    assert store.suspended_nodes() == set()
    store.set_operator_person(person)
    assert store.suspended_nodes() == {"11" * 32, "22" * 32}
    assert store.apply_contact_suspension(person, second)
    assert store.suspended_nodes() == {"22" * 32}
    assert store.apply_contact_suspension(person, empty)
    assert store.suspended_nodes() == set()
    assert not store.apply_contact_suspension(person, first)
    assert store.suspension_status()["received"][person] == {"seq": 3, "count": 0}


def test_store_persists_exact_pending_record_and_promotes_or_recovers(tmp_path):
    store = Store(tmp_path / "pending", b"q" * 32)
    private, person = _key()
    first = _record(private, person, 1, ("11" * 32,))
    second = _record(private, person, 2, ("22" * 32,))

    assert store.set_pending_own_suspension(first)
    assert not store.set_pending_own_suspension(first)
    assert store.pending_own_suspension() == first
    with pytest.raises(ValueError, match="already pending"):
        store.set_pending_own_suspension(second)
    store.commit_pending_own_suspension(first)
    assert store.own_suspension() == first
    assert store.pending_own_suspension() is None

    assert store.set_pending_own_suspension(second)
    with pytest.raises(ValueError, match="not newer than local state"):
        store.recover_own_suspension(first)
    store.recover_own_suspension(second)
    assert store.own_suspension() == second
    assert store.pending_own_suspension() is None


def test_untrust_and_contact_removal_stop_enforcement_without_reopening_replay(tmp_path):
    store = Store(tmp_path / "store", b"u" * 32)
    private, person = _key()
    operator_node = "aa" * 32
    suspended = "bb" * 32
    record = _record(private, person, 7, (suspended,))
    store.upsert_contact(person, operator_node, "active", "operator")
    assert store.apply_contact_suspension(person, record)
    store.set_operator_person(person)
    assert store.suspended_nodes() == {suspended}

    store.unset_operator_person(person)
    assert store.suspended_nodes() == set()
    store.set_operator_person(person)
    with pytest.raises(ValueError, match="trusted operator; run nak operator-untrust first"):
        store.remove_contact(person)
    assert store.contact(person) is not None
    assert person in store.operator_persons()
    assert store.suspended_nodes() == {suspended}
    store.unset_operator_person(person)
    store.remove_contact(person)
    assert store.suspended_nodes() == set()
    assert person not in store.operator_persons()
    assert store.suspension_status()["received"][person] == {"seq": 7, "count": 0}

    store.upsert_contact(person, operator_node, "active", "operator-again")
    store.set_operator_person(person)
    assert not store.apply_contact_suspension(person, _record(
        private, person, 6, (suspended,)))
    assert store.suspended_nodes() == set()


def test_person_proofs_prune_by_signed_timestamp(tmp_path):
    store = Store(tmp_path / "store", b"p" * 32)
    store.consume_person_action_proof("aa" * 64, 100, 219, 120)
    store.consume_person_action_proof("bb" * 64, 221, 221, 120)
    # At now=221 the first proof is outside the window by its signed timestamp even though it was
    # consumed only two seconds ago, so its signature no longer occupies the replay table.
    store.consume_person_action_proof("aa" * 64, 221, 221, 120)


def test_installing_own_record_enforces_it_locally(tmp_path):
    private, person = _key()
    node = object.__new__(nakd.Node)
    node.person = person
    node.store = Store(tmp_path / "own", b"o" * 32)
    node._wake = {}
    node._clock = time.time
    record = _record(private, person, 1, ("33" * 32,))
    node.install_own_suspension(record, **_install_proof(private, record))
    assert node.store.suspended_nodes() == {"33" * 32}
    assert person in node.store.operator_persons()


def test_raw_owner_ops_require_single_use_person_proofs_and_trusted_import(tmp_path):
    party = Party(tmp_path, "owner", 9)
    operator_private, operator_person = _key()
    operator_node = "44" * 32
    party.node.store.upsert_contact(operator_person, operator_node, "active", "operator")
    remote = _record(operator_private, operator_person, 1, ("55" * 32,))
    blob = encode_record(remote)
    own = _record(party.person_priv, party.person_pub, 1, ("66" * 32,))
    try:
        raw_requests = [
            {"op": "operator_trust", "contact": "operator"},
            {"op": "operator_untrust", "contact": "operator"},
            {"op": "install_suspension", "record": own},
        ]
        party.node.store.set_operator_person(operator_person)
        raw_requests.append({"op": "import_suspension", "blob": blob})
        for request in raw_requests:
            result = nakd.handle(party.node, request)
            assert not result["ok"]
            assert "person must authorize" in result["error"]

        trust_proof = _proof(party.person_priv, "operator-trust", {"person": operator_person})
        assert nakd.handle(party.node, {"op": "operator_trust", "contact": "operator",
                                       **trust_proof})["ok"]
        replay = nakd.handle(party.node, {"op": "operator_trust", "contact": "operator",
                                         **trust_proof})
        assert not replay["ok"] and "already been used" in replay["error"]

        imported = nakd.handle(party.node, {
            "op": "import_suspension", "blob": blob,
            **_proof(party.person_priv, "import-suspension",
                     nakd.suspension_record_proof_payload(remote)),
        })
        assert imported["ok"] and imported["person"] == operator_person
        assert party.node.store.suspended_nodes() == {"55" * 32}

        untrust_proof = _proof(
            party.person_priv, "operator-untrust", {"person": operator_person})
        untrusted = nakd.handle(party.node, {
            "op": "operator_untrust", "contact": "operator", **untrust_proof})
        assert untrusted["ok"] and party.node.store.suspended_nodes() == set()

        installed = nakd.handle(party.node, {"op": "install_suspension", "record": own,
                                             **_install_proof(party.person_priv, own)})
        assert installed["ok"] and installed["seq"] == 1

        stranger_private, stranger_person = _key()
        stranger_record = _record(stranger_private, stranger_person, 1, ("77" * 32,))
        party.node.store.upsert_contact(stranger_person, "88" * 32, "active", "stranger-op")
        refused = nakd.handle(party.node, {
            "op": "import_suspension", "blob": encode_record(stranger_record),
            **_proof(party.person_priv, "import-suspension",
                     nakd.suspension_record_proof_payload(stranger_record)),
        })
        assert not refused["ok"] and "trusted operator" in refused["error"]
        assert stranger_person not in party.node.store.operator_persons()
    finally:
        party.node.stop()


def test_raw_remove_refuses_trusted_operator_until_person_proofed_untrust(tmp_path):
    party = Party(tmp_path, "remove-operator", 9)
    operator_private, operator_person = _key()
    operator_node = "48" * 32
    suspended = "49" * 32
    party.node.store.upsert_contact(operator_person, operator_node, "active", "operator")
    party.node.store.apply_contact_suspension(
        operator_person, _record(operator_private, operator_person, 1, (suspended,)))
    party.node.store.set_operator_person(operator_person)
    try:
        refused = nakd.handle(party.node, {"op": "remove", "who": "operator"})
        assert refused == {
            "ok": False,
            "error": "this contact is a trusted operator; run nak operator-untrust first",
        }
        assert party.node.store.contact(operator_person) is not None
        assert operator_person in party.node.store.operator_persons()
        assert party.node.store.suspended_nodes() == {suspended}

        proof = _proof(
            party.person_priv, "operator-untrust", {"person": operator_person})
        untrusted = nakd.handle(party.node, {
            "op": "operator_untrust", "contact": "operator", **proof})
        assert untrusted["ok"]
        removed = nakd.handle(party.node, {"op": "remove", "who": "operator"})
        assert removed == {"ok": True, "removed": operator_person}
        assert party.node.store.contact(operator_person) is None
        assert operator_person not in party.node.store.operator_persons()
        assert party.node.store.suspended_nodes() == set()
    finally:
        party.node.stop()


def test_stage_install_and_stale_recovery_each_require_person_proof(tmp_path):
    party = Party(tmp_path, "pending-owner", 9)
    first = _record(party.person_priv, party.person_pub, 1, ("61" * 32,))
    pending = _record(party.person_priv, party.person_pub, 2, ("62" * 32,))
    server = _record(party.person_priv, party.person_pub, 2, ("63" * 32,))
    try:
        refused = nakd.handle(party.node, {"op": "stage_suspension", "record": first})
        assert not refused["ok"] and "person must authorize" in refused["error"]
        staged = nakd.handle(party.node, {
            "op": "stage_suspension", "record": first,
            **_proof(party.person_priv, "stage-suspension",
                     nakd.suspension_record_proof_payload(first)),
        })
        assert staged["ok"] and party.node.store.pending_own_suspension() == first
        assert party.node.store.own_suspension() is None
        refused = nakd.handle(party.node, {"op": "install_suspension", "record": first})
        assert not refused["ok"] and party.node.store.own_suspension() is None
        installed = nakd.handle(party.node, {
            "op": "install_suspension", "record": first,
            **_install_proof(party.person_priv, first),
        })
        assert installed["ok"] and party.node.store.own_suspension() == first

        assert nakd.handle(party.node, {
            "op": "stage_suspension", "record": pending,
            **_proof(party.person_priv, "stage-suspension",
                     nakd.suspension_record_proof_payload(pending)),
        })["ok"]
        refused = nakd.handle(party.node, {"op": "recover_suspension", "record": server})
        assert not refused["ok"] and party.node.store.own_suspension() == first
        recovered = nakd.handle(party.node, {
            "op": "recover_suspension", "record": server,
            **_proof(party.person_priv, "recover-suspension",
                     nakd.suspension_record_proof_payload(server)),
        })
        assert recovered["ok"] and party.node.store.own_suspension() == server
        assert party.node.store.pending_own_suspension() is None
    finally:
        party.node.stop()


def test_person_action_proof_rejects_timestamp_beyond_future_skew(tmp_path):
    party = Party(tmp_path, "owner", 9)
    operator_private, operator_person = _key()
    del operator_private
    party.node.store.upsert_contact(operator_person, "44" * 32, "active", "operator")
    party.node._clock = lambda: 1000
    proof = _proof(party.person_priv, "operator-trust", {"person": operator_person}, 1121)
    try:
        result = nakd.handle(party.node, {
            "op": "operator_trust", "contact": "operator", **proof})
        assert not result["ok"] and "person must authorize" in result["error"]
    finally:
        party.node.stop()


def test_install_rejects_a_record_not_signed_by_our_person_even_with_proof(tmp_path):
    party = Party(tmp_path, "owner", 9)
    other_private, other_person = _key()
    record = _record(other_private, other_person, 1, ("99" * 32,))
    try:
        result = nakd.handle(party.node, {
            "op": "install_suspension", "record": record,
            **_install_proof(party.person_priv, record),
        })
        assert not result["ok"] and "another person" in result["error"]
    finally:
        party.node.stop()


@pytest.fixture
def net(tmp_path):
    relay = RendezvousRelay("127.0.0.1", 0, max_per_ip_per_min=100000)
    port = relay.start()
    parties = []

    def roster():
        return {party.node.node: {"tier": party.tier} for party in parties}

    def make(name, **kwargs):
        party = Party(tmp_path, name, port, roster_loader=roster, **kwargs)
        parties.append(party)
        return party

    yield make
    for party in parties:
        party.node.stop()
    relay.stop()


def _connect(inviter, joiner, inviter_name, joiner_name):
    joiner.node.redeem(inviter.invite(), petname=inviter_name, wait_s=15)
    inviter.node.accept(inviter.node.store.requests()[0]["id"], petname=joiner_name)
    assert wait(lambda: (joiner.node.store.contact(inviter.person_pub) or {}).get("state") == "active")
    assert wait(lambda: inviter.person_pub in joiner.node._sessions and
                joiner.person_pub in inviter.node._sessions)


def _suspension_publisher(node):
    private, person = _key()
    node.store.upsert_contact(person, "fe" * 32, "active", "test-operator")
    node.store.set_operator_person(person)

    def publish(seq, *nodes):
        record = _record(private, person, seq, nodes,
                         {subject: "abuse" for subject in nodes})
        assert node.store.apply_contact_suspension(person, record)

    return publish


def test_end_to_end_push_ack_enforcement_and_unsuspend(net):
    operator = net("operator")
    receiver = net("receiver")
    sender = net("sender", caps=("nak.msg", "nak.task.post"))
    _connect(operator, receiver, "operator", "receiver")
    _connect(receiver, sender, "receiver", "sender")
    receiver.node.store.set_operator_person(operator.person_pub)

    record = _record(operator.person_priv, operator.person_pub, 1, (sender.node.node,))
    operator.node.install_own_suspension(record, **_install_proof(operator.person_priv, record))
    assert wait(lambda: receiver.node.store.suspended_nodes() == {sender.node.node})
    assert wait(lambda: operator.node.store.suspension_sent_seq(
        receiver.person_pub, receiver.node.node) == 1)

    mid = sender.node.send("receiver", "must be refused")["queued"]
    assert wait(lambda: sender.node.store.refused(mid) == "node suspended by operator")
    assert receiver.node.store.inbox() == []
    with pytest.raises(ValueError, match="node suspended by operator"):
        receiver.node._queue(sender.person_pub, "task.assign", {"task_hash": "aa" * 32})

    restored_record = _record(operator.person_priv, operator.person_pub, 2, (), {})
    operator.node.install_own_suspension(
        restored_record, **_install_proof(operator.person_priv, restored_record))
    assert wait(lambda: receiver.node.store.suspended_nodes() == set())
    restored = sender.node.send("receiver", "allowed again")["queued"]
    assert wait(lambda: sender.node.delivered(restored))


def test_suspended_task_actions_refuse_before_state_or_ledger_and_retry_after_unsuspend(net):
    poster = net("poster", caps=("nak.msg", "nak.task.post", "nak.pay"),
                 constraints={"per_tx_cap": 10, "day_cap": 50})
    worker = net("worker", caps=("nak.msg", "nak.task.claim"))
    _connect(poster, worker, "poster", "worker")
    poster_suspensions = _suspension_publisher(poster.node)
    worker_suspensions = _suspension_publisher(worker.node)

    # Posting a paid task used to open escrow and insert the task before _queue refused it.
    poster_suspensions(1, worker.node.node)
    poster_tasks = poster.node.store.tasks("poster")
    poster_ledger = poster.node.settle.journal()
    poster_outbox = poster.node.store.outbox()
    with pytest.raises(ValueError, match="node suspended by operator"):
        poster.node.post_task("worker", "suspension drill", "Say ok.",
                              [{"contains_all": ["ok"]}], reward=5)
    assert poster.node.store.tasks("poster") == poster_tasks
    assert poster.node.settle.journal() == poster_ledger
    assert poster.node.store.outbox() == poster_outbox

    poster_suspensions(2)
    task_hash = poster.node.post_task(
        "worker", "suspension drill", "Say ok.", [{"contains_all": ["ok"]}], reward=5
    )["task_hash"]
    assert wait(lambda: worker.node.store.task_get(task_hash, "worker", poster.person_pub))
    assert len(poster.node.settle.journal(task_hash)) == 1

    # Claiming used to leave the worker row in `claimed` with no queued claim.
    worker_suspensions(1, poster.node.node)
    offered = worker.node.store.task_get(task_hash, "worker", poster.person_pub)
    worker_ledger = worker.node.settle.journal()
    worker_outbox = worker.node.store.outbox()
    with pytest.raises(ValueError, match="node suspended by operator"):
        worker.node.claim_task(task_hash)
    assert worker.node.store.task_get(task_hash, "worker", poster.person_pub) == offered
    assert worker.node.settle.journal() == worker_ledger
    assert worker.node.store.outbox() == worker_outbox

    worker_suspensions(2)
    assert worker.node.claim_task(task_hash)["state"] == "claimed"
    assert wait(lambda: worker.node.store.task_get(
        task_hash, "worker", poster.person_pub)["state"] == "assigned")

    # Model the shared-ledger submit path and stop background work so only this call can touch it.
    worker.node.stop()
    assert worker.node.store.task_cas(task_hash, "worker", poster.person_pub, ("assigned",),
                                      hold_id="67d08151619349ec80e86fd3a473b571",
                                      funding_state="funded")
    ledger_calls = []
    worker.node.settle.open_on_assign = True

    def ledger_status(seen_hash, *, hold_id=""):
        ledger_calls.append(("status", seen_hash, hold_id))
        task = worker.node.store.task_get(task_hash, "worker", poster.person_pub)
        return {"state": "open", "worker_keyid": worker.node.node, "amount": 5,
                "task_hash": task_hash, "deadline": task["spec"]["deadline"]}

    def ledger_submit(seen_hash, result_hash, *, hold_id=""):
        ledger_calls.append(("submit", seen_hash, hold_id, result_hash))
        return {"event": "submitted"}

    worker.node.settle.status = ledger_status
    worker.node.settle.submit = ledger_submit
    worker_suspensions(3, poster.node.node)
    assigned = worker.node.store.task_get(task_hash, "worker", poster.person_pub)
    worker_outbox = worker.node.store.outbox()
    with pytest.raises(ValueError, match="node suspended by operator"):
        worker.node.submit_task(task_hash, "ok")
    assert worker.node.store.task_get(task_hash, "worker", poster.person_pub) == assigned
    assert worker.node.store.outbox() == worker_outbox
    assert ledger_calls == []

    worker_suspensions(4)
    assert worker.node.submit_task(task_hash, "ok")["state"] == "submitted"
    assert [call[0] for call in ledger_calls][-1] == "submit"
    assert all(call[0] == "status" for call in ledger_calls[:-1])
    assert worker.node.store.task_get(task_hash, "worker", poster.person_pub)["state"] == "submitted"


def test_suspension_arriving_after_preflight_never_strands_post_claim_or_submit(tmp_path):
    parties = []

    def party(name, caps):
        made = Party(tmp_path, name, 9, caps=caps,
                     constraints={"per_tx_cap": 10, "day_cap": 50})
        parties.append(made)
        return made

    def race_at_queue(actor, peer_node, publish):
        original = actor.node._queue
        fired = False

        def raced(person, kind, body, **kwargs):
            nonlocal fired
            if not fired:
                fired = True
                publish(1, peer_node)
            return original(person, kind, body, **kwargs)

        actor.node._queue = raced

    try:
        poster = party("race-poster", ("nak.msg", "nak.task.post", "nak.pay"))
        worker_private, worker_person = _key()
        del worker_private
        worker_node = "41" * 32
        poster.node.store.upsert_contact(worker_person, worker_node, "active", "worker")
        race_at_queue(poster, worker_node, _suspension_publisher(poster.node))
        posted = poster.node.post_task(
            "worker", "race post", "Say ok.", [{"contains_all": ["ok"]}], reward=5)
        assert poster.node.store.task_get(posted["task_hash"], "poster", "")["state"] == "posted"
        assert len(poster.node.settle.journal(posted["task_hash"])) == 1
        assert len(poster.node.store.outbox()) == 1

        claimer = party("race-claimer", ("nak.msg", "nak.task.claim"))
        poster_private, poster_person = _key()
        del poster_private
        poster_node = "42" * 32
        claimer.node.store.upsert_contact(poster_person, poster_node, "active", "poster")
        spec = nakd.T.make_spec(poster_person, "race claim", "Say ok.",
                                [{"contains_all": ["ok"]}], deadline_s=3600)
        task_hash = nakd.T.task_hash(spec)
        assert claimer.node.store.task_insert(
            task_hash, "worker", poster_person, spec, "offered")
        race_at_queue(claimer, poster_node, _suspension_publisher(claimer.node))
        assert claimer.node.claim_task(task_hash)["state"] == "claimed"
        assert claimer.node.store.task_get(task_hash, "worker", poster_person)["state"] == "claimed"
        assert len(claimer.node.store.outbox()) == 1

        submitter = party("race-submitter", ("nak.msg", "nak.task.claim"))
        owner_private, owner_person = _key()
        del owner_private
        owner_node = "43" * 32
        submitter.node.store.upsert_contact(owner_person, owner_node, "active", "poster")
        spec = nakd.T.make_spec(owner_person, "race submit", "Say ok.",
                                [{"contains_all": ["ok"]}], deadline_s=3600)
        task_hash = nakd.T.task_hash(spec)
        assert submitter.node.store.task_insert(
            task_hash, "worker", owner_person, spec, "assigned")
        race_at_queue(submitter, owner_node, _suspension_publisher(submitter.node))
        assert submitter.node.submit_task(task_hash, "ok")["state"] == "submitted"
        assert submitter.node.store.task_get(
            task_hash, "worker", owner_person)["state"] == "submitted"
        assert len(submitter.node.store.outbox()) == 1
    finally:
        for made in parties:
            made.node.stop()


def test_inbound_suspension_decision_survives_claim_and_result_reply_races(tmp_path):
    poster = Party(
        tmp_path, "inbound-race-poster", 9,
        caps=("nak.msg", "nak.task.post", "nak.pay"),
        constraints={"per_tx_cap": 10, "day_cap": 50})
    worker = Party(
        tmp_path, "inbound-race-worker", 9,
        caps=("nak.msg", "nak.task.claim"))
    poster.node.stop()
    worker.node.stop()
    poster.node.store.upsert_contact(
        worker.person_pub, worker.node.node, "active", "worker")
    publish = _suspension_publisher(poster.node)

    def race_reply(expected_kind, seq):
        original = poster.node._queue
        fired = False

        def raced(person, kind, body, **kwargs):
            nonlocal fired
            assert kind == expected_kind
            if not fired:
                fired = True
                publish(seq, worker.node.node)
            return original(person, kind, body, **kwargs)

        poster.node._queue = raced
        return original

    task_hash = poster.node.post_task(
        "worker", "inbound suspension race", "Say ok.",
        [{"contains_all": ["ok"]}], reward=5)["task_hash"]

    original = race_reply("task.assign", 1)
    claim_body = {"task_hash": task_hash, "mid": "claim-race"}
    claim_env = worker.node._sign(
        "task.claim", poster.person_pub, claim_body, touch=False)
    claim_reply = poster.node._receive(
        worker.person_pub, {"t": "env", "env": claim_env, "body": claim_body})
    poster.node._queue = original
    assert claim_reply == {"t": "ack", "mid": "claim-race"}
    assert poster.node.store.task_get(task_hash, "poster", "")["state"] == "assigned"
    assert [frame["kind"] for frame in poster.node.store.pending_for(worker.person_pub)] == [
        "task.post", "task.assign"]

    publish(2)
    original = race_reply("task.accept", 3)
    result_body = {
        "task_hash": task_hash, "epoch": 1, "output": "ok", "mid": "result-race"}
    result_env = worker.node._sign(
        "task.result", poster.person_pub, result_body, touch=False)
    result_reply = poster.node._receive(
        worker.person_pub, {"t": "env", "env": result_env, "body": result_body})
    poster.node._queue = original
    assert result_reply == {"t": "ack", "mid": "result-race"}
    task = poster.node.store.task_get(task_hash, "poster", "")
    assert task["state"] == "accepted"
    assert task["verdict"]["passed"] is True
    assert [entry["event"] for entry in poster.node.settle.journal(task_hash)] == [
        "open", "released"]
    assert [frame["kind"] for frame in poster.node.store.pending_for(worker.person_pub)] == [
        "task.post", "task.assign", "task.accept"]


def _stopped_poster_with_suspended_worker(tmp_path, name):
    poster = Party(tmp_path, name, 9, caps=("nak.msg", "nak.task.post", "nak.pay"),
                   constraints={"per_tx_cap": 10, "day_cap": 50})
    poster.node.stop()
    _private, worker_person = _key()
    worker_node = hashlib.sha256(name.encode()).hexdigest()
    poster.node.store.upsert_contact(worker_person, worker_node, "active", "worker")
    _suspension_publisher(poster.node)(1, worker_node)
    return poster, worker_person, worker_node


def _poster_task(node, worker_person, *, state, reward=0, deadline_s=3600,
                 output=None, verdict=None, receive_state=None, hold_id=""):
    spec = nakd.T.make_spec(node.person, f"suspension finality {time.time_ns()}", "Say ok.",
                            [{"contains_all": ["ok"]}], reward=reward,
                            deadline_s=deadline_s,
                            unit="credits" if node.settle.open_on_assign else "TEST")
    task_hash = nakd.T.task_hash(spec)
    assert node.store.task_insert(
        task_hash, "poster", "", spec, state, [worker_person],
        escrow_auth={"task_hash": task_hash, "amount": reward, "mint": "credits"}
        if reward else None,
        hold_id=hold_id, funding_state="pending" if reward else "")
    fields = {"assignee": worker_person, "epoch": 1}
    if output is not None:
        fields["output"] = output
    if verdict is not None:
        fields["verdict"] = json.dumps(verdict)
    if receive_state is not None:
        fields["receive_state"] = receive_state
    assert node.store.task_cas(task_hash, "poster", "", (state,), **fields)
    return task_hash


def test_human_silence_and_approval_hook_keep_final_state_when_worker_is_suspended(
        tmp_path):
    made = []
    try:
        human, worker, _node = _stopped_poster_with_suspended_worker(tmp_path, "final-human")
        made.append(human)
        task_hash = _poster_task(human.node, worker, state="awaiting_approval")
        ts = int(time.time())
        signature = human.person_priv.sign(
            nakd.V.decision_statement(task_hash, "accept", "", ts)).hex()
        result = human.node.decide_task(task_hash, "accept", ts=ts, signature=signature)
        assert result["state"] == "accepted"
        assert human.node.store.task_get(task_hash, "poster", "")["state"] == "accepted"
        assert human.node.store.pending_for(worker) == []

        silence, worker, _node = _stopped_poster_with_suspended_worker(tmp_path, "final-silence")
        made.append(silence)
        task_hash = _poster_task(silence.node, worker, state="awaiting_approval")
        row = silence.node.store.task_get(task_hash, "poster", "")
        row["spec"]["deadline"] = int(time.time()) - 1
        primary_key = silence.node.store._pk((task_hash, "poster", ""))
        silence.node.store._q(
            "UPDATE tasks SET spec=? WHERE task_hash=? AND role='poster' AND peer=''",
            (silence.node.store._enc("tasks", "spec", primary_key, json.dumps(row["spec"])),
             task_hash))
        assert silence.node.sweep() == 1
        assert silence.node.store.task_get(task_hash, "poster", "")["state"] == "accepted"

        approval_party, worker, _node = _stopped_poster_with_suspended_worker(
            tmp_path, "final-approval")
        made.append(approval_party)
        task_hash = _poster_task(approval_party.node, worker, state="awaiting_approval")
        approve = tmp_path / "approve-task"
        approve.write_text("#!/bin/sh\nexit 0\n")
        approve.chmod(0o700)
        approval_party.node._run_approval_hook(task_hash, str(approve))
        assert approval_party.node.store.task_get(task_hash, "poster", "")["state"] == "accepted"
        assert approval_party.node.store.pending_for(worker) == []
    finally:
        for party in made:
            party.node.stop()


def test_receipt_retry_and_expiry_keep_final_state_when_worker_is_suspended(tmp_path):
    party, worker, _node = _stopped_poster_with_suspended_worker(tmp_path, "final-background")
    try:
        task_hash = _poster_task(
            party.node, worker, state="assigned", output="ok",
            verdict={"passed": True, "rules_passed": True, "reasons": [],
                     "pending_action": "accept"}, receive_state="pending")
        party.node._mark_poster_received = lambda _task: True
        party.node._retry_poster_receipts()
        assert party.node.store.task_get(task_hash, "poster", "")["state"] == "accepted"
        assert party.node.store.pending_for(worker) == []

        expired = _poster_task(party.node, worker, state="assigned")
        row = party.node.store.task_get(expired, "poster", "")
        row["spec"]["deadline"] = int(time.time()) - 1
        primary_key = party.node.store._pk((expired, "poster", ""))
        party.node.store._q(
            "UPDATE tasks SET spec=? WHERE task_hash=? AND role='poster' AND peer=''",
            (party.node.store._enc("tasks", "spec", primary_key, json.dumps(row["spec"])),
             expired))
        assert party.node.sweep() == 1
        assert party.node.store.task_get(expired, "poster", "")["state"] == "expired"
        assert party.node.store.pending_for(worker) == []
    finally:
        party.node.stop()


def test_pending_assignment_and_dispute_reconcile_keep_state_when_worker_is_suspended(
        tmp_path):
    party, worker, worker_node = _stopped_poster_with_suspended_worker(
        tmp_path, "final-ledger-background")
    try:
        party.node.settle.open_on_assign = True
        party.node.store.set_agent_card(worker, {
            "display_name": "Worker", "offers": "tasks", "intents": [], "paid": True,
            "price_hint": "", "models": [], "supports": [], "updated_at": 1,
            "ledger_keyid": worker_node,
        })
        assigned = _poster_task(party.node, worker, state="assigned", reward=5)
        party.node.settle.open = lambda *_args, **_kwargs: {"hold_id": "hold-finality"}
        party.node._finish_pending_assignments()
        row = party.node.store.task_get(assigned, "poster", "")
        assert row["state"] == "assigned" and row["hold_id"] == "hold-finality"
        assert party.node.store.pending_for(worker) == []

        disputed = _poster_task(
            party.node, worker, state="rejected", reward=5, hold_id="hold-disputed",
            verdict={"passed": False, "reasons": ["bad output"]})
        row = party.node.store.task_get(disputed, "poster", "")
        assert party.node._reconcile_poster_hold(
            row, {"state": "disputed"}, lambda _task: None) == 1
        assert party.node.store.task_get(
            disputed, "poster", "")["funding_state"] == "disputed"
        assert party.node.store.pending_for(worker) == []
    finally:
        party.node.stop()


def test_suspension_preflight_bypass_does_not_bypass_live_grant_check(tmp_path):
    party = Party(tmp_path, "grant-check", 9, caps=("nak.msg",))
    _private, person = _key()
    party.node.store.upsert_contact(person, "45" * 32, "active", "peer")
    try:
        with pytest.raises(ValueError, match="cap nak.task.post not granted"):
            party.node._queue(person, "task.post", {"task_hash": "46" * 32},
                              suspension_preflighted=True)
        assert party.node.store.outbox() == []
    finally:
        party.node.stop()


def test_cli_operator_suspend_refuses_without_tty_before_daemon_or_http(monkeypatch):
    preflights = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(nak, "broker_preflight", lambda kind, summary: preflights.append((kind, summary)))
    monkeypatch.setattr(nak, "call", lambda *_args, **_kwargs: pytest.fail("must not call nakd"))
    with pytest.raises(SystemExit, match="interactive-only"):
        nak.main(["operator", "suspend", "11" * 32, "--reason", "abuse"],
                 http_post=lambda *_args: pytest.fail("must not call HTTP"))
    assert preflights and preflights[0][0] == "suspend"


def test_operator_suspend_lost_response_retries_the_exact_pending_record(monkeypatch):
    private, person = _key()
    target = "71" * 32
    state = {"own": None, "pending": None}
    posted = []
    key_opens = []

    class FakePersonKey:
        def __init__(self, *_args):
            key_opens.append(True)

        def public_key(self):
            return private.public_key()

        def sign(self, statement):
            return private.sign(statement)

    @contextmanager
    def approved(_kind, _summary):
        yield

    def fake_call(_sock, request):
        if request["op"] == "contacts":
            return {"contacts": []}
        if request["op"] == "status":
            return {"person": person, "node": "70" * 32}
        if request["op"] == "suspensions":
            return {**state, "received": {}, "sent": {}, "operators": [], "suspended": []}
        if request["op"] == "stage_suspension":
            assert state["pending"] is None
            state["pending"] = request["record"]
            return {"seq": request["record"]["seq"], "new": True}
        if request["op"] == "install_suspension":
            assert request["record"] == state["pending"]
            state["own"], state["pending"] = state["pending"], None
            return {"seq": state["own"]["seq"]}
        raise AssertionError(request)

    def post(url, body):
        assert url.endswith("/operator/suspend")
        posted.append(body["record"])
        if len(posted) == 1:
            raise SystemExit("could not reach the onboard server: response lost")
        return {"suspended": [target], "restored": [], "seq": 1}

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "broker_preflight", lambda *_args: None)
    monkeypatch.setattr(nak, "approval", approved)
    monkeypatch.setattr(nak, "PersonKey", FakePersonKey)
    monkeypatch.setattr(nak, "call", fake_call)
    monkeypatch.setenv("NAK_ONBOARD_BASE", "https://onboard.invalid/base")
    monkeypatch.setenv("NAK_ONBOARD_TOKEN", "test-token")

    command = ["operator", "suspend", target, "--reason", "abuse", "--force-unknown-node"]
    with pytest.raises(SystemExit, match="response lost"):
        nak.main(command, http_post=post)
    pending = state["pending"]
    assert pending is not None and state["own"] is None
    assert nak.main(command, http_post=post) == 0
    assert posted == [pending, pending]
    assert state == {"own": pending, "pending": None}
    assert len(key_opens) == 2


def test_operator_suspend_stale_seq_adopts_signed_server_state_and_requests_rerun(monkeypatch):
    private, person = _key()
    old_node, target, server_node = "72" * 32, "73" * 32, "74" * 32
    own = _record(private, person, 1, (old_node,), {old_node: "old"}, now=100)
    remote = _record(private, person, 3, (old_node, server_node),
                     {old_node: "old", server_node: "server"}, now=300)
    state = {"own": own, "pending": None}
    fetched = []

    class FakePersonKey:
        def __init__(self, *_args):
            pass

        def public_key(self):
            return private.public_key()

        def sign(self, statement):
            return private.sign(statement)

    @contextmanager
    def approved(_kind, _summary):
        yield

    def fake_call(_sock, request):
        if request["op"] == "contacts":
            return {"contacts": []}
        if request["op"] == "status":
            return {"person": person, "node": "75" * 32}
        if request["op"] == "suspensions":
            return {**state, "received": {}, "sent": {}, "operators": [], "suspended": []}
        if request["op"] == "stage_suspension":
            state["pending"] = request["record"]
            return {"seq": request["record"]["seq"], "new": True}
        if request["op"] == "recover_suspension":
            state["own"], state["pending"] = request["record"], None
            return {"seq": request["record"]["seq"]}
        raise AssertionError(request)

    def post(url, _body):
        if url.endswith("/operator/suspend"):
            raise nak._OnboardHTTPError(409, "suspension record seq is stale")
        assert url.endswith("/operator/suspensions")
        fetched.append("operator")
        return {"record": remote}

    def get(url):
        fetched.append(url)
        return {"nodes": remote["nodes"], "seq": remote["seq"]}

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "broker_preflight", lambda *_args: None)
    monkeypatch.setattr(nak, "approval", approved)
    monkeypatch.setattr(nak, "PersonKey", FakePersonKey)
    monkeypatch.setattr(nak, "call", fake_call)
    monkeypatch.setenv("NAK_ONBOARD_BASE", "https://onboard.invalid/base")
    monkeypatch.setenv("NAK_ONBOARD_TOKEN", "test-token")

    with pytest.raises(SystemExit, match="advanced to seq 3.*re-run"):
        nak.main(["operator", "suspend", target, "--reason", "abuse", "--force-unknown-node"],
                 http_post=post, http_get=get)
    assert fetched == ["https://onboard.invalid/base/suspensions", "operator"]
    assert state == {"own": remote, "pending": None}


@pytest.mark.parametrize("status", [400, 403, 404])
def test_operator_suspend_definitive_refusal_clears_exact_pending_with_person_proof(
        monkeypatch, status):
    private, person = _key()
    target = "76" * 32
    state = {"own": None, "pending": None}
    abandoned = []

    class FakePersonKey:
        def __init__(self, *_args):
            pass

        def public_key(self):
            return private.public_key()

        def sign(self, statement):
            return private.sign(statement)

    @contextmanager
    def approved(_kind, _summary):
        yield

    def fake_call(_sock, request):
        if request["op"] == "contacts":
            return {"contacts": []}
        if request["op"] == "status":
            return {"person": person, "node": "75" * 32}
        if request["op"] == "suspensions":
            return {**state, "received": {}, "sent": {}, "operators": [], "suspended": []}
        if request["op"] == "stage_suspension":
            state["pending"] = request["record"]
            return {"seq": request["record"]["seq"], "new": True}
        if request["op"] == "abandon_suspension":
            assert request["record"] == state["pending"]
            private.public_key().verify(
                bytes.fromhex(request["signature"]),
                nakd.person_action_statement(
                    "abandon-suspension",
                    nakd.suspension_record_proof_payload(request["record"]),
                    request["ts"]))
            abandoned.append(request["record"])
            state["pending"] = None
            return {"seq": request["record"]["seq"], "abandoned": True}
        raise AssertionError(request)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "broker_preflight", lambda *_args: None)
    monkeypatch.setattr(nak, "approval", approved)
    monkeypatch.setattr(nak, "PersonKey", FakePersonKey)
    monkeypatch.setattr(nak, "call", fake_call)
    monkeypatch.setenv("NAK_ONBOARD_BASE", "https://onboard.invalid/base")
    monkeypatch.setenv("NAK_ONBOARD_TOKEN", "test-token")

    def post(_url, _body):
        raise nak._OnboardHTTPError(status, "node is not rostered")

    with pytest.raises(SystemExit, match=f"{status}.*node is not rostered.*pending.*cleared"):
        nak.main(["operator", "suspend", target, "--reason", "abuse",
                  "--force-unknown-node"], http_post=post)
    assert len(abandoned) == 1
    assert state == {"own": None, "pending": None}


def test_operator_suspend_manual_abandon_is_tty_and_person_proofed(monkeypatch):
    private, person = _key()
    pending = _record(private, person, 4, ("77" * 32,))
    state = {"own": None, "pending": pending}

    class FakePersonKey:
        def __init__(self, *_args):
            pass

        def public_key(self):
            return private.public_key()

        def sign(self, statement):
            return private.sign(statement)

    @contextmanager
    def approved(_kind, _summary):
        yield

    def fake_call(_sock, request):
        if request["op"] == "status":
            return {"person": person, "node": "75" * 32}
        if request["op"] == "suspensions":
            return {**state, "received": {}, "sent": {}, "operators": [], "suspended": []}
        if request["op"] == "abandon_suspension":
            private.public_key().verify(
                bytes.fromhex(request["signature"]),
                nakd.person_action_statement(
                    "abandon-suspension", nakd.suspension_record_proof_payload(pending),
                    request["ts"]))
            state["pending"] = None
            return {"seq": pending["seq"], "abandoned": True}
        raise AssertionError(request)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "broker_preflight", lambda *_args: None)
    monkeypatch.setattr(nak, "approval", approved)
    monkeypatch.setattr(nak, "PersonKey", FakePersonKey)
    monkeypatch.setattr(nak, "call", fake_call)
    assert nak.main(
        ["operator", "suspend", "--abandon-pending"],
        http_post=lambda *_args: pytest.fail("manual abandon must not call the server")) == 0
    assert state["pending"] is None


def test_raw_operator_target_requires_contact_roster_or_force(tmp_path, monkeypatch):
    unknown = "78" * 32
    rostered = "79" * 32
    contact_node = "7a" * 32
    contacts = [{"person": "7b" * 32, "node": contact_node, "petname": "known"}]
    status = {"person": "7c" * 32, "node": "7d" * 32}
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_call(_sock, request):
        assert request["op"] == "contacts"
        return {"contacts": contacts}

    monkeypatch.setattr(nak, "call", fake_call)
    with pytest.raises(SystemExit, match="neither a contact nor in the local admission roster"):
        nak._resolve_node_reference(Path("unused"), unknown, status)
    assert nak._resolve_node_reference(
        Path("unused"), unknown, status, force_unknown=True) == unknown
    assert nak._resolve_node_reference(Path("unused"), "known", status) == contact_node

    roster = tmp_path / ".nakshatra" / "admission-peers.tsv"
    roster.parent.mkdir(parents=True)
    roster.write_text(f"{rostered}\tmember\toperator\tknown\ttenant\tcoord\n")
    assert nak._resolve_node_reference(Path("unused"), rostered, status) == rostered


def test_operator_suspend_refuses_own_node_and_own_persons_contact(monkeypatch):
    own_person = "01" * 32
    own_node = "02" * 32
    sibling_node = "03" * 32
    status = {"person": own_person, "node": own_node}
    contacts = [{"person": own_person, "node": sibling_node, "petname": "my-other-node"}]

    def fake_call(_sock, request):
        if request["op"] == "contacts":
            return {"contacts": contacts}
        if request["op"] == "status":
            return status
        raise AssertionError(request)

    monkeypatch.setattr(nak, "call", fake_call)
    with pytest.raises(SystemExit, match="own node"):
        nak._resolve_node_reference(Path("unused"), own_node)
    with pytest.raises(SystemExit, match="own node"):
        nak._resolve_node_reference(Path("unused"), "my-other-node")


def test_cli_proof_timestamps_are_taken_after_slow_approval(monkeypatch):
    private, person = _key()
    remote = _record(private, person, 1)
    clock = [1000]
    action_requests = []

    @contextmanager
    def slow_approval(_kind, _summary):
        clock[0] += 200
        yield

    class FakePersonKey:
        def __init__(self, *_args):
            pass

        def sign(self, _statement):
            return b"s" * 64

    def fake_call(_sock, request):
        if request["op"] == "contacts":
            return {"contacts": [{"person": person, "node": "44" * 32,
                                   "petname": "operator"}]}
        action_requests.append(request)
        if request["op"] == "report_cli":
            return {"subject": request["subject"], "status": "received"}
        if request["op"] == "import_suspension":
            return {"seq": 1, "count": 1, "new": True}
        if request["op"] == "operator_trust":
            return {"person": person, "contact": "operator"}
        raise AssertionError(request)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "broker_preflight", lambda *_args: None)
    monkeypatch.setattr(nak, "approval", slow_approval)
    monkeypatch.setattr(nak, "PersonKey", FakePersonKey)
    monkeypatch.setattr(nak.time, "time", lambda: clock[0])
    monkeypatch.setattr(nak, "call", fake_call)

    assert nak.main(["report", "55" * 32, "--reason", "abuse"]) == 0
    assert nak.main(["suspensions", "import", encode_record(remote)]) == 0
    assert nak.main(["operator-trust", "operator"]) == 0
    assert [request["ts"] for request in action_requests] == [1200, 1400, 1600]


def test_report_op_signs_and_caps_at_ten(tmp_path, monkeypatch):
    party = Party(tmp_path, "reporter", 9)
    subject = "77" * 32
    seen = []

    def post(url, body):
        seen.append((url, body))
        unsigned = dict(body)
        signature = bytes.fromhex(unsigned.pop("sig"))
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(body["reporter"])).verify(
            signature, canonical(unsigned))
        return {"status": "recorded"}

    monkeypatch.setattr(nakd, "_http_post_json", post)
    monkeypatch.setenv("NAK_ONBOARD_BASE", "https://onboard.invalid/base")
    try:
        for index in range(10):
            result = party.node.report(subject, "abuse", f"report {index}", via="cli")
            assert result["status"] == "recorded"
        with pytest.raises(ValueError, match="10 per 24 hours"):
            party.node.report(subject, "abuse", "eleventh", via="cli")
        assert len(seen) == 10
        assert seen[0][0] == "https://onboard.invalid/base/report"
        assert seen[0][1]["via"] == "cli"
    finally:
        party.node.stop()


@pytest.mark.parametrize("via", ["cli", "mcp"])
def test_report_refuses_secret_note_and_malformed_hold_id_before_signing_or_quota(
        tmp_path, monkeypatch, via):
    party = Party(tmp_path, f"report-{via}", 9)
    _contact_private, contact_person = _key()
    party.node.store.upsert_contact(contact_person, "77" * 32, "active", "subject")
    monkeypatch.setattr(
        nakd, "_http_post_json", lambda *_args: pytest.fail("refused report must not use HTTP"))
    try:
        with pytest.raises(ValueError, match="secret-shaped token"):
            party.node.report("subject", "abuse", "leaked sk-proj-" + "x" * 24, via=via)
        with pytest.raises(ValueError, match="hold_id must be an identifier"):
            party.node.report("subject", "abuse", evidence={"hold_id": "not a hold/id"}, via=via)
        assert party.node.store._q(
            "SELECT value FROM store_meta WHERE key='report_rate_v1'") == []
    finally:
        party.node.stop()


def test_failed_report_posts_refund_local_quota(tmp_path, monkeypatch):
    party = Party(tmp_path, "reporter", 9)
    subject = "77" * 32
    attempts = 0

    def post(_url, _body):
        nonlocal attempts
        attempts += 1
        if attempts <= 10:
            raise ValueError("server unavailable")
        return {"status": "recorded"}

    monkeypatch.setattr(nakd, "_http_post_json", post)
    try:
        for _index in range(10):
            with pytest.raises(ValueError, match="server unavailable"):
                party.node.report(subject, "abuse", via="cli")
        for _index in range(10):
            assert party.node.report(subject, "abuse", via="cli")["status"] == "recorded"
        with pytest.raises(ValueError, match="10 per 24 hours"):
            party.node.report(subject, "abuse", via="cli")
        assert attempts == 20
    finally:
        party.node.stop()


def test_raw_report_via_cli_spoof_is_mcp_and_person_proof_earns_cli(tmp_path, monkeypatch):
    party = Party(tmp_path, "reporter", 9)
    contact_private, contact_person = _key()
    del contact_private
    subject_node = "ab" * 32
    party.node.store.upsert_contact(contact_person, subject_node, "active", "subject")
    seen = []
    monkeypatch.setattr(nakd, "_http_post_json",
                        lambda _url, body: seen.append(body) or {"status": "recorded"})
    try:
        spoof = nakd.handle(party.node, {"op": "report", "subject": "subject",
                                         "reason": "abuse", "via": "cli"})
        assert spoof["ok"] and seen[-1]["via"] == "mcp"
        bare = nakd.handle(party.node, {"op": "report", "subject": "cd" * 32,
                                       "reason": "abuse", "via": "cli"})
        assert not bare["ok"] and "existing contact" in bare["error"]

        payload = nakd.report_proof_payload("cd" * 32, "fraud", "", {})
        proved = nakd.handle(party.node, {
            "op": "report_cli", "subject": "cd" * 32, "reason": "fraud",
            **_proof(party.person_priv, "report-cli", payload),
        })
        assert proved["ok"] and seen[-1]["via"] == "cli"
    finally:
        party.node.stop()


def test_operator_reports_escape_and_mark_every_untrusted_line(monkeypatch, capsys):
    monkeypatch.setenv("NAK_ONBOARD_TOKEN", "test-token")
    malicious = {"received": "now\x1b[2J", "reporter": "aa" * 32,
                 "subject": "bb" * 32, "reason": "abuse\rFORGED",
                 "via": "mcp", "note": "first\nFORGED\x1b[31mred"}
    assert nak.main(["operator", "reports"],
                    http_post=lambda *_args: {"reports": [malicious]}) == 0
    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert len(output.splitlines()) == 1
    assert output.startswith("UNTRUSTED REPORT ")
    assert r"\nFORGED" in output and r"\rFORGED" in output and r"\x1b[31m" in output


def test_mcp_report_is_opt_in_and_rejects_non_contact(tmp_path):
    party = Party(tmp_path, "mcp-reporter", 9)

    class LocalClient:
        def report(self, subject, reason, note="", *, task_hash="", mid="", hold_id=""):
            try:
                evidence = {key: value for key, value in
                            (("task_hash", task_hash), ("mid", mid), ("hold_id", hold_id)) if value}
                return party.node.report(subject, reason, note, evidence, via="mcp")
            except ValueError as exc:
                raise NakError(str(exc)) from exc

    try:
        assert "nak_report" not in nak_mcp.Server(LocalClient()).tools
        server = nak_mcp.Server(LocalClient(), allow_connect=True)
        assert "nak_report" in server.tools
        with pytest.raises(NakError, match="existing contact"):
            server.call("nak_report", {"subject": "11" * 32, "reason": "abuse"})
    finally:
        party.node.stop()


@pytest.mark.parametrize("reason", ["leaked sk-proj-" + "A" * 24, "code nki1.abc", "-----BEGIN key"])
def test_suspension_reason_never_carries_a_secret_out(reason):
    node = "11" * 32
    with pytest.raises(ValueError, match="secret-shaped token or encoded blob"):
        _record(nodes=(node,), reasons={node: reason})

    # A record signed elsewhere with such a reason is refused on receipt too, never installed or relayed.
    private, person = _key()
    record = {"schema": "nak.suspensions.v1", "person": person, "seq": 1, "issued_at": 123,
              "nodes": [node], "reasons": {node: reason}, "sig": "0" * 128}
    from network.suspension import suspension_statement as statement
    record["sig"] = private.sign(statement(record)).hex()
    with pytest.raises(ValueError, match="secret-shaped token or encoded blob"):
        verify_record(record, person)
    with pytest.raises(ValueError):
        decode_record(encode_record(record))
