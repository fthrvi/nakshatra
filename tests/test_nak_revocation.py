"""Shared, person-signed grant revocation records and session propagation."""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
for candidate in (os.environ.get("NAK_STHAMBHA_PATH", ""), "/home/prithvi/sthambha"):
    if candidate and (Path(candidate) / "sthambha" / "signer.py").exists():
        sys.path.insert(0, candidate)
        break

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402
from sthambha import signer as signer_mod  # noqa: E402

from network import nak, nak_mcp, nakd  # noqa: E402
from network.revocation import (MAX_BLOB, MAX_GRANTS, MAX_SEQ, decode_record, encode_record,
                                make_record, revocation_statement, verify_record)  # noqa: E402
from network.store import Store  # noqa: E402
from transport.relay import RendezvousRelay  # noqa: E402
from test_nakd import Party, wait  # noqa: E402


@pytest.fixture
def net(tmp_path):
    relay = RendezvousRelay("127.0.0.1", 0, max_per_ip_per_min=100000)
    port = relay.start()
    parties = []

    def make(name, **kwargs):
        party = Party(tmp_path, name, port, roster_loader=lambda: {
            item.node.node: {"tier": item.tier} for item in parties}, **kwargs)
        parties.append(party)
        return party

    yield make
    for party in parties:
        party.node.stop()
    relay.stop()


def _key():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, private.public_key().public_bytes_raw().hex()


def _record(private=None, person=None, seq=1, ids=("0123456789abcdef",), now=1_700_000_000):
    private = private or ed25519.Ed25519PrivateKey.generate()
    person = person or private.public_key().public_bytes_raw().hex()
    return make_record(private.sign, person, seq, ids, now)


def _resign(record, private):
    out = dict(record)
    out["sig"] = private.sign(revocation_statement(out)).hex()
    return out


def _cli_home(monkeypatch, tmp_path):
    home = tmp_path
    net_dir = home / ".nakshatra" / "net"
    signer_dir = home / ".sthambha" / "signer"
    net_dir.mkdir(parents=True)
    signer_dir.mkdir(parents=True)
    (net_dir / "agent").write_text("pa")
    private, person = _key()
    signer_mod.save_key(home / ".sthambha" / "person-TEST.key", private)
    (signer_dir / "person.pub").write_text(person)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    return home, net_dir, signer_dir, private, person


def _daemon_material(tmp_path, name):
    root = tmp_path / name
    state_dir = root / "net"
    state_dir.mkdir(parents=True, mode=0o700)
    node_private = ed25519.Ed25519PrivateKey.generate()
    node_key = node_private.private_bytes_raw()
    node_pub = node_private.public_key().public_bytes_raw().hex()
    person_private, person_pub = signer_mod.new_key()
    signer_dir = root / "signer"
    (signer_dir / "agents").mkdir(parents=True, mode=0o700)
    (signer_dir / "node.pub").write_text(node_pub)
    (signer_dir / "person.pub").write_text(person_pub)
    agent_private, agent_pub = signer_mod.new_key()
    signer_mod.save_key(signer_dir / "agents" / "pa.key", agent_private)
    grant = signer_mod.issue_grant(
        person_private, "pa", agent_pub, ("nak.msg",), node=node_pub, ttl_s=7 * 86400)
    (signer_dir / "agents" / "pa.grant.json").write_text(json.dumps(grant))
    signer = signer_mod.Signer(signer_dir, custody="test")
    return state_dir, node_key, signer, person_private, person_pub


def test_record_and_blob_round_trip():
    private, person = _key()
    record = make_record(private.sign, person, 3,
                         ["fedcba9876543210", "0123456789abcdef", "0123456789abcdef"], 123)
    assert record["grant_ids"] == ["0123456789abcdef", "fedcba9876543210"]
    assert verify_record(record, person) == record
    assert decode_record(encode_record(record)) == record


@pytest.mark.parametrize("mutate", [
    lambda r: dict(r, schema="nak.revocations.v0"),
    lambda r: {key: value for key, value in r.items() if key != "issued_at"},
    lambda r: dict(r, extra=True),
    lambda r: dict(r, person="A" * 64),
    lambda r: dict(r, seq=0),
    lambda r: dict(r, seq=True),
    lambda r: dict(r, issued_at=True),
    lambda r: dict(r, grant_ids=["ABCDEF0123456789"]),
    lambda r: dict(r, grant_ids=["fedcba9876543210", "0123456789abcdef"]),
    lambda r: dict(r, grant_ids=["0123456789abcdef", "0123456789abcdef"]),
    lambda r: dict(r, grant_ids=[f"{value:016x}" for value in range(4097)]),
    lambda r: dict(r, sig="0" * 127),
])
def test_verify_record_rejects_every_schema_defect(mutate):
    private, person = _key()
    record = mutate(_record(private, person))
    if (set(record) == {"schema", "person", "seq", "issued_at", "grant_ids", "sig"}
            and isinstance(record.get("sig"), str) and len(record["sig"]) == 128):
        try:
            record = _resign(record, private)
        except Exception:
            pass
    with pytest.raises(ValueError):
        verify_record(record, person)


def test_verify_record_rejects_person_mismatch_and_bad_signature():
    private, person = _key()
    record = _record(private, person)
    _other_private, other_person = _key()
    with pytest.raises(ValueError, match="another person"):
        verify_record(record, other_person)
    record["sig"] = ("0" if record["sig"][0] != "0" else "1") + record["sig"][1:]
    with pytest.raises(ValueError, match="does not verify"):
        verify_record(record, person)


def test_verify_record_rejects_sequence_above_json_safe_integer():
    private, person = _key()
    record = _record(private, person)
    record["seq"] = 2**63
    record = _resign(record, private)
    with pytest.raises(ValueError, match=f"1 through {MAX_SEQ}"):
        verify_record(record, person)


def test_blob_rejects_noncanonical_and_over_size():
    record = _record()
    blob = encode_record(record)
    with pytest.raises(ValueError):
        decode_record(blob + "=")
    with pytest.raises(ValueError, match="256 KiB"):
        decode_record("nkr1." + "a" * MAX_BLOB)


def test_store_union_stale_and_sent_semantics(tmp_path):
    store = Store(tmp_path / "store", b"k" * 32)
    private, person = _key()
    first = _record(private, person, 2, ("1111111111111111", "2222222222222222"))
    later = _record(private, person, 3, ("2222222222222222",))
    stale = _record(private, person, 1, ("3333333333333333",))
    assert store.apply_contact_revocation(person, first)
    assert store.apply_contact_revocation(person, later)
    assert not store.apply_contact_revocation(person, stale)
    assert store.revoked_grants(person) == {"1111111111111111", "2222222222222222"}
    assert store.contact_revocation_seq(person) == 3
    store.mark_revocation_sent(person, 3)
    store.mark_revocation_sent(person, 2)
    assert store.revocation_sent_seq(person) == 3


def test_store_rejects_cumulative_union_over_limit_without_changing_state(tmp_path):
    store = Store(tmp_path / "bounded", b"b" * 32)
    private, person = _key()
    first_ids = tuple(f"{value:016x}" for value in range(MAX_GRANTS))
    second_ids = tuple(f"{value:016x}" for value in range(MAX_GRANTS, 2 * MAX_GRANTS))
    assert store.apply_contact_revocation(person, _record(private, person, 1, first_ids))
    with pytest.raises(ValueError, match=f"exceeds {MAX_GRANTS}"):
        store.apply_contact_revocation(person, _record(private, person, 2, second_ids))
    assert store.contact_revocation_seq(person) == 1
    assert store.revoked_grants(person) == set(first_ids)


def _pair(net):
    a, b = net("rev-a"), net("rev-b")
    b.node.redeem(a.invite(), wait_s=15)
    a.node.accept(a.node.store.requests()[0]["id"], petname="b")
    assert wait(lambda: (b.node.store.contact(a.person_pub) or {}).get("state") == "active")
    assert wait(lambda: a.person_pub in b.node._sessions and b.person_pub in a.node._sessions)
    return a, b


def test_wrong_person_and_malformed_revs_do_not_kill_session(net):
    a, b = _pair(net)
    wrong_private, _wrong_person = _key()
    forged = _record(wrong_private, a.person_pub, 1)
    channel, lock = a.node._sessions[b.person_pub]
    nakd.send_frame(channel, {"t": "revs", "rec": forged}, lock)
    nakd.send_frame(channel, {"t": "revs", "rec": {"broken": True}}, lock)
    time.sleep(0.2)
    assert b.node.store.revoked_grants(a.person_pub) == set()
    mid = a.node.send("b", "session survived invalid revocation frames")["queued"]
    assert wait(lambda: a.node.delivered(mid))


def test_revoked_envelope_nacked_but_fresh_grant_verifies(net):
    a, b = _pair(net)
    body = {"mid": "ab" * 12, "text": "copied key", "aspect": ""}
    old_env = a.node._sign("msg", b.person_pub, body, touch=False)
    old_id = old_env["grant"]["grant_id"]
    b.node.store.apply_contact_revocation(
        a.person_pub, _record(a.person_priv, a.person_pub, 1, (old_id,)))
    refused = b.node._receive_checked(a.person_pub, {"env": old_env, "body": body})
    assert refused["t"] == "nack" and refused["why"] == "does not verify: grant revoked"

    grant_path = a.dir / "signer" / "agents" / "pa.grant.json"
    current = json.loads(grant_path.read_text())
    _agent_private, agent_pub = signer_mod.load_key(a.dir / "signer" / "agents" / "pa.key")
    fresh = signer_mod.issue_grant(a.person_priv, "pa", agent_pub, current["caps"], node=a.node.node,
                                   ttl_s=7 * 86400)
    grant_path.write_text(json.dumps(fresh))
    a.signer = signer_mod.Signer(a.dir / "signer", custody="test")
    fresh_body = {"mid": "cd" * 12, "text": "fresh grant", "aspect": ""}
    fresh_env = a.node._sign("msg", b.person_pub, fresh_body, touch=False)
    assert b.node._receive_checked(a.person_pub, {"env": fresh_env, "body": fresh_body})["t"] == "ack"


def test_end_to_end_push_revokes_and_ack_records_sent_seq(net):
    a, b = _pair(net)
    body = {"mid": "ef" * 12, "text": "stolen sender", "aspect": ""}
    copied_env = a.node._sign("msg", b.person_pub, body, touch=False)
    grant_id = copied_env["grant"]["grant_id"]
    record = _record(a.person_priv, a.person_pub, 1, (grant_id,))
    assert a.node.store.revocation_sent_seq(b.person_pub) == 0
    a.node.install_own_revocation(record)
    assert wait(lambda: b.node.store.revoked_grants(a.person_pub) == {grant_id})
    assert wait(lambda: a.node.store.revocation_sent_seq(b.person_pub) == 1)
    reply = b.node._receive_checked(a.person_pub, {"env": copied_env, "body": body})
    assert reply == {"t": "nack", "mid": body["mid"], "why": "does not verify: grant revoked"}


def test_unsent_ack_does_not_advance_state(net):
    a, b = _pair(net)
    channel, lock = b.node._sessions[a.person_pub]
    nakd.send_frame(channel, {"t": "revs-ack", "seq": 999}, lock)
    time.sleep(0.2)
    assert a.node.store.revocation_sent_seq(b.person_pub) == 0


def test_unacked_revocation_is_sent_only_once_per_session(net, monkeypatch):
    a, b = _pair(net)
    original_send = nakd.send_frame
    sent = []

    def without_revocation_acks(channel, frame, lock=None):
        if frame.get("t") == "revs-ack":
            return None
        if frame.get("t") == "revs":
            sent.append(frame["rec"]["seq"])
        return original_send(channel, frame, lock)

    monkeypatch.setattr(nakd, "send_frame", without_revocation_acks)
    a.node.install_own_revocation(_record(a.person_priv, a.person_pub, 1))
    assert wait(lambda: sent == [1])
    assert wait(lambda: b.node.store.contact_revocation_seq(a.person_pub) == 1)
    for _ in range(20):
        a.node._wake[b.person_pub].set()
        time.sleep(0.01)
    assert sent == [1]
    assert a.node.store.revocation_sent_seq(b.person_pub) == 0
    a.node.install_own_revocation(_record(
        a.person_priv, a.person_pub, 2, ("0123456789abcdef", "fedcba9876543210")))
    assert wait(lambda: sent == [1, 2])
    for _ in range(20):
        a.node._wake[b.person_pub].set()
        time.sleep(0.01)
    assert sent == [1, 2]


def test_install_requires_own_records_to_remain_cumulative(tmp_path):
    private, person = _key()
    node = object.__new__(nakd.Node)
    node.person = person
    node.store = Store(tmp_path / "cumulative", b"c" * 32)
    node._wake = {}
    node.install_own_revocation(_record(
        private, person, 1, ("1111111111111111",)))
    with pytest.raises(ValueError, match="retain every"):
        node.install_own_revocation(_record(
            private, person, 2, ("2222222222222222",)))


def test_racing_out_of_order_own_installs_keep_the_higher_sequence(tmp_path, monkeypatch):
    private, person = _key()
    store = Store(tmp_path / "own-cas", b"o" * 32)
    node = object.__new__(nakd.Node)
    node.person = person
    node.store = store
    node._wake = {}
    start = threading.Barrier(2)
    reads = threading.Barrier(2)
    errors = []
    original_own = store.own_revocation

    # This hook makes the pre-fix read/check/write implementation lose deterministically: both
    # callers read the old value, seq 3 writes first, and seq 2 writes last. The fixed Node no
    # longer performs this read; its store transaction serializes both complete CAS operations.
    def synchronized_old_read():
        current = original_own()
        reads.wait(2)
        if threading.current_thread().name == "install-low":
            time.sleep(0.05)
        return current

    monkeypatch.setattr(store, "own_revocation", synchronized_old_read)

    def install_high():
        start.wait(2)
        node.install_own_revocation(_record(
            private, person, 3, ("2222222222222222", "3333333333333333")))

    def install_low():
        start.wait(2)
        try:
            node.install_own_revocation(_record(private, person, 2, ("2222222222222222",)))
        except ValueError as exc:
            errors.append(str(exc))

    high = threading.Thread(target=install_high, name="install-high")
    low = threading.Thread(target=install_low, name="install-low")
    high.start()
    low.start()
    high.join()
    low.join()
    assert errors in ([], ["revocation seq must be newer than the installed record"])
    assert original_own()["seq"] == 3
    assert original_own()["grant_ids"] == ["2222222222222222", "3333333333333333"]


@pytest.mark.parametrize("listener_kind", ["invite", "address"])
def test_known_contact_cannot_request_again_under_a_revoked_grant(net, listener_kind):
    inviter, known = net(f"known-{listener_kind}-inviter"), net(f"known-{listener_kind}-peer")
    inviter.node.store.upsert_contact(known.person_pub, known.node.node, "active", "known")
    listener_id = "12" * 16
    if listener_kind == "invite":
        inviter.node.store.add_invite(listener_id, "34" * 16, int(time.time()) + 60)
    else:
        inviter.node.store.set_address("stored", listener_id, "34" * 16,
                                       int(time.time()) + 60)
    body = {"person": known.person_pub, "node": known.node.node, "nickname": "Known",
            listener_kind: listener_id, "mid": "56" * 12}
    env = known.node._sign("contact.request", inviter.person_pub, body, touch=False)
    grant_id = env["grant"]["grant_id"]
    inviter.node.store.apply_contact_revocation(
        known.person_pub, _record(known.person_priv, known.person_pub, 1, (grant_id,)))
    reply = inviter.node._take_request(
        {"t": "env", "env": env, "body": body}, listener_kind, listener_id)
    assert reply == {"t": "nack", "why": "request does not verify: grant revoked"}
    assert inviter.node.store.requests() == []
    if listener_kind == "invite":
        assert inviter.node.store.invite(listener_id)["consumed_by"] is None


def test_oversized_union_and_sequence_overflow_do_not_kill_session_or_import(net):
    a, b = _pair(net)
    first_ids = tuple(f"{value:016x}" for value in range(MAX_GRANTS))
    second_ids = tuple(f"{value:016x}" for value in range(MAX_GRANTS, 2 * MAX_GRANTS))
    b.node.store.apply_contact_revocation(
        a.person_pub, _record(a.person_priv, a.person_pub, 1, first_ids))
    channel, lock = a.node._sessions[b.person_pub]
    nakd.send_frame(channel, {"t": "revs", "rec": _record(
        a.person_priv, a.person_pub, 2, second_ids)}, lock)

    overflow = _record(a.person_priv, a.person_pub, 2, ("ffffffffffffffff",))
    overflow["seq"] = 2**63
    overflow = _resign(overflow, a.person_priv)
    with pytest.raises(ValueError, match=f"1 through {MAX_SEQ}"):
        verify_record(overflow, a.person_pub)
    nakd.send_frame(channel, {"t": "revs", "rec": overflow}, lock)
    time.sleep(0.2)
    assert b.node.store.contact_revocation_seq(a.person_pub) == 1
    assert b.node.store.revoked_grants(a.person_pub) == set(first_ids)
    mid = a.node.send("b", "session survived bounded revocation failures")["queued"]
    assert wait(lambda: a.node.delivered(mid))

    raw = json.dumps(overflow, sort_keys=True, separators=(",", ":")).encode()
    blob = "nkr1." + base64.urlsafe_b64encode(raw).decode().rstrip("=")
    result = nakd.handle(b.node, {"op": "import_revocation", "blob": blob})
    assert result["ok"] is False
    assert f"1 through {MAX_SEQ}" in result["error"]


def test_cli_revoke_refuses_without_tty(monkeypatch, tmp_path):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="interactive-only"):
        nak.main(["revoke", "0123456789abcdef"])


def test_cli_revoke_preflight_summary_is_plain_and_names_every_requested_id(monkeypatch):
    summaries = []
    monkeypatch.setattr(nak, "broker_preflight",
                        lambda kind, summary: summaries.append((kind, summary)))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="interactive-only"):
        nak.main(["revoke", "0123456789abcdef", "--grant-id", "fedc\n\u202eba9876543210"])
    assert summaries == [("revoke", "Revoke grants 0123456789abcdef, fedcba9876543210")]


def test_cli_revoke_refuses_without_person_key(monkeypatch, tmp_path):
    home = tmp_path
    net_dir = home / ".nakshatra" / "net"
    net_dir.mkdir(parents=True)
    (net_dir / "agent").write_text("pa")
    signer_dir = home / ".sthambha" / "signer"
    signer_dir.mkdir(parents=True)
    _private, person = _key()
    (signer_dir / "person.pub").write_text(person)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    with pytest.raises(SystemExit, match="cannot load the person key for this decision"):
        nak.main(["revoke", "0123456789abcdef"])
    assert not (signer_dir / "revoked.txt").exists()


def test_cli_revoke_persists_signed_record_while_daemon_is_down(monkeypatch, tmp_path, capsys):
    _home, net_dir, signer_dir, _private, person = _cli_home(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "call", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        SystemExit("refused: the Nakshatra messaging service is not running")))
    assert nak.main(["revoke", "0123456789abcdef"]) == 0
    assert (signer_dir / "revoked.txt").read_text() == "0123456789abcdef\n"
    pending = json.loads((net_dir / "pending-revocation.json").read_text())
    assert verify_record(pending, person)["grant_ids"] == ["0123456789abcdef"]
    assert "will be installed when it starts" in capsys.readouterr().out


def test_concurrent_cli_revokes_preserve_both_ids_and_advance_twice(monkeypatch, tmp_path):
    _home, net_dir, signer_dir, _private, person = _cli_home(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(nak, "call", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        SystemExit("refused: the Nakshatra messaging service is not running")))
    start = threading.Barrier(2)
    results = []
    errors = []

    def revoke(grant_id):
        start.wait()
        try:
            results.append(nak.main(["revoke", grant_id]))
        except BaseException as exc:  # preserve thread failures for the main assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=revoke, args=("aaaaaaaaaaaaaaaa",)),
        threading.Thread(target=revoke, args=("bbbbbbbbbbbbbbbb",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted(results) == [0, 0]
    pending = verify_record(json.loads((net_dir / "pending-revocation.json").read_text()), person)
    assert pending["seq"] == 2
    assert pending["grant_ids"] == ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"]
    assert set((signer_dir / "revoked.txt").read_text().splitlines()) == set(pending["grant_ids"])
    lock = net_dir / "pending-revocation.lock"
    assert lock.is_file() and (lock.stat().st_mode & 0o777) == 0o600


def test_cli_reports_daemon_install_cas_loss_as_failure(monkeypatch, tmp_path):
    _home, net_dir, _signer_dir, private, person = _cli_home(monkeypatch, tmp_path)
    store = Store(net_dir, b"r" * 32)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

    def lose_install(_sock, request):
        proposed = request["record"]
        store.set_own_revocation(_record(
            private, person, proposed["seq"] + 1,
            tuple(proposed["grant_ids"]) + ("cccccccccccccccc",)))
        raise SystemExit("refused: revocation seq must be newer than the installed record")

    monkeypatch.setattr(nak, "call", lose_install)
    with pytest.raises(SystemExit, match="newer record exists; re-run nak revoke"):
        nak.main(["revoke", "aaaaaaaaaaaaaaaa"])
    assert store.own_revocation()["seq"] == 2
    assert not (net_dir / "pending-revocation.json").exists()


def test_cli_revoke_refuses_unsigned_pending_state_before_writing(monkeypatch, tmp_path):
    _home, net_dir, signer_dir, _private, person = _cli_home(monkeypatch, tmp_path)
    planted = {"schema": "nak.revocations.v1", "person": person, "seq": 9,
               "issued_at": 1, "grant_ids": ["aaaaaaaaaaaaaaaa"], "sig": "0" * 128}
    pending = net_dir / "pending-revocation.json"
    pending.write_text(json.dumps(planted))
    pending.chmod(0o600)
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("must refuse before prompting"))
    with pytest.raises(SystemExit, match=rf"{re.escape(str(pending))}.*does not verify"):
        nak.main(["revoke", "bbbbbbbbbbbbbbbb"])
    assert not (signer_dir / "revoked.txt").exists()


def test_cli_revoke_refuses_symlinked_pending_state(monkeypatch, tmp_path):
    _home, net_dir, signer_dir, private, person = _cli_home(monkeypatch, tmp_path)
    target = tmp_path / "signed.json"
    target.write_text(json.dumps(_record(private, person)))
    target.chmod(0o600)
    pending = net_dir / "pending-revocation.json"
    pending.symlink_to(target)
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("must refuse before prompting"))
    with pytest.raises(SystemExit, match=re.escape(str(pending))):
        nak.main(["revoke", "bbbbbbbbbbbbbbbb"])
    assert not (signer_dir / "revoked.txt").exists()


def test_cli_revoke_confirmation_lists_full_resulting_set(monkeypatch, tmp_path, capsys):
    _home, net_dir, signer_dir, private, person = _cli_home(monkeypatch, tmp_path)
    pending = net_dir / "pending-revocation.json"
    pending.write_text(json.dumps(_record(private, person, 1, ("1111111111111111",))))
    pending.chmod(0o600)
    (signer_dir / "revoked.txt").write_text("2222222222222222\n")
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "no")
    assert nak.main(["revoke", "3333333333333333"]) == 1
    assert "1111111111111111 (previously revoked)" in prompts[0]
    assert "2222222222222222 (previously revoked)" in prompts[0]
    assert "3333333333333333 (new)" in prompts[0]
    assert "nothing revoked" in capsys.readouterr().out


def test_daemon_ignores_symlinked_pending_revocation_and_keeps_starting(tmp_path):
    state_dir, node_key, signer, private, person = _daemon_material(tmp_path, "pending-link")
    target = tmp_path / "signed-revocation.json"
    target.write_text(json.dumps(_record(private, person)))
    target.chmod(0o600)
    pending = state_dir / "pending-revocation.json"
    pending.symlink_to(target)
    logs = []

    node = nakd.Node(state_dir, node_key, signer.handle, agent="pa",
                     relay=("127.0.0.1", 9), log=logs.append)
    try:
        assert node.store.own_revocation() is None
        assert pending.is_symlink()
        assert any("leaving unreadable pending revocation record" in line for line in logs)
    finally:
        node.stop()


def test_daemon_rejects_oversize_pending_before_reading_and_keeps_starting(tmp_path):
    state_dir, node_key, signer, _private, _person = _daemon_material(tmp_path, "pending-large")
    pending = state_dir / "pending-revocation.json"
    with pending.open("wb") as out:
        out.truncate(MAX_BLOB + 1)
    pending.chmod(0o600)
    logs = []

    node = nakd.Node(state_dir, node_key, signer.handle, agent="pa",
                     relay=("127.0.0.1", 9), log=logs.append)
    try:
        assert node.store.own_revocation() is None
        assert pending.stat().st_size == MAX_BLOB + 1
        assert any(f"exceeds the {MAX_BLOB}-byte size limit" in line for line in logs)
    finally:
        node.stop()


def test_mcp_exposes_no_revoke_tool():
    assert all("revoke" not in name and "revocation" not in name
               for name in set(nak_mcp.READ_TOOLS) | set(nak_mcp.CONNECT_TOOLS))
