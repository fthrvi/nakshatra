"""Contact-visible person-key custody is a node claim, never person authority."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

_REPO = Path(__file__).resolve().parent.parent
for _candidate in (os.environ.get("NAK_STHAMBHA_PATH", ""), str(Path.home() / "sthambha"),
                   str(Path.home() / "sthambha-signer"), str(_REPO.parent / "sthambha")):
    if _candidate and (Path(_candidate) / "sthambha" / "signer.py").exists():
        sys.path.insert(0, _candidate)
        break

from network import custody_card, nak, nak_mcp, nakd, person_custody
from network.store import Store


def _keypair():
    private = ed25519.Ed25519PrivateKey.generate()
    return private, private.public_key().public_bytes_raw().hex()


def _statement(*, now=1_800_000_000, mode="file"):
    node_key, node = _keypair()
    _person_key, person = _keypair()
    statement = custody_card.issue(
        person, mode, node_key, since=now - 3 * 86400, issued_at=now)
    return statement, person, node, node_key


def test_mode_derivation_validates_file_and_broker_configuration(tmp_path):
    net = tmp_path / "net"
    net.mkdir()
    assert person_custody.custody_statement_mode(net_dir=net) == "file"
    marker = net / "person-custody"
    marker.write_text("file\n")
    marker.chmod(0o600)
    assert person_custody.custody_statement_mode(net_dir=net) == "file"

    token = tmp_path / "broker.token"
    token.write_text("opaque-test-token\n")
    token.chmod(0o600)
    config = net / "person-broker.json"
    config.write_text(json.dumps({
        "url": "http://127.0.0.1:7788", "token_file": str(token),
    }))
    config.chmod(0o600)
    marker.write_text("broker\n")
    assert person_custody.custody_statement_mode(net_dir=net) == "broker-biometric"


def test_broker_config_without_marker_refuses_file_fallback(tmp_path):
    net = tmp_path / "net"
    net.mkdir()
    config = net / "person-broker.json"
    config.write_text("{}")
    config.chmod(0o600)
    with pytest.raises(SystemExit, match="marker is missing|refusing file custody"):
        person_custody.custody_statement_mode(net_dir=net)


def test_statement_verifies_against_known_contact_node():
    statement, person, node, _ = _statement()
    assert custody_card.verify(
        statement, expected_person=person, expected_node=node,
        now=statement["issued_at"])["mode"] == "file"


def test_statement_refuses_bad_node_signature():
    statement, person, node, _ = _statement()
    statement["sig"] = ("00" if statement["sig"][:2] != "00" else "01") + statement["sig"][2:]
    with pytest.raises(ValueError, match="signature does not verify"):
        custody_card.verify(
            statement, expected_person=person, expected_node=node,
            now=statement["issued_at"])


def test_statement_refuses_wrong_person_even_with_valid_node_signature():
    statement, _person, node, _ = _statement()
    _other_key, other_person = _keypair()
    with pytest.raises(ValueError, match="person is not this contact"):
        custody_card.verify(
            statement, expected_person=other_person, expected_node=node,
            now=statement["issued_at"])


def test_statement_refuses_expiry():
    statement, person, node, _ = _statement()
    with pytest.raises(ValueError, match="expired"):
        custody_card.verify(
            statement, expected_person=person, expected_node=node,
            now=statement["expires_at"])


@pytest.mark.parametrize(("change", "message"), [
    (lambda s: s.update(v=2), "version"),
    (lambda s: s.update(kind="nak.future.v1"), "unknown custody statement kind"),
    (lambda s: s.update(person="not-a-key"), "person is malformed"),
    (lambda s: s.update(mode="hardware-ish"), "mode is unsupported"),
    (lambda s: s.update(since=True), "since is malformed"),
    (lambda s: s.update(since=s["issued_at"] + 1), "since is after"),
    (lambda s: s.update(issued_at=s["issued_at"] + 301), "issuance is in the future"),
    (lambda s: s.update(expires_at=s["issued_at"]), "expiry is not after"),
    (lambda s: s.update(expires_at=s["issued_at"] + custody_card.MAX_TTL_S + 1), "exceeds 30 days"),
    (lambda s: s.update(extra=True), "fields are missing or unknown"),
    (lambda s: s.pop("mode"), "fields are missing or unknown"),
])
def test_statement_refuses_every_invalid_shape_without_crashing(change, message):
    statement, person, node, node_key = _statement()
    change(statement)
    statement["sig"] = node_key.sign(custody_card.canonical(statement)).hex()
    with pytest.raises(ValueError, match=message):
        custody_card.verify(
            statement, expected_person=person, expected_node=node,
            now=1_800_000_000)


def test_statement_refuses_non_object_bad_signature_encoding_and_bad_known_node():
    statement, person, node, _ = _statement()
    with pytest.raises(ValueError, match="not an object"):
        custody_card.verify([], expected_person=person, expected_node=node, now=statement["issued_at"])
    statement["sig"] = "not-hex"
    with pytest.raises(ValueError, match="signature is malformed"):
        custody_card.verify(
            statement, expected_person=person, expected_node=node, now=statement["issued_at"])
    statement, person, _node, _ = _statement()
    with pytest.raises(ValueError, match="contact node key is malformed"):
        custody_card.verify(
            statement, expected_person=person, expected_node="bad", now=statement["issued_at"])


def test_store_ignores_older_statement(tmp_path):
    store = Store(tmp_path / "state", b"d" * 32)
    node_key, _node = _keypair()
    _person_key, person = _keypair()
    newer = custody_card.issue(person, "file", node_key, since=100, issued_at=300)
    older = custody_card.issue(person, "file", node_key, since=100, issued_at=200)
    assert store.set_contact_custody_statement(person, newer, received_at=301) is True
    assert store.set_contact_custody_statement(person, older, received_at=302) is False
    assert store.contact_custody_statement(person) == newer


def test_session_refuses_inactive_misbound_and_malformed_custody_frames(tmp_path):
    store = Store(tmp_path / "state", b"e" * 32)
    node_key, node = _keypair()
    _person_key, person = _keypair()
    statement = custody_card.issue(person, "file", node_key, since=100, issued_at=200)
    receiver = object.__new__(nakd.Node)
    receiver.store = store
    receiver._clock = lambda: 200
    store.upsert_contact(person, node, "awaiting", "friend")

    with pytest.raises(ValueError, match="active contacts"):
        receiver._receive_custody_frame(
            person, {"t": "custody", "statement": statement}, session_node=node)
    store.set_contact_state(person, "active")
    with pytest.raises(ValueError, match="not bound"):
        receiver._receive_custody_frame(
            person, {"t": "custody", "statement": statement}, session_node="f" * 64)
    with pytest.raises(ValueError, match="fields are missing or unknown"):
        receiver._receive_custody_frame(
            person, {"t": "custody", "statement": statement, "extra": True}, session_node=node)


def test_bad_or_expired_stored_statement_never_breaks_contact_listing(tmp_path):
    store = Store(tmp_path / "state", b"f" * 32)
    node_key, node = _keypair()
    _person_key, person = _keypair()
    store.upsert_contact(person, node, "active", "friend")
    expired = custody_card.issue(
        person, "file", node_key, since=100, issued_at=200, expires_at=201)
    store.set_contact_custody_statement(person, expired, received_at=200)
    receiver = object.__new__(nakd.Node)
    receiver.store = store
    receiver._clock = lambda: 202
    assert receiver._contact_view(store.contact(person))["key_custody"] == "unknown"


def test_display_strings_and_invalid_fallback():
    file_statement, _, _, _ = _statement(mode="file")
    broker_statement, _, _, _ = _statement(mode="broker-biometric")
    now = file_statement["issued_at"] + 3 * 86400
    assert custody_card.display(file_statement, now=now) == \
        "file on disk (self-reported by their node, 3d ago)"
    assert custody_card.display(broker_statement, now=now) == \
        "phone approval (self-reported by their node, 3d ago)"
    assert custody_card.display(None, now=now) == "unknown"
    assert custody_card.display(file_statement, now=file_statement["expires_at"]) == "unknown"


def test_cli_and_mcp_contact_lists_label_self_report(monkeypatch, capsys):
    row = {"petname": "friend", "person": "a" * 64, "node": "b" * 64,
           "state": "active", "online": True, "path": "relay", "direct": 0,
           "key_custody": "phone approval (self-reported by their node, 3d ago)"}

    def fake_call(_sock, request):
        return {"contacts": [row]} if request["op"] == "contacts" else {"contact": row}

    monkeypatch.setattr(nak, "call", fake_call)
    assert nak.main(["contacts"]) == 0
    assert "key custody: phone approval (self-reported by their node, 3d ago)" in capsys.readouterr().out
    assert nak.main(["contact", "show", "friend"]) == 0
    assert "key custody: phone approval (self-reported by their node, 3d ago)" in capsys.readouterr().out

    class Client:
        def contacts(self):
            return [row]

    rendered = nak_mcp.Server(Client()).call("nak_contacts", {})
    assert "key custody: phone approval (self-reported by their node, 3d ago)" in rendered


def test_backward_clock_correction_keeps_own_issued_at_monotonic(tmp_path, monkeypatch):
    store = Store(tmp_path / "state", b"g" * 32)
    node_key = ed25519.Ed25519PrivateKey.generate()
    node = node_key.public_key().public_bytes_raw().hex()
    _person_key, person = _keypair()
    sender = object.__new__(nakd.Node)
    sender.store = store
    sender.person = person
    sender.node = node
    sender._key = node_key.private_bytes_raw()
    sender._state_dir = tmp_path / "state"
    sender._custody_lock = __import__("threading").Lock()
    mode = {"value": "file"}
    monkeypatch.setattr(person_custody, "custody_statement_mode", lambda **_kw: mode["value"])

    sender._clock = lambda: 1000
    first = sender._own_custody_statement()
    assert first["issued_at"] == 1000

    # The clock is corrected backward by more than the future-skew allowance, then custody changes.
    sender._clock = lambda: 600
    mode["value"] = "broker-biometric"
    second = sender._own_custody_statement()
    assert second["mode"] == "broker-biometric"
    assert second["issued_at"] == 1001
    assert store.own_custody_statement() == second

    # A receiver that stored the first card accepts the rotation once its clock allows it.
    receiver = Store(tmp_path / "receiver", b"h" * 32)
    assert receiver.set_contact_custody_statement(person, first, received_at=1000) is True
    assert receiver.set_contact_custody_statement(person, second, received_at=1001) is True
