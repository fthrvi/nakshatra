"""M0.5 node-store encryption and legacy migration tests (temporary state only)."""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric import ed25519

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from network import custody_card, nakd  # noqa: E402
from network.store import SCHEMA, Store  # noqa: E402


KEY = bytes(range(32))
PERSON = "ab" * 32
NODE = "cd" * 32
TASK = "ef" * 32
MARKER = "KNOWN_PLAINTEXT_MARKER_M05"


def test_every_encrypted_column_round_trips_and_is_ciphertext_in_sqlite(tmp_path):
    store = Store(tmp_path / "net", KEY)
    store.upsert_contact(PERSON, NODE, "active", "pet-name", "nick-name")
    rid = store.add_request(PERSON, NODE, "request-nick", "invite-1")
    starter_rid = store.add_request(PERSON, NODE, "starter-nick", "invite-1", NODE)
    store.add_invite("invite-1", "11" * 32, int(time.time()) + 1000, "invite-note")
    now = int(time.time())
    assert store.decide_request(starter_rid, True, starter={
        "invite_id": "starter-1", "to_keyid": NODE, "amount": 7,
        "signed_invite": "signed-starter-invite", "reason": "starter-reason", "created": now,
    })
    store.starter_update("starter-1", receipt={"v": 1, "replayed": False})
    store.queue("out-1", PERSON, {"kind": "msg", "body": {"text": "outbox-text"}})
    store.refuse("out-1", "outbox-refusal")
    store.add_inbox("in-1", PERSON, "inbox-pet", "agent:alice", "whole", "inbox-text")
    store.add_agent_message({"mid": "12" * 12, "thread": "34" * 16, "intent": "ask", "reply_to": "",
                             "text": "agent-text", "want": "agent-want", "refs": [],
                             "budget": {"amount": 0, "unit": "credits"}, "harness": "test"},
                            PERSON, "in")
    store.set_agent_card(PERSON, {"display_name": "Agent", "offers": "card-offers", "intents": ["ask"],
                                  "paid": True, "price_hint": "none", "models": ["m"],
                                  "supports": ["agent.msg/1"], "updated_at": 1,
                                  "ledger_keyid": "77" * 32})
    custody = custody_card.issue(
        PERSON, "file", ed25519.Ed25519PrivateKey.generate(), since=1, issued_at=2)
    store.set_own_custody_statement(custody)
    store.set_contact_custody_statement(PERSON, custody, received_at=3)
    assert store.task_insert(TASK, "worker", PERSON, {"title": "task-spec"}, "assigned")
    assert store.task_cas(TASK, "worker", PERSON, ("assigned",), state="submitted",
                          output="task-output", verdict=json.dumps({"reason": "task-verdict"}))
    store.raw_log({"received": 123, "trust": "external", "body": MARKER})

    assert store.resolve("pet-name")["nickname"] == "nick-name"
    assert store.requests()[0]["id"] == rid and store.requests()[0]["nickname"] == "request-nick"
    assert store.inbox()[0]["text"] == "inbox-text" and store.inbox()[0]["author"] == "agent:alice"
    assert store.outbox(include_text=True)[0]["text"] == "outbox-text"
    assert store.outbox()[0]["reason"] == "outbox-refusal"
    task = store.task_get(TASK, "worker", PERSON)
    assert task["spec"]["title"] == "task-spec" and task["output"] == "task-output"
    assert task["verdict"]["reason"] == "task-verdict"
    assert store.raw_records()[0]["body"] == MARKER
    assert store.agent_thread("34")[1][0]["want"] == "agent-want"
    assert store.agent_card(PERSON)["offers"] == "card-offers"
    assert store.agent_card(PERSON)["ledger_keyid"] == "77" * 32
    assert store.starter_rows()[0]["signed_invite"] == "signed-starter-invite"
    assert store.starter_rows()[0]["receipt"] == {"v": 1, "replayed": False}
    assert store.own_custody_statement() == custody
    assert store.contact_custody_statement(PERSON) == custody

    for table, (pk_cols, columns) in Store._ENCRYPTED.items():
        rows = store._q(f"SELECT {', '.join((*pk_cols, *columns))} FROM {table}")
        assert rows, table
        for row in rows:
            for value in row[len(pk_cols):]:
                if value is not None:
                    assert value.startswith("enc1:"), (table, value)


def test_ciphertext_cannot_move_between_rows_because_aad_binds_primary_key(tmp_path):
    store = Store(tmp_path / "net", KEY)
    store.add_inbox("row-a", PERSON, "p", "a", "x", "first")
    store.add_inbox("row-b", PERSON, "p", "a", "x", "second")
    a = store._q("SELECT text FROM inbox WHERE nonce='row-a'")[0][0]
    b = store._q("SELECT text FROM inbox WHERE nonce='row-b'")[0][0]
    store._q("UPDATE inbox SET text=? WHERE nonce='row-a'", (b,))
    store._q("UPDATE inbox SET text=? WHERE nonce='row-b'", (a,))
    with pytest.raises(InvalidTag):
        store.inbox()


def test_ciphertext_cannot_move_between_columns_because_aad_binds_column(tmp_path):
    store = Store(tmp_path / "net", KEY)
    store.add_inbox("row", PERSON, "petname", "author", "whole", "text")
    petname, author = store._q("SELECT petname, author FROM inbox WHERE nonce='row'")[0]
    store._q("UPDATE inbox SET petname=?, author=? WHERE nonce='row'", (author, petname))
    with pytest.raises(InvalidTag):
        store.inbox()


def test_agent_message_ciphertext_cannot_move_between_rows_or_columns(tmp_path):
    store = Store(tmp_path / "net", KEY)
    first = {"mid": "10" * 12, "thread": "30" * 16, "intent": "ask", "reply_to": "",
             "text": "first text", "want": "first want"}
    second = {"mid": "20" * 12, "thread": "30" * 16, "intent": "answer", "reply_to": "",
              "text": "second text", "want": "second want"}
    store.add_agent_message(first, PERSON, "in")
    store.add_agent_message(second, PERSON, "in")
    first_text, first_want = store._q(
        "SELECT text,want FROM agent_messages WHERE mid=?", (first["mid"],))[0]
    second_text = store._q("SELECT text FROM agent_messages WHERE mid=?", (second["mid"],))[0][0]

    store._q("UPDATE agent_messages SET text=? WHERE mid=?", (second_text, first["mid"]))
    with pytest.raises(InvalidTag):
        store.agent_thread(first["thread"], PERSON)
    store._q("UPDATE agent_messages SET text=?,want=? WHERE mid=?",
             (first_want, first_text, first["mid"]))
    with pytest.raises(InvalidTag):
        store.agent_thread(first["thread"], PERSON)


def test_agent_card_ciphertext_cannot_move_between_rows_or_columns(tmp_path):
    store = Store(tmp_path / "net", KEY)
    card = {"display_name": "Agent", "offers": "offers", "intents": ["ask"], "paid": False,
            "price_hint": "none", "models": ["m"], "supports": ["agent.msg/1"], "updated_at": 1}
    other = "12" * 32
    store.set_agent_card(PERSON, card)
    store.set_agent_card(other, dict(card, display_name="Other", offers="other offers"))
    display_name, offers = store._q(
        "SELECT display_name,offers FROM agent_cards WHERE person=?", (PERSON,))[0]
    other_offers = store._q("SELECT offers FROM agent_cards WHERE person=?", (other,))[0][0]

    store._q("UPDATE agent_cards SET offers=? WHERE person=?", (other_offers, PERSON))
    with pytest.raises(InvalidTag):
        store.agent_card(PERSON)
    store._q("UPDATE agent_cards SET display_name=?,offers=? WHERE person=?",
             (offers, display_name, PERSON))
    with pytest.raises(InvalidTag):
        store.agent_card(PERSON)


def test_plaintext_that_starts_with_ciphertext_prefix_is_still_encrypted(tmp_path):
    store = Store(tmp_path / "net", KEY)
    store.add_inbox("prefix", PERSON, "p", "a", "x", "enc1:not-actually-ciphertext")
    raw = store._q("SELECT text FROM inbox WHERE nonce='prefix'")[0][0]
    assert raw.startswith("enc1:") and raw != "enc1:not-actually-ciphertext"
    assert store.inbox()[0]["text"] == "enc1:not-actually-ciphertext"


def test_wrong_data_key_fails_closed_instead_of_reencrypting_ciphertext(tmp_path):
    root = tmp_path / "net"
    store = Store(root, KEY)
    store.add_inbox("wrong-key", PERSON, "p", "a", "x", "history")
    ciphertext = store._q("SELECT text FROM inbox WHERE nonce='wrong-key'")[0][0]
    store._db.close()
    with pytest.raises(InvalidTag):
        Store(root, bytes(reversed(KEY)))
    db = sqlite3.connect(root / "net.sqlite")
    try:
        assert db.execute("SELECT text FROM inbox WHERE nonce='wrong-key'").fetchone()[0] == ciphertext
    finally:
        db.close()


def _legacy_store(root: Path) -> None:
    root.mkdir(parents=True)
    db = sqlite3.connect(root / "net.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    db.execute("ALTER TABLE contacts ADD COLUMN name_src TEXT")
    db.execute("ALTER TABLE contacts ADD COLUMN direct INTEGER NOT NULL DEFAULT 0")
    db.execute("ALTER TABLE contacts ADD COLUMN direct_hint TEXT")
    db.execute("ALTER TABLE contacts ADD COLUMN p2p INTEGER NOT NULL DEFAULT 0")
    db.execute("ALTER TABLE outbox ADD COLUMN refused TEXT")
    now = int(time.time())
    db.execute("INSERT INTO contacts (person,node,petname,nickname,state,added,name_src) VALUES (?,?,?,?,?,?,?)",
               (PERSON, NODE, "pet-" + MARKER, "nick-" + MARKER, "active", now, "yours"))
    db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,'pending')",
               ("req", PERSON, NODE, "request-" + MARKER, "inv", now))
    db.execute("INSERT INTO invites VALUES (?,?,?,?,?,NULL)",
               ("inv", "11" * 32, now + 1000, now, "note-" + MARKER))
    db.execute("INSERT INTO outbox VALUES (?,?,?,?,?,?)",
               ("out", PERSON, json.dumps({"kind": "msg", "body": {"text": MARKER}}), now, now,
                "refused-" + MARKER))
    db.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
               (TASK, "worker", PERSON, json.dumps({"title": MARKER}), "submitted", None, None, 0,
                "output-" + MARKER, json.dumps({"reason": MARKER}), now, now))
    db.execute("INSERT INTO inbox VALUES (?,?,?,?,?,?,?)",
               ("in", PERSON, "pet-" + MARKER, "author-" + MARKER, "aspect-" + MARKER,
                "text-" + MARKER, now))
    db.commit()
    db.close()
    (root / "inbox-raw.jsonl").write_text(json.dumps({"received": now, "body": MARKER}) + "\n")


def test_structurally_ciphertext_shaped_legacy_plaintext_migrates(tmp_path):
    root = tmp_path / "net"
    _legacy_store(root)
    shaped = "enc1:" + base64.b64encode(b"x" * 40).decode()
    db = sqlite3.connect(root / "net.sqlite")
    db.execute("UPDATE inbox SET text=? WHERE nonce='in'", (shaped,))
    db.commit()
    db.close()
    store = Store(root, KEY)
    assert store.inbox()[0]["text"] == shaped
    assert store._q("SELECT text FROM inbox WHERE nonce='in'")[0][0] != shaped


def _assert_marker_absent(root: Path) -> None:
    for name in ("net.sqlite", "net.sqlite-wal", "net.sqlite-shm", "inbox-raw.jsonl"):
        path = root / name
        if path.exists():
            assert MARKER.encode() not in path.read_bytes(), name


def test_plaintext_migration_scrubs_db_wal_and_raw_log_and_is_idempotent(tmp_path):
    root = tmp_path / "net"
    _legacy_store(root)
    store = Store(root, KEY)
    assert store.contact(PERSON)["petname"] == "pet-" + MARKER
    assert store.requests()[0]["nickname"] == "request-" + MARKER
    assert store.inbox()[0]["text"] == "text-" + MARKER
    assert store.outbox()[0]["reason"] == "refused-" + MARKER
    assert store.task_get(TASK, "worker", PERSON)["output"] == "output-" + MARKER
    assert store.raw_records()[0]["body"] == MARKER
    _assert_marker_absent(root)
    before = store._q("SELECT text FROM inbox WHERE nonce='in'")[0][0]
    store._db.close()
    reopened = Store(root, KEY)
    assert reopened._q("SELECT text FROM inbox WHERE nonce='in'")[0][0] == before
    _assert_marker_absent(root)


def test_schema_migration_is_idempotent_on_a_forgejo_main_store(tmp_path):
    root = tmp_path / "net"
    module_path = tmp_path / "forgejo_main_store.py"
    source = subprocess.run(
        ["git", "show", "forgejo/main:scripts/network/store.py"], cwd=_REPO,
        check=True, capture_output=True, text=True).stdout
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location("forgejo_main_store", module_path)
    assert spec and spec.loader
    parent_store = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_store)
    legacy = parent_store.Store(root, KEY)
    legacy._db.close()
    db = sqlite3.connect(root / "net.sqlite")
    assert db.execute("SELECT name FROM sqlite_master WHERE name='agent_messages'").fetchone() is None
    db.close()
    first = Store(root, KEY)
    first.set_agent_card(PERSON, {"display_name": "A", "offers": "O", "intents": [], "paid": False,
                                  "price_hint": "", "models": [], "supports": ["agent.msg/1"],
                                  "updated_at": 1})
    first._db.close()
    second = Store(root, KEY)
    assert second.agent_card(PERSON)["display_name"] == "A"


def test_mixed_store_recovery_finishes_plaintext_rows_and_raw_lines(tmp_path):
    root = tmp_path / "net"
    _legacy_store(root)
    first = Store(root, KEY)
    first._db.close()
    db = sqlite3.connect(root / "net.sqlite")
    db.execute("UPDATE contacts SET nickname=? WHERE person=?", ("mixed-" + MARKER, PERSON))
    db.commit()
    db.close()
    with (root / "inbox-raw.jsonl").open("a") as out:
        out.write(json.dumps({"received": 456, "body": "mixed-" + MARKER}) + "\n")
    recovered = Store(root, KEY)
    assert recovered.contact(PERSON)["nickname"] == "mixed-" + MARKER
    assert recovered.raw_records()[-1]["body"] == "mixed-" + MARKER
    _assert_marker_absent(root)


def test_cleanup_pending_survives_crash_between_commit_and_vacuum(tmp_path, monkeypatch):
    root = tmp_path / "net"
    _legacy_store(root)
    original = Store._scrub_plaintext_pages

    def crash_after_commit(store):
        store._db.close()
        raise RuntimeError("simulated crash before scrub")

    monkeypatch.setattr(Store, "_scrub_plaintext_pages", crash_after_commit)
    with pytest.raises(RuntimeError, match="simulated crash"):
        Store(root, KEY)
    db = sqlite3.connect(root / "net.sqlite")
    try:
        assert db.execute("SELECT value FROM store_meta WHERE key='cleanup_pending'").fetchone() == ("1",)
    finally:
        db.close()
    monkeypatch.setattr(Store, "_scrub_plaintext_pages", original)
    recovered = Store(root, KEY)
    assert recovered._db.execute(
        "SELECT value FROM store_meta WHERE key='cleanup_pending'").fetchone() == ("0",)
    _assert_marker_absent(root)


def test_torn_final_raw_log_line_is_dropped_during_atomic_rewrite(tmp_path):
    root = tmp_path / "net"
    store = Store(root, KEY)
    store.raw_log({"received": 1, "body": "kept"})
    store._db.close()
    with (root / "inbox-raw.jsonl").open("ab") as out:
        out.write(b'{"v":"enc1","ct":')
    recovered = Store(root, KEY)
    assert recovered.raw_records() == [{"received": 1, "body": "kept"}]
    assert (root / "inbox-raw.jsonl").read_bytes().endswith(b"\n")


def test_store_and_raw_log_modes_are_private(tmp_path):
    root = tmp_path / "net"
    store = Store(root, KEY)
    store.raw_log({"received": 1, "body": "private"})
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    for name in ("net.sqlite", "net.sqlite-wal", "net.sqlite-shm", "inbox-raw.jsonl"):
        path = root / name
        if path.exists():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_node_refuses_to_create_store_when_signer_has_no_data_key(tmp_path):
    node_key = bytes(range(32))
    node_pub = nakd.ed25519.Ed25519PrivateKey.from_private_bytes(
        node_key).public_key().public_bytes_raw().hex()

    def no_key(request):
        if request.get("op") == "whoami":
            return {"ok": True, "node": node_pub, "person": PERSON, "custody": "test",
                    "agents": ["agent"]}
        return {"ok": False, "error": "data key unavailable"}

    with pytest.raises(RuntimeError, match="signer data key unavailable"):
        nakd.Node(tmp_path / "net", node_key, no_key, agent="agent",
                  relay=("127.0.0.1", 1))
    assert not (tmp_path / "net" / "net.sqlite").exists()


def test_wrong_node_signer_is_rejected_before_data_key_or_store_access(tmp_path):
    node_key = bytes(range(32))
    calls = []

    def wrong_signer(request):
        calls.append(request["op"])
        if request["op"] == "whoami":
            return {"ok": True, "node": "00" * 32, "person": PERSON, "custody": "test",
                    "agents": ["agent"]}
        pytest.fail("wrong-node signer data key was requested")

    with pytest.raises(RuntimeError, match="different node key"):
        nakd.Node(tmp_path / "net", node_key, wrong_signer, agent="agent",
                  relay=("127.0.0.1", 1))
    assert calls == ["whoami"] and not (tmp_path / "net" / "net.sqlite").exists()


def test_resolve_refuses_ambiguous_petname_and_prefix_but_exact_key_wins(tmp_path):
    store = Store(tmp_path / "net", KEY)
    a, b = "ab" * 32, "ab" * 4 + "cd" * 28
    store.upsert_contact(a, "11" * 32, "active", petname="dup")
    store.upsert_contact(b, "22" * 32, "active", petname="dup")
    with pytest.raises(ValueError, match="matches 2 contacts"):
        store.resolve("dup")
    with pytest.raises(ValueError, match="matches 2 contacts"):
        store.resolve("ab" * 4)
    assert store.resolve(a)["person"] == a
    assert store.resolve(b)["person"] == b
    assert store.resolve("ab" * 5)["person"] == a
    assert store.resolve("nobody") is None


@pytest.mark.parametrize("bad", ["%", "_", "ab%", "AB", "zz", "a" * 33, ""])
def test_agent_thread_prefix_must_be_lowercase_hex(tmp_path, bad):
    store = Store(tmp_path / "net", KEY)
    store.add_agent_message({"mid": "12" * 12, "thread": "34" * 16, "intent": "ask", "reply_to": "",
                             "text": "hello"}, "ab" * 32, "in")
    with pytest.raises(ValueError, match="lowercase hex"):
        store.agent_thread(bad)
    assert store.agent_thread("3434")[0] == "34" * 16
