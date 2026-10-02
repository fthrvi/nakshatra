"""The node's own messaging state: contacts, connection requests, outstanding invites, outbox, inbox.

One sqlite file per node, every check-and-write in one transaction (the ledger lesson). The inbox is
ALSO appended to a raw JSONL log before anything else reads it (unification lane, 2026-10-02:
friend messages are untrusted external content — raw-log first, source=nakshatra, trust=external).
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (person TEXT PRIMARY KEY, node TEXT NOT NULL, petname TEXT,
    nickname TEXT, state TEXT NOT NULL, added INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, person TEXT NOT NULL, node TEXT NOT NULL,
    nickname TEXT, invite_nonce TEXT, received INTEGER NOT NULL, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS invites (nonce TEXT PRIMARY KEY, ik TEXT NOT NULL, expires_at INTEGER NOT NULL,
    created INTEGER NOT NULL, note TEXT, consumed_by TEXT);
CREATE TABLE IF NOT EXISTS outbox (nonce TEXT PRIMARY KEY, to_person TEXT NOT NULL, frame TEXT NOT NULL,
    created INTEGER NOT NULL, acked INTEGER);
CREATE TABLE IF NOT EXISTS seen (nonce TEXT PRIMARY KEY, ts INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS inbox (nonce TEXT PRIMARY KEY, from_person TEXT NOT NULL, petname TEXT,
    author TEXT, aspect TEXT, text TEXT, received INTEGER NOT NULL);
"""


class Store:
    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._db = sqlite3.connect(str(self.dir / "net.sqlite"), isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._raw = self.dir / "inbox-raw.jsonl"

    def _q(self, sql, args=()):
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _tx(self, fn):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                out = fn(self._db)
                self._db.execute("COMMIT")
                return out
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    # invites
    def add_invite(self, nonce: str, ik: str, expires_at: int, note: str = "") -> None:
        self._q("INSERT INTO invites VALUES (?,?,?,?,?,NULL)", (nonce, ik, expires_at, int(time.time()), note))

    def open_invites(self, now: Optional[int] = None) -> list:
        now = int(now if now is not None else time.time())
        return [dict(zip(("nonce", "ik", "expires_at"), r)) for r in
                self._q("SELECT nonce, ik, expires_at FROM invites WHERE consumed_by IS NULL AND expires_at > ?", (now,))]

    def consume_invite(self, nonce: str, by_person: str, now: Optional[int] = None) -> bool:
        now = int(now if now is not None else time.time())

        def fn(db):
            r = db.execute("SELECT consumed_by, expires_at FROM invites WHERE nonce=?", (nonce,)).fetchone()
            if not r or r[0] is not None or r[1] <= now:
                return False
            db.execute("UPDATE invites SET consumed_by=? WHERE nonce=?", (by_person, nonce))
            return True
        return self._tx(fn)

    # requests (someone redeemed our invite and asks to connect)
    def add_request(self, person: str, node: str, nickname: str, invite_nonce: str) -> str:
        rid = secrets.token_hex(4)
        self._q("INSERT INTO requests VALUES (?,?,?,?,?,?, 'pending')",
                (rid, person, node, nickname[:40], invite_nonce, int(time.time())))
        return rid

    def requests(self, state: str = "pending") -> list:
        return [dict(zip(("id", "person", "node", "nickname", "received"), r)) for r in
                self._q("SELECT id, person, node, nickname, received FROM requests WHERE state=?", (state,))]

    def decide_request(self, rid: str, accept: bool) -> Optional[dict]:
        def fn(db):
            r = db.execute("SELECT person, node, nickname FROM requests WHERE id=? AND state='pending'", (rid,)).fetchone()
            if not r:
                return None
            db.execute("UPDATE requests SET state=? WHERE id=?", ("accepted" if accept else "declined", rid))
            return {"person": r[0], "node": r[1], "nickname": r[2]}
        return self._tx(fn)

    # contacts
    def upsert_contact(self, person: str, node: str, state: str, petname: str = "", nickname: str = "") -> None:
        self._q("INSERT INTO contacts VALUES (?,?,?,?,?,?) ON CONFLICT(person) DO UPDATE SET node=excluded.node, "
                "state=excluded.state, petname=COALESCE(NULLIF(excluded.petname,''), contacts.petname)",
                (person, node, petname, nickname, state, int(time.time())))

    def set_contact_state(self, person: str, state: str) -> None:
        self._q("UPDATE contacts SET state=? WHERE person=?", (state, person))

    def contact(self, person: str) -> Optional[dict]:
        r = self._q("SELECT person, node, petname, nickname, state FROM contacts WHERE person=?", (person,))
        return dict(zip(("person", "node", "petname", "nickname", "state"), r[0])) if r else None

    def contacts(self) -> list:
        return [dict(zip(("person", "node", "petname", "nickname", "state"), r)) for r in
                self._q("SELECT person, node, petname, nickname, state FROM contacts ORDER BY added")]

    def resolve(self, who: str) -> Optional[dict]:
        """A petname or a person key (or its prefix of at least 8 hex chars)."""
        for c in self.contacts():
            if who and (c["petname"] == who or c["person"] == who or (len(who) >= 8 and c["person"].startswith(who))):
                return c
        return None

    def remove_contact(self, person: str) -> None:
        self._q("DELETE FROM contacts WHERE person=?", (person,))

    # outbox
    def queue(self, nonce: str, to_person: str, frame: dict) -> None:
        self._q("INSERT INTO outbox VALUES (?,?,?,?,NULL)", (nonce, to_person, json.dumps(frame), int(time.time())))

    def pending_for(self, person: str) -> list:
        return [json.loads(r[0]) for r in
                self._q("SELECT frame FROM outbox WHERE to_person=? AND acked IS NULL ORDER BY created", (person,))]

    def ack(self, nonce: str) -> None:
        self._q("UPDATE outbox SET acked=? WHERE nonce=?", (int(time.time()), nonce))

    def delivered(self, nonce: str) -> bool:
        r = self._q("SELECT acked FROM outbox WHERE nonce=?", (nonce,))
        return bool(r and r[0][0])

    # inbox (untrusted external content)
    def raw_log(self, record: dict) -> None:
        with self._lock, open(self._raw, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def first_time(self, nonce: str) -> bool:
        try:
            self._q("INSERT INTO seen VALUES (?,?)", (nonce, int(time.time())))
            return True
        except sqlite3.IntegrityError:
            return False

    def add_inbox(self, nonce: str, from_person: str, petname: str, author: str, aspect: str, text: str) -> None:
        self._q("INSERT OR IGNORE INTO inbox VALUES (?,?,?,?,?,?,?)",
                (nonce, from_person, petname, author, aspect, text, int(time.time())))

    def inbox(self, since: int = 0, limit: int = 50) -> list:
        return [dict(zip(("nonce", "from_person", "petname", "author", "aspect", "text", "received"), r)) | {
                "source": "nakshatra", "trust": "external"} for r in
                self._q("SELECT nonce, from_person, petname, author, aspect, text, received FROM inbox "
                        "WHERE received >= ? ORDER BY received DESC LIMIT ?", (since, limit))]
