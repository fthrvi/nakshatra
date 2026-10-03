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
CREATE TABLE IF NOT EXISTS tasks (task_hash TEXT NOT NULL, role TEXT NOT NULL, peer TEXT NOT NULL,
    spec TEXT NOT NULL, state TEXT NOT NULL, targets TEXT, assignee TEXT, epoch INTEGER NOT NULL DEFAULT 0,
    output TEXT, verdict TEXT, created INTEGER NOT NULL, updated INTEGER NOT NULL,
    PRIMARY KEY (task_hash, role, peer));
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
        try:   # contacts.name_src: "yours" (the person named them) | "self-chosen" (their nickname / note)
            self._db.execute("ALTER TABLE contacts ADD COLUMN name_src TEXT")
        except sqlite3.OperationalError:
            pass
        for col in ("direct INTEGER NOT NULL DEFAULT 0", "direct_hint TEXT",
                    "p2p INTEGER NOT NULL DEFAULT 0"):   # U3b: opt-in direct paths
            try:
                self._db.execute(f"ALTER TABLE contacts ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        try:   # outbox.refused: a peer's terminal nack (added with tasks; older stores lack it)
            self._db.execute("ALTER TABLE outbox ADD COLUMN refused TEXT")
        except sqlite3.OperationalError:
            pass

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
    def upsert_contact(self, person: str, node: str, state: str, petname: str = "", nickname: str = "",
                       name_src: str = "self-chosen") -> None:
        self._q("INSERT INTO contacts (person, node, petname, nickname, state, added, name_src) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(person) DO UPDATE SET node=excluded.node, state=excluded.state, "
                "name_src=CASE WHEN excluded.petname<>'' THEN excluded.name_src ELSE contacts.name_src END, "
                "petname=COALESCE(NULLIF(excluded.petname,''), contacts.petname)",
                (person, node, petname, nickname, state, int(time.time()), name_src))

    def set_contact_state(self, person: str, state: str) -> None:
        self._q("UPDATE contacts SET state=? WHERE person=?", (state, person))

    def contact(self, person: str) -> Optional[dict]:
        r = self._q("SELECT person, node, petname, nickname, state, name_src, direct, direct_hint, p2p FROM contacts WHERE person=?", (person,))
        return dict(zip(("person", "node", "petname", "nickname", "state", "name_src", "direct", "direct_hint", "p2p"), r[0])) if r else None

    def contacts(self) -> list:
        return [dict(zip(("person", "node", "petname", "nickname", "state", "name_src", "direct", "direct_hint", "p2p"), r)) for r in
                self._q("SELECT person, node, petname, nickname, state, name_src, direct, direct_hint, p2p FROM contacts ORDER BY added")]

    def resolve(self, who: str) -> Optional[dict]:
        """A petname or a person key (or its prefix of at least 8 hex chars)."""
        for c in self.contacts():
            if who and (c["petname"] == who or c["person"] == who or (len(who) >= 8 and c["person"].startswith(who))):
                return c
        return None

    def set_direct(self, person: str, on: bool) -> None:
        """Opt this contact in/out of direct paths. Turning it off also forgets their addresses."""
        if on:
            self._q("UPDATE contacts SET direct=1 WHERE person=?", (person,))
        else:
            self._q("UPDATE contacts SET direct=0, direct_hint=NULL, p2p=0 WHERE person=?", (person,))

    def set_direct_addr(self, person: str, hint: str, p2p: bool) -> None:
        self._q("UPDATE contacts SET direct_hint=?, p2p=? WHERE person=? AND direct=1",
                (hint, int(bool(p2p)), person))

    def remove_contact(self, person: str) -> None:
        self._q("DELETE FROM contacts WHERE person=?", (person,))

    # outbox
    def queue(self, nonce: str, to_person: str, frame: dict) -> None:
        self._q("INSERT INTO outbox (nonce, to_person, frame, created) VALUES (?,?,?,?)",
                (nonce, to_person, json.dumps(frame), int(time.time())))

    def pending_for(self, person: str) -> list:
        return [json.loads(r[0]) for r in
                self._q("SELECT frame FROM outbox WHERE to_person=? AND acked IS NULL ORDER BY created", (person,))]

    def ack(self, nonce: str, to_person: str) -> None:
        """Only the contact a message was FOR can acknowledge it."""
        self._q("UPDATE outbox SET acked=? WHERE nonce=? AND to_person=?", (int(time.time()), nonce, to_person))

    def refuse(self, nonce: str, why: str, to_person: Optional[str] = None) -> None:
        """The peer received it and said no (terminal): stop retrying, keep the reason. A peer can
        only refuse what was sent TO it; a local refusal (signer said no) passes no person."""
        if to_person is None:
            self._q("UPDATE outbox SET acked=?, refused=? WHERE nonce=? AND acked IS NULL",
                    (int(time.time()), str(why)[:300], nonce))
        else:
            self._q("UPDATE outbox SET acked=?, refused=? WHERE nonce=? AND to_person=? AND acked IS NULL",
                    (int(time.time()), str(why)[:300], nonce, to_person))

    def refused(self, nonce: str) -> Optional[str]:
        r = self._q("SELECT refused FROM outbox WHERE nonce=?", (nonce,))
        return r[0][0] if r else None

    def delivered(self, nonce: str) -> bool:
        r = self._q("SELECT acked FROM outbox WHERE nonce=?", (nonce,))
        return bool(r and r[0][0])

    # inbox (untrusted external content)
    def raw_log(self, record: dict) -> None:
        with self._lock, open(self._raw, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def is_seen(self, nonce: str) -> bool:
        return bool(self._q("SELECT 1 FROM seen WHERE nonce=?", (nonce,)))

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

    # tasks: one row per (task, my role, counterparty). Poster rows use peer "" (the assignee is a
    # column); worker rows use the poster's person key as peer. Every transition is compare-and-set.
    _TASK_COLS = ("task_hash", "role", "peer", "spec", "state", "targets", "assignee", "epoch", "output",
                  "verdict", "created", "updated")

    def task_insert(self, task_hash: str, role: str, peer: str, spec: dict, state: str,
                    targets: Optional[list] = None) -> bool:
        now = int(time.time())
        try:
            self._q("INSERT INTO tasks VALUES (?,?,?,?,?,?,NULL,0,NULL,NULL,?,?)",
                    (task_hash, role, peer, json.dumps(spec), state,
                     json.dumps(targets) if targets is not None else None, now, now))
            return True
        except sqlite3.IntegrityError:
            return False

    def task_cas(self, task_hash: str, role: str, peer: str, from_states: tuple, **fields) -> bool:
        """Move a task out of one of `from_states`, setting `fields`, atomically. False if it was not
        in one of those states (someone else won, or the step is out of order)."""
        allowed = {"state", "assignee", "epoch", "output", "verdict"}
        if set(fields) - allowed:
            raise ValueError(f"bad task fields {set(fields) - allowed}")
        sets = ", ".join(f"{k}=?" for k in fields) + ", updated=?"
        marks = ",".join("?" * len(from_states))

        def fn(db):
            cur = db.execute(f"UPDATE tasks SET {sets} WHERE task_hash=? AND role=? AND peer=? AND state IN ({marks})",
                             (*fields.values(), int(time.time()), task_hash, role, peer, *from_states))
            return cur.rowcount == 1
        return self._tx(fn)

    def task_get(self, task_hash: str, role: str, peer: str = "") -> Optional[dict]:
        r = self._q(f"SELECT {', '.join(self._TASK_COLS)} FROM tasks WHERE task_hash=? AND role=? AND peer=?",
                    (task_hash, role, peer))
        return self._task_row(r[0]) if r else None

    def task_find(self, prefix: str) -> list:
        if len(prefix) < 8:
            return []
        return [self._task_row(r) for r in self._q(
            f"SELECT {', '.join(self._TASK_COLS)} FROM tasks WHERE task_hash LIKE ? ORDER BY updated DESC",
            (prefix + "%",))]

    def tasks_in_states(self, states: tuple, role: Optional[str] = None) -> list:
        """Every task in these states (no limit: the sweeper must never miss an old one)."""
        marks = ",".join("?" * len(states))
        q = f"SELECT {', '.join(self._TASK_COLS)} FROM tasks WHERE state IN ({marks})"
        args: tuple = tuple(states)
        if role:
            q, args = q + " AND role=?", args + (role,)
        return [self._task_row(r) for r in self._q(q, args)]

    def tasks(self, role: Optional[str] = None, limit: int = 50) -> list:
        q = f"SELECT {', '.join(self._TASK_COLS)} FROM tasks"
        args: tuple = ()
        if role:
            q, args = q + " WHERE role=?", (role,)
        return [self._task_row(r) for r in self._q(q + " ORDER BY updated DESC LIMIT ?", (*args, limit))]

    def _task_row(self, r) -> dict:
        d = dict(zip(self._TASK_COLS, r))
        d["spec"] = json.loads(d["spec"])
        d["targets"] = json.loads(d["targets"]) if d["targets"] else None
        d["verdict"] = json.loads(d["verdict"]) if d["verdict"] else None
        return d
