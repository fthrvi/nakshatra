"""The node's own messaging state: contacts, connection requests, outstanding invites, outbox, inbox.

One sqlite file per node, every check-and-write in one transaction (the ledger lesson). The inbox is
ALSO appended to a raw JSONL log before anything else reads it (unification lane, 2026-10-02:
friend messages are untrusted external content — raw-log first, source=nakshatra, trust=external).
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

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
CREATE TABLE IF NOT EXISTS agent_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, mid TEXT NOT NULL UNIQUE,
    peer TEXT NOT NULL, direction TEXT NOT NULL, thread TEXT NOT NULL, intent TEXT NOT NULL,
    reply_to TEXT NOT NULL, expires INTEGER, text TEXT NOT NULL, want TEXT, refs TEXT, budget TEXT,
    harness TEXT, created INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS agent_messages_peer_thread_idx ON agent_messages(peer, thread, created, id);
CREATE TABLE IF NOT EXISTS agent_cards (person TEXT PRIMARY KEY, display_name TEXT NOT NULL,
    offers TEXT NOT NULL, intents TEXT NOT NULL, paid TEXT NOT NULL, price_hint TEXT NOT NULL,
    models TEXT NOT NULL, supports TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Store:
    _ENCRYPTED = {
        "inbox": (("nonce",), ("text", "petname", "author", "aspect")),
        "contacts": (("person",), ("petname", "nickname")),
        "requests": (("id",), ("nickname",)),
        "invites": (("nonce",), ("note",)),
        "outbox": (("nonce",), ("frame", "refused")),
        "tasks": (("task_hash", "role", "peer"), ("spec", "output", "verdict")),
        "agent_messages": (("mid",), ("text", "want", "refs", "budget", "harness")),
        "agent_cards": (("person",), ("display_name", "offers", "intents", "paid", "price_hint",
                                          "models", "supports", "updated_at")),
    }

    def __init__(self, state_dir: Path, data_key: bytes):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.dir, 0o700)
        if not isinstance(data_key, bytes) or len(data_key) != 32:
            raise ValueError("the Nakshatra store requires a 32-byte signer data key")
        self._aead = ChaCha20Poly1305(data_key)
        self._db_path = self.dir / "net.sqlite"
        self._db = sqlite3.connect(str(self._db_path), isolation_level=None, check_same_thread=False)
        os.chmod(self._db_path, 0o600)
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
        self._key_verified = self._verify_key_marker()
        self._migrate()
        self._fix_modes()

    @staticmethod
    def _pk(values) -> str:
        return json.dumps(list(values), separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _aad(table: str, column: str, primary_key: str) -> bytes:
        return f"{table}\0{column}\0{primary_key}".encode("utf-8")

    def _seal(self, value: str, aad: bytes) -> str:
        nonce = secrets.token_bytes(12)
        return "enc1:" + base64.b64encode(
            nonce + self._aead.encrypt(nonce, str(value).encode("utf-8"), aad)).decode("ascii")

    def _open(self, value, aad: bytes):
        if value is None or not (isinstance(value, str) and value.startswith("enc1:")):
            return value
        raw = base64.b64decode(value[5:], validate=True)
        if len(raw) < 28:  # 12-byte nonce + at least the 16-byte Poly1305 tag
            raise ValueError("encrypted store value is truncated")
        return self._aead.decrypt(raw[:12], raw[12:], aad).decode("utf-8")

    def _enc(self, table: str, column: str, primary_key: str, value):
        if value is None:
            return value
        return self._seal(str(value), self._aad(table, column, primary_key))

    def _dec(self, table: str, column: str, primary_key: str, value):
        return self._open(value, self._aad(table, column, primary_key))

    def _is_encrypted(self, table: str, column: str, primary_key: str, value) -> bool:
        if not (isinstance(value, str) and value.startswith("enc1:")):
            return False
        try:
            raw = base64.b64decode(value[5:], validate=True)
        except ValueError:
            return False
        if len(raw) < 28:
            return False
        try:
            self._dec(table, column, primary_key, value)
        except InvalidTag:
            if self._key_verified:
                raise  # after migration, this is corruption rather than ambiguous legacy text
            return False
        return True

    def _verify_key_marker(self) -> bool:
        row = self._db.execute("SELECT value FROM store_meta WHERE key='key_check'").fetchone()
        if not row:
            return False
        if self._open(row[0], b"store-key-check") != "nak-store-key-v1":
            raise ValueError("Nakshatra store data-key check failed")
        return True

    def _fix_modes(self) -> None:
        os.chmod(self.dir, 0o700)
        for path in (self._db_path, Path(str(self._db_path) + "-wal"),
                     Path(str(self._db_path) + "-shm"), self._raw):
            try:
                os.chmod(path, 0o600)
            except FileNotFoundError:
                pass

    def _migrate(self) -> None:
        """Encrypt every legacy plaintext field in one transaction, then scrub SQLite free space."""
        changed = False
        self._db.execute("BEGIN IMMEDIATE")
        try:
            for table, (pk_cols, columns) in self._ENCRYPTED.items():
                selected = ", ".join((*pk_cols, *columns))
                for row in self._db.execute(f"SELECT {selected} FROM {table}").fetchall():
                    pk_values, values = row[:len(pk_cols)], row[len(pk_cols):]
                    primary_key = self._pk(pk_values)
                    updates = {column: self._enc(table, column, primary_key, value)
                               for column, value in zip(columns, values)
                               if value is not None and not self._is_encrypted(
                                   table, column, primary_key, value)}
                    if not updates:
                        continue
                    sets = ", ".join(f"{column}=?" for column in updates)
                    where = " AND ".join(f"{column}=?" for column in pk_cols)
                    self._db.execute(f"UPDATE {table} SET {sets} WHERE {where}",
                                     (*updates.values(), *pk_values))
                    changed = True
            if not self._key_verified:
                self._db.execute("INSERT INTO store_meta (key, value) VALUES ('key_check', ?)",
                                 (self._seal("nak-store-key-v1", b"store-key-check"),))
            if changed:
                self._db.execute("INSERT INTO store_meta (key, value) VALUES ('cleanup_pending', '1') "
                                 "ON CONFLICT(key) DO UPDATE SET value='1'")
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise
        self._key_verified = True
        raw_changed = self._migrate_raw_log()
        if raw_changed:
            self._db.execute("INSERT INTO store_meta (key, value) VALUES ('cleanup_pending', '1') "
                             "ON CONFLICT(key) DO UPDATE SET value='1'")
        pending = self._db.execute(
            "SELECT value FROM store_meta WHERE key='cleanup_pending'").fetchone()
        if pending and pending[0] == "1":
            self._scrub_plaintext_pages()
            self._db.execute("UPDATE store_meta SET value='0' WHERE key='cleanup_pending'")
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _scrub_plaintext_pages(self) -> None:
        self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._db.execute("VACUUM")
        self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _read_raw_rows(self) -> tuple[list[dict], bool]:
        if not self._raw.exists():
            return [], False
        lines = self._raw.read_bytes().splitlines(keepends=True)
        rows, torn = [], False
        for index, line in enumerate(lines):
            body = line.rstrip(b"\r\n")
            if not body:
                continue
            try:
                rows.append(json.loads(body.decode("utf-8")))
            except (UnicodeDecodeError, ValueError):
                if index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                    torn = True
                    break
                raise
        return rows, torn

    def _migrate_raw_log(self) -> bool:
        if not self._raw.exists():
            return False
        source_rows, changed = self._read_raw_rows()
        rows = []
        for row in source_rows:
            if row.get("v") == "enc1" and isinstance(row.get("ct"), str):
                # Authenticate existing encrypted lines now; a bad key must fail startup.
                self._open(row["ct"], b"inbox-raw")
                rows.append(row)
                continue
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            rows.append({"v": "enc1", "ts": int(row.get("ts", row.get("received", time.time()))),
                         "ct": self._seal(encoded, b"inbox-raw")})
            changed = True
        if not changed:
            return False
        fd, tmp_name = tempfile.mkstemp(prefix=".inbox-raw.", dir=self.dir)
        tmp = Path(tmp_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as out:
                for row in rows:
                    out.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, self._raw)
            dfd = os.open(str(self.dir), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            tmp.unlink(missing_ok=True)
            raise
        return True

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
        note = self._enc("invites", "note", self._pk((nonce,)), note)
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
        nickname = self._enc("requests", "nickname", self._pk((rid,)), nickname[:40])
        self._q("INSERT INTO requests VALUES (?,?,?,?,?,?, 'pending')",
                (rid, person, node, nickname, invite_nonce, int(time.time())))
        return rid

    def requests(self, state: str = "pending") -> list:
        out = []
        for r in self._q("SELECT id, person, node, nickname, received FROM requests WHERE state=?", (state,)):
            row = dict(zip(("id", "person", "node", "nickname", "received"), r))
            row["nickname"] = self._dec("requests", "nickname", self._pk((row["id"],)), row["nickname"])
            out.append(row)
        return out

    def decide_request(self, rid: str, accept: bool) -> Optional[dict]:
        def fn(db):
            r = db.execute("SELECT person, node, nickname FROM requests WHERE id=? AND state='pending'", (rid,)).fetchone()
            if not r:
                return None
            db.execute("UPDATE requests SET state=? WHERE id=?", ("accepted" if accept else "declined", rid))
            return {"person": r[0], "node": r[1],
                    "nickname": self._dec("requests", "nickname", self._pk((rid,)), r[2])}
        return self._tx(fn)

    # contacts
    def upsert_contact(self, person: str, node: str, state: str, petname: str = "", nickname: str = "",
                       name_src: str = "self-chosen") -> None:
        primary_key = self._pk((person,))
        values = (person, node, self._enc("contacts", "petname", primary_key, petname),
                  self._enc("contacts", "nickname", primary_key, nickname), state, int(time.time()), name_src)
        update_names = (", name_src=excluded.name_src, petname=excluded.petname" if petname else "")
        self._q("INSERT INTO contacts (person, node, petname, nickname, state, added, name_src) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(person) DO UPDATE SET node=excluded.node, state=excluded.state" + update_names, values)

    def set_contact_state(self, person: str, state: str) -> None:
        self._q("UPDATE contacts SET state=? WHERE person=?", (state, person))

    def contact(self, person: str) -> Optional[dict]:
        r = self._q("SELECT person, node, petname, nickname, state, name_src, direct, direct_hint, p2p FROM contacts WHERE person=?", (person,))
        if not r:
            return None
        row = dict(zip(("person", "node", "petname", "nickname", "state", "name_src", "direct", "direct_hint", "p2p"), r[0]))
        primary_key = self._pk((person,))
        for column in ("petname", "nickname"):
            row[column] = self._dec("contacts", column, primary_key, row[column])
        return row

    def contacts(self) -> list:
        return [self.contact(r[0]) for r in self._q("SELECT person FROM contacts ORDER BY added")]

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

    def clear_direct_addr(self, person: str) -> None:
        """Forget a peer's withdrawn/failed direct capability without changing our own opt-in."""
        self._q("UPDATE contacts SET direct_hint=NULL, p2p=0 WHERE person=?", (person,))

    def remove_contact(self, person: str) -> None:
        self._q("DELETE FROM contacts WHERE person=?", (person,))

    # outbox
    def queue(self, nonce: str, to_person: str, frame: dict) -> None:
        raw = self._enc("outbox", "frame", self._pk((nonce,)), json.dumps(frame))
        self._q("INSERT INTO outbox (nonce, to_person, frame, created) VALUES (?,?,?,?)",
                (nonce, to_person, raw, int(time.time())))

    def pending_for(self, person: str) -> list:
        return [json.loads(self._dec("outbox", "frame", self._pk((r[0],)), r[1])) for r in
                self._q("SELECT nonce, frame FROM outbox WHERE to_person=? AND acked IS NULL ORDER BY created", (person,))]

    def ack(self, nonce: str, to_person: str) -> None:
        """Only the contact a message was FOR can acknowledge it."""
        self._q("UPDATE outbox SET acked=? WHERE nonce=? AND to_person=?", (int(time.time()), nonce, to_person))

    def refuse(self, nonce: str, why: str, to_person: Optional[str] = None) -> None:
        """The peer received it and said no (terminal): stop retrying, keep the reason. A peer can
        only refuse what was sent TO it; a local refusal (signer said no) passes no person."""
        encrypted = self._enc("outbox", "refused", self._pk((nonce,)), str(why)[:300])
        if to_person is None:
            self._q("UPDATE outbox SET acked=?, refused=? WHERE nonce=? AND acked IS NULL",
                    (int(time.time()), encrypted, nonce))
        else:
            self._q("UPDATE outbox SET acked=?, refused=? WHERE nonce=? AND to_person=? AND acked IS NULL",
                    (int(time.time()), encrypted, nonce, to_person))

    def refused(self, nonce: str) -> Optional[str]:
        r = self._q("SELECT refused FROM outbox WHERE nonce=?", (nonce,))
        return self._dec("outbox", "refused", self._pk((nonce,)), r[0][0]) if r else None

    def delivered(self, nonce: str) -> bool:
        r = self._q("SELECT acked FROM outbox WHERE nonce=?", (nonce,))
        return bool(r and r[0][0])

    def outbox(self, refused_only: bool = False, include_text: bool = False, limit: int = 50) -> list:
        where = " WHERE refused IS NOT NULL" if refused_only else ""
        rows = self._q("SELECT nonce, to_person, frame, created, acked, refused FROM outbox" + where +
                       " ORDER BY created DESC LIMIT ?", (int(limit),))
        out = []
        for nonce, person, raw, created, acked, refused in rows:
            primary_key = self._pk((nonce,))
            frame = json.loads(self._dec("outbox", "frame", primary_key, raw))
            refused = self._dec("outbox", "refused", primary_key, refused)
            row = {"id": nonce, "to": person, "kind": frame.get("kind", ""), "created": created,
                   "delivered": bool(acked and not refused), "refused": bool(refused),
                   "completed": acked, "reason": refused or ""}
            if include_text:
                body = frame.get("body") or {}
                row["text"] = body.get("text") or body.get("output") or ""
            out.append(row)
        return out

    def refusal_summary(self) -> dict:
        count = self._q("SELECT COUNT(*) FROM outbox WHERE refused IS NOT NULL")[0][0]
        latest = self._q("SELECT nonce, to_person, acked, refused FROM outbox "
                         "WHERE refused IS NOT NULL ORDER BY acked DESC LIMIT 1")
        return {"count": count, "latest": ({"id": latest[0][0], "to": latest[0][1],
                                             "time": latest[0][2], "reason": self._dec(
                                                 "outbox", "refused", self._pk((latest[0][0],)), latest[0][3])}
                                            if latest else None)}

    # inbox (untrusted external content)
    def raw_log(self, record: dict) -> None:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        row = {"v": "enc1", "ts": int(record.get("ts", record.get("received", time.time()))),
               "ct": self._seal(encoded, b"inbox-raw")}
        with self._lock:
            fd = os.open(str(self._raw), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
                os.fchmod(fd, 0o600)
            finally:
                os.close(fd)

    def raw_records(self) -> list:
        out = []
        rows, _torn = self._read_raw_rows()
        for row in rows:
            if row.get("v") == "enc1":
                row = json.loads(self._open(row["ct"], b"inbox-raw"))
            out.append(row)
        return out

    def is_seen(self, nonce: str) -> bool:
        return bool(self._q("SELECT 1 FROM seen WHERE nonce=?", (nonce,)))

    def first_time(self, nonce: str) -> bool:
        try:
            self._q("INSERT INTO seen VALUES (?,?)", (nonce, int(time.time())))
            return True
        except sqlite3.IntegrityError:
            return False

    def add_inbox(self, nonce: str, from_person: str, petname: str, author: str, aspect: str, text: str) -> None:
        primary_key = self._pk((nonce,))
        self._q("INSERT OR IGNORE INTO inbox VALUES (?,?,?,?,?,?,?)",
                (nonce, from_person, self._enc("inbox", "petname", primary_key, petname),
                 self._enc("inbox", "author", primary_key, author),
                 self._enc("inbox", "aspect", primary_key, aspect),
                 self._enc("inbox", "text", primary_key, text), int(time.time())))

    def inbox(self, since: int = 0, limit: int = 50) -> list:
        out = []
        for r in self._q("SELECT nonce, from_person, petname, author, aspect, text, received FROM inbox "
                         "WHERE received >= ? ORDER BY received DESC LIMIT ?", (since, limit)):
            row = dict(zip(("nonce", "from_person", "petname", "author", "aspect", "text", "received"), r))
            primary_key = self._pk((row["nonce"],))
            for column in ("petname", "author", "aspect", "text"):
                row[column] = self._dec("inbox", column, primary_key, row[column])
            row.update(source="nakshatra", trust="external")
            out.append(row)
        return out

    # M1 agent conversations. Peer-controlled received rows are returned with trust=external; sent
    # rows are the local node's own record of what it queued.
    _AGENT_COLS = ("id", "mid", "peer", "direction", "thread", "intent", "reply_to", "expires",
                   "text", "want", "refs", "budget", "harness", "created")

    def add_agent_message(self, body: dict, peer: str, direction: str, *, created: Optional[int] = None) -> bool:
        if direction not in {"in", "out"}:
            raise ValueError("agent message direction must be in or out")
        mid = body["mid"]
        primary_key = self._pk((mid,))
        values = {
            "text": body["text"], "want": body.get("want"),
            "refs": json.dumps(body["refs"], separators=(",", ":")) if "refs" in body else None,
            "budget": json.dumps(body["budget"], separators=(",", ":")) if "budget" in body else None,
            "harness": body.get("harness"),
        }
        try:
            self._q("INSERT INTO agent_messages (mid,peer,direction,thread,intent,reply_to,expires,text,want,refs,"
                    "budget,harness,created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (mid, peer, direction, body["thread"], body["intent"], body["reply_to"],
                     body.get("expires"), *(self._enc("agent_messages", key, primary_key, value)
                                            for key, value in values.items()),
                     int(created if created is not None else time.time())))
            return True
        except sqlite3.IntegrityError:
            return False

    def agent_mid_known(self, peer: str, thread: str, mid: str) -> bool:
        return bool(self._q("SELECT 1 FROM agent_messages WHERE peer=? AND thread=? AND mid=?",
                            (peer, thread, mid)))

    def _agent_row(self, values) -> dict:
        row = dict(zip(self._AGENT_COLS, values))
        primary_key = self._pk((row["mid"],))
        for column in ("text", "want", "refs", "budget", "harness"):
            row[column] = self._dec("agent_messages", column, primary_key, row[column])
        for column in ("refs", "budget"):
            row[column] = json.loads(row[column]) if row[column] is not None else None
        row.pop("id", None)
        row["trust"] = "external" if row["direction"] == "in" else "local"
        return row

    def agent_threads(self) -> list:
        rows = self._q(f"SELECT {', '.join(self._AGENT_COLS)} FROM agent_messages ORDER BY created, id")
        grouped = {}
        for values in rows:
            row = self._agent_row(values)
            key = (row["peer"], row["thread"])
            summary = grouped.setdefault(key, {"thread": row["thread"], "peer": row["peer"], "count": 0})
            summary.update(last_intent=row["intent"], last_time=row["created"],
                           unread=row["direction"] == "in")
            summary["count"] += 1
        return sorted(grouped.values(), key=lambda row: (row["last_time"], row["thread"]), reverse=True)

    def agent_thread(self, prefix: str, peer: str | None = None) -> tuple[str, list]:
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("thread id or prefix is required")
        query = "SELECT DISTINCT peer,thread FROM agent_messages WHERE thread LIKE ?"
        args: tuple = (prefix + "%",)
        if peer is not None:
            query, args = query + " AND peer=?", args + (peer,)
        matches = self._q(query + " ORDER BY peer,thread", args)
        if not matches:
            raise ValueError("no such thread")
        if len(matches) > 1:
            raise ValueError("thread prefix is ambiguous; specify the contact")
        matched_peer, thread = matches[0]
        rows = self._q(f"SELECT {', '.join(self._AGENT_COLS)} FROM agent_messages "
                       "WHERE peer=? AND thread=? ORDER BY created, id", (matched_peer, thread))
        return thread, [self._agent_row(row) for row in rows]

    # Contact cards are encrypted in their entirety. ``person`` is the row key: the local person's
    # key denotes their own card, while contact person keys denote received cards.
    _CARD_FIELDS = ("display_name", "offers", "intents", "paid", "price_hint", "models", "supports",
                    "updated_at")

    def set_agent_card(self, person: str, card: dict) -> bool:
        existing = self.agent_card(person)
        if existing and existing["updated_at"] >= card["updated_at"]:
            return False
        primary_key = self._pk((person,))
        encoded = {
            key: json.dumps(card[key], separators=(",", ":")) if key in {"intents", "paid", "models", "supports",
                                                                            "updated_at"} else card[key]
            for key in self._CARD_FIELDS
        }
        values = [self._enc("agent_cards", key, primary_key, encoded[key]) for key in self._CARD_FIELDS]
        self._q("INSERT INTO agent_cards (person,display_name,offers,intents,paid,price_hint,models,supports,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(person) DO UPDATE SET " +
                ",".join(f"{key}=excluded.{key}" for key in self._CARD_FIELDS), (person, *values))
        return True

    def agent_card(self, person: str) -> Optional[dict]:
        rows = self._q("SELECT display_name,offers,intents,paid,price_hint,models,supports,updated_at "
                       "FROM agent_cards WHERE person=?", (person,))
        if not rows:
            return None
        primary_key = self._pk((person,))
        row = {key: self._dec("agent_cards", key, primary_key, value)
               for key, value in zip(self._CARD_FIELDS, rows[0])}
        for key in ("intents", "paid", "models", "supports", "updated_at"):
            row[key] = json.loads(row[key])
        return row

    # tasks: one row per (task, my role, counterparty). Poster rows use peer "" (the assignee is a
    # column); worker rows use the poster's person key as peer. Every transition is compare-and-set.
    _TASK_COLS = ("task_hash", "role", "peer", "spec", "state", "targets", "assignee", "epoch", "output",
                  "verdict", "created", "updated")

    def task_insert(self, task_hash: str, role: str, peer: str, spec: dict, state: str,
                    targets: Optional[list] = None) -> bool:
        now = int(time.time())
        primary_key = self._pk((task_hash, role, peer))
        try:
            self._q("INSERT INTO tasks VALUES (?,?,?,?,?,?,NULL,0,NULL,NULL,?,?)",
                    (task_hash, role, peer, self._enc("tasks", "spec", primary_key, json.dumps(spec)), state,
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
        primary_key = self._pk((task_hash, role, peer))
        fields = {key: (self._enc("tasks", key, primary_key, value)
                        if key in ("output", "verdict") and value is not None else value)
                  for key, value in fields.items()}

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
        primary_key = self._pk((d["task_hash"], d["role"], d["peer"]))
        for column in ("spec", "output", "verdict"):
            d[column] = self._dec("tasks", column, primary_key, d[column])
        d["spec"] = json.loads(d["spec"])
        d["targets"] = json.loads(d["targets"]) if d["targets"] else None
        d["verdict"] = json.loads(d["verdict"]) if d["verdict"] else None
        return d
