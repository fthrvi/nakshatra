"""The node's own messaging state: contacts, connection requests, outstanding invites, outbox, inbox.

One sqlite file per node, every check-and-write in one transaction (the ledger lesson). The inbox is
ALSO appended to a raw JSONL log before anything else reads it (unification lane, 2026-10-02:
friend messages are untrusted external content — raw-log first, source=nakshatra, trust=external).
"""
from __future__ import annotations

import base64
import json
import os
import re
import secrets
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from network.revocation import MAX_GRANTS

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
CREATE TABLE IF NOT EXISTS request_ledger_keys (request_id TEXT PRIMARY KEY, ledger_key TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS invite_codes (invite_id TEXT PRIMARY KEY, code TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS authenticated_contacts (person TEXT PRIMARY KEY, node TEXT NOT NULL,
    authenticated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS starter_obligations (invite_id TEXT PRIMARY KEY, person TEXT NOT NULL,
    to_keyid TEXT, amount INTEGER NOT NULL, signed_invite TEXT NOT NULL, state TEXT NOT NULL,
    reason TEXT, created INTEGER NOT NULL, updated INTEGER NOT NULL, next_attempt INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0, approved INTEGER NOT NULL DEFAULT 0, paid_at INTEGER,
    receipt TEXT);
CREATE TABLE IF NOT EXISTS address_state (slot INTEGER PRIMARY KEY CHECK(slot=1), code TEXT NOT NULL,
    aid TEXT NOT NULL, ik TEXT NOT NULL, expires_at INTEGER NOT NULL, created INTEGER NOT NULL,
    enabled INTEGER NOT NULL, install_line TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS retired_address_aids (aid TEXT PRIMARY KEY, retired_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS request_sources (request_id TEXT PRIMARY KEY, source TEXT NOT NULL,
    source_id TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS request_sources_source_idx ON request_sources(source, source_id);
CREATE TABLE IF NOT EXISTS blocks (person TEXT PRIMARY KEY, until INTEGER NOT NULL, reason TEXT NOT NULL,
    created INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS own_revocation (record_json TEXT NOT NULL, seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS contact_revocations (person TEXT NOT NULL, grant_id TEXT NOT NULL,
    PRIMARY KEY(person, grant_id));
CREATE TABLE IF NOT EXISTS contact_revocation_seq (person TEXT PRIMARY KEY, seq INTEGER NOT NULL,
    received_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS revocation_sent (person TEXT NOT NULL, node TEXT NOT NULL,
    seq INTEGER NOT NULL, PRIMARY KEY(person, node));
"""


class Store:
    _ENCRYPTED = {
        "inbox": (("nonce",), ("text", "petname", "author", "aspect")),
        "contacts": (("person",), ("petname", "nickname")),
        "requests": (("id",), ("nickname",)),
        "invites": (("nonce",), ("note",)),
        "invite_codes": (("invite_id",), ("code",)),
        "outbox": (("nonce",), ("frame", "refused")),
        "tasks": (("task_hash", "role", "peer"), ("spec", "output", "verdict", "escrow_auth")),
        "agent_messages": (("mid",), ("text", "want", "refs", "budget", "harness")),
        "agent_cards": (("person",), ("display_name", "offers", "intents", "paid", "price_hint",
                                          "models", "supports", "updated_at", "ledger_keyid")),
        "starter_obligations": (("invite_id",), ("signed_invite", "reason", "receipt")),
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
        self._migrate_revocation_sent()
        # An address disabled by the pre-hardening build is already retired evidence. Preserve it
        # before any caller can try to re-register the same aid after an upgrade.
        self._db.execute(
            "INSERT OR IGNORE INTO retired_address_aids(aid,retired_at) "
            "SELECT aid,created FROM address_state WHERE enabled=0")
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
        for col in ("escrow_auth TEXT", "hold_id TEXT", "funding_state TEXT", "funding_reason TEXT",
                    "result_hash TEXT", "submit_state TEXT", "submit_reason TEXT",
                    "submit_attempts INTEGER NOT NULL DEFAULT 0", "receive_state TEXT",
                    "receive_reason TEXT", "receive_attempts INTEGER NOT NULL DEFAULT 0"):
            try:
                self._db.execute(f"ALTER TABLE tasks ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        try:
            self._db.execute("ALTER TABLE agent_cards ADD COLUMN ledger_keyid TEXT")
        except sqlite3.OperationalError:
            pass
        self._key_verified = self._verify_key_marker()
        self._migrate_starter_side_tables()
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

    def _has_column(self, table: str, column: str) -> bool:
        return any(row[1] == column for row in self._db.execute(f"PRAGMA table_info({table})"))

    def _migrate_starter_side_tables(self) -> None:
        """Adopt databases opened by the first starter branch without altering legacy tables.

        That version appended columns to ``requests`` and ``invites``. Keeping those columns is
        harmless, but fresh databases must retain the old positional-insert shapes so the previous
        binary can roll back safely.
        """
        if self._has_column("requests", "ledger_key"):
            self._db.execute(
                "INSERT OR IGNORE INTO request_ledger_keys(request_id,ledger_key) "
                "SELECT id,ledger_key FROM requests WHERE ledger_key IS NOT NULL AND ledger_key<>''")
        if self._has_column("invites", "code"):
            rows = self._db.execute(
                "SELECT nonce,code FROM invites WHERE code IS NOT NULL AND code<>''").fetchall()
            for nonce, sealed in rows:
                code = self._dec("invites", "code", self._pk((nonce,)), sealed)
                self._db.execute(
                    "INSERT OR IGNORE INTO invite_codes(invite_id,code) VALUES (?,?)",
                    (nonce, self._enc("invite_codes", "code", self._pk((nonce,)), code)))

    def _migrate_revocation_sent(self) -> None:
        """Replace the branch's person-only ACK cache with fail-closed person+node state.

        Old rows cannot be bound safely: the contact node may already have changed before this
        upgrade. Dropping only this public, replayable side table makes every contact receive the
        cumulative record once more and leaves every pre-existing non-revocation table untouched.
        """
        if self._has_column("revocation_sent", "node"):
            return
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute("DROP TABLE revocation_sent")
            self._db.execute(
                "CREATE TABLE revocation_sent (person TEXT NOT NULL, node TEXT NOT NULL, "
                "seq INTEGER NOT NULL, PRIMARY KEY(person, node))")
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise

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
    def add_invite(self, nonce: str, ik: str, expires_at: int, note: str = "", code: str = "") -> None:
        note = self._enc("invites", "note", self._pk((nonce,)), note)
        sealed_code = self._enc("invite_codes", "code", self._pk((nonce,)), code)

        def fn(db):
            db.execute("INSERT INTO invites (nonce,ik,expires_at,created,note,consumed_by) "
                       "VALUES (?,?,?,?,?,NULL)", (nonce, ik, expires_at, int(time.time()), note))
            db.execute("INSERT INTO invite_codes(invite_id,code) VALUES (?,?)", (nonce, sealed_code))
        self._tx(fn)

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

    def invite(self, nonce: str) -> Optional[dict]:
        rows = self._q(
            "SELECT i.nonce,i.ik,i.expires_at,i.created,i.note,i.consumed_by,c.code "
            "FROM invites i LEFT JOIN invite_codes c ON c.invite_id=i.nonce WHERE i.nonce=?", (nonce,))
        if not rows:
            return None
        row = dict(zip(("nonce", "ik", "expires_at", "created", "note", "consumed_by", "code"), rows[0]))
        primary_key = self._pk((nonce,))
        row["note"] = self._dec("invites", "note", primary_key, row["note"])
        row["code"] = self._dec("invite_codes", "code", primary_key, row["code"])
        return row

    # public request address (additive side table; legacy request/invite shapes stay untouched)
    def set_address(self, code: str, aid: str, ik: str, expires_at: int, install_line: str = "",
                    now: Optional[int] = None) -> None:
        now = int(time.time() if now is None else now)

        def fn(db):
            if db.execute("SELECT 1 FROM retired_address_aids WHERE aid=?", (aid,)).fetchone():
                raise ValueError("that request address aid was retired and cannot be registered again")
            current = db.execute("SELECT aid,enabled FROM address_state WHERE slot=1").fetchone()
            if current and not current[1]:
                db.execute("INSERT OR IGNORE INTO retired_address_aids(aid,retired_at) VALUES (?,?)",
                           (current[0], now))
                if current[0] == aid:
                    raise ValueError("that request address aid was retired and cannot be registered again")
            if current and current[0] != aid:
                db.execute("INSERT OR IGNORE INTO retired_address_aids(aid,retired_at) VALUES (?,?)",
                           (current[0], now))
            db.execute(
                "INSERT INTO address_state(slot,code,aid,ik,expires_at,created,enabled,install_line) "
                "VALUES(1,?,?,?,?,?,1,?) "
                "ON CONFLICT(slot) DO UPDATE SET code=excluded.code,aid=excluded.aid,ik=excluded.ik,"
                "expires_at=excluded.expires_at,created=excluded.created,enabled=1,"
                "install_line=excluded.install_line",
                (code, aid, ik, int(expires_at), now, install_line))
        self._tx(fn)

    def address(self, *, active_only: bool = False, now: Optional[int] = None) -> Optional[dict]:
        rows = self._q(
            "SELECT code,aid,ik,expires_at,created,enabled,install_line FROM address_state WHERE slot=1")
        if not rows:
            return None
        row = dict(zip(("code", "aid", "ik", "expires_at", "created", "enabled", "install_line"), rows[0]))
        row["enabled"] = bool(row["enabled"])
        if active_only and (not row["enabled"] or row["expires_at"] <= int(
                time.time() if now is None else now)):
            return None
        return row

    def retire_address(self, aid: str, *, now: Optional[int] = None) -> bool:
        now = int(time.time() if now is None else now)

        def fn(db):
            current = db.execute("SELECT aid,enabled FROM address_state WHERE slot=1").fetchone()
            if not current or current[0] != aid:
                raise ValueError("that request address is not current")
            if not current[1]:
                return False
            db.execute("INSERT OR IGNORE INTO retired_address_aids(aid,retired_at) VALUES (?,?)",
                       (aid, now))
            db.execute("UPDATE address_state SET enabled=0 WHERE slot=1")
            return True
        return self._tx(fn)

    # requests (someone redeemed our invite and asks to connect)
    def add_request(self, person: str, node: str, nickname: str, invite_nonce: str,
                    ledger_key: str = "", *, source: str = "invite", source_id: str = "",
                    received: Optional[int] = None) -> str:
        rid = secrets.token_hex(4)
        nickname = self._enc("requests", "nickname", self._pk((rid,)), nickname[:40])
        received = int(time.time() if received is None else received)

        def fn(db):
            db.execute("INSERT INTO requests (id,person,node,nickname,invite_nonce,received,state) "
                       "VALUES (?,?,?,?,?,?,'pending')",
                       (rid, person, node, nickname, invite_nonce, received))
            if ledger_key:
                db.execute("INSERT INTO request_ledger_keys(request_id,ledger_key) VALUES (?,?)",
                           (rid, ledger_key))
            if source != "invite":
                db.execute("INSERT INTO request_sources(request_id,source,source_id) VALUES (?,?,?)",
                           (rid, source, source_id))
        self._tx(fn)
        return rid

    def requests(self, state: str = "pending") -> list:
        out = []
        for r in self._q(
                "SELECT r.id,r.person,r.node,r.nickname,r.received,k.ledger_key,s.source,s.source_id "
                "FROM requests r LEFT JOIN request_ledger_keys k ON k.request_id=r.id "
                "LEFT JOIN request_sources s ON s.request_id=r.id WHERE r.state=?", (state,)):
            row = dict(zip(("id", "person", "node", "nickname", "received", "ledger_key",
                            "source", "source_id"), r))
            row["nickname"] = self._dec("requests", "nickname", self._pk((row["id"],)), row["nickname"])
            row["ledger_key"] = row["ledger_key"] or ""
            row["source"] = row["source"] or "invite"
            row["source_id"] = row["source_id"] or ""
            out.append(row)
        return out

    def request(self, rid: str, state: str = "pending") -> Optional[dict]:
        rows = self._q(
            "SELECT r.id,r.person,r.node,r.nickname,r.invite_nonce,r.received,r.state,k.ledger_key,"
            "s.source,s.source_id "
            "FROM requests r LEFT JOIN request_ledger_keys k ON k.request_id=r.id "
            "LEFT JOIN request_sources s ON s.request_id=r.id "
            "WHERE r.id=? AND r.state=?", (rid, state))
        if not rows:
            return None
        row = dict(zip(("id", "person", "node", "nickname", "invite_nonce", "received", "state",
                        "ledger_key", "source", "source_id"), rows[0]))
        row["nickname"] = self._dec("requests", "nickname", self._pk((rid,)), row["nickname"])
        row["ledger_key"] = row["ledger_key"] or ""
        row["source"] = row["source"] or "invite"
        row["source_id"] = row["source_id"] or ""
        return row

    def decide_request(self, rid: str, accept: bool, starter: Optional[dict] = None,
                       *, block_until: int = 0, block_reason: str = "") -> Optional[dict]:
        def fn(db):
            r = db.execute(
                "SELECT r.person,r.node,r.nickname,r.invite_nonce,k.ledger_key,s.source,s.source_id "
                "FROM requests r "
                "LEFT JOIN request_ledger_keys k ON k.request_id=r.id "
                "LEFT JOIN request_sources s ON s.request_id=r.id "
                "WHERE r.id=? AND r.state='pending'", (rid,)).fetchone()
            if not r:
                return None
            db.execute("UPDATE requests SET state=? WHERE id=?", ("accepted" if accept else "declined", rid))
            if not accept and block_until:
                db.execute(
                    "INSERT INTO blocks(person,until,reason,created) VALUES(?,?,?,?) "
                    "ON CONFLICT(person) DO UPDATE SET until=excluded.until,reason=excluded.reason,"
                    "created=excluded.created",
                    (r[0], int(block_until), str(block_reason)[:200], int(time.time())))
            if accept and starter is not None:
                invite_id = starter["invite_id"]
                pk = self._pk((invite_id,))
                db.execute(
                    "INSERT INTO starter_obligations "
                    "(invite_id,person,to_keyid,amount,signed_invite,state,reason,created,updated,next_attempt) "
                    "VALUES (?,?,?,?,?,'pending',?,?,?,?)",
                    (invite_id, r[0], starter.get("to_keyid") or None, int(starter["amount"]),
                     self._enc("starter_obligations", "signed_invite", pk, starter["signed_invite"]),
                     self._enc("starter_obligations", "reason", pk, starter.get("reason", "")),
                     int(starter["created"]), int(starter["created"]), int(starter["created"])))
            return {"person": r[0], "node": r[1],
                    "nickname": self._dec("requests", "nickname", self._pk((rid,)), r[2]),
                    "invite_nonce": r[3], "ledger_key": r[4] or "",
                    "source": r[5] or "invite", "source_id": r[6] or ""}
        return self._tx(fn)

    def pending_address_counts(self, person: str) -> tuple[int, int]:
        rows = self._q(
            "SELECT SUM(CASE WHEN r.person=? THEN 1 ELSE 0 END),COUNT(*) FROM requests r "
            "JOIN request_sources s ON s.request_id=r.id "
            "WHERE r.state='pending' AND s.source='address'", (person,))
        mine, total = rows[0] if rows else (0, 0)
        return int(mine or 0), int(total or 0)

    def recent_address_request_count(self, since: int) -> int:
        rows = self._q(
            "SELECT COUNT(*) FROM requests r JOIN request_sources s ON s.request_id=r.id "
            "WHERE s.source='address' AND r.received>=?", (int(since),))
        return int(rows[0][0])

    @staticmethod
    def _expire_address_requests(db, cutoff: int) -> int:
        cur = db.execute(
            "UPDATE requests SET state='expired' WHERE state='pending' AND received<=? "
            "AND id IN (SELECT request_id FROM request_sources WHERE source='address')",
            (int(cutoff),))
        return cur.rowcount

    def expire_address_requests(self, now: int, ttl_s: int) -> int:
        return self._tx(lambda db: self._expire_address_requests(db, int(now) - int(ttl_s)))

    def admit_address_request(self, person: str, node: str, nickname: str, ledger_key: str,
                              aid: str, *, now: int, pending_ttl_s: int,
                              pending_total_max: int, daily_max: int) -> tuple[Optional[str], str, int]:
        """Check every stranger gate and insert under one BEGIN IMMEDIATE transaction."""
        rid = secrets.token_hex(4)
        sealed_nickname = self._enc("requests", "nickname", self._pk((rid,)), nickname[:40])

        def fn(db):
            expired = self._expire_address_requests(db, int(now) - int(pending_ttl_s))
            blocked = db.execute("SELECT 1 FROM blocks WHERE person=? AND until>?",
                                 (person, int(now))).fetchone()
            if blocked:
                return None, "blocked", expired
            if db.execute("SELECT 1 FROM contacts WHERE person=?", (person,)).fetchone():
                return None, "already a contact", expired
            mine, total = db.execute(
                "SELECT SUM(CASE WHEN r.person=? THEN 1 ELSE 0 END),COUNT(*) FROM requests r "
                "JOIN request_sources s ON s.request_id=r.id "
                "WHERE r.state='pending' AND s.source='address'", (person,)).fetchone()
            if int(mine or 0) >= 1:
                return None, "already has a pending address request", expired
            if int(total or 0) >= int(pending_total_max):
                return None, "pending address request cap reached", expired
            recent = db.execute(
                "SELECT COUNT(*) FROM requests r JOIN request_sources s ON s.request_id=r.id "
                "WHERE s.source='address' AND r.received>=?", (int(now) - 86400,)).fetchone()[0]
            if int(recent) >= int(daily_max):
                return None, "daily address request cap reached", expired
            db.execute("INSERT INTO requests (id,person,node,nickname,invite_nonce,received,state) "
                       "VALUES (?,?,?,?,?,?, 'pending')",
                       (rid, person, node, sealed_nickname, "", int(now)))
            if ledger_key:
                db.execute("INSERT INTO request_ledger_keys(request_id,ledger_key) VALUES (?,?)",
                           (rid, ledger_key))
            db.execute("INSERT INTO request_sources(request_id,source,source_id) VALUES (?, 'address', ?)",
                       (rid, aid))
            return rid, "", expired
        return self._tx(fn)

    # address-request block list
    def block(self, person: str, until: int, reason: str = "manual block",
              now: Optional[int] = None) -> None:
        self._q(
            "INSERT INTO blocks(person,until,reason,created) VALUES(?,?,?,?) "
            "ON CONFLICT(person) DO UPDATE SET until=excluded.until,reason=excluded.reason,"
            "created=excluded.created",
            (person, int(until), str(reason)[:200], int(time.time() if now is None else now)))

    def unblock(self, person: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM blocks WHERE person=?", (person,))
            return cur.rowcount == 1

    def is_blocked(self, person: str, now: Optional[int] = None) -> bool:
        rows = self._q("SELECT until FROM blocks WHERE person=?", (person,))
        return bool(rows and int(rows[0][0]) > int(time.time() if now is None else now))

    def blocks(self, now: Optional[int] = None) -> list[dict]:
        now = int(time.time() if now is None else now)
        return [dict(zip(("person", "until", "reason", "created"), row)) for row in self._q(
            "SELECT person,until,reason,created FROM blocks WHERE until>? ORDER BY until,person", (now,))]

    def mark_contact_authenticated(self, person: str, node: str, now: Optional[int] = None) -> None:
        self._q(
            "INSERT INTO authenticated_contacts(person,node,authenticated_at) VALUES (?,?,?) "
            "ON CONFLICT(person) DO UPDATE SET node=excluded.node,authenticated_at=excluded.authenticated_at",
            (person, node, int(time.time() if now is None else now)))

    def authenticated_contact_node(self, person: str) -> str:
        rows = self._q("SELECT node FROM authenticated_contacts WHERE person=?", (person,))
        return rows[0][0] if rows else ""

    # starter credits
    def starter_rows(self, states: Optional[tuple[str, ...]] = None) -> list[dict]:
        columns = ("invite_id", "person", "to_keyid", "amount", "signed_invite", "state", "reason",
                   "created", "updated", "next_attempt", "attempts", "approved", "paid_at", "receipt")
        sql, args = "SELECT " + ",".join(columns) + " FROM starter_obligations", ()
        if states:
            sql += " WHERE state IN (" + ",".join("?" for _ in states) + ")"
            args = tuple(states)
        rows = []
        for values in self._q(sql + " ORDER BY created,invite_id", args):
            row = dict(zip(columns, values))
            pk = self._pk((row["invite_id"],))
            for column in ("signed_invite", "reason", "receipt"):
                row[column] = self._dec("starter_obligations", column, pk, row[column])
            if row["receipt"]:
                row["receipt"] = json.loads(row["receipt"])
            row["approved"] = int(row["approved"] or 0)
            rows.append(row)
        return rows

    def starter_update(self, invite_id: str, **changes) -> bool:
        allowed = {"state", "reason", "updated", "next_attempt", "attempts", "approved", "paid_at", "receipt"}
        if not changes or set(changes) - allowed:
            raise ValueError("invalid starter obligation update")
        pk = self._pk((invite_id,))
        values = dict(changes)
        for column in ("reason", "receipt"):
            if column in values:
                value = values[column]
                if column == "receipt" and value is not None:
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"))
                values[column] = self._enc("starter_obligations", column, pk, value)
        sets = ",".join(f"{column}=?" for column in values)
        self._q(f"UPDATE starter_obligations SET {sets} WHERE invite_id=?",
                (*values.values(), invite_id))
        return True

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
        """A petname or a person key (or its prefix of at least 8 hex chars).

        An exact person key always wins. Anything else that matches more than one contact is
        refused rather than silently picking the first, so a duplicate petname or a short shared
        prefix can never address the wrong person."""
        if not who:
            return None
        contacts = self.contacts()
        for c in contacts:
            if c["person"] == who:
                return c
        hits = [c for c in contacts
                if c["petname"] == who or (len(who) >= 8 and c["person"].startswith(who))]
        if len(hits) > 1:
            raise ValueError(f"{who!r} matches {len(hits)} contacts; use a longer person key prefix")
        return hits[0] if hits else None

    # Public, person-signed cumulative revocation state. These intentionally remain plaintext:
    # grant ids and signed records are shared with every contact and contain no message content.
    def set_own_revocation(self, record: dict) -> None:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        def fn(db):
            row = db.execute("SELECT record_json,seq FROM own_revocation LIMIT 1").fetchone()
            if row:
                try:
                    current = json.loads(row[0])
                    current_ids = set(current["grant_ids"])
                    current_seq = current["seq"]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError("installed own revocation record is malformed") from exc
                if type(current_seq) is not int or current_seq != int(row[1]):
                    raise ValueError("installed own revocation record is malformed")
                if int(record["seq"]) <= current_seq:
                    raise ValueError("revocation seq must be newer than the installed record")
                if not current_ids.issubset(record["grant_ids"]):
                    raise ValueError("new revocation record must retain every previously revoked grant")
            db.execute("DELETE FROM own_revocation")
            db.execute("INSERT INTO own_revocation(record_json,seq) VALUES (?,?)",
                       (encoded, int(record["seq"])))
        self._tx(fn)

    def own_revocation(self) -> Optional[dict]:
        rows = self._q("SELECT record_json FROM own_revocation LIMIT 1")
        return json.loads(rows[0][0]) if rows else None

    def apply_contact_revocation(self, person: str, record: dict) -> bool:
        seq = record["seq"]
        now = int(time.time())

        def fn(db):
            current = db.execute(
                "SELECT seq FROM contact_revocation_seq WHERE person=?", (person,)).fetchone()
            if current and seq <= current[0]:
                return False
            existing = {row[0] for row in db.execute(
                "SELECT grant_id FROM contact_revocations WHERE person=?", (person,)).fetchall()}
            if len(existing | set(record["grant_ids"])) > MAX_GRANTS:
                raise ValueError(f"cumulative revocation set exceeds {MAX_GRANTS} grant ids")
            db.executemany(
                "INSERT OR IGNORE INTO contact_revocations(person,grant_id) VALUES (?,?)",
                ((person, grant_id) for grant_id in record["grant_ids"]))
            db.execute(
                "INSERT INTO contact_revocation_seq(person,seq,received_at) VALUES (?,?,?) "
                "ON CONFLICT(person) DO UPDATE SET seq=excluded.seq,received_at=excluded.received_at",
                (person, seq, now))
            return True
        return self._tx(fn)

    def revoked_grants(self, person: str) -> set[str]:
        return {row[0] for row in self._q(
            "SELECT grant_id FROM contact_revocations WHERE person=?", (person,))}

    def contact_revocation_seq(self, person: str) -> int:
        rows = self._q("SELECT seq FROM contact_revocation_seq WHERE person=?", (person,))
        return int(rows[0][0]) if rows else 0

    def revocation_sent_seq(self, person: str, node: str) -> int:
        rows = self._q("SELECT seq FROM revocation_sent WHERE person=? AND node=?", (person, node))
        return int(rows[0][0]) if rows else 0

    def mark_revocation_sent(self, person: str, node: str, seq: int) -> None:
        self._q(
            "INSERT INTO revocation_sent(person,node,seq) VALUES (?,?,?) "
            "ON CONFLICT(person,node) DO UPDATE SET seq=MAX(revocation_sent.seq,excluded.seq)",
            (person, node, int(seq)))

    def revocation_status(self) -> dict:
        own = self.own_revocation()
        received = {person: {"seq": int(seq), "count": int(count)}
                    for person, seq, count in self._q(
                        "SELECT s.person,s.seq,COUNT(r.grant_id) FROM contact_revocation_seq s "
                        "LEFT JOIN contact_revocations r ON r.person=s.person GROUP BY s.person,s.seq")}
        sent = {person: int(seq) for person, seq in self._q(
            "SELECT r.person,r.seq FROM revocation_sent r JOIN contacts c "
            "ON c.person=r.person AND c.node=r.node")}
        return {"own": own, "received": received, "sent": sent}

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
        def fn(db):
            db.execute("DELETE FROM contacts WHERE person=?", (person,))
            db.execute("DELETE FROM revocation_sent WHERE person=?", (person,))
        self._tx(fn)

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

    def outbox_frame(self, nonce: str, to_person: str) -> Optional[dict]:
        """The frame we queued under `nonce` to `to_person` (None if it is not ours to that person)."""
        r = self._q("SELECT frame FROM outbox WHERE nonce=? AND to_person=?", (nonce, to_person))
        return json.loads(self._dec("outbox", "frame", self._pk((nonce,)), r[0][0])) if r else None

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
        if not isinstance(prefix, str) or not re.fullmatch(r"[0-9a-f]{1,32}", prefix):
            raise ValueError("thread id or prefix must be 1-32 lowercase hex characters")
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
                    "updated_at", "ledger_keyid")

    def set_agent_card(self, person: str, card: dict) -> bool:
        existing = self.agent_card(person)
        if existing and existing["updated_at"] >= card["updated_at"]:
            return False
        primary_key = self._pk((person,))
        encoded = {
            key: json.dumps(card[key], separators=(",", ":")) if key in {"intents", "paid", "models", "supports",
                                                                            "updated_at"} else card.get(key)
            for key in self._CARD_FIELDS
        }
        values = [self._enc("agent_cards", key, primary_key, encoded[key]) for key in self._CARD_FIELDS]
        self._q("INSERT INTO agent_cards (person,display_name,offers,intents,paid,price_hint,models,supports,updated_at,ledger_keyid) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(person) DO UPDATE SET " +
                ",".join(f"{key}=excluded.{key}" for key in self._CARD_FIELDS), (person, *values))
        return True

    def agent_card(self, person: str) -> Optional[dict]:
        rows = self._q("SELECT display_name,offers,intents,paid,price_hint,models,supports,updated_at,ledger_keyid "
                       "FROM agent_cards WHERE person=?", (person,))
        if not rows:
            return None
        primary_key = self._pk((person,))
        row = {key: self._dec("agent_cards", key, primary_key, value)
               for key, value in zip(self._CARD_FIELDS, rows[0])}
        for key in ("intents", "paid", "models", "supports", "updated_at"):
            row[key] = json.loads(row[key])
        if row["ledger_keyid"] is None:
            row.pop("ledger_keyid")
        return row

    # tasks: one row per (task, my role, counterparty). Poster rows use peer "" (the assignee is a
    # column); worker rows use the poster's person key as peer. Every transition is compare-and-set.
    _TASK_COLS = ("task_hash", "role", "peer", "spec", "state", "targets", "assignee", "epoch", "output",
                  "verdict", "escrow_auth", "hold_id", "funding_state", "funding_reason", "result_hash",
                  "submit_state", "submit_reason", "submit_attempts", "receive_state", "receive_reason",
                  "receive_attempts", "created", "updated")

    def task_insert(self, task_hash: str, role: str, peer: str, spec: dict, state: str,
                    targets: Optional[list] = None, *, escrow_auth: Optional[dict] = None,
                    hold_id: str = "", funding_state: str = "", funding_reason: str = "") -> bool:
        now = int(time.time())
        primary_key = self._pk((task_hash, role, peer))
        try:
            self._q("INSERT INTO tasks (task_hash,role,peer,spec,state,targets,assignee,epoch,output,verdict,"
                    "escrow_auth,hold_id,funding_state,funding_reason,result_hash,submit_state,submit_reason,"
                    "submit_attempts,receive_state,receive_reason,receive_attempts,created,updated) "
                    "VALUES (?,?,?,?,?,?,NULL,0,NULL,NULL,?,?,?,?,NULL,NULL,NULL,0,NULL,NULL,0,?,?)",
                    (task_hash, role, peer, self._enc("tasks", "spec", primary_key, json.dumps(spec)), state,
                     json.dumps(targets) if targets is not None else None,
                     self._enc("tasks", "escrow_auth", primary_key, json.dumps(escrow_auth))
                     if escrow_auth is not None else None,
                     hold_id or None, funding_state or None, funding_reason or None, now, now))
            return True
        except sqlite3.IntegrityError:
            return False

    def task_cas(self, task_hash: str, role: str, peer: str, from_states: tuple, **fields) -> bool:
        """Move a task out of one of `from_states`, setting `fields`, atomically. False if it was not
        in one of those states (someone else won, or the step is out of order)."""
        allowed = {"state", "assignee", "epoch", "output", "verdict", "hold_id", "funding_state",
                   "funding_reason", "result_hash", "submit_state", "submit_reason", "submit_attempts",
                   "receive_state", "receive_reason", "receive_attempts"}
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
        for column in ("spec", "output", "verdict", "escrow_auth"):
            d[column] = self._dec("tasks", column, primary_key, d[column])
        d["spec"] = json.loads(d["spec"])
        d["escrow_auth"] = json.loads(d["escrow_auth"]) if d["escrow_auth"] else None
        d["targets"] = json.loads(d["targets"]) if d["targets"] else None
        d["verdict"] = json.loads(d["verdict"]) if d["verdict"] else None
        return d
