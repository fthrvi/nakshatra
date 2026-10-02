"""Nakshatra settlement: the interface money plugs into, and the local ledger that stands in for it.

A task with a reward moves value in exactly three ways, each at most once per task:

    open     — when the task is posted: the poster's signer authorises the spend (caps apply; over
               a cap it needs the person), and the reward is held in escrow
    release  — when the result is accepted: escrow pays the assigned worker
    refund   — when the result is rejected, or the deadline passes with no accepted result

`SettlementAdapter` is that contract. `LedgerAdapter` is adapter #0: a local sqlite ledger in TEST
units, so the whole flow can be exercised and drilled with no chain and no money. The Solana devnet
escrow (hackathon days 8-9) is adapter #1 behind the same four calls; nothing in nakd changes.

Exactly-once is the ledger's job, not the caller's: every transition is a compare-and-set on the
escrow row, so a replayed accept, a duplicate result or a racing expiry can never pay twice or pay
AND refund.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


class SettlementError(Exception):
    pass


class SettlementAdapter:
    """What every settlement backend implements. Amounts are integers in the unit's base units."""
    name = "abstract"

    def open(self, task_hash: str, poster: str, amount: int, unit: str, deadline: int,
             authorisation: dict) -> dict:
        raise NotImplementedError

    def release(self, task_hash: str, worker: str) -> dict:
        raise NotImplementedError

    def refund(self, task_hash: str, reason: str) -> dict:
        raise NotImplementedError

    def status(self, task_hash: str) -> Optional[dict]:
        raise NotImplementedError


class LedgerAdapter(SettlementAdapter):
    """Adapter #0: escrow as rows in a local ledger. No money moves; the transitions are real."""
    name = "ledger"

    def __init__(self, path: Path):
        self._db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS escrow (task_hash TEXT PRIMARY KEY, poster TEXT NOT NULL,
            amount INTEGER NOT NULL, unit TEXT NOT NULL, deadline INTEGER NOT NULL, state TEXT NOT NULL,
            worker TEXT, authorisation TEXT NOT NULL, opened INTEGER NOT NULL, closed INTEGER, reason TEXT);
        CREATE TABLE IF NOT EXISTS journal (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL,
            task_hash TEXT NOT NULL, event TEXT NOT NULL, amount INTEGER NOT NULL, unit TEXT NOT NULL,
            party TEXT);
        """)
        self._lock = threading.Lock()

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

    def open(self, task_hash, poster, amount, unit, deadline, authorisation):
        if not isinstance(amount, int) or amount <= 0:
            raise SettlementError("escrow needs a positive integer amount")
        auth = authorisation or {}
        if auth.get("task_hash") != task_hash or auth.get("amount") != amount or auth.get("mint") != unit:
            raise SettlementError("the spend authorisation does not cover this task, amount and unit")
        now = int(time.time())

        def fn(db):
            try:
                db.execute("INSERT INTO escrow VALUES (?,?,?,?,?,'open',NULL,?,?,NULL,NULL)",
                           (task_hash, poster, amount, unit, int(deadline), json.dumps(auth), now))
            except sqlite3.IntegrityError:
                raise SettlementError("escrow for this task is already open")
            db.execute("INSERT INTO journal (ts, task_hash, event, amount, unit, party) VALUES (?,?,?,?,?,?)",
                       (now, task_hash, "open", amount, unit, poster))
            return {"adapter": self.name, "event": "open", "task_hash": task_hash, "amount": amount, "unit": unit}
        return self._tx(fn)

    def _close(self, task_hash, to_state, party, reason):
        now = int(time.time())

        def fn(db):
            cur = db.execute("UPDATE escrow SET state=?, worker=COALESCE(?, worker), closed=?, reason=? "
                             "WHERE task_hash=? AND state='open'", (to_state, party if to_state == "released" else None,
                                                                     now, reason, task_hash))
            if cur.rowcount != 1:
                row = db.execute("SELECT state FROM escrow WHERE task_hash=?", (task_hash,)).fetchone()
                raise SettlementError(f"escrow is {row[0] if row else 'missing'}, not open; nothing was {to_state}")
            amount, unit = db.execute("SELECT amount, unit FROM escrow WHERE task_hash=?", (task_hash,)).fetchone()
            db.execute("INSERT INTO journal (ts, task_hash, event, amount, unit, party) VALUES (?,?,?,?,?,?)",
                       (now, task_hash, to_state, amount, unit, party))
            return {"adapter": self.name, "event": to_state, "task_hash": task_hash, "amount": amount,
                    "unit": unit, "to": party, "reason": reason}
        return self._tx(fn)

    def release(self, task_hash, worker):
        return self._close(task_hash, "released", worker, "accepted")

    def refund(self, task_hash, reason):
        row = self.status(task_hash)
        return self._close(task_hash, "refunded", (row or {}).get("poster"), reason)

    def status(self, task_hash):
        with self._lock:
            r = self._db.execute("SELECT task_hash, poster, amount, unit, deadline, state, worker, opened, closed, reason "
                                 "FROM escrow WHERE task_hash=?", (task_hash,)).fetchone()
        return dict(zip(("task_hash", "poster", "amount", "unit", "deadline", "state", "worker", "opened",
                         "closed", "reason"), r)) if r else None

    def journal(self, task_hash: Optional[str] = None) -> list:
        q, args = "SELECT seq, ts, task_hash, event, amount, unit, party FROM journal", ()
        if task_hash:
            q, args = q + " WHERE task_hash=?", (task_hash,)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY seq", args).fetchall()
        return [dict(zip(("seq", "ts", "task_hash", "event", "amount", "unit", "party"), r)) for r in rows]
