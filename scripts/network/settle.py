"""Nakshatra settlement: the interface money plugs into, and the local ledger that stands in for it.

A task with a reward moves value in exactly three ways, each at most once per task:

    open     — the poster's signer authorises at post (caps apply); adapter #0 holds then, while
               Neuron waits until assignment so the hold can bind the selected worker key
    release  — when the result is accepted: escrow pays the assigned worker
    refund   — when the result is rejected, or the deadline passes with no accepted result

`SettlementAdapter` is that contract. `LedgerAdapter` is adapter #0: a local sqlite ledger in TEST
units, so the whole flow can be exercised and drilled with no chain and no money.
`NeuronEscrowAdapter` is the explicit, pinned-HTTPS credits backend.

Exactly-once is the ledger's job, not the caller's: every transition is a compare-and-set on the
escrow row, so a replayed accept, a duplicate result or a racing expiry can never pay twice or pay
AND refund.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import nakshatra_auth


class SettlementError(RuntimeError):
    status = 0  # HTTP status when the ledger refused the call, else 0


class InsufficientFunds(SettlementError):
    status = 402


class HoldNotFound(SettlementError):
    status = 404


class HoldConflict(SettlementError):
    status = 409


class HoldNotOpen(SettlementError):
    """The ledger answered, and the hold is in a terminal state (`state`: paid|refunded). Definitive."""

    def __init__(self, message: str, state: str = "", hold: Optional[dict] = None):
        super().__init__(message)
        self.state = state
        self.hold = hold or {}


class LedgerBusy(SettlementError):
    """Local, retryable: signing timestamps ran too far ahead of the clock; nothing was sent."""


# Largest lead of a request timestamp over the wall clock. The ledger accepts +/-60 s, so a
# backlog beyond this is refused locally instead of being sent into a 401 lockout.
MAX_TS_LEAD_S = 30
DEFINITIVE_STATUSES = frozenset({402, 403, 404, 409, 422})

_STATUS_ERRORS = {402: InsufficientFunds, 404: HoldNotFound, 409: HoldConflict}


class SettlementAdapter:
    """What every settlement backend implements. Amounts are integers in the unit's base units."""
    name = "abstract"

    open_on_assign = False

    def open(self, task_hash: str, poster: str, amount: int, unit: str, deadline: int,
             authorisation: dict, *, worker_keyid: str = "") -> dict:
        raise NotImplementedError

    def release(self, task_hash: str, worker: str, *, hold_id: str = "") -> dict:
        raise NotImplementedError

    def refund(self, task_hash: str, reason: str, *, hold_id: str = "") -> dict:
        raise NotImplementedError

    def dispute(self, task_hash: str, reason: str, note: str, *, hold_id: str = "") -> dict:
        raise NotImplementedError

    def status(self, task_hash: str, *, hold_id: str = "") -> Optional[dict]:
        raise NotImplementedError

    def submit(self, task_hash: str, result_hash: str, *, hold_id: str = "") -> dict:
        raise NotImplementedError

    def received(self, task_hash: str, result_hash: str, *, hold_id: str = "") -> dict:
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

    def open(self, task_hash, poster, amount, unit, deadline, authorisation, *, worker_keyid=""):
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

    def release(self, task_hash, worker, *, hold_id=""):
        return self._close(task_hash, "released", worker, "accepted")

    def refund(self, task_hash, reason, *, hold_id=""):
        row = self.status(task_hash)
        return self._close(task_hash, "refunded", (row or {}).get("poster"), reason)

    def status(self, task_hash, *, hold_id=""):
        with self._lock:
            r = self._db.execute("SELECT task_hash, poster, amount, unit, deadline, state, worker, opened, closed, reason "
                                 "FROM escrow WHERE task_hash=?", (task_hash,)).fetchone()
        return dict(zip(("task_hash", "poster", "amount", "unit", "deadline", "state", "worker", "opened",
                         "closed", "reason"), r)) if r else None

    def submit(self, task_hash, result_hash, *, hold_id=""):
        return {"adapter": self.name, "event": "submitted", "task_hash": task_hash,
                "result_hash": result_hash}

    def received(self, task_hash, result_hash, *, hold_id=""):
        return {"adapter": self.name, "event": "received", "task_hash": task_hash,
                "result_hash": result_hash}

    def journal(self, task_hash: Optional[str] = None) -> list:
        q, args = "SELECT seq, ts, task_hash, event, amount, unit, party FROM journal", ()
        if task_hash:
            q, args = q + " WHERE task_hash=?", (task_hash,)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY seq", args).fetchall()
        return [dict(zip(("seq", "ts", "task_hash", "event", "amount", "unit", "party"), r)) for r in rows]


_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_INVITE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_TASK_STATES = {"open", "disputed", "paid", "refunded"}


class NeuronEscrowAdapter(SettlementAdapter):
    """Fail-closed signed client for Neuron's fixed-price task-hold API."""

    name = "neuron"
    open_on_assign = True

    def __init__(self, url: str, spki_sha256: str, private_key: bytes, keyid: str, *,
                 mode: str = "enforce", timeout: float = 5.0, opener: Optional[Callable] = None):
        self.url = (url or "").rstrip("/")
        self.spki_sha256 = (spki_sha256 or "").strip().lower()
        self.private_key = private_key
        self.keyid = keyid
        self.mode = (mode or "").strip().lower()
        self.timeout = float(timeout)
        self._opener = opener
        self._holds: dict[str, str] = {}
        self._ts_lock = threading.Lock()
        self._last_ts = 0
        if self.mode != "enforce":
            raise SettlementError("Neuron task escrow requires NAKSHATRA_LEDGER_MODE=enforce")
        if not self.url.startswith("https://"):
            raise SettlementError("Neuron task escrow requires an HTTPS ledger URL")
        if _HEX64.fullmatch(self.spki_sha256) is None:
            raise SettlementError("Neuron task escrow requires a 64-hex ledger SPKI pin")
        if not isinstance(private_key, bytes) or len(private_key) != 32 or _HEX64.fullmatch(keyid or "") is None:
            raise SettlementError("Neuron task escrow requires a valid node ledger identity")

    def _open(self, request):
        if self._opener is not None:
            return self._opener(request, timeout=self.timeout,
                                expected_spki_hash=self.spki_sha256)
        return nakshatra_auth.open_pillar_url(
            request, timeout=self.timeout, expected_spki_hash=self.spki_sha256)

    def _next_ts(self) -> int:
        """Whole-second timestamp, strictly increasing per adapter: Ed25519 is deterministic and
        the ledger rejects a replayed signature, so two requests must never share (signature)."""
        with self._ts_lock:
            now = int(time.time())
            ts = max(now, self._last_ts + 1)
            if ts > now + MAX_TS_LEAD_S:
                raise LedgerBusy("Neuron ledger signing backlog; retry shortly")
            self._last_ts = ts
            return ts

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        raw = b"" if body is None else json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        label = path.split("?")[0]
        value = None
        # one retry on 401: another process (nak CLI vs nakd) may have used the same second
        for attempt in (0, 1):
            authorization, sent_ts = nakshatra_auth.build_signed_envelope(
                self.private_key, self.keyid, method, path, raw, self._next_ts())
            request = urllib.request.Request(
                self.url + path, data=raw if method == "POST" else None,
                headers={"Authorization": authorization,
                         **({"Content-Type": "application/json"} if method == "POST" else {})},
                method=method)
            try:
                with self._open(request) as response:
                    value = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                # only a same-second collision (ts not ahead of the clock) is worth a retry
                if exc.code == 401 and attempt == 0 and sent_ts <= int(time.time()):
                    continue
                error = _STATUS_ERRORS.get(exc.code, SettlementError)(
                    f"Neuron ledger {label} refused HTTP {exc.code}")
                error.status = exc.code
                try:
                    payload = json.loads(exc.read())
                except Exception:  # noqa: BLE001 - an error body is optional and untrusted
                    payload = {}
                detail = payload.get("error", "") if isinstance(payload, dict) else ""
                error.ledger_error = str(detail)[:200] if isinstance(detail, str) else ""
                code = payload.get("code", "") if isinstance(payload, dict) else ""
                error.ledger_code = str(code)[:64] if isinstance(code, str) else ""
                raise error from exc
            except Exception as exc:  # noqa: BLE001 - enforce mode fails every dependency error closed
                raise SettlementError(
                    f"Neuron ledger {label} unavailable ({type(exc).__name__})") from exc
        if not isinstance(value, dict):
            raise SettlementError("Neuron ledger returned a non-object task response")
        return value

    @staticmethod
    def _hold(value: dict, *, expected_hold: str = "", expected_state: str = "") -> dict:
        required = {"v", "hold_id", "task_hash", "worker_keyid", "amount", "deadline", "state"}
        if not required.issubset(value) or type(value.get("v")) is not int or value["v"] != 1:
            raise SettlementError("Neuron ledger returned an invalid task-hold response")
        if (not isinstance(value["hold_id"], str) or not value["hold_id"]
                or _HEX64.fullmatch(str(value["task_hash"])) is None
                or _HEX64.fullmatch(str(value["worker_keyid"])) is None
                or isinstance(value["amount"], bool) or not isinstance(value["amount"], int)
                or value["amount"] <= 0 or isinstance(value["deadline"], bool)
                or not isinstance(value["deadline"], int) or value["state"] not in _TASK_STATES):
            raise SettlementError("Neuron ledger returned malformed task-hold fields")
        if expected_hold and value["hold_id"] != expected_hold:
            raise SettlementError("Neuron ledger returned a different task hold")
        if expected_state and value["state"] != expected_state:
            raise HoldNotOpen(f"Neuron task hold is {value['state']}, not {expected_state}",
                              state=value["state"], hold=value)
        return value

    @staticmethod
    def _receipt(value: dict, event: str) -> dict:
        return {"adapter": "neuron", "event": event, "task_hash": value["task_hash"],
                "hold_id": value["hold_id"], "amount": value["amount"], "unit": "credits",
                "worker_keyid": value["worker_keyid"], "deadline": value["deadline"]}

    def open(self, task_hash, poster, amount, unit, deadline, authorisation, *, worker_keyid=""):
        if unit != "credits":
            raise SettlementError("Neuron task escrow accepts credits only")
        auth = authorisation or {}
        if (auth.get("task_hash") != task_hash or auth.get("amount") != amount
                or auth.get("mint") != unit):
            raise SettlementError("the spend authorisation does not cover this task, amount and unit")
        if _HEX64.fullmatch(worker_keyid or "") is None:
            raise SettlementError("the assigned contact has no valid ledger key")
        value = self._hold(self._request("POST", "/spend/task/authorize", {
            "client_nonce": task_hash, "task_hash": task_hash, "worker_keyid": worker_keyid,
            "amount": amount, "deadline": int(deadline),
        }), expected_state="open")
        if set(value) != {"v", "hold_id", "task_hash", "worker_keyid", "amount", "deadline", "state"}:
            raise SettlementError("Neuron ledger returned an invalid authorize schema")
        if (value["task_hash"] != task_hash or value["worker_keyid"] != worker_keyid
                or value["amount"] != amount or value["deadline"] != int(deadline)):
            raise SettlementError("Neuron ledger authorized different task-hold terms")
        self._holds[task_hash] = value["hold_id"]
        return self._receipt(value, "open")

    def _hold_id(self, task_hash: str, hold_id: str) -> str:
        resolved = hold_id or self._holds.get(task_hash, "")
        if not resolved:
            raise SettlementError("task hold id is missing")
        return resolved

    def release(self, task_hash, worker, *, hold_id=""):
        resolved = self._hold_id(task_hash, hold_id)
        value = self._hold(self._request("POST", "/spend/task/pay", {"hold_id": resolved}),
                           expected_hold=resolved, expected_state="paid")
        if value["task_hash"] != task_hash:
            raise SettlementError("Neuron ledger paid a different task")
        return self._receipt(value, "released")

    def refund(self, task_hash, reason, *, hold_id=""):
        resolved = self._hold_id(task_hash, hold_id)
        value = self._hold(self._request("POST", "/spend/task/refund", {
            "hold_id": resolved, "reason": str(reason)[:64],
        }), expected_hold=resolved, expected_state="refunded")
        if value["task_hash"] != task_hash:
            raise SettlementError("Neuron ledger refunded a different task")
        return self._receipt(value, "refunded")

    def dispute(self, task_hash, reason, note, *, hold_id=""):
        resolved = self._hold_id(task_hash, hold_id)
        value = self._hold(self._request("POST", "/spend/task/dispute", {
            "hold_id": resolved, "reason": str(reason)[:64], "note": str(note)[:280],
        }), expected_hold=resolved, expected_state="disputed")
        expected = {"v", "hold_id", "task_hash", "worker_keyid", "amount", "deadline",
                    "state", "replayed"}
        if (set(value) != expected or not isinstance(value["replayed"], bool)
                or value["task_hash"] != task_hash):
            raise SettlementError("Neuron ledger returned an invalid task dispute response")
        return self._receipt(value, "disputed")

    def submit(self, task_hash, result_hash, *, hold_id=""):
        if _HEX64.fullmatch(result_hash or "") is None:
            raise SettlementError("task result hash must be 64 lowercase hex characters")
        resolved = self._hold_id(task_hash, hold_id)
        value = self._hold(self._request("POST", "/spend/task/submit", {
            "hold_id": resolved, "result_hash": result_hash,
        }), expected_hold=resolved, expected_state="open")
        expected = {"v", "hold_id", "task_hash", "worker_keyid", "amount", "deadline", "state",
                    "submitted_at", "result_hash", "replayed"}
        if (set(value) != expected or not isinstance(value["replayed"], bool)
                or value["task_hash"] != task_hash
                or isinstance(value["submitted_at"], bool) or not isinstance(value["submitted_at"], int)
                or value["submitted_at"] <= 0 or value["result_hash"] != result_hash):
            raise SettlementError("Neuron ledger returned an invalid task submit response")
        receipt = self._receipt(value, "submitted")
        receipt.update(result_hash=result_hash, submitted_at=value["submitted_at"])
        return receipt

    def received(self, task_hash, result_hash, *, hold_id=""):
        if _HEX64.fullmatch(result_hash or "") is None:
            raise SettlementError("task result hash must be 64 lowercase hex characters")
        resolved = self._hold_id(task_hash, hold_id)
        value = self._hold(self._request("POST", "/spend/task/received", {
            "hold_id": resolved, "result_hash": result_hash,
        }), expected_hold=resolved, expected_state="open")
        expected = {"v", "hold_id", "task_hash", "worker_keyid", "amount", "deadline", "state",
                    "received_at", "received_hash", "replayed"}
        if (set(value) != expected or not isinstance(value["replayed"], bool)
                or value["task_hash"] != task_hash
                or isinstance(value["received_at"], bool) or not isinstance(value["received_at"], int)
                or value["received_at"] <= 0 or value["received_hash"] != result_hash):
            raise SettlementError("Neuron ledger returned an invalid task received response")
        receipt = self._receipt(value, "received")
        receipt.update(result_hash=result_hash, received_at=value["received_at"])
        return receipt

    def status(self, task_hash, *, hold_id=""):
        resolved = self._hold_id(task_hash, hold_id)
        path = "/spend/task/status?" + urllib.parse.urlencode({"hold_id": resolved})
        value = self._hold(self._request("GET", path), expected_hold=resolved)
        required = {"v", "hold_id", "task_hash", "payer_account", "worker_keyid", "amount",
                    "deadline", "state", "server_time", "submitted_at", "result_hash",
                    "received_at", "received_hash", "refund_after_submit"}
        optional = {"dispute_reason", "locked"}
        if (not required.issubset(value) or not set(value).issubset(required | optional)
                or value["task_hash"] != task_hash
                or not isinstance(value["refund_after_submit"], bool)
                or not isinstance(value["payer_account"], str)
                or isinstance(value["server_time"], bool) or not isinstance(value["server_time"], int)
                or isinstance(value["submitted_at"], bool) or not isinstance(value["submitted_at"], int)
                or value["submitted_at"] < 0
                or not isinstance(value["result_hash"], str)
                or (value["result_hash"] and _HEX64.fullmatch(value["result_hash"]) is None)
                or bool(value["submitted_at"]) != bool(value["result_hash"])
                or isinstance(value["received_at"], bool) or not isinstance(value["received_at"], int)
                or value["received_at"] < 0 or not isinstance(value["received_hash"], str)
                or (value["received_hash"] and _HEX64.fullmatch(value["received_hash"]) is None)
                or bool(value["received_at"]) != bool(value["received_hash"])
                or ("locked" in value and (not isinstance(value["locked"], bool)
                                            or value["locked"] != bool(value["submitted_at"])))
                or ("dispute_reason" in value
                    and value["dispute_reason"] is not None
                    and (not isinstance(value["dispute_reason"], str)
                         or not value["dispute_reason"]
                         or len(value["dispute_reason"]) > 64))):
            raise SettlementError("Neuron ledger returned an invalid task status")
        out = dict(value)
        out.setdefault("locked", bool(value["submitted_at"]))
        out.setdefault("dispute_reason", None)
        out["ledger_state"] = value["state"]
        out["state"] = {"open": "open", "disputed": "disputed", "paid": "released",
                        "refunded": "refunded"}[value["state"]]
        out["unit"] = "credits"
        out["adapter"] = self.name
        return out

    def starter(self, invite_id: str, to_keyid: str, amount: int) -> dict:
        """Pay one signed consumer-invite grant through Neuron's idempotent starter route."""
        if _INVITE_ID.fullmatch(invite_id or "") is None:
            raise SettlementError("starter invite id must be 1-64 URL-safe characters")
        if _HEX64.fullmatch(to_keyid or "") is None:
            raise SettlementError("starter recipient key must be 64 lowercase hex characters")
        if isinstance(amount, bool) or not isinstance(amount, int) or not 1 <= amount <= 100_000_000:
            raise SettlementError("starter amount must be between 1 and 100000000")
        value = self._request("POST", "/spend/starter", {
            "invite_id": invite_id, "to_keyid": to_keyid, "amount": amount,
        })
        expected = {"v", "invite_id", "from_account", "to_account", "to_keyid", "amount", "ts",
                    "replayed"}
        if (set(value) != expected or type(value.get("v")) is not int or value["v"] != 1
                or value.get("invite_id") != invite_id or value.get("to_keyid") != to_keyid
                or type(value.get("amount")) is not int or value["amount"] != amount
                or not isinstance(value.get("from_account"), str) or not value["from_account"]
                or not isinstance(value.get("to_account"), str) or not value["to_account"]
                or type(value.get("ts")) is not int or value["ts"] <= 0
                or not isinstance(value.get("replayed"), bool)):
            raise SettlementError("Neuron ledger returned an invalid starter response")
        return dict(value)

    def starter_grant_status(self, invite_id: str) -> dict:
        if _INVITE_ID.fullmatch(invite_id or "") is None:
            raise SettlementError("starter invite id must be 1-64 URL-safe characters")
        path = "/spend/starter/status?" + urllib.parse.urlencode({"invite_id": invite_id})
        value = self._request("GET", path)
        expected = {"v", "invite_id", "from_account", "to_account", "to_keyid", "amount", "ts"}
        if (set(value) != expected or type(value.get("v")) is not int or value["v"] != 1
                or value.get("invite_id") != invite_id
                or _HEX64.fullmatch(str(value.get("to_keyid", ""))) is None
                or type(value.get("amount")) is not int or not 1 <= value["amount"] <= 100_000_000
                or not isinstance(value.get("from_account"), str) or not value["from_account"]
                or not isinstance(value.get("to_account"), str) or not value["to_account"]
                or type(value.get("ts")) is not int or value["ts"] <= 0):
            raise SettlementError("Neuron ledger returned an invalid starter status")
        return dict(value)

    def journal(self, task_hash: Optional[str] = None) -> list:
        """Neuron owns its journal; task status is available through ``nak escrow status``."""
        return []
