"""Payment-bound worker admission and durable usage settlement.

The module is intentionally inert until worker.py's single paid-admission gate
constructs :class:`PaidAdmission`. Claim tokens are accepted only as method
arguments, are never logged, and only their SHA-256 digest reaches SQLite.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Mapping, Optional

from cryptography.hazmat.primitives.asymmetric import ed25519

import nakshatra_auth


CLAIM_RESPONSE_FIELDS = frozenset({
    "v", "hold_id", "token_sha256", "claim_instance", "payer_keyid",
    "payer_account_at_authorize", "worker_keyid",
    "worker_account_at_authorize", "billable", "session_id", "model_id",
    "billing_mode", "layer_start", "layer_end", "price_multiplier",
    "max_prompt_tokens", "max_output_tokens", "max_token_layers", "max_cost",
    "execution_deadline", "settlement_deadline", "server_time",
    "claim_response_sha256",
})
READY_RESPONSE_FIELDS = frozenset({"v", "ready", "capability", "server_time"})
SETTLE_RESPONSE_FIELDS = frozenset({
    "v", "hold_id", "receipt_id", "accepted_delta", "cumulative_charge",
    "replayed", "state",
})
USAGE_FIELDS = frozenset({
    "v", "kind", "receipt_id", "hold_id", "claim_response_sha256",
    "payer_keyid", "payer_account_at_authorize", "worker_keyid",
    "worker_account_at_authorize", "billable", "session_id", "model_id",
    "billing_mode", "layer_start", "layer_end", "sequence",
    "cumulative_prompt_tokens", "cumulative_generated_tokens",
    "cumulative_token_layers", "cumulative_cost", "work_completed_at",
    "final", "termination", "output_sha256",
})
RECEIPT_DOMAIN = b"nakshatra-spend-usage-receipt-v2\0"
PAID_CAPABILITIES = (
    "spend_online_claim_v1",
    "paid_direct_greedy_v1",
    "prompt_token_layers_v1",
    "usage_receipt_v2",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
_MAX_TEXT_BYTES = 256


class PaidAdmissionError(RuntimeError):
    """A paid call or its durable accounting was refused."""


class PaidAdmissionUnavailable(PaidAdmissionError):
    """A required ledger/store dependency is unavailable."""


class PaidAdmissionConflict(PaidAdmissionError):
    """A session or step identifier was reused with different bytes."""


class PaidExecutionDeadline(PaidAdmissionError):
    """A paid step started or completed outside its execution window."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def _strict_object(raw: bytes, fields: frozenset[str], label: str) -> dict:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise PaidAdmissionUnavailable(f"duplicate {label} field")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaidAdmissionUnavailable(f"ledger returned invalid {label} JSON") from exc
    if not isinstance(value, dict) or set(value) != fields:
        raise PaidAdmissionUnavailable(f"ledger returned an invalid strict {label} schema")
    return value


def _integer(value: object, name: str, *, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise PaidAdmissionUnavailable(f"ledger {name} is invalid")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PaidAdmissionUnavailable(f"ledger {name} is invalid")
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise PaidAdmissionUnavailable(f"ledger {name} exceeds {_MAX_TEXT_BYTES} bytes")
    return value


class SpendLedgerClient:
    """Strict signed client for Neuron's N3 spend routes."""

    def __init__(self, url: str, spki_sha256: str, private_key: bytes,
                 worker_keyid: str, *, timeout: float = 5.0,
                 opener: Optional[Callable] = None):
        self.url = (url or "").rstrip("/")
        self.spki_sha256 = (spki_sha256 or "").strip().lower()
        self.private_key = private_key
        self.worker_keyid = worker_keyid
        self.timeout = float(timeout)
        self._opener = opener
        if not self.url.startswith("https://"):
            raise PaidAdmissionUnavailable("paid ledger URL must use HTTPS")
        if _HEX64.fullmatch(self.spki_sha256) is None:
            raise PaidAdmissionUnavailable("paid ledger SPKI pin must be 64 lowercase hex")
        if len(private_key) != 32 or _HEX64.fullmatch(worker_keyid) is None:
            raise PaidAdmissionUnavailable("paid worker identity is invalid")

    def _open(self, request):
        if self._opener is not None:
            return self._opener(
                request, timeout=self.timeout,
                expected_spki_hash=self.spki_sha256)
        return nakshatra_auth.open_pillar_url(
            request, timeout=self.timeout,
            expected_spki_hash=self.spki_sha256)

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> bytes:
        raw = b"" if body is None else _canonical(body)
        authorization, _timestamp = nakshatra_auth.build_signed_envelope(
            self.private_key, self.worker_keyid, method, path, raw)
        request = urllib.request.Request(
            self.url + path,
            data=(raw if method == "POST" else None),
            headers={
                "Authorization": authorization,
                **({"Content-Type": "application/json"} if method == "POST" else {}),
            },
            method=method,
        )
        try:
            with self._open(request) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            # Do not reflect response bodies: a proxy must never trick the
            # worker into including submitted token material in diagnostics.
            raise PaidAdmissionUnavailable(
                f"ledger {path} refused request with HTTP {exc.code}") from exc
        except Exception as exc:
            raise PaidAdmissionUnavailable(
                f"ledger {path} dependency unavailable ({type(exc).__name__})") from exc

    def probe(self) -> dict:
        value = _strict_object(
            self._request("GET", "/spend/ready"),
            READY_RESPONSE_FIELDS, "spend readiness response")
        if (type(value["v"]) is not int or value["v"] != 1
                or value["ready"] is not True
                or value["capability"] != "spend_online_claim_v1"):
            raise PaidAdmissionUnavailable("ledger claim capability is not ready")
        _integer(value["server_time"], "server_time")
        return value

    def claim(self, claim_token: str, claim_instance: str) -> dict:
        if not isinstance(claim_token, str) or _TOKEN.fullmatch(claim_token) is None:
            raise PaidAdmissionError("spend claim token is invalid")
        value = _strict_object(
            self._request("POST", "/spend/claim", {
                "claim_token": claim_token,
                "claim_instance": claim_instance,
            }), CLAIM_RESPONSE_FIELDS, "spend claim response")
        self._validate_claim(value, claim_token, claim_instance)
        return value

    def _validate_claim(self, value: dict, claim_token: str,
                        claim_instance: str) -> None:
        if (type(value["v"]) is not int or value["v"] != 1
                or value["claim_instance"] != claim_instance):
            raise PaidAdmissionUnavailable("ledger claim response identity is invalid")
        token_sha256 = hashlib.sha256(claim_token.encode("ascii")).hexdigest()
        if value["token_sha256"] != token_sha256:
            raise PaidAdmissionUnavailable("ledger claim response token digest differs")
        for name in ("payer_keyid", "worker_keyid"):
            if not isinstance(value[name], str) or _HEX64.fullmatch(value[name]) is None:
                raise PaidAdmissionUnavailable(f"ledger {name} is invalid")
        if value["worker_keyid"] != self.worker_keyid:
            raise PaidAdmissionUnavailable("ledger claim is for another worker")
        for name in (
            "hold_id", "payer_account_at_authorize", "worker_account_at_authorize",
            "session_id", "model_id", "billing_mode", "claim_instance",
        ):
            _text(value[name], name)
        if value["billing_mode"] != "greedy-direct-v1":
            raise PaidAdmissionUnavailable("ledger claim billing mode is unsupported")
        if not isinstance(value["billable"], bool):
            raise PaidAdmissionUnavailable("ledger billable is invalid")
        integers = {}
        for name in (
            "layer_start", "layer_end", "price_multiplier", "max_prompt_tokens",
            "max_output_tokens", "max_token_layers", "max_cost",
            "execution_deadline", "settlement_deadline", "server_time",
        ):
            integers[name] = _integer(value[name], name)
        if integers["layer_end"] <= integers["layer_start"]:
            raise PaidAdmissionUnavailable("ledger claim layer span is invalid")
        if integers["settlement_deadline"] <= integers["execution_deadline"]:
            raise PaidAdmissionUnavailable("ledger claim deadlines are invalid")
        if integers["server_time"] >= integers["execution_deadline"]:
            raise PaidAdmissionUnavailable("ledger claim is already outside execution time")
        layers = integers["layer_end"] - integers["layer_start"]
        expected_token_layers = (
            integers["max_prompt_tokens"] + integers["max_output_tokens"]
        ) * layers
        expected_cost = (
            expected_token_layers * integers["price_multiplier"]
            if value["billable"] else 0
        )
        if integers["max_token_layers"] != expected_token_layers:
            raise PaidAdmissionUnavailable("ledger max_token_layers is inconsistent")
        if integers["max_cost"] != expected_cost:
            raise PaidAdmissionUnavailable("ledger max_cost is inconsistent")
        response_without_digest = {
            key: item for key, item in value.items()
            if key != "claim_response_sha256"
        }
        expected = hashlib.sha256(_canonical(response_without_digest)).hexdigest()
        if (not isinstance(value["claim_response_sha256"], str)
                or value["claim_response_sha256"] != expected):
            raise PaidAdmissionUnavailable("ledger claim response digest is invalid")

    def settle(self, receipt: dict, signature: str) -> dict:
        value = _strict_object(
            self._request("POST", "/spend/settle", {
                "receipt": receipt, "signature": signature,
            }), SETTLE_RESPONSE_FIELDS, "spend settle response")
        if (type(value["v"]) is not int or value["v"] != 1
                or value["hold_id"] != receipt["hold_id"]
                or value["receipt_id"] != receipt["receipt_id"]):
            raise PaidAdmissionUnavailable("ledger settle response identity differs")
        for name in ("accepted_delta", "cumulative_charge"):
            _integer(value[name], name)
        if not isinstance(value["replayed"], bool):
            raise PaidAdmissionUnavailable("ledger settle replay flag is invalid")
        _text(value["state"], "state")
        return value


class ChainLedgerGuard:
    """Identity-only ledger stand-in; chain mode must never move credits."""

    def __init__(self, worker_keyid: str):
        self.worker_keyid = worker_keyid

    def probe(self) -> dict:
        return {"ready": True}

    def claim(self, _claim_token: str, _claim_instance: str) -> dict:
        raise PaidAdmissionUnavailable("chain mode forbids legacy spend claims")

    def settle(self, _receipt: dict, _signature: str) -> dict:
        raise PaidAdmissionUnavailable("chain mode forbids legacy spend settlement")


class PaidAdmissionStore:
    """Mode-0600 SQLite claim, meter, step journal, and settlement outbox."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._prepare_path()
        self._db = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        os.chmod(self.path, 0o600)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._schema()
        self._secure_sidecars()
        row = self._db.execute("PRAGMA quick_check").fetchone()
        if row is None or row[0] != "ok":
            raise PaidAdmissionUnavailable("paid admission store recovery check failed")

    def _prepare_path(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        if os.path.lexists(self.path) and self.path.is_symlink():
            raise PaidAdmissionUnavailable("paid admission store must not be a symlink")
        if self.path.exists():
            metadata = self.path.stat()
            if metadata.st_uid != os.geteuid() or not stat.S_ISREG(metadata.st_mode):
                raise PaidAdmissionUnavailable("paid admission store is not uid-owned regular data")

    def _secure_sidecars(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                os.chmod(candidate, 0o600)

    def _schema(self) -> None:
        with self._lock:
            self._db.executescript("""
            CREATE TABLE IF NOT EXISTS claim_attempts(
              token_sha256 TEXT PRIMARY KEY, claim_instance TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS claims(
              payer_keyid TEXT NOT NULL, session_id TEXT NOT NULL,
              hold_id TEXT NOT NULL, token_sha256 TEXT NOT NULL,
              response_json TEXT NOT NULL, claim_response_sha256 TEXT NOT NULL,
              execution_deadline INTEGER NOT NULL,
              settlement_deadline INTEGER NOT NULL,
              prompt_tokens INTEGER NOT NULL DEFAULT 0,
              generated_tokens INTEGER NOT NULL DEFAULT 0,
              receipt_sequence INTEGER NOT NULL DEFAULT 0,
              final INTEGER NOT NULL DEFAULT 0,
              last_checkpoint_at INTEGER NOT NULL DEFAULT 0,
              last_checkpoint_prompt_tokens INTEGER NOT NULL DEFAULT 0,
              last_checkpoint_generated_tokens INTEGER NOT NULL DEFAULT 0,
              last_work_completed_at INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY(payer_keyid, session_id));
            CREATE TABLE IF NOT EXISTS steps(
              payer_keyid TEXT NOT NULL, session_id TEXT NOT NULL,
              step_id TEXT NOT NULL, request_sha256 TEXT NOT NULL,
              kv_epoch INTEGER NOT NULL, committed_at INTEGER NOT NULL,
              PRIMARY KEY(payer_keyid, session_id, step_id));
            CREATE TABLE IF NOT EXISTS outbox(
              receipt_id TEXT PRIMARY KEY, payer_keyid TEXT NOT NULL,
              session_id TEXT NOT NULL, sequence INTEGER NOT NULL,
              receipt_json TEXT NOT NULL, signature TEXT NOT NULL,
              state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
              last_error TEXT NOT NULL DEFAULT '', response_json TEXT,
              UNIQUE(payer_keyid, session_id, sequence));
            CREATE TABLE IF NOT EXISTS kv_state(
              singleton INTEGER PRIMARY KEY CHECK(singleton=1),
              payer_keyid TEXT, session_id TEXT, kv_epoch INTEGER NOT NULL);
            INSERT OR IGNORE INTO kv_state(singleton, kv_epoch) VALUES(1, 0);
            """)
            claim_columns = {
                str(row[1]) for row in self._db.execute("PRAGMA table_info(claims)")
            }
            for name in (
                "last_checkpoint_prompt_tokens",
                "last_checkpoint_generated_tokens",
                "last_work_completed_at",
            ):
                if name not in claim_columns:
                    self._db.execute(
                        f"ALTER TABLE claims ADD COLUMN {name} "
                        "INTEGER NOT NULL DEFAULT 0")
            self._db.execute(
                "UPDATE claims SET last_checkpoint_prompt_tokens=prompt_tokens, "
                "last_checkpoint_generated_tokens=generated_tokens, "
                "last_work_completed_at=last_checkpoint_at "
                "WHERE receipt_sequence>0 AND last_work_completed_at=0")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def claim_instance(self, token_sha256: str) -> str:
        with self._lock:
            row = self._db.execute(
                "SELECT claim_instance FROM claim_attempts WHERE token_sha256=?",
                (token_sha256,)).fetchone()
            if row is not None:
                return str(row[0])
            instance = uuid.uuid4().hex
            self._db.execute(
                "INSERT INTO claim_attempts(token_sha256, claim_instance) VALUES(?,?)",
                (token_sha256, instance))
            self._secure_sidecars()
            return instance

    def save_claim(self, response: Mapping[str, object]) -> dict:
        canonical = _canonical(dict(response)).decode("utf-8")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                existing = self._db.execute(
                    "SELECT response_json FROM claims WHERE payer_keyid=? AND session_id=?",
                    (response["payer_keyid"], response["session_id"])).fetchone()
                if existing is not None and existing[0] != canonical:
                    raise PaidAdmissionConflict("paid session already has another claim snapshot")
                self._db.execute(
                    "INSERT OR IGNORE INTO claims(payer_keyid, session_id, hold_id, "
                    "token_sha256, response_json, claim_response_sha256, "
                    "execution_deadline, settlement_deadline) VALUES(?,?,?,?,?,?,?,?)",
                    (response["payer_keyid"], response["session_id"], response["hold_id"],
                     response["token_sha256"], canonical,
                     response["claim_response_sha256"], response["execution_deadline"],
                     response["settlement_deadline"]),)
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
            self._secure_sidecars()
        return dict(response)

    def get_claim(self, payer_keyid: str, session_id: str) -> Optional[dict]:
        with self._lock:
            row = self._db.execute(
                "SELECT response_json FROM claims WHERE payer_keyid=? AND session_id=?",
                (payer_keyid, session_id)).fetchone()
        return json.loads(row[0]) if row is not None else None

    def inspect_step(self, payer_keyid: str, session_id: str,
                     step_id: str, request_sha256: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT request_sha256 FROM steps WHERE payer_keyid=? AND session_id=? "
                "AND step_id=?", (payer_keyid, session_id, step_id)).fetchone()
        if row is None:
            return False
        if row[0] != request_sha256:
            raise PaidAdmissionConflict("paid step_id was reused with different request bytes")
        return True

    def meter(self, payer_keyid: str, session_id: str) -> sqlite3.Row:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM claims WHERE payer_keyid=? AND session_id=?",
                (payer_keyid, session_id)).fetchone()
        if row is None:
            raise PaidAdmissionError("paid session has no durable claim")
        return row

    def commit_step(self, claim: dict, step_id: str, request_sha256: str,
                    prompt_delta: int, generated_delta: int, kv_epoch: int,
                    completed_at: int, receipt: Optional[dict] = None,
                    signature: str = "") -> None:
        canonical_receipt = (
            _canonical(receipt).decode("utf-8") if receipt is not None else "")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                duplicate = self._db.execute(
                    "SELECT request_sha256 FROM steps WHERE payer_keyid=? AND session_id=? "
                    "AND step_id=?", (claim["payer_keyid"], claim["session_id"],
                                      step_id)).fetchone()
                if duplicate is not None:
                    if duplicate[0] != request_sha256:
                        raise PaidAdmissionConflict(
                            "paid step_id was reused with different request bytes")
                    self._db.execute("ROLLBACK")
                    return
                meter = self._db.execute(
                    "SELECT prompt_tokens,generated_tokens,receipt_sequence,final "
                    "FROM claims WHERE payer_keyid=? AND session_id=?",
                    (claim["payer_keyid"], claim["session_id"])).fetchone()
                if meter is None or meter["final"]:
                    raise PaidAdmissionConflict(
                        "paid session became unavailable before usage commit")
                prompt = int(meter["prompt_tokens"]) + prompt_delta
                generated = int(meter["generated_tokens"]) + generated_delta
                if receipt is not None:
                    if (int(receipt["sequence"])
                            != int(meter["receipt_sequence"]) + 1):
                        raise PaidAdmissionConflict(
                            "paid receipt sequence raced another stream")
                    if (int(receipt["cumulative_prompt_tokens"]) != prompt
                            or int(receipt["cumulative_generated_tokens"])
                            != generated):
                        raise PaidAdmissionConflict(
                            "paid cumulative meter changed before commit")
                self._db.execute(
                    "INSERT INTO steps(payer_keyid,session_id,step_id,request_sha256,"
                    "kv_epoch,committed_at) VALUES(?,?,?,?,?,?)",
                    (claim["payer_keyid"], claim["session_id"], step_id,
                     request_sha256, kv_epoch, completed_at))
                if receipt is None:
                    self._db.execute(
                        "UPDATE claims SET prompt_tokens=?,generated_tokens=?, "
                        "last_work_completed_at=? WHERE payer_keyid=? AND session_id=?",
                        (prompt, generated, completed_at, claim["payer_keyid"],
                         claim["session_id"]))
                else:
                    self._db.execute(
                        "UPDATE claims SET prompt_tokens=?,generated_tokens=?, "
                        "receipt_sequence=?,last_checkpoint_at=?, "
                        "last_checkpoint_prompt_tokens=?, "
                        "last_checkpoint_generated_tokens=?, "
                        "last_work_completed_at=? "
                        "WHERE payer_keyid=? AND session_id=?",
                        (prompt, generated, receipt["sequence"], completed_at,
                         prompt, generated, completed_at, claim["payer_keyid"],
                         claim["session_id"]))
                    self._db.execute(
                        "INSERT INTO outbox(receipt_id,payer_keyid,session_id,sequence,"
                        "receipt_json,signature) VALUES(?,?,?,?,?,?)",
                        (receipt["receipt_id"], claim["payer_keyid"],
                         claim["session_id"], receipt["sequence"],
                         canonical_receipt, signature))
                self._db.execute(
                    "UPDATE kv_state SET payer_keyid=?,session_id=?,kv_epoch=? "
                    "WHERE singleton=1",
                    (claim["payer_keyid"], claim["session_id"], kv_epoch))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
            self._secure_sidecars()

    def commit_final(self, claim: dict, receipt: dict, signature: str) -> bool:
        canonical_receipt = _canonical(receipt).decode("utf-8")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT final,receipt_sequence FROM claims WHERE payer_keyid=? "
                    "AND session_id=?", (claim["payer_keyid"], claim["session_id"])).fetchone()
                if row is None or row[0] or int(row[1]) == 0:
                    self._db.execute("ROLLBACK")
                    return False
                meter = self._db.execute(
                    "SELECT prompt_tokens,generated_tokens FROM claims "
                    "WHERE payer_keyid=? AND session_id=?",
                    (claim["payer_keyid"], claim["session_id"])).fetchone()
                if (int(receipt["sequence"]) != int(row[1]) + 1
                        or int(receipt["cumulative_prompt_tokens"]) != int(meter[0])
                        or int(receipt["cumulative_generated_tokens"])
                        != int(meter[1])):
                    raise PaidAdmissionConflict(
                        "paid final receipt raced cumulative usage")
                self._db.execute(
                    "UPDATE claims SET final=1,receipt_sequence=?, "
                    "last_checkpoint_at=?,last_checkpoint_prompt_tokens=?, "
                    "last_checkpoint_generated_tokens=? WHERE payer_keyid=? "
                    "AND session_id=?",
                    (receipt["sequence"], receipt["work_completed_at"],
                     receipt["cumulative_prompt_tokens"],
                     receipt["cumulative_generated_tokens"],
                     claim["payer_keyid"], claim["session_id"]))
                self._db.execute(
                    "INSERT INTO outbox(receipt_id,payer_keyid,session_id,sequence,"
                    "receipt_json,signature) VALUES(?,?,?,?,?,?)",
                    (receipt["receipt_id"], claim["payer_keyid"], claim["session_id"],
                     receipt["sequence"], canonical_receipt, signature))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
            self._secure_sidecars()
            return True

    def pending(self) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._db.execute(
                "SELECT * FROM outbox WHERE state='pending' ORDER BY rowid"))

    def receipt_signature(self, receipt_id: str) -> str:
        with self._lock:
            row = self._db.execute(
                "SELECT signature FROM outbox WHERE receipt_id=?", (receipt_id,)
            ).fetchone()
        if row is None:
            raise PaidAdmissionConflict("paid receipt is absent from the durable outbox")
        return str(row[0])

    def mark_sent(self, receipt_id: str, response: dict) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE outbox SET state='sent',attempts=attempts+1,last_error='',"
                "response_json=? WHERE receipt_id=? AND state='pending'",
                (_canonical(response).decode("utf-8"), receipt_id))

    def mark_attempt_failed(self, receipt_id: str, reason: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE outbox SET attempts=attempts+1,last_error=? "
                "WHERE receipt_id=? AND state='pending'",
                (reason[:128], receipt_id))

    def mark_chain_vouched(self, receipt_id: str, voucher_total: int) -> None:
        """Mark a local meter receipt covered without calling the legacy ledger."""
        response = _canonical({
            "kind": "chain-vouched", "voucher_total": int(voucher_total),
        }).decode("utf-8")
        with self._lock:
            self._db.execute(
                "UPDATE outbox SET state='chain-vouched',attempts=attempts+1,"
                "last_error='',response_json=? WHERE receipt_id=? AND state='pending'",
                (response, receipt_id))


class PaidAdmission:
    """Online claim, cap enforcement, metering, receipt signing, and outbox."""

    def __init__(self, ledger: SpendLedgerClient, store: PaidAdmissionStore,
                 private_key: bytes, *, model_id: str, layer_start: int,
                 layer_end: int, clock: Callable[[], float] = time.time,
                 voucher_store=None, channel_state=None,
                 pay_program: Optional[str] = None):
        self.ledger = ledger
        self.store = store
        self.private_key = private_key
        self.worker_keyid = ledger.worker_keyid
        self.model_id = model_id
        self.layer_start = int(layer_start)
        self.layer_end = int(layer_end)
        self.clock = clock
        self.ready = False
        self.readiness_error = "not probed"
        self._outbox_lock = threading.Lock()
        self._session_locks_lock = threading.Lock()
        self._session_locks: dict[tuple[str, str], threading.RLock] = {}
        self.chain_mode = os.environ.get("NAK_PAY_MODE", "") == "chain"
        self._chain_sessions: dict[tuple[str, str], dict] = {}
        self._pay = None
        self.voucher_store = voucher_store
        self.channel_state = channel_state
        self.pay_program = pay_program
        self.pay_wallet_address = None
        if self.chain_mode:
            try:
                from network import pay_voucher
                from network.pay_wallet import load_wallet
                self._pay = pay_voucher
                self.pay_program = pay_program or pay_voucher.configured_program()
                self.channel_state = channel_state or pay_voucher.configured_state()
                self.voucher_store = voucher_store or pay_voucher.WorkerVoucherStore()
                self.pay_wallet_address = load_wallet(Path.home())["address"]
            except SystemExit as exc:
                raise PaidAdmissionUnavailable(
                    f"chain pay wallet is unavailable ({exc})") from exc
            except Exception as exc:
                raise PaidAdmissionUnavailable(
                    f"chain voucher configuration is unavailable ({exc})") from exc

    def session_lock(self, claim: Mapping[str, object]) -> threading.RLock:
        """Return the in-process lock serializing one durable paid session."""
        key = (str(claim["payer_keyid"]), str(claim["session_id"]))
        with self._session_locks_lock:
            return self._session_locks.setdefault(key, threading.RLock())

    def check_ready(self) -> bool:
        try:
            if not self.chain_mode:
                self.ledger.probe()
            self.ready = True
            self.readiness_error = ""
        except Exception as exc:
            self.ready = False
            self.readiness_error = str(exc)
        return self.ready

    def claim(self, payer_keyid: str, session_id: str,
              claim_token: str = "") -> dict:
        if not self.ready:
            raise PaidAdmissionUnavailable("paid admission readiness is false")
        chain_init = None
        raw_claim_token = claim_token
        if self.chain_mode and claim_token:
            try:
                chain_init = self._pay.parse_wire(claim_token)
                if (chain_init.get("kind") != "init"
                        or set(chain_init) != {
                            "v", "kind", "claim_token", "program", "channel",
                            "channel_base", "hold_id", "session_id", "model_id",
                            "max_prompt_tokens", "max_output_tokens",
                        }):
                    raise self._pay.VoucherError("invalid chain init envelope")
                if chain_init["program"] != self.pay_program:
                    raise self._pay.VoucherError(
                        "channel program differs from configured program")
                raw_claim_token = chain_init["claim_token"]
            except Exception as exc:
                raise PaidAdmissionError(f"chain voucher init refused ({exc})") from exc
        existing = self.store.get_claim(payer_keyid, session_id)
        if existing is not None:
            if raw_claim_token:
                digest = hashlib.sha256(raw_claim_token.encode("ascii")).hexdigest()
                if digest != existing["token_sha256"]:
                    raise PaidAdmissionConflict("paid session token differs from durable claim")
            self._validate_scope(existing, payer_keyid, session_id)
            if self.chain_mode and (payer_keyid, session_id) not in self._chain_sessions:
                if chain_init is None:
                    raise PaidAdmissionError("first chain frame requires channel init")
                self._admit_chain(existing, chain_init)
            return existing
        if not raw_claim_token:
            raise PaidAdmissionError("first paid frame requires spend_claim_token")
        if self.chain_mode and chain_init is None:
            raise PaidAdmissionError("first chain frame requires channel init")
        if _TOKEN.fullmatch(raw_claim_token) is None:
            raise PaidAdmissionError("spend claim token is invalid")
        if self.chain_mode:
            response = self._chain_claim(
                payer_keyid, session_id, raw_claim_token, chain_init)
        else:
            token_digest = hashlib.sha256(raw_claim_token.encode("ascii")).hexdigest()
            instance = self.store.claim_instance(token_digest)
            response = self.ledger.claim(raw_claim_token, instance)
        self._validate_scope(response, payer_keyid, session_id)
        saved = self.store.save_claim(response)
        if self.chain_mode:
            self._admit_chain(saved, chain_init)
        return saved

    def _chain_claim(self, payer_keyid: str, session_id: str,
                     claim_token: str, init: Mapping[str, object]) -> dict:
        """Build the durable local meter scope without reserving ledger credits."""
        if init.get("session_id") != session_id or init.get("model_id") != self.model_id:
            raise PaidAdmissionError("chain voucher init scope differs from this call")
        hold_id = init.get("hold_id")
        if (not isinstance(hold_id, str) or not hold_id
                or len(hold_id.encode("utf-8")) > _MAX_TEXT_BYTES):
            raise PaidAdmissionError("chain voucher hold id is invalid")
        prompt = self._pay._u64(init.get("max_prompt_tokens"), "maximum prompt tokens")
        output = self._pay._u64(init.get("max_output_tokens"), "maximum output tokens")
        if prompt + output == 0:
            raise PaidAdmissionError("chain voucher token caps are empty")
        layers = self.layer_end - self.layer_start
        max_cost = (prompt + output) * layers
        token_digest = hashlib.sha256(claim_token.encode("ascii")).hexdigest()
        response = {
            "v": 1, "hold_id": hold_id, "token_sha256": token_digest,
            "claim_instance": "chain-" + token_digest,
            "payer_keyid": payer_keyid,
            "payer_account_at_authorize": "chain:" + payer_keyid,
            "worker_keyid": self.worker_keyid,
            "worker_account_at_authorize": self.pay_wallet_address,
            "billable": True, "session_id": session_id, "model_id": self.model_id,
            "billing_mode": "greedy-direct-v1", "layer_start": self.layer_start,
            "layer_end": self.layer_end, "price_multiplier": 1,
            "max_prompt_tokens": prompt, "max_output_tokens": output,
            "max_token_layers": max_cost, "max_cost": max_cost,
            "execution_deadline": (1 << 63) - 2,
            "settlement_deadline": (1 << 63) - 1,
            "server_time": int(self.clock()), "claim_response_sha256": "",
        }
        response["claim_response_sha256"] = hashlib.sha256(
            b"nakshatra-chain-meter-v1\0" + _canonical(response)).hexdigest()
        return response

    def _admit_chain(self, claim: dict, init: dict) -> None:
        reservation_id = (
            f"{claim['payer_keyid']}:{claim['session_id']}")
        reserved = False
        try:
            # Acquire before deriving the base. This serializes the latest
            # stored voucher with admission across threads and processes.
            self.voucher_store.acquire_reservation(
                init["channel"], reservation_id)
            reserved = True
            account = self.channel_state.read_channel(
                init["channel"], self.pay_program)
            self._pay.require_open_channel(account, now=self.clock())
            if account.worker != self.pay_wallet_address:
                raise self._pay.VoucherError(
                    "channel worker does not match this worker pay wallet")
            latest = self.voucher_store.latest(init["channel"])
            latest_total = latest.total if latest is not None else account.redeemed
            derived_base = max(account.redeemed, latest_total)
            base = self._pay._u64(init["channel_base"], "channel base")
            if base != derived_base:
                raise self._pay.VoucherError(
                    "sender channel base does not match worker-derived base")
            if base > account.cap:
                raise self._pay.VoucherError("channel base exceeds channel cap")
            if (base + self._pay.required_capacity_micro_usdc(
                    int(claim["max_cost"])) > account.cap):
                raise self._pay.VoucherError(
                    "channel has insufficient uncommitted capacity")
            self._chain_sessions[(claim["payer_keyid"], claim["session_id"])] = {
                "channel": init["channel"], "channel_base": base,
                "pending_total": None, "channel_closing": False,
                "reservation_id": reservation_id,
            }
        except Exception as exc:
            if reserved:
                self.voucher_store.release_reservation(
                    init["channel"], reservation_id)
            raise PaidAdmissionError(f"chain channel admission refused ({exc})") from exc

    def _chain_session(self, claim: Mapping[str, object]) -> dict:
        try:
            return self._chain_sessions[(str(claim["payer_keyid"]),
                                         str(claim["session_id"]))]
        except KeyError as exc:
            raise PaidAdmissionError("chain session has no admitted channel") from exc

    def release_chain_session(self, claim: Mapping[str, object]) -> None:
        """Release this process's durable channel reservation on every exit."""
        if not self.chain_mode:
            return
        key = (str(claim["payer_keyid"]), str(claim["session_id"]))
        session = self._chain_sessions.get(key)
        if session is None:
            return
        try:
            self.voucher_store.release_reservation(
                session["channel"], session["reservation_id"])
        except Exception as exc:
            raise PaidAdmissionError(
                f"chain channel reservation release failed ({exc})") from exc
        self._chain_sessions.pop(key, None)

    def accept_chain_control(self, claim: dict, value: str) -> None:
        """Verify and durably store a voucher before this method returns."""
        if not self.chain_mode:
            return
        try:
            control = self._pay.parse_wire(value)
            if control.get("kind") != "voucher":
                raise self._pay.VoucherError("expected a voucher control envelope")
            session = self._chain_session(claim)
            record = self.voucher_store.accept(
                control, configured_program=self.pay_program,
                state=self.channel_state, expected_channel=session["channel"],
                now=self.clock(), reservation_id=session["reservation_id"])
            pending = session["pending_total"]
            if pending is not None and record.total >= pending:
                session["pending_total"] = None
            for row in self.store.pending():
                if (row["payer_keyid"] != claim["payer_keyid"]
                        or row["session_id"] != claim["session_id"]):
                    continue
                receipt = json.loads(row["receipt_json"])
                covered = self._pay.credits_to_total(
                    session["channel_base"], int(receipt["cumulative_cost"]))
                if record.total >= covered:
                    self.store.mark_chain_vouched(row["receipt_id"], record.total)
        except self._pay.ChannelClosingError as exc:
            session = self._chain_session(claim)
            session["channel_closing"] = True
            raise PaidAdmissionError("channel-closing") from exc
        except Exception as exc:
            raise PaidAdmissionError(f"chain voucher refused ({exc})") from exc

    def require_chain_voucher_before_step(self, claim: dict,
                                          prompt_delta: int,
                                          generated_delta: int) -> None:
        if not self.chain_mode:
            return
        session = self._chain_session(claim)
        try:
            self.voucher_store.assert_reservation(
                session["channel"], session["reservation_id"])
        except Exception as exc:
            raise PaidAdmissionError(
                f"chain channel reservation lost ({exc})") from exc
        if session["channel_closing"]:
            raise PaidAdmissionError("channel-closing")
        row = self.store.meter(claim["payer_keyid"], claim["session_id"])
        latest = self.voucher_store.latest(session["channel"])
        accepted_total = (
            latest.total if latest is not None else session["channel_base"])
        covered_credits = max(
            0, accepted_total - session["channel_base"]) * self._pay.RATE
        layers = int(claim["layer_end"]) - int(claim["layer_start"])
        multiplier = int(claim["price_multiplier"]) if claim["billable"] else 0
        delivered_credits = (
            int(row["prompt_tokens"]) + int(row["generated_tokens"])
            + int(prompt_delta) + int(generated_delta)
        ) * layers * multiplier
        checkpoint_credits = 8 * layers * multiplier
        if (delivered_credits - covered_credits
                > checkpoint_credits + (self._pay.RATE - 1)):
            raise PaidAdmissionError("voucher-missing")

    def chain_receipt_wire(self, claim: dict, receipt: Optional[dict]) -> str:
        if not self.chain_mode or receipt is None:
            return ""
        session = self._chain_session(claim)
        try:
            self.voucher_store.assert_reservation(
                session["channel"], session["reservation_id"])
        except Exception as exc:
            raise PaidAdmissionError(
                f"chain channel reservation lost ({exc})") from exc
        total = self._pay.credits_to_total(
            session["channel_base"], int(receipt["cumulative_cost"]))
        session["pending_total"] = total
        signature = self.store.receipt_signature(receipt["receipt_id"])
        wire = self._pay.receipt_wire(receipt, signature)
        try:
            account = self.channel_state.read_channel(
                session["channel"], self.pay_program)
            self._pay.require_open_channel(account, now=self.clock())
        except self._pay.ChannelClosingError:
            # The completed checkpoint and its receipt still reach the payer;
            # no subsequent checkpoint work may begin.
            session["channel_closing"] = True
        except Exception as exc:
            raise PaidAdmissionError(
                f"chain channel checkpoint refused ({exc})") from exc
        return wire

    def final_chain_receipt_wire(self, claim: dict,
                                 termination: str = "normal") -> str:
        if not self.chain_mode:
            return ""
        return self.chain_receipt_wire(claim, self.finalize(claim, termination))

    def require_chain_final_voucher(self, claim: dict) -> None:
        if not self.chain_mode:
            return
        session = self._chain_session(claim)
        pending = session["pending_total"]
        latest = self.voucher_store.latest(session["channel"])
        if pending is not None and (latest is None or latest.total < pending):
            raise PaidAdmissionError("voucher-missing")

    def _validate_scope(self, claim: dict, payer_keyid: str,
                        session_id: str) -> None:
        expected = {
            "payer_keyid": payer_keyid,
            "worker_keyid": self.worker_keyid,
            "session_id": session_id,
            "model_id": self.model_id,
            "billing_mode": "greedy-direct-v1",
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
        }
        for name, value in expected.items():
            if claim.get(name) != value:
                raise PaidAdmissionError(f"spend claim {name} does not match this call")
        if int(self.clock()) >= int(claim["execution_deadline"]):
            raise PaidExecutionDeadline("spend execution deadline has passed")

    @staticmethod
    def request_digest(message) -> str:
        clone = type(message)()
        clone.CopyFrom(message)
        if hasattr(clone, "spend_claim_token"):
            clone.spend_claim_token = ""
        return hashlib.sha256(clone.SerializeToString()).hexdigest()

    def prepare_step(self, claim: dict, step_id: str, request_sha256: str,
                     n_tokens: int) -> tuple[bool, int, int]:
        if not step_id or len(step_id.encode("utf-8")) > _MAX_TEXT_BYTES:
            raise PaidAdmissionError("paid step_id is invalid")
        if int(self.clock()) >= int(claim["execution_deadline"]):
            raise PaidExecutionDeadline("spend execution deadline has passed")
        if self.store.inspect_step(
                claim["payer_keyid"], claim["session_id"], step_id,
                request_sha256):
            return True, 0, 0
        row = self.store.meter(claim["payer_keyid"], claim["session_id"])
        if row["final"]:
            raise PaidAdmissionError("paid session is already final")
        if int(row["prompt_tokens"]) == 0:
            prompt_delta, generated_delta = n_tokens, 1
        else:
            if n_tokens != 1:
                raise PaidAdmissionError("paid post-prefill steps must contain one token")
            prompt_delta, generated_delta = 0, 1
        if int(row["prompt_tokens"]) + prompt_delta > int(claim["max_prompt_tokens"]):
            raise PaidAdmissionError("paid prompt token cap exceeded")
        if (int(row["generated_tokens"]) + generated_delta
                > int(claim["max_output_tokens"])):
            raise PaidAdmissionError("paid output token cap exceeded")
        return False, prompt_delta, generated_delta

    def _receipt(self, claim: dict, sequence: int, prompt: int, generated: int,
                 *, completed_at: int, final: bool,
                 termination: str) -> tuple[dict, str]:
        layers = int(claim["layer_end"]) - int(claim["layer_start"])
        token_layers = (prompt + generated) * layers
        cost = (token_layers * int(claim["price_multiplier"])
                if claim["billable"] else 0)
        receipt = {
            "v": 2,
            "kind": "nakshatra-worker-usage",
            "receipt_id": "",
            "hold_id": claim["hold_id"],
            "claim_response_sha256": claim["claim_response_sha256"],
            "payer_keyid": claim["payer_keyid"],
            "payer_account_at_authorize": claim["payer_account_at_authorize"],
            "worker_keyid": claim["worker_keyid"],
            "worker_account_at_authorize": claim["worker_account_at_authorize"],
            "billable": claim["billable"],
            "session_id": claim["session_id"],
            "model_id": claim["model_id"],
            "billing_mode": claim["billing_mode"],
            "layer_start": claim["layer_start"],
            "layer_end": claim["layer_end"],
            "sequence": sequence,
            "cumulative_prompt_tokens": prompt,
            "cumulative_generated_tokens": generated,
            "cumulative_token_layers": token_layers,
            "cumulative_cost": cost,
            "work_completed_at": int(completed_at),
            "final": bool(final),
            "termination": termination,
            "output_sha256": "",
        }
        without_id = {key: value for key, value in receipt.items()
                      if key != "receipt_id"}
        receipt["receipt_id"] = "sha256:" + hashlib.sha256(
            RECEIPT_DOMAIN + _canonical(without_id)).hexdigest()
        if set(receipt) != USAGE_FIELDS:
            raise AssertionError("worker usage receipt schema drift")
        signature = ed25519.Ed25519PrivateKey.from_private_bytes(
            self.private_key).sign(_canonical(receipt)).hex()
        return receipt, signature

    def commit_step(self, claim: dict, step_id: str, request_sha256: str,
                    prompt_delta: int, generated_delta: int,
                    kv_epoch: int) -> Optional[dict]:
        completed_now = self.clock()
        if completed_now > int(claim["execution_deadline"]):
            raise PaidExecutionDeadline(
                "paid work completed after the execution deadline")
        completed_at = int(completed_now)
        row = self.store.meter(claim["payer_keyid"], claim["session_id"])
        prompt = int(row["prompt_tokens"]) + prompt_delta
        generated = int(row["generated_tokens"]) + generated_delta
        checkpoint_due = (
            prompt_delta > 0
            or generated - int(row["last_checkpoint_generated_tokens"]) >= 8
            or completed_at - int(row["last_checkpoint_at"]) >= 15
        )
        receipt = None
        signature = ""
        if checkpoint_due:
            receipt, signature = self._receipt(
                claim, int(row["receipt_sequence"]) + 1, prompt, generated,
                completed_at=completed_at, final=False,
                termination="checkpoint")
        self.store.commit_step(
            claim, step_id, request_sha256, prompt_delta, generated_delta,
            kv_epoch, completed_at, receipt, signature)
        return receipt

    def finalize(self, claim: dict, termination: str) -> Optional[dict]:
        row = self.store.meter(claim["payer_keyid"], claim["session_id"])
        if row["final"] or int(row["receipt_sequence"]) == 0:
            return None
        has_uncheckpointed_usage = (
            int(row["prompt_tokens"])
            != int(row["last_checkpoint_prompt_tokens"])
            or int(row["generated_tokens"])
            != int(row["last_checkpoint_generated_tokens"])
        )
        completed_at = int(
            row["last_work_completed_at"] if has_uncheckpointed_usage
            else row["last_checkpoint_at"])
        receipt, signature = self._receipt(
            claim, int(row["receipt_sequence"]) + 1,
            int(row["prompt_tokens"]), int(row["generated_tokens"]),
            completed_at=completed_at, final=True, termination=termination)
        return receipt if self.store.commit_final(claim, receipt, signature) else None

    def flush_outbox(self) -> int:
        """Replay every pending receipt; Neuron makes identical retries idempotent."""
        if self.chain_mode:
            return 0
        sent = 0
        with self._outbox_lock:
            for row in self.store.pending():
                receipt = json.loads(row["receipt_json"])
                try:
                    response = self.ledger.settle(receipt, row["signature"])
                except PaidAdmissionError as exc:
                    self.store.mark_attempt_failed(
                        row["receipt_id"], type(exc).__name__)
                    break
                self.store.mark_sent(row["receipt_id"], response)
                sent += 1
        return sent

    def outbox_loop(self, stop_event: threading.Event,
                    interval_seconds: float = 15.0) -> None:
        """Retry durable settlements until shutdown or the ledger accepts them."""
        while not stop_event.wait(max(0.1, float(interval_seconds))):
            self.flush_outbox()
