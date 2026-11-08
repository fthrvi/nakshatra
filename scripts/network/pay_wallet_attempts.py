"""Durable approval and transaction attempts for broker-held pay wallets.

An approval key is committed before the broker is contacted.  Once signed, the
complete transaction identity is committed before ``sendTransaction``.  A
replacement approval is fenced until every older signature is finalized or is
proved expired by the same status/height/status rule used by the redeemer and
starter paths.

The step-4 custody importer must hold
``pay_wallet_custody.custody_lock()`` before changing custody or wallet
identity.  Wallet acts hold it across journal writes and broadcast.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from network import pay_chain, pay_voucher
from network.pay_autochannel import _OpenLock


_OPERATIONS = {"ensure", "starter-grant", "withdraw-request", "withdraw-execute"}
_ERROR_CODES = {"", "finalized-error", "send-refused", "rpc-unavailable", "status-unknown"}


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"{name} must be non-empty text")
    return value


def instruction_digest(instructions: Iterable[pay_chain.Instruction]) -> str:
    """Hash an instruction plan without its replaceable recent blockhash."""
    digest = hashlib.sha256()
    for instruction in instructions:
        digest.update(pay_chain.pubkey_bytes(instruction.program_id, "program id"))
        digest.update(len(instruction.accounts).to_bytes(2, "little"))
        for account in instruction.accounts:
            digest.update(pay_chain.pubkey_bytes(account.pubkey, "instruction account"))
            digest.update(bytes((account.is_signer, account.is_writable)))
        digest.update(len(instruction.data).to_bytes(4, "little"))
        digest.update(instruction.data)
    return digest.hexdigest()


def operation_scope(operation: str, escrow: str, generation: int, *, act_id: str = "") -> str:
    if operation not in _OPERATIONS:
        raise ValueError("invalid broker wallet operation")
    escrow_key = pay_chain.pubkey_text(escrow, "escrow")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("escrow generation must be a non-negative integer")
    suffix = f":{_text(act_id, 'act id')}" if act_id else ""
    return f"{operation}:{escrow_key}:{generation}{suffix}"


def logical_key(scope: str, intent_sha256: str) -> str:
    _text(scope, "operation scope")
    if (not isinstance(intent_sha256, str) or len(intent_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in intent_sha256)):
        raise ValueError("intent digest must be lowercase SHA-256")
    return f"{scope}:{intent_sha256}"


@dataclass(frozen=True)
class PreparedApproval:
    id: int
    logical_key: str
    idempotency_key: str
    message: bytes
    message_sha256: str
    blockhash: str
    last_valid_block_height: int


@dataclass(frozen=True)
class ReconcileResult:
    unresolved: bool
    succeeded: tuple[dict, ...]


class AttemptJournal:
    """Mode-0600 SQLite journal shared by CLI and daemon wallet sends."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or Path.home() / ".nakshatra" / "pay" / "wallet-attempts.sqlite")
        pay_voucher._secure_db_path(self.path)
        self._db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        os.chmod(self.path, 0o600)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS attempts("
            "id INTEGER PRIMARY KEY, operation_scope TEXT NOT NULL, logical_key TEXT NOT NULL, "
            "operation TEXT NOT NULL, wallet_id TEXT NOT NULL, custody TEXT NOT NULL, "
            "idempotency_key TEXT NOT NULL UNIQUE, message BLOB NOT NULL, "
            "message_sha256 TEXT NOT NULL, blockhash TEXT NOT NULL, "
            "last_valid_block_height INTEGER NOT NULL, signature TEXT UNIQUE, "
            "escrow TEXT NOT NULL, generation INTEGER NOT NULL, "
            "withdrawal_message_digest TEXT NOT NULL DEFAULT '', "
            "withdrawal_act_id TEXT NOT NULL DEFAULT '', "
            "withdrawal_amount INTEGER, withdrawal_at INTEGER, "
            "withdrawal_consumed_by TEXT NOT NULL DEFAULT '', "
            "resolution TEXT, error TEXT NOT NULL DEFAULT '', created INTEGER NOT NULL DEFAULT 0)"
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(attempts)")}
        for name, declaration in (
                ("withdrawal_act_id", "TEXT NOT NULL DEFAULT ''"),
                ("withdrawal_amount", "INTEGER"),
                ("withdrawal_at", "INTEGER"),
                ("withdrawal_consumed_by", "TEXT NOT NULL DEFAULT ''")):
            if name not in columns:
                self._db.execute(f"ALTER TABLE attempts ADD COLUMN {name} {declaration}")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS withdrawal_cycles("
            "act_id TEXT PRIMARY KEY, escrow TEXT NOT NULL, generation INTEGER NOT NULL, "
            "amount INTEGER NOT NULL, request_attempt_id INTEGER, "
            "state TEXT NOT NULL, execute_logical_key TEXT NOT NULL DEFAULT '', "
            "created INTEGER NOT NULL DEFAULT 0)"
        )
        self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS withdrawal_one_active "
            "ON withdrawal_cycles(escrow) WHERE state!='consumed'")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS attempts_scope ON attempts(operation_scope,id)")
        self._secure()

    def _secure(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                os.chmod(candidate, 0o600)

    def prepare(self, *, scope: str, logical: str, operation: str, wallet_id: str,
                custody: str, message: bytes, blockhash: str,
                last_valid_block_height: int, escrow: str, generation: int,
                created: int = 0, withdrawal_act_id: str = "",
                withdrawal_amount: int | None = None) -> PreparedApproval:
        """Commit a stable key and exact broker content before first enqueue."""
        if operation not in _OPERATIONS:
            raise ValueError("invalid broker wallet operation")
        if custody != "broker":
            raise ValueError("broker wallet journal requires broker custody")
        _text(scope, "operation scope")
        _text(logical, "logical key")
        wallet = pay_chain.pubkey_text(wallet_id, "wallet id")
        escrow_key = pay_chain.pubkey_text(escrow, "escrow")
        raw = bytes(message)
        if not raw:
            raise ValueError("compiled message must not be empty")
        digest = hashlib.sha256(raw).hexdigest()
        pay_chain.pubkey_bytes(blockhash, "recent blockhash")
        if (isinstance(last_valid_block_height, bool)
                or not isinstance(last_valid_block_height, int)
                or last_valid_block_height < 0):
            raise ValueError("last valid block height must be non-negative")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("escrow generation must be non-negative")
        if operation.startswith("withdraw-"):
            _text(withdrawal_act_id, "withdrawal act id")
        if operation == "withdraw-request":
            if (isinstance(withdrawal_amount, bool) or not isinstance(withdrawal_amount, int)
                    or not 0 < withdrawal_amount < 2**64):
                raise ValueError("withdrawal amount must be a positive u64")
        existing = self._db.execute(
            "SELECT * FROM attempts WHERE logical_key=? AND resolution IS NULL "
            "ORDER BY id DESC LIMIT 1", (logical,)).fetchone()
        if existing is not None:
            if existing["signature"] is not None:
                raise pay_voucher.VoucherError("previous broker wallet transaction still unresolved")
            self._secure()
            return self._prepared(existing)
        idempotency_key = f"wallet:{operation}:{secrets.token_hex(16)}"
        withdrawal_digest = digest if operation.startswith("withdraw-") else ""
        cursor = self._db.execute(
            "INSERT INTO attempts(operation_scope,logical_key,operation,wallet_id,custody,"
            "idempotency_key,message,message_sha256,blockhash,last_valid_block_height,"
            "escrow,generation,withdrawal_message_digest,withdrawal_act_id,"
            "withdrawal_amount,created) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (scope, logical, operation, wallet, custody, idempotency_key,
             sqlite3.Binary(raw), digest, str(blockhash), int(last_valid_block_height),
             escrow_key, int(generation), withdrawal_digest, withdrawal_act_id,
             withdrawal_amount, int(created)))
        row = self._db.execute(
            "SELECT * FROM attempts WHERE id=?", (cursor.lastrowid,)).fetchone()
        if operation == "withdraw-request":
            cycle = self._db.execute(
                "SELECT * FROM withdrawal_cycles WHERE act_id=?", (withdrawal_act_id,)).fetchone()
            if (cycle is None or cycle["escrow"] != escrow_key
                    or int(cycle["generation"]) != int(generation)
                    or int(cycle["amount"]) != withdrawal_amount):
                self._db.execute("DELETE FROM attempts WHERE id=?", (cursor.lastrowid,))
                raise pay_voucher.VoucherError("withdrawal act does not match its request")
            self._db.execute(
                "UPDATE withdrawal_cycles SET request_attempt_id=COALESCE(request_attempt_id,?) "
                "WHERE act_id=?", (cursor.lastrowid, withdrawal_act_id))
        self._secure()
        return self._prepared(row)

    @staticmethod
    def _prepared(row: sqlite3.Row) -> PreparedApproval:
        return PreparedApproval(
            id=int(row["id"]), logical_key=row["logical_key"],
            idempotency_key=row["idempotency_key"], message=bytes(row["message"]),
            message_sha256=row["message_sha256"], blockhash=row["blockhash"],
            last_valid_block_height=int(row["last_valid_block_height"]))

    def record_signed(self, attempt_id: int, signature: str) -> None:
        """Commit the fourth transaction field before broadcast."""
        try:
            raw_signature = pay_chain.b58decode(signature)
        except (TypeError, ValueError) as exc:
            raise ValueError("transaction signature is not valid base58") from exc
        if len(raw_signature) != 64 or pay_chain.b58encode(raw_signature) != signature:
            raise ValueError("transaction signature must be canonical base58 for 64 bytes")
        cursor = self._db.execute(
            "UPDATE attempts SET signature=? WHERE id=? AND signature IS NULL "
            "AND resolution IS NULL", (str(signature), int(attempt_id)))
        if cursor.rowcount != 1:
            row = self._db.execute(
                "SELECT signature FROM attempts WHERE id=?", (int(attempt_id),)).fetchone()
            if row is None or row["signature"] != signature:
                raise pay_voucher.VoucherError("broker wallet attempt could not be recorded")
        self._secure()

    def resolve_signature(self, signature: str, resolution: str, error: str = "") -> None:
        if resolution not in {"succeeded", "failed", "expired"}:
            raise ValueError("invalid broker wallet resolution")
        if error not in _ERROR_CODES:
            raise ValueError("broker wallet error must be a closed code")
        self._db.execute(
            "UPDATE attempts SET resolution=?,error=? WHERE signature=? AND resolution IS NULL",
            (resolution, error, str(signature)))
        self._secure()

    def _rows(self, *, scope: str | None = None, operation: str | None = None,
              escrow: str | None = None, generation: int | None = None) -> list[dict]:
        clauses, values = [], []
        for column, value in (("operation_scope", scope), ("operation", operation),
                              ("escrow", escrow), ("generation", generation)):
            if value is not None:
                clauses.append(f"{column}=?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return [dict(row) for row in self._db.execute(
            "SELECT * FROM attempts" + where + " ORDER BY id", values)]

    def attempts(self, *, scope: str | None = None) -> list[dict]:
        """Return journal rows for diagnostics and deterministic tests."""
        return self._rows(scope=scope)

    def reconcile(self, scope: str, rpc, *, resumable_logical_key: str = "") -> ReconcileResult:
        """Resolve every attempt in a scope using the two-read expiry fence."""
        height = None
        unresolved = False
        for attempt in self._rows(scope=scope):
            if attempt["resolution"] is not None:
                continue
            signature = attempt["signature"]
            if signature is None:
                if height is None:
                    height = rpc.block_height(commitment="finalized")
                if height > int(attempt["last_valid_block_height"]):
                    self._db.execute(
                        "UPDATE attempts SET resolution='expired' WHERE id=? AND resolution IS NULL",
                        (attempt["id"],))
                    continue
                if attempt["logical_key"] == resumable_logical_key:
                    continue
                unresolved = True
                continue
            status = pay_chain.validate_signature_status(rpc.signature_status(signature))
            if status is None:
                if height is None:
                    height = rpc.block_height(commitment="finalized")
                if height > int(attempt["last_valid_block_height"]):
                    status = pay_chain.validate_signature_status(rpc.signature_status(signature))
                    if status is None:
                        self.resolve_signature(signature, "expired")
                        continue
            if status is not None and status["confirmationStatus"] == "finalized":
                if status["err"] is None:
                    self.resolve_signature(signature, "succeeded")
                else:
                    self.resolve_signature(signature, "failed", "finalized-error")
                continue
            unresolved = True
        self._secure()
        succeeded = tuple(row for row in self._rows(scope=scope)
                          if row["resolution"] == "succeeded")
        return ReconcileResult(unresolved, succeeded)

    def begin_withdraw_request(self, rpc, escrow: str, generation: int,
                               amount: int, *, created: int = 0) -> str:
        """Create or resume the one unfinished withdrawal act for an escrow."""
        escrow_key = pay_chain.pubkey_text(escrow, "escrow")
        if (isinstance(amount, bool) or not isinstance(amount, int) or not 0 < amount < 2**64):
            raise ValueError("withdrawal amount must be a positive u64")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            active = self._db.execute(
                "SELECT * FROM withdrawal_cycles WHERE escrow=? AND state!='consumed'",
                (escrow_key,)).fetchone()
            if active is not None:
                if active["state"] == "requesting" and int(active["amount"]) == amount:
                    self._db.execute("COMMIT")
                    return str(active["act_id"])
                raise pay_voucher.VoucherError(
                    "another withdrawal cycle for this escrow is still active")

            info = rpc.account_info(escrow_key, commitment="finalized")
            if info is None:
                raise pay_voucher.VoucherError("wallet escrow is missing")
            state = pay_chain.decode_escrow(info["data"])
            if int(state["generation"]) != int(generation):
                raise pay_voucher.VoucherError("escrow generation changed during withdrawal")
            if state["pending_withdraw"] is not None:
                raise pay_voucher.VoucherError(
                    "escrow already has a pending withdrawal without an active broker act")
            act_id = secrets.token_hex(16)
            self._db.execute(
                "INSERT INTO withdrawal_cycles(act_id,escrow,generation,amount,state,created) "
                "VALUES(?,?,?,?,?,?)",
                (act_id, escrow_key, int(generation), amount, "requesting", int(created)))
            self._db.execute("COMMIT")
            self._secure()
            return act_id
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def bind_finalized_withdraw_request(self, rpc, attempt_id: int) -> dict:
        """Bind a finalized request row to the exact finalized pending tuple."""
        row = self._db.execute(
            "SELECT * FROM attempts WHERE id=?", (int(attempt_id),)).fetchone()
        if (row is None or row["operation"] != "withdraw-request"
                or row["resolution"] != "succeeded" or row["signature"] is None
                or not row["withdrawal_act_id"]
                or row["withdrawal_message_digest"] != row["message_sha256"]):
            raise pay_voucher.VoucherError(
                "withdraw request is not reconciled finalized provenance")
        info = rpc.account_info(row["escrow"], commitment="finalized")
        if info is None:
            raise pay_voucher.VoucherError("wallet escrow disappeared after withdraw request")
        state = pay_chain.decode_escrow(info["data"])
        pending = state["pending_withdraw"]
        if (int(state["generation"]) != int(row["generation"]) or pending is None
                or int(pending["amount"]) != int(row["withdrawal_amount"])):
            raise pay_voucher.VoucherError(
                "finalized pending withdrawal does not match the approved request")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute(
                "UPDATE attempts SET withdrawal_at=? WHERE id=?",
                (int(pending["withdraw_at"]), int(attempt_id)))
            cursor = self._db.execute(
                "UPDATE withdrawal_cycles SET state='ready',request_attempt_id=? "
                "WHERE act_id=? AND state IN ('requesting','ready')",
                (int(attempt_id), row["withdrawal_act_id"]))
            if cursor.rowcount != 1:
                raise pay_voucher.VoucherError("withdrawal cycle cannot accept request provenance")
            self._db.execute("COMMIT")
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise
        self._secure()
        return dict(self._db.execute(
            "SELECT * FROM attempts WHERE id=?", (int(attempt_id),)).fetchone())

    def withdraw_cycle_for_execute(self, escrow: str, generation: int) -> tuple[dict, dict]:
        """Return only the current unconsumed request cycle; never historical rows."""
        escrow_key = pay_chain.pubkey_text(escrow, "escrow")
        cycle = self._db.execute(
            "SELECT * FROM withdrawal_cycles WHERE escrow=? AND state!='consumed'",
            (escrow_key,)).fetchone()
        if cycle is None or int(cycle["generation"]) != int(generation):
            raise pay_voucher.VoucherError(
                "withdraw execute requires an active finalized withdraw-request record")
        if cycle["state"] == "requesting" or cycle["request_attempt_id"] is None:
            raise pay_voucher.VoucherError(
                "withdraw execute requires an active finalized withdraw-request record")
        request = self._db.execute(
            "SELECT * FROM attempts WHERE id=?", (int(cycle["request_attempt_id"]),)).fetchone()
        if (request is None or request["resolution"] != "succeeded"
                or request["signature"] is None or request["withdrawal_at"] is None
                or request["withdrawal_message_digest"] != request["message_sha256"]):
            raise pay_voucher.VoucherError(
                "withdraw execute requires an active finalized withdraw-request record")
        return dict(cycle), dict(request)

    def assert_withdraw_pending(self, rpc, request: dict) -> None:
        """Require current finalized pending state to equal the request binding."""
        info = rpc.account_info(request["escrow"], commitment="finalized")
        if info is None:
            raise pay_voucher.VoucherError("wallet escrow is missing before withdraw execute")
        state = pay_chain.decode_escrow(info["data"])
        pending = state["pending_withdraw"]
        if (int(state["generation"]) != int(request["generation"]) or pending is None
                or int(pending["amount"]) != int(request["withdrawal_amount"])
                or int(pending["withdraw_at"]) != int(request["withdrawal_at"])):
            raise pay_voucher.VoucherError(
                "current pending withdrawal does not match the approved request record")

    def consume_withdraw_request(self, request_id: int, act_id: str,
                                 execute_logical_key: str) -> None:
        """Link one request record to its sole execute act before broker enqueue."""
        _text(act_id, "withdrawal act id")
        _text(execute_logical_key, "execute logical key")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            request = self._db.execute(
                "SELECT * FROM attempts WHERE id=?", (int(request_id),)).fetchone()
            cycle = self._db.execute(
                "SELECT * FROM withdrawal_cycles WHERE act_id=?", (act_id,)).fetchone()
            if (request is None or cycle is None or request["withdrawal_act_id"] != act_id
                    or request["withdrawal_consumed_by"] not in {"", execute_logical_key}
                    or cycle["execute_logical_key"] not in {"", execute_logical_key}
                    or cycle["state"] not in {"ready", "executing"}):
                raise pay_voucher.VoucherError(
                    "withdraw request record was already consumed by another execute")
            self._db.execute(
                "UPDATE attempts SET withdrawal_consumed_by=? WHERE id=?",
                (execute_logical_key, int(request_id)))
            self._db.execute(
                "UPDATE withdrawal_cycles SET state='executing',execute_logical_key=? "
                "WHERE act_id=?", (execute_logical_key, act_id))
            self._db.execute("COMMIT")
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise
        self._secure()

    def complete_withdraw_execute(self, act_id: str) -> None:
        """Close a cycle permanently after its execute reaches finalized success."""
        cursor = self._db.execute(
            "UPDATE withdrawal_cycles SET state='consumed' "
            "WHERE act_id=? AND state='executing'", (act_id,))
        if cursor.rowcount != 1:
            raise pay_voucher.VoucherError("withdraw execute cycle is not active")
        self._secure()

    def close(self) -> None:
        self._secure()
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()


def attempt_lock(path: str | Path | None = None) -> _OpenLock:
    return _OpenLock(path or Path.home() / ".nakshatra" / "pay" / "wallet-attempts.lock")
