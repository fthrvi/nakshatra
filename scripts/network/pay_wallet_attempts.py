"""Durable approval and transaction attempts for broker-held pay wallets.

An approval key is committed before the broker is contacted.  Once signed, the
complete transaction identity is committed before ``sendTransaction``.  A
replacement approval is fenced until every older signature is finalized or is
proved expired by the same status/height/status rule used by the redeemer and
starter paths.
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
            "resolution TEXT, error TEXT NOT NULL DEFAULT '', created INTEGER NOT NULL DEFAULT 0)"
        )
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
                created: int = 0) -> PreparedApproval:
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
            "escrow,generation,withdrawal_message_digest,created) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (scope, logical, operation, wallet, custody, idempotency_key,
             sqlite3.Binary(raw), digest, str(blockhash), int(last_valid_block_height),
             escrow_key, int(generation), withdrawal_digest, int(created)))
        row = self._db.execute(
            "SELECT * FROM attempts WHERE id=?", (cursor.lastrowid,)).fetchone()
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
        self._db.execute(
            "UPDATE attempts SET resolution=?,error=? WHERE signature=? AND resolution IS NULL",
            (resolution, str(error)[:500], str(signature)))
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
                    self.resolve_signature(signature, "failed", str(status["err"]))
                continue
            unresolved = True
        self._secure()
        succeeded = tuple(row for row in self._rows(scope=scope)
                          if row["resolution"] == "succeeded")
        return ReconcileResult(unresolved, succeeded)

    def require_finalized_withdraw_request(self, rpc, escrow: str, generation: int) -> dict:
        """Return finalized request provenance for an execute, or refuse it."""
        prefix = operation_scope("withdraw-request", escrow, generation)
        result = self.reconcile(prefix, rpc)
        if result.unresolved:
            raise pay_voucher.VoucherError(
                "previous withdraw-request transaction is still unresolved")
        candidates = [row for row in self._rows(
            operation="withdraw-request", escrow=pay_chain.pubkey_text(escrow),
            generation=int(generation)) if row["resolution"] == "succeeded"
            and row["withdrawal_message_digest"] == row["message_sha256"]]
        if not candidates:
            raise pay_voucher.VoucherError(
                "withdraw execute requires a reconciled finalized withdraw-request "
                "for this escrow generation")
        return candidates[-1]

    def close(self) -> None:
        self._secure()
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()


def attempt_lock(path: str | Path | None = None) -> _OpenLock:
    return _OpenLock(path or Path.home() / ".nakshatra" / "pay" / "wallet-attempts.lock")
