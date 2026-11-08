"""Durable broker-wallet attempt fencing with fake RPC only."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from network import nak, pay_chain, pay_wallet_attempts, pay_wallet_custody


BLOCKHASH = pay_chain.b58encode(bytes(range(32)))


def _key(byte: int) -> str:
    public = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes([byte]) * 32).public_key().public_bytes_raw()
    return pay_chain.b58encode(public)


WALLET = _key(11)
ESCROW = _key(12)
PROGRAM = _key(13)


def _escrow_data(*, pending: tuple[int, int] | None = None, generation: int = 7) -> bytes:
    raw = bytearray(163)
    raw[:8] = hashlib.sha256(b"account:Escrow").digest()[:8]
    raw[8:40] = pay_chain.pubkey_bytes(WALLET)
    raw[40:72] = pay_chain.pubkey_bytes(_key(14))
    struct.pack_into("<Q", raw, 104, generation)
    if pending is None:
        raw[112] = 0
        offset = 113
    else:
        raw[112] = 1
        struct.pack_into("<Qq", raw, 113, *pending)
        offset = 129
    raw[offset:offset + 32] = pay_chain.pubkey_bytes(WALLET)
    raw[offset + 32:offset + 34] = b"\x01\x01"
    return bytes(raw)


class Rpc:
    def __init__(self, statuses=(), *, height=50):
        self.statuses = list(statuses)
        self.height = height
        self.status_calls = 0

    def signature_status(self, _signature):
        self.status_calls += 1
        return self.statuses.pop(0) if self.statuses else None

    def block_height(self, *, commitment="finalized"):
        assert commitment == "finalized"
        return self.height


class WithdrawalRpc(Rpc):
    def __init__(self, statuses=(), *, pending=None):
        super().__init__(statuses, height=50)
        self.pending = pending

    def account_info(self, address, *, commitment="finalized"):
        assert address == ESCROW
        assert commitment == "finalized"
        return {"owner": PROGRAM, "data": _escrow_data(pending=self.pending)}


def _prepared(book, *, operation="ensure", intent="a" * 64, lvh=60,
              act_id="", amount=None):
    scope = pay_wallet_attempts.operation_scope(operation, ESCROW, 7, act_id=act_id)
    logical = pay_wallet_attempts.logical_key(scope, intent)
    message = b"compiled-message-" + intent[:1].encode()
    return scope, logical, book.prepare(
        scope=scope, logical=logical, operation=operation, wallet_id=WALLET,
        custody="broker", message=message, blockhash=BLOCKHASH,
        last_valid_block_height=lvh, escrow=ESCROW, generation=7,
        withdrawal_act_id=act_id, withdrawal_amount=amount)


def test_signed_attempt_commits_all_prebroadcast_fields(tmp_path):
    path = tmp_path / "attempts.sqlite"
    with pay_wallet_attempts.AttemptJournal(path) as book:
        scope, _logical, approval = _prepared(book)
        signature = pay_chain.b58encode(b"s" * 64)
        book.record_signed(approval.id, signature)

    db = sqlite3.connect(path)
    row = db.execute(
        "SELECT operation,signature,message_sha256,blockhash,"
        "last_valid_block_height,idempotency_key,escrow,generation FROM attempts"
    ).fetchone()
    assert row == (
        "ensure", signature, hashlib.sha256(approval.message).hexdigest(),
        BLOCKHASH, 60, approval.idempotency_key, ESCROW, 7)
    assert approval.idempotency_key.startswith("wallet:ensure:")
    assert scope.startswith("ensure:")


def test_existing_attempt_journal_migrates_with_withdrawal_cycle_fields(tmp_path):
    path = tmp_path / "legacy.sqlite"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE attempts("
        "id INTEGER PRIMARY KEY, operation_scope TEXT NOT NULL, logical_key TEXT NOT NULL, "
        "operation TEXT NOT NULL, wallet_id TEXT NOT NULL, custody TEXT NOT NULL, "
        "idempotency_key TEXT NOT NULL UNIQUE, message BLOB NOT NULL, "
        "message_sha256 TEXT NOT NULL, blockhash TEXT NOT NULL, "
        "last_valid_block_height INTEGER NOT NULL, signature TEXT UNIQUE, "
        "escrow TEXT NOT NULL, generation INTEGER NOT NULL, "
        "withdrawal_message_digest TEXT NOT NULL DEFAULT '', resolution TEXT, "
        "error TEXT NOT NULL DEFAULT '', created INTEGER NOT NULL DEFAULT 0)"
    )
    db.commit()
    db.close()

    with pay_wallet_attempts.AttemptJournal(path) as book:
        columns = {row[1] for row in book._db.execute(  # noqa: SLF001 - migration assertion
            "PRAGMA table_info(attempts)")}
        assert {"withdrawal_act_id", "withdrawal_amount", "withdrawal_at",
                "withdrawal_consumed_by"} <= columns
        assert book._db.execute(  # noqa: SLF001 - migration assertion
            "SELECT name FROM sqlite_master WHERE name='withdrawal_cycles'").fetchone()


def test_subprocess_exit_after_prebroadcast_record_preserves_attempt(tmp_path):
    path = tmp_path / "killed.sqlite"
    script = """
import os
from network import pay_chain, pay_wallet_attempts
path,wallet,escrow,blockhash,signature = os.environ['ARGS'].split('|')
scope=pay_wallet_attempts.operation_scope('ensure',escrow,7)
logical=pay_wallet_attempts.logical_key(scope,'c'*64)
with pay_wallet_attempts.AttemptJournal(path) as book:
    approval=book.prepare(scope=scope,logical=logical,operation='ensure',wallet_id=wallet,
        custody='broker',message=b'kill-boundary-message',blockhash=blockhash,
        last_valid_block_height=60,escrow=escrow,generation=7)
    book.record_signed(approval.id,signature)
    os._exit(17)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "scripts") + os.pathsep + env.get(
        "PYTHONPATH", "")
    signature = pay_chain.b58encode(b"k" * 64)
    env["ARGS"] = "|".join((str(path), WALLET, ESCROW, BLOCKHASH, signature))
    result = subprocess.run([sys.executable, "-c", script], env=env, check=False)
    assert result.returncode == 17
    row = sqlite3.connect(path).execute(
        "SELECT signature,message_sha256,blockhash,last_valid_block_height FROM attempts"
    ).fetchone()
    assert row == (
        signature, hashlib.sha256(b"kill-boundary-message").hexdigest(), BLOCKHASH, 60)


@pytest.mark.parametrize("confirmation", ["processed", "confirmed"])
def test_observed_attempt_stays_unresolved_past_last_valid_height(tmp_path, confirmation):
    with pay_wallet_attempts.AttemptJournal(tmp_path / "attempts.sqlite") as book:
        scope, _logical, approval = _prepared(book, lvh=60)
        book.record_signed(approval.id, pay_chain.b58encode(b"o" * 64))
        result = book.reconcile(scope, Rpc([{
            "err": None, "confirmationStatus": confirmation}], height=61))
        assert result.unresolved is True
        assert result.succeeded == ()


def test_status_appearing_on_expiry_reread_is_not_expired(tmp_path):
    status = {"err": None, "confirmationStatus": "finalized"}
    with pay_wallet_attempts.AttemptJournal(tmp_path / "attempts.sqlite") as book:
        scope, _logical, approval = _prepared(book, lvh=60)
        book.record_signed(approval.id, pay_chain.b58encode(b"r" * 64))
        rpc = Rpc([None, status], height=61)
        result = book.reconcile(scope, rpc)
        assert not result.unresolved
        assert len(result.succeeded) == 1
        assert rpc.status_calls == 2


def test_two_absent_reads_expire_and_rebuild_gets_new_approval_key(tmp_path):
    with pay_wallet_attempts.AttemptJournal(tmp_path / "attempts.sqlite") as book:
        scope, logical, first = _prepared(book, lvh=60)
        book.record_signed(first.id, pay_chain.b58encode(b"x" * 64))
        assert not book.reconcile(scope, Rpc([None, None], height=61)).unresolved
        second = book.prepare(
            scope=scope, logical=logical, operation="ensure", wallet_id=WALLET,
            custody="broker", message=b"new-blockhash-message", blockhash=BLOCKHASH,
            last_valid_block_height=80, escrow=ESCROW, generation=7)
        assert second.id != first.id
        assert second.idempotency_key != first.idempotency_key


def test_withdraw_execute_requires_finalized_request_provenance(tmp_path):
    with pay_wallet_attempts.AttemptJournal(tmp_path / "attempts.sqlite") as book:
        rpc = WithdrawalRpc([{"err": None, "confirmationStatus": "finalized"}])
        with pytest.raises(ValueError, match="active finalized withdraw-request"):
            book.withdraw_cycle_for_execute(ESCROW, 7)
        act_id = book.begin_withdraw_request(rpc, ESCROW, 7, 500_000)
        scope, _logical, approval = _prepared(
            book, operation="withdraw-request", intent="b" * 64,
            act_id=act_id, amount=500_000)
        signature = pay_chain.b58encode(b"w" * 64)
        book.record_signed(approval.id, signature)
        result = book.reconcile(scope, rpc)
        assert len(result.succeeded) == 1
        rpc.pending = (500_000, 12_345)
        provenance = book.bind_finalized_withdraw_request(rpc, approval.id)
        assert provenance["signature"] == signature
        assert provenance["withdrawal_message_digest"] == provenance["message_sha256"]
        assert provenance["withdrawal_act_id"] == act_id
        assert (provenance["withdrawal_amount"], provenance["withdrawal_at"]) == (
            500_000, 12_345)


def test_withdraw_pending_must_match_bound_request_and_active_request_is_singleton(tmp_path):
    rpc = WithdrawalRpc([{"err": None, "confirmationStatus": "finalized"}])
    with pay_wallet_attempts.AttemptJournal(tmp_path / "attempts.sqlite") as book:
        act_id = book.begin_withdraw_request(rpc, ESCROW, 7, 500_000)
        _scope, _logical, approval = _prepared(
            book, operation="withdraw-request", intent="d" * 64,
            act_id=act_id, amount=500_000)
        book.record_signed(approval.id, pay_chain.b58encode(b"d" * 64))
        book.reconcile(_scope, rpc)
        rpc.pending = (500_000, 22_222)
        book.bind_finalized_withdraw_request(rpc, approval.id)

        with pytest.raises(ValueError, match="still active"):
            book.begin_withdraw_request(rpc, ESCROW, 7, 500_000)
        _cycle, request = book.withdraw_cycle_for_execute(ESCROW, 7)
        rpc.pending = (500_001, 22_222)
        with pytest.raises(ValueError, match="does not match"):
            book.assert_withdraw_pending(rpc, request)


def test_finalized_rpc_error_is_reduced_to_closed_code(tmp_path, caplog):
    secret = "Bearer SECRET-TOKEN"
    path = tmp_path / "attempts.sqlite"
    with pay_wallet_attempts.AttemptJournal(path) as book:
        scope, _logical, approval = _prepared(book)
        book.record_signed(approval.id, pay_chain.b58encode(b"e" * 64))
        result = book.reconcile(scope, Rpc([{
            "err": {"rpc_body": secret}, "confirmationStatus": "finalized"}]))
        assert not result.unresolved
        assert book.attempts(scope=scope)[0]["error"] == "finalized-error"
    assert all(secret.encode() not in candidate.read_bytes()
               for candidate in tmp_path.glob("attempts.sqlite*"))
    assert secret not in caplog.text


def test_custody_flip_during_record_signed_refuses_before_broadcast(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    pay_dir = home / ".nakshatra" / "pay"
    pay_dir.mkdir(parents=True, mode=0o700)
    pay_dir.chmod(0o700)
    (pay_dir / "wallet.pub").write_text(WALLET + "\n")
    custody = pay_dir / "custody"
    custody.write_text("broker\n")
    monkeypatch.setenv("HOME", str(home))
    private = ed25519.Ed25519PrivateKey.from_private_bytes(bytes([11]) * 32)
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda message, _wallet, *, idempotency_key: private.sign(message))
    original = pay_wallet_attempts.AttemptJournal.record_signed

    def record_then_flip(self, attempt_id, signature):
        original(self, attempt_id, signature)
        custody.write_text("file\n")

    monkeypatch.setattr(pay_wallet_attempts.AttemptJournal, "record_signed", record_then_flip)

    class SendRpc(Rpc):
        sent = 0
        def latest_blockhash_info(self):
            return {"blockhash": BLOCKHASH, "lastValidBlockHeight": 60}
        def fee_for_message(self, _message): return 5_000
        def balance(self, _address): return 1_000_000
        def cluster_label(self): return "local"
        def send_and_confirm(self, _transaction, *, last_valid_block_height):
            self.sent += 1
            return _transaction.signature

    rpc = SendRpc()
    with pytest.raises(SystemExit, match="custody mode changed.*nothing was signed"):
        nak._pay_send(
            rpc, pay_chain, [pay_chain.system_transfer(WALLET, _key(20), 1)],
            {"address": WALLET, "custody": "broker"}, operation="ensure",
            escrow=ESCROW, generation=7)
    assert rpc.sent == 0


def test_custody_lock_held_by_second_process_refuses_within_bound(tmp_path):
    home = tmp_path / "home"
    lock_path = home / ".nakshatra" / "pay" / "custody.lock"
    lock_path.parent.mkdir(parents=True, mode=0o700)
    script = (
        "import fcntl,os,sys,time\n"
        "fd=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR,0o600)\n"
        "fcntl.flock(fd,fcntl.LOCK_EX)\n"
        "print('locked',flush=True)\n"
        "time.sleep(10)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path)], stdout=subprocess.PIPE,
        text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        started = time.monotonic()
        with pytest.raises(SystemExit, match="another process"):
            with pay_wallet_custody.custody_lock(home=home, timeout_s=0.15):
                pytest.fail("custody act proceeded without the lock")
        assert time.monotonic() - started < 1.0
    finally:
        process.terminate()
        process.wait(timeout=2)


def test_two_same_amount_withdrawal_acts_each_approve_and_send_once(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    pay_dir = home / ".nakshatra" / "pay"
    pay_dir.mkdir(parents=True, mode=0o700)
    pay_dir.chmod(0o700)
    (pay_dir / "wallet.pub").write_text(WALLET + "\n")
    (pay_dir / "custody").write_text("broker\n")
    monkeypatch.setenv("HOME", str(home))
    private = ed25519.Ed25519PrivateKey.from_private_bytes(bytes([11]) * 32)
    approvals = []

    def sign(message, _wallet, *, idempotency_key):
        approvals.append(idempotency_key)
        return private.sign(message)

    monkeypatch.setattr(pay_wallet_custody, "wallet_sign", sign)

    class ActRpc(WithdrawalRpc):
        def __init__(self):
            super().__init__()
            self.blockhash_counter = 0
            self.send_kinds = []
            self.next_kind = ""
            self.next_pending = None

        def latest_blockhash_info(self):
            self.blockhash_counter += 1
            return {
                "blockhash": pay_chain.b58encode(bytes([self.blockhash_counter]) * 32),
                "lastValidBlockHeight": 100 + self.blockhash_counter,
            }

        def fee_for_message(self, _message): return 5_000
        def balance(self, _address): return 1_000_000
        def cluster_label(self): return "local"

        def send_and_confirm(self, transaction, *, last_valid_block_height):
            assert last_valid_block_height >= 101
            self.send_kinds.append(self.next_kind)
            self.pending = self.next_pending
            return transaction.signature

    rpc = ActRpc()
    wallet = {"address": WALLET, "custody": "broker"}
    request = pay_chain.escrow_withdraw_request(WALLET, 500_000, PROGRAM)
    execute = pay_chain.escrow_withdraw_execute(WALLET, _key(15), PROGRAM)

    rpc.next_kind, rpc.next_pending = "request", (500_000, 10_001)
    nak._pay_send(
        rpc, pay_chain, request, wallet, operation="withdraw-request",
        escrow=ESCROW, generation=7, withdrawal_amount=500_000)
    rpc.next_kind, rpc.next_pending = "execute", None
    nak._pay_send(
        rpc, pay_chain, execute, wallet, operation="withdraw-execute",
        escrow=ESCROW, generation=7)

    with pytest.raises(SystemExit, match="active finalized withdraw-request"):
        nak._pay_send(
            rpc, pay_chain, execute, wallet, operation="withdraw-execute",
            escrow=ESCROW, generation=7)

    rpc.next_kind, rpc.next_pending = "request", (500_000, 10_002)
    nak._pay_send(
        rpc, pay_chain, request, wallet, operation="withdraw-request",
        escrow=ESCROW, generation=7, withdrawal_amount=500_000)
    rpc.next_kind, rpc.next_pending = "execute", None
    nak._pay_send(
        rpc, pay_chain, execute, wallet, operation="withdraw-execute",
        escrow=ESCROW, generation=7)

    assert rpc.send_kinds == ["request", "execute", "request", "execute"]
    assert len([key for key in approvals if key.startswith("wallet:withdraw-request:")]) == 2
    with pay_wallet_attempts.AttemptJournal() as book:
        request_rows = [row for row in book.attempts()
                        if row["operation"] == "withdraw-request"]
        assert len(request_rows) == 2
        assert len({row["withdrawal_act_id"] for row in request_rows}) == 2
        assert all(row["withdrawal_consumed_by"] for row in request_rows)


def test_unsigned_approval_key_is_reused_until_blockhash_expires(tmp_path):
    with pay_wallet_attempts.AttemptJournal(tmp_path / "attempts.sqlite") as book:
        scope, logical, first = _prepared(book, lvh=60)
        result = book.reconcile(
            scope, Rpc(height=60), resumable_logical_key=logical)
        assert not result.unresolved
        reused = book.prepare(
            scope=scope, logical=logical, operation="ensure", wallet_id=WALLET,
            custody="broker", message=b"replacement-must-not-win", blockhash=BLOCKHASH,
            last_valid_block_height=70, escrow=ESCROW, generation=7)
        assert reused.id == first.id
        assert reused.idempotency_key == first.idempotency_key
        assert reused.message == first.message


def test_cli_unknown_broadcast_is_fenced_then_expiry_reapproves(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    pay_dir = home / ".nakshatra/pay"
    pay_dir.mkdir(parents=True, mode=0o700)
    pay_dir.chmod(0o700)
    (pay_dir / "wallet.pub").write_text(WALLET + "\n")
    (pay_dir / "custody").write_text("broker\n")
    monkeypatch.setenv("HOME", str(home))
    private = ed25519.Ed25519PrivateKey.from_private_bytes(bytes([11]) * 32)
    approvals = []

    def sign(message, wallet_id, *, idempotency_key):
        assert wallet_id == WALLET
        approvals.append((idempotency_key, message))
        return private.sign(message)

    monkeypatch.setattr(pay_wallet_custody, "wallet_sign", sign)
    blockhashes = [
        {"blockhash": BLOCKHASH, "lastValidBlockHeight": 60},
        {"blockhash": pay_chain.b58encode(bytes(range(1, 33))),
         "lastValidBlockHeight": 80},
    ]

    class SendRpc(Rpc):
        def __init__(self):
            super().__init__(height=50)
            self.fail_unknown = True

        def latest_blockhash_info(self):
            return blockhashes.pop(0)

        def fee_for_message(self, _message): return 5_000
        def balance(self, _address): return 1_000_000
        def cluster_label(self): return "local"

        def send_and_confirm(self, transaction, *, last_valid_block_height):
            if self.fail_unknown:
                assert last_valid_block_height == 60
                self.fail_unknown = False
                raise pay_chain.VoucherError("transaction final state is unknown")
            assert last_valid_block_height == 80
            return transaction.signature

    rpc = SendRpc()
    instructions = [pay_chain.system_transfer(WALLET, _key(21), 1)]
    wallet = {"address": WALLET, "custody": "broker"}

    with pytest.raises(SystemExit, match="final state is unknown"):
        nak._pay_send(
            rpc, pay_chain, instructions, wallet, operation="ensure",
            escrow=ESCROW, generation=7)
    with pytest.raises(SystemExit, match="still unresolved"):
        nak._pay_send(
            rpc, pay_chain, instructions, wallet, operation="ensure",
            escrow=ESCROW, generation=7)
    assert len(approvals) == 1

    rpc.height = 61
    signature = nak._pay_send(
        rpc, pay_chain, instructions, wallet, operation="ensure",
        escrow=ESCROW, generation=7)
    assert signature
    assert len(approvals) == 2
    assert approvals[0][0] != approvals[1][0]
    assert approvals[0][1] != approvals[1][1]
