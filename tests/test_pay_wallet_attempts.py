"""Durable broker-wallet attempt fencing with fake RPC only."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
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


def _prepared(book, *, operation="ensure", intent="a" * 64, lvh=60):
    scope = pay_wallet_attempts.operation_scope(operation, ESCROW, 7)
    logical = pay_wallet_attempts.logical_key(scope, intent)
    message = b"compiled-message-" + intent[:1].encode()
    return scope, logical, book.prepare(
        scope=scope, logical=logical, operation=operation, wallet_id=WALLET,
        custody="broker", message=message, blockhash=BLOCKHASH,
        last_valid_block_height=lvh, escrow=ESCROW, generation=7)


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
        with pytest.raises(ValueError, match="reconciled finalized withdraw-request"):
            book.require_finalized_withdraw_request(Rpc(height=1), ESCROW, 7)

        scope, _logical, approval = _prepared(
            book, operation="withdraw-request", intent="b" * 64)
        signature = pay_chain.b58encode(b"w" * 64)
        book.record_signed(approval.id, signature)
        result = book.reconcile(scope, Rpc([{
            "err": None, "confirmationStatus": "finalized"}], height=1))
        assert len(result.succeeded) == 1
        provenance = book.require_finalized_withdraw_request(Rpc(height=1), ESCROW, 7)
        assert provenance["signature"] == signature
        assert provenance["withdrawal_message_digest"] == provenance["message_sha256"]


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
