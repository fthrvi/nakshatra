from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import stat
import struct
import time
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from network import pay_autochannel as auto
from network import pay_chain
from network import pay_voucher as voucher


PROGRAM = pay_chain.b58encode(bytes([41]) * 32)
OWNER_SEED = bytes([42]) * 32
OWNER = pay_chain.b58encode(ed25519.Ed25519PrivateKey.from_private_bytes(
    OWNER_SEED).public_key().public_bytes_raw())
NODE_SEED = bytes([43]) * 32
NODE_PUBLIC = ed25519.Ed25519PrivateKey.from_private_bytes(
    NODE_SEED).public_key().public_bytes_raw()
NODE = pay_chain.b58encode(NODE_PUBLIC)
WORKER_SEED = bytes([44]) * 32
WORKER_PUBLIC = ed25519.Ed25519PrivateKey.from_private_bytes(
    WORKER_SEED).public_key().public_bytes_raw()
WORKER = WORKER_PUBLIC.hex()
OTHER_WORKER = ed25519.Ed25519PrivateKey.from_private_bytes(
    bytes([45]) * 32).public_key().public_bytes_raw().hex()
BLOCKHASH = pay_chain.b58encode(bytes(range(32)))


def _escrow_data(owner: str, vault: str, *, committed: int, seq: int,
                 generation: int = 0) -> bytes:
    raw = bytearray(163)
    raw[:8] = hashlib.sha256(b"account:Escrow").digest()[:8]
    raw[8:40] = pay_chain.pubkey_bytes(owner)
    raw[40:72] = pay_chain.pubkey_bytes(vault)
    struct.pack_into("<QIIQQQ", raw, 72, committed, seq, 0, seq, 0, generation)
    raw[112] = 0
    raw[113:145] = pay_chain.pubkey_bytes(owner)
    raw[145:147] = b"\x01\x01"
    return bytes(raw)


def _delegate_data(escrow: str, *, cap: int, spent: int, expires: int,
                   flags: int = 1, disabled: bool = False,
                   generation: int = 0) -> bytes:
    raw = bytearray(115)
    raw[:8] = hashlib.sha256(b"account:Delegate").digest()[:8]
    raw[8:40] = pay_chain.pubkey_bytes(escrow)
    raw[40:72] = NODE_PUBLIC
    struct.pack_into("<QQQqBII", raw, 72, generation, cap, spent, expires,
                     flags, 0, 0)
    raw[113] = int(disabled)
    raw[114] = 1
    return bytes(raw)


def _token_data(amount: int) -> bytes:
    raw = bytearray(165)
    struct.pack_into("<Q", raw, 64, amount)
    return bytes(raw)


class FakeRpc:
    def __init__(self, *, open_cap=10, vault_amount=100, delegate_cap=100,
                 expires=10_000, flags=1, disabled=False, node_balance=100_000,
                 has_delegate=True, shared_seq=None, shared_sent=None,
                 shared_lock=None, fail_definite=False, fail_unknown=False,
                 chain_time=1000, chain_time_error=None,
                 channel_changes=None):
        self.open_cap = open_cap
        self.vault_amount = vault_amount
        self.delegate_cap = delegate_cap
        self.expires = expires
        self.flags = flags
        self.disabled = disabled
        self.node_balance = node_balance
        self.has_delegate = has_delegate
        self.fail_definite = fail_definite
        self.fail_unknown = fail_unknown
        self.current_chain_time = chain_time
        self.chain_time_error = chain_time_error
        self.channel_changes = dict(channel_changes or {})
        self.seq = shared_seq or multiprocessing.Value("q", 0)
        self.sent = shared_sent or multiprocessing.Value("i", 0)
        self.send_lock = shared_lock or multiprocessing.Lock()
        self.derived = pay_chain._derived(OWNER, PROGRAM)
        self.delegate = pay_chain.delegate_pda(
            self.derived["escrow"], NODE_PUBLIC, PROGRAM)[0]
        self.last_seq = 0
        self.cluster_checks = 0

    def ensure_cluster(self):
        self.cluster_checks += 1
        return "local-test"

    def chain_time(self):
        if self.chain_time_error is not None:
            raise self.chain_time_error
        return self.current_chain_time

    def account_info(self, address):
        with self.seq.get_lock():
            seq = self.seq.value
        if address == self.derived["escrow"]:
            self.last_seq = seq
            data = _escrow_data(
                OWNER, self.derived["vault"], committed=seq * self.open_cap,
                seq=seq)
            return {"owner": PROGRAM, "data": data}
        if address == self.delegate and self.has_delegate:
            data = _delegate_data(
                self.derived["escrow"], cap=self.delegate_cap,
                spent=seq * self.open_cap, expires=self.expires,
                flags=self.flags, disabled=self.disabled)
            return {"owner": PROGRAM, "data": data}
        if address == self.derived["vault"]:
            return {"owner": pay_chain.TOKEN_PROGRAM_ID,
                    "data": _token_data(self.vault_amount)}
        return None

    def latest_blockhash(self):
        return BLOCKHASH

    def fee_for_message(self, _message):
        return 5

    def rent_exemption(self, space):
        assert space == auto.CHANNEL_ACCOUNT_SPACE
        return 100

    def balance(self, address):
        assert address == NODE
        return self.node_balance

    def send_and_confirm(self, transaction, *, on_sent=None):
        if self.fail_definite:
            raise pay_chain.TransactionDefiniteFailure("preflight rejected")
        if self.fail_unknown:
            raise voucher.VoucherError("send outcome unknown")
        with self.send_lock:
            with self.seq.get_lock():
                if self.seq.value != self.last_seq:
                    raise pay_chain.TransactionDefiniteFailure("channel PDA already in use")
                if on_sent is not None:
                    on_sent(transaction.signature)
                self.seq.value += 1
            with self.sent.get_lock():
                self.sent.value += 1
        return transaction.signature

    def read_channel(self, address, program_id):
        assert program_id == PROGRAM
        with self.seq.get_lock():
            count = self.seq.value
        for seq in range(count):
            expected = pay_chain.channel_pda(
                self.derived["escrow"], 0, seq, PROGRAM)[0]
            if address == expected:
                account = voucher.ChannelAccount(
                    escrow=self.derived["escrow"], generation=0, seq=seq,
                    worker=pay_chain.b58encode(WORKER_PUBLIC),
                    voucher_signer=NODE, delegate=self.delegate,
                    cap=self.open_cap, redeemed=0, fee_bps_snapshot=300,
                    status="Open", close_at=0, rent_payer=NODE, bump=1)
                return replace(account, **self.channel_changes)
        raise voucher.VoucherError(f"account does not exist: {address}")

    def read_escrow_owner(self, escrow, program_id):
        assert escrow == self.derived["escrow"] and program_id == PROGRAM
        return OWNER


def _limits(path: Path, *, per_tx=10, per_day=50, cap=100):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps({
        "cap_total": cap, "days": 30, "per_tx": per_tx, "per_day": per_day,
    }))
    path.chmod(0o600)


def _paths(tmp_path):
    pay = tmp_path / "pay"
    return {
        "book": pay / "channels.sqlite",
        "limits": pay / "limits.json",
        "ledger": pay / "opens.sqlite",
        "lock": pay / "channel-open.lock",
    }


def _ensure(tmp_path, rpc, *, worker=WORKER, required=1, state=None, clock=None):
    paths = _paths(tmp_path)
    book = voucher.PayerChannelBook(paths["book"])
    try:
        kwargs = {} if clock is None else {"clock": clock}
        return auto.ensure_channel(
            worker, required, owner=OWNER, book=book, state=state or rpc,
            program=PROGRAM, private_key=NODE_SEED, rpc=rpc,
            limits_path=paths["limits"], ledger_path=paths["ledger"],
            lock_path=paths["lock"], **kwargs)
    finally:
        book.close()


def test_per_tx_refuses_required_over_limit_and_caps_open(tmp_path):
    paths = _paths(tmp_path)
    _limits(paths["limits"], per_tx=10, per_day=50)
    rpc = FakeRpc(open_cap=10)
    with pytest.raises(voucher.VoucherError, match="per-transaction.*nak pay limits"):
        _ensure(tmp_path, rpc, required=11)
    assert rpc.sent.value == 0

    chosen = _ensure(tmp_path, rpc, required=5)
    assert chosen.cap == 10
    assert chosen.cap <= 10
    assert rpc.sent.value == 1


def test_open_ledger_counts_pending_unknown_and_recent_only(tmp_path):
    ledger_path = _paths(tmp_path)["ledger"]
    with auto.OpenLedger(ledger_path) as ledger:
        old = ledger.reserve(7, 1000 - auto.DAY_SECONDS - 1)
        pending = ledger.reserve(10, 1000)
        unknown = ledger.reserve(11, 1000)
        ledger.set_status(unknown, "sent", "unknown-signature")
        failed = ledger.reserve(12, 1000)
        ledger.set_status(failed, "failed")
        finalized = ledger.reserve(13, 1000)
        ledger.set_status(finalized, "finalized", "ok")
        assert ledger.total_since(1000) == 10 + 11 + 13
        assert old and pending
    assert stat.S_IMODE(ledger_path.stat().st_mode) == 0o600


def test_per_day_refuses_after_budget_is_reserved(tmp_path):
    paths = _paths(tmp_path)
    _limits(paths["limits"], per_tx=10, per_day=20)
    with auto.OpenLedger(paths["ledger"]) as ledger:
        ledger.reserve(20, 1000)
    rpc = FakeRpc(open_cap=10)
    with pytest.raises(voucher.VoucherError, match="24-hour.*nak pay limits"):
        _ensure(tmp_path, rpc, required=1)
    assert rpc.sent.value == 0


def test_local_clock_jump_does_not_free_per_day_budget(tmp_path):
    paths = _paths(tmp_path)
    _limits(paths["limits"], per_tx=10, per_day=10)
    with auto.OpenLedger(paths["ledger"]) as ledger:
        ledger.reserve(10, 1000)
    rpc = FakeRpc(open_cap=10, expires=1_000_000, chain_time=1000)

    with pytest.raises(voucher.VoucherError, match="24-hour"):
        _ensure(
            tmp_path, rpc, required=1,
            clock=lambda: 1000 + 10 * auto.DAY_SECONDS)

    assert rpc.sent.value == 0


def test_chain_time_advancing_24_hours_frees_per_day_budget(tmp_path):
    paths = _paths(tmp_path)
    _limits(paths["limits"], per_tx=10, per_day=10)
    with auto.OpenLedger(paths["ledger"]) as ledger:
        ledger.reserve(10, 1000)
    rpc = FakeRpc(
        open_cap=10, expires=1_000_000,
        chain_time=1000 + auto.DAY_SECONDS)

    assert _ensure(tmp_path, rpc, required=1).cap == 10
    assert rpc.sent.value == 1


def test_chain_time_never_moves_behind_newest_ledger_row(tmp_path):
    paths = _paths(tmp_path)
    _limits(paths["limits"], per_tx=10, per_day=10)
    with auto.OpenLedger(paths["ledger"]) as ledger:
        ledger.reserve(10, 100_000)
    rpc = FakeRpc(open_cap=10, expires=1_000_000, chain_time=1000)

    with pytest.raises(voucher.VoucherError, match="24-hour"):
        _ensure(tmp_path, rpc, required=1)

    assert rpc.sent.value == 0


def test_chain_time_read_failure_refuses_to_open(tmp_path):
    paths = _paths(tmp_path)
    _limits(paths["limits"])
    rpc = FakeRpc(
        open_cap=10,
        chain_time_error=voucher.VoucherError("finalized chain time unavailable"))

    with pytest.raises(voucher.VoucherError, match="chain time unavailable"):
        _ensure(tmp_path, rpc, required=1)

    assert rpc.sent.value == 0


def test_definite_failure_releases_but_unknown_send_stays_charged(tmp_path):
    paths = _paths(tmp_path)
    _limits(paths["limits"], per_tx=10, per_day=10)
    rpc = FakeRpc(open_cap=10, fail_definite=True)
    with pytest.raises(pay_chain.TransactionDefiniteFailure):
        _ensure(tmp_path, rpc, required=1)
    with auto.OpenLedger(paths["ledger"]) as ledger:
        assert ledger.total_since(1000) == 0
    rpc.fail_definite = False
    assert _ensure(tmp_path, rpc, required=1).cap == 10

    other = tmp_path / "unknown"
    other_paths = _paths(other)
    _limits(other_paths["limits"], per_tx=10, per_day=10)
    unknown = FakeRpc(open_cap=10, fail_unknown=True)
    with pytest.raises(voucher.VoucherError, match="outcome unknown"):
        _ensure(other, unknown, required=1)
    unknown.fail_unknown = False
    with pytest.raises(voucher.VoucherError, match="24-hour"):
        _ensure(other, unknown, worker=OTHER_WORKER, required=1)
    assert unknown.sent.value == 0


def test_existing_usable_channel_is_reused_without_transaction(tmp_path):
    paths = _paths(tmp_path)
    _limits(paths["limits"])
    rpc = FakeRpc(open_cap=10)
    rpc.seq.value = 1
    channel = pay_chain.channel_pda(rpc.derived["escrow"], 0, 0, PROGRAM)[0]
    with voucher.PayerChannelBook(paths["book"]) as book:
        book.put(program=PROGRAM, channel=channel, worker_key=WORKER, cap=10)
    chosen = _ensure(tmp_path, rpc, required=1)
    assert chosen.channel == channel
    assert rpc.sent.value == 0


def test_closing_and_exhausted_channels_are_skipped_then_opened(tmp_path):
    paths = _paths(tmp_path)
    _limits(paths["limits"])
    closing = pay_chain.b58encode(bytes([50]) * 32)
    exhausted = pay_chain.b58encode(bytes([51]) * 32)
    account = lambda status: voucher.ChannelAccount(
        escrow=pay_chain.b58encode(bytes([52]) * 32), generation=0, seq=0,
        worker=pay_chain.b58encode(WORKER_PUBLIC), voucher_signer=NODE,
        delegate=None, cap=10, redeemed=0, fee_bps_snapshot=300,
        status=status, close_at=2000, rent_payer=NODE, bump=1)
    state = voucher.FakeChannelState({
        closing: account("Closing"), exhausted: account("Open"),
    }, {account("Open").escrow: OWNER}, program_id=PROGRAM)
    with voucher.PayerChannelBook(paths["book"]) as book:
        book.put(program=PROGRAM, channel=closing, worker_key=WORKER, cap=10)
        book.put(program=PROGRAM, channel=exhausted, worker_key=WORKER,
                 cap=10, max_signed_total=10)
    rpc = FakeRpc(open_cap=10)
    chosen = _ensure(tmp_path, rpc, required=1, state=state)
    assert chosen.channel not in {closing, exhausted}
    assert rpc.sent.value == 1
    with voucher.PayerChannelBook(paths["book"]) as book:
        assert len(book.list_channels()) == 3


@pytest.mark.parametrize(("changes", "match"), [
    ({"has_delegate": False}, "delegate is not configured"),
    ({"expires": 999}, "delegate is expired"),
    ({"vault_amount": 0}, "collateral.*nak pay deposit"),
    ({"node_balance": 0}, "node key lacks SOL.*nak pay ensure"),
])
def test_clear_preflight_refusals(tmp_path, changes, match):
    paths = _paths(tmp_path)
    _limits(paths["limits"])
    rpc = FakeRpc(open_cap=10, **changes)
    with pytest.raises(voucher.VoucherError, match=match):
        _ensure(tmp_path, rpc, required=1)
    assert rpc.sent.value == 0


@pytest.mark.parametrize(("field", "value"), [
    ("escrow", pay_chain.b58encode(bytes([60]) * 32)),
    ("worker", pay_chain.b58encode(bytes([61]) * 32)),
    ("cap", 9),
    ("voucher_signer", pay_chain.b58encode(bytes([62]) * 32)),
    ("delegate", pay_chain.b58encode(bytes([63]) * 32)),
    ("status", "Closing"),
])
def test_post_open_verification_rejects_each_mutated_field(
        tmp_path, field, value):
    paths = _paths(tmp_path)
    _limits(paths["limits"])
    rpc = FakeRpc(open_cap=10, channel_changes={field: value})

    with voucher.PayerChannelBook(paths["book"]) as book:
        with pytest.raises(
                voucher.VoucherError,
                match="finalized channel does not match the requested open"):
            auto.ensure_channel(
                WORKER, 1, owner=OWNER, book=book, state=rpc,
                program=PROGRAM, private_key=NODE_SEED, rpc=rpc,
                limits_path=paths["limits"], ledger_path=paths["ledger"],
                lock_path=paths["lock"])
        assert book.list_channels() == []

    assert rpc.sent.value == 1
    with auto.OpenLedger(paths["ledger"]) as ledger:
        assert ledger.total_since(rpc.current_chain_time) == 10


def _concurrent_open_target(root: str, rpc: FakeRpc, start, result):
    try:
        start.wait()
        chosen = _ensure(Path(root), rpc, required=1)
        result.put(("ok", chosen.channel))
    except Exception as exc:  # pragma: no cover - relayed to the parent
        result.put(("error", repr(exc)))


def test_two_processes_open_exactly_one_channel(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fcntl process test requires fork")
    paths = _paths(tmp_path)
    _limits(paths["limits"])
    context = multiprocessing.get_context("fork")
    seq = context.Value("q", 0)
    sent = context.Value("i", 0)
    rpc = FakeRpc(
        open_cap=10, shared_seq=seq, shared_sent=sent,
        shared_lock=context.Lock())
    start = context.Event()
    result = context.Queue()
    processes = [
        context.Process(target=_concurrent_open_target,
                        args=(str(tmp_path), rpc, start, result))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    replies = [result.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert [status for status, _value in replies] == ["ok", "ok"]
    assert len({value for _status, value in replies}) == 1
    assert sent.value == 1
