"""Money-path invariants for worker redemption and channel reclaim."""
from __future__ import annotations

import base64
import builtins
import sqlite3
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from network import nak, nakd, pay_chain, pay_redeemer, pay_voucher  # noqa: E402


PROGRAM = "GLWqQsSqUsDkPvkc8rxcoQUyro7ZQd5denht21NtdTTo"
MINT = pay_chain.b58encode(bytes(range(1, 33)))
BLOCKHASH = pay_chain.b58encode(bytes(range(32, 64)))
OWNER = pay_chain.b58encode(bytes([61]) * 32)
WALLET_SEED = bytes([62]) * 32
WALLET = {
    "seed": WALLET_SEED,
    "public_key": ed25519.Ed25519PrivateKey.from_private_bytes(
        WALLET_SEED).public_key().public_bytes_raw(),
    "custody": "file",
}
WALLET["address"] = pay_chain.b58encode(WALLET["public_key"])
ESCROW = pay_chain.escrow_pda(OWNER, PROGRAM)[0]
CHANNEL = pay_chain.channel_pda(ESCROW, 4, 7, PROGRAM)[0]
CHANNEL2 = pay_chain.channel_pda(ESCROW, 4, 8, PROGRAM)[0]
SIGNER_SEED = bytes([63]) * 32
SIGNER = pay_chain.b58encode(ed25519.Ed25519PrivateKey.from_private_bytes(
    SIGNER_SEED).public_key().public_bytes_raw())


class VoucherStore:
    def __init__(self, total=2_000_000):
        signature = ed25519.Ed25519PrivateKey.from_private_bytes(SIGNER_SEED).sign(
            pay_voucher.voucher_message(PROGRAM, CHANNEL, total))
        self.row = pay_voucher.VoucherRecord(CHANNEL, PROGRAM, total, signature)

    def all_latest(self):
        return [self.row]


def channel_state(*, redeemed=0, status="Open", close_at=0):
    return pay_voucher.ChannelAccount(
        ESCROW, 4, 7, WALLET["address"], SIGNER, None, 5_000_000,
        redeemed, 300, status, close_at, SIGNER, 1)


class Rpc:
    def __init__(self, attempt_path: Path, *, now=1_000_000, state=None,
                 status=None, height=10, ata=True, chain_times=None):
        self.attempt_path = attempt_path
        self.now = now
        self.state = state or channel_state()
        self.status = status
        self.height = height
        self.ata = ata
        self.chain_times = list(chain_times or [])
        self.chain_time_calls = 0
        self.sent = 0
        self.instructions_seen = []

    def ensure_cluster(self): return "local"
    def chain_time(self):
        self.chain_time_calls += 1
        return self.chain_times.pop(0) if self.chain_times else self.now
    def read_channel(self, channel, program):
        assert (channel, program) == (CHANNEL, PROGRAM)
        return self.state
    def read_escrow_owner(self, escrow, program):
        assert (escrow, program) == (ESCROW, PROGRAM)
        return OWNER
    def signature_status(self, _signature):
        return self.status
    def block_height(self, *, commitment):
        assert commitment == "finalized"
        return self.height
    def account_info(self, _address, *, commitment):
        assert commitment == "finalized"
        return ({"owner": pay_chain.TOKEN_PROGRAM_ID, "data": bytes(165)}
                if self.ata else None)
    def latest_blockhash_info(self, *, commitment):
        assert commitment == "finalized"
        return {"blockhash": BLOCKHASH, "lastValidBlockHeight": 100}
    def fee_for_message(self, _message, *, commitment):
        assert commitment == "finalized"
        return 5_000
    def balance(self, address, *, commitment):
        assert address == WALLET["address"] and commitment == "finalized"
        return 1_000_000
    def rent_exemption(self, space, *, commitment):
        assert (space, commitment) == (165, "finalized")
        return 200_000
    def send_and_confirm(self, transaction, *, last_valid_block_height):
        assert last_valid_block_height == 100
        # Mutation checked: moving AttemptBook.record below broadcast makes this
        # fail because the signed intent is absent at the first send boundary.
        db = sqlite3.connect(self.attempt_path)
        try:
            row = db.execute(
                "SELECT signature FROM attempts WHERE signature=?",
                (transaction.signature,)).fetchone()
        finally:
            db.close()
        assert row == (transaction.signature,)
        self.sent += 1
        self.status = {"err": None, "confirmationStatus": "finalized"}
        self.state = channel_state(redeemed=2_000_000)
        return transaction.signature


def run(tmp_path, rpc, **kwargs):
    return pay_redeemer.run_redeemer(
        store=VoucherStore(), rpc=rpc, wallet=WALLET, program=PROGRAM, mint=MINT,
        attempts_path=tmp_path / "redeems.sqlite",
        lock_path=tmp_path / "redeem.lock", **kwargs)


def broker_home(tmp_path, monkeypatch, *, node_seed=bytes([71]) * 32):
    home = tmp_path / "broker-home"
    pay_dir = home / ".nakshatra" / "pay"
    pay_dir.mkdir(parents=True, mode=0o700)
    pay_dir.chmod(0o700)
    (pay_dir / "wallet.pub").write_text(WALLET["address"] + "\n", encoding="ascii")
    (pay_dir / "custody").write_text("broker\n", encoding="ascii")
    key_path = home / ".nakshatra" / "keys" / "worker.ed25519"
    key_path.parent.mkdir(parents=True, mode=0o700)
    key_path.write_bytes(node_seed)
    key_path.chmod(0o600)
    monkeypatch.setenv("HOME", str(home))
    return home, node_seed


def test_redeem_records_signed_attempt_before_broadcast(tmp_path):
    rpc = Rpc(tmp_path / "redeems.sqlite")

    result = run(tmp_path, rpc)

    assert rpc.sent == 1
    assert result["sent"] == 1
    with pay_redeemer.AttemptBook(tmp_path / "redeems.sqlite") as book:
        assert book.unresolved(CHANNEL) == []
        assert book.timing(CHANNEL) == (rpc.now, rpc.now)


def test_redeem_after_close_at_is_never_sent(tmp_path):
    rpc = Rpc(tmp_path / "redeems.sqlite", now=50_000,
              state=channel_state(status="Closing", close_at=50_000))
    messages = []

    result = run(tmp_path, rpc, log=messages.append)

    # Mutation checked: changing >= close_at to > (or deleting the guard)
    # broadcasts a transaction the program must reject after the grace elapsed.
    assert rpc.sent == 0 and result["skipped"] == 1
    assert any("UNREDEEMABLE" in message for message in messages)


def test_closing_redeem_rereads_chain_time_and_refuses_at_safety_margin(tmp_path):
    rpc = Rpc(
        tmp_path / "redeems.sqlite", now=49_000,
        state=channel_state(status="Closing", close_at=50_000),
        chain_times=[49_000, 49_400])
    messages = []

    result = run(tmp_path, rpc, log=messages.append)

    # Mutation checked: checking before signing/durable SQLite only broadcasts
    # here after those operations cross the 600-second refusal point.
    assert rpc.chain_time_calls == 2
    assert rpc.sent == 0 and result["skipped"] == 1
    with pay_redeemer.AttemptBook(tmp_path / "redeems.sqlite") as book:
        attempts = book.unresolved(CHANNEL)
        assert len(attempts) == 1 and attempts[0]["kind"] == "redeem"
    assert any("SAFETY MARGIN" in message and "nothing was sent" in message
               for message in messages)


@pytest.mark.parametrize("fresh_state,chain_times,message", [
    (channel_state(status="Closing", close_at=50_000), [49_000, 49_400],
     "SAFETY MARGIN"),
    (channel_state(redeemed=2_000_000), [49_000, 49_001],
     "not above finalized on-chain redeemed"),
])
def test_redeem_rereads_full_finalized_eligibility_before_broadcast(
        tmp_path, fresh_state, chain_times, message):
    class ChangedChannelRpc(Rpc):
        def __init__(self):
            super().__init__(
                tmp_path / "redeems.sqlite", state=channel_state(),
                chain_times=chain_times)
            self.channel_reads = 0

        def read_channel(self, channel, program):
            self.channel_reads += 1
            assert (channel, program) == (CHANNEL, PROGRAM)
            return channel_state() if self.channel_reads == 1 else fresh_state

    rpc = ChangedChannelRpc()
    messages = []

    result = run(tmp_path, rpc, log=messages.append)

    # Mutation checked: reusing the initial Open/unredeemed account snapshot in
    # the post-durability fence broadcasts after either finalized fact changed.
    assert rpc.channel_reads == 2 and rpc.chain_time_calls == 2
    assert rpc.sent == 0 and result["skipped"] == 1
    with pay_redeemer.AttemptBook(tmp_path / "redeems.sqlite") as book:
        attempts = book.unresolved(CHANNEL)
        assert len(attempts) == 1 and attempts[0]["kind"] == "redeem"
    assert any(message in row and "nothing was sent" in row for row in messages)


def test_voucher_accept_deadline_never_overruns_redeemer_refusal():
    state = channel_state(status="Closing", close_at=50_000)
    assert pay_voucher.ACCEPT_MARGIN == (
        pay_voucher.REDEEM_MARGIN + pay_voucher.CLOSING_POLL_S + 120)
    for chain_now in (
            50_000 - pay_voucher.ACCEPT_MARGIN - 1,
            50_000 - pay_voucher.ACCEPT_MARGIN,
            50_000 - pay_voucher.REDEEM_MARGIN - 1,
            50_000 - pay_voucher.REDEEM_MARGIN,
            50_000):
        try:
            pay_voucher.require_redeemable_channel(state, now=chain_now)
            accepted_as_redeemable = True
        except pay_voucher.ChannelClosingError:
            accepted_as_redeemable = False
        redeemer_would_send = pay_redeemer._redeem_before_margin(
            channel=CHANNEL, state=state, chain_now=chain_now,
            log=lambda _line: None)
        # Mutation checked: restoring the old 15-second accept margin makes a
        # voucher acceptable at times when the redeemer already refuses it.
        assert not accepted_as_redeemable or redeemer_would_send


def test_daily_due_uses_finalized_chain_time_not_local_clock(tmp_path, monkeypatch):
    path = tmp_path / "redeems.sqlite"
    first = Rpc(path, now=1_000_000, state=channel_state())
    first_store = VoucherStore(total=10)
    pay_redeemer.run_redeemer(
        store=first_store, rpc=first, wallet=WALLET, program=PROGRAM, mint=MINT,
        attempts_path=path, lock_path=tmp_path / "redeem.lock")
    assert first.sent == 0
    monkeypatch.setattr(pay_redeemer, "DEFAULT_MAX_AGE_S", 1)  # local module state is irrelevant
    monkeypatch.setattr("time.time", lambda: -10**12)
    second = Rpc(path, now=1_000_000 + 86_400, state=channel_state())

    pay_redeemer.run_redeemer(
        store=first_store, rpc=second, wallet=WALLET, program=PROGRAM, mint=MINT,
        attempts_path=path, lock_path=tmp_path / "redeem.lock")

    # Mutation checked: replacing rpc.chain_time() with time.time() leaves this
    # small voucher unsent despite a full finalized-chain day passing.
    assert second.sent == 1


def test_broker_redeem_uses_node_for_fee_payer_by_and_worker_ata(
        tmp_path, monkeypatch):
    node_seed = bytes([72]) * 32
    node_address = pay_chain.b58encode(pay_voucher.public_key_bytes(node_seed))
    home, _ = broker_home(tmp_path, monkeypatch, node_seed=node_seed)
    assert not (home / ".nakshatra/pay/wallet.ed25519").exists()
    seen = {}
    real_redeem = pay_chain.channel_redeem
    real_ata = pay_chain.ata_create_idempotent

    def capture_redeem(by, *args, **kwargs):
        seen["by"] = by
        return real_redeem(by, *args, **kwargs)

    def capture_ata(payer, owner, mint):
        seen["ata"] = (payer, owner, mint)
        return real_ata(payer, owner, mint)

    monkeypatch.setattr(pay_chain, "channel_redeem", capture_redeem)
    monkeypatch.setattr(pay_chain, "ata_create_idempotent", capture_ata)

    class BrokerRpc(Rpc):
        def balance(self, address, *, commitment):
            assert address == node_address and commitment == "finalized"
            return 1_000_000

    rpc = BrokerRpc(tmp_path / "redeems.sqlite", ata=False)

    result = pay_redeemer.run_redeemer(
        store=VoucherStore(), rpc=rpc, node_seed=node_seed,
        program=PROGRAM, mint=MINT, attempts_path=tmp_path / "redeems.sqlite",
        lock_path=tmp_path / "redeem.lock")

    # Mutation checked: requiring the broker wallet seed rejects before this
    # permissionless redeem, while using it as payer/by cannot sign without it.
    assert result["sent"] == 1 and rpc.sent == 1
    assert seen == {
        "by": node_address,
        "ata": (node_address, WALLET["address"], MINT),
    }


def test_broker_redeem_without_ata_or_node_sol_points_to_pay_ensure(
        tmp_path):
    node_seed = bytes([73]) * 32
    node_address = pay_chain.b58encode(pay_voucher.public_key_bytes(node_seed))

    class PoorBrokerRpc(Rpc):
        def balance(self, address, *, commitment):
            assert address == node_address and commitment == "finalized"
            return 5_000

    rpc = PoorBrokerRpc(tmp_path / "redeems.sqlite", ata=False)
    broker_wallet = {key: WALLET[key] for key in ("public_key", "address")}
    broker_wallet["custody"] = "broker"
    messages = []

    result = pay_redeemer.run_redeemer(
        store=VoucherStore(), rpc=rpc, wallet=broker_wallet, node_seed=node_seed,
        program=PROGRAM, mint=MINT, attempts_path=tmp_path / "redeems.sqlite",
        lock_path=tmp_path / "redeem.lock", log=messages.append)

    # Mutation checked: blindly adding ATA creation either attempts an
    # unaffordable transaction or gives no actionable way to fund its payer.
    assert rpc.sent == 0 and result["skipped"] == 1
    assert any("node key" in row and "nak pay ensure" in row
               and "nothing was sent" in row for row in messages)
    with pay_redeemer.AttemptBook(tmp_path / "redeems.sqlite") as attempts:
        assert attempts.unresolved(CHANNEL) == []


@pytest.mark.parametrize("error", [
    None,
    {"InstructionError": [1, "Custom"]},
])
def test_nonfinalized_status_survives_blockhash_expiry_and_blocks_resend(
        tmp_path, error):
    path = tmp_path / "redeems.sqlite"
    with pay_redeemer.AttemptBook(path) as book:
        book.note_voucher(CHANNEL, 1_000)
        book.record(channel=CHANNEL, kind="redeem", signature="old-signature",
                    blockhash=BLOCKHASH, last_valid_block_height=100,
                    chain_now=1_000, total=2_000_000)
    rpc = Rpc(path, now=2_000, height=101, status={
        "err": error,
        "confirmationStatus": "confirmed"})

    result = run(tmp_path, rpc)

    # Mutation checked: expiring a confirmed success or error merely because
    # finalized height passed lastValidBlockHeight sends a replacement while
    # the observed signature can still finalize on the rooted fork.
    assert rpc.sent == 0 and result["skipped"] == 1
    with pay_redeemer.AttemptBook(path) as book:
        assert len(book.unresolved(CHANNEL)) == 1


def test_dry_run_never_records_or_sends(tmp_path):
    rpc = Rpc(tmp_path / "redeems.sqlite")
    messages = []

    result = run(tmp_path, rpc, dry_run=True, log=messages.append)

    assert rpc.sent == 0 and result["sent"] == 0 and result["would_send"] == 1
    with pay_redeemer.AttemptBook(tmp_path / "redeems.sqlite") as book:
        assert book.unresolved(CHANNEL) == []
    assert any("dry-run" in message and "nothing was sent" in message for message in messages)


def _worker_store(tmp_path, *, total, redeemed):
    state = pay_voucher.FakeChannelState(
        {CHANNEL: channel_state(
            redeemed=redeemed, status="Closing", close_at=50_000)},
        {ESCROW: OWNER}, program_id=PROGRAM)
    store = pay_voucher.WorkerVoucherStore(tmp_path / "vouchers.sqlite")
    signature = ed25519.Ed25519PrivateKey.from_private_bytes(SIGNER_SEED).sign(
        pay_voucher.voucher_message(PROGRAM, CHANNEL, total))
    store.accept({
        "program": PROGRAM, "channel": CHANNEL, "total": total,
        "signature": base64.b64encode(signature).decode("ascii"),
    }, configured_program=PROGRAM, state=state, now=1_000)
    return store


def test_cooperative_finalize_waits_for_active_worker_reservation(tmp_path):
    store = _worker_store(tmp_path, total=100, redeemed=100)
    store.acquire_reservation(CHANNEL, "payer:active-session")
    rpc = Rpc(
        tmp_path / "redeems.sqlite", now=1_000,
        state=channel_state(redeemed=100, status="Closing", close_at=50_000))
    try:
        result = pay_redeemer.run_redeemer(
            store=store, rpc=rpc, wallet=WALLET, program=PROGRAM, mint=MINT,
            attempts_path=tmp_path / "redeems.sqlite",
            lock_path=tmp_path / "redeem.lock")
        # Mutation checked: finalizing solely because redeemed == latest drops
        # the channel while this session can still deliver its next checkpoint.
        assert rpc.sent == 0 and result["finalized"] == 0
    finally:
        store.release_reservation(CHANNEL, "payer:active-session")
        store.close()


def test_cooperative_finalize_rereads_latest_voucher_before_send(tmp_path):
    store = _worker_store(tmp_path, total=110, redeemed=100)
    old_signature = ed25519.Ed25519PrivateKey.from_private_bytes(SIGNER_SEED).sign(
        pay_voucher.voucher_message(PROGRAM, CHANNEL, 100))
    old = pay_voucher.VoucherRecord(CHANNEL, PROGRAM, 100, old_signature)
    state = channel_state(redeemed=100, status="Closing", close_at=50_000)
    rpc = Rpc(tmp_path / "redeems.sqlite", now=1_000, state=state)
    try:
        with pay_redeemer.AttemptBook(tmp_path / "redeems.sqlite") as attempts:
            finalized = pay_redeemer._cooperative_finalize(
                rpc=rpc, book=attempts, store=store, voucher=old,
                channel=state, owner=OWNER, executor=WALLET, chain_now=1_000,
                dry_run=False, log=lambda _line: None)
        # Mutation checked: trusting the stale all_latest snapshot finalizes
        # despite the store already holding an unredeemed newer voucher.
        assert finalized is False and rpc.sent == 0
    finally:
        store.close()


def test_locally_closing_channel_is_not_selected_for_new_work(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    state = pay_voucher.FakeChannelState(
        {CHANNEL: channel_state(status="Open")}, {ESCROW: OWNER}, program_id=PROGRAM)
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000)
        book.mark_closing(CHANNEL, 2_000_000)
        with pytest.raises(pay_voucher.NoUsableChannel):
            book.begin_session(worker_hex, program=PROGRAM, state=state,
                               required_micro_usdc=1)
    # Mutation checked: removing the local status check in begin_session selects
    # a channel after close was sent but before finalized RPC reflects Closing.


def test_begin_session_persists_on_chain_closing_state(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    state = pay_voucher.FakeChannelState(
        {CHANNEL: channel_state(status="Closing", close_at=12_345)},
        {ESCROW: OWNER}, program_id=PROGRAM)
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000)
        with pytest.raises(pay_voucher.NoUsableChannel):
            book.begin_session(worker_hex, program=PROGRAM, state=state,
                               required_micro_usdc=1)
        # Mutation checked: merely skipping the authoritative Closing account
        # leaves the payer row Open forever and hides it from daemon reclaim.
        row = book.for_channel(CHANNEL)
        assert row.status == "Closing" and row.close_at == 12_345


def test_begin_session_retries_when_channel_closes_between_read_and_update(
        tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    first, second = sorted((CHANNEL, CHANNEL2))
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        for key in (first, second):
            book.put(program=PROGRAM, channel=key, worker_key=worker_hex,
                     cap=5_000_000, max_signed_total=100)

        class InterleavedClosingState:
            def read_channel(self, channel, program):
                assert program == PROGRAM
                if channel == first:
                    # Process B's durable Closing write lands after process A
                    # read its candidate row but before A updates channel_base.
                    book.mark_closing(first, 12_345)
                    return channel_state(redeemed=7)
                assert channel == second
                return channel_state(redeemed=3)

        chosen = book.begin_session(
            worker_hex, program=PROGRAM, state=InterleavedClosingState(),
            required_micro_usdc=1)

        # Mutation checked: dropping status='Open' from the channel_base UPDATE
        # returns the stale first candidate instead of retrying the next one.
        assert chosen.channel == second and chosen.channel_base == 3
        assert book.for_channel(first).status == "Closing"
        assert book.for_channel(first).channel_base == 0


class PayerRpc:
    def __init__(self, path: Path, payer_book: pay_voucher.PayerChannelBook,
                 *, now=2_000_000, closing=False):
        self.path = path
        self.book = payer_book
        self.now = now
        self.sent = 0
        self.node_seed = bytes([71]) * 32
        self.node = pay_chain.b58encode(pay_voucher.public_key_bytes(self.node_seed))
        self.state = pay_voucher.ChannelAccount(
            ESCROW, 4, 7, WALLET["address"], self.node, None, 5_000_000,
            10, 300, "Closing" if closing else "Open",
            now - 1 if closing else 0, self.node, 1)
        self.status = None

    def ensure_cluster(self): return "local"
    def chain_time(self): return self.now
    def read_channel(self, channel, program):
        assert (channel, program) == (CHANNEL, PROGRAM)
        return self.state
    def read_escrow_owner(self, escrow, program):
        assert (escrow, program) == (ESCROW, PROGRAM)
        return OWNER
    def signature_status(self, _signature): return self.status
    def block_height(self, *, commitment): return 10
    def latest_blockhash_info(self, *, commitment):
        return {"blockhash": BLOCKHASH, "lastValidBlockHeight": 100}
    def fee_for_message(self, _message, *, commitment): return 5_000
    def balance(self, address, *, commitment):
        assert address == self.node
        return 1_000_000
    def send_and_confirm(self, transaction, *, last_valid_block_height):
        with sqlite3.connect(self.path) as db:
            assert db.execute(
                "SELECT 1 FROM attempts WHERE signature=?",
                (transaction.signature,)).fetchone() == (1,)
        # Mutation checked: moving payer-book Closing after broadcast exposes
        # this channel to ensure_channel during the ambiguous send window.
        if self.state.status == "Open":
            assert self.book.for_channel(CHANNEL).status == "Closing"
            self.state = pay_voucher.ChannelAccount(
                self.state.escrow, self.state.generation, self.state.seq,
                self.state.worker, self.state.voucher_signer, self.state.delegate,
                self.state.cap, self.state.redeemed, self.state.fee_bps_snapshot,
                "Closing", self.now + 86_400, self.state.rent_payer, self.state.bump)
        self.sent += 1
        self.status = {"err": None, "confirmationStatus": "finalized"}
        return transaction.signature


def test_close_request_uses_node_authority_and_marks_book_before_send(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000, max_signed_total=10)
        rpc = PayerRpc(tmp_path / "redeems.sqlite", book)

        result = pay_redeemer.request_close(
            channels=[CHANNEL], book=book, rpc=rpc, program=PROGRAM, wallet=WALLET,
            node_seed=rpc.node_seed, node_public=pay_voucher.public_key_bytes(rpc.node_seed),
            attempts_path=tmp_path / "redeems.sqlite",
            lock_path=tmp_path / "redeem.lock")

        assert result["sent"] == 1
        assert book.for_channel(CHANNEL).status == "Closing"
        assert book.for_channel(CHANNEL).close_at == rpc.now + 86_400


def test_broker_close_uses_node_when_it_is_voucher_signer(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    broker_wallet = {key: WALLET[key] for key in ("public_key", "address")}
    broker_wallet["custody"] = "broker"
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000, max_signed_total=10)
        rpc = PayerRpc(tmp_path / "redeems.sqlite", book)

        result = pay_redeemer.request_close(
            channels=[CHANNEL], book=book, rpc=rpc, program=PROGRAM,
            wallet=broker_wallet, node_seed=rpc.node_seed,
            node_public=pay_voucher.public_key_bytes(rpc.node_seed),
            attempts_path=tmp_path / "redeems.sqlite",
            lock_path=tmp_path / "redeem.lock")

        # Mutation checked: a blanket broker-custody refusal hides the node's
        # existing voucher-signer authority and sends nothing.
        assert result["sent"] == 1 and rpc.sent == 1


def test_broker_close_owner_authority_explains_wallet_approval_boundary(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    broker_wallet = {key: WALLET[key] for key in ("public_key", "address")}
    broker_wallet["custody"] = "broker"
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000, max_signed_total=10)
        rpc = PayerRpc(tmp_path / "redeems.sqlite", book)
        rpc.state = pay_voucher.ChannelAccount(
            ESCROW, 4, 7, WALLET["address"], SIGNER, None, 5_000_000,
            10, 300, "Open", 0, rpc.node, 1)
        rpc.read_escrow_owner = lambda escrow, program: WALLET["address"]

        with pytest.raises(
                pay_voucher.VoucherError,
                match=r"needs wallet approval \(broker custody; arrives in M4c 2c\)"):
            pay_redeemer.request_close(
                channels=[CHANNEL], book=book, rpc=rpc, program=PROGRAM,
                wallet=broker_wallet, node_seed=rpc.node_seed,
                node_public=pay_voucher.public_key_bytes(rpc.node_seed),
                attempts_path=tmp_path / "redeems.sqlite",
                lock_path=tmp_path / "redeem.lock")

        # Mutation checked: falling through to wallet["seed"] crashes or signs
        # without the broker approval capability that M4c 2c will provide.
        assert rpc.sent == 0


def test_close_request_uses_local_owner_wallet_for_old_delegate_channel(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000, max_signed_total=10)

        class OwnerWalletRpc(PayerRpc):
            def __init__(self):
                super().__init__(tmp_path / "redeems.sqlite", book)
                assert SIGNER not in {self.node, WALLET["address"]}
                self.state = pay_voucher.ChannelAccount(
                    ESCROW, 4, 7, WALLET["address"], SIGNER, None, 5_000_000,
                    10, 300, "Open", 0, self.node, 1)

            def read_escrow_owner(self, escrow, program):
                assert (escrow, program) == (ESCROW, PROGRAM)
                return WALLET["address"]

            def balance(self, address, *, commitment):
                assert address == WALLET["address"] and commitment == "finalized"
                return 1_000_000

        rpc = OwnerWalletRpc()
        result = pay_redeemer.request_close(
            channels=[CHANNEL], book=book, rpc=rpc, program=PROGRAM, wallet=WALLET,
            node_seed=rpc.node_seed, node_public=pay_voucher.public_key_bytes(rpc.node_seed),
            attempts_path=tmp_path / "redeems.sqlite",
            lock_path=tmp_path / "redeem.lock")

        # Mutation checked: requiring voucher_signer == owner rejects the
        # escrow owner's wallet when an old delegate opened the channel.
        assert result["sent"] == 1 and rpc.sent == 1
        assert book.for_channel(CHANNEL).status == "Closing"


def test_close_request_finalized_failure_restores_open_without_daemon(
        tmp_path, monkeypatch):
    monkeypatch.delenv("NAK_PAY_REDEEM", raising=False)
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    attempts_path = tmp_path / "redeems.sqlite"
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000, max_signed_total=10)

        class FailedCloseRpc(PayerRpc):
            def __init__(self):
                super().__init__(attempts_path, book)
                self.channel_reads = 0

            def read_channel(self, channel, program):
                self.channel_reads += 1
                return super().read_channel(channel, program)

            def send_and_confirm(self, transaction, *, last_valid_block_height):
                with sqlite3.connect(self.path) as db:
                    assert db.execute(
                        "SELECT 1 FROM attempts WHERE signature=?",
                        (transaction.signature,)).fetchone() == (1,)
                assert self.book.for_channel(CHANNEL).status == "Closing"
                self.sent += 1
                self.status = {
                    "err": {"InstructionError": [0, "Custom"]},
                    "confirmationStatus": "finalized",
                }
                raise pay_chain.TransactionDefiniteFailure(
                    "transaction failed at finalized")

        rpc = FailedCloseRpc()
        result = pay_redeemer.request_close(
            channels=[CHANNEL], book=book, rpc=rpc, program=PROGRAM, wallet=WALLET,
            node_seed=rpc.node_seed, node_public=pay_voucher.public_key_bytes(rpc.node_seed),
            attempts_path=attempts_path, lock_path=tmp_path / "redeem.lock")

        # Mutation checked: consulting the pre-callback row skips the finalized
        # Open reread and strands this channel Closing until a daemon cycle.
        restored = book.for_channel(CHANNEL)
        assert result["sent"] == 0 and result["skipped"] == 1
        assert rpc.sent == 1 and rpc.channel_reads == 2
        assert restored.status == "Open" and restored.close_at == 0
        with pay_redeemer.AttemptBook(attempts_path) as attempts:
            assert attempts.unresolved(CHANNEL) == []
        assert book.begin_session(
            worker_hex, program=PROGRAM, state=rpc,
            required_micro_usdc=1).channel == CHANNEL


def test_payer_finalize_uses_chain_deadline_and_marks_closed(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000, max_signed_total=10)
        book.mark_closing(CHANNEL, 1_999_999)
        rpc = PayerRpc(tmp_path / "redeems.sqlite", book, closing=True)

        result = pay_redeemer.finalize_payer_channels(
            book=book, rpc=rpc, program=PROGRAM, node_seed=rpc.node_seed,
            all_channels=False, attempts_path=tmp_path / "redeems.sqlite",
            lock_path=tmp_path / "redeem.lock")

        assert result["sent"] == 1
        assert book.for_channel(CHANNEL).status == "Closed"


def test_daemon_finalize_reconciles_locally_open_channel_from_chain(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000, max_signed_total=10)
        rpc = PayerRpc(tmp_path / "redeems.sqlite", book, closing=True)

        result = pay_redeemer.finalize_payer_channels(
            book=book, rpc=rpc, program=PROGRAM, node_seed=rpc.node_seed,
            all_channels=False, attempts_path=tmp_path / "redeems.sqlite",
            lock_path=tmp_path / "redeem.lock")

        # Mutation checked: filtering local Open rows before finalized account
        # reads makes the normal daemon path examine zero channels forever.
        assert result["examined"] == 1 and result["sent"] == 1
        assert book.for_channel(CHANNEL).status == "Closed"


def test_missing_finalized_channel_is_closed_and_next_row_still_finalizes(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        for channel in (CHANNEL, CHANNEL2):
            book.put(program=PROGRAM, channel=channel, worker_key=worker_hex,
                     cap=5_000_000, max_signed_total=10)
            book.mark_closing(channel, 1_999_999)

        class MissingThenClosingRpc(PayerRpc):
            def __init__(self):
                super().__init__(tmp_path / "redeems.sqlite", book, closing=True)
                self.second = pay_voucher.ChannelAccount(
                    ESCROW, 4, 8, WALLET["address"], self.node, None, 5_000_000,
                    10, 300, "Closing", self.now - 1, self.node, 1)

            def read_channel(self, channel, program):
                assert program == PROGRAM
                if channel == CHANNEL:
                    raise pay_voucher.VoucherError("channel account does not exist")
                assert channel == CHANNEL2
                return self.second

        rpc = MissingThenClosingRpc()
        result = pay_redeemer.finalize_payer_channels(
            book=book, rpc=rpc, program=PROGRAM, node_seed=rpc.node_seed,
            all_channels=True, attempts_path=tmp_path / "redeems.sqlite",
            lock_path=tmp_path / "redeem.lock")

        # Mutation checked: requiring a locally successful finalize raises on
        # the first absent account and strands the later elapsed channel.
        assert result["examined"] == 2 and result["sent"] == 1
        assert book.for_channel(CHANNEL).status == "Closed"
        assert book.for_channel(CHANNEL2).status == "Closed"


def test_finalize_row_error_is_logged_and_does_not_abort_next_row(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    messages = []
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        for channel in (CHANNEL, CHANNEL2):
            book.put(program=PROGRAM, channel=channel, worker_key=worker_hex,
                     cap=5_000_000, max_signed_total=10)
            book.mark_closing(channel, 1_999_999)

        class BrokenThenClosingRpc(PayerRpc):
            def __init__(self):
                super().__init__(tmp_path / "redeems.sqlite", book, closing=True)
                self.second = pay_voucher.ChannelAccount(
                    ESCROW, 4, 8, WALLET["address"], self.node, None, 5_000_000,
                    10, 300, "Closing", self.now - 1, self.node, 1)

            def read_channel(self, channel, program):
                if channel == CHANNEL:
                    raise pay_voucher.VoucherError("malformed finalized channel")
                return self.second

        rpc = BrokenThenClosingRpc()
        result = pay_redeemer.finalize_payer_channels(
            book=book, rpc=rpc, program=PROGRAM, node_seed=rpc.node_seed,
            all_channels=True, attempts_path=tmp_path / "redeems.sqlite",
            lock_path=tmp_path / "redeem.lock", log=messages.append)

        assert result["examined"] == 2 and result["sent"] == 1
        assert book.for_channel(CHANNEL).status == "Closing"
        assert book.for_channel(CHANNEL2).status == "Closed"
        assert any(CHANNEL in message and "malformed finalized channel" in message
                   for message in messages)


def test_failed_finalized_close_restores_open_and_selectable_channel(tmp_path):
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    attempts_path = tmp_path / "redeems.sqlite"
    with pay_voucher.PayerChannelBook(tmp_path / "channels.sqlite") as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000, max_signed_total=10)
        book.mark_closing(CHANNEL, 0)
        with pay_redeemer.AttemptBook(attempts_path) as attempts:
            attempts.record(
                channel=CHANNEL, kind="close", signature="failed-close",
                blockhash=BLOCKHASH, last_valid_block_height=100,
                chain_now=1_999_000)
        rpc = PayerRpc(attempts_path, book, closing=False)
        rpc.status = {
            "err": {"InstructionError": [0, "Custom"]},
            "confirmationStatus": "finalized",
        }

        result = pay_redeemer.finalize_payer_channels(
            book=book, rpc=rpc, program=PROGRAM, node_seed=rpc.node_seed,
            attempts_path=attempts_path, lock_path=tmp_path / "redeem.lock")

        # Mutation checked: leaving the local row Closing after the finalized
        # failure keeps begin_session and the next idle-close sweep blind to it.
        restored = book.for_channel(CHANNEL)
        assert result["skipped"] == 1
        assert restored.status == "Open" and restored.close_at == 0
        assert book.begin_session(
            worker_hex, program=PROGRAM, state=rpc,
            required_micro_usdc=1).channel == CHANNEL


def test_nakd_pay_cycle_contains_all_failures(monkeypatch):
    messages = []
    seen = {}
    node = object.__new__(nakd.Node)
    node._key = bytes([81]) * 32
    node.log = messages.append

    def fail_redeem(**kwargs):
        seen["node_seed"] = kwargs.get("node_seed")
        raise SystemExit("redeem boom")

    monkeypatch.setattr(pay_redeemer, "run_redeemer", fail_redeem)
    monkeypatch.setattr(
        pay_redeemer, "finalize_payer_channels",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("finalize boom")))

    node._pay_maintenance_cycle()

    # Mutation checked: making nakd's redeemer rediscover a broker wallet seed
    # blocks the cycle on a correctly seedless broker-custody configuration.
    assert seen["node_seed"] == node._key
    assert any("redeem cycle failed" in row for row in messages)
    assert any("finalize cycle failed" in row for row in messages)


def test_broker_cli_channels_close_and_finalize_without_wallet_seed(
        tmp_path, monkeypatch, capsys):
    home, node_seed = broker_home(tmp_path, monkeypatch)
    assert not (home / ".nakshatra/pay/wallet.ed25519").exists()
    worker_hex = pay_chain.pubkey_bytes(WALLET["address"]).hex()
    with pay_voucher.PayerChannelBook() as book:
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=worker_hex,
                 cap=5_000_000, max_signed_total=10)

    class CliRpc(PayerRpc):
        def __init__(self):
            with pay_voucher.PayerChannelBook() as current:
                super().__init__(
                    home / ".nakshatra/pay/redeems.sqlite", current,
                    closing=False)
            self.node_seed = node_seed
            self.node = pay_chain.b58encode(pay_voucher.public_key_bytes(node_seed))
            self.state = pay_voucher.ChannelAccount(
                ESCROW, 4, 7, WALLET["address"], self.node, None, 5_000_000,
                10, 300, "Open", 0, self.node, 1)

        def send_and_confirm(self, transaction, *, last_valid_block_height):
            with pay_voucher.PayerChannelBook() as current:
                self.book = current
                return super().send_and_confirm(
                    transaction, last_valid_block_height=last_valid_block_height)

    rpc = CliRpc()
    monkeypatch.setattr(
        nak, "_pay_chain_settings", lambda: (pay_chain, rpc, PROGRAM, MINT))

    nak.main(["pay", "channels"])
    channels_output = capsys.readouterr().out
    assert CHANNEL in channels_output and "redeemed 0.00001 USDC" in channels_output

    nak.main(["pay", "close", CHANNEL])
    close_output = capsys.readouterr().out
    assert "close cycle:" in close_output and "sent 1" in close_output

    finalized = {}

    def capture_finalize(**kwargs):
        finalized.update(kwargs)
        return {"examined": 0, "sent": 0, "would_send": 0, "skipped": 0}

    monkeypatch.setattr(pay_redeemer, "finalize_payer_channels", capture_finalize)
    nak.main(["pay", "finalize"])
    finalize_output = capsys.readouterr().out

    # Mutation checked: the former unconditional load_wallet() rejected all
    # three commands before their node-key/read-only permissionless paths.
    assert finalized["node_seed"] == node_seed
    assert "finalize cycle:" in finalize_output


def test_nakd_pay_cycle_contains_payment_import_failure(monkeypatch):
    messages = []
    node = object.__new__(nakd.Node)
    node._key = bytes([81]) * 32
    node.log = messages.append
    real_import = builtins.__import__

    def fail_pay_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "network" and "pay_redeemer" in fromlist:
            raise SystemExit("damaged payment install")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_pay_import)

    node._pay_maintenance_cycle()

    # Mutation checked: leaving the import above containment lets SystemExit
    # escape and permanently kills the maintenance thread.
    assert any("payment modules unavailable" in row
               and "SystemExit" in row for row in messages)


def test_nakd_closing_voucher_is_redeemed_by_next_poll_before_margin(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    close_at = 50_000
    clock = {"now": close_at - pay_voucher.ACCEPT_MARGIN - 2}
    attempts_path = tmp_path / ".nakshatra/pay/redeems.sqlite"
    rpc = Rpc(
        attempts_path, now=clock["now"],
        state=channel_state(status="Closing", close_at=close_at))
    sent_at = []

    class LoopDriver:
        accepted = False

        def is_set(self):
            return rpc.sent == 1

        def wait(self, seconds):
            assert seconds == pay_voucher.CLOSING_POLL_S
            clock["now"] += 1
            total = 2_000_000
            signature = ed25519.Ed25519PrivateKey.from_private_bytes(
                SIGNER_SEED).sign(
                    pay_voucher.voucher_message(PROGRAM, CHANNEL, total))
            with pay_voucher.WorkerVoucherStore() as vouchers:
                record = vouchers.accept({
                    "program": PROGRAM, "channel": CHANNEL, "total": total,
                    "signature": base64.b64encode(signature).decode("ascii"),
                }, configured_program=PROGRAM, state=rpc,
                    now=clock["now"])
                assert clock["now"] == close_at - pay_voucher.ACCEPT_MARGIN - 1
                assert record.close_at == close_at and record.redeemed == 0
            self.accepted = True
            clock["now"] += seconds - 1
            return False

    driver = LoopDriver()
    node = object.__new__(nakd.Node)
    node._stop = driver
    node.log = lambda _line: None

    def cycle():
        if not driver.accepted:
            return
        rpc.now = clock["now"]
        result = pay_redeemer.run_redeemer(
            rpc=rpc, wallet=WALLET, program=PROGRAM, mint=MINT,
            attempts_path=attempts_path,
            lock_path=tmp_path / ".nakshatra/pay/redeem.lock",
            log=lambda _line: None)
        if result["sent"]:
            sent_at.append(clock["now"])

    node._pay_maintenance_cycle = cycle
    node._pay_maintenance_loop()

    # Mutation checked: a fixed 600-second wait, or failing to persist the
    # Closing observation, leaves this voucher beyond the redeem margin.
    assert sent_at == [close_at - pay_voucher.ACCEPT_MARGIN - 2
                       + pay_voucher.CLOSING_POLL_S]
    assert sent_at[0] < close_at - pay_voucher.REDEEM_MARGIN


def test_nakd_pay_loop_survives_unexpected_cycle_exception():
    messages = []
    calls = []

    class TwoCycles:
        def __init__(self):
            self.checks = 0

        def is_set(self):
            self.checks += 1
            return self.checks > 3

        def wait(self, _seconds):
            return False

    node = object.__new__(nakd.Node)
    node._stop = TwoCycles()
    node.log = messages.append
    node._closing_redeem_pending = lambda: True

    def cycle():
        calls.append(1)
        if len(calls) == 1:
            raise SystemExit("unexpected cycle escape")

    node._pay_maintenance_cycle = cycle
    node._pay_maintenance_loop()

    # Mutation checked: omitting the loop's outer BaseException guard stops
    # after the first escaped payment exception instead of running cycle two.
    assert len(calls) == 2
    assert any("maintenance loop recovered" in row for row in messages)
