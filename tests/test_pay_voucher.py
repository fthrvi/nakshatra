from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import base64
import queue
import sqlite3
import stat
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nakshatra_pb2 as pb  # noqa: E402
import paid_admission as admission  # noqa: E402
import worker  # noqa: E402
from network import pay_voucher as pay  # noqa: E402
from network import pay_wallet  # noqa: E402


TOKEN = "A" * 43


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NAK_PAY_MODE", raising=False)
    monkeypatch.delenv("NAKSHATRA_PAY_PROGRAM_ID", raising=False)
    monkeypatch.delenv("NAKSHATRA_PAY_RPC_URL", raising=False)


def _addr(byte: int) -> str:
    return pay.b58encode(bytes([byte]) * 32)


PROGRAM = _addr(1)
CHANNEL = _addr(2)
ESCROW = _addr(3)
WORKER_WALLET = _addr(4)
RENT_PAYER = _addr(5)


def _key(seed_byte: int) -> tuple[bytes, bytes, str, str]:
    seed = bytes([seed_byte]) * 32
    public = ed25519.Ed25519PrivateKey.from_private_bytes(
        seed).public_key().public_bytes_raw()
    return seed, public, public.hex(), pay.b58encode(public)


PAYER_SEED, PAYER_PUBLIC, PAYER_HEX, PAYER_B58 = _key(7)
OWNER_SEED, OWNER_PUBLIC, OWNER_HEX, OWNER_B58 = _key(8)
WRONG_SEED, WRONG_PUBLIC, WRONG_HEX, WRONG_B58 = _key(9)
WORKER_SEED, WORKER_PUBLIC, WORKER_HEX, WORKER_B58 = _key(10)
PAY_WALLET_SEED, PAY_WALLET_PUBLIC, _PAY_WALLET_HEX, PAY_WALLET_B58 = _key(11)


def _install_pay_wallet(home: Path) -> None:
    pay_dir = home / ".nakshatra" / "pay"
    pay_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(pay_dir, 0o700)
    seed_path = pay_dir / "wallet.ed25519"
    seed_path.write_bytes(PAY_WALLET_SEED)
    os.chmod(seed_path, 0o600)
    (pay_dir / "wallet.pub").write_text(PAY_WALLET_B58 + "\n", encoding="ascii")
    (pay_dir / "custody").write_text("file\n", encoding="ascii")


def _channel(*, signer=PAYER_B58, cap=10_000, redeemed=0,
             status="Open", close_at=0) -> pay.ChannelAccount:
    return pay.ChannelAccount(
        escrow=ESCROW, generation=4, seq=9, worker=PAY_WALLET_B58,
        voucher_signer=signer, delegate=None, cap=cap, redeemed=redeemed,
        fee_bps_snapshot=300, status=status, close_at=close_at,
        rent_payer=RENT_PAYER, bump=251,
    )


def _state(*, program=PROGRAM, signer=PAYER_B58, cap=10_000, redeemed=0):
    return pay.FakeChannelState(
        {CHANNEL: _channel(signer=signer, cap=cap, redeemed=redeemed)},
        {ESCROW: OWNER_B58}, program_id=program)


def _voucher(seed=PAYER_SEED, *, program=PROGRAM, channel=CHANNEL,
             total=123, claimed_signer=None) -> dict:
    return pay.parse_wire(pay.voucher_wire(
        program=program, channel=channel, total=total,
        signature=pay.sign_voucher(seed, program, channel, total),
        signer=claimed_signer))


def test_voucher_message_fixed_vector_and_exact_length():
    program = bytes(range(32))
    channel = bytes(range(32, 64))
    message = pay.voucher_message(program, channel, 0x0102030405060708)
    assert len(message) == 88
    assert message == (
        b"NAK_PAY_VOUCHER1" + program + channel
        + bytes.fromhex("0807060504030201"))
    assert message.hex() == (
        "4e414b5f5041595f564f554348455231"
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
        "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
        "0807060504030201")


def test_sign_verify_and_tampered_total():
    signature = pay.sign_voucher(PAYER_SEED, PROGRAM, CHANNEL, 88)
    assert pay.verify_voucher(PAYER_B58, signature, PROGRAM, CHANNEL, 88)
    assert not pay.verify_voucher(WRONG_B58, signature, PROGRAM, CHANNEL, 88)
    assert not pay.verify_voucher(PAYER_B58, signature, PROGRAM, CHANNEL, 89)
    assert not pay.verify_voucher(PAYER_B58, signature, _addr(99), CHANNEL, 88)
    assert not pay.verify_voucher(PAYER_B58, signature, PROGRAM, _addr(98), 88)


def test_worker_store_trusts_only_chain_signers_and_is_monotonic(tmp_path):
    store = pay.WorkerVoucherStore(tmp_path / "worker" / "vouchers.sqlite")
    state = _state()
    accepted = store.accept(
        _voucher(total=123, claimed_signer=WRONG_B58),
        configured_program=PROGRAM, state=state, expected_channel=CHANNEL)
    assert accepted.total == 123  # sender-claimed signer was ignored

    # Equal cumulative vouchers are idempotent.
    assert store.accept(
        _voucher(total=123), configured_program=PROGRAM,
        state=state, expected_channel=CHANNEL).total == 123

    with pytest.raises(pay.VoucherError, match="below"):
        store.accept(_voucher(total=122), configured_program=PROGRAM,
                     state=state, expected_channel=CHANNEL)
    with pytest.raises(pay.VoucherError, match="configured program"):
        store.accept(_voucher(total=124), configured_program=_addr(44),
                     state=state, expected_channel=CHANNEL)
    with pytest.raises(pay.VoucherError, match="another channel"):
        store.accept(_voucher(total=124), configured_program=PROGRAM,
                     state=state, expected_channel=_addr(45))
    with pytest.raises(pay.VoucherError, match="on-chain accepted"):
        store.accept(_voucher(WRONG_SEED, total=124, claimed_signer=WRONG_B58),
                     configured_program=PROGRAM, state=state,
                     expected_channel=CHANNEL)

    tampered = _voucher(total=124)
    tampered["total"] = 125
    with pytest.raises(pay.VoucherError, match="on-chain accepted"):
        store.accept(tampered, configured_program=PROGRAM, state=state,
                     expected_channel=CHANNEL)

    assert stat.S_IMODE((tmp_path / "worker").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "worker" / "vouchers.sqlite").stat().st_mode) == 0o600
    store.close()


def test_escrow_owner_is_an_authoritative_alternate_signer(tmp_path):
    store = pay.WorkerVoucherStore(tmp_path / "vouchers.sqlite")
    accepted = store.accept(
        _voucher(OWNER_SEED, total=77), configured_program=PROGRAM,
        state=_state(), expected_channel=CHANNEL)
    assert accepted.total == 77
    store.close()


def test_floor_is_applied_to_cumulative_credits_not_checkpoint_deltas():
    checkpoints = [1] * 250
    cumulative = 0
    totals = []
    for delta in checkpoints:
        cumulative += delta
        totals.append(pay.credits_to_total(17, cumulative))
    assert totals[98:102] == [17, 18, 18, 18]
    assert totals[-1] == 19
    assert totals[-1] - 17 == sum(checkpoints) // pay.RATE
    assert sum(delta // pay.RATE for delta in checkpoints) == 0


def test_payer_book_keeps_ceiling_while_allowing_safe_resign(tmp_path):
    book = pay.PayerChannelBook(tmp_path / "pay" / "channels.sqlite")
    book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX,
             cap=1000, channel_base=10, max_signed_total=10)
    book.record(CHANNEL, 12)
    # Totals at or below the persisted ceiling were already authorized and
    # may be re-signed after a crash; they do not lower the ceiling.
    book.record(CHANNEL, 11)
    with pytest.raises(pay.VoucherError, match="decrease"):
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX,
                 cap=1000, channel_base=10, max_signed_total=11)
    assert book.for_worker(WORKER_HEX).max_signed_total == 12
    assert stat.S_IMODE((tmp_path / "pay").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "pay" / "channels.sqlite").stat().st_mode) == 0o600
    book.close()


def test_payer_book_migrates_legacy_total_to_signing_ceiling(tmp_path):
    path = tmp_path / "legacy" / "channels.sqlite"
    path.parent.mkdir(mode=0o700)
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE channels(program TEXT NOT NULL, channel TEXT PRIMARY KEY,"
        "worker_key TEXT NOT NULL UNIQUE, cap TEXT NOT NULL,"
        "channel_base TEXT NOT NULL, last_vouched_total TEXT NOT NULL)")
    db.execute(
        "INSERT INTO channels VALUES(?,?,?,?,?,?)",
        (PROGRAM, CHANNEL, WORKER_HEX, "1000", "10", "12"))
    db.commit()
    db.close()

    book = pay.PayerChannelBook(path)
    row = book.for_worker(WORKER_HEX)
    assert (row.channel_base, row.max_signed_total) == (10, 12)
    columns = {entry[1] for entry in book._db.execute(
        "PRAGMA table_info(channels)")}
    assert "max_signed_total" in columns
    assert "last_vouched_total" not in columns
    book.close()


def test_payer_crash_ceiling_is_not_reused_as_the_next_session_base(
        tmp_path, monkeypatch):
    """A persisted-but-unsent authorization may be safely re-signed below its ceiling."""
    monkeypatch.setenv("NAK_PAY_MODE", "chain")
    _install_pay_wallet(Path(os.environ["HOME"]))
    state = _state(cap=1000, redeemed=0)
    payer_path = tmp_path / "crash" / "payer.sqlite"
    book = pay.PayerChannelBook(payer_path)
    book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX,
             cap=1000, channel_base=0, max_signed_total=0)
    # This is the durable write immediately before a signature/send.  The
    # process dies before the worker can persist the resulting voucher.
    book.record(CHANNEL, 4)
    assert book.for_worker(WORKER_HEX).max_signed_total == 4
    book.close()
    vouchers = pay.WorkerVoucherStore(tmp_path / "crash" / "worker.sqlite")
    paid_store = admission.PaidAdmissionStore(tmp_path / "crash" / "paid.sqlite")
    manager = admission.PaidAdmission(
        _Ledger(_claim("crash-restart", layer_end=100)), paid_store,
        WORKER_SEED, model_id="model", layer_start=0, layer_end=100,
        clock=lambda: 100, voucher_store=vouchers, channel_state=state,
        pay_program=PROGRAM)
    assert manager.check_ready()

    restarted = pay.PayerChannelBook(payer_path)
    consumer = pay.ConsumerVoucherManager(
        book=restarted, state=state, program=PROGRAM, private_key=PAYER_SEED,
        plan=[{"worker_keyid": WORKER_HEX, "layer_start": 0,
               "layer_end": 100}],
        max_cost_by_worker={WORKER_HEX: 200},
        session_id="crash-restart", hold_id="hold", model_id="model",
        max_prompt_tokens=1, max_output_tokens=1)
    claim = manager.claim(
        PAYER_HEX, "crash-restart", consumer.first_wire(WORKER_HEX, TOKEN))
    assert manager._chain_session(claim)["channel_base"] == 0

    receipt = _run_step(manager, claim, 1, 1)
    consumer.observe(WORKER_HEX, manager.chain_receipt_wire(claim, receipt))
    consumer.acknowledge_delivery(prompt_tokens=1, generated_tokens=1)
    control = pay.parse_wire(consumer.outgoing(WORKER_HEX))
    assert control["total"] == 2
    assert control["total"] <= 0 + (receipt["cumulative_cost"] // pay.RATE)
    row = restarted.for_worker(WORKER_HEX)
    assert row.channel_base == 0
    assert row.max_signed_total == 4

    with pytest.raises(pay.VoucherError, match="below the on-chain redeemed"):
        restarted.adopt_session_base(CHANNEL, 0, on_chain_redeemed=1)

    consumer.book.close()
    manager.store.close()
    vouchers.close()


def test_payer_adopts_worker_stored_voucher_before_signing_new_work(
        tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_PAY_MODE", "chain")
    _install_pay_wallet(Path(os.environ["HOME"]))
    state = _state(cap=1000, redeemed=0)
    vouchers = pay.WorkerVoucherStore(tmp_path / "adopt" / "worker.sqlite")
    vouchers.accept(
        _voucher(total=2), configured_program=PROGRAM, state=state,
        expected_channel=CHANNEL)
    paid_store = admission.PaidAdmissionStore(tmp_path / "adopt" / "paid.sqlite")
    manager = admission.PaidAdmission(
        _Ledger(_claim("adopt", layer_end=100)), paid_store, WORKER_SEED,
        model_id="model", layer_start=0, layer_end=100, clock=lambda: 100,
        voucher_store=vouchers, channel_state=state, pay_program=PROGRAM)
    assert manager.check_ready()

    book = pay.PayerChannelBook(tmp_path / "adopt" / "payer.sqlite")
    book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX,
             cap=1000, channel_base=0, max_signed_total=4)
    consumer = pay.ConsumerVoucherManager(
        book=book, state=state, program=PROGRAM, private_key=PAYER_SEED,
        plan=[{"worker_keyid": WORKER_HEX, "layer_start": 0,
               "layer_end": 100}],
        max_cost_by_worker={WORKER_HEX: 200}, session_id="adopt",
        hold_id="hold", model_id="model", max_prompt_tokens=1,
        max_output_tokens=1)
    claim = manager.claim(
        PAYER_HEX, "adopt", consumer.first_wire(WORKER_HEX, TOKEN))
    assert manager._chain_session(claim)["channel_base"] == 2

    receipt = _run_step(manager, claim, 1, 1)
    consumer.observe(WORKER_HEX, manager.chain_receipt_wire(claim, receipt))
    assert consumer.channels[WORKER_HEX].channel_base == 2
    consumer.acknowledge_delivery(prompt_tokens=1, generated_tokens=1)
    assert pay.parse_wire(consumer.outgoing(WORKER_HEX))["total"] == 4

    consumer.book.close()
    manager.store.close()
    vouchers.close()


def test_payer_refuses_worker_base_above_ever_signed_ceiling(tmp_path):
    book = pay.PayerChannelBook(tmp_path / "ceiling.sqlite")
    book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX,
             cap=1000, channel_base=0, max_signed_total=3)
    with pytest.raises(pay.VoucherError, match="above the payer signing ceiling"):
        book.adopt_session_base(CHANNEL, 4, on_chain_redeemed=0)
    book.close()


def test_consumer_preflight_requires_chain_authority_and_capacity(tmp_path):
    plan = [{"worker_keyid": WORKER_HEX, "layer_start": 0, "layer_end": 4}]
    book = pay.PayerChannelBook(tmp_path / "authority.sqlite")
    book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX, cap=10)
    with pytest.raises(pay.VoucherError, match="accepted voucher signer"):
        pay.ConsumerVoucherManager(
            book=book, state=_state(cap=10), program=PROGRAM,
            private_key=WRONG_SEED, plan=plan,
            max_cost_by_worker={WORKER_HEX: 100})
    book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX,
             cap=10, channel_base=10, max_signed_total=10)
    with pytest.raises(pay.VoucherError, match="insufficient"):
        pay.ConsumerVoucherManager(
            book=book, state=_state(cap=10, redeemed=10), program=PROGRAM,
            private_key=PAYER_SEED, plan=plan,
            max_cost_by_worker={WORKER_HEX: 100})
    book.close()


def _channel_fixture() -> bytes:
    return b"".join((
        hashlib.sha256(b"account:Channel").digest()[:8],
        pay.b58decode(ESCROW), struct.pack("<Q", 4), struct.pack("<Q", 9),
        pay.b58decode(PAY_WALLET_B58), pay.b58decode(PAYER_B58), b"\0",
        struct.pack("<Q", 10_000), struct.pack("<Q", 123),
        struct.pack("<H", 300), b"\0", struct.pack("<q", 0),
        pay.b58decode(RENT_PAYER), bytes([251]), b"\0" * 32,
    ))


def test_channel_decoder_layout_and_neuron_compatibility():
    fixture = _channel_fixture()
    decoded = pay.decode_channel(fixture)
    assert decoded == _channel(cap=10_000, redeemed=123)

    neuron = Path("/home/prithvi/neuron/python/chain/program_client.py")
    if not neuron.exists():
        pytest.skip("read-only neuron checkout is absent")
    chain_dir = str(neuron.parent)
    solders_dir = Path(
        "/home/prithvi/.cache/nak-solana/venv-chain/lib/python3.12/site-packages")
    added_solders = solders_dir.exists()
    if added_solders:
        sys.path.insert(0, str(solders_dir))
    sys.path.insert(0, chain_dir)
    try:
        spec = importlib.util.spec_from_file_location("neuron_program_client", neuron)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        other = module.decode_channel(fixture)
        other_values = {
            name: (str(value) if name in {
                "escrow", "worker", "voucher_signer", "delegate", "rent_payer"
            } and value is not None else value)
            for name, value in vars(other).items()
        }
        assert vars(decoded) == other_values
        assert pay.voucher_message(PROGRAM, CHANNEL, 123) == module.voucher_message(
            PROGRAM, CHANNEL, 123)
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        pytest.skip(f"neuron decoder dependencies unavailable: {exc}")
    finally:
        sys.path.remove(chain_dir)
        if added_solders:
            sys.path.remove(str(solders_dir))


class _Ledger:
    def __init__(self, claim):
        self.worker_keyid = claim["worker_keyid"]
        self._claim = claim
        self.calls = []

    def probe(self):
        self.calls.append("/spend/ready")
        return {"ready": True}

    def claim(self, token, claim_instance):
        self.calls.append("/spend/claim")
        assert token == TOKEN
        value = dict(self._claim)
        value["claim_instance"] = claim_instance
        value["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
        return value

    def settle(self, receipt, signature):
        self.calls.append("/spend/settle")
        return {"receipt_id": receipt["receipt_id"]}


def _claim(session="session", *, layer_end=4) -> dict:
    max_cost = 41 * layer_end
    value = {
        "v": 1, "hold_id": "hold", "token_sha256": "", "claim_instance": "",
        "payer_keyid": PAYER_HEX, "payer_account_at_authorize": "payer",
        "worker_keyid": WORKER_HEX, "worker_account_at_authorize": "worker",
        "billable": True, "session_id": session, "model_id": "model",
        "billing_mode": "greedy-direct-v1", "layer_start": 0,
        "layer_end": layer_end,
        "price_multiplier": 1, "max_prompt_tokens": 1, "max_output_tokens": 40,
        "max_token_layers": max_cost, "max_cost": max_cost,
        "execution_deadline": 1000, "settlement_deadline": 2000,
        "server_time": 1, "claim_response_sha256": "digest",
    }
    return value


def _chain_admission(tmp_path, monkeypatch, *, session="session", cap=1000,
                     redeemed=0, channel_worker=PAY_WALLET_B58, layers=4):
    monkeypatch.setenv("NAK_PAY_MODE", "chain")
    _install_pay_wallet(Path(os.environ["HOME"]))
    account = _channel(cap=cap, redeemed=redeemed)
    if channel_worker != account.worker:
        account = pay.ChannelAccount(
            account.escrow, account.generation, account.seq, channel_worker,
            account.voucher_signer, account.delegate, account.cap,
            account.redeemed, account.fee_bps_snapshot, account.status,
            account.close_at, account.rent_payer, account.bump)
    state = pay.FakeChannelState(
        {CHANNEL: account}, {ESCROW: OWNER_B58}, program_id=PROGRAM)
    vouchers = pay.WorkerVoucherStore(tmp_path / session / "worker-vouchers.sqlite")
    store = admission.PaidAdmissionStore(tmp_path / session / "paid.sqlite")
    ledger = _Ledger(_claim(session, layer_end=layers))
    manager = admission.PaidAdmission(
        ledger, store, WORKER_SEED, model_id="model",
        layer_start=0, layer_end=layers, clock=lambda: 100,
        voucher_store=vouchers, channel_state=state, pay_program=PROGRAM)
    assert manager.check_ready()
    book = pay.PayerChannelBook(tmp_path / session / "payer-channels.sqlite")
    book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX,
             cap=cap, channel_base=redeemed, max_signed_total=redeemed)
    consumer = pay.ConsumerVoucherManager(
        book=book, state=state, program=PROGRAM, private_key=PAYER_SEED,
        plan=[{"worker_keyid": WORKER_HEX, "layer_start": 0,
               "layer_end": layers}],
        max_cost_by_worker={WORKER_HEX: 41 * layers},
        session_id=session, hold_id="hold", model_id="model",
        max_prompt_tokens=1, max_output_tokens=40)
    claim = manager.claim(
        PAYER_HEX, session,
        consumer.first_wire(WORKER_HEX, TOKEN))
    return manager, consumer, claim, vouchers, ledger


def _run_step(manager, claim, index, n_tokens):
    step_id = f"step-{index}"
    duplicate, prompt_delta, generated_delta = manager.prepare_step(
        claim, step_id, f"digest-{index}", n_tokens)
    assert not duplicate
    manager.require_chain_voucher_before_step(
        claim, prompt_delta, generated_delta)
    receipt = manager.commit_step(
        claim, step_id, f"digest-{index}", prompt_delta, generated_delta, 1)
    return receipt


def test_fake_paid_session_voucher_chain_and_missing_bound(tmp_path, monkeypatch):
    manager, consumer, claim, worker_vouchers, _ledger = _chain_admission(
        tmp_path, monkeypatch)
    receipt_totals = []
    for index in range(1, 18):
        pending = consumer.outgoing(WORKER_HEX)
        if pending:
            manager.accept_chain_control(claim, pending)
            consumer.mark_sent(WORKER_HEX, pending)
        receipt = _run_step(manager, claim, index, 1)
        if receipt is not None:
            consumer.observe(
                WORKER_HEX, manager.chain_receipt_wire(claim, receipt))
            consumer.acknowledge_delivery(prompt_tokens=1, generated_tokens=index)
            receipt_totals.append(
                pay.parse_wire(consumer.outgoing(WORKER_HEX))["total"])
    assert receipt_totals == sorted(receipt_totals)
    assert worker_vouchers.latest(CHANNEL).total == receipt_totals[-2]
    final_wire = consumer.outgoing(WORKER_HEX)
    manager.accept_chain_control(claim, final_wire)
    assert worker_vouchers.latest(CHANNEL).total == receipt_totals[-1]

    # Silence is bounded in credits even while every rounded voucher total is
    # still zero: one 8-token checkpoint plus RATE-1 rounding credits.
    stopped, consumer2, claim2, store2, _ledger2 = _chain_admission(
        tmp_path, monkeypatch, session="stopped")
    first = _run_step(stopped, claim2, 1, 1)
    assert first is not None
    stopped.chain_receipt_wire(claim2, first)
    completed_without_voucher = 0
    for index in range(2, 60):
        try:
            _run_step(stopped, claim2, index, 1)
        except admission.PaidAdmissionError as exc:
            assert str(exc) == "voucher-missing"
            break
        completed_without_voucher += 1
    meter = stopped.store.meter(PAYER_HEX, "stopped")
    delivered = (
        int(meter["prompt_tokens"]) + int(meter["generated_tokens"])) * 4
    assert delivered <= 8 * 4 + (pay.RATE - 1)

    consumer.book.close()
    manager.store.close()
    worker_vouchers.close()
    stopped.store.close()
    consumer2.book.close()
    store2.close()


def test_worker_reserves_one_session_per_channel_and_advances_base(
        tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_PAY_MODE", "chain")
    _install_pay_wallet(Path(os.environ["HOME"]))
    state = _state(cap=1000, redeemed=10)
    vouchers = pay.WorkerVoucherStore(tmp_path / "shared" / "vouchers.sqlite")

    def manager_for(session):
        store = admission.PaidAdmissionStore(tmp_path / session / "paid.sqlite")
        manager = admission.PaidAdmission(
            _Ledger(_claim(session, layer_end=100)), store, WORKER_SEED,
            model_id="model", layer_start=0, layer_end=100, clock=lambda: 100,
            voucher_store=vouchers, channel_state=state, pay_program=PROGRAM)
        assert manager.check_ready()
        return manager

    def init_for(session, base, ceiling=None):
        return pay.init_wire(
            TOKEN, program=PROGRAM, channel=CHANNEL, channel_base=base,
            hold_id="hold", session_id=session, model_id="model",
            max_prompt_tokens=1, max_output_tokens=40,
            max_signed_total=base if ceiling is None else ceiling)

    first = manager_for("reservation-a")
    claim_a = first.claim(PAYER_HEX, "reservation-a", init_for("reservation-a", 10))
    second = manager_for("reservation-b")
    with pytest.raises(admission.PaidAdmissionError, match="channel busy"):
        second.claim(PAYER_HEX, "reservation-b", init_for("reservation-b", 10))
    with pytest.raises(pay.VoucherError, match="not held by this session"):
        vouchers.accept(
            _voucher(total=11), configured_program=PROGRAM, state=state,
            expected_channel=CHANNEL,
            reservation_id=f"{PAYER_HEX}:reservation-b")
    assert vouchers.latest(CHANNEL) is None

    # One 100-layer prefill+decode checkpoint is 200 credits, so A's final
    # cumulative voucher advances the channel base from 10 to 12.
    receipt = _run_step(first, claim_a, 1, 1)
    assert receipt is not None and receipt["cumulative_cost"] == 200
    first.chain_receipt_wire(claim_a, receipt)
    first.accept_chain_control(
        claim_a, pay.voucher_wire(
            program=PROGRAM, channel=CHANNEL, total=12,
            signature=pay.sign_voucher(PAYER_SEED, PROGRAM, CHANNEL, 12)))
    first.release_chain_session(claim_a)

    with pytest.raises(admission.PaidAdmissionError,
                       match="exceeds the payer signing ceiling"):
        second.claim(PAYER_HEX, "reservation-b", init_for("reservation-b", 10))
    claim_b = second.claim(
        PAYER_HEX, "reservation-b", init_for("reservation-b", 10, 12))
    assert second._chain_session(claim_b)["channel_base"] == 12

    second.release_chain_session(claim_b)
    first.store.close()
    second.store.close()
    vouchers.close()


def test_worker_reservation_identity_includes_boot_id(tmp_path, monkeypatch):
    monkeypatch.setattr(pay, "_process_start_time", lambda _pid: "12345")
    boot = {"value": "11111111-1111-1111-1111-111111111111"}
    monkeypatch.setattr(pay, "_boot_id", lambda: boot["value"])
    store = pay.WorkerVoucherStore(tmp_path / "boot" / "vouchers.sqlite")
    pid = os.getpid()

    store._db.execute(
        "INSERT INTO reservations(channel,reservation_id,pid,process_start,"
        "boot_id,acquired_at) VALUES(?,?,?,?,?,?)",
        (CHANNEL, "old-boot", pid, "12345",
         "00000000-0000-0000-0000-000000000000", 1))
    store.acquire_reservation(CHANNEL, "new-boot")
    row = store._db.execute(
        "SELECT * FROM reservations WHERE channel=?", (CHANNEL,)).fetchone()
    assert row["reservation_id"] == "new-boot"
    assert row["boot_id"] == boot["value"]

    with pytest.raises(pay.VoucherError, match="channel busy"):
        store.acquire_reservation(CHANNEL, "same-boot-contender")

    def unreadable_boot():
        raise pay.VoucherError("cannot verify voucher reservation boot")

    monkeypatch.setattr(pay, "_boot_id", unreadable_boot)
    with pytest.raises(pay.VoucherError, match="cannot verify.*boot"):
        store.acquire_reservation(CHANNEL, "unknown-boot-contender")
    assert store._db.execute(
        "SELECT reservation_id FROM reservations WHERE channel=?",
        (CHANNEL,)).fetchone()[0] == "new-boot"
    store.close()


def test_voucher_silence_bound_counts_credits_before_rounding(
        tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_PAY_MODE", "chain")
    _install_pay_wallet(Path(os.environ["HOME"]))
    state = _state(cap=1000, redeemed=10)
    vouchers = pay.WorkerVoucherStore(tmp_path / "rounding" / "vouchers.sqlite")
    vouchers.accept(
        _voucher(total=10), configured_program=PROGRAM, state=state,
        expected_channel=CHANNEL)
    store = admission.PaidAdmissionStore(tmp_path / "rounding" / "paid.sqlite")
    manager = admission.PaidAdmission(
        _Ledger(_claim("rounding", layer_end=1)), store, WORKER_SEED,
        model_id="model", layer_start=0, layer_end=1, clock=lambda: 100,
        voucher_store=vouchers, channel_state=state, pay_program=PROGRAM)
    assert manager.check_ready()
    claim = manager.claim(
        PAYER_HEX, "rounding", pay.init_wire(
            TOKEN, program=PROGRAM, channel=CHANNEL, channel_base=10,
            hold_id="hold", session_id="rounding", model_id="model",
            max_prompt_tokens=1, max_output_tokens=200))

    stopped = False
    for index in range(1, 150):
        try:
            receipt = _run_step(manager, claim, index, 1)
        except admission.PaidAdmissionError as exc:
            assert str(exc) == "voucher-missing"
            stopped = True
            break
        if receipt is not None:
            wire = manager.chain_receipt_wire(claim, receipt)
            assert pay.parse_wire(wire)["receipt"]["cumulative_cost"] <= 107
    assert stopped
    meter = manager.store.meter(PAYER_HEX, "rounding")
    delivered = int(meter["prompt_tokens"]) + int(meter["generated_tokens"])
    assert delivered <= 8 + (pay.RATE - 1)

    manager.release_chain_session(claim)
    store.close()
    vouchers.close()


def test_chain_mode_never_calls_credit_moving_ledger_routes(tmp_path, monkeypatch):
    manager, consumer, claim, vouchers, ledger = _chain_admission(
        tmp_path, monkeypatch, session="no-double-charge")
    receipt = _run_step(manager, claim, 1, 1)
    consumer.observe(WORKER_HEX, manager.chain_receipt_wire(claim, receipt))
    consumer.acknowledge_delivery(prompt_tokens=1, generated_tokens=1)
    manager.accept_chain_control(claim, consumer.outgoing(WORKER_HEX))
    manager.flush_outbox()

    assert not ({"/spend/claim", "/spend/settle"} & set(ledger.calls))
    states = [row[0] for row in manager.store._db.execute(
        "SELECT state FROM outbox ORDER BY sequence")]
    assert states == ["chain-vouched"]
    consumer.book.close()
    manager.store.close()
    vouchers.close()


def test_chain_admission_refuses_channel_for_another_payee(tmp_path, monkeypatch):
    with pytest.raises(admission.PaidAdmissionError, match="worker pay wallet"):
        _chain_admission(
            tmp_path, monkeypatch, session="wrong-payee",
            channel_worker=WORKER_WALLET)


def test_worker_derives_channel_base_and_refuses_sender_value(tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_PAY_MODE", "chain")
    _install_pay_wallet(Path(os.environ["HOME"]))
    state = _state(cap=1000)
    vouchers = pay.WorkerVoucherStore(tmp_path / "base" / "vouchers.sqlite")
    store = admission.PaidAdmissionStore(tmp_path / "base" / "paid.sqlite")
    manager = admission.PaidAdmission(
        _Ledger(_claim("base")), store, WORKER_SEED, model_id="model",
        layer_start=0, layer_end=4, clock=lambda: 100,
        voucher_store=vouchers, channel_state=state, pay_program=PROGRAM)
    assert manager.check_ready()
    forged = pay.init_wire(
        TOKEN, program=PROGRAM, channel=CHANNEL, channel_base=1000,
        hold_id="hold", session_id="base", model_id="model",
        max_prompt_tokens=1, max_output_tokens=40)
    with pytest.raises(admission.PaidAdmissionError,
                       match="exceeds the worker-derived base"):
        manager.claim(PAYER_HEX, "base", forged)
    store.close()
    vouchers.close()


def test_channel_base_at_cap_cannot_admit_ceil_max_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_PAY_MODE", "chain")
    _install_pay_wallet(Path(os.environ["HOME"]))
    state = _state(cap=1000, redeemed=1000)
    vouchers = pay.WorkerVoucherStore(tmp_path / "full" / "vouchers.sqlite")
    store = admission.PaidAdmissionStore(tmp_path / "full" / "paid.sqlite")
    manager = admission.PaidAdmission(
        _Ledger(_claim("full")), store, WORKER_SEED, model_id="model",
        layer_start=0, layer_end=4, clock=lambda: 100,
        voucher_store=vouchers, channel_state=state, pay_program=PROGRAM)
    assert manager.check_ready()
    init = pay.init_wire(
        TOKEN, program=PROGRAM, channel=CHANNEL, channel_base=1000,
        hold_id="hold", session_id="full", model_id="model",
        max_prompt_tokens=1, max_output_tokens=40)
    with pytest.raises(admission.PaidAdmissionError, match="insufficient"):
        manager.claim(PAYER_HEX, "full", init)
    store.close()
    vouchers.close()


def test_closing_channel_refused_at_admission_and_voucher_accept(
        tmp_path, monkeypatch):
    monkeypatch.setenv("NAK_PAY_MODE", "chain")
    _install_pay_wallet(Path(os.environ["HOME"]))
    closing = _channel(status="Closing", close_at=99)
    state = pay.FakeChannelState(
        {CHANNEL: closing}, {ESCROW: OWNER_B58}, program_id=PROGRAM)
    vouchers = pay.WorkerVoucherStore(tmp_path / "closing" / "vouchers.sqlite")
    store = admission.PaidAdmissionStore(tmp_path / "closing" / "paid.sqlite")
    manager = admission.PaidAdmission(
        _Ledger(_claim("closing")), store, WORKER_SEED, model_id="model",
        layer_start=0, layer_end=4, clock=lambda: 100,
        voucher_store=vouchers, channel_state=state, pay_program=PROGRAM)
    assert manager.check_ready()
    init = pay.init_wire(
        TOKEN, program=PROGRAM, channel=CHANNEL, channel_base=0,
        hold_id="hold", session_id="closing", model_id="model",
        max_prompt_tokens=1, max_output_tokens=40)
    with pytest.raises(admission.PaidAdmissionError, match="not open"):
        manager.claim(PAYER_HEX, "closing", init)
    with pytest.raises(pay.VoucherError, match="not open"):
        vouchers.accept(
            _voucher(total=1), configured_program=PROGRAM, state=state,
            expected_channel=CHANNEL, now=100)
    store.close()
    vouchers.close()


def test_voucher_refuses_channel_too_near_close_deadline(tmp_path):
    state = pay.FakeChannelState(
        {CHANNEL: _channel(status="Open", close_at=115)},
        {ESCROW: OWNER_B58}, program_id=PROGRAM)
    vouchers = pay.WorkerVoucherStore(tmp_path / "deadline" / "vouchers.sqlite")
    with pytest.raises(pay.VoucherError, match="close deadline"):
        vouchers.accept(
            _voucher(total=1), configured_program=PROGRAM, state=state,
            expected_channel=CHANNEL, now=100)

    state.channels[CHANNEL] = _channel(status="Open", close_at=116)
    assert vouchers.accept(
        _voucher(total=1), configured_program=PROGRAM, state=state,
        expected_channel=CHANNEL, now=100).total == 1
    vouchers.close()


def test_mid_session_close_stops_after_current_checkpoint(
        tmp_path, monkeypatch):
    manager, consumer, claim, vouchers, _ledger = _chain_admission(
        tmp_path, monkeypatch, session="mid-close")
    first = _run_step(manager, claim, 1, 1)
    consumer.observe(WORKER_HEX, manager.chain_receipt_wire(claim, first))
    consumer.acknowledge_delivery(prompt_tokens=1, generated_tokens=1)
    manager.accept_chain_control(claim, consumer.outgoing(WORKER_HEX))

    manager.channel_state.channels[CHANNEL] = _channel(
        status="Closing", close_at=200)
    completed = 0
    for index in range(2, 20):
        try:
            receipt = _run_step(manager, claim, index, 1)
        except admission.PaidAdmissionError as exc:
            assert str(exc) == "channel-closing"
            break
        completed += 1
        if receipt is not None:
            manager.chain_receipt_wire(claim, receipt)
    else:
        pytest.fail("worker did not stop after the channel close request")

    assert completed <= 8
    final = manager.finalize(claim, "channel-closing")
    assert final is not None
    assert final["termination"] == "channel-closing"
    consumer.book.close()
    manager.store.close()
    vouchers.close()


class _Context:
    def __init__(self):
        self._code = None
        self._details = ""

    def set_code(self, value):
        self._code = value

    def set_details(self, value):
        self._details = value


class _Daemon:
    def __init__(self):
        self._owner = None

    def info(self):
        return {"n_embd": 4, "n_layers": 4, "gpu_offload_status": {}}

    @property
    def current_owner(self):
        return self._owner

    def call(self, _cmd, _n_tokens, _payload, start_pos=0, flags=0):
        del start_pos, flags
        return 0, struct.pack("<Ii", 0, 42)

    def acquire_session(self, principal, stale_after_s=None):
        del stale_after_s
        if self._owner is None or self._owner == principal:
            self._owner = principal
            return True
        return False

    def release_session(self, principal):
        if self._owner == principal:
            self._owner = None


def test_final_partial_receipt_is_vouched_before_stream_close(tmp_path, monkeypatch):
    manager, consumer, _claim_value, vouchers, _ledger = _chain_admission(
        tmp_path, monkeypatch, session="final-partial")
    servicer = worker.WorkerServicer(
        _Daemon(), "solo", 0, 4, "model", paid_admission=manager,
        refuse_unregistered_peers=False, refuse_unpinned_peers=False)
    servicer._check_grpc_auth = lambda *_args, **_kwargs: (
        worker.AuthenticatedKeyId(PAYER_HEX, worker._wgrpcauth.ROLE_CONSUMER))
    requests = queue.Queue()
    sentinel = object()

    def request_iter():
        while True:
            item = requests.get(timeout=5)
            if item is sentinel:
                return
            yield item

    responses = servicer.Inference(request_iter(), _Context())
    init = pb.InferenceStep(
        session_id="final-partial", step_id="step-1", prefix_length=0,
        spend_claim_token=consumer.first_wire(WORKER_HEX, TOKEN))
    init.token_ids.ids.append(1)
    requests.put(init)
    first = next(responses)
    consumer.observe(WORKER_HEX, first.spend_claim_token)
    consumer.acknowledge_delivery(prompt_tokens=1, generated_tokens=1)

    for index in range(2, 9):
        step = pb.InferenceStep(
            session_id="final-partial", step_id=f"step-{index}",
            prefix_length=index - 1,
            spend_claim_token=(consumer.outgoing(WORKER_HEX) if index == 2 else ""))
        step.token_ids.ids.append(1)
        requests.put(step)
        response = next(responses)
        if index == 2:
            consumer.mark_sent(WORKER_HEX, step.spend_claim_token)
        assert not response.spend_claim_token

    requests.put(pb.InferenceStep(
        session_id="final-partial", step_id="voucher-final-request",
        prefix_length=8, error=pay.CONTROL_FINAL_REQUEST))
    final_response = next(responses)
    assert final_response.error == pay.CONTROL_FINAL_RECEIPT
    final_control = pay.parse_wire(final_response.spend_claim_token)
    assert final_control["receipt"]["cumulative_generated_tokens"] == 8
    consumer.observe(WORKER_HEX, final_response.spend_claim_token)
    consumer.acknowledge_delivery(prompt_tokens=1, generated_tokens=8)
    final_voucher = consumer.outgoing(WORKER_HEX)
    requests.put(pb.InferenceStep(
        session_id="final-partial", step_id="voucher-final-ack",
        prefix_length=8, spend_claim_token=final_voucher,
        error=pay.CONTROL_REQUEST))
    assert next(responses).error == pay.CONTROL_OK
    consumer.mark_sent(WORKER_HEX, final_voucher)
    requests.put(sentinel)
    with pytest.raises(StopIteration):
        next(responses)
    assert vouchers._db.execute(
        "SELECT count(*) FROM reservations").fetchone()[0] == 0

    latest = vouchers.latest(CHANNEL)
    assert latest is not None
    assert latest.total == pay.credits_to_total(
        0, final_control["receipt"]["cumulative_cost"])
    consumer.book.close()
    manager.store.close()
    vouchers.close()


def test_worker_error_releases_channel_reservation(tmp_path, monkeypatch):
    manager, consumer, _claim_value, vouchers, _ledger = _chain_admission(
        tmp_path, monkeypatch, session="release-error")
    servicer = worker.WorkerServicer(
        _Daemon(), "solo", 0, 4, "model", paid_admission=manager,
        refuse_unregistered_peers=False, refuse_unpinned_peers=False)
    servicer._check_grpc_auth = lambda *_args, **_kwargs: (
        worker.AuthenticatedKeyId(PAYER_HEX, worker._wgrpcauth.ROLE_CONSUMER))
    first = pb.InferenceStep(
        session_id="release-error", step_id="step-1", prefix_length=0,
        spend_claim_token=consumer.first_wire(WORKER_HEX, TOKEN))
    first.token_ids.ids.append(1)
    malformed = pb.InferenceStep(
        session_id="release-error", step_id="step-2", prefix_length=1,
        spend_claim_token="not-a-voucher")
    malformed.token_ids.ids.append(1)

    responses = list(servicer.Inference(iter((first, malformed)), _Context()))
    assert len(responses) == 1
    assert vouchers._db.execute(
        "SELECT count(*) FROM reservations").fetchone()[0] == 0

    consumer.book.close()
    manager.store.close()
    vouchers.close()


def test_missing_final_voucher_exposes_only_last_partial_checkpoint(
        tmp_path, monkeypatch):
    manager, consumer, claim, vouchers, _ledger = _chain_admission(
        tmp_path, monkeypatch, session="missing-final", layers=16)
    first = _run_step(manager, claim, 1, 1)
    consumer.observe(WORKER_HEX, manager.chain_receipt_wire(claim, first))
    consumer.acknowledge_delivery(prompt_tokens=1, generated_tokens=1)
    first_voucher = consumer.outgoing(WORKER_HEX)
    manager.accept_chain_control(claim, first_voucher)
    consumer.mark_sent(WORKER_HEX, first_voucher)
    for index in range(2, 9):
        assert _run_step(manager, claim, index, 1) is None

    final_wire = manager.final_chain_receipt_wire(claim)
    final_receipt = pay.parse_wire(final_wire)["receipt"]
    covered = vouchers.latest(CHANNEL)
    assert covered is not None
    unpaid_credits = (
        final_receipt["cumulative_cost"] - first["cumulative_cost"])
    assert unpaid_credits == 7 * 16
    assert unpaid_credits < 8 * 16
    with pytest.raises(admission.PaidAdmissionError, match="voucher-missing"):
        manager.require_chain_final_voucher(claim)
    assert worker._PAY_FINAL_VOUCHER_WAIT_S == 15.0
    consumer.book.close()
    manager.store.close()
    vouchers.close()


DEVNET_GENESIS = "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG"
MAINNET_GENESIS = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"


def _rpc_server(genesis, account_data):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request["method"])
            if request["method"] == "getGenesisHash":
                result = genesis
            else:
                result = {"context": {"apiVersion": "2.0.15", "slot": 341197053},
                          "value": {
                    "owner": PROGRAM,
                    "data": [base64.b64encode(account_data).decode("ascii"), "base64"],
                    "lamports": 2039280,
                    "executable": False,
                    "rentEpoch": 18446744073709551615,
                    "space": len(account_data),
                }}
            body = json.dumps({"jsonrpc": "2.0", "id": request["id"],
                               "result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, calls


@pytest.mark.parametrize(
    ("genesis", "accepted"),
    [(DEVNET_GENESIS, True), (MAINNET_GENESIS, False), (_addr(77), True)],
)
def test_rpc_checks_genesis_before_account_reads(genesis, accepted):
    server, calls = _rpc_server(genesis, _channel_fixture())
    try:
        state = pay.JsonRpcChannelState(
            f"http://127.0.0.1:{server.server_port}", timeout=2)
        if accepted:
            assert state.read_channel(CHANNEL, PROGRAM).cap == 10_000
        else:
            with pytest.raises(pay.VoucherError, match="mainnet"):
                state.read_channel(CHANNEL, PROGRAM)
        assert calls[0] == "getGenesisHash"
    finally:
        server.shutdown()
        server.server_close()


def test_malformed_rpc_genesis_fails_closed():
    server, calls = _rpc_server({"not": "a genesis hash"}, _channel_fixture())
    try:
        state = pay.JsonRpcChannelState(
            f"http://127.0.0.1:{server.server_port}", timeout=2)
        with pytest.raises(pay.VoucherError, match="malformed getGenesisHash"):
            state.read_channel(CHANNEL, PROGRAM)
        assert calls == ["getGenesisHash"]
    finally:
        server.shutdown()
        server.server_close()


def test_rpc_rejects_legacy_value_only_account_wrapper():
    class LegacyShape(pay.JsonRpcChannelState):
        def _ensure_cluster(self):
            pass

        def _rpc(self, method, params):
            del method, params
            return {"value": None}

    with pytest.raises(pay.VoucherError, match="malformed account result"):
        LegacyShape("http://127.0.0.1:1").read_channel(CHANNEL, PROGRAM)


def test_unset_mode_preserves_legacy_wire_bytes(tmp_path, monkeypatch):
    monkeypatch.delenv("NAK_PAY_MODE", raising=False)
    omitted = pb.InferenceStep(
        session_id="s", step_id="step-1", prefix_length=3,
        token_ids=pb.TokenIds(ids=[1]))
    explicit_empty = pb.InferenceStep(
        session_id="s", step_id="step-1", prefix_length=3,
        token_ids=pb.TokenIds(ids=[1]), spend_claim_token="")
    assert omitted.SerializeToString() == explicit_empty.SerializeToString()

    store = admission.PaidAdmissionStore(tmp_path / "legacy.sqlite")
    manager = admission.PaidAdmission(
        _Ledger(_claim()), store, WORKER_SEED, model_id="model",
        layer_start=0, layer_end=4, clock=lambda: 100)
    assert manager.chain_mode is False
    assert manager.check_ready()
    claim = manager.claim(PAYER_HEX, "session", TOKEN)
    _duplicate, prompt_delta, generated_delta = manager.prepare_step(
        claim, "step-1", "digest-1", 1)
    receipt = manager.commit_step(
        claim, "step-1", "digest-1", prompt_delta, generated_delta, 1)
    expected = {
        "v": 2, "kind": "nakshatra-worker-usage", "receipt_id": "",
        "hold_id": "hold", "claim_response_sha256": "digest",
        "payer_keyid": PAYER_HEX, "payer_account_at_authorize": "payer",
        "worker_keyid": WORKER_HEX, "worker_account_at_authorize": "worker",
        "billable": True, "session_id": "session", "model_id": "model",
        "billing_mode": "greedy-direct-v1", "layer_start": 0, "layer_end": 4,
        "sequence": 1, "cumulative_prompt_tokens": 1,
        "cumulative_generated_tokens": 1, "cumulative_token_layers": 8,
        "cumulative_cost": 8, "work_completed_at": 100, "final": False,
        "termination": "checkpoint", "output_sha256": "",
    }
    without_id = {key: value for key, value in expected.items()
                  if key != "receipt_id"}
    expected["receipt_id"] = "sha256:" + hashlib.sha256(
        admission.RECEIPT_DOMAIN + json.dumps(
            without_id, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode()).hexdigest()
    assert receipt == expected
    assert manager.chain_receipt_wire(claim, receipt) == ""
    store.close()


def test_consumer_refuses_receipt_claiming_more_tokens_than_received(
        tmp_path, monkeypatch):
    manager, consumer, claim, worker_vouchers, _ledger = _chain_admission(
        tmp_path, monkeypatch, session="overclaim")
    receipt = None
    for index in range(1, 18):
        receipt = _run_step(manager, claim, index, 1)
        if receipt is not None:
            break
    assert receipt is not None
    consumer.observe(WORKER_HEX, manager.chain_receipt_wire(claim, receipt))
    generated = int(receipt["cumulative_generated_tokens"])
    with pytest.raises(pay.VoucherError, match="has not received"):
        consumer.acknowledge_delivery(prompt_tokens=1, generated_tokens=generated - 1)
    assert not consumer.outgoing(WORKER_HEX)
    consumer.book.close()
    manager.store.close()
    worker_vouchers.close()


@pytest.mark.parametrize("url", [
    "http://api.devnet.solana.com",
    "http://10.42.0.1:8899",
    "ftp://127.0.0.1:8899",
    "https://",
])
def test_rpc_refuses_plaintext_remote_and_odd_schemes(url):
    with pytest.raises(pay.VoucherError, match="https://"):
        pay.JsonRpcChannelState(url)


@pytest.mark.parametrize("url", [
    "https://api.devnet.solana.com",
    "http://127.0.0.1:8899",
    "http://localhost:8899",
    "http://[::1]:8899",
])
def test_rpc_accepts_tls_or_loopback(url):
    assert pay.JsonRpcChannelState(url).rpc_url == url


def test_rpc_oversized_response_fails_closed():
    class Endless:
        def __init__(self):
            self.asked = None

        def read(self, size=-1):
            self.asked = size
            return b" " * (size if size and size > 0 else 10 * pay.RPC_RESPONSE_MAX)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    body = Endless()
    state = pay.JsonRpcChannelState(
        "http://127.0.0.1:1", opener=lambda _request, timeout: body)
    with pytest.raises(pay.VoucherError, match="too large"):
        state.read_channel(CHANNEL, PROGRAM)
    assert body.asked == pay.RPC_RESPONSE_MAX + 1


def test_first_step_prefill_longer_than_a_checkpoint_is_admitted(tmp_path, monkeypatch):
    manager, consumer, claim, worker_vouchers, _ledger = _chain_admission(
        tmp_path, monkeypatch, session="prefill")
    # A 100-token prompt on a 4-layer stage is 404 credits before any receipt can exist.
    manager.require_chain_voucher_before_step(claim, 100, 1)
    consumer.book.close()
    manager.store.close()
    worker_vouchers.close()


def test_rpc_refuses_redirects():
    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(302)
            self.send_header("Location", "http://198.51.100.7:8899/")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        state = pay.JsonRpcChannelState(
            f"http://127.0.0.1:{server.server_port}", timeout=2)
        with pytest.raises(pay.VoucherError, match="redirects are refused"):
            state.read_channel(CHANNEL, PROGRAM)
    finally:
        server.shutdown()
        server.server_close()
