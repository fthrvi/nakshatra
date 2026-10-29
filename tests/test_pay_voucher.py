from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import struct
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nakshatra_pb2 as pb  # noqa: E402
import paid_admission as admission  # noqa: E402
from network import pay_voucher as pay  # noqa: E402


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


def _channel(*, signer=PAYER_B58, cap=10_000, redeemed=0) -> pay.ChannelAccount:
    return pay.ChannelAccount(
        escrow=ESCROW, generation=4, seq=9, worker=WORKER_WALLET,
        voucher_signer=signer, delegate=None, cap=cap, redeemed=redeemed,
        fee_bps_snapshot=300, status="Open", close_at=0,
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


def test_payer_book_never_lowers_a_vouched_total(tmp_path):
    book = pay.PayerChannelBook(tmp_path / "pay" / "channels.sqlite")
    book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX,
             cap=1000, channel_base=10, last_vouched_total=10)
    book.record(CHANNEL, 12)
    with pytest.raises(pay.VoucherError, match="decrease"):
        book.record(CHANNEL, 11)
    with pytest.raises(pay.VoucherError, match="decrease"):
        book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX,
                 cap=1000, channel_base=10, last_vouched_total=11)
    assert book.for_worker(WORKER_HEX).last_vouched_total == 12
    assert stat.S_IMODE((tmp_path / "pay").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "pay" / "channels.sqlite").stat().st_mode) == 0o600
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
        pay.b58decode(WORKER_WALLET), pay.b58decode(PAYER_B58), b"\0",
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

    def probe(self):
        return {"ready": True}

    def claim(self, token, claim_instance):
        assert token == TOKEN
        value = dict(self._claim)
        value["claim_instance"] = claim_instance
        value["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
        return value

    def settle(self, receipt, signature):
        return {"receipt_id": receipt["receipt_id"]}


def _claim(session="session") -> dict:
    value = {
        "v": 1, "hold_id": "hold", "token_sha256": "", "claim_instance": "",
        "payer_keyid": PAYER_HEX, "payer_account_at_authorize": "payer",
        "worker_keyid": WORKER_HEX, "worker_account_at_authorize": "worker",
        "billable": True, "session_id": session, "model_id": "model",
        "billing_mode": "greedy-direct-v1", "layer_start": 0, "layer_end": 4,
        "price_multiplier": 1, "max_prompt_tokens": 1, "max_output_tokens": 40,
        "max_token_layers": 164, "max_cost": 164,
        "execution_deadline": 1000, "settlement_deadline": 2000,
        "server_time": 1, "claim_response_sha256": "digest",
    }
    return value


def _chain_admission(tmp_path, monkeypatch, *, session="session"):
    monkeypatch.setenv("NAK_PAY_MODE", "chain")
    state = _state(cap=1000)
    vouchers = pay.WorkerVoucherStore(tmp_path / session / "worker-vouchers.sqlite")
    store = admission.PaidAdmissionStore(tmp_path / session / "paid.sqlite")
    manager = admission.PaidAdmission(
        _Ledger(_claim(session)), store, WORKER_SEED, model_id="model",
        layer_start=0, layer_end=4, clock=lambda: 100,
        voucher_store=vouchers, channel_state=state, pay_program=PROGRAM)
    assert manager.check_ready()
    book = pay.PayerChannelBook(tmp_path / session / "payer-channels.sqlite")
    book.put(program=PROGRAM, channel=CHANNEL, worker_key=WORKER_HEX, cap=1000)
    consumer = pay.ConsumerVoucherManager(
        book=book, state=state, program=PROGRAM, private_key=PAYER_SEED,
        plan=[{"worker_keyid": WORKER_HEX, "layer_start": 0, "layer_end": 4}],
        max_cost_by_worker={WORKER_HEX: 164})
    claim = manager.claim(
        PAYER_HEX, session,
        consumer.first_wire(WORKER_HEX, TOKEN))
    return manager, consumer, claim, vouchers


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
    manager, consumer, claim, worker_vouchers = _chain_admission(
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

    # A consumer that stops after the first checkpoint can make the worker do
    # fewer than one additional 8-token checkpoint, never an unbounded stream.
    stopped, consumer2, claim2, store2 = _chain_admission(
        tmp_path, monkeypatch, session="stopped")
    first = _run_step(stopped, claim2, 1, 1)
    assert first is not None
    stopped.chain_receipt_wire(claim2, first)
    completed_without_voucher = 0
    for index in range(2, 20):
        try:
            _run_step(stopped, claim2, index, 1)
        except admission.PaidAdmissionError as exc:
            assert str(exc) == "voucher-missing"
            break
        completed_without_voucher += 1
    assert completed_without_voucher <= 8

    consumer.book.close()
    manager.store.close()
    worker_vouchers.close()
    stopped.store.close()
    consumer2.book.close()
    store2.close()


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
