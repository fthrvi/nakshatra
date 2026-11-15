"""Broker-custody CLI coverage for the bounded escrow deposit template."""
from __future__ import annotations

import hashlib
import struct
import sys
import tempfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from network import (nak, pay_chain, pay_wallet, pay_wallet_attempts,
                     pay_wallet_custody)  # noqa: E402


PROGRAM = "GLWqQsSqUsDkPvkc8rxcoQUyro7ZQd5denht21NtdTTo"
MINT = pay_chain.b58encode(bytes(range(1, 33)))
BLOCKHASH = pay_chain.b58encode(bytes(range(32, 64)))
NODE_SEED = bytes(range(101, 133))


@pytest.fixture(autouse=True)
def _temp_home(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="nak-deposit-test-", dir="/tmp") as raw:
        home = Path(raw) / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("NAK_PAY_RPC", "http://127.0.0.1:1")
        monkeypatch.setenv("NAK_PAY_PROGRAM", PROGRAM)
        monkeypatch.setenv("NAK_PAY_MINT", MINT)
        yield home


def _escrow_data(owner: str, vault: str, *, generation: int = 7) -> bytes:
    raw = bytearray(163)
    raw[:8] = hashlib.sha256(b"account:Escrow").digest()[:8]
    raw[8:40] = pay_chain.pubkey_bytes(owner)
    raw[40:72] = pay_chain.pubkey_bytes(vault)
    struct.pack_into("<Q", raw, 104, generation)
    raw[112] = 0
    raw[113:145] = pay_chain.pubkey_bytes(owner)
    raw[145:147] = b"\x01\x01"
    return bytes(raw)


def _delegate_data(escrow: str, node: str, *, expires: int) -> bytes:
    raw = bytearray(115)
    raw[:8] = hashlib.sha256(b"account:Delegate").digest()[:8]
    raw[8:40] = pay_chain.pubkey_bytes(escrow)
    raw[40:72] = pay_chain.pubkey_bytes(node)
    struct.pack_into("<QQQqBII", raw, 72, 7, 5_000_000, 0, expires, 3, 0, 0)
    raw[114] = 1
    return bytes(raw)


def _token_data(amount: int) -> bytes:
    raw = bytearray(165)
    struct.pack_into("<Q", raw, 64, amount)
    return bytes(raw)


class FakeRpc:
    def __init__(self, *, configured: bool, wallet_token_balance: int | None,
                 node_balance: int = nak.NODE_SOL_TARGET, generation: int = 7,
                 delegate_configured: bool = True, wallet_balance: int = 100_000_000):
        self.configured = configured
        self.wallet_token_balance = wallet_token_balance
        self.node_balance = node_balance
        self.wallet_balance = wallet_balance
        self.generation = generation
        self.delegate_configured = delegate_configured
        self.account_requests: list[tuple[str, str]] = []
        self.balance_requests: list[str] = []
        self.fee_requests: list[bytes] = []
        self.rent_requests: list[int] = []
        self.sent = []
        self.escrow_account_data = None
        self.ata_account_data = None
        self.vault_account_data = None
        self.vault_present = True
        self.vault_owner = pay_chain.TOKEN_PROGRAM_ID
        self.signature_status_result = {
            "err": None, "confirmationStatus": "finalized"}
        self.blockhash_infos = []
        self.owner = self.node = self.escrow = self.vault = self.delegate = self.ata = ""

    def configure(self, wallet: dict, node_public: bytes) -> None:
        self.owner = wallet["address"]
        self.node = pay_chain.b58encode(node_public)
        derived = pay_chain._derived(self.owner, PROGRAM)
        self.escrow = derived["escrow"]
        self.vault = derived["vault"]
        self.delegate = pay_chain.delegate_pda(
            self.escrow, node_public, PROGRAM)[0]
        self.ata = pay_chain.associated_token_address(self.owner, MINT)[0]

    def ensure_cluster(self):
        return "local"

    def account_info(self, address, *, commitment="confirmed"):
        self.account_requests.append((address, commitment))
        if not self.configured:
            return None
        if address == self.escrow:
            data = (self.escrow_account_data if self.escrow_account_data is not None
                    else _escrow_data(self.owner, self.vault, generation=self.generation))
            return {"owner": PROGRAM, "data": data}
        if address == self.vault:
            if not self.vault_present:
                return None
            data = (self.vault_account_data if self.vault_account_data is not None
                    else _token_data(0))
            return {"owner": self.vault_owner, "data": data}
        if address == self.delegate and self.delegate_configured:
            return {"owner": PROGRAM, "data": _delegate_data(
                self.escrow, self.node, expires=30 * 86400 + 1_000_000)}
        if address == pay_chain.owner_state_pda(self.owner, PROGRAM)[0]:
            return {"owner": PROGRAM, "data": bytes(17)}
        if address == self.ata and (self.wallet_token_balance is not None
                                    or self.ata_account_data is not None):
            data = (self.ata_account_data if self.ata_account_data is not None
                    else _token_data(self.wallet_token_balance))
            return {"owner": pay_chain.TOKEN_PROGRAM_ID,
                    "data": data}
        return None

    def balance(self, address):
        self.balance_requests.append(address)
        return self.node_balance if address == self.node else self.wallet_balance

    @staticmethod
    def chain_time():
        return 1_000_000

    def rent_exemption(self, size):
        self.rent_requests.append(size)
        return 1_000 + size

    @staticmethod
    def latest_blockhash():
        return BLOCKHASH

    def latest_blockhash_info(self):
        if self.blockhash_infos:
            return self.blockhash_infos.pop(0)
        return {"blockhash": BLOCKHASH, "lastValidBlockHeight": 999}

    def fee_for_message(self, message):
        self.fee_requests.append(bytes(message))
        return 5_000

    @staticmethod
    def block_height(*, commitment="finalized"):
        assert commitment == "finalized"
        return 10

    def signature_status(self, _signature):
        return self.signature_status_result

    def send_and_confirm(self, transaction, *, last_valid_block_height=None):
        self.sent.append((transaction, last_valid_block_height))
        self.configured = True
        return transaction.signature

    @staticmethod
    def cluster_label():
        return "local"


def _wallet(home: Path, *, custody: str) -> tuple[dict, ed25519.Ed25519PrivateKey, bytes]:
    wallet = pay_wallet.ensure_wallet(home)
    private = ed25519.Ed25519PrivateKey.from_private_bytes(wallet["seed"])
    worker = home / ".nakshatra/keys/worker.ed25519"
    worker.parent.mkdir(parents=True, mode=0o700)
    worker.write_bytes(NODE_SEED)
    worker.chmod(0o600)
    node = ed25519.Ed25519PrivateKey.from_private_bytes(
        NODE_SEED).public_key().public_bytes_raw()
    if custody == "broker":
        (home / ".nakshatra/pay/wallet.ed25519").unlink()
        (home / ".nakshatra/pay/custody").write_text("broker\n")
    return wallet, private, node


def _install_rpc(monkeypatch, rpc: FakeRpc) -> None:
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)


def _manual_instruction(
        program_id: str, data: bytes,
        accounts: list[tuple[str, bool, bool]]) -> pay_chain.Instruction:
    return pay_chain.Instruction(
        program_id, data,
        tuple(pay_chain.AccountMeta(pubkey, signer, writable)
              for pubkey, signer, writable in accounts))


def test_cli_broker_deposit_signs_exact_two_instruction_template_once(
        _temp_home, monkeypatch):
    wallet, private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(configured=True, wallet_token_balance=2_000_000)
    rpc.configure(wallet, node)
    _install_rpc(monkeypatch, rpc)
    approvals = []

    def sign(message, wallet_id, *, idempotency_key):
        assert wallet_id == wallet["address"]
        approvals.append((message, idempotency_key))
        return private.sign(message)

    monkeypatch.setattr(pay_wallet_custody, "wallet_sign", sign)

    assert nak.main(["pay", "deposit", "--amount", "1"]) == 0

    instructions = pay_chain.escrow_deposit(
        wallet["address"], MINT, 1_000_000, PROGRAM)
    derived = pay_chain._derived(wallet["address"], PROGRAM)
    assert len(instructions) == 2
    assert (instructions[0].program_id, instructions[0].data) == (
        pay_chain.ASSOCIATED_TOKEN_PROGRAM_ID, b"\x01")
    assert [(meta.pubkey, meta.is_signer, meta.is_writable)
            for meta in instructions[0].accounts] == [
        (wallet["address"], True, True), (rpc.ata, False, True),
        (wallet["address"], False, False), (MINT, False, False),
        (pay_chain.SYSTEM_PROGRAM_ID, False, False),
        (pay_chain.TOKEN_PROGRAM_ID, False, False),
    ]
    assert [(meta.pubkey, meta.is_signer, meta.is_writable)
            for meta in instructions[1].accounts] == [
        (wallet["address"], True, False), (derived["config"], False, False),
        (MINT, False, False), (derived["escrow"], False, False),
        (rpc.ata, False, True), (derived["vault"], False, True),
        (pay_chain.TOKEN_PROGRAM_ID, False, False),
        (derived["event"], False, False), (PROGRAM, False, False),
    ]
    assert instructions[1].data == bytes.fromhex(
        "8964fcdb8ccd92d7") + struct.pack("<Q", 1_000_000)
    expected = pay_chain.compile_message(
        instructions, wallet["address"], BLOCKHASH).data
    assert len(approvals) == 1
    assert approvals[0][0] == expected == rpc.sent[0][0].message
    assert approvals[0][1].startswith("wallet:deposit:")
    assert rpc.account_requests.count((derived["escrow"], "finalized")) == 1
    assert rpc.account_requests.count((rpc.ata, "finalized")) == 1
    with pay_wallet_attempts.AttemptJournal() as journal:
        rows = journal.attempts()
    assert [(row["operation"], row["escrow"], row["generation"],
             row["withdrawal_act_id"], row["withdrawal_amount"])
            for row in rows] == [
        ("deposit", derived["escrow"], 7, "", None),
    ]


def test_cli_two_same_amount_broker_deposits_are_distinct_acts(
        _temp_home, monkeypatch):
    wallet, private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(configured=True, wallet_token_balance=2_000_000)
    rpc.configure(wallet, node)
    rpc.blockhash_infos = [
        {"blockhash": BLOCKHASH, "lastValidBlockHeight": 999},
        {"blockhash": pay_chain.b58encode(bytes(range(31, 63))),
         "lastValidBlockHeight": 1_000},
    ]
    _install_rpc(monkeypatch, rpc)
    approvals = []

    def sign(message, _wallet_id, *, idempotency_key):
        approvals.append((bytes(message), idempotency_key))
        return private.sign(message)

    monkeypatch.setattr(pay_wallet_custody, "wallet_sign", sign)

    assert nak.main(["pay", "deposit", "--amount", "1"]) == 0
    assert nak.main(["pay", "deposit", "--amount", "1"]) == 0

    assert len(approvals) == 2
    assert len(rpc.sent) == 2
    assert rpc.sent[0][0].signature != rpc.sent[1][0].signature
    with pay_wallet_attempts.AttemptJournal() as journal:
        rows = [row for row in journal.attempts() if row["operation"] == "deposit"]
    assert len(rows) == 2
    assert len({row["operation_scope"] for row in rows}) == 2
    assert all(row["resolution"] == "succeeded" for row in rows)


def test_cli_broker_deposit_refuses_prior_act_unresolved_without_broker_contact(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(configured=True, wallet_token_balance=2_000_000)
    rpc.configure(wallet, node)
    rpc.signature_status_result = None
    _install_rpc(monkeypatch, rpc)
    instructions = pay_chain.escrow_deposit(wallet["address"], MINT, 1_000_000, PROGRAM)
    scope = pay_wallet_attempts.operation_scope(
        "deposit", rpc.escrow, 7, act_id="earlier-deposit-act")
    logical = pay_wallet_attempts.logical_key(
        scope, pay_wallet_attempts.instruction_digest(instructions))
    with pay_wallet_attempts.AttemptJournal() as journal:
        approval = journal.prepare(
            scope=scope, logical=logical, operation="deposit",
            wallet_id=wallet["address"], custody="broker",
            message=pay_chain.compile_message(
                instructions, wallet["address"], BLOCKHASH).data,
            blockhash=BLOCKHASH, last_valid_block_height=999,
            escrow=rpc.escrow, generation=7)
        journal.record_signed(approval.id, pay_chain.b58encode(b"u" * 64))
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(
            SystemExit,
            match=r"previous deposit is unresolved; retry after it settles; nothing was signed$"):
        nak.main(["pay", "deposit", "--amount", "1"])
    assert contacted == []
    assert rpc.sent == []


@pytest.mark.parametrize("amount", ["0", "5.000001"])
def test_cli_broker_deposit_refuses_amount_outside_ceiling_before_signing(
        _temp_home, monkeypatch, amount):
    _wallet(_temp_home, custody="broker")
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(
            SystemExit,
            match=r"broker custody deposit amount.*5 USDC; nothing was signed$"):
        nak.main(["pay", "deposit", "--amount", amount])
    assert contacted == []


def test_cli_broker_deposit_refuses_short_ata_balance_before_signing(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(configured=True, wallet_token_balance=999_999)
    rpc.configure(wallet, node)
    _install_rpc(monkeypatch, rpc)
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(
            SystemExit, match=r"too little USDC.*nothing was signed$"):
        nak.main(["pay", "deposit", "--amount", "1"])
    assert (rpc.ata, "finalized") in rpc.account_requests
    assert contacted == [] and rpc.sent == []


def test_cli_broker_deposit_refuses_missing_escrow_before_signing(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(configured=False, wallet_token_balance=2_000_000)
    rpc.configure(wallet, node)
    _install_rpc(monkeypatch, rpc)
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(
            SystemExit, match=r"wallet escrow is missing.*nothing was signed$"):
        nak.main(["pay", "deposit", "--amount", "1"])
    assert (rpc.escrow, "finalized") in rpc.account_requests
    assert contacted == [] and rpc.sent == []


def test_cli_broker_deposit_refuses_malformed_escrow_before_signing(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(configured=True, wallet_token_balance=2_000_000)
    rpc.configure(wallet, node)
    rpc.escrow_account_data = b"not-an-escrow"
    _install_rpc(monkeypatch, rpc)
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(SystemExit, match=r"Escrow account.*nothing was signed$"):
        nak.main(["pay", "deposit", "--amount", "1"])
    assert contacted == [] and rpc.sent == []


def test_cli_broker_deposit_refuses_malformed_ata_before_signing(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(configured=True, wallet_token_balance=2_000_000)
    rpc.configure(wallet, node)
    rpc.ata_account_data = b"not-a-token-account"
    _install_rpc(monkeypatch, rpc)
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(SystemExit, match=r"165 bytes.*nothing was signed$"):
        nak.main(["pay", "deposit", "--amount", "1"])
    assert contacted == [] and rpc.sent == []


def test_cli_broker_deposit_refuses_missing_vault_before_signing(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(configured=True, wallet_token_balance=2_000_000)
    rpc.configure(wallet, node)
    rpc.vault_present = False
    _install_rpc(monkeypatch, rpc)
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(SystemExit, match=r"canonical vault is missing; nothing was signed$"):
        nak.main(["pay", "deposit", "--amount", "1"])
    assert contacted == [] and rpc.sent == []


def test_cli_broker_deposit_refuses_wrong_owner_vault_before_signing(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(configured=True, wallet_token_balance=2_000_000)
    rpc.configure(wallet, node)
    rpc.vault_owner = pay_chain.SYSTEM_PROGRAM_ID
    _install_rpc(monkeypatch, rpc)
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(SystemExit, match=r"classic SPL Token program; nothing was signed$"):
        nak.main(["pay", "deposit", "--amount", "1"])
    assert contacted == [] and rpc.sent == []


def test_cli_broker_deposit_refuses_low_sol_before_signing(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(
        configured=True, wallet_token_balance=2_000_000, wallet_balance=0)
    rpc.configure(wallet, node)
    _install_rpc(monkeypatch, rpc)
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(SystemExit, match=r"too little SOL.*nothing was signed$"):
        nak.main(["pay", "deposit", "--amount", "1"])
    assert contacted == [] and rpc.sent == []


def test_cli_broker_deposit_refuses_implicit_ensure_signing(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(
        configured=True, wallet_token_balance=2_000_000,
        delegate_configured=False)
    rpc.configure(wallet, node)
    _install_rpc(monkeypatch, rpc)
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(
            SystemExit, match=r"requires.*configured first.*nothing was signed$"):
        nak.main(["pay", "deposit", "--amount", "1"])
    assert contacted == [] and rpc.sent == []


def test_cli_file_custody_deposit_path_still_sends_ensure_then_deposit(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="file")
    rpc = FakeRpc(
        configured=False, wallet_token_balance=None, wallet_balance=9_460)
    rpc.configure(wallet, node)
    _install_rpc(monkeypatch, rpc)
    sends = []
    original_send = nak._pay_send

    def capture_send(rpc_arg, chain_arg, instructions, wallet_arg, **kwargs):
        sends.append((tuple(instructions), dict(kwargs)))
        return original_send(rpc_arg, chain_arg, instructions, wallet_arg, **kwargs)

    monkeypatch.setattr(nak, "_pay_send", capture_send)

    assert nak.main(["pay", "deposit", "--amount", "1"]) == 0

    derived = {
        "config": pay_chain.config_pda(PROGRAM)[0],
        "owner_state": pay_chain.owner_state_pda(wallet["address"], PROGRAM)[0],
        "escrow": pay_chain.escrow_pda(wallet["address"], PROGRAM)[0],
        "event": pay_chain.event_authority_pda(PROGRAM)[0],
    }
    derived["vault"] = pay_chain.vault_pda(derived["escrow"], PROGRAM)[0]
    delegate = pay_chain.delegate_pda(derived["escrow"], node, PROGRAM)[0]
    expected_ensure = (
        _manual_instruction(PROGRAM, hashlib.sha256(b"global:escrow_init").digest()[:8], [
            (wallet["address"], True, True), (derived["config"], False, False),
            (derived["owner_state"], False, True), (MINT, False, False),
            (derived["escrow"], False, True), (derived["vault"], False, True),
            (pay_chain.TOKEN_PROGRAM_ID, False, False),
            (pay_chain.SYSTEM_PROGRAM_ID, False, False),
            (derived["event"], False, False), (PROGRAM, False, False),
        ]),
        _manual_instruction(
            PROGRAM,
            hashlib.sha256(b"global:delegate_set").digest()[:8]
            + node + struct.pack("<QqB", 5_000_000, 1_000_000 + 30 * 86400, 3), [
                (wallet["address"], True, True), (derived["config"], False, False),
                (derived["escrow"], False, False), (delegate, False, True),
                (pay_chain.SYSTEM_PROGRAM_ID, False, False),
                (derived["event"], False, False), (PROGRAM, False, False),
            ]),
    )
    expected_deposit = (
        _manual_instruction(pay_chain.ASSOCIATED_TOKEN_PROGRAM_ID, b"\x01", [
            (wallet["address"], True, True), (rpc.ata, False, True),
            (wallet["address"], False, False), (MINT, False, False),
            (pay_chain.SYSTEM_PROGRAM_ID, False, False),
            (pay_chain.TOKEN_PROGRAM_ID, False, False),
        ]),
        _manual_instruction(
            PROGRAM,
            hashlib.sha256(b"global:escrow_deposit").digest()[:8]
            + struct.pack("<Q", 1_000_000), [
                (wallet["address"], True, False), (derived["config"], False, False),
                (MINT, False, False), (derived["escrow"], False, False),
                (rpc.ata, False, True), (derived["vault"], False, True),
                (pay_chain.TOKEN_PROGRAM_ID, False, False),
                (derived["event"], False, False), (PROGRAM, False, False),
            ]),
    )
    assert sends == [
        (expected_ensure, {
            "rent": 4_460, "operation": "ensure",
            "escrow": derived["escrow"], "generation": 0}),
        (expected_deposit, {
            "rent": 1_165, "operation": "deposit", "escrow": "", "generation": 0}),
    ]
    assert len(rpc.sent) == 2
    expected_messages = [
        pay_chain.compile_message(expected_ensure, wallet["address"], BLOCKHASH).data,
        pay_chain.compile_message(expected_deposit, wallet["address"], BLOCKHASH).data,
    ]
    assert rpc.fee_requests == expected_messages
    assert [sent[0].message for sent in rpc.sent] == expected_messages
    assert [sent[1] for sent in rpc.sent] == [None, None]
    assert rpc.rent_requests == [17, 163, 165, 115, 165]
    assert rpc.balance_requests == [rpc.node, wallet["address"], wallet["address"]]


def test_cli_broker_escrow_init_and_revoke_remain_refused(
        _temp_home, monkeypatch):
    wallet, _private, node = _wallet(_temp_home, custody="broker")
    rpc = FakeRpc(configured=False, wallet_token_balance=None)
    rpc.configure(wallet, node)
    _install_rpc(monkeypatch, rpc)
    contacted = []
    monkeypatch.setattr(
        pay_wallet_custody, "wallet_sign",
        lambda *_args, **_kwargs: contacted.append(True))

    with pytest.raises(SystemExit, match="broker custody refuses escrow initialization"):
        nak.main(["pay", "ensure"])
    with pytest.raises(SystemExit, match="broker custody refuses delegate revoke"):
        nak.main(["pay", "revoke", "00" * 32])
    assert contacted == [] and rpc.sent == []
