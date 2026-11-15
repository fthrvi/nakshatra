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
                 delegate_configured: bool = True):
        self.configured = configured
        self.wallet_token_balance = wallet_token_balance
        self.node_balance = node_balance
        self.generation = generation
        self.delegate_configured = delegate_configured
        self.account_requests: list[tuple[str, str]] = []
        self.sent = []
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
            return {"owner": PROGRAM, "data": _escrow_data(
                self.owner, self.vault, generation=self.generation)}
        if address == self.vault:
            return {"owner": pay_chain.TOKEN_PROGRAM_ID, "data": _token_data(0)}
        if address == self.delegate and self.delegate_configured:
            return {"owner": PROGRAM, "data": _delegate_data(
                self.escrow, self.node, expires=30 * 86400 + 1_000_000)}
        if address == pay_chain.owner_state_pda(self.owner, PROGRAM)[0]:
            return {"owner": PROGRAM, "data": bytes(17)}
        if address == self.ata and self.wallet_token_balance is not None:
            return {"owner": pay_chain.TOKEN_PROGRAM_ID,
                    "data": _token_data(self.wallet_token_balance)}
        return None

    def balance(self, address):
        return self.node_balance if address == self.node else 100_000_000

    @staticmethod
    def chain_time():
        return 1_000_000

    @staticmethod
    def rent_exemption(size):
        return 1_000 + size

    @staticmethod
    def latest_blockhash():
        return BLOCKHASH

    @staticmethod
    def latest_blockhash_info():
        return {"blockhash": BLOCKHASH, "lastValidBlockHeight": 999}

    @staticmethod
    def fee_for_message(_message):
        return 5_000

    @staticmethod
    def block_height(*, commitment="finalized"):
        assert commitment == "finalized"
        return 10

    @staticmethod
    def signature_status(_signature):
        return {"err": None, "confirmationStatus": "finalized"}

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
    rpc = FakeRpc(configured=False, wallet_token_balance=None)
    rpc.configure(wallet, node)
    _install_rpc(monkeypatch, rpc)

    assert nak.main(["pay", "deposit", "--amount", "1"]) == 0

    assert len(rpc.sent) == 2
    assert bytes.fromhex("8964fcdb8ccd92d7") + struct.pack(
        "<Q", 1_000_000) in rpc.sent[1][0].message


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
