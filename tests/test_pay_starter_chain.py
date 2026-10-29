"""On-chain inviter-funded starter grants and their crash fences."""
from __future__ import annotations

import hashlib
import os
import struct
import threading
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from network import nakd, pay_chain, pay_wallet, pay_voucher


PROGRAM = pay_chain.b58encode(bytes([81]) * 32)
BLOCKHASH = pay_chain.b58encode(bytes([82]) * 32)
RECIPIENT_SEED = bytes([83]) * 32
RECIPIENT = ed25519.Ed25519PrivateKey.from_private_bytes(
    RECIPIENT_SEED).public_key().public_bytes_raw()


def _escrow_data(owner: str, vault: str, *, committed: int = 0) -> bytes:
    raw = bytearray(163)
    raw[:8] = hashlib.sha256(b"account:Escrow").digest()[:8]
    raw[8:40] = pay_chain.pubkey_bytes(owner)
    raw[40:72] = pay_chain.pubkey_bytes(vault)
    struct.pack_into("<QIIQQQ", raw, 72, committed, 0, 0, 0, 0, 0)
    raw[113:145] = pay_chain.pubkey_bytes(owner)
    raw[145:147] = b"\x01\x01"
    return bytes(raw)


def _delegate_data(escrow: str, *, cap: int, expires: int,
                   flags: int = 1, disabled: bool = False) -> bytes:
    raw = bytearray(pay_chain.DELEGATE_ACCOUNT_SPACE)
    raw[:8] = hashlib.sha256(b"account:Delegate").digest()[:8]
    raw[8:40] = pay_chain.pubkey_bytes(escrow)
    raw[40:72] = RECIPIENT
    struct.pack_into("<QQQqBII", raw, 72, 0, cap, 0, expires, flags, 0, 0)
    raw[113] = int(disabled)
    raw[114] = 1
    return bytes(raw)


def _token_data(amount: int) -> bytes:
    raw = bytearray(165)
    struct.pack_into("<Q", raw, 64, amount)
    return bytes(raw)


class StarterStore:
    def __init__(self, row):
        self.row = row
        self.updates = []
        self.crash_on_intent = False

    def starter_update(self, invite_id, **changes):
        assert invite_id == self.row["invite_id"]
        self.row.update(changes)
        self.updates.append(dict(changes))
        if self.crash_on_intent and changes.get("sol_lamports"):
            self.crash_on_intent = False
            raise RuntimeError("crash after durable SOL intent")
        return True


class AttemptStore(StarterStore):
    def starter_rows(self, states=None):
        if states and self.row["state"] not in states:
            return []
        return [self.row]

    def contact(self, _person):
        return {"node": self.row["to_keyid"], "petname": "newcomer"}

    def authenticated_contact_node(self, _person):
        return self.row["to_keyid"]


class StarterRpc:
    def __init__(self, wallet, *, vault_amount=10, committed=0,
                 wallet_balance=50_000_000, node_balance=0,
                 existing_cap=None, existing_expires=0, fail_unknown=False):
        self.wallet = wallet
        self.derived = pay_chain._derived(wallet["address"], PROGRAM)
        self.delegate_address = pay_chain.delegate_pda(
            self.derived["escrow"], RECIPIENT, PROGRAM)[0]
        self.vault_amount = vault_amount
        self.committed = committed
        self.wallet_balance = wallet_balance
        self.node_balance = node_balance
        self.delegate_cap = existing_cap
        self.delegate_expires = existing_expires
        self.fail_unknown = fail_unknown
        self.sent = []
        self.chain_now = 1_700_000_000

    def ensure_cluster(self):
        return "local"

    def chain_time(self):
        return self.chain_now

    def account_info(self, address):
        if address == self.derived["escrow"]:
            return {"owner": PROGRAM, "data": _escrow_data(
                self.wallet["address"], self.derived["vault"], committed=self.committed)}
        if address == self.derived["vault"]:
            return {"owner": pay_chain.TOKEN_PROGRAM_ID,
                    "data": _token_data(self.vault_amount)}
        if address == self.delegate_address and self.delegate_cap is not None:
            return {"owner": PROGRAM, "data": _delegate_data(
                self.derived["escrow"], cap=self.delegate_cap,
                expires=self.delegate_expires)}
        return None

    def balance(self, address):
        if address == self.wallet["address"]:
            return self.wallet_balance
        assert address == pay_chain.b58encode(RECIPIENT)
        return self.node_balance

    def latest_blockhash(self):
        return BLOCKHASH

    def fee_for_message(self, _message):
        return 5_000

    def rent_exemption(self, space):
        assert space == pay_chain.DELEGATE_ACCOUNT_SPACE
        return 1_000_000

    def send_and_confirm(self, transaction, *, on_sent=None):
        self.sent.append(transaction)
        if on_sent:
            on_sent(transaction.signature)
        self.delegate_cap = max(self.delegate_cap or 0, 1_000_000)
        self.delegate_expires = self.chain_now + nakd.STARTER_DELEGATE_DAYS * 86400
        if self.fail_unknown:
            self.fail_unknown = False
            raise pay_voucher.VoucherError("transaction final state is unknown")
        return transaction.signature


@pytest.fixture
def chain_node(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_PAY_RPC", "http://127.0.0.1:8899")
    monkeypatch.setenv("NAK_PAY_PROGRAM", PROGRAM)
    wallet = pay_wallet.ensure_wallet(home)
    row = {
        "invite_id": "starter-one", "to_keyid": RECIPIENT.hex(),
        "amount": 100_000_000, "sol_lamports": 0, "chain_signature": "",
    }
    node = object.__new__(nakd.Node)
    node.store = StarterStore(row)
    return node, wallet, row


def test_chain_starter_sends_one_delegate_and_sol_transaction(chain_node, monkeypatch):
    node, wallet, row = chain_node
    rpc = StarterRpc(wallet, vault_amount=2_000_000)
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)

    receipt = node._chain_starter(row)

    assert len(rpc.sent) == 1
    raw = rpc.sent[0].message
    args = (RECIPIENT + struct.pack("<QqB", 1_000_000,
                                    rpc.chain_now + 30 * 86400,
                                    pay_chain.DELEGATE_OPEN_CHANNEL))
    assert pay_chain.anchor_discriminator("delegate_set") + args in raw
    assert struct.pack("<I", 2) + struct.pack("<Q", nakd.STARTER_NODE_SOL) in raw
    assert row["sol_lamports"] == nakd.STARTER_NODE_SOL
    assert receipt["signature"] == rpc.sent[0].signature
    assert receipt["escrow"] == rpc.derived["escrow"]
    assert receipt["delegate"] == rpc.delegate_address


def test_unknown_outcome_is_adopted_without_second_send(chain_node, monkeypatch):
    node, wallet, row = chain_node
    rpc = StarterRpc(wallet, vault_amount=2_000_000, fail_unknown=True)
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)

    with pytest.raises(pay_voucher.VoucherError, match="unknown"):
        node._chain_starter(row)
    receipt = node._chain_starter(row)

    assert len(rpc.sent) == 1
    assert receipt["existing"] is True
    assert receipt["signature"] == row["chain_signature"]


def test_crash_after_sol_intent_never_builds_second_top_up(chain_node, monkeypatch):
    node, wallet, row = chain_node
    rpc = StarterRpc(wallet, vault_amount=2_000_000)
    node.store.crash_on_intent = True
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)

    with pytest.raises(RuntimeError, match="durable SOL intent"):
        node._chain_starter(row)
    assert row["sol_lamports"] == nakd.STARTER_NODE_SOL and not rpc.sent
    node._chain_starter(row)

    assert len(rpc.sent) == 1
    assert struct.pack("<I", 2) + struct.pack("<Q", nakd.STARTER_NODE_SOL) not in rpc.sent[0].message


def test_existing_larger_expired_cap_is_not_lowered(chain_node, monkeypatch):
    node, wallet, row = chain_node
    larger = 2_000_000
    rpc = StarterRpc(
        wallet, vault_amount=3_000_000, existing_cap=larger,
        existing_expires=1_600_000_000, node_balance=nakd.STARTER_NODE_SOL)
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)

    node._chain_starter(row)

    assert struct.pack("<Q", larger) in rpc.sent[0].message
    assert rpc.delegate_cap == larger


@pytest.mark.parametrize(("changes", "match"), [
    ({"vault_amount": 999_999}, "inviter escrow lacks 1.000000 USDC"),
    ({"wallet_balance": 1, "vault_amount": 2_000_000}, "wallet lacks SOL"),
])
def test_chain_starter_preflight_refusals(chain_node, monkeypatch, changes, match):
    node, wallet, row = chain_node
    rpc = StarterRpc(wallet, **changes)
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)

    with pytest.raises(ValueError, match=match):
        node._chain_starter(row)

    assert not rpc.sent


def test_chain_starter_refuses_broker_custody(chain_node):
    node, _wallet, row = chain_node
    (Path(os.environ["HOME"]) / ".nakshatra/pay/custody").write_text("broker\n")

    with pytest.raises(ValueError, match="file-custody wallet"):
        node._chain_starter(row)


def test_chain_starter_refuses_fractional_micro_usdc(chain_node):
    node, _wallet, row = chain_node
    row["amount"] = 101

    with pytest.raises(ValueError, match="whole number of micro-USDC"):
        node._chain_starter(row)


def _attempt_node():
    row = {
        "invite_id": "accepted", "person": "ab" * 32,
        "to_keyid": RECIPIENT.hex(), "amount": 1_000_000,
        "signed_invite": "signed", "state": "pending", "reason": "",
        "created": 100, "updated": 100, "next_attempt": 0,
        "attempts": 0, "approved": 0, "paid_at": None, "receipt": None,
        "sol_lamports": 0, "chain_signature": "",
    }
    node = object.__new__(nakd.Node)
    node.store = AttemptStore(row)
    node._starter_lock = threading.Lock()
    node.person = "cd" * 32
    node.node = "ef" * 32
    node.agent = "pa"
    node.log = lambda _message: None
    node._verified_starter_invite = lambda _code: {
        "kind": "consumer", "consumer": {
            "code": row["invite_id"], "starter": row["amount"]}}
    node._grant_status = lambda: {"valid": True, "caps": ["nak.pay"]}
    node._signer = lambda _request: {"ok": True, "status": "ok"}
    node._chain_starter = lambda _row: {
        "ts": 500, "signature": "chain-signature", "escrow": "escrow",
        "delegate": "delegate", "cap": 10_000, "expires_at": 900,
    }

    class RefuseLedger:
        def starter(self, *_args):
            raise AssertionError("ledger starter path must not run in chain mode")

    node.settle = RefuseLedger()
    return node, row


def test_attempt_starter_chain_mode_keeps_approval_gates_and_skips_ledger(monkeypatch):
    node, row = _attempt_node()
    calls = []
    node._chain_starter = lambda item: calls.append(dict(item)) or {
        "ts": 500, "signature": "chain-signature", "escrow": "escrow",
        "delegate": "delegate", "cap": 10_000, "expires_at": 900,
    }
    monkeypatch.setenv("NAK_PAY_MODE", "chain")

    assert node._attempt_starter(row["invite_id"], now=200) is True

    assert len(calls) == 1
    assert row["approved"] == 1 and row["state"] == "paid"
    assert row["receipt"]["signature"] == "chain-signature"


@pytest.mark.parametrize("gate", [
    "unapproved", "missing-grant", "unauthenticated", "tampered", "expired",
])
def test_attempt_starter_chain_mode_every_existing_gate_blocks(monkeypatch, gate):
    node, row = _attempt_node()
    paid = []
    node._chain_starter = lambda _row: paid.append(True) or {"ts": 500}
    if gate == "unapproved":
        node._signer = lambda _request: {"ok": False, "error": "not approved"}
    elif gate == "missing-grant":
        node._grant_status = lambda: {"valid": True, "caps": []}
    elif gate == "unauthenticated":
        node.store.authenticated_contact_node = lambda _person: ""
    elif gate == "tampered":
        node._verified_starter_invite = lambda _code: {
            "kind": "consumer", "consumer": {"code": "other", "starter": row["amount"]}}
    monkeypatch.setenv("NAK_PAY_MODE", "chain")

    now = row["created"] + nakd.STARTER_RETRY_WINDOW_S + 1 if gate == "expired" else 200
    assert node._attempt_starter(row["invite_id"], now=now) is False

    assert not paid
