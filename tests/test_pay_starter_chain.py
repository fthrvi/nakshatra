"""On-chain inviter-funded starter grants and their crash fences."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
import threading
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from network import nakd, pay_chain, pay_wallet, pay_voucher
from network.store import Store


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


def _delegate_data(escrow: str, *, cap: int, expires: int, spent: int = 0,
                   flags: int = 1, disabled: bool = False) -> bytes:
    raw = bytearray(pay_chain.DELEGATE_ACCOUNT_SPACE)
    raw[:8] = hashlib.sha256(b"account:Delegate").digest()[:8]
    raw[8:40] = pay_chain.pubkey_bytes(escrow)
    raw[40:72] = RECIPIENT
    struct.pack_into("<QQQqBII", raw, 72, 0, cap, spent, expires, flags, 0, 0)
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
        self.lock = threading.Lock()

    def starter_rows(self, states=None):
        with self.lock:
            if states and self.row.get("state") not in states:
                return []
            return [dict(self.row)]

    def starter_update(self, invite_id, **changes):
        assert invite_id == self.row["invite_id"]
        with self.lock:
            self.row.update(changes)
            self.updates.append(dict(changes))
            if self.crash_on_intent and changes.get("chain_attempts"):
                self.crash_on_intent = False
                raise RuntimeError("crash after durable starter attempt")
        return True


class AttemptStore(StarterStore):
    def contact(self, _person):
        return {"node": self.row["to_keyid"], "petname": "newcomer"}

    def authenticated_contact_node(self, _person):
        return self.row["to_keyid"]


class StarterRpc:
    def __init__(self, wallet, *, vault_amount=10, committed=0,
                 wallet_balance=50_000_000, node_balance=0,
                 existing_cap=None, existing_spent=0, existing_expires=0,
                 fail_unknown=False, confirmed_only_delegate=False):
        self.wallet = wallet
        self.derived = pay_chain._derived(wallet["address"], PROGRAM)
        self.delegate_address = pay_chain.delegate_pda(
            self.derived["escrow"], RECIPIENT, PROGRAM)[0]
        self.vault_amount = vault_amount
        self.committed = committed
        self.wallet_balance = wallet_balance
        self.node_balance = node_balance
        self.delegate_cap = existing_cap
        self.delegate_spent = existing_spent
        self.delegate_expires = existing_expires
        self.fail_unknown = fail_unknown
        self.confirmed_only_delegate = confirmed_only_delegate
        self.delegate_finalized = not confirmed_only_delegate
        self.sent = []
        self.chain_now = 1_700_000_000
        self.finalized_height = 100
        self.next_last_valid_height = 120
        self.blockhash_counter = 0
        self.statuses = {}
        self.commitments = []

    def ensure_cluster(self):
        return "local"

    def chain_time(self):
        return self.chain_now

    def account_info(self, address, *, commitment="confirmed"):
        self.commitments.append(("account", commitment, address))
        if address == self.derived["escrow"]:
            return {"owner": PROGRAM, "data": _escrow_data(
                self.wallet["address"], self.derived["vault"], committed=self.committed)}
        if address == self.derived["vault"]:
            return {"owner": pay_chain.TOKEN_PROGRAM_ID,
                    "data": _token_data(self.vault_amount)}
        if address == self.delegate_address and self.delegate_cap is not None:
            if commitment == "finalized" and not self.delegate_finalized:
                return None
            return {"owner": PROGRAM, "data": _delegate_data(
                self.derived["escrow"], cap=self.delegate_cap,
                spent=self.delegate_spent, expires=self.delegate_expires)}
        return None

    def balance(self, address, *, commitment="confirmed"):
        self.commitments.append(("balance", commitment, address))
        if address == self.wallet["address"]:
            return self.wallet_balance
        assert address == pay_chain.b58encode(RECIPIENT)
        return self.node_balance

    def latest_blockhash_info(self, *, commitment="confirmed"):
        self.commitments.append(("blockhash", commitment, ""))
        self.blockhash_counter += 1
        blockhash = pay_chain.b58encode(bytes([81 + self.blockhash_counter]) * 32)
        return {"blockhash": blockhash,
                "lastValidBlockHeight": self.next_last_valid_height}

    def block_height(self, *, commitment="finalized"):
        self.commitments.append(("height", commitment, ""))
        return self.finalized_height

    def signature_status(self, signature):
        return self.statuses.get(signature)

    def fee_for_message(self, _message, *, commitment="confirmed"):
        self.commitments.append(("fee", commitment, ""))
        return 5_000

    def rent_exemption(self, space, *, commitment="confirmed"):
        self.commitments.append(("rent", commitment, ""))
        assert space == pay_chain.DELEGATE_ACCOUNT_SPACE
        return 1_000_000

    def send_and_confirm(self, transaction, *, on_sent=None):
        self.sent.append(transaction)
        if on_sent:
            on_sent(transaction.signature)
        marker = pay_chain.anchor_discriminator("delegate_set")
        offset = transaction.message.index(marker) + len(marker) + 32
        self.delegate_cap, = struct.unpack_from("<Q", transaction.message, offset)
        self.delegate_expires = self.chain_now + nakd.STARTER_DELEGATE_DAYS * 86400
        self.delegate_finalized = True
        transfer = struct.pack("<I", 2) + struct.pack("<Q", nakd.STARTER_NODE_SOL)
        if transfer in transaction.message:
            self.node_balance += nakd.STARTER_NODE_SOL
        if self.fail_unknown:
            self.fail_unknown = False
            self.delegate_finalized = False
            self.statuses[transaction.signature] = {
                "err": None, "confirmationStatus": "confirmed"}
            raise pay_voucher.VoucherError("transaction final state is unknown")
        self.statuses[transaction.signature] = {
            "err": None, "confirmationStatus": "finalized"}
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
        "amount": 100_000_000, "chain_attempts": [],
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
    assert row["chain_attempts"][0]["sol_lamports"] == nakd.STARTER_NODE_SOL
    assert receipt["signature"] == rpc.sent[0].signature
    assert receipt["escrow"] == rpc.derived["escrow"]
    assert receipt["delegate"] == rpc.delegate_address


def test_unknown_outcome_waits_for_recorded_attempt_to_finalize(chain_node, monkeypatch):
    node, wallet, row = chain_node
    rpc = StarterRpc(wallet, vault_amount=2_000_000, fail_unknown=True)
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)

    with pytest.raises(pay_voucher.VoucherError, match="unknown"):
        node._chain_starter(row)
    with pytest.raises(pay_voucher.VoucherError, match="still unresolved"):
        node._chain_starter(row)

    assert len(rpc.sent) == 1
    signature = row["chain_attempts"][0]["signature"]
    rpc.statuses[signature] = {"err": None, "confirmationStatus": "finalized"}
    receipt = node._chain_starter(row)
    assert receipt["existing"] is True
    assert receipt["signature"] == signature


def test_crash_before_broadcast_waits_for_expiry_then_rebuilds_with_sol(
        chain_node, monkeypatch):
    node, wallet, row = chain_node
    rpc = StarterRpc(wallet, vault_amount=2_000_000)
    node.store.crash_on_intent = True
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)

    with pytest.raises(RuntimeError, match="durable starter attempt"):
        node._chain_starter(row)
    assert row["chain_attempts"][0]["sol_lamports"] == nakd.STARTER_NODE_SOL
    assert not rpc.sent
    with pytest.raises(pay_voucher.VoucherError, match="still unresolved"):
        node._chain_starter(row)
    assert not rpc.sent

    rpc.finalized_height = row["chain_attempts"][0]["last_valid_block_height"] + 1
    node._chain_starter(row)
    assert len(rpc.sent) == 1
    assert struct.pack("<I", 2) + struct.pack("<Q", nakd.STARTER_NODE_SOL) in rpc.sent[0].message
    assert rpc.node_balance == nakd.STARTER_NODE_SOL


def test_two_reconcilers_under_real_pay_lock_broadcast_once(chain_node, monkeypatch):
    node, wallet, row = chain_node
    rpc = StarterRpc(wallet, vault_amount=2_000_000, fail_unknown=True)
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)
    other = object.__new__(nakd.Node)
    other.store = node.store
    results = []

    def run(target, stale):
        try:
            results.append(target._chain_starter(stale))
        except pay_voucher.VoucherError as exc:
            results.append(str(exc))

    stale_rows = (dict(row), dict(row))
    threads = [threading.Thread(target=run, args=(target, stale))
               for target, stale in zip((node, other), stale_rows)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert len(rpc.sent) == 1
    assert any("still unresolved" in str(result) for result in results)


def test_confirmed_only_delegate_is_not_treated_as_paid(chain_node, monkeypatch):
    node, wallet, row = chain_node
    rpc = StarterRpc(
        wallet, vault_amount=2_000_000, existing_cap=1_000_000,
        existing_expires=1_800_000_000, confirmed_only_delegate=True)
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)

    node._chain_starter(row)

    assert len(rpc.sent) == 1
    assert all(commitment == "finalized" for _kind, commitment, _address in rpc.commitments)


def test_finalized_error_allows_a_new_attempt(chain_node, monkeypatch):
    node, wallet, row = chain_node
    rpc = StarterRpc(wallet, vault_amount=2_000_000)
    failed_signature = pay_chain.b58encode(b"f" * 64)
    row["chain_attempts"] = [{
        "signature": failed_signature, "blockhash": BLOCKHASH,
        "last_valid_block_height": 120, "cap_total": 1_000_000,
        "sol_lamports": nakd.STARTER_NODE_SOL,
        "expires_at": rpc.chain_now + 30 * 86400,
        "escrow": rpc.derived["escrow"], "delegate": rpc.delegate_address,
    }]
    rpc.statuses[failed_signature] = {
        "err": {"InstructionError": [0, "Custom"]},
        "confirmationStatus": "finalized"}
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)

    node._chain_starter(row)

    assert len(rpc.sent) == 1
    assert row["chain_attempts"][0]["resolution"] == "failed"
    assert len(row["chain_attempts"]) == 2


def test_forgejo_main_pending_ledger_row_migrates_intact_and_still_updates(tmp_path):
    state = tmp_path / "store"
    store = Store(state, b"k" * 32)
    request_id = store.add_request("ab" * 32, "cd" * 32, "newcomer", "invite", "cd" * 32)
    store.decide_request(request_id, True, starter={
        "invite_id": "old-row", "to_keyid": "cd" * 32, "amount": 100,
        "signed_invite": "signed", "reason": "", "created": 10,
    })
    store._db.close()
    db = sqlite3.connect(state / "net.sqlite")
    db.execute("ALTER TABLE starter_obligations DROP COLUMN chain_attempts")
    db.execute("ALTER TABLE starter_obligations DROP COLUMN chain_created")
    main_columns = tuple(row[1] for row in db.execute(
        "PRAGMA table_info(starter_obligations)"))
    before = db.execute(
        f"SELECT {','.join(main_columns)} FROM starter_obligations").fetchone()
    db.commit()
    db.close()

    migrated = Store(state, b"k" * 32)
    migrated_columns = tuple(row[1] for row in migrated._db.execute(
        "PRAGMA table_info(starter_obligations)"))
    after = migrated._db.execute(
        f"SELECT {','.join(main_columns)} FROM starter_obligations").fetchone()
    assert main_columns == (
        "invite_id", "person", "to_keyid", "amount", "signed_invite", "state", "reason",
        "created", "updated", "next_attempt", "attempts", "approved", "paid_at", "receipt")
    assert after == before
    assert "sol_lamports" not in migrated_columns
    assert "chain_signature" not in migrated_columns
    row = migrated.starter_rows()[0]
    assert row["state"] == "pending" and row["signed_invite"] == "signed"
    assert row["chain_attempts"] == [] and row["chain_created"] is None
    attempt = {"signature": "sig", "last_valid_block_height": 7}
    migrated.starter_update("old-row", chain_attempts=[attempt])
    assert migrated.starter_rows()[0]["chain_attempts"] == [attempt]
    migrated.starter_update(
        "old-row", state="paid", reason="", updated=11, next_attempt=11,
        paid_at=11, receipt={"v": 1, "replayed": False})
    paid = migrated.starter_rows()[0]
    assert paid["state"] == "paid" and paid["receipt"] == {"v": 1, "replayed": False}


def test_expired_larger_cap_renews_with_exactly_signed_amount_spendable(
        chain_node, monkeypatch):
    node, wallet, row = chain_node
    larger = 100_000_000
    spent = 40_000_000
    rpc = StarterRpc(
        wallet, vault_amount=2_000_000, existing_cap=larger, existing_spent=spent,
        existing_expires=1_600_000_000, node_balance=nakd.STARTER_NODE_SOL)
    monkeypatch.setattr(pay_chain, "PayRpc", lambda _url: rpc)

    node._chain_starter(row)

    signed_cap = spent + 1_000_000
    args = (RECIPIENT + struct.pack(
        "<QqB", signed_cap, rpc.chain_now + 30 * 86400,
        pay_chain.DELEGATE_OPEN_CHANNEL))
    assert pay_chain.anchor_discriminator("delegate_set") + args in rpc.sent[0].message
    assert signed_cap - spent == 1_000_000
    assert struct.pack("<Q", larger) not in rpc.sent[0].message


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
        "chain_attempts": [], "chain_created": None,
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
    node._starter_chain_time = lambda: 200
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
    elif gate == "expired":
        row["chain_created"] = 100
        node._starter_chain_time = lambda: 100 + nakd.STARTER_RETRY_WINDOW_S + 1
    monkeypatch.setenv("NAK_PAY_MODE", "chain")

    assert node._attempt_starter(row["invite_id"], now=200) is False

    assert not paid


def test_chain_retry_window_ignores_local_clock_skew_backward(monkeypatch):
    node, row = _attempt_node()
    row["chain_created"] = 1_000
    node._starter_chain_time = lambda: 1_000 + nakd.STARTER_RETRY_WINDOW_S + 1
    paid = []
    node._chain_starter = lambda _row: paid.append(True) or {"ts": 1}
    monkeypatch.setenv("NAK_PAY_MODE", "chain")

    assert node._attempt_starter(row["invite_id"], now=-31_536_000) is False
    assert row["state"] == "failed"
    assert not paid


def test_chain_retry_window_ignores_local_clock_skew_forward(monkeypatch):
    node, row = _attempt_node()
    node._starter_chain_time = lambda: 5_000
    monkeypatch.setenv("NAK_PAY_MODE", "chain")

    assert node._attempt_starter(row["invite_id"], now=31_536_000) is True
    assert row["chain_created"] == 5_000
    assert row["state"] == "paid"
