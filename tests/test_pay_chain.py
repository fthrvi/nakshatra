"""Pure-Python Solana transactions and the ``nak pay`` chain CLI."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from network import nak, pay_chain, pay_wallet  # noqa: E402
from network import pay_voucher  # noqa: E402


PROGRAM = "GLWqQsSqUsDkPvkc8rxcoQUyro7ZQd5denht21NtdTTo"
MINT = pay_chain.b58encode(bytes(range(1, 33)))
BLOCKHASH = pay_chain.b58encode(bytes(range(32, 64)))


@pytest.fixture(autouse=True)
def _temp_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_PAY_PROGRAM", PROGRAM)
    monkeypatch.setenv("NAK_PAY_MINT", MINT)
    monkeypatch.delenv("NAK_PAY_RPC", raising=False)
    monkeypatch.delenv("NAKSHATRA_PAY_RPC_URL", raising=False)
    return home


def _make_keys(home: Path, *, custody: str = "file") -> dict:
    wallet = pay_wallet.ensure_wallet(home)
    if custody != "file":
        (home / ".nakshatra/pay/custody").write_text(custody + "\n")
    key_path = home / ".nakshatra/keys/worker.ed25519"
    key_path.parent.mkdir(parents=True, mode=0o700)
    key_path.write_bytes(bytes(range(101, 133)))
    key_path.chmod(0o600)
    return wallet


def test_pda_and_ata_fixed_vectors():
    owner = pay_chain.b58encode(bytes(range(32)))
    assert pay_chain.config_pda(PROGRAM) == (
        "8Q1BNKohnutkCsWhPSKVydM5LBNbzVom8Zss5hNsa7r8", 253)
    assert pay_chain.escrow_pda(owner, PROGRAM) == (
        "4VP7HBng8QazeGroEKNS1jqR5JWofEeSzs2swagzgNqA", 255)
    assert pay_chain.associated_token_address(owner, MINT) == (
        "4sLu39JCMJJhUygCgADqJywzuBKN1WkxbaiEr6u6EBuj", 254)
    # Solana/curve25519-dalek accepts the non-canonical sign on x=0 and
    # non-canonical y values after field reduction.
    assert pay_chain._is_ed25519_point(((1 << 255) + 1).to_bytes(32, "little"))
    assert pay_chain._is_ed25519_point((2**255 - 19).to_bytes(32, "little"))


def test_discriminators_and_compact_u16():
    assert pay_chain.anchor_discriminator("escrow_init") == bytes(
        [11, 31, 91, 199, 154, 124, 137, 116])
    assert pay_chain.anchor_discriminator("delegate_set") == bytes(
        [139, 16, 230, 103, 254, 166, 20, 165])
    assert pay_chain.compact_u16(0) == b"\0"
    assert pay_chain.compact_u16(127) == b"\x7f"
    assert pay_chain.compact_u16(128) == b"\x80\x01"
    assert pay_chain.compact_u16(0xFFFF) == b"\xff\xff\x03"


def test_rpc_environment_name_precedence_and_deprecated_alias(monkeypatch):
    monkeypatch.setenv("NAK_PAY_RPC", "http://127.0.0.1:1111")
    monkeypatch.setenv("NAKSHATRA_PAY_RPC_URL", "http://127.0.0.1:2222")
    assert pay_chain.configured_rpc_url(os.environ) == "http://127.0.0.1:1111"
    assert pay_voucher.configured_state().rpc_url == "http://127.0.0.1:1111"
    monkeypatch.delenv("NAK_PAY_RPC")
    assert pay_chain.configured_rpc_url(os.environ) == "http://127.0.0.1:2222"
    assert pay_voucher.configured_state().rpc_url == "http://127.0.0.1:2222"


def test_message_order_and_transaction_signature_verifies():
    seed = bytes(range(32))
    private = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
    payer = pay_chain.b58encode(private.public_key().public_bytes_raw())
    writable = pay_chain.b58encode(bytes(range(2, 34)))
    readonly = pay_chain.b58encode(bytes(range(3, 35)))
    ix = pay_chain.Instruction(PROGRAM, b"payload", (
        pay_chain.AccountMeta(readonly), pay_chain.AccountMeta(writable, False, True),
        pay_chain.AccountMeta(payer, True, False),
    ))
    message = pay_chain.compile_message([ix], payer, BLOCKHASH)
    assert message.account_keys[0] == payer
    assert message.account_keys[1] == writable
    tx = pay_chain.sign_transaction([ix], payer, BLOCKHASH, {payer: seed})
    ed25519.Ed25519PublicKey.from_public_bytes(
        private.public_key().public_bytes_raw()).verify(tx.signatures[0], tx.message)
    assert tx.data == b"\x01" + tx.signatures[0] + tx.message


def _confirmation_rpc(monkeypatch, statuses):
    signature = pay_chain.b58encode(b"s" * 64)
    transaction = pay_chain.SignedTransaction(b"transaction", b"message", (), signature)
    rpc = pay_chain.PayRpc("http://127.0.0.1:1")
    monkeypatch.setattr(rpc, "_ensure_cluster", lambda: None)
    remaining = list(statuses)
    calls = []

    def fake_rpc(method, _params):
        calls.append(method)
        if method == "sendTransaction":
            return signature
        assert method == "getSignatureStatuses"
        status = remaining.pop(0) if remaining else None
        return {"value": [status]}

    now = [0.0]
    monkeypatch.setattr(rpc, "_rpc", fake_rpc)
    monkeypatch.setattr(pay_chain.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(pay_chain.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    return rpc, transaction, calls


def test_send_and_confirm_does_not_accept_confirmed_then_null(monkeypatch):
    rpc, transaction, _calls = _confirmation_rpc(monkeypatch, [
        {"err": None, "confirmationStatus": "confirmed"},
        None,
    ])

    with pytest.raises(pay_chain.VoucherError, match=rf"final state is unknown.*{transaction.signature}"):
        rpc.send_and_confirm(transaction, timeout=0.3)


def test_send_and_confirm_waits_through_confirmed_for_finalized(monkeypatch):
    rpc, transaction, calls = _confirmation_rpc(monkeypatch, [
        {"err": None, "confirmationStatus": "confirmed"},
        {"err": None, "confirmationStatus": "finalized"},
    ])

    assert rpc.send_and_confirm(transaction, timeout=1) == transaction.signature
    assert calls == ["sendTransaction", "getSignatureStatuses", "getSignatureStatuses"]


def test_send_and_confirm_fails_immediately_on_transaction_error(monkeypatch):
    rpc, transaction, calls = _confirmation_rpc(monkeypatch, [
        {"err": {"InstructionError": [0, "Custom"]}, "confirmationStatus": "confirmed"},
        {"err": None, "confirmationStatus": "finalized"},
    ])

    with pytest.raises(pay_chain.VoucherError, match="transaction failed"):
        rpc.send_and_confirm(transaction, timeout=1)
    assert calls == ["sendTransaction", "getSignatureStatuses"]


def test_all_builders_match_neuron_and_solders_byte_for_byte():
    python = Path("/home/prithvi/.cache/nak-solana/venv-chain/bin/python")
    if not python.exists():
        pytest.skip("the solders cross-check venv is absent")
    code = r'''
from network import pay_chain as c
from python.chain import program_client as n
from solders.hash import Hash
from solders.message import Message
from solders.pubkey import Pubkey
from solders.system_program import TransferParams,transfer
P=str(n.PROGRAM_ID); owner=c.b58encode(bytes(range(32))); mint=c.b58encode(bytes(range(1,33)))
key=c.b58encode(bytes(range(2,34))); worker=c.b58encode(bytes(range(3,35))); bh=c.b58encode(bytes(range(4,36)))
d=c._derived(owner,P); base={'event_authority':d['event'],'program':P}; source=c.associated_token_address(owner,mint)[0]
delegate=c.delegate_pda(d['escrow'],key,P)[0]; channel=c.channel_pda(d['escrow'],4,5,P)[0]
assert c.config_pda(P) == (str(n.config_pda()[0]),n.config_pda()[1])
assert c.escrow_pda(owner,P) == (str(n.escrow_pda(owner)[0]),n.escrow_pda(owner)[1])
ata_ref=Pubkey.find_program_address([bytes(Pubkey.from_string(owner)),bytes(n.TOKEN_PROGRAM_ID),bytes(Pubkey.from_string(mint))],Pubkey.from_string(c.ASSOCIATED_TOKEN_PROGRAM_ID))
assert c.associated_token_address(owner,mint) == (str(ata_ref[0]),ata_ref[1])
cases=[
([c.system_transfer(owner,key,123)],transfer(TransferParams(from_pubkey=Pubkey.from_string(owner),to_pubkey=Pubkey.from_string(key),lamports=123))),
(c.escrow_init(owner,mint,P),n.escrow_init({'owner':owner,'config':d['config'],'owner_state':d['owner_state'],'mint':mint,'escrow':d['escrow'],'vault':d['vault'],**base})),
(c.escrow_deposit(owner,mint,10,P),n.escrow_deposit({'owner':owner,'config':d['config'],'mint':mint,'escrow':d['escrow'],'source':source,'vault':d['vault'],'token_program':n.TOKEN_PROGRAM_ID,**base},10)),
(c.escrow_withdraw_request(owner,10,P),n.escrow_withdraw_request({'owner':owner,'config':d['config'],'escrow':d['escrow'],'vault':d['vault'],**base},10)),
(c.escrow_withdraw_execute(owner,mint,P),n.escrow_withdraw_execute({'owner':owner,'config':d['config'],'mint':mint,'escrow':d['escrow'],'vault':d['vault'],'destination':source,'token_program':n.TOKEN_PROGRAM_ID,**base})),
(c.delegate_set(owner,key,12,33,15,P),n.delegate_set({'owner':owner,'config':d['config'],'escrow':d['escrow'],'delegate':delegate,'system_program':n.SYSTEM_PROGRAM_ID,**base},key,12,33,15)),
(c.delegate_revoke(owner,key,P),n.delegate_revoke({'owner':owner,'config':d['config'],'escrow':d['escrow'],'delegate':delegate,**base},key)),
(c.channel_open(key,owner,worker,10,4,5,P,delegate=delegate),n.channel_open({'signer':key,'config':d['config'],'escrow':d['escrow'],'vault':d['vault'],'delegate':delegate,'channel':channel,'system_program':n.SYSTEM_PROGRAM_ID,**base},worker,10)),
(c.channel_close_request(key,owner,4,5,P),n.channel_close_request({'signer':key,'config':d['config'],'escrow':d['escrow'],'channel':channel,**base})),]
for ours, reference in cases:
    actual=ours[-1]
    assert actual.data == reference.data
    assert [(a.pubkey,a.is_signer,a.is_writable) for a in actual.accounts] == [(str(a.pubkey),a.is_signer,a.is_writable) for a in reference.accounts]
    sold=[n.Instruction(Pubkey.from_string(x.program_id),x.data,[n.AccountMeta(Pubkey.from_string(a.pubkey),a.is_signer,a.is_writable) for a in x.accounts]) for x in ours]
    assert c.compile_message(ours,owner,bh).data == bytes(Message.new_with_blockhash(sold,Pubkey.from_string(owner),Hash.from_string(bh)))
print('ok')
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join((
        "/home/prithvi/.local/lib/python3.12/site-packages",
        "/home/prithvi/neuron", str(ROOT / "scripts")))
    result = subprocess.run([str(python), "-c", code], env=env, text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def _escrow_data(owner: str, vault: str) -> bytes:
    raw = bytearray(163)
    raw[:8] = hashlib.sha256(b"account:Escrow").digest()[:8]
    raw[8:40] = pay_chain.pubkey_bytes(owner)
    raw[40:72] = pay_chain.pubkey_bytes(vault)
    raw[145:147] = b"\x01\x01"
    return bytes(raw)


def _delegate_data(escrow: str, key: str, *, expires: int) -> bytes:
    raw = bytearray(115)
    raw[:8] = hashlib.sha256(b"account:Delegate").digest()[:8]
    raw[8:40] = pay_chain.pubkey_bytes(escrow)
    raw[40:72] = pay_chain.pubkey_bytes(key)
    struct.pack_into("<QQQqBII", raw, 72, 0, 5_000_000, 0, expires, 3, 0, 0)
    raw[114] = 1
    return bytes(raw)


def test_decode_escrow_accepts_stale_some_payload_after_pending_is_cleared():
    owner = pay_chain.b58encode(bytes(range(32)))
    vault = pay_chain.b58encode(bytes(range(1, 33)))
    raw = bytearray(_escrow_data(owner, vault))
    raw[147:163] = bytes(range(1, 17))

    decoded = pay_chain.decode_escrow(bytes(raw))

    assert decoded["owner"] == owner
    assert decoded["vault"] == vault
    assert decoded["pending_withdraw"] is None


def test_decode_escrow_rejects_invalid_pending_option_tag():
    owner = pay_chain.b58encode(bytes(range(32)))
    vault = pay_chain.b58encode(bytes(range(1, 33)))
    raw = bytearray(_escrow_data(owner, vault))
    raw[112] = 2

    with pytest.raises(ValueError, match="invalid pending withdrawal option"):
        pay_chain.decode_escrow(bytes(raw))


class FakeRpc:
    def __init__(self, *, genesis="local-test", balance=100_000_000,
                 node_balance=0, token_balance=0):
        self.genesis = genesis
        self.balance = balance
        self.node_balance = node_balance
        self.token_balance = token_balance
        self.methods: list[str] = []
        self.sent = 0
        self.ready = False
        self.owner = self.node = self.escrow = self.vault = self.delegate = ""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                request = json.loads(self.rfile.read(int(self.headers["content-length"])))
                method, params = request["method"], request["params"]
                outer.methods.append(method)
                if method == "getGenesisHash": result = outer.genesis
                elif method == "getAccountInfo": result = outer.account(params[0])
                elif method == "getMinimumBalanceForRentExemption": result = 1_000 + params[0]
                elif method == "getLatestBlockhash": result = {"context":{"slot":1},"value":{"blockhash":BLOCKHASH,"lastValidBlockHeight":999}}
                elif method == "getFeeForMessage": result = {"context":{"slot":1},"value":5_000}
                elif method == "getBalance":
                    value = outer.node_balance if params[0] == outer.node else outer.balance
                    result = {"context":{"slot":1},"value":value}
                elif method == "sendTransaction":
                    raw = base64.b64decode(params[0]); result = pay_chain.b58encode(raw[1:65])
                    outer.sent += 1; outer.ready = True
                    outer.node_balance = nak._NODE_SOL_TARGET
                elif method == "getSignatureStatuses": result = {"context":{"slot":1},"value":[{"slot":1,"confirmations":None,"err":None,"confirmationStatus":"finalized"}]}
                else: raise AssertionError(method)
                body = json.dumps({"jsonrpc":"2.0","id":request["id"],"result":result}).encode()
                self.send_response(200); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def log_message(self, _fmt, *_args): return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def configure(self, wallet: dict, node: bytes):
        self.owner = wallet["address"]
        self.escrow = pay_chain.escrow_pda(self.owner, PROGRAM)[0]
        self.vault = pay_chain.vault_pda(self.escrow, PROGRAM)[0]
        self.node = pay_chain.b58encode(node)
        self.delegate = pay_chain.delegate_pda(self.escrow, node, PROGRAM)[0]

    def account(self, address: str):
        data = None
        if self.ready and address == self.escrow: data = _escrow_data(self.owner, self.vault)
        elif self.ready and address == self.delegate: data = _delegate_data(self.escrow, self.node, expires=int(time.time()) + 30*86400)
        elif self.ready and address == self.vault:
            data = bytearray(165)
            struct.pack_into("<Q", data, 64, self.token_balance)
            data = bytes(data)
        elif self.ready and address == pay_chain.owner_state_pda(self.owner, PROGRAM)[0]: data = bytes(17)
        value = None if data is None else {"data":[base64.b64encode(data).decode(),"base64"],"executable":False,"lamports":1,"owner":PROGRAM if len(data) != 165 else pay_chain.TOKEN_PROGRAM_ID,"rentEpoch":0,"space":len(data)}
        return {"context":{"slot":1},"value":value}

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_exc):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)


def test_cli_ensure_is_idempotent_and_limits_are_private(_temp_home, monkeypatch):
    wallet = _make_keys(_temp_home)
    node = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(101, 133))).public_key().public_bytes_raw()
    rpc = FakeRpc(); rpc.configure(wallet, node)
    with rpc as url:
        monkeypatch.setenv("NAK_PAY_RPC", url)
        assert nak.main(["pay", "ensure"]) == 0
        assert nak.main(["pay", "ensure"]) == 0
    assert rpc.sent == 1
    limits = _temp_home / ".nakshatra/pay/limits.json"
    assert stat.S_IMODE(limits.stat().st_mode) == 0o600


def test_cli_ensure_tops_up_node_and_reports_wallet_shortfall(
        _temp_home, monkeypatch, capsys):
    wallet = _make_keys(_temp_home)
    node = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes(range(101, 133))).public_key().public_bytes_raw()
    rpc = FakeRpc(balance=100_000_000, node_balance=9_000_000)
    rpc.configure(wallet, node)
    with rpc as url:
        monkeypatch.setenv("NAK_PAY_RPC", url)
        assert nak.main(["pay", "ensure"]) == 0
    assert rpc.node_balance == nak._NODE_SOL_TARGET
    assert rpc.sent == 1

    poor = FakeRpc(balance=1, node_balance=0)
    poor.configure(wallet, node)
    with poor as url:
        monkeypatch.setenv("NAK_PAY_RPC", url)
        with pytest.raises(
                SystemExit,
                match=rf"wallet {wallet['address']}.*shortfall.*nothing was sent"):
            nak.main(["pay", "ensure"])
    assert poor.sent == 0
    assert capsys.readouterr().out == ""


def test_cli_low_sol_sends_nothing(_temp_home, monkeypatch):
    wallet = _make_keys(_temp_home)
    node = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(101, 133))).public_key().public_bytes_raw()
    rpc = FakeRpc(balance=0); rpc.configure(wallet, node)
    with rpc as url:
        monkeypatch.setenv("NAK_PAY_RPC", url)
        with pytest.raises(SystemExit, match=r"wallet .* needs .* SOL.*nothing was sent"):
            nak.main(["pay", "ensure"])
    assert rpc.sent == 0


def test_cli_mainnet_and_broker_are_refused(_temp_home, monkeypatch):
    wallet = _make_keys(_temp_home)
    node = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(101, 133))).public_key().public_bytes_raw()
    rpc = FakeRpc(genesis=pay_chain.MAINNET_GENESIS_HASH); rpc.configure(wallet, node)
    with rpc as url:
        monkeypatch.setenv("NAK_PAY_RPC", url)
        with pytest.raises(SystemExit, match="mainnet"):
            nak.main(["pay", "ensure"])
    assert rpc.sent == 0
    (_temp_home / ".nakshatra/pay/custody").write_text("broker\n")
    with pytest.raises(SystemExit, match="broker.*not available yet"):
        nak.main(["pay", "ensure"])


@pytest.mark.parametrize("argv,match", [
    (["pay","limits","--cap","5","--days","0","--per-tx","1","--per-day","5"], "days"),
    (["pay","limits","--cap","5","--days","30","--per-tx","6","--per-day","5"], "per-tx"),
    (["pay","limits","--cap","5","--days","30","--per-tx","1.0000001","--per-day","5"], "6 USDC decimals"),
])
def test_cli_limits_validation(_temp_home, argv, match):
    _make_keys(_temp_home)
    with pytest.raises(SystemExit, match=match):
        nak.main(argv)


@pytest.mark.parametrize(("amount", "warned"), [("20", False), ("20.000001", True)])
def test_cli_deposit_warns_when_resulting_escrow_exceeds_20_usdc(
        _temp_home, monkeypatch, capsys, amount, warned):
    wallet = _make_keys(_temp_home)
    node = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes(range(101, 133))).public_key().public_bytes_raw()
    rpc = FakeRpc(); rpc.configure(wallet, node)

    with rpc as url:
        monkeypatch.setenv("NAK_PAY_RPC", url)
        assert nak.main(["pay", "deposit", "--amount", amount]) == 0

    warning = "run `nak pay backup --confirm`"
    assert (warning in capsys.readouterr().err) is warned


def test_cli_status_warns_above_20_usdc_until_backup_is_confirmed(
        _temp_home, monkeypatch, capsys):
    wallet = _make_keys(_temp_home)
    node = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes(range(101, 133))).public_key().public_bytes_raw()
    rpc = FakeRpc(token_balance=20_000_001); rpc.configure(wallet, node); rpc.ready = True

    with rpc as url:
        monkeypatch.setenv("NAK_PAY_RPC", url)
        assert nak.main(["pay", "status"]) == 0
        captured = capsys.readouterr()
        assert "node balance: 0 SOL" in captured.out
        assert "run `nak pay backup --confirm`" in captured.err

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert nak.main(["pay", "backup", "--confirm"]) == 0
        capsys.readouterr()

        assert nak.main(["pay", "status"]) == 0
        assert capsys.readouterr().err == ""


@pytest.mark.parametrize("amount", [
    "0.00000099999999999999999999999999999999999999",
    "1.0000000000000000000000000000001",
    "1.0000001",
    "-1",
    "nan",
    "1e3",
    "1e400",
    "18446744073709551616",
])
def test_pay_usdc_parser_rejects_inexact_or_unsafe_amounts(amount):
    with pytest.raises(SystemExit, match="positive with at most 6 USDC decimals"):
        nak._pay_usdc(amount, "--amount")


@pytest.mark.slow
def test_local_validator_pay_flow(tmp_path, _temp_home, monkeypatch):
    """CLI + builders against a private validator and locally loaded programs."""
    needed = [
        Path("/home/prithvi/wt/neu-pay/programs/nakshatra_pay/target/deploy/nakshatra_pay.so"),
        Path("/home/prithvi/.cache/nak-solana/spl_token.so"),
        Path("/home/prithvi/.cache/nak-solana/spl_ata.so"),
        Path("/home/prithvi/.cache/nak-solana/venv-chain/bin/python"),
    ]
    if not all(path.exists() for path in needed) or shutil.which("solana-test-validator") is None:
        pytest.skip("local validator programs/tooling are incomplete")
    wallet = _make_keys(_temp_home)
    node_seed = (_temp_home / ".nakshatra/keys/worker.ed25519").read_bytes()
    node_public = ed25519.Ed25519PrivateKey.from_private_bytes(
        node_seed).public_key().public_bytes_raw()
    authority_seed = os.urandom(32)
    mint_seed = os.urandom(32)
    authority = pay_chain.b58encode(ed25519.Ed25519PrivateKey.from_private_bytes(
        authority_seed).public_key().public_bytes_raw())
    mint = pay_chain.b58encode(ed25519.Ed25519PrivateKey.from_private_bytes(
        mint_seed).public_key().public_bytes_raw())
    node = pay_chain.b58encode(node_public)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    if not 12_000 <= port <= 64_000 - 90 or 8_800 <= port <= 8_999:
        pytest.skip("could not reserve a safe local validator port block")
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "validator.log"
    log = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(name, None)
    process = subprocess.Popen([
        shutil.which("solana-test-validator"), "--ledger", str(tmp_path / "ledger"),
        "--reset", "--quiet", "--rpc-port", str(port), "--faucet-port", str(port + 2),
        "--gossip-port", str(port + 3), "--dynamic-port-range", f"{port + 10}-{port + 80}",
        "--mint", authority, "--bpf-program", pay_chain.TOKEN_PROGRAM_ID, str(needed[1]),
        "--bpf-program", pay_chain.ASSOCIATED_TOKEN_PROGRAM_ID, str(needed[2]),
        "--upgradeable-program", PROGRAM, str(needed[0]), authority,
    ], env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    try:
        rpc = pay_chain.PayRpc(url, timeout=1)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.skip("local validator failed: " + log_path.read_text(errors="replace")[-1000:])
            try:
                if rpc._rpc("getHealth", []) == "ok" and rpc._rpc("getSlot", []) >= 2:
                    break
            except pay_chain.VoucherError:
                pass
            time.sleep(0.2)
        else:
            pytest.skip("local validator did not become healthy")

        setup = r'''
import base64,os,runpy,time
from solders.instruction import AccountMeta,Instruction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import TransferParams,transfer
from python.chain import program_client as pay
from python.chain.solana_rpc import SolanaRPC
t=runpy.run_path('/home/prithvi/neuron/python/tests/test_pay_program_validator.py')
send=t['_send']; create_mint=t['_create_mint']; create_token=t['_create_token_account']; mint_to=t['_mint_to']
rpc=SolanaRPC(os.environ['RPC'],timeout=3); authority=Keypair.from_seed(bytes.fromhex(os.environ['AUTH_SEED'])); mint=Keypair.from_seed(bytes.fromhex(os.environ['MINT_SEED']))
owner=Pubkey.from_string(os.environ['OWNER']); node=Pubkey.from_string(os.environ['NODE'])
create_mint(rpc,authority,mint)
source=Pubkey.from_string(os.environ['SOURCE_ATA'])
ata=Instruction(Pubkey.from_string(os.environ['ATA_PROGRAM']),b'\x01',[
 AccountMeta(authority.pubkey(),True,True),AccountMeta(source,False,True),AccountMeta(owner,False,False),
 AccountMeta(mint.pubkey(),False,False),AccountMeta(pay.SYSTEM_PROGRAM_ID,False,False),AccountMeta(pay.TOKEN_PROGRAM_ID,False,False)])
send(rpc,[ata],authority); mint_to(rpc,authority,mint.pubkey(),source,2_000_000)
fee_wallet=Keypair(); create_token(rpc,authority,fee_wallet,mint.pubkey(),authority.pubkey())
send(rpc,[transfer(TransferParams(from_pubkey=authority.pubkey(),to_pubkey=owner,lamports=100_000_000)),transfer(TransferParams(from_pubkey=authority.pubkey(),to_pubkey=node,lamports=100_000_000))],authority)
config=pay.config_pda()[0]; fee_vault=pay.fee_vault_pda(config)[0]
send(rpc,[pay.config_init({'authority':authority.pubkey(),'config':config,'program_data':pay.program_data_pda()[0],'mint':mint.pubkey(),'fee_wallet':fee_wallet.pubkey(),'fee_vault':fee_vault,'token_program':pay.TOKEN_PROGRAM_ID,'system_program':pay.SYSTEM_PROGRAM_ID,'event_authority':pay.event_authority_pda()[0],'program':pay.PROGRAM_ID},300)],authority)
'''
        setup_env = env.copy()
        setup_env.update({
            "PYTHONPATH": "/home/prithvi/.local/lib/python3.12/site-packages:/home/prithvi/neuron",
            "RPC": url, "AUTH_SEED": authority_seed.hex(), "MINT_SEED": mint_seed.hex(),
            "OWNER": wallet["address"], "NODE": node,
            "SOURCE_ATA": pay_chain.associated_token_address(wallet["address"], mint)[0],
            "ATA_PROGRAM": pay_chain.ASSOCIATED_TOKEN_PROGRAM_ID,
        })
        result = subprocess.run([str(needed[3]), "-c", setup], env=setup_env,
                                text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr

        monkeypatch.setenv("NAK_PAY_RPC", url)
        monkeypatch.setenv("NAK_PAY_MINT", mint)
        assert nak.main(["pay", "ensure"]) == 0
        assert nak.main(["pay", "deposit", "--amount", "2"]) == 0
        escrow_address = pay_chain.escrow_pda(wallet["address"], PROGRAM)[0]
        escrow = pay_chain.decode_escrow(rpc.account_info(escrow_address)["data"])
        delegate = pay_chain.delegate_pda(escrow_address, node_public, PROGRAM)[0]
        assert rpc.account_info(delegate) is not None
        channel_address = pay_chain.channel_pda(
            escrow_address, escrow["generation"], escrow["channel_seq"], PROGRAM)[0]
        open_ix = pay_chain.channel_open(
            node, wallet["address"], authority, 1_000_000,
            escrow["generation"], escrow["channel_seq"], PROGRAM, delegate=delegate)
        blockhash = rpc.latest_blockhash()
        tx = pay_chain.sign_transaction(open_ix, node, blockhash, {node: node_seed})
        rpc.send_and_confirm(tx)
        assert rpc.account_info(channel_address) is not None
        assert nak.main(["pay", "withdraw", "--amount", "0.5"]) == 0
        pending = pay_chain.decode_escrow(
            rpc.account_info(escrow_address)["data"])["pending_withdraw"]
        assert pending and pending["amount"] == 500_000
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=5)
        log.close()
