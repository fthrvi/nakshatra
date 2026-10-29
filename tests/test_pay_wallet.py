"""Onboarding's local Solana wallet: immutable seed, offline CLI, devnet-only RPC."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from network import join_setup, nak, pay_wallet  # noqa: E402


@pytest.fixture(autouse=True)
def _temporary_home_and_loopback_only(tmp_path, monkeypatch):
    """Every test gets mktemp-HOME semantics and may connect only to loopback."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NAK_NET_DIR", raising=False)
    monkeypatch.delenv("NAK_SIGNER_DIR", raising=False)
    monkeypatch.delenv("NAK_PAY_RPC", raising=False)

    real_connect = socket.socket.connect

    def loopback_connect(sock, address):
        host = address[0] if isinstance(address, tuple) and address else ""
        if host not in {"127.0.0.1", "::1"}:
            pytest.fail(f"test attempted a non-loopback network connection to {address!r}")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_connect)
    return home


def _home() -> Path:
    return Path(os.environ["HOME"])


def _snapshot(root: Path) -> dict[str, tuple[int, int, bytes | str | None]]:
    result = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        data: bytes | str | None
        if path.is_symlink():
            data = os.readlink(path)
        elif stat.S_ISREG(info.st_mode):
            data = path.read_bytes()
        else:
            data = None
        result[path.relative_to(root).as_posix()] = (info.st_mode, info.st_mtime_ns, data)
    return result


def _identity_files(home: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(home).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(home.rglob("*"))
        if path.is_file() and ".nakshatra/pay" not in path.as_posix()
    }


def test_first_join_creates_wallet_and_second_join_keeps_seed_identical():
    home = _home()
    first = join_setup.ensure_identity(home)
    seed_path = home / ".nakshatra" / "pay" / "wallet.ed25519"
    public_path = seed_path.with_name("wallet.pub")
    custody_path = seed_path.with_name("custody")
    seed = seed_path.read_bytes()
    expected_public = (
        ed25519.Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw())

    assert len(seed) == 32
    assert stat.S_IMODE(seed_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(seed_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(public_path.stat().st_mode) == 0o644
    assert public_path.read_text().strip() == pay_wallet.base58_encode(expected_public)
    assert custody_path.read_text().strip() == "file"
    assert first["pay_address"] == public_path.read_text().strip()

    seed_mtime = seed_path.stat().st_mtime_ns
    second = join_setup.ensure_identity(home)
    assert seed_path.read_bytes() == seed
    assert seed_path.stat().st_mtime_ns == seed_mtime
    assert second["pay_address"] == first["pay_address"]


@pytest.mark.parametrize("kind", ["short", "symlink", "public-mode", "owner-exec-mode"])
def test_corrupt_seed_refuses_before_any_join_write(kind, tmp_path):
    home = _home()
    pay_dir = home / ".nakshatra" / "pay"
    pay_dir.mkdir(parents=True, mode=0o700)
    seed_path = pay_dir / "wallet.ed25519"
    if kind == "short":
        seed_path.write_bytes(b"x" * 31)
        seed_path.chmod(0o600)
    elif kind == "symlink":
        outside = tmp_path / "outside-seed"
        outside.write_bytes(b"x" * 32)
        outside.chmod(0o600)
        seed_path.symlink_to(outside)
    elif kind == "owner-exec-mode":
        seed_path.write_bytes(b"x" * 32)
        seed_path.chmod(0o700)
    else:
        seed_path.write_bytes(b"x" * 32)
        seed_path.chmod(0o644)
    before = _snapshot(tmp_path)

    with pytest.raises(SystemExit, match="refusing"):
        join_setup.ensure_identity(home)

    assert _snapshot(tmp_path) == before


def test_established_identity_adds_wallet_without_touching_identity_files():
    home = _home()
    join_setup.ensure_identity(home)
    shutil.rmtree(home / ".nakshatra" / "pay")
    before = _identity_files(home)

    result = join_setup.ensure_identity(home)

    assert pay_wallet.load_wallet(home)["address"] == result["pay_address"]
    assert _identity_files(home) == before


@pytest.mark.parametrize(("raw", "encoded"), [
    (b"", ""),
    (b"\0", "1"),
    (b"\0" * 32, "1" * 32),
    (b"hello world", "StV1DL6CwTryKyV"),
])
def test_base58_vectors(raw, encoded):
    assert pay_wallet.base58_encode(raw) == encoded
    assert pay_wallet.base58_decode(encoded) == raw


def test_existing_mismatched_public_address_is_never_rewritten():
    home = _home()
    pay_wallet.ensure_wallet(home)
    public_path = home / ".nakshatra" / "pay" / "wallet.pub"
    public_path.write_text("1" * 32 + "\n")
    before = public_path.read_bytes()

    with pytest.raises(SystemExit, match="does not match"):
        pay_wallet.ensure_wallet(home)

    assert public_path.read_bytes() == before


def test_nak_pay_address_and_status_are_offline(capsys):
    wallet = pay_wallet.ensure_wallet(_home())

    assert nak.main(["pay", "address"]) == 0
    assert capsys.readouterr().out.strip() == wallet["address"]
    assert nak.main(["pay", "status"]) == 0
    output = capsys.readouterr().out
    assert f"address: {wallet['address']}" in output
    assert "custody: file" in output
    assert "balance: set NAK_PAY_RPC to a devnet RPC to show balances" in output


@pytest.mark.parametrize(("stdin_tty", "stdout_tty"), [(False, True), (True, False)])
def test_nak_pay_export_refuses_without_input_and_output_ttys(
        stdin_tty, stdout_tty, monkeypatch, capsys):
    pay_wallet.ensure_wallet(_home())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin_tty)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout_tty)
    monkeypatch.setattr("builtins.input", lambda _prompt: "export")
    with pytest.raises(SystemExit, match="interactive-only"):
        nak.main(["pay", "export", "--keypair"])
    assert capsys.readouterr().out == ""


def test_nak_pay_export_prints_the_standard_64_byte_keypair_after_confirmation(
        monkeypatch, capsys):
    wallet = pay_wallet.ensure_wallet(_home())

    class Terminal:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", Terminal())
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "export")
    assert nak.main(["pay", "export", "--keypair"]) == 0
    exported = bytes(json.loads(capsys.readouterr().out))
    assert exported == wallet["seed"] + wallet["public_key"]
    assert len(exported) == 64


def test_bip39_wordlist_and_official_256_bit_vectors():
    wordlist = Path(pay_wallet.__file__).with_name("bip39_english.txt")
    assert hashlib.sha256(wordlist.read_bytes()).hexdigest() == (
        "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda")
    assert pay_wallet.bip39_mnemonic(bytes(32)) == " ".join(["abandon"] * 23 + ["art"])
    assert pay_wallet.bip39_mnemonic(bytes([0xff]) * 32) == " ".join(["zoo"] * 23 + ["vote"])


@pytest.mark.parametrize(("stdin_tty", "stdout_tty"), [(False, True), (True, False)])
def test_nak_pay_backup_refuses_without_input_and_output_ttys(
        stdin_tty, stdout_tty, monkeypatch, capsys):
    pay_wallet.ensure_wallet(_home())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin_tty)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout_tty)

    with pytest.raises(SystemExit, match="interactive-only"):
        nak.main(["pay", "backup"])
    assert capsys.readouterr().out == ""


def test_nak_pay_backup_prints_24_words_only_after_confirmation(monkeypatch, capsys):
    monkeypatch.setattr(pay_wallet.os, "urandom", lambda size: bytes(size))
    pay_wallet.ensure_wallet(_home())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "backup")

    assert nak.main(["pay", "backup"]) == 0

    assert capsys.readouterr().out.strip() == " ".join(["abandon"] * 23 + ["art"])


def test_nak_pay_backup_confirm_records_private_ack_without_seed(monkeypatch, capsys):
    wallet = pay_wallet.ensure_wallet(_home())
    mnemonic = pay_wallet.bip39_mnemonic(wallet["seed"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    assert nak.main(["pay", "backup", "--confirm"]) == 0

    output = capsys.readouterr().out
    assert mnemonic not in output
    ack = _home() / ".nakshatra" / "pay" / "backup-confirmed"
    assert ack.read_text() == "confirmed\n"
    assert stat.S_IMODE(ack.stat().st_mode) == 0o600


class _RPCServer:
    def __init__(self, genesis: str, balance: int = 0):
        self.methods = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib callback name
                size = int(self.headers.get("content-length", "0"))
                request = json.loads(self.rfile.read(size))
                owner.methods.append(request["method"])
                result = (genesis if request["method"] == "getGenesisHash"
                          else {"context": {"slot": 1}, "value": balance})
                body = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def test_nak_pay_status_reads_local_validator_balance(monkeypatch, capsys):
    pay_wallet.ensure_wallet(_home())
    rpc = _RPCServer("local-validator", balance=1_250_000_000)
    with rpc as url:
        monkeypatch.setenv("NAK_PAY_RPC", url)
        assert nak.main(["pay", "status"]) == 0

    assert rpc.methods == ["getGenesisHash", "getBalance"]
    assert "balance: 1.25 SOL" in capsys.readouterr().out


def test_nak_pay_status_refuses_mainnet_genesis_on_loopback(monkeypatch):
    pay_wallet.ensure_wallet(_home())
    rpc = _RPCServer(nak._SOLANA_MAINNET_GENESIS, balance=9_000_000_000)
    with rpc as url:
        monkeypatch.setenv("NAK_PAY_RPC", url)
        with pytest.raises(SystemExit, match="mainnet.*DEVNET-ONLY"):
            nak.main(["pay", "status"])

    assert rpc.methods == ["getGenesisHash"]
