"""File/broker wallet custody tests using an in-process fake and temporary homes."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from unittest import mock

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import client  # noqa: E402
import paid_admission  # noqa: E402
import paid_client  # noqa: E402
from network import (broker_transport, pay_autochannel, pay_chain, pay_voucher,  # noqa: E402
                     pay_wallet, pay_wallet_custody)


TOKEN = "wallet-broker-secret-token"
NODE_ID = "ab" * 32
BLOCKHASH = pay_chain.b58encode(bytes(range(32, 64)))


@pytest.fixture(autouse=True)
def _temp_home_loopback_only(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NAK_NET_DIR", raising=False)
    monkeypatch.delenv("NAK_SIGNER_DIR", raising=False)
    monkeypatch.delenv("NAK_PAY_MODE", raising=False)
    real_connect = socket.socket.connect

    def loopback_connect(sock, address):
        host = address[0] if isinstance(address, tuple) and address else ""
        if host not in {"127.0.0.1", "::1"}:
            pytest.fail(f"test attempted a non-loopback connection to {address!r}")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_connect)
    return home


def _home() -> Path:
    return Path(os.environ["HOME"])


class FakeBroker:
    def __init__(self, private: ed25519.Ed25519PrivateKey, *, status="approved",
                 corrupt="", lost_enqueue=False, lost_sign=False, enqueue_acks=None):
        self.private = private
        self.wallet_id = pay_wallet.base58_encode(private.public_key().public_bytes_raw())
        self.principal = f"nakshatra-wallet:{NODE_ID}"
        self.status = status
        self.corrupt = corrupt
        self.lost_enqueue = lost_enqueue
        self.lost_sign = lost_sign
        self.enqueues: list[dict] = []
        self.enqueue_attempts: list[dict] = []
        self.enqueue_attempt_bytes: list[bytes] = []
        self.enqueue_acks = list(enqueue_acks or [])
        self.sign_bodies: list[dict] = []
        self.by_key: dict[str, tuple[bytes, str]] = {}
        self.signed = False
        self.on_result = lambda: None
        self._patch = mock.patch.object(
            pay_wallet_custody, "_broker_json", side_effect=self.request)

    @property
    def url(self):
        return "http://127.0.0.1:1"

    def request(self, config, method, path, body=None, *, timeout=3,
                retry_transient=False):
        del timeout
        assert config.token == TOKEN
        if method == "GET" and path == "/principal/v1":
            return {"principal": self.principal}
        if method == "POST" and path == "/enqueue/v1":
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            self.enqueue_attempts.append(body)
            self.enqueue_attempt_bytes.append(json.dumps(body, separators=(",", ":")).encode())
            prior = self.by_key.get(body["idempotency_key"])
            if prior is not None and prior[0] != canonical:
                raise SystemExit(
                    "approval service error (HTTP 409, code idempotency-conflict)")
            if prior is None:
                request_id = f"{len(self.by_key) + 1:016x}"
                self.by_key[body["idempotency_key"]] = (canonical, request_id)
                self.enqueues.append(body)
            else:
                request_id = prior[1]
            if self.lost_enqueue:
                assert retry_transient
                self.lost_enqueue = False
                raise broker_transport.BrokerTransient(
                    "wallet broker is unavailable; nothing was signed")
            if self.enqueue_acks:
                acknowledgement = self.enqueue_acks.pop(0)
                if acknowledgement == "malformed-json":
                    raise SystemExit(pay_wallet_custody._INVALID_BROKER_DATA)
                return acknowledgement
            return {"request_id": request_id}
        if method == "GET" and path.startswith("/result/v1?request_id="):
            request_id = path.rsplit("=", 1)[1]
            self.on_result()
            result = {"status": self.status, "request_id": request_id}
            if self.corrupt == "result-request-id": result["request_id"] = "f" * 16
            return result
        assert method == "POST" and path == "/wallet-sign/v1"
        self.sign_bodies.append(body)
        if self.signed:
            if retry_transient:
                raise broker_transport.BrokerAlreadySigned
            raise SystemExit("approval service error (HTTP 409, code already-signed)")
        message = base64.b64decode(body["exact_content"]["message_b64"])
        signature = self.private.sign(message)
        response = {
            "request_id": body["request_id"],
            "wallet_id": self.wallet_id,
            "message_sha256": hashlib.sha256(message).hexdigest(),
            "signatures": [signature.hex()],
            "items_b64": [base64.b64encode(message).decode()],
        }
        if self.corrupt == "request-id": response["request_id"] = "f" * 16
        if self.corrupt == "wallet-id": response["wallet_id"] = "1" * 32
        if self.corrupt == "digest": response["message_sha256"] = "0" * 64
        if self.corrupt == "message": response["items_b64"] = [base64.b64encode(b"other").decode()]
        if self.corrupt == "signature": response["signatures"] = [(b"x" * 64).hex()]
        if self.corrupt == "count": response["items_b64"].append(response["items_b64"][0])
        self.signed = True
        if self.lost_sign:
            assert retry_transient
            self.lost_sign = False
            raise broker_transport.BrokerTransient(
                "wallet broker is unavailable; nothing was signed")
        return response

    def __enter__(self):
        self._patch.start()
        return self

    def __exit__(self, *_args):
        self._patch.stop()


def _broker_wallet(broker: FakeBroker) -> tuple[Path, bytes]:
    home = _home()
    pay_dir = home / ".nakshatra" / "pay"
    pay_dir.mkdir(parents=True, mode=0o700)
    (pay_dir / "wallet.pub").write_text(broker.wallet_id + "\n")
    (pay_dir / "custody").write_text("broker\n")
    signer = home / ".sthambha" / "signer"
    signer.mkdir(parents=True)
    node_public = signer / "node.pub"
    node_public.write_text(NODE_ID + "\n")
    node_public.chmod(0o644)
    net = home / ".nakshatra" / "net"
    net.mkdir(parents=True)
    token = net / "wallet-broker.token"
    token.write_text(TOKEN + "\n")
    token.chmod(0o600)
    config = net / "wallet-broker.json"
    config.write_text(json.dumps({
        "url": broker.url, "token_file": str(token),
    }))
    config.chmod(0o600)
    return home, broker.private.private_bytes_raw()


def _sign(broker: FakeBroker, message=b"compiled-solana-message", *, key="wallet:act-0001"):
    home, _seed = _broker_wallet(broker)
    return pay_wallet_custody.wallet_sign(
        message, broker.wallet_id, idempotency_key=key, home=home)


def test_broker_closed_request_and_signature_contract():
    private = ed25519.Ed25519PrivateKey.generate()
    with FakeBroker(private) as broker:
        message = b"compiled-solana-message"
        assert _sign(broker, message) == private.sign(message)
        assert broker.enqueues == [{
            "what": "nakshatra.wallet.v1",
            "idempotency_key": "wallet:act-0001",
            "tier": "high",
            "review": {"description": "Solana payment transaction"},
            "exact_content": {
                "v": 1, "wallet_id": broker.wallet_id,
                "message_b64": base64.b64encode(message).decode(),
            },
            "ttl": 60,
        }]
        assert broker.sign_bodies == [{
            "request_id": "0000000000000001",
            "exact_content": broker.enqueues[0]["exact_content"],
        }]


def test_wrong_dynamic_principal_refuses_before_enqueue():
    private = ed25519.Ed25519PrivateKey.generate()
    with FakeBroker(private) as broker:
        broker.principal = "nakshatra-wallet:" + "cd" * 32
        with pytest.raises(SystemExit, match="wrong principal"):
            _sign(broker)
        assert broker.enqueues == [] and broker.sign_bodies == []


def test_lost_enqueue_response_recovers_with_the_same_stable_key_and_content(monkeypatch):
    private = ed25519.Ed25519PrivateKey.generate()
    monkeypatch.setattr(pay_wallet_custody, "_BROKER_RETRY_INITIAL_S", 0)
    with FakeBroker(private, lost_enqueue=True) as broker:
        assert _sign(broker) == private.sign(b"compiled-solana-message")
        assert len(broker.enqueues) == 1
        assert len(broker.enqueue_attempts) == 2
        assert broker.enqueue_attempts[0] == broker.enqueue_attempts[1]


@pytest.mark.parametrize("first_ack", [
    {},
    "malformed-json",
    {"request_id": "not-a-request-id"},
])
def test_unusable_enqueue_ack_recovers_with_byte_identical_body_and_key(
        monkeypatch, first_ack):
    private = ed25519.Ed25519PrivateKey.generate()
    monkeypatch.setattr(pay_wallet_custody, "_BROKER_RETRY_INITIAL_S", 0)
    key = "wallet:stable-ack"
    with FakeBroker(private, enqueue_acks=[first_ack]) as broker:
        assert _sign(broker, key=key) == private.sign(b"compiled-solana-message")
        assert len(broker.enqueue_attempt_bytes) == 2
        assert broker.enqueue_attempt_bytes[0] == broker.enqueue_attempt_bytes[1]
        assert [body["idempotency_key"] for body in broker.enqueue_attempts] == [key, key]


def test_always_malformed_enqueue_ack_fails_uncertain_within_original_bound(
        monkeypatch):
    private = ed25519.Ed25519PrivateKey.generate()
    now = iter([0.0, 0.0, 0.25, 0.5, 0.75, 1.0])
    monkeypatch.setattr(pay_wallet_custody, "_TTL_S", 1)
    monkeypatch.setattr(pay_wallet_custody, "_BROKER_RETRY_INITIAL_S", 0)
    monkeypatch.setattr(pay_wallet_custody.time, "monotonic", lambda: next(now))
    key = "wallet:always-malformed"
    with FakeBroker(private, enqueue_acks=["malformed-json"] * 10) as broker:
        with pytest.raises(SystemExit, match="enqueue remained uncertain.*same idempotency_key"):
            _sign(broker, key=key)
        assert 1 < len(broker.enqueue_attempt_bytes) < 10
        assert len(set(broker.enqueue_attempt_bytes)) == 1
        assert {body["idempotency_key"] for body in broker.enqueue_attempts} == {key}
        assert len(broker.by_key) == 1


def test_same_key_with_different_content_is_a_closed_conflict():
    private = ed25519.Ed25519PrivateKey.generate()
    with FakeBroker(private) as broker:
        home, _seed = _broker_wallet(broker)
        pay_wallet_custody.wallet_sign(
            b"first", broker.wallet_id, idempotency_key="wallet:stable-1", home=home)
        broker.signed = False
        with pytest.raises(SystemExit, match="code idempotency-conflict"):
            pay_wallet_custody.wallet_sign(
                b"second", broker.wallet_id,
                idempotency_key="wallet:stable-1", home=home)
        assert len(broker.enqueues) == 1


@pytest.mark.parametrize("status", ["declined", "expired"])
def test_not_approved_never_calls_sign(status):
    private = ed25519.Ed25519PrivateKey.generate()
    with FakeBroker(private, status=status) as broker:
        with pytest.raises(SystemExit, match="not approved; nothing was signed"):
            _sign(broker)
        assert broker.sign_bodies == []


def test_consumed_result_requires_a_new_human_approval():
    private = ed25519.Ed25519PrivateKey.generate()
    with FakeBroker(private, status="consumed") as broker:
        with pytest.raises(SystemExit, match="new human approval"):
            _sign(broker)
        assert broker.sign_bodies == []


def test_consumed_with_lost_sign_response_refuses_instead_of_reenqueueing(monkeypatch):
    private = ed25519.Ed25519PrivateKey.generate()
    monkeypatch.setattr(pay_wallet_custody, "_BROKER_RETRY_INITIAL_S", 0)
    with FakeBroker(private, lost_sign=True) as broker:
        with pytest.raises(SystemExit, match="signature was lost in transit.*new human approval"):
            _sign(broker)
        assert len(broker.enqueues) == 1
        assert len(broker.sign_bodies) == 2


@pytest.mark.parametrize(("corrupt", "match"), [
    ("result-request-id", "substituted the approval request id"),
    ("request-id", "substituted the signing request id"),
    ("wallet-id", "substituted wallet_id"),
    ("message", "substituted the signed message"),
    ("digest", "wrong message digest"),
    ("signature", "wrong wallet signature"),
    ("count", "wrong number"),
])
def test_broker_echo_digest_signature_and_count_are_verified(corrupt, match):
    private = ed25519.Ed25519PrivateKey.generate()
    with FakeBroker(private, corrupt=corrupt) as broker:
        with pytest.raises(SystemExit, match=match):
            _sign(broker)


def test_custody_change_mid_act_refuses_without_file_fallback():
    private = ed25519.Ed25519PrivateKey.generate()
    with FakeBroker(private) as broker:
        home, _seed = _broker_wallet(broker)
        broker.on_result = lambda: (
            home / ".nakshatra/pay/custody").write_text("file\n")
        with pytest.raises(SystemExit, match="changed from broker to file"):
            pay_wallet_custody.wallet_sign(
                b"message", broker.wallet_id,
                idempotency_key="wallet:act-change", home=home)
        assert broker.sign_bodies == []


def test_broker_setup_rerun_and_paid_admission_need_no_seed(monkeypatch, tmp_path):
    private = ed25519.Ed25519PrivateKey.generate()
    with FakeBroker(private) as broker:
        home, _seed = _broker_wallet(broker)
        before = {
            path.relative_to(home): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in home.rglob("*") if path.is_file()
        }
        identity = pay_wallet.ensure_wallet(home)
        after = {
            path.relative_to(home): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in home.rglob("*") if path.is_file()
        }
        assert identity == {
            "public_key": private.public_key().public_bytes_raw(),
            "address": broker.wallet_id, "custody": "broker",
        }
        assert before == after
        assert not (home / ".nakshatra/pay/wallet.ed25519").exists()
        with pytest.raises(SystemExit, match="broker custody.*export is unavailable"):
            pay_wallet.load_wallet(home)

        monkeypatch.setenv("NAK_PAY_MODE", "chain")
        store = paid_admission.PaidAdmissionStore(tmp_path / "paid.sqlite")
        ledger = type("Ledger", (), {"worker_keyid": "cd" * 32})()
        manager = paid_admission.PaidAdmission(
            ledger, store, bytes(range(32)), model_id="model", layer_start=0,
            layer_end=1, voucher_store=object(), channel_state=object(),
            pay_program="program")
        assert manager.pay_wallet_address == broker.wallet_id
        store.close()


def test_client_chain_voucher_setup_needs_only_broker_public_identity(
        monkeypatch, tmp_path):
    private = ed25519.Ed25519PrivateKey.generate()
    broker = FakeBroker(private)
    home, _seed = _broker_wallet(broker)
    worker_seed = bytes.fromhex("11" * 32)
    worker_key = ed25519.Ed25519PrivateKey.from_private_bytes(
        worker_seed).public_key().public_bytes_raw().hex()
    worker_path = home / ".nakshatra" / "keys" / "worker.ed25519"
    worker_path.parent.mkdir(parents=True)
    worker_path.write_bytes(worker_seed)
    worker_path.chmod(0o600)
    config = tmp_path / "chain.yaml"
    config.write_text(
        "model: {id: model}\nworkers:\n"
        "  - id: one\n"
        f"    public_key_hex: {worker_key}\n"
        f"    peer_spki_hash: {'a' * 64}\n"
        "    address: 127.0.0.1\n    port: 5540\n"
        "    layer_range: [0, 2]\n    mode: solo\n")
    envelope = paid_client.make_paid_envelope(
        claim_token="A" * 43, hold_id="hold", session_id="session",
        model_id="model", max_prompt_tokens=3, max_output_tokens=2,
        plan=[{"worker_keyid": worker_key, "layer_start": 0, "layer_end": 2}])
    admission_read, admission_write = os.pipe()
    progress_read, progress_write = os.pipe()
    os.write(admission_write, json.dumps(envelope).encode())
    os.close(admission_write)

    monkeypatch.setenv("NAK_PAY_MODE", "chain")
    monkeypatch.setattr(client, "_TLS_AVAILABLE", True)
    monkeypatch.setattr(client, "_open_chain_channel", lambda *_args: object())
    monkeypatch.setattr(client.pb_grpc, "NakshatraStub", lambda _channel: object())

    def info(_stub, request, timeout):
        assert timeout == 10.0
        signature = ed25519.Ed25519PrivateKey.from_private_bytes(worker_seed).sign(
            paid_client.INFO_ATTESTATION_DOMAIN + request.identity_challenge)
        return client.pb.InfoResponse(
            protocol_version="0.1.0", layer_start=0, layer_end=2,
            hidden_size=8, has_token_embd=True, has_lm_head=True,
            protocol_capabilities=list(paid_client.REQUIRED_PAID_CAPABILITIES),
            worker_keyid=worker_key, identity_signature=signature)

    monkeypatch.setattr(client, "info_with_startup_wait", info)
    monkeypatch.setattr(client, "tokenize_local", lambda *_args: ([1, 2, 3], object()))
    monkeypatch.setattr(pay_voucher, "configured_program", lambda: "program")
    monkeypatch.setattr(pay_voucher, "configured_state", lambda: "state")
    monkeypatch.setattr(pay_chain, "configured_rpc_url", lambda _env: "loopback-fake")
    monkeypatch.setattr(
        pay_chain, "PayRpc",
        lambda _url: type("Rpc", (), {"chain_time": lambda self: 1})())
    book = type("Book", (), {"close": lambda self: None})()
    monkeypatch.setattr(pay_voucher, "PayerChannelBook", lambda: book)
    owners = []

    def ensure_channel(*_args, **kwargs):
        owners.append(kwargs["owner"])
        return "channel"

    monkeypatch.setattr(pay_autochannel, "ensure_channel", ensure_channel)
    monkeypatch.setattr(
        pay_voucher, "ConsumerVoucherManager",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after setup")))
    monkeypatch.setattr(sys, "argv", [
        "client.py", "--config", str(config), "--model-id", "model",
        "--model-path", "/unused.gguf", "--prompt", "prompt",
        "--max-tokens", "2", "--use-streaming",
        "--paid-admission-fd", str(admission_read),
        "--paid-progress-fd", str(progress_write),
    ])
    try:
        with pytest.raises(SystemExit, match="stop after setup"):
            client.main()
    finally:
        os.close(progress_read)
        os.close(progress_write)
    assert owners == [broker.wallet_id]


def test_file_custody_signature_and_compile_once_assembly_match_sign_transaction(monkeypatch):
    home = _home()
    wallet = pay_wallet.ensure_wallet(home)
    instruction = pay_chain.system_transfer(wallet["address"], wallet["address"], 0)
    compiled = pay_chain.compile_message([instruction], wallet["address"], BLOCKHASH)
    signature = pay_wallet_custody.wallet_sign(
        compiled.data, wallet["address"], idempotency_key="wallet:file-0001", home=home)
    assembled = pay_chain.assemble_transaction(compiled.data, signature)
    reference = pay_chain.sign_transaction(
        [instruction], wallet["address"], BLOCKHASH,
        {wallet["address"]: wallet["seed"]})
    assert signature == reference.signatures[0]
    assert assembled == reference


def test_logs_and_exceptions_never_contain_seed_or_bearer_token(caplog):
    private = ed25519.Ed25519PrivateKey.generate()
    with FakeBroker(private, corrupt="signature") as broker:
        _home_path, seed = _broker_wallet(broker)
        with pytest.raises(SystemExit) as caught:
            pay_wallet_custody.wallet_sign(
                b"message", broker.wallet_id,
                idempotency_key="wallet:no-leaks", home=_home_path)
        rendered = str(caught.value)
        assert TOKEN not in rendered
        assert seed.hex() not in rendered
        assert TOKEN not in caplog.text
        assert seed.hex() not in caplog.text


@pytest.mark.parametrize("key", ["", "short", " leading-key", "wallet/key/not-allowed"])
def test_idempotency_key_must_match_the_broker_contract(key):
    private = ed25519.Ed25519PrivateKey.generate()
    with FakeBroker(private) as broker:
        home, _seed = _broker_wallet(broker)
        with pytest.raises(SystemExit, match="idempotency_key is missing or malformed"):
            pay_wallet_custody.wallet_sign(
                b"message", broker.wallet_id, idempotency_key=key, home=home)
        assert broker.enqueues == []
