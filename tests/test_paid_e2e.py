"""Real-loopback payment admission integration tests.

These tests intentionally keep only the model engine fake.  Neuron's N3
ledger service, signed HTTPS clients, the gateway, client.py, worker
servicers, gRPC auth, TLS/SPKI pinning, durable outboxes, and SQLite stores
are the production implementations.
"""
from __future__ import annotations

import hashlib
import json
import os
import socketserver
import ssl
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent import futures
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

import grpc
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
NEURON_PYTHON = REPO.parents[1] / "neuron" / "python"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(NEURON_PYTHON))

import ledger_client  # noqa: E402
import nakshatra_auth as auth  # noqa: E402
import nakshatra_grpc_auth as grpc_auth  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402
import nakshatra_pb2_grpc as pb_grpc  # noqa: E402
import nakshatra_serve as serve  # noqa: E402
import nakshatra_tls as tls  # noqa: E402
import paid_admission  # noqa: E402
import worker  # noqa: E402
from core.ledger_store import SPEND_USAGE_FIELDS  # noqa: E402
from ledger_server import V2LedgerService, _make_v2_handler  # noqa: E402


MODEL = "paid-e2e-model"
PROMPT_TOKENS = 3
OUTPUT_TOKENS = 2
TOTAL_LAYERS = 4
FUNDED = 10_000


@dataclass(frozen=True)
class _Identity:
    private: bytes
    public: str
    operator: str
    role: str


def _identity(label: str, operator: str, role: str) -> _Identity:
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()
    )
    private = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return _Identity(private, public, operator, role)


def _write_consumer_key(home: Path, identity: _Identity) -> Path:
    path = home / ".nakshatra" / "keys" / "worker.ed25519"
    path.parent.mkdir(parents=True, mode=0o700)
    path.write_bytes(identity.private)
    path.chmod(0o600)
    return path


class _RosterResolver:
    """Small immutable roster adapter for the production gRPC verifier."""

    def __init__(self, identities: list[_Identity]):
        self._identities = {item.public: item for item in identities}

    def resolve(self, keyid: str):
        item = self._identities.get(keyid)
        return item.public if item is not None else None

    def resolve_identity(self, keyid: str):
        item = self._identities.get(keyid)
        if item is None:
            return None
        return grpc_auth.ResolvedIdentity(item.public, item.public, item.role)

    def stats(self):
        return {"cached_identities": len(self._identities)}


class _DeterministicDaemon:
    """Daemon protocol double: deterministic tensors/tokens, real servicer."""

    def __init__(self, mode: str, on_first_call=None):
        self.mode = mode
        self.calls = []
        self._on_first_call = on_first_call
        self._owner = None
        self._lock = threading.Lock()

    def info(self):
        return {"n_embd": 1, "n_layers": TOTAL_LAYERS, "gpu_offload_status": {}}

    def acquire_session(self, principal, stale_after_s=None):
        del stale_after_s
        with self._lock:
            if self._owner is None or self._owner == principal:
                self._owner = principal
                return True
            return False

    def release_session(self, principal):
        with self._lock:
            if self._owner == principal:
                self._owner = None

    @property
    def current_owner(self):
        with self._lock:
            return self._owner

    def mark_step_served(self):
        return None

    def call(self, cmd, n_tokens, payload, start_pos=0, flags=0):
        self.calls.append((cmd, n_tokens, bytes(payload), start_pos, flags))
        if len(self.calls) == 1 and self._on_first_call is not None:
            self._on_first_call()
        if self.mode in ("last", "solo"):
            token_id = 64 + len(self.calls)  # A, B, ...
            return 0, struct.pack("<Ii", 0, token_id)
        hidden = struct.pack(
            f"<{n_tokens}f", *(float(start_pos + i + 1) for i in range(n_tokens))
        )
        return 0, struct.pack("<I", 0) + hidden


class _TokenizerHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            request = json.loads(line)
            if request.get("op") == "tokenize":
                response = {"ids": [1, 2, 3]}
            elif request.get("op") == "detokenize":
                response = {
                    "text": "".join(
                        chr(int(token))
                        for token in request.get("ids", [])
                        if 32 <= int(token) <= 126
                    )
                }
            elif request.get("op") == "ping":
                response = {"ok": True}
            else:
                response = {"error": "unknown operation"}
            self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")
            self.wfile.flush()


class _TokenizerServer:
    def __init__(self, path: Path):
        self.path = path
        self.server = socketserver.ThreadingUnixStreamServer(
            str(path), _TokenizerHandler
        )
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class _PinnedLedgerServer:
    def __init__(self, service: V2LedgerService, tls_dir: Path):
        self.service = service
        self.cert, self.key = tls.generate_self_signed_cert(
            "nakshatra.local", tls_dir
        )
        self.spki = tls.compute_spki_hash(self.cert)
        self.port = 0
        self.server = None
        self.thread = None
        self.start()

    @property
    def url(self):
        return f"https://127.0.0.1:{self.port}"

    def start(self):
        if self.server is not None:
            return
        server = ThreadingHTTPServer(
            ("127.0.0.1", self.port), _make_v2_handler(self.service)
        )
        server.daemon_threads = True
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(self.cert, self.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        self.port = int(server.server_address[1])
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        server, thread = self.server, self.thread
        if server is None:
            return
        self.server = self.thread = None
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    def close(self):
        self.stop()


@dataclass
class _WorkerServer:
    identity: _Identity
    daemon: _DeterministicDaemon
    servicer: worker.WorkerServicer
    server: grpc.Server
    port: int
    spki: str
    manager: paid_admission.PaidAdmission | None

    def close(self):
        self.server.stop(grace=0).wait(timeout=5)
        if self.manager is not None:
            self.manager.store.close()


def _start_worker(
    root: Path,
    identity: _Identity,
    resolver: _RosterResolver,
    *,
    mode: str,
    layer_start: int,
    layer_end: int,
    ledger: _PinnedLedgerServer | None,
    on_first_call=None,
) -> _WorkerServer:
    root.mkdir(parents=True, exist_ok=True)
    daemon = _DeterministicDaemon(mode, on_first_call=on_first_call)
    manager = None
    if ledger is not None:
        store = paid_admission.PaidAdmissionStore(root / "paid.sqlite3")
        ledger_client = paid_admission.SpendLedgerClient(
            ledger.url,
            ledger.spki,
            identity.private,
            identity.public,
            timeout=0.5,
        )
        manager = paid_admission.PaidAdmission(
            ledger_client,
            store,
            identity.private,
            model_id=MODEL,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        assert manager.check_ready(), manager.readiness_error
    servicer = worker.WorkerServicer(
        daemon,
        mode,
        layer_start,
        layer_end,
        MODEL,
        peer_resolver=resolver,
        auth_required=True,
        paid_admission=manager,
        worker_priv=identity.private,
        worker_pub_hex=identity.public,
        refuse_unregistered_peers=False,
        refuse_unpinned_peers=True,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_NakshatraServicer_to_server(servicer, server)
    cert, key = tls.generate_self_signed_cert(
        "nakshatra.local", root / "tls"
    )
    spki = tls.compute_spki_hash(cert)
    port = server.add_secure_port(
        "127.0.0.1:0", tls.build_grpc_server_credentials(cert, key)
    )
    assert port > 0
    server.start()
    return _WorkerServer(identity, daemon, servicer, server, port, spki, manager)


class _E2EBackend(serve.ChainChatBackend):
    def __init__(self, tokenizer_socket: Path):
        super().__init__(str(SCRIPTS), timeout_s=20, paid_admission=True)
        self._test_tokenizer_socket = str(tokenizer_socket)

    def _tokenizer_socket(self, tokenizer_gguf: str) -> str:
        del tokenizer_gguf
        return self._test_tokenizer_socket


@dataclass
class _PaidStack:
    consumer: _Identity
    worker_identities: list[_Identity]
    workers: list[_WorkerServer]
    service: V2LedgerService
    ledger: _PinnedLedgerServer
    backend: _E2EBackend
    gateway: ThreadingHTTPServer
    gateway_thread: threading.Thread
    tokenizer: _TokenizerServer
    chain_path: Path

    @property
    def gateway_url(self):
        return f"http://127.0.0.1:{self.gateway.server_address[1]}"

    def close(self):
        self.gateway.shutdown()
        self.gateway.server_close()
        self.gateway_thread.join(timeout=5)
        for item in self.workers:
            item.close()
        self.ledger.close()
        self.service.close()
        self.tokenizer.close()


@contextmanager
def _paid_stack(
    tmp_path: Path,
    monkeypatch,
    spans: list[tuple[int, int, str]],
    *,
    dial_identity_as: int | None = None,
    stop_ledger_on_compute: bool = False,
):
    consumer = _identity("paid-e2e-consumer", "consumer", grpc_auth.ROLE_CONSUMER)
    identity_count = max(
        len(spans), (dial_identity_as + 1 if dial_identity_as is not None else 1)
    )
    worker_identities = [
        _identity(f"paid-e2e-worker-{index}", f"worker-{index}", grpc_auth.ROLE_PEER)
        for index in range(identity_count)
    ]
    roster = tmp_path / "roster.tsv"
    roster.write_text(
        "\n".join(
            f"{item.public}\t{item.operator}\t{item.operator}\t"
            f"{'trusted' if item is consumer else 'known'}\ttest"
            for item in [consumer, *worker_identities]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NAK_SPEND_ROUTES", "on")
    service = V2LedgerService(
        tmp_path / "ledger.sqlite3", roster, start_sweeper=False
    )
    service.store.deposit("paid-e2e-funding", consumer.operator, FUNDED, 1)
    ledger = _PinnedLedgerServer(service, tmp_path / "ledger-tls")

    home = tmp_path / "home"
    home.mkdir()
    key_path = _write_consumer_key(home, consumer)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAKSHATRA_LEDGER_MODE", "enforce")
    monkeypatch.setenv("NAKSHATRA_LEDGER_URL", ledger.url)
    monkeypatch.setenv("NAKSHATRA_LEDGER_SPKI_SHA256", ledger.spki)
    monkeypatch.setenv("NAKSHATRA_LEDGER_TIMEOUT", "0.5")
    monkeypatch.setenv("NAKSHATRA_CHAIN_TLS_MODE", "required")
    monkeypatch.setenv("NAKSHATRA_PAID_STATUS_TIMEOUT", "3")
    monkeypatch.setenv("NAKSHATRA_PAID_STATUS_INTERVAL", "0.01")
    monkeypatch.setenv("NKS_REGISTRY_KEYID", consumer.public)
    monkeypatch.setenv("NKS_STREAM_SPEC", "0")
    monkeypatch.setenv("NKS_ASYNC_PIPELINE", "0")
    monkeypatch.setenv("NAKSHATRA_SPECULATIVE", "0")
    monkeypatch.delenv("NAKSHATRA_ACT_QUANT", raising=False)
    monkeypatch.setattr(auth, "WORKER_KEY_PATH", key_path)

    resolver = _RosterResolver([consumer, *worker_identities])
    workers = []
    callback = ledger.stop if stop_ledger_on_compute else None
    tokenizer = None
    gateway = None
    gateway_thread = None
    try:
        for index, (layer_start, layer_end, mode) in enumerate(spans):
            workers.append(
                _start_worker(
                    tmp_path / f"worker-{index}",
                    worker_identities[index],
                    resolver,
                    mode=mode,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    ledger=ledger,
                    on_first_call=callback if index == 0 else None,
                )
            )
        chain_workers = []
        for index, item in enumerate(workers):
            declared = (
                worker_identities[dial_identity_as]
                if dial_identity_as is not None
                else item.identity
            )
            layer_start, layer_end, mode = spans[index]
            chain_workers.append(
                {
                    "id": f"stage-{index}",
                    "address": "127.0.0.1",
                    "port": item.port,
                    "layer_range": [layer_start, layer_end],
                    "mode": mode,
                    "peer_spki_hash": item.spki,
                    "public_key_hex": declared.public,
                }
            )
        chain_path = tmp_path / "chain.yaml"
        chain_path.write_text(
            yaml.safe_dump(
                {"model": {"id": MODEL}, "workers": chain_workers},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        tokenizer = _TokenizerServer(tmp_path / "tokenizer.sock")
        backend = _E2EBackend(tokenizer.path)
        backend._ledger_hook_cached = ledger_client.LedgerHook(
            url=ledger.url,
            enabled=True,
            mode="enforce",
            spki_hash=ledger.spki,
        )
        entry = serve.ModelEntry(
            name=MODEL,
            tokenizer_gguf=str(tmp_path / "fake-tokenizer.gguf"),
            chain_yaml=str(chain_path),
            num_layers=TOTAL_LAYERS,
            hidden_size=1,
        )
        gateway = serve.build_server(
            "127.0.0.1", 0, {MODEL: entry}, backend
        )
        gateway_thread = threading.Thread(
            target=gateway.serve_forever, daemon=True
        )
        gateway_thread.start()
        yield _PaidStack(
            consumer,
            worker_identities,
            workers,
            service,
            ledger,
            backend,
            gateway,
            gateway_thread,
            tokenizer,
            chain_path,
        )
    finally:
        if gateway is not None:
            gateway.shutdown()
            gateway.server_close()
            gateway_thread.join(timeout=5)
        for item in workers:
            item.close()
        ledger.close()
        service.close()
        if tokenizer is not None:
            tokenizer.close()


def _chat(base_url: str):
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": OUTPUT_TOKENS,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read()), time.monotonic()
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), time.monotonic()


def _only_hold(service: V2LedgerService) -> dict:
    row = service.store._db.execute(
        "SELECT hold_id FROM spend_holds ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    return service.store.spend_status(row["hold_id"])


def _assert_zero_sum(service: V2LedgerService):
    total = service.store._db.execute(
        "SELECT COALESCE(SUM(balance), 0) FROM accounts"
    ).fetchone()[0]
    assert total == 0


@pytest.mark.parametrize(
    "spans",
    [
        [(0, TOTAL_LAYERS, "solo")],
        [(0, 2, "first"), (2, TOTAL_LAYERS, "last")],
    ],
    ids=["single-stage", "two-stage"],
)
def test_paid_happy_path_settles_before_release_and_preserves_value(
    tmp_path, monkeypatch, spans
):
    with _paid_stack(tmp_path, monkeypatch, spans) as stack:
        settled_at = []
        original = stack.service.settle_spend

        def observed_settle(*args, **kwargs):
            result = original(*args, **kwargs)
            if result["state"] == "settled":
                settled_at.append(time.monotonic())
            return result

        stack.service.settle_spend = observed_settle
        status, response, response_at = _chat(stack.gateway_url)

        assert status == 200
        assert response["choices"][0]["message"]["content"] == "AB"
        assert response["usage"] == {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": OUTPUT_TOKENS,
            "total_tokens": PROMPT_TOKENS + OUTPUT_TOKENS,
        }
        assert settled_at and max(settled_at) <= response_at

        audit = response["nakshatra"]["audit_receipt"]
        hold = stack.service.store.spend_status(audit["hold_id"])
        expected_total = (PROMPT_TOKENS + OUTPUT_TOKENS) * TOTAL_LAYERS
        assert hold["state"] == "settled"
        assert hold["escrow_balance"] == 0
        assert response["nakshatra"]["credits_charged"] == expected_total
        assert stack.service.store.balance(stack.consumer.operator) == FUNDED - expected_total
        for index, (start, end, _mode) in enumerate(spans):
            expected_share = (PROMPT_TOKENS + OUTPUT_TOKENS) * (end - start)
            assert stack.service.store.balance(f"worker-{index}") == expected_share
        _assert_zero_sum(stack.service)

        rows = stack.service.store._db.execute(
            "SELECT receipt_json FROM spend_usage_receipts ORDER BY rowid"
        ).fetchall()
        assert rows
        receipts = [json.loads(row["receipt_json"]) for row in rows]
        assert all(set(item) == SPEND_USAGE_FIELDS for item in receipts)
        assert all(item["v"] == 2 for item in receipts)
        assert all(item["kind"] == "nakshatra-worker-usage" for item in receipts)
        assert {item["worker_keyid"] for item in receipts} == {
            stack.worker_identities[index].public for index in range(len(spans))
        }


def test_paid_runtime_worker_identity_mismatch_never_computes_and_releases_hold(
    tmp_path, monkeypatch
):
    # Dial worker 0 while authorizing worker 1's key for the same span.
    with _paid_stack(
        tmp_path,
        monkeypatch,
        [(0, TOTAL_LAYERS, "solo")],
        dial_identity_as=1,
    ) as stack:
        status, response, _ = _chat(stack.gateway_url)
        hold = _only_hold(stack.service)

        assert status == 503
        assert "AB" not in json.dumps(response)
        assert stack.workers[0].daemon.calls == []
        assert hold["state"] == "settled"
        assert hold["terminal_at"] is not None
        assert [entry["state"] for entry in hold["workers"]] == ["abandoned"]
        assert hold["escrow_balance"] == 0
        assert stack.service.store.balance(stack.consumer.operator) == FUNDED
        _assert_zero_sum(stack.service)


def test_paid_ledger_partition_after_claim_hides_text_then_outbox_settles(
    tmp_path, monkeypatch
):
    with _paid_stack(
        tmp_path,
        monkeypatch,
        [(0, TOTAL_LAYERS, "solo")],
        stop_ledger_on_compute=True,
    ) as stack:
        status, response, _ = _chat(stack.gateway_url)
        assert status == 503
        assert "AB" not in json.dumps(response)
        assert len(stack.workers[0].daemon.calls) == OUTPUT_TOKENS

        manager = stack.workers[0].manager
        pending = manager.store._db.execute(
            "SELECT COUNT(*) FROM outbox WHERE state='pending'"
        ).fetchone()[0]
        assert pending >= 1

        stack.ledger.start()
        assert manager.flush_outbox() == pending
        hold = _only_hold(stack.service)
        assert int(time.time()) < hold["settlement_deadline"]
        assert hold["state"] == "settled"
        assert hold["escrow_balance"] == 0
        assert manager.store._db.execute(
            "SELECT COUNT(*) FROM outbox WHERE state='pending'"
        ).fetchone()[0] == 0
        _assert_zero_sum(stack.service)


def test_consumer_direct_worker_call_without_claim_token_is_cold_rejected(
    tmp_path, monkeypatch
):
    with _paid_stack(
        tmp_path, monkeypatch, [(0, TOTAL_LAYERS, "solo")]
    ) as stack:
        endpoint = f"127.0.0.1:{stack.workers[0].port}"
        channel = tls.open_pinned_channel(endpoint, stack.workers[0].spki)
        try:
            stub = pb_grpc.NakshatraStub(channel)
            step = pb.InferenceStep(session_id="direct", step_id="step-0")
            step.token_ids.ids.extend([1, 2, 3])
            header = grpc_auth.build_grpc_auth_header(
                stack.consumer.private,
                stack.consumer.public,
                "/nakshatra.Nakshatra/Inference",
                step.SerializeToString(),
                is_streaming=True,
            )
            with pytest.raises(grpc.RpcError) as rejected:
                list(stub.Inference(iter([step]), metadata=[("authorization", header)]))
            assert rejected.value.code() == grpc.StatusCode.PERMISSION_DENIED
            assert stack.workers[0].daemon.calls == []
        finally:
            channel.close()


def test_paid_gate_off_keeps_legacy_peer_response_bytes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    caller = _identity("paid-e2e-peer", "peer", grpc_auth.ROLE_PEER)
    serving = _identity("paid-e2e-serving", "serving", grpc_auth.ROLE_PEER)
    resolver = _RosterResolver([caller, serving])
    server = _start_worker(
        tmp_path / "legacy-worker",
        serving,
        resolver,
        mode="solo",
        layer_start=0,
        layer_end=TOTAL_LAYERS,
        ledger=None,
    )
    endpoint = f"127.0.0.1:{server.port}"
    channel = tls.open_pinned_channel(endpoint, server.spki)
    try:
        assert server.servicer.paid_gate_enabled is False
        stub = pb_grpc.NakshatraStub(channel)
        step = pb.InferenceStep(session_id="legacy", step_id="step-0")
        step.token_ids.ids.extend([1, 2, 3])
        header = grpc_auth.build_grpc_auth_header(
            caller.private,
            caller.public,
            "/nakshatra.Nakshatra/Inference",
            step.SerializeToString(),
            is_streaming=True,
        )
        responses = list(
            stub.Inference(iter([step]), metadata=[("authorization", header)])
        )
        expected = pb.InferenceStep(
            session_id="legacy", step_id="step-0", prefix_length=3
        )
        expected.token_ids.ids.append(65)
        assert [item.SerializeToString() for item in responses] == [
            expected.SerializeToString()
        ]
    finally:
        channel.close()
        server.close()
