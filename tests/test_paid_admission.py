from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import grpc
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nakshatra_auth as auth  # noqa: E402
import nakshatra_grpc_auth as grpc_auth  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402
import paid_admission as paid  # noqa: E402
import worker  # noqa: E402


TOKEN = "A" * 43
PAYER = "b" * 64
MODEL = "paid-test-model"


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()


def _claim(worker_keyid: str, claim_instance: str = "instance-1", *,
           session_id: str = "paid-session", token: str = TOKEN,
           server_time: int = 100, max_output_tokens: int = 16) -> dict:
    response = {
        "v": 1,
        "hold_id": "hold-1",
        "token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
        "claim_instance": claim_instance,
        "payer_keyid": PAYER,
        "payer_account_at_authorize": "payer-account",
        "worker_keyid": worker_keyid,
        "worker_account_at_authorize": "worker-account",
        "billable": True,
        "session_id": session_id,
        "model_id": MODEL,
        "billing_mode": "greedy-direct-v1",
        "layer_start": 0,
        "layer_end": 4,
        "price_multiplier": 2,
        "max_prompt_tokens": 16,
        "max_output_tokens": max_output_tokens,
        "max_token_layers": (16 + max_output_tokens) * 4,
        "max_cost": (16 + max_output_tokens) * 8,
        "execution_deadline": 1000,
        "settlement_deadline": 87400,
        "server_time": server_time,
    }
    response["claim_response_sha256"] = hashlib.sha256(
        _canonical(response)).hexdigest()
    return response


class _Ledger:
    def __init__(self, worker_keyid: str):
        self.worker_keyid = worker_keyid
        self.claim_calls = []
        self.settle_calls = []
        self.fail_settle = False

    def probe(self):
        return {
            "v": 1, "ready": True,
            "capability": "spend_online_claim_v1", "server_time": 100,
        }

    def claim(self, token: str, claim_instance: str):
        self.claim_calls.append((hashlib.sha256(token.encode()).hexdigest(),
                                 claim_instance))
        return _claim(self.worker_keyid, claim_instance)

    def settle(self, receipt: dict, signature: str):
        if self.fail_settle:
            raise paid.PaidAdmissionUnavailable("ledger unavailable")
        self.settle_calls.append((dict(receipt), signature))
        return {
            "v": 1,
            "hold_id": receipt["hold_id"],
            "receipt_id": receipt["receipt_id"],
            "accepted_delta": 0,
            "cumulative_charge": receipt["cumulative_cost"],
            "replayed": False,
            "state": "final" if receipt["final"] else "claimed",
        }


def _manager(tmp_path: Path, *, ledger=None, clock=lambda: 200.0):
    private, public = auth.generate_keypair()
    store = paid.PaidAdmissionStore(tmp_path / "paid.sqlite3")
    ledger = ledger or _Ledger(public)
    manager = paid.PaidAdmission(
        ledger, store, private, model_id=MODEL,
        layer_start=0, layer_end=4, clock=clock)
    assert manager.check_ready()
    return manager, ledger, private, public


def test_claim_store_is_0600_and_claim_survives_restart(tmp_path):
    manager, ledger, private, public = _manager(tmp_path)
    claim = manager.claim(PAYER, "paid-session", TOKEN)
    first_instance = ledger.claim_calls[0][1]
    persisted = b"".join(
        path.read_bytes() for path in tmp_path.iterdir()
        if path.name.startswith("paid.sqlite3"))
    assert TOKEN.encode("ascii") not in persisted
    manager.store.close()

    assert stat.S_IMODE(os.stat(tmp_path / "paid.sqlite3").st_mode) == 0o600
    store = paid.PaidAdmissionStore(tmp_path / "paid.sqlite3")
    restarted = paid.PaidAdmission(
        ledger, store, private, model_id=MODEL,
        layer_start=0, layer_end=4, clock=lambda: 201)
    assert restarted.check_ready()
    assert restarted.claim(PAYER, "paid-session") == claim
    assert ledger.claim_calls == [(hashlib.sha256(TOKEN.encode()).hexdigest(),
                                   first_instance)]
    store.close()


def test_spend_client_uses_exact_n3_routes_and_request_schemas():
    private, public = auth.generate_keypair()
    seen = []

    class Response:
        def __init__(self, value):
            self.raw = _canonical(value)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self.raw

    def opener(request, *, timeout, expected_spki_hash):
        path = request.full_url.removeprefix("https://ledger.example")
        body = json.loads(request.data) if request.data is not None else None
        seen.append((request.get_method(), path, body, expected_spki_hash))
        if path == "/spend/ready":
            return Response({
                "v": 1, "ready": True,
                "capability": "spend_online_claim_v1", "server_time": 100,
            })
        if path == "/spend/claim":
            return Response(_claim(
                public, body["claim_instance"], token=body["claim_token"]))
        assert path == "/spend/settle"
        receipt = body["receipt"]
        return Response({
            "v": 1, "hold_id": receipt["hold_id"],
            "receipt_id": receipt["receipt_id"], "accepted_delta": 8,
            "cumulative_charge": 8, "replayed": False, "state": "claimed",
        })

    client = paid.SpendLedgerClient(
        "https://ledger.example", "1" * 64, private, public, opener=opener)
    client.probe()
    claim = client.claim(TOKEN, "stable-instance")
    receipt = {"hold_id": claim["hold_id"], "receipt_id": "sha256:" + "2" * 64}
    client.settle(receipt, "3" * 128)
    assert [(method, path) for method, path, _body, _pin in seen] == [
        ("GET", "/spend/ready"),
        ("POST", "/spend/claim"),
        ("POST", "/spend/settle"),
    ]
    assert seen[1][2] == {
        "claim_token": TOKEN, "claim_instance": "stable-instance"}
    assert seen[2][2] == {"receipt": receipt, "signature": "3" * 128}
    assert all(pin == "1" * 64 for _method, _path, _body, pin in seen)


@pytest.mark.parametrize("field", [
    "v", "layer_start", "layer_end", "price_multiplier",
    "max_prompt_tokens", "max_output_tokens", "max_token_layers", "max_cost",
    "execution_deadline", "settlement_deadline", "server_time",
])
@pytest.mark.parametrize("replacement", [True, False])
def test_every_claim_integer_field_rejects_booleans(field, replacement):
    private, public = auth.generate_keypair()
    client = paid.SpendLedgerClient(
        "https://ledger.example", "1" * 64, private, public,
        opener=lambda *_args, **_kwargs: None)
    response = _claim(public, "stable-instance")
    response[field] = replacement
    with pytest.raises(paid.PaidAdmissionUnavailable):
        client._validate_claim(response, TOKEN, "stable-instance")


def test_receipts_match_strict_n2_schema_id_and_signature(tmp_path):
    now = [200]
    manager, ledger, private, public = _manager(
        tmp_path, clock=lambda: now[0])
    claim = manager.claim(PAYER, "paid-session", TOKEN)
    duplicate, prompt, generated = manager.prepare_step(
        claim, "step-0", "1" * 64, 3)
    assert (duplicate, prompt, generated) == (False, 3, 1)
    first = manager.commit_step(claim, "step-0", "1" * 64, prompt, generated, 1)
    now[0] += 15
    duplicate, prompt, generated = manager.prepare_step(
        claim, "step-1", "2" * 64, 1)
    assert (duplicate, prompt, generated) == (False, 0, 1)
    second = manager.commit_step(claim, "step-1", "2" * 64, prompt, generated, 1)
    final = manager.finalize(claim, "normal")

    assert set(first) == paid.USAGE_FIELDS
    assert (first["cumulative_prompt_tokens"], first["cumulative_generated_tokens"]) == (3, 1)
    assert (second["cumulative_prompt_tokens"], second["cumulative_generated_tokens"]) == (3, 2)
    assert final["final"] is True
    assert final["sequence"] == 3
    assert final["cumulative_cost"] == second["cumulative_cost"]
    for receipt in (first, second, final):
        fields = {k: v for k, v in receipt.items() if k != "receipt_id"}
        assert receipt["receipt_id"] == "sha256:" + hashlib.sha256(
            paid.RECEIPT_DOMAIN + _canonical(fields)).hexdigest()
    for row in manager.store.pending():
        receipt = json.loads(row["receipt_json"])
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public)).verify(
            bytes.fromhex(row["signature"]), _canonical(receipt))
    assert manager.flush_outbox() == 3
    assert manager.flush_outbox() == 0
    assert len(ledger.settle_calls) == 3
    manager.store.close()


def test_outbox_replays_once_after_restart_and_ledger_partition(tmp_path):
    manager, ledger, private, public = _manager(tmp_path)
    claim = manager.claim(PAYER, "paid-session", TOKEN)
    _, prompt, generated = manager.prepare_step(
        claim, "step-0", "3" * 64, 2)
    manager.commit_step(claim, "step-0", "3" * 64, prompt, generated, 1)
    manager.finalize(claim, "cancelled")
    ledger.fail_settle = True
    assert manager.flush_outbox() == 0
    manager.store.close()

    ledger.fail_settle = False
    store = paid.PaidAdmissionStore(tmp_path / "paid.sqlite3")
    restarted = paid.PaidAdmission(
        ledger, store, private, model_id=MODEL,
        layer_start=0, layer_end=4, clock=lambda: 300)
    assert restarted.check_ready()
    assert restarted.flush_outbox() == 2
    assert restarted.flush_outbox() == 0
    assert len(ledger.settle_calls) == 2
    store.close()


def test_paid_session_cannot_continue_after_restart_without_resident_kv(tmp_path):
    manager, ledger, private, public = _manager(tmp_path)
    claim = manager.claim(PAYER, "paid-session", TOKEN)
    _, prompt, generated = manager.prepare_step(
        claim, "step-0", "4" * 64, 2)
    manager.commit_step(claim, "step-0", "4" * 64, prompt, generated, 1)
    manager.store.close()

    store = paid.PaidAdmissionStore(tmp_path / "paid.sqlite3")
    restarted = paid.PaidAdmission(
        ledger, store, private, model_id=MODEL,
        layer_start=0, layer_end=4, clock=lambda: 300)
    assert restarted.check_ready()
    daemon = _Daemon()
    servicer = worker.WorkerServicer(
        daemon, "last", 0, 4, MODEL, paid_admission=restarted,
        refuse_unregistered_peers=False, refuse_unpinned_peers=False)
    servicer._check_grpc_auth = lambda *_args, **_kwargs: (
        worker.AuthenticatedKeyId(PAYER, grpc_auth.ROLE_CONSUMER))
    step = pb.InferenceStep(
        session_id="paid-session", step_id="step-1", prefix_length=2)
    step.token_ids.ids.append(42)
    context = _Context()
    response = list(servicer.Inference(iter([step]), context))
    assert len(response) == 1
    assert b"KV_RESIDENCY_LOST" in response[0].error
    assert context._code == grpc.StatusCode.FAILED_PRECONDITION
    assert daemon.calls == []
    store.close()


class _Daemon:
    def __init__(self):
        self.calls = []
        self._owner = None

    def info(self):
        return {"n_embd": 4, "n_layers": 4, "gpu_offload_status": {}}

    def acquire_session(self, principal, stale_after_s=None):
        if self._owner is None or self._owner == principal:
            self._owner = principal
            return True
        return False

    def release_session(self, principal):
        if self._owner == principal:
            self._owner = None

    @property
    def current_owner(self):
        return self._owner

    def call(self, cmd, n_tokens, payload, start_pos=0, flags=0):
        self.calls.append((cmd, n_tokens, start_pos, flags))
        return 0, struct.pack("<Ii", 0, 42)


class _Context:
    def __init__(self, metadata=None):
        self._code = None
        self._details = ""
        self._metadata = metadata or []

    def invocation_metadata(self):
        return self._metadata

    def peer(self):
        return "ipv4:127.0.0.1:1"

    def set_code(self, code):
        self._code = code

    def set_details(self, details):
        self._details = details

    def abort(self, code, details):
        self._code = code
        self._details = details
        raise RuntimeError(details)


def _gateway_paid_client():
    path = (Path(__file__).resolve().parents[2] / "nak-g1" /
            "scripts" / "paid_client.py")
    if not path.is_file():
        pytest.skip("gateway paid_client.py is not available")
    spec = importlib.util.spec_from_file_location("gateway_paid_client", path)
    if spec is None or spec.loader is None:
        pytest.skip("gateway paid_client.py is not importable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pytest.skip(f"gateway paid_client.py dependency unavailable: {exc}")
    return module


def test_info_identity_attestation_interoperates_with_gateway_verifier():
    private, public = auth.generate_keypair()
    servicer = worker.WorkerServicer(
        _Daemon(), "last", 0, 4, MODEL,
        worker_priv=private, worker_pub_hex=public)
    challenge = bytes(range(32))
    response = servicer.Info(
        pb.InfoRequest(identity_challenge=challenge), _Context())

    assert response.worker_keyid == public
    assert len(response.identity_signature) == 64
    ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public)).verify(
        response.identity_signature,
        b"nakshatra-paid-info-v1\0" + challenge)
    assert _gateway_paid_client().verify_info_attestation(
        response, challenge) == public


def test_empty_info_challenge_is_byte_identical_to_legacy_probe():
    private, public = auth.generate_keypair()
    keyed = worker.WorkerServicer(
        _Daemon(), "last", 0, 4, MODEL,
        worker_priv=private, worker_pub_hex=public)
    legacy = worker.WorkerServicer(_Daemon(), "last", 0, 4, MODEL)

    keyed_response = keyed.Info(pb.InfoRequest(), _Context())
    legacy_response = legacy.Info(pb.InfoRequest(), _Context())
    assert keyed_response.SerializeToString() == legacy_response.SerializeToString()
    assert keyed_response.worker_keyid == ""
    assert keyed_response.identity_signature == b""


@pytest.mark.parametrize("challenge", [b"x", b"x" * 31, b"x" * 33])
def test_info_identity_rejects_malformed_nonempty_challenge(challenge):
    private, public = auth.generate_keypair()
    servicer = worker.WorkerServicer(
        _Daemon(), "last", 0, 4, MODEL,
        worker_priv=private, worker_pub_hex=public)
    context = _Context()
    with pytest.raises(RuntimeError, match="exactly 32 bytes"):
        servicer.Info(pb.InfoRequest(identity_challenge=challenge), context)
    assert context._code == grpc.StatusCode.INVALID_ARGUMENT


def test_info_identity_requires_available_matching_worker_key():
    private, public = auth.generate_keypair()
    other_private, _other_public = auth.generate_keypair()
    for servicer in (
        worker.WorkerServicer(_Daemon(), "last", 0, 4, MODEL),
        worker.WorkerServicer(
            _Daemon(), "last", 0, 4, MODEL,
            worker_priv=other_private, worker_pub_hex=public),
    ):
        context = _Context()
        with pytest.raises(RuntimeError, match="identity is unavailable"):
            servicer.Info(
                pb.InfoRequest(identity_challenge=b"x" * 32), context)
        assert context._code == grpc.StatusCode.FAILED_PRECONDITION


def test_new_paid_client_rejects_old_worker_without_truncate_capability():
    gateway = _gateway_paid_client()
    old_info = SimpleNamespace(protocol_capabilities=[
        capability for capability in gateway.REQUIRED_PAID_CAPABILITIES
        if capability != "truncate_session_owner"
    ])
    with pytest.raises(
            gateway.PaidAdmissionError, match="truncate_session_owner"):
        gateway.require_paid_capabilities(
            [({"id": "old-worker"}, object(), old_info)])


def _servicer(tmp_path: Path, *, role=grpc_auth.ROLE_CONSUMER):
    manager, ledger, _private, _public = _manager(tmp_path)
    daemon = _Daemon()
    servicer = worker.WorkerServicer(
        daemon, "last", 0, 4, MODEL,
        paid_admission=manager,
        refuse_unregistered_peers=False,
        refuse_unpinned_peers=False,
    )
    identity = worker.AuthenticatedKeyId(PAYER, role)
    servicer._check_grpc_auth = lambda *_args, **_kwargs: identity
    return servicer, daemon, manager, ledger


def _step(*, token=TOKEN):
    step = pb.InferenceStep(
        session_id="paid-session", step_id="step-0",
        spend_claim_token=token)
    step.token_ids.ids.extend([1, 2, 3])
    return step


@pytest.mark.parametrize("mutation", [
    "missing_token", "invalid_token", "chain", "next_server", "pushed",
    "all_logits", "fabric",
])
def test_every_paid_admission_rejection_has_zero_daemon_calls(tmp_path, mutation):
    servicer, daemon, manager, _ledger = _servicer(tmp_path)
    step = _step()
    if mutation == "missing_token":
        step.spend_claim_token = ""
    elif mutation == "invalid_token":
        step.spend_claim_token = "short"
    elif mutation == "chain":
        step.chain.add(address="peer:1", session_id="downstream")
    elif mutation == "next_server":
        step.next_server.address = "peer:1"
    elif mutation == "pushed":
        step.pushed = True
    elif mutation == "all_logits":
        step.all_logits = True
    elif mutation == "fabric":
        servicer.fabric_transport_enabled = True
    context = _Context()
    assert list(servicer.Inference(iter([step]), context)) == []
    assert context._code == grpc.StatusCode.PERMISSION_DENIED
    assert daemon.calls == []
    manager.store.close()


def test_paid_stream_meters_prompt_and_output_and_finalizes(tmp_path):
    servicer, daemon, manager, ledger = _servicer(tmp_path)
    first = _step()
    second = pb.InferenceStep(
        session_id="paid-session", step_id="step-1", prefix_length=3)
    second.token_ids.ids.append(42)
    responses = list(servicer.Inference(iter([first, second]), _Context()))
    assert [list(response.token_ids.ids) for response in responses] == [[42], [42]]
    meter = manager.store.meter(PAYER, "paid-session")
    assert (meter["prompt_tokens"], meter["generated_tokens"]) == (3, 2)
    assert meter["receipt_sequence"] == 2
    assert meter["final"] == 1
    assert len(daemon.calls) == 2
    assert len(ledger.settle_calls) == 2
    manager.store.close()


@pytest.mark.parametrize("mode", ["first", "middle", "last", "solo"])
def test_prefill_meters_first_emitted_token_on_every_stage(tmp_path, mode):
    manager, _ledger, _private, _public = _manager(tmp_path)
    daemon = _Daemon()
    servicer = worker.WorkerServicer(
        daemon, mode, 0, 4, MODEL, paid_admission=manager,
        refuse_unregistered_peers=False, refuse_unpinned_peers=False)
    servicer._check_grpc_auth = lambda *_args, **_kwargs: (
        worker.AuthenticatedKeyId(PAYER, grpc_auth.ROLE_CONSUMER))

    responses = list(servicer.Inference(iter([_step()]), _Context()))
    meter = manager.store.meter(PAYER, "paid-session")

    assert len(responses) == 1
    assert meter["prompt_tokens"] == 3
    assert meter["generated_tokens"] == 1
    assert meter["final"] == 1
    manager.store.close()


def test_receipts_checkpoint_after_eight_generated_tokens(tmp_path):
    manager, _ledger, _private, _public = _manager(tmp_path)
    claim = manager.claim(PAYER, "paid-session", TOKEN)
    emitted = []
    for index in range(9):
        n_tokens = 3 if index == 0 else 1
        _, prompt, generated = manager.prepare_step(
            claim, f"step-{index}", f"{index + 1:064x}", n_tokens)
        receipt = manager.commit_step(
            claim, f"step-{index}", f"{index + 1:064x}",
            prompt, generated, 1)
        if receipt is not None:
            emitted.append(receipt)
    final = manager.finalize(claim, "normal")

    assert [item["cumulative_generated_tokens"] for item in emitted] == [1, 9]
    assert [item["sequence"] for item in emitted] == [1, 2]
    assert final["sequence"] == 3
    assert final["cumulative_generated_tokens"] == 9
    manager.store.close()


def test_same_paid_session_is_serialized_through_finalization(tmp_path):
    class BlockingDaemon(_Daemon):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def call(self, cmd, n_tokens, payload, start_pos=0, flags=0):
            self.calls.append((cmd, n_tokens, start_pos, flags))
            self.entered.set()
            assert self.release.wait(timeout=5)
            return 0, struct.pack("<Ii", 0, 42)

    manager, ledger, _private, _public = _manager(tmp_path)
    daemon = BlockingDaemon()
    servicer = worker.WorkerServicer(
        daemon, "last", 0, 4, MODEL, paid_admission=manager,
        refuse_unregistered_peers=False, refuse_unpinned_peers=False)
    servicer._check_grpc_auth = lambda *_args, **_kwargs: (
        worker.AuthenticatedKeyId(PAYER, grpc_auth.ROLE_CONSUMER))
    first = pb.ForwardRequest(
        hidden_in=struct.pack("<2i", 1, 2), n_tokens=2,
        has_token_ids=True, session_id="paid-session",
        step_id="forward-0", spend_claim_token=TOKEN)
    second = pb.ForwardRequest(
        hidden_in=struct.pack("<2i", 3, 4), n_tokens=2,
        has_token_ids=True, session_id="paid-session", step_id="forward-1")
    contexts = [_Context(), _Context()]
    responses = [None, None]
    errors = []

    def invoke(index, request):
        try:
            responses[index] = servicer.Forward(request, contexts[index])
        except Exception as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    winner = threading.Thread(target=invoke, args=(0, first))
    loser = threading.Thread(target=invoke, args=(1, second))
    winner.start()
    assert daemon.entered.wait(timeout=5)
    loser.start()
    loser.join(timeout=0.1)
    assert loser.is_alive()
    assert len(daemon.calls) == 1
    daemon.release.set()
    winner.join(timeout=5)
    loser.join(timeout=5)

    assert not errors
    assert responses[0].hidden_out == struct.pack("<i", 42)
    assert responses[1] == pb.ForwardResponse()
    assert contexts[1]._code == grpc.StatusCode.PERMISSION_DENIED
    assert "already final" in contexts[1]._details
    assert len(daemon.calls) == 1
    assert len(ledger.settle_calls) == 2
    manager.store.close()


def test_late_daemon_completion_releases_no_output_or_usage(tmp_path):
    now = [999.0]

    class AdvancingDaemon(_Daemon):
        def call(self, cmd, n_tokens, payload, start_pos=0, flags=0):
            result = super().call(cmd, n_tokens, payload, start_pos, flags)
            if len(self.calls) == 2:
                now[0] = 1001.0
            return result

    manager, ledger, _private, _public = _manager(
        tmp_path, clock=lambda: now[0])
    daemon = AdvancingDaemon()
    servicer = worker.WorkerServicer(
        daemon, "last", 0, 4, MODEL, paid_admission=manager,
        refuse_unregistered_peers=False, refuse_unpinned_peers=False)
    servicer._check_grpc_auth = lambda *_args, **_kwargs: (
        worker.AuthenticatedKeyId(PAYER, grpc_auth.ROLE_CONSUMER))
    first = _step()
    second = pb.InferenceStep(
        session_id="paid-session", step_id="step-1", prefix_length=3)
    second.token_ids.ids.append(42)
    context = _Context()

    responses = list(servicer.Inference(iter([first, second]), context))
    meter = manager.store.meter(PAYER, "paid-session")
    settled = [receipt for receipt, _signature in ledger.settle_calls]

    assert [list(response.token_ids.ids) for response in responses] == [[42]]
    assert context._code == grpc.StatusCode.DEADLINE_EXCEEDED
    assert "completed after" in context._details
    assert len(daemon.calls) == 2
    assert (meter["prompt_tokens"], meter["generated_tokens"]) == (3, 1)
    assert meter["final"] == 1
    assert [receipt["final"] for receipt in settled] == [False, True]
    assert [receipt["work_completed_at"] for receipt in settled] == [999, 999]
    assert [receipt["cumulative_generated_tokens"] for receipt in settled] == [1, 1]
    manager.store.close()


def test_paid_forward_is_stateless_billed_and_stateful_rejection_is_cold(tmp_path):
    servicer, daemon, manager, ledger = _servicer(tmp_path)
    request = pb.ForwardRequest(
        hidden_in=struct.pack("<2i", 1, 2), n_tokens=2,
        has_token_ids=True, session_id="paid-session",
        step_id="forward-0", spend_claim_token=TOKEN)
    response = servicer.Forward(request, _Context())
    assert response.hidden_out == struct.pack("<i", 42)
    assert len(daemon.calls) == 1
    assert len(ledger.settle_calls) == 2
    manager.store.close()

    other = tmp_path / "stateful"
    other.mkdir()
    servicer, daemon, manager, _ledger = _servicer(other)
    request.keep_kv = True
    context = _Context()
    assert servicer.Forward(request, context) == pb.ForwardResponse()
    assert context._code == grpc.StatusCode.PERMISSION_DENIED
    assert daemon.calls == []
    manager.store.close()


def test_request_digest_excludes_claim_token_only():
    one = _step(token="A" * 43)
    two = _step(token="B" * 43)
    assert paid.PaidAdmission.request_digest(one) == paid.PaidAdmission.request_digest(two)
    two.token_ids.ids.append(4)
    assert paid.PaidAdmission.request_digest(one) != paid.PaidAdmission.request_digest(two)


def _projection(value):
    response = MagicMock()
    response.read.return_value = json.dumps(value).encode()
    response.__enter__ = lambda self: self
    response.__exit__ = lambda *_args: None
    return response


def test_consumer_resolution_is_default_off_and_peer_overlap_wins():
    peer_public = "c" * 64
    consumer_public = "d" * 64
    default_resolver = grpc_auth.PillarPeerKeyResolver("https://pillar")
    with patch.object(
            grpc_auth.auth, "open_pillar_url",
            return_value=_projection({"peers": []})) as opened:
        default_resolver.refresh_once()
    assert opened.call_count == 1
    assert default_resolver.resolve(consumer_public) is None

    resolver = grpc_auth.PillarPeerKeyResolver(
        "https://pillar", enable_consumers=True)
    with patch.object(
            grpc_auth.auth, "open_pillar_url",
            side_effect=[
                _projection({"peers": [{
                    "node_id": peer_public,
                    "public_key_hex": peer_public,
                    "address": "worker:5500",
                }]}),
                _projection({"readers": [peer_public, consumer_public]}),
            ]):
        resolver.refresh_once()
    assert resolver.resolve_identity(peer_public).role == grpc_auth.ROLE_PEER
    assert resolver.resolve_identity(consumer_public).role == grpc_auth.ROLE_CONSUMER
    assert resolver.known_node_ids() == {peer_public}


def test_consumer_authority_is_narrowly_inference_only(tmp_path):
    private, public = auth.generate_keypair()

    class Resolver:
        def resolve(self, keyid):
            return public if keyid == public else None

        def resolve_identity(self, keyid):
            if keyid != public:
                return None
            return grpc_auth.ResolvedIdentity(
                public, public, grpc_auth.ROLE_CONSUMER)

    daemon = _Daemon()
    servicer = worker.WorkerServicer(
        daemon, "last", 0, 4, MODEL, auth_required=True,
        peer_resolver=Resolver(), refuse_unregistered_peers=False,
        refuse_unpinned_peers=False)
    body = b"first-frame"
    allowed = {
        "/nakshatra.Nakshatra/Forward": False,
        "/nakshatra.Nakshatra/Inference": True,
        "/nakshatra.Nakshatra/TruncateKV": False,
    }
    for method, streaming in allowed.items():
        header = grpc_auth.build_grpc_auth_header(
            private, public, method, body, is_streaming=streaming)
        identity = servicer._check_grpc_auth(
            _Context([("authorization", header)]), body,
            method_path=method, is_streaming=streaming)
        assert str(identity) == public
        assert identity.role == grpc_auth.ROLE_CONSUMER

    denied = (
        "/nakshatra.Nakshatra/SignParticipation",
        "/nakshatra.Nakshatra/Sleep",
        "/nakshatra.Nakshatra/Wake",
    )
    for method in denied:
        header = grpc_auth.build_grpc_auth_header(
            private, public, method, body)
        with pytest.raises(RuntimeError, match="not authorized"):
            servicer._check_grpc_auth(
                _Context([("authorization", header)]), body,
                method_path=method, is_streaming=False)
