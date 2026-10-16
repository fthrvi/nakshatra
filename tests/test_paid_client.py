from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ledger_client  # noqa: E402
import client  # noqa: E402
import nakshatra_serve as serve  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402
import paid_client as paid  # noqa: E402


PRIV1 = bytes.fromhex("11" * 32)
PRIV2 = bytes.fromhex("22" * 32)


def _public_key(private: bytes) -> str:
    return ed25519.Ed25519PrivateKey.from_private_bytes(private).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


KEY1 = _public_key(PRIV1)
KEY2 = _public_key(PRIV2)
PAYER = "c" * 64
SPKI1 = "d" * 64
SPKI2 = "e" * 64


def _plan():
    return [
        {"worker_keyid": KEY1, "layer_start": 0, "layer_end": 2},
        {"worker_keyid": KEY2, "layer_start": 2, "layer_end": 4},
    ]


def _request():
    return paid.authorize_request(
        client_nonce="nonce", session_id="session", model_id="model",
        prompt_tokens=3, output_tokens=2, plan=_plan())


def _authorization():
    return {
        "v": 1,
        "claim_token": "A" * 43,
        "hold_id": "hold",
        "max_token_layers": 20,
        "max_cost": 20,
        "execution_deadline": 200,
        "settlement_deadline": 300,
        "workers": [
            {"worker_keyid": KEY1, "billable": True,
             "max_token_layers": 10, "max_cost": 10},
            {"worker_keyid": KEY2, "billable": True,
             "max_token_layers": 10, "max_cost": 10},
        ],
    }


def _status(*, state="settled", final=True, include_second=True):
    workers = []
    for key, start, end in ((KEY1, 0, 2), (KEY2, 2, 4)):
        workers.append({
            "worker_keyid": key,
            "worker_account_at_authorize": "worker",
            "billable": True,
            "layer_start": start,
            "layer_end": end,
            "price_multiplier": 1,
            "max_prompt_tokens": 3,
            "max_output_tokens": 2,
            "max_token_layers": 10,
            "max_cost": 10,
            "state": "final" if final else "claimed",
            "claimed_at": 110,
            "cumulative_prompt_tokens": 3,
            "cumulative_generated_tokens": 2,
            "cumulative_token_layers": 10,
            "cumulative_cost": 10,
            "final": final,
            "final_receipt_id": "sha256:" + key[0] * 64 if final else None,
            "claim_response_sha256": "d" * 64,
        })
    if not include_second:
        workers.pop()
    return {
        "v": 1,
        "hold_id": "hold",
        "payer_keyid": PAYER,
        "payer_account_at_authorize": "payer",
        "session_id": "session",
        "model_id": "model",
        "billing_mode": "greedy-direct-v1",
        "max_prompt_tokens": 3,
        "max_output_tokens": 2,
        "max_token_layers": 20,
        "max_cost": 20,
        "state": state,
        "release_requested": True,
        "authorized_at": 100,
        "execution_deadline": 200,
        "settlement_deadline": 300,
        "terminal_at": 150 if state in {"settled", "expired", "cancelled", "disputed"} else None,
        "escrow_balance": 0 if state in {"settled", "expired", "cancelled", "disputed"} else 20,
        "workers": workers,
    }


def test_exact_planner_and_runtime_capabilities_are_byte_equal():
    workers = [
        {"public_key_hex": KEY2, "layer_range": [2, 4]},
        {"public_key_hex": KEY1, "layer_range": [0, 2]},
    ]
    plan = paid.exact_chain_plan(workers)
    assert plan == _plan()
    caps = list(paid.REQUIRED_PAID_CAPABILITIES)
    stubs = [
        ({"_paid_attested_worker_keyid": KEY1}, object(), SimpleNamespace(
            layer_start=0, layer_end=2, protocol_capabilities=caps)),
        ({"_paid_attested_worker_keyid": KEY2}, object(), SimpleNamespace(
            layer_start=2, layer_end=4, protocol_capabilities=caps)),
    ]
    paid.require_exact_runtime_plan(plan, stubs)
    paid.require_paid_capabilities(stubs)
    stubs[1][2].layer_start = 3
    with pytest.raises(paid.PaidAdmissionError, match="topology changed"):
        paid.require_exact_runtime_plan(plan, stubs)


def test_authorize_a_serve_b_is_rejected_from_attested_key_before_compute():
    challenge = bytes(range(paid.INFO_CHALLENGE_BYTES))
    signature = ed25519.Ed25519PrivateKey.from_private_bytes(PRIV2).sign(
        paid.INFO_ATTESTATION_DOMAIN + challenge)
    info = SimpleNamespace(
        worker_keyid=KEY2, identity_signature=signature,
        layer_start=0, layer_end=2,
        protocol_capabilities=list(paid.REQUIRED_PAID_CAPABILITIES),
    )
    proven = paid.verify_info_attestation(info, challenge)
    assert proven == KEY2
    runtime = [({"_paid_attested_worker_keyid": proven}, object(), info)]
    with pytest.raises(paid.PaidAdmissionError, match="topology changed"):
        paid.require_exact_runtime_plan(
            [{"worker_keyid": KEY1, "layer_start": 0, "layer_end": 2}],
            runtime,
        )


def test_paid_client_production_path_authorize_a_serve_b_exits_before_compute(
        tmp_path, monkeypatch):
    config = tmp_path / "chain.yaml"
    config.write_text(
        "model: {id: model}\nworkers:\n"
        f"  - id: one\n    public_key_hex: {KEY1}\n"
        f"    peer_spki_hash: {SPKI1}\n"
        "    address: 127.0.0.1\n    port: 5540\n"
        "    layer_range: [0, 2]\n    mode: solo\n")
    envelope = paid.make_paid_envelope(
        claim_token="A" * 43, hold_id="hold", session_id="session",
        model_id="model", max_prompt_tokens=3, max_output_tokens=2,
        plan=[{"worker_keyid": KEY1, "layer_start": 0, "layer_end": 2}],
    )
    admission_read, admission_write = os.pipe()
    progress_read, progress_write = os.pipe()
    os.write(admission_write, json.dumps(envelope).encode())
    os.close(admission_write)
    opened = []

    monkeypatch.setattr(client, "_TLS_AVAILABLE", True)
    monkeypatch.setattr(
        client, "_open_chain_channel",
        lambda address, pin, mode: opened.append((address, pin, mode)) or object())
    monkeypatch.setattr(client.pb_grpc, "NakshatraStub", lambda _channel: object())

    def info(_stub, request, timeout):
        assert timeout == 10.0
        signature = ed25519.Ed25519PrivateKey.from_private_bytes(PRIV2).sign(
            paid.INFO_ATTESTATION_DOMAIN + request.identity_challenge)
        return pb.InfoResponse(
            protocol_version="0.1.0", layer_start=0, layer_end=2,
            hidden_size=8, has_token_embd=True, has_lm_head=True,
            protocol_capabilities=list(paid.REQUIRED_PAID_CAPABILITIES),
            worker_keyid=KEY2, identity_signature=signature,
        )

    monkeypatch.setattr(client, "info_with_startup_wait", info)
    monkeypatch.setattr(
        client, "tokenize_local",
        lambda *_args, **_kwargs: pytest.fail("identity mismatch reached compute"))
    monkeypatch.setattr(sys, "argv", [
        "client.py", "--config", str(config), "--model-id", "model",
        "--model-path", "/unused.gguf", "--prompt", "prompt",
        "--max-tokens", "2", "--use-streaming",
        "--paid-admission-fd", str(admission_read),
        "--paid-progress-fd", str(progress_write),
    ])
    try:
        with pytest.raises(SystemExit, match="pre-compute admission refused"):
            client.main()
    finally:
        os.close(progress_read)
        os.close(progress_write)
    assert opened == [("127.0.0.1:5540", SPKI1, "required")]


def test_info_attestation_rejects_wrong_signature_and_unattested_runtime():
    challenge = bytes(range(paid.INFO_CHALLENGE_BYTES))
    wrong = ed25519.Ed25519PrivateKey.from_private_bytes(PRIV1).sign(
        paid.INFO_ATTESTATION_DOMAIN + challenge)
    info = SimpleNamespace(worker_keyid=KEY2, identity_signature=wrong)
    with pytest.raises(paid.PaidAdmissionError, match="signature"):
        paid.verify_info_attestation(info, challenge)
    with pytest.raises(paid.PaidAdmissionError, match="topology changed"):
        paid.require_exact_runtime_plan(
            [{"worker_keyid": KEY1, "layer_start": 0, "layer_end": 2}],
            [({}, object(), SimpleNamespace(layer_start=0, layer_end=2))],
        )


def test_paid_mode_refuses_push_and_non_greedy_locally():
    assert paid.paid_mode_refusal(
        push=True, speculative=True, stream_spec=False,
        async_pipeline=False, eagle=False) == "push, non-greedy"


def test_paid_client_refuses_alternate_failover_before_worker_contact(
        tmp_path, monkeypatch):
    config = tmp_path / "chain.yaml"
    config.write_text(
        "model: {id: model}\nworkers:\n"
        f"  - id: one\n    public_key_hex: {KEY1}\n"
        f"    peer_spki_hash: {SPKI1}\n"
        "    address: 127.0.0.1\n    port: 5540\n"
        "    layer_range: [0, 2]\n    mode: solo\n"
        "    alternates:\n"
        f"      - id: backup\n        public_key_hex: {KEY2}\n"
        f"        peer_spki_hash: {SPKI2}\n"
        "        address: 127.0.0.1\n        port: 5541\n"
        "        layer_range: [0, 2]\n        mode: solo\n"
    )
    envelope = paid.make_paid_envelope(
        claim_token="A" * 43, hold_id="hold", session_id="session",
        model_id="model", max_prompt_tokens=3, max_output_tokens=2,
        plan=[{"worker_keyid": KEY1, "layer_start": 0, "layer_end": 2}],
    )
    admission_read, admission_write = os.pipe()
    progress_read, progress_write = os.pipe()
    os.write(admission_write, json.dumps(envelope).encode())
    os.close(admission_write)
    monkeypatch.setattr(client, "_TLS_AVAILABLE", True)
    monkeypatch.setattr(
        client, "_open_chain_channel",
        lambda *_args, **_kwargs: pytest.fail("alternate plan reached worker contact"))
    monkeypatch.setattr(sys, "argv", [
        "client.py", "--config", str(config), "--model-id", "model",
        "--model-path", "/unused.gguf", "--prompt", "prompt",
        "--max-tokens", "2", "--use-streaming",
        "--paid-admission-fd", str(admission_read),
        "--paid-progress-fd", str(progress_write),
    ])
    try:
        with pytest.raises(SystemExit, match="alternate failover"):
            client.main()
    finally:
        os.close(progress_read)
        os.close(progress_write)


def test_proto_field_numbers_match_the_worker_merge_contract():
    info_request = pb.InfoRequest.DESCRIPTOR.fields_by_name
    info_response = pb.InfoResponse.DESCRIPTOR.fields_by_name
    forward = pb.ForwardRequest.DESCRIPTOR.fields_by_name
    inference = pb.InferenceStep.DESCRIPTOR.fields_by_name
    truncate = pb.TruncateRequest.DESCRIPTOR.fields_by_name
    assert (forward["session_id"].number, forward["step_id"].number,
            forward["spend_claim_token"].number) == (9, 10, 11)
    assert inference["spend_claim_token"].number == 18
    assert truncate["session_id"].number == 2
    assert info_request["identity_challenge"].number == 1
    assert (info_response["worker_keyid"].number,
            info_response["identity_signature"].number) == (13, 14)


def test_direct_step_places_claim_token_only_in_the_explicit_field():
    class Stream:
        worker_id = "worker"

        def __init__(self):
            self.request = None

        def step(self, request):
            self.request = request
            response = pb.InferenceStep()
            response.token_ids.ids.append(7)
            return response

    stream = Stream()
    result = client.call_inference_step(
        stream, (5).to_bytes(4, "little", signed=True), 1, True,
        session_id="session", step_idx=4, prefix_length=0,
        spend_claim_token="opaque")
    assert int.from_bytes(result, "little", signed=True) == 7
    assert stream.request.session_id == "session"
    assert stream.request.step_id == "step-4"
    assert stream.request.spend_claim_token == "opaque"


def test_authorize_and_status_schemas_are_exact():
    request = _request()
    authorization = paid.validate_authorization(_authorization(), request)
    status = paid.validate_status(
        _status(), request=request, authorization=authorization)
    paid.require_releasable_status(status, used={KEY1, KEY2})
    bad = _authorization() | {"unknown": True}
    with pytest.raises(paid.PaidAdmissionError, match="unknown"):
        paid.validate_authorization(bad, request)


AUTHORIZE_INTEGER_PATHS = [
    (field,) for field in (
        "v", "max_token_layers", "max_cost", "execution_deadline",
        "settlement_deadline",
    )
] + [
    ("workers", 0, field) for field in ("max_token_layers", "max_cost")
]

STATUS_INTEGER_PATHS = [
    (field,) for field in (
        "v", "max_prompt_tokens", "max_output_tokens", "max_token_layers",
        "max_cost", "authorized_at", "execution_deadline",
        "settlement_deadline", "terminal_at", "escrow_balance",
    )
] + [
    ("workers", 0, field) for field in (
        "layer_start", "layer_end", "price_multiplier", "max_prompt_tokens",
        "max_output_tokens", "max_token_layers", "max_cost", "claimed_at",
        "cumulative_prompt_tokens", "cumulative_generated_tokens",
        "cumulative_token_layers", "cumulative_cost",
    )
]


def _mutate_path(value, path, replacement):
    target = value
    for item in path[:-1]:
        target = target[item]
    target[path[-1]] = replacement
    return value


@pytest.mark.parametrize("path", AUTHORIZE_INTEGER_PATHS)
@pytest.mark.parametrize("replacement", [True, False])
def test_every_authorize_integer_field_rejects_booleans(path, replacement):
    raw = _mutate_path(_authorization(), path, replacement)
    with pytest.raises(paid.PaidAdmissionError):
        paid.validate_authorization(raw, _request())


@pytest.mark.parametrize("path", STATUS_INTEGER_PATHS)
@pytest.mark.parametrize("replacement", [True, False])
def test_every_status_integer_field_rejects_booleans(path, replacement):
    authorization = paid.validate_authorization(_authorization(), _request())
    raw = _mutate_path(_status(), path, replacement)
    with pytest.raises(paid.PaidAdmissionError):
        paid.validate_status(raw, request=_request(), authorization=authorization)


def test_identical_authorize_retry_accepts_neurons_rotated_token_semantics():
    request = _request()
    first = paid.validate_authorization(_authorization(), request)
    rotated_raw = dict(_authorization(), claim_token="B" * 43)
    rotated = paid.validate_authorization(rotated_raw, request)
    assert rotated["hold_id"] == first["hold_id"]
    assert rotated["claim_token"] != first["claim_token"]


def test_ledger_hook_uses_exact_spend_routes_and_query(monkeypatch):
    hook = ledger_client.LedgerHook(enabled=False)
    hook.enabled = hook.fail_closed = True
    hook._private_key = object()
    hook.url = "https://ledger.example/ledger"
    hook.spki_hash = "a" * 64
    posts = []
    gets = []
    monkeypatch.setattr(
        hook, "_required_post",
        lambda path, body, label, **kwargs:
        posts.append((path, body, label, kwargs)) or {})
    monkeypatch.setattr(
        hook, "_required_get",
        lambda path, label: gets.append((path, label)) or {})
    hook.authorize_spend(_request())
    hook.abandon_spend("hold id", KEY1)
    hook.release_spend("hold id")
    hook.spend_status("hold id")
    assert [item[0] for item in posts] == [
        "/spend/authorize", "/spend/abandon", "/spend/release"]
    assert posts[0][3] == {"retry_ambiguous_transport": True}
    assert posts[1][3] == posts[2][3] == {}
    assert gets == [("/spend/status?hold_id=hold+id", "spend status")]


def _strict_hook():
    hook = ledger_client.LedgerHook(enabled=False)
    hook.enabled = hook.fail_closed = True
    hook._private_key = object()
    hook.url = "https://ledger.example"
    hook.spki_hash = "a" * 64
    return hook


def test_authorize_retries_one_ambiguous_transport_with_same_request(monkeypatch):
    hook = _strict_hook()
    request = _request()
    seen = []
    rotated = dict(_authorization(), claim_token="B" * 43)

    def post(path, body):
        seen.append((path, body, json.dumps(body, sort_keys=True)))
        if len(seen) == 1:
            raise OSError("response lost after commit")
        return rotated

    monkeypatch.setattr(hook, "_post", post)
    result = hook.authorize_spend(request)
    assert result["claim_token"] == "B" * 43
    assert len(seen) == 2
    assert seen[0][0] == seen[1][0] == "/spend/authorize"
    assert seen[0][1] is request and seen[1][1] is request
    assert seen[0][2] == seen[1][2]


def test_authorize_409_is_explicit_conflict_and_never_retried(monkeypatch):
    hook = _strict_hook()
    calls = []

    def conflict(path, body):
        calls.append((path, body))
        raise urllib.error.HTTPError(
            "https://ledger.example/spend/authorize", 409, "Conflict", {},
            BytesIO(b'{"error":"hold already claimed"}'))

    monkeypatch.setattr(hook, "_post", conflict)
    with pytest.raises(ledger_client.LedgerConflict, match="HTTP 409"):
        hook.authorize_spend(_request())
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", [
    lambda status: status.update(state="expired"),
    lambda status: status["workers"][1].update(final=False, state="claimed",
                                                final_receipt_id=None),
    lambda status: status["workers"].pop(),
    lambda status: status.update(state="disputed"),
    lambda status: status.update(unknown=True),
])
def test_expired_partial_missing_dispute_and_unknown_never_release(mutation):
    request = _request()
    authorization = paid.validate_authorization(_authorization(), request)
    status = _status()
    mutation(status)
    with pytest.raises(paid.PaidAdmissionError):
        validated = paid.validate_status(
            status, request=request, authorization=authorization)
        paid.require_releasable_status(validated, used={KEY1, KEY2})


def test_audit_receipt_is_visibly_noneconomic_and_legacy_settle_refuses_it(
        tmp_path, monkeypatch):
    request = _request()
    authorization = paid.validate_authorization(_authorization(), request)
    status = paid.validate_status(
        _status(), request=request, authorization=authorization)
    receipt = paid.audit_receipt(status, output_text="hello", used={KEY1, KEY2})
    assert receipt["economic"] is False
    assert receipt["kind"] == "nakshatra-run-audit"
    assert [item["receipt_id"] for item in receipt["worker_receipts"]] == [
        "sha256:" + KEY1[0] * 64, "sha256:" + KEY2[0] * 64]
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(receipt))
    hook = ledger_client.LedgerHook(enabled=False)
    hook.enabled = True
    hook.fail_closed = True
    hook._private_key = object()
    monkeypatch.setattr(hook, "_post", lambda *_args: pytest.fail(
        "audit receipt reached legacy /settle"))
    with pytest.raises(ledger_client.LedgerUnavailable, match="audit-only"):
        hook.settle(str(path))


class _PaidHook:
    def __init__(self, status=None, status_error=None, authorization=None):
        self.status_value = status or _status()
        self.status_error = status_error
        self.authorization_value = authorization or _authorization()
        self.calls = []

    def require_paid_spend(self):
        return None

    def authorize_spend(self, request):
        self.calls.append(("authorize", request))
        self.status_value["session_id"] = request["session_id"]
        return self.authorization_value

    def abandon_spend(self, hold_id, worker_keyid):
        self.calls.append(("abandon", hold_id, worker_keyid))
        return self.status_value

    def release_spend(self, hold_id):
        self.calls.append(("release", hold_id))
        return self.status_value

    def spend_status(self, hold_id):
        self.calls.append(("status", hold_id))
        if self.status_error:
            raise self.status_error
        return self.status_value

    def settle(self, _path):
        pytest.fail("paid aggregate reached legacy settle")


def _paid_backend(tmp_path, hook):
    config = tmp_path / "chain.yaml"
    config.write_text(
        "workers:\n"
        f"  - id: one\n    public_key_hex: {KEY1}\n    peer_spki_hash: {SPKI1}\n    address: 127.0.0.1\n"
        "    port: 1\n    layer_range: [0, 2]\n"
        f"  - id: two\n    public_key_hex: {KEY2}\n    peer_spki_hash: {SPKI2}\n    address: 127.0.0.1\n"
        "    port: 2\n    layer_range: [2, 4]\n")
    backend = serve.ChainChatBackend(paid_admission=True)
    backend._ledger_hook_cached = hook
    backend._paid_prompt_tokens = lambda _entry, _prompt: 3
    entry = serve.ModelEntry(
        name="model", tokenizer_gguf="/tokenizer", chain_yaml=str(config))
    return backend, entry


def test_paid_gateway_refuses_missing_spki_before_authorize(tmp_path, caplog):
    config = tmp_path / "chain.yaml"
    config.write_text(
        "workers:\n"
        f"  - id: one\n    public_key_hex: {KEY1}\n    address: 127.0.0.1\n"
        "    port: 1\n    layer_range: [0, 2]\n")
    hook = _PaidHook()
    backend = serve.ChainChatBackend(paid_admission=True)
    backend._ledger_hook_cached = hook
    backend._paid_prompt_tokens = lambda _entry, _prompt: 3
    entry = serve.ModelEntry(
        name="model", tokenizer_gguf="/tokenizer", chain_yaml=str(config))
    with pytest.raises(serve.CreditGateError) as caught:
        backend.generate(entry, "prompt", 2, {})
    assert caught.value.status == 503
    assert hook.calls == []
    assert "paid pre-client planning failure" in caplog.text
    assert ("PaidAdmissionError: paid chain hop has no valid TLS SPKI pin"
            in caplog.text)


def test_paid_from_roster_keeps_local_authority_and_enriches_pillar_transport(
        tmp_path, monkeypatch):
    source = tmp_path / "roster-chain.yaml"
    source.write_text(
        "model: {id: model}\nworkers:\n"
        f"  - id: local-roster-name\n    public_key_hex: {KEY1}\n"
        "    peer_spki_hash: ''\n"
        "    address: stale.local\n    port: 1\n"
        "    layer_range: [0, 2]\n    mode: solo\n"
    )
    endpoint = {
        "kind": client.ACTIVATION_ENDPOINT_KIND,
        "peer_id": "12D3KooPinnedRosterWorker",
    }
    peer = {
        "node_id": "pillar-node-id",
        "public_key_hex": KEY1,
        "peer_spki_hash": SPKI1,
        "address": "fresh.example:5540",
        "endpoint": endpoint,
        "tier": "self",  # never imported; the local roster chose the worker
    }
    backend = serve.ChainChatBackend(paid_admission=True)
    backend._roster_chain_yaml = lambda _entry, paid=False: str(source)
    monkeypatch.setenv("NAKSHATRA_PILLAR_URL", "https://pillar.example")
    monkeypatch.setenv("STHAMBHA_PILLAR_SPKI_SHA256", "aa" * 32)
    monkeypatch.setattr(client, "_fetch_registry_peers", lambda *_args: [peer])
    entry = serve.ModelEntry(
        name="model", tokenizer_gguf="/tokenizer", from_roster=True,
        hidden_size=1)

    path, plan, owned = backend._paid_chain_config(entry)
    try:
        document = client.yaml.safe_load(Path(path).read_text())
    finally:
        Path(path).unlink()

    assert owned is True
    assert plan == [{"worker_keyid": KEY1, "layer_start": 0, "layer_end": 2}]
    worker = document["workers"][0]
    assert worker["id"] == "local-roster-name"
    assert worker["public_key_hex"] == KEY1
    assert "tier" not in worker
    assert worker["node_id"] == "pillar-node-id"
    assert worker["endpoint"] == endpoint
    assert worker["peer_spki_hash"] == SPKI1
    assert (worker["address"], worker["port"]) == ("fresh.example", 5540)


def _successful_client_run(cmd, *, env, pass_fds, **_kwargs):
    token_fd = int(cmd[cmd.index("--paid-admission-fd") + 1])
    progress_fd = int(cmd[cmd.index("--paid-progress-fd") + 1])
    envelope = paid.read_paid_envelope(os.dup(token_fd))
    assert envelope["claim_token"] not in cmd
    assert envelope["claim_token"] not in env.values()
    paid.write_invocation_event(progress_fd, KEY1)
    paid.write_invocation_event(progress_fd, KEY2)
    return SimpleNamespace(
        returncode=0, stdout="[chain] gen:  'hello'\n", stderr="")


def test_gateway_paid_success_uses_fd_and_never_legacy_settle(
        tmp_path, monkeypatch):
    hook = _PaidHook()
    backend, entry = _paid_backend(tmp_path, hook)
    monkeypatch.setattr(subprocess, "run", _successful_client_run)
    result = backend.generate(entry, "prompt", 2, {})
    assert result.text == "hello"
    assert result.audit_receipt["economic"] is False
    assert [call[0] for call in hook.calls] == ["authorize", "release", "status"]


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="requires procfs")
def test_paid_token_absent_from_proc_logs_errors_and_grandchild_fds(
        tmp_path, monkeypatch, caplog):
    hook = _PaidHook()
    backend, entry = _paid_backend(tmp_path, hook)
    completed = []
    real_run = subprocess.run
    probe = r'''
import json, os, subprocess, sys
sys.path.insert(0, sys.argv[1])
import paid_client
token_fd = int(sys.argv[sys.argv.index("--paid-admission-fd") + 1])
progress_fd = int(sys.argv[sys.argv.index("--paid-progress-fd") + 1])
token_target = os.readlink("/proc/self/fd/" + str(token_fd))
envelope = paid_client.read_paid_envelope(token_fd)
token = envelope["claim_token"].encode()
if token in open("/proc/self/cmdline", "rb").read():
    raise SystemExit("secret appeared in cmdline")
if token in open("/proc/self/environ", "rb").read():
    raise SystemExit("secret appeared in environ")
grandchild = subprocess.run(
    [sys.executable, "-c", "import json,os; print(json.dumps([os.readlink('/proc/self/fd/'+n) for n in os.listdir('/proc/self/fd') if os.path.exists('/proc/self/fd/'+n)]))"],
    capture_output=True, text=True, check=True)
if token.decode() in grandchild.stdout or token_target in grandchild.stdout:
    raise SystemExit("secret descriptor reached grandchild")
for key in envelope["plan"]:
    paid_client.write_invocation_event(progress_fd, key["worker_keyid"])
print("[chain] gen:  'hello'")
'''

    def probe_cmd(_entry, _prompt, _max_tokens, *, paid_admission_fd,
                  paid_progress_fd, **_kwargs):
        return [
            sys.executable, "-c", probe, str(ROOT / "scripts"),
            "--paid-admission-fd", str(paid_admission_fd),
            "--paid-progress-fd", str(paid_progress_fd),
        ]

    def recording_run(*args, **kwargs):
        result = real_run(*args, **kwargs)
        completed.append(result)
        return result

    monkeypatch.setattr(backend, "_cmd", probe_cmd)
    monkeypatch.setattr(subprocess, "run", recording_run)
    result = backend.generate(entry, "prompt", 2, {})
    assert result.text == "hello"
    assert len(completed) == 1
    captured = (completed[0].stdout or "") + (completed[0].stderr or "")
    assert "A" * 43 not in captured
    assert "A" * 43 not in caplog.text


def test_gateway_production_hook_retries_authorize_and_forwards_rotated_token(
        tmp_path, monkeypatch):
    hook = _strict_hook()
    backend, entry = _paid_backend(tmp_path, hook)
    seen_authorize = []
    status = _status()

    def post(path, body):
        if path == "/spend/authorize":
            seen_authorize.append(body)
            status["session_id"] = body["session_id"]
            if len(seen_authorize) == 1:
                raise OSError("committed response was lost")
            return dict(_authorization(), claim_token="B" * 43)
        assert path == "/spend/release"
        return status

    monkeypatch.setattr(hook, "_post", post)
    monkeypatch.setattr(hook, "_required_get", lambda *_args: status)

    def client_run(cmd, *, env, pass_fds, **_kwargs):
        token_fd = int(cmd[cmd.index("--paid-admission-fd") + 1])
        progress_fd = int(cmd[cmd.index("--paid-progress-fd") + 1])
        envelope = paid.read_paid_envelope(os.dup(token_fd))
        assert envelope["claim_token"] == "B" * 43
        paid.write_invocation_event(progress_fd, KEY1)
        paid.write_invocation_event(progress_fd, KEY2)
        return SimpleNamespace(
            returncode=0, stdout="[chain] gen:  'hello'\n", stderr="")

    monkeypatch.setattr(subprocess, "run", client_run)
    result = backend.generate(entry, "prompt", 2, {})
    assert result.text == "hello"
    assert len(seen_authorize) == 2
    assert seen_authorize[0] is seen_authorize[1]
    assert seen_authorize[0]["client_nonce"] == seen_authorize[1]["client_nonce"]
    assert seen_authorize[0]["session_id"] == seen_authorize[1]["session_id"]


def test_gateway_authorize_retry_409_after_ambiguous_commit_fails_closed(
        tmp_path, monkeypatch):
    hook = _strict_hook()
    backend, entry = _paid_backend(tmp_path, hook)
    calls = []

    def post(path, body):
        calls.append((path, body))
        if len(calls) == 1:
            raise OSError("committed response was lost")
        raise urllib.error.HTTPError(
            "https://ledger.example/spend/authorize", 409, "Conflict", {},
            BytesIO(b'{"error":"hold already claimed"}'))

    monkeypatch.setattr(hook, "_post", post)
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs:
        pytest.fail("409 authorize conflict reached compute"))
    with pytest.raises(serve.CreditGateError) as caught:
        backend.generate(entry, "prompt", 2, {})
    assert caught.value.status == 503
    assert len(calls) == 2
    assert calls[0][1] is calls[1][1]


@pytest.mark.parametrize("path", AUTHORIZE_INTEGER_PATHS)
@pytest.mark.parametrize("replacement", [True, False])
def test_gateway_authorize_boolean_returns_503_without_compute_or_text(
        tmp_path, monkeypatch, path, replacement):
    raw = _mutate_path(_authorization(), path, replacement)
    hook = _PaidHook(authorization=raw)
    backend, entry = _paid_backend(tmp_path, hook)
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs:
        pytest.fail("invalid authorize response reached compute"))
    with pytest.raises(serve.CreditGateError) as caught:
        backend.generate(entry, "prompt", 2, {})
    assert caught.value.status == 503


@pytest.mark.parametrize("path", STATUS_INTEGER_PATHS)
@pytest.mark.parametrize("replacement", [True, False])
def test_gateway_status_boolean_returns_503_without_releasing_text(
        tmp_path, monkeypatch, path, replacement):
    raw = _mutate_path(_status(), path, replacement)
    hook = _PaidHook(status=raw)
    backend, entry = _paid_backend(tmp_path, hook)
    monkeypatch.setattr(subprocess, "run", _successful_client_run)
    with pytest.raises(serve.CreditGateError) as caught:
        backend.generate(entry, "prompt", 2, {})
    assert caught.value.status == 503


def test_precompute_client_rejection_abandons_every_stage_then_releases(
        tmp_path, monkeypatch):
    hook = _PaidHook()
    backend, entry = _paid_backend(tmp_path, hook)
    monkeypatch.setattr(
        subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="capability mismatch"))
    with pytest.raises(serve.CreditGateError) as caught:
        backend.generate(entry, "prompt", 2, {})
    assert caught.value.status == 503
    assert ("abandon", "hold", KEY1) in hook.calls
    assert ("abandon", "hold", KEY2) in hook.calls
    assert ("release", "hold") in hook.calls


def test_paid_failure_log_redacts_token_keys_and_base64_and_tails_stderr(
        tmp_path, monkeypatch, caplog):
    hook = _PaidHook()
    backend, entry = _paid_backend(tmp_path, hook)
    seen_token = []
    hex_key = "de" * 32
    base64_blob = "c2Vuc2l0aXZlLXBheW1lbnQtYXV0aG9yaXR5LWJsb2ItMDEyMzQ1Njc4OQ=="

    def failed_client(cmd, *, env, pass_fds, **_kwargs):
        token_fd = int(cmd[cmd.index("--paid-admission-fd") + 1])
        envelope = paid.read_paid_envelope(os.dup(token_fd))
        seen_token.append(envelope["claim_token"])
        lines = ["outside retained stderr tail"]
        lines.extend(f"diagnostic line {number}" for number in range(38))
        lines.extend([
            f"claim_token={envelope['claim_token']}",
            f"worker_key={hex_key}",
            f"response_blob={base64_blob}",
        ])
        error = RuntimeError(
            f"client transport failed with claim_token={envelope['claim_token']}")
        error.stderr = "\n".join(lines)
        raise error

    monkeypatch.setattr(subprocess, "run", failed_client)
    with pytest.raises(serve.CreditGateError):
        backend.generate(entry, "prompt", 2, {})

    assert seen_token
    assert seen_token[0] not in caplog.text
    assert hex_key not in caplog.text
    assert base64_blob not in caplog.text
    assert "<redacted>" in caplog.text
    assert "RuntimeError: client transport failed with claim_token=<redacted>" \
        in caplog.text
    assert "diagnostic line 1" in caplog.text
    assert "outside retained stderr tail" not in caplog.text


@pytest.mark.parametrize("status", [
    _status(state="expired"),
    _status(final=False),
    _status(include_second=False),
    _status(state="disputed"),
    _status() | {"unknown": True},
])
def test_gateway_paid_finality_failures_return_503_without_text(
        tmp_path, monkeypatch, status):
    hook = _PaidHook(status=status)
    backend, entry = _paid_backend(tmp_path, hook)
    monkeypatch.setattr(subprocess, "run", _successful_client_run)
    monkeypatch.setenv("NAKSHATRA_PAID_STATUS_TIMEOUT", "0")
    with pytest.raises(serve.CreditGateError) as caught:
        backend.generate(entry, "prompt", 2, {})
    assert caught.value.status == 503


def test_gateway_paid_status_outage_returns_503_without_text(tmp_path, monkeypatch):
    hook = _PaidHook(status_error=OSError("ledger down"))
    backend, entry = _paid_backend(tmp_path, hook)
    monkeypatch.setattr(subprocess, "run", _successful_client_run)
    with pytest.raises(serve.CreditGateError) as caught:
        backend.generate(entry, "prompt", 2, {})
    assert caught.value.status == 503


class _FailedPaidFinalityBackend(serve.ChatBackend):
    def requires_credit_buffer(self):
        return True

    def generate(self, *_args, **_kwargs):
        raise serve.CreditGateError("paid finality unavailable", 503)

    def generate_stream(self, *_args, **_kwargs):
        pytest.fail("failed paid finality opened a streaming body")


def _paid_stream_http_response(path, payload):
    entry = serve.ModelEntry(name="model")
    server = serve.build_server(
        "127.0.0.1", 0, {entry.name: entry}, _FailedPaidFinalityBackend())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=2)
        error = caught.value
        return error.code, error.headers.get_content_type(), error.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(("path", "payload"), [
    ("/v1/chat/completions", {
        "model": "model", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }),
    ("/api/chat", {
        "model": "model", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }),
])
def test_http_paid_stream_finality_failure_is_503_before_sse_or_ndjson(
        path, payload):
    status, content_type, body = _paid_stream_http_response(path, payload)
    assert status == 503
    assert content_type == "application/json"
    assert b"data:" not in body
    assert b"text/event-stream" not in body
    assert b'"done": false' not in body
    assert b"paid finality unavailable" in body


def test_gateway_timeout_abandons_only_stage_proven_uninvoked(tmp_path, monkeypatch):
    hook = _PaidHook()
    backend, entry = _paid_backend(tmp_path, hook)

    def timeout_after_first(cmd, **_kwargs):
        progress_fd = int(cmd[cmd.index("--paid-progress-fd") + 1])
        paid.write_invocation_event(progress_fd, KEY1)
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "run", timeout_after_first)
    with pytest.raises(serve.CreditGateError) as caught:
        backend.generate(entry, "prompt", 2, {})
    assert caught.value.status == 503
    assert ("abandon", "hold", KEY2) in hook.calls
    assert ("abandon", "hold", KEY1) not in hook.calls
    assert ("release", "hold") in hook.calls
