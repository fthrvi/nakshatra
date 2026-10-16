from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ledger_client  # noqa: E402
import client  # noqa: E402
import nakshatra_serve as serve  # noqa: E402
import nakshatra_pb2 as pb  # noqa: E402
import paid_admission as paid  # noqa: E402


KEY1 = "a" * 64
KEY2 = "b" * 64
PAYER = "c" * 64


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
        ({"public_key_hex": KEY1}, object(), SimpleNamespace(
            layer_start=0, layer_end=2, protocol_capabilities=caps)),
        ({"public_key_hex": KEY2}, object(), SimpleNamespace(
            layer_start=2, layer_end=4, protocol_capabilities=caps)),
    ]
    paid.require_exact_runtime_plan(plan, stubs)
    paid.require_paid_capabilities(stubs)
    stubs[1][2].layer_start = 3
    with pytest.raises(paid.PaidAdmissionError, match="topology changed"):
        paid.require_exact_runtime_plan(plan, stubs)


def test_paid_mode_refuses_push_fabric_and_non_greedy_locally():
    assert paid.paid_mode_refusal(
        push=True, fabric=True, speculative=True, stream_spec=False,
        async_pipeline=False, eagle=False) == "push, fabric, non-greedy"


def test_proto_field_numbers_match_the_worker_merge_contract():
    forward = pb.ForwardRequest.DESCRIPTOR.fields_by_name
    inference = pb.InferenceStep.DESCRIPTOR.fields_by_name
    truncate = pb.TruncateRequest.DESCRIPTOR.fields_by_name
    assert (forward["session_id"].number, forward["step_id"].number,
            forward["spend_claim_token"].number) == (9, 10, 11)
    assert inference["spend_claim_token"].number == 18
    assert truncate["session_id"].number == 2


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
        lambda path, body, label: posts.append((path, body, label)) or {})
    monkeypatch.setattr(
        hook, "_required_get",
        lambda path, label: gets.append((path, label)) or {})
    hook.authorize_spend(_request())
    hook.abandon_spend("hold id", KEY1)
    hook.release_spend("hold id")
    hook.spend_status("hold id")
    assert [item[0] for item in posts] == [
        "/spend/authorize", "/spend/abandon", "/spend/release"]
    assert gets == [("/spend/status?hold_id=hold+id", "spend status")]


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
        "sha256:" + "a" * 64, "sha256:" + "b" * 64]
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
    def __init__(self, status=None, status_error=None):
        self.status_value = status or _status()
        self.status_error = status_error
        self.calls = []

    def require_paid_spend(self):
        return None

    def authorize_spend(self, request):
        self.calls.append(("authorize", request))
        self.status_value["session_id"] = request["session_id"]
        return _authorization()

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
        f"  - id: one\n    public_key_hex: {KEY1}\n    address: 127.0.0.1\n"
        "    port: 1\n    layer_range: [0, 2]\n"
        f"  - id: two\n    public_key_hex: {KEY2}\n    address: 127.0.0.1\n"
        "    port: 2\n    layer_range: [2, 4]\n")
    backend = serve.ChainChatBackend(paid_admission=True)
    backend._ledger_hook_cached = hook
    backend._paid_prompt_tokens = lambda _entry, _prompt: 3
    entry = serve.ModelEntry(
        name="model", tokenizer_gguf="/tokenizer", chain_yaml=str(config))
    return backend, entry


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
