"""Payment-bound gateway/client invariants shared by the gateway and client.

This module is intentionally pure: it plans exact worker identities/spans,
validates the strict Neuron wire shapes, and decides whether buffered output
may be released.  It never performs inference or ledger I/O.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from typing import Iterable, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


REQUIRED_PAID_CAPABILITIES = frozenset({
    "spend_online_claim_v1",
    "paid_direct_greedy_v1",
    "prompt_token_layers_v1",
    "usage_receipt_v2",
    "truncate_session_owner",
    "kv_residency_epoch_v1",
})

AUTHORIZE_FIELDS = frozenset({
    "v", "claim_token", "hold_id", "max_token_layers", "max_cost",
    "execution_deadline", "settlement_deadline", "workers",
})
AUTHORIZE_WORKER_FIELDS = frozenset({
    "worker_keyid", "billable", "max_token_layers", "max_cost",
})
STATUS_FIELDS = frozenset({
    "v", "hold_id", "payer_keyid", "payer_account_at_authorize",
    "session_id", "model_id", "billing_mode", "max_prompt_tokens",
    "max_output_tokens", "max_token_layers", "max_cost", "state",
    "release_requested", "authorized_at", "execution_deadline",
    "settlement_deadline", "terminal_at", "escrow_balance", "workers",
})
STATUS_WORKER_FIELDS = frozenset({
    "worker_keyid", "worker_account_at_authorize", "billable",
    "layer_start", "layer_end", "price_multiplier", "max_prompt_tokens",
    "max_output_tokens", "max_token_layers", "max_cost", "state",
    "claimed_at", "cumulative_prompt_tokens", "cumulative_generated_tokens",
    "cumulative_token_layers", "cumulative_cost", "final",
    "final_receipt_id", "claim_response_sha256",
})
_HOLD_STATES = frozenset({
    "open", "closing", "settled", "expired", "cancelled", "disputed",
})
_ENTRY_STATES = frozenset({
    "planned", "claimed", "final", "abandoned", "expired",
})
_KEYID_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RECEIPT_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_ENVELOPE_BYTES = 64 * 1024
_MAX_WORKERS = 128
_MAX_TOKENS = 1_000_000
_MAX_LAYERS = 1_000_000
INFO_ATTESTATION_DOMAIN = b"nakshatra-paid-info-v1\0"
INFO_CHALLENGE_BYTES = 32


class PaidAdmissionError(RuntimeError):
    """The paid protocol could not establish an exact, releasable run."""


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise PaidAdmissionError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: object, name: str) -> str:
    try:
        encoded = value.encode("utf-8") if isinstance(value, str) else b""
    except UnicodeEncodeError as exc:
        raise PaidAdmissionError(f"{name} must be valid UTF-8") from exc
    if (not isinstance(value, str) or not value or len(encoded) > 256
            or any(ord(char) < 0x20 or ord(char) == 0x7f for char in value)):
        raise PaidAdmissionError(f"{name} must be a non-empty bounded string")
    return value


def _keyid(value: object, name: str = "worker_keyid") -> str:
    if not isinstance(value, str) or _KEYID_RE.fullmatch(value) is None:
        raise PaidAdmissionError(f"{name} must be a lowercase Ed25519 key id")
    return value


def _exact_object(value: object, fields: frozenset[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise PaidAdmissionError(f"unknown {name} schema")
    return value


def _worker_span(worker: Mapping[str, object]) -> tuple[int, int]:
    raw = worker.get("layer_range")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        start, end = raw
    else:
        start, end = worker.get("layer_start"), worker.get("layer_end")
    start = _integer(start, "layer_start")
    end = _integer(end, "layer_end", minimum=1)
    if end <= start:
        raise PaidAdmissionError("layer_end must be greater than layer_start")
    return start, end


def worker_keyid(worker: Mapping[str, object]) -> str:
    """Return the economic identity from one already-selected chain hop."""
    for field in ("public_key_hex", "worker_keyid"):
        value = worker.get(field)
        if value:
            return _keyid(str(value).strip().lower())
    node_id = str(worker.get("node_id") or "").strip().lower()
    if _KEYID_RE.fullmatch(node_id):
        return node_id
    raise PaidAdmissionError("paid chain hop has no full worker public key")


def make_info_challenge() -> bytes:
    """Return a fresh challenge for one paid worker's channel-bound Info."""
    return secrets.token_bytes(INFO_CHALLENGE_BYTES)


def verify_info_attestation(info: object, challenge: bytes) -> str:
    """Return the Ed25519 key proven by Info on this pinned TLS channel."""
    if not isinstance(challenge, bytes) or len(challenge) != INFO_CHALLENGE_BYTES:
        raise PaidAdmissionError("paid Info challenge is invalid")
    keyid = _keyid(getattr(info, "worker_keyid", None), "worker_keyid")
    signature = getattr(info, "identity_signature", None)
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise PaidAdmissionError("paid worker identity signature is invalid")
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(keyid)).verify(
            signature, INFO_ATTESTATION_DOMAIN + challenge)
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise PaidAdmissionError("paid worker identity signature is invalid") from exc
    return keyid


def exact_chain_plan(workers: Iterable[Mapping[str, object]]) -> list[dict]:
    """Freeze the precise economic chain in layer order.

    Labels and addresses deliberately do not enter the authorization schema.
    The full worker key and half-open span are the shared planner contract.
    """
    stages = []
    for worker in workers:
        start, end = _worker_span(worker)
        stages.append({
            "worker_keyid": worker_keyid(worker),
            "layer_start": start,
            "layer_end": end,
        })
    stages.sort(key=lambda stage: stage["layer_start"])
    if not stages:
        raise PaidAdmissionError("paid chain is empty")
    if len(stages) > _MAX_WORKERS:
        raise PaidAdmissionError("paid chain exceeds the worker limit")
    seen = set()
    cursor = stages[0]["layer_start"]
    for stage in stages:
        if stage["worker_keyid"] in seen:
            raise PaidAdmissionError("paid chain repeats a worker key")
        if stage["layer_start"] != cursor:
            raise PaidAdmissionError("paid chain has a layer gap or overlap")
        if stage["layer_end"] > _MAX_LAYERS:
            raise PaidAdmissionError("paid chain exceeds the layer limit")
        seen.add(stage["worker_keyid"])
        cursor = stage["layer_end"]
    return stages


def require_paid_tls_pins(workers: Iterable[Mapping[str, object]]) -> None:
    """Require one signed-roster SPKI hash for every paid direct hop."""
    seen = False
    for worker in workers:
        seen = True
        pin = worker.get("peer_spki_hash")
        if not isinstance(pin, str) or _SHA256_RE.fullmatch(pin) is None:
            raise PaidAdmissionError("paid chain hop has no valid TLS SPKI pin")
    if not seen:
        raise PaidAdmissionError("paid chain is empty")


def plan_sha256(plan: Sequence[Mapping[str, object]]) -> str:
    canonical = json.dumps(list(plan), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def runtime_chain_plan(sorted_stubs: Iterable[tuple]) -> list[dict]:
    """Describe the attested identities/spans the client is about to call."""
    stages = []
    for worker, _stub, info in sorted_stubs:
        stages.append({
            "worker_keyid": _keyid(
                worker.get("_paid_attested_worker_keyid"), "worker_keyid"),
            "layer_start": _integer(info.layer_start, "layer_start"),
            "layer_end": _integer(info.layer_end, "layer_end", minimum=1),
        })
    return exact_chain_plan(stages)


def require_exact_runtime_plan(
    authorized_plan: Sequence[Mapping[str, object]], sorted_stubs: Iterable[tuple]
) -> None:
    try:
        runtime = runtime_chain_plan(sorted_stubs)
    except PaidAdmissionError as exc:
        raise PaidAdmissionError(
            "paid chain topology changed after authorization") from exc
    if list(authorized_plan) != runtime:
        raise PaidAdmissionError("paid chain topology changed after authorization")


def require_paid_capabilities(sorted_stubs: Iterable[tuple]) -> None:
    missing = []
    for worker, _stub, info in sorted_stubs:
        absent = sorted(REQUIRED_PAID_CAPABILITIES - set(info.protocol_capabilities))
        if absent:
            missing.append((str(worker.get("id") or "<unknown>"), absent))
    if missing:
        detail = "; ".join(f"{worker}: {','.join(absent)}" for worker, absent in missing)
        raise PaidAdmissionError(f"paid worker capability set incomplete ({detail})")


def paid_mode_refusal(*, push: bool, speculative: bool, stream_spec: bool,
                      async_pipeline: bool, eagle: bool, fabric: bool = False) -> str:
    selected = []
    if push:
        selected.append("push")
    if fabric:
        selected.append("fabric")
    if speculative or stream_spec or async_pipeline or eagle:
        selected.append("non-greedy")
    return ", ".join(selected)


def make_paid_envelope(*, claim_token: str, hold_id: str, session_id: str,
                       model_id: str, max_prompt_tokens: int,
                       max_output_tokens: int,
                       plan: Sequence[Mapping[str, object]]) -> dict:
    if _TOKEN_RE.fullmatch(claim_token) is None:
        raise PaidAdmissionError("ledger returned an invalid claim token")
    normalized = exact_chain_plan(plan)
    return {
        "v": 1,
        "claim_token": claim_token,
        "hold_id": _text(hold_id, "hold_id"),
        "session_id": _text(session_id, "session_id"),
        "model_id": _text(model_id, "model_id"),
        "max_prompt_tokens": _integer(max_prompt_tokens, "max_prompt_tokens"),
        "max_output_tokens": _integer(max_output_tokens, "max_output_tokens"),
        "plan": normalized,
        "plan_sha256": plan_sha256(normalized),
    }


def read_paid_envelope(fd: int) -> dict:
    """Consume the opaque claim from one inherited descriptor, then close it."""
    chunks = []
    total = 0
    try:
        while True:
            chunk = os.read(fd, min(8192, _MAX_ENVELOPE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_ENVELOPE_BYTES:
                raise PaidAdmissionError("paid admission envelope is too large")
    finally:
        os.close(fd)
    try:
        value = json.loads(b"".join(chunks))
        value = _exact_object(
            value,
            frozenset({
                "v", "claim_token", "hold_id", "session_id", "model_id",
                "max_prompt_tokens", "max_output_tokens", "plan", "plan_sha256",
            }),
            "paid admission envelope",
        )
        if value["v"] != 1 or isinstance(value["v"], bool):
            raise PaidAdmissionError("unknown paid admission envelope version")
        rebuilt = make_paid_envelope(
            claim_token=value["claim_token"], hold_id=value["hold_id"],
            session_id=value["session_id"], model_id=value["model_id"],
            max_prompt_tokens=value["max_prompt_tokens"],
            max_output_tokens=value["max_output_tokens"],
            plan=value["plan"],
        )
        if value["plan_sha256"] != rebuilt["plan_sha256"]:
            raise PaidAdmissionError("paid admission plan digest mismatch")
        return rebuilt
    except PaidAdmissionError:
        raise
    except Exception as exc:
        raise PaidAdmissionError("invalid paid admission envelope") from exc


def authorize_request(*, client_nonce: str, session_id: str, model_id: str,
                      prompt_tokens: int, output_tokens: int,
                      plan: Sequence[Mapping[str, object]]) -> dict:
    prompt_cap = _integer(prompt_tokens, "max_prompt_tokens")
    output_cap = _integer(output_tokens, "max_output_tokens")
    if prompt_cap > _MAX_TOKENS or output_cap > _MAX_TOKENS:
        raise PaidAdmissionError("paid token cap exceeds the ledger limit")
    if prompt_cap + output_cap == 0:
        raise PaidAdmissionError("prompt and output caps cannot both be zero")
    return {
        "client_nonce": _text(client_nonce, "client_nonce"),
        "session_id": _text(session_id, "session_id"),
        "model_id": _text(model_id, "model_id"),
        "billing_mode": "greedy-direct-v1",
        "max_prompt_tokens": prompt_cap,
        "max_output_tokens": output_cap,
        "workers": exact_chain_plan(plan),
    }


def validate_authorization(value: object, request: Mapping[str, object]) -> dict:
    result = _exact_object(value, AUTHORIZE_FIELDS, "spend authorize response")
    if result["v"] != 1 or isinstance(result["v"], bool):
        raise PaidAdmissionError("unknown spend authorize response version")
    if not isinstance(result["claim_token"], str) or _TOKEN_RE.fullmatch(
            result["claim_token"]) is None:
        raise PaidAdmissionError("ledger returned an invalid claim token")
    _text(result["hold_id"], "hold_id")
    _integer(result["max_token_layers"], "max_token_layers")
    _integer(result["max_cost"], "max_cost")
    execution = _integer(result["execution_deadline"], "execution_deadline", minimum=1)
    settlement = _integer(result["settlement_deadline"], "settlement_deadline", minimum=1)
    if settlement <= execution:
        raise PaidAdmissionError("ledger returned invalid spend deadlines")
    raw_workers = result["workers"]
    if not isinstance(raw_workers, list):
        raise PaidAdmissionError("unknown spend authorize workers schema")
    expected = {stage["worker_keyid"]: stage for stage in request["workers"]}
    seen = set()
    token_total = cost_total = 0
    prompt = int(request["max_prompt_tokens"])
    output = int(request["max_output_tokens"])
    for raw in raw_workers:
        worker = _exact_object(raw, AUTHORIZE_WORKER_FIELDS, "authorize worker")
        key = _keyid(worker["worker_keyid"])
        if key not in expected or key in seen or not isinstance(worker["billable"], bool):
            raise PaidAdmissionError("authorize worker set differs from the plan")
        seen.add(key)
        stage = expected[key]
        cap = (prompt + output) * (stage["layer_end"] - stage["layer_start"])
        if _integer(worker["max_token_layers"], "max_token_layers") != cap:
            raise PaidAdmissionError("authorize worker token-layer cap differs from the plan")
        expected_cost = cap if worker["billable"] else 0
        if _integer(worker["max_cost"], "max_cost") != expected_cost:
            raise PaidAdmissionError("authorize worker cost cap differs from the plan")
        token_total += cap
        cost_total += expected_cost
    if seen != set(expected):
        raise PaidAdmissionError("authorize worker set differs from the plan")
    if result["max_token_layers"] != token_total or result["max_cost"] != cost_total:
        raise PaidAdmissionError("authorize aggregate caps differ from the plan")
    return result


def validate_status(value: object, *, request: Mapping[str, object],
                    authorization: Mapping[str, object]) -> dict:
    status = _exact_object(value, STATUS_FIELDS, "spend status response")
    if status["v"] != 1 or isinstance(status["v"], bool):
        raise PaidAdmissionError("unknown spend status response version")
    for field in (
        "max_prompt_tokens", "max_output_tokens", "max_token_layers", "max_cost",
        "authorized_at", "execution_deadline", "settlement_deadline",
        "escrow_balance",
    ):
        _integer(status[field], field, minimum=(1 if field in {
            "authorized_at", "execution_deadline", "settlement_deadline"} else 0))
    if status["terminal_at"] is not None:
        _integer(status["terminal_at"], "terminal_at", minimum=1)
    immutable = {
        "hold_id": authorization["hold_id"],
        "session_id": request["session_id"],
        "model_id": request["model_id"],
        "billing_mode": "greedy-direct-v1",
        "max_prompt_tokens": request["max_prompt_tokens"],
        "max_output_tokens": request["max_output_tokens"],
        "max_token_layers": authorization["max_token_layers"],
        "max_cost": authorization["max_cost"],
        "execution_deadline": authorization["execution_deadline"],
        "settlement_deadline": authorization["settlement_deadline"],
    }
    if any(status.get(field) != expected for field, expected in immutable.items()):
        raise PaidAdmissionError("spend status immutable fields differ from authorization")
    _keyid(status["payer_keyid"], "payer_keyid")
    _text(status["payer_account_at_authorize"], "payer_account_at_authorize")
    if status["state"] not in _HOLD_STATES or not isinstance(status["release_requested"], bool):
        raise PaidAdmissionError("unknown spend hold state")
    raw_workers = status["workers"]
    if not isinstance(raw_workers, list):
        raise PaidAdmissionError("unknown spend status workers schema")
    expected_plan = {stage["worker_keyid"]: stage for stage in request["workers"]}
    auth_workers = {worker["worker_keyid"]: worker for worker in authorization["workers"]}
    seen = set()
    for raw in raw_workers:
        worker = _exact_object(raw, STATUS_WORKER_FIELDS, "status worker")
        key = _keyid(worker["worker_keyid"])
        if key not in expected_plan or key in seen:
            raise PaidAdmissionError("status worker set differs from the plan")
        seen.add(key)
        stage = expected_plan[key]
        auth_worker = auth_workers[key]
        for field in (
            "layer_start", "layer_end", "price_multiplier", "max_prompt_tokens",
            "max_output_tokens", "max_token_layers", "max_cost",
            "cumulative_prompt_tokens", "cumulative_generated_tokens",
            "cumulative_token_layers", "cumulative_cost",
        ):
            _integer(worker[field], field, minimum=(1 if field == "layer_end" else 0))
        if worker["claimed_at"] is not None:
            _integer(worker["claimed_at"], "claimed_at", minimum=1)
        if (worker["layer_start"], worker["layer_end"]) != (
                stage["layer_start"], stage["layer_end"]):
            raise PaidAdmissionError("status worker span differs from the plan")
        for field in ("billable", "max_token_layers", "max_cost"):
            if worker[field] != auth_worker[field]:
                raise PaidAdmissionError("status worker caps differ from authorization")
        if worker["max_prompt_tokens"] != request["max_prompt_tokens"] or \
                worker["max_output_tokens"] != request["max_output_tokens"] or \
                worker["price_multiplier"] != 1:
            raise PaidAdmissionError("status worker pricing differs from authorization")
        if worker["state"] not in _ENTRY_STATES or not isinstance(worker["final"], bool):
            raise PaidAdmissionError("unknown spend worker state")
        _text(worker["worker_account_at_authorize"], "worker_account_at_authorize")
        receipt_id = worker["final_receipt_id"]
        if (receipt_id is not None
                and (not isinstance(receipt_id, str)
                     or _RECEIPT_ID_RE.fullmatch(receipt_id) is None)):
            raise PaidAdmissionError("invalid final_receipt_id")
        claim_digest = worker["claim_response_sha256"]
        if (claim_digest is not None
                and (not isinstance(claim_digest, str)
                     or _SHA256_RE.fullmatch(claim_digest) is None)):
            raise PaidAdmissionError("invalid claim_response_sha256")
    if seen != set(expected_plan):
        raise PaidAdmissionError("status is missing a planned worker")
    return status


def require_releasable_status(status: Mapping[str, object], *, used: set[str]) -> None:
    if status["state"] != "settled" or status["escrow_balance"] != 0:
        raise PaidAdmissionError("paid hold is not exactly settled")
    if status["terminal_at"] is None:
        raise PaidAdmissionError("settled paid hold has no terminal time")
    by_key = {worker["worker_keyid"]: worker for worker in status["workers"]}
    if not used or not used.issubset(by_key):
        raise PaidAdmissionError("paid status is missing a used stage")
    generated_count = None
    for key, worker in by_key.items():
        if key in used:
            if (worker["state"] != "final" or not worker["final"]
                    or not isinstance(worker["final_receipt_id"], str)
                    or _RECEIPT_ID_RE.fullmatch(worker["final_receipt_id"]) is None
                    or not isinstance(worker["claim_response_sha256"], str)
                    or _SHA256_RE.fullmatch(worker["claim_response_sha256"]) is None):
                raise PaidAdmissionError("paid status has a nonfinal used stage")
            if worker["cumulative_prompt_tokens"] != status["max_prompt_tokens"]:
                raise PaidAdmissionError("paid status prompt meter differs from authorization")
            if worker["cumulative_generated_tokens"] > status["max_output_tokens"]:
                raise PaidAdmissionError("paid status output meter exceeds authorization")
            if generated_count is None:
                generated_count = worker["cumulative_generated_tokens"]
            elif generated_count != worker["cumulative_generated_tokens"]:
                raise PaidAdmissionError("paid stages disagree on generated-token count")
            layers = worker["layer_end"] - worker["layer_start"]
            token_layers = (
                worker["cumulative_prompt_tokens"]
                + worker["cumulative_generated_tokens"]
            ) * layers
            expected_cost = token_layers * worker["price_multiplier"] \
                if worker["billable"] else 0
            if (worker["cumulative_token_layers"] != token_layers
                    or worker["cumulative_cost"] != expected_cost
                    or worker["cumulative_cost"] > worker["max_cost"]):
                raise PaidAdmissionError("paid status charge does not match the plan")
        elif worker["state"] != "abandoned" or worker["final"]:
            raise PaidAdmissionError("paid status has an unaccounted unused stage")


def audit_receipt(status: Mapping[str, object], *, output_text: str,
                  used: set[str]) -> dict:
    """Build evidence only; this shape is rejected before either settle route."""
    require_releasable_status(status, used=used)
    worker_receipts = []
    for worker in status["workers"]:
        if worker["worker_keyid"] not in used:
            continue
        worker_receipts.append({
            "receipt_id": worker["final_receipt_id"],
            "worker_keyid": worker["worker_keyid"],
            "layer_start": worker["layer_start"],
            "layer_end": worker["layer_end"],
            "cumulative_prompt_tokens": worker["cumulative_prompt_tokens"],
            "cumulative_generated_tokens": worker["cumulative_generated_tokens"],
            "cumulative_token_layers": worker["cumulative_token_layers"],
            "cumulative_cost": worker["cumulative_cost"],
            "final": True,
        })
    return {
        "v": 2,
        "kind": "nakshatra-run-audit",
        "economic": False,
        "hold_id": status["hold_id"],
        "session_id": status["session_id"],
        "model_id": status["model_id"],
        "state": "settled",
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        "charged": sum(item["cumulative_cost"] for item in worker_receipts),
        "worker_receipts": worker_receipts,
    }


def parse_invocation_events(raw: bytes, plan: Sequence[Mapping[str, object]]) -> set[str]:
    allowed = {stage["worker_keyid"] for stage in plan}
    used = set()
    for line in raw.splitlines():
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception as exc:
            raise PaidAdmissionError("invalid paid invocation record") from exc
        if not isinstance(event, dict) or set(event) != {"invoked"}:
            raise PaidAdmissionError("invalid paid invocation record")
        key = _keyid(event["invoked"])
        if key not in allowed:
            raise PaidAdmissionError("paid invocation record contains an unplanned stage")
        used.add(key)
    return used


def write_invocation_event(fd: int, worker_key: str) -> None:
    raw = json.dumps({"invoked": _keyid(worker_key)}, separators=(",", ":")).encode() + b"\n"
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        view = view[written:]
