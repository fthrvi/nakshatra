"""Client for the biometric Nakshatra release-signing approval broker."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from network import broker_transport  # noqa: E402

try:  # Package import for installed code; direct import for release/*.py command-line tools.
    from . import releasekit as rk
except ImportError:  # pragma: no cover - exercised by the command-line module layout.
    import releasekit as rk


_PRINCIPAL = "nakshatra-release"
_WHAT = "nakshatra.release.v1"
_WAIT_S = 600
_POLL_S = 1
_TIMEOUT_S = 3
_REQUEST_ID = re.compile(r"[0-9a-f]{16}\Z")
_PUBLIC_KEY = re.compile(r"[0-9a-f]{64}\Z")
_SIGNATURE = re.compile(r"[0-9a-f]{128}\Z")
_LATEST_KEYS = {"schema", "channel", "version", "manifest_sha256", "created", "expires_at"}
_KEYROLL_KEYS = {"schema", "old_pubkey", "new_pubkey", "issued_at", "reason"}


class BrokerSigningError(RuntimeError):
    """A closed local refusal from the release-signing boundary."""


def _closed(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except SystemExit as exc:
        raise BrokerSigningError(str(exc)) from None


def _config_dir(config_dir: Path | None) -> Path:
    return Path(config_dir) if config_dir is not None else Path.home() / ".nakshatra-release"


def _read_config(config_dir: Path | None) -> broker_transport.BrokerConfig:
    directory = _config_dir(config_dir)
    path = directory / "release-broker.json"
    return _closed(
        broker_transport.load_broker_config, path, subject="release",
        missing_message="release broker configuration is missing: {path}",
        token_missing_message="release broker token file is missing: {path}")


def load_release_public(config_dir: Path | None = None) -> str:
    """Load the locally pinned broker release public key."""
    path = _config_dir(config_dir) / "release.pub"
    try:
        raw = _closed(broker_transport.read_owned_regular, path, "release public key")
    except FileNotFoundError as exc:
        raise BrokerSigningError(f"release public key is missing: {path}") from exc
    try:
        public_hex = raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise BrokerSigningError(f"release public key {path} is not ASCII text") from exc
    if _PUBLIC_KEY.fullmatch(public_hex) is None:
        raise BrokerSigningError(
            f"release public key {path} must contain 64 lowercase hex characters")
    return public_hex


_ERROR_IDS = {
    ("/principal/v1", 401, "unauthorized"): "auth",
    ("/enqueue/v1", 401, "unauthorized"): "auth",
    ("/result/v1", 401, "unauthorized"): "auth",
    ("/release-sign/v1", 401, "unauthorized"): "auth",
    ("/release-sign/v1", 404, "not found"): "route-missing",
    ("/release-sign/v1", 403, "nakshatra-release principal required"): "wrong-principal",
    ("/release-sign/v1", 409, "release signing request was already signed"): "already-signed",
}


def _error_info(path: str, exc: urllib.error.HTTPError) -> tuple[str, bool]:
    try:
        document = json.loads(exc.read(4096).decode("utf-8"))
    except (AttributeError, UnicodeError, ValueError, json.JSONDecodeError):
        return "", False
    if not isinstance(document, dict):
        return "", False
    value = document.get("error", "")
    if not isinstance(value, str):
        return "", document.get("retryable") is True
    return (_ERROR_IDS.get((urllib.parse.urlsplit(path).path, int(exc.code), value), ""),
            document.get("retryable") is True)


def _request(config: broker_transport.BrokerConfig, method: str, path: str,
             body: dict | None = None, *, timeout: float = _TIMEOUT_S,
             retry_transient: bool = False) -> dict:
    return broker_transport.request_json(
        config, method, path, body, timeout=timeout, error_info=_error_info,
        unavailable_message="release broker is unavailable; nothing was signed",
        invalid_message="release broker returned invalid data; nothing was signed",
        retry_transient=retry_transient)


def _uncertain(phase: str, request_id: str, detail: str) -> BrokerSigningError:
    state = ("approval may already have been granted" if phase == "result"
             else "approval was granted and signing may already have completed")
    return BrokerSigningError(
        f"{detail}; {state}; request_id={request_id} can be retried without a new prompt")


def _until(config: broker_transport.BrokerConfig, method: str, path: str,
           body: dict | None, *, deadline: float, request_id: str, phase: str) -> dict:
    try:
        return broker_transport.request_json_until(
            lambda req_method, req_path, req_body, **kwargs: _request(
                config, req_method, req_path, req_body, **kwargs),
            method, path, body, deadline=deadline, request_id=request_id, phase=phase,
            assert_current=lambda: None, uncertain_failure=_uncertain,
            deadline_detail="release approval deadline expired after a transient failure",
            already_signed_detail=(
                "the release-signing request was consumed but its signatures were lost in transit; "
                "nothing was written; start the act again for a new approval"),
            timeout_s=_TIMEOUT_S)
    except SystemExit as exc:
        raise BrokerSigningError(str(exc)) from None


def _decode_response(response: dict, request_id: str,
                     expected_count: int) -> tuple[list[bytes], list[bytes]]:
    if response.get("request_id") != request_id:
        raise BrokerSigningError("release broker substituted the signing request id")
    signature_values, item_values = response.get("signatures"), response.get("items_b64")
    if (not isinstance(signature_values, list) or not isinstance(item_values, list)
            or len(signature_values) != expected_count or len(item_values) != expected_count):
        raise BrokerSigningError("release broker returned the wrong number of signatures or items")
    signatures: list[bytes] = []
    items: list[bytes] = []
    try:
        for value in signature_values:
            if not isinstance(value, str) or _SIGNATURE.fullmatch(value) is None:
                raise ValueError("invalid signature encoding")
            signatures.append(bytes.fromhex(value))
        for value in item_values:
            if not isinstance(value, str):
                raise ValueError("invalid item encoding")
            items.append(base64.b64decode(value, validate=True))
    except (ValueError, binascii.Error) as exc:
        raise BrokerSigningError(f"release broker returned malformed signing data: {exc}") from exc
    return signatures, items


def _sign(kind: str, wire_items: list[dict], summary: str,
          config_dir: Path | None) -> tuple[list[bytes], list[bytes], str]:
    config = _read_config(config_dir)
    public_hex = load_release_public(config_dir)
    principal = _closed(_request, config, "GET", "/principal/v1")
    if principal.get("principal") != _PRINCIPAL:
        raise BrokerSigningError(
            "release broker authenticated the wrong principal; expected 'nakshatra-release'")
    exact_content = {"v": 1, "kind": kind, "items": wire_items}
    deadline = time.monotonic() + _WAIT_S
    enqueued = _closed(_request, config, "POST", "/enqueue/v1", {
        "what": _WHAT,
        "tier": "high",
        "review": {"description": summary},
        "exact_content": exact_content,
        "idempotency_key": f"{_PRINCIPAL}:{secrets.token_hex(16)}",
        "ttl": _WAIT_S,
    })
    request_id = enqueued.get("request_id")
    if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
        raise BrokerSigningError("release broker returned an invalid request id")
    result_path = "/result/v1?" + urllib.parse.urlencode({"request_id": request_id})
    while True:
        if time.monotonic() >= deadline:
            raise _uncertain(
                "result", request_id, "release approval deadline expired without a terminal result")
        result = _until(
            config, "GET", result_path, None, deadline=deadline,
            request_id=request_id, phase="result")
        status = result.get("status")
        if status in {"approved", "consumed"}:
            break
        if status in {"declined", "expired"}:
            raise BrokerSigningError("release signing was not approved; nothing was signed")
        if status != "pending":
            raise _uncertain(
                "result", request_id, "release broker returned an invalid terminal status")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _uncertain(
                "result", request_id, "release approval deadline expired while status was pending")
        time.sleep(min(_POLL_S, remaining))
    response = _until(
        config, "POST", "/release-sign/v1",
        {"request_id": request_id, "exact_content": exact_content},
        deadline=deadline, request_id=request_id, phase="sign")
    signatures, items = _decode_response(response, request_id, len(wire_items))
    return signatures, items, public_hex


def _signature_object(public_hex: str, signature: bytes) -> dict:
    return {"alg": "Ed25519", "keyid": public_hex[:16],
            "value": base64.b64encode(signature).decode("ascii")}


def _verified_signature(public_hex: str, signature: bytes, item: bytes, index: int) -> dict:
    value = _signature_object(public_hex, signature)
    if not rk.verify_signature(value, public_hex, item):
        raise BrokerSigningError(f"release broker returned a wrong signature for item {index}")
    return value


def sign_release(manifest: dict, config_dir: Path | None = None) -> tuple[dict, dict]:
    """Approve and sign one manifest and its bound latest pointer in a single broker act."""
    if not isinstance(manifest, dict) or "sig" in manifest:
        raise BrokerSigningError("release manifest must be one unsigned object")
    manifest_raw = rk.canonical(manifest)
    latest_template = {
        "schema": "nak-release-latest/1",
        "channel": manifest.get("channel"),
        "version": manifest.get("version"),
        "manifest_sha256": "0" * 64,
        "created": manifest.get("created"),
        "expires_at": manifest.get("created") + 60 * 86400
        if type(manifest.get("created")) is int else None,
    }
    latest_template_raw = rk.canonical(latest_template)
    signatures, returned, public_hex = _sign(
        "release",
        [{"b64": base64.b64encode(manifest_raw).decode("ascii")},
         {"b64": base64.b64encode(latest_template_raw).decode("ascii"), "record_ref": 0}],
        f"Sign Nakshatra {manifest.get('channel')} release {manifest.get('version')}",
        config_dir)
    if returned[0] != manifest_raw:
        raise BrokerSigningError("release broker substituted signing item 0")
    manifest_sig = _verified_signature(public_hex, signatures[0], returned[0], 0)
    signed_manifest = dict(manifest, sig=manifest_sig)
    manifest_text = json.dumps(signed_manifest, indent=1, sort_keys=True)
    expected_latest = dict(
        latest_template,
        manifest_sha256=hashlib.sha256(manifest_text.encode("utf-8")).hexdigest())
    expected_latest_raw = rk.canonical(expected_latest)
    if returned[1] != expected_latest_raw:
        raise BrokerSigningError("release broker substituted or incorrectly filled signing item 1")
    latest_sig = _verified_signature(public_hex, signatures[1], returned[1], 1)
    return signed_manifest, dict(expected_latest, sig=latest_sig)


def refresh_latest(latest: dict, config_dir: Path | None = None) -> dict:
    """Approve and re-sign one already-bound latest pointer."""
    if not isinstance(latest, dict):
        raise BrokerSigningError("latest pointer must be an object")
    unsigned = {key: value for key, value in latest.items() if key != "sig"}
    if set(unsigned) != _LATEST_KEYS:
        raise BrokerSigningError("latest pointer has malformed fields")
    raw = rk.canonical(unsigned)
    signatures, returned, public_hex = _sign(
        "refresh", [{"b64": base64.b64encode(raw).decode("ascii")}],
        f"Refresh Nakshatra {unsigned['channel']} release {unsigned['version']}", config_dir)
    if returned[0] != raw:
        raise BrokerSigningError("release broker substituted refresh signing item")
    signature = _verified_signature(public_hex, signatures[0], returned[0], 0)
    return dict(unsigned, sig=signature)


def sign_keyroll_new(statement: dict, config_dir: Path | None = None) -> dict:
    """Approve the new-key proof-of-possession signature on a key-roll statement."""
    if not isinstance(statement, dict) or set(statement) != _KEYROLL_KEYS:
        raise BrokerSigningError("key-roll statement has malformed fields")
    public_hex = load_release_public(config_dir)
    if statement.get("new_pubkey") != public_hex:
        raise BrokerSigningError("key-roll new_pubkey does not match release.pub")
    raw = rk.canonical_keyroll(statement)
    signatures, returned, signed_by = _sign(
        "keyroll", [{"b64": base64.b64encode(raw).decode("ascii")}],
        f"Roll Nakshatra release key {statement['old_pubkey'][:16]} to {public_hex[:16]}",
        config_dir)
    if signed_by != public_hex or returned[0] != raw:
        raise BrokerSigningError("release broker substituted key-roll signing item")
    return _verified_signature(public_hex, signatures[0], returned[0], 0)
