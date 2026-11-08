"""The single file-or-broker boundary for Solana pay-wallet signatures."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from network import broker_transport, pay_chain, pay_wallet, person_custody


_WHAT = "nakshatra.wallet.v1"
_TTL_S = 60
_BROKER_TIMEOUT_S = 3
_BROKER_POLL_S = 1
_BROKER_RETRY_INITIAL_S = 0.1
_BROKER_RETRY_MAX_S = 1.0
_NOT_APPROVED = "wallet signing was not approved; nothing was signed"
_INVALID_BROKER_DATA = "wallet broker returned invalid data; nothing was signed"
_UNCERTAIN_ENQUEUE = (
    "wallet approval enqueue remained uncertain; retry this exact act with the "
    "same idempotency_key; nothing was signed"
)
_IDEMPOTENCY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}\Z")
_NODE_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_ID_RE = re.compile(r"[0-9a-f]{16}\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}\Z")


@dataclass(frozen=True)
class _WalletAct:
    """One custody decision carried unchanged from preflight through signing."""

    home: Path
    wallet_id: str
    mode: str
    public_key: ed25519.Ed25519PublicKey
    seed: bytes | None = field(repr=False)
    config: broker_transport.BrokerConfig | None
    principal: str | None


def _net_dir(home: Path) -> Path:
    return Path(os.environ.get("NAK_NET_DIR", home / ".nakshatra" / "net"))


def _signer_dir(home: Path) -> Path:
    configured = os.environ.get("NAK_SIGNER_DIR")
    if configured:
        return Path(configured)
    default = person_custody._signer_dir()  # noqa: SLF001 - reuse the established public path
    try:
        return home / default.relative_to(Path.home())
    except ValueError:
        return default


def _broker_config_path(home: Path, net_dir: Path | None = None) -> Path:
    return (Path(net_dir) if net_dir is not None else _net_dir(home)) / "wallet-broker.json"


def _load_node_id(home: Path, signer_dir: Path | None = None) -> str:
    path = (Path(signer_dir) if signer_dir is not None else _signer_dir(home)) / "node.pub"
    try:
        raw = broker_transport.read_owned_regular(
            path, "node public identity", max_bytes=256)
        node_id = raw.decode("ascii").strip()
    except FileNotFoundError as exc:
        raise SystemExit(f"wallet broker node identity is missing at {path}; nothing was signed") from exc
    except UnicodeError as exc:
        raise SystemExit(f"wallet broker node identity at {path} is invalid; nothing was signed") from exc
    if _NODE_ID_RE.fullmatch(node_id) is None:
        raise SystemExit(f"wallet broker node identity at {path} is invalid; nothing was signed")
    return node_id


def _load_broker_config(home: Path, net_dir: Path | None = None) -> broker_transport.BrokerConfig:
    path = _broker_config_path(home, net_dir)
    return broker_transport.load_broker_config(
        path, subject="wallet",
        missing_message="wallet custody is broker but configuration is missing: {path}",
        token_missing_message="wallet broker token file is missing: {path}")


_BROKER_ERROR_IDS = {
    ("/principal/v1", 401, "unauthorized"): "auth",
    ("/enqueue/v1", 401, "unauthorized"): "auth",
    ("/result/v1", 401, "unauthorized"): "auth",
    ("/wallet-sign/v1", 401, "unauthorized"): "auth",
    ("/enqueue/v1", 400, "idempotency_key is missing or malformed"): "bad-idempotency",
    ("/enqueue/v1", 409, "idempotency_key already names different content"): "idempotency-conflict",
    ("/enqueue/v1", 409, "idempotent request is already consumed"): "request-consumed",
    ("/enqueue/v1", 409, "idempotent request is no longer available"): "request-gone",
    ("/result/v1", 403, "request belongs to a different principal"): "wrong-owner",
    ("/wallet-sign/v1", 403, "nakshatra wallet principal required"): "wrong-principal",
    ("/wallet-sign/v1", 409, "wallet signing request was already signed"): "already-signed",
    ("/wallet-sign/v1", 409, "wallet signing request was already consumed"): "already-signed",
    ("/wallet-sign/v1", 409,
     "wallet signing requires the matching live approved request"): "approval-mismatch",
    ("/wallet-sign/v1", 503, "approval ledger is unavailable"): "ledger-unavailable",
}


def _broker_error_info(path: str, exc: urllib.error.HTTPError) -> tuple[str, bool]:
    try:
        document = json.loads(exc.read(4096).decode("utf-8"))
    except (AttributeError, UnicodeError, ValueError, json.JSONDecodeError):
        return "", False
    if not isinstance(document, dict):
        return "", False
    value = document.get("error", "")
    if not isinstance(value, str):
        return "", document.get("retryable") is True
    route = urllib.parse.urlsplit(path).path
    return (_BROKER_ERROR_IDS.get((route, int(exc.code), value), ""),
            document.get("retryable") is True)


def _broker_json(config: broker_transport.BrokerConfig, method: str, path: str,
                 body: dict | None = None, *, timeout: float = _BROKER_TIMEOUT_S,
                 retry_transient: bool = False) -> dict:
    return broker_transport.request_json(
        config, method, path, body, opener=broker_transport.BROKER_OPENER,
        timeout=timeout, error_info=_broker_error_info,
        unavailable_message="wallet broker is unavailable; nothing was signed",
        invalid_message=_INVALID_BROKER_DATA,
        non_object_message=_INVALID_BROKER_DATA,
        retry_transient=retry_transient)


def _read_current_identity(act: _WalletAct) -> dict:
    custody_path = pay_wallet._paths(act.home)[3]  # noqa: SLF001 - custody boundary owns this check
    current_mode = pay_wallet._read_custody(custody_path)  # noqa: SLF001
    if current_mode != act.mode:
        raise SystemExit(
            f"wallet custody mode changed from {act.mode} to {current_mode} "
            "during this act; nothing was signed")
    identity = pay_wallet.load_wallet_identity(act.home)
    if identity.get("address") != act.wallet_id:
        raise SystemExit("wallet identity changed during this act; nothing was signed")
    return identity


def _preflight(wallet_id: str, *, home: Path, net_dir: Path | None,
               signer_dir: Path | None) -> _WalletAct:
    try:
        decoded = pay_wallet.base58_decode(wallet_id)
    except (TypeError, ValueError) as exc:
        raise SystemExit("wallet_id must be a canonical base58 Ed25519 public key") from exc
    if len(decoded) != 32 or pay_wallet.base58_encode(decoded) != wallet_id:
        raise SystemExit("wallet_id must be a canonical base58 Ed25519 public key")
    identity = pay_wallet.load_wallet_identity(home)
    if identity["address"] != wallet_id:
        raise SystemExit("wallet_id does not match wallet.pub; nothing was signed")
    public_key = ed25519.Ed25519PublicKey.from_public_bytes(identity["public_key"])
    mode = identity["custody"]
    if mode == "file":
        seed = identity.get("seed")
        if not isinstance(seed, bytes) or len(seed) != 32:
            raise SystemExit("file-custody wallet seed is unavailable; nothing was signed")
        return _WalletAct(home, wallet_id, mode, public_key, seed, None, None)

    config = _load_broker_config(home, net_dir)
    node_id = _load_node_id(home, signer_dir)
    principal = f"nakshatra-wallet:{node_id}"
    authenticated = _broker_json(config, "GET", "/principal/v1")
    if authenticated.get("principal") != principal:
        raise SystemExit(
            f"wallet broker authenticated the wrong principal; expected {principal!r}; "
            "nothing was signed")
    return _WalletAct(home, wallet_id, mode, public_key, None, config, principal)


def _uncertain_failure(phase: str, request_id: str, detail: str) -> SystemExit:
    if phase == "result":
        state = "approval may already have been granted"
    else:
        state = "approval was granted and signing may already have completed"
    return SystemExit(
        f"{detail}; {state}; request_id={request_id} may be retried only as the same act")


def _broker_json_until(config: broker_transport.BrokerConfig, method: str, path: str,
                       body: dict | None, *, deadline: float, request_id: str,
                       phase: str, act: _WalletAct) -> dict:
    return broker_transport.request_json_until(
        lambda req_method, req_path, req_body, **kwargs: _broker_json(
            config, req_method, req_path, req_body, **kwargs),
        method, path, body, deadline=deadline, request_id=request_id, phase=phase,
        assert_current=lambda: _read_current_identity(act),
        uncertain_failure=_uncertain_failure,
        deadline_detail="wallet approval deadline expired after a transient failure",
        invalid_state_detail="wallet broker returned an invalid signing state",
        already_signed_detail=(
            "the wallet-signing request was consumed, but its signature was lost in transit; "
            "a new human approval is required; nothing was signed"),
        timeout_s=_BROKER_TIMEOUT_S, retry_initial_s=_BROKER_RETRY_INITIAL_S,
        retry_max_s=_BROKER_RETRY_MAX_S)


def _enqueue_until(config: broker_transport.BrokerConfig, body: dict, *,
                   deadline: float, act: _WalletAct) -> str:
    """Recover a possibly accepted enqueue using only the caller's stable key."""
    delay = _BROKER_RETRY_INITIAL_S
    while True:
        _read_current_identity(act)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit(_UNCERTAIN_ENQUEUE)
        try:
            acknowledgement = _broker_json(
                config, "POST", "/enqueue/v1", body,
                timeout=min(_BROKER_TIMEOUT_S, remaining), retry_transient=True)
            request_id = acknowledgement.get("request_id")
            if isinstance(request_id, str) and _REQUEST_ID_RE.fullmatch(request_id) is not None:
                return request_id
        except broker_transport.BrokerTransient:
            pass
        except SystemExit as exc:
            if str(exc) != _INVALID_BROKER_DATA:
                raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit(_UNCERTAIN_ENQUEUE) from None
        time.sleep(min(delay, remaining))
        delay = min(_BROKER_RETRY_MAX_S, max(delay * 2, _BROKER_RETRY_INITIAL_S))


def _broker_sign(raw_message: bytes, idempotency_key: str, act: _WalletAct) -> bytes:
    if act.config is None:
        raise SystemExit("wallet broker configuration was not carried from preflight")
    exact_content = {
        "v": 1,
        "wallet_id": act.wallet_id,
        "message_b64": base64.b64encode(raw_message).decode("ascii"),
    }
    enqueue_body = {
        "what": _WHAT,
        "idempotency_key": idempotency_key,
        "tier": "high",
        "review": {"description": "Solana payment transaction"},
        "exact_content": exact_content,
        "ttl": _TTL_S,
    }
    deadline = time.monotonic() + _TTL_S
    request_id = _enqueue_until(act.config, enqueue_body, deadline=deadline, act=act)

    result_path = "/result/v1?" + urllib.parse.urlencode({"request_id": request_id})
    while True:
        result = _broker_json_until(
            act.config, "GET", result_path, None, deadline=deadline,
            request_id=request_id, phase="result", act=act)
        returned_id = result.get("request_id")
        if returned_id is not None and returned_id != request_id:
            raise SystemExit("wallet broker substituted the approval request id; nothing was signed")
        status = result.get("status")
        if status == "approved":
            break
        if status in {"declined", "expired"}:
            raise SystemExit(_NOT_APPROVED)
        if status == "consumed":
            raise SystemExit(
                "the wallet-signing request was consumed, but its signature is unavailable; "
                "a new human approval is required; nothing was signed")
        if status != "pending":
            raise _uncertain_failure(
                "result", request_id,
                "wallet broker returned an invalid terminal status; nothing was signed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _uncertain_failure(
                "result", request_id,
                "wallet approval deadline expired while status was pending")
        time.sleep(min(_BROKER_POLL_S, remaining))

    response = _broker_json_until(
        act.config, "POST", "/wallet-sign/v1",
        {"request_id": request_id, "exact_content": exact_content},
        deadline=deadline, request_id=request_id, phase="sign", act=act)
    if response.get("request_id") != request_id:
        raise SystemExit("wallet broker substituted the signing request id; nothing was accepted")
    if response.get("wallet_id") != act.wallet_id:
        raise SystemExit("wallet broker substituted wallet_id; nothing was accepted")
    expected_digest = hashlib.sha256(raw_message).hexdigest()
    digest = response.get("message_sha256")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None or digest != expected_digest:
        raise SystemExit("wallet broker returned the wrong message digest; nothing was accepted")

    signatures, items = response.get("signatures"), response.get("items_b64")
    if (not isinstance(signatures, list) or not isinstance(items, list)
            or len(signatures) != 1 or len(items) != 1):
        raise SystemExit(
            "wallet broker returned the wrong number of signatures or items; nothing was accepted")
    signature_hex, item_b64 = signatures[0], items[0]
    try:
        if not isinstance(signature_hex, str) or _SIGNATURE_RE.fullmatch(signature_hex) is None:
            raise ValueError("invalid signature encoding")
        signature = bytes.fromhex(signature_hex)
        if not isinstance(item_b64, str):
            raise ValueError("invalid item encoding")
        returned_message = base64.b64decode(item_b64, validate=True)
        if base64.b64encode(returned_message).decode("ascii") != item_b64:
            raise ValueError("non-canonical item encoding")
    except (ValueError, binascii.Error) as exc:
        raise SystemExit(
            f"wallet broker returned malformed signing data ({exc}); nothing was accepted") from exc
    if returned_message != raw_message:
        raise SystemExit("wallet broker substituted the signed message; nothing was accepted")
    try:
        act.public_key.verify(signature, returned_message)
    except InvalidSignature as exc:
        raise SystemExit("wallet broker returned a wrong wallet signature; nothing was accepted") from exc
    _read_current_identity(act)
    return signature


def wallet_sign(raw_message: bytes, wallet_id: str, *, idempotency_key: str,
                home: Path | None = None, net_dir: Path | None = None,
                signer_dir: Path | None = None) -> bytes:
    """Sign one compiled Solana message through the custody backend pinned at preflight."""
    if not isinstance(raw_message, bytes) or not raw_message:
        raise SystemExit("raw_message must be non-empty bytes; nothing was signed")
    if not isinstance(idempotency_key, str) or _IDEMPOTENCY_RE.fullmatch(idempotency_key) is None:
        raise SystemExit("idempotency_key is missing or malformed; nothing was signed")
    selected_home = Path.home() if home is None else Path(home)
    act = _preflight(
        wallet_id, home=selected_home, net_dir=net_dir, signer_dir=signer_dir)
    _read_current_identity(act)
    if act.mode == "broker":
        return _broker_sign(raw_message, idempotency_key, act)
    if act.seed is None:
        raise SystemExit("file-custody wallet seed is unavailable; nothing was signed")
    signature = pay_chain._private_key(act.seed).sign(raw_message)  # noqa: SLF001
    _read_current_identity(act)
    return signature
