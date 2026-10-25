"""The single boundary for signatures made with the person's authority key."""
from __future__ import annotations

import base64
import binascii
import contextvars
import hashlib
import json
import os
import re
import secrets
import stat
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


KINDS = {
    "contact-accept",
    "address-control",
    "task-verdict",
    "grant-issue",
    "grant-renew",
    "spend-cap",
    "invite",
    "revoke",
    "report",
    "suspend",
}
PREFLIGHT_ONLY_KINDS = {"resolve"}

_PRINCIPAL = "nakshatra-person"
_WHAT = "nakshatra.person.v1"
_BROKER_TIMEOUT_S = 3
_BROKER_DEFAULT_WAIT_S = 300
_BROKER_POLL_S = 1
_BROKER_V1_MIN_TTL_S = 10  # pillar_signer_broker.V1_MIN_TTL; compatibility is asserted in tests.
_PROOF_SAFETY_MARGIN_S = 30
_NOT_APPROVED = "person signing was not approved; nothing was signed"
_HOME_OVERRIDE: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "person_custody_home", default=None)
_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_ZERO_RECORD_HASH = "0" * 64
_SUSPENSION_PROOF_KINDS = {
    "stage-suspension",
    "install-suspension",
    "abandon-suspension",
    "recover-suspension",
    "import-suspension",
}
_PERSON_ACT_SENTINEL = object()
_BROKER_RETRY_INITIAL_S = 0.1
_BROKER_RETRY_MAX_S = 1.0


@dataclass(frozen=True)
class _BrokerConfig:
    url: str
    token: str


@dataclass(frozen=True, init=False)
class PersonAct:
    """One custody decision carried unchanged from preflight through signing."""
    kind: str
    summary: str
    mode: str
    config: _BrokerConfig | None
    public_key: ed25519.Ed25519PublicKey | None
    _sentinel: object = field(repr=False, compare=False)

    def __init__(self, kind: str, summary: str, mode: str,
                 config: _BrokerConfig | None,
                 public_key: ed25519.Ed25519PublicKey | None, *,
                 _sentinel: object | None = None):
        if _sentinel is not _PERSON_ACT_SENTINEL:
            raise TypeError("PersonAct construction is private; use broker_preflight()")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "public_key", public_key)
        object.__setattr__(self, "_sentinel", _sentinel)

    def for_signing(self, kind: str, summary: str) -> "PersonAct":
        """Refine presentation/kind while preserving the pinned custody configuration."""
        _check_request(kind, summary)
        _assert_act_mode(self)
        return _new_person_act(kind, summary, self.mode, self.config, self.public_key)


def _new_person_act(kind: str, summary: str, mode: str,
                    config: _BrokerConfig | None,
                    public_key: ed25519.Ed25519PublicKey | None) -> PersonAct:
    return PersonAct(
        kind, summary, mode, config, public_key, _sentinel=_PERSON_ACT_SENTINEL)


def _home() -> Path:
    return _HOME_OVERRIDE.get() or Path.home()


def _net_dir() -> Path:
    home_override = _HOME_OVERRIDE.get()
    if home_override is not None:
        return home_override / ".nakshatra" / "net"
    return Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net"))


def _custody_path() -> Path:
    return _net_dir() / "person-custody"


def _broker_config_path() -> Path:
    return _net_dir() / "person-broker.json"


def _person_key_path(home: Path | None = None) -> Path:
    return (home or _home()) / ".sthambha" / "person-TEST.key"


def _signer_dir() -> Path:
    home_override = _HOME_OVERRIDE.get()
    if home_override is not None:
        return home_override / ".sthambha" / "signer"
    return Path(os.environ.get("NAK_SIGNER_DIR", Path.home() / ".sthambha" / "signer"))


def _check_request(kind: str, _summary: str) -> None:
    if kind not in KINDS:
        expected = ", ".join(sorted(KINDS))
        raise SystemExit(f"unknown person-signing kind {kind!r}; expected one of: {expected}")


_BIDI_CONTROLS = {
    "\u061c", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\u2066", "\u2067", "\u2068", "\u2069",
}


def plain(text: object, limit: int) -> str:
    """Make untrusted text safe for one-line approval summaries."""
    if limit <= 0:
        return ""
    clean = "".join(
        char for char in str(text)
        if unicodedata.category(char) != "Cc" and char not in _BIDI_CONTROLS
    )
    clean = " ".join(clean.split())
    if len(clean) <= limit:
        return clean
    if limit == 1:
        return "…"
    return clean[:limit - 1].rstrip() + "…"


def _read_owned_regular(path: Path, label: str, *, exact_mode: int | None = None,
                        max_bytes: int = 16 * 1024) -> bytes:
    """Read a small euid-owned file without following or racing a symlink."""
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise SystemExit(f"cannot read {label} {path}: {exc}") from exc
    mode = stat.S_IMODE(before.st_mode)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"{label} {path} must be a regular file")
    if before.st_uid != os.geteuid():
        raise SystemExit(f"{label} {path} must be owned by the effective user")
    if exact_mode is not None:
        if mode != exact_mode:
            raise SystemExit(f"{label} {path} must have mode {exact_mode:04o}")
    elif mode & 0o022:
        raise SystemExit(f"{label} {path} must not be group/world writable")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise SystemExit(f"cannot read {label} {path}: {exc}") from exc
    try:
        after = os.fstat(fd)
        if (not stat.S_ISREG(after.st_mode) or after.st_uid != os.geteuid()
                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)):
            raise SystemExit(f"{label} {path} changed while it was being opened")
        after_mode = stat.S_IMODE(after.st_mode)
        if ((exact_mode is not None and after_mode != exact_mode)
                or (exact_mode is None and after_mode & 0o022)):
            raise SystemExit(f"{label} {path} permissions changed while it was being opened")
        if not 0 < after.st_size <= max_bytes:
            raise SystemExit(f"{label} {path} is empty or too large")
        raw = b""
        while len(raw) <= max_bytes:
            chunk = os.read(fd, min(4096, max_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
        if not raw or len(raw) > max_bytes:
            raise SystemExit(f"{label} {path} is empty or too large")
        return raw
    finally:
        os.close(fd)


def custody_mode() -> str:
    """Return the configured backend once. Missing marker preserves legacy file custody.

    A broker configuration is durable evidence that custody was configured.  Once it exists,
    losing (or being unable to read) the marker must never silently reactivate a leftover file key.
    The marker may be absent only on a legacy installation with no broker configuration at all.
    """
    path = _custody_path()
    try:
        mode = _read_owned_regular(path, "person custody marker").decode("utf-8").strip()
    except FileNotFoundError:
        try:
            _broker_config_path().lstat()
        except FileNotFoundError:
            return "file"
        except OSError as exc:
            raise SystemExit(
                f"person broker configuration exists but cannot be inspected; "
                f"refusing file custody: {_broker_config_path()}") from exc
        raise SystemExit(
            "person broker configuration exists but the person custody marker is missing; "
            "refusing file custody")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"person custody marker {path} is not UTF-8 text") from exc
    if mode not in {"file", "broker"}:
        raise SystemExit(
            f"invalid person custody mode {mode!r} in {path}; expected one word: file or broker")
    return mode


def _assert_act_mode(act: PersonAct) -> str:
    """Reject forged acts and custody changes before any signing-key operation."""
    if not isinstance(act, PersonAct) or getattr(act, "_sentinel", None) is not _PERSON_ACT_SENTINEL:
        raise SystemExit("invalid person custody act; nothing was signed")
    current = custody_mode()
    if current != act.mode:
        raise SystemExit(
            f"person custody mode changed from {act.mode} to {current} during this act; "
            "nothing was signed")
    return current


def _raw_file_key(path: Path) -> tuple[ed25519.Ed25519PrivateKey, str]:
    """Parse the historical raw-hex person-key format without signer-state coupling."""
    seed = bytes.fromhex(Path(path).read_text().strip())
    key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
    return key, key.public_key().public_bytes_raw().hex()


def _person_public_key() -> ed25519.Ed25519PublicKey:
    path = _signer_dir() / "person.pub"
    try:
        person_hex = path.read_text().strip()
        if _HEX64_RE.fullmatch(person_hex) is None:
            raise ValueError("expected 64 lowercase hex characters")
        return ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(person_hex))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot load the public person identity from {path}: {exc}") from exc


def _load_matching_person_key(*, error_context: str = "this decision") -> ed25519.Ed25519PrivateKey:
    """Load the legacy file key and bind it to the active signer's public authority."""
    from sthambha import signer as signer_mod

    signer_dir = _signer_dir()
    person_key = _person_key_path()
    try:
        signer_person = (signer_dir / "person.pub").read_text().strip()
        person_priv, person_pub = signer_mod.load_key(person_key)
    except (OSError, ValueError, signer_mod.SignerError) as exc:
        raise SystemExit(f"cannot load the person key for {error_context}: {exc}") from exc
    if person_pub != signer_person:
        raise SystemExit(f"the person key at {person_key} does not match {signer_dir / 'person.pub'}")
    return person_priv


def _read_private_token(path: Path) -> str:
    try:
        raw = _read_owned_regular(
            path, "person broker token file", exact_mode=0o600, max_bytes=16 * 1024)
        token = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise SystemExit(f"person broker token file {path} is not UTF-8 text") from exc
    if not token:
        raise SystemExit(f"person broker token file {path} is empty")
    return token


def _broker_config() -> _BrokerConfig:
    path = _broker_config_path()
    try:
        raw = _read_owned_regular(path, "person broker configuration")
        document = json.loads(raw.decode("utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"person custody is broker but configuration is missing: {path}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid person broker configuration {path}") from exc
    if not isinstance(document, dict) or set(document) != {"url", "token_file"}:
        raise SystemExit(
            f"invalid person broker configuration {path}: expected only url and token_file")
    url, token_file = document.get("url"), document.get("token_file")
    if not isinstance(url, str) or not isinstance(token_file, str) or not token_file:
        raise SystemExit(f"invalid person broker configuration {path}: url and token_file must be text")
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise SystemExit(f"invalid person broker URL in {path}: {exc}") from exc
    if (parsed.scheme != "http" or not parsed.hostname or port is None or parsed.username is not None
            or parsed.password is not None or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment):
        raise SystemExit(
            f"invalid person broker URL in {path}: expected http://host:port with no path")
    return _BrokerConfig(url=url.rstrip("/"), token=_read_private_token(Path(token_file)))


class _NoBrokerRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _req, _fp, _code, _msg, _headers, _newurl):
        return None


_BROKER_OPENER = urllib.request.build_opener(_NoBrokerRedirects)


_BROKER_ERROR_IDS = {
    # HTTP authentication wrappers shared by these routes.
    ("/principal/v1", 401, "unauthorized"): "auth",
    ("/enqueue/v1", 401, "unauthorized"): "auth",
    ("/result/v1", 401, "unauthorized"): "auth",
    ("/person-sign/v1", 401, "unauthorized"): "auth",
    # enqueue_v1
    ("/enqueue/v1", 404, "v1_disabled"): "v1-off",
    ("/enqueue/v1", 400, "JSON object required"): "bad-json",
    ("/enqueue/v1", 403, "requester does not match the authenticated principal"): "principal-mismatch",
    ("/enqueue/v1", 400, "what must be a versioned registry action"): "bad-action",
    ("/enqueue/v1", 400, "tier must be low, medium, or high"): "bad-tier",
    ("/enqueue/v1", 400, "idempotency_key is missing or malformed"): "bad-idempotency",
    ("/enqueue/v1", 400, "review must contain exactly description"): "bad-review",
    ("/enqueue/v1", 400, "review.description must be 1-500 characters"): "bad-review-text",
    ("/enqueue/v1", 400, "exact_content is required"): "missing-content",
    ("/enqueue/v1", 503, "person key not installed"): "key-unavailable",
    ("/enqueue/v1", 400,
     "nakshatra-person requires what=nakshatra.person.v1 and tier=high"): "bad-person-envelope",
    ("/enqueue/v1", 400, "nakshatra person request was refused"): "person-refused",
    ("/enqueue/v1", 400, "exact_content is too large (max 20000 canonical bytes)"): "content-too-large",
    ("/enqueue/v1", 400, "ttl must be 10-300 seconds"): "bad-ttl",
    ("/enqueue/v1", 409, "idempotency_key already names different content"): "idempotency-conflict",
    ("/enqueue/v1", 503, "approval ledger is unavailable"): "ledger-unavailable",
    ("/enqueue/v1", 409, "idempotent request is no longer available"): "request-gone",
    ("/enqueue/v1", 409, "idempotent request is already approved"): "request-terminal",
    ("/enqueue/v1", 409, "idempotent request is already declined"): "request-terminal",
    ("/enqueue/v1", 409, "idempotent request is already expired"): "request-terminal",
    ("/enqueue/v1", 409, "idempotent request is already consumed"): "request-terminal",
    # result_v1
    ("/result/v1", 404, "v1_disabled"): "v1-off",
    ("/result/v1", 403, "request belongs to a different principal"): "wrong-owner",
    # person_sign_v1 and its broker-ledger refusals.
    ("/person-sign/v1", 404, "not found"): "route-missing",
    ("/person-sign/v1", 403, "nakshatra-person principal required"): "wrong-principal",
    ("/person-sign/v1", 503, "person key not installed"): "key-unavailable",
    ("/person-sign/v1", 400, "person signing request has an invalid shape"): "bad-sign-request",
    ("/person-sign/v1", 400, "request_id is required"): "missing-request-id",
    ("/person-sign/v1", 400, "nakshatra person request was refused"): "person-refused",
    ("/person-sign/v1", 409, "person signing request is malformed"): "bad-sign-request",
    ("/person-sign/v1", 409,
     "person signing requires the matching live approved request"): "approval-mismatch",
    ("/person-sign/v1", 409,
     "person signing requires a biometric approval door"): "non-biometric",
    ("/person-sign/v1", 409,
     "person signing approval receipt is unavailable"): "receipt-unavailable",
    ("/person-sign/v1", 409,
     "person signing approval receipt does not match its door"): "receipt-door-mismatch",
    ("/person-sign/v1", 409, "person signing request was already signed"): "already-signed",
    ("/person-sign/v1", 409,
     "person signing callback returned an invalid batch"): "invalid-sign-batch",
    ("/person-sign/v1", 409,
     "person signing callback returned malformed output"): "invalid-sign-output",
    ("/person-sign/v1", 503, "approval ledger is unavailable"): "ledger-unavailable",
    ("/person-sign/v1", 500, "person signing failed"): "signing-failed",
}


def _broker_error_info(path: str, exc: urllib.error.HTTPError) -> tuple[str, bool]:
    """Map a response to a closed local identifier and its explicit retry hint."""
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


class _BrokerTransient(Exception):
    """A transport or broker-declared transient failure safe to retry in one act."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _broker_json(config: _BrokerConfig, method: str, path: str,
                 body: dict | None = None, *, timeout: float = _BROKER_TIMEOUT_S,
                 retry_transient: bool = False) -> dict:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = {"Authorization": f"Bearer {config.token}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        config.url + path, data=data, headers=headers, method=method)
    try:
        with _BROKER_OPENER.open(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_id, broker_retryable = _broker_error_info(path, exc)
        message = (f"approval service error (HTTP {exc.code}, code {error_id})"
                   if error_id else f"approval service error (HTTP {exc.code})")
        if retry_transient and (int(exc.code) in {502, 503, 504} or broker_retryable):
            raise _BrokerTransient(message) from None
        raise SystemExit(message) from None
    except (OSError, urllib.error.URLError, TimeoutError, ConnectionError):
        message = "person broker is unavailable; nothing was signed"
        if retry_transient:
            raise _BrokerTransient(message) from None
        raise SystemExit(message) from None
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise SystemExit(
            "person broker is unavailable or returned invalid data; nothing was signed") from None
    if not isinstance(result, dict):
        raise SystemExit("person broker returned invalid data; nothing was signed")
    return result


def _uncertain_broker_failure(phase: str, request_id: str, detail: str) -> SystemExit:
    if phase == "result":
        state = "approval may already have been granted"
    else:
        state = "approval was granted and signing may already have completed"
    return SystemExit(
        f"{detail}; {state}; request_id={request_id} can be retried without a new prompt")


def _broker_json_until(config: _BrokerConfig, method: str, path: str,
                       body: dict | None, *, deadline: float,
                       request_id: str, phase: str, act: PersonAct) -> dict:
    """Retry one result/sign request with bounded backoff inside the act deadline."""
    delay = _BROKER_RETRY_INITIAL_S
    while True:
        _assert_act_mode(act)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _uncertain_broker_failure(
                phase, request_id, "person approval deadline expired after a transient failure")
        try:
            return _broker_json(
                config, method, path, body,
                timeout=min(_BROKER_TIMEOUT_S, remaining), retry_transient=True)
        except _BrokerTransient as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _uncertain_broker_failure(phase, request_id, exc.message) from None
            sleep_for = min(delay, remaining)
            time.sleep(sleep_for)
            delay = min(_BROKER_RETRY_MAX_S, max(delay * 2, _BROKER_RETRY_INITIAL_S))
        except SystemExit as exc:
            raise _uncertain_broker_failure(phase, request_id, str(exc)) from None


def broker_preflight(kind: str, summary: str) -> PersonAct:
    """Refuse an unavailable broker before prompts, writes, or state-changing calls."""
    if kind not in PREFLIGHT_ONLY_KINDS:
        _check_request(kind, summary)
    mode = custody_mode()
    config = None
    if mode == "broker":
        config = _broker_config()
        principal = _broker_json(config, "GET", "/principal/v1")
        if principal.get("principal") != _PRINCIPAL:
            raise SystemExit(
                "person broker authenticated the wrong principal; expected "
                 "'nakshatra-person'; nothing was signed")
    public_key = _person_public_key() if mode == "broker" else None
    return _new_person_act(kind, summary, mode, config, public_key)


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _parse_canonical_object(raw: bytes, label: str) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label}: expected canonical JSON object") from exc
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise SystemExit(f"invalid {label}: expected canonical JSON object")
    return value


def _fill_suspension_template(record_raw: bytes, record_signature: bytes,
                              template_raw: bytes, person_hex: str) -> bytes:
    from network import suspension

    unsigned = _parse_canonical_object(record_raw, "suspension record reference")
    if set(unsigned) != {"schema", "person", "seq", "issued_at", "nodes", "reasons"}:
        raise SystemExit("invalid suspension record reference: expected an unsigned nak.suspensions.v1 record")
    if unsigned.get("person") != person_hex:
        raise SystemExit("invalid suspension record reference: record person does not match signer/person.pub")
    record = dict(unsigned, sig=record_signature.hex())
    try:
        suspension._validate(record, person_hex)  # noqa: SLF001 - the protocol requires this exact shape
    except ValueError as exc:
        raise SystemExit(f"invalid suspension record reference: {exc}") from exc
    signed_record = _canonical(record)

    template = _parse_canonical_object(template_raw, "suspension proof template")
    if (set(template) != {"kind", "payload", "ts"}
            or template.get("kind") not in _SUSPENSION_PROOF_KINDS
            or template.get("payload") != {"record_sha256": _ZERO_RECORD_HASH}
            or type(template.get("ts")) is not int):
        raise SystemExit(
            "invalid suspension proof template: expected a closed person-action with a zero record hash")
    filled = dict(template)
    filled["payload"] = {"record_sha256": hashlib.sha256(signed_record).hexdigest()}
    return _canonical(filled)


def _validate_context(kind: str, items: list, context: dict | None) -> dict | None:
    if context is None:
        return None
    if not isinstance(context, dict) or set(context) != {"verdict_reason"}:
        raise SystemExit("invalid person-signing context: only verdict_reason is allowed")
    reason = context.get("verdict_reason")
    if kind != "task-verdict" or not isinstance(reason, str) or len(reason) > 300:
        raise SystemExit("invalid verdict_reason context for person signing")
    digest = hashlib.sha256(reason.encode("utf-8")).hexdigest()
    found = False
    for item in items:
        raw = item if isinstance(item, bytes) else item[2]
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("reason_sha256") == digest:
            found = True
    if not found:
        raise SystemExit("verdict_reason context does not match any verdict item")
    return dict(context)


def _validated_items(items: list, person_hex: str) -> tuple[list, list[dict]]:
    if not isinstance(items, list) or not 1 <= len(items) <= 8:
        raise SystemExit("person-signing batch must contain 1 through 8 items")
    normalized = []
    wire = []
    for index, item in enumerate(items):
        if isinstance(item, bytes):
            if not item:
                raise SystemExit("person-signing statement must not be empty")
            normalized.append(item)
            wire.append({"b64": base64.b64encode(item).decode("ascii")})
            continue
        if (not isinstance(item, tuple) or len(item) != 3 or item[0] != "ref"
                or type(item[1]) is not int or not isinstance(item[2], bytes) or not item[2]):
            raise SystemExit("person-signing batch items must be bytes or ('ref', index, template_bytes)")
        ref, template = item[1], item[2]
        if not 0 <= ref < index or not isinstance(normalized[ref], bytes):
            raise SystemExit("person-signing record_ref must name an earlier plain item")
        _fill_suspension_template(normalized[ref], b"\0" * 64, template, person_hex)
        normalized.append(("ref", ref, template))
        wire.append({"b64": base64.b64encode(template).decode("ascii"), "record_ref": ref})
    return normalized, wire


def _error_context(kind: str) -> str:
    return "renewal" if kind in {"grant-renew", "spend-cap"} else "this decision"


def _proof_skew_s(kind: str) -> int | None:
    """Use nakd's actual verifier constants for statements that age during approval."""
    if kind in {"contact-accept", "address-control"}:
        from network import nakd
        return nakd.ACCEPT_PROOF_MAX_SKEW_S
    if kind in {"report", "suspend"}:
        from network import nakd
        return nakd.PERSON_ACTION_PROOF_MAX_SKEW_S
    if kind == "task-verdict":
        from network import verdict
        return verdict.DECISION_MAX_SKEW_S
    return None


def _approval_wait_s(kind: str) -> int:
    skew = _proof_skew_s(kind)
    if skew is None:
        return _BROKER_DEFAULT_WAIT_S
    desired = skew - _PROOF_SAFETY_MARGIN_S
    if desired < _BROKER_V1_MIN_TTL_S:
        raise SystemExit(
            "person broker minimum TTL is incompatible with nakd proof freshness: "
            f"kind={kind}, skew={skew}s, safety_margin={_PROOF_SAFETY_MARGIN_S}s, "
            f"broker_min={_BROKER_V1_MIN_TTL_S}s; nothing was signed")
    return max(desired, _BROKER_V1_MIN_TTL_S)


def _refuse_stale_signed_items(kind: str, results: list[tuple[bytes, bytes]], *,
                               now: int | None = None) -> None:
    """Last local fence before a caller can send or persist a newly signed short proof."""
    skew = _proof_skew_s(kind)
    if skew is None:
        return
    current = int(time.time()) if now is None else int(now)
    for raw, _signature in results:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            continue
        ts = value.get("ts") if isinstance(value, dict) else None
        if type(ts) is int and current - ts > skew:
            raise SystemExit(
                f"person approval completed after the {skew}s nakd proof window; "
                "nothing was sent or written")


def _file_sign_batch(kind: str, items: list, person_hex: str,
                     act: PersonAct) -> list[tuple[bytes, bytes]]:
    _assert_act_mode(act)
    key = _load_matching_person_key(error_context=_error_context(kind))
    results: list[tuple[bytes, bytes]] = []
    for item in items:
        if isinstance(item, bytes):
            final = item
        else:
            _marker, ref, template = item
            final = _fill_suspension_template(
                results[ref][0], results[ref][1], template, person_hex)
        _assert_act_mode(act)
        results.append((final, key.sign(final)))
    return results


def _broker_sign_batch(kind: str, items: list, wire_items: list[dict], *, summary: str,
                       context: dict | None, person_key: ed25519.Ed25519PublicKey,
                       person_hex: str, config: _BrokerConfig,
                       act: PersonAct) -> list[tuple[bytes, bytes]]:
    wait_s = _approval_wait_s(kind)
    exact_content = {"v": 1, "kind": kind, "items": wire_items}
    if context is not None:
        exact_content["context"] = context
    deadline = time.monotonic() + wait_s
    _assert_act_mode(act)
    enqueued = _broker_json(config, "POST", "/enqueue/v1", {
        "what": _WHAT,
        "tier": "high",
        "review": {"description": summary},
        "exact_content": exact_content,
        "idempotency_key": f"{_PRINCIPAL}:{secrets.token_hex(16)}",
        "ttl": wait_s,
    })
    request_id = enqueued.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise SystemExit("person broker returned no request id; nothing was signed")

    result_path = "/result/v1?" + urllib.parse.urlencode({"request_id": request_id})
    while True:
        if time.monotonic() >= deadline:
            raise _uncertain_broker_failure(
                "result", request_id, "person approval deadline expired without a terminal result")
        result = _broker_json_until(
            config, "GET", result_path, None, deadline=deadline,
            request_id=request_id, phase="result", act=act)
        status = result.get("status")
        if status in {"approved", "consumed"}:
            if time.monotonic() >= deadline:
                raise _uncertain_broker_failure(
                    "result", request_id, "person approval deadline expired")
            break
        if status in {"declined", "expired"}:
            raise SystemExit(_NOT_APPROVED)
        if status != "pending":
            raise _uncertain_broker_failure(
                "result", request_id,
                "person broker returned an invalid terminal status; nothing was signed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _uncertain_broker_failure(
                "result", request_id, "person approval deadline expired while status was pending")
        time.sleep(min(_BROKER_POLL_S, remaining))

    response = _broker_json_until(
        config, "POST", "/person-sign/v1",
        {"request_id": request_id, "exact_content": exact_content},
        deadline=deadline, request_id=request_id, phase="sign", act=act)
    if response.get("request_id") != request_id:
        raise SystemExit("person broker substituted the signing request id; nothing was accepted")
    signatures_raw, returned_raw = response.get("signatures"), response.get("items_b64")
    if (not isinstance(signatures_raw, list) or not isinstance(returned_raw, list)
            or len(signatures_raw) != len(items) or len(returned_raw) != len(items)):
        raise SystemExit("person broker returned the wrong number of signatures or items; nothing was accepted")
    signatures = []
    returned = []
    try:
        for signature in signatures_raw:
            if not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{128}", signature) is None:
                raise ValueError("invalid signature encoding")
            signatures.append(bytes.fromhex(signature))
        for encoded in returned_raw:
            if not isinstance(encoded, str):
                raise ValueError("invalid item encoding")
            returned.append(base64.b64decode(encoded, validate=True))
    except (ValueError, binascii.Error) as exc:
        raise SystemExit(f"person broker returned malformed signing data: {exc}; nothing was accepted") from exc

    for index, item in enumerate(items):
        expected = item if isinstance(item, bytes) else _fill_suspension_template(
            returned[item[1]], signatures[item[1]], item[2], person_hex)
        if returned[index] != expected:
            raise SystemExit(f"person broker substituted signing item {index}; nothing was accepted")
        try:
            person_key.verify(signatures[index], returned[index])
        except InvalidSignature as exc:
            raise SystemExit(f"person broker returned a wrong signature for item {index}; "
                             "nothing was accepted") from exc
    return list(zip(returned, signatures))


def person_sign_batch(kind: str, items: list, *, summary: str,
                      context: dict | None = None,
                      act: PersonAct | None = None) -> list[tuple[bytes, bytes]]:
    """Sign one atomic 1-8 item act, resolving suspension-record templates in order."""
    _check_request(kind, summary)
    act = broker_preflight(kind, summary) if act is None else act
    _assert_act_mode(act)
    if act.kind != kind:
        raise SystemExit("person custody act kind changed during signing; nothing was signed")
    mode = act.mode
    public_key = act.public_key or _person_public_key()
    person_hex = public_key.public_bytes_raw().hex()
    normalized, wire = _validated_items(items, person_hex)
    checked_context = _validate_context(kind, normalized, context)
    if mode == "broker":
        if act.config is None:
            raise SystemExit("person broker configuration was not carried from preflight")
        results = _broker_sign_batch(
            kind, normalized, wire, summary=summary, context=checked_context,
            person_key=public_key, person_hex=person_hex, config=act.config, act=act)
    else:
        results = _file_sign_batch(kind, normalized, person_hex, act)
    _refuse_stale_signed_items(kind, results)
    return results


def person_sign(kind: str, statement: bytes, *, summary: str,
                context: dict | None = None, act: PersonAct | None = None) -> bytes:
    """Sign one classified statement through the configured person-custody backend."""
    return person_sign_batch(
        kind, [statement], summary=summary, context=context, act=act)[0][1]


@contextmanager
def _signing_home(home: Path):
    """Keep explicit-home bootstrap work inside that home, including in tests."""
    token = _HOME_OVERRIDE.set(Path(home))
    try:
        yield
    finally:
        _HOME_OVERRIDE.reset(token)


class PersonKey:
    """The private-key-shaped adapter required by joincode and sthambha.signer."""

    def __init__(self, kind: str, summary: str, *, key_path: Path | None = None,
                 act: PersonAct | None = None):
        _check_request(kind, summary)
        self._kind = kind
        self._summary = summary
        self._act = broker_preflight(kind, summary) if act is None else act
        _assert_act_mode(self._act)
        if self._act.kind != kind:
            raise SystemExit("person custody act kind changed before signing; nothing was signed")
        mode = self._act.mode
        self._mode = mode
        if key_path is not None and mode == "broker":
            raise SystemExit("person custody is broker; --person-key is not allowed")
        if mode == "broker":
            self._key = None
            if self._act.public_key is None:
                raise SystemExit("person broker public identity was not carried from preflight")
            self._public = self._act.public_key
        elif key_path is not None:
            # Invites historically parsed this raw file without requiring signer/person.pub.
            self._key, _person_pub = _raw_file_key(Path(key_path))
            self._public = self._key.public_key()
        else:
            self._key = _load_matching_person_key(error_context=_error_context(kind))
            self._public = self._key.public_key()

    def sign(self, data: bytes) -> bytes:
        if not isinstance(data, bytes) or not data:
            raise SystemExit("person-signing statement must not be empty")
        _assert_act_mode(self._act)
        if self._key is not None:
            _assert_act_mode(self._act)
            result = [(data, self._key.sign(data))]
            _refuse_stale_signed_items(self._kind, result)
            return result[0][1]
        return person_sign(self._kind, data, summary=self._summary, act=self._act)

    def public_key(self) -> ed25519.Ed25519PublicKey:
        return self._public


def custody_label() -> str:
    return "test" if custody_mode() == "file" else "broker"
