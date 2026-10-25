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
from dataclasses import dataclass
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
_BROKER_WAIT_S = 300
_BROKER_POLL_S = 1
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


@dataclass(frozen=True)
class _BrokerConfig:
    url: str
    token: str


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


def custody_mode() -> str:
    """Return the configured person-key backend. Missing configuration preserves file custody."""
    path = _custody_path()
    try:
        mode = path.read_text().strip()
    except FileNotFoundError:
        return "file"
    except OSError as exc:
        raise SystemExit(f"cannot read person custody mode from {path}: {exc}") from exc
    if mode not in {"file", "broker"}:
        raise SystemExit(
            f"invalid person custody mode {mode!r} in {path}; expected one word: file or broker")
    return mode


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
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise SystemExit(f"cannot read person broker token file {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"person broker token file {path} must be a regular file")
        if info.st_uid != os.getuid():
            raise SystemExit(f"person broker token file {path} must be owned by the current user")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise SystemExit(f"person broker token file {path} must have mode 0600")
        if not 0 < info.st_size <= 16 * 1024:
            raise SystemExit(f"person broker token file {path} is empty or too large")
        raw = b""
        while len(raw) <= 16 * 1024:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(fd)
    try:
        token = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise SystemExit(f"person broker token file {path} is not UTF-8 text") from exc
    if not token:
        raise SystemExit(f"person broker token file {path} is empty")
    return token


def _broker_config() -> _BrokerConfig:
    path = _broker_config_path()
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"person custody is broker but configuration is missing: {path}") from exc
    except OSError as exc:
        raise SystemExit(f"cannot read person broker configuration {path}: {exc}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid person broker configuration {path}: {exc}") from exc
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


def _broker_json(config: _BrokerConfig, method: str, path: str,
                 body: dict | None = None, *, timeout: float = _BROKER_TIMEOUT_S) -> dict:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = {"Authorization": f"Bearer {config.token}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        config.url + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            error = json.loads(exc.read().decode("utf-8")).get("error", "")
        except (AttributeError, UnicodeError, ValueError, json.JSONDecodeError):
            error = ""
        detail = f": {error}" if error else ""
        raise SystemExit(f"person broker refused {path} (HTTP {exc.code}){detail}; nothing was signed") from exc
    except (OSError, urllib.error.URLError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"person broker is unavailable or returned invalid data at {path}: {exc}; "
                         "nothing was signed") from exc
    if not isinstance(result, dict):
        raise SystemExit(f"person broker returned a non-object at {path}; nothing was signed")
    return result


def broker_preflight(kind: str, summary: str) -> None:
    """Refuse an unavailable broker before prompts, writes, or state-changing calls."""
    if kind not in PREFLIGHT_ONLY_KINDS:
        _check_request(kind, summary)
    if custody_mode() != "broker":
        return
    config = _broker_config()
    principal = _broker_json(config, "GET", "/principal/v1")
    if principal.get("principal") != _PRINCIPAL:
        raise SystemExit(
            f"person broker authenticated the wrong principal {principal.get('principal')!r}; "
            "expected 'nakshatra-person'; nothing was signed")
    _person_public_key()


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


def _file_sign_batch(kind: str, items: list, person_hex: str) -> list[tuple[bytes, bytes]]:
    key = _load_matching_person_key(error_context=_error_context(kind))
    results: list[tuple[bytes, bytes]] = []
    for item in items:
        if isinstance(item, bytes):
            final = item
        else:
            _marker, ref, template = item
            final = _fill_suspension_template(
                results[ref][0], results[ref][1], template, person_hex)
        results.append((final, key.sign(final)))
    return results


def _broker_sign_batch(kind: str, items: list, wire_items: list[dict], *, summary: str,
                       context: dict | None, person_key: ed25519.Ed25519PublicKey,
                       person_hex: str) -> list[tuple[bytes, bytes]]:
    config = _broker_config()
    exact_content = {"v": 1, "kind": kind, "items": wire_items}
    if context is not None:
        exact_content["context"] = context
    enqueued = _broker_json(config, "POST", "/enqueue/v1", {
        "what": _WHAT,
        "tier": "high",
        "review": {"description": summary},
        "exact_content": exact_content,
        "idempotency_key": f"{_PRINCIPAL}:{secrets.token_hex(16)}",
        "ttl": _BROKER_WAIT_S,
    })
    request_id = enqueued.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise SystemExit("person broker returned no request id; nothing was signed")

    deadline = time.monotonic() + _BROKER_WAIT_S
    result_path = "/result/v1?" + urllib.parse.urlencode({"request_id": request_id})
    while True:
        result = _broker_json(config, "GET", result_path)
        status = result.get("status")
        if status == "approved":
            break
        if status in {"declined", "expired"}:
            raise SystemExit(_NOT_APPROVED)
        if status in {"unknown", "consumed"}:
            raise SystemExit(f"person broker returned terminal status {status!r}; nothing was signed")
        if time.monotonic() >= deadline:
            raise SystemExit(_NOT_APPROVED)
        time.sleep(_BROKER_POLL_S)

    response = _broker_json(config, "POST", "/person-sign/v1", {
        "request_id": request_id, "exact_content": exact_content,
    })
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
                      context: dict | None = None) -> list[tuple[bytes, bytes]]:
    """Sign one atomic 1-8 item act, resolving suspension-record templates in order."""
    _check_request(kind, summary)
    mode = custody_mode()
    if mode == "broker":
        broker_preflight(kind, summary)
    public_key = _person_public_key()
    person_hex = public_key.public_bytes_raw().hex()
    normalized, wire = _validated_items(items, person_hex)
    checked_context = _validate_context(kind, normalized, context)
    if custody_mode() != mode:
        raise SystemExit("person custody mode changed during signing; nothing was signed")
    if mode == "broker":
        return _broker_sign_batch(kind, normalized, wire, summary=summary,
                                  context=checked_context, person_key=public_key,
                                  person_hex=person_hex)
    return _file_sign_batch(kind, normalized, person_hex)


def person_sign(kind: str, statement: bytes, *, summary: str,
                context: dict | None = None) -> bytes:
    """Sign one classified statement through the configured person-custody backend."""
    return person_sign_batch(kind, [statement], summary=summary, context=context)[0][1]


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

    def __init__(self, kind: str, summary: str, *, key_path: Path | None = None):
        _check_request(kind, summary)
        self._kind = kind
        self._summary = summary
        mode = custody_mode()
        self._mode = mode
        if key_path is not None and mode == "broker":
            raise SystemExit("person custody is broker; --person-key is not allowed")
        if mode == "broker":
            broker_preflight(kind, summary)
            self._key = None
            self._public = _person_public_key()
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
        if custody_mode() != self._mode:
            raise SystemExit("person custody mode changed during signing; nothing was signed")
        if self._key is not None:
            return self._key.sign(data)
        return person_sign(self._kind, data, summary=self._summary)

    def public_key(self) -> ed25519.Ed25519PublicKey:
        return self._public


def custody_label() -> str:
    return "test" if custody_mode() == "file" else "broker"
