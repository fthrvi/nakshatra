"""The single boundary for signatures made with the person's authority key."""
from __future__ import annotations

import contextvars
import os
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

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

_BROKER_UNAVAILABLE = (
    "person custody is set to broker, but the phone-approval service is not available yet; "
    "nothing was signed."
)
_HOME_OVERRIDE: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "person_custody_home", default=None)


@dataclass
class _Approval:
    kind: str
    summary: str
    key: ed25519.Ed25519PrivateKey | None = None


_APPROVAL: contextvars.ContextVar[_Approval | None] = contextvars.ContextVar(
    "person_custody_approval", default=None)


def _home() -> Path:
    return _HOME_OVERRIDE.get() or Path.home()


def _custody_path() -> Path:
    home_override = _HOME_OVERRIDE.get()
    if home_override is not None:
        net_dir = home_override / ".nakshatra" / "net"
    else:
        net_dir = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net"))
    return net_dir / "person-custody"


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


def _broker_sign(kind: str, statement: bytes, summary: str) -> bytes:
    """BROKER STUB: the future phone-approval transport belongs entirely in this function."""
    raise SystemExit(_BROKER_UNAVAILABLE)


def broker_preflight(kind: str, summary: str) -> None:
    """Refuse an unavailable broker before prompts, writes, or state-changing calls."""
    if kind not in PREFLIGHT_ONLY_KINDS:
        _check_request(kind, summary)
    if custody_mode() == "broker":
        raise SystemExit(_BROKER_UNAVAILABLE)


def _error_context(kind: str) -> str:
    return "renewal" if kind in {"grant-renew", "spend-cap"} else "this decision"


def _approval_key(kind: str, summary: str) -> ed25519.Ed25519PrivateKey | None:
    active = _APPROVAL.get()
    if active is None:
        return None
    if active.kind != kind:
        raise SystemExit("person-signing request does not match the active approval")
    if active.key is None:
        active.key = _load_matching_person_key(error_context=_error_context(kind))
    return active.key


def person_sign(kind: str, statement: bytes, *, summary: str) -> bytes:
    """Sign one classified statement through the configured person-custody backend."""
    _check_request(kind, summary)
    if not isinstance(statement, bytes) or not statement:
        raise SystemExit("person-signing statement must not be empty")
    approved_key = _approval_key(kind, summary)
    if approved_key is not None:
        return approved_key.sign(statement)
    mode = custody_mode()
    if mode == "broker":
        return _broker_sign(kind, statement, summary)
    return _load_matching_person_key(error_context=_error_context(kind)).sign(statement)


@contextmanager
def approval(kind: str, summary: str):
    """Group statements into one human approval.

    The future broker transport must treat one ``approval`` block as ONE phone approval covering
    every statement signed inside it. File custody lazily loads and checks one shared key.
    """
    _check_request(kind, summary)
    broker_preflight(kind, summary)
    if _APPROVAL.get() is not None:
        raise SystemExit("nested person approvals are not supported")
    token = _APPROVAL.set(_Approval(kind, summary))
    try:
        yield
    finally:
        _APPROVAL.reset(token)


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
        if key_path is not None and custody_mode() == "broker":
            raise SystemExit("person custody is broker; --person-key is not allowed")
        approved_key = _approval_key(kind, summary)
        if approved_key is not None:
            if key_path is not None:
                raise SystemExit("an explicit person key cannot be used inside an approval")
            self._key = approved_key
        elif key_path is not None:
            # Invites historically parsed this raw file without requiring signer/person.pub.
            self._key, _person_pub = _raw_file_key(Path(key_path))
        else:
            broker_preflight(kind, summary)
            self._key = _load_matching_person_key(error_context=_error_context(kind))

    def sign(self, data: bytes) -> bytes:
        if not isinstance(data, bytes) or not data:
            raise SystemExit("person-signing statement must not be empty")
        return self._key.sign(data)

    def public_key(self) -> ed25519.Ed25519PublicKey:
        return self._key.public_key()


def custody_label() -> str:
    return "test" if custody_mode() == "file" else "broker"
