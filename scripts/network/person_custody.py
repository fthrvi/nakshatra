"""The single boundary for signatures made with the person's authority key."""
from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
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
}

_BROKER_UNAVAILABLE = (
    "person custody is set to broker, but the phone-approval service is not available yet; "
    "nothing was signed."
)
_HOME_OVERRIDE: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "person_custody_home", default=None)
_KEY_OVERRIDE: contextvars.ContextVar[ed25519.Ed25519PrivateKey | None] = contextvars.ContextVar(
    "person_custody_key", default=None)


def _home() -> Path:
    return _HOME_OVERRIDE.get() or Path.home()


def _custody_path() -> Path:
    return _home() / ".nakshatra" / "net" / "person-custody"


def _person_key_path(home: Path | None = None) -> Path:
    return (home or _home()) / ".sthambha" / "person-TEST.key"


def _signer_dir() -> Path:
    home_override = _HOME_OVERRIDE.get()
    if home_override is not None:
        return home_override / ".sthambha" / "signer"
    return Path(os.environ.get("NAK_SIGNER_DIR", Path.home() / ".sthambha" / "signer"))


def _check_request(kind: str, summary: str) -> None:
    if kind not in KINDS:
        expected = ", ".join(sorted(KINDS))
        raise SystemExit(f"unknown person-signing kind {kind!r}; expected one of: {expected}")
    if not isinstance(summary, str) or not summary.strip() or "\n" in summary or "\r" in summary:
        raise SystemExit("person-signing summary must be one non-empty plain-words line")


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


def _load_matching_person_key() -> ed25519.Ed25519PrivateKey:
    """Load the legacy file key and bind it to the active signer's public authority."""
    from sthambha import signer as signer_mod

    signer_dir = _signer_dir()
    person_key = _person_key_path()
    try:
        signer_person = (signer_dir / "person.pub").read_text().strip()
        person_priv, person_pub = signer_mod.load_key(person_key)
    except (OSError, ValueError, signer_mod.SignerError) as exc:
        raise SystemExit(f"cannot load the person key for this decision: {exc}") from exc
    if person_pub != signer_person:
        raise SystemExit(f"the person key at {person_key} does not match {signer_dir / 'person.pub'}")
    return person_priv


def _broker_sign(kind: str, statement: bytes, summary: str) -> bytes:
    """BROKER STUB: the future phone-approval transport belongs entirely in this function."""
    raise SystemExit(_BROKER_UNAVAILABLE)


def person_sign(kind: str, statement: bytes, *, summary: str) -> bytes:
    """Sign one classified statement through the configured person-custody backend."""
    _check_request(kind, summary)
    key_override = _KEY_OVERRIDE.get()
    if key_override is not None:
        return key_override.sign(statement)
    mode = custody_mode()
    if mode == "broker":
        return _broker_sign(kind, statement, summary)
    return _load_matching_person_key().sign(statement)


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
        self._explicit_file_override = key_path is not None
        if key_path is None:
            mode = custody_mode()
            if mode == "broker":
                _broker_sign(kind, b"", summary)
            _load_matching_person_key()  # fail before callers stage work; person_sign reloads at signing time
            self._key = None
            self._public_from_signer = True
        else:
            # The explicit invite override intentionally keeps the old direct-file parse semantics.
            seed = bytes.fromhex(Path(key_path).read_text().strip())
            self._key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
            self._public_from_signer = False

    def sign(self, data: bytes) -> bytes:
        if not self._explicit_file_override:
            return person_sign(self._kind, data, summary=self._summary)
        assert self._key is not None
        token = _KEY_OVERRIDE.set(self._key)
        try:
            return person_sign(self._kind, data, summary=self._summary)
        finally:
            _KEY_OVERRIDE.reset(token)

    def public_key(self) -> ed25519.Ed25519PublicKey:
        if not self._public_from_signer:
            assert self._key is not None
            return self._key.public_key()
        path = _signer_dir() / "person.pub"
        try:
            return ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(path.read_text().strip()))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot load the signer's person public key from {path}: {exc}") from exc


def custody_label() -> str:
    return "test" if custody_mode() == "file" else "broker"
