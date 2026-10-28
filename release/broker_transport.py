"""Shared, fail-closed transport helpers for local approval-broker clients."""
from __future__ import annotations

import json
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class BrokerConfig:
    url: str
    token: str


class NoBrokerRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _req, _fp, _code, _msg, _headers, _newurl):
        return None


BROKER_OPENER = urllib.request.build_opener(NoBrokerRedirects)


class BrokerTransient(Exception):
    """A transport or broker-declared transient failure safe to retry."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BrokerAlreadySigned(Exception):
    """The broker consumed a request, but its successful reply was lost."""


def read_owned_regular(path: Path, label: str, *, exact_mode: int | None = None,
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


def load_broker_config(path: Path, *, subject: str, missing_message: str,
                       token_missing_message: str | None = None) -> BrokerConfig:
    """Load the common closed config shape and its protected bearer token."""
    try:
        raw = read_owned_regular(path, f"{subject} broker configuration")
        document = json.loads(raw.decode("utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(missing_message.format(path=path)) from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {subject} broker configuration {path}") from exc
    if not isinstance(document, dict) or set(document) != {"url", "token_file"}:
        raise SystemExit(
            f"invalid {subject} broker configuration {path}: expected only url and token_file")
    url, token_file = document.get("url"), document.get("token_file")
    if not isinstance(url, str) or not isinstance(token_file, str) or not token_file:
        raise SystemExit(
            f"invalid {subject} broker configuration {path}: url and token_file must be text")
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise SystemExit(f"invalid {subject} broker URL in {path}: {exc}") from exc
    if (parsed.scheme != "http" or not parsed.hostname or port is None
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise SystemExit(
            f"invalid {subject} broker URL in {path}: expected http://host:port with no path")
    token_path = Path(token_file)
    try:
        token_raw = read_owned_regular(
            token_path, f"{subject} broker token file", exact_mode=0o600,
            max_bytes=16 * 1024)
    except FileNotFoundError as exc:
        if token_missing_message is None:
            raise
        raise SystemExit(token_missing_message.format(path=token_path)) from exc
    try:
        token = token_raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise SystemExit(f"{subject} broker token file {token_path} is not UTF-8 text") from exc
    if not token:
        raise SystemExit(f"{subject} broker token file {token_path} is empty")
    return BrokerConfig(url.rstrip("/"), token)


def request_json(config: BrokerConfig, method: str, path: str, body: dict | None = None, *,
                 opener=BROKER_OPENER, timeout: float = 3,
                 error_info: Callable[[str, urllib.error.HTTPError], tuple[str, bool]],
                 unavailable_message: str, invalid_message: str,
                 non_object_message: str | None = None,
                 retry_transient: bool = False) -> dict:
    """Make one authenticated JSON request without redirects or remote-detail reflection."""
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = {"Authorization": f"Bearer {config.token}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(config.url + path, data=data, headers=headers, method=method)
    try:
        with opener.open(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_id, broker_retryable = error_info(path, exc)
        message = (f"approval service error (HTTP {exc.code}, code {error_id})"
                   if error_id else f"approval service error (HTTP {exc.code})")
        if retry_transient and error_id == "already-signed":
            raise BrokerAlreadySigned from None
        if retry_transient and (int(exc.code) in {502, 503, 504} or broker_retryable):
            raise BrokerTransient(message) from None
        raise SystemExit(message) from None
    except (OSError, urllib.error.URLError, TimeoutError, ConnectionError):
        if retry_transient:
            raise BrokerTransient(unavailable_message) from None
        raise SystemExit(unavailable_message) from None
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise SystemExit(invalid_message) from None
    if not isinstance(result, dict):
        raise SystemExit(non_object_message or invalid_message)
    return result


def request_json_until(call: Callable[..., dict], method: str, path: str,
                       body: dict | None, *, deadline: float, request_id: str,
                       phase: str, assert_current: Callable[[], None],
                       uncertain_failure: Callable[[str, str, str], BaseException],
                       deadline_detail: str, already_signed_detail: str,
                       invalid_state_detail: str = "approval broker returned an invalid signing state",
                       timeout_s: float = 3, retry_initial_s: float = 0.1,
                       retry_max_s: float = 1.0) -> dict:
    """Retry one result/sign request with bounded backoff inside an act deadline."""
    delay = retry_initial_s
    while True:
        assert_current()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise uncertain_failure(phase, request_id, deadline_detail)
        try:
            return call(method, path, body, timeout=min(timeout_s, remaining), retry_transient=True)
        except BrokerAlreadySigned:
            if phase != "sign":
                raise SystemExit(invalid_state_detail) from None
            raise SystemExit(already_signed_detail) from None
        except BrokerTransient as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise uncertain_failure(phase, request_id, exc.message) from None
            time.sleep(min(delay, remaining))
            delay = min(retry_max_s, max(delay * 2, retry_initial_s))
        except SystemExit as exc:
            raise uncertain_failure(phase, request_id, str(exc)) from None
