"""nak — talk to your node's messaging service.

    nak status
    nak invite --person-key ~/.sthambha/person-TEST.key [--ttl 86400] [--note "for Rajesh"]
    nak redeem <nki1.… | nka1.…> [--nickname "B"] [--as rajesh]  ask to connect (they must accept)
    nak requests                                           people asking to connect to you
    nak accept <id> [--as rajesh]  |  nak decline <id>
    nak contacts  |  nak remove <name>
    nak send <name> "text"
    nak inbox [--limit 20]

Invites are signed with your PERSON key, which only you hold; the node never sees it. Share an
invite only with the person you are inviting: whoever holds it can ask to connect (asking is all
it does — nothing is shared until you accept).
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import ipaddress
import json
import math
import os
import secrets
import shlex
import stat
import re
import sqlite3
import sys
import time
import unicodedata
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from release_trust import (ReleaseTrustError, load_release_package_catalog,
                           load_release_package_signers)  # noqa: E402
from network.person_custody import (  # noqa: E402
    PersonKey, _person_key_path, _read_owned_regular, broker_preflight, custody_mode,
    person_sign, person_sign_batch, plain)
from network.units import AmountError, MAX_SPEND_CAP, format_amount, parse_amount  # noqa: E402
from network.dlp import outbound_problem  # noqa: E402


def call(sock_path: Path, req: dict) -> dict:
    from network.client import NakClient, NakError
    req = dict(req)
    try:
        r = NakClient(sock_path).call(req.pop("op"), **req)
    except NakError as e:
        raise SystemExit(f"refused: {e}")
    return dict(r, ok=True)


def _release_for_invite(url_override: str = ""):
    """(release dict for the invite, the join one-liner with {CODE}) from THIS node's own install:
    the same source, channel and pinned release key, and the sha256 of the installer files of the
    exact version it runs (recorded in its signed manifest). ("", "") if not release-installed."""
    prefix = Path(os.environ.get("NAK_NODE_PREFIX", Path.home() / ".nakshatra-node"))
    try:
        cfg = json.loads((prefix / "config.json").read_text())
        man = json.loads((prefix / "current" / "manifest.json").read_text())
    except (OSError, ValueError):
        return None, ""
    # A PUBLIC release host (if this node has one configured) beats the node's own source, which may be a
    # private mesh address a friend outside the mesh can't reach: echo URL > ~/.nakshatra/net/release-url
    pub = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "release-url"
    url = (url_override or (pub.read_text().strip() if pub.exists() else "") or cfg.get("source") or "").rstrip("/")
    boot = man.get("bootstrap") or {}
    if not url.startswith(("http://", "https://")):
        return None, ""
    release = {"url": url, "channel": cfg["channel"], "pubkey": cfg["pubkey"], "version": man["version"]}
    if not {"install.py", "releasekit.py"} <= set(boot):
        return release, ""
    base = f"{url}/{cfg['channel']}/{man['version']}"
    line = (f"mkdir -p ~/nak-join && cd ~/nak-join && curl -fsSO {base}/install.py -O {base}/releasekit.py && "
            f"printf '%s  install.py\\n%s  releasekit.py\\n' {boot['install.py']} {boot['releasekit.py']} "
            f"| sha256sum -c --quiet && python3 install.py join '{{CODE}}'")
    return release, line


def _ago(ts: int) -> str:
    d = int(time.time()) - int(ts)
    return f"{d}s ago" if d < 120 else f"{d // 60}m ago" if d < 7200 else f"{d // 3600}h ago"


def _visible_untrusted(value: object, limit: int = 500) -> str:
    """Render remote text on one inert line, with controls made visibly non-operative."""
    rendered = []
    for char in str(value):
        code = ord(char)
        if char == "\n":
            token = r"\n"
        elif char == "\r":
            token = r"\r"
        elif char == "\t":
            token = r"\t"
        elif code == 0x1b:
            token = r"\x1b"
        elif unicodedata.category(char) in {"Cc", "Cf"}:
            token = f"\\u{code:04x}" if code <= 0xffff else f"\\U{code:08x}"
        else:
            token = char
        if sum(map(len, rendered)) + len(token) > limit:
            rendered.append("…")
            break
        rendered.append(token)
    return "".join(rendered)


def _sanitized_untrusted_json(value):
    if isinstance(value, str):
        return _visible_untrusted(value)
    if isinstance(value, list):
        return [_sanitized_untrusted_json(item) for item in value]
    if isinstance(value, dict):
        return {_visible_untrusted(key, 64): _sanitized_untrusted_json(item)
                for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _visible_untrusted(value)


# The public circuit relay (VPS). Its PeerId is the relay's identity: a sidecar refuses any other key there.
DEFAULT_P2P_RELAY_ID = "12D3KooWDfHFtM8PP7TV46nssrix2hTL3kzMkv7kKMhULDakFGB7"
DEFAULT_P2P_RELAYS = (f"/ip4/45.63.109.137/udp/29700/quic-v1/p2p/{DEFAULT_P2P_RELAY_ID},"
                      f"/ip4/45.63.109.137/tcp/29700/p2p/{DEFAULT_P2P_RELAY_ID}")
DEFAULT_ONBOARD_BASE = "https://prithviloka.net/onboard"
DEFAULT_OPERATOR_LEDGER = "http://127.0.0.1:18097"
ONBOARD_STAGE_PURPOSE = "onboard-stage-v2"
ONBOARD_CANCEL_PURPOSE = "onboard-cancel-v2"


class _OnboardHTTPError(SystemExit):
    """An HTTP refusal whose status remains available to recovery logic."""

    def __init__(self, status: int, detail: str = ""):
        self.status = int(status)
        message = f"onboard server refused the request ({self.status})"
        if detail:
            message += f": {detail}"
        super().__init__(message)


class _LedgerHTTPError(Exception):
    """A signed ledger refusal whose status and escaped detail remain inspectable."""

    def __init__(self, status: int, detail: str = ""):
        self.status = int(status)
        self.detail = _visible_untrusted(detail, 500)
        message = f"ledger refused HTTP {self.status}"
        if self.detail:
            message += f": {self.detail}"
        super().__init__(message)


class _NoLedgerRedirects(HTTPRedirectHandler):
    """Never forward a host-unbound Sthambha authorization header through a redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "ledger redirect refused", headers, fp)


def _http_post_json(url: str, body: dict) -> dict:
    req = Request(url, data=json.dumps(body, separators=(",", ":")).encode(),
                  headers={"content-type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode())
    except HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("error", "")
        except (ValueError, AttributeError):
            detail = ""
        detail = _visible_untrusted(detail, 500)
        raise _OnboardHTTPError(e.code, detail)
    except (OSError, URLError, ValueError) as e:
        raise SystemExit(f"could not reach the onboard server: {e}")
    if not isinstance(result, dict):
        raise SystemExit("onboard server returned an invalid response")
    return result


def _http_get_json(url: str) -> dict:
    req = Request(url, headers={"accept": "application/json"}, method="GET")
    try:
        with urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode())
    except HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("error", "")
        except (ValueError, AttributeError):
            detail = ""
        raise _OnboardHTTPError(e.code, _visible_untrusted(detail, 500))
    except (OSError, URLError, ValueError) as e:
        raise SystemExit(f"could not reach the onboard server: {e}")
    if not isinstance(result, dict):
        raise SystemExit("onboard server returned an invalid response")
    return result


def _onboard_token(*, required: bool = True) -> str | None:
    token = os.environ.get("NAK_ONBOARD_TOKEN")
    if token is not None:
        token = token.strip()
        if not token:
            raise SystemExit("NAK_ONBOARD_TOKEN is empty")
        return token
    path = Path.home() / ".nakshatra" / "onboard.token"
    try:
        info = path.lstat()
    except FileNotFoundError:
        if not required:
            return None
        raise SystemExit(f"worker invites need NAK_ONBOARD_TOKEN or {path}: file not found")
    except OSError as e:
        raise SystemExit(f"onboard operator access needs NAK_ONBOARD_TOKEN or {path}: {e}")
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise SystemExit(f"refusing {path}: operator token file must be a regular 0600 file")
    token = path.read_text().strip()
    if not token:
        raise SystemExit(f"refusing {path}: operator token is empty")
    return token


def _stage_auth(kind: str, tier: str, starter: int, ttl: int, node_sign) -> dict:
    """Choose the unchanged owner credential or a member's node-signed stage envelope."""
    token = _onboard_token(required=False)
    if token is not None:
        return {"token": token}
    if tier != "stranger":
        raise SystemExit("without the operator token you may invite at tier stranger only")
    if starter > 10_000_000:
        raise SystemExit("without the operator token --starter may not exceed 10000000")
    body = {"v": 2, "purpose": ONBOARD_STAGE_PURPOSE,
            "kind": kind, "tier": tier, "starter": starter,
            "ttl": ttl, "ts": int(time.time())}
    signed = dict(node_sign(body))
    signed.pop("ok", None)
    if not isinstance(signed, dict) or not isinstance(signed.get("sig"), str):
        raise SystemExit("nakd did not return a signed stage request")
    return signed


def _stage_worker(redeem: str, tier: str, auth: dict, http_post,
                  *, now: int | None = None) -> tuple[dict, int]:
    staged_at = int(time.time() if now is None else now)
    try:
        parsed = urlsplit(redeem)
        host, port = parsed.hostname, parsed.port
        authority = parsed.netloc.rsplit("@", 1)[-1]
    except (TypeError, ValueError):
        parsed = host = port = authority = None
    if parsed is None or parsed.scheme not in {"http", "https"} or not host or authority.endswith(":") or \
            (port is not None and not 1 <= port <= 65535):
        raise SystemExit("worker redeem URL must be an http(s) base URL")
    result = http_post(redeem.rstrip("/") + "/worker-stage", {**auth, "tier": tier})
    code = result.get("code") if isinstance(result, dict) else None
    if not isinstance(code, str) or not code.strip() or len(code) > 64:
        raise SystemExit("onboard server returned an invalid worker code")
    expires_at = result.get("expires_at", staged_at + 3600)
    if isinstance(expires_at, bool) or not isinstance(expires_at, int) or expires_at <= staged_at:
        raise SystemExit("onboard server returned an invalid worker code expiry")
    return {"redeem": redeem.rstrip("/"), "code": code, "tier": tier}, expires_at


def _stage_consumer(redeem: str, tier: str, starter: int, inviter_pubkey: str,
                    auth: dict, http_post, *, now: int | None = None) -> tuple[dict, int]:
    """Stage consumer admission; the server records (but does not execute) any starter transfer."""
    staged_at = int(time.time() if now is None else now)
    try:
        parsed = urlsplit(redeem)
        host, port = parsed.hostname, parsed.port
        authority = parsed.netloc.rsplit("@", 1)[-1]
    except (TypeError, ValueError):
        parsed = host = port = authority = None
    if parsed is None or parsed.scheme not in {"http", "https"} or not host or authority.endswith(":") or \
            (port is not None and not 1 <= port <= 65535):
        raise SystemExit("consumer redeem URL must be an http(s) base URL")
    if type(starter) is not int or not 0 <= starter <= 100_000_000:
        raise SystemExit("--starter must be an integer between 0 and 100000000")
    result = http_post(redeem.rstrip("/") + "/consumer-stage", {
        **auth, "tier": tier, "starter": starter,
        "inviter_pubkey": inviter_pubkey,
    })
    code = result.get("code") if isinstance(result, dict) else None
    if (not isinstance(code, str) or not 1 <= len(code) <= 64
            or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for ch in code)):
        raise SystemExit("onboard server returned an invalid consumer code")
    expires_at = result.get("expires_at", staged_at + 3600)
    if isinstance(expires_at, bool) or not isinstance(expires_at, int) or expires_at <= staged_at:
        raise SystemExit("onboard server returned an invalid consumer code expiry")
    return {"redeem": redeem.rstrip("/"), "code": code, "tier": tier,
            "starter": starter}, expires_at


def _cancel_staged_invite(sock: Path, staged: dict, node: str, http_post) -> None:
    """Undo remote staging when person approval or later local completion fails."""
    token = _onboard_token(required=False)
    if token is not None:
        body = {"token": token, "code": staged["code"]}
    else:
        unsigned = {"v": 2, "purpose": ONBOARD_CANCEL_PURPOSE,
                    "code": staged["code"], "inviter_pubkey": node,
                    "ts": int(time.time())}
        body = call(sock, {"op": "node_sign_stage", "body": unsigned})
        body.pop("ok", None)
    http_post(staged["redeem"].rstrip("/") + "/invite-cancel", body)


def _cancel_local_invite(sock: Path, code: str) -> None:
    """Close a possibly registered invite after a lost reply or later failure."""
    import joincode
    nonce = joincode.decode_invite(code, now=0)["nonce"]
    call(sock, {"op": "cancel_invite", "nonce": nonce})


def _p2p_installer(_sp, action: str) -> None:
    """Ask the signed release's installer to operate on exactly the P2P sidecar service."""
    prefix = Path(os.environ.get("NAK_NODE_PREFIX", Path.home() / ".nakshatra-node"))
    installer = _verified_release_installer(prefix)
    try:
        _sp.run([sys.executable, str(installer), "--prefix", str(prefix), action, "nakshatra-p2p"], check=True)
    except (OSError, _sp.CalledProcessError) as e:
        raise SystemExit(f"could not {action} nakshatra-p2p: {e}")


def _verified_release_installer(prefix: Path) -> Path:
    """Verify the current manifest against the pinned key and the installer against its signed hash."""
    from cryptography.exceptions import InvalidSignature
    try:
        cfg = json.loads((prefix / "config.json").read_text())
        man = json.loads((prefix / "current" / "manifest.json").read_text())
        sig = man["sig"]
        if sig.get("alg") != "Ed25519" or man.get("version") != cfg.get("current"):
            raise ValueError("release identity mismatch")
        body = {k: v for k, v in man.items() if k != "sig"}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        from cryptography.hazmat.primitives.asymmetric import ed25519
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(cfg["pubkey"])).verify(
            base64.b64decode(sig["value"], validate=True), canonical)
        relpath = Path(man["installer"])
        if relpath.is_absolute() or ".." in relpath.parts or relpath.name != "install.py":
            raise ValueError("unsafe installer path")
        installer = prefix / "current" / relpath
        expected = man["bootstrap"]["install.py"]
        if hashlib.sha256(installer.read_bytes()).hexdigest() != expected:
            raise ValueError("installer hash mismatch")
    except (OSError, ValueError, KeyError, TypeError, InvalidSignature) as e:
        raise SystemExit(f"installed release at {prefix} is missing, stale, or fails signature verification: {e}")
    return installer


@contextmanager
def _p2p_ownership_lock():
    """Serialize every decision that can create, take over, restart, or release P2P."""
    path = Path.home() / ".nakshatra" / "p2p.lock"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _enable_p2p_service_locked(_sp, *, owner_token: str = "") -> None:
    """Create and enable P2P while the caller holds ``_p2p_ownership_lock``."""
    env = Path.home() / ".nakshatra" / "p2p.env"
    consumer_env = Path.home() / ".nakshatra" / "p2p-consumer.env"
    created = False
    p2p_config_changed = False
    old_env = env.read_bytes() if env.exists() else None
    old_consumer_env = consumer_env.read_bytes() if consumer_env.exists() else None
    p2p_was_live = old_env is not None and _p2p_unit_enabled(_sp)
    protect_gateway = False
    if not env.exists():
        env.parent.mkdir(parents=True, exist_ok=True)
        ownership = f"{GATEWAY_PROFILE_OWNER_PREFIX}{owner_token}\n" if owner_token else ""
        consumer = 'NAKSHATRA_P2P_NAKD_INBOUND=""\n' if owner_token else ""
        _write_private(
            env, f"P2P_RELAYS={DEFAULT_P2P_RELAYS}\n{consumer}{ownership}".encode()
        )
        created = True
    else:
        # Releases before the consumer-role override put this dialer authorization in p2p.env.
        # It must never survive an upgrade or role transition into an operator/worker generation.
        p2p_config_changed = _remove_env_assignments(
            env, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
    if owner_token and "NAK_CONTAINER_CONSUMER_P2P_PORT" not in os.environ:
        _write_p2p_consumer_env()
    if not owner_token:
        # An explicit `nak p2p on` takes the gateway-created consumer sidecar over for the
        # operator. Restore the signed unit's default nakd inbound in the owned base file; a live
        # consumer-only gateway remains protected by the final override until it is stopped.
        p2p_config_changed = _remove_env_assignment(
            env, "NAKSHATRA_P2P_NAKD_INBOUND") or p2p_config_changed
    if not owner_token:
        protect_gateway = (
            "NAK_CONTAINER_CONSUMER_P2P_PORT" not in os.environ
            and _gateway_unit_enabled(_sp)
            and not _worker_unit_enabled(_sp)
        )
        if protect_gateway:
            before = consumer_env.read_bytes() if consumer_env.exists() else None
            # P2P is now operator-owned. Keep the final empty-inbound override which protects a
            # still-live gateway, but retire the gateway-owned dialer fallback authorization.
            _write_p2p_consumer_env(relay_fallback=False)
            p2p_config_changed = p2p_config_changed or consumer_env.read_bytes() != before
        elif consumer_env.exists():
            consumer_env.unlink()
            p2p_config_changed = True
    try:
        _p2p_installer(_sp, "enable")
        if p2p_config_changed and not created:
            _p2p_installer(_sp, "restart")
    except (Exception, SystemExit) as enable_error:
        if p2p_was_live:
            # This was a takeover/reload of an existing sidecar, not a new service. Never stop the
            # old generation as compensation: restore its base config and restart it, while keeping
            # a live consumer-only gateway protected even if its pre-transition override was stale.
            if old_env is not None:
                _write_private(env, old_env)
                _remove_env_assignments(
                    env, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
            if protect_gateway:
                _write_p2p_consumer_env()
            elif old_consumer_env is None:
                consumer_env.unlink(missing_ok=True)
            else:
                _write_private(consumer_env, old_consumer_env)
            try:
                _p2p_installer(_sp, "restart")
            except (Exception, SystemExit) as rollback_error:
                raise SystemExit(
                    f"{enable_error}; rollback could not restart nakshatra-p2p: {rollback_error}"
                ) from enable_error
            raise

        # A failed enable may have started the unit before its stability check failed: stop it again.
        disabled = False
        disable_error = None
        try:
            _p2p_installer(_sp, "disable")
        except (Exception, SystemExit) as error:
            disable_error = error
        else:
            disabled = True
        # Restore prior files only after a confirmed stop. If the stop failed, the sidecar may be
        # live, so its current role-safe argv and recovery opt-in must remain authoritative.
        if disabled:
            if old_env is None:
                env.unlink(missing_ok=True)
            else:
                _write_private(env, old_env)
                _remove_env_assignments(
                    env, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
            if old_consumer_env is None:
                consumer_env.unlink(missing_ok=True)
            else:
                _write_private(consumer_env, old_consumer_env)
        if disable_error is not None:
            raise SystemExit(
                f"{enable_error}; rollback could not disable nakshatra-p2p: {disable_error}"
            ) from enable_error
        raise
    if not owner_token:
        # A successful explicit `nak p2p on` is an operator/worker takeover. Retire any
        # gateway marker so a later `nak gateway off` cannot disable that sidecar.
        _gateway_p2p_paths()[1].unlink(missing_ok=True)


def _enable_p2p_service(_sp, *, owner_token: str = "") -> None:
    """Create the opt-in and enable the service as one ownership transaction."""
    with _p2p_ownership_lock():
        _enable_p2p_service_locked(_sp, owner_token=owner_token)


WORKER_SERVICE = "nakshatra-worker"
WORKER_BACKENDS = {"cpu", "cuda", "rocm", "vulkan", "metal"}
GATEWAY_SERVICE = "nakshatra-gateway"
GATEWAY_PROFILE = "inference.env"
GATEWAY_PROFILE_OWNER = "inference-profile-created"
GATEWAY_P2P_OWNER = "p2p-profile-created"
GATEWAY_PROFILE_OWNER_PREFIX = "# nak-gateway-owned: "
WORKER_PAID_ENV = (
    "NAKSHATRA_PAID_ADMISSION",
    "NAKSHATRA_PAID_LEDGER_URL",
    "NAKSHATRA_PAID_LEDGER_SPKI_SHA256",
    "NAKSHATRA_PAID_STORE",
)
GATEWAY_PAID_ENV = "NAKSHATRA_GATEWAY_PAID"


def _gateway_paths() -> tuple[Path, Path]:
    root = Path.home() / ".nakshatra" / "gateway"
    return root, Path.home() / ".nakshatra" / "gateway.env"


def _gateway_profile_paths() -> tuple[Path, Path]:
    root, _env_path = _gateway_paths()
    return Path.home() / ".nakshatra" / GATEWAY_PROFILE, root / GATEWAY_PROFILE_OWNER


def _gateway_p2p_paths() -> tuple[Path, Path]:
    root, _env_path = _gateway_paths()
    return Path.home() / ".nakshatra" / "p2p.env", root / GATEWAY_P2P_OWNER


def _p2p_consumer_env_path() -> Path:
    return Path.home() / ".nakshatra" / "p2p-consumer.env"


def _write_p2p_consumer_env(worker_args: str = "", *, relay_fallback: bool = True) -> None:
    """Write the last-loaded, canonical consumer P2P override."""
    consumer_args = "-relay-activation-fallback" if relay_fallback else ""
    _write_private(
        _p2p_consumer_env_path(),
        ('NAKSHATRA_P2P_NAKD_INBOUND=""\n'
         f'NAKSHATRA_P2P_CONSUMER_ARGS="{consumer_args}"\n'
         f'NAKSHATRA_P2P_WORKER_ARGS="{worker_args}"\n').encode(),
    )


def _inference_profile_needed_without_gateway() -> bool:
    """Whether another current opt-in still requires the inference release profile.

    The unconscious service has no smaller profile and therefore needs ``inference.env``. A
    worker does not: ``worker.env`` opts it into the smaller worker profile, while the release
    spec's ``worker.satisfied_by=["inference"]`` only permits inference to act as a superset when
    both are active. Removing a gateway-owned inference marker leaves a worker on its own lock.
    """
    return (Path.home() / ".nakshatra" / "unconscious.release").is_file()


def _gateway_profile_token(profile: Path) -> str:
    try:
        lines = profile.read_text().splitlines()
    except (OSError, UnicodeError):
        return ""
    tokens = [line.removeprefix(GATEWAY_PROFILE_OWNER_PREFIX) for line in lines
              if line.startswith(GATEWAY_PROFILE_OWNER_PREFIX)]
    if len(tokens) != 1 or not re.fullmatch(r"[0-9a-f]{64}", tokens[0]):
        return ""
    return tokens[0]


def _gateway_marker_token(marker: Path) -> str:
    try:
        token = marker.read_text().strip()
    except (OSError, UnicodeError):
        return ""
    return token if re.fullmatch(r"[0-9a-f]{64}", token) else ""


def _release_gateway_profile_ownership() -> None:
    """Retire the marker and remove only the exact profile generation the gateway created."""
    profile, marker = _gateway_profile_paths()
    profile_token = _gateway_profile_token(profile)
    marker_token = _gateway_marker_token(marker)
    owned = bool(profile_token and marker_token
                 and secrets.compare_digest(profile_token, marker_token))
    if owned and not _inference_profile_needed_without_gateway():
        profile.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)


def _release_gateway_p2p_ownership(_sp) -> None:
    """Disable only the exact P2P opt-in generation created by this gateway."""
    with _p2p_ownership_lock():
        p2p_env, marker = _gateway_p2p_paths()
        consumer_env = _p2p_consumer_env_path()
        removed_consumer_override = consumer_env.exists()
        removed_legacy_relay = False
        if p2p_env.exists():
            removed_legacy_relay = _remove_env_assignments(
                p2p_env, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
        env_token = _gateway_profile_token(p2p_env)
        marker_token = _gateway_marker_token(marker)
        owned = bool(env_token and marker_token and secrets.compare_digest(env_token, marker_token))
        worker_enabled = _worker_unit_enabled(_sp)
        if owned and not worker_enabled:
            _gateway_p2p_before_disable()
            # The hook is a regression-test seam for the old check/act race. Re-read all authority
            # state under the same lock immediately before the irreversible service stop.
            env_token = _gateway_profile_token(p2p_env)
            marker_token = _gateway_marker_token(marker)
            owned = bool(env_token and marker_token
                         and secrets.compare_digest(env_token, marker_token))
            worker_enabled = _worker_unit_enabled(_sp)
            if owned and not worker_enabled:
                # Prepare the only safe retained generation before asking the installer to stop
                # it. If the stop fails, removing the gateway's empty nakd override makes this
                # profile byte-equivalent to the pre-gateway operator P2P argv (762b0da).
                consumer_env.unlink(missing_ok=True)
                _remove_env_assignment(p2p_env, "NAKSHATRA_P2P_NAKD_INBOUND")
                try:
                    _p2p_installer(_sp, "disable")
                except (Exception, SystemExit) as disable_error:
                    # The installer retains p2p.env when it cannot prove the sidecar stopped.
                    # Retire gateway ownership and restart so the retained unit is verified under
                    # legacy operator config before control returns to the caller.
                    marker.unlink(missing_ok=True)
                    try:
                        _p2p_installer(_sp, "restart")
                    except (Exception, SystemExit) as restart_error:
                        raise SystemExit(
                            f"{disable_error}; retained p2p promotion could not restart: "
                            f"{restart_error}"
                        ) from disable_error
                    raise SystemExit(
                        f"{disable_error}; retained p2p was promoted to operator-owned legacy "
                        "configuration and restarted"
                    ) from disable_error
                p2p_env.unlink(missing_ok=True)
                removed_consumer_override = False
        consumer_env.unlink(missing_ok=True)
        if (removed_consumer_override or removed_legacy_relay) and p2p_env.exists():
            _p2p_installer(_sp, "restart")
        marker.unlink(missing_ok=True)


def _gateway_p2p_before_disable() -> None:
    """Test seam invoked between gateway P2P eligibility checks; production is a no-op."""


def _gateway_values() -> dict[str, str]:
    _root, env_path = _gateway_paths()
    values = {}
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def _env_file_values(path: Path) -> dict[str, str]:
    values = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        assignment = _env_assignment(line)
        if assignment is not None:
            key, value = assignment
            try:
                parsed = shlex.split(value, posix=True)
            except ValueError:
                parsed = []
            values[key] = parsed[0] if len(parsed) == 1 else value.strip().strip("\"'")
    return values


def _gateway_package_defaults() -> tuple[str, str]:
    """Load an untrusted package location and an optional prior local narrowing."""
    standing = Path.home() / ".nakshatra" / "consumer-standing.json"
    try:
        value = json.loads(standing.read_text())
    except (OSError, ValueError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    sources = []
    for path in (Path.home() / ".nakshatra" / "worker.env", _gateway_paths()[1]):
        values = {}
        try:
            lines = path.read_text().splitlines()
        except OSError:
            lines = ()
        for line in lines:
            if line and not line.lstrip().startswith("#") and "=" in line:
                key, item = line.split("=", 1)
                values[key.strip()] = item.strip().strip("\"'")
        sources.append(values)
    urls = [value.get("package_url", "")] + [item.get("PACKAGE_URL", "") for item in sources]
    url = next((item.strip() for item in urls if isinstance(item, str) and item.strip()), "")
    # Only the gateway's own prior, release-validated choice is reused as a narrowing.
    prior_narrow = sources[-1].get("PACKAGE_TRUSTED_PUBKEY", "")
    narrow = prior_narrow.strip() if isinstance(prior_narrow, str) else ""
    return url, narrow


def _wallet_hook():
    values = _gateway_values()
    url = (
        os.environ.get("NAKSHATRA_LEDGER_URL", "").strip()
        or values.get("NAKSHATRA_LEDGER_URL", "").strip()
    )
    if not url:
        raise SystemExit("wallet needs a ledger URL; run: nak gateway on …")
    pin = (
        os.environ.get("NAKSHATRA_LEDGER_SPKI_SHA256", "").strip()
        or values.get("NAKSHATRA_LEDGER_SPKI_SHA256", "").strip()
        or None
    )
    from accounting import wallet
    return wallet(url=url, spki_hash=pin)


def _exact_usdc_units(text: str, *, scale: int, error: str) -> int:
    """Convert plain USDC text to integer units without Decimal context rounding."""
    if not isinstance(text, str) or re.fullmatch(r"[0-9]+(?:\.[0-9]*)?", text) is None:
        raise SystemExit(error)
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise SystemExit(error) from exc
    sign, digits, exponent = value.as_tuple()
    if (not value.is_finite() or sign or value.is_zero()
            or not isinstance(exponent, int) or exponent < -6):
        raise SystemExit(error)
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    units = coefficient * (10 ** (exponent + scale))
    if units >= 2**64:
        raise SystemExit(error)
    return units


def _usdc_credits(text: str) -> int:
    return _exact_usdc_units(
        text, scale=8,
        error="withdraw amount must be positive with at most 6 USDC decimals")


def _wallet_command(args) -> None:
    from ledger_client import LedgerUnavailable
    try:
        hook = _wallet_hook()
        if args.wallet_cmd == "address":
            row = hook.wallet()
            address = row.get("deposit_address")
            print(address if isinstance(address, str) and address
                  else "pending: deposit address creation queued")
        elif args.wallet_cmd == "status":
            print(json.dumps(_display_amount_fields(
                hook.wallet_status(), "balance", "withdrawable"), indent=1, sort_keys=True))
        elif args.wallet_cmd == "bind":
            print(json.dumps(hook.bind(args.address), indent=1, sort_keys=True))
        else:
            credits = _usdc_credits(args.usdc)
            print(json.dumps(_display_amount_fields(
                hook.withdraw(credits), "amount"), indent=1, sort_keys=True))
    except LedgerUnavailable as exc:
        raise SystemExit(f"wallet refused: {exc}") from exc


_SOLANA_DEVNET_GENESIS = "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG"
_SOLANA_MAINNET_GENESIS = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"


class _NoPayRedirects(HTTPRedirectHandler):
    """Do not let an explicitly chosen RPC silently redirect to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "pay RPC redirect refused", headers, fp)


def _pay_rpc_url(value: str) -> tuple[str, bool]:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"NAK_PAY_RPC is not a valid URL: {exc}") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.fragment or (port is not None and not 1 <= port <= 65535)):
        raise SystemExit("NAK_PAY_RPC must be an http(s) URL without credentials or a fragment")
    local = parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
    return value, local


def _pay_rpc_request(opener, url: str, method: str, params: list) -> object:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                      separators=(",", ":")).encode()
    request = Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    try:
        with opener.open(request, timeout=10) as response:
            reply = json.loads(response.read().decode())
    except (HTTPError, URLError, OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"pay RPC request {method} failed: {exc}") from exc
    if not isinstance(reply, dict):
        raise SystemExit(f"pay RPC request {method} returned a non-object response")
    if reply.get("error") is not None:
        raise SystemExit(f"pay RPC request {method} was refused: {_visible_untrusted(reply['error'], 300)}")
    if "result" not in reply:
        raise SystemExit(f"pay RPC request {method} returned no result")
    return reply["result"]


def _format_sol(lamports: int) -> str:
    whole, fraction = divmod(lamports, 1_000_000_000)
    if not fraction:
        return str(whole)
    return f"{whole}.{fraction:09d}".rstrip("0")


_PAY_DEFAULTS = {
    "cap_total": 5_000_000,
    "days": 30,
    "per_tx": 1_000_000,
    "per_day": 5_000_000,
}
_NODE_SOL_MIN = 10_000_000
_NODE_SOL_TARGET = 30_000_000


def _pay_usdc(value: str, name: str) -> int:
    return _exact_usdc_units(
        value, scale=6,
        error=f"{name} must be positive with at most 6 USDC decimals")


def _pay_amount(value: int) -> str:
    whole, fraction = divmod(int(value), 1_000_000)
    return str(whole) if not fraction else f"{whole}.{fraction:06d}".rstrip("0")


def _pay_limits_path() -> Path:
    return Path.home() / ".nakshatra" / "pay" / "limits.json"


def _pay_backup_ack_path() -> Path:
    return _pay_limits_path().with_name("backup-confirmed")


def _pay_backup_confirmed() -> bool:
    path = _pay_backup_ack_path()
    if not os.path.lexists(path):
        return False
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"refusing {path}: backup acknowledgement must be a regular file")
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise SystemExit(
                f"refusing {path}: backup acknowledgement must be uid-owned mode 0600")
        value = path.read_text(encoding="ascii")
    except SystemExit:
        raise
    except (OSError, UnicodeError) as exc:
        raise SystemExit(f"cannot read backup acknowledgement at {path}: {exc}") from exc
    if value != "confirmed\n":
        raise SystemExit(f"refusing invalid backup acknowledgement at {path}")
    return True


def _save_pay_backup_confirmation() -> None:
    if _pay_backup_confirmed():
        return
    path = _pay_backup_ack_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(b"confirmed\n")
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        temp.unlink(missing_ok=True)


def _warn_unbacked_pay_balance(balance: int) -> None:
    if balance > 20_000_000 and not _pay_backup_confirmed():
        print(
            "warning: escrow holds more than 20 USDC; back up the wallet seed, then "
            "run `nak pay backup --confirm` to acknowledge the backup",
            file=sys.stderr,
        )


def _validate_pay_limits(values: dict) -> dict:
    required = {"cap_total", "days", "per_tx", "per_day"}
    if set(values) != required:
        raise SystemExit("pay limits must contain cap_total, days, per_tx, and per_day")
    for key in ("cap_total", "per_tx", "per_day"):
        if isinstance(values[key], bool) or not isinstance(values[key], int) or not 0 < values[key] < 2**64:
            raise SystemExit(f"pay limit {key} must be a positive USDC amount")
    if isinstance(values["days"], bool) or not isinstance(values["days"], int) or not 1 <= values["days"] <= 3650:
        raise SystemExit("pay limit days must be between 1 and 3650")
    if values["per_tx"] > values["per_day"]:
        raise SystemExit("pay limit per-tx cannot exceed per-day")
    if values["per_day"] > values["cap_total"]:
        raise SystemExit("pay limit per-day cannot exceed the on-chain cap")
    return dict(values)


def _load_pay_limits() -> dict:
    path = _pay_limits_path()
    if not path.exists():
        return dict(_PAY_DEFAULTS)
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"refusing {path}: pay limits must be a regular file")
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise SystemExit(f"refusing {path}: pay limits must be uid-owned mode 0600")
        values = json.loads(path.read_text(encoding="utf-8"))
    except SystemExit:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"cannot read pay limits at {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise SystemExit("pay limits must be a JSON object")
    return _validate_pay_limits(values)


def _save_pay_limits(values: dict) -> None:
    values = _validate_pay_limits(values)
    path = _pay_limits_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if os.path.lexists(path) and path.is_symlink():
        raise SystemExit(f"refusing {path}: pay limits must not be a symlink")
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    body = (json.dumps(values, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(body)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        temp.unlink(missing_ok=True)


def _pay_chain_settings(*, require_rpc: bool = True):
    from network import pay_chain

    rpc_url = pay_chain.configured_rpc_url(os.environ)
    if require_rpc and not rpc_url:
        raise SystemExit("NAK_PAY_RPC is required for this on-chain payment command")
    program = pay_chain.configured_pubkey(os.environ, "NAK_PAY_PROGRAM")
    mint = pay_chain.configured_pubkey(os.environ, "NAK_PAY_MINT")
    try:
        rpc = pay_chain.PayRpc(rpc_url) if rpc_url else None
        if rpc is not None:
            rpc.ensure_cluster()
    except (ValueError, pay_chain.VoucherError) as exc:
        raise SystemExit(str(exc)) from exc
    return pay_chain, rpc, program, mint


def _pay_wallet_and_node(*, signing: bool = True):
    from network.pay_wallet import load_wallet
    from nakshatra_auth import load_worker_key

    wallet = load_wallet(Path.home())
    if signing and wallet["custody"] == "broker":
        raise SystemExit(
            "wallet custody is broker; on-chain wallet signing is not available yet")
    try:
        node_seed, node_hex = load_worker_key(
            Path.home() / ".nakshatra" / "keys" / "worker.ed25519")
    except (OSError, ValueError, PermissionError) as exc:
        raise SystemExit(f"cannot load the node worker key: {exc}") from exc
    return wallet, node_seed, bytes.fromhex(node_hex)


def _pay_send(rpc, chain, instructions, wallet: dict, *, rent: int = 0) -> str:
    try:
        blockhash = rpc.latest_blockhash()
        message = chain.compile_message(instructions, wallet["address"], blockhash)
        fee = rpc.fee_for_message(message.data)
        balance = rpc.balance(wallet["address"])
        needed = int(rent) + fee
        if balance < needed:
            shortfall = needed - balance
            raise SystemExit(
                f"wallet {wallet['address']} has too little SOL: needs "
                f"{_format_sol(needed)} SOL, has {_format_sol(balance)} SOL, "
                f"shortfall {_format_sol(shortfall)} SOL; nothing was sent")
        tx = chain.sign_transaction(
            instructions, wallet["address"], blockhash,
            {wallet["address"]: wallet["seed"]})
        signature = rpc.send_and_confirm(tx)
        cluster = rpc.cluster_label()
    except SystemExit:
        raise
    except (ValueError, chain.VoucherError) as exc:
        raise SystemExit(f"pay transaction refused: {exc}") from exc
    print(f"cluster: {cluster}")
    print(f"transaction: {signature}")
    print(f"explorer: {chain.explorer_link(signature, cluster)}")
    return signature


def _pay_ensure() -> None:
    wallet, _node_seed, node_public = _pay_wallet_and_node()
    chain, rpc, program, mint = _pay_chain_settings()
    limits = _load_pay_limits()
    path = _pay_limits_path()
    if not path.exists():
        _save_pay_limits(limits)
    d = chain._derived(wallet["address"], program)
    node_address = chain.b58encode(node_public)
    delegate_address = chain.delegate_pda(d["escrow"], node_public, program)[0]
    try:
        escrow_info = rpc.account_info(d["escrow"])
        delegate_info = rpc.account_info(delegate_address)
        owner_state_info = rpc.account_info(d["owner_state"])
        vault_info = rpc.account_info(d["vault"])
        instructions = []
        rent = 0
        node_balance = rpc.balance(node_address)
        if node_balance < _NODE_SOL_MIN:
            node_top_up = _NODE_SOL_TARGET - node_balance
            instructions.append(chain.system_transfer(
                wallet["address"], node_address, node_top_up))
            # The transfer is in this transaction, so include it in the
            # wallet's pre-send affordability check with account rent.
            rent += node_top_up
        escrow_state = None
        if escrow_info is None:
            instructions.extend(chain.escrow_init(wallet["address"], mint, program))
            if owner_state_info is None:
                rent += rpc.rent_exemption(17)
            rent += rpc.rent_exemption(163) + rpc.rent_exemption(165)
        elif escrow_info["owner"] != program:
            raise SystemExit("escrow address exists but is not owned by NAK_PAY_PROGRAM")
        elif vault_info is None:
            raise SystemExit("escrow exists but its canonical vault is missing")
        else:
            escrow_state = chain.decode_escrow(escrow_info["data"])
            if escrow_state["owner"] != wallet["address"] or escrow_state["vault"] != d["vault"]:
                raise SystemExit("escrow account does not match the configured wallet and vault")
            if vault_info["owner"] != chain.TOKEN_PROGRAM_ID:
                raise SystemExit("escrow vault is not owned by the classic SPL Token program")

        current = None
        if delegate_info is not None:
            if delegate_info["owner"] != program:
                raise SystemExit("delegate address exists but is not owned by NAK_PAY_PROGRAM")
            current = chain.decode_delegate(delegate_info["data"])
        expires_soon = current is None or current["expires_at"] <= int(time.time()) + 7 * 86400
        changed = (escrow_info is None or current is None
                   or (escrow_state is not None and current["generation"] != escrow_state["generation"])
                   or current["cap_total"] != limits["cap_total"]
                   or current["flags"] != chain.DELEGATE_FLAGS_ALL or current["disabled"])
        if expires_soon or changed:
            if current is not None and current["spent"] > limits["cap_total"]:
                raise SystemExit("saved on-chain cap is below the delegate's amount already spent")
            if current is None:
                rent += rpc.rent_exemption(115)
            expires_at = int(time.time()) + limits["days"] * 86400
            instructions.extend(chain.delegate_set(
                wallet["address"], node_public, limits["cap_total"], expires_at,
                chain.DELEGATE_FLAGS_ALL, program))
        if not instructions:
            print("pay escrow and node delegate are already configured")
            return
        _pay_send(rpc, chain, instructions, wallet, rent=rent)
    except SystemExit:
        raise
    except (ValueError, chain.VoucherError) as exc:
        raise SystemExit(f"pay ensure refused: {exc}") from exc


def _pay_status(wallet: dict) -> None:
    from network import pay_chain as chain

    print(f"address: {wallet['address']}")
    print(f"custody: {wallet['custody']}")
    rpc_url = chain.configured_rpc_url(os.environ)
    if not rpc_url:
        print("balance: set NAK_PAY_RPC to a devnet RPC to show balances")
        return
    try:
        rpc = chain.PayRpc(rpc_url)
        rpc.ensure_cluster()
        print(f"balance: {_format_sol(rpc.balance(wallet['address']))} SOL")
        node_hex = None
        try:
            _node_seed, node_hex = __import__("nakshatra_auth").load_worker_key(
                Path.home() / ".nakshatra" / "keys" / "worker.ed25519")
            node_address = chain.b58encode(bytes.fromhex(node_hex))
            print(f"node balance: {_format_sol(rpc.balance(node_address))} SOL")
        except (OSError, ValueError, PermissionError) as exc:
            print(f"node balance: unavailable ({exc})")
        program_value = os.environ.get("NAK_PAY_PROGRAM", "").strip()
        mint_value = os.environ.get("NAK_PAY_MINT", "").strip()
        if not program_value or not mint_value:
            print("escrow: set NAK_PAY_PROGRAM and NAK_PAY_MINT to show on-chain payment state")
            return
        program = chain.configured_pubkey(os.environ, "NAK_PAY_PROGRAM")
        chain.configured_pubkey(os.environ, "NAK_PAY_MINT")
        if node_hex is None:
            raise SystemExit("cannot show delegate without a valid node worker key")
        d = chain._derived(wallet["address"], program)
        escrow_info = rpc.account_info(d["escrow"])
        if escrow_info is None:
            print("escrow: not initialized")
            return
        if escrow_info["owner"] != program:
            raise SystemExit("escrow address exists but is not owned by NAK_PAY_PROGRAM")
        escrow = chain.decode_escrow(escrow_info["data"])
        vault_info = rpc.account_info(escrow["vault"])
        if vault_info is None:
            raise SystemExit("escrow exists but its canonical vault is missing")
        if vault_info["owner"] != chain.TOKEN_PROGRAM_ID:
            raise SystemExit("escrow vault is not owned by the classic SPL Token program")
        balance = chain.token_amount(vault_info["data"])
        print(f"escrow balance: {_pay_amount(balance)} USDC")
        _warn_unbacked_pay_balance(balance)
        print(f"escrow committed: {_pay_amount(escrow['committed'])} USDC")
        delegate_address = chain.delegate_pda(d["escrow"], bytes.fromhex(node_hex), program)[0]
        delegate_info = rpc.account_info(delegate_address)
        if delegate_info is None:
            print("delegate: not configured")
        else:
            delegate = chain.decode_delegate(delegate_info["data"])
            print(f"delegate cap: {_pay_amount(delegate['cap_total'])} USDC")
            print(f"delegate expires: {delegate['expires_at']}")
    except (OSError, ValueError, PermissionError, chain.VoucherError) as exc:
        raise SystemExit(f"pay status refused: {exc}") from exc


def _pay_command(args) -> None:
    from network.pay_wallet import bip39_mnemonic, load_wallet

    if args.pay_cmd == "backup":
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise SystemExit("nak pay backup is interactive-only; run it in a terminal")
        wallet = load_wallet(Path.home())
        if args.confirm:
            _save_pay_backup_confirmation()
            print("wallet backup acknowledgement recorded")
            return
        if input('type "backup" to print the private wallet seed phrase: ').strip() != "backup":
            raise SystemExit("wallet backup cancelled; nothing was printed")
        print(bip39_mnemonic(wallet["seed"]))
        return

    if args.pay_cmd == "export":
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise SystemExit("nak pay export is interactive-only; run it in a terminal")
        if input('type "export" to print the private wallet keypair: ').strip() != "export":
            raise SystemExit("wallet export cancelled; nothing was printed")
        wallet = load_wallet(Path.home())
        print(json.dumps(list(wallet["seed"] + wallet["public_key"]), separators=(",", ":")))
        return

    wallet = load_wallet(Path.home())
    if args.pay_cmd == "address":
        print(wallet["address"])
        return
    if args.pay_cmd == "status":
        _pay_status(wallet)
        return
    if args.pay_cmd == "ensure":
        _pay_ensure()
        return
    if args.pay_cmd == "limits":
        values = {"cap_total": _pay_usdc(args.cap, "--cap"), "days": args.days,
                  "per_tx": _pay_usdc(args.per_tx, "--per-tx"),
                  "per_day": _pay_usdc(args.per_day, "--per-day")}
        _save_pay_limits(values)
        _pay_ensure()
        return

    if args.pay_cmd == "channels":
        chain, rpc, _program, _mint = _pay_chain_settings()
        from network.pay_voucher import PayerChannelBook
        with PayerChannelBook() as book:
            rows = book.list_channels()
        if not rows:
            print("no payer channels")
        for row in rows:
            try:
                state = rpc.read_channel(row.channel, row.program)
                print(f"{row.worker_key}  {row.channel}  cap {_pay_amount(state.cap)} USDC  "
                      f"redeemed {_pay_amount(state.redeemed)} USDC  {state.status.lower()}")
            except chain.VoucherError as exc:
                print(f"{row.worker_key}  {row.channel}  unavailable: {exc}")
        return

    if args.pay_cmd in {"deposit", "withdraw"}:
        _pay_ensure()
    wallet, _node_seed, _node_public = _pay_wallet_and_node()
    chain, rpc, program, mint = _pay_chain_settings()
    if args.pay_cmd == "deposit":
        amount = _pay_usdc(args.amount, "--amount")
        instructions = chain.escrow_deposit(wallet["address"], mint, amount, program)
        vault = chain.vault_pda(chain.escrow_pda(wallet["address"], program)[0], program)[0]
        vault_info = rpc.account_info(vault)
        if vault_info is None:
            raise SystemExit("escrow exists but its canonical vault is missing")
        if vault_info["owner"] != chain.TOKEN_PROGRAM_ID:
            raise SystemExit("escrow vault is not owned by the classic SPL Token program")
        current_balance = chain.token_amount(vault_info["data"])
        _warn_unbacked_pay_balance(current_balance + amount)
        ata = chain.associated_token_address(wallet["address"], mint)[0]
        rent = 0 if rpc.account_info(ata) is not None else rpc.rent_exemption(165)
        _pay_send(rpc, chain, instructions, wallet, rent=rent)
    elif args.pay_cmd == "withdraw":
        if args.execute:
            instructions = chain.escrow_withdraw_execute(wallet["address"], mint, program)
            ata = chain.associated_token_address(wallet["address"], mint)[0]
            rent = 0 if rpc.account_info(ata) is not None else rpc.rent_exemption(165)
            _pay_send(rpc, chain, instructions, wallet, rent=rent)
        else:
            amount = _pay_usdc(args.amount, "--amount")
            _pay_send(rpc, chain, chain.escrow_withdraw_request(
                wallet["address"], amount, program), wallet)
            escrow = chain.escrow_pda(wallet["address"], program)[0]
            current = rpc.account_info(escrow)
            pending = chain.decode_escrow(current["data"])["pending_withdraw"] if current else None
            print(f"withdraw_at: {pending['withdraw_at'] if pending else int(time.time()) + 21600}")


def _balance_command() -> None:
    from ledger_client import LedgerUnavailable
    try:
        row = _display_amount_fields(
            _wallet_hook().balance(), "balance", "available", "credit_limit", "limit")
    except LedgerUnavailable as exc:
        raise SystemExit(f"balance refused: {exc}") from exc
    print(json.dumps(row, indent=1, sort_keys=True))


def _gateway_installer(_sp, action: str) -> None:
    prefix = Path(os.environ.get("NAK_NODE_PREFIX", Path.home() / ".nakshatra-node"))
    installer = _verified_release_installer(prefix)
    try:
        _sp.run([sys.executable, str(installer), "--prefix", str(prefix),
                 action, GATEWAY_SERVICE], check=True)
    except (OSError, _sp.CalledProcessError) as e:
        raise SystemExit(f"could not {action} {GATEWAY_SERVICE}: {e}")


def _bare_http_url(value: str, name: str, *, allow_path: bool = False) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as e:
        raise SystemExit(f"--{name} is not a valid URL: {e}") from e
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or (port is not None and not 1 <= port <= 65535)
            or (not allow_path and parsed.path not in {"", "/"})):
        kind = "base URL without credentials, query, or fragment" if allow_path else "bare origin"
        raise SystemExit(f"--{name} must be an http(s) {kind}")
    return value.rstrip("/")


def _paid_probe(ledger_url: str, ledger_spki: str) -> dict:
    """Run the authenticated claim-capability probe used by worker.py itself."""
    ledger = _bare_http_url(ledger_url, "ledger-url", allow_path=True)
    if urlsplit(ledger).scheme != "https":
        raise SystemExit("paid admission requires an https --ledger-url")
    pin = (ledger_spki or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", pin):
        raise SystemExit("--ledger-spki must be a 64-character SHA-256 hex digest")
    try:
        from nakshatra_auth import load_worker_key
        from paid_admission import SpendLedgerClient
        private, keyid = load_worker_key(
            Path.home() / ".nakshatra" / "keys" / "worker.ed25519")
        return SpendLedgerClient(ledger, pin, private, keyid).probe()
    except Exception as exc:
        raise SystemExit(f"paid admission preflight failed: {exc}") from exc


def _paid_store_preflight(path: Path) -> Path:
    """Open, recover-check, and leave the durable paid store owner-only."""
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        from paid_admission import PaidAdmissionStore
        store = PaidAdmissionStore(path)
        store.close()
        metadata = path.stat()
    except Exception as exc:
        raise SystemExit(f"paid store preflight failed for {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise SystemExit(f"paid store preflight failed for {path}: store is not a 0600 regular file")
    return path


def _replace_env_assignments(path: Path, updates: dict[str, str | None]) -> None:
    """Atomically replace selected assignments while preserving every unrelated byte."""
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"cannot update {path}: {exc}") from exc
    kept = []
    for line in body.splitlines(keepends=True):
        try:
            assignment = _env_assignment(line.decode())
        except UnicodeDecodeError:
            assignment = None
        if assignment is None or assignment[0] not in updates:
            kept.append(line)
    result = b"".join(kept)
    additions = []
    for name, value in updates.items():
        if value is not None:
            if not value or any(c in value for c in "\r\n\0"):
                raise SystemExit(f"{name} must be one non-empty environment-file value")
            quoted = '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
            additions.append(f"{name}={quoted}\n".encode())
    if additions and result and not result.endswith((b"\n", b"\r")):
        result += b"\n"
    _write_private(path, result + b"".join(additions))


def _worker_paid_info(env_path: Path, timeout_s: float = 3.0):
    """Return local worker Info over its real gRPC transport, or raise."""
    import grpc
    import nakshatra_pb2 as pb
    import nakshatra_pb2_grpc as pbg

    values = _env_file_values(env_path)
    port = _worker_port(env_path)
    listen = values.get("LISTEN", "[::]").strip()
    if listen in {"0.0.0.0", "*"}:
        host = "127.0.0.1"
    elif listen in {"[::]", "::"}:
        host = "[::1]"
    elif listen.startswith("[") and listen.endswith("]"):
        host = listen
    elif ":" in listen:
        host = f"[{listen}]"
    else:
        host = listen
    target = f"{host}:{port}"
    try:
        with grpc.insecure_channel(target) as channel:
            return pbg.NakshatraStub(channel).Info(pb.InfoRequest(), timeout=timeout_s)
    except Exception:
        cert = Path.home() / ".nakshatra" / "tls" / "worker-cert.pem"
        roots = cert.read_bytes()
        credentials = grpc.ssl_channel_credentials(root_certificates=roots)
        with grpc.secure_channel(
                target, credentials,
                options=[("grpc.ssl_target_name_override", "nakshatra.local")]) as channel:
            return pbg.NakshatraStub(channel).Info(pb.InfoRequest(), timeout=timeout_s)


def _wait_worker_paid_ready(env_path: Path, timeout_s: float = 90.0) -> bool:
    from paid_admission import PAID_CAPABILITIES
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            info = _worker_paid_info(env_path)
            if set(PAID_CAPABILITIES) <= set(info.protocol_capabilities):
                return True
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _gateway_probe_url(env_path: Path) -> str:
    values = _env_file_values(env_path)
    bind = values.get("BIND", "127.0.0.1")
    port = values.get("PORT", "11600")
    if bind in {"0.0.0.0", "*"}:
        bind = "127.0.0.1"
    elif bind in {"::", "[::]"}:
        bind = "[::1]"
    elif ":" in bind and not bind.startswith("["):
        bind = f"[{bind}]"
    return f"http://{bind}:{port}/v1/models"


def _wait_gateway_ready(env_path: Path, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with urlopen(_gateway_probe_url(env_path), timeout=2.0) as response:
                value = json.loads(response.read().decode())
            if isinstance(value, dict) and isinstance(value.get("data"), list):
                return True
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _loopback_bind(value: str) -> bool:
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _gateway_on(_sp, args) -> None:
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if not _loopback_bind(args.bind) and not args.allow_lan:
        raise SystemExit("refusing a non-loopback gateway bind without --allow-lan")
    pillar = _bare_http_url(args.pillar_url, "pillar-url")
    ledger = _bare_http_url(args.ledger_url, "ledger-url", allow_path=True)
    for value, name in ((pillar, "pillar-url"), (ledger, "ledger-url")):
        parsed = urlsplit(value)
        try:
            local = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            local = parsed.hostname == "localhost"
        if parsed.scheme != "https" and not local:
            raise SystemExit(f"--{name} must use https outside loopback")
    pin = (args.pillar_spki or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", pin):
        raise SystemExit("--pillar-spki must be a 64-character SHA-256 hex digest")
    if args.models != "auto" and not Path(args.models).is_file():
        raise SystemExit("--models must be 'auto' or an existing serve_models YAML file")
    package_url = package_pubkey = ""
    if args.models == "auto":
        try:
            release_package_signers = load_release_package_signers()
        except ReleaseTrustError as e:
            raise SystemExit(str(e)) from e
        default_url, default_pubkey = _gateway_package_defaults()
        package_url = (getattr(args, "package_url", None) or default_url).strip()
        package_pubkey = (getattr(args, "package_pubkey", None) or default_pubkey).strip().lower()
        if not package_url:
            raise SystemExit(
                "--models auto needs --package-url (or a package URL saved by invite/join)")
        try:
            parsed_package = urlsplit(package_url)
            package_host = parsed_package.hostname
        except ValueError as e:
            raise SystemExit(f"--package-url is not a valid URL: {e}") from e
        if parsed_package.scheme not in {"http", "https"} or not package_host:
            raise SystemExit("--package-url must be an http(s) package base or package.json URL")
        if package_pubkey and not re.fullmatch(r"[0-9a-f]{64}", package_pubkey):
            raise SystemExit(
                "--package-pubkey must be the package signer's Ed25519 public key "
                "(64 hex characters)")
        if package_pubkey and package_pubkey not in release_package_signers:
            raise SystemExit(
                "--package-pubkey may only narrow to a package signer trusted by the signed release")
    if args.models != "auto":
        # The service repeats this check at config load. Refuse here too so
        # `nak gateway on` never enables an enforce-mode registry that can
        # bypass settlement through a local engine proxy.
        from nakshatra_serve import ModelConfigError, _load_models
        try:
            _load_models(args.models, enforce_ledger=True)
        except ModelConfigError as e:
            raise SystemExit(f"refusing gateway models config: {e}") from e

    root, env_path = _gateway_paths()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    old_env = env_path.read_bytes() if env_path.exists() else None
    profile, profile_owner = _gateway_profile_paths()
    made_profile = not profile.exists()
    if made_profile:
        owner_token = secrets.token_hex(32)
        _write_private(
            profile,
            ("# Opted into the signed release inference profile.\n"
             f"{GATEWAY_PROFILE_OWNER_PREFIX}{owner_token}\n").encode(),
        )
        try:
            _write_private(profile_owner, f"{owner_token}\n".encode())
        except (Exception, SystemExit):
            profile.unlink(missing_ok=True)
            profile_owner.unlink(missing_ok=True)
            raise
    made_p2p = False
    host_consumer_override = False
    consumer_env = _p2p_consumer_env_path()
    old_consumer_env = consumer_env.read_bytes() if consumer_env.exists() else None
    consumer_override_changed = False
    package_body = ""
    if args.models == "auto":
        package_body = f"PACKAGE_URL={_env_value(package_url, 'package-url')}\n"
        if package_pubkey:
            package_body += f"PACKAGE_TRUSTED_PUBKEY={package_pubkey}\n"
    body = (
        f"PORT={args.port}\n"
        f"BIND={_env_value(args.bind, 'bind')}\n"
        f"MODELS={_env_value(args.models, 'models')}\n"
        f"{package_body}"
        f"NAKSHATRA_PILLAR_URL={_env_value(pillar, 'pillar-url')}\n"
        f"STHAMBHA_PILLAR_SPKI_SHA256={pin}\n"
        "NKS_TRUST_PINNED_PILLAR=1\n"
        "NAKSHATRA_CHAIN_TLS_MODE=required\n"
        "NAKSHATRA_CREDITS=1\n"
        "NAKSHATRA_LEDGER_MODE=enforce\n"
        f"NAKSHATRA_LEDGER_URL={_env_value(ledger, 'ledger-url')}\n"
        f"NAKSHATRA_LEDGER_SPKI_SHA256={pin}\n"
    ).encode()
    action = "restart" if old_env is not None and _gateway_unit_enabled(_sp) else "enable"
    gateway_attempted = False
    try:
        _write_private(env_path, body)
        # The existence decision, gateway ownership marker, opt-in creation, and service enable
        # are one transaction relative to explicit P2P and worker commands.
        with _p2p_ownership_lock():
            p2p_env, p2p_owner = _gateway_p2p_paths()
            made_p2p = not p2p_env.exists()
            scrubbed_legacy_relay = False
            if not made_p2p:
                scrubbed_legacy_relay = _remove_env_assignments(
                    p2p_env, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
            host_consumer_override = (
                "NAK_CONTAINER_CONSUMER_P2P_PORT" not in os.environ
                and not _worker_unit_enabled(_sp)
            )
            if host_consumer_override:
                _write_p2p_consumer_env()
                consumer_override_changed = consumer_env.read_bytes() != old_consumer_env
            if made_p2p:
                p2p_owner_token = secrets.token_hex(32)
                _write_private(p2p_owner, f"{p2p_owner_token}\n".encode())
                _enable_p2p_service_locked(_sp, owner_token=p2p_owner_token)
            elif consumer_override_changed or scrubbed_legacy_relay:
                _p2p_installer(_sp, "restart")
        gateway_attempted = True
        _gateway_installer(_sp, action)
    except (Exception, SystemExit) as transition_error:
        # A restart failure belongs to an already-live gateway generation. Restore its prior
        # configuration, but keep the consumer override continuously in place: the failed restart
        # may have left either generation running. A second restart is the compensating action.
        if action == "restart":
            if old_env is not None:
                _write_private(env_path, old_env)
            if gateway_attempted:
                try:
                    _gateway_installer(_sp, "restart")
                except (Exception, SystemExit) as rollback_error:
                    raise SystemExit(
                        f"{transition_error}; rollback could not restart {GATEWAY_SERVICE}: "
                        f"{rollback_error}"
                    ) from transition_error
            raise

        # An enable operation can fail after systemd has started the gateway but before its
        # stability check completes. Do not remove any protection until a compensating disable has
        # confirmed that the gateway stopped.
        if gateway_attempted:
            try:
                _gateway_installer(_sp, "disable")
            except (Exception, SystemExit) as rollback_error:
                raise SystemExit(
                    f"{transition_error}; rollback could not disable {GATEWAY_SERVICE}: "
                    f"{rollback_error}"
                ) from transition_error

        cleanup_errors = []
        if old_env is None:
            env_path.unlink(missing_ok=True)
        else:
            _write_private(env_path, old_env)
        if made_profile:
            _release_gateway_profile_ownership()
        if made_p2p:
            try:
                _release_gateway_p2p_ownership(_sp)
            except (Exception, SystemExit) as error:
                cleanup_errors.append(f"could not restore p2p ownership: {error}")
        elif consumer_override_changed:
            with _p2p_ownership_lock():
                if old_consumer_env is None:
                    consumer_env.unlink(missing_ok=True)
                else:
                    _write_private(consumer_env, old_consumer_env)
                if _p2p_enabled():
                    try:
                        _p2p_installer(_sp, "restart")
                    except (Exception, SystemExit) as error:
                        cleanup_errors.append(f"could not restart restored p2p: {error}")
        if cleanup_errors:
            raise SystemExit(f"{transition_error}; rollback " + "; ".join(cleanup_errors)) \
                from transition_error
        raise
    print(f"gateway ON at http://{args.bind}:{args.port}/v1 (credits enforced)")


def _unit_enabled_or_active(_sp, service: str) -> bool:
    """Conservatively recognize a unit that can still be serving traffic.

    Opt-in files are configuration, not liveness. In particular, a disabled worker may retain a
    recovery copy of worker.env after a failed transition, while an active transient unit may no
    longer report enabled. Treat either positive systemd state as live.
    """
    positive = {
        "enabled", "enabled-runtime", "linked", "linked-runtime", "alias",
        "active", "activating", "reloading",
    }
    for command in ("is-enabled", "is-active"):
        try:
            result = _sp.run(
                ["systemctl", "--user", command, f"{service}.service"],
                capture_output=True, text=True, check=False)
        except (OSError, AttributeError):
            continue
        if result.returncode == 0 and (getattr(result, "stdout", "") or "").strip() in positive:
            return True
    return False


def _gateway_unit_enabled(_sp) -> bool:
    return _unit_enabled_or_active(_sp, GATEWAY_SERVICE)


def _p2p_unit_enabled(_sp) -> bool:
    return _unit_enabled_or_active(_sp, "nakshatra-p2p")


def _gateway_off(_sp) -> None:
    p2p_env = Path.home() / ".nakshatra" / "p2p.env"
    if p2p_env.exists():
        _remove_env_assignments(
            p2p_env, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
    _gateway_installer(_sp, "disable")
    _gateway_paths()[1].unlink(missing_ok=True)
    _release_gateway_profile_ownership()
    _release_gateway_p2p_ownership(_sp)
    print("gateway OFF")


def _gateway_paid(_sp, action: str) -> dict | None:
    _root, env_path = _gateway_paths()
    if action == "status":
        values = _env_file_values(env_path)
        return {
            "enabled": values.get(GATEWAY_PAID_ENV) == "1",
            "configured": env_path.is_file(),
            "ledger_mode": values.get("NAKSHATRA_LEDGER_MODE"),
            "ledger_url": values.get("NAKSHATRA_LEDGER_URL"),
        }
    with _p2p_ownership_lock():
        if not env_path.is_file():
            raise SystemExit("gateway paid mode needs a configured gateway; run `nak gateway on` first")
        old_env = env_path.read_bytes()
        values = _env_file_values(env_path)
        if action == "on":
            if values.get("NAKSHATRA_LEDGER_MODE") != "enforce":
                raise SystemExit("gateway paid mode requires NAKSHATRA_LEDGER_MODE=enforce")
            _paid_probe(
                values.get("NAKSHATRA_LEDGER_URL", ""),
                values.get("NAKSHATRA_LEDGER_SPKI_SHA256", ""),
            )
            _replace_env_assignments(env_path, {GATEWAY_PAID_ENV: "1"})
            try:
                _gateway_installer(_sp, "restart")
                if not _wait_gateway_ready(env_path):
                    raise SystemExit("gateway paid restart did not become ready")
            except (Exception, SystemExit) as transition_error:
                try:
                    _write_private(env_path, old_env)
                except (Exception, SystemExit) as restore_error:
                    raise SystemExit(
                        f"{transition_error}; rollback could not restore {env_path}: "
                        f"{restore_error}"
                    ) from transition_error
                try:
                    _gateway_installer(_sp, "restart")
                except (Exception, SystemExit) as rollback_error:
                    raise SystemExit(
                        f"{transition_error}; rollback could not restart {GATEWAY_SERVICE}: "
                        f"{rollback_error}"
                    ) from transition_error
                raise
            print("gateway paid ON")
            return None

        changed = values.get(GATEWAY_PAID_ENV) == "1"
        _replace_env_assignments(env_path, {GATEWAY_PAID_ENV: None})
        if changed:
            # Fail safe: the on-disk routing gate remains off even if this
            # restart fails. A later service recovery therefore comes up legacy.
            _gateway_installer(_sp, "restart")
        print("gateway paid OFF")
        return None


def _gateway_status(_sp) -> dict:
    _root, env_path = _gateway_paths()
    values = _gateway_values()
    unit = {}
    for prop, command in (("active", "is-active"), ("enabled", "is-enabled")):
        try:
            result = _sp.run(["systemctl", "--user", command,
                              f"{GATEWAY_SERVICE}.service"],
                             capture_output=True, text=True, check=False)
            unit[prop] = (result.stdout or "").strip() or ("yes" if result.returncode == 0 else "no")
        except OSError as e:
            unit[prop] = f"unavailable: {e}"
    return {
        "configured": env_path.is_file(),
        "bind": values.get("BIND"), "port": int(values["PORT"]) if values.get("PORT", "").isdigit() else None,
        "models": values.get("MODELS"), "pillar_url": values.get("NAKSHATRA_PILLAR_URL"),
        "ledger_url": values.get("NAKSHATRA_LEDGER_URL"),
        "ledger_mode": values.get("NAKSHATRA_LEDGER_MODE"),
        "paid_admission": values.get(GATEWAY_PAID_ENV) == "1", "unit": unit,
    }


def _worker_paths() -> tuple[Path, Path, Path]:
    home = Path.home()
    root = home / ".nakshatra" / "worker"
    return root, root / "build.json", home / ".nakshatra" / "worker.env"


def _release_engine_tree(prefix: Path) -> tuple[str, dict]:
    """Verify the installed release, then return its strict engine TREE and manifest."""
    _verified_release_installer(prefix)
    try:
        man = json.loads((prefix / "current" / "manifest.json").read_text())
        base = (prefix / "current" / "nakshatra" / "engine" / "BASE").read_text()
    except (OSError, ValueError) as e:
        raise SystemExit(f"installed release has no readable engine identity: {e}")
    trees = re.findall(r"^TREE=([0-9a-f]{40})$", base, re.MULTILINE)
    if len(trees) != 1:
        raise SystemExit("installed release engine/BASE has no single valid TREE")
    return trees[0], man


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _write_private_new(path: Path, data: bytes) -> None:
    """Atomically create one owner-only file without replacing a concurrent winner."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    linked = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(tmp, path)
        linked = True
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            if not linked:
                raise


def _worker_build(_sp) -> dict:
    prefix = Path(os.environ.get("NAK_NODE_PREFIX", Path.home() / ".nakshatra-node"))
    tree, man = _release_engine_tree(prefix)
    script = prefix / "current" / "nakshatra" / "deploy" / "provision-worker.sh"
    if not script.is_file():
        raise SystemExit(f"installed release is missing {script}")
    root, record_path, _env_path = _worker_paths()
    engine_dir = root / "engine"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = dict(os.environ, WORKER_DIR=str(engine_dir), WORKER_BUILD_ONLY="1")
    markers: dict[str, str] = {}
    try:
        proc = _sp.Popen(["bash", str(script)], env=env, stdout=_sp.PIPE, stderr=_sp.STDOUT,
                         text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.startswith("NAK_BUILD_") and "=" in line:
                key, value = line.strip().split("=", 1)
                markers[key] = value
        rc = proc.wait()
    except OSError as e:
        raise SystemExit(f"could not run the installed release's worker build: {e}")
    if rc != 0:
        raise SystemExit(f"worker build failed with exit status {rc}; build.json was not changed")
    try:
        daemon = Path(markers["NAK_BUILD_DAEMON"]).resolve(strict=True)
        backend = markers["NAK_BUILD_BACKEND"]
        n_gpu_layers = int(markers["NAK_BUILD_N_GPU_LAYERS"])
    except (KeyError, OSError, ValueError) as e:
        raise SystemExit(f"worker build completed without a valid result: {e}")
    if not daemon.is_relative_to(engine_dir.resolve()) or not daemon.is_file() or not os.access(daemon, os.X_OK):
        raise SystemExit(f"worker build returned an unsafe or non-executable daemon path: {daemon}")
    if backend not in WORKER_BACKENDS or n_gpu_layers < 0:
        raise SystemExit(f"worker build returned invalid runtime flags: backend={backend!r}, n_gpu_layers={n_gpu_layers}")
    record = {"daemon_bin": str(daemon), "backend": backend, "n_gpu_layers": n_gpu_layers,
              "engine_tree": tree, "release_version": man.get("version", ""), "built_at": int(time.time())}
    _write_private(record_path, (json.dumps(record, indent=1, sort_keys=True) + "\n").encode())
    print(f"worker build ready: {backend}, engine {tree[:12]}…, daemon {daemon}")
    return record


def _worker_installer(_sp, action: str) -> None:
    prefix = Path(os.environ.get("NAK_NODE_PREFIX", Path.home() / ".nakshatra-node"))
    installer = _verified_release_installer(prefix)
    try:
        _sp.run([sys.executable, str(installer), "--prefix", str(prefix), action, WORKER_SERVICE], check=True)
    except (OSError, _sp.CalledProcessError) as e:
        raise SystemExit(f"could not {action} {WORKER_SERVICE}: {e}")


def _p2p_enabled() -> bool:
    """The release's opt-in file is the authoritative sidecar-enabled marker."""
    return (Path.home() / ".nakshatra" / "p2p.env").is_file()


def _env_assignment(raw: str) -> tuple[str, str] | None:
    """Normalize one systemd EnvironmentFile assignment without evaluating it."""
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].lstrip()
    if "=" not in line:
        return None
    name, value = line.split("=", 1)
    name = name.rstrip(" \t\r")
    return name, value


def _set_env_assignment(path: Path, name: str, value: str | None) -> None:
    """Replace one assignment in a release-owned env file, preserving the rest."""
    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        raise SystemExit(f"cannot update {path}: {e}") from e
    lines = [
        line for line in lines
        if (assignment := _env_assignment(line)) is None or assignment[0] != name
    ]
    if value is not None:
        lines.append(f'{name}="{value}"')
    _write_private(path, (("\n".join(lines) + "\n") if lines else "").encode())


def _remove_env_assignment(path: Path, name: str) -> bool:
    """Atomically remove one assignment without changing any other bytes in the env file."""
    return _remove_env_assignments(path, name)


def _remove_env_assignments(path: Path, *names: str) -> bool:
    """Atomically remove selected assignments without changing any other bytes."""
    try:
        body = path.read_bytes()
    except OSError as e:
        raise SystemExit(f"cannot update {path}: {e}") from e
    lines = body.splitlines(keepends=True)
    kept = []
    for line in lines:
        try:
            assignment = _env_assignment(line.decode())
        except UnicodeDecodeError:
            assignment = None
        if assignment is None or assignment[0] not in names:
            kept.append(line)
    if len(kept) == len(lines):
        return False
    _write_private(path, b"".join(kept))
    return True


def _worker_port(env_path: Path) -> int:
    try:
        body = env_path.read_text()
    except OSError as e:
        raise SystemExit(f"cannot read worker port from {env_path}: {e}") from e
    match = re.search(r"(?m)^PORT=(?:\"(\d+)\"|'(\d+)'|(\d+))\s*$", body)
    if not match:
        raise SystemExit(f"{env_path} has no valid PORT assignment")
    port = int(next(value for value in match.groups() if value is not None))
    if not 1 <= port <= 65535:
        raise SystemExit(f"{env_path} has an out-of-range PORT assignment")
    return port


def _env_value(value: str, name: str) -> str:
    if not value or any(c in value for c in "\r\n\0") or any(c.isspace() for c in value):
        raise SystemExit(f"--{name} must be one non-empty value without whitespace")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _worker_relay_enabled(env_path: Path) -> bool:
    try:
        body = env_path.read_text()
    except OSError:
        return False
    match = re.search(r"(?m)^NAKSHATRA_P2P_WORKER_ARGS=(.*)$", body)
    if not match:
        return False
    try:
        values = shlex.split(match.group(1))
        args = shlex.split(values[0]) if len(values) == 1 else values
        return "-allow-relayed-activation" in args
    except ValueError:
        return False


def _worker_p2p_args(port: int, allow_relayed: bool) -> str:
    args = f"-inbound=127.0.0.1:{port} -require-direct-activation"
    if allow_relayed:
        args += " -allow-relayed-activation"
    return args


def _worker_enable(_sp, args) -> None:
    with _p2p_ownership_lock():
        _worker_enable_locked(_sp, args)


def _worker_enable_locked(_sp, args) -> None:
    root, _record, env_path = _worker_paths()
    # systemd's ReadWritePaths requires the path to exist before it can construct the sandbox. Keep
    # both writable roots present even when build.json is missing so the launcher gets to print the
    # useful refusal and TLS can create this node's certificate on its first start.
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    (Path.home() / ".nakshatra" / "tls").mkdir(parents=True, exist_ok=True, mode=0o700)
    old_env = env_path.read_bytes() if env_path.exists() else None
    p2p_env, p2p_owner = _gateway_p2p_paths()
    old_p2p_env = p2p_env.read_bytes() if p2p_env.exists() else None
    old_p2p_owner = p2p_owner.read_bytes() if p2p_owner.exists() else None
    consumer_env = _p2p_consumer_env_path()
    old_consumer_env = consumer_env.read_bytes() if consumer_env.exists() else None
    took_over_p2p = False
    worker_was_live = old_env is not None and _worker_unit_enabled(_sp)
    if old_env is None:
        missing = [name for name in ("model_id", "package_url", "package_pubkey", "pillar_url", "layers", "port")
                   if getattr(args, name) in (None, "")]
        if missing:
            raise SystemExit("first `nak worker on` needs --model-id, --package-url, --package-pubkey, "
                             "--pillar-url, --layers a-b, and --port")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", args.package_pubkey):
            raise SystemExit("--package-pubkey must be the package signer's Ed25519 public key (64 hex characters)")
        match = re.fullmatch(r"(\d+)-(\d+)", args.layers)
        if not match or int(match.group(1)) >= int(match.group(2)):
            raise SystemExit("--layers must be a non-empty half-open range a-b (for example 0-16)")
        start, end = map(int, match.groups())
        if not 1 <= args.port <= 65535:
            raise SystemExit("--port must be between 1 and 65535")
        if not math.isfinite(args.idle_unload_s) or args.idle_unload_s < 0:
            raise SystemExit("--idle-unload-s must be a finite number >= 0")
        if not math.isfinite(args.release_grace_s):
            raise SystemExit("--release-grace-s must be finite (negative disables it)")
        model = _env_value(args.model_id, "model-id")
        package = _env_value(args.package_url, "package-url")
        pillar = _env_value(args.pillar_url, "pillar-url")
        from worker_service import WorkerServiceError, validate_pillar_transport
        trusted_pillar_nets = args.trusted_pillar_nets
        try:
            _pillar_url, pillar_spki_value = validate_pillar_transport(
                args.pillar_url, args.pillar_spki or "", trusted_pillar_nets
            )
        except WorkerServiceError as e:
            raise SystemExit(str(e)) from e
        pillar_spki = _env_value(pillar_spki_value, "pillar-spki") if pillar_spki_value else '""'
        trusted_nets = _env_value(trusted_pillar_nets, "trusted-pillar-nets")
        slice_key = hashlib.sha256(
            f"{args.model_id}\0{args.package_url}\0{start}\0{end}".encode()).hexdigest()[:16]
        sub_gguf = root / f"slice-{slice_key}-L{start}-{end}.gguf"
        listen = _env_value(args.listen_address, "listen-address")
        body = (f"MODEL_ID={model}\nPACKAGE_URL={package}\nPACKAGE_TRUSTED_PUBKEY={args.package_pubkey.lower()}\n"
                f"PILLAR_URL={pillar}\n"
                f"PILLAR_SPKI_SHA256={pillar_spki}\nTRUSTED_PILLAR_NETS={trusted_nets}\n"
                f"MODE={args.mode}\nPORT={args.port}\n"
                f"LAYER_START={start}\nLAYER_END={end}\nSUB_GGUF={sub_gguf}\n"
                f"LISTEN={listen}\nIDLE_UNLOAD_S={args.idle_unload_s:g}\n"
                f"RELEASE_GRACE_S={args.release_grace_s:g}\n"
                f"WARM_DIRECTIVES={'1' if args.warm_directives == 'on' else '0'}\n")
        _write_private(env_path, body.encode())
    port = _worker_port(env_path)
    allow_relayed = _worker_relay_enabled(env_path)
    _remove_env_assignments(
        env_path, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
    _set_env_assignment(
        env_path,
        "NAKSHATRA_P2P_WORKER_ARGS",
        _worker_p2p_args(port, allow_relayed),
    )
    consumer_env.unlink(missing_ok=True)
    if p2p_env.exists():
        _remove_env_assignments(
            p2p_env, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
    action = "restart" if worker_was_live else "enable"
    worker_attempted = False
    try:
        worker_attempted = True
        _worker_installer(_sp, action)
        if _p2p_enabled():
            took_over_p2p = True
            _remove_env_assignments(
                p2p_env, "NAKSHATRA_P2P_NAKD_INBOUND",
                "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
            p2p_owner.unlink(missing_ok=True)
            _p2p_installer(_sp, "restart")
        else:
            took_over_p2p = True
            p2p_owner.unlink(missing_ok=True)
    except (Exception, SystemExit) as transition_error:
        # A pre-existing worker must never be stopped as compensation for a failed reconfiguration.
        # Put its prior config back, keep the consumer override absent, and restart both readers.
        if action == "restart":
            if old_env is not None:
                _write_private(env_path, old_env)
                _remove_env_assignments(
                    env_path, "NAKSHATRA_P2P_GATEWAY_ARGS",
                    "NAKSHATRA_P2P_CONSUMER_ARGS")
            consumer_env.unlink(missing_ok=True)
            rollback_errors = []
            if _p2p_enabled():
                try:
                    _p2p_installer(_sp, "restart")
                except (Exception, SystemExit) as error:
                    rollback_errors.append(f"could not restart restored p2p: {error}")
            if worker_attempted:
                try:
                    _worker_installer(_sp, "restart")
                except (Exception, SystemExit) as error:
                    rollback_errors.append(f"could not restart {WORKER_SERVICE}: {error}")
            if rollback_errors:
                raise SystemExit(f"{transition_error}; rollback " + "; ".join(rollback_errors)) \
                    from transition_error
            raise

        # enable may have started the unit before its stability check failed. Restore prior files
        # only after the compensating disable confirms that the worker is stopped.
        try:
            _worker_installer(_sp, "disable")
        except (Exception, SystemExit) as disable_error:
            restart_error = None
            if _p2p_enabled():
                try:
                    _p2p_installer(_sp, "restart")
                except (Exception, SystemExit) as error:
                    restart_error = error
            detail = (
                f"{transition_error}; rollback could not disable {WORKER_SERVICE}: {disable_error}"
            )
            if restart_error is not None:
                detail += f"; could not restart p2p with retained worker inbound: {restart_error}"
            raise SystemExit(detail) from transition_error

        if old_env is None:
            env_path.unlink(missing_ok=True)
        else:
            _write_private(env_path, old_env)
            _remove_env_assignments(
                env_path, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
        if took_over_p2p:
            if old_p2p_env is None:
                p2p_env.unlink(missing_ok=True)
            else:
                _write_private(p2p_env, old_p2p_env)
                _remove_env_assignments(
                    p2p_env, "NAKSHATRA_P2P_GATEWAY_ARGS",
                    "NAKSHATRA_P2P_CONSUMER_ARGS")
            if old_p2p_owner is None:
                p2p_owner.unlink(missing_ok=True)
            else:
                _write_private(p2p_owner, old_p2p_owner)
        if old_consumer_env is None:
            consumer_env.unlink(missing_ok=True)
        else:
            _write_private(consumer_env, old_consumer_env)
        if _p2p_enabled():
            try:
                _p2p_installer(_sp, "restart")
            except (Exception, SystemExit) as rollback_error:
                raise SystemExit(
                    f"{transition_error}; rollback could not restart restored p2p: {rollback_error}"
                ) from transition_error
        raise
    print(f"worker ON ({env_path})")
    if not _p2p_enabled():
        print("p2p is not enabled; this worker is reachable only by its literal address")


def _worker_disable(_sp) -> None:
    with _p2p_ownership_lock():
        _worker_disable_locked(_sp)


def _worker_disable_locked(_sp) -> None:
    _root, _record, env_path = _worker_paths()
    p2p_env = Path.home() / ".nakshatra" / "p2p.env"
    if p2p_env.exists():
        _remove_env_assignments(
            p2p_env, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
    try:
        _worker_installer(_sp, "disable")
    except (Exception, SystemExit):
        raise
    # The worker is now confirmed stopped. A mock/dev installer may retain worker.env, so remove
    # its stale inbound assignment explicitly; a real installer has already removed the opt-in.
    if env_path.exists():
        _remove_env_assignments(
            env_path, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
        _set_env_assignment(env_path, "NAKSHATRA_P2P_WORKER_ARGS", None)
    protect_gateway = (
        "NAK_CONTAINER_CONSUMER_P2P_PORT" not in os.environ
        and _gateway_unit_enabled(_sp)
    )
    if protect_gateway:
        # This final override must exist before the sidecar reloads without worker inbound.
        _write_p2p_consumer_env()
    if _p2p_enabled():
        _p2p_installer(_sp, "restart")
    print("worker OFF")


def _worker_relay(_sp, enabled: bool) -> None:
    with _p2p_ownership_lock():
        _root, _record, env_path = _worker_paths()
        if not env_path.is_file():
            raise SystemExit("worker relay needs a configured worker; run `nak worker on` first")
        old_env = env_path.read_bytes()
        port = _worker_port(env_path)
        _remove_env_assignments(
            env_path, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
        _set_env_assignment(
            env_path, "NAKSHATRA_P2P_WORKER_ARGS", _worker_p2p_args(port, enabled)
        )
        p2p_enabled = _p2p_enabled()
        p2p_env = Path.home() / ".nakshatra" / "p2p.env"
        if p2p_enabled:
            _remove_env_assignments(
                p2p_env, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
        try:
            if p2p_enabled:
                _p2p_installer(_sp, "restart")
        except (Exception, SystemExit) as restart_error:
            _write_private(env_path, old_env)
            _remove_env_assignments(
                env_path, "NAKSHATRA_P2P_GATEWAY_ARGS", "NAKSHATRA_P2P_CONSUMER_ARGS")
            try:
                _p2p_installer(_sp, "restart")
            except (Exception, SystemExit) as rollback_error:
                raise SystemExit(
                    f"relay restart failed: {restart_error}; "
                    f"rollback restart failed: {rollback_error}"
                ) from rollback_error
            raise
    print(f"worker relay {'ON' if enabled else 'OFF'}")


def _worker_status(_sp, *, preflight: bool = False) -> dict:
    _root, record_path, env_path = _worker_paths()
    build = None
    build_state = "missing"
    if record_path.exists():
        try:
            build = json.loads(record_path.read_text())
            prefix = Path(os.environ.get("NAK_NODE_PREFIX", Path.home() / ".nakshatra-node"))
            current_tree, _man = _release_engine_tree(prefix)
            build_state = "ready" if build.get("engine_tree") == current_tree else "stale"
        except (OSError, ValueError, SystemExit) as e:
            build_state = f"invalid: {e}"
    unit = {}
    for prop, command in (("active", "is-active"), ("enabled", "is-enabled")):
        try:
            result = _sp.run(["systemctl", "--user", command, f"{WORKER_SERVICE}.service"],
                             capture_output=True, text=True, check=False)
            unit[prop] = (result.stdout or "").strip() or ("yes" if result.returncode == 0 else "no")
        except OSError as e:
            unit[prop] = f"unavailable: {e}"
    paid_values = _env_file_values(env_path)
    status = {
        "configured": env_path.exists(),
        "allow_relayed_activation": _worker_relay_enabled(env_path),
        "paid_admission": paid_values.get("NAKSHATRA_PAID_ADMISSION") == "1",
        "paid_ledger_url": paid_values.get("NAKSHATRA_PAID_LEDGER_URL"),
        "paid_store": paid_values.get("NAKSHATRA_PAID_STORE"),
        "build_state": build_state,
        "build": build,
        "unit": unit,
    }
    # The preflight runs GPU vendor tools (nvidia-smi, rocm-smi); only on an explicit request, never on a status poll.
    if preflight:
        status["preflight"] = _worker_preflight(_sp)
    return status


def _worker_preflight(_sp) -> dict:
    """Probe this node without changing it, using only the signed release catalog."""
    prefix = Path(os.environ.get("NAK_NODE_PREFIX", Path.home() / ".nakshatra-node"))
    error = None
    try:
        catalog = load_release_package_catalog(prefix)
    except ReleaseTrustError as e:
        catalog = {}
        error = str(e)
    from worker_preflight import collect_preflight
    return collect_preflight(
        catalog, catalog_error=error, state_dir=_worker_paths()[0], run=_sp.run)


def _worker_paid(_sp, action: str, args=None) -> dict | None:
    root, _record, env_path = _worker_paths()
    if action == "status":
        values = _env_file_values(env_path)
        store = values.get("NAKSHATRA_PAID_STORE")
        if values.get("NAKSHATRA_PAID_ADMISSION") == "1" and not store:
            store = str(root / "paid.sqlite")
        owner_only = False
        if store:
            try:
                metadata = Path(store).lstat()
                owner_only = (
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_uid == os.geteuid()
                    and stat.S_IMODE(metadata.st_mode) == 0o600
                )
            except OSError:
                pass
        return {
            "enabled": values.get("NAKSHATRA_PAID_ADMISSION") == "1",
            "configured": env_path.is_file(),
            "ledger_url": values.get("NAKSHATRA_PAID_LEDGER_URL"),
            "store": store,
            "store_owner_only": owner_only,
        }
    with _p2p_ownership_lock():
        if not env_path.is_file():
            raise SystemExit("worker paid mode needs a configured worker; run `nak worker on` first")
        old_env = env_path.read_bytes()
        values = _env_file_values(env_path)
        if action == "on":
            ledger = _bare_http_url(args.ledger_url, "ledger-url", allow_path=True)
            pin = (args.ledger_spki or "").strip().lower()
            store_arg = args.store or (root / "paid.sqlite")
            _paid_probe(ledger, pin)
            store_path = Path(store_arg).expanduser()
            if not store_path.is_absolute():
                store_path = Path.cwd() / store_path
            store_path = Path(os.path.abspath(store_path))
            worker_root = root.resolve()
            if (not store_path.is_relative_to(worker_root)
                    or not store_path.parent.resolve().is_relative_to(worker_root)):
                raise SystemExit(
                    f"--store must be inside {root} (the signed worker unit's writable root)"
                )
            store = _paid_store_preflight(store_path)
            _replace_env_assignments(env_path, {
                "NAKSHATRA_PAID_ADMISSION": "1",
                "NAKSHATRA_PAID_LEDGER_URL": ledger,
                "NAKSHATRA_PAID_LEDGER_SPKI_SHA256": pin,
                "NAKSHATRA_PAID_STORE": str(store),
            })
            try:
                _worker_installer(_sp, "restart")
                if not _wait_worker_paid_ready(env_path):
                    raise SystemExit("worker paid restart did not advertise paid readiness")
            except (Exception, SystemExit) as transition_error:
                try:
                    _write_private(env_path, old_env)
                except (Exception, SystemExit) as restore_error:
                    raise SystemExit(
                        f"{transition_error}; rollback could not restore {env_path}: "
                        f"{restore_error}"
                    ) from transition_error
                try:
                    _worker_installer(_sp, "restart")
                except (Exception, SystemExit) as rollback_error:
                    raise SystemExit(
                        f"{transition_error}; rollback could not restart {WORKER_SERVICE}: "
                        f"{rollback_error}"
                    ) from transition_error
                raise
            print("worker paid ON")
            return None

        changed = any(name in values for name in WORKER_PAID_ENV)
        _replace_env_assignments(env_path, {name: None for name in WORKER_PAID_ENV})
        if changed:
            # Consumer resolution is disabled on disk before the restart. Do
            # not restore it if the stop/start path itself is unhealthy.
            _worker_installer(_sp, "restart")
        print("worker paid OFF")
        return None


def _owner_lease_path() -> Path:
    return Path.home() / ".nakshatra" / "owner-lease.env"


def _worker_unit_enabled(_sp) -> bool:
    return _unit_enabled_or_active(_sp, WORKER_SERVICE)


def _owner_lease_values(path: Path) -> dict[str, str]:
    try:
        body = path.read_text()
    except OSError:
        return {}
    values = {}
    for name in ("OWNER_LEASE_PORT", "OWNER_LEASE_TOKEN", "OWNER_LEASE_TTL_S",
                 "OWNER_LEASE_RESUME_S"):
        match = re.search(rf"(?m)^{name}=([^\r\n]+)$", body)
        if match:
            values[name] = match.group(1).strip().strip("\"'")
    return values


def _owner_lease_on(_sp, args) -> None:
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if not math.isfinite(args.ttl_s) or args.ttl_s <= 0:
        raise SystemExit("--ttl-s must be finite and > 0")
    if not math.isfinite(args.resume_s) or args.resume_s < 0:
        raise SystemExit("--resume-s must be finite and >= 0")
    path = _owner_lease_path()
    old = path.read_bytes() if path.exists() else None
    previous = _owner_lease_values(path)
    token = previous.get("OWNER_LEASE_TOKEN", "").lower()
    if len(token) < 64 or len(token) % 2 or not re.fullmatch(r"[0-9a-f]+", token):
        token = secrets.token_hex(32)
    body = (
        f"OWNER_LEASE_PORT={args.port}\n"
        f"OWNER_LEASE_TOKEN={token}\n"
        f"OWNER_LEASE_TTL_S={args.ttl_s:g}\n"
        f"OWNER_LEASE_RESUME_S={args.resume_s:g}\n"
    ).encode()
    _write_private(path, body)
    try:
        if _worker_unit_enabled(_sp):
            _worker_installer(_sp, "restart")
    except (Exception, SystemExit):
        if old is None:
            path.unlink(missing_ok=True)
        else:
            _write_private(path, old)
        raise
    print(f"owner lease ON: 127.0.0.1:{args.port}; token file: {path}")


def _owner_lease_off(_sp) -> None:
    path = _owner_lease_path()
    old = path.read_bytes() if path.exists() else None
    path.unlink(missing_ok=True)
    try:
        if _worker_unit_enabled(_sp):
            _worker_installer(_sp, "restart")
    except (Exception, SystemExit):
        if old is not None:
            _write_private(path, old)
        raise
    print("owner lease OFF")


def _owner_lease_status() -> dict:
    path = _owner_lease_path()
    values = _owner_lease_values(path)
    result = {"enabled": path.is_file(), "token_file": str(path)}
    if path.is_file():
        result.update({
            "port": int(values.get("OWNER_LEASE_PORT", "5571")),
            "ttl_s": float(values.get("OWNER_LEASE_TTL_S", "6")),
            "resume_s": float(values.get("OWNER_LEASE_RESUME_S", "30")),
            "token_configured": bool(values.get("OWNER_LEASE_TOKEN")),
            "owner_only": not bool(path.stat().st_mode & 0o077),
        })
    return result


def _default_p2p_socket() -> Path:
    from client_defaults import default_p2p_socket_path
    return default_p2p_socket_path()


def _grant_paths() -> tuple[Path, Path, str]:
    net_dir = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net"))
    signer_dir = Path(os.environ.get("NAK_SIGNER_DIR", Path.home() / ".sthambha" / "signer"))
    try:
        agent = (net_dir / "agent").read_text().strip()
    except OSError as exc:
        raise SystemExit(f"cannot identify this node's delegated agent from {net_dir / 'agent'}: {exc}") from exc
    try:
        from sthambha import signer as signer_mod
    except ImportError as exc:
        raise SystemExit(f"cannot load the signer while resolving agent {agent!r}: {exc}") from exc
    try:
        signer_mod.validate_agent_name(agent)
    except (OSError, ValueError, signer_mod.SignerError) as exc:
        raise SystemExit(f"refusing invalid delegated agent state for {agent!r}: {exc}") from exc
    return signer_dir, net_dir, agent


def _broker_report_subject(subject: str) -> str:
    """Resolve a broker-mode report target from the local store, without a daemon RPC."""
    if isinstance(subject, str) and re.fullmatch(r"[0-9a-f]{64}", subject) is not None:
        return subject
    signer_dir = Path(os.environ.get("NAK_SIGNER_DIR", Path.home() / ".sthambha" / "signer"))
    net_dir = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net"))
    data_key_path = signer_dir / "datakey"
    try:
        data_key = _read_owned_regular(
            data_key_path, "Nakshatra store data key", exact_mode=0o600, max_bytes=32)
        from network.store import resolve_contact_readonly
        contact = resolve_contact_readonly(net_dir, data_key, subject)
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise SystemExit(
            "broker custody could not resolve the report target from the local contact store; "
            "nothing was signed") from exc
    if not contact or re.fullmatch(r"[0-9a-f]{64}", str(contact.get("node", ""))) is None:
        raise SystemExit(
            "broker custody requires a local contact name or a 64-character node key for reports; "
            "nothing was signed")
    return contact["node"]


def _delegate_reuse_command(state: Path, person_key: Path, agent: str, caps: list[str], ttl_days: int,
                            idle_days: int, lease_days: int, no_lease: bool,
                            *, custody: str | None = None) -> str:
    if (custody if custody is not None else custody_mode()) == "broker":
        command = ("nak grant renew --caps " + shlex.quote(",".join(caps)) +
                   f" --ttl {ttl_days} --yes")
        if no_lease:
            return command + " --no-lease"
        return command + f" --idle-days {idle_days} --lease-days {lease_days}"
    command = ("python -m sthambha.signer delegate --state " + shlex.quote(str(state)) +
            " --person-key " + shlex.quote(str(person_key)) + " --agent " + shlex.quote(agent) +
            " --caps " + shlex.quote(",".join(caps)) + f" --ttl {ttl_days * 86400} --reuse-key")
    if no_lease:
        return command + " --no-lease"
    return command + f" --idle-days {idle_days} --lease-days {lease_days}"


OLD_SIGNER_MESSAGE = "signer too old for sliding grants; update the node or use --no-lease"


def _cli_amount(text: str, *, max_base_units: int = MAX_SPEND_CAP) -> int:
    try:
        amount, legacy = parse_amount(text, max_base_units=max_base_units)
    except AmountError as exc:
        raise SystemExit(str(exc)) from exc
    if legacy:
        print(f"notice: read bare integer {text!r} as base units for backward compatibility; "
              "append 'base', or use a decimal point, $ or USDC for human units", file=sys.stderr)
    return amount


def _display_amount_fields(row: dict, *names: str) -> dict:
    shown = dict(row)
    for name in names:
        value = shown.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            shown.setdefault(f"{name}_usdc", format_amount(value))
            shown[f"{name}_display"] = format_amount(value)
    return shown


def _grant_renew(args) -> int:
    """Human CLI boundary for re-signing one grant without rotating its agent key."""
    # The accurate statement summary is built after reading the current grant. This early, generic
    # preflight keeps broker custody ahead of the historical TTY check without moving file-mode work.
    for flag, name in (("--per-tx-cap", "per_tx_cap"), ("--day-cap", "day_cap"),
                       ("--verdict-auto-max", "verdict_auto_max")):
        value = getattr(args, name)
        if value is not None:
            try:
                setattr(args, name, _cli_amount(value))
            except SystemExit as exc:
                raise SystemExit(f"{flag} {exc}") from exc
    custody_act = broker_preflight("grant-renew", "Renew the delegated agent grant")
    if not args.yes and not sys.stdin.isatty():
        raise SystemExit("nak grant renew is interactive-only; run it in a terminal or pass --yes explicitly")
    if not 1 <= args.ttl <= 365:
        raise SystemExit("--ttl must be between 1 and 365 days")
    if not 1 <= args.idle_days <= 90:
        raise SystemExit("--idle-days must be between 1 and 90 days")
    if not 1 <= args.lease_days <= 7:
        raise SystemExit("--lease-days must be between 1 and 7 days")
    for flag, value in (("--per-tx-cap", args.per_tx_cap), ("--day-cap", args.day_cap),
                        ("--verdict-auto-max", args.verdict_auto_max)):
        if value is not None and not 0 <= value <= MAX_SPEND_CAP:
            raise SystemExit(f"{flag} must be between 0 and {MAX_SPEND_CAP} base units")
    signer_dir, _net_dir, agent = _grant_paths()
    from sthambha import signer as signer_mod
    if not args.no_lease and not callable(getattr(signer_mod, "load_or_create_lease_key", None)):
        raise SystemExit(OLD_SIGNER_MESSAGE)
    try:
        with signer_mod.open_agent_files(signer_dir, agent) as agent_files:
            return _grant_renew_confined(
                args, signer_dir, agent, signer_mod, agent_files, custody_act)
    except (OSError, ValueError, signer_mod.SignerError) as exc:
        raise SystemExit(f"cannot renew confined agent {agent!r}: {exc}") from exc


def _grant_renew_confined(args, signer_dir: Path, agent: str, signer_mod, agent_files,
                           custody_act) -> int:
    """Renew while holding the one no-follow agents-directory descriptor."""
    try:
        current = agent_files.load_grant()
    except signer_mod.SignerError as exc:
        raise SystemExit(f"cannot read the current grant at {agent_files.grant_path}: {exc}") from exc
    current_caps = sorted({str(cap) for cap in current.get("caps") or [] if str(cap)})
    if args.caps is None:
        caps = current_caps
    else:
        caps = sorted({cap.strip() for cap in args.caps.split(",") if cap.strip()})
        if not caps:
            raise SystemExit("--caps must name at least one capability")
    if not caps:
        raise SystemExit("the current grant has no capabilities; name the intended set with --caps")
    try:
        _, agent_pub = agent_files.load_key()
        signer_person = (signer_dir / "person.pub").read_text().strip()
        node_pub = (signer_dir / "node.pub").read_text().strip()
    except (OSError, ValueError, signer_mod.SignerError) as exc:
        raise SystemExit(f"cannot load signer identity for renewal: {exc}") from exc
    if current.get("subject") != agent or current.get("subject_pub") != agent_pub:
        raise SystemExit("the current grant is not bound to the existing agent key; refusing renewal")
    if not signer_mod._verify(current, signer_person):  # noqa: SLF001 — authenticate stale/expired metadata
        raise SystemExit("the current grant signature does not verify; refusing to copy its authority")
    if current.get("node") not in ("*", node_pub):
        raise SystemExit("the current grant is bound to a different node; refusing renewal")
    constraints = current.get("constraints") or {}
    if not isinstance(constraints, dict):
        raise SystemExit("the current grant has malformed constraints; refusing renewal")
    constraints = dict(constraints)
    cap_changes = []
    for name, value in (("per_tx_cap", args.per_tx_cap), ("day_cap", args.day_cap)):
        if value is not None and constraints.get(name) != value:
            cap_changes.append(
                f"{name} {format_amount(int(constraints.get(name, 0)))} -> {format_amount(value)}")
            constraints[name] = value
    from network import verdict as verdict_policy
    policy_changes = []
    requested_policy = (
        ("verdict_mode", args.verdict),
        ("verdict_auto_max", args.verdict_auto_max),
        ("verdict_ask_strangers", None if args.verdict_ask_strangers is None
         else args.verdict_ask_strangers == "yes"),
    )
    for name, value in requested_policy:
        if value is not None and constraints.get(name, {
                "verdict_mode": verdict_policy.DEFAULT_MODE,
                "verdict_auto_max": verdict_policy.DEFAULT_AUTO_MAX,
                "verdict_ask_strangers": verdict_policy.DEFAULT_ASK_STRANGERS,
        }[name]) != value:
            policy_changes.append(f"{name} -> {str(value).lower()}")
            constraints[name] = value
    effective_policy = verdict_policy.policy(constraints)
    widening = sorted(set(caps) - set(current_caps))
    if cap_changes:
        person_kind = "spend-cap"
    else:
        person_kind = "grant-renew"
    summary_parts = [
        f"Renew agent {plain(agent, 64)}",
        f"ttl {args.ttl} days",
        f"caps {','.join(caps)}",
    ]
    if widening:
        summary_parts.append(f"widened caps {','.join(widening)}")
    if cap_changes:
        summary_parts.append(f"spend caps {', '.join(cap_changes)}")
    if policy_changes:
        summary_parts.append(f"verdict policy {', '.join(policy_changes)}")
    summary = "; ".join(summary_parts)
    person_key = _person_key_path()
    fallback = _delegate_reuse_command(
        signer_dir, person_key, agent, caps, args.ttl,
        args.idle_days, args.lease_days, args.no_lease, custody=custody_act.mode)
    if custody_act.mode == "file" and not person_key.is_file():
        if cap_changes or policy_changes:
            print(f"person key is not available at {person_key}; the grant was not changed.\n"
                  "Spend caps can only be changed where the person key is available; verdict policy "
                  "has the same human-only boundary.",
                  file=sys.stderr)
            return 1
        print(f"person key is not available at {person_key}; the grant was not changed.\n"
              "Run this human delegation command where the person key and signer state are available:\n"
              f"  {fallback}", file=sys.stderr)
        return 1
    if not args.yes:
        detail = f"; explicitly adding {','.join(widening)}" if widening else ""
        if cap_changes:
            detail += f"; spend caps {', '.join(cap_changes)}"
        if policy_changes:
            detail += f"; verdict policy {', '.join(policy_changes)}"
        detail += f"; answers: {verdict_policy.describe(effective_policy)}"
        behaviour = ("as a fixed-expiry legacy grant"
                     if args.no_lease else
                     f"as a charter active while used (a {args.lease_days}-day wire lease; "
                     f"lapses after {args.idle_days} idle days)")
        answer = input(f"Renew agent {agent!r} for {args.ttl} days with caps {','.join(caps)} "
                       f"{behaviour}{detail}? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("grant unchanged")
            return 1
    person_priv = PersonKey(
        person_kind, summary, act=custody_act.for_signing(person_kind, summary))
    lease = None
    pending_lease_key = None
    lease_path = signer_dir / "lease.key"
    lease_written = False
    if not args.no_lease:
        if lease_path.exists():
            from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed25519
            lease_seed = _read_owned_regular(
                lease_path, "lease key", exact_mode=0o600, max_bytes=32)
            if len(lease_seed) != 32:
                raise SystemExit(f"cannot renew with malformed lease key {lease_path}")
            _lease_priv = _ed25519.Ed25519PrivateKey.from_private_bytes(lease_seed)
            lease_pub = _lease_priv.public_key().public_bytes_raw().hex()
        else:
            # A new lease key is part of the approved grant, but not part of preflight.
            # Keep it only in memory until the PERSON signature has actually returned.
            pending_lease_key, lease_pub = signer_mod.new_key()
        lease = {"issuer_pub": lease_pub, "max_s": args.lease_days * 86400,
                 "idle_s": args.idle_days * 86400}
    grant_args = {"node": current["node"], "ttl_s": args.ttl * 86400, "constraints": constraints}
    if lease is not None:
        grant_args["lease"] = lease
    try:
        renewed = signer_mod.issue_grant(person_priv, agent, agent_pub, caps, **grant_args)
        if pending_lease_key is not None:
            _write_private_new(lease_path, pending_lease_key.private_bytes_raw())
            lease_written = True
        agent_files.write_grant(renewed)
    except BaseException:
        if lease_written:
            try:
                lease_path.unlink(missing_ok=True)
            except BaseException:
                pass
        raise
    if cap_changes:
        print(f"spend caps now {format_amount(renewed['constraints'].get('per_tx_cap', 0))} per payment, "
              f"{format_amount(renewed['constraints'].get('day_cap', 0))} per day")
    print("verdict policy: " + verdict_policy.describe(verdict_policy.policy(renewed.get("constraints"))))
    authority = ("fixed-expiry legacy grant"
                 if args.no_lease else
                 f"charter active while used; {args.lease_days}-day wire leases; "
                 f"lapses after {args.idle_days} idle days")
    expiry_name = "grant" if args.no_lease else "charter"
    print(f"renewed agent {agent} with caps {','.join(renewed['caps'])}: {authority}; {expiry_name} through "
          f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(renewed['expires_at']))} "
          f"(grant {renewed['grant_id']}); signer reload is automatic")
    return 0


def _print_grant_status(grant: dict) -> None:
    print(f"agent {grant['agent']}  caps {','.join(grant['caps']) or '-'}  " +
          (grant.get("status") or "grant expires " +
           time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(grant['expires_at']))))
    if grant.get("grant_id"):
        print(f"grant id: {grant['grant_id']}")
    if grant.get("warning"):
        print(f"WARNING: {grant['warning']}")
    spend_caps = grant.get("spend_caps") or grant.get("constraints")
    if isinstance(spend_caps, dict):
        print("spend caps: "
              f"{format_amount(int(spend_caps.get('per_tx_cap', 0)))} per payment; "
              f"{format_amount(int(spend_caps.get('day_cap', 0)))} per day")
    if grant.get("verdict_policy"):
        from network.verdict import describe
        print("paid-task verdicts: " + describe(grant["verdict_policy"]))


def _stored_own_revocation(net_dir: Path, person_pub: str) -> dict | None:
    """Load and authenticate prior cumulative state before it can influence a new signature."""
    from network.revocation import read_private_json, verify_record

    pending = net_dir / "pending-revocation.json"
    try:
        pending_record = read_private_json(pending)
        if pending_record is not None:
            pending_record = verify_record(pending_record, person_pub)
    except ValueError as exc:
        raise SystemExit(f"refused: invalid previous revocation record in {pending}: {exc}") from exc

    db_path = net_dir / "net.sqlite"
    if not db_path.is_file():
        return pending_record
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = db.execute("SELECT record_json FROM own_revocation LIMIT 1").fetchone()
        finally:
            db.close()
    except sqlite3.OperationalError as exc:
        if "no such table: own_revocation" in str(exc):
            return pending_record
        raise SystemExit(f"refused: cannot read previous revocation record in {db_path}: {exc}") from exc
    except (sqlite3.Error, OSError) as exc:
        raise SystemExit(f"refused: cannot read previous revocation record in {db_path}: {exc}") from exc
    if not row:
        return pending_record
    try:
        stored_record = verify_record(json.loads(row[0]), person_pub)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"refused: invalid previous revocation record in {db_path}: {exc}") from exc
    if pending_record is None:
        return stored_record
    # Both records are authenticated. Preserve their cumulative union even if a prior crash left
    # the pending file and installed store at different sequences.
    newest = pending_record if pending_record["seq"] >= stored_record["seq"] else stored_record
    return dict(newest, grant_ids=sorted(
        set(pending_record["grant_ids"]) | set(stored_record["grant_ids"])))


@contextmanager
def _revocation_lock(net_dir: Path):
    """Serialize every local read/sign/install transaction across CLI processes."""
    from network.revocation import RevocationLockError, revocation_lock
    try:
        with revocation_lock(net_dir):
            yield
    except RevocationLockError as exc:
        raise SystemExit(f"refused: {exc}") from exc


def _revoke(args, sock: Path) -> int:
    """The human-only boundary that revokes locally first, then publishes cumulatively."""
    requested_values = [args.grant_id, *(args.more_grant_ids or [])]
    summary = "Revoke grants " + ", ".join(
        plain(value, 32) or "(empty)" for value in requested_values)
    custody_act = broker_preflight("revoke", summary)
    if not sys.stdin.isatty():
        raise SystemExit("nak revoke is interactive-only; run it in a terminal")
    requested = requested_values
    bad = [value for value in requested if re.fullmatch(r"[0-9a-f]{16}", value or "") is None]
    if bad:
        raise SystemExit("grant ids must be 16 lowercase hex characters")
    requested = sorted(set(requested))
    signer_dir, net_dir, _agent = _grant_paths()
    try:
        person_pub = (signer_dir / "person.pub").read_text().strip()
    except OSError as exc:
        raise SystemExit(f"cannot load the person identity for revocation: {exc}") from exc
    person_priv = PersonKey("revoke", summary, act=custody_act)
    from network.revocation import (
        MAX_GRANTS, MAX_SEQ, attach_signature, build_unsigned_record,
        revocation_statement)
    with _revocation_lock(net_dir):
        # These are deliberately inside the lock. They are cumulative inputs shared by every CLI
        # process, so reading any of them before flock would reintroduce a lost-update window.
        previous = _stored_own_revocation(net_dir, person_pub)
        prior_ids = set(previous["grant_ids"]) if previous else set()
        seq = previous["seq"] + 1 if previous else 1
        if seq > MAX_SEQ:
            raise SystemExit(f"cannot create another revocation record: sequence limit {MAX_SEQ} reached")

        revoked_path = signer_dir / "revoked.txt"
        existing = set()
        try:
            if revoked_path.exists():
                existing = {line.strip() for line in revoked_path.read_text().splitlines()
                            if re.fullmatch(r"[0-9a-f]{16}", line.strip())}
        except OSError as exc:
            raise SystemExit(f"could not read {revoked_path}: {exc}") from exc

        already_revoked = prior_ids | existing
        resulting = sorted(already_revoked | set(requested))
        if len(resulting) > MAX_GRANTS:
            raise SystemExit(f"cannot revoke more than {MAX_GRANTS} distinct grant ids")
        rendered = ", ".join(
            f"{grant_id} ({'previously revoked' if grant_id in already_revoked else 'new'})"
            for grant_id in resulting)
        answer = input(f"REVOKE resulting grant set: {rendered}; tell every contact? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("nothing revoked")
            return 1

        unsigned_record = build_unsigned_record(
            person_pub, seq, resulting, int(time.time()))
        record = attach_signature(
            unsigned_record, person_priv.sign(revocation_statement(unsigned_record)))

        try:
            revoked_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(str(revoked_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND |
                         getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "a") as out:
                for grant_id in requested:
                    out.write(grant_id + "\n")
                out.flush()
                os.fsync(out.fileno())
            os.chmod(revoked_path, 0o600)
        except OSError as exc:
            raise SystemExit(f"could not update {revoked_path}: {exc}") from exc

        try:
            call(sock, {"op": "install_revocation", "record": record})
            (net_dir / "pending-revocation.json").unlink(missing_ok=True)
            delivery = "contacts will be told over their next live session"
        except SystemExit as exc:
            if "messaging service is not running" in str(exc):
                _write_private(net_dir / "pending-revocation.json",
                               (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
                delivery = ("nak-net is down; the record will be installed when it starts "
                            "(or re-run this command)")
            else:
                current = _stored_own_revocation(net_dir, person_pub)
                if current and current["seq"] >= record["seq"]:
                    raise SystemExit(
                        "refused: revocation was not installed because a newer record exists; "
                        "re-run nak revoke to merge it") from exc
                raise
    print(f"revoked {', '.join(requested)}; {delivery}")
    return 0


def _resolve_posted_task(sock: Path, ref: str) -> str:
    rows = [row for row in call(sock, {"op": "tasks", "role": "poster"})["tasks"]
            if row.get("task_hash", "").startswith(str(ref))]
    if len(rows) != 1:
        raise SystemExit("refused: no single posted task matches that id")
    return rows[0]["task_hash"]


def _admission_roster_has_node(node: str) -> bool:
    path = Path.home() / ".nakshatra" / "admission-peers.tsv"
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SystemExit(f"refused: cannot read local admission roster {path}: {exc}") from exc
    for line in lines:
        fields = line.strip().split()
        if fields and not fields[0].startswith("#") and fields[0] == node:
            return True
    return False


def _resolve_node_reference(sock: Path, ref: str, status: dict | None = None,
                            *, force_unknown: bool = False) -> str:
    rows = call(sock, {"op": "contacts"})["contacts"]
    status = status or call(sock, {"op": "status"})
    if ref == status.get("node") or ref == status.get("person"):
        raise SystemExit("refused: cannot suspend this person's own node")
    hits = [row for row in rows if row.get("node") == ref or row.get("petname") == ref or
            row.get("person") == ref or
            (len(ref) >= 8 and str(row.get("person", "")).startswith(ref))]
    if len(hits) == 1:
        node = hits[0]["node"]
        person = hits[0].get("person")
    if len(hits) > 1:
        raise SystemExit("refused: no single contact matches that node reference")
    if not hits:
        if re.fullmatch(r"[0-9a-f]{64}", ref or ""):
            node, person = ref, None
            if not force_unknown and not _admission_roster_has_node(node):
                raise SystemExit(
                    "refused: raw node is neither a contact nor in the local admission roster; "
                    "use --force-unknown-node only after verifying the key")
        else:
            raise SystemExit("refused: no single contact matches that node reference")
    if node == status.get("node") or (person and person == status.get("person")):
        raise SystemExit("refused: cannot suspend this person's own node")
    return node


def _resolve_contact_reference(sock: Path, ref: str) -> dict:
    rows = call(sock, {"op": "contacts"})["contacts"]
    hits = [row for row in rows if row.get("petname") == ref or row.get("person") == ref or
            (len(ref) >= 8 and str(row.get("person", "")).startswith(ref))]
    if len(hits) != 1:
        raise SystemExit("refused: no single contact matches that reference")
    return hits[0]


def _operator_onboard_base() -> str:
    base = os.environ.get("NAK_ONBOARD_BASE", "").strip()
    if not base:
        state = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net"))
        try:
            base = (state / "onboard-base").read_text().strip()
        except FileNotFoundError:
            base = DEFAULT_ONBOARD_BASE
        except OSError as exc:
            raise SystemExit(f"cannot read onboard-base: {exc}") from exc
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or \
            parsed.password or parsed.query or parsed.fragment:
        raise SystemExit("onboard base must be an http(s) URL without credentials, query, or fragment")
    return base.rstrip("/")


def _server_suspension_record(base: str, token: str, http_post, http_get) -> dict:
    """Fetch the signed server record, with compatibility for the redacted public view."""
    public = http_get(base + "/suspensions")
    record = public.get("record") if isinstance(public.get("record"), dict) else None
    if public.get("schema") == "nak.suspensions.v1":
        record = public
    if record is None:
        # Hardened servers expose only nodes+seq publicly.  Their authenticated operator view is
        # the same signed record and is the compatible recovery source.
        operator_view = http_post(base + "/operator/suspensions", {"token": token})
        record = operator_view.get("record")
    if not isinstance(record, dict):
        raise SystemExit("onboard server did not return a signed suspension record")
    return record


def _operator_ledger_base(value: str) -> str:
    """Validate the adjudication endpoint without resolving an attacker-controlled hostname."""
    base = _bare_http_url(value, "ledger")
    parsed = urlsplit(base)
    if parsed.scheme == "http":
        # Plain http only to a literal loopback IP: a hostname (even "localhost") can resolve anywhere.
        try:
            loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise SystemExit("--ledger must use https unless its host is a loopback IP address")
    return base


def _operator_ledger_request(base: str, private: bytes, keyid: str, method: str,
                             path: str, body: dict | None = None) -> dict:
    """Make one directly signed operator request to the ledger's spend listener."""
    import nakshatra_auth

    raw = (b"" if body is None else json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    authorization, _sent_ts = nakshatra_auth.build_signed_envelope(
        private, keyid, method, path, raw)
    request = Request(
        base + path,
        data=raw if method == "POST" else None,
        headers={"Authorization": authorization,
                 "Accept": "application/json",
                 **({"Content-Type": "application/json"} if method == "POST" else {})},
        method=method,
    )
    try:
        # No environment proxies: the signed adjudication must go straight to the ledger.
        with build_opener(ProxyHandler({}), _NoLedgerRedirects()).open(request, timeout=20) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            raw_error = exc.read(4096)
            error_text = raw_error.decode("utf-8")
            try:
                payload = json.loads(error_text)
            except ValueError:
                payload = None
            detail = payload.get("error", "") if isinstance(payload, dict) else error_text
        except (OSError, UnicodeDecodeError, AttributeError):
            detail = ""
        raise _LedgerHTTPError(exc.code, detail if isinstance(detail, str) else "") from exc
    except (OSError, URLError, UnicodeDecodeError, ValueError) as exc:
        raise SystemExit(f"could not reach the ledger: {_visible_untrusted(exc, 500)}") from exc
    if not isinstance(value, dict):
        raise SystemExit("ledger returned an invalid non-object response")
    return value


def _operator_worker_key() -> tuple[bytes, str]:
    from nakshatra_auth import load_worker_key

    try:
        return load_worker_key(Path.home() / ".nakshatra" / "keys" / "worker.ed25519")
    except (OSError, ValueError, PermissionError) as exc:
        raise SystemExit(f"cannot load the node worker key: {exc}") from exc


def _operator_hold_status(base: str, private: bytes, keyid: str, hold_id: str) -> dict:
    path = "/spend/task/status?" + urlencode({"hold_id": hold_id})
    value = _operator_ledger_request(base, private, keyid, "GET", path)
    required = {"hold_id", "task_hash", "payer_account", "worker_keyid", "amount", "state"}
    if (not required.issubset(value)
            or value.get("hold_id") != hold_id
            or not isinstance(value.get("task_hash"), str)
            or not isinstance(value.get("payer_account"), str)
            or not isinstance(value.get("worker_keyid"), str)
            or isinstance(value.get("amount"), bool)
            or not isinstance(value.get("amount"), int)
            or value["amount"] < 0
            or value.get("state") not in {"open", "disputed", "paid", "refunded"}):
        raise SystemExit("ledger returned an invalid task status")
    return value


def _short_id(value: object) -> str:
    safe = _visible_untrusted(value, 64)
    return safe if len(safe) <= 16 else safe[:16] + "…"


def _print_operator_hold(value: dict) -> None:
    reason = value.get("dispute_reason")
    suffix = f"  reason {_visible_untrusted(reason, 64)}" if reason else ""
    print(f"{_visible_untrusted(value['hold_id'], 128)}  {value['state']}  "
          f"{format_amount(value['amount'])}  payer {_short_id(value['payer_account'])}  "
          f"worker {_short_id(value['worker_keyid'])}  task {_short_id(value['task_hash'])}{suffix}")


def _operator_disputes(args, sock: Path) -> int:
    holds = list(args.hold)
    for task in args.task:
        local = call(sock, {"op": "escrow_status", "task": task})
        hold_id = local.get("hold_id")
        if not isinstance(hold_id, str) or not hold_id:
            raise SystemExit(f"task {_visible_untrusted(task, 72)} has no task hold")
        holds.append(hold_id)
    if not holds:
        raise SystemExit("nak operator disputes needs a task id or --hold HOLD_ID")
    base = _operator_ledger_base(args.ledger)
    private, keyid = _operator_worker_key()
    for hold_id in holds:
        try:
            _print_operator_hold(_operator_hold_status(base, private, keyid, hold_id))
        except _LedgerHTTPError as exc:
            raise SystemExit(str(exc)) from exc
    return 0


def _operator_resolve(args) -> int:
    summary = (f"Resolve ledger task hold {plain(args.hold_id, 72)} as {args.outcome} "
               f"for {json.dumps(plain(args.reason, 64), ensure_ascii=False)}")
    custody_act = broker_preflight("resolve", summary)
    if not sys.stdin.isatty():
        raise SystemExit("nak operator resolve is interactive-only; run it in a terminal")
    if not isinstance(args.reason, str) or not args.reason or len(args.reason) > 64:
        raise SystemExit("--reason must be non-empty text of at most 64 characters")
    if outbound_problem(args.reason):
        raise SystemExit("--reason is sent to the ledger; it looks like it carries a secret or "
                         "encoded blob — refusing")
    base = _operator_ledger_base(args.ledger)
    # Resolution is deliberately preflight-only: ledger requests use the worker/operator key,
    # never the person key. Keep the custody mode pinned through this human act nonetheless.
    custody_act.assert_current()
    private, keyid = _operator_worker_key()
    try:
        status = _operator_hold_status(base, private, keyid, args.hold_id)
    except _LedgerHTTPError as exc:
        raise SystemExit(str(exc)) from exc
    if status["state"] != "disputed":
        raise SystemExit(f"refused: task hold is {status['state']}, not disputed")

    prompt = (f"RESOLVE hold {_visible_untrusted(args.hold_id, 128)}  "
              f"amount {format_amount(status['amount'])}  "
              f"payer {_short_id(status['payer_account'])}  "
              f"worker {_short_id(status['worker_keyid'])}  "
              f"outcome {args.outcome.upper()}? [y/N] ")
    if input(prompt).strip().lower() not in {"y", "yes"}:
        print("task hold unchanged")
        return 1

    expected_state = "paid" if args.outcome == "pay" else "refunded"
    body = {"hold_id": args.hold_id, "outcome": args.outcome, "reason": args.reason}
    replayed = False
    custody_act.assert_current()
    try:
        result = _operator_ledger_request(
            base, private, keyid, "POST", "/spend/task/resolve", body)
    except _LedgerHTTPError as exc:
        if exc.status != 409:
            raise SystemExit(str(exc)) from exc
        # A lost successful response followed by the same human decision is safe to adopt.  A
        # genuinely conflicting 409 stays a refusal because its terminal state will not match.
        try:
            result = _operator_hold_status(base, private, keyid, args.hold_id)
        except _LedgerHTTPError:
            raise SystemExit(str(exc)) from exc
        replayed = result.get("state") == expected_state
        if not replayed:
            raise SystemExit(str(exc)) from exc
    if (result.get("hold_id") != args.hold_id or result.get("state") != expected_state
            or ("replayed" in result and not isinstance(result.get("replayed"), bool))):
        raise SystemExit("ledger returned an invalid task resolution response")
    replayed = replayed or bool(result.get("replayed", False))
    print(f"task hold {_visible_untrusted(args.hold_id, 128)} is {expected_state}"
          + (" (replayed)" if replayed else ""))
    return 0


def _operator_suspend(args, sock: Path, http_post, http_get) -> int:
    verb = "suspend" if args.operator_cmd == "suspend" else "unsuspend"
    reason = getattr(args, "reason", "")
    abandoning = bool(getattr(args, "abandon_pending", False))
    if abandoning and (args.target or reason or getattr(args, "force_unknown_node", False)):
        raise SystemExit("--abandon-pending cannot be combined with a target, --reason, or --force-unknown-node")
    if not abandoning and not args.target:
        raise SystemExit(f"nak operator {verb} requires a node or contact")
    summary = ("Abandon pending suspension update" if abandoning else
               f"{verb.title()} node/contact {plain(args.target, 72)}")
    if reason:
        summary += f" for {json.dumps(plain(reason, 120), ensure_ascii=False)}"
    custody_act = broker_preflight("suspend", summary)
    if not sys.stdin.isatty():
        raise SystemExit(f"nak operator {verb} is interactive-only; run it in a terminal")
    if not abandoning and verb == "suspend" and \
            (not isinstance(reason, str) or not reason or len(reason) > 120):
        raise SystemExit("--reason must be non-empty text of at most 120 characters")
    if reason and outbound_problem(reason):
        raise SystemExit("--reason is sent to the onboard server and every contact; "
                         "it looks like it carries a secret or encoded blob — refusing")

    status = call(sock, {"op": "status"})
    state = call(sock, {"op": "suspensions"})
    previous = state.get("own")
    pending = state.get("pending")
    from network.suspension import MAX_SEQ, verify_record
    if previous:
        try:
            previous = verify_record(previous, status["person"])
        except ValueError as exc:
            raise SystemExit(f"refused: installed suspension record is invalid: {exc}") from exc
    if pending:
        try:
            pending = verify_record(pending, status["person"])
        except ValueError as exc:
            raise SystemExit(f"refused: pending suspension record is invalid: {exc}") from exc
    from network.nakd import person_action_statement, suspension_record_proof_payload
    if abandoning:
        if pending is None:
            print("no pending suspension update")
            return 0
        answer = input(f"ABANDON pending operator record seq {pending['seq']}? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("pending suspension update unchanged")
            return 1
        person_key = PersonKey("suspend", summary, act=custody_act)
        if person_key.public_key().public_bytes_raw().hex() != status["person"]:
            raise SystemExit("the approved person key does not match nakd's person identity")
        abandon_ts = int(time.time())
        abandon_sig = person_key.sign(person_action_statement(
            "abandon-suspension", suspension_record_proof_payload(pending), abandon_ts)).hex()
        call(sock, {"op": "abandon_suspension", "record": pending,
                    "ts": abandon_ts, "signature": abandon_sig})
        print(f"abandoned pending operator record seq {pending['seq']}; server state was not changed")
        return 0

    node = _resolve_node_reference(
        sock, args.target, status, force_unknown=bool(getattr(args, "force_unknown_node", False)))
    nodes = set(previous["nodes"] if previous else [])
    reasons = dict(previous["reasons"] if previous else {})
    if verb == "suspend":
        nodes.add(node)
        reasons[node] = reason
    else:
        if node not in nodes:
            raise SystemExit("refused: that node is not currently suspended")
        nodes.remove(node)
        reasons.pop(node, None)
    seq = previous["seq"] + 1 if previous else 1
    if seq > MAX_SEQ:
        raise SystemExit(f"cannot create another suspension record: sequence limit {MAX_SEQ} reached")
    retrying = pending is not None
    if retrying and (pending["seq"] != seq or set(pending["nodes"]) != nodes
                     or pending["reasons"] != reasons):
        raise SystemExit(
            f"refused: suspension record seq {pending['seq']} is pending; re-run its original "
            "suspend/unsuspend command before starting a different change")
    action = "SUSPEND" if verb == "suspend" else "UNSUSPEND"
    retry_label = "RETRY " if retrying else ""
    answer = input(
        f"{retry_label}{action} node {node[:16]}… and publish operator record seq {seq}? [y/N] ")
    if answer.strip().lower() not in {"y", "yes"}:
        print("suspension state unchanged")
        return 1

    base = _operator_onboard_base()
    token = _onboard_token()
    if retrying:
        record = pending
        person_key = PersonKey("suspend", summary, act=custody_act)
        person_pub = person_key.public_key().public_bytes_raw().hex()
        if person_pub != status["person"]:
            raise SystemExit("the approved person key does not match nakd's person identity")
        install_ts = int(time.time())
        install_sig = person_key.sign(person_action_statement(
            "install-suspension", suspension_record_proof_payload(record), install_ts)).hex()
    else:
        from network.suspension import attach_signature, build_unsigned_record, suspension_statement
        person_key = PersonKey("suspend", summary, act=custody_act)
        person_pub = person_key.public_key().public_bytes_raw().hex()
        if person_pub != status["person"]:
            raise SystemExit("the approved person key does not match nakd's person identity")
        stage_ts = int(time.time())
        unsigned_record = build_unsigned_record(person_pub, seq, nodes, reasons, stage_ts)
        zero_payload = {"record_sha256": "0" * 64}
        stage_template = person_action_statement("stage-suspension", zero_payload, stage_ts)
        install_ts = stage_ts
        install_template = person_action_statement("install-suspension", zero_payload, install_ts)
        signed = person_sign_batch(
            "suspend",
            [suspension_statement(unsigned_record),
             ("ref", 0, stage_template),
             ("ref", 0, install_template)],
            summary=summary, act=custody_act)
        record = attach_signature(unsigned_record, signed[0][1])
        stage_sig = signed[1][1].hex()
        install_sig = signed[2][1].hex()
        # This write precedes the server request.  A timeout or lost response therefore leaves the
        # exact signed bytes available for an idempotent retry instead of minting seq+1 again.
        call(sock, {"op": "stage_suspension", "record": record,
                    "ts": stage_ts, "signature": stage_sig})

    try:
        response = http_post(base + "/operator/suspend", {"token": token, "record": record})
    except _OnboardHTTPError as exc:
        if exc.status in {400, 403, 404}:
            abandon_summary = f"Abandon rejected suspension update seq {record['seq']}"
            abandon_act = broker_preflight("suspend", abandon_summary)
            abandon_ts = int(time.time())
            abandon_sig = person_sign(
                "suspend",
                person_action_statement(
                    "abandon-suspension", suspension_record_proof_payload(record), abandon_ts),
                summary=abandon_summary, act=abandon_act).hex()
            call(sock, {"op": "abandon_suspension", "record": record,
                        "ts": abandon_ts, "signature": abandon_sig})
            raise SystemExit(f"{exc}; pending suspension update cleared; server state was unchanged")
        if exc.status != 409:
            raise
        remote = _server_suspension_record(base, token, http_post, http_get)
        try:
            remote = verify_record(remote, status["person"])
        except ValueError as verify_exc:
            raise SystemExit(
                f"refused: server suspension recovery record is invalid: {verify_exc}") from verify_exc
        local_seq = previous["seq"] if previous else 0
        if remote["seq"] <= local_seq:
            raise SystemExit("refused: server stale-sequence recovery did not return newer state")
        recovery_summary = f"Recover server suspension state at seq {remote['seq']}"
        recovery_act = broker_preflight("suspend", recovery_summary)
        recovery_ts = int(time.time())
        recovery_sig = person_sign(
            "suspend",
            person_action_statement(
                "recover-suspension", suspension_record_proof_payload(remote), recovery_ts),
            summary=recovery_summary, act=recovery_act).hex()
        recovered = call(sock, {"op": "recover_suspension", "record": remote,
                                "ts": recovery_ts, "signature": recovery_sig})
        raise SystemExit(
            f"server suspension state advanced to seq {recovered['seq']}; installed it locally; "
            "re-run the command with the new state")
    call(sock, {"op": "install_suspension", "record": record,
                "ts": install_ts, "signature": install_sig})
    changed = response.get("suspended", []) if verb == "suspend" else response.get("restored", [])
    print(f"operator record seq {seq} installed; {verb}ed {len(changed)} node(s)")
    return 0


class _NoAbbrevArgumentParser(argparse.ArgumentParser):
    """Keep every parser level strict; argparse otherwise abbreviates long options."""

    def __init__(self, *args, **kwargs):
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)


def main(argv=None, *, http_post=None, http_get=None) -> int:
    default_sock = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "nakd.sock"
    # ArgumentParser uses type(self) for every nested subparser, so the strict setting propagates
    # through gateway, wallet, worker, owner-lease, and task rather than stopping at the root.
    ap = _NoAbbrevArgumentParser(prog="nak", description=__doc__.split("\n")[0])
    ap.add_argument("--sock", type=Path, default=default_sock)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("grant", help="inspect or explicitly renew this node's agent grant")
    gsub = p.add_subparsers(dest="grant_cmd", required=True)
    gsub.add_parser("status", help="show the delegated agent, caps, expiry, and warning")
    q = gsub.add_parser("renew", help="human-only renewal that preserves the existing agent key")
    q.add_argument("--ttl", type=int, default=365, metavar="DAYS", help="charter lifetime in days (1-365)")
    q.add_argument("--idle-days", type=int, default=30, metavar="DAYS",
                   help="lapse after this many unused days (1-90; default 30)")
    q.add_argument("--lease-days", type=int, default=7, metavar="DAYS",
                   help="wire credential lifetime in days (1-7; default 7)")
    q.add_argument("--no-lease", action="store_true",
                   help="issue a legacy fixed-expiry grant instead of a sliding charter")
    q.add_argument("--caps", help="explicit comma-separated cap set; omitted preserves current caps")
    q.add_argument("--per-tx-cap", default=None, metavar="AMOUNT",
                   help="nak.pay limit per payment (for example '0.25 USDC' or '25000000 base')")
    q.add_argument("--day-cap", default=None, metavar="AMOUNT",
                   help="nak.pay daily limit (for example '1.5 USDC' or '150000000 base')")
    q.add_argument("--verdict", choices=["auto", "ask", "threshold"], default=None,
                   help="paid answer policy; omitted preserves it")
    q.add_argument("--verdict-auto-max", default=None, metavar="AMOUNT",
                   help="threshold auto-accept maximum in USDC or explicit base units; omitted preserves it")
    q.add_argument("--verdict-ask-strangers", choices=["yes", "no"], default=None,
                   help="always ask for unrostered/stranger workers; omitted preserves it")
    q.add_argument("--yes", action="store_true", help="confirm explicitly when stdin is not a terminal")
    p = sub.add_parser("revoke", help="human-only local and shared agent-grant revocation")
    p.add_argument("grant_id")
    p.add_argument("--grant-id", dest="more_grant_ids", action="append", default=[])
    p = sub.add_parser("revocations", help="inspect or move person-signed revocation records")
    rsub = p.add_subparsers(dest="revocations_cmd")
    rsub.add_parser("export", help="print this person's latest nkr1 record")
    q = rsub.add_parser("import", help="install an existing contact's nkr1 record")
    q.add_argument("blob")
    p = sub.add_parser("suspensions", help="inspect or move person-signed suspension records")
    ssub = p.add_subparsers(dest="suspensions_cmd")
    ssub.add_parser("export", help="print this operator's latest nks1 record")
    q = ssub.add_parser("import", help="install an existing contact's nks1 record")
    q.add_argument("blob")
    p = sub.add_parser("report", help="report a node or contact to the network operator")
    p.add_argument("subject")
    p.add_argument("--reason", required=True,
                   choices=["spam", "fraud", "abuse", "bad-output", "other"])
    p.add_argument("--note", default="")
    p.add_argument("--task", default="", help="related 64-hex task hash")
    p.add_argument("--mid", default="", help="related message id")
    p = sub.add_parser("operator", help="review reports and publish suspension state")
    osub = p.add_subparsers(dest="operator_cmd", required=True)
    q = osub.add_parser("reports", help="show recent member reports")
    q.add_argument("--since", default="7d")
    q.add_argument("--json", action="store_true")
    q = osub.add_parser("disputes", help="inspect named task holds on the ledger")
    q.add_argument("task", nargs="*", help="local task id whose hold should be inspected")
    q.add_argument("--hold", action="append", default=[], metavar="HOLD_ID",
                   help="ledger hold id to inspect (repeatable)")
    q.add_argument("--ledger", default=DEFAULT_OPERATOR_LEDGER,
                   help=f"ledger origin (default: {DEFAULT_OPERATOR_LEDGER})")
    q = osub.add_parser("resolve", help="human-only adjudication of one disputed task hold")
    q.add_argument("hold_id")
    q.add_argument("outcome", choices=["pay", "refund"])
    q.add_argument("--reason", required=True)
    q.add_argument("--ledger", default=DEFAULT_OPERATOR_LEDGER,
                   help=f"ledger origin (default: {DEFAULT_OPERATOR_LEDGER})")
    q = osub.add_parser("suspend", help="suspend one rostered node")
    q.add_argument("target", nargs="?")
    q.add_argument("--reason", default="")
    q.add_argument("--force-unknown-node", action="store_true")
    q.add_argument("--abandon-pending", action="store_true",
                   help="person-authorized removal of a staged update the server did not apply")
    q = osub.add_parser("unsuspend", help="restore one suspended node's prior tier")
    q.add_argument("target")
    q.add_argument("--force-unknown-node", action="store_true")
    p = sub.add_parser("operator-trust", help="trust a contact's signed suspension records")
    p.add_argument("contact")
    p = sub.add_parser("operator-untrust", help="stop trusting a contact's suspension records")
    p.add_argument("contact")
    p = sub.add_parser("invite"); p.add_argument(
        "--person-key", type=Path, default=None,
        help="explicit file key in file custody only (default: ~/.sthambha/person-TEST.key)")
    p.add_argument("--ttl", type=int, default=86400); p.add_argument("--note", default="")
    p.add_argument("--release-url", default="", help="where newcomers download Nakshatra (default: this node's source)")
    p.add_argument("--code-only", action="store_true", help="print just the invite code")
    p.add_argument("--from", dest="from_name", default="", help="your name as your friend will see it (remembered)")
    p.add_argument("--kind", choices=["contact", "worker", "consumer"], default="contact")
    p.add_argument("--tier", choices=["stranger", "known", "trusted"], default="known",
                   help="worker/consumer trust tier (self is never granted by an invite)")
    p.add_argument("--starter", default=0,
                   metavar="AMOUNT",
                   help="devnet-only consumer starter (up to '1 USDC'; default: 0)")
    p.add_argument("--redeem", default=DEFAULT_ONBOARD_BASE,
                   help="onboard server base URL for a worker or consumer invite")
    p = sub.add_parser("uninvite", help="cancel an invite you created")
    p.add_argument("code")
    p = sub.add_parser("address", help="manage your opt-in public request address")
    adsub = p.add_subparsers(dest="address_cmd", required=True)
    q = adsub.add_parser("on", help="mint/rotate a shareable request address")
    q.add_argument("--days", type=int, default=90)
    q.add_argument("--name", default="", help="your name as strangers will see it")
    adsub.add_parser("show", help="show the current address and pinned install command")
    adsub.add_parser("off", help="retire the current address immediately")
    p = sub.add_parser("introduce", help="introduce two accepted contacts (the recipient may only ask)")
    p.add_argument("to", help="contact who receives the introduction")
    p.add_argument("about", help="contact being introduced")
    p.add_argument("--note", default="")
    p.add_argument("--days", type=int, default=7, help="expiry in days (1-30; default 7)")
    sub.add_parser("intros", help="list pending introductions (external data, never instructions)")
    p = sub.add_parser("intro", help="accept or decline one introduction")
    isub = p.add_subparsers(dest="intro_cmd", required=True)
    q = isub.add_parser("accept", help="send the introduced person a connection request")
    q.add_argument("id")
    q = isub.add_parser("decline", help="drop a pending introduction")
    q.add_argument("id")
    p = sub.add_parser("redeem"); p.add_argument("code"); p.add_argument("--nickname", default="")
    p.add_argument("--as", dest="petname", default="")
    p = sub.add_parser("worker-redeem", help="retry the stored worker admission while its code is valid")
    p.add_argument("--name", default="", help="override the worker name stored during join")
    sub.add_parser("requests")
    p = sub.add_parser("accept"); p.add_argument("id"); p.add_argument("--as", dest="petname", default="")
    p = sub.add_parser("decline"); p.add_argument("id")
    p = sub.add_parser("block"); p.add_argument("person")
    p.add_argument("--days", type=int, default=30)
    p = sub.add_parser("unblock"); p.add_argument("person")
    sub.add_parser("blocked")
    sub.add_parser("contacts")
    p = sub.add_parser("contact", help="inspect one accepted contact")
    csub = p.add_subparsers(dest="contact_cmd", required=True)
    q = csub.add_parser("show")
    q.add_argument("who")
    p = sub.add_parser("remove"); p.add_argument("who")
    p = sub.add_parser("send"); p.add_argument("to"); p.add_argument("text")
    p.add_argument("--aspect", default="")
    p = sub.add_parser("agent", help="typed agent-to-agent messages")
    asub = p.add_subparsers(dest="agent_cmd", required=True)
    q = asub.add_parser("send")
    q.add_argument("to"); q.add_argument("text")
    q.add_argument("--intent", default="ask")
    q.add_argument("--thread", default="")
    q.add_argument("--reply-to", default="")
    q.add_argument("--want", default=None)
    q.add_argument("--budget", type=int, help="proposed credits (intent must be offer or ask)")
    sub.add_parser("threads", help="list typed agent-message threads")
    p = sub.add_parser("thread", help="show one typed agent-message thread")
    p.add_argument("thread")
    p.add_argument("--contact", default="", help="contact name or person-key prefix when ids collide")
    p = sub.add_parser("card", help="show or publish the person's outward-facing agent card")
    csub = p.add_subparsers(dest="card_cmd", required=True)
    q = csub.add_parser("show"); q.add_argument("contact", nargs="?", default="")
    q = csub.add_parser("set")
    q.add_argument("--name", required=True); q.add_argument("--offers", required=True)
    q.add_argument("--intents", default="", help="comma-separated agent.msg intents (default: all)")
    q.add_argument("--models", default="", help="comma-separated model names")
    q.add_argument("--price-hint", default="")
    q.add_argument("--paid", action="store_true",
                   help="advertise paid work using this node's ledger key (Neuron escrow only)")
    p = sub.add_parser("inbox"); p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("outbox", help="show recent queued, delivered, and refused outbound items")
    p.add_argument("--refused", action="store_true", help="show only refused items")
    p.add_argument("--text", action="store_true", help="also show message/result text")
    p.add_argument("--limit", type=int, default=50)
    sub.add_parser("ledger", help="escrow journal: open / released / refunded")
    p = sub.add_parser("escrow", help="inspect task escrow")
    esub = p.add_subparsers(dest="escrow_cmd", required=True)
    q = esub.add_parser("status"); q.add_argument("task")
    p = sub.add_parser("wallet", help="deposit and withdraw USDC-backed inference credits")
    walsub = p.add_subparsers(dest="wallet_cmd", required=True)
    walsub.add_parser("address", help="show or queue this operator's deposit token account")
    walsub.add_parser("status", help="show balance, cashable credits, and pending withdrawals")
    q = walsub.add_parser("bind", help="bind the Solana address that receives withdrawals")
    q.add_argument("address")
    q = walsub.add_parser("withdraw", help="request USDC to the bound payout address")
    q.add_argument("usdc", help="USDC amount (at least 1, up to 6 decimal places)")
    p = sub.add_parser("pay", help="inspect or export this device's non-custodial Solana wallet")
    paysub = p.add_subparsers(dest="pay_cmd", required=True)
    paysub.add_parser("address", help="print the Solana pay address")
    paysub.add_parser("status", help="show the address, custody mode, and optional devnet SOL balance")
    paysub.add_parser("ensure", help="idempotently initialize escrow and renew this node's delegate")
    q = paysub.add_parser("limits", help="save hard/soft USDC limits and run ensure")
    q.add_argument("--cap", required=True, help="on-chain lifetime delegate cap in USDC")
    q.add_argument("--days", required=True, type=int, help="delegate lifetime in days (1-3650)")
    q.add_argument("--per-tx", required=True, help="local per-transaction cap in USDC")
    q.add_argument("--per-day", required=True, help="local daily cap in USDC")
    q = paysub.add_parser("deposit", help="deposit USDC from the wallet's associated token account")
    q.add_argument("--amount", required=True, help="USDC amount")
    q = paysub.add_parser("withdraw", help="request a withdrawal, or execute it after six hours")
    withdraw = q.add_mutually_exclusive_group(required=True)
    withdraw.add_argument("--amount", help="USDC amount to request")
    withdraw.add_argument("--execute", action="store_true", help="execute the pending withdrawal")
    paysub.add_parser("channels", help="list the local payer channel book with on-chain state")
    q = paysub.add_parser("backup", help="interactively print or acknowledge a BIP39 seed backup")
    q.add_argument("--confirm", action="store_true",
                   help="record that the seed phrase has been backed up without printing it")
    q = paysub.add_parser("export", help="interactively export the Phantom/Solflare keypair")
    q.add_argument("--keypair", action="store_true", required=True,
                   help="print the 64-byte seed + public key as a JSON integer array")
    sub.add_parser("balance", help="show this operator's signed ledger balance in USDC and base units")
    p = sub.add_parser("direct", help="allow/stop direct connections with a contact (reveals your LAN/IPv6)")
    p.add_argument("who"); p.add_argument("onoff", choices=["on", "off"])
    p = sub.add_parser("direct-listen", help="accept direct connections on this port (or 'off'); restarts nak-net")
    p.add_argument("port")
    p = sub.add_parser("p2p", help="use the local libp2p sidecar for both-NAT direct paths; restarts nak-net")
    p.add_argument("onoff", choices=["on", "off"])
    p.add_argument("--socket", type=Path, help="sidecar UNIX socket (default: $XDG_RUNTIME_DIR/nakshatra/p2p.sock)")
    p = sub.add_parser("gateway", help="operate this node's local OpenAI consumer gateway")
    gsub = p.add_subparsers(dest="gateway_cmd", required=True)
    q = gsub.add_parser("on", help="configure and enable the fail-closed consumer gateway")
    q.add_argument("--port", type=int, default=11600)
    q.add_argument("--bind", default="127.0.0.1")
    q.add_argument("--allow-lan", action="store_true",
                   help="acknowledge that a non-loopback bind exposes inference to the LAN")
    q.add_argument("--pillar-url", required=True)
    q.add_argument("--pillar-spki", required=True,
                   help="TLS SPKI SHA-256 pin for the public VPS front")
    q.add_argument("--ledger-url", required=True,
                   help="public ledger base on the same pinned VPS front")
    q.add_argument("--models", default="auto",
                   help="auto (signed pillar discovery) or a local serve_models YAML")
    q.add_argument("--package-url",
                   help="signed model-package base URL (required for auto unless join state has it)")
    q.add_argument("--package-pubkey",
                   help="optionally narrow the signed release's package signer allowlist (64 hex)")
    gsub.add_parser("off", help="disable the gateway and remove its opt-in")
    gsub.add_parser("status", help="show configuration and unit state without key material")
    q = gsub.add_parser("paid", help="atomically control payment-bound gateway routing")
    gpsub = q.add_subparsers(dest="gateway_paid_cmd", required=True)
    gpsub.add_parser("on", help="preflight and enable payment-bound routing")
    gpsub.add_parser("off", help="disable payment-bound routing before restarting")
    gpsub.add_parser("status", help="show the payment-bound routing gate")
    p = sub.add_parser("worker", help="build and operate this node's signed-release inference worker")
    wsub = p.add_subparsers(dest="worker_cmd", required=True)
    q = wsub.add_parser("preflight", help="check hardware and runtime against signed model packages")
    q.add_argument("--json", action="store_true", help="print the machine-readable report")
    wsub.add_parser("build", help="compile the current release's verified engine for this GPU")
    q = wsub.add_parser("on", help="configure (once) and enable the worker service")
    q.add_argument("--model-id")
    q.add_argument("--package-url")
    q.add_argument("--package-pubkey",
                   help="Ed25519 public key (hex) of the model package signer; remote packages must be signed by it")
    q.add_argument("--pillar-url")
    q.add_argument("--pillar-spki",
                   help="pillar TLS public-key SHA-256 pin (required for https://)")
    q.add_argument("--trusted-pillar-nets", default="10.42.0.0/24",
                   help="comma-separated encrypted-overlay CIDRs allowed for http:// pillars")
    q.add_argument("--listen-address", default="[::]",
                   help="gRPC bind address (default: [::]; use 127.0.0.1 for sidecar-only access)")
    q.add_argument("--layers", help="half-open layer range a-b (for example 0-16)")
    q.add_argument("--port", type=int)
    q.add_argument("--idle-unload-s", type=float, default=600.0,
                   help="seconds before idle-backstop unload; 0 disables this backstop")
    q.add_argument("--release-grace-s", type=float, default=30.0,
                   help="seconds after session release before unload; 0 is immediate, negative disables")
    q.add_argument("--warm-directives", choices=["on", "off"], default="on",
                   help="honor pillar warm-pool directives (default: on)")
    # worker.py cannot derive its pipeline role from a layer range alone because the CLI does not
    # know the model's total layer count. Preserve the proven outsider-worker role as the default,
    # while exposing the worker.py choice for last/middle/whole-model assignments.
    q.add_argument("--mode", choices=["first", "middle", "last", "solo"], default="first")
    wsub.add_parser("off", help="disable the worker service and remove its worker.env opt-in")
    wsub.add_parser("status", help="show the verified build record and systemd unit state")
    q = wsub.add_parser("paid", help="atomically control payment-bound worker admission")
    wpsub = q.add_subparsers(dest="worker_paid_cmd", required=True)
    q = wpsub.add_parser("on", help="preflight and enable payment-bound admission")
    q.add_argument("--ledger-url", required=True,
                   help="HTTPS spend API base URL")
    q.add_argument("--ledger-spki", required=True,
                   help="TLS SPKI SHA-256 pin for the spend front")
    q.add_argument("--store", type=Path,
                   help="durable SQLite store inside ~/.nakshatra/worker "
                        "(default: paid.sqlite)")
    wpsub.add_parser("off", help="disable consumer resolution and paid admission")
    wpsub.add_parser("status", help="show paid admission state without the TLS pin")
    q = wsub.add_parser("relay", help="allow or refuse capped relay activation streams")
    q.add_argument("relay_cmd", choices=["on", "off"])
    q = wsub.add_parser("owner-lease", help="control the local owner-sentinel lease")
    lsub = q.add_subparsers(dest="owner_lease_cmd", required=True)
    q = lsub.add_parser("on", help="create the private lease token and enable default-deny")
    q.add_argument("--port", type=int, default=5571)
    q.add_argument("--ttl-s", type=float, default=6.0)
    q.add_argument("--resume-s", type=float, default=30.0)
    lsub.add_parser("off", help="remove the lease configuration")
    lsub.add_parser("status", help="show lease configuration without revealing its token")
    p = sub.add_parser("task", help="post / list / claim / submit / decide tasks")
    tsub = p.add_subparsers(dest="tcmd", required=True)
    q = tsub.add_parser("post"); q.add_argument("to"); q.add_argument("title"); q.add_argument("instructions")
    q.add_argument("--rule", action="append", default=[], help='JSON rule, e.g. \'{"max_words": 60}\' (repeatable)')
    q.add_argument("--reward", default=0, metavar="AMOUNT",
                   help="devnet reward in USDC or explicit base units (default: 0)")
    q.add_argument("--deadline", type=int, default=3600)
    q = tsub.add_parser("list"); q.add_argument("--role", default="")
    q = tsub.add_parser("claim"); q.add_argument("task")
    q = tsub.add_parser("submit"); q.add_argument("task"); q.add_argument("output")
    q = tsub.add_parser(
        "decide", help="human decision for a paid answer awaiting approval",
        description="Human-only paid-answer decision. A task keeps the verdict policy it was posted under.")
    q.add_argument("task"); q.add_argument("decision", choices=["accept", "reject"])
    q.add_argument("--reason", default="")
    q = sub.add_parser("tasks", help="list tasks (alias for `nak task list`)")
    q.add_argument("--role", default="")
    a = ap.parse_args(argv)
    s = a.sock

    if a.cmd == "grant":
        if a.grant_cmd == "renew":
            return _grant_renew(a)
        _print_grant_status(call(s, {"op": "status"})["grant"])
    elif a.cmd == "revoke":
        return _revoke(a, s)
    elif a.cmd == "revocations":
        if a.revocations_cmd == "export":
            from network.revocation import encode_record
            own = call(s, {"op": "revocations"}).get("own")
            if not own:
                raise SystemExit("no revocation record has been created")
            print(encode_record(own))
        elif a.revocations_cmd == "import":
            result = call(s, {"op": "import_revocation", "blob": a.blob})
            print(f"imported seq {result['seq']} with {result['count']} revoked grant(s)"
                  + ("" if result["new"] else " (already known)"))
        else:
            result = call(s, {"op": "revocations"})
            own = result.get("own")
            print(f"own seq {own['seq'] if own else 0}: " +
                  (", ".join(own["grant_ids"]) if own and own["grant_ids"] else "no revoked grants"))
            contacts = {row["person"]: row for row in call(s, {"op": "contacts"})["contacts"]}
            people = sorted(set(contacts) | set(result.get("received", {})) | set(result.get("sent", {})))
            for person in people:
                row = result.get("received", {}).get(person, {})
                contact = contacts.get(person, {})
                print(f"{contact.get('petname') or person[:16]}: received seq {row.get('seq', 0)} "
                      f"({row.get('count', 0)} grant(s)); own acked seq "
                      f"{result.get('sent', {}).get(person, 0)}")
    elif a.cmd == "suspensions":
        if a.suspensions_cmd == "export":
            from network.suspension import encode_record
            own = call(s, {"op": "suspensions"}).get("own")
            if not own:
                raise SystemExit("no suspension record has been created")
            print(encode_record(own))
        elif a.suspensions_cmd == "import":
            summary = "Import a trusted operator suspension record"
            custody_act = broker_preflight("suspend", summary)
            if not sys.stdin.isatty():
                raise SystemExit("nak suspensions import is interactive-only; run it in a terminal")
            from network.nakd import person_action_statement, suspension_record_proof_payload
            from network.suspension import decode_record
            try:
                record = decode_record(a.blob)
            except ValueError as exc:
                raise SystemExit(f"refused: invalid suspension record: {exc}") from exc
            answer = input(f"IMPORT suspension record seq {record['seq']} from "
                           f"{record['person'][:16]}…? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                print("suspension state unchanged")
                return 1
            person_key = PersonKey("suspend", summary, act=custody_act)
            proof_ts = int(time.time())
            proof_sig = person_key.sign(person_action_statement(
                "import-suspension", suspension_record_proof_payload(record), proof_ts)).hex()
            result = call(s, {"op": "import_suspension", "blob": a.blob,
                              "ts": proof_ts, "signature": proof_sig})
            print(f"imported seq {result['seq']} with {result['count']} suspended node(s)" +
                  ("" if result["new"] else " (already known)"))
        else:
            result = call(s, {"op": "suspensions"})
            own = result.get("own")
            print(f"own seq {own['seq'] if own else 0}: " +
                  (", ".join(own["nodes"]) if own and own["nodes"] else "no suspended nodes"))
            pending = result.get("pending")
            if pending:
                print(f"pending server confirmation seq {pending['seq']}: " +
                      (", ".join(pending["nodes"]) if pending["nodes"] else
                       "no suspended nodes"))
            trusted = set(result.get("operators", []))
            for person, row in sorted(result.get("received", {}).items()):
                label = "trusted operator" if person in trusted else "untrusted contact"
                print(f"{person[:16]}…: received seq {row['seq']} ({row['count']} node(s), {label})")
    elif a.cmd == "report":
        if len(a.note) > 500:
            raise SystemExit("--note must be at most 500 characters")
        evidence = {key: value for key, value in (("task_hash", a.task), ("mid", a.mid)) if value}
        summary = f"Report node/contact {plain(a.subject, 72)} for {a.reason}"
        custody_act = broker_preflight("report", summary)
        if not sys.stdin.isatty():
            raise SystemExit("nak report is interactive-only; run it in a terminal")
        # Preserve the established file-custody bytes exactly: file mode signs the operator's
        # typed subject and lets nakd resolve it as it always has.  Broker mode must present the
        # actual node key to the approval parser, so resolve it from nakd's local encrypted store
        # without adding a daemon operation that older nakd versions do not understand.
        subject = (_broker_report_subject(a.subject)
                   if getattr(custody_act, "mode", "file") == "broker" else a.subject)
        from network.nakd import person_action_statement, report_proof_payload
        person_key = PersonKey("report", summary, act=custody_act)
        proof_ts = int(time.time())
        proof_sig = person_key.sign(person_action_statement(
            "report-cli", report_proof_payload(subject, a.reason, a.note, evidence),
            proof_ts)).hex()
        result = call(s, {"op": "report_cli", "subject": subject, "reason": a.reason,
                          "note": a.note, "evidence": evidence, "ts": proof_ts,
                          "signature": proof_sig})
        print(f"report {result.get('status', 'recorded')} for node {result['subject'][:16]}…; "
              "reports are claims for operator review and do not suspend nodes")
    elif a.cmd == "operator":
        http_post = http_post or _http_post_json
        http_get = http_get or _http_get_json
        if a.operator_cmd in {"suspend", "unsuspend"}:
            return _operator_suspend(a, s, http_post, http_get)
        if a.operator_cmd == "disputes":
            return _operator_disputes(a, s)
        if a.operator_cmd == "resolve":
            return _operator_resolve(a)
        if not re.fullmatch(r"[1-9][0-9]*[smhdw]", a.since or ""):
            raise SystemExit("--since must look like 30m, 24h, or 7d")
        result = http_post(_operator_onboard_base() + "/operator/reports",
                           {"token": _onboard_token(), "since": a.since})
        reports = result.get("reports")
        if not isinstance(reports, list):
            raise SystemExit("onboard server returned an invalid reports response")
        if a.json:
            print(json.dumps({"trust": "UNTRUSTED external report data",
                              "reports": _sanitized_untrusted_json(reports)},
                             indent=1, sort_keys=True))
        else:
            for report in reports:
                if not isinstance(report, dict):
                    continue
                label = " [UNTRUSTED MCP-FILED CLAIM]" if report.get("via") == "mcp" else ""
                note = (f" — {_visible_untrusted(report.get('note'), 500)}"
                        if report.get("note") else "")
                print(f"UNTRUSTED REPORT {_visible_untrusted(report.get('received', '?'), 40)} "
                      f"{_visible_untrusted(report.get('reporter', ''), 64)[:16]}… -> "
                      f"{_visible_untrusted(report.get('subject', ''), 64)[:16]}… "
                      f"{_visible_untrusted(report.get('reason', '?'), 32)}{label}{note}")
    elif a.cmd in {"operator-trust", "operator-untrust"}:
        trusting = a.cmd == "operator-trust"
        action = "Trust" if trusting else "Stop trusting"
        summary = f"{action} contact {plain(a.contact, 72)} as a suspension operator"
        custody_act = broker_preflight("suspend", summary)
        if not sys.stdin.isatty():
            raise SystemExit(f"nak {a.cmd} is interactive-only; run it in a terminal")
        contact = _resolve_contact_reference(s, a.contact)
        prompt = "TRUST" if trusting else "STOP TRUSTING"
        answer = input(f"{prompt} {plain(a.contact, 72)} as a suspension operator? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("operator trust unchanged")
            return 1
        from network.nakd import person_action_statement
        person_key = PersonKey("suspend", summary, act=custody_act)
        proof_ts = int(time.time())
        proof_kind = "operator-trust" if trusting else "operator-untrust"
        proof_sig = person_key.sign(person_action_statement(
            proof_kind, {"person": contact["person"]}, proof_ts)).hex()
        op = "operator_trust" if trusting else "operator_untrust"
        result = call(s, {"op": op, "contact": contact["person"],
                          "ts": proof_ts, "signature": proof_sig})
        past = "trusted" if trusting else "stopped trusting"
        print(f"{past} suspension operator [UNTRUSTED CONTACT LABEL] "
              f"{_visible_untrusted(result['contact'], 72)} ({result['person'][:16]}…)")
    elif a.cmd == "pay":
        _pay_command(a)
    elif a.cmd == "wallet":
        _wallet_command(a)
    elif a.cmd == "balance":
        _balance_command()
    elif a.cmd == "gateway":
        import subprocess as _sp
        if a.gateway_cmd == "on":
            _gateway_on(_sp, a)
        elif a.gateway_cmd == "off":
            _gateway_off(_sp)
        elif a.gateway_cmd == "paid":
            result = _gateway_paid(_sp, a.gateway_paid_cmd)
            if result is not None:
                print(json.dumps(result, indent=1, sort_keys=True))
        else:
            print(json.dumps(_gateway_status(_sp), indent=1, sort_keys=True))
    elif a.cmd == "worker":
        import subprocess as _sp
        if a.worker_cmd == "preflight":
            result = _worker_preflight(_sp)
            if a.json:
                print(json.dumps(result, indent=1, sort_keys=True))
            else:
                from worker_preflight import render_plain
                print(render_plain(result))
        elif a.worker_cmd == "build":
            _worker_build(_sp)
        elif a.worker_cmd == "on":
            _worker_enable(_sp, a)
        elif a.worker_cmd == "off":
            _worker_disable(_sp)
        elif a.worker_cmd == "status":
            print(json.dumps(_worker_status(_sp), indent=1, sort_keys=True))
        elif a.worker_cmd == "relay":
            _worker_relay(_sp, a.relay_cmd == "on")
        elif a.worker_cmd == "paid":
            result = _worker_paid(_sp, a.worker_paid_cmd, a)
            if result is not None:
                print(json.dumps(result, indent=1, sort_keys=True))
        elif a.owner_lease_cmd == "on":
            _owner_lease_on(_sp, a)
        elif a.owner_lease_cmd == "off":
            _owner_lease_off(_sp)
        else:
            print(json.dumps(_owner_lease_status(), indent=1, sort_keys=True))
    elif a.cmd == "status":
        r = call(s, {"op": "status"})
        own_custody = {"file": "file on disk", "broker-biometric": "phone approval"}.get(
            r.get("key_custody"), r.get("key_custody", "unknown"))
        print(f"person {r['person'][:16]}…  node {r['node'][:16]}…  "
              f"key custody {own_custody}  relay {r['relay']}")
        print(f"{len(r['contacts'])} contacts, {r['pending_requests']} pending requests, {r['open_invites']} open invites")
        waiting = int(r.get("awaiting_approval") or 0)
        if waiting:
            noun = "answer" if waiting == 1 else "answers"
            print(f"{waiting} paid {noun} waiting for your decision (each keeps its posting-time policy)")
        paid_starters = int(r.get("starter_paid_count") or 0)
        if paid_starters:
            print(f"starter credits: {paid_starters} paid")
        for starter in r.get("starters") or []:
            if starter.get("state") == "paid":
                continue
            reason = f": {starter.get('reason')}" if starter.get("reason") else ""
            amount = starter.get("amount")
            shown_amount = (f" {format_amount(amount)} (devnet spend-only);"
                            if isinstance(amount, int) and not isinstance(amount, bool) else "")
            print(f"starter {starter.get('invite_id')} to {starter.get('petname')}: "
                  f"{shown_amount} {starter.get('state')}{reason}")
        print(f"direct listener {r.get('direct_port') or 'off'}  p2p "
              f"{(r.get('p2p') or {}).get('dial', 'off')}")
        g = r["grant"]
        _print_grant_status(g)
        print(f"refused outbox items: {r['outbox_refused']}")
        if r.get("newest_refusal"):
            q = r["newest_refusal"]
            print(f"newest refusal [{_ago(q['time'])}] {q['id']} to {q['to'][:12]}: {q['reason']}")
    elif a.cmd == "invite":
        import joincode
        try:
            a.starter = (_cli_amount(a.starter, max_base_units=100_000_000)
                         if isinstance(a.starter, str) else a.starter)
        except SystemExit as exc:
            raise SystemExit(f"--starter {exc}") from exc
        if a.kind != "consumer" and a.starter:
            raise SystemExit("--starter is only valid with --kind consumer")
        name_file = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "name"
        from_name = a.from_name or (name_file.read_text().strip() if name_file.exists() else "")
        summary_parts = [f"Create {a.kind} invite"]
        if a.note:
            summary_parts.append(f"with your note {json.dumps(plain(a.note, 80), ensure_ascii=False)}")
        if from_name:
            name_source = "the name you supplied" if a.from_name else "your saved name"
            summary_parts.append(
                f"using {name_source} {json.dumps(plain(from_name[:40], 40), ensure_ascii=False)}")
        invite_summary = "; ".join(summary_parts)
        custody_act = broker_preflight("invite", invite_summary)
        if a.person_key is not None and custody_act.mode == "broker":
            raise SystemExit("person custody is broker; --person-key is not allowed")
        http_post = http_post or _http_post_json
        st = call(s, {"op": "status"})
        invite_key_path = (a.person_key or _person_key_path()) if custody_act.mode == "file" else None
        priv = PersonKey("invite", invite_summary, key_path=invite_key_path, act=custody_act)
        release, line = _release_for_invite(a.release_url)
        if not from_name:
            print("tip: add --from <your name> so your friend sees who invited them (remembered after once)",
                  file=sys.stderr)
        issued_at = int(time.time())
        worker = consumer = staged = None
        code = None
        register_attempted = False
        try:
            if a.kind == "worker":
                auth = _stage_auth(
                    a.kind, a.tier, a.starter, a.ttl,
                    lambda body: call(s, {"op": "node_sign_stage",
                                          "body": {**body, "inviter_pubkey": st["node"]}}))
                worker, staged_expiry = _stage_worker(
                    a.redeem, a.tier, auth, http_post, now=issued_at)
                staged = worker
            elif a.kind == "consumer":
                auth = _stage_auth(
                    a.kind, a.tier, a.starter, a.ttl,
                    lambda body: call(s, {"op": "node_sign_stage",
                                          "body": {**body, "inviter_pubkey": st["node"]}}))
                consumer, staged_expiry = _stage_consumer(
                    a.redeem, a.tier, a.starter, st["node"], auth, http_post,
                    now=issued_at)
                staged = consumer
            else:
                staged_expiry = issued_at + a.ttl
            expires_at = min(issued_at + a.ttl, staged_expiry)
            code = joincode.encode_invite(
                priv, st["node"], [{"node": st["node"], "addrs": [st["relay"]]}],
                ttl_s=expires_at - issued_at, now=issued_at, note=a.note,
                release=release, from_name=from_name,
                kind=a.kind, worker=worker, consumer=consumer)
            register_attempted = True
            call(s, {"op": "register_invite", "code": code})
            if a.from_name:
                _write_private(name_file, (a.from_name.strip()[:40] + "\n").encode())
        except BaseException as original:
            cleanup_failures = []
            if register_attempted and code is not None:
                try:
                    _cancel_local_invite(s, code)
                except BaseException as cancel_exc:
                    cleanup_failures.append(f"local {type(cancel_exc).__name__}")
            if staged is not None:
                try:
                    _cancel_staged_invite(s, staged, st["node"], http_post)
                except BaseException as cancel_exc:
                    cleanup_failures.append(f"server {type(cancel_exc).__name__}")
            if cleanup_failures:
                raise SystemExit(
                    "invite did not complete and cleanup could not be confirmed; manual "
                    f"cancellation is required ({', '.join(cleanup_failures)})") from original
            raise
        if a.kind == "consumer":
            print(f"devnet starter: {format_amount(a.starter)} (spend-only)", file=sys.stderr)
        if a.code_only or not line:
            print(code)
            if a.kind in {"worker", "consumer"}:
                print(f"{a.kind} invite expires at " +
                      f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(expires_at))} "
                      f"(Unix {expires_at})", file=sys.stderr)
        else:
            expiry = (f"expires at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(expires_at))} "
                      f"(Unix {expires_at})" if a.kind in {"worker", "consumer"}
                      else f"expires in {a.ttl // 3600}h")
            print(f"Send your friend this ONE line (single use, {expiry}; it is a secret — only them):\n",
                  file=sys.stderr)
            print(line.replace("{CODE}", code))
            print("\nThey paste it in a terminal (Linux or WSL with systemd). When their request arrives:"
                  "  nak requests  →  nak accept <id> --as <name>", file=sys.stderr)
    elif a.cmd == "uninvite":
        import joincode
        inv = joincode.decode_invite(a.code, now=0)
        st = call(s, {"op": "status"})
        if inv.get("inviter") != st.get("person") or inv.get("inviter_node") != st.get("node"):
            raise SystemExit("that invite was not made by this person and node")
        if inv.get("kind") in {"worker", "consumer"}:
            details = inv[inv["kind"]]
            token = _onboard_token(required=False)
            if token is not None:
                cancel_body = {"token": token, "code": details["code"]}
            else:
                unsigned = {"v": 2, "purpose": ONBOARD_CANCEL_PURPOSE,
                            "code": details["code"],
                            "inviter_pubkey": st["node"], "ts": int(time.time())}
                cancel_body = call(s, {"op": "node_sign_stage", "body": unsigned})
                cancel_body.pop("ok", None)
            (http_post or _http_post_json)(
                details["redeem"].rstrip("/") + "/invite-cancel", cancel_body)
        result = call(s, {"op": "cancel_invite", "nonce": inv["nonce"]})
        if not result.get("cancelled"):
            raise SystemExit("invite is not open locally")
        print("invite cancelled")
    elif a.cmd == "address":
        if a.address_cmd == "off":
            row = call(s, {"op": "address_status"})
            if not row.get("enabled"):
                print("request address is already off")
                return 0
            address_summary = f"Turn request address off {row['aid'][:8]}…"
            custody_act = broker_preflight("address-control", address_summary)
            from network.nakd import address_control_statement
            ts = int(time.time())
            signature = PersonKey(
                "address-control", address_summary, act=custody_act).sign(
                address_control_statement("address_off", row["aid"], ts)).hex()
            call(s, {"op": "address_off", "aid": row["aid"],
                     "ts": ts, "signature": signature})
            print("request address off; the old address is dead immediately")
        elif a.address_cmd == "show":
            row = call(s, {"op": "address_status"})
            if not row.get("enabled"):
                print("request address is off")
            else:
                print(row["code"])
                if row.get("install_line"):
                    print(row["install_line"])
                print(f"expires at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(row['expires_at']))}",
                      file=sys.stderr)
        else:
            if not 1 <= a.days <= 365:
                raise SystemExit("--days must be between 1 and 365")
            import joincode
            name_file = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "name"
            from_name = a.name.strip()[:40] or (name_file.read_text().strip() if name_file.exists() else "")
            address_summary = (
                "Turn on (or rotate) your public request address for "
                f"{plain(from_name, 40) or 'strangers'}"
            )
            custody_act = broker_preflight("address-control", address_summary)
            st = call(s, {"op": "status"})
            release, line = _release_for_invite()
            if not release or not line:
                raise SystemExit("cannot publish an address until this node has a sha-pinned public release line")
            person_key = PersonKey("address-control", address_summary, act=custody_act)
            person_pub = person_key.public_key().public_bytes_raw().hex()
            unsigned_address = joincode.build_unsigned_address(
                person_pub, st["node"],
                [{"node": st["node"], "addrs": [st["relay"]]}], days=a.days,
                from_name=from_name, release=release)
            from network.nakd import address_control_statement
            ts = int(time.time())
            control_statement = address_control_statement(
                "register_address", unsigned_address["aid"], ts)
            signed = person_sign_batch(
                "address-control", [joincode._canon(unsigned_address), control_statement],
                summary=address_summary, act=custody_act)
            code = joincode.attach_address_signature(unsigned_address, signed[0][1])
            install_line = line.replace("{CODE}", code)
            signature = signed[1][1].hex()
            if a.name:
                _write_private(name_file, (from_name + "\n").encode())
            call(s, {"op": "register_address", "code": code, "install_line": install_line,
                     "ts": ts, "signature": signature})
            print(code)
            print(install_line)
            print("request address on (rotated): anyone with this line may ask; nothing is shared until you accept",
                  file=sys.stderr)
    elif a.cmd == "redeem":
        r = call(s, {"op": "redeem", "code": a.code, "nickname": a.nickname, "petname": a.petname, "wait_s": 20})
        if r["state"] == "request refused":
            print(f"request refused: {r.get('why') or 'not accepted'}")
        else:
            print(f"your request was sent; nothing is shared until {r.get('name') or 'they'} accept")
    elif a.cmd == "introduce":
        r = call(s, {"op": "introduce", "to": a.to, "about": a.about,
                     "note": a.note, "days": a.days})
        print(f"introduction {r['intro_id']} queued to {r['to']} and mirrored to {r['about']}; "
              "it is external data and creates no connection")
    elif a.cmd == "intros":
        rows = call(s, {"op": "intros"})["intros"]
        if not rows:
            print("no pending introductions")
        for row in rows:
            state = "valid" if row["valid"] else f"invalid: {row['problem']}"
            note = f" — {row['note']}" if row.get("note") else ""
            subject = (row["about"]["display_name"] if row.get("role") == "recipient" else
                       f"person {row['to'][:16]}… may ask to connect")
            print(f"{row['intro_id']}  {subject}  introduced by {row['introduced_by']}  "
                  f"trust=external  {state}{note}")
    elif a.cmd == "intro":
        op = "intro_accept" if a.intro_cmd == "accept" else "intro_decline"
        r = call(s, {"op": op, "id": a.id})
        if a.intro_cmd == "accept":
            print(f"connection request queued for {r['to']}; nothing flows unless they accept")
        else:
            print(f"declined {r['declined']}")
    elif a.cmd == "worker-redeem":
        from network import join_setup
        join_setup.retry_worker_redeem(Path.home(), name=a.name, http_post=http_post)
    elif a.cmd == "requests":
        for q in call(s, {"op": "requests"})["requests"] or []:
            kind = q.get("kind") or q.get("source") or "invite"
            tier = q.get("tier") or ""
            request_type = f"{kind} {tier}".strip()
            if kind in {"worker", "consumer"}:
                request_type += ", vouched by you"
            if kind == "intro":
                request_type += f", introduced by {q.get('introduced_by') or q.get('introducer', '')[:12]}"
            print(f"{q['id']}  {request_type:31}  {q['nickname'] or '(no nickname)'}  "
                  f"person {q['person'][:16]}…  {_ago(q['received'])}")
    elif a.cmd in ("accept", "decline"):
        proof = {}
        if a.cmd == "accept":
            pending = [row for row in call(s, {"op": "requests"})["requests"]
                       if row.get("id") == a.id]
            if pending and pending[0].get("requires_person"):
                stranger = pending[0].get("person", "unknown")
                nickname = plain(pending[0].get("nickname") or "", 40)
                accept_summary = f"Accept stranger {plain(stranger, 64)[:16]}…"
                if nickname:
                    accept_summary += f" (calls themselves {json.dumps(nickname, ensure_ascii=False)})"
                if a.petname:
                    accept_summary += f" as {plain(a.petname, 40)}"
                custody_act = broker_preflight("contact-accept", accept_summary)
                if not sys.stdin.isatty():
                    raise SystemExit(f"a person must accept this one: nak accept {a.id}")
                answer = input(f"ACCEPT person-gated request {a.id}? [y/N] ")
                if answer.strip().lower() not in {"y", "yes"}:
                    print("request unchanged")
                    return 1
                from network.nakd import accept_statement
                ts = int(time.time())
                signature = PersonKey(
                    "contact-accept", accept_summary, act=custody_act).sign(
                    accept_statement(a.id, a.petname, ts)).hex()
                proof = {"ts": ts, "signature": signature}
        r = call(s, {"op": a.cmd, "id": a.id,
                     **({"petname": a.petname, **proof} if a.cmd == "accept" else {})})
        print(json.dumps(r))
    elif a.cmd == "block":
        print(json.dumps(call(s, {"op": "block", "person": a.person, "days": a.days})))
    elif a.cmd == "unblock":
        print(json.dumps(call(s, {"op": "unblock", "person": a.person})))
    elif a.cmd == "blocked":
        for row in call(s, {"op": "blocked"})["blocked"]:
            print(f"{row['person']}  until {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(row['until']))}  "
                  f"{row['reason']}")
    elif a.cmd == "contacts":
        for c in call(s, {"op": "contacts"})["contacts"]:
            via = f" via {c['path']}" if c.get("path") else ""
            d = " [direct ok]" if c.get("direct") else ""
            print(f"{c['petname'] or '-':16} {c['state']:9} {('online' + via) if c['online'] else 'offline':16} "
                  f"{c['person'][:16]}…{d}  key custody: {c.get('key_custody', 'unknown')}")
    elif a.cmd == "contact":
        c = call(s, {"op": "contact", "who": a.who})["contact"]
        via = f" via {c['path']}" if c.get("path") else ""
        print(f"contact {c['petname'] or c['person'][:16]} "
              f"({('online' + via) if c.get('online') else 'offline'})")
        print(f"person {c['person']}\nnode {c['node']}")
        print(f"key custody: {c.get('key_custody', 'unknown')}")
    elif a.cmd == "remove":
        print(json.dumps(call(s, {"op": "remove", "who": a.who})))
    elif a.cmd == "send":
        r = call(s, {"op": "send", "to": a.to, "text": a.text, "aspect": a.aspect})
        print(f"queued {r['queued']} to {r['to']}")
    elif a.cmd == "agent":
        r = call(s, {"op": "agent_send", "to": a.to, "text": a.text, "intent": a.intent,
                     "thread": a.thread, "reply_to": a.reply_to,
                     **({"want": a.want} if a.want is not None else {}),
                     **({"budget": {"amount": a.budget, "unit": "credits"}}
                        if a.budget is not None else {})})
        print(f"queued {r['queued']} to {r['to']} in thread {r['thread']}")
    elif a.cmd == "threads":
        for row in call(s, {"op": "threads"})["threads"]:
            unread = " unread-ish" if row["unread"] else ""
            print(f"{row['thread']}  {row['contact']} ({row['peer'][:16]}…, {row['name_source']})  "
                  f"{row['last_intent']}  "
                  f"{_ago(row['last_time'])}  {row['count']} message(s){unread}")
    elif a.cmd == "thread":
        result = call(s, {"op": "thread", "thread": a.thread, "contact": a.contact})
        print(f"thread {result['thread']} peer={result['peer'][:16]}…")
        for message in result["messages"]:
            reply = f" reply_to={message['reply_to']}" if message["reply_to"] else ""
            want = f"\n  want: {message['want']}" if message.get("want") is not None else ""
            budget = message.get("budget")
            proposed = (f"\n  proposed budget: {budget['amount']} credits"
                        if isinstance(budget, dict) else "")
            sender = ("me" if message["from_me"] else
                      f"{message['name']} [{message['from_key']}…, {message['name_source']}]")
            print(f"[{_ago(message['created'])}] {sender} {message['intent']} mid={message['mid']}{reply}\n"
                  f"  {message['text']}{want}{proposed}")
    elif a.cmd == "card":
        if a.card_cmd == "set":
            intents = [value.strip() for value in a.intents.split(",") if value.strip()] or None
            models = [value.strip() for value in a.models.split(",") if value.strip()]
            result = call(s, {"op": "card_set", "name": a.name, "offers": a.offers,
                              "intents": intents, "models": models, "price_hint": a.price_hint,
                              "paid": a.paid})
            print(f"agent card saved and sent to {result['sent']} contact(s)")
        else:
            result = call(s, {"op": "card", "contact": a.contact})
            card = result["card"]
            print(f"{result.get('contact') or 'my card'}: {card['display_name']}\n"
                  f"offers: {card['offers']}\nintents: {', '.join(card['intents'])}\n"
                  f"models: {', '.join(card['models']) or '-'}\npaid: {'yes' if card['paid'] else 'no'}\n"
                  f"price hint: {card['price_hint'] or '-'}\nsupports: {', '.join(card['supports'])}")
    elif a.cmd == "direct":
        r = call(s, {"op": "direct", "who": a.who, "on": a.onoff == "on"})
        if a.onoff == "on":
            print("direct ON for this contact: they will learn your LAN/IPv6 addresses (not via the relay any "
                  "more). Used only if they turn it on for you too." +
                  ("" if r.get("listening") else " Note: this node is not listening; run: nak direct-listen 51830"))
        else:
            print("direct OFF: their addresses are forgotten; traffic goes through the relay")
    elif a.cmd == "direct-listen":
        import subprocess as _sp
        f = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "direct-port"
        if a.port == "off":
            f.unlink(missing_ok=True)
        else:
            _write_private(f, (str(int(a.port)) + "\n").encode())
        _sp.run(["systemctl", "--user", "restart", "nak-net.service"], check=False)
        print(f"direct listener {'off' if a.port == 'off' else 'on port ' + a.port}; nak-net restarted")
    elif a.cmd == "p2p":
        import subprocess as _sp
        f = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "p2p-dial"
        with _p2p_ownership_lock():
            if a.onoff == "off":
                f.unlink(missing_ok=True)
                _sp.run(["systemctl", "--user", "restart", "nak-net.service"], check=False)
                if a.socket is None:          # a custom --socket means a sidecar we don't manage
                    p2p_env = Path.home() / ".nakshatra" / "p2p.env"
                    if p2p_env.exists():
                        _remove_env_assignments(
                            p2p_env, "NAKSHATRA_P2P_GATEWAY_ARGS",
                            "NAKSHATRA_P2P_CONSUMER_ARGS")
                    _p2p_installer(_sp, "disable")
            else:
                dial = a.socket or _default_p2p_socket()
                if not dial.is_absolute():
                    raise SystemExit("--socket must be an absolute path")
                if a.socket is None:          # the bundled sidecar listens on the default socket only
                    _enable_p2p_service_locked(_sp)
                _write_private(f, (str(dial) + "\n").encode())
                _sp.run(["systemctl", "--user", "restart", "nak-net.service"], check=False)
            # The worker reads the P2P state only at startup. Keep the ownership check and restart in
            # this same transaction so gateway release cannot race it.
            if _worker_unit_enabled(_sp):
                _worker_installer(_sp, "restart")
                print("worker restarted so it advertises the new p2p state")
        print(f"p2p {'OFF' if a.onoff == 'off' else 'ON via ' + str(dial)}; nak-net restarted")
    elif a.cmd == "ledger":
        r = call(s, {"op": "ledger"})
        print(f"settlement adapter: {r['adapter']}  (TEST units — no real money)")
        for j in r["journal"]:
            print(f"[{_ago(j['ts'])}] {j['task_hash'][:12]} {j['event']:9} {j['amount']} {j['unit']}"
                  f"  {('→ ' + j['party'][:12]) if j['party'] else ''}")
    elif a.cmd == "escrow":
        print(json.dumps(call(s, {"op": "escrow_status", "task": a.task}), sort_keys=True))
    elif a.cmd == "tasks":
        from network.client import render_tasks
        print(render_tasks(call(s, {"op": "tasks", "role": a.role})["tasks"]))
    elif a.cmd == "task":
        if a.tcmd == "post":
            rules = [json.loads(r) for r in a.rule] or [{"max_words": 200}]
            try:
                reward = _cli_amount(a.reward) if isinstance(a.reward, str) else a.reward
            except SystemExit as exc:
                raise SystemExit(f"--reward {exc}") from exc
            r = call(s, {"op": "task_post", "to": a.to, "title": a.title, "instructions": a.instructions,
                         "acceptance": rules, "reward": reward, "deadline_s": a.deadline})
            esc = (f"; escrow open {format_amount(r['escrow']['amount'])} (devnet credits)"
                   if r.get("escrow") else "")
            print(f"posted {r['task_hash'][:12]} to {', '.join(r['posted_to'])}{esc}")
        elif a.tcmd == "list":
            from network.client import render_tasks
            print(render_tasks(call(s, {"op": "tasks", "role": a.role})["tasks"]))
        elif a.tcmd == "claim":
            print(json.dumps(call(s, {"op": "task_claim", "task": a.task})))
        elif a.tcmd == "submit":
            print(json.dumps(call(s, {"op": "task_submit", "task": a.task, "output": a.output})))
        elif a.tcmd == "decide":
            custody_act = broker_preflight(
                "task-verdict", f"{a.decision.title()} paid task {plain(a.task, 12)}")
            if not sys.stdin.isatty():
                raise SystemExit("nak task decide is interactive-only; run it in a terminal")
            task_hash = _resolve_posted_task(s, a.task)
            task_summary = f"{a.decision.title()} paid task {task_hash[:12]}…"
            # Bind the configured public authority before prompting; the actual signature below
            # still goes through person_sign so verdict_reason reaches the broker context.
            decision_act = custody_act.for_signing("task-verdict", task_summary)
            PersonKey("task-verdict", task_summary, act=decision_act)
            answer = input(f"{a.decision.upper()} paid task {task_hash[:12]}? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                print("task unchanged")
                return 1
            from network import verdict as verdict_policy
            ts = int(time.time())
            signature = person_sign(
                "task-verdict",
                verdict_policy.decision_statement(task_hash, a.decision, a.reason, ts),
                summary=task_summary, context={"verdict_reason": a.reason},
                act=decision_act).hex()
            print(json.dumps(call(s, {"op": "task_decide", "task": task_hash,
                                      "decision": a.decision, "reason": a.reason,
                                      "ts": ts, "signature": signature})))
    elif a.cmd == "inbox":
        for m in reversed(call(s, {"op": "inbox", "limit": a.limit})["messages"]):
            print(f"[{_ago(m['received'])}] {m['petname'] or m['from_person'][:12]}: {m['text']}")
    elif a.cmd == "outbox":
        for row in call(s, {"op": "outbox", "refused": a.refused, "text": a.text,
                            "limit": a.limit})["items"]:
            state = "refused" if row["refused"] else ("delivered" if row["delivered"] else "queued")
            detail = f": {row['reason']}" if row.get("reason") else ""
            print(f"[{_ago(row['created'])}] {row['id']} to {row['to'][:12]} {row['kind']} {state}{detail}")
            if a.text and row.get("text"):
                print(f"  {row['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
