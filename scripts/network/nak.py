"""nak — talk to your node's messaging service.

    nak status
    nak invite --person-key ~/.sthambha/person-TEST.key [--ttl 86400] [--note "for Rajesh"]
    nak redeem <nki1.…> [--nickname "B"] [--as rajesh]     ask to connect (they must accept)
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
import hashlib
import ipaddress
import json
import math
import os
import secrets
import stat
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from release_trust import ReleaseTrustError, load_release_package_signers  # noqa: E402


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


# The public circuit relay (VPS). Its PeerId is the relay's identity: a sidecar refuses any other key there.
DEFAULT_P2P_RELAY_ID = "12D3KooWDfHFtM8PP7TV46nssrix2hTL3kzMkv7kKMhULDakFGB7"
DEFAULT_P2P_RELAYS = (f"/ip4/45.63.109.137/udp/29700/quic-v1/p2p/{DEFAULT_P2P_RELAY_ID},"
                      f"/ip4/45.63.109.137/tcp/29700/p2p/{DEFAULT_P2P_RELAY_ID}")
DEFAULT_ONBOARD_BASE = "https://prithviloka.net/onboard"


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
        raise SystemExit(f"onboard server refused the request ({e.code})" + (f": {detail}" if detail else ""))
    except (OSError, URLError, ValueError) as e:
        raise SystemExit(f"could not reach the onboard server: {e}")
    if not isinstance(result, dict):
        raise SystemExit("onboard server returned an invalid response")
    return result


def _onboard_token() -> str:
    token = os.environ.get("NAK_ONBOARD_TOKEN")
    if token is not None:
        token = token.strip()
        if not token:
            raise SystemExit("NAK_ONBOARD_TOKEN is empty")
        return token
    path = Path.home() / ".nakshatra" / "onboard.token"
    try:
        info = path.lstat()
    except OSError as e:
        raise SystemExit(f"worker invites need NAK_ONBOARD_TOKEN or {path}: {e}")
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise SystemExit(f"refusing {path}: operator token file must be a regular 0600 file")
    token = path.read_text().strip()
    if not token:
        raise SystemExit(f"refusing {path}: operator token is empty")
    return token


def _stage_worker(redeem: str, tier: str, http_post, *, now: int | None = None) -> tuple[dict, int]:
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
    result = http_post(redeem.rstrip("/") + "/worker-stage", {"token": _onboard_token(), "tier": tier})
    code = result.get("code") if isinstance(result, dict) else None
    if not isinstance(code, str) or not code.strip() or len(code) > 64:
        raise SystemExit("onboard server returned an invalid worker code")
    expires_at = result.get("expires_at", staged_at + 3600)
    if isinstance(expires_at, bool) or not isinstance(expires_at, int) or expires_at <= staged_at:
        raise SystemExit("onboard server returned an invalid worker code expiry")
    return {"redeem": redeem.rstrip("/"), "code": code, "tier": tier}, expires_at


def _stage_consumer(redeem: str, tier: str, starter: int, inviter_pubkey: str,
                    http_post, *, now: int | None = None) -> tuple[dict, int]:
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
    if type(starter) is not int or not 0 <= starter <= 10**12:
        raise SystemExit("--starter must be an integer between 0 and 1000000000000")
    result = http_post(redeem.rstrip("/") + "/consumer-stage", {
        "token": _onboard_token(), "tier": tier, "starter": starter,
        "inviter_pubkey": inviter_pubkey,
    })
    code = result.get("code") if isinstance(result, dict) else None
    if not isinstance(code, str) or not code.strip() or len(code) > 64:
        raise SystemExit("onboard server returned an invalid consumer code")
    expires_at = result.get("expires_at", staged_at + 3600)
    if isinstance(expires_at, bool) or not isinstance(expires_at, int) or expires_at <= staged_at:
        raise SystemExit("onboard server returned an invalid consumer code expiry")
    return {"redeem": redeem.rstrip("/"), "code": code, "tier": tier,
            "starter": starter}, expires_at


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


def _enable_p2p_service(_sp) -> None:
    """Create the opt-in if needed, then atomically hand the one-service enable to the installer."""
    env = Path.home() / ".nakshatra" / "p2p.env"
    created = False
    if not env.exists():
        env.parent.mkdir(parents=True, exist_ok=True)
        env.write_text(f"P2P_RELAYS={DEFAULT_P2P_RELAYS}\n")
        created = True
    try:
        _p2p_installer(_sp, "enable")
    except (Exception, SystemExit):
        # A failed enable may have started the unit before its stability check failed: stop it again.
        try:
            _p2p_installer(_sp, "disable")
        except (Exception, SystemExit):
            pass
        if created:
            env.unlink(missing_ok=True)
        raise


WORKER_SERVICE = "nakshatra-worker"
WORKER_BACKENDS = {"cpu", "cuda", "rocm", "vulkan", "metal"}
GATEWAY_SERVICE = "nakshatra-gateway"
GATEWAY_PROFILE = "inference.env"
GATEWAY_PROFILE_OWNER = "inference-profile-created"
GATEWAY_PROFILE_OWNER_PREFIX = "# nak-gateway-owned: "


def _gateway_paths() -> tuple[Path, Path]:
    root = Path.home() / ".nakshatra" / "gateway"
    return root, Path.home() / ".nakshatra" / "gateway.env"


def _gateway_profile_paths() -> tuple[Path, Path]:
    root, _env_path = _gateway_paths()
    return Path.home() / ".nakshatra" / GATEWAY_PROFILE, root / GATEWAY_PROFILE_OWNER


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


def _usdc_credits(text: str) -> int:
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise SystemExit("withdraw amount must be a positive USDC decimal") from exc
    micros = value * 1_000_000
    if not value.is_finite() or value <= 0 or micros != micros.to_integral_value():
        raise SystemExit("withdraw amount must be positive with at most 6 USDC decimals")
    return int(micros) * 100


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
            print(json.dumps(hook.wallet_status(), indent=1, sort_keys=True))
        elif args.wallet_cmd == "bind":
            print(json.dumps(hook.bind(args.address), indent=1, sort_keys=True))
        else:
            credits = _usdc_credits(args.usdc)
            print(json.dumps(hook.withdraw(credits), indent=1, sort_keys=True))
    except LedgerUnavailable as exc:
        raise SystemExit(f"wallet refused: {exc}") from exc


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
    _write_private(env_path, body)
    action = "restart" if old_env is not None and _gateway_unit_enabled(_sp) else "enable"
    try:
        _gateway_installer(_sp, action)
    except (Exception, SystemExit):
        if old_env is None:
            env_path.unlink(missing_ok=True)
        else:
            _write_private(env_path, old_env)
        if made_profile:
            _release_gateway_profile_ownership()
        raise
    print(f"gateway ON at http://{args.bind}:{args.port}/v1 (credits enforced)")


def _gateway_unit_enabled(_sp) -> bool:
    try:
        result = _sp.run(
            ["systemctl", "--user", "is-enabled", f"{GATEWAY_SERVICE}.service"],
            capture_output=True, text=True, check=False)
    except OSError:
        return False
    return result.returncode == 0 and (result.stdout or "").strip() not in {
        "disabled", "masked", "not-found",
    }


def _gateway_off(_sp) -> None:
    _gateway_installer(_sp, "disable")
    _gateway_paths()[1].unlink(missing_ok=True)
    _release_gateway_profile_ownership()
    print("gateway OFF")


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
        "ledger_mode": values.get("NAKSHATRA_LEDGER_MODE"), "unit": unit,
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


def _set_env_assignment(path: Path, name: str, value: str | None) -> None:
    """Replace one assignment in a release-owned env file, preserving the rest."""
    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        raise SystemExit(f"cannot update {path}: {e}") from e
    prefix = f"{name}="
    lines = [line for line in lines if not line.startswith(prefix)]
    if value is not None:
        lines.append(f'{name}="{value}"')
    _write_private(path, (("\n".join(lines) + "\n") if lines else "").encode())


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


def _worker_enable(_sp, args) -> None:
    root, _record, env_path = _worker_paths()
    # systemd's ReadWritePaths requires the path to exist before it can construct the sandbox. Keep
    # both writable roots present even when build.json is missing so the launcher gets to print the
    # useful refusal and TLS can create this node's certificate on its first start.
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    (Path.home() / ".nakshatra" / "tls").mkdir(parents=True, exist_ok=True, mode=0o700)
    old_env = env_path.read_bytes() if env_path.exists() else None
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
    _set_env_assignment(
        env_path,
        "NAKSHATRA_P2P_WORKER_ARGS",
        f"-inbound=127.0.0.1:{port} -require-direct-activation",
    )
    try:
        _worker_installer(_sp, "enable")
        if _p2p_enabled():
            _p2p_installer(_sp, "restart")
    except (Exception, SystemExit):
        # enable may have started the unit before its stability check failed. Ask the same verified
        # installer to stop it, then restore exactly the opt-in state the operator had beforehand.
        try:
            _worker_installer(_sp, "disable")
        except (Exception, SystemExit):
            pass
        if old_env is None:
            env_path.unlink(missing_ok=True)
        else:
            _write_private(env_path, old_env)
        if _p2p_enabled():
            try:
                _p2p_installer(_sp, "restart")
            except (Exception, SystemExit):
                pass
        raise
    print(f"worker ON ({env_path})")
    if not _p2p_enabled():
        print("p2p is not enabled; this worker is reachable only by its literal address")


def _worker_disable(_sp) -> None:
    _root, _record, env_path = _worker_paths()
    old_env = env_path.read_bytes() if env_path.exists() else None
    if old_env is not None:
        _set_env_assignment(env_path, "NAKSHATRA_P2P_WORKER_ARGS", None)
    try:
        _worker_installer(_sp, "disable")
    except (Exception, SystemExit):
        if old_env is not None:
            _write_private(env_path, old_env)
        raise
    if _p2p_enabled():
        _p2p_installer(_sp, "restart")
    print("worker OFF")


def _worker_status(_sp) -> dict:
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
    return {"configured": env_path.exists(), "build_state": build_state, "build": build, "unit": unit}


def _owner_lease_path() -> Path:
    return Path.home() / ".nakshatra" / "owner-lease.env"


def _worker_unit_enabled(_sp) -> bool:
    try:
        result = _sp.run(
            ["systemctl", "--user", "is-enabled", f"{WORKER_SERVICE}.service"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and (result.stdout or "").strip() not in {
        "disabled", "masked", "not-found",
    }


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


def main(argv=None, *, http_post=None) -> int:
    default_sock = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "nakd.sock"
    ap = argparse.ArgumentParser(prog="nak", description=__doc__.split("\n")[0])
    ap.add_argument("--sock", type=Path, default=default_sock)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("invite"); p.add_argument("--person-key", type=Path,
                                                 default=Path.home() / ".sthambha" / "person-TEST.key")
    p.add_argument("--ttl", type=int, default=86400); p.add_argument("--note", default="")
    p.add_argument("--release-url", default="", help="where newcomers download Nakshatra (default: this node's source)")
    p.add_argument("--code-only", action="store_true", help="print just the invite code")
    p.add_argument("--from", dest="from_name", default="", help="your name as your friend will see it (remembered)")
    p.add_argument("--kind", choices=["contact", "worker", "consumer"], default="contact")
    p.add_argument("--tier", choices=["stranger", "known", "trusted"], default="known",
                   help="worker/consumer trust tier (self is never granted by an invite)")
    p.add_argument("--starter", type=int, default=0,
                   help="consumer starter credits requested from the inviter (default: 0)")
    p.add_argument("--redeem", default=DEFAULT_ONBOARD_BASE,
                   help="onboard server base URL for a worker or consumer invite")
    p = sub.add_parser("redeem"); p.add_argument("code"); p.add_argument("--nickname", default="")
    p.add_argument("--as", dest="petname", default="")
    p = sub.add_parser("worker-redeem", help="retry the stored worker admission while its code is valid")
    p.add_argument("--name", default="", help="override the worker name stored during join")
    sub.add_parser("requests")
    p = sub.add_parser("accept"); p.add_argument("id"); p.add_argument("--as", dest="petname", default="")
    p = sub.add_parser("decline"); p.add_argument("id")
    sub.add_parser("contacts")
    p = sub.add_parser("remove"); p.add_argument("who")
    p = sub.add_parser("send"); p.add_argument("to"); p.add_argument("text")
    p.add_argument("--aspect", default="")
    p = sub.add_parser("inbox"); p.add_argument("--limit", type=int, default=20)
    sub.add_parser("ledger", help="escrow journal: open / released / refunded")
    p = sub.add_parser("wallet", help="deposit and withdraw USDC-backed inference credits")
    walsub = p.add_subparsers(dest="wallet_cmd", required=True)
    walsub.add_parser("address", help="show or queue this operator's deposit token account")
    walsub.add_parser("status", help="show balance, cashable credits, and pending withdrawals")
    q = walsub.add_parser("bind", help="bind the Solana address that receives withdrawals")
    q.add_argument("address")
    q = walsub.add_parser("withdraw", help="request USDC to the bound payout address")
    q.add_argument("usdc", help="USDC amount (at least 1, up to 6 decimal places)")
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
    p = sub.add_parser("worker", help="build and operate this node's signed-release inference worker")
    wsub = p.add_subparsers(dest="worker_cmd", required=True)
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
    q = wsub.add_parser("owner-lease", help="control the local owner-sentinel lease")
    lsub = q.add_subparsers(dest="owner_lease_cmd", required=True)
    q = lsub.add_parser("on", help="create the private lease token and enable default-deny")
    q.add_argument("--port", type=int, default=5571)
    q.add_argument("--ttl-s", type=float, default=6.0)
    q.add_argument("--resume-s", type=float, default=30.0)
    lsub.add_parser("off", help="remove the lease configuration")
    lsub.add_parser("status", help="show lease configuration without revealing its token")
    p = sub.add_parser("task", help="post / list / claim / submit tasks")
    tsub = p.add_subparsers(dest="tcmd", required=True)
    q = tsub.add_parser("post"); q.add_argument("to"); q.add_argument("title"); q.add_argument("instructions")
    q.add_argument("--rule", action="append", default=[], help='JSON rule, e.g. \'{"max_words": 60}\' (repeatable)')
    q.add_argument("--reward", type=int, default=0); q.add_argument("--deadline", type=int, default=3600)
    q = tsub.add_parser("list"); q.add_argument("--role", default="")
    q = tsub.add_parser("claim"); q.add_argument("task")
    q = tsub.add_parser("submit"); q.add_argument("task"); q.add_argument("output")
    a = ap.parse_args(argv)
    s = a.sock

    if a.cmd == "wallet":
        _wallet_command(a)
    elif a.cmd == "gateway":
        import subprocess as _sp
        if a.gateway_cmd == "on":
            _gateway_on(_sp, a)
        elif a.gateway_cmd == "off":
            _gateway_off(_sp)
        else:
            print(json.dumps(_gateway_status(_sp), indent=1, sort_keys=True))
    elif a.cmd == "worker":
        import subprocess as _sp
        if a.worker_cmd == "build":
            _worker_build(_sp)
        elif a.worker_cmd == "on":
            _worker_enable(_sp, a)
        elif a.worker_cmd == "off":
            _worker_disable(_sp)
        elif a.worker_cmd == "status":
            print(json.dumps(_worker_status(_sp), indent=1, sort_keys=True))
        elif a.owner_lease_cmd == "on":
            _owner_lease_on(_sp, a)
        elif a.owner_lease_cmd == "off":
            _owner_lease_off(_sp)
        else:
            print(json.dumps(_owner_lease_status(), indent=1, sort_keys=True))
    elif a.cmd == "status":
        r = call(s, {"op": "status"})
        print(f"person {r['person'][:16]}…  node {r['node'][:16]}…  custody {r['custody']}  relay {r['relay']}")
        print(f"{len(r['contacts'])} contacts, {r['pending_requests']} pending requests, {r['open_invites']} open invites")
        print(f"direct listener {r.get('direct_port') or 'off'}  p2p "
              f"{(r.get('p2p') or {}).get('dial', 'off')}")
    elif a.cmd == "invite":
        import joincode
        from cryptography.hazmat.primitives.asymmetric import ed25519
        if a.kind != "consumer" and a.starter:
            raise SystemExit("--starter is only valid with --kind consumer")
        http_post = http_post or _http_post_json
        st = call(s, {"op": "status"})
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(a.person_key.read_text().strip()))
        release, line = _release_for_invite(a.release_url)
        name_file = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "name"
        from_name = a.from_name or (name_file.read_text().strip() if name_file.exists() else "")
        if a.from_name:
            name_file.write_text(a.from_name.strip()[:40] + "\n")
        if not from_name:
            print("tip: add --from <your name> so your friend sees who invited them (remembered after once)",
                  file=sys.stderr)
        issued_at = int(time.time())
        worker = consumer = None
        if a.kind == "worker":
            worker, staged_expiry = _stage_worker(a.redeem, a.tier, http_post, now=issued_at)
        elif a.kind == "consumer":
            consumer, staged_expiry = _stage_consumer(
                a.redeem, a.tier, a.starter, st["node"], http_post, now=issued_at)
        else:
            staged_expiry = issued_at + a.ttl
        expires_at = min(issued_at + a.ttl, staged_expiry)
        code = joincode.encode_invite(priv, st["node"], [{"node": st["node"], "addrs": [st["relay"]]}],
                                      ttl_s=expires_at - issued_at, now=issued_at, note=a.note,
                                      release=release, from_name=from_name,
                                      kind=a.kind, worker=worker, consumer=consumer)
        call(s, {"op": "register_invite", "code": code})
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
    elif a.cmd == "redeem":
        r = call(s, {"op": "redeem", "code": a.code, "nickname": a.nickname, "petname": a.petname, "wait_s": 20})
        print(f"{r['state']}. They have to accept before either of you can message.")
    elif a.cmd == "worker-redeem":
        from network import join_setup
        join_setup.retry_worker_redeem(Path.home(), name=a.name, http_post=http_post)
    elif a.cmd == "requests":
        for q in call(s, {"op": "requests"})["requests"] or []:
            print(f"{q['id']}  {q['nickname'] or '(no nickname)'}  person {q['person'][:16]}…  {_ago(q['received'])}")
    elif a.cmd in ("accept", "decline"):
        r = call(s, {"op": a.cmd, "id": a.id, **({"petname": a.petname} if a.cmd == "accept" else {})})
        print(json.dumps(r))
    elif a.cmd == "contacts":
        for c in call(s, {"op": "contacts"})["contacts"]:
            via = f" via {c['path']}" if c.get("path") else ""
            d = " [direct ok]" if c.get("direct") else ""
            print(f"{c['petname'] or '-':16} {c['state']:9} {('online' + via) if c['online'] else 'offline':16} "
                  f"{c['person'][:16]}…{d}")
    elif a.cmd == "remove":
        print(json.dumps(call(s, {"op": "remove", "who": a.who})))
    elif a.cmd == "send":
        r = call(s, {"op": "send", "to": a.to, "text": a.text, "aspect": a.aspect})
        print(f"queued {r['queued']} to {r['to']}")
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
            f.write_text(str(int(a.port)) + "\n")
        _sp.run(["systemctl", "--user", "restart", "nak-net.service"], check=False)
        print(f"direct listener {'off' if a.port == 'off' else 'on port ' + a.port}; nak-net restarted")
    elif a.cmd == "p2p":
        import subprocess as _sp
        f = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "p2p-dial"
        if a.onoff == "off":
            f.unlink(missing_ok=True)
            _sp.run(["systemctl", "--user", "restart", "nak-net.service"], check=False)
            if a.socket is None:              # a custom --socket means a sidecar we don't manage
                _p2p_installer(_sp, "disable")
        else:
            dial = a.socket or _default_p2p_socket()
            if not dial.is_absolute():
                raise SystemExit("--socket must be an absolute path")
            if a.socket is None:              # the bundled sidecar listens on the default socket only
                _enable_p2p_service(_sp)
            f.write_text(str(dial) + "\n")
            _sp.run(["systemctl", "--user", "restart", "nak-net.service"], check=False)
        # The worker reads the p2p state (its typed endpoint / PeerId) only at startup: restart it so it advertises
        # the new state instead of a stale endpoint (U4 C1 final review). Only if this node runs a worker.
        if (Path.home() / ".nakshatra" / "worker.env").is_file():
            _worker_installer(_sp, "restart")
            print("worker restarted so it advertises the new p2p state")
        print(f"p2p {'OFF' if a.onoff == 'off' else 'ON via ' + str(dial)}; nak-net restarted")
    elif a.cmd == "ledger":
        r = call(s, {"op": "ledger"})
        print(f"settlement adapter: {r['adapter']}  (TEST units — no real money)")
        for j in r["journal"]:
            print(f"[{_ago(j['ts'])}] {j['task_hash'][:12]} {j['event']:9} {j['amount']} {j['unit']}"
                  f"  {('→ ' + j['party'][:12]) if j['party'] else ''}")
    elif a.cmd == "task":
        if a.tcmd == "post":
            rules = [json.loads(r) for r in a.rule] or [{"max_words": 200}]
            r = call(s, {"op": "task_post", "to": a.to, "title": a.title, "instructions": a.instructions,
                         "acceptance": rules, "reward": a.reward, "deadline_s": a.deadline})
            esc = f"; escrow open {r['escrow']['amount']} {r['escrow']['unit']}" if r.get("escrow") else ""
            print(f"posted {r['task_hash'][:12]} to {', '.join(r['posted_to'])}{esc}")
        elif a.tcmd == "list":
            from network.client import render_tasks
            print(render_tasks(call(s, {"op": "tasks", "role": a.role})["tasks"]))
        elif a.tcmd == "claim":
            print(json.dumps(call(s, {"op": "task_claim", "task": a.task})))
        elif a.tcmd == "submit":
            print(json.dumps(call(s, {"op": "task_submit", "task": a.task, "output": a.output})))
    elif a.cmd == "inbox":
        for m in reversed(call(s, {"op": "inbox", "limit": a.limit})["messages"]):
            print(f"[{_ago(m['received'])}] {m['petname'] or m['from_person'][:12]}: {m['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
