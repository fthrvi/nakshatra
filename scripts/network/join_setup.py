"""Join setup — what `install.py join` runs inside the freshly installed release.

Makes the node a person's node, then asks the inviter to connect:

    1. node key        ~/.nakshatra/keys/worker.ed25519      (kept if present)
    2. person key      ~/.sthambha/person-TEST.key           TEST custody: a file, not a vault
    3. signer state    ~/.sthambha/signer/                   node.pub, person.pub
    4. an agent        "agent": may message and claim tasks  (NOT post or pay; the person decides that)
    5. services        nak-signer + nak-net (systemd --user)
    6. the request     redeem the invite; the friend still has to accept

Idempotent: run it again with another invite and it reuses everything, only redeeming the new one.
NAK_JOIN_NO_SYSTEMD=1 skips step 5 (tests run nakd in-process).
"""
from __future__ import annotations

import argparse
from http.client import HTTPException
import json
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

AGENT = "agent"
AGENT_CAPS = ("nak.msg", "nak.task.claim")
AGENT_TTL_S = 365 * 86400
AGENT_LEASE_S = 7 * 86400
AGENT_IDLE_S = 30 * 86400
OLD_SIGNER_MESSAGE = "signer too old for sliding grants; update the node or use --no-lease"
WORKER_RETRY_FILE = Path(".nakshatra") / "worker-invite.json"
CONSUMER_STANDING_FILE = Path(".nakshatra") / "consumer-standing.json"


class WorkerRedeemError(RuntimeError):
    pass


def _say(msg: str) -> None:  # noqa: D401
    print(f"  · {msg}", flush=True)


def _http_post_json(url: str, body: dict) -> dict:
    try:
        req = Request(url, data=json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
                      headers={"content-type": "application/json"}, method="POST")
        with urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
    except HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("error", "")
        except (ValueError, AttributeError):
            detail = ""
        raise WorkerRedeemError(f"onboard server refused worker admission ({e.code})" +
                                (f": {detail}" if detail else ""))
    except (HTTPException, OSError, URLError, ValueError) as e:
        raise WorkerRedeemError(f"could not reach the onboard server: {e}")
    if not isinstance(result, dict):
        raise WorkerRedeemError("onboard server returned an invalid worker response")
    return result


def _store_worker_invite(home: Path, code: str, name: str) -> Path:
    path = home / WORKER_RETRY_FILE
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"invite": code, "name": name[:40]}, f, separators=(",", ":"))
            f.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _store_consumer_standing(home: Path, standing: dict) -> Path:
    """Persist only non-authoritative consumer facts later CLI setup may reuse.

    The onboarding response is unsigned. Its package URL is only a location;
    package signer trust comes exclusively from the signed node release.
    """
    package_url = standing.get("package_url")
    if package_url is not None:
        if not isinstance(package_url, str) or not package_url.strip():
            raise WorkerRedeemError("onboard server returned an invalid package_url")
    saved = {
        "role": "consumer",
        "tier": standing["tier"],
        "operator": standing["operator"],
    }
    if package_url is not None:
        saved["package_url"] = package_url.strip()
    path = home / CONSUMER_STANDING_FILE
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(saved, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _load_worker_invite(home: Path) -> dict:
    path = home / WORKER_RETRY_FILE
    try:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise WorkerRedeemError(f"refusing {path}: stored worker invite must be a regular 0600 file")
        saved = json.loads(path.read_text())
    except WorkerRedeemError:
        raise
    except (OSError, ValueError) as e:
        raise WorkerRedeemError(f"could not read a stored worker invite at {path}: {e}")
    if not isinstance(saved, dict) or not isinstance(saved.get("invite"), str) or \
            not isinstance(saved.get("name"), str):
        raise WorkerRedeemError(f"stored worker invite at {path} is malformed")
    return saved


def redeem_worker_invite(home: Path, code: str, name: str, *, http_post=None,
                         capabilities_fn=None, now: int | None = None) -> dict:
    """Redeem the worker half of a signed invite with proof of possession of the node key."""
    import joincode
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from fabric.worker_join import detect_capabilities
    try:
        inv = joincode.decode_invite(code, now=now)
    except ValueError as e:
        raise WorkerRedeemError(str(e))
    if inv.get("kind", "contact") != "worker":
        raise WorkerRedeemError("stored invite is not a worker invite")
    node_key = home / ".nakshatra" / "keys" / "worker.ed25519"
    try:
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(node_key.read_bytes())
    except (OSError, ValueError) as e:
        raise WorkerRedeemError(f"could not read the node key at {node_key}: {e}")
    body = {"v": 2, "code": inv["worker"]["code"],
            "pubkey": priv.public_key().public_bytes_raw().hex(), "name": name[:40],
            "capabilities": (capabilities_fn or detect_capabilities)(),
            "ts": int(now if now is not None else time.time())}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    body["sig"] = priv.sign(canonical).hex()
    try:
        result = (http_post or _http_post_json)(inv["worker"]["redeem"].rstrip("/") + "/worker-redeem", body)
    except WorkerRedeemError:
        raise
    except (HTTPException, OSError, URLError, ValueError) as e:
        raise WorkerRedeemError(f"could not reach the onboard server: {e}")
    if not isinstance(result, dict) or not isinstance(result.get("tier"), str) or \
            not isinstance(result.get("junction"), str):
        raise WorkerRedeemError("onboard server returned no worker standing")
    if result["tier"] != inv["worker"]["tier"]:
        raise WorkerRedeemError("onboard server returned a different tier than the signed invite")
    return result


def redeem_consumer_invite(home: Path, code: str, name: str, *, http_post=None,
                           now: int | None = None) -> dict:
    """Redeem consumer admission with the worker-v2 node-key proof, but no worker probes."""
    import joincode
    from cryptography.hazmat.primitives.asymmetric import ed25519
    try:
        inv = joincode.decode_invite(code, now=now)
    except ValueError as e:
        raise WorkerRedeemError(str(e))
    if inv.get("kind", "contact") != "consumer":
        raise WorkerRedeemError("invite is not a consumer invite")
    node_key = home / ".nakshatra" / "keys" / "worker.ed25519"
    try:
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(node_key.read_bytes())
    except (OSError, ValueError) as e:
        raise WorkerRedeemError(f"could not read the node key at {node_key}: {e}")
    body = {"v": 2, "code": inv["consumer"]["code"],
            "pubkey": priv.public_key().public_bytes_raw().hex(), "name": name[:40],
            # Redeem v2 deliberately keeps one proof schema. Empty means this identity offers no compute.
            "capabilities": {}, "ts": int(now if now is not None else time.time())}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    body["sig"] = priv.sign(canonical).hex()
    try:
        result = (http_post or _http_post_json)(
            inv["consumer"]["redeem"].rstrip("/") + "/consumer-redeem", body)
    except WorkerRedeemError:
        raise
    except (HTTPException, OSError, URLError, ValueError) as e:
        raise WorkerRedeemError(f"could not reach the onboard server: {e}")
    if not isinstance(result, dict) or result.get("role") != "consumer" or \
            not isinstance(result.get("tier"), str) or not isinstance(result.get("operator"), str):
        raise WorkerRedeemError("onboard server returned no consumer standing")
    if result["tier"] != inv["consumer"]["tier"]:
        raise WorkerRedeemError("onboard server returned a different tier than the signed invite")
    try:
        _store_consumer_standing(home, result)
    except OSError as e:
        raise WorkerRedeemError(f"could not persist consumer standing: {e}") from e
    return result


def retry_worker_redeem(home: Path, *, name: str = "", http_post=None) -> dict:
    """CLI entry point: retry the persisted invite and delete it only after admission succeeds."""
    try:
        saved = _load_worker_invite(home)
    except WorkerRedeemError as e:
        raise SystemExit(f"worker redeem failed: {e}")
    try:
        result = redeem_worker_invite(home, saved["invite"], name or saved["name"], http_post=http_post)
    except WorkerRedeemError as e:
        raise SystemExit(f"worker redeem failed: {e}. The invite is still stored; retry: nak worker-redeem")
    (home / WORKER_RETRY_FILE).unlink()
    print(f"worker standing: tier {result['tier']}, junction {result['junction']}")
    return result


def _delegate_command(state: Path, person_key: Path, agent: str, caps, *, reuse_key: bool) -> str:
    argv = ["python", "-m", "sthambha.signer", "delegate", "--state", str(state),
            "--person-key", str(person_key), "--agent", agent,
            "--caps", ",".join(sorted(set(caps))), "--ttl", str(AGENT_TTL_S)]
    if reuse_key:
        argv.append("--reuse-key")
    return shlex.join(argv)


def _path_present(path: Path) -> bool:
    """Like lexists: even a dangling symlink is pre-existing state."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _signer_state_present(state: Path) -> bool:
    """Whether the signer path contains any authority residue, without changing it."""
    try:
        info = state.lstat()
    except FileNotFoundError:
        return False
    if state.is_symlink() or not stat.S_ISDIR(info.st_mode):
        return True
    try:
        next(state.iterdir())
    except StopIteration:
        return False
    return True


def _sthambha_authority_present(root: Path, state: Path) -> bool:
    """Whether ~/.sthambha contains anything except an empty signer directory."""
    try:
        info = root.lstat()
    except FileNotFoundError:
        return False
    if root.is_symlink() or not stat.S_ISDIR(info.st_mode):
        return True
    for entry in root.iterdir():
        if entry != state or _signer_state_present(entry):
            return True
    return False


def _established_identity(home: Path, state: Path, net: Path, person_path: Path,
                          node_key: Path) -> dict:
    """Read and validate an established node without changing a single file."""
    from sthambha import signer as S

    person_pub_path, node_pub_path, agent_path = (
        state / "person.pub", state / "node.pub", net / "agent")
    missing_identity = [str(path) for path in (person_pub_path, node_pub_path, agent_path)
                        if not path.exists()]
    if missing_identity:
        command = _delegate_command(state, person_path, AGENT, AGENT_CAPS, reuse_key=False)
        raise SystemExit("established node identity is incomplete; refusing automatic repair and leaving all "
                         "files untouched. Restore the missing identity file(s): " +
                         ", ".join(missing_identity) + ". Human recovery must name the authority explicitly: " +
                         command)
    try:
        agent = S.validate_agent_name(agent_path.read_text().strip())
        person_pub = person_pub_path.read_text().strip()
        node_pub = node_pub_path.read_text().strip()
        if not node_key.exists():
            raise ValueError(f"node key {node_key} is missing")
        if S._node_pub_from_key(node_key) != node_pub:  # noqa: SLF001 — signer owns this format
            raise ValueError("the node key does not match signer node.pub")
        if person_path.exists():
            _, stored_person_pub = S.load_key(person_path)
            if stored_person_pub != person_pub:
                raise ValueError("the TEST custody person key does not match signer person.pub")
    except (OSError, UnicodeError, ValueError, S.SignerError) as exc:
        raise SystemExit(f"established node identity is inconsistent; refusing automatic repair and leaving all "
                         f"files untouched: {exc}") from exc

    try:
        agent_files = S.open_agent_files(state, agent)
    except (OSError, S.SignerError) as exc:
        raise SystemExit("established node identity is inconsistent; refusing automatic repair and leaving all "
                         f"files untouched: {exc}") from exc
    with agent_files:
        key_exists, grant_exists = agent_files.key_exists(), agent_files.grant_exists()
        grant = None
        caps = list(AGENT_CAPS)
        if grant_exists:
            try:
                grant = agent_files.load_grant()
                if isinstance(grant.get("caps"), list) and all(isinstance(cap, str) for cap in grant["caps"]):
                    caps = grant["caps"]
            except (OSError, ValueError, AttributeError, S.SignerError):
                grant = None

        problem = ""
        if not key_exists and not grant_exists:
            problem = "both the agent key and grant are missing"
        elif not key_exists:
            problem = "the agent key is missing while its grant remains"
        elif not grant_exists:
            problem = "the agent grant is missing while its key remains"
        else:
            try:
                _, agent_pub = agent_files.load_key()
                if not isinstance(grant, dict):
                    raise ValueError("the grant is not valid JSON")
                if grant.get("schema") != S.GRANT_SCHEMA:
                    raise ValueError("the grant schema is invalid")
                if grant.get("subject") != agent:
                    raise ValueError("the grant names a different agent")
                if grant.get("subject_pub") != agent_pub:
                    raise ValueError("the grant is bound to a different agent key")
                if grant.get("node") not in ("*", node_pub):
                    raise ValueError("the grant is bound to a different node")
                if not S._verify(grant, person_pub):  # noqa: SLF001 — structural validation, not time validity
                    raise ValueError("the grant signature does not verify against signer person.pub")
            except (OSError, UnicodeError, ValueError, S.SignerError) as exc:
                problem = str(exc)

    if problem:
        # Reusing a surviving key is safe only when the grant is the missing half. Other structural
        # mismatches need an explicit human key rotation, which omits --reuse-key deliberately.
        reuse_key = key_exists and not grant_exists
        command = _delegate_command(state, person_path, agent, caps, reuse_key=reuse_key)
        raise SystemExit(f"established node delegation for agent {agent!r} is incomplete or mismatched "
                         f"({problem}); refusing to mint, rotate, or rename it and leaving all files untouched. "
                         f"Human recovery must name the authority explicitly: {command}")

    now = int(time.time())
    expires = int(grant.get("expires_at", 0))
    missing = sorted(set(AGENT_CAPS) - set(grant.get("caps") or []))
    notices = []
    if expires <= now:
        notices.append("grant expired")
    elif expires < now + 7 * 86400:
        notices.append("grant expires in under 7 days")
    if missing:
        notices.append("grant intentionally remains narrower than join defaults "
                       f"(missing {', '.join(missing)})")
    agent_result = f"'{agent}' kept"
    if notices:
        command = "nak grant renew"
        if missing:
            widened = sorted(set(grant.get("caps") or []) | set(AGENT_CAPS))
            command += " --caps " + ",".join(widened)
        agent_result += f"; {'; '.join(notices)}; human renewal only: {command}"
    return {
        "node_key": "kept",
        "person_key": ("kept" if person_path.exists() else "not on this box (human custody kept)"),
        "signer": "kept",
        "agent": agent_result,
        "person": person_pub,
        "node": node_pub,
    }


def ensure_identity(home: Path) -> dict:
    """Steps 1-4. Create the agent delegation once; later authority changes stay human-run."""
    from sthambha import signer as S
    try:
        agent = S.validate_agent_name(AGENT)
    except S.SignerError as exc:
        raise SystemExit(f"refusing invalid join agent {AGENT!r}: {exc}") from exc
    # This is the authority boundary. Decide it before mkdir/chmod/key generation or any other write.
    state = home / ".sthambha" / "signer"
    net = home / ".nakshatra" / "net"
    person_path = home / ".sthambha" / "person-TEST.key"
    node_key = home / ".nakshatra" / "keys" / "worker.ed25519"
    try:
        # Snapshot every path used to classify this run before taking either branch. The only
        # pre-existing key first-init may consume is worker.ed25519; validate it before any write.
        sthambha_authority_present = _sthambha_authority_present(home / ".sthambha", state)
        net_agent_present = _path_present(net / "agent")
        net_store_present = _path_present(net / "net.sqlite")
        node_key_present = _path_present(node_key)
    except OSError as exc:
        raise SystemExit(f"cannot inspect existing identity state; leaving all files untouched: {exc}") from exc
    first_initialisation = not (sthambha_authority_present or net_agent_present or net_store_present)
    if not first_initialisation:
        return _established_identity(home, state, net, person_path, node_key)
    if not callable(getattr(S, "load_or_create_lease_key", None)):
        # Compatibility must be decided before first-init creates any identity or authority state.
        raise SystemExit(OLD_SIGNER_MESSAGE)

    node_pub = None
    if node_key_present:
        try:
            node_pub = S._node_pub_from_key(node_key)  # noqa: SLF001 — validate before first-init writes
        except (OSError, ValueError, S.SignerError) as exc:
            raise SystemExit(f"pre-existing worker key is invalid; leaving all files untouched: {exc}") from exc

    done = {}
    if not node_key_present:
        node_key.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(str(node_key), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))                 # a raw Ed25519 seed, the format nakd/meshd read
        node_pub = S._node_pub_from_key(node_key)  # noqa: SLF001 — the signer's own reader for this file
        done["node_key"] = "created"
    else:
        done["node_key"] = "kept"
    person_priv = None
    if not person_path.exists() and (state / "person.pub").exists():
        person_pub = (state / "person.pub").read_text().strip()
        done["person_key"] = "not on this box (human custody kept)"
    elif not person_path.exists():
        priv, _ = S.new_key()
        S.save_key(person_path, priv)
        done["person_key"] = "created (TEST custody: a file on this machine)"
        person_priv, person_pub = S.load_key(person_path)
    else:
        done["person_key"] = "kept"
        person_priv, person_pub = S.load_key(person_path)
    try:
        agent_files = S.open_agent_files(state, agent, create=True)
    except (OSError, S.SignerError) as exc:
        raise SystemExit(f"cannot open confined signer agent state: {exc}") from exc
    with agent_files:
        os.chmod(state, 0o700)
        if not (state / "person.pub").exists():
            (state / "node.pub").write_text(node_pub)
            (state / "person.pub").write_text(person_pub)
            done["signer"] = "set up"
        else:
            if ((state / "person.pub").read_text().strip() != person_pub or
                    (state / "node.pub").read_text().strip() != node_pub):
                raise SystemExit("the signer here belongs to a different person or node key; refusing to mix them")
            done["signer"] = "kept"
        key_exists, grant_exists = agent_files.key_exists(), agent_files.grant_exists()
        if not key_exists and not grant_exists:
            if person_priv is None:
                raise SystemExit(f"agent {agent!r} has no delegation and the person key is not on this box; "
                                 "create it with an explicit human delegation")
            agent_priv, agent_pub = S.new_key()
            _lease_priv, lease_pub = S.load_or_create_lease_key(state)
            grant = S.issue_grant(
                person_priv, agent, agent_pub, AGENT_CAPS, node=node_pub, ttl_s=AGENT_TTL_S,
                lease={"issuer_pub": lease_pub, "max_s": AGENT_LEASE_S, "idle_s": AGENT_IDLE_S})
            agent_files.write_key_and_grant(agent_priv, grant)
            done["agent"] = (f"'{agent}' may {', '.join(AGENT_CAPS)} under a 365-day charter; "
                             "active while used and lapses after 30 idle days")
        elif key_exists != grant_exists:
            raise SystemExit(f"agent {agent!r} has only one of its key/grant files; refusing automatic repair. "
                             "Restore the missing signer-state file or delegate a replacement explicitly.")
        else:
            grant = agent_files.load_grant()
            now = int(time.time())
            expires = int(grant.get("expires_at", 0))
            missing = sorted(set(AGENT_CAPS) - set(grant.get("caps") or []))
            notices = []
            if expires <= now:
                notices.append("grant expired")
            elif expires < now + 7 * 86400:
                notices.append("grant expires in under 7 days")
            if missing:
                notices.append("grant intentionally remains narrower than join defaults "
                               f"(missing {', '.join(missing)})")
            if notices:
                command = "nak grant renew"
                if missing:
                    widened = sorted(set(grant.get("caps") or []) | set(AGENT_CAPS))
                    command += " --caps " + ",".join(widened)
                done["agent"] = f"'{agent}' kept; {'; '.join(notices)}; human renewal only: {command}"
            else:
                done["agent"] = f"'{agent}' kept"
    net.mkdir(parents=True, exist_ok=True, mode=0o700)
    agent_path = net / "agent"
    if not agent_path.exists():
        fd = os.open(agent_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as out:
            out.write(agent + "\n")
    os.chmod(agent_path, 0o600)
    done["person"], done["node"] = person_pub, node_pub
    return done


def write_identity_gated_units(home: Path) -> None:
    """Re-run the current signed release's unit writer after join creates its opt-in files."""
    prefix = Path(os.environ.get("NAK_NODE_PREFIX", home / ".nakshatra-node"))
    manifest_path = prefix / "current" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        shipped = manifest["installer"]
        if not isinstance(shipped, str) or not shipped:
            raise ValueError("missing installer path")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not find the current signed-release installer: {exc}") from exc
    installer = prefix / "current" / shipped
    if not installer.is_file():
        raise SystemExit(f"the current signed-release installer is missing: {installer}")
    try:
        result = subprocess.run(
            [sys.executable, str(installer), "--prefix", str(prefix), "write-units"],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"could not write services after identity setup: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-500:] or f"exit status {result.returncode}"
        raise SystemExit(f"could not write services after identity setup: {detail}")


def start_services(sock: Path, timeout_s: float = 90) -> None:
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "--now", "nak-signer.service", "nak-net.service"], check=False)
    subprocess.run(["systemctl", "--user", "restart", "nak-net.service"], check=False)
    from network.client import NakClient, NakError
    end = time.time() + timeout_s
    while time.time() < end:                        # a leftover socket FILE proves nothing: wait for an answer
        try:
            NakClient(sock, timeout=5).status()
            return
        except NakError:
            time.sleep(1)
    raise SystemExit("the messaging service did not come up; see: journalctl --user -u nak-net -n 30")


def _mcp_command() -> str:
    """Return the command the user's host-side AI client can actually execute."""
    return "nakc mcp" if os.environ.get("NAK_CONTAINER") == "1" else "~/.local/bin/nak-mcp"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Nakshatra join setup (run by install.py join)")
    ap.add_argument("--invite", required=True)
    ap.add_argument("--name", default="friend")
    ap.add_argument("--home", type=Path, default=Path.home())
    a = ap.parse_args(argv)
    print("setting up this node:")
    done = ensure_identity(a.home)
    for k in ("node_key", "person_key", "signer", "agent"):
        _say(f"{k.replace('_', ' ')}: {done[k]}")
    sock = a.home / ".nakshatra" / "net" / "nakd.sock"
    if os.environ.get("NAK_JOIN_NO_SYSTEMD") != "1":
        write_identity_gated_units(a.home)
        start_services(sock)
        _say("services: nak-signer + nak-net running (they start again after a reboot)")
    from network.client import NakClient, NakError
    try:
        r = NakClient(sock).redeem(a.invite, nickname=a.name[:40], wait_s=30)
    except NakError as e:
        raise SystemExit(f"could not send the connection request: {e}")
    print(f"\nconnection request: {r['state']}. Your friend appears here as '{r.get('name') or '?'}'.")
    import joincode
    try:
        inv = joincode.decode_invite(a.invite)
    except ValueError as e:
        raise SystemExit(f"invite became invalid during setup: {e}")
    if inv.get("kind", "contact") == "worker":
        retry_path = _store_worker_invite(a.home, a.invite, a.name)
        try:
            standing = redeem_worker_invite(a.home, a.invite, a.name)
        except WorkerRedeemError as e:
            print(f"\nworker redeem failed: {e}", file=sys.stderr)
            print(f"contact setup is complete. The worker invite remains at {retry_path}.", file=sys.stderr)
            print("retry while its one-time code is valid:  nak worker-redeem", file=sys.stderr)
        else:
            retry_path.unlink()
            print(f"worker standing: tier {standing['tier']}, junction {standing['junction']}")
    elif inv.get("kind", "contact") == "consumer":
        try:
            standing = redeem_consumer_invite(a.home, a.invite, a.name)
        except WorkerRedeemError as e:
            print(f"\nconsumer redeem failed: {e}", file=sys.stderr)
            print("contact setup is complete; paste the signed join line again while its code is valid.",
                  file=sys.stderr)
        else:
            starter = int(inv["consumer"].get("starter", 0))
            suffix = f"; starter request {starter} pending hub transfer" if starter else ""
            print(f"consumer standing: tier {standing['tier']}, operator {standing['operator']}{suffix}")
    print("\nnext:")
    print("  1. ask your friend to accept you (they run: nak requests, then nak accept <id>)")
    print("  2. check:  nak contacts      (shows them as active once they accept)")
    print(f"  3. talk:   nak send '{r.get('name') or '<their name>'}' \"hello\"     ·   nak inbox")
    mcp_command = _mcp_command()
    print("  4. plug in your AI (optional):")
    argv = [os.path.expanduser(part) for part in shlex.split(mcp_command)]
    openclaw_entry = json.dumps({"command": argv[0], "args": argv[1:]}, separators=(",", ":"))
    print(f"       OpenClaw: openclaw mcp set nakshatra {shlex.quote(openclaw_entry)} && openclaw config set tools.toolSearch false")
    print(f"       Hermes:   add an mcp_servers entry with command {mcp_command}")
    if (a.home / ".sthambha" / "person-TEST.key").exists():
        custody_note = "your person key is a TEST file at ~/.sthambha/person-TEST.key: back it up, and never share it"
    else:
        custody_note = "your person key remains in human custody off this box"
    print(f"\nyou are {done['person'][:16]}… on node {done['node'][:16]}… — {custody_note}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
