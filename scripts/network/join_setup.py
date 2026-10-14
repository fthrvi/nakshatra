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
AGENT_TTL_S = 90 * 86400
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


def ensure_identity(home: Path) -> dict:
    """Steps 1-4. Returns what was created vs kept."""
    from sthambha import signer as S
    done = {}
    node_key = home / ".nakshatra" / "keys" / "worker.ed25519"
    if not node_key.exists():
        node_key.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(str(node_key), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))                 # a raw Ed25519 seed, the format nakd/meshd read
        done["node_key"] = "created"
    else:
        done["node_key"] = "kept"
    person_path = home / ".sthambha" / "person-TEST.key"
    if not person_path.exists():
        priv, _ = S.new_key()
        S.save_key(person_path, priv)
        done["person_key"] = "created (TEST custody: a file on this machine)"
    else:
        done["person_key"] = "kept"
    person_priv, person_pub = S.load_key(person_path)
    state = home / ".sthambha" / "signer"
    (state / "agents").mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state, 0o700)
    node_pub = S._node_pub_from_key(node_key)  # noqa: SLF001 — the signer's own reader for this file
    if not (state / "person.pub").exists():
        (state / "node.pub").write_text(node_pub)
        (state / "person.pub").write_text(person_pub)
        done["signer"] = "set up"
    else:
        if (state / "person.pub").read_text().strip() != person_pub or (state / "node.pub").read_text().strip() != node_pub:
            raise SystemExit("the signer here belongs to a different person or node key; refusing to mix them")
        done["signer"] = "kept"
    net = home / ".nakshatra" / "net"
    agent = AGENT
    if (net / "agent").exists():                    # an existing node already signs as some agent: keep it
        agent = (net / "agent").read_text().strip() or AGENT
    grant_path = state / "agents" / f"{agent}.grant.json"
    fresh = False
    if grant_path.exists():
        g = json.loads(grant_path.read_text())
        fresh = g.get("expires_at", 0) > time.time() + 7 * 86400 and set(AGENT_CAPS) <= set(g.get("caps", []))
    if not fresh:
        agent_priv, agent_pub = S.new_key()
        S.save_key(state / "agents" / f"{agent}.key", agent_priv)
        grant = S.issue_grant(person_priv, agent, agent_pub, AGENT_CAPS, node=node_pub, ttl_s=AGENT_TTL_S)
        grant_path.write_text(json.dumps(grant, indent=1))
        done["agent"] = f"'{agent}' may {', '.join(AGENT_CAPS)} for 90 days"
    else:
        done["agent"] = f"'{agent}' kept"
    net.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not (net / "agent").exists():
        (net / "agent").write_text(agent + "\n")
    done["person"], done["node"] = person_pub, node_pub
    return done


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
    print("  4. plug in your AI (optional):")
    print("       OpenClaw: openclaw mcp add nakshatra --command ~/.local/bin/nak-mcp && openclaw config set tools.toolSearch false")
    print("       Hermes:   add an mcp_servers entry with command ~/.local/bin/nak-mcp")
    print(f"\nyou are {done['person'][:16]}… on node {done['node'][:16]}… — your person key is a TEST file at "
          f"~/.sthambha/person-TEST.key: back it up, and never share it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
