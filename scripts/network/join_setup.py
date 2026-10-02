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
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

AGENT = "agent"
AGENT_CAPS = ("nak.msg", "nak.task.claim")
AGENT_TTL_S = 90 * 86400


def _say(msg: str) -> None:  # noqa: D401
    print(f"  · {msg}", flush=True)


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
