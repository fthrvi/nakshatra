"""The newcomer's road: one pasted line → verified installer → keys + agent → connection request."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from test_nakd import Party, net, signer_mod, wait  # noqa: F401  (net is a fixture)

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "release"))
import install as I  # noqa: E402
import joincode  # noqa: E402
from network import join_setup, nak, nakd  # noqa: E402

REL = {"url": "http://10.42.0.1:8960", "channel": "canary", "pubkey": "e" * 64, "version": "0.6.0"}


def _invite(p, **kw):
    return joincode.encode_invite(p.person_priv, p.node.node, [{"node": p.node.node, "addrs": []}], release=REL, **kw)


def test_installer_verifies_invites_without_cryptography(net):
    a = net("inviter")
    code = _invite(a)
    inv = I.parse_invite(code)
    assert inv["release"] == REL and inv["inviter"] == a.person_pub
    body = json.loads(joincode._unb64u(code[5:]))
    body["release"]["url"] = "http://evil.example"                         # swap the download source
    forged = "nki1." + joincode._b64u(json.dumps(body).encode())
    with pytest.raises(I.InstallError, match="signature does not verify"):
        I.parse_invite(forged)
    with pytest.raises(I.InstallError, match="expired"):
        I.parse_invite(_invite(a, ttl_s=1), now=int(time.time()) + 5)
    plain = joincode.encode_invite(a.person_priv, a.node.node, [{"node": a.node.node, "addrs": []}])
    with pytest.raises(I.InstallError, match="where to get Nakshatra"):
        I.parse_invite(plain)
    with pytest.raises(I.InstallError, match="not a Nakshatra invite"):
        I.parse_invite("hello")


def test_identity_setup_is_idempotent_and_grants_only_message_and_claim(tmp_path):
    first = join_setup.ensure_identity(tmp_path)
    assert first["node_key"] == "created" and first["person_key"].startswith("created")
    assert oct((tmp_path / ".sthambha" / "signer").stat().st_mode & 0o777) == "0o700"
    grant = json.loads((tmp_path / ".sthambha" / "signer" / "agents" / "agent.grant.json").read_text())
    signer_mod.verify_grant(grant, first["person"], cap="nak.msg", node=first["node"])
    signer_mod.verify_grant(grant, first["person"], cap="nak.task.claim", node=first["node"])
    for cap in ("nak.task.post", "nak.pay"):
        with pytest.raises(signer_mod.SignerError):
            signer_mod.verify_grant(grant, first["person"], cap=cap, node=first["node"])
    again = join_setup.ensure_identity(tmp_path)
    assert again == dict(first, node_key="kept", person_key="kept", signer="kept", agent="'agent' kept")


def test_existing_node_keeps_its_own_agent(tmp_path):
    join_setup.ensure_identity(tmp_path)
    net_dir = tmp_path / ".nakshatra" / "net"
    (net_dir / "agent").write_text("bro-agent\n")       # a node that already signs as another agent
    r = join_setup.ensure_identity(tmp_path)
    assert r["agent"].startswith("'bro-agent'")
    assert (tmp_path / ".sthambha" / "signer" / "agents" / "bro-agent.grant.json").exists()


def test_join_end_to_end_request_reaches_the_inviter(net, tmp_path, monkeypatch, capsys):
    a = net("inviter")
    code = _invite(a, note="for rajesh")
    a.node.register_invite(code)
    home = tmp_path / "rajesh"
    join_setup.ensure_identity(home)                # what the installer's setup does first
    sg = signer_mod.Signer(home / ".sthambha" / "signer", custody="test")
    node_key = (home / ".nakshatra" / "keys" / "worker.ed25519").read_bytes()
    node = nakd.Node(home / ".nakshatra" / "net", node_key, sg.handle, agent="agent", relay=a.node.relay).start()
    stop = threading.Event()
    sock = home / ".nakshatra" / "net" / "nakd.sock"
    threading.Thread(target=nakd.serve_control, args=(node, sock, stop), daemon=True).start()
    assert wait(lambda: sock.exists())
    monkeypatch.setenv("NAK_JOIN_NO_SYSTEMD", "1")
    assert join_setup.main(["--invite", code, "--name", "Rajesh", "--home", str(home)]) == 0
    out = capsys.readouterr().out
    assert "request sent" in out and "nak accept" in out
    reqs = a.node.store.requests()
    assert len(reqs) == 1 and reqs[0]["nickname"] == "Rajesh"
    a.node.accept(reqs[0]["id"], petname="rajesh")
    assert wait(lambda: (node.store.contact(a.person_pub) or {}).get("state") == "active")
    stop.set(); node.stop()


def test_invite_line_checks_the_installer_fingerprint_before_running_it(tmp_path, monkeypatch):
    prefix = tmp_path / "node"
    (prefix / "current").mkdir(parents=True)
    files = {"install.py": b"print('real installer')\n", "releasekit.py": b"# kit\n"}
    shas = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    (prefix / "config.json").write_text(json.dumps({"source": "http://10.42.0.1:8960", "channel": "canary",
                                                    "pubkey": "e" * 64}))
    (prefix / "current" / "manifest.json").write_text(json.dumps({"version": "0.6.0", "bootstrap": shas}))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    release, line = nak._release_for_invite()
    assert release == REL and "http://10.42.0.1:8960/canary/0.6.0/install.py" in line and "{CODE}" in line
    check = line.split(" && ")[3]                    # the printf | sha256sum -c step, exactly as sent
    d = tmp_path / "dl"
    d.mkdir()
    for k, v in files.items():
        (d / k).write_bytes(v)
    assert subprocess.run(["bash", "-c", check], cwd=d).returncode == 0
    (d / "install.py").write_bytes(b"print('tampered')\n")
    assert subprocess.run(["bash", "-c", check], cwd=d, capture_output=True).returncode != 0
