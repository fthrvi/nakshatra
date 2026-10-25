"""The newcomer's road: one pasted line → verified installer → keys + agent → connection request."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def _register_address(p, code):
    address = joincode.decode_address(code)
    base = f"{REL['url']}/{REL['channel']}/{REL['version']}"
    line = (f"mkdir -p ~/nak-join && cd ~/nak-join && curl -fsSO {base}/install.py "
            f"-O {base}/releasekit.py && printf '%s  install.py\\n%s  releasekit.py\\n' "
            f"{'1' * 64} {'2' * 64} | sha256sum -c --quiet && python3 install.py join '{code}'")
    ts = int(time.time())
    signature = p.person_priv.sign(
        nakd.address_control_statement("register_address", address["aid"], ts)).hex()
    p.node.register_address(code, line, ts=ts, signature=signature)


def _resign_invite(p, code, **changes):
    body = json.loads(joincode._unb64u(code[5:]))
    body.update(changes)
    body["sig"] = {"alg": "Ed25519", "keyid": p.person_pub[:16],
                   "value": base64.b64encode(p.person_priv.sign(joincode._canon(body))).decode()}
    return "nki1." + joincode._b64u(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def _identity_snapshot(home: Path) -> dict[str, tuple[int, bytes | None]]:
    return {
        path.relative_to(home).as_posix(): (path.stat().st_mode & 0o777,
                                            path.read_bytes() if path.is_file() else None)
        for path in sorted(home.rglob("*"))
    }


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
    worker = _invite(a, kind="worker", worker={
        "redeem": "https://onboard.example/base", "code": "one-time", "tier": "trusted"})
    assert I.parse_invite(worker)["worker"]["tier"] == "trusted"
    consumer = _invite(a, kind="consumer", consumer={
        "redeem": "https://onboard.example/base", "code": "one-time",
        "tier": "known", "starter": 50})
    assert I.parse_invite(consumer)["consumer"]["starter"] == 50
    with pytest.raises(I.InstallError, match="unexpected worker details"):
        I.parse_invite(_resign_invite(a, code, worker=None))
    for redeem in ("http:///missing-host", "http://host:bad", "http://host:0", "http://host:65536",
                   "http://host:"):
        with pytest.raises(I.InstallError, match="worker invite details are malformed"):
            I.parse_invite(_resign_invite(a, worker,
                                          worker={"redeem": redeem, "code": "one-time", "tier": "trusted"}))
    with pytest.raises(I.InstallError, match="not a Nakshatra invite"):
        I.parse_invite("hello")


def test_installer_accepts_nka_address_and_rejects_prefix_kind_confusion(net):
    a = net("inviter")
    code = joincode.encode_address(
        a.person_priv, a.node.node, [{"node": a.node.node, "addrs": []}],
        days=90, from_name="Biswa", release=REL)
    parsed = I.parse_invite(code)
    assert parsed["kind"] == "address" and parsed["release"] == REL
    with pytest.raises(I.InstallError, match="must use the nka1"):
        I.parse_invite("nki1." + code.removeprefix("nka1."))
    invite = _invite(a)
    with pytest.raises(I.InstallError, match="request address fields"):
        I.parse_invite("nka1." + invite.removeprefix("nki1."))


def test_genuine_first_identity_setup_mints_once_and_is_idempotent(tmp_path):
    first = join_setup.ensure_identity(tmp_path)
    assert first["node_key"] == "created" and first["person_key"].startswith("created")
    assert oct((tmp_path / ".sthambha" / "signer").stat().st_mode & 0o777) == "0o700"
    assert oct((tmp_path / ".nakshatra" / "net" / "agent").stat().st_mode & 0o777) == "0o600"
    grant = json.loads((tmp_path / ".sthambha" / "signer" / "agents" / "agent.grant.json").read_text())
    assert grant["lease"] == {
        "issuer_pub": signer_mod.load_or_create_lease_key(tmp_path / ".sthambha" / "signer")[1],
        "max_s": 7 * 86400,
        "idle_s": 30 * 86400,
    }
    assert 364 * 86400 <= grant["expires_at"] - int(time.time()) <= 365 * 86400
    signer_mod.verify_grant(grant, first["person"], cap="nak.msg", node=first["node"])
    signer_mod.verify_grant(grant, first["person"], cap="nak.task.claim", node=first["node"])
    for cap in ("nak.task.post", "nak.pay"):
        with pytest.raises(signer_mod.SignerError):
            signer_mod.verify_grant(grant, first["person"], cap=cap, node=first["node"])
    again = join_setup.ensure_identity(tmp_path)
    assert again == dict(first, node_key="kept", person_key="kept", signer="kept", agent="'agent' kept")


def test_first_identity_setup_old_signer_fails_cleanly_before_any_state_change(tmp_path, monkeypatch):
    before = _identity_snapshot(tmp_path)
    monkeypatch.delattr(signer_mod, "load_or_create_lease_key")

    with pytest.raises(SystemExit) as refused:
        join_setup.ensure_identity(tmp_path)

    assert str(refused.value) == "signer too old for sliding grants; update the node or use --no-lease"
    assert _identity_snapshot(tmp_path) == before


def test_first_identity_setup_keeps_a_preexisting_worker_key(tmp_path):
    worker_key = tmp_path / ".nakshatra" / "keys" / "worker.ed25519"
    worker_key.parent.mkdir(parents=True)
    original = bytes(range(32))
    worker_key.write_bytes(original)

    result = join_setup.ensure_identity(tmp_path)

    assert result["node_key"] == "kept"
    assert worker_key.read_bytes() == original
    assert (tmp_path / ".sthambha" / "signer" / "agents" / "agent.grant.json").is_file()


def test_first_identity_setup_accepts_an_empty_signer_directory(tmp_path):
    (tmp_path / ".sthambha" / "signer").mkdir(parents=True)

    result = join_setup.ensure_identity(tmp_path)

    assert result["signer"] == "set up"
    assert (tmp_path / ".sthambha" / "signer" / "agents" / "agent.key").is_file()


def test_person_key_alone_is_established_authority_and_is_left_untouched(tmp_path):
    person_key = tmp_path / ".sthambha" / "person-TEST.key"
    person_key.parent.mkdir(parents=True)
    person_key.write_bytes(b"person authority residue")
    before = _identity_snapshot(tmp_path)

    with pytest.raises(SystemExit) as refused:
        join_setup.ensure_identity(tmp_path)

    message = str(refused.value)
    assert "python -m sthambha.signer delegate" in message
    assert str(person_key) in message
    assert _identity_snapshot(tmp_path) == before


def test_unknown_sthambha_file_is_established_authority_and_is_left_untouched(tmp_path):
    unknown = tmp_path / ".sthambha" / "future-authority.state"
    unknown.parent.mkdir(parents=True)
    unknown.write_bytes(b"future authority residue")
    before = _identity_snapshot(tmp_path)

    with pytest.raises(SystemExit) as refused:
        join_setup.ensure_identity(tmp_path)

    assert "python -m sthambha.signer delegate" in str(refused.value)
    assert _identity_snapshot(tmp_path) == before


def test_malformed_worker_key_fails_repeatably_before_any_first_init_write(tmp_path):
    worker_key = tmp_path / ".nakshatra" / "keys" / "worker.ed25519"
    worker_key.parent.mkdir(parents=True)
    worker_key.write_bytes(b"short")
    before = _identity_snapshot(tmp_path)

    errors = []
    for _ in range(2):
        with pytest.raises(SystemExit) as refused:
            join_setup.ensure_identity(tmp_path)
        errors.append(str(refused.value))
        assert _identity_snapshot(tmp_path) == before

    assert errors[0] == errors[1]
    assert "pre-existing worker key is invalid" in errors[0]
    assert "established node identity" not in errors[0]


@pytest.mark.parametrize("present", range(32), ids=lambda value: f"subset-{value:05b}")
def test_established_authority_five_file_matrix_never_repairs_or_remints(tmp_path, present):
    join_setup.ensure_identity(tmp_path)
    state = tmp_path / ".sthambha" / "signer"
    markers = (
        state / "person.pub",                                      # P
        state / "node.pub",                                        # N
        tmp_path / ".nakshatra" / "net" / "agent",                # A
        state / "agents" / "agent.key",                            # K
        state / "agents" / "agent.grant.json",                     # G
    )
    for bit, path in enumerate(markers):
        if not present & (1 << bit):
            path.unlink()
    before = _identity_snapshot(tmp_path)

    if present == 0b11111:
        result = join_setup.ensure_identity(tmp_path)
        assert result["signer"] == "kept" and result["agent"].startswith("'agent' kept")
    else:
        with pytest.raises(SystemExit) as refused:
            join_setup.ensure_identity(tmp_path)
        assert "python -m sthambha.signer delegate" in str(refused.value)

    assert _identity_snapshot(tmp_path) == before


@pytest.mark.parametrize("residue", ["spend.sqlite", "datakey", "net.sqlite"])
def test_authority_residue_alone_fails_closed_without_writes(tmp_path, residue):
    if residue == "net.sqlite":
        path = tmp_path / ".nakshatra" / "net" / residue
    else:
        path = tmp_path / ".sthambha" / "signer" / residue
    path.parent.mkdir(parents=True)
    path.write_bytes(("residue:" + residue).encode())
    before = _identity_snapshot(tmp_path)

    with pytest.raises(SystemExit) as refused:
        join_setup.ensure_identity(tmp_path)

    assert "python -m sthambha.signer delegate" in str(refused.value)
    assert _identity_snapshot(tmp_path) == before


def test_existing_short_narrow_grant_is_preserved_byte_for_byte(tmp_path):
    first = join_setup.ensure_identity(tmp_path)
    state = tmp_path / ".sthambha" / "signer"
    key_path = state / "agents" / "agent.key"
    grant_path = state / "agents" / "agent.grant.json"
    person_priv, _ = signer_mod.load_key(tmp_path / ".sthambha" / "person-TEST.key")
    _, agent_pub = signer_mod.load_key(key_path)
    narrow = signer_mod.issue_grant(person_priv, "agent", agent_pub, ("nak.msg",), node=first["node"],
                                    ttl_s=6 * 86400)
    signer_mod.save_grant(grant_path, narrow)
    before = _identity_snapshot(tmp_path)

    result = join_setup.ensure_identity(tmp_path)

    assert _identity_snapshot(tmp_path) == before
    assert "expires in under 7 days" in result["agent"]
    assert "remains narrower" in result["agent"]
    assert "nak grant renew --caps nak.msg,nak.task.claim" in result["agent"]

    (tmp_path / ".sthambha" / "person-TEST.key").unlink()
    without_person_key = join_setup.ensure_identity(tmp_path)
    assert without_person_key["person_key"] == "not on this box (human custody kept)"
    assert not (tmp_path / ".sthambha" / "person-TEST.key").exists()
    assert key_path.read_bytes() == before[".sthambha/signer/agents/agent.key"][1]
    assert grant_path.read_bytes() == before[".sthambha/signer/agents/agent.grant.json"][1]


def test_existing_expired_grant_is_preserved_separately(tmp_path):
    first = join_setup.ensure_identity(tmp_path)
    state = tmp_path / ".sthambha" / "signer"
    key_path = state / "agents" / "agent.key"
    grant_path = state / "agents" / "agent.grant.json"
    person_priv, _ = signer_mod.load_key(tmp_path / ".sthambha" / "person-TEST.key")
    _, agent_pub = signer_mod.load_key(key_path)
    expired = signer_mod.issue_grant(person_priv, "agent", agent_pub, ("nak.msg",), node=first["node"],
                                     ttl_s=1, now=int(time.time()) - 10)
    signer_mod.save_grant(grant_path, expired)
    before = _identity_snapshot(tmp_path)

    result = join_setup.ensure_identity(tmp_path)

    assert _identity_snapshot(tmp_path) == before
    assert "grant expired" in result["agent"]
    assert "nak grant renew --caps nak.msg,nak.task.claim" in result["agent"]


def test_established_node_agent_rename_never_mints_or_renames(tmp_path):
    join_setup.ensure_identity(tmp_path)
    net_dir = tmp_path / ".nakshatra" / "net"
    (net_dir / "agent").write_text("bro-agent\n")
    before = _identity_snapshot(tmp_path)

    with pytest.raises(SystemExit) as refused:
        join_setup.ensure_identity(tmp_path)

    message = str(refused.value)
    assert "refusing to mint, rotate, or rename" in message
    assert "--agent bro-agent --caps nak.msg,nak.task.claim --ttl 31536000" in message
    assert "--reuse-key" not in message
    assert _identity_snapshot(tmp_path) == before


def test_established_node_rejects_path_shaped_agent_before_building_signer_paths(tmp_path):
    join_setup.ensure_identity(tmp_path)
    agent_path = tmp_path / ".nakshatra" / "net" / "agent"
    agent_path.write_text("../agents/agent\n")
    before = _identity_snapshot(tmp_path)

    with pytest.raises(SystemExit, match="bad agent name"):
        join_setup.ensure_identity(tmp_path)

    assert _identity_snapshot(tmp_path) == before


@pytest.mark.parametrize("bad_name", [
    "agent.name", "agent name", "agent\tname", "agent\x1fname", "agént", "a" * 65,
    "agent\x00name",
])
def test_join_rejects_noncanonical_established_agent_without_mutation(tmp_path, bad_name):
    join_setup.ensure_identity(tmp_path)
    agent_path = tmp_path / ".nakshatra" / "net" / "agent"
    agent_path.write_text(bad_name + "\n")
    before = _identity_snapshot(tmp_path)

    with pytest.raises(SystemExit, match="bad agent name"):
        join_setup.ensure_identity(tmp_path)

    assert _identity_snapshot(tmp_path) == before


@pytest.mark.parametrize("bad_name", [
    "agent.name", "agent name", "agent\tname", "agent\x1fname", "agént", "a" * 65,
    "agent\x00name",
])
def test_join_validates_configured_agent_before_first_filesystem_mutation(
        tmp_path, monkeypatch, bad_name):
    monkeypatch.setattr(join_setup, "AGENT", bad_name)

    with pytest.raises(SystemExit, match="bad agent name"):
        join_setup.ensure_identity(tmp_path)

    assert _identity_snapshot(tmp_path) == {}


def test_join_refuses_key_symlink_swapped_at_validation_open_boundary(tmp_path, monkeypatch):
    join_setup.ensure_identity(tmp_path)
    state = tmp_path / ".sthambha" / "signer"
    key_path = state / "agents" / "agent.key"
    grant_path = state / "agents" / "agent.grant.json"
    outside = tmp_path / "outside-agent.key"
    outside.write_bytes(key_path.read_bytes())
    outside_before = outside.read_bytes()
    grant_before = grant_path.read_bytes()

    def swap_key(_state, name):
        assert name == "agent"
        key_path.unlink()
        key_path.symlink_to(outside)

    monkeypatch.setattr(signer_mod, "_agent_open_hook", swap_key)
    with pytest.raises(SystemExit, match="non-regular agent state file"):
        join_setup.ensure_identity(tmp_path)

    assert outside.read_bytes() == outside_before
    assert grant_path.read_bytes() == grant_before


def test_established_node_with_key_and_grant_deleted_fails_closed(tmp_path):
    join_setup.ensure_identity(tmp_path)
    state = tmp_path / ".sthambha" / "signer"
    (state / "agents" / "agent.key").unlink()
    (state / "agents" / "agent.grant.json").unlink()
    before = _identity_snapshot(tmp_path)

    with pytest.raises(SystemExit) as refused:
        join_setup.ensure_identity(tmp_path)

    message = str(refused.value)
    expected = (f"python -m sthambha.signer delegate --state {state} --person-key "
                f"{tmp_path / '.sthambha' / 'person-TEST.key'} --agent agent "
                "--caps nak.msg,nak.task.claim --ttl 31536000")
    assert expected in message and "--reuse-key" not in message
    assert _identity_snapshot(tmp_path) == before


def test_established_node_with_only_agent_key_missing_names_caps_without_reuse(tmp_path):
    join_setup.ensure_identity(tmp_path)
    state = tmp_path / ".sthambha" / "signer"
    grant_path = state / "agents" / "agent.grant.json"
    grant = json.loads(grant_path.read_text())
    grant["caps"] = ["nak.msg"]
    grant_path.write_text(json.dumps(grant))
    (state / "agents" / "agent.key").unlink()
    before = _identity_snapshot(tmp_path)

    with pytest.raises(SystemExit) as refused:
        join_setup.ensure_identity(tmp_path)

    message = str(refused.value)
    assert "agent key is missing while its grant remains" in message
    assert "--caps nak.msg --ttl 31536000" in message
    assert "--reuse-key" not in message
    assert _identity_snapshot(tmp_path) == before


def test_established_node_with_only_grant_missing_reuses_surviving_key(tmp_path):
    join_setup.ensure_identity(tmp_path)
    state = tmp_path / ".sthambha" / "signer"
    (state / "agents" / "agent.grant.json").unlink()
    before = _identity_snapshot(tmp_path)

    with pytest.raises(SystemExit) as refused:
        join_setup.ensure_identity(tmp_path)

    message = str(refused.value)
    assert "agent grant is missing while its key remains" in message
    assert "--caps nak.msg,nak.task.claim --ttl 31536000 --reuse-key" in message
    assert _identity_snapshot(tmp_path) == before


def test_join_end_to_end_request_reaches_the_inviter(net, tmp_path, monkeypatch, capsys):
    a = net("inviter")
    code = _invite(a, note="for rajesh", from_name="Biswa")
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
    assert "request sent" in out and "nak accept" in out and "appears here as 'Biswa'" in out
    assert "~/.local/bin/nak-mcp" in out and "nakc mcp" not in out
    c = node.store.contact(a.person_pub)
    assert c["petname"] == "Biswa" and c["name_src"] == "self-chosen"          # NOT the note "for rajesh"
    reqs = a.node.store.requests()
    assert len(reqs) == 1 and reqs[0]["nickname"] == "Rajesh"
    a.node.accept(reqs[0]["id"], petname="rajesh")
    assert wait(lambda: (node.store.contact(a.person_pub) or {}).get("state") == "active")
    stop.set(); node.stop()


def test_join_setup_address_path_sends_request_and_prints_person_boundary(net, tmp_path, monkeypatch, capsys):
    a = net("inviter")
    code = joincode.encode_address(
        a.person_priv, a.node.node, [{"node": a.node.node, "addrs": []}],
        days=90, from_name="Biswa", release=REL)
    _register_address(a, code)
    home = tmp_path / "stranger"
    join_setup.ensure_identity(home)
    sg = signer_mod.Signer(home / ".sthambha" / "signer", custody="test")
    node_key = (home / ".nakshatra" / "keys" / "worker.ed25519").read_bytes()
    node = nakd.Node(home / ".nakshatra" / "net", node_key, sg.handle, agent="agent",
                     relay=a.node.relay).start()
    stop = threading.Event()
    sock = home / ".nakshatra" / "net" / "nakd.sock"
    threading.Thread(target=nakd.serve_control, args=(node, sock, stop), daemon=True).start()
    assert wait(sock.exists)
    monkeypatch.setenv("NAK_JOIN_NO_SYSTEMD", "1")
    try:
        assert join_setup.main(["--invite", code, "--name", "Unknown", "--home", str(home)]) == 0
        out = capsys.readouterr().out
        assert "your request was sent; nothing is shared until Biswa accepts" in out
        assert wait(lambda: len(a.node.store.requests()) == 1)
        request = a.node.store.requests()[0]
        assert request["nickname"] == "Unknown" and request["source"] == "address"
    finally:
        stop.set()
        node.stop()


@pytest.mark.parametrize("container", [False, True], ids=["host", "container"])
def test_fresh_join_writes_and_enables_identity_gated_units(tmp_path, monkeypatch, container):
    """First join must revisit units skipped before the identity opt-ins existed."""
    from network import client as network_client

    home = tmp_path / "home"
    prefix = tmp_path / "node"
    units = home / ".config" / "systemd" / "user"
    release = prefix / "releases" / "0.1.0"
    installer = release / "nakshatra" / "release" / "install.py"
    installer.parent.mkdir(parents=True)
    shutil.copy2(_REPO / "release" / "install.py", installer)
    shutil.copy2(_REPO / "release" / "releasekit.py", installer.parent / "releasekit.py")
    (prefix / "current").symlink_to("releases/0.1.0")
    spec = json.loads((_REPO / "release" / "spec.json").read_text())
    manifest = {
        "version": "0.1.0",
        "installer": "nakshatra/release/install.py",
        "components": spec["components"],
        "services": {name: spec["services"][name] for name in ("nak-signer", "nak-net")},
    }
    (release / "manifest.json").write_text(json.dumps(manifest))
    (prefix / "config.json").write_text(json.dumps({"current": "0.1.0"}))
    home.mkdir()
    systemctl_log = tmp_path / "systemctl.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_systemctl = fake_bin / "systemctl"
    fake_systemctl.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$NAK_TEST_SYSTEMCTL_LOG\"\n"
    )
    fake_systemctl.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    monkeypatch.setenv("NAK_TEST_SYSTEMCTL_LOG", str(systemctl_log))
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.delenv("NAK_JOIN_NO_SYSTEMD", raising=False)
    if container:
        monkeypatch.setenv("NAK_CONTAINER", "1")
    else:
        monkeypatch.delenv("NAK_CONTAINER", raising=False)

    install = I.Installer(prefix, systemd=False, make_venv=False, unit_dir=units)
    assert install.write_units(manifest) == []
    assert not (units / "nak-signer.service").exists()
    assert not (units / "nak-net.service").exists()

    class Client:
        status_calls = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def status(self):
            type(self).status_calls += 1
            return {"ok": True}

        def redeem(self, *_args, **_kwargs):
            return {"state": "request sent", "name": "Inviter"}

    monkeypatch.setattr(network_client, "NakClient", Client)
    monkeypatch.setattr(joincode, "decode_invite", lambda _code: {"kind": "contact"})

    assert join_setup.main(["--invite", "nki1.test", "--name", "Newcomer", "--home", str(home)]) == 0

    assert (units / "nak-signer.service").is_file()
    assert (units / "nak-net.service").is_file()
    assert I.UNIT_MARKER in (units / "nak-signer.service").read_text()
    # The node's settlement backend (NAKSHATRA_ESCROW=neuron + pinned ledger) is the operator's
    # choice, written to net.env; without this hook a container node could never fund a Neuron hold.
    assert f"EnvironmentFile=-{home}/.nakshatra/net.env" in (units / "nak-net.service").read_text()
    calls = systemctl_log.read_text().splitlines()
    assert "--user enable --now nak-signer.service nak-net.service" in calls
    assert "--user restart nak-net.service" in calls
    assert Client.status_calls == 1


def test_join_next_steps_use_the_host_nakc_wrapper_inside_container(monkeypatch):
    monkeypatch.setenv("NAK_CONTAINER", "1")
    assert join_setup._mcp_command() == "nakc mcp"

    monkeypatch.delenv("NAK_CONTAINER")
    assert join_setup._mcp_command() == "~/.local/bin/nak-mcp"


def test_invite_line_checks_the_installer_fingerprint_before_running_it(tmp_path, monkeypatch):
    prefix = tmp_path / "node"
    (prefix / "current").mkdir(parents=True)
    files = {"install.py": b"print('real installer')\n", "releasekit.py": b"# kit\n"}
    shas = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    (prefix / "config.json").write_text(json.dumps({"source": "http://10.42.0.1:8960", "channel": "canary",
                                                    "pubkey": "e" * 64}))
    (prefix / "current" / "manifest.json").write_text(json.dumps({"version": "0.6.0", "bootstrap": shas}))
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    monkeypatch.setenv("NAK_NET_DIR", str(tmp_path / "no-net"))   # never read the real ~/.nakshatra/net
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


def test_invite_prefers_the_public_release_host(tmp_path, monkeypatch):
    prefix = tmp_path / "node"
    (prefix / "current").mkdir(parents=True)
    (prefix / "config.json").write_text(json.dumps({"source": "http://10.42.0.1:8960", "channel": "canary",
                                                    "pubkey": "e" * 64}))
    (prefix / "current" / "manifest.json").write_text(json.dumps({"version": "0.6.3", "bootstrap": {
        "install.py": "a" * 64, "releasekit.py": "b" * 64}}))
    net_dir = tmp_path / "net"
    net_dir.mkdir()
    (net_dir / "release-url").write_text("http://45.63.109.137:8960\n")
    monkeypatch.setenv("NAK_NODE_PREFIX", str(prefix))
    monkeypatch.setenv("NAK_NET_DIR", str(net_dir))
    release, line = nak._release_for_invite()
    assert release["url"] == "http://45.63.109.137:8960" and "45.63.109.137:8960/canary/0.6.3/install.py" in line


def test_nak_worker_invite_stages_code_with_injected_http(tmp_path, monkeypatch, capsys):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    person = ed25519.Ed25519PrivateKey.generate()
    person_path = tmp_path / "person.key"
    person_path.write_text(person.private_bytes_raw().hex())
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NAK_NET_DIR", str(tmp_path / "net"))
    monkeypatch.setenv("NAK_ONBOARD_TOKEN", "operator-secret")
    monkeypatch.setattr(nak, "_release_for_invite", lambda _url="": (REL, ""))
    registered = []

    def fake_call(_sock, req):
        if req["op"] == "status":
            return {"node": "a" * 64, "relay": "relay.example:51820"}
        registered.append(req["code"])
        return {}

    posted = []

    def fake_post(url, body):
        posted.append((url, body))
        return {"code": "staged-once", "expires_at": 1600}

    monkeypatch.setattr(nak, "call", fake_call)
    monkeypatch.setattr(nak.time, "time", lambda: 1000)
    assert nak.main(["invite", "--person-key", str(person_path), "--kind", "worker", "--tier", "known",
                     "--from", "Alice",
                     "--redeem", "https://onboard.example/base", "--code-only"], http_post=fake_post) == 0
    output = capsys.readouterr()
    code = output.out.strip()
    inv = joincode.decode_invite(code, now=1001)
    assert posted == [("https://onboard.example/base/worker-stage",
                       {"token": "operator-secret", "tier": "known"})]
    assert inv["worker"] == {"redeem": "https://onboard.example/base", "code": "staged-once", "tier": "known"}
    assert inv["expires_at"] == 1600 and "Unix 1600" in output.err
    assert registered == [code]
    assert oct((tmp_path / "net" / "name").stat().st_mode & 0o777) == "0o600"


def test_nak_consumer_invite_stages_signed_starter_request(tmp_path, monkeypatch, capsys):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    person = ed25519.Ed25519PrivateKey.generate()
    person_path = tmp_path / "person.key"
    person_path.write_text(person.private_bytes_raw().hex())
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NAK_NET_DIR", str(tmp_path / "net"))
    monkeypatch.setenv("NAK_ONBOARD_TOKEN", "operator-secret")
    monkeypatch.setattr(nak, "_release_for_invite", lambda _url="": (REL, ""))
    registered = []

    def fake_call(_sock, req):
        if req["op"] == "status":
            return {"node": "a" * 64, "relay": "relay.example:51820"}
        registered.append(req["code"])
        return {}

    posted = []

    def fake_post(url, body):
        posted.append((url, body))
        return {"code": "consumer-once", "expires_at": 1600}

    monkeypatch.setattr(nak, "call", fake_call)
    monkeypatch.setattr(nak.time, "time", lambda: 1000)
    assert nak.main(["invite", "--person-key", str(person_path), "--kind", "consumer",
                     "--tier", "known", "--starter", "1234",
                     "--redeem", "https://onboard.example/base", "--code-only"],
                    http_post=fake_post) == 0
    output = capsys.readouterr()
    code = output.out.strip()
    inv = joincode.decode_invite(code, now=1001)
    assert posted == [("https://onboard.example/base/consumer-stage", {
        "token": "operator-secret", "tier": "known", "starter": 1234,
        "inviter_pubkey": "a" * 64,
    })]
    assert inv["consumer"] == {"redeem": "https://onboard.example/base",
                               "code": "consumer-once", "tier": "known", "starter": 1234}
    assert inv["expires_at"] == 1600 and "consumer invite expires" in output.err
    assert registered == [code]


def test_nak_member_invite_posts_node_signed_stranger_envelope(tmp_path, monkeypatch, capsys):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    person = ed25519.Ed25519PrivateKey.generate()
    person_path = tmp_path / "person.key"
    person_path.write_text(person.private_bytes_raw().hex())
    node = ed25519.Ed25519PrivateKey.generate()
    node_pub = node.public_key().public_bytes_raw().hex()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NAK_NET_DIR", str(tmp_path / "net"))
    monkeypatch.delenv("NAK_ONBOARD_TOKEN", raising=False)
    monkeypatch.setattr(nak, "_release_for_invite", lambda _url="": (REL, ""))
    registered = []

    def fake_call(_sock, req):
        if req["op"] == "status":
            return {"node": node_pub, "relay": "relay.example:51820"}
        if req["op"] == "node_sign_stage":
            body = {**req["body"], "nonce": "01" * 16}
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False).encode()
            return {**body, "sig": node.sign(canonical).hex(), "ok": True}
        registered.append(req["code"])
        return {}

    posted = []

    def fake_post(url, body):
        posted.append((url, body))
        unsigned = {key: value for key, value in body.items() if key != "sig"}
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode()
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(body["inviter_pubkey"])).verify(
            bytes.fromhex(body["sig"]), canonical)
        return {"code": "member-worker", "expires_at": 1600}

    monkeypatch.setattr(nak, "call", fake_call)
    monkeypatch.setattr(nak.time, "time", lambda: 1000)
    assert nak.main(["invite", "--person-key", str(person_path), "--kind", "worker",
                     "--tier", "stranger", "--from", "Alice",
                     "--redeem", "https://onboard.example/base", "--code-only"],
                    http_post=fake_post) == 0
    code = capsys.readouterr().out.strip()
    assert posted[0][0] == "https://onboard.example/base/worker-stage"
    envelope = posted[0][1]
    assert set(envelope) == {"v", "purpose", "nonce", "kind", "tier", "starter",
                             "inviter_pubkey", "ttl", "ts", "sig"}
    assert {key: envelope[key] for key in ("v", "purpose", "nonce", "kind", "tier", "starter",
                                            "inviter_pubkey", "ttl", "ts")} == {
        "v": 2, "purpose": "onboard-stage-v2", "kind": "worker",
        "nonce": "01" * 16, "tier": "stranger", "starter": 0,
        "inviter_pubkey": node_pub, "ttl": 86400, "ts": 1000,
    }
    assert registered == [code]


def test_node_sign_stage_only_signs_exact_domain_separated_envelopes(net, monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ed25519

    inviter = net("signing-oracle")
    now = int(time.time())
    stage = {
        "v": 2, "purpose": "onboard-stage-v2", "kind": "consumer", "tier": "stranger",
        "starter": 10_000_000, "inviter_pubkey": inviter.node.node, "ttl": 86400, "ts": now,
    }
    worker_stage = {**stage, "kind": "worker", "starter": 0}
    cancel = {
        "v": 2, "purpose": "onboard-cancel-v2", "code": "valid_Code-2",
        "inviter_pubkey": inviter.node.node, "ts": now,
    }
    nonces = iter(("01" * 16, "02" * 16, "03" * 16))
    monkeypatch.setattr(nakd.secrets, "token_hex", lambda size: next(nonces) if size == 16 else None)
    public = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(inviter.node.node))
    expected_keys = (
        {"v", "purpose", "nonce", "kind", "tier", "starter", "inviter_pubkey", "ttl", "ts", "sig"},
        {"v", "purpose", "nonce", "kind", "tier", "starter", "inviter_pubkey", "ttl", "ts", "sig"},
        {"v", "purpose", "nonce", "code", "inviter_pubkey", "ts", "sig"},
    )
    for body, keys in zip((stage, worker_stage, cancel), expected_keys):
        signed = inviter.node.node_sign_stage(body)
        assert set(signed) == keys
        sig = signed.pop("sig")
        public.verify(bytes.fromhex(sig), json.dumps(
            signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        assert {key: value for key, value in signed.items() if key != "nonce"} == body

    refused = [
        {**stage, "tier": "trusted"},
        {**stage, "starter": 10_000_001},
        {**stage, "extra": "field"},
        {**stage, "purpose": "onboard-cancel-v2"},
        {**stage, "ts": now - 121},
        {**stage, "kind": "worker", "starter": 1},
        {**stage, "nonce": "04" * 16},
        {**cancel, "code": "not/a/code"},
        {**cancel, "purpose": "onboard-stage-v2"},
        {**cancel, "nonce": "05" * 16},
    ]
    for body in refused:
        with pytest.raises(ValueError, match="onboard signing request"):
            inviter.node.node_sign_stage(body)


def test_node_sign_stage_generates_a_fresh_nonce_for_each_call(net, monkeypatch):
    inviter = net("signing-nonce")
    body = {
        "v": 2, "purpose": "onboard-stage-v2", "kind": "worker", "tier": "stranger",
        "starter": 0, "inviter_pubkey": inviter.node.node, "ttl": 86400,
        "ts": int(time.time()),
    }
    generated = iter(("11" * 16, "22" * 16))
    sizes = []

    def token_hex(size):
        sizes.append(size)
        return next(generated)

    monkeypatch.setattr(nakd.secrets, "token_hex", token_hex)

    first = inviter.node.node_sign_stage(body)
    second = inviter.node.node_sign_stage(body)

    assert first["nonce"] == "11" * 16
    assert second["nonce"] == "22" * 16
    assert first["nonce"] != second["nonce"]
    assert sizes == [16, 16]
    assert "nonce" not in body


def test_member_stage_refuses_non_stranger_and_large_starter_before_signing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NAK_ONBOARD_TOKEN", raising=False)
    signed = []
    with pytest.raises(SystemExit, match="tier stranger only"):
        nak._stage_auth("worker", "known", 0, 86400, signed.append)
    with pytest.raises(SystemExit, match="may not exceed 10000000"):
        nak._stage_auth("consumer", "stranger", 10_000_001, 86400, signed.append)
    assert signed == []


def test_member_consumer_stage_includes_capped_starter_in_signed_envelope(tmp_path, monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    node = ed25519.Ed25519PrivateKey.generate()
    node_pub = node.public_key().public_bytes_raw().hex()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NAK_ONBOARD_TOKEN", raising=False)

    def sign(body):
        body = {**body, "inviter_pubkey": node_pub, "nonce": "06" * 16}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode()
        return {**body, "sig": node.sign(canonical).hex(), "ok": True}

    monkeypatch.setattr(nak.time, "time", lambda: 1000)
    auth = nak._stage_auth("consumer", "stranger", 5_000_000, 3600, sign)
    posted = []
    consumer, expires_at = nak._stage_consumer(
        "https://onboard.example/base", "stranger", 5_000_000, node_pub, auth,
        lambda url, body: posted.append((url, body)) or {"code": "member-consumer", "expires_at": 1600},
        now=1000)
    assert expires_at == 1600 and consumer["starter"] == 5_000_000
    assert posted[0][1]["kind"] == "consumer" and posted[0][1]["starter"] == 5_000_000
    assert "ok" not in posted[0][1]


@pytest.mark.parametrize("status,reason", [
    (403, "invite right paused"),
    (409, "replayed request"),
])
def test_onboard_error_reason_is_preserved(monkeypatch, status, reason):
    import io
    from urllib.error import HTTPError

    def refused(request, timeout):
        raise HTTPError(request.full_url, status, "refused", {},
                        io.BytesIO(json.dumps({"error": reason}).encode()))

    monkeypatch.setattr(nak, "urlopen", refused)
    with pytest.raises(SystemExit, match=rf"\({status}\): {reason}"):
        nak._http_post_json("https://onboard.example/worker-stage", {"not": "a secret"})


def test_uninvite_cancels_server_then_local_invite(net, tmp_path, monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    inviter = net("member-cancel")
    code = _invite(inviter, kind="worker", worker={
        "redeem": "https://onboard.example/base", "code": "cancel-me", "tier": "stranger"})
    registered = inviter.node.register_invite(code)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NAK_ONBOARD_TOKEN", raising=False)

    def fake_call(_sock, req):
        result = nakd.handle(inviter.node, req)
        assert result["ok"], result
        return result

    posted = []
    monkeypatch.setattr(nak, "call", fake_call)
    assert nak.main(["uninvite", code], http_post=lambda url, body: posted.append((url, body)) or {"ok": True}) == 0
    assert posted[0][0] == "https://onboard.example/base/invite-cancel"
    body = posted[0][1]
    unsigned = {key: value for key, value in body.items() if key != "sig"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode()
    ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(inviter.node.node)).verify(
        bytes.fromhex(body["sig"]), canonical)
    assert body["purpose"] == "onboard-cancel-v2"
    assert set(body) == {"v", "purpose", "nonce", "code", "inviter_pubkey", "ts", "sig"}
    assert len(body["nonce"]) == 32 and all(ch in "0123456789abcdef" for ch in body["nonce"])
    assert body["code"] == "cancel-me" and body["inviter_pubkey"] == inviter.node.node
    assert all(row["nonce"] != registered["nonce"] for row in inviter.node.store.open_invites())

    contact_code = _invite(inviter)
    contact = inviter.node.register_invite(contact_code)
    assert nak.main(["uninvite", contact_code],
                    http_post=lambda _url, _body: pytest.fail("contact cancel must stay local")) == 0
    assert all(row["nonce"] != contact["nonce"] for row in inviter.node.store.open_invites())


def test_stranger_worker_request_is_person_gated_but_known_is_not(net):
    inviter, stranger, known = net("gate-inviter"), net("gate-stranger"), net("gate-known")

    def request_for(party, tier):
        code = _invite(inviter, kind="worker", worker={
            "redeem": "https://onboard.example/base", "code": f"worker-{tier}", "tier": tier})
        row = inviter.node.register_invite(code)
        assert inviter.node.store.consume_invite(row["nonce"], party.person_pub)
        rid = inviter.node.store.add_request(
            party.person_pub, party.node.node, tier, row["nonce"], party.node.node)
        return rid

    stranger_id = request_for(stranger, "stranger")
    stranger_row = next(row for row in nakd.handle(inviter.node, {"op": "requests"})["requests"]
                        if row["id"] == stranger_id)
    assert stranger_row["kind"] == "worker" and stranger_row["tier"] == "stranger"
    assert stranger_row["requires_person"] is True
    with pytest.raises(ValueError, match="a person must accept this one"):
        inviter.node.accept(stranger_id, "stranger")
    ts = int(time.time())
    signature = inviter.person_priv.sign(
        nakd.accept_statement(stranger_id, "stranger", ts)).hex()
    assert inviter.node.accept(stranger_id, "stranger", ts=ts, signature=signature)["person"] == \
        stranger.person_pub

    known_id = request_for(known, "known")
    known_row = next(row for row in nakd.handle(inviter.node, {"op": "requests"})["requests"]
                     if row["id"] == known_id)
    assert known_row["kind"] == "worker" and known_row["tier"] == "known"
    assert known_row["requires_person"] is False
    assert inviter.node.accept(known_id, "known")["person"] == known.person_pub


def test_legacy_request_without_invite_metadata_fails_closed_for_mcp(net):
    from network.client import NakError
    from network import nak_mcp

    inviter, requester = net("legacy-inviter"), net("legacy-requester")
    code = _invite(inviter)
    invite = inviter.node.register_invite(code)
    inviter.node.store._db.execute(
        "DELETE FROM invite_codes WHERE invite_id=?", (invite["nonce"],))
    assert inviter.node.store.consume_invite(invite["nonce"], requester.person_pub)
    request_id = inviter.node.store.add_request(
        requester.person_pub, requester.node.node, "known person",
        invite["nonce"], requester.node.node)

    listed = nakd.handle(inviter.node, {"op": "requests"})["requests"]
    row = next(row for row in listed if row["id"] == request_id)
    assert row["kind"] == "unknown" and row["requires_person"] is True
    with pytest.raises(ValueError, match="a person must accept this one"):
        inviter.node.accept(request_id, "known person")

    class DaemonClient:
        def accept(self, req_id, petname):
            reply = nakd.handle(inviter.node, {"op": "accept", "id": req_id, "petname": petname})
            if not reply["ok"]:
                raise NakError(reply["error"])
            return reply

    server = nak_mcp.Server(DaemonClient(), allow_connect=True)
    result = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "nak_accept",
                   "arguments": {"id": request_id, "name": "known person"}},
    })["result"]
    assert result["isError"]
    assert f"a person must accept this one: nak accept {request_id}" in result["content"][0]["text"]
    assert inviter.node.store.request(request_id) is not None


def test_consumer_redeem_uses_node_key_proof_without_worker_capabilities(net, tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    a = net("inviter")
    code = _invite(a, kind="consumer", consumer={
        "redeem": "https://onboard.example/base", "code": "consumer-once",
        "tier": "known", "starter": 0})
    identity = join_setup.ensure_identity(tmp_path)
    seen = {}

    def fake_post(url, body):
        seen.update(url=url, body=body)
        unsigned = {k: v for k, v in body.items() if k != "sig"}
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode()
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(body["pubkey"])).verify(
            bytes.fromhex(body["sig"]), canonical)
        return {"tier": "known", "operator": "op-consumer", "role": "consumer"}

    result = join_setup.redeem_consumer_invite(
        tmp_path, code, "Laptop", http_post=fake_post, now=int(time.time()))
    assert result["operator"] == "op-consumer"
    assert seen["url"] == "https://onboard.example/base/consumer-redeem"
    assert seen["body"]["pubkey"] == identity["node"]
    assert seen["body"]["capabilities"] == {}


def test_consumer_redeem_persists_url_but_ignores_onboarding_package_key(net, tmp_path):
    a = net("inviter-package")
    code = _invite(a, kind="consumer", consumer={
        "redeem": "https://onboard.example/base", "code": "consumer-package",
        "tier": "known", "starter": 0})
    join_setup.ensure_identity(tmp_path)

    result = join_setup.redeem_consumer_invite(
        tmp_path, code, "Laptop", now=int(time.time()),
        http_post=lambda _url, _body: {
            "tier": "known", "operator": "op-consumer", "role": "consumer",
            "package_url": "https://models.example/llama32-3b",
            "package_pubkey": "attacker-controlled-not-a-key",
        })

    assert result["package_url"].endswith("llama32-3b")
    path = tmp_path / join_setup.CONSUMER_STANDING_FILE
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text()) == {
        "operator": "op-consumer",
        "package_url": "https://models.example/llama32-3b",
        "role": "consumer",
        "tier": "known",
    }


def test_worker_stage_expiry_defaults_to_one_hour_and_rejects_bad_urls(monkeypatch):
    monkeypatch.setenv("NAK_ONBOARD_TOKEN", "operator-secret")
    worker, expires_at = nak._stage_worker(
        "https://onboard.example/base", "known", {"token": "operator-secret"},
        lambda _url, _body: {"code": "staged-once"}, now=1000)
    assert worker["code"] == "staged-once" and expires_at == 4600
    for url in ("http:///missing-host", "http://host:bad", "http://host:0", "http://host:65536", "http://host:"):
        with pytest.raises(SystemExit, match=r"http\(s\)"):
            nak._stage_worker(url, "known", {"token": "operator-secret"},
                              lambda _url, _body: {"code": "never"}, now=1000)


def test_starter_is_rejected_for_non_consumer_invites():
    with pytest.raises(SystemExit, match="only valid"):
        nak.main(["invite", "--kind", "worker", "--starter", "1"])


def test_worker_token_file_must_be_0600(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NAK_ONBOARD_TOKEN", raising=False)
    token = tmp_path / ".nakshatra" / "onboard.token"
    token.parent.mkdir()
    token.write_text("secret\n")
    token.chmod(0o644)
    with pytest.raises(SystemExit, match="0600"):
        nak._onboard_token()
    token.chmod(0o600)
    assert nak._onboard_token() == "secret"


def test_worker_redeem_without_stored_invite_is_a_clear_cli_error(tmp_path):
    with pytest.raises(SystemExit, match="could not read a stored worker invite"):
        join_setup.retry_worker_redeem(tmp_path)


def test_worker_redeem_converts_invalid_url_error_to_retry_guidance(net, tmp_path, monkeypatch):
    from http.client import InvalidURL
    from fabric import worker_join
    a = net("inviter")
    code = _invite(a, kind="worker", worker={
        "redeem": "https://onboard.example/base", "code": "one-use", "tier": "known"})
    join_setup.ensure_identity(tmp_path)
    saved = join_setup._store_worker_invite(tmp_path, code, "GPU Friend")
    monkeypatch.setattr(worker_join, "detect_capabilities", lambda: {"gpu": "fake", "vram_mb": 1})

    def fail_url(_url, _body):
        raise InvalidURL("nonnumeric port: 'bad'")

    with pytest.raises(SystemExit, match="invite is still stored; retry: nak worker-redeem"):
        join_setup.retry_worker_redeem(tmp_path, http_post=fail_url)
    assert saved.exists()


def test_worker_http_client_wraps_a_malformed_port():
    with pytest.raises(join_setup.WorkerRedeemError, match="could not reach the onboard server"):
        join_setup._http_post_json("http://host:bad/worker-redeem", {})


class _WorkerOnboardHandler(BaseHTTPRequestHandler):
    failures_left = 1
    seen = []

    def do_POST(self):  # noqa: N802
        from cryptography.hazmat.primitives.asymmetric import ed25519
        size = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(size))
        unsigned = {k: v for k, v in body.items() if k != "sig"}
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(body["pubkey"])).verify(
            bytes.fromhex(body["sig"]), canonical)
        type(self).seen.append(body)
        if type(self).failures_left:
            type(self).failures_left -= 1
            payload, status = {"error": "temporary outage"}, 503
        else:
            payload, status = {"tier": "known", "junction": "junction.example:7443", "tenant": "op-deadbeef"}, 200
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, _format, *_args):
        pass


def test_worker_join_failure_keeps_contact_and_retry_uses_signed_node_key(net, tmp_path, monkeypatch, capsys):
    from fabric import worker_join
    _WorkerOnboardHandler.failures_left = 1
    _WorkerOnboardHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _WorkerOnboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    a = net("inviter")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    code = _invite(a, kind="worker", worker={"redeem": base, "code": "one-use", "tier": "known"})
    a.node.register_invite(code)
    home = tmp_path / "worker"
    join_setup.ensure_identity(home)
    sg = signer_mod.Signer(home / ".sthambha" / "signer", custody="test")
    node_key = (home / ".nakshatra" / "keys" / "worker.ed25519").read_bytes()
    node = nakd.Node(home / ".nakshatra" / "net", node_key, sg.handle, agent="agent", relay=a.node.relay).start()
    stop = threading.Event()
    sock = home / ".nakshatra" / "net" / "nakd.sock"
    threading.Thread(target=nakd.serve_control, args=(node, sock, stop), daemon=True).start()
    assert wait(lambda: sock.exists())
    monkeypatch.setenv("NAK_JOIN_NO_SYSTEMD", "1")
    monkeypatch.setattr(worker_join, "detect_capabilities",
                        lambda: {"gpu": "fake-gpu", "vram_mb": 12345, "backend": "test", "kind": "gpu-worker"})
    try:
        assert join_setup.main(["--invite", code, "--name", "GPU Friend", "--home", str(home)]) == 0
        output = capsys.readouterr()
        assert "contact setup is complete" in output.err and "nak worker-redeem" in output.err
        retry = home / join_setup.WORKER_RETRY_FILE
        assert retry.exists() and (retry.stat().st_mode & 0o777) == 0o600
        assert (node.store.contact(a.person_pub) or {})["state"] == "awaiting"

        result = join_setup.retry_worker_redeem(home)
        assert result["tier"] == "known" and result["junction"] == "junction.example:7443"
        assert not retry.exists()
        body = _WorkerOnboardHandler.seen[-1]
        assert body["v"] == 2 and body["code"] == "one-use" and body["name"] == "GPU Friend"
        assert body["pubkey"] == node.node and body["capabilities"]["gpu"] == "fake-gpu"
        assert len(body["sig"]) == 128 and abs(int(time.time()) - body["ts"]) <= 10
    finally:
        stop.set()
        node.stop()
        server.shutdown()
        server.server_close()
