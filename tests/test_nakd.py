"""nakd: invite → request → accept → message, over a real local relay with two real signers."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
for cand in (os.environ.get("NAK_STHAMBHA_PATH", ""), str(Path.home() / "sthambha-signer"),
             str(_REPO.parent / "sthambha")):
    if cand and (Path(cand) / "sthambha" / "signer.py").exists():
        sys.path.insert(0, cand)
        break
signer_mod = pytest.importorskip("sthambha.signer")

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

import joincode  # noqa: E402
from network import nakd  # noqa: E402
from transport.relay import RendezvousRelay  # noqa: E402


def wait(pred, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


class Party:
    def __init__(self, root: Path, name: str, relay_port: int, caps=("nak.msg",)):
        self.dir = root / name
        self.node_key = ed25519.Ed25519PrivateKey.generate().private_bytes_raw()
        node_pub = ed25519.Ed25519PrivateKey.from_private_bytes(self.node_key).public_key().public_bytes_raw().hex()
        self.person_priv, self.person_pub = signer_mod.new_key()
        sdir = self.dir / "signer"
        (sdir / "agents").mkdir(parents=True, mode=0o700)
        (sdir / "node.pub").write_text(node_pub)
        (sdir / "person.pub").write_text(self.person_pub)
        agent_priv, agent_pub = signer_mod.new_key()
        signer_mod.save_key(sdir / "agents" / "pa.key", agent_priv)
        grant = signer_mod.issue_grant(self.person_priv, "pa", agent_pub, caps, node=node_pub)
        (sdir / "agents" / "pa.grant.json").write_text(json.dumps(grant))
        self.signer = signer_mod.Signer(sdir, custody="test")
        self.node = nakd.Node(self.dir / "net", self.node_key, self.signer.handle, agent="pa",
                              relay=("127.0.0.1", relay_port)).start()

    def invite(self, ttl=600, note=""):
        code = joincode.encode_invite(self.person_priv, self.node.node,
                                      [{"node": self.node.node, "addrs": ["127.0.0.1"]}], ttl_s=ttl, note=note)
        self.node.register_invite(code)
        return code


@pytest.fixture
def net(tmp_path):
    relay = RendezvousRelay("127.0.0.1", 0, max_per_ip_per_min=100000)
    port = relay.start()
    parties = []

    def make(name, **kw):
        p = Party(tmp_path, name, port, **kw)
        parties.append(p)
        return p
    yield make
    for p in parties:
        p.node.stop()
    relay.stop()


def test_connect_only_after_accept_then_message_both_ways(net):
    a, b = net("biswa"), net("rajesh")
    code = a.invite(note="rajesh")
    r = b.node.redeem(code, nickname="R", petname="biswa", wait_s=15)
    assert r["state"] == "request sent"
    reqs = a.node.store.requests()
    assert len(reqs) == 1 and reqs[0]["person"] == b.person_pub and reqs[0]["nickname"] == "R"
    # nothing is shared before acceptance: A has no contact, B cannot send
    assert a.node.store.contacts() == []
    with pytest.raises(ValueError, match="not been accepted"):
        b.node.send("biswa", "hello?")
    a.node.accept(reqs[0]["id"], petname="rajesh")
    assert wait(lambda: (b.node.store.contact(a.person_pub) or {}).get("state") == "active")

    mid = b.node.send("biswa", "namaste from rajesh")["queued"]
    assert wait(lambda: b.node.delivered(mid))
    inbox = a.node.store.inbox()
    assert [m["text"] for m in inbox] == ["namaste from rajesh"]
    assert inbox[0]["source"] == "nakshatra" and inbox[0]["trust"] == "external"
    assert inbox[0]["petname"] == "rajesh" and inbox[0]["author"].startswith("agent:pa@")
    raw = (a.dir / "net" / "inbox-raw.jsonl").read_text().splitlines()
    assert len(raw) == 1 and json.loads(raw[0])["trust"] == "external"

    mid2 = a.node.send("rajesh", "got it", aspect="whole")["queued"]
    assert wait(lambda: a.node.delivered(mid2))
    assert b.node.store.inbox()[0]["text"] == "got it" and b.node.store.inbox()[0]["aspect"] == "whole"


def test_invite_is_single_use(net):
    a, b, c = net("a"), net("b"), net("c")
    code = a.invite()
    assert b.node.redeem(code, wait_s=15)["state"] == "request sent"
    c.node.redeem(code, wait_s=3)
    time.sleep(1)
    assert [q["person"] for q in a.node.store.requests()] == [b.person_pub]
    nonce = joincode.decode_invite(code)["nonce"]
    assert a.node.store.consume_invite(nonce, c.person_pub) is False


def test_decline_leaves_no_contact(net):
    a, b = net("a"), net("b")
    b.node.redeem(a.invite(), wait_s=15)
    a.node.decline(a.node.store.requests()[0]["id"])
    assert a.node.store.contacts() == [] and a.node.store.requests() == []
    with pytest.raises(ValueError):
        a.node.send(b.person_pub, "hi")


def test_only_your_own_invites_register(net):
    a, b = net("a"), net("b")
    foreign = joincode.encode_invite(b.person_priv, b.node.node, [{"node": b.node.node, "addrs": []}])
    with pytest.raises(ValueError, match="not made for this"):
        a.node.register_invite(foreign)
    with pytest.raises(ValueError, match="your own invite"):
        a.node.redeem(a.invite())


def test_stranger_cannot_reach_you(net):
    a, s = net("a"), net("stranger")
    # the stranger fabricates a contact row for A and tries to talk; A has never accepted them
    s.node.store.upsert_contact(a.person_pub, a.node.node, "active", "a")
    s.node._ensure_session(a.person_pub)
    mid = s.node.send("a", "let me in")["queued"]
    time.sleep(2)
    assert not s.node.delivered(mid) and a.node.store.inbox() == []


def _pair(net):
    a, b = net("a"), net("b")
    b.node.redeem(a.invite(), wait_s=15)
    a.node.accept(a.node.store.requests()[0]["id"], petname="b")
    assert wait(lambda: (b.node.store.contact(a.person_pub) or {}).get("state") == "active")
    return a, b


def test_receive_refuses_tampered_and_misbound_envelopes(net):
    a, b = _pair(net)
    body = {"text": "real", "aspect": "", "mid": "m1"}
    env = b.node._sign("msg", a.person_pub, body)
    ok = a.node._receive(b.person_pub, {"t": "env", "env": env, "body": body})
    assert ok["t"] == "ack"
    tampered = dict(body, text="forged", mid="m1")
    assert a.node._receive(b.person_pub, {"t": "env", "env": env, "body": tampered})["t"] == "nack"
    # B's validly signed envelope presented as if from another contact is refused
    other = net("c")
    a.node.store.upsert_contact(other.person_pub, other.node.node, "active", "c")
    assert a.node._receive(other.person_pub, {"t": "env", "env": env, "body": body})["t"] == "nack"
    # addressed to someone else
    env2 = b.node._sign("msg", other.person_pub, dict(body, mid="m2"))
    assert a.node._receive(b.person_pub, {"t": "env", "env": env2, "body": dict(body, mid="m2")})["t"] == "nack"
    assert [m["text"] for m in a.node.store.inbox()] == ["real"]


def test_duplicate_delivery_is_stored_once(net):
    a, b = _pair(net)
    body = {"text": "once", "aspect": "", "mid": "dup1"}
    for _ in range(3):
        env = b.node._sign("msg", a.person_pub, body)
        assert a.node._receive(b.person_pub, {"t": "env", "env": env, "body": body})["t"] == "ack"
    assert [m["text"] for m in a.node.store.inbox()] == ["once"]


def test_outbox_survives_peer_offline(net, tmp_path):
    a, b = _pair(net)
    b.node.stop()
    time.sleep(0.3)
    mid = a.node.send("b", "while you were away")["queued"]
    time.sleep(1)
    assert not a.node.delivered(mid)
    # B comes back with the same state dir and keys
    b.node = nakd.Node(b.dir / "net", b.node_key, b.signer.handle, agent="pa", relay=a.node.relay).start()
    assert wait(lambda: a.node.delivered(mid), timeout=40)
    assert b.node.store.inbox()[0]["text"] == "while you were away"


def test_control_socket_round_trip(net, tmp_path):
    import threading
    a = net("a")
    stop = threading.Event()
    sock = tmp_path / "c.sock"
    t = threading.Thread(target=nakd.serve_control, args=(a.node, sock, stop), daemon=True)
    t.start()
    assert wait(lambda: sock.exists())
    assert oct(sock.stat().st_mode & 0o777) == "0o600"
    from network import nak
    r = nak.call(sock, {"op": "status"})
    assert r["person"] == a.person_pub and r["contacts"] == []
    with pytest.raises(SystemExit, match="not a contact"):
        nak.call(sock, {"op": "send", "to": "nobody", "text": "x"})
    stop.set()


def _serve(node, tmp_path):
    import threading
    stop = threading.Event()
    sock = tmp_path / f"{node.node[:8]}.sock"
    threading.Thread(target=nakd.serve_control, args=(node, sock, stop), daemon=True).start()
    assert wait(lambda: sock.exists())
    return sock, stop


def test_outbound_floor_refuses_secrets_and_invites(net):
    a, b = _pair(net)
    for bad in ("my key is sk-ant-" + "x" * 30, "-----BEGIN OPENSSH PRIVATE KEY-----", "A" * 200,
                "join me: " + a.invite()):
        with pytest.raises(ValueError):
            a.node.send("b", bad)
    a.node.send("b", "a normal note about dinner at 7, ok?")


def test_mcp_front_door(net, tmp_path):
    from network.client import NakClient
    from network.nak_mcp import Server
    a, b = _pair(net)
    sock, stop = _serve(a.node, tmp_path)
    srv = Server(NakClient(sock), allow_connect=False, aspect="whole")
    init = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
    assert init["result"]["serverInfo"]["name"] == "nakshatra"
    names = {t["name"] for t in srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]}
    assert names == {"nak_status", "nak_contacts", "nak_inbox", "nak_requests", "nak_send",
                     "nak_tasks", "nak_task_post", "nak_task_claim", "nak_task_submit"}   # no connect tools
    call = lambda n, args: srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                       "params": {"name": n, "arguments": args}})["result"]
    r = call("nak_accept", {"id": "x"})
    assert r["isError"] and "disabled" in r["content"][0]["text"]
    r = call("nak_send", {"to": "stranger", "text": "hi"})
    assert r["isError"] and "not a contact" in r["content"][0]["text"]
    r = call("nak_send", {"to": "b", "text": "hello from the mcp"})
    assert not r["isError"]
    mid = r["content"][0]["text"].split("(id ")[1].split(")")[0]
    assert wait(lambda: a.node.delivered(mid))
    got = b.node.store.inbox()[0]
    assert got["text"] == "hello from the mcp" and got["aspect"] == "whole"
    # B's view of the inbox through its own front door is framed as untrusted
    sock_b, stop_b = _serve(b.node, tmp_path)
    inbox = Server(NakClient(sock_b)).call("nak_inbox", {})
    assert inbox.startswith("UNTRUSTED EXTERNAL CONTENT") and "hello from the mcp" in inbox
    stop.set(); stop_b.set()


def test_mcp_stdio_process(net, tmp_path):
    import subprocess
    a = net("a")
    sock, stop = _serve(a.node, tmp_path)
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "nak_status", "arguments": {}}}]
    p = subprocess.run([sys.executable, str(_REPO / "scripts" / "network" / "nak_mcp.py"), "--sock", str(sock)],
                       input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True, timeout=30,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    out = [json.loads(l) for l in p.stdout.splitlines()]
    assert [o["id"] for o in out] == [1, 2]
    assert a.person_pub[:16] in out[1]["result"]["content"][0]["text"]
    stop.set()


# ── tasks ───────────────────────────────────────────────────────────────────────────────────

def _worker_row(node, h):
    return [r for r in node.store.tasks("worker") if r["task_hash"] == h]


def _connect(net, a, b):
    b.node.redeem(a.invite(), wait_s=15)
    a.node.accept(a.node.store.requests()[0]["id"], petname=b.dir.name)
    assert wait(lambda: (b.node.store.contact(a.person_pub) or {}).get("state") == "active")


def test_task_post_claim_assign_result_accept(net):
    a = net("poster", caps=("nak.msg", "nak.task.post"))
    b = net("worker", caps=("nak.msg", "nak.task.claim"))
    _connect(net, a, b)
    r = a.node.post_task("worker", "capital", "What is the capital of Nepal? One line.",
                         [{"max_words": 8}, {"contains_all": ["Kathmandu"]}], reward=5)
    h = r["task_hash"]
    assert wait(lambda: _worker_row(b.node, h))
    offer = _worker_row(b.node, h)[0]
    assert offer["state"] == "offered" and offer["spec"]["reward"] == {"amount": 5, "unit": "TEST"}
    with pytest.raises(ValueError, match="assigned to you"):
        b.node.submit_task(h[:12], "too early")
    b.node.claim_task(h[:12])
    assert wait(lambda: _worker_row(b.node, h)[0]["state"] == "assigned")
    assert a.node.store.task_get(h, "poster")["assignee"] == b.person_pub
    sub = b.node.submit_task(h[:12], "Kathmandu is the capital.")
    assert sub["precheck_passed"]
    assert wait(lambda: _worker_row(b.node, h)[0]["state"] == "accepted")
    t = a.node.store.task_get(h, "poster")
    assert t["state"] == "accepted" and t["verdict"]["passed"] and t["output"] == "Kathmandu is the capital."


def test_task_result_failing_rules_is_rejected_with_reasons(net):
    a = net("poster", caps=("nak.msg", "nak.task.post"))
    b = net("worker", caps=("nak.msg", "nak.task.claim"))
    _connect(net, a, b)
    h = a.node.post_task("worker", "short", "Say hi.", [{"max_words": 2}])["task_hash"]
    assert wait(lambda: _worker_row(b.node, h))
    b.node.claim_task(h)
    assert wait(lambda: _worker_row(b.node, h)[0]["state"] == "assigned")
    assert not b.node.submit_task(h, "hello there my friend")["precheck_passed"]
    assert wait(lambda: _worker_row(b.node, h)[0]["state"] == "rejected")
    assert "4 words > max 2" in _worker_row(b.node, h)[0]["verdict"]["reasons"][0]


def test_first_claim_wins_second_is_told_lost(net):
    a = net("poster", caps=("nak.msg", "nak.task.post"))
    b = net("w1", caps=("nak.msg", "nak.task.claim"))
    c = net("w2", caps=("nak.msg", "nak.task.claim"))
    _connect(net, a, b)
    _connect(net, a, c)
    h = a.node.post_task("*", "one job", "Do it.", [{"max_words": 3}])["task_hash"]
    assert wait(lambda: _worker_row(b.node, h) and _worker_row(c.node, h))
    b.node.claim_task(h)
    assert wait(lambda: _worker_row(b.node, h)[0]["state"] == "assigned")
    c.node.claim_task(h)
    assert wait(lambda: _worker_row(c.node, h)[0]["state"] == "lost")
    assert a.node.store.task_get(h, "poster")["assignee"] == b.person_pub


def test_worker_without_claim_cap_cannot_claim(net):
    a = net("poster", caps=("nak.msg", "nak.task.post"))
    b = net("worker", caps=("nak.msg",))            # no nak.task.claim delegated
    _connect(net, a, b)
    h = a.node.post_task("worker", "x", "Do it.", [{"max_words": 3}])["task_hash"]
    assert wait(lambda: _worker_row(b.node, h))
    b.node.claim_task(h)                              # queued; the signer refuses to sign it
    assert wait(lambda: any(b.node.store.refused(m) for m in _outbox_mids(b.node)))
    assert a.node.store.task_get(h, "poster")["state"] == "posted"
    # and the session with that contact still carries ordinary messages
    mid = b.node.send(a.person_pub, "still here")["queued"]
    assert wait(lambda: b.node.delivered(mid)) and not b.node.store.refused(mid)


def _outbox_mids(node):
    return [r[0] for r in node.store._q("SELECT nonce FROM outbox")]


def test_poster_cannot_post_secrets_or_to_strangers(net):
    a = net("poster", caps=("nak.msg", "nak.task.post"))
    with pytest.raises(ValueError, match="no accepted contacts"):
        a.node.post_task("*", "x", "y", [{"max_words": 3}])
    b = net("worker", caps=("nak.msg", "nak.task.claim"))
    _connect(net, a, b)
    with pytest.raises(ValueError, match="secret"):
        a.node.post_task("worker", "x", "use key sk-ant-" + "z" * 30, [{"max_words": 3}])
    with pytest.raises(ValueError, match="accepted contact"):
        a.node.post_task(["worker", "nobody"], "x", "y", [{"max_words": 3}])


def test_forged_task_steps_are_refused(net):
    a = net("poster", caps=("nak.msg", "nak.task.post"))
    b = net("worker", caps=("nak.msg", "nak.task.claim"))
    _connect(net, a, b)
    h = a.node.post_task("worker", "x", "Do it.", [{"max_words": 3}])["task_hash"]
    assert wait(lambda: _worker_row(b.node, h))
    # a result without being assigned is refused at the poster
    body = {"task_hash": h, "epoch": 1, "output": "done", "mid": "f1"}
    env = b.node._sign("task.result", a.person_pub, body)
    assert a.node._receive(b.person_pub, {"t": "env", "env": env, "body": body})["t"] == "nack"
    # a spec whose poster field lies about who posted it is refused at the worker
    from network import tasks as T
    fake = T.make_spec(b.person_pub, "fake", "Do it.", [{"max_words": 3}])
    body = {"task_hash": T.task_hash(fake), "spec": fake, "mid": "f2"}
    env = a.node._sign("task.post", b.person_pub, body)
    assert b.node._receive(a.person_pub, {"t": "env", "env": env, "body": body})["t"] == "nack"


def test_mcp_task_tools_round_trip(net, tmp_path):
    from network.client import NakClient
    from network.nak_mcp import Server
    a = net("poster", caps=("nak.msg", "nak.task.post"))
    b = net("worker", caps=("nak.msg", "nak.task.claim"))
    _connect(net, a, b)
    sa, stop_a = _serve(a.node, tmp_path)
    sb, stop_b = _serve(b.node, tmp_path)
    pa, pb = Server(NakClient(sa)), Server(NakClient(sb))
    out = pa.call("nak_task_post", {"to": "worker", "title": "greet", "instructions": "Say namaste.",
                                    "acceptance": '[{"contains_all": ["namaste"]}]'})
    h = out.split("task ")[1].split(" ")[0]
    assert wait(lambda: "offered" in pb.call("nak_tasks", {}))
    view = pb.call("nak_tasks", {})
    assert "UNTRUSTED" in view and "<untrusted_request" in view and "must mention ['namaste']" in view
    pb.call("nak_task_claim", {"task": h})
    assert wait(lambda: "assigned" in pb.call("nak_tasks", {}))
    assert "passes" in pb.call("nak_task_submit", {"task": h, "output": "Namaste!"})
    assert wait(lambda: "ACCEPTED" in pa.call("nak_tasks", {"role": "poster"}))
    stop_a.set(); stop_b.set()
