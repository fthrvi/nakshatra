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
import json
import os
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


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


def main(argv=None) -> int:
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
    p = sub.add_parser("redeem"); p.add_argument("code"); p.add_argument("--nickname", default="")
    p.add_argument("--as", dest="petname", default="")
    sub.add_parser("requests")
    p = sub.add_parser("accept"); p.add_argument("id"); p.add_argument("--as", dest="petname", default="")
    p = sub.add_parser("decline"); p.add_argument("id")
    sub.add_parser("contacts")
    p = sub.add_parser("remove"); p.add_argument("who")
    p = sub.add_parser("send"); p.add_argument("to"); p.add_argument("text")
    p.add_argument("--aspect", default="")
    p = sub.add_parser("inbox"); p.add_argument("--limit", type=int, default=20)
    sub.add_parser("ledger", help="escrow journal: open / released / refunded")
    p = sub.add_parser("direct", help="allow/stop direct connections with a contact (reveals your LAN/IPv6)")
    p.add_argument("who"); p.add_argument("onoff", choices=["on", "off"])
    p = sub.add_parser("direct-listen", help="accept direct connections on this port (or 'off'); restarts nak-net")
    p.add_argument("port")
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

    if a.cmd == "status":
        r = call(s, {"op": "status"})
        print(f"person {r['person'][:16]}…  node {r['node'][:16]}…  custody {r['custody']}  relay {r['relay']}")
        print(f"{len(r['contacts'])} contacts, {r['pending_requests']} pending requests, {r['open_invites']} open invites")
    elif a.cmd == "invite":
        import joincode
        from cryptography.hazmat.primitives.asymmetric import ed25519
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
        code = joincode.encode_invite(priv, st["node"], [{"node": st["node"], "addrs": [st["relay"]]}],
                                      ttl_s=a.ttl, note=a.note, release=release, from_name=from_name)
        call(s, {"op": "register_invite", "code": code})
        if a.code_only or not line:
            print(code)
        else:
            print("Send your friend this ONE line (single use, expires in "
                  f"{a.ttl // 3600}h; it is a secret — only them):\n", file=sys.stderr)
            print(line.replace("{CODE}", code))
            print("\nThey paste it in a terminal (Linux or WSL with systemd). When their request arrives:"
                  "  nak requests  →  nak accept <id> --as <name>", file=sys.stderr)
    elif a.cmd == "redeem":
        r = call(s, {"op": "redeem", "code": a.code, "nickname": a.nickname, "petname": a.petname, "wait_s": 20})
        print(f"{r['state']}. They have to accept before either of you can message.")
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
