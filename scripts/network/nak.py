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
import socket
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def call(sock_path: Path, req: dict) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(60)
    try:
        s.connect(str(sock_path))
    except OSError as e:
        raise SystemExit(f"nakd is not running ({sock_path}): {e}")
    with s:
        f = s.makefile("rwb")
        f.write(json.dumps(req).encode() + b"\n")
        f.flush()
        line = f.readline()
    r = json.loads(line) if line else {"ok": False, "error": "nakd closed the connection"}
    if not r.get("ok"):
        raise SystemExit(f"refused: {r.get('error')}")
    return r


def _ago(ts: int) -> str:
    d = int(time.time()) - int(ts)
    return f"{d}s ago" if d < 120 else f"{d // 60}m ago" if d < 7200 else f"{d // 3600}h ago"


def main(argv=None) -> int:
    default_sock = Path(os.environ.get("NAK_NET_DIR", Path.home() / ".nakshatra" / "net")) / "nakd.sock"
    ap = argparse.ArgumentParser(prog="nak", description=__doc__.split("\n")[0])
    ap.add_argument("--sock", type=Path, default=default_sock)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("invite"); p.add_argument("--person-key", type=Path, required=True)
    p.add_argument("--ttl", type=int, default=86400); p.add_argument("--note", default="")
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
        code = joincode.encode_invite(priv, st["node"], [{"node": st["node"], "addrs": [st["relay"]]}],
                                      ttl_s=a.ttl, note=a.note)
        r = call(s, {"op": "register_invite", "code": code})
        print(code)
        print(f"\nsingle use, expires in {a.ttl // 3600}h. Share it only with the person you are inviting.",
              file=sys.stderr)
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
            print(f"{c['petname'] or '-':16} {c['state']:9} {'online' if c['online'] else 'offline':8} {c['person'][:16]}…")
    elif a.cmd == "remove":
        print(json.dumps(call(s, {"op": "remove", "who": a.who})))
    elif a.cmd == "send":
        r = call(s, {"op": "send", "to": a.to, "text": a.text, "aspect": a.aspect})
        print(f"queued {r['queued']} to {r['to']}")
    elif a.cmd == "inbox":
        for m in reversed(call(s, {"op": "inbox", "limit": a.limit})["messages"]):
            print(f"[{_ago(m['received'])}] {m['petname'] or m['from_person'][:12]}: {m['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
