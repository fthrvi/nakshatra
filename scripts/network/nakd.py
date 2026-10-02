"""nakd — the node's messaging service: invites, connection requests, contacts, encrypted messages.

How two people connect (nothing flows until BOTH have agreed):

  1. A asks for an invite (a human act: it is signed with A's PERSON key, which this daemon never
     holds). The invite carries a one-time key `ik`. A's node waits on the relay at the invite's
     rendezvous, pinned to ik's public half.
  2. B redeems it. B's node dials that rendezvous, proves it holds `ik`, and inside the encrypted
     channel sends a signed `contact.request` carrying B's person key, node key and a nickname of
     B's choosing. That is a REQUEST only. A's node records it as pending; nothing else happens.
  3. A accepts (or declines). On accept both sides store each other as contacts and A's node sends
     a signed `contact.accept` over the normal per-contact session. B's side flips to active.
  4. From then on, only contacts can message. Each pair of nodes meets on the relay at a
     rendezvous derived from both node keys; the channel is pinned to the contact's node key and
     every message is an envelope signed by an agent key that the sender's person delegated.

What each party sees:
  * The relay operator sees two IP addresses meet; everything else is ciphertext.
  * Your contact sees your person key, node key and the nickname you chose. Never your IP (the
    relay is in between), never your name unless you put it in the nickname.
  * Nobody who is not your contact can reach you: a message from anyone else fails the pinned
    handshake or the contact check and is dropped before it is stored.

Inbound text is UNTRUSTED EXTERNAL CONTENT (unification lane, 2026-10-02): it is raw-logged first,
labelled source=nakshatra trust=external, and never triggers any action here.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

import joincode  # noqa: E402
from mesh.pairing import pair_role  # noqa: E402
from network.store import Store  # noqa: E402
from network import tasks as T  # noqa: E402
from transport.relay import connect as relay_connect  # noqa: E402
from transport.secure_channel import SecureChannelError, secure_handshake  # noqa: E402

try:
    from sthambha.signer import SignerError, canonical, verify_envelope
except ImportError:  # the release ships both; a dev checkout may point at a sthambha worktree
    _alt = os.environ.get("NAK_STHAMBHA_PATH")
    if not _alt:
        raise
    sys.path.insert(0, _alt)
    from sthambha.signer import SignerError, canonical, verify_envelope  # noqa: E402

MSG_DOMAIN = "nak-msg-v1"
MAX_FRAME = 1 << 20
MAX_TEXT = 8000
WAIT_S = 100.0          # under the relay's 120 s waiting TTL, so we re-register before it reaps us
PING_S = 30.0
IDLE_S = 95.0           # no frame (not even a ping) for this long = the peer is gone
DEFAULT_RELAY = ("45.63.109.137", 51820)

# Floor for every agent on every node: refuse outbound text carrying a recognisable secret or a bulk
# encoded payload. Same formats as Prithvi's web egress DLP (mind/web_tools.py); a harness with its
# own stricter check (Prithvi's agency_guard path) runs that first — this is the node's own floor.
import re as _re
_SECRET_FMT = _re.compile(
    r"(-----BEGIN |ssh-rsa\s|AKIA[0-9A-Z]{16}|"
    r"gh[posru]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-ant-[A-Za-z0-9_-]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|nsec1[a-z0-9]{20,}|"
    r"bcn_[a-f0-9]{16,}|prithvi-canary-[A-Za-z0-9]{8,}|nki1\.)")
_BLOB = _re.compile(r"[A-Za-z0-9+/=_-]{128,}")


def outbound_problem(text: str) -> str:
    """A reason to refuse sending this text, or ''. (An invite code counts: invites are handed over
    by people, never forwarded by an agent.)"""
    if _SECRET_FMT.search(text):
        return "message contains a secret-shaped token; not sending it"
    if _BLOB.search(text):
        return "message contains a long encoded blob; not sending it"
    return ""


# ── framing over a SecureChannel ────────────────────────────────────────────────────────────────

def send_frame(ch, obj: dict, lock: Optional[threading.Lock] = None) -> None:
    data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(data) > MAX_FRAME:
        raise ValueError("frame too large")
    if lock:
        with lock:
            ch.sendall(struct.pack(">I", len(data)) + data)
    else:
        ch.sendall(struct.pack(">I", len(data)) + data)


def _read_exact(ch, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        more = ch.recv(n - len(buf))
        if not more:
            return None
        buf += more
    return buf


def recv_frame(ch) -> Optional[dict]:
    hdr = _read_exact(ch, 4)
    if hdr is None:
        return None
    n = struct.unpack(">I", hdr)[0]
    if n > MAX_FRAME:
        raise SecureChannelError("frame too large")
    data = _read_exact(ch, n)
    if data is None:
        return None
    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise SecureChannelError("frame is not an object")
    return obj


def body_hash(body: dict) -> str:
    return hashlib.sha256(canonical(body)).hexdigest()


# ── signer access ───────────────────────────────────────────────────────────────────────────────

class SignerClient:
    """The local signer over its unix socket (one JSON line each way). The daemon never sees a key."""

    def __init__(self, sock_path: Path):
        self.path = str(sock_path)

    def __call__(self, req: dict) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10)
        try:
            s.connect(self.path)
            f = s.makefile("rwb")
            f.write(json.dumps(req).encode() + b"\n")
            f.flush()
            line = f.readline()
        finally:
            s.close()
        return json.loads(line) if line else {"ok": False, "error": "signer closed the connection"}


# ── the node ────────────────────────────────────────────────────────────────────────────────────

class Node:
    def __init__(self, state_dir: Path, node_key: bytes, signer: Callable[[dict], dict], *,
                 agent: str, relay: tuple = DEFAULT_RELAY, log: Callable[[str], None] = lambda m: None):
        self.store = Store(state_dir)
        self._key = node_key
        self.node = ed25519.Ed25519PrivateKey.from_private_bytes(node_key).public_key().public_bytes_raw().hex()
        self._signer = signer
        self.agent = agent
        self.relay = relay
        self.log = log
        who = signer({"op": "whoami"})
        if not who.get("ok"):
            raise RuntimeError(f"signer: {who.get('error')}")
        if who["node"] != self.node:
            raise RuntimeError("the signer is set up for a different node key")
        if agent not in who.get("agents", []):
            raise RuntimeError(f"the signer has no delegation for agent {agent!r}")
        self.person = who["person"]
        self.custody = who.get("custody", "")
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}       # key -> worker thread
        self._sessions: dict[str, tuple] = {}                 # contact person -> (channel, send lock)
        self._wake: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # lifecycle
    def start(self) -> "Node":
        for inv in self.store.open_invites():
            self._spawn(f"invite:{inv['nonce']}", self._invite_loop, inv)
        for c in self.store.contacts():
            self._ensure_session(c["person"])
        return self

    def stop(self) -> None:
        self._stop.set()
        for ev in list(self._wake.values()):
            ev.set()
        with self._lock:
            for ch, _ in list(self._sessions.values()):
                _hard_close(ch)

    def _spawn(self, key: str, fn, *args) -> None:
        with self._lock:
            t = self._threads.get(key)
            if t and t.is_alive():
                return
            t = threading.Thread(target=fn, args=args, daemon=True, name=key)
            self._threads[key] = t
        t.start()

    def _sign(self, kind: str, to: str, body: dict) -> dict:
        r = self._signer({"op": "sign_envelope", "agent": self.agent, "kind": kind, "to": to,
                          "body_sha256": body_hash(body)})
        if not r.get("ok"):
            raise RuntimeError(f"signer refused: {r.get('error')}")
        return r["envelope"]

    def _dial(self, rid: bytes, my_key: bytes, pin: str, initiator: bool, binding: bytes):
        sock = relay_connect(self.relay[0], self.relay[1], rid, timeout=15)
        sock.settimeout(WAIT_S)          # wait for the partner, then handshake
        try:
            ch = secure_handshake(sock, my_key, pin, initiator, binding)
        except BaseException:
            _close(sock)
            raise
        return sock, ch

    def _backoff(self, attempt: int) -> None:
        self._stop.wait(min(30.0, 0.3 * (2 ** min(attempt, 7))) * (0.5 + random.random()))

    # ── invites (inviter side) ────────────────────────────────────────────────────────────────
    def register_invite(self, code: str) -> dict:
        """Accept an invite made with this person's key for this node and start waiting for it."""
        inv = joincode.decode_invite(code)
        if inv["inviter"] != self.person or inv["inviter_node"] != self.node:
            raise ValueError("that invite was not made for this person and node")
        self.store.add_invite(inv["nonce"], inv["ik"], inv["expires_at"], inv.get("note", ""))
        row = {"nonce": inv["nonce"], "ik": inv["ik"], "expires_at": inv["expires_at"]}
        self._spawn(f"invite:{inv['nonce']}", self._invite_loop, row)
        return {"nonce": inv["nonce"], "expires_at": inv["expires_at"]}

    def _invite_loop(self, inv: dict) -> None:
        nonce = inv["nonce"]
        pin = joincode.invite_pub({"ik": inv["ik"]})
        rid = joincode.invite_rendezvous({"nonce": nonce})
        attempt = 0
        while not self._stop.is_set() and time.time() < inv["expires_at"]:
            if not any(i["nonce"] == nonce for i in self.store.open_invites()):
                return
            try:
                sock, ch = self._dial(rid, self._key, pin, False, b"nak-invite-v1|" + nonce.encode())
            except (OSError, SecureChannelError):
                attempt += 1
                self._backoff(attempt)
                continue
            attempt = 0
            try:
                sock.settimeout(30)
                frame = recv_frame(ch)
                reply = self._take_request(frame, nonce)
                send_frame(ch, reply)
                if reply.get("t") == "ack":
                    return
            except (OSError, SecureChannelError, ValueError) as e:
                self.log(f"invite {nonce[:8]}: {e}")
            finally:
                _close(sock)

    def _take_request(self, frame: Optional[dict], nonce: str) -> dict:
        if not frame or frame.get("t") != "env":
            return {"t": "nack", "why": "expected a signed request"}
        env, body = frame.get("env") or {}, frame.get("body") or {}
        if env.get("kind") != "contact.request" or not isinstance(body, dict):
            return {"t": "nack", "why": "expected contact.request"}
        try:
            person, node = body["person"], body["node"]
            verify_envelope(env, person)
        except (KeyError, SignerError) as e:
            return {"t": "nack", "why": f"request does not verify: {e}"}
        if env.get("from_node") != node or env.get("to") != self.person or env.get("body_sha256") != body_hash(body) \
                or body.get("invite") != nonce:
            return {"t": "nack", "why": "request is not bound to this invite"}
        if person == self.person:
            return {"t": "nack", "why": "cannot connect to yourself"}
        if not self.store.consume_invite(nonce, person):
            return {"t": "nack", "why": "invite already used or expired"}
        rid = self.store.add_request(person, node, str(body.get("nickname", ""))[:40], nonce)
        self.log(f"connection request {rid} from {person[:12]} ({body.get('nickname', '')!r})")
        return {"t": "ack", "mid": body.get("mid", "")}

    # ── redeeming someone else's invite ──────────────────────────────────────────────────────
    def redeem(self, code: str, nickname: str = "", petname: str = "", wait_s: float = 0.0) -> dict:
        """Ask the inviter to connect. Returns at once; delivery retries until the invite expires."""
        inv = joincode.decode_invite(code)
        if inv["inviter"] == self.person:
            raise ValueError("that is your own invite")
        existing = self.store.contact(inv["inviter"])
        if existing and existing["state"] == "active":
            return {"state": "already connected", "person": inv["inviter"]}
        self.store.upsert_contact(inv["inviter"], inv["inviter_node"], "awaiting", petname or inv.get("note", "")[:40])
        done = threading.Event()
        self._spawn(f"redeem:{inv['nonce']}", self._redeem_loop, inv, nickname[:40], done)
        if wait_s:
            done.wait(wait_s)
        return {"state": "request sent" if done.is_set() else "request queued", "person": inv["inviter"]}

    def _redeem_loop(self, inv: dict, nickname: str, done: threading.Event) -> None:
        body = {"person": self.person, "node": self.node, "nickname": nickname, "invite": inv["nonce"],
                "mid": secrets.token_hex(12)}
        ik = bytes.fromhex(inv["ik"])
        rid = joincode.invite_rendezvous(inv)
        attempt = 0
        while not self._stop.is_set() and time.time() < inv["expires_at"]:
            try:
                sock, ch = self._dial(rid, ik, inv["inviter_node"], True, b"nak-invite-v1|" + inv["nonce"].encode())
            except (OSError, SecureChannelError):
                attempt += 1
                self._backoff(attempt)
                continue
            try:
                sock.settimeout(30)
                send_frame(ch, {"t": "env", "env": self._sign("contact.request", inv["inviter"], body), "body": body})
                reply = recv_frame(ch) or {}
            except (OSError, SecureChannelError, ValueError) as e:
                reply = {"t": "nack", "why": str(e)}
            finally:
                _close(sock)
            if reply.get("t") == "ack":
                done.set()
                self._ensure_session(inv["inviter"])     # be there to hear the accept
                return
            self.log(f"redeem {inv['nonce'][:8]}: {reply.get('why')}")
            if "already used" in str(reply.get("why", "")):
                return
            attempt += 1
            self._backoff(attempt)

    # ── deciding requests ────────────────────────────────────────────────────────────────────
    def accept(self, req_id: str, petname: str = "") -> dict:
        r = self.store.decide_request(req_id, True)
        if not r:
            raise ValueError("no pending request with that id")
        self.store.upsert_contact(r["person"], r["node"], "active", petname or r["nickname"], r["nickname"])
        self._queue(r["person"], "contact.accept", {"person": self.person, "node": self.node})
        self._ensure_session(r["person"])
        return {"person": r["person"], "petname": petname or r["nickname"]}

    def decline(self, req_id: str) -> dict:
        if not self.store.decide_request(req_id, False):
            raise ValueError("no pending request with that id")
        return {"declined": req_id}

    def remove(self, who: str) -> dict:
        c = self.store.resolve(who)
        if not c:
            raise ValueError("no such contact")
        self.store.remove_contact(c["person"])
        with self._lock:
            s = self._sessions.get(c["person"])
        if s:
            _hard_close(s[0])
        return {"removed": c["person"]}

    # ── sending ──────────────────────────────────────────────────────────────────────────────
    def send(self, who: str, text: str, aspect: str = "") -> dict:
        c = self.store.resolve(who)
        if not c:
            raise ValueError("not a contact: only people who accepted a connection can be messaged")
        if c["state"] != "active":
            raise ValueError("that connection has not been accepted yet")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
            raise ValueError(f"text must be 1-{MAX_TEXT} characters")
        why = outbound_problem(text)
        if why:
            self.log(f"send to {c['person'][:12]} refused: {why}")
            raise ValueError(why)
        mid = self._queue(c["person"], "msg", {"text": text, "aspect": str(aspect)[:32]})
        return {"queued": mid, "to": c["petname"] or c["person"][:12]}

    def _queue(self, person: str, kind: str, body: dict) -> str:
        mid = secrets.token_hex(12)
        self.store.queue(mid, person, {"kind": kind, "body": dict(body, mid=mid)})
        ev = self._wake.get(person)
        if ev:
            ev.set()
        return mid

    def delivered(self, mid: str) -> bool:
        return self.store.delivered(mid)

    # ── per-contact sessions ─────────────────────────────────────────────────────────────────
    def _ensure_session(self, person: str) -> None:
        self._wake.setdefault(person, threading.Event())
        self._spawn(f"session:{person}", self._session_loop, person)

    def _session_loop(self, person: str) -> None:
        attempt = 0
        while not self._stop.is_set():
            c = self.store.contact(person)
            if not c:
                return
            role = pair_role(self.node, c["node"], MSG_DOMAIN)
            try:
                sock, ch = self._dial(role.rendezvous_id, self._key, c["node"], role.is_initiator,
                                      MSG_DOMAIN.encode())
            except (OSError, SecureChannelError):
                attempt += 1
                self._backoff(attempt)
                continue
            attempt = 0
            self.log(f"session up with {c['petname'] or person[:12]}")
            self._run_session(person, sock, ch)
            self._backoff(0)

    def _run_session(self, person: str, sock, ch) -> None:
        lock = threading.Lock()
        with self._lock:
            self._sessions[person] = (ch, lock)
        wake = self._wake.setdefault(person, threading.Event())
        down = threading.Event()
        sock.settimeout(IDLE_S)

        def reader():
            try:
                while not down.is_set():
                    frame = recv_frame(ch)
                    if frame is None:
                        return
                    t = frame.get("t")
                    if t == "env":
                        send_frame(ch, self._receive(person, frame), lock)
                    elif t == "ack":
                        self.store.ack(str(frame.get("mid", "")))
                    elif t == "nack":   # received and refused: terminal, stop retrying, keep why
                        self.store.refuse(str(frame.get("mid", "")), str(frame.get("why", "")))
                        self.log(f"{person[:12]} refused {str(frame.get('mid', ''))[:8]}: {frame.get('why')}")
            except (OSError, SecureChannelError, ValueError) as e:
                self.log(f"session {person[:12]}: {e}")
            finally:
                down.set()
                wake.set()

        threading.Thread(target=reader, daemon=True).start()
        last_ping = 0.0
        try:
            while not down.is_set() and not self._stop.is_set():
                if not self.store.contact(person):
                    return
                for item in self.store.pending_for(person):
                    try:
                        env = self._sign(item["kind"], person, item["body"])
                    except RuntimeError as e:
                        # The signer said no (no delegation for this kind, revoked, expired). That is
                        # terminal for THIS item; it must not take the whole contact session down.
                        self.store.refuse(item["body"].get("mid", ""), f"not signed: {e}")
                        self.log(f"{item['kind']} to {person[:12]} not sent: {e}")
                        continue
                    send_frame(ch, {"t": "env", "env": env, "body": item["body"]}, lock)
                if time.time() - last_ping > PING_S:
                    send_frame(ch, {"t": "ping"}, lock)
                    last_ping = time.time()
                wake.wait(PING_S)
                wake.clear()
                # anything still unacked after a wake is re-sent; the receiver dedupes by mid
        except (OSError, SecureChannelError, ValueError, RuntimeError) as e:
            self.log(f"session {person[:12]} send: {e}")
        finally:
            down.set()
            with self._lock:
                if self._sessions.get(person, (None,))[0] is ch:
                    del self._sessions[person]
            _hard_close(ch)

    def _receive(self, person: str, frame: dict) -> dict:
        """Verify one inbound envelope from a contact. Returns the ack/nack to send back."""
        env, body = frame.get("env") or {}, frame.get("body") or {}
        c = self.store.contact(person)
        mid = str(body.get("mid", "")) if isinstance(body, dict) else ""
        if not c or not mid:
            return {"t": "nack", "mid": mid, "why": "not a contact"}
        try:
            verify_envelope(env, c["person"])
        except SignerError as e:
            return {"t": "nack", "mid": mid, "why": f"does not verify: {e}"}
        if env.get("from_node") != c["node"] or env.get("to") != self.person or env.get("body_sha256") != body_hash(body):
            return {"t": "nack", "mid": mid, "why": "envelope not bound to this session"}
        kind = env.get("kind")
        if kind == "contact.accept":
            if c["state"] == "awaiting":
                self.store.set_contact_state(person, "active")
                self.log(f"{c['petname'] or person[:12]} accepted the connection")
            return {"t": "ack", "mid": mid}
        if c["state"] != "active":
            return {"t": "nack", "mid": mid, "why": "connection not accepted yet"}
        if isinstance(kind, str) and kind.startswith("task."):
            if self.store.is_seen(mid):
                return {"t": "ack", "mid": mid}
            try:
                self._task_receive(c, kind, body, env)
            except ValueError as e:
                return {"t": "nack", "mid": mid, "why": str(e)[:300]}
            self.store.first_time(mid)
            return {"t": "ack", "mid": mid}
        if kind != "msg":
            return {"t": "nack", "mid": mid, "why": f"{kind} is not handled here"}
        text = body.get("text")
        if not isinstance(text, str) or len(text) > MAX_TEXT:
            return {"t": "nack", "mid": mid, "why": "bad text"}
        if self.store.first_time(mid):
            self.store.raw_log({"received": int(time.time()), "source": "nakshatra", "trust": "external",
                                "from_person": person, "envelope": env, "body": body})
            self.store.add_inbox(mid, person, c["petname"], env.get("author", ""), str(body.get("aspect", ""))[:32], text)
        return {"t": "ack", "mid": mid}

    # ── tasks ────────────────────────────────────────────────────────────────────────────────
    # Poster: post_task → (claim arrives) assign the first valid claimer → (result arrives) judge by
    # the declared rules → accept/reject. Worker: offer arrives → claim_task → assign arrives →
    # submit_task → verdict arrives. Every step is a signed envelope on the contact session.

    def post_task(self, to, title: str, instructions: str, acceptance: list, reward: int = 0,
                  deadline_s: int = 3600) -> dict:
        names = to if isinstance(to, list) else [to]
        contacts = [c for c in self.store.contacts() if c["state"] == "active"]
        if names in (["*"], ["all"]):
            targets = contacts
        else:
            targets = [c for n in names for c in [self.store.resolve(str(n))] if c and c["state"] == "active"]
            if len(targets) != len(names):
                raise ValueError("every target must be an accepted contact")
        if not targets:
            raise ValueError("no accepted contacts to post to")
        why = outbound_problem(f"{title}\n{instructions}")
        if why:
            raise ValueError(why)
        spec = T.make_spec(self.person, title, instructions, acceptance, reward=reward, deadline_s=deadline_s)
        h = T.task_hash(spec)
        self.store.task_insert(h, "poster", "", spec, "posted", [c["person"] for c in targets])
        for c in targets:
            self._queue(c["person"], "task.post", {"task_hash": h, "spec": spec})
        self.log(f"posted task {h[:12]} '{title}' to {len(targets)} contact(s)")
        return {"task_hash": h, "posted_to": [c["petname"] or c["person"][:12] for c in targets]}

    def _worker_task(self, ref: str) -> dict:
        rows = [r for r in self.store.task_find(str(ref)) if r["role"] == "worker"]
        if len(rows) != 1:
            raise ValueError("no single offered task matches that id")
        return rows[0]

    def claim_task(self, ref: str) -> dict:
        t = self._worker_task(ref)
        if t["spec"]["deadline"] <= time.time():
            raise ValueError("that task's deadline has passed")
        if not self.store.task_cas(t["task_hash"], "worker", t["peer"], ("offered",), state="claimed"):
            raise ValueError(f"cannot claim a task that is {t['state']}")
        self._queue(t["peer"], "task.claim", {"task_hash": t["task_hash"]})
        return {"task_hash": t["task_hash"], "state": "claimed"}

    def submit_task(self, ref: str, output: str) -> dict:
        t = self._worker_task(ref)
        if not isinstance(output, str) or not output or len(output) > T.MAX_OUTPUT:
            raise ValueError(f"output must be 1-{T.MAX_OUTPUT} characters")
        why = outbound_problem(output)
        if why:
            raise ValueError(why)
        passed, reasons = T.evaluate(t["spec"], output)       # the worker can see it would fail first
        if not self.store.task_cas(t["task_hash"], "worker", t["peer"], ("assigned",),
                                   state="submitted", output=output):
            raise ValueError(f"cannot submit a task that is {t['state']} (it must be assigned to you)")
        self._queue(t["peer"], "task.result", {"task_hash": t["task_hash"], "epoch": t["epoch"], "output": output})
        return {"task_hash": t["task_hash"], "state": "submitted", "precheck_passed": passed, "precheck": reasons}

    def _task_receive(self, c: dict, kind: str, body: dict, env: dict) -> None:
        """Apply one verified task envelope from an active contact. ValueError = refuse (nack)."""
        h = body.get("task_hash")
        if not (isinstance(h, str) and len(h) == 64):
            raise ValueError("task_hash missing")
        who = c["petname"] or c["person"][:12]
        if kind == "task.post":
            spec = body.get("spec")
            T.validate_spec(spec)
            if T.task_hash(spec) != h or spec["poster"] != c["person"]:
                raise ValueError("task spec is not bound to its hash and poster")
            if self.store.task_insert(h, "worker", c["person"], spec, "offered"):
                self.store.raw_log({"received": int(time.time()), "source": "nakshatra", "trust": "external",
                                    "from_person": c["person"], "envelope": env, "body": body})
                self.log(f"task offer {h[:12]} '{spec['title']}' from {who}")
            return
        if kind == "task.claim":
            t = self.store.task_get(h, "poster", "")
            if not t:
                raise ValueError("no such task")
            if c["person"] not in (t["targets"] or []):
                raise ValueError("this task was not offered to you")
            if t["spec"]["deadline"] <= time.time():
                raise ValueError("deadline passed")
            if self.store.task_cas(h, "poster", "", ("posted",), state="assigned", assignee=c["person"], epoch=1):
                self._queue(c["person"], "task.assign", {"task_hash": h, "epoch": 1, "worker": c["person"]})
                self.log(f"task {h[:12]} assigned to {who}")
            elif t["assignee"] != c["person"]:
                self._queue(c["person"], "task.reject", {"task_hash": h, "epoch": 0,
                                                         "reasons": ["already assigned to someone else"]})
            return
        if kind == "task.assign":
            if body.get("worker") != self.person or not isinstance(body.get("epoch"), int):
                raise ValueError("assignment is not for this node's person")
            if not self.store.task_cas(h, "worker", c["person"], ("claimed",), state="assigned", epoch=body["epoch"]):
                raise ValueError("assignment for a task this node did not claim")
            self.log(f"task {h[:12]} assigned to us by {who}")
            return
        if kind == "task.result":
            t = self.store.task_get(h, "poster", "")
            if not t or t["assignee"] != c["person"] or body.get("epoch") != t["epoch"]:
                raise ValueError("result for a task not assigned to you")
            output = body.get("output")
            passed, reasons = T.evaluate(t["spec"], output)
            verdict = {"passed": passed, "reasons": reasons,
                       "output_sha256": hashlib.sha256(str(output).encode("utf-8")).hexdigest()}
            if not self.store.task_cas(h, "poster", "", ("assigned",), state="accepted" if passed else "rejected",
                                       output=output, verdict=json.dumps(verdict)):
                raise ValueError("task is not awaiting a result")
            self.store.raw_log({"received": int(time.time()), "source": "nakshatra", "trust": "external",
                                "from_person": c["person"], "envelope": env, "body": body})
            self._queue(c["person"], "task.accept" if passed else "task.reject",
                        {"task_hash": h, "epoch": t["epoch"], **verdict})
            self.log(f"task {h[:12]} result from {who}: {'ACCEPTED' if passed else 'REJECTED ' + str(reasons)}")
            return
        if kind in ("task.accept", "task.reject"):
            t = self.store.task_get(h, "worker", c["person"])
            if not t:
                raise ValueError("no such task here")
            reasons = body.get("reasons") if isinstance(body.get("reasons"), list) else []
            verdict = json.dumps({"passed": kind == "task.accept", "reasons": [str(r)[:300] for r in reasons][:12]})
            if body.get("epoch") == 0:
                self.store.task_cas(h, "worker", c["person"], ("claimed",), state="lost", verdict=verdict)
            else:
                self.store.task_cas(h, "worker", c["person"], ("submitted",),
                                    state="accepted" if kind == "task.accept" else "rejected", verdict=verdict)
            return
        raise ValueError(f"unknown task step {kind}")

    # ── read side ────────────────────────────────────────────────────────────────────────────
    def status(self) -> dict:
        with self._lock:
            online = set(self._sessions)
        return {"person": self.person, "node": self.node, "custody": self.custody,
                "relay": f"{self.relay[0]}:{self.relay[1]}",
                "contacts": [dict(c, online=c["person"] in online) for c in self.store.contacts()],
                "pending_requests": len(self.store.requests()),
                "open_invites": len(self.store.open_invites())}


def _close(sock) -> None:
    try:
        sock.close()
    except OSError:
        pass


def _hard_close(ch) -> None:
    """Unblock a reader parked in recv() on another thread, then close."""
    s = getattr(ch, "_sock", None)
    if s is not None:
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    ch.close()


# ── control socket (the CLI and, on Day 5, the MCP front door talk to this) ───────────────────

OPS = {"status", "register_invite", "redeem", "requests", "accept", "decline", "contacts", "remove", "send",
       "delivered", "inbox", "task_post", "tasks", "task_claim", "task_submit"}


def handle(node: Node, req: dict) -> dict:
    op = req.get("op")
    try:
        if op not in OPS:
            raise ValueError(f"unknown op {op!r}")
        if op == "status":
            return {"ok": True, **node.status()}
        if op == "register_invite":
            return {"ok": True, **node.register_invite(req["code"])}
        if op == "redeem":
            return {"ok": True, **node.redeem(req["code"], req.get("nickname", ""), req.get("petname", ""),
                                              float(req.get("wait_s", 0)))}
        if op == "requests":
            return {"ok": True, "requests": node.store.requests()}
        if op == "accept":
            return {"ok": True, **node.accept(req["id"], req.get("petname", ""))}
        if op == "decline":
            return {"ok": True, **node.decline(req["id"])}
        if op == "contacts":
            return {"ok": True, "contacts": node.status()["contacts"]}
        if op == "remove":
            return {"ok": True, **node.remove(req["who"])}
        if op == "send":
            return {"ok": True, **node.send(req["to"], req["text"], req.get("aspect", ""))}
        if op == "delivered":
            return {"ok": True, "delivered": node.delivered(req["mid"])}
        if op == "task_post":
            return {"ok": True, **node.post_task(req["to"], req["title"], req["instructions"], req["acceptance"],
                                                 int(req.get("reward", 0)), int(req.get("deadline_s", 3600)))}
        if op == "tasks":
            rows = node.store.tasks(req.get("role") or None, int(req.get("limit", 50)))
            for r in rows:
                c = node.store.contact(r["peer"] or (r["assignee"] or ""))
                r["peer_name"] = (c or {}).get("petname", "")
            return {"ok": True, "tasks": rows}
        if op == "task_claim":
            return {"ok": True, **node.claim_task(req["task"])}
        if op == "task_submit":
            return {"ok": True, **node.submit_task(req["task"], req["output"])}
        if op == "inbox":
            return {"ok": True, "messages": node.store.inbox(int(req.get("since", 0)), int(req.get("limit", 50)))}
    except (KeyError, ValueError, RuntimeError) as e:
        return {"ok": False, "error": str(e) if not isinstance(e, KeyError) else f"missing {e}"}
    return {"ok": False, "error": "unreachable"}


def serve_control(node: Node, sock_path: Path, stop: threading.Event) -> None:
    """Owner-only unix socket (0600, same uid). One JSON request per line."""
    sock_path = Path(sock_path)
    if sock_path.exists():
        sock_path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)
    try:
        srv.bind(str(sock_path))
    finally:
        os.umask(old)
    srv.listen(8)
    srv.settimeout(0.5)

    def one(conn):
        with conn:
            creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            if struct.unpack("3i", creds)[1] != os.getuid():
                return
            f = conn.makefile("rwb")
            for line in f:
                if len(line) > 64 * 1024:
                    return
                try:
                    reply = handle(node, json.loads(line))
                except (ValueError, TypeError):
                    reply = {"ok": False, "error": "malformed request"}
                f.write(json.dumps(reply, ensure_ascii=False).encode() + b"\n")
                f.flush()

    while not stop.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        threading.Thread(target=one, args=(conn,), daemon=True).start()
    srv.close()


def _load_node_key(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) == 32:
        return data
    raw = bytes.fromhex(data.decode("ascii").strip())
    if len(raw) != 32:
        raise SystemExit(f"{path} is not a 32-byte Ed25519 key")
    return raw


def main(argv=None) -> int:
    import argparse
    home = Path.home()
    ap = argparse.ArgumentParser(prog="nakd", description="Nakshatra messaging node")
    ap.add_argument("--state", type=Path, default=Path(os.environ.get("NAK_NET_DIR", home / ".nakshatra" / "net")))
    ap.add_argument("--node-key", type=Path, default=home / ".nakshatra" / "keys" / "worker.ed25519")
    ap.add_argument("--signer", type=Path, default=Path(os.environ.get(
        "NAK_SIGNER_SOCK", home / ".sthambha" / "signer" / "signer.sock")))
    ap.add_argument("--agent", default=os.environ.get("NAK_AGENT", ""))
    ap.add_argument("--relay", default=os.environ.get("NAK_RELAY", f"{DEFAULT_RELAY[0]}:{DEFAULT_RELAY[1]}"))
    a = ap.parse_args(argv)
    if not a.agent and (a.state / "agent").exists():
        a.agent = (a.state / "agent").read_text().strip()      # set once per node: `echo prithvi > …/net/agent`
    if not a.agent:
        raise SystemExit("--agent, NAK_AGENT or <state>/agent is required: the delegated agent this node signs as")
    host, port = a.relay.rsplit(":", 1)
    stop = threading.Event()
    node = Node(a.state, _load_node_key(a.node_key), SignerClient(a.signer), agent=a.agent,
                relay=(host, int(port)), log=lambda m: print(f"[nakd] {m}", flush=True)).start()
    print(f"[nakd] node {node.node[:16]} person {node.person[:16]} relay {a.relay} custody {node.custody}", flush=True)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        serve_control(node, a.state / "nakd.sock", stop)
    except KeyboardInterrupt:
        pass
    node.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
