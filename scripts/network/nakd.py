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

import base64
import hashlib
import json
import os
import random
import secrets
import shlex
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from typing import Callable, Optional

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nakshatra_auth import WORKER_KEY_PATH  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

import joincode  # noqa: E402
from mesh.pairing import pair_role  # noqa: E402
from network.agent import (AGENT_MSG_VERSION, CARD_SUPPORT, CARD_VERSION, INTENTS,
                           validate_agent_message, validate_card)  # noqa: E402
from network.store import Store  # noqa: E402
from network import tasks as T  # noqa: E402
from network import verdict as V  # noqa: E402
from network.settle import DEFINITIVE_STATUSES, HoldNotOpen, SettlementError  # noqa: E402
import accounting as _accounting  # noqa: E402
from transport.connect import (accept_direct, local_endpoints, open_channel, open_direct,  # noqa: E402
                               open_via_sidecar)
from transport.secure_channel import SecureChannelError  # noqa: E402

try:
    from sthambha.signer import MAX_SKEW_S, SignerError, canonical, verify_envelope
except ImportError:  # the release ships both; a dev checkout may point at a sthambha worktree
    _alt = os.environ.get("NAK_STHAMBHA_PATH")
    if not _alt:
        raise
    sys.path.insert(0, _alt)
    from sthambha.signer import MAX_SKEW_S, SignerError, canonical, verify_envelope  # noqa: E402

MSG_DOMAIN = "nak-msg-v1"
MAX_FRAME = 1 << 20
MAX_TEXT = 8000
SWEEP_S = 30.0          # how often expired tasks are swept (refund + notify)
APPROVAL_TIMEOUT_S = 700.0
APPROVAL_REPEAT_S = 6 * 3600
APPROVAL_FINAL_S = 2 * 3600
APPROVAL_MAX_PROMPTS = 8
GRANT_WARN_S = 7 * 86400
# Worst case is 5 advertised endpoints × 3 s, a 10 s handshake, then the sidecar's 15 s DCUtR wait
# plus its pinned handshake. Fifty seconds covers that sequence; only both-sides direct opt-ins wait.
DIRECT_WAIT_S = 50.0
P2P_TIMEOUT_S = 20.0
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


class DataKeyError(RuntimeError):
    """The signer answered but cannot supply the mandatory encrypted-store key."""


# ── the node ────────────────────────────────────────────────────────────────────────────────────

class EscrowRefused(ValueError):
    """Opening the escrow failed. `definitive` = the ledger answered with a refusal (no hold can
    exist); otherwise the outcome is unknown (timeout, 5xx, backlog) and a hold may exist."""

    def __init__(self, message: str, definitive: bool = False, hold: Optional[dict] = None,
                 hold_state: str = ""):
        super().__init__(message)
        self.definitive = definitive
        self.hold, self.hold_state = hold or {}, hold_state


REOPENED = "task reopened: "     # nack prefix: the claim was refused AND the task is claimable again


class Node:
    def __init__(self, state_dir: Path, node_key: bytes, signer: Callable[[dict], dict], *,
                 agent: str, relay: tuple = DEFAULT_RELAY, log: Callable[[str], None] = lambda m: None,
                 settlement=None, direct_port: Optional[int] = None, p2p: Optional[dict] = None,
                 roster_loader: Optional[Callable[[], dict]] = None):
        self._key = node_key
        self.node = ed25519.Ed25519PrivateKey.from_private_bytes(node_key).public_key().public_bytes_raw().hex()
        self._signer = signer
        self.agent = agent
        self.relay = relay
        self.log = log
        # Authenticate the signer's public identity before its data key can touch legacy state. A
        # valid but wrong signer must never get a chance to migrate this node's plaintext history.
        who = signer({"op": "whoami"})
        if not who.get("ok"):
            raise RuntimeError(f"signer: {who.get('error')}")
        if who["node"] != self.node:
            raise RuntimeError("the signer is set up for a different node key")
        if agent not in who.get("agents", []):
            raise RuntimeError(f"the signer has no delegation for agent {agent!r}")
        self.person = who["person"]
        self.custody = who.get("custody", "")
        key_reply = signer({"op": "data_key"})
        if not key_reply.get("ok"):
            raise DataKeyError(f"signer data key unavailable: {key_reply.get('error')}")
        try:
            data_key = base64.b64decode(key_reply["data_key"], validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise DataKeyError("signer returned an invalid data key") from exc
        if len(data_key) != 32:
            raise DataKeyError("signer returned an invalid data key")
        self.store = Store(state_dir, data_key)
        # Adapter #0 (local ledger, TEST units) unless a real one is passed; see network/settle.py.
        # THE accounting front door (accounting.py, U5) picks the escrow backend; tests may pass one.
        self.settle = settlement or _accounting.escrow(
            Path(state_dir), private_key=self._key, keyid=self.node)
        self._harden_state_files(Path(state_dir))
        self.direct_port = direct_port        # U3b: listen for direct connections (opt-in, per node)
        self.p2p = dict(p2p) if p2p else None  # {"dial": "/run/user/<uid>/nakshatra/p2p.sock"}
        self._paths: dict[str, str] = {}      # contact person -> "direct" | "p2p" | "relay"
        self._listener = None
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}       # key -> worker thread
        self._sessions: dict[str, tuple] = {}                 # contact person -> (channel, send lock)
        self._wake: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._approval_lock = threading.Lock()
        self._approval_running: set[str] = set()
        self._roster_loader = roster_loader
        self._last_grant_warning_day: Optional[str] = None
        self._warn_grant(force=True)

    @staticmethod
    def _harden_state_files(state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(state_dir, 0o700)
        names = ("net.sqlite", "net.sqlite-wal", "net.sqlite-shm", "inbox-raw.jsonl",
                 "settle.sqlite", "settle.sqlite-wal", "settle.sqlite-shm",
                 "agent", "name", "direct-port", "p2p-dial")
        for name in names:
            try:
                os.chmod(state_dir / name, 0o600)
            except FileNotFoundError:
                pass

    # lifecycle
    def start(self) -> "Node":
        for inv in self.store.open_invites():
            self._spawn(f"invite:{inv['nonce']}", self._invite_loop, inv)
        for c in self.store.contacts():
            self._ensure_session(c["person"])
        self._spawn("sweeper", self._sweep_loop)
        self._rearm_approvals()
        if self.direct_port:
            self._spawn("direct-listener", self._listen_loop)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
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

    def _sign(self, kind: str, to: str, body: dict, *, touch: bool = True) -> dict:
        r = self._signer({"op": "sign_envelope", "agent": self.agent, "kind": kind, "to": to,
                          "body_sha256": body_hash(body), "touch": touch})
        if not r.get("ok"):
            raise RuntimeError(f"signer refused: {r.get('error')}")
        return r["envelope"]

    def _touch_signing_activity(self, kind: str, to: str, body: dict) -> None:
        """Touch once at the initiating boundary; background delivery signs touch=false."""
        r = self._signer({"op": "sign_envelope", "agent": self.agent, "kind": kind, "to": to,
                          "body_sha256": body_hash(body), "touch": True})
        if not r.get("ok"):
            raise RuntimeError(f"signer refused: {r.get('error')}")

    def _grant_status(self, kind: str = "") -> dict:
        req = {"op": "grant_status", "agent": self.agent}
        if kind:
            req["kind"] = kind
        r = self._signer(req)
        if not r.get("ok"):
            raise RuntimeError(f"cannot inspect grant for agent {self.agent!r}: {r.get('error')}")
        constraints = r.get("constraints")
        if not isinstance(constraints, dict):
            constraints = self._local_grant_constraints()
        r["verdict_policy"] = V.policy(constraints)
        return r

    def _local_grant_constraints(self) -> dict:
        """Best-effort status view of the signed local grant; enforcement uses the signed spend auth."""
        signer_dir = Path(os.environ.get("NAK_SIGNER_DIR", Path.home() / ".sthambha" / "signer"))
        try:
            grant = json.loads((signer_dir / "agents" / f"{self.agent}.grant.json").read_text())
            from sthambha.signer import _verify  # noqa: PLC0415, SLF001 - authenticate public metadata
            if grant.get("subject") == self.agent and _verify(grant, self.person):
                return grant.get("constraints") if isinstance(grant.get("constraints"), dict) else {}
        except (ImportError, OSError, ValueError, TypeError):
            pass
        return {}

    @staticmethod
    def _utc(ts: int) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(int(ts)))

    @classmethod
    def _grant_status_text(cls, grant: dict) -> str:
        lease = grant.get("lease") or {}
        if not lease.get("enabled"):
            return f"grant expires {cls._utc(grant.get('expires_at', 0))}"
        problem = str(grant.get("problem") or "")
        if grant.get("valid") is False and problem.startswith("lapsed: unused since "):
            unused_since = problem.removeprefix("lapsed: unused since ")
            return f"LAPSED: unused since {unused_since}. Wake it with: nak grant renew"
        idle_days = int(lease.get("idle_s", 0)) // 86400
        day_word = "day" if idle_days == 1 else "days"
        return (f"active while used — lapses if unused for {idle_days} {day_word} "
                f"(next: {cls._utc(lease.get('lapses_at', 0))}); charter until "
                f"{cls._utc(grant.get('expires_at', 0))}; renew wakes it")

    @classmethod
    def _grant_warning_text(cls, grant: dict, now: Optional[int] = None) -> str:
        now = int(now if now is not None else time.time())
        if grant.get("revoked"):
            return f"grant {grant.get('grant_id') or '(unknown)'} is revoked"
        if grant.get("valid") is False:
            problem = str(grant.get("problem") or "signer rejected it")
            if problem.startswith("lapsed: unused since "):
                return "LAPSED: unused since " + problem.removeprefix("lapsed: unused since ")
            return f"grant invalid: {problem}"
        if now + MAX_SKEW_S < int(grant.get("not_before", 0)):
            return f"grant is not valid until {cls._utc(grant['not_before'])}"
        expires = int(grant.get("expires_at", 0))
        if now >= expires:
            return f"grant expired at {cls._utc(expires)}"
        left = expires - now
        if left < GRANT_WARN_S:
            return f"grant expires at {cls._utc(expires)} ({left // 86400}d {left % 86400 // 3600}h left)"
        return ""

    def _renewal_command(self, grant: dict, required_cap: str = "") -> str:
        caps = sorted(set(grant.get("caps") or []))
        if required_cap and required_cap not in caps:
            caps.append(required_cap)
            return "nak grant renew --caps " + shlex.quote(",".join(sorted(caps)))
        return "nak grant renew"

    def _require_grant(self, kind: str) -> dict:
        grant = self._grant_status(kind)
        if grant.get("valid"):
            return grant
        cap = str(grant.get("required_cap") or "")
        if grant.get("revoked"):
            problem = f"grant {grant.get('grant_id') or '(unknown)'} is revoked"
        elif cap and cap not in grant.get("caps", []):
            problem = f"cap {cap} not granted"
        else:
            problem = str(grant.get("problem") or "grant is not valid for this node")
        raise ValueError(f"agent {self.agent!r} cannot sign {kind}: {problem}. Human renewal: "
                         f"{self._renewal_command(grant, cap)}")

    def _warn_grant(self, force: bool = False, now: Optional[int] = None) -> None:
        now = int(now if now is not None else time.time())
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if not force and self._last_grant_warning_day == day:
            return
        try:
            grant = self._grant_status()
            warning = self._grant_warning_text(grant, now)
        except RuntimeError as exc:
            warning = str(exc)
        self._last_grant_warning_day = day
        if warning:
            self.log(f"WARNING: agent {self.agent!r} {warning}; human renewal: "
                     f"{self._renewal_command(grant) if 'grant' in locals() else 'check the signer service'}")

    def _dial(self, rid: bytes, my_key: bytes, pin: str, initiator: bool, binding: bytes):
        # THE one way to reach a peer (transport/connect.py) — the same path meshd's tunnels use.
        return open_channel(relay=self.relay, rendezvous_id=rid, my_key=my_key, peer_pub_hex=pin,
                            initiator=initiator, binding=binding, connect_timeout=15, wait_timeout=WAIT_S)

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
        self._require_grant("contact.request")
        # The inviter appears under the name YOU give them, else the name they gave themselves. The note
        # is about whom the invite is for, never a name for the inviter.
        self.store.upsert_contact(inv["inviter"], inv["inviter_node"], "awaiting",
                                  petname or inv.get("from_name", "") or inv["inviter"][:12],
                                  name_src="yours" if petname else "self-chosen")
        done = threading.Event()
        self._spawn(f"redeem:{inv['nonce']}", self._redeem_loop, inv, nickname[:40], done)
        if wait_s:
            done.wait(wait_s)
        c = self.store.contact(inv["inviter"]) or {}
        return {"state": "request sent" if done.is_set() else "request queued", "person": inv["inviter"],
                "name": c.get("petname", "")}

    def _redeem_loop(self, inv: dict, nickname: str, done: threading.Event) -> None:
        body = {"person": self.person, "node": self.node, "nickname": nickname, "invite": inv["nonce"],
                "mid": secrets.token_hex(12)}
        ik = bytes.fromhex(inv["ik"])
        rid = joincode.invite_rendezvous(inv)
        attempt = 0
        touched = False
        while not self._stop.is_set() and time.time() < inv["expires_at"]:
            try:
                sock, ch = self._dial(rid, ik, inv["inviter_node"], True, b"nak-invite-v1|" + inv["nonce"].encode())
            except (OSError, SecureChannelError):
                attempt += 1
                self._backoff(attempt)
                continue
            try:
                sock.settimeout(30)
                env = self._sign("contact.request", inv["inviter"], body, touch=not touched)
                touched = True
                send_frame(ch, {"t": "env", "env": env, "body": body})
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
        self._require_grant("contact.accept")
        r = self.store.decide_request(req_id, True)
        if not r:
            raise ValueError("no pending request with that id")
        self.store.upsert_contact(r["person"], r["node"], "active", petname or r["nickname"], r["nickname"],
                                  name_src="yours" if petname else "self-chosen")
        self._queue(r["person"], "contact.accept", {"person": self.person, "node": self.node})
        self._send_card_if_set(r["person"])
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
        self._require_grant("msg")
        mid = self._queue(c["person"], "msg", {"text": text, "aspect": str(aspect)[:32]})
        return {"queued": mid, "to": c["petname"] or c["person"][:12]}

    def agent_send(self, who: str, text: str, *, intent: str = "ask", thread: str = "",
                   reply_to: str = "", want=None, expires=None, budget=None, refs=None,
                   harness=None) -> dict:
        c = self.store.resolve(who)
        if not c:
            raise ValueError("not a contact: only people who accepted a connection can be messaged")
        if c["state"] != "active":
            raise ValueError("that connection has not been accepted yet")
        name = c["petname"] or c["person"][:12]
        if self.store.agent_card(c["person"]) is None:
            raise ValueError(f"{name} runs an older node (no agent card yet); use nak_send")
        self._require_grant("agent.msg")
        body = {"v": AGENT_MSG_VERSION, "mid": secrets.token_hex(12),
                "thread": thread or secrets.token_hex(16), "reply_to": reply_to or "",
                "intent": intent, "text": text}
        for key, value in (("want", want), ("expires", expires), ("budget", budget), ("refs", refs),
                           ("harness", harness)):
            if value is not None:
                body[key] = value
        body = validate_agent_message(
            body, MAX_TEXT, outbound=True,
            reply_known=lambda known_thread, known_mid: self.store.agent_mid_known(
                c["person"], known_thread, known_mid))
        for value in (body["text"], body.get("want", "")):
            why = outbound_problem(value)
            if why:
                self.log(f"agent send to {c['person'][:12]} refused: {why}")
                raise ValueError(why)
        self._touch_signing_activity("agent.msg", c["person"], body)
        self.store.add_agent_message(body, c["person"], "out")
        self.store.queue(body["mid"], c["person"], {"kind": "agent.msg", "body": body})
        ev = self._wake.get(c["person"])
        if ev:
            ev.set()
        return {"queued": body["mid"], "to": name, "thread": body["thread"],
                "expires": body.get("expires")}

    def _card_body(self) -> Optional[dict]:
        card = self.store.agent_card(self.person)
        return dict(card, v=CARD_VERSION) if card else None

    def _send_card_if_set(self, person: str, *, touch: bool = True) -> Optional[str]:
        body = self._card_body()
        return self._queue(person, "contact.card", body, touch=touch) if body else None

    def card_set(self, display_name: str, offers: str, *, intents=None, models=None,
                 price_hint: str = "", paid: bool = False) -> dict:
        if paid and self.settle.name != "neuron":
            raise ValueError("paid cards require NAKSHATRA_ESCROW=neuron")
        previous = self.store.agent_card(self.person)
        updated_at = max(int(time.time()), int(previous["updated_at"]) + 1 if previous else 0)
        card = {"v": CARD_VERSION, "mid": "0" * 24, "display_name": display_name,
                "offers": offers, "intents": list(intents if intents is not None else INTENTS),
                "paid": bool(paid), "price_hint": price_hint, "models": list(models or []),
                "supports": [CARD_SUPPORT], "updated_at": updated_at}
        if paid:
            card["ledger_keyid"] = self.node
        validate_card(card)
        stored = {key: value for key, value in card.items() if key not in {"v", "mid"}}
        self.store.set_agent_card(self.person, stored)
        sent = []
        for contact in self.store.contacts():
            if contact["state"] == "active":
                sent.append(self._send_card_if_set(contact["person"]))
        return {"card": stored, "sent": len(sent)}

    def card(self, who: str = "") -> dict:
        if not who:
            card = self.store.agent_card(self.person)
            if not card:
                raise ValueError("no agent card is set; use: nak card set ...")
            return {"card": card, "person": self.person, "mine": True}
        c = self.store.resolve(who)
        if not c:
            raise ValueError("no such contact")
        card = self.store.agent_card(c["person"])
        if not card:
            raise ValueError(f"{c['petname'] or c['person'][:12]} has no agent card yet")
        return {"card": card, "person": c["person"], "mine": False,
                "contact": c["petname"] or c["person"][:12]}

    def threads(self) -> list:
        rows = self.store.agent_threads()
        for row in rows:
            contact = self.store.contact(row["peer"])
            row["contact"] = (contact or {}).get("petname") or row["peer"][:12]
            row["name_source"] = (contact or {}).get("name_src") or "self-chosen"
        return rows

    def thread(self, prefix: str, who: str = "") -> tuple[str, list]:
        peer = None
        if who:
            contact = self.store.resolve(who)
            if not contact:
                raise ValueError("no such contact")
            peer = contact["person"]
        thread, rows = self.store.agent_thread(prefix, peer)
        for row in rows:
            contact = self.store.contact(row["peer"])
            name = (contact or {}).get("petname") or row["peer"][:12]
            row["from_me"] = row["direction"] == "out"
            if row["from_me"]:
                row["from"] = "me"
            else:
                row["from"] = name
                row["name"] = name
                row["from_key"] = row["peer"][:16]
                row["name_source"] = (contact or {}).get("name_src") or "self-chosen"
        return thread, rows

    def _queue(self, person: str, kind: str, body: dict, *, touch: bool = True) -> str:
        self._require_grant(kind)
        mid = secrets.token_hex(12)
        queued_body = dict(body, mid=mid)
        if touch:
            # Record the initiating human/agent activity now. Delivery happens in the daemon's
            # background sender and is always touch=false, including its retries.
            self._touch_signing_activity(kind, person, queued_body)
        self.store.queue(mid, person, {"kind": kind, "body": queued_body})
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

    def _p2p_available(self) -> bool:
        """Advertise/use p2p only while the opted-in sidecar's uid-gated socket actually exists."""
        try:
            return bool(self.p2p and Path(str(self.p2p.get("dial", ""))).is_socket())
        except OSError:
            return False

    def _session_loop(self, person: str) -> None:
        attempt = 0
        while not self._stop.is_set():
            c = self.store.contact(person)
            if not c:
                return
            if person in self._sessions:          # e.g. they reached us directly: one session per contact
                self._stop.wait(1.0)
                continue
            sock = ch = None
            path = "relay"
            p2p_failed = False
            role0 = pair_role(self.node, c["node"], MSG_DOMAIN)
            can_p2p = bool(c.get("direct") and c.get("p2p") and self._p2p_available())
            if c.get("direct") and (c.get("direct_hint") or can_p2p) and not role0.is_initiator:
                # Exactly ONE side dials direct (the pairing initiator, as on the relay); the other waits
                # briefly for that call before falling back, so the two never cross and refuse each other.
                end = time.time() + DIRECT_WAIT_S
                while time.time() < end and person not in self._sessions and not self._stop.is_set():
                    self._stop.wait(0.2)
                if person in self._sessions:
                    continue
            if c.get("direct") and c.get("direct_hint") and role0.is_initiator:
                try:
                    sock, ch = open_direct(c["direct_hint"], my_key=self._key, peer_pub_hex=c["node"],
                                           purpose=MSG_DOMAIN.encode())
                    path = "direct"
                except (OSError, SecureChannelError) as e:
                    self.log(f"direct to {c['petname'] or person[:12]} failed ({e})")
            if ch is None and can_p2p and role0.is_initiator:
                try:
                    sock, ch = open_via_sidecar(self.p2p["dial"], c["node"], my_key=self._key,
                                                purpose=MSG_DOMAIN.encode(), timeout=P2P_TIMEOUT_S)
                    path = "p2p"
                except (OSError, SecureChannelError, ValueError) as e:
                    # A failed attempt means the cached capability may be stale. Forget it locally and
                    # mark it negative to the peer once the relay fallback is established.
                    self.store.clear_direct_addr(person)
                    p2p_failed = True
                    self.log(f"direct (p2p) to {c['petname'] or person[:12]} failed ({e}); using the relay")
            if ch is None:
                role = pair_role(self.node, c["node"], MSG_DOMAIN)
                try:
                    sock, ch = self._dial(role.rendezvous_id, self._key, c["node"], role.is_initiator,
                                          MSG_DOMAIN.encode())
                except (OSError, SecureChannelError):
                    attempt += 1
                    self._backoff(attempt)
                    continue
            path_label = "direct (p2p)" if path == "p2p" else path
            self.log(f"session up with {c['petname'] or person[:12]} ({path_label})")
            started = time.time()
            self._run_session(person, sock, ch, path=path, p2p_failed=p2p_failed)
            # A session that drops at once is a failure, not a success: keep backing off unless it
            # actually lasted (a peer that accepts and hangs up must not drive a tight loop).
            attempt = 0 if time.time() - started > 60 else attempt + 1
            self._backoff(attempt)

    def _run_session(self, person: str, sock, ch, path: str = "relay", p2p_failed: bool = False) -> None:
        lock = threading.Lock()
        with self._lock:
            if person in self._sessions:          # lost a race with another path: keep the first
                _hard_close(ch)
                return
            self._sessions[person] = (ch, lock)
            self._paths[person] = path
        advertised = False

        def advertise_endpoints_once() -> None:
            # Endpoint disclosure happens only after the peer has proved its opt-in by sending either
            # {t:addr?, direct:true} or its own addr frame. Older addr? frames remain parse-compatible,
            # but cannot trigger disclosure because they carry no affirmative capability bit.
            nonlocal advertised
            cc = self.store.contact(person) or {}
            if p2p_failed or advertised or not (cc.get("direct") and self.direct_port):
                return
            send_frame(ch, {"t": "addr", "endpoints": local_endpoints(self.direct_port),
                            "p2p": self._p2p_available()}, lock)
            advertised = True

        c0 = self.store.contact(person) or {}
        if p2p_failed:
            try:
                send_frame(ch, {"t": "addr", "endpoints": [], "p2p": False, "failed": True}, lock)
            except (OSError, SecureChannelError):
                pass
        elif c0.get("direct"):
            try:
                # Capability first, with no endpoints. Both peers must opt in before either reveals any.
                send_frame(ch, {"t": "addr?", "direct": True}, lock)
            except (OSError, SecureChannelError):
                pass
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
                        self.store.ack(str(frame.get("mid", "")), person)
                    elif t == "addr?":  # a direct:true capability proves their side opted in
                        cc = self.store.contact(person) or {}
                        if frame.get("direct") is True and cc.get("direct"):
                            advertise_endpoints_once()
                    elif t == "addr":   # their direct endpoints; kept only if WE opted this contact in
                        hint = frame.get("endpoints")
                        cc = self.store.contact(person) or {}
                        if hint == [] and frame.get("p2p") is False:
                            self.store.clear_direct_addr(person)
                        elif not p2p_failed and cc.get("direct") and isinstance(hint, str) and len(hint) <= 1000:
                            self.store.set_direct_addr(person, hint, frame.get("p2p") is True)
                            # A positive advertisement also proves remote opt-in. This completes the
                            # exchange when our earlier capability arrived before they enabled direct.
                            advertise_endpoints_once()
                    elif t == "nack":   # received and refused: terminal, stop retrying, keep why
                        self.store.refuse(str(frame.get("mid", "")), str(frame.get("why", "")), person)
                        if str(frame.get("why", "")).startswith(REOPENED):
                            self._claim_reopened(str(frame.get("mid", "")), person)
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
                        env = self._sign(item["kind"], person, item["body"], touch=False)
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
                    self._paths.pop(person, None)
            _hard_close(ch)

    def _receive(self, person: str, frame: dict) -> dict:
        """Verify one inbound envelope from a contact. Returns the ack/nack to send back. Anything a
        peer sends that makes processing blow up is refused, never allowed to kill the session."""
        try:
            return self._receive_checked(person, frame)
        except Exception as e:  # noqa: BLE001 — peer-controlled input; refuse, keep the reader alive
            body = frame.get("body") if isinstance(frame, dict) else None
            mid = str(body.get("mid", "")) if isinstance(body, dict) else ""
            self.log(f"refused malformed frame from {person[:12]}: {type(e).__name__}")
            return {"t": "nack", "mid": mid, "why": "malformed"}

    def _receive_checked(self, person: str, frame: dict) -> dict:
        env, body = frame.get("env") or {}, frame.get("body") or {}
        if not isinstance(env, dict) or not isinstance(body, dict):
            return {"t": "nack", "mid": "", "why": "malformed"}
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
                # This publication is forced by the peer's accept, not initiated locally.
                self._send_card_if_set(person, touch=False)
            return {"t": "ack", "mid": mid}
        if c["state"] != "active":
            return {"t": "nack", "mid": mid, "why": "connection not accepted yet"}
        if kind == "contact.card":
            card_body = validate_card(body)
            if card_body.get("ledger_keyid") and card_body["ledger_keyid"] != c["node"]:
                self.log(f"refused card from {c['petname'] or person[:12]}: ledger_keyid is not its node key")
                return {"t": "nack", "mid": mid, "why": "card ledger_keyid is not this contact's node key"}
            if self.store.first_time(mid):
                self.store.raw_log({"received": int(time.time()), "source": "nakshatra", "trust": "external",
                                    "from_person": person, "envelope": env, "body": body})
                card = {key: value for key, value in card_body.items() if key not in {"v", "mid"}}
                self.store.set_agent_card(person, card)
            return {"t": "ack", "mid": mid}
        if kind == "agent.msg":
            message = validate_agent_message(body, MAX_TEXT)
            if message.get("expires") is not None and message["expires"] <= int(time.time()):
                if self.store.first_time(mid):
                    self.log(f"dropped expired agent.msg {mid[:8]} from {person[:12]}")
                return {"t": "ack", "mid": mid}
            if self.store.first_time(mid):
                self.store.raw_log({"received": int(time.time()), "source": "nakshatra", "trust": "external",
                                    "from_person": person, "envelope": env, "body": body})
                self.store.add_agent_message(message, person, "in")
            return {"t": "ack", "mid": mid}
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

    # ── direct paths (U3b) ───────────────────────────────────────────────────────────────────
    def set_direct(self, who: str, on: bool) -> dict:
        """Opt a contact in/out of direct connections. On means: we tell THEM our LAN/IPv6 addresses
        (inside the encrypted session) and accept their direct connections. It reveals where we are,
        so it is per contact and off by default. A direct path is used only if BOTH sides opted in."""
        c = self.store.resolve(who)
        if not c or c["state"] != "active":
            raise ValueError("direct paths are only for accepted contacts")
        self.store.set_direct(c["person"], on)
        with self._lock:
            sess = self._sessions.get(c["person"])
            path = self._paths.get(c["person"])
        if sess:
            try:
                if on:
                    # No endpoints here: the peer must answer with its own direct:true capability first.
                    send_frame(sess[0], {"t": "addr?", "direct": True}, sess[1])
                else:
                    send_frame(sess[0], {"t": "addr", "endpoints": [], "p2p": False}, sess[1])
            except (OSError, SecureChannelError):
                pass
            if not on and path in ("direct", "p2p"):
                # Revocation takes effect immediately. The session loops reconnect over the relay.
                _hard_close(sess[0])
        return {"person": c["person"], "direct": bool(on), "listening": bool(self.direct_port)}

    def _listen_loop(self) -> None:
        srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)   # one socket for v4 + v6
        except OSError:
            pass
        srv.bind(("::", int(self.direct_port)))
        srv.listen(16)
        srv.settimeout(1.0)
        self._listener = srv
        self.log(f"accepting direct connections on port {self.direct_port}")
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._direct_incoming, args=(conn,), daemon=True).start()

    def _direct_incoming(self, conn) -> None:
        by_node = {}

        def allow(peer_node: str) -> bool:
            for c in self.store.contacts():
                if c["node"] == peer_node and c["state"] == "active" and c.get("direct") \
                        and c["person"] not in self._sessions:
                    by_node[peer_node] = c
                    return True
            return False
        try:
            peer, ch = accept_direct(conn, my_key=self._key, purpose=MSG_DOMAIN.encode(), allow=allow)
        except (OSError, SecureChannelError, PermissionError) as e:
            self.log(f"refused a direct connection: {e}")
            return
        c = by_node[peer]
        incoming_path = "p2p" if self.p2p and _is_loopback_peer(conn) else "direct"
        label = "direct (p2p)" if incoming_path == "p2p" else "direct, incoming"
        self.log(f"session up with {c['petname'] or c['person'][:12]} ({label})")
        self._run_session(c["person"], conn, ch, path=incoming_path)

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
        self._require_grant("task.post")
        spec = T.make_spec(self.person, title, instructions, acceptance, reward=reward,
                           deadline_s=deadline_s,
                           unit="credits" if self.settle.open_on_assign else "TEST")
        h = T.task_hash(spec)
        escrow = None
        authorisation = None
        if spec["reward"]["amount"] > 0:
            authorisation = self._approve_reward(h, spec)  # cap check always happens before anything is sent
            if not self.settle.open_on_assign:
                escrow = self._open_escrow(h, spec, authorisation=authorisation)
        funding = "unfunded" if self.settle.open_on_assign and spec["reward"]["amount"] > 0 else ""
        self.store.task_insert(h, "poster", "", spec, "posted", [c["person"] for c in targets],
                               escrow_auth=authorisation if self.settle.open_on_assign else None,
                               funding_state=funding)
        for c in targets:
            self._queue(c["person"], "task.post", {"task_hash": h, "spec": spec})
        self.log(f"posted task {h[:12]} '{title}' to {len(targets)} contact(s)")
        return {"task_hash": h, "posted_to": [c["petname"] or c["person"][:12] for c in targets], "escrow": escrow}

    def _approve_reward(self, h: str, spec: dict) -> dict:
        amount, unit = spec["reward"]["amount"], spec["reward"]["unit"]
        r = self._signer({"op": "approve_spend", "agent": self.agent, "task_hash": h, "epoch": 0,
                          "amount": amount, "mint": unit, "to": "escrow", "touch": True})
        if not r.get("ok"):
            raise ValueError(f"the signer refused the reward: {r.get('error')}")
        if r.get("status") == "needs_human":
            raise ValueError(f"this reward needs your person's approval (over the agent's cap): {r.get('description')}")
        return r["authorisation"]

    def _open_escrow(self, h: str, spec: dict, *, authorisation: dict,
                     worker_keyid: str = "") -> dict:
        amount, unit = spec["reward"]["amount"], spec["reward"]["unit"]
        try:
            return self.settle.open(h, self.person, amount, unit, spec["deadline"], authorisation,
                                    worker_keyid=worker_keyid)
        except SettlementError as e:
            if isinstance(e, HoldNotOpen):      # a replayed authorize found the hold already terminal
                raise EscrowRefused(f"escrow: {e}", definitive=True, hold=e.hold, hold_state=e.state)
            raise EscrowRefused(f"escrow: {e}", definitive=e.status in DEFINITIVE_STATUSES)

    def _set_funding(self, h: str, role: str, peer: str, state: str, reason: str = "") -> bool:
        task = self.store.task_get(h, role, peer)
        return bool(task and self.store.task_cas(
            h, role, peer, (task["state"],), funding_state=state, funding_reason=reason))

    def _settle(self, h: str, spec: dict, how: str, party: str = "", reason: str = "",
                hold_id: str = "") -> Optional[dict]:
        """release|refund the task's escrow, if it has one. Exactly-once is the adapter's job."""
        if spec["reward"]["amount"] <= 0:
            return None
        if self.settle.open_on_assign and not hold_id:
            return None                         # a Neuron task expired before anyone was assigned
        try:
            receipt = (self.settle.release(h, party, hold_id=hold_id) if how == "release"
                       else self.settle.refund(h, reason, hold_id=hold_id))
            if self.settle.open_on_assign:
                self._set_funding(h, "poster", "", "paid" if how == "release" else "refunded")
            return receipt
        except SettlementError as e:
            self.log(f"settlement {how} for {h[:12]} refused: {e}")
            if self.settle.open_on_assign:
                self._set_funding(h, "poster", "", "pending", str(e))
            return {"error": str(e)}

    def _sweep_loop(self) -> None:
        while not self._stop.wait(SWEEP_S):
            try:
                self._warn_grant()
                self.sweep()
            except Exception as e:  # noqa: BLE001 — a sweep bug must never kill the node
                self.log(f"sweep failed: {e}")

    @staticmethod
    def _task_constraints(task: dict) -> dict:
        auth = task.get("escrow_auth") or {}
        grant = auth.get("grant") if isinstance(auth, dict) else {}
        return grant.get("constraints") if isinstance(grant, dict) and \
            isinstance(grant.get("constraints"), dict) else {}

    def _worker_is_stranger(self, contact: dict) -> bool:
        try:
            if self._roster_loader is not None:
                peers = self._roster_loader() or {}
            else:
                from fabric import worker_join  # noqa: PLC0415
                peers = worker_join._load_admission().load_peers() or {}  # noqa: SLF001
            row = peers.get(contact.get("node")) or {}
            return row.get("tier") not in {"known", "trusted", "self"}
        except Exception:  # noqa: BLE001 - absent/unreadable roster is deliberately stranger
            return True

    @staticmethod
    def _credits_text(amount: int) -> str:
        whole, fraction = divmod(int(amount), 100_000_000)
        digits = f"{fraction:08d}".rstrip("0")
        if len(digits) < 2:
            digits = digits.ljust(2, "0")
        return f"{whole}.{digits}"

    def _approval_description(self, task: dict) -> str:
        contact = self.store.contact(task["assignee"]) or {}
        worker = " ".join(str(contact.get("petname") or task["assignee"][:12]).split())[:40]
        title = " ".join(str(task["spec"]["title"]).split())[:60]
        reward = self._credits_text(task["spec"]["reward"]["amount"])
        return (f"Pay {worker} {reward} USDC for task '{title}' ({task['task_hash'][:12]})? "
                "The answer passed the task's rules.")

    @staticmethod
    def _approval_due(task: dict, now: float) -> bool:
        approval = (task.get("verdict") or {}).get("approval") or {}
        count = int(approval.get("prompt_count") or 0)
        last = int(approval.get("last_prompted_at") or 0)
        if now >= int(task["spec"]["deadline"]):
            return False
        if count >= APPROVAL_MAX_PROMPTS:
            return False
        if count == 0:
            return True
        final_at = int(task["spec"]["deadline"]) - APPROVAL_FINAL_S
        if now >= final_at and last < final_at:
            return True
        return now < final_at and now - last >= APPROVAL_REPEAT_S

    def _arm_approval(self, task_hash: str) -> None:
        command = os.environ.get("NAK_APPROVAL_CMD", "")
        if not command:
            return
        if not Path(command).is_absolute():
            self.log("NAK_APPROVAL_CMD must be an absolute path; approval remains unanswered")
            return
        with self._approval_lock:
            if task_hash in self._approval_running:
                return
            task = self.store.task_get(task_hash, "poster", "")
            if not task or task["state"] != "awaiting_approval" or not self._approval_due(task, time.time()):
                return
            self._approval_running.add(task_hash)
        threading.Thread(target=self._run_approval_hook, args=(task_hash, command), daemon=True,
                         name=f"approval:{task_hash}").start()

    def _run_approval_hook(self, task_hash: str, command: str) -> None:
        try:
            task = self.store.task_get(task_hash, "poster", "")
            if not task or task["state"] != "awaiting_approval":
                return
            verdict = dict(task.get("verdict") or {})
            approval = dict(verdict.get("approval") or {})
            now = int(time.time())
            approval["last_prompted_at"] = now
            approval["prompt_count"] = int(approval.get("prompt_count") or 0) + 1
            verdict["approval"] = approval
            if not self.store.task_cas(task_hash, "poster", "", ("awaiting_approval",),
                                       verdict=json.dumps(verdict)):
                return
            try:
                result = subprocess.run([command, self._approval_description(task)], check=False,
                                        timeout=APPROVAL_TIMEOUT_S,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                code = result.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.log(f"approval hook for {task_hash[:12]} unanswered ({type(exc).__name__})")
                return
            if code == 0:
                try:
                    self._apply_task_decision(task_hash, True, "", source="approval hook", touch=True)
                except ValueError as exc:
                    self.log(f"approval hook for {task_hash[:12]} was superseded: {exc}")
            elif code == 10:
                try:
                    self._apply_task_decision(task_hash, False, "approval hook rejected",
                                              source="approval hook", touch=True)
                except ValueError as exc:
                    self.log(f"approval hook for {task_hash[:12]} was superseded: {exc}")
            else:
                self.log(f"approval hook for {task_hash[:12]} unanswered (exit {code})")
        finally:
            with self._approval_lock:
                self._approval_running.discard(task_hash)

    def _rearm_approvals(self) -> None:
        for task in self.store.tasks_in_states(("awaiting_approval",), role="poster"):
            self._arm_approval(task["task_hash"])

    def _apply_task_decision(self, task_hash: str, accept: bool, reason: str, *, source: str,
                             touch: bool, output: Optional[str] = None,
                             verdict: Optional[dict] = None, from_states: tuple = ("awaiting_approval",)) -> dict:
        task = self.store.task_get(task_hash, "poster", "")
        if not task:
            raise ValueError("no such posted task")
        desired = "accepted" if accept else "rejected"
        if task["state"] in {"accepted", "rejected"}:
            if task["state"] != desired:
                raise ValueError(f"task was already {task['state']}")
            self.reconcile_escrow()     # adopt/retry a ledger transition lost around a crash
            return {"task_hash": task_hash, "state": task["state"], "idempotent": True}
        if task["state"] not in from_states:
            raise ValueError(f"task is {task['state']}, not awaiting a decision")
        self._require_grant("task.accept")
        final = dict(verdict if verdict is not None else (task.get("verdict") or {}))
        rules_passed = bool(final.get("rules_passed", final.get("passed")))
        final["rules_passed"] = rules_passed
        final["passed"] = accept
        final["reasons"] = [] if accept else [str(reason or "rejected by person")[:300]]
        final["decision"] = source
        fields = {"state": desired, "verdict": json.dumps(final)}
        if output is not None:
            fields["output"] = output
        if not self.store.task_cas(task_hash, "poster", "", from_states, **fields):
            return self._apply_task_decision(task_hash, accept, reason, source=source, touch=touch)
        receipt = self._settle(task_hash, task["spec"], "release" if accept else "refund",
                               party=task["assignee"] or "", reason=reason or desired,
                               hold_id=task["hold_id"] or "")
        if receipt is not None and not receipt.get("error"):
            final["settlement"] = receipt
            self.store.task_cas(task_hash, "poster", "", (desired,), verdict=json.dumps(final))
        self._queue(task["assignee"], "task.accept" if accept else "task.reject",
                    {"task_hash": task_hash, "epoch": task["epoch"], **final}, touch=touch)
        label = "paid by silence" if source == "silence" else source
        self.log(f"task {task_hash[:12]} {'ACCEPTED' if accept else 'REJECTED'} ({label})")
        return {"task_hash": task_hash, "state": desired, "idempotent": False}

    def decide_task(self, ref: str, decision: str, reason: str = "") -> dict:
        rows = [r for r in self.store.task_find(str(ref)) if r["role"] == "poster"]
        if len(rows) != 1:
            raise ValueError("no single posted task matches that id")
        if decision not in {"accept", "reject"}:
            raise ValueError("decision must be accept or reject")
        if not isinstance(reason, str) or len(reason) > 300:
            raise ValueError("decision reason must be text of at most 300 characters")
        why = outbound_problem(reason)
        if why:
            raise ValueError(why)
        task = rows[0]
        if task["state"] == "awaiting_approval" and task["spec"]["deadline"] <= time.time():
            self._apply_task_decision(task["task_hash"], True, "", source="silence", touch=False)
            task = self.store.task_get(task["task_hash"], "poster", "")
        return self._apply_task_decision(task["task_hash"], decision == "accept", reason,
                                         source="human", touch=True)

    def _mark_worker_submit(self, task: dict) -> bool:
        if (not self.settle.open_on_assign or task["spec"]["reward"]["amount"] <= 0
                or not task.get("hold_id") or not task.get("result_hash")):
            return True
        try:
            self.settle.submit(task["task_hash"], task["result_hash"], hold_id=task["hold_id"])
        except SettlementError as exc:
            self.store.task_cas(task["task_hash"], "worker", task["peer"], ("submitted",),
                                submit_state="pending", submit_reason=str(exc)[:200])
            self.log(f"ledger submit mark for {task['task_hash'][:12]} pending: {exc}")
            return False
        self.store.task_cas(task["task_hash"], "worker", task["peer"], ("submitted",),
                            submit_state="marked", submit_reason="")
        return True

    def _retry_worker_submits(self) -> None:
        for task in self.store.tasks_in_states(("submitted",), role="worker"):
            if task.get("submit_state") != "marked":
                self._mark_worker_submit(task)

    def _claim_reopened(self, mid: str, person: str) -> None:
        """The poster refused our claim and put the task back up: make it claimable again here.
        Only for a task.claim WE sent to that poster, and only while it is still `claimed`."""
        frame = self.store.outbox_frame(mid, person) or {}
        h = (frame.get("body") or {}).get("task_hash")
        if frame.get("kind") == "task.claim" and isinstance(h, str):
            self.store.task_cas(h, "worker", person, ("claimed",), state="offered")

    def _open_assignment(self, h: str, person: str, *, touch: bool = False) -> None:
        """Open (or, idempotently, re-open) the task hold for an `assigned` task that has none yet,
        record it, and only then tell the worker. client_nonce is the task hash, so a repeat after an
        ambiguous failure returns the SAME hold. Definitive ledger refusal: put the task back to
        `posted` (no hold can exist) and raise with the REOPENED marker. Unknown outcome (timeout,
        5xx, backlog): leave it assigned/pending so the next claim replay or sweep retries it."""
        t = self.store.task_get(h, "poster", "")
        card = self.store.agent_card(person) or {}
        if not t or t["state"] != "assigned" or t["assignee"] != person or t["hold_id"]:
            return
        if card.get("ledger_keyid") != (self.store.contact(person) or {}).get("node"):
            self.store.task_cas(h, "poster", "", ("assigned",), state="posted", assignee=None,
                                epoch=0, funding_state="unfunded", funding_reason="")
            raise ValueError(REOPENED + "the contact's card ledger key is not its node key")
        try:
            receipt = self._open_escrow(h, t["spec"], authorisation=t["escrow_auth"],
                                        worker_keyid=card["ledger_keyid"])
        except EscrowRefused as e:
            if not e.definitive:
                self.store.task_cas(h, "poster", "", ("assigned",), funding_state="pending",
                                    funding_reason=str(e)[:200])
                raise ValueError(f"escrow pending: {e}")
            hold = e.hold
            if (e.hold_state == "paid" and hold.get("task_hash") == h
                    and hold.get("worker_keyid") == card["ledger_keyid"]
                    and self.store.task_cas(h, "poster", "", ("assigned",), hold_id=hold["hold_id"],
                                            funding_state="paid", funding_reason="")):
                self.log(f"adopted ledger-paid hold for {h[:12]}")
                raise ValueError(f"escrow already paid by the ledger: {e}")
            self.store.task_cas(h, "poster", "", ("assigned",), state="posted", assignee=None,
                                epoch=0, funding_state="unfunded", funding_reason="")
            raise ValueError(REOPENED + str(e))
        hold_id = receipt["hold_id"]
        # Record the hold durably BEFORE the assignment is sent or observed as funded.
        if not self.store.task_cas(h, "poster", "", ("assigned",), hold_id=hold_id,
                                   funding_state="funded", funding_reason=""):
            # the task left `assigned` (expired) while the hold was opening: refund it now
            try:
                self.settle.refund(h, "task no longer assigned", hold_id=hold_id)
            except SettlementError as e:
                self.log(f"could not refund orphan hold for {h[:12]}: {e}")
            return
        if t["spec"]["deadline"] <= time.time():
            return          # hold recorded; the next sweep expires the task and refunds it
        self._queue(person, "task.assign", {"task_hash": h, "epoch": 1, "worker": person,
                                            "hold_id": hold_id}, touch=touch)
        self.log(f"task {h[:12]} assigned to {person[:12]}")

    def _finish_pending_assignments(self) -> None:
        """Retry assignments left `assigned` with no hold (the ledger was unreachable or its answer
        was lost): the idempotent authorize either returns the hold we already opened or refuses."""
        if not self.settle.open_on_assign:
            return
        for t in self.store.tasks_in_states(("assigned",), role="poster"):
            if t["hold_id"] or t["spec"]["reward"]["amount"] <= 0 or not t["assignee"]:
                continue
            try:
                self._open_assignment(t["task_hash"], t["assignee"], touch=False)
            except ValueError as e:
                self.log(f"assignment of {t['task_hash'][:12]} still not funded: {e}")

    def reconcile_escrow(self) -> int:
        """Settle escrow left OPEN for a task already in a final state (a crash between the state
        change and the settlement). The ledger's own CAS keeps this exactly-once. Returns how many."""
        fixed = 0
        self._retry_worker_submits()
        self._rearm_approvals()
        self._finish_pending_assignments()
        for t in self.store.tasks_in_states(("accepted", "rejected", "expired"), role="poster"):
            if t["spec"]["reward"]["amount"] <= 0:
                continue
            if self.settle.open_on_assign and t["funding_state"] in {"paid", "refunded"}:
                continue
            if self.settle.open_on_assign and not t["hold_id"]:
                continue
            try:
                st = self.settle.status(t["task_hash"], hold_id=t["hold_id"] or "")
            except SettlementError as e:
                self.log(f"escrow status for {t['task_hash'][:12]} unavailable: {e}")
                continue
            desired = "released" if t["state"] == "accepted" else "refunded"
            receipt = None
            if st and st["state"] == "open":
                receipt = self._settle(
                    t["task_hash"], t["spec"], "release" if desired == "released" else "refund",
                    party=t["assignee"] or "", reason=t["state"], hold_id=t["hold_id"] or "")
                if receipt and receipt.get("error"):
                    if not self._adopt_terminal(t):
                        continue
                    receipt = None
                    fixed += 1
                    continue
            elif st and st["state"] == desired and self.settle.open_on_assign:
                self._set_funding(t["task_hash"], "poster", "",
                                  "paid" if desired == "released" else "refunded")
                event = "released" if desired == "released" else "refunded"
                receipt = {"adapter": self.settle.name, "event": event,
                           "task_hash": t["task_hash"], "hold_id": t["hold_id"],
                           "amount": st["amount"], "unit": st.get("unit", "credits")}
            else:
                if st and st["state"] in ("released", "refunded") and self._adopt_terminal(t):
                    fixed += 1
                continue
            fixed += 1
            if self.settle.open_on_assign and t["assignee"] and receipt:
                verdict = dict(t["verdict"] or {})
                verdict["settlement"] = receipt
                self.store.task_cas(t["task_hash"], "poster", "", (t["state"],),
                                    verdict=json.dumps(verdict))
                self._queue(t["assignee"], "task.accept" if t["state"] == "accepted" else "task.reject",
                            {"task_hash": t["task_hash"], "epoch": t["epoch"], **verdict}, touch=False)
            self.log(f"reconciled stranded escrow for {t['task_hash'][:12]} ({t['state']})")
        return fixed

    def _adopt_terminal(self, t: dict) -> bool:
        """The ledger refused our release/refund with 409 (the hold already reached the other terminal
        state, e.g. its own expiry sweep refunded it). Re-read it and record what the ledger says."""
        if not self.settle.open_on_assign:
            return False
        try:
            st = self.settle.status(t["task_hash"], hold_id=t["hold_id"] or "")
        except SettlementError as e:
            self.log(f"escrow status for {t['task_hash'][:12]} unavailable: {e}")
            return False
        if not st or st.get("state") not in ("released", "refunded"):
            return False
        self._set_funding(t["task_hash"], "poster", "",
                          "paid" if st["state"] == "released" else "refunded")
        self.log(f"adopted ledger state {st['state']} for {t['task_hash'][:12]} ({t['state']})")
        return True

    def sweep(self, now: Optional[float] = None) -> int:
        """Expire tasks whose deadline passed without an accepted result: refund the poster's escrow and
        tell the assignee; mark this node's own offers/claims expired. Returns how many expired."""
        now = time.time() if now is None else now
        n = 0
        self.reconcile_escrow()
        for t in self.store.tasks_in_states(("posted", "assigned", "awaiting_approval", "offered", "claimed")):
            if t["spec"]["deadline"] > now:
                continue
            if t["role"] == "poster" and t["state"] == "awaiting_approval":
                self._apply_task_decision(t["task_hash"], True, "", source="silence", touch=False)
                n += 1
            elif t["role"] == "poster" and t["state"] in ("posted", "assigned"):
                if (self.settle.open_on_assign and t["state"] == "assigned" and not t["hold_id"]
                        and t["spec"]["reward"]["amount"] > 0):
                    continue    # a hold may exist unrecorded; reconcile re-runs the idempotent authorize
                if self.store.task_cas(t["task_hash"], "poster", "", ("posted", "assigned"), state="expired",
                                       verdict=json.dumps({"passed": False, "reasons": ["deadline passed"]})):
                    receipt = self._settle(t["task_hash"], t["spec"], "refund", reason="expired",
                                           hold_id=t["hold_id"] or "")
                    if t["assignee"]:
                        body = {"task_hash": t["task_hash"], "epoch": t["epoch"],
                                "reasons": ["deadline passed"]}
                        if self.settle.open_on_assign and receipt and not receipt.get("error"):
                            body["settlement"] = receipt
                        self._queue(t["assignee"], "task.reject", body, touch=False)
                    n += 1
            elif t["role"] == "worker" and t["state"] in ("offered", "claimed", "assigned"):
                n += self.store.task_cas(t["task_hash"], "worker", t["peer"], ("offered", "claimed", "assigned"),
                                         state="expired")
        return n

    def _worker_task(self, ref: str) -> dict:
        rows = [r for r in self.store.task_find(str(ref)) if r["role"] == "worker"]
        if len(rows) != 1:
            raise ValueError("no single offered task matches that id")
        return rows[0]

    def escrow_status(self, ref: str) -> dict:
        rows = self.store.task_find(str(ref))
        if len(rows) != 1:
            raise ValueError("no single task matches that id")
        task = rows[0]
        amount = task["spec"]["reward"]["amount"]
        if amount <= 0:
            return {"task_hash": task["task_hash"], "funding": "free", "settlement": None}
        if self.settle.open_on_assign and not task["hold_id"]:
            return {"task_hash": task["task_hash"], "funding": "unfunded", "settlement": None}
        try:
            status = self.settle.status(task["task_hash"], hold_id=task["hold_id"] or "")
        except SettlementError as e:
            raise ValueError(f"escrow: {e}") from e
        funding = {"open": "funded", "released": "paid", "refunded": "refunded"}.get(
            (status or {}).get("state"), "unfunded")
        return {"task_hash": task["task_hash"], "funding": funding,
                "hold_id": task["hold_id"], "settlement": status}

    def _verify_worker_hold(self, task: dict) -> bool:
        """Pay-first: only an exact open Neuron hold makes a rewarded assignment workable."""
        if task["spec"]["reward"]["amount"] <= 0 or not self.settle.open_on_assign:
            return True
        reason = ""
        if not task["hold_id"]:
            reason = "assignment did not include a task hold"
        else:
            try:
                status = self.settle.status(task["task_hash"], hold_id=task["hold_id"])
            except SettlementError as e:
                status = None
                reason = str(e)
            if status is not None:
                expected_amount = task["spec"]["reward"]["amount"]
                if status.get("state") != "open":
                    reason = f"task hold is {status.get('ledger_state') or status.get('state')}"
                elif status.get("worker_keyid") != self.node:
                    reason = "task hold names another worker"
                elif status.get("amount") != expected_amount:
                    reason = "task hold amount does not match the reward"
                elif status.get("task_hash") != task["task_hash"]:
                    reason = "task hold is bound to another task"
                elif status.get("deadline", 0) < task["spec"]["deadline"]:
                    reason = "task hold expires before the task deadline"
        self._set_funding(task["task_hash"], "worker", task["peer"],
                          "unfunded" if reason else "funded", reason)
        return not reason

    def claim_task(self, ref: str) -> dict:
        t = self._worker_task(ref)
        if t["spec"]["deadline"] <= time.time():
            raise ValueError("that task's deadline has passed")
        self._require_grant("task.claim")
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
        if t["spec"]["deadline"] <= time.time():
            raise ValueError("that task's deadline has passed; the poster will not take a result now")
        if not self._verify_worker_hold(t):
            current = self.store.task_get(t["task_hash"], "worker", t["peer"])
            raise ValueError(f"unfunded: {current['funding_reason'] or 'the task hold is not valid'}")
        self._require_grant("task.result")
        passed, reasons = T.evaluate(t["spec"], output)       # the worker can see it would fail first
        result_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()
        funded = self.settle.open_on_assign and t["spec"]["reward"]["amount"] > 0
        if not self.store.task_cas(t["task_hash"], "worker", t["peer"], ("assigned",),
                                   state="submitted", output=output, result_hash=result_hash,
                                   submit_state="pending" if funded else ""):
            raise ValueError(f"cannot submit a task that is {t['state']} (it must be assigned to you)")
        submitted = self.store.task_get(t["task_hash"], "worker", t["peer"])
        if funded:
            self._mark_worker_submit(submitted)       # failure is durable/retried and never blocks delivery
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
            funding = "unfunded" if spec["reward"]["amount"] > 0 and self.settle.open_on_assign else ""
            if self.store.task_insert(h, "worker", c["person"], spec, "offered",
                                      funding_state=funding):
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
            # Both the winner response (assign) and loser response (reject) require nak.task.post.
            # Check before changing assignment state so a local grant problem cannot produce a nack
            # after the protocol step has already committed.
            self._require_grant("task.assign")
            paid = t["spec"]["reward"]["amount"] > 0 and self.settle.open_on_assign
            if not paid:
                if self.store.task_cas(h, "poster", "", ("posted",), state="assigned",
                                       assignee=c["person"], epoch=1):
                    self._queue(c["person"], "task.assign",
                                {"task_hash": h, "epoch": 1, "worker": c["person"]}, touch=False)
                    self.log(f"task {h[:12]} assigned to {who}")
                elif t["assignee"] != c["person"]:
                    self._queue(c["person"], "task.reject", {"task_hash": h, "epoch": 0,
                                                             "reasons": ["already assigned to someone else"]},
                                touch=False)
                return
            card = self.store.agent_card(c["person"])
            if not card or card.get("paid") is not True or not card.get("ledger_keyid"):
                raise ValueError("paid task assignment requires the contact's paid card and ledger key")
            if card["ledger_keyid"] != c["node"]:
                raise ValueError("the contact's card ledger key is not its node key")
            # Winner selection FIRST (CAS posted -> assigned, funding pending, no hold yet): a losing
            # claimant gets the clean reject and no hold is ever authorized for it.
            won = self.store.task_cas(h, "poster", "", ("posted",), state="assigned",
                                      assignee=c["person"], epoch=1,
                                      funding_state="pending", funding_reason="opening task hold")
            if not won:
                cur = self.store.task_get(h, "poster", "")
                if not (cur and cur["state"] == "assigned" and cur["assignee"] == c["person"]
                        and not cur["hold_id"]):
                    if not cur or cur["assignee"] != c["person"]:
                        self._queue(c["person"], "task.reject", {"task_hash": h, "epoch": 0,
                                                                 "reasons": ["already assigned to someone else"]},
                                    touch=False)
                    return
            self._open_assignment(h, c["person"], touch=False)
            return
        if kind == "task.assign":
            if body.get("worker") != self.person or not isinstance(body.get("epoch"), int):
                raise ValueError("assignment is not for this node's person")
            hold_id = body.get("hold_id", "")
            if hold_id and not isinstance(hold_id, str):
                raise ValueError("assignment task hold is invalid")
            if not self.store.task_cas(h, "worker", c["person"], ("claimed",), state="assigned",
                                       epoch=body["epoch"], hold_id=hold_id or None):
                raise ValueError("assignment for a task this node did not claim")
            assigned = self.store.task_get(h, "worker", c["person"])
            if not self._verify_worker_hold(assigned):
                reason = self.store.task_get(h, "worker", c["person"])["funding_reason"]
                self.log(f"task {h[:12]} assigned but unfunded: {reason}")
                return
            self.log(f"task {h[:12]} assigned to us by {who}")
            return
        if kind == "task.result":
            t = self.store.task_get(h, "poster", "")
            if not t or t["assignee"] != c["person"] or body.get("epoch") != t["epoch"]:
                raise ValueError("result for a task not assigned to you")
            if t["state"] != "assigned":
                raise ValueError("task is not awaiting a result")
            if t["spec"]["deadline"] <= time.time():
                self.sweep()                                 # expire + refund now rather than at the next tick
                raise ValueError("deadline passed; the task expired and its escrow was refunded")
            # The verdict is nak.task.post-authorised. Validate it before task state or escrow moves;
            # sender-loop signing still rechecks in case the grant expires while the reply waits.
            self._require_grant("task.accept")
            output = body.get("output")
            passed, reasons = T.evaluate(t["spec"], output)
            verdict = {"passed": passed, "reasons": reasons,
                       "output_sha256": hashlib.sha256(str(output).encode("utf-8")).hexdigest()}
            self.store.raw_log({"received": int(time.time()), "source": "nakshatra", "trust": "external",
                                "from_person": c["person"], "envelope": env, "body": body})
            paid = t["spec"]["reward"]["amount"] > 0 and self.settle.open_on_assign
            action = V.decide_policy(
                self._task_constraints(t), reward=t["spec"]["reward"]["amount"],
                stranger=self._worker_is_stranger(c), rules_passed=passed, escrowed=paid)
            if action == V.ASK:
                now = int(time.time())
                verdict["rules_passed"] = passed
                verdict["approval"] = {"worker": c["person"],
                                       "reward": t["spec"]["reward"]["amount"],
                                       "result_preview_hash": verdict["output_sha256"],
                                       "asked_at": now, "last_prompted_at": 0, "prompt_count": 0}
                if not self.store.task_cas(h, "poster", "", ("assigned",), state="awaiting_approval",
                                           output=output, verdict=json.dumps(verdict)):
                    raise ValueError("task is not awaiting a result")
                self.log(f"task {h[:12]} result from {who}: awaiting approval")
                self._arm_approval(h)
                return
            self._apply_task_decision(
                h, action == V.ACCEPT, "" if passed else "; ".join(reasons),
                source="automatic", touch=False, output=output, verdict=verdict,
                from_states=("assigned",))
            return
        if kind in ("task.accept", "task.reject"):
            t = self.store.task_get(h, "worker", c["person"])
            if not t:
                raise ValueError("no such task here")
            reasons = body.get("reasons") if isinstance(body.get("reasons"), list) else []
            v = {"passed": kind == "task.accept", "reasons": [str(r)[:300] for r in reasons][:12]}
            settlement = body.get("settlement")
            if isinstance(settlement, dict):       # the poster's receipt (verifiable on-chain later)
                v["settlement"] = {k: body["settlement"].get(k) for k in ("adapter", "event", "amount", "unit")}
            verdict = json.dumps(v)
            if body.get("epoch") == 0:
                self.store.task_cas(h, "worker", c["person"], ("claimed",), state="lost", verdict=verdict)
            elif kind == "task.reject" and v["reasons"] == ["deadline passed"]:
                self.store.task_cas(h, "worker", c["person"], ("assigned", "submitted", "expired"),
                                    state="expired", verdict=verdict,
                                    **({"funding_state": "refunded"} if isinstance(settlement, dict) else {}))
            else:
                final_state = "accepted" if kind == "task.accept" else "rejected"
                funding = ""
                if t["spec"]["reward"]["amount"] > 0 and self.settle.open_on_assign:
                    funding = ("paid" if kind == "task.accept" and isinstance(settlement, dict)
                               else "refunded" if kind == "task.reject" and isinstance(settlement, dict)
                               else "funded")
                self.store.task_cas(h, "worker", c["person"], ("submitted", "accepted", "rejected"),
                                    state=final_state, verdict=verdict,
                                    **({"funding_state": funding} if funding else {}))
            return
        raise ValueError(f"unknown task step {kind}")

    # ── read side ────────────────────────────────────────────────────────────────────────────
    def status(self) -> dict:
        with self._lock:
            online = set(self._sessions)
        grant = self._grant_status()
        warning = self._grant_warning_text(grant)
        grant["status"] = self._grant_status_text(grant)
        lapsed = grant["status"].startswith("LAPSED:")
        grant["warning"] = "" if lapsed else (f"{warning}; run: nak grant renew" if warning else "")
        grant = {k: grant[k] for k in ("agent", "caps", "expires_at", "not_before", "revoked",
                                       "grant_id", "valid", "problem", "lease", "status", "warning",
                                       "verdict_policy")
                 if k in grant}
        refused = self.store.refusal_summary()
        waiting = len(self.store.tasks_in_states(("awaiting_approval",), role="poster"))
        return {"person": self.person, "node": self.node, "custody": self.custody,
                "relay": f"{self.relay[0]}:{self.relay[1]}",
                "contacts": [dict(c, online=c["person"] in online, path=self._paths.get(c["person"], ""))
                             for c in self.store.contacts()],
                "direct_port": self.direct_port,
                "p2p": self.p2p,
                "pending_requests": len(self.store.requests()),
                "open_invites": len(self.store.open_invites()),
                "grant": grant, "awaiting_approval": waiting,
                "outbox_refused": refused["count"],
                "newest_refusal": refused["latest"]}


def _close(sock) -> None:
    try:
        sock.close()
    except OSError:
        pass


def _is_loopback_peer(sock) -> bool:
    """The sidecar's -inbound arrives from loopback; ordinary production direct dials do not."""
    import ipaddress
    try:
        ip = ipaddress.ip_address(sock.getpeername()[0].split("%", 1)[0])
        return ip.is_loopback or bool(getattr(ip, "ipv4_mapped", None) and ip.ipv4_mapped.is_loopback)
    except (OSError, ValueError, TypeError, IndexError):
        return False


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
       "agent_send", "threads", "thread", "card", "card_set", "delivered", "inbox", "outbox",
       "task_post", "tasks", "task_claim", "task_submit", "task_decide", "ledger", "escrow_status", "direct"}


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
        if op == "agent_send":
            optional = {key: req[key] for key in ("want", "expires", "budget", "refs", "harness") if key in req}
            return {"ok": True, **node.agent_send(req["to"], req["text"], intent=req.get("intent", "ask"),
                                                  thread=req.get("thread", ""),
                                                  reply_to=req.get("reply_to", ""), **optional)}
        if op == "threads":
            return {"ok": True, "threads": node.threads()}
        if op == "thread":
            thread, messages = node.thread(req["thread"], req.get("contact", ""))
            return {"ok": True, "thread": thread, "peer": messages[0]["peer"], "messages": messages}
        if op == "card":
            return {"ok": True, **node.card(req.get("contact", ""))}
        if op == "card_set":
            return {"ok": True, **node.card_set(req["name"], req["offers"], intents=req.get("intents"),
                                                models=req.get("models"), price_hint=req.get("price_hint", ""),
                                                paid=bool(req.get("paid", False)))}
        if op == "delivered":
            return {"ok": True, "delivered": node.delivered(req["mid"])}
        if op == "task_post":
            return {"ok": True, **node.post_task(req["to"], req["title"], req["instructions"], req["acceptance"],
                                                 int(req.get("reward", 0)), int(req.get("deadline_s", 3600)))}
        if op == "tasks":
            rows = node.store.tasks(req.get("role") or None, int(req.get("limit", 50)))
            for r in rows:
                r.pop("escrow_auth", None)
                for internal in ("hold_id", "funding_state", "funding_reason"):
                    if not r.get(internal):
                        r.pop(internal, None)
                c = node.store.contact(r["peer"] or (r["assignee"] or ""))
                r["peer_name"] = (c or {}).get("petname", "")
                r["name_source"] = (c or {}).get("name_src") or "self-chosen"
            return {"ok": True, "tasks": rows}
        if op == "direct":
            return {"ok": True, **node.set_direct(req["who"], bool(req.get("on")))}
        if op == "ledger":
            return {"ok": True, "adapter": node.settle.name, "journal": node.settle.journal(req.get("task") or None)}
        if op == "escrow_status":
            return {"ok": True, **node.escrow_status(req["task"])}
        if op == "task_claim":
            return {"ok": True, **node.claim_task(req["task"])}
        if op == "task_submit":
            return {"ok": True, **node.submit_task(req["task"], req["output"])}
        if op == "task_decide":
            return {"ok": True, **node.decide_task(req["task"], req["decision"], req.get("reason", ""))}
        if op == "inbox":
            msgs = node.store.inbox(int(req.get("since", 0)), int(req.get("limit", 50)))
            for m in msgs:
                m["name_source"] = (node.store.contact(m["from_person"]) or {}).get("name_src") or "self-chosen"
            return {"ok": True, "messages": msgs}
        if op == "outbox":
            return {"ok": True, "items": node.store.outbox(bool(req.get("refused")),
                                                               bool(req.get("text")),
                                                               int(req.get("limit", 50)))}
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
    os.umask(0o077)
    home = Path.home()
    ap = argparse.ArgumentParser(prog="nakd", description="Nakshatra messaging node")
    ap.add_argument("--state", type=Path, default=Path(os.environ.get("NAK_NET_DIR", home / ".nakshatra" / "net")))
    ap.add_argument("--node-key", type=Path, default=WORKER_KEY_PATH)
    ap.add_argument("--signer", type=Path, default=Path(os.environ.get(
        "NAK_SIGNER_SOCK", home / ".sthambha" / "signer" / "signer.sock")))
    ap.add_argument("--agent", default=os.environ.get("NAK_AGENT", ""))
    ap.add_argument("--relay", default=os.environ.get("NAK_RELAY", f"{DEFAULT_RELAY[0]}:{DEFAULT_RELAY[1]}"))
    ap.add_argument("--direct-port", type=int, help="direct listener port (overrides <state>/direct-port)")
    ap.add_argument("--p2p-dial", help="local sidecar UNIX socket path (overrides <state>/p2p-dial)")
    a = ap.parse_args(argv)
    if not a.agent and (a.state / "agent").exists():
        a.agent = (a.state / "agent").read_text().strip()      # set once per node: `echo prithvi > …/net/agent`
    if not a.agent:
        raise SystemExit("--agent, NAK_AGENT or <state>/agent is required: the delegated agent this node signs as")
    host, port = a.relay.rsplit(":", 1)
    stop = threading.Event()
    # The signer is a separate service and an update restarts both at once: wait for it (up to a
    # minute) instead of dying on the race and leaving the node unreachable until systemd retries.
    deadline = time.time() + 60
    while True:
        try:
            dp_file = a.state / "direct-port"                # opt-in: `nak direct-listen 51830`
            direct_port = a.direct_port if a.direct_port is not None else \
                (int(dp_file.read_text().strip()) if dp_file.exists() else None)
            p2p_file = a.state / "p2p-dial"                  # opt-in: `nak p2p on`
            p2p_dial = a.p2p_dial or (p2p_file.read_text().strip() if p2p_file.exists() else "")
            node = Node(a.state, _load_node_key(a.node_key), SignerClient(a.signer), agent=a.agent,
                        relay=(host, int(port)), log=lambda m: print(f"[nakd] {m}", flush=True),
                        direct_port=direct_port, p2p={"dial": p2p_dial} if p2p_dial else None).start()
            break
        except DataKeyError as e:
            raise SystemExit(f"nakd cannot start without the signer's data key: {e}") from e
        except (OSError, RuntimeError) as e:
            if time.time() > deadline:
                raise
            print(f"[nakd] waiting for the signer: {e}", flush=True)
            time.sleep(2)
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
