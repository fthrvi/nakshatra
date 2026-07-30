"""direct_path.py — rendezvous-ASSISTED UDP hole-punching (Mesh-LLM adoption #2).

Today every NAT'd peer pair reaches each other through the TCP rendezvous relay
(`transport/relay.py`) — a stable host both sides dial *out* to, which then pipes
raw bytes between them. That's correct and it's the only thing that works when
both peers are behind restrictive NATs on different networks (v1.1 §8.2's
connectivity recon). But it's also a permanent detour: every byte does
peer→relay→peer instead of peer→peer, even in the common case where a direct UDP
path would actually work (full-cone/moderate NATs, or — the easy case — two homes
on the *same* LAN). Research measured ~129ms of eliminable "dogleg" latency on
that detour (Mesh-LLM's iroh-QUIC gets this via its own hole-punch; see
`docs/vs-mesh-llm.md`).

This module is the *punch*, implemented natively (stdlib only, no iroh/QUIC dep)
to keep Nakshatra's closed/signed posture (`docs/THREAT_MODEL.md`) rather than
importing a new transport dependency: **simultaneous-open UDP**, HMAC-authenticated
with the pair's shared token so a bystander who guesses/observes the rendezvous
can't forge a HELLO and hijack the path (this mesh is invite-gated — the punch
handshake keeps that spirit, the same way `mesh/pairing.py`'s rendezvous id is
public-but-unforgeable because the crypto above it is what actually authenticates).

SCOPE (important): this module does NOT talk to the rendezvous relay and does NOT
learn peer endpoints itself. It takes `peer_endpoints` — the peer's candidate
(host, port) pairs — as an INPUT, however the caller learned them (a future relay
protocol extension; see `maybe_direct()` below and docs/findings/direct-path.md
§"meshd call-site plan" for the exact wiring this module is waiting for). That
keeps this module pure and unit-testable on loopback sockets with zero real
network, per the same discipline as `fabric/topology_order.py` / `edge_health.py`.

Protocol (one UDP socket per side, both already bound):

    A                                   B
    |-- HELLO(nonce_a) ---------------->|   (burst, sent to every candidate,
    |<-- HELLO(nonce_b) -----------------|    LAN-private ones first — §order_candidates)
    |-- HELLO_ACK(nonce_a2, echo=nonce_b)|
    |<-- HELLO_ACK(nonce_b2, echo=nonce_a)|
    |-- ACK(echo=nonce_b2) ------------->|
    |<-- ACK(echo=nonce_a2) --------------|

Both sides run the SAME `punch()` state machine concurrently (that's the
"simultaneous" part — nobody has to be told who dials first). Whichever leg of
the 3-way completes first for a given side is enough for THAT side to declare the
path live and start using it — sending the closing ACK is what lets the far side
(if it was the one that first saw our HELLO) close its own leg too. Every packet
is authenticated: a forged HELLO/HELLO_ACK/ACK with the wrong token is silently
dropped (`_unpack` returns None), so an attacker sniffing the rendezvous can watch
UDP flow but cannot inject itself into the path.

Fallback guarantee: `punch()` NEVER raises, NEVER blocks past `timeout`, and NEVER
touches the relay. On any failure (peer never answers, NAT is symmetric and the
observed endpoints aren't dialable, bad auth) it returns
`PathResult(direct=False, reason=...)` and the caller keeps using
`transport/relay.py` exactly as it does today — this module adds a path, it never
removes one.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import socket
import struct
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

Endpoint = Tuple[str, int]

MAGIC = b"NKSD1"        # Nakshatra Direct-path v1 (companion to relay.py's "NKSR1")
NONCE_LEN = 8
_MAC_LEN = hashlib.sha256().digest_size   # 32
_NO_ECHO = b"\x00" * NONCE_LEN

_T_HELLO = 0x01
_T_HELLO_ACK = 0x02
_T_ACK = 0x03
_T_PING = 0x04
_T_PONG = 0x05
_TYPE_NAMES = {_T_HELLO: "HELLO", _T_HELLO_ACK: "HELLO_ACK", _T_ACK: "ACK",
               _T_PING: "PING", _T_PONG: "PONG"}


@dataclass(frozen=True)
class PathResult:
    """Outcome of a punch attempt. direct=False ⇒ caller falls back to the relay."""
    direct: bool
    endpoint: Optional[Endpoint] = None   # the address that actually confirmed (may
                                           # differ from every candidate — NAT rewrites
                                           # source ports; we trust the observed sender)
    rtt_ms: Optional[float] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class _Msg:
    mtype: int
    sender_id: str
    nonce: bytes          # this message's own nonce (for the recipient to echo back)
    echo_nonce: bytes     # the nonce being acknowledged, or _NO_ECHO


def _norm_token(token: Union[bytes, str]) -> bytes:
    if isinstance(token, bytes):
        return token
    if isinstance(token, str):
        return token.encode("utf-8")
    raise TypeError(f"token must be bytes or str, got {type(token).__name__}")


def _pack(msg: _Msg, token: bytes) -> bytes:
    id_b = msg.sender_id.encode("utf-8")
    if not (0 < len(id_b) <= 255):
        raise ValueError("sender_id must be 1..255 utf-8 bytes")
    if len(msg.nonce) != NONCE_LEN or len(msg.echo_nonce) != NONCE_LEN:
        raise ValueError(f"nonce/echo_nonce must be {NONCE_LEN} bytes")
    body = MAGIC + bytes([msg.mtype]) + struct.pack(">B", len(id_b)) + id_b + msg.nonce + msg.echo_nonce
    mac = hmac.new(token, body, hashlib.sha256).digest()
    return body + mac


def _unpack(data: bytes, token: bytes) -> Optional[_Msg]:
    """Parse + authenticate. Returns None on ANY malformed/forged input — the
    caller treats that identically to "no packet" (silently ignored), which is
    what makes a forged HELLO unable to hijack a path: it just never confirms."""
    hdr_len = len(MAGIC) + 1 + 1
    if len(data) < hdr_len or data[:len(MAGIC)] != MAGIC:
        return None
    mtype = data[len(MAGIC)]
    id_len = data[len(MAGIC) + 1]
    fixed_len = hdr_len + id_len + NONCE_LEN + NONCE_LEN + _MAC_LEN
    if len(data) != fixed_len or mtype not in _TYPE_NAMES:
        return None
    body, mac = data[:-_MAC_LEN], data[-_MAC_LEN:]
    expected = hmac.new(token, body, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        return None
    off = hdr_len
    try:
        sender_id = data[off:off + id_len].decode("utf-8")
    except UnicodeDecodeError:
        return None
    off += id_len
    nonce = data[off:off + NONCE_LEN]
    off += NONCE_LEN
    echo_nonce = data[off:off + NONCE_LEN]
    return _Msg(mtype=mtype, sender_id=sender_id, nonce=nonce, echo_nonce=echo_nonce)


def _send(sock: socket.socket, addr: Endpoint, msg: _Msg, token: bytes) -> None:
    try:
        sock.sendto(_pack(msg, token), addr)
    except OSError:
        pass  # best-effort burst; a dropped send just means one fewer shot this round


# ── candidate ordering ───────────────────────────────────────────────────

def _is_lan_private(host: str) -> bool:
    """True for RFC1918/loopback/link-local — anything ipaddress calls private.
    Unparseable hosts (a hostname, not an IP literal) are treated as NOT known-
    private so they sort after the addresses we actually know are local."""
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


def order_candidates(endpoints: Sequence[Endpoint]) -> List[Endpoint]:
    """LAN-private candidates first, public after — stable within each group.

    Two peers on the same home LAN should punch straight through without ever
    touching a public/hairpin path. We don't stage this with an artificial delay
    (that would slow down the common same-LAN case for every OTHER pair too);
    `punch()` sends this ordered list every burst, and because a LAN round-trip
    is physically far faster than a public/relay-observed one, the tie resolves
    itself — whichever HELLO_ACK comes back first wins, and that's the LAN one.
    """
    private = [e for e in endpoints if _is_lan_private(e[0])]
    public = [e for e in endpoints if not _is_lan_private(e[0])]
    return private + public


# ── the punch ─────────────────────────────────────────────────────────────

def punch(sock: socket.socket, local_id: str, peer_id: str,
          peer_endpoints: Sequence[Endpoint], token: Union[bytes, str],
          timeout: float, *, retry_interval: float = 0.05) -> PathResult:
    """Simultaneous-open UDP hole-punch. Blocks up to `timeout` seconds.

    `sock` must already be bound (the caller owns its lifetime/port — this
    function never binds or closes it, so it can be reused for probe_rtt after).
    Both peers are expected to call this concurrently with each other's
    candidate endpoints; either side may complete first (see module docstring).

    Returns PathResult(direct=True, endpoint=<observed source addr>, rtt_ms=...)
    the moment ANY leg of the 3-way confirms, or PathResult(direct=False,
    reason=...) on timeout / no candidates. Never raises on network errors —
    those degrade to "no packet" and eventually a timeout, exactly like a peer
    that's silent.
    """
    token_b = _norm_token(token)
    candidates = order_candidates(peer_endpoints)
    if not candidates:
        return PathResult(direct=False, reason="no candidate endpoints")

    deadline = time.monotonic() + timeout
    # outstanding HELLOs *we* sent: candidate -> (send_time, nonce we sent)
    hello_out: Dict[Endpoint, Tuple[float, bytes]] = {}
    # HELLO_ACKs *we* sent in response to a peer HELLO: source addr -> (send_time, nonce we sent)
    hello_ack_out: Dict[Endpoint, Tuple[float, bytes]] = {}
    last_burst = 0.0

    while True:
        now = time.monotonic()
        if now >= deadline:
            return PathResult(direct=False,
                              reason=f"timeout after {timeout:.2f}s: no confirmed "
                                     f"path to {peer_id} ({len(candidates)} candidate(s) tried)")
        if now - last_burst >= retry_interval:
            for ep in candidates:                      # LAN-first order, see order_candidates
                nonce = os.urandom(NONCE_LEN)
                hello_out[ep] = (now, nonce)
                _send(sock, ep, _Msg(_T_HELLO, local_id, nonce, _NO_ECHO), token_b)
            last_burst = now

        remaining = max(0.0, min(retry_interval, deadline - time.monotonic()))
        sock.settimeout(remaining if remaining > 0 else 0.001)
        try:
            data, addr = sock.recvfrom(2048)
        except (socket.timeout, BlockingIOError):
            continue
        except OSError:
            continue
        msg = _unpack(data, token_b)
        if msg is None or msg.sender_id != peer_id:
            continue   # malformed / bad HMAC / not our expected peer — dropped, not fatal

        if msg.mtype == _T_HELLO:
            # Someone (possibly the peer we're punching to, arriving from a source
            # port the NAT rewrote — we trust the observed `addr`, not the candidate
            # list) says hello. Answer, echoing their nonce, and remember we did so
            # this side confirms once THEY send the closing ACK.
            my_nonce = os.urandom(NONCE_LEN)
            hello_ack_out[addr] = (time.monotonic(), my_nonce)
            _send(sock, addr, _Msg(_T_HELLO_ACK, local_id, my_nonce, msg.nonce), token_b)

        elif msg.mtype == _T_HELLO_ACK:
            pending = hello_out.get(addr)
            if pending is None or msg.echo_nonce != pending[1]:
                continue   # acking a HELLO we never sent (or an old one) — ignore
            send_time, _ = pending
            rtt_ms = (time.monotonic() - send_time) * 1000.0
            _send(sock, addr, _Msg(_T_ACK, local_id, os.urandom(NONCE_LEN), msg.nonce), token_b)
            return PathResult(direct=True, endpoint=addr, rtt_ms=rtt_ms)

        elif msg.mtype == _T_ACK:
            pending = hello_ack_out.get(addr)
            if pending is None or msg.echo_nonce != pending[1]:
                continue   # acking a HELLO_ACK we never sent — ignore
            send_time, _ = pending
            rtt_ms = (time.monotonic() - send_time) * 1000.0
            return PathResult(direct=True, endpoint=addr, rtt_ms=rtt_ms)

        # PING/PONG are out of scope for the handshake loop — probe_rtt handles those.


# ── RTT probe on an already-established path ────────────────────────────

def probe_rtt(sock: socket.socket, endpoint: Endpoint, token: Union[bytes, str],
              local_id: str = "probe", timeout: float = 1.0, retries: int = 3) -> Optional[float]:
    """Round-trip time to `endpoint` over an already-punched (or any) UDP path.
    The peer must be answering PINGs — see `answer_ping()`. Returns ms, or None
    if no authenticated PONG arrived within `timeout` after `retries` attempts."""
    token_b = _norm_token(token)
    for _ in range(max(1, retries)):
        nonce = os.urandom(NONCE_LEN)
        t0 = time.monotonic()
        _send(sock, endpoint, _Msg(_T_PING, local_id, nonce, _NO_ECHO), token_b)
        deadline = t0 + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data, addr = sock.recvfrom(2048)
            except (socket.timeout, OSError):
                break
            msg = _unpack(data, token_b)
            if msg is None or msg.mtype != _T_PONG or msg.echo_nonce != nonce or addr != endpoint:
                continue
            return (time.monotonic() - t0) * 1000.0
    return None


def answer_ping(sock: socket.socket, local_id: str, token: Union[bytes, str],
                timeout: float = 2.0) -> bool:
    """Listen for one authenticated PING and answer with PONG. Meant to run in a
    loop/thread on a live direct path (the meshd integration would fold this into
    the per-tunnel keepalive alongside the existing mux health check). Returns
    True if a PING was answered, False on timeout."""
    token_b = _norm_token(token)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        sock.settimeout(remaining)
        try:
            data, addr = sock.recvfrom(2048)
        except (socket.timeout, OSError):
            return False
        msg = _unpack(data, token_b)
        if msg is None or msg.mtype != _T_PING:
            continue
        _send(sock, addr, _Msg(_T_PONG, local_id, os.urandom(NONCE_LEN), msg.nonce), token_b)
        return True


# ── integration seam (NOT wired into meshd in this branch) ─────────────────

def maybe_direct(sock: socket.socket, local_id: str, peer_id: str,
                 peer_endpoints: Sequence[Endpoint], token: Union[bytes, str],
                 *, timeout: float = 3.0) -> PathResult:
    """The exact call meshd COULD make once it has observed peer endpoints to try
    (a rendezvous protocol extension — the relay would need to hand back each
    side's (addr, port) as seen from the outside, alongside today's byte-pipe
    pairing; see docs/findings/direct-path.md for the wire-extension sketch and
    the precise call site in `MeshNode._ensure_tunnel`).

    Env-gated, default OFF (`NKS_DIRECT_PATH=1` opts in) so importing/calling this
    from meshd today changes NOTHING until the flag flips — with it off this is a
    true no-op: it returns immediately, without sending a single packet, before
    even opening `punch()`'s loop. NOT called from meshd.py in this branch (by
    design — the branch ships the mechanism; the wiring is a reviewed follow-up)."""
    if os.environ.get("NKS_DIRECT_PATH", "").strip().lower() not in ("1", "true", "yes"):
        return PathResult(direct=False, reason="NKS_DIRECT_PATH disabled (default OFF)")
    if not peer_endpoints:
        return PathResult(direct=False, reason="no observed peer endpoints")
    return punch(sock, local_id, peer_id, peer_endpoints, token, timeout=timeout)
