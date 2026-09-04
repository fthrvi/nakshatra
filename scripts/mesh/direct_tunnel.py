"""Skip the relay when the peer is actually reachable.

Every NAT'd pair reaches each other through the rendezvous relay today — correct, and the
only thing that works when both sides are behind restrictive NAT. But it is also a permanent
detour: **every byte goes peer→relay→peer even when a direct path exists**, including two
boxes on the same LAN routing through a VPS to talk three feet apart. That dogleg was
measured at ~129 ms.

⚠️⚠️ THE SECURITY IS UNCHANGED, AND THAT IS THE WHOLE REASON THIS IS SAFE. `secure_handshake`
takes any socket: the same pinned-Ed25519 mutual authentication, the same X25519→ChaCha20
channel, the same `session_binding` that blocks relay substitution. A direct socket and a
relayed socket are indistinguishable above the handshake. This changes WHICH PIPE, never what
protects it — so a failed direct attempt costs a fallback, never a weakened session.

⚠️⚠️ AND THE HINT IS ATTACKER-CONTROLLED. `endpoint_hint` arrives in a listing, and on a
public Nostr relay anyone may publish one. Dialing it unfiltered is an SSRF primitive: aim a
node at 169.254.169.254 and it will happily open a TCP connection to its own cloud metadata,
or sweep an internal subnet one listing at a time. Every candidate goes through
`usable_endpoints()` first — the same filter the punch path uses.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endpointfilter import usable_endpoints  # noqa: E402

#: A direct dial must be FAST or worthless: the relay is the fallback and it works. Spending
#: 30s discovering a peer is unreachable costs more than the dogleg it was trying to save.
DIAL_TIMEOUT_S = 3.0


def parse_hint(hint: str) -> List[Tuple[str, int]]:
    """`endpoint_hint` → candidate endpoints. Never raises; junk yields nothing.

    Accepts `host:port`, `[v6]:port`, and comma-separated lists of either.
    """
    out: List[Tuple[str, int]] = []
    if not isinstance(hint, str):
        return out
    for part in hint.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if part.startswith("["):                    # [2001:db8::1]:51820
                host, _, rest = part[1:].partition("]")
                port = int(rest.lstrip(":"))
            else:
                host, _, p = part.rpartition(":")
                port = int(p)
            if host:
                out.append((host, port))
        except (ValueError, TypeError):
            continue                                    # a malformed hint is not an error
    return out


def dial_direct(hint: str, *, allow_lan: bool = True,
                timeout: float = DIAL_TIMEOUT_S) -> Tuple[Optional[socket.socket], str]:
    """Try each safe candidate in order. Returns (socket, why) — socket None on failure.

    ⚠️ Order is preserved from the hint: the publisher listed its preference (LAN first,
    then public) and re-sorting here silently discards that.
    """
    cands, dropped = usable_endpoints(parse_hint(hint), allow_lan=allow_lan)
    if not cands:
        return None, f"no safe endpoint in hint ({'; '.join(dropped) or 'empty'})"
    tried = []
    for host, port in cands:
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.settimeout(None)
            return s, f"direct to {host}:{port}"
        except OSError as e:
            tried.append(f"{host}:{port} ({type(e).__name__})")
    return None, "no candidate answered: " + ", ".join(tried)


def open_pipe(hint: str, relay_connect_fn, *, allow_lan: bool = True,
              timeout: float = DIAL_TIMEOUT_S) -> Tuple[socket.socket, str, bool]:
    """Direct if we can, relay if we cannot. Returns (sock, why, was_direct).

    ⚠️ The relay is the FALLBACK AND THE FLOOR. §8.2 recon established that both peers
    behind NAT on different networks is the COMMON case, not the exception — so this is an
    optimisation on a path that must keep working when it fails. `relay_connect_fn` is
    injected rather than imported so this is testable without a relay, and so a caller can
    never accidentally get a direct-only version.
    """
    sock, why = dial_direct(hint, allow_lan=allow_lan, timeout=timeout)
    if sock is not None:
        return sock, why, True
    return relay_connect_fn(), f"relay (direct unavailable: {why})", False
