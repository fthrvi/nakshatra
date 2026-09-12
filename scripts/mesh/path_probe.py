"""What kind of path do we have to this peer? — measurement, not transport.

⚠️⚠️ THIS DOES NOT REPLACE THE TUNNEL, AND THAT IS DELIBERATE. `maybe_direct` returns a
`PathResult` — an address that answered, plus an RTT. It does NOT return a connected stream.
A punched NAT mapping is UDP; meshd's tunnel is `relay_connect` → TCP → `secure_handshake` →
`MuxTunnel`, and MuxTunnel assumes TCP semantics (ordering, no loss). Handing it a punched
UDP endpoint would produce a tunnel that works on a quiet LAN and corrupts under real loss —
the worst failure shape there is, because it passes every test you would think to write.

Using the punch for what it actually gives — a VERIFIED REACHABLE ADDRESS AND AN RTT — is
useful today and honest: it tells the placement layer which peers are close, and it tells
`pathchoice` whether a direct path exists at all. Making it carry the tunnel needs a
reliability layer over UDP, which is a separate piece of work with its own risks.

Measured, on a real two-site WAN: relay 170.9 ms, direct 28.6 ms, 3.77x throughput. That is
what is on the table — but only once something can safely carry bytes over it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from endpointfilter import usable_endpoints
from natclass import classify_nat
from pathchoice import choose_path
from stunshape import shape_observations


def assess_peer(local: Dict[str, Any], remote: Dict[str, Any],
                stun_replies: Sequence[Any] = (),
                peer_endpoints: Sequence[Any] = (),
                *, allow_lan: bool = True,
                punch=None, punch_args: Optional[dict] = None) -> Dict[str, Any]:
    """Decide what path we have, measuring only if a punch function is supplied.

    `punch` is injected rather than imported so this stays testable without a socket —
    the same decide/act split the join phases use. Pass `direct_path.maybe_direct` in
    production; pass nothing (or a stub) in a test and it reports what it knows without
    touching the network.
    """
    notes: List[str] = []

    # 1. What do our own STUN observations say about our NAT?
    obs, stun_problems = shape_observations(list(stun_replies))
    notes.extend(stun_problems)
    nat, nat_why = classify_nat(obs)
    notes.append(f"local nat: {nat} ({nat_why})")
    local = {**local, "nat": local.get("nat") or nat}

    # 2. Which of the peer's advertised endpoints are even worth trying?
    endpoints, ep_problems = usable_endpoints(list(peer_endpoints), allow_lan=allow_lan)
    notes.extend(ep_problems)

    # 3. What does the decision layer say, before spending a single packet?
    path, why = choose_path(local, remote)
    result: Dict[str, Any] = {"path": path, "why": why, "nat": nat,
                              "endpoints": endpoints, "notes": notes,
                              "direct_confirmed": False, "rtt_ms": None}

    # ⚠️ Only measure when the decision says direct is plausible AND there is somewhere to
    # send. Punching at a symmetric NAT is packets into a wall, and doing it anyway would
    # make the measurement lie about why it failed.
    if path != "direct" or not endpoints or punch is None:
        return result

    try:
        pr = punch(peer_endpoints=endpoints, **(punch_args or {}))
    except Exception as e:                                   # noqa: BLE001
        # ⚠️ A failed punch is an observation, not an error: the answer is "no direct path",
        # which the relay already covers.
        result["notes"].append(f"punch failed: {type(e).__name__}: {e}")
        return result

    result["direct_confirmed"] = bool(getattr(pr, "direct", False))
    result["rtt_ms"] = getattr(pr, "rtt_ms", None)
    if not result["direct_confirmed"]:
        # The decision said direct was plausible and the wire disagreed. The wire wins.
        result["path"] = "relay"
        result["why"] = f"punch did not confirm ({getattr(pr, 'reason', 'no reason')})"
        result["notes"].append("decision said direct, measurement said no — measurement wins")
    return result
