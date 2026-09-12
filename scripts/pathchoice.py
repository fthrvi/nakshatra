"""pathchoice.py — relay, direct, or IPv6, and honest when there is no path.

Going direct is worth real money: 170.9 ms -> 28.6 ms and 3.77x throughput, measured on a
two-site WAN. But the decision is asymmetric and that asymmetry drives every rule here —
**a wrong "direct" stalls a run; a wrong "relay" costs latency.** So every uncertainty
resolves toward the slower answer.

⚠️ IPv6 OUTRANKS NAT ENTIRELY. If both ends hold a globally routable v6 there is no
traversal problem to solve, so the NAT type is irrelevant — a symmetric NAT on one side does
not stop two v6 hosts talking. That ordering is easy to get backwards, because "symmetric
NAT" reads as a blocker in every other context.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ipv6 import select_global_ipv6

#: NAT types through which hole-punching is plausible. Everything else — including
#: "symmetric", "unknown", and anything unrecognised — is not.
PUNCHABLE = frozenset({"none", "full-cone", "restricted", "port-restricted"})


def _side(s: Any) -> Dict[str, Any]:
    return s if isinstance(s, dict) else {}


def _routable_v6(s: Dict[str, Any]) -> str | None:
    v = s.get("ipv6")
    return select_global_ipv6([v]) if isinstance(v, str) and v else None


def _relay(s: Dict[str, Any]) -> str:
    r = s.get("relay")
    return r if isinstance(r, str) and r else ""


def choose_path(local: Any, remote: Any) -> Tuple[str, str]:
    """Return (path, why) with path in {"ipv6", "direct", "relay", "none"}."""
    lo, re = _side(local), _side(remote)

    lv6, rv6 = _routable_v6(lo), _routable_v6(re)
    if lv6 and rv6:
        return "ipv6", f"both sides globally routable over IPv6 ({lv6} <-> {rv6}) — no traversal needed"

    # ⚠️ A missing or unrecognised `nat` is "unknown", and unknown is NEVER punchable. A side
    # whose NAT was never measured is not a side whose NAT is open; treating absence as
    # permissive makes "we forgot to probe" indistinguishable from "we probed and it is open".
    lnat = lo.get("nat") if isinstance(lo.get("nat"), str) else "unknown"
    rnat = re.get("nat") if isinstance(re.get("nat"), str) else "unknown"
    if lnat in PUNCHABLE and rnat in PUNCHABLE:
        return "direct", f"hole-punchable pair ({lnat} <-> {rnat})"

    blocker = lnat if lnat not in PUNCHABLE else rnat
    if _relay(lo) or _relay(re):
        return "relay", f"{blocker!r} NAT is not punchable — falling back to the relay"

    # ⚠️ Say there is no path, rather than returning a default that fails later somewhere
    # less informative. "relay" with no relay configured is a lie that surfaces as a timeout.
    return "none", (f"{blocker!r} NAT is not punchable and neither side has a relay — "
                    "no path between these peers")
