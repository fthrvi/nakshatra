"""Pick a discovery relay, and say precisely why when there is none.

⚠️ The `why` matters as much as the choice. An operator whose only relay is failing must
learn that it EXISTS and is broken — not that they configured nothing. Those are different
problems with different fixes, and a single "no relay available" collapses them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

MAX_FAILS = 3
MAX_AGE_S = 86400


def select_relay(candidates: Any, *, now: float) -> Tuple[str, str]:
    if not isinstance(candidates, list) or not candidates:
        return "", "no relays configured"

    # ⚠️ Reasons are counted separately and reported in a FIXED precedence, because a
    # candidate can be rejected for several at once. Without an explicit order the message
    # depends on iteration order, which is how two runs disagree about the same list.
    counts = {"scheme": 0, "fails": 0, "stale": 0}
    usable: List[Dict[str, Any]] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        url = c.get("url")
        if not isinstance(url, str) or not url.startswith(("wss://", "ws://")):
            counts["scheme"] += 1
            continue
        fails = c.get("fails")
        if isinstance(fails, int) and fails >= MAX_FAILS:
            counts["fails"] += 1
            continue
        last_ok = c.get("last_ok")
        if not isinstance(last_ok, (int, float)) or isinstance(last_ok, bool) \
                or (now - last_ok) > MAX_AGE_S:
            counts["stale"] += 1
            continue
        usable.append(c)

    if not usable:
        # Precedence: a relay that is failing is more actionable than one that is merely
        # stale, and both are more actionable than a typo'd scheme.
        if counts["fails"]:
            return "", (f"{counts['fails']} relay(s) skipped for repeated failure "
                        f"(>= {MAX_FAILS}) — they exist and are broken, not absent")
        if counts["stale"]:
            return "", f"{counts['stale']} relay(s) have a stale or missing last_ok"
        if counts["scheme"]:
            return "", f"{counts['scheme']} relay(s) are not ws:// or wss://"
        return "", "no usable relays"

    def key(c: Dict[str, Any]):
        rtt = c.get("rtt_ms")
        num = isinstance(rtt, (int, float)) and not isinstance(rtt, bool)
        # ⚠️ wss before ws at equal rtt: a plaintext relay lets anyone on the path see which
        # nodes are looking for whom, which is a map of the network.
        return (0 if num else 1, rtt if num else 0.0,
                0 if c["url"].startswith("wss://") else 1, c["url"])

    best = min(usable, key=key)
    return best["url"], (f"lowest rtt among {len(usable)} usable "
                         f"(rtt={best.get('rtt_ms')}, fails={best.get('fails', 0)})")
