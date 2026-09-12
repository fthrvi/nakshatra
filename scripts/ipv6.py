"""IPv6 address selection for NAT traversal.

This module provides utilities to select a globally routable IPv6 address
from a list of candidates, preferring stable addresses over temporary/privacy ones.
"""

import ipaddress
from typing import Optional


def select_global_ipv6(addrs: list[str]) -> Optional[str]:
    """Select the best globally-routable IPv6 address from a list.

    Returns the best globally-routable IPv6 address from the given list of
    candidate addresses, or None if no suitable address is found.

    Preference order:
    1. Globally routable (not link-local, loopback, ULA, IPv4-mapped, multicast, etc.)
    2. Among survivors, prefer addresses that appear more stable:
       - EUI-64 derived addresses (containing 'ff:fe' in the interface identifier)
       - Addresses with three or more zero groups
       - (Note: This is a heuristic and can be wrong; a privacy address still works,
         it just changes daily, so picking one costs a re-announce rather than a failure.)

    Ties break by sorted string order for determinism.

    Args:
        addrs: List of IPv6 address strings to evaluate.

    Returns:
        The best globally-routable IPv6 address as a string, or None if none found.
    """
    if not isinstance(addrs, list):
        return None

    candidates = []
    for addr_str in addrs:
        if not isinstance(addr_str, str):
            continue

        try:
            addr = ipaddress.IPv6Address(addr_str)
        except (ipaddress.AddressValueError, ValueError):
            continue

        # Skip non-globally-routable addresses
        if addr.is_link_local:
            continue
        if addr.is_loopback:
            continue
        if addr.is_unspecified:
            continue
        # Check for ULA (fc00::/7) using network membership
        if addr in ipaddress.IPv6Network('fc00::/7'):
            continue
        # Check for IPv4-mapped addresses
        if str(addr).startswith('::ffff:'):
            continue
        if addr.is_multicast:
            continue

        # This is a globally routable address
        candidates.append(addr)

    if not candidates:
        return None

    # Score candidates: higher score = more preferred
    def score(addr: ipaddress.IPv6Address) -> tuple[int, str]:
        # Primary score: stability heuristic
        score_val = 0
        # Check for EUI-64 pattern (ff:fe in the interface identifier)
        # Interface identifier is the last 64 bits (last 4 groups)
        # ⚠️⚠️ WORK ON THE EXPLODED FORM, AND COUNT GROUPS — NOT CHARACTERS.
        # This counted `str(addr).count("0")`, the literal character, which is not the same
        # thing at all: the compressed `2601:8c0:681:f790::388c` contains two '0' characters,
        # while the PRIVACY address `2601:8c0:681:f790:4dd4:c670:418c:7404` contains four —
        # so the rotating address outscored the stable one and the heuristic ran backwards.
        # Caught by running it against this machine's real addresses; twelve synthetic tests
        # had not, because none of them used a compressed form.
        groups = addr.exploded.split(":")          # always 8 groups, 4 hex digits each
        iid = groups[4:]                           # the interface identifier: last 64 bits
        if "fffe" in "".join(iid):
            score_val += 2                         # EUI-64 derived — stable by construction
        if sum(1 for g in iid if g == "0000") >= 3:
            score_val += 1                         # a low, mostly-zero IID: a delegated/static
                                                   # address, not one SLAAC randomised

        # Secondary: string representation for deterministic tie-breaking
        return (-score_val, str(addr))  # Negative score for ascending sort

    candidates.sort(key=score)
    return str(candidates[0])