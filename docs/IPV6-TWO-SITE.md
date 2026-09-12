# The two-site IPv6 run — what was measured, and why it stopped

**2026-09-04. T4.3. Partially executed: one site is real, the second does not exist yet.**

## What the plan assumed

> ⭐ *Possibly the cheapest win on this plan.* The home connection has stable public IPv6 and
> no CGNAT. If both ends of a pair have IPv6, NAT traversal is unnecessary: connect directly.

The first half is confirmed. The second half has no second end.

## Site 1 — the hub. Confirmed.

    2601:8c0:681:f790::388c                    /128, stable, delegated
    2601:8c0:681:f790:4dd4:c670:418c:7404      /64, temporary (privacy)
    2601:8c0:681:f790:903f:e789:1f8d:9296      /64, dynamic mngtmpaddr

    ping6 2606:4700:4700::1111 → 2/2 received, rtt avg 22.6 ms

Globally routable, no CGNAT, working v6 transit. Exactly as the audit said.

## Site 2 — does not exist yet

| candidate | v6? | why not |
|---|---|---|
| VPS `45.63.109.137` (Vultr) | ✗ | `buzz.prithviloka.net` resolves only to `::ffff:45.63.109.137` — an IPv4-mapped address, no AAAA. Vultr offers v6; it has never been enabled or published. |
| elitedesk `10.42.0.5` | ✗ | same LAN as the hub. Two boxes behind one router is not two sites — it measures a switch, not the internet. |
| CARC | ✗ | compute nodes have no outbound internet at all (verified on easley053). |
| MacBook | — | roams; when it is at home it is site 1 again. |

⚠️ **The honest conclusion: this test cannot be completed today**, and running it between the
hub and elitedesk would produce a number that looks like a result and measures nothing. A
0.3 ms LAN RTT would be reported as "IPv6 direct path confirmed" and the first real
cross-site run would contradict it.

## What the attempt DID find — a real bug

Pointing the selector at this machine's actual addresses, it chose the **privacy** address
over the stable one.

The stability heuristic counted `str(addr).count("0")` — the literal **character** — rather
than zero **groups**. The compressed `2601:8c0:681:f790::388c` contains two `0` characters;
the rotating `...:4dd4:c670:418c:7404` contains four. So the address that changes daily
outscored the one that does not, and the heuristic ran exactly backwards.

⚠️ **Twelve synthetic tests missed it, because not one used a compressed address.** Real data
found it on the first call. Fixed to count zero groups in the exploded interface identifier,
and the three real addresses are now a regression fixture — including a permutation test, so
the answer cannot depend on the order the kernel happened to list them in.

Cost of the bug had it shipped: a node re-announcing itself every time its privacy address
rotated, and peers holding an address that had already expired. Not fatal, and invisible —
which is worse.

## To actually complete T4.3

One of:

1. **Enable IPv6 on the Vultr VPS** and publish an AAAA. Cheapest by far — the VPS is already
   the relay's public face, and it is genuinely a second site on a different network.
2. **The UNM lab site**, if it has v6 transit.
3. **Any peer's machine** once a stranger joins — which is the real answer, and needs the
   join path this branch just built.

Then the measurement is: `select_global_ipv6` on both ends, connect directly on the chosen
addresses, and compare RTT and throughput against the relay path. The reference to beat is the
measured relay number: **170.9 ms → 28.6 ms, 3.77× throughput** on a real two-site WAN via
hole-punching. IPv6 should match or beat that *without* punching, because there is nothing to
traverse.

⚠️ And the comparison must be same-pair, same-hour. A v6 number from today against a v4 number
from June measures the internet's mood, not the protocol.
