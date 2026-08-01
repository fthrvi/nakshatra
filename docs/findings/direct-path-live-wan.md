# direct-path, live over a real WAN — 170.9ms → 28.6ms (2026-08-01)

The one item `direct-path.md` left open: *"Real WAN RTT, jitter, loss, MTU —
loopback RTT is [not] a live-network question."* ijru moving to another site
gave us two genuinely NAT'd peers on different residential networks, so this is
that test.

## The dogleg, measured

| leg | RTT |
|---|---|
| hub → VPS relay | 63.8 ms |
| ijru → VPS relay | 116.7 ms |
| **sum (predicted relay path)** | **180.5 ms** |
| **hub → ijru over the overlay (actual)** | **180.9 ms** |

The prediction lands within **0.4 ms**, which settles what the overlay was doing:
every packet between two boxes ~25 ms apart was doglegging through a VPS in
another state. Neither box knew — WireGuard was working perfectly, just very far
around.

## The punch

Simultaneous-open UDP, HMAC-authenticated, `scripts/mesh/direct_path.py`
unmodified. **5/5 attempts succeeded**, first one in 40 ms of wall clock. Both
NATs turned out to preserve the source port (endpoint-independent mapping), so
the candidate list never needed more than the peer's public address.

## Direct vs relayed, 20 packets each

| path | min | **avg** | max | jitter (mdev) | loss |
|---|---|---|---|---|---|
| relayed (today's overlay) | 124.6 | **170.9 ms** | 244.0 | 25.0 ms | 0% |
| **direct (punched)** | 20.0 | **28.6 ms** | 40.9 | **5.9 ms** | 0% |

- **6.0× lower RTT**, and **4.2× less jitter** — the second number matters more
  than it looks for a chain that pays a round trip per decode step, because
  jitter is what turns a p50 into a stall.
- **MTU 1500 clean** on the direct path (1472-byte payload with DF set, no
  fragmentation) — so there is no hidden path-MTU tax to pay back.

The research note that motivated this module estimated **~129 ms** of eliminable
dogleg. Measured: **142 ms**.

## Why this matters beyond networking

The split 30B chain currently serves **4.85 tok/s** across the relayed link (it
was 51.16 tok/s when both stages shared a LAN). The chain pays one round trip per
decode step, so RTT is very nearly the whole cost. Moving that path from 170.9 ms
to 28.6 ms is a ~6× reduction in the dominant term.

## NOT done here, deliberately — needs an operator call

Repointing the live WireGuard peer at the direct endpoint is the obvious payoff,
and it is **not** a change to make unattended: ijru is now at another site, so a
WG config that fails to come back strands a box nobody can walk over to. The
punch proves the path exists and holds; adopting it for the tunnel is a separate,
supervised change (endpoint + PersistentKeepalive, with a timed revert).
