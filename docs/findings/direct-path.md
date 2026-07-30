# direct-path — rendezvous-assisted UDP hole-punching (Mesh-LLM adoption #2)

**Status:** module + tests DONE (`scripts/mesh/direct_path.py`, `tests/test_direct_path.py`).
Integration seam (`maybe_direct`) exists and is env-gated OFF by default; **meshd.py is
NOT wired in this branch** — see §4 for the exact call site.
**Branch:** `discovery/direct-path` (BRANCHES.md row, infra lane ride, agent B).

## 1. The problem this closes

Every NAT'd Nakshatra peer pair reaches each other today through
`transport/relay.py`: a TCP rendezvous relay that pairs two outbound
connections by a shared `rendezvous_id` and pipes bytes between them
(`RendezvousRelay._handle`, relay.py:154-167). That's *correct* — it's the
only thing that works when both peers are behind restrictive NAT on
different networks, and the v1.1 §8.2 connectivity recon found that's the
common case. But it's also a permanent detour: **every byte goes
peer→relay→peer** even when a direct path would work — including the trivial
case of two boxes on the *same* LAN going through a VPS to talk to each
other 3 feet away. The research note this branch responds to measured
**~129ms of eliminable "dogleg" latency** on that detour.

Mesh-LLM (the independent occupant surveyed in `docs/vs-mesh-llm.md`) gets a
direct path via iroh-QUIC's built-in hole-punching. Rather than take the iroh
dependency (this repo's posture is "closed, signed, stdlib-first" — see
`docs/THREAT_MODEL.md` and the vs-mesh-llm "borrow the mechanism, keep the
wall" pattern already used for P1-P4), this module implements the *mechanism*
natively: **simultaneous-open UDP, HMAC-authenticated with a pair-specific
shared token**, so a bystander who observes the public rendezvous can watch
packets fly but can't forge one into the path.

## 2. What was built

`scripts/mesh/direct_path.py` — pure module, stdlib only (`hmac`, `hashlib`,
`ipaddress`, `socket`, `struct`), no imports from meshd/transport/discovery:

- **`PathResult(direct, endpoint, rtt_ms, reason)`** — the outcome type.
- **`punch(sock, local_id, peer_id, peer_endpoints, token, timeout, *,
  retry_interval=0.05)`** — the state machine. Both sides call this
  concurrently with a socket they already own. Each burst re-sends `HELLO`
  to every candidate (LAN-ordered, see below); on any authenticated `HELLO`
  it replies `HELLO_ACK` (echoing the sender's nonce); on a `HELLO_ACK` that
  echoes a nonce *we* actually sent, it replies `ACK` and declares the path
  live immediately (that side already has proof of a live round trip); on an
  `ACK` that echoes a `HELLO_ACK` *we* sent, likewise. Every message is
  `MAGIC + type + sender_id + nonce + echo_nonce` HMAC-SHA256'd over the pair
  token (`_pack`/`_unpack`); a bad MAC or an unexpected `sender_id` is
  silently dropped — indistinguishable from "no packet arrived", which is
  exactly what makes forged frames unable to hijack a path (they just never
  complete the 3-way).
- **`order_candidates(endpoints)`** — RFC1918/loopback/link-local candidates
  first, public after, stable within each group (`ipaddress.ip_address(...).is_private`).
  `punch()` sends this ordered list every burst rather than staging a delay
  window: a LAN round trip is physically far faster than a public/relay-
  observed one, so the tie resolves itself to the LAN path without slowing
  down the (much more common, per the recon) cross-NAT case.
- **`probe_rtt(sock, endpoint, token, ...)`** / **`answer_ping(sock,
  local_id, token, ...)`** — a PING/PONG pair for measuring/health-checking
  an already-established path (same auth, same framing).
- **`maybe_direct(sock, local_id, peer_id, peer_endpoints, token, *,
  timeout=3.0)`** — the integration seam. Gated on `NKS_DIRECT_PATH`
  (default unset = OFF); with it off, returns
  `PathResult(direct=False, reason="NKS_DIRECT_PATH disabled (default OFF)")`
  **before opening `punch()`'s loop at all** — zero packets sent, a true
  no-op (asserted directly in `test_maybe_direct_default_off_sends_nothing`,
  which checks nothing arrived at the peer socket).

`tests/test_direct_path.py` — 12 tests, two real UDP sockets on `127.0.0.1`,
1.4s total, no sleep longer than the 0.5s worst-case timeout in the
one-side-silent case:

| test | proves |
|---|---|
| `test_order_candidates_lan_before_public` (+ hostname variant) | pure ordering function, LAN-first, stable |
| `test_punch_both_sides_success` | full simultaneous 3-way over real sockets, both sides confirm, RTT measured |
| `test_punch_lan_candidate_preferred_when_mixed_with_unreachable_public` | a dead decoy candidate doesn't block the live one |
| `test_punch_one_side_silent_falls_back` | a bound-but-non-participating peer → clean timeout, `direct=False` |
| `test_punch_no_candidates_fails_immediately` | empty candidate list fails fast, no network I/O |
| `test_punch_rejects_bad_hmac_forged_hello_ack` | an attacker who intercepts a HELLO and forges a `HELLO_ACK` under the wrong token cannot complete the handshake |
| `test_pack_unpack_roundtrip_and_tamper_rejected` | wire-level: flipping one bit anywhere (payload or MAC) breaks verification |
| `test_probe_rtt_measures_round_trip` / `..._none_when_peer_silent` | the RTT helper |
| `test_maybe_direct_default_off_sends_nothing` / `..._enabled_runs_the_real_punch` | the flag seam behaves as documented in both states |

Full repo collection before/after: **941 → 953 tests** (+12, exactly the new
file), same **18 pre-existing collection errors** (unrelated `hivemind`/
`petals` optional-dependency tests — `INITIAL_PEERS` env var, struct-unpack
fixture mismatches; present on `main` before this branch). Existing
mesh/discovery/transport suites (`test_meshd_discovery.py`,
`test_mesh_pairing.py`, `test_relay.py`, `test_identity_handshake.py`,
`test_secure_channel.py`, `test_mux_tunnel.py`, `test_discovery.py`) all
still green (48 passed, 1 skipped) — nothing in the existing protocol was
touched.

## 3. What real-NAT scenarios localhost CANNOT prove — read this before trusting the green tests

Every test here runs two sockets bound to `127.0.0.1` on the same kernel.
That's enough to prove the **state machine and the authentication are
correct** — the HELLO/HELLO_ACK/ACK sequencing, the RTT measurement, the
HMAC rejection of forged frames, the LAN-first ordering, the flag no-op
contract. It proves **nothing** about whether the punch actually defeats a
real NAT, because there is no NAT in the loop. Specifically, honestly, what
this test suite cannot exercise:

- **Full-cone NAT** (the friendliest case: external mapping is per
  internal-endpoint, reusable by anyone) — punching against it should behave
  like the localhost test does. Untested against a real box.
- **(Address-)restricted-cone NAT** — the external mapping only accepts
  return traffic from an address the internal host already sent to. This is
  *why* the punch has to be simultaneous (both sides "send to" each other
  before either "receives from" the other) — the whole reason this module
  exists instead of a one-shot connect. Localhost has no such restriction to
  violate, so a bug that only manifests under restricted-cone rules (e.g. a
  timing window where one side's outbound packet hasn't left before the
  other's arrives) would not show up here.
- **Symmetric NAT** — allocates a *new* external (addr, port) mapping **per
  destination**, not per source. This is the hard case: the port a peer
  observed via the rendezvous relay (talking to the relay's IP) is *not* the
  port that same NAT will use when a punch packet targets a different peer's
  IP. Against symmetric NAT the whole rendezvous-observed-endpoint strategy
  structurally cannot work — no amount of protocol cleverness fixes it, only
  port-prediction heuristics (unreliable) or falling back to the relay (what
  this module already does). Two symmetric NATs facing each other is close
  to the worst case and is common on cellular/CGNAT links. **This needs a
  live two-site test** (e.g. hub ↔ an office laptop on a different ISP/NAT,
  per `project_lab_mesh_site.md`'s mac4-lab precedent) to know which of
  Biswa's actual node pairs land in this bucket.
- **CGNAT / double-NAT** (common on ISP "shared IPv4" plans and most LTE)
  compounds the above — even full-cone-behind-full-cone can behave like
  restricted or symmetric once a carrier-grade NAT is added in front of the
  home router.
- **Real WAN RTT, jitter, loss, MTU/fragmentation** — loopback RTT is
  sub-millisecond and lossless; a real cross-site UDP path has all three,
  and `punch()`'s fixed `retry_interval=0.05` / burst-everything strategy was
  tuned for a fast localhost test, not for minimizing wasted packets over a
  slow/lossy WAN link (that tuning is itself a live-network question).
- **Firewalls that silently drop vs. actively reject** — a host with nothing
  listening on 127.0.0.1 doesn't raise an OS error on `sendto` in these
  tests (verified: `test_punch_one_side_silent_falls_back` uses a real bound
  idle socket, not an unbound port, specifically so it degrades the same way
  a real silently-dropping NAT/firewall would — but a NAT that sends back an
  ICMP port-unreachable, vs. one that black-holes, are different real-world
  behaviors this can't distinguish).

**Bottom line:** the module is correct and safe to land (it can't do
anything worse than a slower relay path — see §5), but "does it actually cut
the 129ms dogleg on Biswa's real mesh" is an open empirical question that
needs hub↔pi, hub↔elitedesk, or hub↔lab-site measurement, not more unit
tests.

## 4. The meshd call-site plan (not wired in this branch)

`MeshNode._ensure_tunnel` (`scripts/mesh/meshd.py:194-255`) is where a tunnel
to an admitted peer comes up today. The relevant slice, inside the nested
`run_tunnel()` closure:

```python
sock = relay_connect(self.cfg.rendezvous_host, self.cfg.rendezvous_port,
                     role.rendezvous_id, timeout=30)
chan = secure_handshake(sock, self.priv, pin.ed25519_pubkey_hex,
                        role.is_initiator, session_binding=binding)
# <-- direct-path upgrade attempt would go HERE, after chan exists -->
mux = MuxTunnel(chan)
```

Why *after* `secure_handshake` and not before: `punch()`'s HMAC `token` must
be a secret only the two peers know — **not** derivable from the public
listing (mesh_id + Ed25519 pubkeys are gossiped in the clear over discovery,
same as `mesh/pairing.rendezvous_id`, which is deliberately *not* secret
because it only needs to be unguessable-enough to pair relay sockets, not to
authenticate). `secure_handshake` (`transport/secure_channel.py:88-139`)
already derives an ECDH shared secret (`shared = x_priv.exchange(...)`) via
X25519, then HKDFs it into per-direction ChaCha20-Poly1305 keys. The
straightforward extension: HKDF the *same* `shared` + `salt` with a new
`info` label (e.g. `HS_TAG + b"|direct-path-token"`) into a third, symmetric
32-byte key — that's the `token` `punch()` wants, provably known only to the
two authenticated peers, never sent on the wire. This needs a small,
additive change to `secure_handshake` (return the extra derived key
alongside the `SecureChannel`, or expose a `SecureChannel.derive(label)`
method) — not made in this branch; it's a one-function seam in
`secure_channel.py`, reviewed separately since it touches the live tunnel
crypto path.

The remaining pieces the meshd integration needs, none built here:

1. **Local candidate discovery.** meshd doesn't enumerate its own interfaces
   today. The standard stdlib trick (`socket.socket(AF_INET, SOCK_DGRAM);
   s.connect(("8.8.8.8", 80)); s.getsockname()[0]` — no packet actually
   sent, UDP `connect()` just asks the kernel to pick a route) gives the
   LAN-facing local IP without a new dependency.
2. **Learning the peer's *observed* public endpoint.** Today
   `RendezvousRelay._handle` (relay.py:154-167) pairs sockets and forwards
   bytes; it never tells either side what source `(addr, port)` it saw. A
   small, backward-compatible extension — the relay sends back one line
   (`b"OBSERVED " + addr + b":" + port`) right after pairing, for each side
   — is the "small message extension the rendezvous COULD serve" mentioned
   in this branch's brief. It costs the relay nothing extra to trust (it
   already sees the TCP peer address to accept the connection) and doesn't
   change the relay's trust model (still forwards-only, still can't forge
   the punch HMAC).
3. **Exchanging candidate lists.** Once `chan` is up, each side sends its
   local candidates (from #1) and the plan is to ALSO learn the peer's
   `chan`-reported observed endpoint (from #2, learned by each side about
   *itself*, then shared over the now-encrypted `chan`) — one small framed
   message over `chan.sendall`/`chan.recv` before handing `chan` to
   `MuxTunnel`.
4. **The punch itself.** Open a UDP socket, call
   `maybe_direct(udp_sock, self.node_id, peer.node_id, candidates, token,
   timeout=2.0)` (env `NKS_DIRECT_PATH=1` to opt in). On `direct=True`: the
   relay-based `chan`/`mux` stays up as the fallback (never torn down purely
   on a punch success — only replaced once the new path has proven itself
   over more than one probe, e.g. a few `probe_rtt` rounds) while a NEW
   authenticated channel is negotiated **over the raw UDP punch socket**.
5. **UDP-native secure channel — the one real gap.** `SecureChannel`
   (`transport/secure_channel.py:148+`) and `MuxTunnel`
   (`transport/mux_tunnel.py`) are built on `sendall`/`recv` against a
   **stream** socket (TCP semantics: ordered, no datagram boundaries). A
   punched UDP socket doesn't give you that for free — reordering and loss
   need to be handled before `MuxTunnel` can ride it unmodified. Options:
   (a) a minimal reliable-datagram shim (seq+ack, small window) under the
   existing `sendall`/`recv` interface, or (b) route the direct UDP path
   through something that already speaks datagram-native (this is the exact
   niche QUIC — and iroh — fill; staying stdlib-only here means building
   (a)). **This is explicitly NOT solved by this branch** — `direct_path.py`
   proves connectivity (the punch) and gives you an authenticated raw
   datagram socket; turning that into a `MuxTunnel`-compatible stream is
   the next diff, gated behind its own review since it's new wire protocol
   on the data path.

Given #5, the honest characterization of what THIS branch ships is: **the
hole-punch mechanism, proven correct in isolation, ready to be handed a
token and endpoints** — not an end-to-end swap of the live gRPC tunnel onto
UDP. That last mile is real work and deliberately out of scope per the
brief ("no changes to the live meshd service").

## 5. The fallback guarantee

- `punch()` and `maybe_direct()` **never raise** on network errors — a
  send failure, a timeout, a malformed/forged packet all degrade to "try
  again next burst" or, at the deadline, `PathResult(direct=False,
  reason=...)`. There is no code path where a punch failure propagates as
  an exception into a caller that isn't ready for one.
- `maybe_direct()` is off by default (`NKS_DIRECT_PATH` unset) and, when
  off, sends **zero packets** — verified directly
  (`test_maybe_direct_default_off_sends_nothing` asserts the peer socket
  receives literally nothing).
- `transport/relay.py`, `mesh/pairing.py`, and `scripts/mesh/meshd.py` are
  **byte-for-byte unchanged** in this branch (`git diff main -- scripts/mesh/meshd.py
  scripts/transport/relay.py scripts/mesh/pairing.py` is empty as of this
  commit) — the relay path a live mesh depends on today is untouched, not
  just "still passing tests."
- Per §4, even the planned integration keeps the relay-based `chan`/`mux`
  alive until a direct path has proven itself over multiple RTT probes, and
  only ever *adds* a path — it does not remove the relay fallback from the
  code, ever. A NAT-hostile peer pair (§3's symmetric-NAT case) simply never
  gets `direct=True` and stays on the relay exactly as it does today.
