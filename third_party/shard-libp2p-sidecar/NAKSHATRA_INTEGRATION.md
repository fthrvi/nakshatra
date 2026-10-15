# libp2p sidecar — vendored from `leyten/shard`, for nakshatra's permissionless tier

> **Attribution / License.** This directory is **vendored from
> [`leyten/shard`](https://github.com/leyten/shard) `sidecar/` @ `e2469732`**, licensed
> **Apache-2.0** (see `LICENSE`). Finding **#22** (speed-stack plan / shard delta).
> We adopt it because it's the part of shard that is *both proven and engine-agnostic* — a
> transparent TCP↔libp2p tunnel that doesn't care what runs on the wire, so it carries our
> existing gRPC/fabric without engine changes.
>
> ⚠️ **No longer unmodified as of 2026-09-07.** Nakshatra carries two deliberate patches in `main.go`:
> upstream's relay-reservation loop calls `relayclient.Reserve()` exactly once per process and
> never again — a reservation silently expires (~1hr TTL observed) with nothing here to notice,
> and the first sign of trouble is a downstream dial failing `NO_RESERVATION` on the relay's
> side, potentially much later. Added `renewRelayReservation`/`halfLife`: a background goroutine
> per relay that re-reserves at half the granted TTL (with a 30s retry backoff on failure), for
> as long as the process runs. Verified against a real relay with a deliberately short TTL
> (15s) — the reservation renewed continuously across multiple cycles with no gap. The second patch
> adds the uid-gated UNIX-socket `-dial-listen` protocol used by nakd: relay circuits rendezvous DCUtR, but
> contact bytes are accepted only on the resulting direct connection. Both patches have Go tests.

## What it is
A Go `go-libp2p` daemon that runs beside a node's engine and gives it, for free:
- a **per-node ed25519 identity → libp2p PeerId** (no shared secret — replaces shard's old PSK),
- **NAT traversal**: DCUtR hole-punching (QUIC) + circuit-relay-v2 fallback (home GPUs behind NAT join),
- a **transparent TCP↔libp2p tunnel** (the engine keeps speaking plain TCP locally).

Interface (`main.go`):
- `-key <path>` — persist the node key (stable PeerId across restarts).
- `-inbound HOST:PORT` — inbound libp2p streams are dialed to the local engine here (a **worker**: its gRPC port).
- `-nakd-inbound HOST:PORT` — direct-only nakd streams are dialed to nakd here; unlike activation streams, limited relay circuits are refused.
- `-forward LOCAL=PEER_MULTIADDR` — listen on a LOCAL tcp port, carry each conn to PEER (a **coordinator**: one per upstream worker, repeatable).
- `-relay` — also be a public relay + AutoNAT server (run on a reachable box).
- `-dial-listen PATH` — accept `DIAL <peerid>` from nakd on a mode-0600 UNIX socket, open the dedicated nakd protocol, and return only a direct stream.
- `-relay-activation-fallback` — after the direct wait, permit dynamic `FORWARD` activation
  streams on the existing limited relay circuit.
- `-allow-relayed-activation` — mandatory worker opt-in for every relayed activation stream,
  independent of the worker's `-require-direct-activation` direct-path policy. The release wiring
  requires both dialer and worker flags.
- `-direct-wait DURATION` — direct-path wait before fallback (default remains `15s`).

## How it plugs into nakshatra (the wiring — next step, needs Go to build)
The beauty: **nothing in our engine changes.** We only rewrite the chain's worker *addresses* to
local tunnel endpoints and let the sidecar route by PeerId.

- **Each worker box** runs: `sidecar -key node.key -inbound 127.0.0.1:<worker_grpc_port>` and advertises
  its `/p2p/<PeerId>` multiaddr (into the roster / `peers.tsv`).
- **The coordinator** (client.py) runs one sidecar with a `-forward` per chain worker:
  `-forward 127.0.0.1:6000=<wA multiaddr> -forward 127.0.0.1:6001=<wB multiaddr> …`, then the
  cluster YAML's worker addresses become `127.0.0.1:6000`, `127.0.0.1:6001` — local tunnel endpoints.
  Our `call_forward`/fabric talks to those; the sidecar carries it over libp2p to the real PeerId.

So `client.py` and `worker.py` are untouched; the chain config points at tunnel endpoints.

## Dual-transport architecture (why we keep BOTH)
- **Trusted / owned tier** (Prithvi's brain, our boxes) → **WireGuard mesh + signed admission**
  (`infra/wireguard/`, `transport/junction.py`). Sovereign, operator-controlled.
- **Permissionless tier** (strangers' GPUs joining to earn credits) → **this libp2p sidecar**.
  Public PeerId identity, DCUtR NAT, no mesh enrollment needed. The credit ledger + admission tiers
  (`worker_join.eligible_workers`) still gate *who earns* and *which models* a stranger may serve.

A stranger joins via libp2p, is admitted at a low tier (`general`/`known`), serves only public
models (the identity firewall keeps Prithvi's mind off stranger GPUs), and earns credits.

## Build (when ready)
Needs Go (≥1.25 per `go.mod`): `cd third_party/shard-libp2p-sidecar && go build -o sidecar .`
Then a systemd `--user` unit per role, like `neuron-ledger.service`.

## Status
The signed node release builds the pinned source as a static linux/amd64 artifact and ships the
opt-in `nakshatra-p2p` service. Nakd uses it for both-NAT contact sessions after both contacts enable
direct paths and advertise p2p. Typed inference endpoints use its dynamic `FORWARD` listener; relay
fallback remains separately opt-in at both the consumer gateway and worker.

Relay service defaults are explicitly capped at 64 MiB per direction per circuit, 6 hours,
64 circuits total, and 2 per peer. Established circuits are conservatively charged their full
128 MiB bidirectional admitted cap against persistent 20 GiB global and 2 GiB per-peer UTC-day
budgets. `-relay-allow-file` accepts the roster's node-key hex (the release sidecar key is derived
from that same key) or its libp2p PeerId; it does not accept the separate operator/release/package
signing keys. Relay startup requires this file; the production binary has no open-relay flag. The
production root-owned, group-readable
`root:nakshatra-sidecar-relay` mode 0640 allow file is accepted because it is read-only input. The
state file, unlike the allow file, must be owned by the running relay uid and is exclusively locked
for the process lifetime. `nak gateway on` enables the
dialer fallback only in the gateway-owned `p2p.env`; `nak worker relay on|off` atomically toggles
the worker opt-in and restarts an enabled P2P sidecar.
