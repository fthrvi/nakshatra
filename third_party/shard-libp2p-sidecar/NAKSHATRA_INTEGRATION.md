# libp2p sidecar — vendored from `leyten/shard`, for nakshatra's permissionless tier

> **Attribution / License.** This directory is **vendored verbatim from
> [`leyten/shard`](https://github.com/leyten/shard) `sidecar/` @ `e2469732`**, licensed
> **Apache-2.0** (see `LICENSE`). Unmodified. Finding **#22** (speed-stack plan / shard delta).
> We adopt it because it's the part of shard that is *both proven and engine-agnostic* — a
> transparent TCP↔libp2p tunnel that doesn't care what runs on the wire, so it carries our
> existing gRPC/fabric without engine changes.

## What it is
A Go `go-libp2p` daemon that runs beside a node's engine and gives it, for free:
- a **per-node ed25519 identity → libp2p PeerId** (no shared secret — replaces shard's old PSK),
- **NAT traversal**: DCUtR hole-punching (QUIC) + circuit-relay-v2 fallback (home GPUs behind NAT join),
- a **transparent TCP↔libp2p tunnel** (the engine keeps speaking plain TCP locally).

Interface (`main.go`):
- `-key <path>` — persist the node key (stable PeerId across restarts).
- `-inbound HOST:PORT` — inbound libp2p streams are dialed to the local engine here (a **worker**: its gRPC port).
- `-forward LOCAL=PEER_MULTIADDR` — listen on a LOCAL tcp port, carry each conn to PEER (a **coordinator**: one per upstream worker, repeatable).
- `-relay` — also be a public relay + AutoNAT server (run on a reachable box).

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
Vendored + documented (this commit). **Remaining:** install Go → build → a `sidecar.service` per
node → rewrite a from-roster chain to tunnel endpoints → prove our gRPC tunnels over libp2p between
two NAT'd boxes (wants the **2nd box**). Then strangers can join the permissionless tier.
