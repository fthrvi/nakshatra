# sidecar — shard's libp2p transport daemon

A Go (`go-libp2p`) daemon that runs next to the Python engine on every node. It owns the
node's keypair identity (an ed25519 key → libp2p PeerId) and moves activation frames
between adjacent pipeline stages over authenticated, encrypted libp2p streams. Per the
boundary law ([../docs/INTEGRATION.md](../docs/INTEGRATION.md)) it knows only **peers and
bytes** — nothing about `$ZERO`, accounts, payments, or the orchestrator.

It's a single static binary (no CGO), so we build once and `scp` it to any linux/amd64
node — a node needs no Go toolchain to run it.

## Build

```sh
cd sidecar
GOTOOLCHAIN=local GOPROXY=https://goproxy.io,direct go build -o /tmp/sidecar .
```

Two load-bearing build pins (in the spirit of `phase0/setup_box.sh`'s `kernels` pin):

- **`go-libp2p v0.33.2`** — builds on Go 1.22 (our toolchain); newer releases want Go 1.23/1.25+.
  It still ships DCUtR + circuit-relay-v2, so step 2 (NAT traversal) isn't blocked by the pin.
- **`GOPROXY=goproxy.io`** — `proxy.golang.org` returns `403` on the `klauspost/compress` zip
  (and Go only falls back to `direct` on `404`, not `403`), which stalls the whole resolve.
  `goproxy.io` serves it fine.

## Run as a tunnel (how the engine uses it)

The sidecar is a transparent TCP↔libp2p tunnel. Pin the libp2p port to the node's
public-mapped port; the engine then dials/listens on localhost and the sidecar carries
each connection to/from the right ring neighbour:

```sh
# a stage: accept inbound streams -> the local engine, and carry the engine's
# next-hop connection to the downstream peer over libp2p
sidecar -key /root/node.key -listen /ip4/0.0.0.0/tcp/29600 \
        -inbound 127.0.0.1:29610 \
        -forward 127.0.0.1:29611=/ip4/<peer_ip>/tcp/<peer_port>/p2p/<peer_id>
```

The engine runs unchanged except one import: `import wire` → `import shard.transport as
wire` (see `../shard/transport.py`). Proven: gpt-oss-120B across 4 scattered boxes over
libp2p, no `SHARD_PSK`, bit-identical to the trusted-wire receipt
(`../docs/receipts/gpt-oss-120b-libp2p-20260619.json`).

## Self-test — identity + connect-by-key + round-trip (connectivity check)

```sh
# terminal A (listener): prints its PeerId + dialable multiaddr, writes the addr to a file
/tmp/sidecar -key /tmp/a.key -addrfile /tmp/addrA

# terminal B (dialer): connects by key, round-trips a 2 MiB activation-sized frame
/tmp/sidecar -key /tmp/b.key -peer "$(cat /tmp/addrA)" -size 2097152
# -> ROUND-TRIP OK: 2097152 bytes echoed by <A's PeerId> ...
```

`ROUND-TRIP OK` with two distinct PeerIds = the transport is up, authenticated by key, and
intact. No `SHARD_PSK` anywhere.

## Nakd direct-path socket

`-dial-listen "$XDG_RUNTIME_DIR/nakshatra/p2p.sock"` enables the uid-gated UNIX-socket protocol used by nakd. The
directory and socket are forced to modes 0700 and 0600, and a directory owned by another uid is refused. A local client
writes `DIAL <peerid>\n`. The sidecar may use a configured `-relays` circuit to rendezvous DCUtR,
but waits up to `-direct-wait` (15 s by default) for a non-relayed connection and opens the stream
only there. It replies `OK direct\n` before piping bytes, or `ERR <reason>\n` and closes. User data
is never carried over the relay circuit.

## Capped activation relay fallback

Dynamic inference `FORWARD` tunnels may opt into `-relay-activation-fallback`. They still wait
for a direct path first (`-direct-wait`, unchanged at 15 seconds by default), then use the existing
limited circuit connection. The worker must independently pass `-allow-relayed-activation` alongside
`-require-direct-activation`; otherwise it resets the relayed activation stream.

`-relay` servers are explicitly capped even without extra limit flags: 64 MiB in each direction per
circuit (`-relay-circuit-bytes`), 6 hours (`-relay-circuit-duration`), 64 concurrent circuits total
(`-relay-max-circuits`), and 2 per peer (`-relay-circuits-per-peer`). The global UTC-day budget is
20 GiB (`-relay-daily-bytes`) and the per-peer budget is 2 GiB (`-relay-peer-daily-bytes`).
Each attempt receives an identity-bound admission token carrying source and destination from the
reservation check through establish, failure, and close. Pending plus active attempts share the
global and per-peer concurrency limits. A successful circuit is conservatively charged its full
possible traffic in both directions (128 MiB with defaults); a failed attempt releases its exact
reservation without charging another attempt. Admission happens only after the destination's relay
reservation is confirmed.

`-relay` requires `-relay-allow-file PATH` and refuses to start without it. `-relay-open` is an
explicit test/lab escape hatch and emits a loud warning; do not use it in production. The allow
file contains one identity per line,
either a libp2p PeerId or the 64-lowercase/uppercase-hex Ed25519 **node public key**. When this flag
is present, reservations require the destination to be listed and circuits require both source and
destination to be listed. Blank lines and `#` comment lines are ignored, and CRLF is accepted. The
loader uses one `O_NOFOLLOW` descriptor for validation and reading, caps the file at 1 MiB and
10,000 entries, and detects changes by device/inode/size/nanosecond-mtime plus SHA-256. Symlinks,
missing/unreadable files, and any malformed non-comment line deny everyone and log its line number.
The allow file is read-only input and need not be owned by the relay uid: the production
`root:nakshatra-sidecar-relay` mode `0640` file is accepted.

State is written with mode 0600 to `-relay-state`; by default this is `<key>.relay-state.json` (or
`./relay-state.json` when no `-key` was supplied). A sibling lock is exclusively flocked for the
relay lifetime, so a second process using the same state refuses to start. Existing state must be a
non-symlink regular file owned by the running uid and not group/world writable. Corruption refuses
startup. Writes use temp + fsync + rename + directory fsync. The persisted maximum UTC day prevents
a backward clock adjustment from granting a fresh budget.

The VPS relay launch is:

```bash
nakshatra-sidecar -relay -key ... -listen /ip4/0.0.0.0/tcp/29700 -quic -announce ... -relay-allow-file /var/lib/nakshatra-sidecar-relay/relay.allow -relay-state /var/lib/nakshatra-sidecar-relay/relay-state.json
```

For signed Nakshatra releases, `scripts/sidecar_key.py` deterministically builds the sidecar
`node.key` from `~/.nakshatra/keys/worker.ed25519`. Therefore the correct allow-file hex is the
roster row's node-public-key field (the first field in `admission-peers.tsv`), and it derives the
same PeerId printed by the sidecar. It is **not** `operator.pub.hex`, a release signing key, or a
package signing key. A legacy sidecar key created independently is different: list its printed
PeerId, or migrate it through `sidecar_key.py` and update every address that pins the old PeerId.
