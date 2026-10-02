# Nakshatra components: the one list (2026-10-03)

**Read this before building anything.** Every job has ONE home. If what you are about to build fits a
row, extend that row's component. If no row fits, add a row in the same change. This file exists
because the same things got built twice:
- hole punching existed, unwired, while it was being proposed as new;
- four join paths and four accounting paths grew side by side.

Status was checked on the machines on 2026-10-03, not taken from older docs:
- **LIVE** = running now;
- **WIRED** = called by something that runs;
- **BUILT** = code + tests, nothing calls it;
- **DEAD** = known not to work (kept only as a pointer).

Unification work in progress (U1–U6): `trisul/plans/2026-10-03-nakshatra-unification-plan.md`
(branch `network/agent-net-survey`).

## Inference: one model split across machines
| Job | Component | Status | Runs where |
|---|---|---|---|
| Serve a layer slice | `scripts/worker.py` + patched llama.cpp daemon (`third_party`) | LIVE | inference nodes |
| Drive one request through a chain | `scripts/client.py` (per-request; no standing coordinator) | WIRED | spawned by the gateway |
| Decide where layers go | `placement.py`, `fabric/serve_planner.py`, `serve_chain.py`, `topology_order.py` | WIRED | gateway |
| OpenAI-compatible endpoint | `nakshatra_serve.py` | LIVE | hub: `nakshatra-unconscious` :11599, `nakshatra-deep-bw` :11601 (deep rung: no backend while ijru is down) |
| Summon / reap + leases | `fabric/serve_lifecycle.py` (`PillarLeaseClient` → Sthambha) | WIRED | gateway |
| Declarative reconcile | `serving` lifecycle-reconcile | BUILT (timer not armed) | — |
| Speculative decoding, remote proposals | `speculative.py`, `remote_proposals.py` | BUILT, behind flags | — |
| Signed run receipts, participation signatures, disputes | `receipt.py`, `identity_binding.py`, `dispute.py` | WIRED | gateway |

## Network: finding and reaching peers
| Job | Component | Status | Runs where |
|---|---|---|---|
| Publish / discover signed listings | `mesh/meshd.py`, `discovery/*` (FileRelay; Nostr with `--nostr-relay`) | LIVE | hub `nakshatra-meshd` |
| Rendezvous relay (pairs two outbound sockets, pipes ciphertext) | `transport/relay.py` | LIVE | VPS :51820 (`nakshatra-rendezvous`), hub |
| Pinned handshake → encrypted channel | `transport/secure_channel.py`, `mesh/pairing.py` | LIVE | everywhere |
| Many streams over one channel | `transport/mux_tunnel.py` | LIVE | meshd tunnels |
| Admission-gated relay (default deny, trust tiers) | `transport/junction.py` + trisul `infra/control-plane/admission.py` | LIVE | VPS :9778 |
| **Hole punching + relay fallback (libp2p DCUtR, circuit-relay-v2)** | `third_party/shard-libp2p-sidecar` | **LIVE**: VPS relay :29700 (+QUIC); blackwell exposes Ollama through it | VPS, blackwell |
| Native UDP hole punch (HMAC-authenticated) | `mesh/direct_path.py` | BUILT, **not called** (proven 3.77× on WAN 2026-08-01) | — |
| Relay / direct / IPv6 decision; NAT class | `pathchoice.py`, `ipv6.py`, `natclass.py`, `stunshape.py`, `mesh/path_probe.py` | BUILT, not called | — |
| Dial a peer (relay + pinned handshake) | three copies today: `meshd._ensure_tunnel`, `network/nakd._dial`, `transport/tunnel_endpoint.py` | LIVE | → **U3 folds them into one** |

## People and their agents (added 2026-10)
| Job | Component | Status | Runs where |
|---|---|---|---|
| Contacts, invites + accept, messages, inbox (untrusted) | `network/nakd.py`, `store.py`, `client.py`, `nak.py` | LIVE | hub, bro, test box (`nak-net`) |
| Agent front door for any harness | `network/nak_mcp.py` | LIVE | bro's OpenClaw |
| Tasks: post → claim → assign → result → accept / reject | `network/tasks.py` + nakd | LIVE | hub ↔ bro |
| Person → agent grants, typed envelopes, spend caps | Sthambha `sthambha/signer.py` | LIVE | `nak-signer` on every node |
| Signed invites | `joincode.py` (`nki1.`) | LIVE | — |

## Identity
| Key | Used by | Note |
|---|---|---|
| `~/.nakshatra/keys/worker.ed25519` | worker, pillar auth (`Sthambha-Ed25519`), nakd, signer's `node.pub` | **the node key** |
| `~/.nakshatra/mesh.key` | meshd (`--identity-file` default) | a SECOND node identity → **U2 folds it into the node key** |
| sidecar key (`~/.config/nakshatra-sidecar/*.key`) | libp2p PeerId | a third → U2 derives it from the node key |
| `~/.nakshatra/keys/nostr.secp256k1` | Nostr discovery | another curve, required by Nostr; bind it to the node key in the listing |
| person key (signer state) | the human; issues agent grants | separate on purpose: machine ≠ person |

## Joining
| Who joins | Component | Status |
|---|---|---|
| A person (gets a node, asks a friend to connect) | `release/install.py join` (one pasted line) | LIVE (verified from zero) |
| Your own machine into your fleet | `fabric/join.py` → Sthambha `/join` | WIRED |
| An outsider's GPU, at a trust tier | `fabric/worker_join.py` + junction + trisul `infra/onboarding/worker.sh` | WIRED |
| `scripts/join/` (`nakshatra join`) + legacy `encode_join` / `decode_join` | — | **DEAD** → U4 retires it |

## Accounting
| Job | Component | Status |
|---|---|---|
| Meter inference: `gate` / `settle(receipt)` | `ledger_client.py` → Neuron ledger | LIVE (`NAKSHATRA_CREDITS=1` → :8097) |
| Settle once per run; tier credit limits | `settlekey.py`, `creditlimit.py` | BUILT / WIRED |
| Task escrow: open / release / refund | `network/settle.py` (local ledger adapter) | LIVE (TEST units) |
| → U5 puts metering and escrow under ONE interface | | |
| Ledger service | Neuron `python/ledger_server.py` :8097 | LIVE, **unauthenticated, racy, takes forged receipts** (fix before money) |
| Chain | Neuron Substrate | LIVE, being stopped (Biswa 10-03) |

## Shipping code to machines
| Job | Component | Status |
|---|---|---|
| Signed releases, installer, rollback, hourly self-update, `nak` commands | `release/` | LIVE (0.7.0 from main) |
| Public release host | VPS :8960 (`nakshatra-releases`), `release/publish.sh` | LIVE |
| The hub's meshd / relay / serve | still run from the `~/nakshatra` working tree | → U6 moves them onto the release |
