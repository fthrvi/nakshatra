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
| **Reach a peer** (relay + pinned handshake + purpose binding) | **`transport/connect.py`**: `open_channel` (relay), `open_direct` / `accept_direct` (direct TCP over LAN or IPv6, using `mesh/direct_tunnel` and its SSRF filter). meshd, nakd and tunnel_endpoint all use it. | LIVE | every node |
| Direct connections between contacts (U3b) | nakd: opt-in per contact on BOTH sides (`nak direct <name> on`) and per node (`nak direct-listen 51830`). Addresses are swapped only inside the encrypted session. One dialer (the pairing initiator); the same pinned handshake with a direct-only binding; the relay is the fallback. | LIVE (0.9.0) | contacts who opt in |
| UDP hole punch for both-NAT pairs | `mesh/direct_path.py` (needs a reliable stream on top) and the libp2p sidecar | BUILT / LIVE for Sutra only | → next |

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
| ~~`~/.nakshatra/mesh.key`~~ | meshd, before U2 | **retired 2026-10-03**: meshd now defaults to the node key (file kept only for rollback) |
| sidecar key (`~/.config/nakshatra-sidecar/*.key`) | libp2p PeerId | `scripts/sidecar_key.py` writes it FROM the node key (same libp2p format). The live blackwell sidecar still has its own key, because switching changes its PeerId and Sutra's tunnel dials it: coordinate with that lane. |
| `~/.nakshatra/keys/nostr.secp256k1` | Nostr transport only | another curve, required by Nostr. Already bound: the Nostr event carries the listing, and the listing is signed by the node key. |
| person key (signer state) | the human; issues agent grants | separate on purpose: machine ≠ person |
| `~/.nakshatra-worker/` key | outsider-GPU workers onboarded by trisul `worker.sh` | another node-key location → U4 folds it into the node key |

## Joining (who joins → the ONE live path for it today)
| Who joins | Live path | Installer | Key | Status |
|---|---|---|---|---|
| Your own **device** onto your private mesh | trisul `infra/onboarding` `mesh-invite.sh` / knock (Biswa approves) → `onboard_server` `/redeem` | WireGuard `join.sh` | WireGuard | LIVE (VPS `onboard.service`) |
| An **outsider's GPU** into your compute pool, at a trust tier | trisul `worker-invite.sh` → `worker.sh` → `onboard_server` `/worker-redeem` → `admission/peers.tsv` → junction | **unsigned** tarballs (`worker-llama-stack.tgz`, `worker-scripts.tgz`) | `~/.nakshatra-worker/` (a separate key location) | LIVE (blackwell's registrar came in this way) |
| Your own **machine** into your fleet | `fabric/join.py` → Sthambha `/join` | by hand | node key | WIRED |
| A **person**, as a contact | `release/install.py join` (one pasted line) | **signed release** | node key | LIVE |
| (engine for the outsider-GPU path) | `scripts/join/` six-phase decision engine | — | — | BUILT: decisions tested, observation layer stubbed. **Not dead**: it is the planned engine for the worker kind. |

**U4 (agreed direction): one invite, one installer, one key, several kinds.**
- One installer: the signed release, everywhere. It retires the unsigned worker tarballs. The worker stack moves into the release first (U6).
- One invite: the signed `nki1.` invite carries a `kind` (contact | worker | device). A worker invite also carries the one-time code from `onboard_server` (which stays the roster authority).
- One node key: `~/.nakshatra/keys/worker.ed25519`, replacing `~/.nakshatra-worker/`.
- Order: U6 (worker stack in the release) → the U4 worker kind → the device kind.

## Accounting
| Job | Component | Status |
|---|---|---|
| Meter inference: `gate` / `settle(receipt)` | `ledger_client.py` → Neuron ledger | LIVE (`NAKSHATRA_CREDITS=1` → :8097) |
| Settle once per run; tier credit limits | `settlekey.py`, `creditlimit.py` | BUILT / WIRED |
| Task escrow: open / release / refund | `network/settle.py` (local ledger adapter) | LIVE (TEST units) |
| **The front door for both halves** | **`scripts/accounting.py`**: `metering()` (the gateway's credit hook) and `escrow(state_dir)` (task escrow). New backends (Solana devnet escrow, the repaired Neuron ledger) are added HERE. | LIVE (U5, 2026-10-03) |
| Ledger service | Neuron `python/ledger_server.py` :8097 | LIVE, **unauthenticated, racy, takes forged receipts** (fix before money) |
| Chain | Neuron Substrate | LIVE, being stopped (Biswa 10-03) |

## Shipping code to machines
| Job | Component | Status |
|---|---|---|
| Signed releases, installer, rollback, hourly self-update, `nak` commands | `release/` | LIVE (0.7.0 from main) |
| Public release host | VPS :8960 (`nakshatra-releases`), `release/publish.sh` | LIVE |
| meshd + relay | **from the signed release** (`nakshatra-meshd` / `nakshatra-relay`, opt-in per node via `~/.nakshatra/{meshd,relay}.env`) | LIVE on the hub since 0.8.0 (U6a) |
| Units for a new release | written by the installer SHIPPED IN that release (`install.py write-units`), never the older running one | LIVE since 0.8.1 |
| Inference gateway + GPU workers | still from the `~/nakshatra` working tree / hand-built stacks / unsigned worker tarballs | → U6b: the worker stack into the release (Biswa's go first: it is Prithvi's brain) |
