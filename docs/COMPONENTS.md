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
| Native UDP hole punch (HMAC-authenticated; retained for the WireGuard/inference path) | `mesh/direct_path.py` | BUILT, **not called** (proven 3.77× on WAN 2026-08-01) | — |
| Relay / direct / IPv6 decision; NAT class | `pathchoice.py`, `ipv6.py`, `natclass.py`, `stunshape.py`, `mesh/path_probe.py` | BUILT, not called | — |
| **Reach a peer** (relay + pinned handshake + purpose binding) | **`transport/connect.py`**: `open_channel` (relay), `open_direct` / `accept_direct` (direct TCP over LAN or IPv6, using `mesh/direct_tunnel` and its SSRF filter). meshd, nakd and tunnel_endpoint all use it. | LIVE | every node |
| Direct connections between contacts (U3b) | nakd: opt-in per contact on BOTH sides (`nak direct <name> on`) and per node (`nak direct-listen 51830`). Addresses and the p2p capability are swapped only inside the encrypted session. One dialer tries LAN/IPv6, then the sidecar, then the relay; every path uses the same pinned handshake and direct-only binding. | LIVE | contacts who opt in |
| UDP hole punch for both-NAT contacts | `third_party/shard-libp2p-sidecar` DCUtR via its circuit-relay-v2 rendezvous; `-dial-listen` gives nakd only a non-relayed stream, and the signed release ships it as opt-in `nakshatra-p2p` | LIVE | contacts with direct + p2p on at both ends |

**P2P live check (2026-10-08, 0.11.3):** both the hub and bro run `nakshatra-p2p` (`nak p2p on`) with relay reservations. A
hub→bro `DIAL` returned `OK direct` in 0.6 s, `DIRECT via /ip6/…/quic-v1 (ipv6)`. Before 0.11.3 it timed out:
DCUtR alone cannot join two peers behind the same NAT. The sidecar now also dials the peer's identified LAN/IPv6
addresses, under a connection gater (no loopback, link-local or multicast) and with work caps. Not yet tested: two
different NATs over IPv4 only (needs a second network).

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
| An **outsider's GPU** into your compute pool, at a trust tier | signed `nak invite --kind worker` → `release/install.py join` → node-key-signed onboard `/worker-redeem`; runtime remains trisul `worker.sh` until phase B | **signed release** for identity/invite; legacy worker tarballs for runtime | `~/.nakshatra/keys/worker.ed25519` | U4 phase A (signed identity + admission); runtime LIVE via `worker.sh` |
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
| Restarting services on update | only a service whose fingerprint changed: its unit + the commits of the components it runs + the env lock content. A service with `drain` waits for its port to go idle (gateway: :11599, up to 15 min) | LIVE since 0.10.7. Verified: an identical release restarted nothing (0.10.9) |
| Inference gateway (`nakshatra-unconscious`) | **from the signed release** with the `inference` profile (hash-locked grpcio/numpy/llama_cpp_python…), opt-in via `~/.nakshatra/{inference.env,unconscious.release}`; the hub's settings live in drop-ins | LIVE on the hub since 0.10.2 (U6b, 2026-10-03; cutover/rollback scripts in trisul `infra/nakshatra/`) |
| **The engine (patched llama.cpp)**: ONE source of truth | **`engine/`**: upstream b9992 + a patch series; `engine/source.sh` rebuilds it and refuses unless the tree hash matches exactly. Shipped in every release. Retired copies: the hub-only fork branch (now exported), the stale `experiments/v0.0` daemon (no `solo` mode), and `rebuild-worker.sh`, which would have reverted to it. | LIVE in releases since 0.10.3 (U6c); compiled from it on the hub |
| Build the worker for this GPU | `deploy/provision-worker.sh` (CUDA/ROCm/Vulkan/Metal/CPU, no sudo), building from `engine/` when run from a release | BUILT. The hub's live binary is still the one hand-built 09-12 from the identical tree; rebuild through this path at the next worker change. |
| Run a donated GPU worker from the signed release | `nak worker build`; then `nak worker on … --pillar-url https://… --pillar-spki <sha256>` creates the opt-in `nakshatra-worker` service (`~/.nakshatra/worker.env`). `scripts/worker_service.py` binds the machine-local daemon to the current release's engine TREE before it execs `worker.py`. Plain HTTP is accepted only for loopback or `TRUSTED_PILLAR_NETS` (default `10.42.0.0/24`, the encrypted WireGuard mesh). | BUILT (U4 B1). Uses the one node key; every HTTPS pillar call is SPKI-pinned; no unsigned script copy and no `--skip-sha256`. |
| Verify model slices/packages before use | `scripts/worker.py` streams each peer-fetched slice through SHA-256 and requires the expected digest from the pillar's file index. `scripts/packaging/fetch_package.py` streams each fragment against its manifest digest; a public remote manifest must be signed by an operator-pinned `--package-trusted-pubkey`. `--skip-sha256` is only an explicit `--local-dev` escape hatch and refuses non-local/non-loopback sources. | WIRED (U4 B3); outsider provisioning never supplies `--skip-sha256`. |

### Courteous unload + owner lease — worker side BUILT (2026-10-13)
Measured on bro (RTX 5070) with a 60 s test timeout. After 60 s idle the worker went to `residency -> standby` and freed its VRAM (GPU 4890 → 2327 MiB), and the pillar projected `standby`. A hub prompt then printed `[chain] warming … (standby) … 2.11s`; the worker re-verified the slice SHA-256, respawned the daemon and went `loaded`, after which tokens ran at 8 ms/step. The default idle backstop is `IDLE_UNLOAD_S=600` (0 disables). Session release now schedules the faster `RELEASE_GRACE_S=30` unload (0 immediate; negative disables); a new step/session cancels it. A crashed client's KV session can no longer pin VRAM: it is released once stale (300 s) and the idle backstop has passed.

The opt-in `~/.nakshatra/owner-lease.env` adds the loopback-only sentinel endpoint described in `COURTEOUS-WORKER.md`. Its default-deny state is `away`; live permission moves it to `standby`, demand loads it to `loaded`, and a lapse cancels work, frees session ownership, unloads, and immediately heartbeats `away`. The residency vocabulary is therefore exactly `loaded | standby | away`; the worker never admits work while away, and coordinators must exclude away workers. The pillar ranks `loaded > unknown > standby` only as a tie-breaker after health, admission, coverage and tier. Remaining work: the Windows sentinel and demand-based warm pools in the planner.

## Known installer follow-ups (Codex review, 2026-10-03; NOT regressions of p2p, all pre-existing)
- `current` switches before services drain, so an in-flight gateway request can spawn helpers from the new release.
- Rollback (automatic or explicit) restarts the previous release's services without draining.
- "Healthy" = stable process for 5 s; there is no readiness probe (e.g. the gateway's /v1/models, the sidecar's relay reservation).
- A request arriving between the last idle sample and the restart is cut (a narrow race).
- An interruption between the switch and saving the config leaves an unvalidated release active until the next run.

## First outsider GPU (bro, RTX 5070, WSL; 2026-10-13)
Proven:
- **Standby warm-up is tolerated.** The pillar's per-hop `residency` lets `client.py` add `NAKSHATRA_WARMUP_TIMEOUT_S` (default 60 s) to only a standby hop's first RPC and retry one warm-up `UNAVAILABLE` before normal recovery.
- **END TO END:** the hub asked "The capital of France is" and bro answered " Paris. The capital of Italy is Rome…" at 63 tok/s, with an 8 ms per-token RPC. The pillar served the chain, the roster firewall admitted bro (`known`) and the worker's TLS SPKI was pinned. Activations went over the sidecar FORWARD, **DIRECT via IPv6 QUIC**; the relay was used for discovery only.
- **One way in.** The hub ran `nak invite --kind worker --tier known` and bro ran `install.py join`. The node-key-signed `/worker-redeem` on the VPS onboard server then added bro to the roster at `known`.
- **Signed packages.** `llama32-3b` is signed by a dedicated package-signer key (`2a4d231e…`) and served from the untrusted release host (`:8960/packages/`). A remote fetch verifies the signature and every artifact's SHA-256.
- **Pinned pillar.** Off-overlay workers reach the pillar through an nginx TLS front on the VPS (`https://45.63.109.137:7778`, forwarding to the Pi `10.42.0.3:7777`, SPKI `17c38321…`).
- **Idle unload (step 1).** The worker now stops its GPU daemon after 600 idle seconds by default, while gRPC, TLS, heartbeat, and the verified slice stay up in standby. The first inference request re-verifies the startup SHA-256 and performs a single-flight reload. `nak worker on --idle-unload-s 0` disables that backstop; release-grace unload remains independently enabled unless `RELEASE_GRACE_S` is negative.

Outsider bugs fixed in 0.12.1–0.12.3:
- The CUDA build is capped by RAM (`WORKER_BUILD_JOBS` overrides).
- The installer's sdist build is capped too (`CMAKE_BUILD_PARALLEL_LEVEL`).
- nvcc is now found in `/usr/local/cuda*/bin`.

Fixed by `client/admission-defaults`:
- **Client ergonomics — FIXED (`1721315`, `252b228`).** The socket and keyid default from the release state, and the keyid fallback is load-only. The hub moved to the release registrar on 2026-10-13: the legacy hand-installed unit was kept as `nakshatra-registrar.service.legacy-20261013`, the same node key was reused so the pillar's `hub` lock holds, and the config is `~/.nakshatra/registrar.env`. A hub inference with ZERO manual env ("Water boils at" → "100°C (212°F)…", 63 tok/s) works. Pre-existing, not a regression: the registrar advertises `unknown offering 0.0GB` on the hub (the legacy one did too), so its VRAM probe via `rocm-smi` needs a look.

Status of the follow-ups (second inference with the defaults, 2026-10-13: "Jupiter. It is a gas giant…" at 51 tok/s):
- **Pillar deploy drift.** The Pi pillar ran Sthambha `3a4f7bf` (09-30), which drops typed endpoints, so the chain showed `endpoint: null`. It was redeployed to `c8d62c1` with `scripts/deploy-pi.sh`. Pillar deploys need to follow Sthambha main the way releases do.
- **Roster sync — FIXED (trisul `bf3256e3e`).** `roster-merge.py pull` runs on a 5-minute hub timer (`nakshatra-roster-pull.timer`). It merges valid VPS self-joins into `~/.nakshatra/admission-peers.tsv` (git wins; `self` is never taken from live), and `admission.py` prefers that file when `MESH_PEERS` is unset. It also unions the admitted keys into the Pi pillar's `peers.allow`.
- **Worker profile weight — FIXED (0.12.4).** The worker has its own hash-locked `worker` profile (grpcio/protobuf/numpy/gguf), and `inference` satisfies it as a superset. On bro the update installed in about 12 s with no compile, and `llama_cpp` is absent from the worker venv. ⚠️ Upgrade path: a node that ran `nak worker on` before 0.12.4 has both `worker.env` and `inference.env`, and the OLD installer refuses that combination. Fail-closed: the node stays on its old release. Fix: move `~/.nakshatra/inference.env` aside on a worker-only node, then update.
- **Pillar front — OPEN (2026-10-13).** It is public, limited to the four routes a worker uses (`POST /peer`, `GET /peers`, `GET /files`, `GET /health`; everything else returns 404). Limits: 2 r/s with a burst of 20 and 8 connections per IP. The Pi pillar refuses first-time registration from keys not in `peers.allow`; a signed stranger key was tested and got a 403. `STHAMBHA_TRUSTED_PROXIES=10.50.0.1/32` makes the pillar's own per-IP limits see the real client.
