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
| Agent front door for any harness | `network/nak_mcp.py`: messaging/tasks plus local-gateway models/ask, signed balance, and redacted worker status | BUILT (consumer gateway branch) | any invited node |
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
| An **outsider's GPU** into your compute pool, at a trust tier | signed `nak invite --kind worker` → `release/install.py join` → node-key-signed onboard `/worker-redeem` → `nak worker build` + `nak worker on` (public pinned pillar front `:7778`) | **signed release** (the unsigned `/worker.sh` is retired) | `~/.nakshatra/keys/worker.ed25519` | **LIVE end to end 2026-10-13** (bro: invite → serve → idle unload → owner lease → warm pools) |
| Your own **machine** into your fleet | `fabric/join.py` → Sthambha `/join` | by hand | node key | WIRED |
| A **person**, as a contact | `release/install.py join` (one pasted line) | **signed release** | node key | LIVE |
| (engine for the outsider-GPU path) | `scripts/join/` six-phase decision engine | — | — | BUILT: decisions tested, observation layer stubbed. **Not dead**: it is the planned engine for the worker kind. |

**U4 (agreed direction): one invite, one installer, one key, several kinds.**
- One installer: the signed release, everywhere. It retires the unsigned worker tarballs. The worker stack moves into the release first (U6).
- One invite: the signed `nki1.` invite carries a `kind` (contact | worker). A worker invite also carries the one-time code from `onboard_server` (which stays the roster authority).
- **No `device` kind (Biswa, 2026-10-13).** Nakshatra grants inference/compute admission ONLY. Mesh membership is personal to each person and stays in that person's own mesh tooling (for Biswa: trisul `infra/onboarding` + `infra/wireguard`), never in a Nakshatra invite, release or key.
- One node key: `~/.nakshatra/keys/worker.ed25519`, replacing `~/.nakshatra-worker/`.
- Order: U6 (worker stack in the release) → the U4 worker kind. **U4 is complete.**

## Accounting
| Job | Component | Status |
|---|---|---|
| Meter inference: `gate` / `settle(receipt)` | `ledger_client.py` → Neuron ledger; signs every call with the node key (strict load, never creates it; `NAKSHATRA_LEDGER_KEYID` optional) | LIVE (`NAKSHATRA_CREDITS=1` in `~/.nakshatra/unconscious.env` → :8097). The hub remains fail-open. `NAKSHATRA_LEDGER_MODE=enforce` is the outsider-gateway policy: missing key, refusal, timeout, 5xx, or failed settlement denies; the client omits requester/account assertions and the ledger derives the operator from the roster. NOTE: it is ON for Prithvi's unconscious gateway; every settle so far charged 0 (own nodes). Since 0.15.4 (2026-10-13 21:14) the hub's gateway signs as `self`. Canary through the installed `LedgerHook`: the free own-node settle was recorded (`requester=self, charged=0, settled`), a replay added nothing, and `gate` answers `(True, '')`. The v2 server answers signed GETs for the signer and ignores the stale `?account=prithvi`. One race to know about: the roster-pull timer reads the SHARED trisul checkout, so a roster commit takes effect only after that checkout syncs. The hub's receipts carry no worker signatures yet (the gateway sets no `NAKSHATRA_ROSTER`), so only free own-node settles can be recorded. |
| Settle once per run; tier credit limits | `settlekey.py`, `creditlimit.py` | BUILT / WIRED |
| Task escrow: open / release / refund | `network/settle.py` (local ledger adapter) | LIVE (TEST units) |
| **The front door for both halves** | **`scripts/accounting.py`**: `metering()` (the gateway's credit hook) and `escrow(state_dir)` (task escrow). New backends (Solana devnet escrow, the repaired Neuron ledger) are added HERE. | LIVE (U5, 2026-10-03) |
| Ledger service | Neuron `python/ledger_server.py` :8097, **v2 since 2026-10-13**. It runs on an authenticated SQLite store (`~/.neuron/credits/ledger.sqlite`) through the user-unit drop-in `~/.config/systemd/user/neuron-ledger.service.d/10-v2.conf`; the legacy `ledger.json` and its `.bak.*` are untouched. Every POST/GET must be signed by a key in the admission roster, and the requester is derived from the roster row, never the body. Credit goes only to stages with valid rostered worker signatures; `receipt_id` comes from the work's content, so replays are idempotent; disputes are final; free own-node settles are recorded; amounts are bounded integers; the credit limit is checked inside the transaction; the journal is append-only. Live smoke: an unsigned settle/transfer gets 401 and `/balances` gets 403. Rollback: remove the drop-in, `daemon-reload`, restart. Backed up nightly (trisul `backup-sources.tsv`). | **HARDENED (TEST units only)** |
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
| Outsider consumer gateway (`nakshatra-gateway`) | Separate signed-release service, opt-in via `nak gateway on`; loopback `:11600` by default, SPKI-pinned/signed pillar discovery, signed fail-closed ledger, OpenAI `/v1` | BUILT (code/docs/tests; not deployed) |
| **The engine (patched llama.cpp)**: ONE source of truth | **`engine/`**: upstream b9992 + a patch series; `engine/source.sh` rebuilds it and refuses unless the tree hash matches exactly. Shipped in every release. Retired copies: the hub-only fork branch (now exported), the stale `experiments/v0.0` daemon (no `solo` mode), and `rebuild-worker.sh`, which would have reverted to it. | LIVE in releases since 0.10.3 (U6c); compiled from it on the hub |
| Build the worker for this GPU | `deploy/provision-worker.sh` (CUDA/ROCm/Vulkan/Metal/CPU, no sudo), building from `engine/` when run from a release | BUILT. The hub's live binary is still the one hand-built 09-12 from the identical tree; rebuild through this path at the next worker change. |
| Run a donated GPU worker from the signed release | `nak worker preflight` first checks GPU/CPU capacity, disk and runtime against the signed package catalog; then `nak worker build` and `nak worker on … --pillar-url https://… --pillar-spki <sha256>` create the opt-in `nakshatra-worker` service (`~/.nakshatra/worker.env`). `scripts/worker_service.py` binds the machine-local daemon to the current release's engine TREE before it execs `worker.py`. Plain HTTP is accepted only for loopback or `TRUSTED_PILLAR_NETS` (default `10.42.0.0/24`, the encrypted WireGuard mesh). | BUILT (U4 B1 + GPU preflight). Uses the one node key; every HTTPS pillar call is SPKI-pinned; no unsigned script copy and no `--skip-sha256`. |
| Verify model slices/packages before use | `scripts/worker.py` streams each peer-fetched slice through SHA-256 and requires the expected digest from the pillar's file index. `scripts/packaging/fetch_package.py` streams each fragment against its manifest digest; a public remote manifest must be signed by an operator-pinned `--package-trusted-pubkey`. `--skip-sha256` is only an explicit `--local-dev` escape hatch and refuses non-local/non-loopback sources. | WIRED (U4 B3); outsider provisioning never supplies `--skip-sha256`. |

### Courteous unload + owner lease — worker side BUILT (2026-10-13)
Measured on bro (RTX 5070) with a 60 s test timeout. After 60 s idle the worker went to `residency -> standby` and freed its VRAM (GPU 4890 → 2327 MiB), and the pillar projected `standby`. A hub prompt then printed `[chain] warming … (standby) … 2.11s`; the worker re-verified the slice SHA-256, respawned the daemon and went `loaded`, after which tokens ran at 8 ms/step. The default idle backstop is `IDLE_UNLOAD_S=600` (0 disables). Session release now schedules the faster `RELEASE_GRACE_S=30` unload (0 immediate; negative disables); a new step/session cancels it. A crashed client's KV session can no longer pin VRAM: it is released once stale (300 s) and the idle backstop has passed.

The opt-in `~/.nakshatra/owner-lease.env` adds the loopback-only sentinel endpoint described in `COURTEOUS-WORKER.md`. Its default-deny state is `away`; live permission moves it to `standby`, demand loads it to `loaded`, and a lapse cancels work, frees session ownership, unloads, and immediately heartbeats `away`. The residency vocabulary is therefore exactly `loaded | standby | away`; the worker never admits work while away, and coordinators must exclude away workers. The pillar ranks `loaded > unknown > standby` only as a tie-breaker after health, admission, coverage and tier. Remaining work: the Windows sentinel and demand-based warm pools in the planner.

Pillar `/peer` responses may also carry a short-lived warm-pool advisory: `keep` suppresses idle/release unload, `prewarm` digest-verifies and loads a standby worker through the normal single-flight path, and `release` unloads immediately when no session or RPC is active. Advisories never override an owner lease: stale/away workers do not prewarm, and a lease lapse unloads even under `keep`. They expire back to normal policy and may be disabled per worker with `nak worker on --warm-directives off` (`WARM_DIRECTIVES=0` in `worker.env`).

### Owner lease + release-on-done — step 2 LIVE (0.14.x, 2026-10-13)
Measured on bro (RTX 5070, WSL) with the Linux sentinel standing in for the owner:
- **Release on done.** 30 s after a request finished, the worker went to `standby` (VRAM 4892 → 2329 MiB). `NAKSHATRA_WORKER_RELEASE_GRACE_S`, default 30; the 600 s idle unload stays as the backstop.
- **Default deny.** With the lease on, the worker starts `away`. After 30 s of clear heartbeats it moves to `standby` and serves ("The Nile river flows through 11 countries…", 2.3 s warm).
- **Owner busy.** When the pause file appeared, the worker went `away` in about 0.5 s, mid-load, and freed VRAM (4912 → 2508 MiB). The pillar dropped it from `/chain`, and the client's `/peers` fallback now skips it with "(N away: their owners are using them)" (`50ff853`).
- **Owner done.** Back to `standby` after exactly the 30 s resume window.
- **Sentinel silent.** The worker went `away` after 6.7 s (TTL 6 s): silence means stop.
- **Windows sentinel.** Parsed and dry-run on blackwell's real Windows: `{clear:true}`. The GPU signal uses Windows' per-process GPU counters, because `nvidia-smi` reports `[N/A]` per process on consumer drivers. The WSL VM (`vmwp`) and the desktop (`dwm`) are ignored. It was INSTALLED on blackwell on 2026-10-13 with Biswa's ok: a per-user scheduled task `Nakshatra Owner Sentinel` (no execution limit, runs on battery), with files in `%LOCALAPPDATA%\nakshatra\owner-sentinel`. bro's lease is ON. Windows → WSL loopback forwarding reaches the 127.0.0.1-only endpoint (a wrong token gets 401). Two bugs were found live and fixed in 0.14.4. (1) Default task settings would have killed the sentinel after 72 h. (2) The GPU-counter sample (~1 s) plus a full 2 s sleep put every heartbeat in the stale band, which restarted the 30 s resume window forever; now only a lapse restarts it, and both sentinels keep their cadence. Proven: "Kathmandu is the capital of" → "Nepal…" through bro under the live Windows sentinel. The task now runs a local launcher that checks the signed-release copy in the configured WSL distro at startup and every six hours, atomically refreshes the local sentinel when its SHA-256 changes, and otherwise keeps supervising the existing local copy.
- **Hardening applied in review:** the lease token is compared as bytes, the handler times out after 5 s, and an abandoned KV session can no longer pin VRAM forever.

### Warm pools — step 3 LIVE (Sthambha `2a80590`, worker 0.15.x, 2026-10-13)
The pillar tracks per-model `/chain` demand over a 10-minute window (authenticated callers only; the public front does not expose `/chain`). It computes a Little's-law warm target, `ceil(rpm × 20 s / 60 × 1.5)` clamped to floor/cap (`~/.neuron/pillar/warm-pool.json`, default floor 0 and cap 4), and returns a short-lived `warm` directive (`keep`/`prewarm`/`release`, ≤120 s) in each worker's heartbeat response. It chooses already-loaded workers first, warms whole chains, and never directs an `away` worker. Operators can read `GET /warm`. Workers honour directives only under the owner lease (prewarm needs a live lease; a lapse always wins). Opt-out: `WARM_DIRECTIVES=0`. Measured on bro: one prompt gave llama32-3b `target 1, loaded 1`, and bro stayed loaded 70 s after the request (past the 30 s release grace) on renewed `keep`. A `release` now frees the daemon immediately when no step is in flight and no fresh session owns it.

### Outsider workers EARN (2026-10-13, client 0.15.5 + Neuron `ea81881`)
First real outsider payment: hub → bro (llama32-3b, streaming, `--receipt-out`). bro signed its stage, and the v2 ledger credited `op-686a9540` **+280** test units (10 tokens × 28 layers) and debited `self` −280. Both moves are in the append-only journal (earn/spend).
Three fixes made it work, all found live:
1. **client.** `SignParticipation` was sent unsigned, so authenticated workers answered UNAUTHENTICATED. It is now signed like Forward.
2. **ledger.** It matched signatures to chain stages by exact id. Real chains use a 14-hex short id, and a release worker's node id IS its pubkey (the roster name is a label). Matching is now same range plus an exact or ≥12-hex prefix id, with identity bound by node_id == the rostered pubkey.
3. **receipt id.** It now includes `run_id`: deterministic output made two genuine runs collide, so the second went unpaid. A replay under an invented run_id still credits nothing, because the signature binds the real run.
Signatures only exist for the streaming `Inference` path (by design; unary `Forward` carries no run id). The gateway already uses `--use-streaming --receipt-out`.
Known nit: a credited account is created with tier `stranger` regardless of its roster tier. Credits are unaffected, but the tier should come from the roster when the account is first created.

### Outsider CONSUMERS + Solana credits (2026-10-14)
- **Consumer gateway** (`nak gateway on|off|status`, release service `nakshatra-gateway`, loopback `127.0.0.1:11600/v1`, 0.15.6). An invited node can USE the network without serving. It talks to the network ONLY through the public VPS front: signed `GET /chain`/`/peers` and the signer-scoped `/ledger/*` paths (trisul `infra/control-plane/vps/`, hub user unit `nakshatra-ledger-tunnel`). It runs in ENFORCE mode: no output byte reaches the client before the ledger returns `{receipt_id, charged:int>=0}`; engine_url/stub/unknown-proxy paths are 403. The hub's own `nakshatra-unconscious` (Prithvi's backend) is byte-identical to before, proven by golden tests against `cd3ae8b`.
  - Proven live: a gateway on the public front only asked llama32-3b ("Kathmandu."), paid 168 credits, and bro earned them.
  - A node that is its own (only) worker routes that hop over loopback (`9fc07fe`); enforce-mode replies carry real `usage` counts.
- **Harnesses**: `docs/HARNESSES.md`. MCP `nak-mcp` (messaging, tasks, `nak_ask`, `nak_models`, `nak_balance`, `nak_worker_status`) + the loopback OpenAI endpoint. bro's OpenClaw already uses `nak-mcp`.
- **Consumer admission** (S2): `nak invite --kind consumer --tier known [--starter N]` → 7-column roster row (`… consumer`); routing/planner/registrar never summon it; it never enters the Pi `peers.allow`. One-time codes on the onboard server are now atomic (one lock, atomic saves).
- **Solana USDC credits** — design `docs/design/2026-10-14-solana-usdc-credits.md` (Biswa decided: plain SPL vault + indexer, RATE 1 USDC = 10^8 credits, overdrafts $0.01/$0.25/$2.50, devnet payout caps 5/50 USDC, 7-day hold, known+ to cash earnings). DEVNET ONLY; mainnet is Biswa's separate decision.
  - Live in the ledger (Neuron `400643b`): treasury account, deposit/withdraw tables, holds, zero-sum checked before every commit, reserved names, `DepositConflict`. Indexer `python/chain/indexer.py` (not installed): credits only after the per-account on-chain balance cross-check, genesis-pinned (devnet/loopback localnet; mainnet refused), https-only remote RPC, optional confirm RPC.
  - Real test: a local `solana-test-validator` with SPL programs preloaded from `~/.cache/nak-solana` (`NAK_SPL_PROGRAM_DIR`) — a finalized SPL transfer is indexed into the ledger.
  - Next slices: S4 payouts (`nak-treasury`), S5 routes + `nak wallet` + MCP `nak_wallet`, S6 reconcile, S7 devnet drill with bro.
- **Node container** (`deploy/container/`, `nakc`): non-root, no capabilities, read-only root, one state volume + optional read-only `/share`, gateway on host loopback only, only the p2p port public and only while the worker is on, supervisor runs only installer-signed units. Smoke-tested on the hub (offline). Protects the friend's files FROM the node; does not protect a compromised host.
- **Windows owner sentinel**: the launcher now updates itself (0.15.6, Launcher-Version, File.Replace, successor handoff under a Global per-user SID mutex). Installed on blackwell 2026-10-14; task restart 999/1 min.

### Consumers end to end — status 2026-10-14 03:20 (releases 0.15.7–0.15.9)
- **Proven with a real friend-style join:** `nak invite --kind consumer --tier stranger` → a fresh node CONTAINER (`nakc join`) installed the signed release over the public URL, made its keys, redeemed the consumer code (onboard server, atomic one-time codes) → roster row `op-3ac68dc17064d7ab stranger consumer` → hub roster pull → ledger roster; NOT in the Pi `peers.allow`, but in the new `readers.allow`.
- **Pillar readers** (Sthambha `28aadea`, deployed to the Pi): keys in `~/.neuron/pillar/readers.allow` may only `GET /peers`, `/chain`, `/files` (40 other route families 403). trisul roster-merge fills it from role=consumer rows.
- **Consumer tokenizer** (0.15.8): a consumer fetches only the package's signed `shared/metadata.gguf`; the package signer is pinned in the SIGNED RELEASE (`release/package-signers.json`), cache re-verified on every load, no redirects, size-capped.
- **Discovery** (0.15.8): signed pillar GETs carry a signed random nonce (two same-second requests used to collide → 409); auto-discovery retries with backoff and refreshes every 5 min; empty → 503 "no models discovered yet".
- **Gateway activation** (0.15.9): `nak gateway on` owns its `inference` profile opt-in (token-bound, removed only if it created it); in the container use `nakc gateway on …` (binds 0.0.0.0:11599 inside with --allow-lan; the host publish stays 127.0.0.1:11600).
- **Still to see:** the container consumer asking a paid question. Blocked only because bro (the only llama32-3b worker) is offline: WSL stops the idle `nakshatra-bro` distro once nothing holds it open, and the sentinel deliberately never boots it. Container `nakshatra-node` is stopped; its volume keeps the joined identity — `nakc start` + `nakc gateway on …` when bro is up.
- **Solana credits, all slices S1–S6 landed in Neuron** (`28952fb`), nothing on-chain switched on: S4 payouts sign IN-PROCESS with solders 0.23.0 (hash-pinned, offline venv `~/.cache/nak-solana/venv-chain`), validate every field of the message before signing, never rebuild before finality, kill-check before every write, singleton lock; S5 routes default OFF (`NAK_FUNDS_ROUTES`); S6 reconcile is read-only, point-in-time, full-lifecycle, signed with a pinned key, and fails closed on any error. **S7 devnet drill needs Biswa present** (one above-cap payout must ask his Face ID).

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
- **Worker profile weight — FIXED (0.12.4).** The worker has its own hash-locked `worker` profile (grpcio/protobuf/numpy/gguf), and `inference` satisfies it as a superset. On bro the update installed in about 12 s with no compile, and `llama_cpp` is absent from the worker venv. ⚠️ Upgrade path: a node that ran `nak worker on` before 0.12.4 has both `worker.env` and `inference.env`, and the OLD installer refuses that combination. Fail-closed: the node stays on its old release. Fix: move `~/.nakshatra/inference.env` aside on a worker-only node, then update. (Checked 2026-10-14: no node is affected — bro and the hub are on 0.15.x; current installers resolve the combination via `satisfied_by`.)
- **Pillar front — OPEN (2026-10-13).** It is public, limited to the four routes a worker uses (`POST /peer`, `GET /peers`, `GET /files`, `GET /health`; everything else returns 404). Limits: 2 r/s with a burst of 20 and 8 connections per IP. The Pi pillar refuses first-time registration from keys not in `peers.allow`; a signed stranger key was tested and got a 403. `STHAMBHA_TRUSTED_PROXIES=10.50.0.1/32` makes the pillar's own per-IP limits see the real client.
