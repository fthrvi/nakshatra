# Nakshatra — Infrastructure Map (2026-06-11)

What every plane does, what's **proven on real hardware**, and where each piece lives.
Status legend:  ✅ proven live · 🟢 built + tested · 🟡 staged / config-driven · ⬜ not built

```
                         ╔════════════════════════════════════════════════════════╗
                         ║                A CLIENT WANTS TO GENERATE              ║
                         ║         (Prithvi's mind, or `scripts/client.py`)       ║
                         ╚════════════════════════════════════════════════════════╝
                                                  │
        ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
        │                                          │                                         │
        ▼                                          ▼                                         ▼
┌───────────────────┐               ┌──────────────────────────┐              ┌──────────────────────────┐
│  1. DISCOVERY      │              │   2. ADMISSION / TRUST    │              │   3. SOVEREIGN TRANSPORT │
│   (find peers)     │   listings   │   (who am I talking to?)  │   pinned     │   (reach across NATs)    │
│                    │ ───────────► │                           │ ───────────► │                          │
│ Nostr relays (NIP- │   signed     │ Ed25519 identity pin      │   identity   │ rendezvous relay (untrust│
│ 01, kind 30078)    │              │ (admission binds the      │              │ -ed) → mux tunnel →      │
│ ▸ nostr.py        ✅│              │  tunnel to THIS key)     🟢│              │ X25519+ChaCha20 channel  │
│ ▸ nakshatra_listing│              │ ▸ identity_handshake.py 🟢│              │ ▸ relay.py            ✅ │
│   (signed, drift_  🟢│              │ DRIFT-CLASS GATE:         │              │ ▸ mux_tunnel.py       ✅ │
│   class + compute) │              │  only same-engine-build   │              │ ▸ secure_channel.py   ✅ │
│ ▸ rank: measured-  🟢│              │  peers may chain/recover  │              │ ▸ tunnel_endpoint.py  ✅ │
│   compute + class  │              │ ▸ drift_gauge.py        ✅│              │                          │
└───────────────────┘               └──────────────────────────┘              └────────────┬─────────────┘
   "who's out there +                  "prove it's them, and that             relay can't READ (ChaCha20)
    are they MY build?"                 they won't diverge my stream"          or FORGE (Ed25519) the bytes
                                                                                            │
                                                                                            ▼
        ┌───────────────────────────────────────────────────────────────────────────────────────────┐
        │                            4. DATA PLANE  (the actual inference)                            │
        │                                gRPC over the secure tunnel                                  │
        │                                                                                             │
        │   client tokenizes ──► Worker A [layers 0..8) ──hidden state──► Worker B [layers 8..16)     │
        │     "capital of France"   embeddings + blocks      (~hidden_size      blocks + lm_head        │
        │                            (local weights)          floats/token)      (local weights)        │
        │                                  │                    on the wire           │                 │
        │                                  └──────────────── KV cache stays on each worker ───► logits  │
        │                                                                                  │            │
        │   Forward RPC: (hidden_in, n_tokens, keep_kv, start_pos) → hidden_out      client samples     │
        │   ▸ worker.py ✅   ▸ nakshatra_pb2 (wire v1) ✅   ▸ auto-provision from layer package ✅      │
        │                                                                              token → repeat   │
        └───────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼  (a worker drops mid-generation)
        ┌───────────────────────────────────────────────────────────────────────────────────────────┐
        │                       5. RECOVERY PLANE  (O(t) fault tolerance — fork A, DONE)              │
        │                                                                                             │
        │   client caches each activation it relays ──► on B's death, pick a SAME-DRIFT-CLASS B'      │
        │   (ActivationReplayCache, bounded)             ──► replay ONLY cached inputs to B' (rebuild  │
        │   ▸ activation_cache.py ✅                          its KV) ── survivor A is NEVER re-run    │
        │   ▸ drift_aware.py      ✅ (same-class gate)    ──► resume A(untouched)→B'                    │
        │   ▸ validate/ot_recovery.py ✅ PROVEN          Cost: O(t) on the ONE failed link,           │
        │                                                 not O(chain × T) full restart.               │
        │   PROOF: kill B after step 4 → B' caught up via 4 replays, A=8 forwards (never re-run),     │
        │          output BYTE-IDENTICAL to the no-failure baseline.                                   │
        └───────────────────────────────────────────────────────────────────────────────────────────┘
```

## Is it working right now? (2026-06-11 live checks)

| Plane | Module | Status | Evidence |
|------|--------|--------|----------|
| Discovery | `discovery/nostr.py` | ✅ live | NIP-01 secp256k1 sign + verify + tamper-reject round-trip |
| Discovery | `discovery/nakshatra_listing.py` | ✅ live | signed listing build→json→reparse→verify; tamper rejected |
| Discovery | rank by measured compute | ✅ live | faster decode-ms/layer ranks first over a candidate set |
| Admission | `identity_handshake.py` | 🟢 | 8 tests; ran live on Pillar + Vultr relays |
| Drift gauge | `drift_gauge.py` | ✅ | cross-machine finding: same build → identical fingerprint |
| Transport | relay / mux / secure_channel | ✅ proven | **§7 PASS**: two NAT'd machines, no shared VPN, byte-identical `' Paris'` over a public Vultr relay, encrypted |
| Data plane | `worker.py` + wire v1 | ✅ proven | §10 single-box AND cross-machine (this box + Mac4) |
| Recovery | activation_cache + drift_aware | ✅ proven | `ot_recovery.py` O(t) end-to-end, byte-identical |
| Recovery | drift_class config/registry wiring | 🟢 | this session's polish; 4 regression tests |

**Test suites green this session:** drift + cache `22 passed`; nostr + discovery + handshake + relay + mux + secure_channel + gauge `46 passed, 1 skipped`.

## Always-on capstone (2026-06-11) — the mesh now stands itself up

The two operational gaps are **CLOSED**. There is a long-running daemon + the
discovery→tunnel auto-bringup, installable as systemd `--user` services:

- 🟢 **`scripts/mesh/meshd.py`** — the node daemon: publish-heartbeat (re-sign +
  republish the signed listing every `--refresh`s) → discover (verify + pin +
  rank + **same-drift-class** + **heartbeat-TTL** staleness) → **auto-dial an
  Ed25519-pinned X25519+ChaCha20 tunnel** to each admitted peer. Writes a status
  file each loop.
- 🟢 **`scripts/mesh/pairing.py`** — coordination-free rendezvous id + role: both
  sides derive the same id and opposite client/server roles from the two pubkeys
  + who serves a worker (no negotiation round-trip).
- 🟢 **`deploy/systemd/{nakshatra-relay,nakshatra-meshd}.service`** + `install-mesh.sh`
  — `Restart=always`, enabled at login. `./deploy/install-mesh.sh status` shows it.
- ✅ **PROVEN twice on this box:**
  - `validate/mesh_capstone.py` — two in-process meshd nodes auto-form a tunnel; a
    real gRPC `Info` traverses it → `MESH_CAPSTONE_OK`.
  - `validate/mesh_join_standing.py` — a worker peer joins the **deployed**
    standing services; the running orchestrator auto-tunnels to it and carries
    real gRPC → `MESH_JOIN_OK`. Verified: enabled-at-login, crash→auto-restart,
    peer-leaves→tunnel-pruned (heartbeat TTL).

**Live status now:** `nakshatra-relay` + `nakshatra-meshd` active on this box; the
node publishes a signature-valid listing with its real gauge drift-class
(`prithvi-q8@gauge1:327418908285`). 24/7 across logout needs `sudo loginctl
enable-linger $USER` (one-time).

## What is still NOT auto-running (smaller, honest gaps)

- 🟡 **Discovery substrate is FileRelay** (a shared/local directory of signed
  listings) by default — zero-dep and always-on, but local. `meshd --nostr-relay
  wss://…` swaps in a real **public Nostr** relay (signed-listing schema is
  identical; needs `websocket-client`). The signing/verify/rank path is proven
  live either way.
- 🟡 **Reachability across NATs** still needs a public rendezvous relay (Vultr/Pi);
  the local standing relay binds `127.0.0.1`/`::`. The §7 cross-NAT run proved the
  remote-relay path — wiring meshd's `--rendezvous` at a standing public relay is
  config, not code.
- 🟡 **O(t) productionization:** proven via the Forward-relay driver. Folding it
  into `client.py`'s recovery branch as the default, and the worker-push variant
  (worker-side cache + catch-up RPC + proto field), are staged.
- ⬜ **Hole-punching** to skip the relay when a peer is directly reachable.
- ⬜ **Standing-relay hardening:** rendezvous-id allowlist + rate-limit; active
  liveness probe on idle tunnels (today a dead idle tunnel is reaped via the
  discovery TTL, not by sensing the broken pipe).
- ⬜ **Bidirectional auto-tunnel** per pair (today one directed tunnel/pair —
  enough for client-orchestrated chains; push-mode worker↔worker is the extension).

## One-line summary
**The whole pipeline — discover → pin → drift-gate → encrypted relay tunnel →
layer-split inference → O(t) recovery — is BUILT, PROVEN on real hardware, and now
STANDS ITSELF UP:** `meshd` + systemd services publish, discover, and auto-form
encrypted tunnels with no hand-holding (proven against the deployed services via
`mesh_join_standing.py`). What remains is reach/scale polish: a standing *public*
relay + Nostr publisher for cross-NAT strangers (the local always-on path is done).
