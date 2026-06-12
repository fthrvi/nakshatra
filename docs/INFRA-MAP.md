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

## What is NOT auto-running (the honest gaps — all integration, not capability)

- ⬜ **No standing live mesh right now.** The transport stack is *proven* but not left running — each demo was brought up by hand (Pillar relay via bg-SSH that dies on disconnect; Vultr relay was transient). There is no always-on Nostr publisher + relay daemon yet.
- 🟡 **Nostr discovery → tunnel auto-bringup** not wired: listings carry `wg_pubkey`/`transport`/`relay_hint`/`drift_class`, but a peer doesn't yet auto-dial a tunnel on admission (config-driven today).
- 🟡 **O(t) productionization:** proven via the Forward-relay driver (client in the per-hop loop = the v1.1 §8.5 path). Folding it into `client.py`'s recovery branch as the default, and the worker-push variant (worker-side cache + catch-up RPC + proto field), are staged.
- ⬜ **Hole-punching** to skip the relay when a peer is directly reachable (optimization).
- ⬜ **Standing-relay hardening:** rendezvous-id allowlist + rate-limit (the spike relays were open).

## One-line summary
**The whole pipeline — discover → pin → drift-gate → encrypted relay tunnel → layer-split inference → O(t) recovery — is BUILT and each stage has been PROVEN on real hardware (this box + Mac4 + a public VPS).** What's left is *operations*: leaving a relay + publisher running, and auto-wiring discovery→tunnel so it forms without hand-holding.
