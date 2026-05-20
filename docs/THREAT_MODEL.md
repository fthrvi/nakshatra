# Nakshatra worker threat model

**Status:** Living document, updated as phases ship.
**Last updated:** 2026-05-20 (Phase A shipped — 8 cheap-win defensive limits).

This file is the worker-side companion to `sthambha/docs/THREAT_MODEL.md`.
Sthambha's threat model covers the pillar (control plane); this one covers
the worker (data plane). Together they describe what the four-project
stack defends against at L2 and L3.

The Sthambha threat model already has a **"Worker-side (Nakshatra)
hardening — DEFENDED as of Phase F + G"** section. That section documents
what the worker did about its **outbound** posture (pillar requests, TLS
SPKI pinning, sandbox introspection). This document documents the
worker's **inbound** posture — its gRPC and HTTP servers — which Phases F
and G did not address.

## Topology assumptions

Same three modes as Sthambha (see `sthambha/docs/THREAT_MODEL.md`
§Topology assumptions):

- **Mode A — all-LAN.** Worker gRPC + HTTP ports are reachable only on
  the trusted physical switch.
- **Mode B — mixed.** Some workers behind WireGuard; cross-site traffic
  encrypted at the network layer.
- **Mode C — public.** Worker ports could be reachable by anyone on the
  internet. **This is the threat surface this document targets.**

A worker without per-request auth on its inbound surface is acceptable in
Mode A and tolerable in Mode B (the WireGuard tunnel authenticates peers
at L3). It is **catastrophic in Mode C** — anyone reaching the port can
issue Forward calls, fetch model weights, or spawn slicer subprocesses.

## Worker surfaces

| Surface | Bound on | Today | Mode-C goal |
|---|---|---|---|
| gRPC `Info` | `--port` (default 5500) | Anonymous | ANONYMOUS (intentional — peer discovery) |
| gRPC `Forward` / `Inference` | `--port` | Anonymous | AUTHENTICATED against pillar-registered peers |
| HTTP `GET /healthz` | `--file-server-port` (default grpc+1000) | Anonymous, verbose | ANONYMOUS minimal (`{status}`); `/healthz/full` AUTHENTICATED |
| HTTP `GET /file/<basename>` | file-server-port | Anonymous, simple sanitize | AUTHENTICATED against pillar-registered peers |
| HTTP `POST /slice` | file-server-port | Anonymous, spawns subprocess | OPERATOR (signed by operator key installed at pillar) |
| HTTP `GET /slice/<task_id>` | file-server-port | Anonymous | AUTHENTICATED (caller knows task_id from POST response) |

## What's defended

### Outbound (worker → pillar) — DEFENDED as of Phase F + G (2026-05-19)

| Threat | Defense | Phase |
|---|---|---|
| Worker daemon registers/heartbeats unsigned (impersonation) | `nakshatra_auth.py` Ed25519 sign on every pillar call; TOFU registration; persistent key at `~/.nakshatra/keys/worker.ed25519` mode 600 | F (01b6097) |
| Worker trusts hardcoded pillar URL → DNS/ARP spoof | SPKI hash pinning via `STHAMBHA_PILLAR_SPKI_SHA256` env | F3 (01b6097) |
| Worker process runs unsandboxed → leaked key + co-tenant read | `nakshatra_sandbox.py` runtime introspection + `STHAMBHA_REFUSE_NONCOMPLIANT_SANDBOX` opt-in refuse | G (026485c) |
| Worker silently runs with broken attestation | WARN on `attestation_observed=false` from pillar response | J1 (carried from sthambha sprint) |
| Pillar's TLS cert acceptable as TLS 1.2 | TLS 1.3 default in `build_pillar_ssl_context`; `STHAMBHA_TLS_ALLOW_1_2` opt-in | I2 (carried) |
| Attestation blob shape changes silently in future | `attestation_version: 1` field set by `build_attestation_blob` | J2 (carried) |

### Inbound (gRPC + HTTP file server) — IN PROGRESS (Phase A onward)

These threats are the focus of the 2026-05-20 worker hardening sprint.
The table below will fill in as phases ship; rows tagged **PENDING**
identify audit gaps the plan addresses.

| Threat | Defense | Phase |
|---|---|---|
| gRPC `Forward` / `Inference` accepts unauthenticated calls | PENDING — gRPC Ed25519 auth + TLS, tier model | B |
| `Info` exposed publicly | INTENTIONAL — anonymous tier (peer discovery requires this) | B |
| HTTP `/file/<basename>` leaks model weights to anyone | PENDING — AUTHENTICATED tier against pillar-registered peers | C |
| HTTP `POST /slice` spawns subprocess for any caller | PENDING — OPERATOR tier (operator key required) | C |
| `Inference.chain[].address` lets attacker pivot worker to any gRPC endpoint (SSRF) | PENDING — accept only addresses for peers the pillar has registered | B |
| `_peer_streams` cache grows unbounded with attacker-supplied addresses | `OrderedDict` + LRU cap (`MAX_PEER_STREAMS=64`); oldest evicted with `_peer_evictions` counter | **A3** (2026-05-20) |
| `Inference` stream held open indefinitely by slow client → ThreadPoolExecutor exhaustion | `_iter_with_idle_timeout` wraps request iterator; `DEADLINE_EXCEEDED` after `INFERENCE_STREAM_IDLE_TIMEOUT_S=60s` | **A2** (2026-05-20) |
| gRPC message-size cap not set explicitly (default 4 MiB; should be intentional) | Explicit `WORKER_GRPC_MAX_MESSAGE_BYTES=16 MiB` set on both `max_receive_message_length` and `max_send_message_length` | **A1** (2026-05-20) |
| Slice subprocess can run for 3600s eating CPU; no concurrent-spawn cap | `MAX_CONCURRENT_SLICES=1` enforced at POST `/slice` (HTTP 429); timeout reduced to `SLICE_SUBPROCESS_TIMEOUT_S=1800s` | **A8** (2026-05-20) |
| `STHAMBHA_PILLAR_SPKI_SHA256="abc"` (too short) silently disables pinning | `validate_spki_hash_env` strict 64-hex-char check; `sys.exit` on malformed value | **A4** (2026-05-20) |
| Worker happily starts unsigned by default; only WARNs | `STHAMBHA_REFUSE_UNSIGNED=true` + pillar URL set + no key → exit 2 via `should_refuse_unsigned_startup` | **A5** (2026-05-20) |
| `recent_rpc_ms` can contain NaN/Inf if clock skews backwards | `safe_rpc_ms` (`math.isfinite` + negative reject) at both append (DaemonClient.call) and emit (heartbeat_loop) sites | **A6** (2026-05-20) |
| Pillar omits `model_sha256` from `/files`; worker fetches without verification | `should_refuse_unverified_fetch` filters candidates lacking `model_sha256`; default true; `STHAMBHA_REFUSE_UNVERIFIED_FETCH=false` opts out for Mode A/B | **A7** (2026-05-20) |
| `/slice` body `force_keep_token_embd = bool(value)` — lenient bool coercion (same shape as L4 on Sthambha side) | PENDING — `as_strict_bool` helper | D |
| `/slice` `full_gguf_path` accepts arbitrary paths (`/etc/passwd` probe via subprocess error messages) | PENDING — path sanitization + allowlist root | C |
| No audit log for slice spawns, register events, fetch attempts | PENDING — `~/.nakshatra/audit.jsonl` | D |
| Heartbeat thread blocks 5s on each `urlopen`; no backoff on repeated failures | PENDING — exponential backoff + jitter | D |
| `attestation_nonce_hex` from pillar accepted as `str(...)` — unbounded, non-hex | PENDING — strict hex validation, length cap | D |
| `_attestation_nonce` echoed back blindly in next signed envelope | PENDING — re-validate before send | D |
| `step.prefix_length` from gRPC accepted without bounds check | PENDING — `as_safe_int` helper at parse boundary | D |
| `step.next_server.address` / `step.chain[].address` unbounded length | PENDING — 256-byte cap (mirror Sthambha O3) | D |
| `/healthz` discloses GPU model, RAM, idem cache stats, push stats, latency averages → fingerprinting + latency side-channel | PENDING — tiered: `/healthz` minimal, `/healthz/full` AUTHENTICATED | C |

## What's NOT defended

These are intentional limits. Documented so future readers know the floor.

### Operator-layer (orchestration) responsibility

| Threat | Mitigation |
|---|---|
| Worker gRPC/HTTP ports reachable outside the trust zone | **Operator owns firewall.** Mode A: bind to LAN only; Mode B: bind to WireGuard interface; Mode C: bind to localhost + sandbox netns + operator reverse proxy. |
| Daemon subprocess shares worker process privileges | **By design.** Phase G's sandbox compliance check verifies that the surrounding OS sandbox provides isolation. Worker-side process isolation would require root + ptrace. |
| Operator misconfigures `--public-address` to expose worker outside the sandbox | **Documentation responsibility.** `docs/SANDBOX-EXAMPLES.md` on the Sthambha side covers reference templates. |

### Hardware / kernel-level (same as Sthambha)

| Threat | Why we can't defend at this layer |
|---|---|
| Compromised co-tenant escapes the OS sandbox (kernel CVE in runc / podman / namespace) | Out of scope. Same as for any container workload. **Mitigation: patched kernels + dedicated hosts for Mode C.** |
| Root-on-worker-host adversary reads `~/.nakshatra/keys/worker.ed25519` | Forward-compat seam: future `worker-auth` lift could mirror Sthambha's `Signer` protocol (I9) for HSM/Yubikey-backed worker keys. Not in this sprint. |
| GPU side-channels (timing / power / EM) | Research-grade. Public-cloud GPU inference doesn't defend either. **Mitigation: `multi_tenant_capability: "exclusive"` from the planner side.** |

### Worker self-reports compliance (carried from Sthambha I8)

Same caveat as Sthambha I8: the worker's sandbox compliance report is
self-attested. Phase G implements introspection (`/proc` + cgroup
reads); Phase I8 (carried) adds nonce-bound fingerprint signing. **An
attacker who patched the worker binary can still lie** about both the
compliance report and the fingerprint, because the Ed25519 key is in
process memory, not in a TPM. Hardware attestation (TPM-PCR-quote /
TDX / SEV-SNP) is the eventual close; the Phase I9 Signer protocol
shape is forward-compatible.

### Application-layer (same as Sthambha)

| Threat | Why this is the operator's call |
|---|---|
| Malicious worker returns garbage hidden states (cross-vendor drift, model-load mismatch, deliberate corruption) | Belongs in the offer/reward layer (chain side, paused). **Mitigation: sthambha's `chain_status='drifty'` flag (Phase 5 on the Sthambha side) + planner refusal of known-bad backends.** |
| Authenticated peer floods Inference RPCs after passing rate limit | The rate-limit window resets; operator evicts persistently spammy keys via `sthambha-cli peer evict <node_id>`. |

## Trust dependencies

The worker's security floor depends on these holding:

1. **The pillar is honest enough.** The worker trusts the pillar to provide
   correct `expected_sha` and well-formed `address` strings in `/files`
   responses. Phase A adds the `STHAMBHA_REFUSE_UNVERIFIED_FETCH` opt-in
   to harden this in Mode C; full pillar-byzantine resistance is out of
   scope (Mode C topology assumes a single pillar federation owned by the
   operator group).
2. **The operator's orchestration enforces the OS sandbox.** Phase G's
   compliance check is **observational** (read-only `/proc`); it does not
   enforce. The orchestration layer (Docker / Podman / systemd-nspawn /
   k8s) does. Same trust shape as Sthambha (operator owns orchestration).
3. **The `cryptography` library** does what it says (mirror of Sthambha
   trust dep 3).
4. **The worker host's clock** is reasonably accurate (NTP). The Phase F
   signed envelopes use `int(time.time())` and the pillar's 60s window
   depends on both sides agreeing on the time.

## How to contribute a defense

Same shape as Sthambha — see `sthambha/docs/THREAT_MODEL.md` §How to
contribute. Defenses for worker-side gaps land here; defenses for
pillar-side gaps land on the Sthambha doc; defenses that span both
(e.g., wire-contract changes) get a row in both, in the same commit.

If the contribution adds a runtime dep, write an ADR in
`~/trisul/decisions/` (the only worker-side dep today is `cryptography`,
inherited from ADR 0006). ADR for the worker-hardening sprint itself is
deferred until Phase B (when gRPC auth + TLS introduces architectural
shifts worth ratifying).

## History

- **Phase F (2026-05-19, sthambha sprint commit 01b6097)** — worker Ed25519 keypair, signed pillar requests, TOFU registration, SPKI pinning of the pillar's cert. Outbound auth complete.
- **Phase G (2026-05-19, sthambha sprint commit 026485c)** — `nakshatra_sandbox.py` introspects worker runtime + validates against `SandboxSpec`; worker reports compliance summary to pillar; planner Mode-C filter refuses non-compliant peers.
- **Phase I (2026-05-19, sthambha sprint commit d816290)** — TLS 1.3 default in worker's pillar context; soft attestation (nonce-bound runtime fingerprint).
- **Phase J (2026-05-19, sthambha sprint commit f81d6d1)** — worker WARN on `attestation_observed=false`; `attestation_version` field.

### Worker hardening sprint (2026-05-20 → ongoing)

Phases A-D are planned in `~/trisul/plans/2026-05-20-nakshatra-worker-hardening-sprint.md`.

- **Phase A (2026-05-20)** — 8 cheap-win defensive limits: explicit gRPC message-size cap (A1), `Inference` stream idle timeout (A2), `_peer_streams` LRU cap (A3), `STHAMBHA_PILLAR_SPKI_SHA256` strict validation (A4), `STHAMBHA_REFUSE_UNSIGNED` startup gate (A5), `safe_rpc_ms` NaN/Inf guard (A6), `STHAMBHA_REFUSE_UNVERIFIED_FETCH` (default true) on `fetch_sub_gguf_from_peer` (A7), `MAX_CONCURRENT_SLICES=1` + reduced subprocess timeout (A8). 30 new tests added; 66 total worker-side tests passing.

## Related

- `~/sthambha/docs/THREAT_MODEL.md` — pillar threat model (control plane).
- `~/sthambha/docs/four-project-architecture.md` — L1-L4 split.
- `~/sthambha/docs/SANDBOX-EXAMPLES.md` — operator orchestration templates.
- `~/trisul/decisions/0006-sthambha-request-auth.md` — Ed25519 wire contract (shared with pillar).
- `~/trisul/plans/2026-05-20-nakshatra-worker-hardening-sprint.md` — this sprint's plan.
- `~/trisul/sessions/2026-05-19.md` — Sthambha hardening sprint retro (the parent sprint).
