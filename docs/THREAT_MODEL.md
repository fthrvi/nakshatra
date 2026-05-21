# Nakshatra worker threat model

**Status:** Living document, updated as phases ship.
**Last updated:** 2026-05-20 (Phase D shipped — strict-type sweep + audit log + heartbeat backoff).

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
| gRPC `Forward` / `Inference` accepts unauthenticated calls | Ed25519 signature in `authorization` metadata; `verify_grpc_call` parses + verifies against `PillarPeerKeyResolver`-cached pubkeys; tier model: `Info` ANONYMOUS, `Forward`/`Inference` AUTHENTICATED. `NAKSHATRA_AUTH_REQUIRED` env: default true when `--pillar-url` set, false otherwise (legacy Mode A) | **B-auth** (2026-05-20) |
| `Info` exposed publicly | INTENTIONAL — `ANONYMOUS_GRPC_METHODS` includes only `/nakshatra.Nakshatra/Info` for peer discovery / capability negotiation | **B-auth** (2026-05-20) |
| HTTP `/file/<basename>` leaks model weights to anyone | `_check_tier(TIER_AUTHENTICATED)` requires Sthambha-Ed25519 signature whose keyid resolves through `PillarPeerKeyResolver`. Range requests + full fetches both gated. | **C2** (2026-05-20) |
| HTTP `POST /slice` spawns subprocess for any caller | `_check_tier(TIER_OPERATOR)` requires signature against operator pubkey at `~/.nakshatra/keys/operator.pub.hex` (independent of network auth — rotation is a filesystem operation). No operator pubkey installed → every POST refused. | **C2+C3** (2026-05-20) |
| `Inference.chain[].address` lets attacker pivot worker to any gRPC endpoint (SSRF) | `PillarPeerKeyResolver.is_registered_address` allowlist gates push targets. `NAKSHATRA_REFUSE_UNREGISTERED_PEERS=true` (default); refusals emit `push_failed:` so the client downgrades to client-relay. Stale cache (> 5min) returns False for every check — refuse beats push to attacker-supplied endpoint. | **B-ssrf** (2026-05-20) |
| TLS on the gRPC server (worker → worker / client → worker MITM defense) | `nakshatra_tls.ensure_cert` at boot generates self-signed RSA-2048 cert at `~/.nakshatra/tls/`; gRPC server uses `add_secure_port` when `NAKSHATRA_TLS_REQUIRED=true` (default when pillar configured). SPKI hash logged + audited at boot. Boot WARN on explicit-disable-with-pillar (Sthambha Phase O lesson — ship the WARN, don't silently fail-open). | **Phase 2** (2026-05-21) |
| Pillar `/peers` projection doesn't expose `peer_spki_hash` for inter-worker pinning | Pillar carries `peer_spki_hash` field on `PeerStatus`; `/peers` + new `GET /peers/<id>` projections expose it. Strict-hex parse at `POST /peer` (64 lowercase hex via `as_bounded_hex`, malformed → empty + WARN, prior value preserved). Worker `PillarPeerKeyResolver._spki_cache` + `expected_spki(address)` consume the projection. | **Phase 1** (2026-05-21, sthambha commits cf6323e + f8ff30f) |
| Outbound gRPC stream to a peer with the wrong cert (MITM substituting cert / operator-rotated cert without re-registering) | `_open_outbound_channel` TLS-probes the peer, compares observed SPKI against `expected_spki(address)`, refuses on mismatch. `NAKSHATRA_REFUSE_UNPINNED_PEERS=true` (default) refuses outbound to peers the pillar has no hash for. Stale resolver cache (> 5min) refuses every lookup. `spki_pin_mismatch` audit event carries expected + actual hashes; per-reason counters in `auth_stats` (`spki_unpinned_refusals`, `spki_mismatch_refusals`, `spki_probe_failures`). | **Phase 3** (2026-05-21) |
| `_peer_streams` cache grows unbounded with attacker-supplied addresses | `OrderedDict` + LRU cap (`MAX_PEER_STREAMS=64`); oldest evicted with `_peer_evictions` counter | **A3** (2026-05-20) |
| `Inference` stream held open indefinitely by slow client → ThreadPoolExecutor exhaustion | `_iter_with_idle_timeout` wraps request iterator; `DEADLINE_EXCEEDED` after `INFERENCE_STREAM_IDLE_TIMEOUT_S=60s` | **A2** (2026-05-20) |
| gRPC message-size cap not set explicitly (default 4 MiB; should be intentional) | Explicit `WORKER_GRPC_MAX_MESSAGE_BYTES=16 MiB` set on both `max_receive_message_length` and `max_send_message_length` | **A1** (2026-05-20) |
| Slice subprocess can run for 3600s eating CPU; no concurrent-spawn cap | `MAX_CONCURRENT_SLICES=1` enforced at POST `/slice` (HTTP 429); timeout reduced to `SLICE_SUBPROCESS_TIMEOUT_S=1800s` | **A8** (2026-05-20) |
| `STHAMBHA_PILLAR_SPKI_SHA256="abc"` (too short) silently disables pinning | `validate_spki_hash_env` strict 64-hex-char check; `sys.exit` on malformed value | **A4** (2026-05-20) |
| Worker happily starts unsigned by default; only WARNs | `STHAMBHA_REFUSE_UNSIGNED=true` + pillar URL set + no key → exit 2 via `should_refuse_unsigned_startup` | **A5** (2026-05-20) |
| `recent_rpc_ms` can contain NaN/Inf if clock skews backwards | `safe_rpc_ms` (`math.isfinite` + negative reject) at both append (DaemonClient.call) and emit (heartbeat_loop) sites | **A6** (2026-05-20) |
| Pillar omits `model_sha256` from `/files`; worker fetches without verification | `should_refuse_unverified_fetch` filters candidates lacking `model_sha256`; default true; `STHAMBHA_REFUSE_UNVERIFIED_FETCH=false` opts out for Mode A/B | **A7** (2026-05-20) |
| `/slice` body `force_keep_token_embd = bool(value)` — lenient bool coercion (same shape as L4 on Sthambha side) | `as_strict_bool` accepts only literal True/False; "false" string → False (default) | **D2** (2026-05-20) |
| `/slice` `full_gguf_path` accepts arbitrary paths (`/etc/passwd` probe via subprocess error messages) | `validate_slice_path` resolves symlinks, refuses paths outside `NAKSHATRA_SLICE_ROOT` (default = file-server dir), refuses NUL + dangerous Unicode (bidi/zero-width/Math-Alphanumeric/BOM/variation-selectors/C0/C1 ranges; mirror Sthambha L2/M3) | **C4** (2026-05-20) |
| No audit log for slice spawns, register events, fetch attempts | `nakshatra_audit.AuditLogger` at `~/.nakshatra/audit.jsonl`; events: `worker_started`, `slice_spawned`/`_completed`/`_failed`, `register_success`/`_failed`, `attestation_observed_false`, `fetch_started`/`_completed`/`_failed`, `auth_failure_grpc`, `auth_failure_http`. Size-bounded rotation at 256 MiB to `audit.jsonl.1`. | **D5** (2026-05-20) |
| Heartbeat thread blocks 5s on each `urlopen`; no backoff on repeated failures | Exponential backoff via `_next_heartbeat_interval`: 30s base, doubles per consecutive failure, cap 600s, ±25% jitter. Resets on success. | **D6** (2026-05-20) |
| `attestation_nonce_hex` from pillar accepted as `str(...)` — unbounded, non-hex | `as_bounded_hex(value, max_chars=64)` in `register_with_pillar`: non-hex / oversized / non-string → empty string (don't echo back) | **D4** (2026-05-20) |
| `_attestation_nonce` echoed back blindly in next signed envelope | Closed at ingest by D4 above — only validated-hex values are stored as `_attestation_nonce` | **D4** (2026-05-20) |
| `step.prefix_length` from gRPC accepted without bounds check | `as_safe_int(value, lo=0, hi=1<<20)` clamps prefix_length before passing to daemon | **D3** (2026-05-20) |
| `step.next_server.address` / `step.chain[].address` unbounded length | `as_bounded_str(addr, 256)` at parse layer; oversized → empty (push step elided) | **D3** (2026-05-20) |
| Auth-failure events flood the audit log under recon storm | `(event, ip, reason)` LRU dedup window (60s default, 8192 entries cap). Repeat failures suppressed in file but counted in `dedup_suppressed` stat. Mirror Sthambha L1. | **D7** (2026-05-20) |
| `/healthz` discloses GPU model, RAM, idem cache stats, push stats, latency averages → fingerprinting + latency side-channel | `/healthz` anonymous minimal body (`status`, `mode`, `layer_start/end`, `uptime_seconds`, `protocol_version`). Verbose body moves to `/healthz/full` behind AUTHENTICATED tier. | **C5** (2026-05-20) |

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
- **Phase B-auth (2026-05-20)** — gRPC Ed25519 verification + SSRF defense. New module `scripts/nakshatra_grpc_auth.py`: `verify_grpc_call`, `build_grpc_auth_header`, `parse_auth_header`, `resolve_auth_required`, `PillarPeerKeyResolver` (background-refreshed cache of pillar's `/peers` projection; stale-deadline 5min). `WorkerServicer` wires auth check on `Forward` (unary) + `Inference` first-frame (streaming); SSRF check on push to `chain[].address`. New envs: `NAKSHATRA_AUTH_REQUIRED` (default secure when pillar configured), `NAKSHATRA_REFUSE_UNREGISTERED_PEERS` (default true), `NAKSHATRA_PEER_REFRESH_INTERVAL` (default 60s). Canonical-string method tokens `POST` (unary) / `STREAM` (streaming) prevent HTTP↔gRPC and stream↔unary signature replay. Cross-repo wire-contract test confirms byte-identical signatures with `nakshatra_auth.sign_request`. 29 new tests added; 95 total worker-side tests passing.
- **Phase B-tls (deferred → DEFENDED in 2026-05-21 SPKI sprint Phase 2+3)** — TLS on the gRPC server + cross-worker SPKI pinning. Originally out of scope for the auth-core ship; landed in the SPKI federation sprint when Sthambha's `/peers` projection grew the `peer_spki_hash` field (Phase 1) and the worker grew the cert generation + outbound pin check (Phase 2 + 3).
- **Phase C (2026-05-20)** — HTTP file-server tier model + path sanitization + operator-key handling. Three tiers (`TIER_ANONYMOUS`, `TIER_AUTHENTICATED`, `TIER_OPERATOR`); `_check_tier` middleware on `FileServerHandler`; `/healthz` minimal/full split; `validate_slice_path` root-bounds + Unicode-safe; operator pubkey loaded from `~/.nakshatra/keys/operator.pub.hex`. 29 new tests added; 124 total worker-side tests passing.
- **Phase D (2026-05-20)** — strict-type sweep + audit log + heartbeat backoff. New modules: `scripts/nakshatra_validation.py` (`as_strict_bool`, `as_safe_int`, `as_safe_float`, `as_str_enum`, `as_bounded_hex`, `as_bounded_str`, `as_str_list` — mirror Sthambha N helpers), `scripts/nakshatra_audit.py` (append-only JSONL, 256 MiB rotation, `(event, ip, reason)` 60s dedup window). Wired at /slice body parse (D2), Inference step parse (D3), pillar response parse (D4). Audit events emitted at worker start, slice lifecycle, register, fetch, auth failures. Heartbeat exponential backoff 30s→600s with ±25% jitter (D6). 50 new tests added; 174 total worker-side tests passing.

### SPKI federation sprint (2026-05-21)

Closes the cross-worker MITM gap by adding TLS at the transport layer + SPKI pinning sourced from the pillar's `/peers` projection. Plan: `~/trisul/plans/2026-05-21-spki-federation-sprint.md`.

- **Phase 1 (sthambha, 2026-05-21, commits `cf6323e` + `f8ff30f` + `0ceece5`)** — `peer_spki_hash` field on `PeerStatus`; `as_bounded_hex(64)` parse layer at `POST /peer`; `/peers` + new `GET /peers/<id>` projections; cross-repo wire test asserts round-trip through register → persistence → projection.
- **Phase 2 (nakshatra, 2026-05-21, commits `3824b50` + `34b96a2` + `cb78da8`)** — `scripts/nakshatra_tls.py` mirrors sthambha/tls.py with worker-cert defaults at `~/.nakshatra/tls/`. `ensure_cert` is idempotent + refuses half-rotated state. Worker boot wraps `grpc.server.add_secure_port` (instead of `add_insecure_port`) when `NAKSHATRA_TLS_REQUIRED=true` (default when pillar configured). Boot WARN on explicit-disable-with-pillar. Worker declares `peer_spki_hash` on every `/peer` registration. `nakshatra-cli tls fingerprint` operator parity. 25 new tests; 199 worker-side tests passing.
- **Phase 3 (nakshatra, 2026-05-21, commits `6833bd2` + `84a13cd`)** — `PillarPeerKeyResolver` extended with `_spki_cache` + `expected_spki(address)` populated from the `/peers` projection. `_open_outbound_channel` TLS-probes the peer, compares SPKI, refuses on mismatch via `PinError`. `NAKSHATRA_REFUSE_UNPINNED_PEERS=true` (default) refuses outbound to peers the pillar has no hash for. `spki_pin_mismatch` audit event for operator forensics. Per-reason counters in `auth_stats`. 26 new tests; 225 worker-side tests passing.

## Operator-key install

The operator key gates `POST /slice` (the only route that spawns a
subprocess). It's intentionally **separate** from the network-auth
keypair: rotating the operator key is a filesystem operation, not a
network event, so a compromised pillar can't grant slicer access.

To install on a worker host:

```
mkdir -p ~/.nakshatra/keys
echo "<64-char-hex-pubkey>" > ~/.nakshatra/keys/operator.pub.hex
chmod 600 ~/.nakshatra/keys/operator.pub.hex
```

The hex value is the Ed25519 public key of whichever keypair the
operator uses to sign slice requests. Pair the worker-side public key
with the operator's private key on a separate, controlled machine.
Future `nakshatra-cli operator install` wraps this; today it's a
manual write.

Without an installed operator pubkey, every `POST /slice` is refused
with HTTP 403 — slicer functionality is opt-in.

## `NAKSHATRA_SLICE_ROOT`

`POST /slice` accepts a `full_gguf_path` field naming the source GGUF
to cut. `validate_slice_path` requires that path (after symlink
resolution) to live under the directory named by `NAKSHATRA_SLICE_ROOT`.
Default: the file-server's directory (parent of `--sub-gguf`). Operators
can set this explicitly to widen or narrow scope.

## Related

- `~/sthambha/docs/THREAT_MODEL.md` — pillar threat model (control plane).
- `~/sthambha/docs/four-project-architecture.md` — L1-L4 split.
- `~/sthambha/docs/SANDBOX-EXAMPLES.md` — operator orchestration templates.
- `~/trisul/decisions/0006-sthambha-request-auth.md` — Ed25519 wire contract (shared with pillar).
- `~/trisul/plans/2026-05-20-nakshatra-worker-hardening-sprint.md` — this sprint's plan.
- `~/trisul/sessions/2026-05-19.md` — Sthambha hardening sprint retro (the parent sprint).
