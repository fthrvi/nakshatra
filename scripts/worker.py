#!/usr/bin/env python3
"""Nakshatra worker (M5) — gRPC server that wraps a llama-nakshatra-worker
daemon subprocess running our patched libllama.

The daemon owns the heavy state (model, KV cache); this Python process owns
the gRPC surface. Each Forward RPC is one round-trip to the daemon over
stdin/stdout.

CLI:
  --port           gRPC listen port
  --sub-gguf       path to the sub-GGUF this worker holds
  --mode           first | middle | last (matches the cluster config)
  --layer-start, --layer-end
                   declared by the cluster config; reported via Info
  --model-id       human-readable model id (matches cluster config)
  --daemon-bin     path to llama-nakshatra-worker binary
  --n-ctx          context length cap (default 256)
"""
import argparse
import collections
import hashlib
import hmac
import json
import math
import os
import platform
import plistlib
import queue
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from concurrent import futures
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import NamedTuple, Optional
from urllib import request as urlrequest, error as urlerror
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from worker_service import (
    DEFAULT_TRUSTED_PILLAR_NETS,
    WorkerServiceError,
    validate_pillar_transport,
)
from heartbeat import node_state

# Phase F (worker hardening sprint, 2026-05-19): Ed25519-signed pillar
# requests + optional TLS SPKI pinning. cryptography is a runtime dep
# pinned in setup.cfg to match Sthambha's range.
try:
    import nakshatra_auth as _wauth
    _AUTH_AVAILABLE = True
except ImportError as _e:
    _AUTH_AVAILABLE = False
    _AUTH_IMPORT_ERR = _e

# Phase G (sandbox enforcement, 2026-05-19): worker verifies its
# runtime matches the SandboxSpec from /join. Failure-soft import for
# the same reason as nakshatra_auth — the worker can start without
# the module, but Mode-C operators need it.
try:
    import nakshatra_sandbox as _wsandbox
    _SANDBOX_AVAILABLE = True
except ImportError as _e:
    _SANDBOX_AVAILABLE = False
    _SANDBOX_IMPORT_ERR = _e

# Phase B (worker hardening sprint, 2026-05-20): gRPC Ed25519 auth +
# SSRF defense via pillar's /peers projection. Failure-soft import so
# the worker can still start in legacy Mode A without the module.
try:
    import nakshatra_grpc_auth as _wgrpcauth
    _GRPC_AUTH_AVAILABLE = True
except ImportError as _e:
    _GRPC_AUTH_AVAILABLE = False
    _GRPC_AUTH_IMPORT_ERR = _e

try:
    import paid_admission as _paid
    _PAID_ADMISSION_AVAILABLE = True
except ImportError as _e:
    _PAID_ADMISSION_AVAILABLE = False
    _PAID_ADMISSION_IMPORT_ERR = _e

# The payment gate may install consumer identities, but their authority stays
# limited to inference. Control/lifecycle/signing RPCs remain peer-only.
_CONSUMER_INFERENCE_GRPC_METHODS = frozenset({
    "/nakshatra.Nakshatra/Forward",
    "/nakshatra.Nakshatra/Inference",
    "/nakshatra.Nakshatra/TruncateKV",
})
_INFO_ATTESTATION_DOMAIN = b"nakshatra-paid-info-v1\0"
_INFO_CHALLENGE_BYTES = 32


class AuthenticatedKeyId(str):
    """Verified key id retaining the resolver's least-authority role."""

    def __new__(cls, value: str, role: str):
        instance = super().__new__(cls, value)
        instance.role = role
        return instance


class IdempotencyConflict(RuntimeError):
    """The same principal/step id arrived with different request bytes."""

# Phase D (worker hardening sprint, 2026-05-20): strict-type helpers +
# audit log. Failure-soft so degraded bringup still works.
try:
    import nakshatra_validation as _wval
    _VALIDATION_AVAILABLE = True
except ImportError as _e:
    _VALIDATION_AVAILABLE = False
    _VALIDATION_IMPORT_ERR = _e

try:
    import nakshatra_audit as _waudit
    _AUDIT_AVAILABLE = True
except ImportError as _e:
    _AUDIT_AVAILABLE = False
    _AUDIT_IMPORT_ERR = _e

# 2026-05-21 SPKI federation Phase 2: self-signed TLS cert for the
# worker's gRPC server + SPKI declaration on /peer. Failure-soft import
# so a worker missing the module can still boot in plaintext Mode A
# (operator just sees a WARN; Phase 3 pinning refuses unpinned peers).
try:
    import nakshatra_tls as _wtls
    _TLS_AVAILABLE = True
except ImportError as _e:
    _TLS_AVAILABLE = False
    _TLS_IMPORT_ERR = _e

# 2026-05-29 fabric Phase D — network-fabric data plane. Failure-soft
# import so a worker missing the fabric package can still boot in gRPC
# mode (the default); only --transport=fabric needs these.
try:
    from fabric.join import JoinClient, NoPlanError, JoinError
    from fabric.backend import FabricBackend
    from fabric.transport import FabricLink
    _FABRIC_AVAILABLE = True
except ImportError as _fe:
    _FABRIC_AVAILABLE = False
    _FABRIC_IMPORT_ERR = _fe


def _audit(event: str, **payload):
    """Phase D5 — fire-and-forget audit log emission. No-op when the
    audit module isn't importable. Centralised here so callers don't
    have to guard each call site."""
    if _AUDIT_AVAILABLE:
        try:
            _waudit.audit(event, **payload)
        except Exception:
            pass  # never let audit failure crash the worker


class TlsBoot(NamedTuple):
    """Outcome of :func:`setup_tls`. Carries the four values the
    boot path downstream actually consumes:

    - ``required``: whether the gRPC server should bind a secure port
    - ``cert_path`` / ``key_path``: PEM paths for ``add_secure_port``
      (None when ``required`` is False)
    - ``spki_hash``: hex hash to declare on ``/peer`` registration
      (empty string when no cert was prepared)
    """
    required: bool
    cert_path: Optional[Path]
    key_path: Optional[Path]
    spki_hash: str


_TRUE_VALUES = frozenset(("true", "1", "yes"))
_FALSE_VALUES = frozenset(("false", "0", "no"))


def _release_service_running() -> bool:
    return os.environ.get("NAKSHATRA_RELEASE_SERVICE", "").strip().lower() in _TRUE_VALUES


def _security_required_without_module(env_name: str, pillar_url: str) -> bool:
    """Mirror the TLS/auth modules' policy when one cannot be imported.

    A configured pillar defaults both protections on, while an explicit false
    remains the legacy Mode-A escape hatch. Signed-release services are never
    allowed to take that escape hatch.
    """
    if _release_service_running():
        return True
    value = os.environ.get(env_name, "").strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return bool((pillar_url or "").strip())


def setup_tls(args) -> TlsBoot:
    """2026-05-26 drive-by — factor the worker boot's TLS init block
    into one place. Same behavior as the inline block it replaced
    (commit predating this one): resolve TLS-required policy, generate
    or load the cert via ``ensure_cert``, surface the SPKI hash for
    /peer registration, sys.exit on partial-cert state or generation
    failure, WARN-but-continue when TLS is explicitly disabled despite
    a configured pillar.

    Defined here at module scope (not nested in main) so unit tests can
    construct a fake ``args`` namespace and call it directly without
    invoking the rest of the boot pipeline."""
    if not _TLS_AVAILABLE:
        if _security_required_without_module(
            "NAKSHATRA_TLS_REQUIRED", args.pillar_url
        ):
            sys.exit(
                "[worker] TLS bringup refused: TLS is required but "
                f"nakshatra_tls could not be imported: {_TLS_IMPORT_ERR}"
            )
        print(f"[worker] WARN: nakshatra_tls import failed "
              f"({_TLS_IMPORT_ERR}); gRPC server will be plaintext",
              flush=True)
        return TlsBoot(False, None, None, "")

    required = (
        _release_service_running()
        or _wtls.resolve_tls_required(
            os.environ.get("NAKSHATRA_TLS_REQUIRED"), args.pillar_url
        )
    )
    if required:
        try:
            cert_path, key_path, spki_hash = _wtls.ensure_cert()
            print(f"[worker] TLS cert: {cert_path} "
                  f"(spki_sha256={spki_hash})", flush=True)
            _audit("tls_cert_ready",
                   cert_path=str(cert_path),
                   spki_sha256=spki_hash)
            return TlsBoot(True, cert_path, key_path, spki_hash)
        except FileExistsError as e:
            # Partial cert state — surface explicitly rather than
            # rotating one side and breaking pinned peers.
            sys.exit(f"[worker] TLS bringup refused: {e}")
        except Exception as e:
            sys.exit(f"[worker] TLS bringup failed: {e}")
    if args.pillar_url:
        # Pillar configured but TLS explicitly disabled — Mode-A
        # bringup. The WARN matches Sthambha's escape-hatch shape;
        # operators see it once at boot, not buried in audit.
        print(f"[worker] WARN: NAKSHATRA_TLS_REQUIRED is off but a "
              f"pillar is configured. gRPC server will speak plaintext; "
              f"peers that pin this worker's SPKI will refuse outbound "
              f"connections (Phase 3 pin check).",
              flush=True)
        _audit("tls_disabled_with_pillar",
               pillar_url=pillar_location_for_log(args.pillar_url))
    return TlsBoot(False, None, None, "")


def resolve_worker_auth_required(args) -> bool:
    """Resolve inbound gRPC auth policy, refusing a missing implementation.

    Import failures are tolerated only for a genuinely unauthenticated legacy
    Mode-A worker. A required or signed-release worker must never silently
    expose its inference surface without request authentication.
    """
    if not _GRPC_AUTH_AVAILABLE:
        if _security_required_without_module(
            "NAKSHATRA_AUTH_REQUIRED", args.pillar_url
        ):
            sys.exit(
                "[worker] gRPC auth bringup refused: authentication is required "
                "but nakshatra_grpc_auth could not be imported: "
                f"{_GRPC_AUTH_IMPORT_ERR}"
            )
        return False
    return (
        _release_service_running()
        or _wgrpcauth.resolve_auth_required(
            os.environ.get("NAKSHATRA_AUTH_REQUIRED"), args.pillar_url
        )
    )


# Convention: fabric UDP port = gRPC port + this offset, unless the
# operator overrides via --fabric-port. Keeps the data plane off the
# control port without a separate discovery step in the prototype.
# Phase F may replace this with an explicit fabric endpoint in the
# capability declaration.
FABRIC_PORT_OFFSET = 100


def _neighbor_fabric_addr(neighbor_address: str) -> tuple[str, int]:
    """Derive a neighbor's fabric UDP (host, port) from the gRPC
    ``host:port`` address the pillar served in the /join neighbor
    block. Prototype convention — fabric port = gRPC port +
    FABRIC_PORT_OFFSET. Raises ValueError on a malformed address so
    the caller refuses boot rather than dialing a bogus endpoint."""
    host, _, port_s = neighbor_address.rpartition(":")
    if not host or not port_s:
        raise ValueError(
            f"neighbor address {neighbor_address!r} not host:port")
    return host, int(port_s) + FABRIC_PORT_OFFSET


def setup_fabric(args, forward_fn, n_embd, *, worker_priv, node_id,
                 pillar_spki_hash=None):
    """2026-05-29 fabric Phase D — bring up the network-fabric data
    plane for ``--transport=fabric``. Returns ``(backend, join_client)``
    with the FabricBackend's links wired and ready to ``serve()``, or
    ``sys.exit``s on a fatal prerequisite (no fabric module, no pillar,
    no plan covering this peer).

    Refuse-boot policy (sprint open question 2): ``--transport=fabric``
    is an explicit opt-in; if the prerequisites aren't met we exit loud
    rather than silently degrading to gRPC. The operator who passed the
    flag meant it.

    Defined at module scope (not nested in main) so it's unit-testable
    with a fake ``forward_fn`` + a mocked JoinClient, mirroring how
    :func:`setup_tls` is tested."""
    if not _FABRIC_AVAILABLE:
        sys.exit(
            f"[worker] --transport=fabric but the fabric package failed "
            f"to import ({_FABRIC_IMPORT_ERR}); cannot start.")
    if not args.pillar_url:
        sys.exit("[worker] --transport=fabric requires --pillar-url "
                 "(the fabric data plane is keyed off the pillar's /join).")
    if not worker_priv or not node_id:
        sys.exit("[worker] --transport=fabric requires a worker Ed25519 "
                 "key + node_id to sign the OWNER-tier /join request.")

    local_port = args.fabric_port or (args.port + FABRIC_PORT_OFFSET)
    capability = {
        "node_id": node_id,
        "address": args.public_address or f"0.0.0.0:{args.port}",
        "public_key_hex": _wauth.public_key_hex_from_private(worker_priv)
        if _AUTH_AVAILABLE else "",
    }
    jc = JoinClient(
        args.pillar_url, node_id, worker_priv, capability,
        pillar_spki_hash=pillar_spki_hash,
    )
    try:
        resp = jc.join()
    except NoPlanError as e:
        sys.exit(
            f"[worker] --transport=fabric refused: no chain plan covers "
            f"this peer yet ({e}). Create + execute a plan via "
            f"sthambha-cli first, then start the fabric worker.")
    except JoinError as e:
        sys.exit(f"[worker] /join failed: {e}")

    keyring = jc.keyring()
    backend = FabricBackend(forward_fn, args.mode, n_embd)

    # Bind one local UDP socket and reuse it for every directional
    # link — UDP is connectionless, so one socket serves all peers.
    import socket as _socket
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", local_port))

    def _link_for(nb):
        if nb is None or not nb.peer_id:
            return None
        key = keyring.get((node_id, nb.peer_id))
        if key is None:
            return None  # no key → can't pin; neighbor stays unwired
        peer_addr = _neighbor_fabric_addr(nb.address)
        return FabricLink(sock, peer_addr, key, _chain_id_from(resp.plan_id))

    # backward neighbor is the inbound link (we receive FORWARD from
    # the previous worker); forward neighbor is where we send onward.
    inbound_link = _link_for(resp.backward)
    forward_link = _link_for(resp.forward)

    # 2026-05-29 fabric Phase F — feedback link wiring.
    # 2-worker-chain shortcut: the feedback wormhole runs between
    # last → first, which is the SAME peer-pair the existing forward
    # / inbound links already address. Reuse them:
    #   - mode=first: feedback comes back FROM the last worker, who
    #     is this worker's forward neighbor. forward_link already
    #     has peer_addr=last + the right per-pair key.
    #   - mode=last: feedback ships TO the first worker, who is this
    #     worker's backward neighbor. inbound_link already has
    #     peer_addr=first + the right key.
    # Chains > 2 workers need a dedicated first↔last keying path
    # (the pillar's keyring is per-chain-edge only); v0 prototype
    # explicitly targets 2-worker chains and the localhost smoke
    # exercises exactly that shape.
    feedback_link = None
    if args.mode == "first":
        feedback_link = forward_link
    elif args.mode == "last":
        feedback_link = inbound_link

    backend.set_links(
        inbound=inbound_link,
        forward=forward_link,
        feedback=feedback_link,
    )
    print(f"[worker] fabric data plane up: plan={resp.plan_id} "
          f"local_udp=:{local_port} mode={args.mode} "
          f"inbound={'yes' if backend.inbound_link else 'no'} "
          f"forward={'yes' if backend.forward_link else 'no'}",
          flush=True)
    _audit("fabric_data_plane_up",
           node_id=node_id, plan_id=resp.plan_id, local_udp_port=local_port)
    return backend, jc


def _chain_id_from(plan_id: str) -> int:
    """Map the pillar's string plan_id to the u64 chain_id the packet
    schema carries. Prototype: stable hash of the plan_id into 64 bits.
    Phase F may switch to a pillar-assigned numeric chain_id if the
    plan store grows one; for now both sides of a link derive the same
    value from the same plan_id string deterministically."""
    import hashlib
    h = hashlib.blake2b(plan_id.encode(), digest_size=8).digest()
    return int.from_bytes(h, "little")


CMD_TOKEN_DECODE = 1
CMD_EMBD_DECODE  = 2
CMD_INFO         = 3
CMD_KV_TRUNCATE  = 4   # speculative decode: discard a rejected tail (payload = u32 n_keep)
CMD_EAGLE_HIDDEN = 5   # EAGLE→live: return float32[n_tokens,3,n_embd] draft hidden states
CMD_SLEEP        = 6   # sleep-mode: release the GPU (free ctx+model), keep process resident
CMD_WAKE         = 7   # sleep-mode: re-acquire the GPU (reload from warm page cache)
# #16: int8 hidden-state wire codec. Default OFF. When "int8", non-last workers quantize the
# hidden state they emit and every worker dequantizes a hidden state it receives — ~4× less WAN
# bytes/hop. All workers + the client must agree (set the same env). The daemon always sees f32.
_ACT_QUANT = os.environ.get("NAKSHATRA_ACT_QUANT", "").strip().lower() == "int8"


class ForwardResult(NamedTuple):
    """2026-05-29 — result of the shared daemon decode path used by both
    the gRPC ``Forward`` RPC and the fabric backend (Phase D). Carries
    the transport-neutral outcome; each caller maps it to its own
    response type (gRPC status codes vs fabric drop counters).

    - ``ok`` True: ``payload`` holds the stripped hidden_state OR int32
      token-id bytes (rtype prefix already removed).
    - ``ok`` False: ``error`` is operator-readable; ``client_error``
      distinguishes a bad-input case (gRPC INVALID_ARGUMENT) from a
      daemon failure (gRPC INTERNAL)."""
    ok: bool
    payload: bytes
    error: str
    client_error: bool


# ── Phase A (worker hardening sprint, 2026-05-20) ─────────────────────
# Defensive limits + bounded state. Mirror of Sthambha A1-A8 on the
# worker side. See ~/trisul/plans/2026-05-20-nakshatra-worker-hardening-
# sprint.md for the design context.

WORKER_GRPC_MAX_MESSAGE_BYTES = 256 * 1024 * 1024     # A1: explicit cap — 256MB covers a full
# 2048-ctx prefill hidden-state (n_tok × hidden × 4B; e.g. 2048×5120×4 ≈ 42MB). The 16MB prior cap
# (and the 4MB gRPC default on the CLIENT channels) was dropping any real prompt (RESOURCE_EXHAUSTED).
INFERENCE_STREAM_IDLE_TIMEOUT_S = 60.0                # A2: bound slow clients
MAX_PEER_STREAMS = 64                                 # A3: LRU cap
SPKI_HASH_LENGTH = 64                                 # A4: sha256 hex
MAX_CONCURRENT_SLICES = 1                             # A8: cap subprocess fan-out
SLICE_SUBPROCESS_TIMEOUT_S = 1800                     # A8: was 3600
# worker/kv-session-ownership-gate (2026-09-12): if a session's owner never
# calls release_session() (worker thread killed, box crashed mid-stream), the
# daemon must not stay locked to it forever — that would turn one bad session
# into a permanent denial-of-service for every session after it. A stale
# owner (idle longer than this) is treated as abandoned and reclaimed.
KV_SESSION_STALE_AFTER_S = 300.0


def validate_spki_hash_env(value: Optional[str]) -> Optional[str]:
    """Phase A4 — validate the STHAMBHA_PILLAR_SPKI_SHA256 env value.

    Catches the silent-disable-by-typo case: an operator sets the env
    to "abc" and pinning silently turns off because the empty-string
    coalesce treats it as set-but-invalid. Returns the canonicalised
    lowercase hash, or None if unset. Raises ValueError if set but
    malformed — startup should refuse rather than proceed with broken
    pinning.
    """
    s = (value or "").strip().lower()
    if not s:
        return None
    if len(s) != SPKI_HASH_LENGTH:
        raise ValueError(
            f"STHAMBHA_PILLAR_SPKI_SHA256 must be {SPKI_HASH_LENGTH} hex "
            f"chars (sha256); got {len(s)} chars"
        )
    try:
        bytes.fromhex(s)
    except ValueError:
        raise ValueError(
            f"STHAMBHA_PILLAR_SPKI_SHA256 must be hex; got non-hex chars"
        )
    return s


def should_refuse_unsigned_startup(
    refuse_env: Optional[str],
    auth_available: bool,
    has_worker_key: bool,
    pillar_url: str,
) -> bool:
    """Phase A5 — STHAMBHA_REFUSE_UNSIGNED gate.

    Returns True when the operator opted into refuse-unsigned AND the
    worker would in fact send unsigned requests to a pillar. The caller
    is expected to sys.exit(2) on True. Mode A/B operators can leave
    the env unset (default false) for legacy unsigned bringup.
    """
    val = (refuse_env or "").strip().lower()
    if val not in ("true", "1", "yes"):
        return False
    if not pillar_url:
        return False  # no pillar = nothing to sign
    return (not auth_available) or (not has_worker_key)


def safe_rpc_ms(ms: float) -> Optional[float]:
    """Phase A6 — NaN/Inf guard on recent_rpc_ms.

    DaemonClient.recent_rpc_ms is sourced from (time.time() - t0) * 1000.
    A backward clock jump or corrupted t0 can produce non-finite or
    negative values; the heartbeat shouldn't ship those to the pillar.
    Mirror of Sthambha O5 `as_safe_float`.
    """
    try:
        m = float(ms)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(m):
        return None
    if m < 0:
        return None
    return m


def should_refuse_unverified_fetch(
    refuse_env: Optional[str], expected_sha: Optional[str]
) -> bool:
    """Phase A7 / U4 B3 — refuse a missing or malformed expected SHA-256.

    ``refuse_env`` is retained in the signature so older callers/tests fail
    safely, but it can no longer disable verification.  Every remotely fetched
    peer slice must have a complete SHA-256 attested by the pillar.
    """
    del refuse_env
    value = (expected_sha or "").strip()
    if len(value) != 64:
        return True
    try:
        bytes.fromhex(value)
    except ValueError:
        return True
    return False


def validate_local_dev_fetch_policy(*, local_dev: bool, skip_sha256: bool,
                                    package_url: str, auto_fetch: bool) -> None:
    """Validate the narrowly-scoped local-development escape hatch.

    Skipping the worker's local registration hash is never a production mode.
    It requires ``--local-dev``; that mode may only consume an existing local
    slice or a filesystem/loopback package and cannot use pillar peer-fetch.
    """
    if skip_sha256 and not local_dev:
        raise ValueError("--skip-sha256 requires explicit --local-dev")
    if not local_dev:
        return
    from packaging.fetch_package import is_local_package_location

    if auto_fetch:
        raise ValueError(
            "--local-dev refuses --auto-fetch because peer slices are remote")
    if package_url and not is_local_package_location(package_url):
        raise ValueError(
            "--local-dev package source must be a local path, file:// URI, "
            "localhost, or a literal loopback address")


def _iter_with_idle_timeout(it, idle_seconds: float):
    """Phase A2 — wrap a blocking iterator with a per-step idle deadline.

    Each next() yields the iterator's next item, or raises TimeoutError
    when no item arrives in idle_seconds. Used to bound how long a slow
    or malicious gRPC client can hold a server thread.

    Implementation: a pump thread reads the iterator and pushes onto a
    queue; the caller blocks on queue.get(timeout). Sentinel marks normal
    end; error tuple ("__error__", exc) propagates exceptions.
    """
    q: queue.Queue = queue.Queue()
    sentinel = object()
    err_tag = "__error__"

    def pump():
        try:
            for item in it:
                q.put(item)
        except Exception as e:
            q.put((err_tag, e))
            return
        q.put(sentinel)

    threading.Thread(target=pump, daemon=True).start()

    while True:
        try:
            item = q.get(timeout=idle_seconds)
        except queue.Empty:
            raise TimeoutError(
                f"no step received in {idle_seconds}s (idle timeout)"
            )
        if item is sentinel:
            return
        if isinstance(item, tuple) and len(item) == 2 and item[0] == err_tag:
            raise item[1]
        yield item


def _running_slice_count() -> int:
    """Phase A8 — number of slice tasks currently in 'running' status.

    Caller must NOT hold _SLICE_LOCK; this function acquires it.
    """
    with _SLICE_LOCK:
        return sum(
            1 for t in _SLICE_TASKS.values() if t.get("status") == "running"
        )


# ── Phase C (worker hardening sprint, 2026-05-20) ─────────────────────
# HTTP tier model + path sanitization + operator key.
#
# Three tiers (mirror Sthambha's C tier model):
#
#   ANONYMOUS    — discoverable without auth; minimal body (peer
#                  discovery / liveness only).
#   AUTHENTICATED — Sthambha-Ed25519 signature whose keyid resolves
#                  through the pillar's /peers projection.
#   OPERATOR     — signature against a separately-installed operator
#                  pubkey at OPERATOR_PUBKEY_PATH. Independent of the
#                  pillar's TOFU lock; rotating an operator key is a
#                  filesystem operation, not a network one.

TIER_ANONYMOUS = "anonymous"
TIER_AUTHENTICATED = "authenticated"
TIER_OPERATOR = "operator"

OPERATOR_PUBKEY_PATH = Path.home() / ".nakshatra" / "keys" / "operator.pub.hex"

# Phase C4 — dangerous Unicode ranges to refuse in `full_gguf_path`.
# Mirrors Sthambha L2/M3 denylists. UTF-8 letters/digits/punctuation in
# any script are accepted; ranges below are abuse-only.
_DANGEROUS_UNICODE_RANGES = [
    (0x0000, 0x001F),  # C0 controls (TAB allowed separately)
    (0x007F, 0x009F),  # DEL + C1 controls
    (0x200B, 0x200F),  # zero-width / bidi controls
    (0x202A, 0x202E),  # bidi LRE/RLE/PDF/LRO/RLO
    (0x2060, 0x206F),  # word joiner, invisible separators
    (0xFE00, 0xFE0F),  # variation selectors
    (0xFEFF, 0xFEFF),  # BOM
    (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols (L2-bypass risk)
    (0xE0000, 0xE007F),  # tag characters
    (0xE0100, 0xE01EF),  # variation selectors supplement
]


def _load_operator_pubkey(
    path: Path = OPERATOR_PUBKEY_PATH,
) -> Optional[str]:
    """Phase C3 — read the operator pubkey hex from disk.

    Returns the canonicalised lowercase hex (64 chars), or None if the
    file is missing / unreadable / malformed. Operators install by
    writing the 64-character hex to OPERATOR_PUBKEY_PATH with mode 600.
    A future ``nakshatra-cli operator install`` will wrap this.
    """
    try:
        if not path.is_file():
            return None
        raw = path.read_text().strip().lower()
    except OSError:
        return None
    if len(raw) != 64:
        return None
    try:
        bytes.fromhex(raw)
    except ValueError:
        return None
    return raw


def _has_dangerous_unicode(s: str) -> bool:
    """Phase C4 — True if ``s`` contains a codepoint in
    `_DANGEROUS_UNICODE_RANGES` or a NUL byte."""
    if "\x00" in s:
        return True
    for ch in s:
        cp = ord(ch)
        for lo, hi in _DANGEROUS_UNICODE_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def validate_slice_path(raw_path: str, root: Path) -> Path:
    """Phase C4 — sanitize an operator-supplied ``full_gguf_path``.

    Returns the resolved Path on success; raises ValueError on:
      - empty path
      - NUL byte or dangerous Unicode
      - relative path that doesn't resolve under ``root``
      - symlink escape past ``root``
      - non-existent / non-file target

    ``root`` is the filesystem root operators have declared via
    NAKSHATRA_SLICE_ROOT (defaults to the file-server's directory).
    """
    if not raw_path or not raw_path.strip():
        raise ValueError("full_gguf_path must not be empty")
    if _has_dangerous_unicode(raw_path):
        raise ValueError("full_gguf_path contains a dangerous Unicode codepoint")
    try:
        resolved = Path(raw_path).resolve()
        root_resolved = root.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"path resolve failed: {e}")
    # Path.is_relative_to (3.9+) — root-bound check after symlink resolution.
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(
            f"full_gguf_path must resolve under {root_resolved!r}; "
            f"got {resolved!r}"
        )
    if not resolved.is_file():
        raise ValueError(f"full_gguf_path is not a regular file: {resolved!r}")
    return resolved


# ── HTTP auth check helpers ──────────────────────────────────────────


def _parse_http_auth_header(header: str) -> tuple[str, str, int]:
    """Parse the Sthambha-Ed25519 HTTP Authorization header.

    Symmetric with `nakshatra_grpc_auth.parse_auth_header`; we keep a
    separate copy here to avoid a hard dependency from worker.py's HTTP
    surface on the gRPC module (failure-soft imports keep the file-
    server runnable even when the gRPC auth deps are absent)."""
    if not header:
        raise ValueError("missing authorization header")
    import re as _re
    m = _re.fullmatch(
        r'\s*Sthambha-Ed25519\s+keyid="([^"]*)",sig="([^"]*)",ts="([^"]*)"\s*',
        header,
    )
    if not m:
        raise ValueError("malformed authorization header")
    keyid, sig_b64, ts_str = m.groups()
    if not keyid:
        raise ValueError("empty keyid")
    if not sig_b64:
        raise ValueError("empty signature")
    try:
        ts = int(ts_str)
    except (TypeError, ValueError):
        raise ValueError("timestamp not an integer")
    return keyid, sig_b64, ts


def verify_http_request(
    auth_header: str,
    method: str, path: str, body: bytes,
    *,
    operator_pubkey: Optional[str],
    peer_resolver,  # PillarPeerKeyResolver or None
    tier: str,
    now_seconds: Optional[int] = None,
    window_s: float = 60.0,
) -> Optional[str]:
    """Phase C — verify an HTTP request's Sthambha-Ed25519 signature for
    the given tier. Returns the verified keyid on success; raises
    ValueError on any failure.

    For TIER_ANONYMOUS this is a no-op (returns None).
    For TIER_AUTHENTICATED the keyid must resolve through ``peer_resolver``.
    For TIER_OPERATOR the keyid is irrelevant; the signature must verify
    against ``operator_pubkey``.
    """
    if tier == TIER_ANONYMOUS:
        return None
    keyid, sig_b64, ts = _parse_http_auth_header(auth_header)
    current = now_seconds if now_seconds is not None else int(time.time())
    if abs(current - ts) > window_s:
        raise ValueError(f"timestamp out of window (skew={current - ts}s)")

    if tier == TIER_OPERATOR:
        if not operator_pubkey:
            raise ValueError(
                "no operator pubkey installed at ~/.nakshatra/keys/operator.pub.hex"
            )
        # Lazy import: avoid pulling nakshatra_auth at module top in case
        # the auth module is unavailable in a degraded bringup.
        if not _AUTH_AVAILABLE:
            raise ValueError("nakshatra_auth unavailable; cannot verify operator signature")
        if not _wauth.verify_request(
            operator_pubkey, method, path, body, ts, sig_b64
        ):
            raise ValueError("operator signature mismatch")
        return keyid

    if tier == TIER_AUTHENTICATED:
        if peer_resolver is None:
            raise ValueError("no peer resolver configured")
        pub_hex = peer_resolver.resolve(keyid)
        if not pub_hex:
            raise ValueError(f"unknown keyid: {keyid!r}")
        if not _AUTH_AVAILABLE:
            raise ValueError("nakshatra_auth unavailable; cannot verify signature")
        if not _wauth.verify_request(pub_hex, method, path, body, ts, sig_b64):
            raise ValueError("signature mismatch")
        return keyid

    raise ValueError(f"unknown tier: {tier!r}")


# Module-level state populated by main() so the per-request
# FileServerHandler can read it without constructor plumbing.
_HTTP_AUTH_STATE: dict = {
    "auth_required": False,
    "operator_pubkey": None,
    "peer_resolver": None,
    "slice_root": None,
}


class DaemonClient:
    """Manages a long-lived llama-nakshatra-worker subprocess over stdin/stdout."""

    def __init__(self, daemon_bin: str, sub_gguf: str, mode: str, n_ctx: int, n_threads: int = 0, n_gpu_layers: int = 0):
        self.proc = subprocess.Popen(
            [daemon_bin, sub_gguf, mode, str(n_ctx), str(n_threads), str(n_gpu_layers)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.lock = threading.Lock()
        # worker/kv-session-ownership-gate (2026-09-12): `self.lock` above only
        # makes ONE call() atomic on the wire (no garbled bytes if two threads
        # call concurrently) — it does NOT protect the daemon's KV cache state.
        # All serving/verify decode lives on a single fixed KV slot (seq 0 —
        # see worker_daemon.cpp), so if two DIFFERENT logical sessions get
        # routed to this same DaemonClient with overlapping lifetimes, each
        # individual call() stays byte-clean but their turns can interleave
        # and silently mutate each other's shared KV context. `_owner_lock` +
        # `_owner_session` add a narrow, additive, fail-closed gate on top:
        # acquire_session()/release_session() bracket one whole logical
        # session (a full Inference stream, or one Forward call), and a call
        # from a DIFFERENT session while the daemon is owned is refused
        # loudly instead of allowed to interleave. Queuing a refused session
        # is explicitly out of scope — refusing clearly is the fix for today.
        self._owner_lock = threading.Lock()
        self._owner_session: Optional[str] = None
        self._owner_since: float = 0.0
        # Stderr buffer (Phase 3.6): keeps last N lines so we can verify what
        # the daemon ACTUALLY did (e.g., GPU offload count) vs. what we asked.
        self.stderr_lines = collections.deque(maxlen=500)
        # Phase H: rolling per-RPC timing. Each Forward() records its
        # daemon-call duration here; main thread averages the last N for
        # the heartbeat payload so the pillar can latency-rank peers.
        self.recent_rpc_ms = collections.deque(maxlen=20)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # Wait for the "[daemon] ready" line so info+forward are valid.
        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise

    def _drain_stderr(self):
        for line in iter(self.proc.stderr.readline, b""):
            text = line.decode("utf-8", "replace")
            self.stderr_lines.append(text)
            sys.stderr.write(f"[daemon] {text}")
            sys.stderr.flush()

    def gpu_offload_status(self) -> dict:
        """Parse stderr buffer for what the daemon ACTUALLY did with the GPU.

        Returns:
          {
            "n_offloaded": int,    # N from "offloaded N/M layers to GPU"
            "total_layers": int,   # M from same
            "uses_gpu": bool,      # n_offloaded > 0
            "backend_hints": [],   # any backend names spotted in stderr
            "log_lines": [],       # relevant excerpt for diagnostics
          }
        """
        n_offloaded = 0
        total_layers = 0
        uses_gpu = False
        backend_hints = set()
        relevant = []

        # Pattern: "load_tensors: offloaded N/M layers to GPU"
        offload_re = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers", re.IGNORECASE)
        # Backend signature lines llama.cpp prints (Metal/ROCm/CUDA buffers)
        backend_signals = (
            ("metal",   re.compile(r"\bMetal\b|ggml-metal|MPS", re.IGNORECASE)),
            ("rocm",    re.compile(r"\bROCm\b|HIP\b|hipMalloc|amdhip", re.IGNORECASE)),
            ("cuda",    re.compile(r"\bCUDA\b|cudaMalloc|cuBLAS",     re.IGNORECASE)),
            ("vulkan",  re.compile(r"\bVulkan\b|vk_buffer|MoltenVK",  re.IGNORECASE)),
            ("cpu",     re.compile(r"\bCPU\b\s+(KV|compute|output)\s+buffer", re.IGNORECASE)),
        )

        for line in self.stderr_lines:
            m = offload_re.search(line)
            if m:
                n_offloaded = int(m.group(1))
                total_layers = int(m.group(2))
                uses_gpu = n_offloaded > 0
                relevant.append(line.rstrip())
            for name, rx in backend_signals:
                if rx.search(line):
                    backend_hints.add(name)
                    if name != "cpu" or "compute buffer" in line.lower():
                        relevant.append(line.rstrip())

        return {
            "n_offloaded": n_offloaded,
            "total_layers": total_layers,
            "uses_gpu": uses_gpu,
            "backend_hints": sorted(backend_hints),
            "log_lines": relevant[:20],
        }

    def _wait_ready(self, timeout: float = 60.0):
        # Daemon prints to stderr; we wait until it has loaded the model.
        # Simple heuristic: send INFO and wait for response.
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                _, _ = self.call(CMD_INFO, 0, b"")
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"daemon exited rc={self.proc.returncode}")
                time.sleep(0.5)
        raise TimeoutError("daemon never became ready")

    def call(self, cmd: int, n_tokens: int, payload: bytes, start_pos: int = 0, flags: int = 0):
        with self.lock:
            t0 = time.time()
            hdr = struct.pack("<IIIII", cmd, n_tokens, start_pos, flags, len(payload))
            self.proc.stdin.write(hdr + payload)
            self.proc.stdin.flush()
            head = self.proc.stdout.read(8)
            if len(head) != 8:
                raise RuntimeError(f"short read from daemon (got {len(head)} bytes)")
            status, plen = struct.unpack("<II", head)
            data = self.proc.stdout.read(plen) if plen else b""
            # Phase H: track per-call timing for latency-aware chain builds.
            # Skip cmd=3 (INFO) — those don't reflect inference latency.
            # Phase A6 (2026-05-20): NaN/Inf guard. Clock skews backward
            # or t0 corruption can produce non-finite or negative values
            # — drop those before they pollute the heartbeat payload.
            if cmd != CMD_INFO:
                clean = safe_rpc_ms((time.time() - t0) * 1000.0)
                if clean is not None:
                    self.recent_rpc_ms.append(clean)
            return status, data

    def info(self):
        s, p = self.call(CMD_INFO, 0, b"")
        if s != 0 or len(p) < 24:
            raise RuntimeError(f"info call failed status={s} len={len(p)}")
        layer_start, layer_end, n_embd, has_embd, has_lm, n_vocab = struct.unpack("<6i", p[:24])
        return dict(layer_start=layer_start, layer_end=layer_end, n_embd=n_embd,
                    has_token_embd=bool(has_embd), has_lm_head=bool(has_lm), n_vocab=n_vocab)

    def close(self):
        # EOF on stdin is the daemon's normal clean-shutdown path.  Only
        # escalate to SIGKILL when it ignores that path for five seconds;
        # an incidental close/wait error is not permission to kill a process
        # which may already have exited normally.
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

    # -- worker/kv-session-ownership-gate (2026-09-12) -----------------------

    def acquire_session(self, session_id: str,
                         stale_after_s: float = KV_SESSION_STALE_AFTER_S) -> bool:
        """Claim exclusive use of this daemon's KV state for `session_id`.

        Returns True when the caller now owns the daemon — either it was free,
        it was already owned by this same session_id (idempotent re-entry, so
        a stream's later steps needn't re-acquire), or the previous owner went
        stale (see KV_SESSION_STALE_AFTER_S) and is reclaimed. Returns False
        when a DIFFERENT, still-live session currently owns it — the caller
        must fail closed rather than proceed and interleave with it.
        """
        now = time.time()
        with self._owner_lock:
            if self._owner_session is None or self._owner_session == session_id:
                self._owner_session = session_id
                self._owner_since = now
                return True
            if now - self._owner_since > stale_after_s:
                # Abandoned owner — its session ended (crash/kill/timeout)
                # without calling release_session(). Reclaiming here is what
                # stops one bad session from permanently locking out the
                # daemon for everyone after it.
                sys.stderr.write(
                    f"[daemon] reclaiming stale KV-session ownership from "
                    f"{self._owner_session!r} (idle {now - self._owner_since:.0f}s) "
                    f"for {session_id!r}\n"
                )
                self._owner_session = session_id
                self._owner_since = now
                return True
            return False

    def release_session(self, session_id: str) -> None:
        """Release ownership if `session_id` currently holds it.

        A no-op for any other session id — this must never let a late or
        duplicate release from a session that has already lost ownership
        (e.g. reclaimed as stale) clobber a newer, legitimate owner.
        """
        with self._owner_lock:
            if self._owner_session == session_id:
                self._owner_session = None
                self._owner_since = 0.0

    @property
    def current_owner(self) -> Optional[str]:
        with self._owner_lock:
            return self._owner_session


class DaemonUnavailableError(RuntimeError):
    """The local inference daemon could not be made ready for a request."""


class OwnerLeaseConfig(NamedTuple):
    port: int
    token: str
    ttl_s: float
    resume_s: float


class WarmDirective(NamedTuple):
    model: str
    action: str
    until: float


_INVALID_WARM_DIRECTIVE = object()
WARM_DIRECTIVE_MAX_S = 300.0


def parse_warm_directive(response: dict, model_id: str, *, now=None):
    """Validate one pillar advisory, distinguishing absent from malformed."""
    if "warm" not in response:
        return None
    raw = response.get("warm")
    if not isinstance(raw, dict):
        return _INVALID_WARM_DIRECTIVE
    if raw.get("model") != model_id or raw.get("action") not in {
            "keep", "prewarm", "release"}:
        return _INVALID_WARM_DIRECTIVE
    until = raw.get("until")
    if isinstance(until, bool) or not isinstance(until, (int, float)):
        return _INVALID_WARM_DIRECTIVE
    try:
        until = float(until)
    except (OverflowError, ValueError):
        return _INVALID_WARM_DIRECTIVE
    current = (now or time.time)()
    if not math.isfinite(until) or until <= current:
        return _INVALID_WARM_DIRECTIVE
    return WarmDirective(
        model=model_id,
        action=raw["action"],
        until=min(until, current + WARM_DIRECTIVE_MAX_S),
    )


def owner_lease_state(last_clear_at: Optional[float], now: float, ttl_s: float,
                      *, explicitly_clear: bool = True,
                      reason: str = "") -> tuple[str, str]:
    """Pure owner-lease policy, using the mesh heartbeat's live/stale/dead bands."""
    if not explicitly_clear:
        return "lapsed", reason or "owner sentinel reported busy"
    if last_clear_at is None:
        return "lapsed", "owner lease has not received a clear heartbeat"
    # live <= 3/4 TTL < stale <= TTL < lapsed. The stale band used to start at TTL/2 (3 s), but a
    # Windows sentinel's ~2-3 s cadence (1 s GPU-counter sample) kept crossing it, and stale refuses
    # new sessions, so requests bounced at random (blackwell, 2026-10-13). The lapse point (TTL) is unchanged.
    state, _detail = node_state(
        last_clear_at, now, interval=ttl_s / 4.0,
        miss_stale=3, miss_dead=4,
    )
    if state == "live":
        return "live", reason or "owner lease heartbeat is live"
    if state == "stale":
        return "stale", "owner lease heartbeat is stale; refusing new sessions"
    return "lapsed", "owner lease heartbeat lapsed; silence means stop"


def owner_lease_config_from_env(environ=None) -> Optional[OwnerLeaseConfig]:
    """Return the launcher's validated lease config, or None when it is disabled."""
    env = os.environ if environ is None else environ
    if env.get("NAKSHATRA_OWNER_LEASE_ENABLED", "").strip().lower() not in {
            "1", "true", "yes"}:
        return None
    try:
        port = int(env.get("NAKSHATRA_OWNER_LEASE_PORT", "5571"))
        ttl_s = float(env.get("NAKSHATRA_OWNER_LEASE_TTL_S", "6"))
        resume_s = float(env.get("NAKSHATRA_OWNER_LEASE_RESUME_S", "30"))
    except ValueError as e:
        raise ValueError(f"invalid owner lease setting: {e}") from e
    token = env.get("NAKSHATRA_OWNER_LEASE_TOKEN", "").strip().lower()
    if not 1 <= port <= 65535:
        raise ValueError("NAKSHATRA_OWNER_LEASE_PORT must be between 1 and 65535")
    if not math.isfinite(ttl_s) or ttl_s <= 0:
        raise ValueError("NAKSHATRA_OWNER_LEASE_TTL_S must be finite and > 0")
    if not math.isfinite(resume_s) or resume_s < 0:
        raise ValueError("NAKSHATRA_OWNER_LEASE_RESUME_S must be finite and >= 0")
    if len(token) < 64 or len(token) % 2 or not re.fullmatch(r"[0-9a-f]+", token):
        raise ValueError(
            "NAKSHATRA_OWNER_LEASE_TOKEN must be 32+ random bytes encoded as hex"
        )
    return OwnerLeaseConfig(port, token, ttl_s, resume_s)


class IdleDaemonController:
    """Keep the heavyweight daemon resident only while it is being used.

    The Python worker, TLS listener, registration loop, and slice file remain
    alive in standby.  This controller is deliberately the sole owner of
    daemon creation and teardown so concurrent cold requests cannot race two
    GPU loads.  Session ownership lives here (rather than in a replaceable
    DaemonClient instance), which makes the no-unload invariant explicit.
    """

    def __init__(self, daemon_factory, sub_gguf: str, startup_sha256: str,
                 idle_unload_s: float = 600.0, *, release_grace_s: float = 30.0,
                 owner_lease_enabled: bool = False, now=None,
                 residency_event: threading.Event = None,
                 model_id: str = "", warm_directives_enabled: bool = True,
                 wall_now=None):
        if not math.isfinite(idle_unload_s) or idle_unload_s < 0:
            raise ValueError("idle_unload_s must be a finite number >= 0")
        self._daemon_factory = daemon_factory
        self._sub_gguf = sub_gguf
        self._startup_sha256 = startup_sha256.lower()
        self.idle_unload_s = float(idle_unload_s)
        if not math.isfinite(release_grace_s):
            raise ValueError("release_grace_s must be finite")
        self.release_grace_s = float(release_grace_s)
        self._now = now or time.monotonic
        self._wall_now = wall_now or time.time
        self._residency_event = residency_event
        self._spawn_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._owner_session: Optional[str] = None
        self._owner_since = 0.0
        self._in_flight = 0
        self._last_step_at = self._now()
        self._release_unload_at: Optional[float] = None
        self._closed = False
        self._model_id = model_id
        self._warm_directives_enabled = bool(warm_directives_enabled)
        self._warm_directive: Optional[WarmDirective] = None
        self._owner_lease_enabled = bool(owner_lease_enabled)
        self._lease_state = "lapsed" if owner_lease_enabled else "disabled"
        self._lease_reason = (
            "owner lease has not received a clear heartbeat"
            if owner_lease_enabled else ""
        )
        self._lease_admit = not owner_lease_enabled
        self._residency = "away" if owner_lease_enabled else "loaded"
        self.recent_rpc_ms = collections.deque(maxlen=20)
        self._last_gpu_offload_status = {}
        self._daemon = self._spawn_initial()
        self._daemon_info = None
        if owner_lease_enabled:
            # Capture immutable shape metadata before default-deny unloads the
            # initial process. The gRPC surface can then start in away without
            # a request lazily loading weights before the first clear lease.
            self._daemon_info = self._daemon.info()
            try:
                self._last_gpu_offload_status = dict(
                    self._daemon.gpu_offload_status()
                )
            except (AttributeError, TypeError):
                pass
            self._daemon.close()
            self._daemon = None

    def _spawn_initial(self):
        try:
            daemon = self._daemon_factory()
        except Exception as e:
            raise DaemonUnavailableError(f"daemon initial spawn failed: {e}") from e
        # Preserve one timing window across unload/reload transitions.
        try:
            daemon.recent_rpc_ms = self.recent_rpc_ms
        except AttributeError:
            # A few argument-parsing tests inject a bare sentinel and stop
            # before the daemon is used. Real daemon implementations expose
            # this deque.
            pass
        return daemon

    @property
    def residency(self) -> str:
        with self._state_lock:
            return self._residency

    @property
    def current_owner(self) -> Optional[str]:
        with self._state_lock:
            return self._owner_session

    @property
    def in_flight(self) -> int:
        with self._state_lock:
            return self._in_flight

    @property
    def lease_state(self) -> str:
        with self._state_lock:
            return self._lease_state

    @property
    def lease_reason(self) -> str:
        with self._state_lock:
            return self._lease_reason

    def _warm_action_locked(self) -> Optional[str]:
        directive = self._warm_directive
        if (not self._warm_directives_enabled or directive is None
                or directive.until <= self._wall_now()):
            return None
        return directive.action

    @property
    def warm_directive(self) -> Optional[WarmDirective]:
        with self._state_lock:
            if self._warm_action_locked() is None:
                return None
            return self._warm_directive

    def apply_warm_directive(self, directive: Optional[WarmDirective]) -> None:
        """Apply one validated advisory without weakening owner-lease policy."""
        if not self._warm_directives_enabled:
            return
        with self._state_lock:
            # Renewals carry a fresh "until" every heartbeat; log only a change of model/action.
            def _kind(d):
                return None if d is None else (d.model, d.action)
            changed = _kind(directive) != _kind(self._warm_directive)
            self._warm_directive = directive
            action = self._warm_action_locked()
            can_prewarm = (
                action == "prewarm"
                and self._daemon is None
                and self._residency == "standby"
                and (not self._owner_lease_enabled
                     or (self._lease_state == "live" and self._lease_admit))
            )
        if changed:
            model = directive.model if directive is not None else self._model_id
            action_for_log = directive.action if directive is not None else "none"
            until = directive.until if directive is not None else 0
            print(
                f"[worker] warm directive: model={model} "
                f"action={action_for_log} until={until:g}",
                flush=True,
            )
        if can_prewarm:
            try:
                self.ensure_loaded()
            except DaemonUnavailableError as e:
                sys.stderr.write(f"[worker] warm prewarm failed: {e}\n")
        elif action == "release":
            self.unload_if_idle(force_window=True)

    @property
    def service_alive(self) -> bool:
        with self._state_lock:
            if self._closed:
                return False
            daemon = self._daemon
        if daemon is None:
            return True  # standby is a healthy worker state
        proc = getattr(daemon, "proc", None)
        return proc is None or proc.poll() is None

    def record_startup_sha256(self, digest: str) -> None:
        """Install the immutable digest used for every later cold reload."""
        digest = (digest or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("startup sub-GGUF SHA-256 must be 64 lowercase hex characters")
        with self._state_lock:
            if self._startup_sha256 and self._startup_sha256 != digest:
                raise ValueError("startup sub-GGUF SHA-256 is already recorded")
            self._startup_sha256 = digest

    def _announce_residency(self, residency: str) -> None:
        changed = False
        with self._state_lock:
            if self._residency != residency:
                self._residency = residency
                changed = True
        if changed:
            print(f"[worker] residency -> {residency}", flush=True)
            if self._residency_event is not None:
                self._residency_event.set()

    def _verify_slice_for_reload(self) -> None:
        if not self._startup_sha256:
            raise DaemonUnavailableError(
                "daemon reload refused: no startup SHA-256 was recorded"
            )
        try:
            observed = sha256_of_file(self._sub_gguf)
        except Exception as e:
            raise DaemonUnavailableError(
                f"daemon reload refused: could not verify sub-GGUF: {e}"
            ) from e
        if observed.lower() != self._startup_sha256:
            raise DaemonUnavailableError(
                "daemon reload refused: sub-GGUF SHA-256 mismatch "
                f"(expected {self._startup_sha256}, observed {observed.lower()})"
            )

    def ensure_loaded(self):
        with self._state_lock:
            if self._closed:
                raise DaemonUnavailableError("daemon reload refused: worker is stopping")
            if self._daemon is not None:
                return self._daemon
            if self._owner_lease_enabled and not self._lease_admit:
                raise DaemonUnavailableError(self._lease_reason)

        # Exactly one contender verifies and spawns.  Losers re-check after
        # acquiring the same lock and reuse the winner's daemon.
        with self._spawn_lock:
            with self._state_lock:
                if self._closed:
                    raise DaemonUnavailableError("daemon reload refused: worker is stopping")
                if self._daemon is not None:
                    return self._daemon
                if self._owner_lease_enabled and not self._lease_admit:
                    raise DaemonUnavailableError(self._lease_reason)
            try:
                self._verify_slice_for_reload()
                daemon = self._daemon_factory()
                try:
                    daemon.recent_rpc_ms = self.recent_rpc_ms
                except AttributeError:
                    pass
            except DaemonUnavailableError as e:
                sys.stderr.write(
                    f"[worker] daemon reload failed; worker remains in standby: {e}\n"
                )
                raise
            except Exception as e:
                message = f"daemon reload failed; worker remains in standby: {e}"
                sys.stderr.write(f"[worker] {message}\n")
                raise DaemonUnavailableError(message) from e
            with self._state_lock:
                if self._closed:
                    daemon.close()
                    raise DaemonUnavailableError("daemon reload refused: worker is stopping")
                if self._owner_lease_enabled and not self._lease_admit:
                    daemon.close()
                    raise DaemonUnavailableError(self._lease_reason)
                self._daemon = daemon
                self._last_step_at = self._now()
            self._announce_residency("loaded")
            return daemon

    def acquire_session(self, session_id: str,
                        stale_after_s: float = KV_SESSION_STALE_AFTER_S) -> bool:
        now = self._now()
        with self._state_lock:
            if (self._owner_lease_enabled and not self._lease_admit
                    and self._owner_session != session_id):
                raise DaemonUnavailableError(self._lease_reason)
            self._release_unload_at = None
            if self._owner_session is None or self._owner_session == session_id:
                self._owner_session = session_id
                self._owner_since = now
            elif now - self._owner_since > stale_after_s:
                sys.stderr.write(
                    f"[daemon] reclaiming stale KV-session ownership from "
                    f"{self._owner_session!r} (idle {now - self._owner_since:.0f}s) "
                    f"for {session_id!r}\n"
                )
                self._owner_session = session_id
                self._owner_since = now
            else:
                return False
        try:
            self.ensure_loaded()
        except Exception:
            self.release_session(session_id)
            raise
        return True

    def release_session(self, session_id: str) -> None:
        unload_now = False
        with self._state_lock:
            if self._owner_session == session_id:
                self._owner_session = None
                self._owner_since = 0.0
                if self._in_flight == 0 and self.release_grace_s >= 0:
                    self._release_unload_at = self._now() + self.release_grace_s
                    unload_now = self.release_grace_s == 0
                if self._in_flight == 0 and self._warm_action_locked() == "release":
                    unload_now = True
        if unload_now:
            self.unload_if_idle()

    def mark_step_served(self) -> None:
        """Refresh idle accounting for a response served from a local cache."""
        with self._state_lock:
            self._last_step_at = self._now()
            self._release_unload_at = None

    def call(self, cmd: int, n_tokens: int, payload: bytes,
             start_pos: int = 0, flags: int = 0):
        is_inference = cmd in (CMD_TOKEN_DECODE, CMD_EMBD_DECODE, CMD_EAGLE_HIDDEN)
        if is_inference:
            with self._state_lock:
                self._release_unload_at = None
                self._in_flight += 1
        served = False
        try:
            daemon = self.ensure_loaded()
            result = daemon.call(cmd, n_tokens, payload, start_pos=start_pos, flags=flags)
            with self._state_lock:
                if self._owner_lease_enabled and self._lease_state == "lapsed":
                    raise DaemonUnavailableError(self._lease_reason)
            served = is_inference
            return result
        except Exception as e:
            with self._state_lock:
                lapsed = self._owner_lease_enabled and self._lease_state == "lapsed"
                reason = self._lease_reason
            if lapsed and not isinstance(e, DaemonUnavailableError):
                raise DaemonUnavailableError(reason) from e
            raise
        finally:
            if is_inference:
                release_now = False
                with self._state_lock:
                    self._in_flight -= 1
                    if served:
                        self._last_step_at = self._now()
                    release_now = (
                        self._in_flight == 0
                        and self._owner_session is None
                        and self._warm_action_locked() == "release"
                    )
                if release_now:
                    self.unload_if_idle()

    def info(self):
        with self._state_lock:
            cached = self._daemon_info
        if cached is not None:
            return dict(cached)
        info = self.ensure_loaded().info()
        with self._state_lock:
            self._daemon_info = dict(info)
        return info

    def gpu_offload_status(self) -> dict:
        with self._state_lock:
            daemon = self._daemon
            previous = dict(self._last_gpu_offload_status)
        if daemon is None:
            return previous
        status = daemon.gpu_offload_status()
        with self._state_lock:
            self._last_gpu_offload_status = dict(status)
        return status

    def unload_if_idle(self, *, force_window: bool = False) -> bool:
        """Cleanly enter standby when every safety guard permits it.

        ``force_window`` skips only the idle/release-grace timing requirement;
        in-flight work, live session ownership, keep directives, and shutdown
        state remain authoritative.
        """
        with self._spawn_lock:
            with self._state_lock:
                now = self._now()
                release_due = (
                    self._release_unload_at is not None
                    and now >= self._release_unload_at
                )
                idle_due = (
                    self.idle_unload_s > 0
                    and self._release_unload_at is None
                    and now - self._last_step_at >= self.idle_unload_s
                )
                warm_action = self._warm_action_locked()
                if warm_action == "keep":
                    return False
                directive_release = warm_action == "release"
                if (self._closed or self._daemon is None or self._in_flight
                        or not (force_window or directive_release
                                or release_due or idle_due)):
                    return False
                if self._owner_session is not None:
                    # A stream that ended without release_session() (client crashed) would
                    # otherwise pin VRAM forever. Same staleness rule acquire_session() uses
                    # to reclaim ownership.
                    if now - self._owner_since <= KV_SESSION_STALE_AFTER_S:
                        return False
                    sys.stderr.write(
                        f"[worker] releasing stale KV-session {self._owner_session!r} "
                        f"(idle {now - self._owner_since:.0f}s) before daemon unload\n")
                    self._owner_session = None
                    self._owner_since = 0.0
                daemon = self._daemon
                self._daemon = None
                self._release_unload_at = None
            try:
                daemon.close()
            except Exception as e:
                # The controller cannot safely reuse a daemon whose shutdown
                # path failed.  Stay in standby and let a request create a
                # freshly verified instance.
                sys.stderr.write(f"[worker] daemon clean stop failed: {e}\n")
            self._announce_residency("standby")
            return True

    def update_owner_lease(self, state: str, reason: str, *, resume: bool = False) -> None:
        """Apply one policy observation; lapse is an immediate destructive release."""
        if not self._owner_lease_enabled:
            return
        if state not in {"live", "stale", "lapsed"}:
            raise ValueError(f"invalid owner lease state {state!r}")
        if state == "lapsed":
            # Publish default-deny immediately. If a digest verification/load
            # currently owns _spawn_lock, that contender observes this gate
            # before installing its new daemon and tears it back down.
            with self._state_lock:
                self._lease_state = state
                self._lease_reason = reason
                self._lease_admit = False
                self._owner_session = None
                self._owner_since = 0.0
                self._release_unload_at = None
            self._announce_residency("away")
            with self._spawn_lock:
                with self._state_lock:
                    daemon = self._daemon
                    self._daemon = None
                if daemon is not None:
                    try:
                        daemon.close()
                    except Exception as e:
                        sys.stderr.write(f"[worker] owner-lease daemon stop failed: {e}\n")
            return
        with self._state_lock:
            self._lease_state = state
            self._lease_reason = reason
            if state == "stale":
                self._lease_admit = False
                return
            if resume:
                self._lease_admit = True
        if state == "live" and resume and self.residency == "away":
            self._announce_residency("standby")

    def idle_monitor_loop(self, stop_event: threading.Event) -> None:
        if self.idle_unload_s == 0 and self.release_grace_s < 0:
            return
        while not stop_event.is_set():
            with self._state_lock:
                deadlines = []
                warm_action = self._warm_action_locked()
                if warm_action == "keep":
                    remaining = max(
                        0.05,
                        self._warm_directive.until - self._wall_now(),
                    )
                elif warm_action == "release":
                    remaining = 1.0
                elif self._release_unload_at is not None:
                    deadlines.append(self._release_unload_at)
                elif self.idle_unload_s > 0:
                    deadlines.append(self._last_step_at + self.idle_unload_s)
                if warm_action not in {"keep", "release"}:
                    remaining = max(0.05, min(deadlines) - self._now()) if deadlines else 1.0
            if stop_event.wait(timeout=min(remaining, 1.0)):
                return
            self.unload_if_idle()

    def close(self) -> None:
        with self._spawn_lock:
            with self._state_lock:
                self._closed = True
                daemon = self._daemon
                self._daemon = None
            if daemon is not None:
                daemon.close()


class OwnerLease:
    """Stateful receipt history around the pure owner_lease_state policy."""

    def __init__(self, config: OwnerLeaseConfig, daemon: IdleDaemonController, *, now=None):
        self.config = config
        self.daemon = daemon
        self._now = now or time.monotonic
        self._lock = threading.Lock()
        self._last_clear_at: Optional[float] = None
        self._resume_since: Optional[float] = None
        self._explicitly_clear = True
        self._reason = ""

    def snapshot(self) -> dict:
        with self._lock:
            now = self._now()
            state, reason = owner_lease_state(
                self._last_clear_at, now, self.config.ttl_s,
                explicitly_clear=self._explicitly_clear, reason=self._reason,
            )
            resume_ready = bool(
                state == "live" and self._resume_since is not None
                and now - self._resume_since >= self.config.resume_s
            )
            return {"state": state, "reason": reason,
                    "resume_ready": resume_ready,
                    "last_clear_at": self._last_clear_at}

    def heartbeat(self, *, at: float, clear: bool, reason: str) -> dict:
        del at  # sender wall time is audit context; receipt monotonic time owns expiry
        with self._lock:
            now = self._now()
            prior_state, _ = owner_lease_state(
                self._last_clear_at, now, self.config.ttl_s,
                explicitly_clear=self._explicitly_clear, reason=self._reason,
            )
            self._reason = reason
            if not clear:
                self._explicitly_clear = False
                self._resume_since = None
            else:
                # Only a real lapse restarts the resume window. "stale" is the jitter band
                # (a heartbeat a little late): it already refuses new sessions, but counting
                # it as a restart kept a Windows sentinel (1 s GPU-counter sample + 2 s
                # sleep) from ever accruing 30 s of resume time on blackwell.
                if prior_state == "lapsed" or self._resume_since is None:
                    self._resume_since = now
                self._explicitly_clear = True
                self._last_clear_at = now
        return self.reconcile()

    def reconcile(self) -> dict:
        snap = self.snapshot()
        # Operators need to see WHY a worker is away (busy owner vs. a silent sentinel); log
        # each change of state/reason, never every heartbeat and never the token.
        seen = (snap["state"], snap["reason"], snap["resume_ready"])
        if seen != getattr(self, "_last_logged", None):
            self._last_logged = seen
            print(f"[worker] owner lease: {snap['state']} ({snap['reason']})"
                  + ("; resume window satisfied" if snap["resume_ready"] else ""), flush=True)
        self.daemon.update_owner_lease(
            snap["state"], snap["reason"], resume=snap["resume_ready"]
        )
        return snap

    def monitor_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.wait(timeout=min(1.0, self.config.ttl_s / 4.0)):
            self.reconcile()


def owner_lease_http_response(lease: OwnerLease, path: str,
                              authorization: str, body: bytes) -> tuple[int, dict]:
    """Pure request validation/action used by the tiny HTTP adapter."""
    prefix = "Bearer "
    candidate = authorization[len(prefix):] if authorization.startswith(prefix) else ""
    # Compare bytes: hmac.compare_digest raises TypeError on non-ASCII str, which a hostile local
    # caller could otherwise use to crash the handler instead of getting a clean 401.
    if not hmac.compare_digest(candidate.encode("utf-8", "surrogateescape"),
                               lease.config.token.encode("utf-8")):
        return 401, {"error": "unauthorized"}
    if path != "/lease":
        return 404, {"error": "not found"}
    try:
        if not body or len(body) > 4096:
            raise ValueError("body length must be 1..4096 bytes")
        payload = json.loads(body)
        at = payload["at"]
        clear = payload["clear"]
        reason = payload.get("reason", "")
        if isinstance(at, bool) or not isinstance(at, (int, float)):
            raise ValueError("at must be a unix timestamp")
        if not isinstance(clear, bool):
            raise ValueError("clear must be boolean")
        if not isinstance(reason, str) or len(reason.encode()) > 512:
            raise ValueError("reason must be a string of at most 512 bytes")
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        return 400, {"error": str(e)}
    snap = lease.heartbeat(at=float(at), clear=clear, reason=reason)
    return 200, {"state": snap["state"], "residency": lease.daemon.residency}


def start_owner_lease_server(lease: OwnerLease, *, server_class=HTTPServer):
    """Start the authenticated endpoint, hard-coded to IPv4 loopback."""

    class Handler(BaseHTTPRequestHandler):
        # One request at a time; a stalled local sender must not hold the endpoint (a missed
        # heartbeat only ever releases the GPU, but the owner's real sentinel should get through).
        timeout = 5

        def _json(self, status: int, payload: dict):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            body = self.rfile.read(length) if 0 < length <= 4096 else b""
            status, payload = owner_lease_http_response(
                lease, self.path, self.headers.get("Authorization", ""), body
            )
            self._json(status, payload)

        def log_message(self, _format, *_args):
            return

    httpd = server_class(("127.0.0.1", lease.config.port), Handler)
    thread = threading.Thread(
        target=httpd.serve_forever, daemon=True, name="owner-lease-http"
    )
    thread.start()
    return httpd, thread


# v0.5 §9.7: idempotency cache is sized in MB, internally converted to an
# entry cap using this per-entry estimate. ~10 KB matches a typical hidden-state
# response (3072-dim × f32 + proto overhead). Adjust if the average grows.
IDEM_BYTES_PER_ENTRY = 10 * 1024


def _grpc_call_aborted(context) -> bool:
    """True iff this call already ended via context.abort() (a status code is set). Older grpc lacks code()."""
    try:
        return context.code() is not None
    except Exception:
        return False


def _startup_blocker(registered: bool, resolver, node_id: str, required_keyids=()) -> str:
    """Why this worker is NOT ready to authenticate callers yet ('' when it is).

    Ready means the pillar accepted our registration AND our peer-key cache can read the pillar's key table - proven by the
    cache resolving our OWN node id (the pillar only lists us after accepting the registration, and only a successful /peers
    refresh fills the cache). Anything less used to be reported as ready: register_with_pillar's False (e.g. a 401 from a TOFU key
    mismatch) was discarded and a failed first refresh was swallowed, so a worker that could authenticate NO ONE answered Info fine
    and the first request died `unknown keyid`. `required_keyids` adds consumers that must already resolve (NAKSHATRA_READY_KEYIDS)."""
    if not registered:
        return "registration with the pillar has not succeeded"
    if resolver is not None:
        for keyid in (node_id, *required_keyids):
            if keyid and resolver.resolve(keyid) is None:
                return f"peer-key cache does not hold {keyid!r} yet"
    return ""


def _await_startup_ready(servicer, register, resolver, node_id: str, required_keyids=(), *, registered: bool = False,
                         poll_s: float = 2.0, log_every_s: float = 30.0, sleep=time.sleep, now=time.monotonic) -> bool:
    """Block until _startup_blocker() is empty, retrying registration and the /peers refresh; then clear `servicer.starting`.
    Runs in a daemon thread when the first pass was not enough. Returns True once ready (False only if `servicer.starting`
    was cleared by someone else first)."""
    last_log = -log_every_s
    while servicer.starting:
        if not registered:
            registered = bool(register())
        if registered and resolver is not None and any(resolver.resolve(k) is None for k in (node_id, *required_keyids) if k):
            try:
                resolver.refresh_once()
            except Exception:  # noqa: BLE001 - reported through the blocker below
                pass
        why = _startup_blocker(registered, resolver, node_id, required_keyids)
        if not why:
            servicer.starting_reason = ""
            servicer.starting = False
            print("[worker] ready: registered with the pillar and the peer-key cache is populated", flush=True)
            return True
        servicer.starting_reason = why
        if now() - last_log >= log_every_s:
            print(f"[worker] still starting: {why} (retrying)", flush=True)
            last_log = now()
        sleep(poll_s)
    return False


class WorkerServicer(pb_grpc.NakshatraServicer):
    def __init__(self, daemon: DaemonClient, mode: str, layer_start: int, layer_end: int, model_id: str,
                 idem_max_entries: int = 6400, idem_ttl_seconds: float = 60.0,
                 peer_resolver=None, auth_required: bool = False,
                 refuse_unregistered_peers: bool = True,
                 refuse_unpinned_peers: bool = True,
                 paid_admission=None, worker_priv: Optional[bytes] = None,
                 worker_pub_hex: str = ""):
        self.daemon = daemon
        self.mode = mode
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.model_id = model_id
        # Phase B (2026-05-20): peer-key resolver caches pillar's /peers
        # projection. When auth_required is True, every non-Info gRPC
        # call must present a Sthambha-Ed25519 signature whose keyid
        # resolves through this resolver.
        self.peer_resolver = peer_resolver
        self.auth_required = auth_required
        self.paid_admission = paid_admission
        self.paid_gate_enabled = paid_admission is not None
        self.worker_priv = worker_priv
        self.worker_pub_hex = worker_pub_hex
        self._worker_identity_ready = False
        if (isinstance(worker_priv, bytes) and len(worker_priv) == 32
                and isinstance(worker_pub_hex, str)
                and re.fullmatch(r"[0-9a-f]{64}", worker_pub_hex)):
            try:
                derived = ed25519.Ed25519PrivateKey.from_private_bytes(
                    worker_priv).public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw).hex()
                self._worker_identity_ready = hmac.compare_digest(
                    derived, worker_pub_hex)
            except Exception:
                self._worker_identity_ready = False
        self.fabric_transport_enabled = False
        # Session ownership and resident KV are deliberately distinct. A
        # stream releases active ownership when it closes, while seq-0 remains
        # resident until the next cold decode replaces it.
        self._kv_gate_lock = threading.RLock()
        self._kv_epoch = 0
        self._kv_resident_principal = None
        # worker/cold-start-readiness: the gRPC port is bound BEFORE registration installs the peer-key resolver
        # (sha256 of a 10 GB slice + pillar registration take seconds). Until main() flips this, an authenticated
        # worker is STARTING - not "unauthenticated". Default False so every direct construction (tests, solo,
        # Mode A) behaves exactly as before; main() sets True only when a pillar registration is still pending.
        self.starting = False
        self.starting_reason = ""  # WHY it is not ready yet (shown in Info's abort details and /healthz)
        self.refuse_unregistered_peers = refuse_unregistered_peers
        # 2026-05-21 SPKI Phase 3.3: refuse outbound channels to peers
        # for which the pillar has not (yet) distributed a SPKI hash.
        # Default true closes the cross-worker MITM gap; operators with
        # legacy Mode-A clusters set NAKSHATRA_REFUSE_UNPINNED_PEERS=false
        # and accept plaintext outbound.
        self.refuse_unpinned_peers = refuse_unpinned_peers
        self._authz_rejections = 0
        self._ssrf_rejections = 0
        # 2026-05-29 fabric Phase F — first-worker gRPC→fabric bridge
        # (sprint OQ8). main() sets this when args.transport=fabric AND
        # mode=first, pointing at FabricBackend.first_worker_round_trip.
        # Forward then ships the decoded hidden_state via fabric +
        # blocks waiting for the last worker's token via the feedback
        # wormhole, returning the token in the gRPC reply. When None
        # (gRPC mode or non-first mode), Forward behaves as before.
        self.fabric_first_worker_bridge = None
        # 2026-05-21 SPKI Phase 3.3+3.5: per-reason counters for the
        # outbound pin check, surfaced in /healthz.
        self._spki_unpinned_refusals = 0
        self._spki_mismatch_refusals = 0
        self._spki_probe_failures = 0
        self.daemon_info = daemon.info()
        print(f"[worker] daemon info: {self.daemon_info}", flush=True)
        self.n_embd = self.daemon_info["n_embd"]
        # v0.5 M0.5.2: idempotency cache. Maps authenticated
        # (keyid, session_id) principal -> {"steps": {step_id: {response,
        # kv_epoch}},
        # "last_touch": float}. On a duplicate (session_id, step_id) we return the cached
        # response without touching the daemon. TTL-expires per-session; hard-capped by
        # total entries across all sessions.
        self._idem_lock = threading.Lock()
        self._idem_cache: dict = {}
        self._idem_ttl = idem_ttl_seconds
        self._idem_max = idem_max_entries
        self._idem_hits = 0
        self._idem_misses = 0
        # v0.5 M0.5.3: peer-stream cache for rpc_push. A downstream bidi
        # stream belongs to exactly one upstream principal and is closed with
        # that upstream session; it may never cross a logical-session boundary.
        # Map: (address, principal) -> (channel, request queue, response iterator)
        # Phase A3 (2026-05-20): OrderedDict + LRU cap. An attacker who
        # repeatedly pushes to fresh addresses can no longer balloon
        # this dict; oldest stream is evicted on overflow.
        self._peer_lock = threading.Lock()
        self._peer_streams: "collections.OrderedDict" = collections.OrderedDict()
        self._push_count = 0
        self._push_errors = 0
        self._peer_evictions = 0
        # 2026-09-12 participation binding — see SignParticipation's docstring. Maps
        # session_id -> {"last_touch": float, "generated_token_ids": [int, ...]} for the
        # sessions THIS worker has actually served, populated by Inference() itself (never
        # by the request being verified). Mirrors the idempotency cache's TTL+cap shape —
        # same eviction pattern, separate map, because this one must outlive a session's
        # idempotency entries (SignParticipation is called AFTER generation ends, by which
        # point early steps may already be idle-evicted from _idem_cache).
        self._participation_lock = threading.Lock()
        self._served_sessions: "collections.OrderedDict" = collections.OrderedDict()
        self._served_ttl = max(idem_ttl_seconds, 900.0)   # a generation can outlive a short idem TTL
        self._served_max_sessions = 2048

    def _idem_evict(self, now: float):
        """Drop expired sessions, then cap total entries. Caller holds _idem_lock."""
        expired = [s for s, e in self._idem_cache.items() if now - e["last_touch"] > self._idem_ttl]
        for s in expired:
            del self._idem_cache[s]
        total = sum(len(e["steps"]) for e in self._idem_cache.values())
        while total > self._idem_max and self._idem_cache:
            oldest = min(self._idem_cache.keys(), key=lambda s: self._idem_cache[s]["last_touch"])
            total -= len(self._idem_cache[oldest]["steps"])
            del self._idem_cache[oldest]

    def _idem_get(self, session_principal, step_id: str,
                  request_sha256: str):
        with self._idem_lock:
            sess = self._idem_cache.get(session_principal)
            if sess and step_id in sess["steps"]:
                if sess["steps"][step_id]["request_sha256"] != request_sha256:
                    raise IdempotencyConflict(
                        "step_id was reused with different request bytes")
                self._idem_hits += 1
                sess["last_touch"] = time.time()
                return sess["steps"][step_id]
            self._idem_misses += 1
            return None

    def _idem_put(self, session_principal, step_id: str, response,
                  kv_epoch: int, request_sha256: str):
        with self._idem_lock:
            now = time.time()
            sess = self._idem_cache.setdefault(
                session_principal, {"steps": {}, "last_touch": now})
            sess["steps"][step_id] = {
                "response": response,
                "kv_epoch": int(kv_epoch),
                "request_sha256": request_sha256,
            }
            sess["last_touch"] = now
            self._idem_evict(now)

    def idem_stats(self) -> dict:
        with self._idem_lock:
            sessions = len(self._idem_cache)
            entries = sum(len(e["steps"]) for e in self._idem_cache.values())
            return {
                "hits": self._idem_hits, "misses": self._idem_misses,
                "sessions": sessions, "entries": entries,
                "max_entries": self._idem_max, "ttl_seconds": self._idem_ttl,
            }

    def _served_evict(self, now: float):
        """Drop expired sessions, then cap total sessions. Caller holds _participation_lock.
        Mirrors `_idem_evict` — same shape, separate map (see __init__ for why)."""
        expired = [s for s, e in self._served_sessions.items()
                  if now - e["last_touch"] > self._served_ttl]
        for s in expired:
            del self._served_sessions[s]
        while len(self._served_sessions) > self._served_max_sessions:
            oldest = min(self._served_sessions, key=lambda s: self._served_sessions[s]["last_touch"])
            del self._served_sessions[oldest]

    def _record_session_activity(self, session_id: str,
                                 token_id: Optional[int] = None,
                                 authenticated_keyid=None):
        """Called ONLY from the real serving path (Inference), right after this worker
        actually ran a decode for `session_id` — never from SignParticipation, which only
        ever READS this. `token_id` is appended when this worker just produced the run's
        final token for that step (mode=="last"), so the accumulated list is exactly the
        sequence `receipt.output_sha256()` would hash for a run this worker genuinely
        finished end to end."""
        if not session_id:
            return
        with self._participation_lock:
            now = time.time()
            # 2026-09-13: there is no sessionless fallback left to keep separate from (see
            # SignParticipation) — this map is now the ONLY evidence SignParticipation ever
            # consults, keyed strictly by the real session_id this worker served.
            principal = self._session_principal(
                authenticated_keyid, session_id)
            entry = self._served_sessions.setdefault(
                principal, {"last_touch": now, "generated_token_ids": []})
            entry["last_touch"] = now
            if token_id is not None:
                entry["generated_token_ids"].append(int(token_id))
            self._served_evict(now)

    def _participation_evidence(self, run_id: str, authenticated_keyid=None):
        """Return (was_this_run_seen, generated_token_ids_or_None) for SignParticipation to
        judge against. Read-only — never mutates `_served_sessions` beyond eviction."""
        with self._participation_lock:
            now = time.time()
            self._served_evict(now)
            principal = self._session_principal(
                authenticated_keyid, run_id)
            entry = self._served_sessions.get(principal)
            if entry is None:
                return False, None
            return True, list(entry["generated_token_ids"])

    def _open_outbound_channel(self, address: str):
        """2026-05-21 SPKI Phase 3 — open a gRPC channel for outbound
        push, pinned against the SPKI hash the pillar attests for the
        destination peer.

        Three refusal arms:
        - ``unpinned_peer`` — pillar has no SPKI on file for this
          address (peer is pre-Phase-2 or the cache hasn't seen it
          yet) and ``refuse_unpinned_peers`` is True.
        - ``spki_mismatch`` — observed cert SPKI doesn't match the
          attested one. Audit-emits ``spki_pin_mismatch`` for
          operator forensics; the worker can detect a substituted
          peer cert (operator swap without re-registering, or MITM).
        - ``probe_failed`` — TLS probe to the peer's port failed
          (peer down, TLS-disabled, wrong port). Surfaces as
          push_failed; client downgrades to client-relay.

        Returns a ``grpc.Channel`` on success. Caller is responsible
        for closing it (LRU eviction in ``_get_peer_stream``).
        """
        if not _TLS_AVAILABLE:
            # No TLS module — legacy Mode-A bringup. The Phase 2 boot
            # path already emitted a WARN; just open insecure.
            return grpc.insecure_channel(address, options=[
                ("grpc.max_receive_message_length", WORKER_GRPC_MAX_MESSAGE_BYTES),
                ("grpc.max_send_message_length", WORKER_GRPC_MAX_MESSAGE_BYTES)])
        expected = (self.peer_resolver.expected_spki(address)
                    if self.peer_resolver is not None else None)
        try:
            return _wtls.open_pinned_channel(
                address, expected,
                refuse_unpinned=self.refuse_unpinned_peers,
            )
        except _wtls.PinError as e:
            if e.reason == "unpinned_peer":
                self._spki_unpinned_refusals += 1
                _audit("spki_pin_unpinned", peer=address)
            elif e.reason == "spki_mismatch":
                self._spki_mismatch_refusals += 1
                # 3.5 — operator-facing forensic event. Includes the
                # full expected + actual hashes so the operator can
                # tell whether they rotated a cert without
                # re-registering, or whether something more sinister
                # is going on.
                _audit("spki_pin_mismatch",
                       peer=address,
                       expected=e.details.get("expected"),
                       actual=e.details.get("actual"))
            elif e.reason == "probe_failed":
                self._spki_probe_failures += 1
                _audit("spki_probe_failed",
                       peer=address,
                       error=e.details.get("error"))
            raise

    def _get_peer_stream(self, address: str, owner_principal):
        """Open or reuse a bidi Inference stream to a peer worker.

        v0.5 M0.5.3 — server-to-server push. Returns (req_queue, response_iterator).
        Sending None to the queue closes the stream.

        Phase A3 (2026-05-20): bounded by MAX_PEER_STREAMS via LRU
        eviction. Re-touching an existing entry refreshes it (move to
        end); opening a fresh stream may evict the oldest. Eviction
        closes the request side (None sentinel) and the channel.
        """
        with self._peer_lock:
            cache_key = (address, owner_principal)
            cached = self._peer_streams.get(cache_key)
            if cached is not None:
                self._peer_streams.move_to_end(cache_key)
                return cached[1], cached[2]
            # 2026-05-21 SPKI Phase 3.3 + 3.4 + 3.5: pin the outbound
            # TLS handshake against the SPKI hash the pillar attests
            # for this peer. _open_outbound_channel raises PinError on
            # refuse; the calling Inference handler catches Exception
            # around _get_peer_stream and emits the structured
            # push_failed: error message that lets the client downgrade
            # the session. spki_pin_mismatch audit emission lives in
            # _open_outbound_channel so it fires before the exception
            # propagates.
            channel = self._open_outbound_channel(address)
            stub = pb_grpc.NakshatraStub(channel)
            req_q: queue.Queue = queue.Queue()

            def request_gen():
                while True:
                    step = req_q.get()
                    if step is None:
                        return
                    yield step

            resp_iter = stub.Inference(request_gen())
            self._peer_streams[cache_key] = (channel, req_q, resp_iter)
            while len(self._peer_streams) > MAX_PEER_STREAMS:
                old_key, (old_ch, old_q, _old_iter) = (
                    self._peer_streams.popitem(last=False)
                )
                self._peer_evictions += 1
                try: old_q.put(None)
                except Exception: pass
                try: old_ch.close()
                except Exception: pass
                sys.stderr.write(
                    f"[peer-streams] evicted oldest {old_key[0]!r} "
                    f"(cap={MAX_PEER_STREAMS})\n"
                )
            return req_q, resp_iter

    def _close_peer_streams(self, owner_principal) -> None:
        """Close every downstream stream owned by one upstream session."""
        closing = []
        with self._peer_lock:
            for cache_key in list(self._peer_streams):
                if cache_key[1] == owner_principal:
                    closing.append(self._peer_streams.pop(cache_key))
        for channel, req_q, _resp_iter in closing:
            try:
                req_q.put(None)
            except Exception:
                pass
            try:
                channel.close()
            except Exception:
                pass

    def push_stats(self) -> dict:
        with self._peer_lock:
            return {
                "active_peers": len(self._peer_streams),
                "push_count": self._push_count,
                "push_errors": self._push_errors,
                "evictions": self._peer_evictions,
            }

    def auth_stats(self) -> dict:
        """Phase B (2026-05-20): expose gRPC auth + SSRF defense counters
        for the /healthz endpoint. Useful for operators tracking attacker
        probes against a Mode-C deployment.

        2026-05-21 SPKI Phase 3.3+3.5: extended with outbound pin
        check counters. ``spki_*`` reflect refusals at the channel-
        open layer (separate from authz/ssrf which are inbound)."""
        return {
            "auth_required": bool(self.auth_required),
            "starting": bool(self.starting),
            "starting_reason": self.starting_reason,
            "authz_rejections": self._authz_rejections,
            "ssrf_rejections": self._ssrf_rejections,
            "refuse_unpinned_peers": bool(self.refuse_unpinned_peers),
            "spki_unpinned_refusals": self._spki_unpinned_refusals,
            "spki_mismatch_refusals": self._spki_mismatch_refusals,
            "spki_probe_failures": self._spki_probe_failures,
            "paid_admission": {
                "enabled": bool(self.paid_gate_enabled),
                "ready": bool(self.paid_admission is not None
                              and self.paid_admission.ready),
                "reason": (self.paid_admission.readiness_error
                           if self.paid_admission is not None else "disabled"),
            },
            "resolver": (
                self.peer_resolver.stats() if self.peer_resolver else None
            ),
        }

    def _check_grpc_auth(self, context, body_bytes: bytes, *,
                          method_path: str, is_streaming: bool) -> Optional[str]:
        """Phase B (2026-05-20): verify the gRPC call's authorization.

        Returns the verified keyid on success, or None when auth is not
        required (legacy Mode A bringup). Aborts the call with
        UNAUTHENTICATED on any failure — caller will not see a normal
        return when auth fails.
        """
        if not self.auth_required:
            return None
        if self.starting:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "worker starting: " + (self.starting_reason or "peer-key resolver not installed yet") + " (retry)",
            )
            return None  # unreachable; abort raises
        if not _GRPC_AUTH_AVAILABLE:
            self._authz_rejections += 1
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "auth required but nakshatra_grpc_auth module unavailable",
            )
            return None  # unreachable; abort raises
        if self.peer_resolver is None:
            self._authz_rejections += 1
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "auth required but no peer resolver configured",
            )
            return None
        metadata = dict(context.invocation_metadata() or [])
        auth_header = metadata.get("authorization", "")
        try:
            keyid = _wgrpcauth.verify_grpc_call(
                method_path, auth_header, body_bytes,
                pubkey_resolver=self.peer_resolver.resolve,
                is_streaming=is_streaming,
            )
            role = _wgrpcauth.ROLE_PEER
            resolve_identity = getattr(
                self.peer_resolver, "resolve_identity", None)
            if callable(resolve_identity):
                identity = resolve_identity(keyid)
                if identity is None:
                    raise _wgrpcauth.AuthError(f"unknown keyid: {keyid!r}")
                role = identity.role
                if (role == _wgrpcauth.ROLE_CONSUMER
                        and method_path not in _CONSUMER_INFERENCE_GRPC_METHODS):
                    raise _wgrpcauth.AuthError(
                        f"consumer keyid {keyid!r} is not authorized for "
                        f"{method_path}")
            return AuthenticatedKeyId(keyid, role)
        except _wgrpcauth.AuthError as e:
            self._authz_rejections += 1
            peer = "?"
            try:
                peer = context.peer()
            except Exception:
                pass
            _audit("auth_failure_grpc", ip=peer, reason=str(e),
                   method_path=method_path, is_streaming=is_streaming)
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED, f"authz failed: {e}"
            )
            return None  # unreachable

    @staticmethod
    def _session_principal(authenticated_keyid, session_id: str):
        """Bind every logical session to its authenticated caller."""
        if authenticated_keyid is None:
            return session_id
        return (str(authenticated_keyid), session_id)

    def _ensure_kv_state(self) -> None:
        """Initialize isolation state for narrow test/adaptor constructions."""
        if not hasattr(self, "_kv_gate_lock"):
            self._kv_gate_lock = threading.RLock()
            self._kv_epoch = 0
            self._kv_resident_principal = None

    def _acquire_kv_session(self, authenticated_keyid, session_id: str):
        self._ensure_kv_state()
        principal = self._session_principal(authenticated_keyid, session_id)
        with self._kv_gate_lock:
            if not self.daemon.acquire_session(principal):
                return None
            return principal

    def _release_kv_session(self, principal) -> None:
        if principal is None:
            return
        self._ensure_kv_state()
        with self._kv_gate_lock:
            self.daemon.release_session(principal)

    def _mark_cold_kv_replacement(self, principal) -> int:
        """Record a successful cold decode as the only resident seq-0 KV."""
        self._ensure_kv_state()
        with self._kv_gate_lock:
            self._kv_epoch += 1
            self._kv_resident_principal = principal
            return self._kv_epoch

    def _resident_epoch(self, principal) -> Optional[int]:
        self._ensure_kv_state()
        with self._kv_gate_lock:
            if self._kv_resident_principal != principal:
                return None
            return self._kv_epoch

    def _truncate_kv_if_owned(self, principal, n_keep: int):
        self._ensure_kv_state()
        with self._kv_gate_lock:
            if self.daemon.current_owner != principal:
                return None
            if self._kv_resident_principal != principal:
                return None
            return self.daemon.call(
                CMD_KV_TRUNCATE, 0, struct.pack("<I", n_keep))

    @staticmethod
    def _kv_lost_step(step):
        return pb.InferenceStep(
            session_id=step.session_id,
            step_id=step.step_id,
            error=(b"KV_RESIDENCY_LOST: cached step no longer owns resident KV; "
                   b"perform a cold replay with a new step_id"),
        )

    def Info(self, request, context):
        # Info is auth-exempt, so it is what a readiness probe can call. While the worker is still registering it must
        # NOT report healthy - a bound port + a TLS handshake was all a probe could see, and the first authenticated
        # request then hit "no peer resolver configured" (2026-09-21 cold start).
        challenge = bytes(getattr(request, "identity_challenge", b""))
        if challenge and len(challenge) != _INFO_CHALLENGE_BYTES:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "identity_challenge must be exactly 32 bytes")
        if self.starting:
            context.abort(grpc.StatusCode.UNAVAILABLE,
                          "worker starting: " + (self.starting_reason or "registration with the pillar not finished"))
        worker_keyid = ""
        identity_signature = b""
        if challenge:
            if not self._worker_identity_ready:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "worker Ed25519 identity is unavailable")
            worker_keyid = self.worker_pub_hex
            identity_signature = ed25519.Ed25519PrivateKey.from_private_bytes(
                self.worker_priv).sign(_INFO_ATTESTATION_DOMAIN + challenge)
        return pb.InfoResponse(
            protocol_version="0.1.0",
            backend="llamacpp-cpu-patched",
            model_id=self.model_id,
            model_content_hash=b"\x00" * 32,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            hidden_size=self.n_embd,
            wire_dtype="f32",
            kv_cache_tokens_free=256,
            has_token_embd=(self.mode in ("first", "solo")),
            has_lm_head=(self.mode in ("last", "solo")),
            worker_keyid=worker_keyid,
            identity_signature=identity_signature,
            # v0.5 §9.1 closure — advertised features so the client can
            # negotiate push at session start without runtime probing.
            # v1.0 §7 — append control/vN tokens so the client can negotiate
            # the control-protocol version on the same handshake.
            protocol_capabilities=[
                "streaming",
                "rpc_push",
                "idempotency_cache",
                "recovery_replay",
                # slice 1: this worker's daemon supports all_logits verify (flag 0x2)
                # + TruncateKV (cmd 4). The client only engages speculative decode
                # when EVERY worker in the chain advertises this.
                "speculative",
                # 2026-06-21: GPU sleep/wake (Sleep/Wake RPCs → daemon cmd 6/7)
                # and EAGLE→live draft-hidden (ForwardRequest.eagle_hidden → cmd 5).
                # Clients/lifecycle probe these before using them; absent on old workers.
                "sleep_wake",
                "eagle_hidden",
                # 2026-07-29: speculative decode ON THE STREAM — this worker's Inference
                # handler honours InferenceStep.all_logits (returns one argmax per input
                # position on mode=last, instead of only the final one). The client only
                # engages --stream-spec when EVERY worker in the chain advertises this;
                # an old worker silently ignoring the field would return a single token
                # where K+1 were expected, so absence must hard-refuse, never degrade quietly.
                "stream_spec",
                # Payment release peer-isolation capabilities. These are
                # enforced for peers even while paid admission remains off.
                "truncate_session_owner",
                "kv_residency_epoch_v1",
            ] + (list(_paid.PAID_CAPABILITIES)
                 if (self.paid_admission is not None
                     and self.paid_admission.ready) else [])
              + _control_version_caps(),
        )

    def _run_forward(self, hidden_in: bytes, n_tokens: int,
                     has_token_ids: bool, keep_kv: bool,
                     start_pos: int, all_logits: bool = False,
                     eagle_hidden: bool = False) -> "ForwardResult":
        """Shared daemon decode path — the transport-neutral core of a
        single forward step. Both the gRPC ``Forward`` RPC and the
        fabric backend (Phase D) call this so the two transports can
        never drift in their daemon-call semantics (cmd selection,
        size validation, rtype-prefix stripping).

        Returns a :class:`ForwardResult`. Does NOT touch gRPC context
        or fabric counters — the caller maps the result to its own
        transport's error reporting. KV semantics are the caller's to
        decide: it passes ``keep_kv`` + ``start_pos`` derived from the
        protobuf fields (gRPC) or the fabric header + per-chain state
        (fabric)."""
        flags = (0x1 if keep_kv else 0x0) | (0x2 if all_logits else 0x0)
        if eagle_hidden:
            # EAGLE→live: cmd=5 returns the captured pre-final hidden states
            # (float32 [n_tokens, 3, n_embd] = embd / l_out-0 / l_out-1) for the
            # draft head. First-worker path → input is token IDs. The payload is
            # NOT activation-quantized: the head consumes raw f32×3.
            if len(hidden_in) != n_tokens * 4:
                return ForwardResult(
                    False, b"", "eagle_hidden requires token_ids input "
                    f"(got {len(hidden_in)} bytes, expected {n_tokens * 4})",
                    client_error=True)
            status, resp = self.daemon.call(
                CMD_EAGLE_HIDDEN, n_tokens, hidden_in, start_pos=start_pos, flags=flags)
            if status != 0 or len(resp) < 4:
                return ForwardResult(
                    False, b"",
                    f"daemon eagle_hidden failed status={status} resp_len={len(resp)}",
                    client_error=False)
            return ForwardResult(True, resp[4:], "", client_error=False)
        if has_token_ids:
            if len(hidden_in) != n_tokens * 4:
                return ForwardResult(
                    False, b"", "hidden_in size mismatch for token_ids mode",
                    client_error=True)
            cmd = CMD_TOKEN_DECODE
        elif _ACT_QUANT:
            # #16: the incoming hidden state is an int8 wire blob → dequantize to f32 for the daemon.
            from act_quant import dequantize_int8, quant_blob_size
            exp = quant_blob_size(n_tokens, self.n_embd)
            if len(hidden_in) != exp:
                return ForwardResult(
                    False, b"", f"hidden_in size mismatch (int8): got {len(hidden_in)}, expected {exp}",
                    client_error=True)
            hidden_in = dequantize_int8(hidden_in, n_tokens, self.n_embd)
            cmd = CMD_EMBD_DECODE
        else:
            expected = n_tokens * self.n_embd * 4
            if len(hidden_in) != expected:
                return ForwardResult(
                    False, b"",
                    f"hidden_in size mismatch: got {len(hidden_in)}, "
                    f"expected {expected}",
                    client_error=True)
            cmd = CMD_EMBD_DECODE
        status, resp = self.daemon.call(
            cmd, n_tokens, hidden_in, start_pos=start_pos, flags=flags)
        if status != 0 or len(resp) < 4:
            return ForwardResult(
                False, b"",
                f"daemon decode failed status={status} resp_len={len(resp)}",
                client_error=False)
        # Strip the 4-byte rtype prefix; payload is hidden_state OR int32 token id.
        payload = resp[4:]
        # #16: a worker that emits a HIDDEN STATE (not the last worker's token) quantizes it for the wire.
        if _ACT_QUANT and self.mode not in ("last", "solo"):
            from act_quant import quantize_int8
            payload = quantize_int8(payload, n_tokens, self.n_embd)
        return ForwardResult(True, payload, "", client_error=False)

    def Forward(self, request, context):
        # Phase B (2026-05-20): authenticate the unary call. The signed
        # body is the serialized request — same canonical-string shape
        # as the HTTP side, just with method=POST + path=gRPC method name.
        authenticated_keyid = self._check_grpc_auth(
            context, request.SerializeToString(),
            method_path="/nakshatra.Nakshatra/Forward",
            is_streaming=False,
        )
        caller_role = getattr(authenticated_keyid, "role", None)
        paid_admission = getattr(self, "paid_admission", None)
        fabric_transport_enabled = getattr(
            self, "fabric_transport_enabled", False)
        # worker/kv-session-ownership-gate (2026-09-12): legacy Forward has no
        # required session_id (paid Forward now does), but it still calls into
        # the SAME daemon/KV state as an Inference stream. Without this, a
        # Forward call landing mid-stream could interleave a turn into
        # another session's seq-0 KV cache exactly like two Inference streams
        # could. Claim a synthetic, single-use session for just this call —
        # refused (fail closed) if a real session currently owns the daemon,
        # released immediately after so it never blocks anyone else.
        explicit_session = (request.session_id or "").strip()
        forward_session_id = explicit_session or (
            f"forward:{uuid.uuid4()}" if authenticated_keyid is None
            else "forward-legacy")
        forward_principal = self._session_principal(
            authenticated_keyid, forward_session_id)
        paid_claim = None
        paid_session_lock = None
        paid_committed = False
        paid_selected = bool(request.spend_claim_token)
        if (paid_admission is not None and authenticated_keyid is not None
                and explicit_session):
            paid_selected = paid_selected or paid_admission.store.get_claim(
                str(authenticated_keyid), explicit_session) is not None
        paid_selected = paid_selected or caller_role == _wgrpcauth.ROLE_CONSUMER
        if paid_selected:
            try:
                if paid_admission is None or not paid_admission.ready:
                    raise _paid.PaidAdmissionUnavailable(
                        "paid admission is not ready")
                if authenticated_keyid is None:
                    raise _paid.PaidAdmissionError(
                        "paid admission requires an authenticated caller")
                if (not explicit_session
                        or len(explicit_session.encode("utf-8")) > 256):
                    raise _paid.PaidAdmissionError(
                        "paid Forward requires a bounded session_id")
                if (not request.step_id
                        or len(request.step_id.encode("utf-8")) > 256):
                    raise _paid.PaidAdmissionError(
                        "paid Forward requires a bounded step_id")
                if (request.keep_kv or int(request.start_pos) != 0
                        or request.all_logits or request.eagle_hidden
                        or self.fabric_first_worker_bridge is not None
                        or fabric_transport_enabled):
                    raise _paid.PaidAdmissionError(
                        "paid Forward must be stateless, cold, direct, greedy, "
                        "and outside the fabric transport")
                n_tokens = int(request.n_tokens)
                if n_tokens <= 0:
                    raise _paid.PaidAdmissionError(
                        "paid Forward requires at least one input token")
                if request.has_token_ids:
                    valid_size = len(request.hidden_in) == n_tokens * 4
                elif _ACT_QUANT:
                    from act_quant import quant_blob_size
                    valid_size = len(request.hidden_in) == quant_blob_size(
                        n_tokens, self.n_embd)
                else:
                    valid_size = (
                        len(request.hidden_in) == n_tokens * self.n_embd * 4)
                if not valid_size:
                    raise _paid.PaidAdmissionError(
                        "paid Forward input dimensions are invalid")
                paid_claim = paid_admission.claim(
                    str(authenticated_keyid), explicit_session,
                    request.spend_claim_token)
                paid_session_lock = paid_admission.session_lock(paid_claim)
                paid_session_lock.acquire()
                request_sha256 = paid_admission.request_digest(request)
                duplicate, paid_prompt_delta, paid_generated_delta = (
                    paid_admission.prepare_step(
                        paid_claim, request.step_id, request_sha256, n_tokens))
                if duplicate:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details(
                        "KV_RESIDENCY_LOST: durable paid step requires cold replay")
                    paid_session_lock.release()
                    paid_session_lock = None
                    return pb.ForwardResponse()
            except _paid.PaidExecutionDeadline as exc:
                if paid_session_lock is not None:
                    paid_session_lock.release()
                    paid_session_lock = None
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details(str(exc))
                return pb.ForwardResponse()
            except _paid.PaidAdmissionError as exc:
                if paid_session_lock is not None:
                    paid_session_lock.release()
                    paid_session_lock = None
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details(str(exc))
                return pb.ForwardResponse()
            except Exception:
                if paid_session_lock is not None:
                    paid_session_lock.release()
                    paid_session_lock = None
                raise
        # A stateful Forward may continue only the KV currently resident for
        # this exact authenticated principal. Refuse before any daemon call;
        # clearing another principal and then decoding with keep_kv would be
        # equally unsafe.
        if (request.keep_kv and authenticated_keyid is not None
                and self._resident_epoch(forward_principal) is None):
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(
                "KV_RESIDENCY_LOST: stateful Forward does not own resident KV")
            if paid_session_lock is not None:
                paid_session_lock.release()
            return pb.ForwardResponse()
        acquired_principal = None
        try:
            acquired_principal = self._acquire_kv_session(
                authenticated_keyid, forward_session_id)
        except DaemonUnavailableError as e:
            if paid_session_lock is not None:
                paid_session_lock.release()
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(str(e))
            return pb.ForwardResponse()
        except Exception:
            if paid_session_lock is not None:
                paid_session_lock.release()
            raise
        if acquired_principal is None:
            if paid_session_lock is not None:
                paid_session_lock.release()
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(
                f"worker busy serving another session "
                f"(owner={self.daemon.current_owner!r})"
            )
            return pb.ForwardResponse()
        try:
            try:
                result = self._run_forward(
                    request.hidden_in, request.n_tokens, request.has_token_ids,
                    request.keep_kv, int(request.start_pos), request.all_logits,
                    eagle_hidden=request.eagle_hidden,
                )
                if result.ok and not request.keep_kv:
                    forward_kv_epoch = self._mark_cold_kv_replacement(
                        forward_principal)
                    if paid_claim is not None:
                        paid_admission.commit_step(
                            paid_claim, request.step_id, request_sha256,
                            paid_prompt_delta, paid_generated_delta,
                            forward_kv_epoch)
                        paid_committed = True
                        paid_admission.flush_outbox()
            except DaemonUnavailableError as e:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details(str(e))
                return pb.ForwardResponse()
            except _paid.PaidExecutionDeadline as e:
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details(str(e))
                return pb.ForwardResponse()
            except _paid.PaidAdmissionError as e:
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details(str(e))
                return pb.ForwardResponse()
        finally:
            self._release_kv_session(acquired_principal)
            try:
                if paid_claim is not None and paid_committed:
                    paid_admission.finalize(paid_claim, "normal")
                    paid_admission.flush_outbox()
            finally:
                if paid_session_lock is not None:
                    paid_session_lock.release()
        if not result.ok:
            context.set_code(
                grpc.StatusCode.INVALID_ARGUMENT if result.client_error
                else grpc.StatusCode.INTERNAL)
            context.set_details(result.error)
            return pb.ForwardResponse()
        # 2026-09-13 participation binding: legacy Forward cannot be bound to
        # an output_sha256. Paid Forward has stable session/step IDs, but its
        # participation is the narrowly scoped usage receipt signed above;
        # consumers remain unable to call the peer SignParticipation RPC. An
        # earlier version of this comment described
        # recording a bare timestamp here as evidence for a SignParticipation fallback; that
        # fallback accepted ANY (run_id, output_sha256) pair as long as recent Forward
        # activity existed, with no check that this node produced THAT run or THAT output —
        # i.e. it signed exactly the forged proofs it was meant to refuse. It has been
        # deleted (see SignParticipation's docstring) rather than narrowed, because Forward
        # activity is fundamentally unable to supply the missing correlation: there is no
        # per-call identifier here to check against. Deliberately a no-op now — kept as a
        # comment, not a stub, so a future edit does not "restore" the removed timestamp.
        # 2026-05-29 fabric Phase F — first-worker gRPC→fabric bridge.
        # When the bridge is wired (mode=first + --transport=fabric),
        # the gRPC reply carries the chain's FINAL token (after the
        # fabric round-trip through mid + last workers) instead of
        # this worker's intermediate hidden_state. Mid/last workers
        # never have the bridge wired and keep returning hidden_state
        # / token directly per the legacy path.
        if self.fabric_first_worker_bridge is not None and not request.eagle_hidden:
            token_bytes = self.fabric_first_worker_bridge(
                result.payload, step_id=int(request.start_pos),
                layer_idx=self.layer_end,
            )
            if token_bytes is None:
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details(
                    "fabric chain timed out before FEEDBACK arrived")
                return pb.ForwardResponse()
            return pb.ForwardResponse(hidden_out=token_bytes)
        return pb.ForwardResponse(hidden_out=result.payload)

    def TruncateKV(self, request, context):
        # Speculative decoding (slice 1): drop a rejected speculative tail from the
        # daemon's KV (cmd=4 → llama_memory_seq_rm, keep [0, n_keep)). The coordinator
        # fans this out to EVERY worker after accept(); a missed worker would silently
        # corrupt the next step's attention with a stale KV tail (no error raised).
        authenticated_keyid = self._check_grpc_auth(
            context, request.SerializeToString(),
            method_path="/nakshatra.Nakshatra/TruncateKV",
            is_streaming=False,
        )
        session_id = (request.session_id or "").strip()
        if session_id:
            principal = self._session_principal(
                authenticated_keyid, session_id)
        else:
            # Old-client compatibility: infer only from the one active owner,
            # and only for an authenticated peer matching that owner. A
            # consumer never gets this compatibility path.
            principal = self.daemon.current_owner
            role = getattr(authenticated_keyid, "role", None)
            if authenticated_keyid is not None:
                if (role != _wgrpcauth.ROLE_PEER
                        or not isinstance(principal, tuple)
                        or principal[0] != str(authenticated_keyid)):
                    principal = None
            if principal is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(
                    "session_id is required unless the authenticated peer is "
                    "the sole active KV owner")
                return pb.TruncateResponse()
        n_keep = int(request.n_keep)
        if n_keep < 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"n_keep must be >= 0, got {n_keep}")
            return pb.TruncateResponse()
        result = self._truncate_kv_if_owned(principal, n_keep)
        if result is None:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details("caller does not own the active resident KV session")
            return pb.TruncateResponse()
        status, _ = result
        if status != 0:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"daemon KV truncate failed status={status}")
        return pb.TruncateResponse()

    def Sleep(self, request, context):
        # sleep-mode (2026-06-21): release the GPU (daemon cmd=6 frees ctx+model;
        # the process + ROCm backend stay resident). Idempotent. The lifecycle
        # calls this on idle-grace instead of a full reap — wake is ~440ms vs ~7.5s.
        self._check_grpc_auth(
            context, request.SerializeToString(),
            method_path="/nakshatra.Nakshatra/Sleep", is_streaming=False)
        status, _ = self.daemon.call(CMD_SLEEP, 0, b"")
        if status != 0:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"daemon sleep failed status={status}")
        return pb.SleepResponse()

    def Wake(self, request, context):
        # sleep-mode (2026-06-21): re-acquire the GPU (daemon cmd=7 reloads from
        # the warm page cache). Idempotent (no-op + 0s if already awake).
        self._check_grpc_auth(
            context, request.SerializeToString(),
            method_path="/nakshatra.Nakshatra/Wake", is_streaming=False)
        t0 = time.time()
        status, _ = self.daemon.call(CMD_WAKE, 0, b"")
        if status != 0:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"daemon wake failed status={status}")
            return pb.WakeResponse()
        return pb.WakeResponse(wake_seconds=time.time() - t0)

    def SignParticipation(self, request, context):
        """Sign, with this node's OWN mesh key, the stage this worker actually served.

        ⚠️⚠️ IT SIGNS WHAT IT BELIEVES, NOT WHAT IT WAS ASKED. The request carries the
        coordinator's idea of this worker's span; the response carries THIS WORKER'S. If they
        disagree, the mismatch surfaces in `signature_coverage` as a gap — which is the
        detector that exists for exactly this. A worker that echoed the requested span would
        turn its key into a rubber stamp and make the whole scheme decorative.

        ⚠️ LOADS THE KEY, NEVER CREATES ONE. `load_or_create_worker_key` would mint a fresh
        identity if the file were missing — and a new key here means everything earned under
        the old one becomes unreachable, silently, in the middle of a signing call. A node
        with no key cannot be paid; that is a FAILED_PRECONDITION for an operator to fix, not
        something to paper over by inventing an identity.

        ⚠️⚠️ 2026-09-12 — IT ALSO REFUSES TO SIGN WORK IT NEVER SAW. Before this, `run_id` and
        `output_sha256` were taken ENTIRELY from the caller with zero correlation to any real
        `Forward`/`Inference` call this worker served — the layer-span check above stops a
        coordinator from putting words in this worker's mouth about WHICH layers, but nothing
        stopped ANY authenticated peer from calling `SignParticipation(run_id="anything",
        output_sha256="anything")` cold and getting a validly-signed fake proof. There is no
        run_id on the wire for the (default) unary Forward path — the proto documents Forward
        as a stateless testing aid — so the real correlator is `session_id`, which the
        streaming Inference RPC already carries and which client.py already uses AS run_id
        (`run_id = session_id or uuid4()`, see client.py). `_record_session_activity` is
        populated ONLY by Inference() actually running a decode; this method only ever reads
        it, in `_participation_evidence`.

        ⚠️⚠️⚠️ 2026-09-13 — THE 2026-09-12 FIX STILL HAD A HOLE, AND IT HAS NOW BEEN REMOVED
        RATHER THAN PATCHED AGAIN. The 09-12 version kept a "sessionless fallback" for a
        never-served run_id: if this node had ANY recent Forward activity (`_last_activity_ts`
        within `_served_ttl`, ~15 min), it signed the request's `output_sha256` UNCHECKED —
        Forward carries no verifiable run output, so that branch could not verify the
        request's `run_id` or `output_sha256` against anything real. The exploit: call Forward
        once (any cheap, unrelated request stamps the timestamp), then call
        SignParticipation(run_id="<anything never served>", output_sha256="<anything>") within
        15 minutes and receive a validly-signed proof for fabricated work. That fallback has
        been deleted, not narrowed — see the comment where `_last_activity_ts` used to be set,
        in Forward(), for why it cannot be repaired: Forward has no verified full-run output
        to bind to a specific output_sha256, so ANY signal derived from "Forward ran
        recently" is a proxy for activity, never for the truth of a specific (run_id,
        output_sha256) pair — which is the one thing this method must not guess at.

        The invariant now: SignParticipation signs a `(run_id, output_sha256)` pair ONLY when
        `_participation_evidence(run_id)` shows this worker's OWN `Inference()` handler
        actually recorded that run_id, AND — whenever this worker recorded the tokens it
        itself produced as the run's final-token producer — the claimed `output_sha256`
        reproduces the exact hash of those tokens. No other path signs. Concretely:
          1. This run_id was never served by this node's own Inference() at all → refuse,
             unconditionally. No fallback of any kind, no matter how recent other activity is.
          2. This run_id WAS served and this worker was the one producing the final tokens
             (mode="last") → the claimed output_sha256 must match what this worker actually
             emitted, byte for byte (`receipt.output_sha256` over the same token sequence).
             A worker cannot be tricked into vouching for content it never produced.
          3. This run_id was served but this worker only ever forwarded hidden state (not
             last) → session-level evidence (this worker really processed this run_id) is the
             strongest this worker can independently judge, since it never held the final
             tokens to hash; allowed, same as before, but only for a run_id it actually saw.

        This makes the unary Forward RPC permanently unable to obtain a SignParticipation
        proof for any run_id — by design, not by omission. Forward is a stateless
        single-forward-pass aid; even paid Forward's session/step identifiers do not
        carry a verified full-run output, so no amount of "recent
        activity" should ever be allowed to substitute for it. A worker that only ever
        served a run_id through Forward — which, by construction, never happens, since
        Forward is never used as a peer-participation evidence key — has, correctly,
        no way to get a signature for it. Only the streaming Inference() path can ever
        produce genuine evidence.
        """
        authenticated_keyid = self._check_grpc_auth(
            context, request.SerializeToString(),
            method_path="/nakshatra.Nakshatra/SignParticipation", is_streaming=False)

        key_path = _wauth.WORKER_KEY_PATH
        if not key_path.exists():
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(f"no worker identity at {key_path}; this node can serve but "
                                "cannot be credited — run the join/registration step")
            return pb.SignParticipationResponse()
        priv = key_path.read_bytes()
        if len(priv) != 32:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("worker key is malformed; refusing to sign with it")
            return pb.SignParticipationResponse()

        run_id = (request.run_id or "").strip()
        out_sha = (request.output_sha256 or "").strip()
        if not run_id or not out_sha:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("run_id and output_sha256 are both required")
            return pb.SignParticipationResponse()

        seen, generated_token_ids = self._participation_evidence(
            run_id, authenticated_keyid)
        if not seen:
            # 2026-09-13: NO fallback here, sessionless or otherwise. This run_id was never
            # recorded by this node's own Inference() handler, so there is nothing to verify
            # the request against — signing anyway (even gated on "recent Forward activity",
            # the previous version's mistake) means vouching for a run this node cannot prove
            # it did. See the docstring's 2026-09-13 section for the exploit this closes.
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(
                f"run_id {run_id!r} was never served by this node's streaming Inference "
                "path — refusing to sign work it cannot independently verify (no sessionless "
                "fallback; unary Forward cannot supply a verified run output for this "
                "evidence)")
            return pb.SignParticipationResponse()
        elif generated_token_ids:
            # This worker WAS the run's final-token producer (mode=="last") for this exact
            # session — it can independently recompute the same hash the client claims.
            from receipt import output_sha256 as _output_sha256
            observed = _output_sha256(generated_token_ids)
            if observed != out_sha:
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details(
                    f"output_sha256 {out_sha!r} does not match what this node actually "
                    f"produced for run_id {run_id!r} ({observed!r}) — refusing to sign")
                return pb.SignParticipationResponse()

        from identity_binding import pub_of, sign_participation
        node_id = os.environ.get("NAKSHATRA_NODE_ID") or socket.gethostname()
        priv_hex = priv.hex()
        entry = sign_participation(priv_hex, run_id=run_id, node_id=node_id,
                                   layer_start=int(self.layer_start),
                                   layer_end=int(self.layer_end),
                                   output_sha256=out_sha)
        return pb.SignParticipationResponse(
            node_id=node_id, pubkey=pub_of(priv_hex),
            layer_start=int(self.layer_start), layer_end=int(self.layer_end),
            sig=entry["sig"])

    def Inference(self, request_iterator, context):
        """v0.5 M0.5.1 — streaming inference RPC.

        Per-stream session: the first step decodes with KV-cache cleared, every
        subsequent step keeps the KV cache and advances start_pos. Mirrors
        Forward's daemon-call behaviour exactly; this is a transport-layer
        change, not a semantic one.

        Routes each step to the daemon based on payload type:
          token_ids    -> CMD_TOKEN_DECODE (typical first-worker input)
          hidden_state -> CMD_EMBD_DECODE  (middle/last input)

        Responds with the typed payload appropriate for this worker's mode:
          mode=last  -> token_ids (a single sampled token id)
          otherwise  -> hidden_state (one vector per input token)
        """
        first_step = True
        # worker/kv-session-ownership-gate (2026-09-12): set once this stream
        # has claimed the daemon, so `finally` below knows whose ownership to
        # release. Left None if we never acquire (e.g. refused, or the stream
        # errors before the first frame) so we never release someone else's.
        owned_principal = None
        session_principal = None
        stream_session_id = None
        authenticated_keyid = None
        stream_kv_epoch = None
        paid_claim = None
        paid_session_lock = None
        paid_termination = "cancelled"
        paid_admission = getattr(self, "paid_admission", None)
        fabric_transport_enabled = getattr(
            self, "fabric_transport_enabled", False)
        try:
            for step in _iter_with_idle_timeout(
                request_iterator, INFERENCE_STREAM_IDLE_TIMEOUT_S
            ):
                # Phase B (2026-05-20): authenticate the first frame.
                # Once the first frame's signature is accepted, gRPC
                # stream identity binds subsequent frames to the same
                # authenticated peer — no per-frame check needed.
                if first_step:
                    authenticated_keyid = self._check_grpc_auth(
                        context, step.SerializeToString(),
                        method_path="/nakshatra.Nakshatra/Inference",
                        is_streaming=True,
                    )
                    stream_session_id = (step.session_id or "").strip()
                    if not stream_session_id or len(stream_session_id.encode()) > 256:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details(
                            "session_id is required and must not exceed 256 bytes")
                        yield pb.InferenceStep(
                            session_id=step.session_id, step_id=step.step_id,
                            error=b"invalid session_id",
                        )
                        return
                    session_principal = self._session_principal(
                        authenticated_keyid, stream_session_id)
                    caller_role = getattr(authenticated_keyid, "role", None)
                    token_present = bool(step.spend_claim_token)
                    existing_paid = None
                    if (paid_admission is not None
                            and authenticated_keyid is not None):
                        existing_paid = paid_admission.store.get_claim(
                            str(authenticated_keyid), stream_session_id)
                    paid_selected = (
                        caller_role == _wgrpcauth.ROLE_CONSUMER
                        or token_present or existing_paid is not None)
                    if paid_selected:
                        # Direct-only is checked before the online claim so a
                        # forbidden topology cannot burn a claim or touch KV.
                        if (paid_admission is None
                                or not paid_admission.ready):
                            raise _paid.PaidAdmissionUnavailable(
                                "paid admission is not ready")
                        if (step.chain or step.next_server.address or step.pushed
                                or step.all_logits
                                or self.fabric_first_worker_bridge is not None
                                or fabric_transport_enabled):
                            raise _paid.PaidAdmissionError(
                                "paid sessions require direct greedy inference; "
                                "chain, next_server, push, speculation, and fabric are forbidden")
                        paid_claim = paid_admission.claim(
                            str(authenticated_keyid), stream_session_id,
                            step.spend_claim_token)
                        paid_session_lock = paid_admission.session_lock(paid_claim)
                        paid_session_lock.acquire()
                    else:
                        # Peers acquire immediately. Paid requests defer this
                        # until every cap/schema rejection has passed, ensuring
                        # no rejected payment can wake or call the daemon.
                        owned_principal = self._acquire_kv_session(
                            authenticated_keyid, stream_session_id)
                    if paid_claim is None and owned_principal is None:
                        busy_msg = (
                            f"worker busy serving another session "
                            f"(owner={self.daemon.current_owner!r})"
                        ).encode()
                        context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                        context.set_details(busy_msg.decode())
                        yield pb.InferenceStep(
                            session_id=step.session_id, step_id=step.step_id,
                            error=busy_msg,
                        )
                        return
                elif step.session_id != stream_session_id:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "session_id must remain stable within an Inference stream")
                    yield pb.InferenceStep(
                        session_id=step.session_id, step_id=step.step_id,
                        error=(b"session_id must remain stable within an "
                               b"Inference stream"),
                    )
                    return
                elif paid_claim is not None:
                    if step.spend_claim_token:
                        raise _paid.PaidAdmissionError(
                            "spend_claim_token is allowed only on the first paid frame")
                    if (step.chain or step.next_server.address or step.pushed
                            or step.all_logits
                            or self.fabric_first_worker_bridge is not None
                            or fabric_transport_enabled):
                        raise _paid.PaidAdmissionError(
                            "paid sessions require direct greedy inference; "
                            "chain, next_server, push, speculation, and fabric are forbidden")
                if (not step.step_id
                        or len(step.step_id.encode("utf-8")) > 256):
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "step_id is required and must not exceed 256 bytes")
                    yield pb.InferenceStep(
                        session_id=step.session_id, step_id=step.step_id,
                        error=b"invalid step_id",
                    )
                    return
                # v0.5 M0.5.2: idempotency cache. If this (session_id, step_id) has
                # been served before, return the cached response without touching
                # the daemon. Note: a cache hit advances first_step too — the
                # daemon's KV state was already updated by the original call, so
                # subsequent steps in this stream should treat themselves as
                # "not first" with respect to keep_kv semantics.
                digest_step = pb.InferenceStep()
                digest_step.CopyFrom(step)
                digest_step.spend_claim_token = ""
                request_sha256 = hashlib.sha256(
                    digest_step.SerializeToString()).hexdigest()
                try:
                    cached = self._idem_get(
                        session_principal, step.step_id, request_sha256)
                except IdempotencyConflict as exc:
                    context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                    context.set_details(str(exc))
                    yield pb.InferenceStep(
                        session_id=step.session_id, step_id=step.step_id,
                        error=str(exc).encode())
                    return
                if cached is not None:
                    if paid_claim is not None and owned_principal is None:
                        owned_principal = self._acquire_kv_session(
                            authenticated_keyid, stream_session_id)
                        if owned_principal is None:
                            raise _paid.PaidAdmissionError(
                                "worker busy serving another session")
                    resident_epoch = self._resident_epoch(session_principal)
                    if (resident_epoch is None
                            or resident_epoch != cached["kv_epoch"]):
                        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                        context.set_details(
                            "KV_RESIDENCY_LOST: cold replay required")
                        yield self._kv_lost_step(step)
                        return
                    mark_served = getattr(self.daemon, "mark_step_served", None)
                    if mark_served is not None:
                        mark_served()
                    first_step = False
                    stream_kv_epoch = resident_epoch
                    yield cached["response"]
                    continue

                if step.HasField("token_ids"):
                    ids = list(step.token_ids.ids)
                    n_tokens = len(ids)
                    if n_tokens == 0:
                        yield pb.InferenceStep(
                            session_id=step.session_id, step_id=step.step_id,
                            error=b"empty token_ids payload",
                        )
                        return
                    input_bytes = struct.pack(f"<{n_tokens}i", *ids)
                    cmd = CMD_TOKEN_DECODE
                elif step.HasField("hidden_state"):
                    hs = step.hidden_state
                    n_tokens = hs.n_tokens
                    # act-quant on the wire (streaming): the incoming hidden is int8 (quant_blob);
                    # dequantize to f32 for the daemon. Size check uses the int8 blob size.
                    if _ACT_QUANT:
                        from act_quant import quant_blob_size, dequantize_int8
                        expected = quant_blob_size(n_tokens, self.n_embd)
                    else:
                        expected = n_tokens * self.n_embd * 4
                    if len(hs.raw) != expected:
                        yield pb.InferenceStep(
                            session_id=step.session_id, step_id=step.step_id,
                            error=f"hidden_state size mismatch: got {len(hs.raw)}, expected {expected}".encode(),
                        )
                        return
                    input_bytes = dequantize_int8(hs.raw, n_tokens, self.n_embd) if _ACT_QUANT else hs.raw
                    cmd = CMD_EMBD_DECODE
                else:
                    yield pb.InferenceStep(
                        session_id=step.session_id, step_id=step.step_id,
                        error=b"InferenceStep payload must be token_ids or hidden_state",
                    )
                    return

                paid_prompt_delta = 0
                paid_generated_delta = 0
                if paid_claim is not None:
                    duplicate, paid_prompt_delta, paid_generated_delta = (
                        paid_admission.prepare_step(
                            paid_claim, step.step_id, request_sha256, n_tokens))
                    if duplicate:
                        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                        context.set_details(
                            "KV_RESIDENCY_LOST: durable paid step requires cold replay")
                        yield self._kv_lost_step(step)
                        return
                    paid_meter = paid_admission.store.meter(
                        paid_claim["payer_keyid"], paid_claim["session_id"])
                    if (int(paid_meter["receipt_sequence"]) > 0
                            and self._resident_epoch(session_principal) is None):
                        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                        context.set_details(
                            "KV_RESIDENCY_LOST: paid session cannot continue "
                            "after resident KV was lost")
                        yield self._kv_lost_step(step)
                        return
                    if owned_principal is None:
                        owned_principal = self._acquire_kv_session(
                            authenticated_keyid, stream_session_id)
                        if owned_principal is None:
                            raise _paid.PaidAdmissionError(
                                "worker busy serving another session")

                # 2026-07-29 stream-spec: OR in the same 0x2 all_logits bit the unary
                # Forward path sets (_run_forward above) — one daemon.call primitive,
                # two transports. Default false -> flags unchanged -> byte-identical.
                cold_decode = first_step
                if not cold_decode and self._resident_epoch(session_principal) is None:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details("KV_RESIDENCY_LOST: cold replay required")
                    yield self._kv_lost_step(step)
                    return
                flags = (0x0 if cold_decode else 0x1) | (0x2 if step.all_logits else 0x0)
                first_step = False
                # Phase D3 (2026-05-20): bound prefix_length. Proto
                # already enforces int type, but unbounded value
                # propagates to the daemon which may panic or wedge.
                # n_ctx is the worker's context cap; clamp to it.
                if _VALIDATION_AVAILABLE:
                    bounded_prefix = _wval.as_safe_int(
                        step.prefix_length, default=0, lo=0, hi=1 << 20
                    )
                else:
                    bounded_prefix = max(0, min(int(step.prefix_length), 1 << 20))
                status, resp = self.daemon.call(
                    cmd, n_tokens, input_bytes,
                    start_pos=bounded_prefix, flags=flags,
                )
                if status != 0 or len(resp) < 4:
                    paid_termination = "daemon_error"
                    yield pb.InferenceStep(
                        session_id=step.session_id, step_id=step.step_id,
                        error=f"daemon decode failed status={status} resp_len={len(resp)}".encode(),
                    )
                    return
                if cold_decode:
                    stream_kv_epoch = self._mark_cold_kv_replacement(
                        session_principal)
                else:
                    stream_kv_epoch = self._resident_epoch(session_principal)
                    if stream_kv_epoch is None:
                        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                        context.set_details("KV_RESIDENCY_LOST: cold replay required")
                        yield self._kv_lost_step(step)
                        return
                # Daemon prefixes payload with rtype; Forward drops it, we do too.
                payload = resp[4:]

                # 2026-05-29 fabric: streaming-mode bridge for the
                # first worker. Same OQ8 shape as the unary Forward
                # bridge (see WorkerServicer.Forward) — when wired
                # AND mode=first, ship the local decode's hidden via
                # fabric, block for the chain's final token to come
                # back via FEEDBACK, and yield it on the gRPC stream
                # as token_ids. Skips the next_server / chain push
                # path below (fabric owns worker↔worker now).
                if (self.mode in ("first", "solo")
                        and self.fabric_first_worker_bridge is not None):
                    token_bytes = self.fabric_first_worker_bridge(
                        payload, step_id=step.step_id,
                        layer_idx=self.layer_end,
                    )
                    if token_bytes is None:
                        yield pb.InferenceStep(
                            session_id=step.session_id,
                            step_id=step.step_id,
                            error=b"fabric chain timed out before "
                                  b"FEEDBACK arrived",
                        )
                        return
                    token_id = struct.unpack("<i", token_bytes[:4])[0]
                    out = pb.InferenceStep(
                        session_id=step.session_id,
                        step_id=step.step_id,
                        prefix_length=step.prefix_length + n_tokens,
                    )
                    out.token_ids.ids.append(token_id)
                    # 2026-09-12 participation binding: this IS the fabric-bridged first
                    # worker's real output for this session — see SignParticipation.
                    self._record_session_activity(
                        step.session_id, token_id=token_id,
                        authenticated_keyid=authenticated_keyid)
                    self._idem_put(
                        session_principal, step.step_id, out, stream_kv_epoch,
                        request_sha256)
                    yield out
                    continue

                out = pb.InferenceStep(
                    session_id=step.session_id,
                    step_id=step.step_id,
                    prefix_length=step.prefix_length + n_tokens,
                )
                final_token_id = None
                if self.mode in ("last", "solo"):
                    if step.all_logits:
                        # stream-spec verify: one argmax per input position (a
                        # single-final-token response here would silently truncate
                        # a K+1-position verify to 1 — the capability gate in
                        # client.py is what makes this branch safe to take). NOT the
                        # run's accepted output — a verify pass, not a generation step —
                        # so it does not feed participation evidence below.
                        n_ids = len(payload) // 4
                        out.token_ids.ids.extend(struct.unpack(f"<{n_ids}i", payload))
                    else:
                        final_token_id = struct.unpack("<i", payload[:4])[0]
                        out.token_ids.ids.append(final_token_id)
                else:
                    if _ACT_QUANT:   # non-last: quantize the emitted hidden to int8 for the wire
                        from act_quant import quantize_int8
                        payload = quantize_int8(payload, n_tokens, self.n_embd)
                    out.hidden_state.raw = payload
                    out.hidden_state.batch = 1
                    out.hidden_state.n_tokens = n_tokens
                if paid_claim is not None:
                    # The usage journal and signed outbox entry are durable
                    # before any successful paid response can escape.
                    paid_admission.commit_step(
                        paid_claim, step.step_id, request_sha256,
                        paid_prompt_delta, paid_generated_delta,
                        stream_kv_epoch)
                    paid_admission.flush_outbox()
                # 2026-09-12 participation binding — see SignParticipation's docstring.
                # Records that THIS worker served THIS session (every mode), and — only for
                # a real accepted final token — the exact id, so `_participation_evidence`
                # can reconstruct the same sequence `receipt.output_sha256()` hashes.
                self._record_session_activity(
                    step.session_id, token_id=final_token_id,
                    authenticated_keyid=authenticated_keyid)

                # v0.5 M0.5.3: server-to-server push. If the incoming step
                # carries a next_server hint AND we produced a hidden_state
                # (non-terminal worker), open/reuse a peer stream and forward
                # our output downstream instead of returning it to our caller.
                # We then wait for the peer's response and yield THAT back, so
                # the response chain unwinds up the call graph.
                # v0.5 M0.5.3 push: decide whether to forward our output
                # downstream. Prefer the v2 `chain` field; fall back to v1's
                # `next_server` for back-compat.
                next_addr = ""
                next_session = ""
                remaining_chain = []
                # Phase D3 (2026-05-20): address-length cap (256 bytes).
                # Without this, unbounded address strings can balloon
                # _peer_streams (already LRU-capped per A3, but the
                # individual entry size also matters). Mirror Sthambha
                # O3 MAX_PEER_ADDRESS_BYTES.
                _ADDR_MAX = 256
                if step.chain:
                    raw_addr = step.chain[0].address
                    if _VALIDATION_AVAILABLE:
                        next_addr = _wval.as_bounded_str(raw_addr, _ADDR_MAX, default="")
                    else:
                        next_addr = raw_addr if len(raw_addr.encode()) <= _ADDR_MAX else ""
                    next_session = step.chain[0].session_id
                    remaining_chain = list(step.chain[1:])
                elif step.next_server.address:
                    raw_addr = step.next_server.address
                    if _VALIDATION_AVAILABLE:
                        next_addr = _wval.as_bounded_str(raw_addr, _ADDR_MAX, default="")
                    else:
                        next_addr = raw_addr if len(raw_addr.encode()) <= _ADDR_MAX else ""
                    next_session = step.next_server.session_id
                if next_addr and self.mode not in ("last", "solo"):
                    # Phase B5 (2026-05-20): SSRF defense. Worker pushes
                    # only to peers the pillar has registered. An attacker-
                    # supplied next_addr to internal-only endpoints (e.g.
                    # "127.0.0.1:22") would otherwise turn the worker into
                    # a probe. refuse_unregistered_peers=False keeps the
                    # legacy behavior for Mode-A clusters with no pillar.
                    if (self.refuse_unregistered_peers
                            and self.peer_resolver is not None
                            and not self.peer_resolver.is_registered_address(next_addr)):
                        self._ssrf_rejections += 1
                        err_msg = (
                            f"push_failed: refusing unregistered peer "
                            f"address {next_addr!r} (Mode-C SSRF defense; "
                            f"set NAKSHATRA_REFUSE_UNREGISTERED_PEERS=false "
                            f"to disable)"
                        ).encode()
                        yield pb.InferenceStep(
                            session_id=step.session_id, step_id=step.step_id,
                            error=err_msg,
                        )
                        continue
                    # The next worker processes the SAME token positions we
                    # just did — it's running a different layer range over the
                    # same sequence. So pushed prefix_length is our INPUT
                    # prefix_length, not our advanced output prefix_length.
                    # (Bug found 2026-05-13 on first M0.5.3 push run: node-d
                    # daemon rejected with "sequence positions inconsistent"
                    # when we advanced by n_tokens here.)
                    push_step = pb.InferenceStep(
                        session_id=next_session or step.session_id,
                        step_id=step.step_id,
                        prefix_length=step.prefix_length,
                        pushed=True,
                    )
                    push_step.hidden_state.raw = payload
                    push_step.hidden_state.batch = 1
                    push_step.hidden_state.n_tokens = n_tokens
                    # v2: forward the rest of the chain so the next worker
                    # knows who to push to AFTER it. v1 next_server callers
                    # send no chain; remaining_chain stays empty.
                    if remaining_chain:
                        push_step.chain.extend(remaining_chain)
                    try:
                        req_q, resp_iter = self._get_peer_stream(
                            next_addr, session_principal)
                        req_q.put(push_step)
                        peer_resp = next(resp_iter)
                        with self._peer_lock:
                            self._push_count += 1
                    except Exception as e:
                        with self._peer_lock:
                            self._push_errors += 1
                            # Evict the broken peer stream so a future session
                            # opens a fresh one instead of reusing the wedged
                            # channel. The downgraded session won't use push
                            # again, but later sessions might.
                            cached = self._peer_streams.pop(
                                (next_addr, session_principal), None)
                            if cached is not None:
                                try: cached[1].put(None)  # close request side
                                except Exception: pass
                                try: cached[0].close()    # close channel
                                except Exception: pass
                        sys.stderr.write(f"[push] to {next_addr!r} failed: {e}\n")
                        # v0.5 §9.5 closure: emit a structured "push_failed:"
                        # error so the client can downgrade this session to
                        # client-relay mode. Yielding our raw hidden state
                        # here would corrupt the chain (in push mode the
                        # client expects a final token id, not a mid-chain
                        # hidden state).
                        err_msg = f"push_failed: peer={next_addr} error={e}".encode()
                        yield pb.InferenceStep(
                            session_id=step.session_id, step_id=step.step_id,
                            error=err_msg,
                        )
                        continue
                    # Cache the peer's response under OUR (session_id, step_id)
                    # so a replay of our step returns the same final result.
                    self._idem_put(
                        session_principal, step.step_id, peer_resp,
                        stream_kv_epoch, request_sha256)
                    yield peer_resp
                    continue

                # v0.5 M0.5.2: store before yielding so a concurrent duplicate
                # gets the cached response. Only cache fully-formed successful
                # responses; error responses are not cached (the error was the
                # whole point of failure, replays should retry).
                self._idem_put(
                    session_principal, step.step_id, out, stream_kv_epoch,
                    request_sha256)
                yield out
            paid_termination = "normal"
        except _paid.PaidExecutionDeadline as e:
            paid_termination = "execution_deadline"
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(str(e))
            return
        except _paid.PaidAdmissionError as e:
            paid_termination = "cancelled"
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(str(e))
            return
        except DaemonUnavailableError as e:
            paid_termination = "daemon_error"
            sys.stderr.write(f"[inference] daemon unavailable: {e}\n")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(str(e))
            return
        except TimeoutError as e:
            paid_termination = "idle_timeout"
            sys.stderr.write(f"[inference] idle timeout: {e}\n")
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(str(e))
            return
        except Exception as e:
            paid_termination = "daemon_error"
            # context.abort() signals by raising a bare Exception AFTER it set the status. Rewriting that into INTERNAL
            # threw away UNAUTHENTICATED/UNAVAILABLE and left the client an unactionable "Inference stream error: ".
            if _grpc_call_aborted(context):
                raise
            # type + traceback: str(e) is empty for queue.Empty / bare EOFError, which made a cold-start abort
            # undiagnosable ("stream aborted:" and nothing else).
            import traceback
            sys.stderr.write(f"[inference] stream aborted: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Inference stream error: {type(e).__name__}: {e}")
            return
        finally:
            # Release scarce KV ownership before a possibly slow settlement
            # attempt; the claim/outbox rows are already durable here.
            if owned_principal is not None:
                self._close_peer_streams(session_principal)
                self._release_kv_session(owned_principal)
            try:
                if paid_claim is not None:
                    try:
                        paid_admission.finalize(
                            paid_claim, paid_termination)
                        paid_admission.flush_outbox()
                    except _paid.PaidAdmissionError:
                        # The durable nonfinal receipt remains replayable; never
                        # leak receipt or payer material into daemon logs.
                        pass
            finally:
                if paid_session_lock is not None:
                    paid_session_lock.release()
            # worker/kv-session-ownership-gate: release on EVERY exit path —
            # normal completion, an early `return` above, the idle-timeout or
            # generic-exception handlers, or the client dropping the stream.
            # A session that errors or times out must not permanently lock
            # out the daemon for the next one.


# Phase I8: persistent nonce slot for soft-attestation. Pillar issues a
# fresh nonce in each /peer response; we stash it here so the next
# heartbeat includes the matching signed attestation. Empty string on
# first contact — pillar accepts and returns a starter nonce.
_attestation_nonce: str = ""


def decide_registered_backend(gpu_vram_gb, declared_backend, offload_status):
    """(backend_to_register, offload_verified) for a worker that declared a GPU backend.

    Downgrade to "cpu" ONLY on a positive report of zero layers offloaded (the daemon printed `offloaded 0/N`, i.e. the
    binary really lacks that backend). A daemon that logged NO offload line at all (n_total == 0) is UNVERIFIED, not
    "zero": newer llama.cpp builds are quiet on a successful load, and treating silence as zero registered a 12 GB RTX 5070
    as a CPU node (2026-09-21). Unverified keeps the operator's declaration and is flagged so the pillar can tell.
    """
    declared_gpu = gpu_vram_gb > 0 and declared_backend != "cpu"
    if not declared_gpu:
        return declared_backend, True
    total = offload_status.get("total_layers", 0)
    if total > 0:
        if offload_status.get("uses_gpu"):
            return declared_backend, True
        return "cpu", True
    return declared_backend, False


def register_with_pillar(pillar_url: str, payload: dict, log_prefix: str = "[worker]",
                          priv_key: Optional[bytes] = None,
                          node_id: Optional[str] = None,
                          spki_hash: Optional[str] = None,
                          warm_directive_handler=None,
                          model_id: Optional[str] = None,
                          now=None):
    """POST to <pillar_url>/peer. Best-effort: log on failure, never raise.

    Phase F2: when ``priv_key`` + ``node_id`` are provided, the request
    gets a signed Authorization header. TOFU first-registration includes
    ``public_key_hex`` in the body (worker_auth helper handles that
    upstream by inserting the field into ``payload`` before this call).

    Phase F3: when ``spki_hash`` is provided and the URL is HTTPS, the
    pillar's TLS cert SPKI is verified against it. Mismatch → refusal.

    Phase I8: includes a soft-attestation blob bound to the pillar-issued
    nonce. Pillar audits hash changes across observations.
    """
    global _attestation_nonce
    # Phase I8: build the attestation blob if the sandbox module is
    # available. nonce_hex is empty on first contact; pillar tolerates.
    if _SANDBOX_AVAILABLE:
        try:
            payload = dict(payload)  # don't mutate caller's dict
            payload["attestation"] = _wsandbox.build_attestation_blob(
                _attestation_nonce)
        except Exception as e:
            print(f"{log_prefix} attestation build failed: {e}", flush=True)
    body_bytes = json.dumps(payload).encode()
    path = "/peer"
    headers = {"Content-Type": "application/json"}
    if _AUTH_AVAILABLE and priv_key and node_id:
        header_val, _ts = _wauth.build_signed_envelope(
            priv_key, node_id, "POST", path, body_bytes)
        headers["Authorization"] = header_val

    full_url = f"{pillar_url.rstrip('/')}{path}"
    try:
        req = urlrequest.Request(
            full_url, data=body_bytes, headers=headers, method="POST",
        )
        if full_url.startswith("https://"):
            if not _AUTH_AVAILABLE:
                print(f"{log_prefix} HTTPS pillar URL but auth module missing; "
                      f"refusing connection", flush=True)
                return False
        if not _AUTH_AVAILABLE:
            raise RuntimeError("pillar auth module unavailable")
        with _wauth.open_pillar_url(
            req, timeout=5, expected_spki_hash=spki_hash,
        ) as resp:
            body = resp.read().decode()
            parsed = None
            # Phase I8: stash the fresh nonce for the next heartbeat.
            # Phase J1: when we SENT an attestation but the pillar
            # didn't observe it (nonce stale, fingerprint malformed),
            # WARN loudly so operators see the silent-broken state
            # rather than relying on attestation that doesn't actually
            # work.
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    # Phase D4 (2026-05-20): strict-hex + length cap on
                    # the pillar-supplied nonce. A malicious pillar
                    # could send a non-hex or oversized blob; we echo
                    # this back in the next signed envelope, so
                    # bound-validate at ingest.
                    raw_nonce = parsed.get("attestation_nonce_hex", "")
                    if _VALIDATION_AVAILABLE:
                        new_nonce = _wval.as_bounded_hex(
                            raw_nonce, max_chars=64, default=""
                        )
                    else:
                        new_nonce = str(raw_nonce) if isinstance(raw_nonce, str) else ""
                        if len(new_nonce) > 64:
                            new_nonce = ""
                    if new_nonce:
                        _attestation_nonce = new_nonce
                    observed = parsed.get("attestation_observed", None)
                    sent_attestation = "attestation" in payload
                    if sent_attestation and observed is False:
                        print(f"{log_prefix} WARN: pillar did not observe "
                              f"our attestation. Likely stale nonce "
                              f"(retry next heartbeat) or fingerprint "
                              f"malformed. Pillar audit log records "
                              f"attestation_nonce_mismatch or "
                              f"attestation_fingerprint_too_long.",
                              flush=True)
                        _audit("attestation_observed_false",
                               url=pillar_location_for_log(pillar_url),
                               log_prefix=log_prefix)
            except json.JSONDecodeError:
                pass
            if (warm_directive_handler is not None and model_id
                    and isinstance(parsed, dict)):
                directive = parse_warm_directive(parsed, model_id, now=now)
                if directive is _INVALID_WARM_DIRECTIVE:
                    print(
                        f"{log_prefix} ignoring invalid warm directive",
                        flush=True,
                    )
                else:
                    warm_directive_handler(directive)
            print(f"{log_prefix} registered with pillar: {body}", flush=True)
            _audit("register_success", url=pillar_location_for_log(pillar_url),
                   log_prefix=log_prefix)
            return True
    except (urlerror.URLError, OSError, TimeoutError) as e:
        safe_pillar = pillar_location_for_log(pillar_url)
        safe_error = _pillar_error_for_log(e, pillar_url)
        print(f"{log_prefix} pillar registration failed ({safe_pillar}): {safe_error}",
              flush=True)
        _audit("register_failed", url=safe_pillar, error=safe_error,
               log_prefix=log_prefix)
        return False


def activation_endpoint_from_env(environ=None) -> Optional[dict]:
    """Return the additive Pillar endpoint for the local libp2p sidecar.

    The release derives ``NAKSHATRA_P2P_PEER_ID`` from the node key.  The
    Pillar independently verifies that derivation against ``public_key_hex``;
    this function only validates the line-safe representation before placing
    it in the signed registration body.
    """
    environ = os.environ if environ is None else environ
    peer_id = str(environ.get("NAKSHATRA_P2P_PEER_ID", "") or "").strip()
    if not peer_id:
        return None
    if (not peer_id.isascii() or len(peer_id) > 160
            or any(ch.isspace() or ord(ch) < 33 for ch in peer_id)):
        raise ValueError("NAKSHATRA_P2P_PEER_ID is invalid")
    return {"kind": "libp2p-activation-v1", "peer_id": peer_id}


HEARTBEAT_INITIAL_INTERVAL_S = 30.0
HEARTBEAT_MAX_INTERVAL_S = 600.0   # 10 minutes
HEARTBEAT_BACKOFF_FACTOR = 2.0
HEARTBEAT_JITTER_FRACTION = 0.25  # ±25%


def _next_heartbeat_interval(
    base: float, consecutive_failures: int, *, rng=None,
) -> float:
    """Phase D6 (2026-05-20) — exponential backoff + jitter.

    On consecutive_failures=0, returns base (no backoff).
    On N failures, returns min(base * factor^N, MAX) with ±25% jitter.
    Pure function so the tests can pin rng to verify the formula.
    """
    import random
    rng = rng or random
    if consecutive_failures <= 0:
        delay = base
    else:
        delay = min(
            base * (HEARTBEAT_BACKOFF_FACTOR ** consecutive_failures),
            HEARTBEAT_MAX_INTERVAL_S,
        )
    jitter = 1.0 + (rng.random() * 2 - 1) * HEARTBEAT_JITTER_FRACTION
    return max(1.0, delay * jitter)


def heartbeat_loop(pillar_url: str, payload: dict, interval: float = 30.0,
                   stop_event: threading.Event = None,
                   daemon_for_timing=None,
                   priv_key: Optional[bytes] = None,
                   node_id: Optional[str] = None,
                   spki_hash: Optional[str] = None,
                   residency_event: threading.Event = None,
                   warm_directive_handler=None,
                   model_id: Optional[str] = None,
                   now=None):
    """Re-register with pillar every `interval` seconds. Run as daemon thread.

    Phase H: if `daemon_for_timing` is supplied, refresh the heartbeat
    payload's `recent_rpc_ms` field with the average of the daemon's
    last N call timings. The pillar then has up-to-date latency data
    for peer-ranking.

    Phase F2: ``priv_key`` + ``node_id`` + ``spki_hash`` are plumbed
    through to register_with_pillar so heartbeats stay signed + cert-
    pinned. Heartbeat payloads omit ``public_key_hex`` after the first
    successful registration — the pillar's TOFU lock already remembers
    the key.

    Phase D6 (2026-05-20): exponential backoff on repeated failures.
    Successful heartbeats reset the counter; the loop converges on
    ``interval`` when the pillar is healthy. With a pillar in outage
    the worker stops hammering: 30s → 60s → 120s → ... up to 600s.
    Jitter ±25% staggers re-attempts across a cluster restart.
    """
    stop_event = stop_event or threading.Event()
    consecutive_failures = 0
    while not stop_event.is_set():
        next_interval = _next_heartbeat_interval(
            interval, consecutive_failures
        )
        if residency_event is None:
            if stop_event.wait(timeout=next_interval):
                break
        else:
            # A residency transition is registry-significant: publish it now,
            # not after the normal 30-second cadence.  The shutdown path sets
            # both events so this wait also exits promptly.
            residency_event.wait(timeout=next_interval)
            residency_event.clear()
            if stop_event.is_set():
                break
        if daemon_for_timing is not None:
            payload["residency"] = daemon_for_timing.residency
        if daemon_for_timing is not None and daemon_for_timing.recent_rpc_ms:
            avg = sum(daemon_for_timing.recent_rpc_ms) / len(daemon_for_timing.recent_rpc_ms)
            # Phase A6: belt-and-suspenders — avg of all-finite values
            # is finite, but guard against deque mutation races.
            avg_clean = safe_rpc_ms(avg)
            if avg_clean is not None:
                payload["recent_rpc_ms"] = avg_clean
        ok = register_with_pillar(
            pillar_url, payload, log_prefix="[heartbeat]",
            priv_key=priv_key, node_id=node_id, spki_hash=spki_hash,
            warm_directive_handler=warm_directive_handler,
            model_id=model_id, now=now,
        )
        if ok:
            consecutive_failures = 0
        else:
            consecutive_failures += 1


_FILE_SERVER_DIR = ""
_HEALTH_STATE: dict = {}

# Sthambha planner Step 4: slice task registry. Worker accepts long-running
# partial_gguf.py jobs via POST /slice, returns a task_id, runs the cut
# in a background thread, exposes status via GET /slice/<task_id>. Output
# lands in _FILE_SERVER_DIR so Phase-4 auto-fetch can serve it as soon as
# the slice finishes. Transport choice (HTTP, not the gRPC the planner doc
# §7 Q2 originally recommended) recorded in trisul ADR 0003 — minimises
# coupling between L2 (this worker) and L3 (sthambha pillar) ahead of the
# fabric work that will eventually subsume both transports.
_SLICE_TASKS: dict = {}
_SLICE_LOCK = threading.Lock()
_SLICE_MAX_HISTORY = 64   # ring-buffer cap on completed task entries
_PARTIAL_GGUF_PATH = ""   # resolved at file-server start, see start_file_server()


def _slug_for_filename(s: str) -> str:
    """Sanitize a string for use as a filename component.

    Keeps alphanumerics, dots, dashes, underscores; collapses everything else
    to a single dash. Then strips traversal sequences ("..", leading dots)
    so the result is safe to concatenate with a directory path.
    """
    out = []
    prev_dash = False
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-").lstrip(".")
    # Collapse any ".." traversal sequences. Single dots between alnum runs
    # (e.g. "llama-3.3-70b") stay; consecutive dots become a single dot.
    while ".." in slug:
        slug = slug.replace("..", ".")
    return slug.strip("-.") or "model"


def _slice_output_filename(model_id: str, layer_start: int, layer_end: int) -> str:
    """Pillar-derived (not client-supplied) — no path injection.

    Convention matches what Phase-4 auto-fetch already scans for: a basename
    in _FILE_SERVER_DIR carrying `nakshatra.layer_range_start/end` metadata.
    partial_gguf.py writes those KVs (see experiments/v0.0/partial_gguf.py).
    """
    return f"{_slug_for_filename(model_id)}.l{layer_start}-{layer_end}.gguf"


def _slice_task_snapshot(task_id: str) -> dict:
    """Return a JSON-serialisable view of a task's current state."""
    with _SLICE_LOCK:
        t = _SLICE_TASKS.get(task_id)
        if t is None:
            return {}
        return {
            "task_id": task_id,
            "status": t["status"],
            "model_id": t["model_id"],
            "layer_start": t["layer_start"],
            "layer_end": t["layer_end"],
            "output_filename": t["output_filename"],
            "started_at": t.get("started_at"),
            "finished_at": t.get("finished_at"),
            "sha256": t.get("sha256"),
            "size_bytes": t.get("size_bytes"),
            "error": t.get("error"),
        }


def _slice_update(task_id: str, **fields):
    """Update a task entry if it still exists; no-op if evicted/cleared.

    The ring-buffer cap can evict an old terminal entry to make room for
    a new task, but the worker thread for the still-running slice keeps a
    reference to its task_id. If the registry no longer holds the entry,
    the update is silently dropped — the subprocess work is done, just
    nothing to record.
    """
    with _SLICE_LOCK:
        entry = _SLICE_TASKS.get(task_id)
        if entry is not None:
            entry.update(fields)


def _run_slice_task(task_id: str, model_id: str, src: str, dst: str,
                    layer_start: int, layer_end: int,
                    force_keep_token_embd: bool, force_keep_output: bool):
    """Worker thread body. Runs partial_gguf.py and updates the registry."""
    _audit("slice_spawned", task_id=task_id, model_id=model_id,
           src=src, dst=dst,
           layer_start=layer_start, layer_end=layer_end)
    try:
        cmd = [
            sys.executable, _PARTIAL_GGUF_PATH, src, dst,
            "--start", str(layer_start),
            "--end", str(layer_end),
        ]
        if force_keep_token_embd:
            cmd.append("--keep-token-embd")
        if force_keep_output:
            cmd.append("--keep-output")
        proc = subprocess.run(
            cmd, capture_output=True, timeout=SLICE_SUBPROCESS_TIMEOUT_S,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[-2000:]
            _slice_update(task_id, status="failed", finished_at=time.time(),
                           error=f"partial_gguf exit {proc.returncode}: {err.strip()}")
            return
        if not os.path.isfile(dst):
            _slice_update(task_id, status="failed", finished_at=time.time(),
                           error=f"partial_gguf returned 0 but {dst!r} not present")
            return
        sha = sha256_of_file(dst)
        # Write the sha256 sidecar Phase-4 cache-scan reads to skip re-hashing.
        try:
            with open(dst + ".sha256", "w") as f:
                f.write(sha + "\n")
        except Exception:
            pass
        _slice_update(task_id, status="completed", finished_at=time.time(),
                       sha256=sha, size_bytes=os.path.getsize(dst))
        _audit("slice_completed", task_id=task_id, sha256=sha,
               size_bytes=os.path.getsize(dst))
    except subprocess.TimeoutExpired:
        _slice_update(task_id, status="failed", finished_at=time.time(),
                       error=f"partial_gguf timed out after {SLICE_SUBPROCESS_TIMEOUT_S}s")
        _audit("slice_failed", task_id=task_id, error="timeout",
               timeout_s=SLICE_SUBPROCESS_TIMEOUT_S)
    except Exception as e:
        _slice_update(task_id, status="failed", finished_at=time.time(),
                       error=f"slice worker crashed: {e}")
        _audit("slice_failed", task_id=task_id, error=str(e))


def detect_gpus() -> list:
    """Best-effort enumeration of physical GPUs on the host (iGPU + dGPU).

    Inventory data for ops dashboards. Independent of which GPU the daemon
    actually offloaded to (that lives in DaemonClient.gpu_offload_status()).
    Returns a list of {name, vendor, integrated, vram_mb} dicts; [] on
    unsupported OS or tool failure. Slow (~1-2s) — call once at startup.
    """
    try:
        if platform.system() == "Darwin":
            # ioreg sees hidden Intel iGPUs that SPDisplaysDataType drops when
            # a discrete AMD GPU is active on Intel iMacs. We walk every
            # IOPCIDevice in the I/O Registry and pick those whose PCI class
            # code starts with 0x03 (display controller).
            out = subprocess.run(
                ["ioreg", "-a", "-r", "-c", "IOPCIDevice"],
                capture_output=True, timeout=5, check=False,
            ).stdout
            plist = plistlib.loads(out) if out else []
            vendor_map = {0x8086: "intel", 0x1002: "amd", 0x10de: "nvidia", 0x106B: "apple"}
            # vendors whose GPUs in a Mac chassis are always integrated
            integrated_vendors = {0x8086, 0x106B}
            gpus = []

            def walk(entries):
                for e in entries:
                    cc = e.get("class-code") or b""
                    # vendor-id and class-code come as little-endian 4-byte
                    # values; display class is the third byte = 0x03.
                    if len(cc) >= 3 and cc[2] == 0x03:
                        model = e.get("model") or b""
                        if isinstance(model, bytes):
                            name = model.rstrip(b"\x00").decode("utf-8", "replace") or "unknown"
                        else:
                            name = str(model) or e.get("IOName", "unknown")
                        vid_b = e.get("vendor-id") or b""
                        vid = int.from_bytes(vid_b[:2], "little") if vid_b else None
                        vendor = vendor_map.get(vid, f"0x{vid:04x}" if vid is not None else "")
                        ioname = e.get("IOName", "")
                        integrated = (vid in integrated_vendors) or ioname == "IGPU"
                        gpus.append({
                            "name": name, "vendor": vendor,
                            "integrated": integrated, "vram_mb": None,
                        })
                    for c in e.get("IORegistryEntryChildren") or []:
                        walk([c])

            walk(plist)
            if gpus:
                return gpus
            # Apple Silicon: GPU is on the SoC, not enumerated via IOPCIDevice.
            # Fall back to SPDisplaysDataType which still surfaces it under a
            # synthetic entry (e.g. "Apple M1 Pro", integrated by definition).
            out = subprocess.run(
                ["system_profiler", "-json", "SPDisplaysDataType"],
                capture_output=True, timeout=5, text=True, check=False,
            ).stdout
            data = json.loads(out) if out else {}
            for d in data.get("SPDisplaysDataType", []):
                name = d.get("_name") or d.get("sppci_model", "unknown")
                vendor = d.get("spdisplays_vendor", "")
                if vendor.startswith("sppci_vendor_"):
                    vendor = vendor[len("sppci_vendor_"):]
                gpus.append({
                    "name": name, "vendor": vendor.lower(),
                    "integrated": True, "vram_mb": None,
                })
            return gpus
        if platform.system() == "Linux":
            out = subprocess.run(
                ["lspci"], capture_output=True, timeout=5, text=True, check=False,
            ).stdout
            gpus = []
            for line in out.splitlines():
                if not any(k in line for k in ("VGA compatible", "3D controller", "Display controller")):
                    continue
                # "01:00.0 VGA compatible controller: AMD/ATI Navi … [Radeon RX …]"
                bus, _, rest = line.partition(" ")
                _, _, name = rest.partition(": ")
                name = name.strip()
                # iGPUs are typically on PCIe bus 00:xx and named for Intel or
                # AMD APU codenames; dGPUs sit on a separate root complex.
                integrated = bus.startswith("00:") and any(
                    k in name for k in ("Intel", "Renoir", "Cezanne", "Raphael", "Rembrandt", "Phoenix")
                )
                vendor = "intel" if "Intel" in name else ("amd" if ("AMD" in name or "ATI" in name) else "")
                gpus.append({
                    "name": name, "vendor": vendor,
                    "integrated": integrated, "vram_mb": None,
                })
            return gpus
    except Exception as e:
        sys.stderr.write(f"[healthz] detect_gpus failed: {e}\n")
    return []


class FileServerHandler(BaseHTTPRequestHandler):
    """Phase-4 file server + worker health endpoint.

    Endpoints:
        GET /file/<basename>     — sends the file (full or Range-restricted)
        GET /healthz             — rich JSON health for ops dashboards; 200 OK
                                   if daemon alive, 503 if not.
        GET /health, /           — aliases for /healthz.
    """
    server_version = "NakshatraFileServer/0.1"

    def log_message(self, format, *args):
        # Quiet by default; uncomment for debugging
        # sys.stderr.write(f"[fileserver] {self.address_string()} {format % args}\n")
        pass

    # Phase C (worker hardening sprint, 2026-05-20) — per-route tier
    # gate. Returns the verified keyid (or None for anonymous) on
    # success; emits HTTP 401/403 and returns False on failure.
    def _check_tier(self, tier: str, *, body: bytes = b"") -> bool:
        """Verify the call against the declared tier. Sends the error
        response itself; returns True iff the call should proceed."""
        if tier == TIER_ANONYMOUS:
            return True
        if not _HTTP_AUTH_STATE.get("auth_required"):
            # Legacy Mode-A bringup: tier checks are skipped for
            # AUTHENTICATED routes but OPERATOR still requires the
            # operator pubkey (independent of network auth).
            if tier == TIER_OPERATOR:
                pass  # fall through to verify
            else:
                return True
        auth_header = self.headers.get("Authorization", "")
        try:
            verify_http_request(
                auth_header,
                method=self.command,
                path=self.path,
                body=body,
                operator_pubkey=_HTTP_AUTH_STATE.get("operator_pubkey"),
                peer_resolver=_HTTP_AUTH_STATE.get("peer_resolver"),
                tier=tier,
            )
            return True
        except ValueError as e:
            status = 401 if tier == TIER_AUTHENTICATED else 403
            ip = self.client_address[0] if self.client_address else "?"
            _audit("auth_failure_http", ip=ip, reason=str(e),
                   tier=tier, method=self.command, path=self.path)
            self.send_error(status, f"{tier}: {e}")
            return False

    def _minimal_health_payload(self):
        """Phase C5 — anonymous tier payload. Operators / clients
        verify liveness + basic identity without leaking GPU/RAM/idem
        cache stats (those go on /healthz/full)."""
        daemon = _HEALTH_STATE.get("daemon")
        daemon_alive = bool(
            daemon is not None and (
                daemon.service_alive if hasattr(daemon, "service_alive")
                else daemon.proc.poll() is None
            )
        )
        residency = getattr(daemon, "residency", "loaded" if daemon_alive else "standby")
        started_at = _HEALTH_STATE.get("started_at", time.time())
        return {
            "status": "ok" if daemon_alive else "down",
            "residency": residency,
            "mode": _HEALTH_STATE.get("mode", ""),
            "layer_start": _HEALTH_STATE.get("layer_start", -1),
            "layer_end": _HEALTH_STATE.get("layer_end", -1),
            "uptime_seconds": round(time.time() - started_at, 1),
            "starting": bool(getattr(_HEALTH_STATE.get("servicer"), "starting", False)),
            "protocol_version": "0.1.0",
        }

    def _health_payload(self):
        daemon = _HEALTH_STATE.get("daemon")
        daemon_alive = bool(
            daemon is not None and (
                daemon.service_alive if hasattr(daemon, "service_alive")
                else daemon.proc.poll() is None
            )
        )
        residency = getattr(daemon, "residency", "loaded" if daemon_alive else "standby")
        started_at = _HEALTH_STATE.get("started_at", time.time())
        samples = list(daemon.recent_rpc_ms) if daemon is not None else []
        recent_avg = (sum(samples) / len(samples)) if samples else None
        gpu = daemon.gpu_offload_status() if daemon is not None else None
        return {
            "status": "ok" if daemon_alive else "down",
            "residency": residency,
            "node_id": _HEALTH_STATE.get("node_id", ""),
            "model_id": _HEALTH_STATE.get("model_id", ""),
            "mode": _HEALTH_STATE.get("mode", ""),
            "layer_start": _HEALTH_STATE.get("layer_start", -1),
            "layer_end": _HEALTH_STATE.get("layer_end", -1),
            "grpc_port": _HEALTH_STATE.get("grpc_port", 0),
            "file_server_port": _HEALTH_STATE.get("file_server_port", 0),
            "uptime_seconds": round(time.time() - started_at, 1),
            "daemon_alive": daemon_alive,
            "recent_rpc_ms_avg": round(recent_avg, 2) if recent_avg is not None else None,
            "recent_rpc_ms_samples": len(samples),
            "gpu": {
                "uses_gpu": gpu["uses_gpu"] if gpu else False,
                "offloaded": (f"{gpu['n_offloaded']}/{gpu['total_layers']}"
                              if gpu and gpu["total_layers"] > 0 else None),
                "backends": gpu["backend_hints"] if gpu else [],
            },
            "gpus_present": _HEALTH_STATE.get("gpus_present", []),
            "idem_cache": (_HEALTH_STATE["servicer"].idem_stats()
                           if _HEALTH_STATE.get("servicer") is not None else None),
            "rpc_push": (_HEALTH_STATE["servicer"].push_stats()
                         if _HEALTH_STATE.get("servicer") is not None else None),
            "auth": (_HEALTH_STATE["servicer"].auth_stats()
                     if _HEALTH_STATE.get("servicer") is not None else None),
            "protocol_version": "0.1.0",
        }

    def do_GET(self):
        # Phase C5: minimal anonymous /healthz, verbose /healthz/full
        # behind AUTHENTICATED. Anonymous body discloses only liveness +
        # mode + layer range + uptime; everything else is on the
        # authenticated path.
        if self.path in ("/", "/health", "/healthz"):
            if not self._check_tier(TIER_ANONYMOUS):
                return
            payload = self._minimal_health_payload()
            body = json.dumps(payload).encode("utf-8")
            self.send_response(
                200 if payload["status"] == "ok" else 503
            )
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/healthz/full":
            if not self._check_tier(TIER_AUTHENTICATED):
                return
            payload = self._health_payload()
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200 if payload["daemon_alive"] else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/slice/"):
            if not self._check_tier(TIER_AUTHENTICATED):
                return
            task_id = self.path[len("/slice/"):]
            snap = _slice_task_snapshot(task_id)
            if not snap:
                self.send_error(404, "task not found")
                return
            body = json.dumps(snap).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if not self.path.startswith("/file/"):
            self.send_error(404, "not found")
            return

        # Phase C2: /file/<basename> is AUTHENTICATED — model weights
        # are not public artefacts on a Mode-C deployment.
        if not self._check_tier(TIER_AUTHENTICATED):
            return

        # Sanitize: only allow simple basenames, no traversal
        filename = self.path[len("/file/"):]
        if "/" in filename or ".." in filename or filename.startswith("."):
            self.send_error(400, "bad filename")
            return
        path = os.path.join(_FILE_SERVER_DIR, filename)
        if not os.path.isfile(path):
            self.send_error(404, "file not found")
            return

        size = os.path.getsize(path)
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            try:
                spec = range_header[6:].strip()
                start_str, _, end_str = spec.partition("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else size - 1
                if start < 0 or end >= size or start > end:
                    self.send_error(416, "bad range")
                    return
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except Exception as e:
                self.send_error(400, f"range error: {e}")
            return

        # Full-file send
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile, 8 * 1024 * 1024)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if self.path != "/slice":
            self.send_error(404, "not found")
            return
        if not _PARTIAL_GGUF_PATH or not os.path.isfile(_PARTIAL_GGUF_PATH):
            self.send_error(503, "slicer not configured on this worker")
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0 or length > 64 * 1024:
            self.send_error(400, "body required (<= 64KB)")
            return
        raw_body = self.rfile.read(length)
        # Phase C2: OPERATOR tier — slicing spawns subprocess; only the
        # operator key (filesystem-scoped, not network-resolved) can
        # authorise. Signature covers the raw body bytes.
        if not self._check_tier(TIER_OPERATOR, body=raw_body):
            return
        try:
            body = json.loads(raw_body)
        except Exception as e:
            self.send_error(400, f"bad json: {e}")
            return

        model_id = (body.get("model_id") or "").strip()
        full_gguf_path = (body.get("full_gguf_path") or "").strip()
        try:
            layer_start = int(body.get("layer_start"))
            layer_end = int(body.get("layer_end"))
        except (TypeError, ValueError):
            self.send_error(400, "layer_start and layer_end must be integers")
            return
        # Phase D2 (2026-05-20): strict-bool. `bool("false")` is truthy
        # so the prior coercion silently treated "false" as True. The
        # strict helper accepts only literal True/False.
        if _VALIDATION_AVAILABLE:
            force_keep_token_embd = _wval.as_strict_bool(
                body.get("force_keep_token_embd"), default=False
            )
            force_keep_output = _wval.as_strict_bool(
                body.get("force_keep_output"), default=False
            )
        else:
            force_keep_token_embd = body.get("force_keep_token_embd") is True
            force_keep_output = body.get("force_keep_output") is True

        if not model_id:
            self.send_error(400, "model_id required")
            return
        # Phase C4: root-bound + unicode-safe path validation. Replaces
        # the prior `os.path.isfile()` check that accepted ANY readable
        # path — attacker could pass /etc/passwd to probe filesystem
        # via subprocess error messages.
        slice_root = _HTTP_AUTH_STATE.get("slice_root") or Path(_FILE_SERVER_DIR)
        try:
            resolved_gguf = validate_slice_path(full_gguf_path, slice_root)
        except ValueError as e:
            self.send_error(400, f"full_gguf_path: {e}")
            return
        full_gguf_path = str(resolved_gguf)
        if layer_end <= layer_start or layer_start < 0:
            self.send_error(400, "layer_end must be > layer_start >= 0")
            return

        # Phase A8 (2026-05-20): cap concurrent slice subprocesses. Each
        # spawn can run for SLICE_SUBPROCESS_TIMEOUT_S seconds eating CPU;
        # unbounded fan-out is a DoS amplifier.
        if _running_slice_count() >= MAX_CONCURRENT_SLICES:
            self.send_error(
                429,
                f"too many concurrent slices (cap={MAX_CONCURRENT_SLICES})",
            )
            return

        output_filename = _slice_output_filename(model_id, layer_start, layer_end)
        output_path = os.path.join(_FILE_SERVER_DIR, output_filename)
        task_id = uuid.uuid4().hex
        with _SLICE_LOCK:
            # Cheap ring-buffer: drop oldest COMPLETED/FAILED if we're at cap.
            # Never drop a running task.
            if len(_SLICE_TASKS) >= _SLICE_MAX_HISTORY:
                terminal = [
                    (tid, t) for tid, t in _SLICE_TASKS.items()
                    if t["status"] in ("completed", "failed")
                ]
                if terminal:
                    terminal.sort(key=lambda kv: kv[1].get("finished_at") or 0.0)
                    del _SLICE_TASKS[terminal[0][0]]
            _SLICE_TASKS[task_id] = {
                "status": "running",
                "model_id": model_id,
                "layer_start": layer_start,
                "layer_end": layer_end,
                "output_filename": output_filename,
                "output_path": output_path,
                "started_at": time.time(),
                "finished_at": None,
                "sha256": None,
                "size_bytes": None,
                "error": None,
            }
        threading.Thread(
            target=_run_slice_task,
            args=(task_id, model_id, full_gguf_path, output_path,
                  layer_start, layer_end, force_keep_token_embd, force_keep_output),
            daemon=True,
        ).start()

        resp = json.dumps({
            "task_id": task_id,
            "status": "running",
            "output_filename": output_filename,
            "poll_url": f"/slice/{task_id}",
        }).encode("utf-8")
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(resp)


class _ThreadingHTTPServer(HTTPServer):
    """Allow multiple concurrent fetches (without this, parallel byte-range
    requests would serialize through one handler thread)."""
    daemon_threads = True
    def process_request(self, request, client_address):
        threading.Thread(
            target=self._handle_request_thread, args=(request, client_address),
            daemon=True,
        ).start()
    def _handle_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            pass
        finally:
            try: self.shutdown_request(request)
            except Exception: pass


def start_file_server(serving_dir: str, port: int):
    """Start the Phase-4 file server in a background thread.

    Also resolves _PARTIAL_GGUF_PATH so POST /slice has a slicer to invoke.
    Override via env var NAKSHATRA_PARTIAL_GGUF; default: discover relative
    to this script at ../experiments/v0.0/partial_gguf.py.
    """
    global _FILE_SERVER_DIR, _PARTIAL_GGUF_PATH
    _FILE_SERVER_DIR = str(Path(serving_dir).resolve())
    env_path = os.environ.get("NAKSHATRA_PARTIAL_GGUF", "").strip()
    if env_path:
        _PARTIAL_GGUF_PATH = str(Path(env_path).resolve())
    else:
        default_path = Path(__file__).resolve().parent.parent / "experiments" / "v0.0" / "partial_gguf.py"
        _PARTIAL_GGUF_PATH = str(default_path) if default_path.is_file() else ""
    server = _ThreadingHTTPServer(("0.0.0.0", port), FileServerHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    slicer_status = _PARTIAL_GGUF_PATH or "DISABLED (partial_gguf.py not found)"
    print(f"[fileserver] listening on :{port}, serving from {_FILE_SERVER_DIR}", flush=True)
    print(f"[fileserver] slicer: {slicer_status}", flush=True)


def package_location_for_log(location: str) -> str:
    """Return a package location safe for journals and audit records.

    URL userinfo, query parameters, and fragments may carry credentials. Keep
    only scheme, host/port, and path. Local package paths are returned without
    query/fragment-shaped suffixes for the same conservative logging rule.
    """
    if _AUTH_AVAILABLE:
        return _wauth.redact_url_for_log(location)
    return "<redacted package location>"


def pillar_location_for_log(location: str) -> str:
    """Return a pillar URL safe for journals and audit records."""
    if _AUTH_AVAILABLE:
        return _wauth.redact_url_for_log(location)
    return "<redacted pillar URL>"


def _pillar_error_for_log(error: object, pillar_url: str) -> str:
    """Remove the configured credential-bearing pillar base from an error."""
    if _AUTH_AVAILABLE:
        return _wauth.redact_url_in_text(error, pillar_url)
    return "<redacted pillar error>"


_CONTROL_CAPS_CACHE: Optional[list] = None


def _control_version_caps() -> list:
    """v1.0 §7 — control/vN capability tokens for the Info handshake. Failure-soft:
    returns [] if the wire module can't be imported, so Info never breaks."""
    global _CONTROL_CAPS_CACHE
    if _CONTROL_CAPS_CACHE is None:
        try:
            from wire.handshake import advertise_capabilities
            _CONTROL_CAPS_CACHE = advertise_capabilities()
        except Exception:
            _CONTROL_CAPS_CACHE = []
    return _CONTROL_CAPS_CACHE


def provision_from_package(package_url: str, layer_start: int, layer_end: int,
                           dest_path: str, *, require_signature: bool = False,
                           trusted_pubkeys: Optional[set] = None,
                           local_dev: bool = False) -> str:
    """v1.0 §5 (P2) — provision this worker's assigned layer range from a
    content-addressed layer package instead of fetching a whole pre-cut sub-GGUF
    from a peer.

    Fetches ONLY this position's fragments (metadata + embeddings iff start==0 +
    head iff end==n_layers + layers[start,end)), verifies each SHA-256
    **fail-closed**, and assembles a loader-ready sub-GGUF at dest_path carrying
    the same nakshatra.layer_range_* metadata a peer-fetched sub-GGUF would — so
    the daemon, cache-scan advertising, and peer-fetch path all work unchanged.

    Unlike fetch_sub_gguf_from_peer this needs no peer holding the exact pre-cut
    range — a freshly discovered node can self-provision any range from a package
    (docs/v1.0-discovery-and-distribution.md §5.3). `package_url` is a local dir,
    a package.json path, or an http(s) base."""
    _audit("package_provision_started", package_url=package_location_for_log(package_url),
           layer_start=layer_start, layer_end=layer_end, dest_path=dest_path)
    from packaging.fetch_package import fetch_and_assemble
    fetch_and_assemble(package_url, layer_start, layer_end, dest_path,
                       require_signature=require_signature,
                       trusted_pubkeys=trusted_pubkeys,
                       local_dev=local_dev)
    _audit("package_provision_completed", dest=dest_path)
    return dest_path


def fetch_sub_gguf_from_peer(pillar_url: str, model_id: str,
                              layer_start: int, layer_end: int,
                              dest_path: str,
                              own_node_id: str = "",
                              priv_key: Optional[bytes] = None,
                              spki_hash: Optional[str] = None,
                              verified_sha_out: Optional[dict] = None) -> str:
    """Query Sthambha for a peer with the requested file and download it.

    Returns the local path on success; raises RuntimeError on failure.
    Verifies SHA-256 against the pillar's recorded hash.  Missing or malformed
    hashes are refused before any peer bytes are fetched.

    Skips candidates whose node_id matches `own_node_id` (don't fetch
    from yourself). Retries the next candidate on connection failure
    so a stale-but-still-listed peer doesn't block the bootstrap.
    """
    _audit("fetch_started", model_id=model_id,
           layer_start=layer_start, layer_end=layer_end,
           dest_path=dest_path)
    # 1. Ask the pillar for the file index.  This bootstrap request has the
    # same authentication contract as registration/heartbeats: it is always
    # signed by the persistent worker identity, and HTTPS uses the configured
    # pillar SPKI pin.  There is deliberately no unsigned fallback.
    if not _AUTH_AVAILABLE:
        raise RuntimeError(
            "authenticated pillar /files query requires nakshatra_auth")
    if not priv_key or not own_node_id:
        raise RuntimeError(
            "authenticated pillar /files query requires the worker key and node id")
    try:
        validate_pillar_transport(
            pillar_url,
            spki_hash or "",
            os.environ.get("TRUSTED_PILLAR_NETS", DEFAULT_TRUSTED_PILLAR_NETS),
        )
    except WorkerServiceError as e:
        raise RuntimeError(
            f"refusing untrusted pillar transport for /files: {e}"
        ) from e
    query = urlencode({"model": model_id})
    files_path = f"/files?{query}"
    files_url = f"{pillar_url.rstrip('/')}{files_path}"
    auth_header, _ts = _wauth.build_signed_envelope(
        priv_key, own_node_id, "GET", files_path, b"")
    headers = {"Authorization": auth_header}
    req = urlrequest.Request(files_url, headers=headers, method="GET")
    try:
        with _wauth.open_pillar_url(
            req, timeout=10, expected_spki_hash=spki_hash,
        ) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        safe_url = f"{pillar_location_for_log(pillar_url).rstrip('/')}/files"
        raise RuntimeError(
            f"could not query pillar at {safe_url}: {_pillar_error_for_log(e, pillar_url)}"
        )

    # 2. Find candidate online peers holding (model_id, layer_start, layer_end)
    candidates = [
        f for f in data.get("files", [])
        if f.get("model_id") == model_id
        and int(f.get("layer_start", -1)) == layer_start
        and int(f.get("layer_end", -1)) == layer_end
        and f.get("is_online")
        and f.get("node_id") != own_node_id  # don't fetch from yourself
    ]
    if not candidates:
        raise RuntimeError(
            f"no online peer holds {model_id} layers [{layer_start},{layer_end})"
            f" — file not available on the network"
        )

    # Phase A7 / U4 B3: the pillar's model_sha256 is the trust anchor for
    # peer slices.  The old STHAMBHA_REFUSE_UNVERIFIED_FETCH=false escape
    # hatch is intentionally ignored: outsider workers never execute bytes
    # whose expected digest was omitted or malformed.
    verified_candidates = [
        c for c in candidates
        if not should_refuse_unverified_fetch(None, c.get("model_sha256", ""))
    ]
    if not verified_candidates:
        raise RuntimeError(
            f"refusing to fetch {model_id} layers [{layer_start},{layer_end})"
            f": no candidate peer has a valid pillar-attested model_sha256"
        )
    candidates = verified_candidates

    # 3. Try each candidate in order; on connection failure, fall through
    last_error = None
    for chosen in candidates:
        addr = chosen.get("address", "")
        if ":" not in addr:
            last_error = f"malformed address {addr!r}"
            continue
        host, _, port_str = addr.rpartition(":")
        grpc_port = int(port_str)
        file_port = grpc_port + 1000  # convention: file server on grpc_port + 1000
        expected_sha = chosen["model_sha256"].strip().lower()
        src_basename = Path(chosen.get("file_path", "")).name or f"w-{layer_start}-{layer_end}.gguf"

        fetch_url = f"http://{host}:{file_port}/file/{src_basename}"
        print(f"[fetch] trying {chosen.get('node_id')} → {fetch_url}", flush=True)

        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path + ".tmp"
        h = hashlib.sha256()
        bytes_received = 0
        t0 = time.time()
        try:
            with urlrequest.urlopen(fetch_url, timeout=60) as resp:
                content_length = int(resp.headers.get("Content-Length", "0"))
                with open(tmp_path, "wb") as out:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        h.update(chunk)
                        bytes_received += len(chunk)
                        if bytes_received and bytes_received % (200 * 1024 * 1024) < 8 * 1024 * 1024:
                            pct = (bytes_received / content_length * 100) if content_length else 0
                            elapsed = time.time() - t0
                            rate_mbps = (bytes_received / 1e6) / max(elapsed, 0.001)
                            print(f"[fetch]   {bytes_received/1e9:.1f}/{content_length/1e9:.1f} GB ({pct:.0f}%) at {rate_mbps:.1f} MB/s",
                                  flush=True)
        except Exception as e:
            print(f"[fetch] {chosen.get('node_id')} failed ({e}); trying next candidate", flush=True)
            try: os.unlink(tmp_path)
            except Exception: pass
            last_error = str(e)
            continue

        # Verify the streamed SHA-256 before the temporary file can become live.
        actual_sha = h.hexdigest()
        if actual_sha != expected_sha:
            print(f"[fetch] {chosen.get('node_id')} sha mismatch (got {actual_sha[:12]}, expected {expected_sha[:12]}); trying next", flush=True)
            try: os.unlink(tmp_path)
            except Exception: pass
            last_error = "sha mismatch"
            continue

        # Atomic move into place
        os.rename(tmp_path, dest_path)
        if verified_sha_out is not None:
            # Let main() reuse the digest accumulated during download when it
            # registers the slice; do not read a multi-GB file a second time.
            verified_sha_out["sha256"] = actual_sha
        elapsed = time.time() - t0
        print(f"[fetch] saved {dest_path} ({bytes_received:,} bytes, sha={actual_sha[:12]}..., "
              f"{elapsed:.1f}s, {(bytes_received/1e6)/max(elapsed,0.001):.1f} MB/s) from {chosen.get('node_id')}", flush=True)
        _audit("fetch_completed", dest=dest_path, bytes_received=bytes_received,
               sha=actual_sha, elapsed_s=round(elapsed, 2),
               source_node_id=chosen.get("node_id"))
        return dest_path

    _audit("fetch_failed", model_id=model_id,
           layer_start=layer_start, layer_end=layer_end,
           last_error=last_error)
    raise RuntimeError(f"all {len(candidates)} candidate peers failed; last error: {last_error}")


def scan_cache_dir(cache_dir: str, model_id: str,
                    sha_cache: dict = None) -> list:
    """Phase 4a: scan a directory for Nakshatra sub-GGUFs, return list of
    CachedFile-shaped dicts (model_id, model_sha256, layer_start, layer_end,
    size_bytes, file_path).

    Reads each .gguf's metadata for `nakshatra.layer_range_start/end` and
    treats files that don't carry these as not-Nakshatra (skipped).

    `sha_cache` (optional) is a {file_path: sha} dict — useful when the
    caller has already hashed a file (avoids re-streaming 8 GB).

    SHA-256 is otherwise streamed with sidecar caching: a file at
    `<gguf_path>.sha256` records the hash; if it exists and is newer
    than the gguf, we trust it. Cuts ~5-30s per cached file on restart.
    """
    results = []
    sha_cache = sha_cache or {}
    if not os.path.isdir(cache_dir):
        return results

    try:
        import gguf
    except ImportError:
        print(f"[cache-scan] gguf lib not available; only single sub-GGUF will be advertised",
              flush=True)
        return results

    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".gguf"))
    for filename in files:
        path = os.path.join(cache_dir, filename)
        try:
            reader = gguf.GGUFReader(path)
            layer_start = layer_end = None
            for field in reader.fields.values():
                if field.name == "nakshatra.layer_range_start":
                    layer_start = int(field.parts[field.data[0]][0])
                elif field.name == "nakshatra.layer_range_end":
                    layer_end = int(field.parts[field.data[0]][0])
            if layer_start is None or layer_end is None:
                continue  # not a Nakshatra sub-GGUF

            size = os.path.getsize(path)

            # SHA: cache > sidecar > recompute
            if path in sha_cache:
                sha = sha_cache[path]
            else:
                sidecar = path + ".sha256"
                if (os.path.isfile(sidecar) and
                    os.path.getmtime(sidecar) >= os.path.getmtime(path)):
                    try:
                        sha = open(sidecar).read().strip().split()[0]
                        if len(sha) != 64:
                            raise ValueError("malformed sidecar sha")
                    except Exception:
                        sha = None
                    if sha is None:
                        sha = sha256_of_file(path)
                        try: open(sidecar, "w").write(sha + "\n")
                        except Exception: pass
                else:
                    print(f"[cache-scan] hashing {filename} ({size/1e9:.1f} GB)…", flush=True)
                    t0 = time.time()
                    sha = sha256_of_file(path)
                    try: open(sidecar, "w").write(sha + "\n")
                    except Exception: pass
                    print(f"[cache-scan]   sha={sha[:12]}... ({time.time()-t0:.1f}s)", flush=True)

            results.append({
                "model_id": model_id,
                "model_sha256": sha,
                "layer_start": layer_start,
                "layer_end": layer_end,
                "size_bytes": size,
                "file_path": path,
            })
            print(f"[cache-scan] {filename}: layers=[{layer_start},{layer_end}) sha={sha[:12]}", flush=True)
        except Exception as e:
            print(f"[cache-scan] {filename}: skipped ({e})", flush=True)
    return results


def detect_free_vram_gb(backend: str):
    """Phase B: best-effort query of actual free GPU VRAM via the backend's
    SMI tool. Returns dict {free_gb, total_gb, detected_via} on success,
    None if detection isn't supported or fails. Catches the home-pc-OOM
    case where total VRAM is large but most is held by another process
    (ollama, browser, etc.).

    Backends:
      rocm:   rocm-smi --showmeminfo vram --json
      cuda:   nvidia-smi --query-gpu=memory.total,memory.free
      metal:  no clean stdlib path; returns None (operator declares)
    """
    backend = (backend or "").lower()
    try:
        if backend == "rocm":
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode()
            data = json.loads(out)
            for card_id, card in data.items():
                if not card_id.startswith("card"):
                    continue
                total_b = int(card.get("VRAM Total Memory (B)", "0"))
                used_b = int(card.get("VRAM Total Used Memory (B)", "0"))
                if total_b > 0:
                    return {
                        "free_gb": (total_b - used_b) / (1024 ** 3),
                        "total_gb": total_b / (1024 ** 3),
                        "detected_via": "rocm-smi",
                    }
        elif backend == "cuda":
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()
            line = out.splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            total_mb, free_mb = int(parts[0]), int(parts[1])
            return {
                "free_gb": free_mb / 1024,
                "total_gb": total_mb / 1024,
                "detected_via": "nvidia-smi",
            }
        # metal / vulkan / cpu: no auto-detect
    except Exception:
        return None
    return None


def detect_ram_gb() -> float:
    """Best-effort total RAM detection (stdlib only). Returns 0.0 on failure."""
    sys_name = platform.system()
    try:
        if sys_name == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        elif sys_name == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2).decode().strip()
            return int(out) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def detect_disk_avail_gb(path: str = "/") -> float:
    """Free disk space at `path` in GB."""
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return 0.0


def detect_cpu_model() -> str:
    sys_name = platform.system()
    try:
        if sys_name == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys_name == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=2)
            return out.decode().strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def sha256_of_file(path: str) -> str:
    """Stream SHA-256 of a file. ~5s for 8 GB on SSD, fine for one-time startup cost."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def grpc_listen_endpoint(address: str, port: int) -> str:
    """Return a gRPC host:port endpoint, bracketing IPv6 literals once."""
    address = address.strip()
    if address.startswith("[") and address.endswith("]"):
        return f"{address}:{port}"
    if ":" in address:
        return f"[{address}]:{port}"
    return f"{address}:{port}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5500)
    ap.add_argument("--listen-address", type=str,
                    default=(os.environ.get("NAKSHATRA_LISTEN_ADDRESS", "").strip()
                             or "[::]"),
                    help="gRPC bind address (default: [::]; may also be set with "
                         "NAKSHATRA_LISTEN_ADDRESS).")
    ap.add_argument("--sub-gguf", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["first", "middle", "last", "solo"], required=True)
    # 2026-05-29 fabric Phase D — transport selector. Default grpc keeps
    # the existing v0.1 70B cluster (home-pc:5530) unchanged. fabric
    # opts into the network-fabric data plane: the worker calls /join,
    # builds per-pair UDP links, and serves worker↔worker hops over the
    # fabric packet schema. gRPC stays up for Info + the client bridge
    # (sprint open question 8) — fabric isn't a full gRPC replacement
    # in the v0 prototype.
    ap.add_argument("--transport", type=str, choices=["grpc", "fabric"],
                    default="grpc",
                    help="data-plane transport. grpc (default): today's "
                         "path. fabric: network-fabric UDP data plane "
                         "(requires a live chain plan covering this peer; "
                         "refuses boot otherwise).")
    ap.add_argument("--fabric-port", type=int, default=0,
                    help="UDP port for the fabric data plane when "
                         "--transport=fabric. 0 (default) means grpc "
                         "port + 100.")
    ap.add_argument("--layer-start", type=int, required=True)
    ap.add_argument("--layer-end", type=int, required=True)
    ap.add_argument("--model-id", type=str, default="nakshatra-v0.1")
    ap.add_argument("--daemon-bin", type=str, default="/home/operator/llama.cpp/build/bin/llama-nakshatra-worker")
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--n-threads", type=int, default=0,
                    help="threads for llama_decode; 0 = let llama.cpp pick a default")
    ap.add_argument("--n-gpu-layers", type=int, default=0,
                    help="layers to offload to GPU; 0 = CPU only, 99 = all on GPU")
    ap.add_argument("--pillar-url", type=str,
                    default=os.environ.get("NAKSHATRA_PILLAR_URL", ""),
                    help="Sthambha pillar URL (e.g. http://node-pi:7777). If set, "
                         "worker registers self + sends heartbeat every 30s.")
    ap.add_argument("--paid-admission", action="store_true",
                    help="atomically enable consumer resolution and payment-bound "
                         "admission (default: off)")
    ap.add_argument("--paid-ledger-url", type=str,
                    default=os.environ.get("NAKSHATRA_PAID_LEDGER_URL", ""),
                    help="HTTPS Neuron spend API URL used only with --paid-admission")
    ap.add_argument("--paid-ledger-spki-sha256", type=str,
                    default=os.environ.get("NAKSHATRA_PAID_LEDGER_SPKI_SHA256", ""),
                    help="pinned spend-front SPKI SHA-256 used only with --paid-admission")
    ap.add_argument("--paid-store", type=str,
                    default=os.environ.get("NAKSHATRA_PAID_STORE", ""),
                    help="durable payment SQLite path used only with --paid-admission")
    ap.add_argument("--public-address", type=str, default="",
                    help="Address other peers should use to reach this worker "
                         "(default: hostname:port). Override for special routing.")
    ap.add_argument("--node-id", type=str, default="",
                    help="Stable node identifier for the registry. Defaults to "
                         "hostname-port (e.g. node-b-5530).")
    # Phase 3.5 — hardware declarations. Operator-declared (network trusts you).
    ap.add_argument("--gpu-vendor", type=str, default="",
                    help="GPU vendor string e.g. AMD / NVIDIA / Apple / Intel.")
    ap.add_argument("--gpu-model", type=str, default="",
                    help="GPU model string e.g. 'Radeon Pro 5700 XT'.")
    ap.add_argument("--gpu-vram-gb", type=float, default=0.0,
                    help="Total GPU VRAM in GB. 0 = no GPU declared.")
    ap.add_argument("--gpu-backend", type=str, default="cpu",
                    help="Inference backend: rocm/cuda/metal/vulkan/cpu.")
    ap.add_argument("--vram-offered-gb", type=float, default=-1.0,
                    help="VRAM you're offering to the network. Default = gpu-vram-gb.")
    ap.add_argument("--ram-offered-gb", type=float, default=-1.0,
                    help="RAM offered to the network. Default = half of system RAM.")
    ap.add_argument("--cpu-threads-offered", type=int, default=0,
                    help="CPU threads offered. Default = --n-threads value.")
    ap.add_argument("--disk-for-cache-gb", type=float, default=0.0,
                    help="Disk space available for layer cache.")
    # Compute-lease (scale-to-zero) ownership declaration → pillar grace.
    ap.add_argument("--ownership-class", type=str, default="",
                    choices=["", "dedicated", "borrowed"],
                    help="'dedicated' (your own box, longer warm window) or "
                         "'borrowed' (someone else's machine, release fast). "
                         "Empty = don't declare (pillar defaults to borrowed).")
    ap.add_argument("--idle-grace-s", type=float, default=0.0,
                    help="Idle seconds before the pillar reaps this node's lease. "
                         "0 = pillar uses the ownership-class default.")
    ap.add_argument("--skip-sha256", action="store_true",
                    help="LOCAL DEVELOPMENT ONLY: skip the local sub-GGUF hash. "
                         "Requires --local-dev and refuses remote package/peer fetches.")
    ap.add_argument("--local-dev", action="store_true",
                    help="Explicit local-development mode. May only use an existing "
                         "local slice or a local/file/loopback package source; never "
                         "use on an outsider worker.")
    ap.add_argument("--file-server-port", type=int, default=0,
                    help="Phase 4: HTTP file-server port for peer-to-peer fetch. "
                         "Default = grpc port + 1000 (e.g. 5530 → 6530).")
    ap.add_argument("--no-file-server", action="store_true",
                    help="Disable the Phase-4 HTTP file server (this worker won't "
                         "let peers fetch its sub-GGUF).")
    ap.add_argument("--auto-fetch", action="store_true",
                    help="If --sub-gguf doesn't exist, query --pillar-url for who "
                         "has the file and download it before spawning the daemon.")
    ap.add_argument("--cache-dir", type=str, default="",
                    help="Directory to scan for additional cached sub-GGUFs that "
                         "this worker can serve to peers. Default: parent of "
                         "--sub-gguf. Workers advertise EVERY Nakshatra sub-GGUF "
                         "found here in cached_files (Phase 4a redundancy).")
    ap.add_argument("--no-cache-scan", action="store_true",
                    help="Disable cache-dir scan; only advertise --sub-gguf.")
    # v1.0 §5 (P2) — provision from a content-addressed layer package.
    ap.add_argument("--package-url", type=str,
                    default=os.environ.get("NAKSHATRA_PACKAGE_URL", ""),
                    help="v1.0 §5: if --sub-gguf is missing, provision the assigned "
                         "[layer-start,layer-end) range from a layer package (a dir, "
                         "package.json path, or http(s) base) — fetch only this "
                         "position's fragments, verify each SHA-256 fail-closed, and "
                         "assemble the sub-GGUF. Needs no peer holding the exact range. "
                         "Takes precedence over --auto-fetch.")
    ap.add_argument("--package-require-signature", action="store_true",
                    help="Refuse an unsigned package manifest (use with --package-url).")
    ap.add_argument("--package-trusted-pubkey", action="append", default=None,
                    help="Only accept a package manifest signed by this Ed25519 hex "
                         "pubkey (repeatable; use with --package-url).")
    ap.add_argument("--no-vram-autodetect", action="store_true",
                    help="Disable Phase-B auto-detection of free VRAM via "
                         "rocm-smi/nvidia-smi. Use the declared --vram-offered-gb "
                         "as-is (operator override).")
    # v0.5 §9.7 closure: idempotency cache sizing.
    ap.add_argument("--idempotency-cache-mb", type=int, default=64,
                    help="Idempotency cache memory cap, MB (default 64). "
                         "Converted internally to an entry count at "
                         f"~{IDEM_BYTES_PER_ENTRY//1024} KB per cached step.")
    ap.add_argument("--idempotency-cache-ttl", type=float, default=60.0,
                    help="Idempotency cache per-session TTL in seconds (default 60).")
    args = ap.parse_args()

    try:
        idle_unload_s = float(
            os.environ.get("NAKSHATRA_WORKER_IDLE_UNLOAD_S", "600")
        )
        if not math.isfinite(idle_unload_s) or idle_unload_s < 0:
            raise ValueError(
                "NAKSHATRA_WORKER_IDLE_UNLOAD_S must be a finite number >= 0"
            )
        release_grace_s = float(
            os.environ.get("NAKSHATRA_WORKER_RELEASE_GRACE_S", "30")
        )
        if not math.isfinite(release_grace_s):
            raise ValueError(
                "NAKSHATRA_WORKER_RELEASE_GRACE_S must be finite "
                "(negative disables it)"
            )
        warm_directives_raw = os.environ.get(
            "NAKSHATRA_WORKER_WARM_DIRECTIVES", "1"
        ).strip().lower()
        if warm_directives_raw in {"1", "true", "yes", "on"}:
            warm_directives_enabled = True
        elif warm_directives_raw in {"0", "false", "no", "off"}:
            warm_directives_enabled = False
        else:
            raise ValueError(
                "NAKSHATRA_WORKER_WARM_DIRECTIVES must be 1 or 0"
            )
        owner_lease_config = owner_lease_config_from_env()
        validate_local_dev_fetch_policy(
            local_dev=args.local_dev,
            skip_sha256=args.skip_sha256,
            package_url=args.package_url,
            auto_fetch=args.auto_fetch,
        )
        spki_hash = validate_spki_hash_env(
            os.environ.get("STHAMBHA_PILLAR_SPKI_SHA256")
        )
    except ValueError as e:
        ap.error(str(e))
    if args.pillar_url.startswith("https://") and not spki_hash:
        if _release_service_running():
            sys.exit("[worker] startup refused: signed-release HTTPS pillar has no SPKI pin")
        print(f"[worker] WARN: HTTPS pillar URL but no SPKI hash pinned "
              f"via STHAMBHA_PILLAR_SPKI_SHA256; cert identity NOT verified",
              flush=True)
    if args.local_dev:
        print(
            "[worker] !!! LOCAL DEVELOPMENT MODE !!! Only local/file/loopback "
            "model sources are permitted; do not use this mode on an outsider worker.",
            flush=True,
        )
    if args.skip_sha256:
        print(
            "[worker] !!! SHA-256 REGISTRATION HASH DISABLED FOR LOCAL DEVELOPMENT !!!",
            flush=True,
        )

    # Phase D5 (2026-05-20): initialise the audit log singleton early so
    # every event in main() (start, fetch, slice, register, auth fail)
    # has a place to land. Failure-soft: missing directory or permission
    # error leaves the worker running without forensic trail.
    if _AUDIT_AVAILABLE:
        try:
            audit_path = os.environ.get("NAKSHATRA_AUDIT_PATH", "").strip()
            _waudit.init_audit(Path(audit_path) if audit_path else None)
            _audit("worker_started",
                   model_id=args.model_id, mode=args.mode,
                   layer_start=args.layer_start, layer_end=args.layer_end,
                   port=args.port,
                   pillar_url=pillar_location_for_log(args.pillar_url))
        except Exception as e:
            print(f"[worker] audit init failed (continuing without forensic log): {e}",
                  flush=True)

    # Establish the persistent worker identity before any bootstrap request.
    # A fresh slice may require a signed pillar /files query before the daemon
    # or registration path exists, so credentials cannot be initialized later.
    node_id = args.node_id or f"{socket.gethostname()}-{args.port}"
    worker_priv: Optional[bytes] = None
    worker_pub_hex = ""
    if args.pillar_url:
        if _AUTH_AVAILABLE:
            try:
                worker_priv, worker_pub_hex = _wauth.load_or_create_worker_key()
            except Exception as e:
                print(f"[worker] WARN: could not load worker key ({e}); "
                      f"registration will be unsigned and authenticated bootstrap "
                      f"fetches will be refused.", flush=True)
        else:
            print(f"[worker] WARN: nakshatra_auth import failed "
                  f"({_AUTH_IMPORT_ERR}); registration will be unsigned and "
                  f"authenticated bootstrap fetches will be refused", flush=True)

    # Phase 4: if sub-GGUF is missing AND we have a pillar to ask, fetch it
    # before doing anything else. This is what lets a fresh machine bootstrap
    # without manual scp.
    provisioned_sha256 = ""
    if not os.path.exists(args.sub_gguf):
        if args.package_url:
            package_log_location = package_location_for_log(args.package_url)
            print(f"[worker] sub-gguf missing at {args.sub_gguf}; provisioning layers "
                  f"[{args.layer_start},{args.layer_end}) from package "
                  f"{package_log_location}",
                  flush=True)
            trusted = (set(args.package_trusted_pubkey)
                       if args.package_trusted_pubkey else None)
            try:
                provision_from_package(
                    args.package_url, args.layer_start, args.layer_end, args.sub_gguf,
                    require_signature=args.package_require_signature,
                    trusted_pubkeys=trusted,
                    local_dev=args.local_dev,
                )
            except Exception as e:
                detail = str(e).replace(args.package_url, package_log_location)
                sys.exit(f"[worker] package provisioning failed: {detail}")
        elif args.auto_fetch and args.pillar_url:
            print(f"[worker] sub-gguf missing at {args.sub_gguf}; auto-fetching from peer", flush=True)
            if not worker_priv:
                sys.exit(
                    "[worker] auto-fetch refused: authenticated pillar /files "
                    "query requires a usable worker key"
                )
            fetch_result = {}
            try:
                fetch_sub_gguf_from_peer(
                    args.pillar_url, args.model_id,
                    args.layer_start, args.layer_end,
                    args.sub_gguf,
                    own_node_id=node_id,
                    priv_key=worker_priv,
                    spki_hash=spki_hash,
                    verified_sha_out=fetch_result,
                )
                provisioned_sha256 = fetch_result.get("sha256", "")
            except Exception as e:
                sys.exit(f"[worker] auto-fetch failed: {e}")
        else:
            sys.exit(f"[worker] sub-gguf does not exist: {args.sub_gguf} "
                     f"(use --package-url to provision from a layer package, or "
                     f"--auto-fetch with --pillar-url to bootstrap from a peer)")

    residency_event = threading.Event()

    def spawn_daemon():
        print(f"[worker] spawning daemon: {args.daemon_bin} {args.sub_gguf} {args.mode} {args.n_ctx} threads={args.n_threads} gpu_layers={args.n_gpu_layers}", flush=True)
        return DaemonClient(
            args.daemon_bin, args.sub_gguf, args.mode, args.n_ctx,
            args.n_threads, args.n_gpu_layers,
        )

    daemon = IdleDaemonController(
        spawn_daemon, args.sub_gguf, "", idle_unload_s,
        release_grace_s=release_grace_s,
        owner_lease_enabled=owner_lease_config is not None,
        residency_event=residency_event,
        model_id=args.model_id,
        warm_directives_enabled=warm_directives_enabled,
    )

    # 2026-05-26 drive-by — TLS bringup factored into setup_tls(args).
    # Same behavior (resolve policy, ensure_cert, WARN on Mode-A
    # downgrade), just out of main()'s critical path.
    tls = setup_tls(args)

    # Phase A1 (2026-05-20): explicit gRPC message-size cap. Default is
    # 4 MiB; we set the explicit cap to WORKER_GRPC_MAX_MESSAGE_BYTES so
    # operators can grep the constant + adjust without spelunking gRPC
    # defaults. 16 MiB covers our largest hidden_state batches with room.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_length", WORKER_GRPC_MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", WORKER_GRPC_MAX_MESSAGE_BYTES),
        ],
    )
    idem_max_entries = max(1, (args.idempotency_cache_mb * 1024 * 1024) // IDEM_BYTES_PER_ENTRY)

    # Phase B (2026-05-20): resolve NAKSHATRA_AUTH_REQUIRED + decide
    # whether to refuse SSRF pushes to unregistered peers. The peer
    # resolver itself is constructed below once we know the worker key.
    auth_required = resolve_worker_auth_required(args)
    refuse_unregistered_peers = (
        os.environ.get("NAKSHATRA_REFUSE_UNREGISTERED_PEERS", "true")
        .strip().lower() in ("true", "1", "yes")
    )
    # 2026-05-21 SPKI Phase 3.3: outbound SPKI pin policy. Default
    # true closes the cross-worker MITM gap; operators with legacy
    # Mode-A clusters opt out. Independent of refuse_unregistered_peers:
    # the SSRF allowlist controls WHO you may push to, the pin
    # controls WHAT IDENTITY they must present.
    refuse_unpinned_peers = (
        os.environ.get("NAKSHATRA_REFUSE_UNPINNED_PEERS", "true")
        .strip().lower() in ("true", "1", "yes")
    )

    paid_admission = None
    paid_store = None
    if args.paid_admission:
        if not _PAID_ADMISSION_AVAILABLE:
            print(
                f"[worker] paid admission NOT ready: module unavailable "
                f"({_PAID_ADMISSION_IMPORT_ERR})", flush=True)
        elif not auth_required:
            print(
                "[worker] paid admission NOT ready: authenticated gRPC callers "
                "are required", flush=True)
        elif args.transport != "grpc":
            print(
                "[worker] paid admission NOT ready: paid v1 requires direct "
                "gRPC transport", flush=True)
        elif not worker_priv or not worker_pub_hex:
            print(
                "[worker] paid admission NOT ready: persistent worker identity "
                "is unavailable", flush=True)
        elif not args.paid_store:
            print(
                "[worker] paid admission NOT ready: --paid-store is required",
                flush=True)
        else:
            try:
                paid_store = _paid.PaidAdmissionStore(args.paid_store)
                paid_ledger = _paid.SpendLedgerClient(
                    args.paid_ledger_url,
                    args.paid_ledger_spki_sha256,
                    worker_priv,
                    worker_pub_hex,
                )
                paid_admission = _paid.PaidAdmission(
                    paid_ledger, paid_store, worker_priv,
                    model_id=args.model_id,
                    layer_start=args.layer_start,
                    layer_end=args.layer_end,
                )
                if paid_admission.check_ready():
                    replayed = paid_admission.flush_outbox()
                    print(
                        f"[worker] paid admission ready; replayed {replayed} "
                        "durable settlement receipt(s)", flush=True)
                else:
                    print(
                        f"[worker] paid admission NOT ready: "
                        f"{paid_admission.readiness_error}", flush=True)
            except Exception as exc:
                if paid_store is not None:
                    paid_store.close()
                    paid_store = None
                paid_admission = None
                print(
                    f"[worker] paid admission NOT ready: {type(exc).__name__}: "
                    f"{exc}", flush=True)

    servicer = WorkerServicer(daemon, args.mode, args.layer_start, args.layer_end, args.model_id,
                              idem_max_entries=idem_max_entries,
                              idem_ttl_seconds=args.idempotency_cache_ttl,
                              auth_required=auth_required,
                              refuse_unregistered_peers=refuse_unregistered_peers,
                              refuse_unpinned_peers=refuse_unpinned_peers,
                              paid_admission=paid_admission,
                              worker_priv=worker_priv,
                              worker_pub_hex=worker_pub_hex)
    servicer.paid_gate_enabled = bool(args.paid_admission)
    servicer.fabric_transport_enabled = args.transport == "fabric"
    # Registration (and with it the peer-key resolver) happens AFTER the port is bound; until then report STARTING.
    servicer.starting = bool(auth_required and args.pillar_url)
    print(f"[worker] idempotency cache: {args.idempotency_cache_mb} MB "
          f"(~{idem_max_entries} entries @ {IDEM_BYTES_PER_ENTRY//1024} KB), "
          f"ttl={args.idempotency_cache_ttl}s", flush=True)
    pb_grpc.add_NakshatraServicer_to_server(servicer, server)
    # 2026-05-21 SPKI Phase 2.4: bind TLS port when cert was prepared
    # above; otherwise legacy insecure path.
    listen_endpoint = grpc_listen_endpoint(args.listen_address, args.port)
    if tls.required and tls.cert_path and tls.key_path:
        creds = _wtls.build_grpc_server_credentials(
            tls.cert_path, tls.key_path)
        server.add_secure_port(listen_endpoint, creds)
        listen_proto = "TLS"
    else:
        server.add_insecure_port(listen_endpoint)
        listen_proto = "plaintext"
    print(f"[worker] M5 listening on {listen_endpoint} ({listen_proto})  "
          f"mode={args.mode}  layers=[{args.layer_start},{args.layer_end})  "
          f"model={args.model_id}", flush=True)
    server.start()

    owner_lease = None
    owner_lease_httpd = None
    owner_lease_http_thread = None
    if owner_lease_config is not None:
        owner_lease = OwnerLease(owner_lease_config, daemon)
        owner_lease.reconcile()
        try:
            owner_lease_httpd, owner_lease_http_thread = start_owner_lease_server(
                owner_lease
            )
        except OSError as e:
            daemon.close()
            server.stop(grace=0)
            sys.exit(
                f"[worker] startup refused: owner lease endpoint could not bind "
                f"127.0.0.1:{owner_lease_config.port}: {e}"
            )
        print(
            f"[worker] owner lease enabled on "
            f"127.0.0.1:{owner_lease_config.port}; default deny (away)",
            flush=True,
        )

    # Idle reload integrity is independent of pillar registration policy.
    # Record the bytes that the initial daemon loaded, then require an exact
    # match before every later spawn.  With idle unload disabled, retain the
    # legacy startup path and hash only if registration needs it below.
    startup_slice_sha256 = ""
    sub_gguf_sha256 = ""
    sub_gguf_size = 0
    if idle_unload_s > 0 or release_grace_s >= 0 or owner_lease_config is not None:
        try:
            sub_gguf_size = os.path.getsize(args.sub_gguf)
            if provisioned_sha256:
                startup_slice_sha256 = provisioned_sha256
                print(
                    f"[worker] recording verified download sha256="
                    f"{startup_slice_sha256[:16]}... for idle reload",
                    flush=True,
                )
            else:
                t0 = time.time()
                print(
                    f"[worker] recording startup sha256 of {args.sub_gguf} "
                    f"({sub_gguf_size/1e9:.1f} GB) for idle reload…",
                    flush=True,
                )
                startup_slice_sha256 = sha256_of_file(args.sub_gguf)
                print(
                    f"[worker] startup sha256={startup_slice_sha256[:16]}... "
                    f"({time.time()-t0:.1f}s)",
                    flush=True,
                )
            daemon.record_startup_sha256(startup_slice_sha256)
            if not args.skip_sha256:
                sub_gguf_sha256 = startup_slice_sha256
        except Exception as e:
            daemon.close()
            server.stop(grace=0)
            if owner_lease_httpd is not None:
                owner_lease_httpd.shutdown()
                owner_lease_httpd.server_close()
            sys.exit(
                f"[worker] startup refused: cannot record sub-GGUF SHA-256 "
                f"required for idle reload: {e}"
            )

    # Phase 4: file server lets peers fetch this worker's sub-GGUF over HTTP
    # byte-range. New peers joining the cluster don't need their files
    # pre-shipped — they query the pillar's file index, find a peer that has
    # the file, and download it. Convention: file-server port = grpc port + 1000.
    file_server_port = args.file_server_port or (args.port + 1000)
    _HEALTH_STATE.update({
        "daemon": daemon,
        "servicer": servicer,
        "started_at": time.time(),
        "node_id": node_id,
        "model_id": args.model_id,
        "mode": args.mode,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "grpc_port": args.port,
        "file_server_port": file_server_port if not args.no_file_server else 0,
        "gpus_present": detect_gpus(),
    })
    if not args.no_file_server:
        try:
            start_file_server(serving_dir=str(Path(args.sub_gguf).parent),
                              port=file_server_port)
        except Exception as e:
            print(f"[fileserver] failed to start: {e} (peers won't be able to fetch from this worker)", flush=True)
            file_server_port = 0
            _HEALTH_STATE["file_server_port"] = 0
    else:
        file_server_port = 0

    # Pillar registration (Phase 3b). Best-effort — worker still serves
    # requests if the pillar is unreachable (back-compat with static YAML
    # cluster configs). Heartbeat thread keeps the registry view fresh.
    stop_event = threading.Event()
    heartbeat_thread = None
    if paid_admission is not None and paid_admission.ready:
        threading.Thread(
            target=paid_admission.outbox_loop,
            args=(stop_event,),
            daemon=True,
            name="paid-settlement-outbox",
        ).start()
    if args.pillar_url:
        public_addr = args.public_address or f"{socket.gethostname()}:{args.port}"

        # Phase 3.5: compute SHA-256 of the sub-GGUF (one-time at startup)
        try:
            if not sub_gguf_size:
                sub_gguf_size = os.path.getsize(args.sub_gguf)
            if not args.skip_sha256:
                if sub_gguf_sha256:
                    print(
                        f"[worker] reusing startup sha256={sub_gguf_sha256[:16]}... "
                        "for pillar registration",
                        flush=True,
                    )
                elif provisioned_sha256:
                    sub_gguf_sha256 = provisioned_sha256
                    print(
                        f"[worker] reusing download sha256={sub_gguf_sha256[:16]}... "
                        "for pillar registration",
                        flush=True,
                    )
                else:
                    t0 = time.time()
                    print(f"[worker] computing sha256 of {args.sub_gguf} ({sub_gguf_size/1e9:.1f} GB)…", flush=True)
                    sub_gguf_sha256 = sha256_of_file(args.sub_gguf)
                    print(f"[worker] sha256={sub_gguf_sha256[:16]}... ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[worker] sub-GGUF inspection failed (continuing without hash): {e}", flush=True)

        # Hardware auto-detection (best-effort) + operator overrides
        ram_total = detect_ram_gb()
        disk_avail = detect_disk_avail_gb(os.path.dirname(args.sub_gguf) or "/")
        cpu_model = detect_cpu_model()
        cpu_threads = os.cpu_count() or 0

        # Phase 3.6 — verify declared GPU backend against daemon reality.
        # The daemon's stderr tells us what actually happened on model load
        # (it prints "offloaded N/M layers to GPU"). If we declared a GPU
        # backend on the CLI but the daemon offloaded 0 layers, the binary
        # was almost certainly built without that backend. We DOWNGRADE the
        # declaration to "cpu" before posting to the pillar — better to be
        # truthful than to lie about capability.
        offload_status = daemon.gpu_offload_status()
        actual_backend, offload_verified = decide_registered_backend(
            args.gpu_vram_gb, args.gpu_backend, offload_status)
        declared_gpu = (args.gpu_vram_gb > 0 and args.gpu_backend != "cpu")
        if declared_gpu and not offload_verified and actual_backend == args.gpu_backend:
            print(f"[worker] NOTE: declared --gpu-backend={args.gpu_backend} but the daemon logged no offload "
                  f"line (a quiet build) - keeping the declaration, registering it as UNVERIFIED", flush=True)
        elif declared_gpu and actual_backend != args.gpu_backend:
            print(f"[worker] WARNING: declared --gpu-backend={args.gpu_backend} "
                  f"but daemon offloaded {offload_status['n_offloaded']}/"
                  f"{offload_status['total_layers']} layers — daemon binary likely "
                  f"lacks {args.gpu_backend} support; downgrading registration to cpu",
                  flush=True)
        elif declared_gpu:
            print(f"[worker] verified: daemon offloaded "
                  f"{offload_status['n_offloaded']}/{offload_status['total_layers']} "
                  f"layers via {args.gpu_backend} (backends seen in log: "
                  f"{offload_status['backend_hints']})", flush=True)

        gpus = []
        if args.gpu_vram_gb > 0:
            gpus.append({
                "vendor": args.gpu_vendor or "unknown",
                "model": args.gpu_model or "unknown",
                "vram_total_gb": args.gpu_vram_gb,
                "backend": actual_backend,
                "actual_layers_offloaded": offload_status["n_offloaded"],
                "offload_verified": offload_verified,
                "total_layers_loaded": offload_status["total_layers"],
            })
        hardware = {
            "platform": platform.system().lower(),
            "arch": platform.machine(),
            "cpu_model": cpu_model,
            "cpu_threads": cpu_threads,
            "ram_total_gb": ram_total,
            "disk_avail_gb": disk_avail,
            "gpus": gpus,
        }

        # Budget: operator declares; sensible defaults
        vram_offered = args.vram_offered_gb if args.vram_offered_gb >= 0 else args.gpu_vram_gb
        ram_offered = args.ram_offered_gb if args.ram_offered_gb >= 0 else max(ram_total / 2, 0.0)
        cpu_offered = args.cpu_threads_offered or args.n_threads or cpu_threads

        # Phase B: auto-detect actual free VRAM and downgrade if declared
        # is over-optimistic. Catches the case where another process
        # (ollama, browser, prior daemon) is hogging the GPU.
        if not args.no_vram_autodetect:
            vram_detected = detect_free_vram_gb(args.gpu_backend)
            if vram_detected:
                free = vram_detected["free_gb"]
                total = vram_detected["total_gb"]
                via = vram_detected["detected_via"]
                if free < vram_offered:
                    print(f"[worker] vram-autodetect ({via}): only {free:.1f} GB free of "
                          f"{total:.1f} GB; downgrading vram_offered_gb {vram_offered:.1f} → {free:.1f}",
                          flush=True)
                    vram_offered = free
                else:
                    print(f"[worker] vram-autodetect ({via}): {free:.1f} GB free of "
                          f"{total:.1f} GB; declared {vram_offered:.1f} GB OK", flush=True)
                # Also override gpu_vram_gb display if operator left it default
                if args.gpu_vram_gb <= 0:
                    args.gpu_vram_gb = total
            else:
                if args.gpu_backend in ("rocm", "cuda"):
                    print(f"[worker] vram-autodetect: failed for backend={args.gpu_backend} "
                          f"(SMI tool missing or unreadable); trusting declared values",
                          flush=True)
                # metal/vulkan/cpu: no detection; silent

        budget = {
            "vram_offered_gb": vram_offered,
            "ram_offered_gb": ram_offered,
            "cpu_threads_offered": cpu_offered,
            "disk_for_cache_gb": args.disk_for_cache_gb,
        }

        # Phase 4a: scan the cache dir for ALL Nakshatra sub-GGUFs (not
        # just the one this worker is serving). Any peer with multiple
        # cached files advertises them all — natural redundancy.
        cached_files = []
        if not args.no_cache_scan and not args.skip_sha256:
            cache_dir = args.cache_dir or str(Path(args.sub_gguf).resolve().parent)
            sha_seed = {str(Path(args.sub_gguf).resolve()): sub_gguf_sha256} if sub_gguf_sha256 else {}
            cached_files = scan_cache_dir(cache_dir, args.model_id, sha_cache=sha_seed)
            print(f"[worker] cache-scan found {len(cached_files)} sub-GGUF(s) in {cache_dir}", flush=True)
        elif args.skip_sha256:
            print(
                "[worker] local-dev --skip-sha256 also disables cache scanning "
                "so no model file is hashed indirectly",
                flush=True,
            )

        # Fallback: if scan found nothing (legacy / non-Nakshatra files),
        # advertise just the one we're serving.
        if not cached_files and sub_gguf_size > 0:
            cached_files = [{
                "model_id": args.model_id,
                "model_sha256": sub_gguf_sha256,
                "layer_start": args.layer_start,
                "layer_end": args.layer_end,
                "size_bytes": sub_gguf_size,
                "file_path": str(Path(args.sub_gguf).resolve()),
            }]

        register_payload = {
            "node_id": node_id,
            "node_type": "compute",
            "address": public_addr,
            "layer_offerings": [{
                "model_id": args.model_id,
                "model_sha256": sub_gguf_sha256,
                "layer_start": args.layer_start,
                "layer_end": args.layer_end,
            }],
            "hardware": hardware,
            "budget": budget,
            "cached_files": cached_files,
            "recent_rpc_ms": 0.0,  # Phase H — populated by heartbeat as data accrues
            "residency": daemon.residency,
        }
        try:
            activation_endpoint = activation_endpoint_from_env()
        except ValueError as e:
            sys.exit(f"[worker] startup refused: {e}")
        if activation_endpoint is not None:
            register_payload["endpoint"] = activation_endpoint
        # Compute-lease ownership (scale-to-zero): declare to the pillar how
        # patiently it may keep this node summoned. Sent in the fabric block the
        # pillar already parses; other fabric fields keep their safe defaults.
        if args.ownership_class or args.idle_grace_s > 0:
            register_payload["fabric"] = {
                "ownership_class": args.ownership_class or "borrowed",
                "idle_grace_s": float(args.idle_grace_s),
            }
        # 2026-05-21 SPKI Phase 2.6: declare the worker's gRPC server
        # SPKI hash so the pillar can distribute it via /peers. Peer
        # workers will pin the TLS handshake against this value in
        # Phase 3. Empty string when TLS is disabled — pillar will
        # store "" and Phase 3 pinning refuses outbound to such peers
        # when NAKSHATRA_REFUSE_UNPINNED_PEERS=true (also a Phase 3
        # default).
        if tls.spki_hash:
            register_payload["peer_spki_hash"] = tls.spki_hash
        # (sandbox_compliance is filled in below once we run the
        # startup compliance check; see Phase G integration just below.)
        # Phase G: snapshot the worker's runtime sandbox compliance at
        # startup. We don't have a SandboxSpec yet (that comes from a
        # /join with a live plan), so we validate against a generic
        # "is this thing containerized at all" check. The full per-plan
        # validation runs later when /join lands.
        sandbox_summary: dict = {}
        sandbox_facts = None
        if _SANDBOX_AVAILABLE:
            sandbox_facts = _wsandbox.collect_runtime_facts()
            generic_spec = {
                "seccomp_profile": "any",
                "cpu_threads_limit": 0,
                "ram_limit_gb": 0,
                "mode_c_compatible": True,
            }
            startup_report = _wsandbox.validate_against_runtime(
                generic_spec, sandbox_facts)
            sandbox_summary = _wsandbox.compliance_summary_for_peer_body(
                startup_report)
            print(f"[worker] {startup_report.format_human()}", flush=True)
            # Operator-controlled Mode-C gate (Phase G3).
            refuse_noncompliant = (
                os.environ.get("STHAMBHA_REFUSE_NONCOMPLIANT_SANDBOX", "")
                .strip().lower() in ("true", "1", "yes")
            )
            if refuse_noncompliant and not startup_report.is_mode_c_compliant():
                print(f"[worker] REFUSING TO START: "
                      f"STHAMBHA_REFUSE_NONCOMPLIANT_SANDBOX=true and "
                      f"runtime is not Mode-C compliant. Run inside a "
                      f"container that satisfies the SandboxSpec — see "
                      f"docs/SANDBOX-EXAMPLES/ on the sthambha repo.",
                      flush=True)
                sys.exit(2)
        else:
            print(f"[worker] WARN: nakshatra_sandbox import failed "
                  f"({_SANDBOX_IMPORT_ERR}); skipping sandbox compliance check",
                  flush=True)
        if sandbox_summary:
            register_payload["sandbox_compliance"] = sandbox_summary

        # Phase F2: the persistent Ed25519 key was loaded before bootstrap
        # provisioning so /files could never race ahead unsigned. Include its
        # public half in the first registration for the pillar's TOFU lock.
        if worker_pub_hex:
            register_payload["public_key_hex"] = worker_pub_hex
        # Phase A5 (2026-05-20): refuse-unsigned startup gate. Operators
        # opt in with STHAMBHA_REFUSE_UNSIGNED=true; default keeps the
        # legacy lenient bringup (worker WARNs and proceeds).
        if should_refuse_unsigned_startup(
            os.environ.get("STHAMBHA_REFUSE_UNSIGNED"),
            _AUTH_AVAILABLE,
            has_worker_key=bool(worker_priv),
            pillar_url=args.pillar_url,
        ):
            sys.exit(
                "[worker] startup refused: STHAMBHA_REFUSE_UNSIGNED=true "
                "but no worker Ed25519 key is available (auth module "
                "missing or key load failed)."
            )
        def register_current_residency():
            # Registration retries may overlap an idle transition. Never
            # replay the startup value after the daemon has entered standby.
            register_payload["residency"] = daemon.residency
            return register_with_pillar(
                args.pillar_url, register_payload,
                priv_key=worker_priv, node_id=node_id, spki_hash=spki_hash,
                warm_directive_handler=(
                    daemon.apply_warm_directive
                    if warm_directives_enabled else None
                ),
                model_id=args.model_id,
            )

        registered_ok = bool(register_current_residency())

        # Phase B (2026-05-20): start the peer-key resolver in the
        # background. The resolver caches the pillar's /peers projection
        # so the gRPC auth check can map an authenticated request's
        # keyid to its registered Ed25519 public key. Without this the
        # worker has nothing to verify against.
        peer_refresh_interval = float(
            os.environ.get("NAKSHATRA_PEER_REFRESH_INTERVAL", "60.0")
        )
        resolver = None
        if (_GRPC_AUTH_AVAILABLE and (auth_required or refuse_unregistered_peers)
                and args.pillar_url):
            resolver = _wgrpcauth.PillarPeerKeyResolver(
                args.pillar_url,
                refresh_interval_s=peer_refresh_interval,
                priv_key=worker_priv, own_node_id=node_id,
                spki_hash=spki_hash,
                enable_consumers=bool(
                    args.paid_admission
                    and paid_admission is not None
                    and paid_admission.ready),
            )
            resolver.start_background_refresh()
            servicer.peer_resolver = resolver
            print(
                f"[worker] gRPC peer-key resolver started "
                f"(refresh every {peer_refresh_interval:.0f}s; "
                f"auth_required={auth_required}, "
                f"refuse_unregistered_peers={refuse_unregistered_peers})",
                flush=True,
            )
        elif auth_required:
            print(
                "[worker] WARN: NAKSHATRA_AUTH_REQUIRED=true but no "
                "--pillar-url is set; auth will reject every gRPC call "
                "since there's no resolver. Did you forget --pillar-url?",
                flush=True,
            )
        # READY = registered AND the peer-key cache holds our own key (see _startup_blocker). Not merely "attempted".
        _ready_keyids = tuple(k.strip() for k in os.environ.get("NAKSHATRA_READY_KEYIDS", "").split(",") if k.strip())
        _blocker = _startup_blocker(registered_ok, resolver, node_id, _ready_keyids)
        if not _blocker:
            servicer.starting = False
        elif servicer.starting:
            servicer.starting_reason = _blocker
            print(f"[worker] NOT ready yet: {_blocker} - answering UNAVAILABLE and retrying in the background", flush=True)
            threading.Thread(
                target=_await_startup_ready, name="startup-ready", daemon=True,
                args=(servicer,
                      register_current_residency,
                      resolver, node_id, _ready_keyids),
                kwargs={"registered": registered_ok},
            ).start()

        # Phase C (2026-05-20): populate the HTTP auth state read by
        # FileServerHandler. Same resolver as gRPC auth (shared cache).
        # Operator pubkey is loaded once at startup; rotation requires
        # restart (good — operator key rotations are infrequent and
        # serious enough that a planned restart is fine).
        operator_pubkey = _load_operator_pubkey()
        slice_root = Path(
            os.environ.get("NAKSHATRA_SLICE_ROOT", "").strip()
            or str(Path(args.sub_gguf).resolve().parent)
        )
        _HTTP_AUTH_STATE.update({
            "auth_required": auth_required,
            "operator_pubkey": operator_pubkey,
            "peer_resolver": resolver,
            "slice_root": slice_root,
        })
        if operator_pubkey:
            print(
                f"[worker] operator pubkey installed: "
                f"{operator_pubkey[:16]}… (POST /slice gated by signature "
                f"against this key)",
                flush=True,
            )
        else:
            print(
                f"[worker] no operator pubkey at {OPERATOR_PUBKEY_PATH}; "
                f"POST /slice will refuse all callers. Install with: "
                f"echo '<hex>' > {OPERATOR_PUBKEY_PATH} && chmod 600 {OPERATOR_PUBKEY_PATH}",
                flush=True,
            )

        # Heartbeat payload omits public_key_hex after the first call —
        # the pillar's TOFU lock remembers it.
        heartbeat_payload = {k: v for k, v in register_payload.items()
                             if k != "public_key_hex"}
        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            args=(args.pillar_url, heartbeat_payload, 30.0, stop_event),
            kwargs={"daemon_for_timing": daemon,
                    "priv_key": worker_priv, "node_id": node_id,
                    "spki_hash": spki_hash,
                    "residency_event": residency_event,
                    "warm_directive_handler": (
                        daemon.apply_warm_directive
                        if warm_directives_enabled else None
                    ),
                    "model_id": args.model_id},
            daemon=True,
        )
        heartbeat_thread.start()
        print(f"[worker] heartbeat → {pillar_location_for_log(args.pillar_url)} "
              f"every 30s as {node_id} @ {public_addr}"
              + (f" (signed; pubkey={worker_pub_hex[:16]}…)" if worker_pub_hex else " (UNSIGNED)"),
              flush=True)

    idle_thread = None
    if idle_unload_s > 0 or release_grace_s >= 0:
        idle_thread = threading.Thread(
            target=daemon.idle_monitor_loop,
            args=(stop_event,),
            daemon=True,
            name="daemon-idle-unload",
        )
        idle_thread.start()
        features = []
        if idle_unload_s > 0:
            features.append(f"idle backstop {idle_unload_s:g}s")
        if release_grace_s >= 0:
            features.append(f"release grace {release_grace_s:g}s")
        print(f"[worker] daemon unload enabled: {', '.join(features)} "
              "(gRPC/TLS/heartbeat stay resident)", flush=True)
    else:
        print("[worker] idle/release unload disabled; daemon remains loaded", flush=True)

    owner_lease_thread = None
    if owner_lease is not None:
        owner_lease_thread = threading.Thread(
            target=owner_lease.monitor_loop,
            args=(stop_event,),
            daemon=True,
            name="owner-lease-monitor",
        )
        owner_lease_thread.start()

    # 2026-05-29 fabric Phase D — bring up the fabric data plane when
    # --transport=fabric. gRPC server above stays up for Info + the
    # client→first-worker bridge (sprint open question 8); the fabric
    # backend handles worker↔worker FORWARD/FEEDBACK hops. setup_fabric
    # refuses boot (sys.exit) if no chain plan covers this peer yet.
    fabric_backend = None
    if args.transport == "fabric":
        fabric_backend, _fabric_join = setup_fabric(
            args, servicer._run_forward, servicer.n_embd,
            worker_priv=worker_priv, node_id=node_id,
            pillar_spki_hash=spki_hash,
        )
        if fabric_backend.inbound_link is not None:
            fabric_thread = threading.Thread(
                target=fabric_backend.serve, daemon=True,
                name="fabric-serve",
            )
            fabric_thread.start()
            print("[worker] fabric serve loop started", flush=True)
        if args.mode in ("first", "solo") and fabric_backend.forward_link is not None:
            # 2026-05-29 fabric Phase F — wire the gRPC→fabric bridge.
            # Forward will ship hidden_state via fabric + wait for
            # FEEDBACK at the feedback_link, then return the token in
            # the gRPC reply. Requires both forward + feedback links;
            # setup_fabric builds forward from the /join response, the
            # feedback link comes from the chain-head/tail resolution
            # done in setup_fabric (Phase F).
            servicer.fabric_first_worker_bridge = (
                fabric_backend.first_worker_round_trip)
            print("[worker] fabric: first-worker gRPC→fabric bridge wired",
                  flush=True)

    try:
        server.wait_for_termination()
    finally:
        stop_event.set()
        residency_event.set()
        if owner_lease_httpd is not None:
            owner_lease_httpd.shutdown()
            owner_lease_httpd.server_close()
        if fabric_backend is not None:
            fabric_backend.stop()
        daemon.close()
        if paid_store is not None:
            paid_store.close()


if __name__ == "__main__":
    main()
