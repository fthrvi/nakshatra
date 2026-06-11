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

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc

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
    required = (
        _TLS_AVAILABLE
        and _wtls.resolve_tls_required(
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
        _audit("tls_disabled_with_pillar", pillar_url=args.pillar_url)
    elif not _TLS_AVAILABLE:
        print(f"[worker] WARN: nakshatra_tls import failed "
              f"({_TLS_IMPORT_ERR}); gRPC server will be plaintext",
              flush=True)
    return TlsBoot(False, None, None, "")


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


def setup_fabric(args, forward_fn, n_embd, *, worker_priv, node_id):
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

WORKER_GRPC_MAX_MESSAGE_BYTES = 16 * 1024 * 1024     # A1: explicit cap
INFERENCE_STREAM_IDLE_TIMEOUT_S = 60.0                # A2: bound slow clients
MAX_PEER_STREAMS = 64                                 # A3: LRU cap
SPKI_HASH_LENGTH = 64                                 # A4: sha256 hex
MAX_CONCURRENT_SLICES = 1                             # A8: cap subprocess fan-out
SLICE_SUBPROCESS_TIMEOUT_S = 1800                     # A8: was 3600


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
    """Phase A7 — STHAMBHA_REFUSE_UNVERIFIED_FETCH gate.

    Default true (refuse when pillar omits model_sha256). A malicious
    pillar can serve poisoned weights by omitting the hash. Mode A/B
    operators can opt out with STHAMBHA_REFUSE_UNVERIFIED_FETCH=false.
    """
    val = (refuse_env if refuse_env is not None else "true").strip().lower()
    refuse = val in ("true", "1", "yes")
    return refuse and not (expected_sha or "").strip()


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
        # Stderr buffer (Phase 3.6): keeps last N lines so we can verify what
        # the daemon ACTUALLY did (e.g., GPU offload count) vs. what we asked.
        self.stderr_lines = collections.deque(maxlen=500)
        # Phase H: rolling per-RPC timing. Each Forward() records its
        # daemon-call duration here; main thread averages the last N for
        # the heartbeat payload so the pillar can latency-rank peers.
        self.recent_rpc_ms = collections.deque(maxlen=20)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # Wait for the "[daemon] ready" line so info+forward are valid.
        self._wait_ready()

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
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# v0.5 §9.7: idempotency cache is sized in MB, internally converted to an
# entry cap using this per-entry estimate. ~10 KB matches a typical hidden-state
# response (3072-dim × f32 + proto overhead). Adjust if the average grows.
IDEM_BYTES_PER_ENTRY = 10 * 1024


class WorkerServicer(pb_grpc.NakshatraServicer):
    def __init__(self, daemon: DaemonClient, mode: str, layer_start: int, layer_end: int, model_id: str,
                 idem_max_entries: int = 6400, idem_ttl_seconds: float = 60.0,
                 peer_resolver=None, auth_required: bool = False,
                 refuse_unregistered_peers: bool = True,
                 refuse_unpinned_peers: bool = True):
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
        # v0.5 M0.5.2: idempotency cache. Maps session_id -> {"steps": {step_id: response_proto},
        # "last_touch": float}. On a duplicate (session_id, step_id) we return the cached
        # response without touching the daemon. TTL-expires per-session; hard-capped by
        # total entries across all sessions.
        self._idem_lock = threading.Lock()
        self._idem_cache: dict = {}
        self._idem_ttl = idem_ttl_seconds
        self._idem_max = idem_max_entries
        self._idem_hits = 0
        self._idem_misses = 0
        # v0.5 M0.5.3: peer-stream cache for rpc_push. One persistent bidi
        # Inference stream per next-hop peer address, reused across sessions.
        # Map: address (str) -> (grpc.Channel, queue.Queue request side, response iterator)
        # Phase A3 (2026-05-20): OrderedDict + LRU cap. An attacker who
        # repeatedly pushes to fresh addresses can no longer balloon
        # this dict; oldest stream is evicted on overflow.
        self._peer_lock = threading.Lock()
        self._peer_streams: "collections.OrderedDict" = collections.OrderedDict()
        self._push_count = 0
        self._push_errors = 0
        self._peer_evictions = 0

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

    def _idem_get(self, session_id: str, step_id: str):
        with self._idem_lock:
            sess = self._idem_cache.get(session_id)
            if sess and step_id in sess["steps"]:
                self._idem_hits += 1
                sess["last_touch"] = time.time()
                return sess["steps"][step_id]
            self._idem_misses += 1
            return None

    def _idem_put(self, session_id: str, step_id: str, response):
        with self._idem_lock:
            now = time.time()
            sess = self._idem_cache.setdefault(session_id, {"steps": {}, "last_touch": now})
            sess["steps"][step_id] = response
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
            return grpc.insecure_channel(address)
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

    def _get_peer_stream(self, address: str):
        """Open or reuse a bidi Inference stream to a peer worker.

        v0.5 M0.5.3 — server-to-server push. Returns (req_queue, response_iterator).
        Sending None to the queue closes the stream.

        Phase A3 (2026-05-20): bounded by MAX_PEER_STREAMS via LRU
        eviction. Re-touching an existing entry refreshes it (move to
        end); opening a fresh stream may evict the oldest. Eviction
        closes the request side (None sentinel) and the channel.
        """
        with self._peer_lock:
            cached = self._peer_streams.get(address)
            if cached is not None:
                self._peer_streams.move_to_end(address)
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
            self._peer_streams[address] = (channel, req_q, resp_iter)
            while len(self._peer_streams) > MAX_PEER_STREAMS:
                old_addr, (old_ch, old_q, _old_iter) = (
                    self._peer_streams.popitem(last=False)
                )
                self._peer_evictions += 1
                try: old_q.put(None)
                except Exception: pass
                try: old_ch.close()
                except Exception: pass
                sys.stderr.write(
                    f"[peer-streams] evicted oldest {old_addr!r} "
                    f"(cap={MAX_PEER_STREAMS})\n"
                )
            return req_q, resp_iter

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
            "authz_rejections": self._authz_rejections,
            "ssrf_rejections": self._ssrf_rejections,
            "refuse_unpinned_peers": bool(self.refuse_unpinned_peers),
            "spki_unpinned_refusals": self._spki_unpinned_refusals,
            "spki_mismatch_refusals": self._spki_mismatch_refusals,
            "spki_probe_failures": self._spki_probe_failures,
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
            return keyid
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

    def Info(self, request, context):
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
            has_token_embd=(self.mode == "first"),
            has_lm_head=(self.mode == "last"),
            # v0.5 §9.1 closure — advertised features so the client can
            # negotiate push at session start without runtime probing.
            # v1.0 §7 — append control/vN tokens so the client can negotiate
            # the control-protocol version on the same handshake.
            protocol_capabilities=[
                "streaming",
                "rpc_push",
                "idempotency_cache",
                "recovery_replay",
            ] + _control_version_caps(),
        )

    def _run_forward(self, hidden_in: bytes, n_tokens: int,
                     has_token_ids: bool, keep_kv: bool,
                     start_pos: int) -> "ForwardResult":
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
        flags = 0x1 if keep_kv else 0x0
        if has_token_ids:
            if len(hidden_in) != n_tokens * 4:
                return ForwardResult(
                    False, b"", "hidden_in size mismatch for token_ids mode",
                    client_error=True)
            cmd = CMD_TOKEN_DECODE
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
        # Strip the 4-byte rtype prefix; payload is hidden_state OR
        # int32 token id. Caller knows this worker's mode via Info.
        return ForwardResult(True, resp[4:], "", client_error=False)

    def Forward(self, request, context):
        # Phase B (2026-05-20): authenticate the unary call. The signed
        # body is the serialized request — same canonical-string shape
        # as the HTTP side, just with method=POST + path=gRPC method name.
        self._check_grpc_auth(
            context, request.SerializeToString(),
            method_path="/nakshatra.Nakshatra/Forward",
            is_streaming=False,
        )
        result = self._run_forward(
            request.hidden_in, request.n_tokens, request.has_token_ids,
            request.keep_kv, int(request.start_pos),
        )
        if not result.ok:
            context.set_code(
                grpc.StatusCode.INVALID_ARGUMENT if result.client_error
                else grpc.StatusCode.INTERNAL)
            context.set_details(result.error)
            return pb.ForwardResponse()
        # 2026-05-29 fabric Phase F — first-worker gRPC→fabric bridge.
        # When the bridge is wired (mode=first + --transport=fabric),
        # the gRPC reply carries the chain's FINAL token (after the
        # fabric round-trip through mid + last workers) instead of
        # this worker's intermediate hidden_state. Mid/last workers
        # never have the bridge wired and keep returning hidden_state
        # / token directly per the legacy path.
        if self.fabric_first_worker_bridge is not None:
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
        try:
            for step in _iter_with_idle_timeout(
                request_iterator, INFERENCE_STREAM_IDLE_TIMEOUT_S
            ):
                # Phase B (2026-05-20): authenticate the first frame.
                # Once the first frame's signature is accepted, gRPC
                # stream identity binds subsequent frames to the same
                # authenticated peer — no per-frame check needed.
                if first_step:
                    self._check_grpc_auth(
                        context, step.SerializeToString(),
                        method_path="/nakshatra.Nakshatra/Inference",
                        is_streaming=True,
                    )
                # v0.5 M0.5.2: idempotency cache. If this (session_id, step_id) has
                # been served before, return the cached response without touching
                # the daemon. Note: a cache hit advances first_step too — the
                # daemon's KV state was already updated by the original call, so
                # subsequent steps in this stream should treat themselves as
                # "not first" with respect to keep_kv semantics.
                cached = self._idem_get(step.session_id, step.step_id)
                if cached is not None:
                    first_step = False
                    yield cached
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
                    expected = n_tokens * self.n_embd * 4
                    if len(hs.raw) != expected:
                        yield pb.InferenceStep(
                            session_id=step.session_id, step_id=step.step_id,
                            error=f"hidden_state size mismatch: got {len(hs.raw)}, expected {expected}".encode(),
                        )
                        return
                    input_bytes = hs.raw
                    cmd = CMD_EMBD_DECODE
                else:
                    yield pb.InferenceStep(
                        session_id=step.session_id, step_id=step.step_id,
                        error=b"InferenceStep payload must be token_ids or hidden_state",
                    )
                    return

                flags = 0x0 if first_step else 0x1
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
                    yield pb.InferenceStep(
                        session_id=step.session_id, step_id=step.step_id,
                        error=f"daemon decode failed status={status} resp_len={len(resp)}".encode(),
                    )
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
                if (self.mode == "first"
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
                    self._idem_put(step.session_id, step.step_id, out)
                    yield out
                    continue

                out = pb.InferenceStep(
                    session_id=step.session_id,
                    step_id=step.step_id,
                    prefix_length=step.prefix_length + n_tokens,
                )
                if self.mode == "last":
                    token_id = struct.unpack("<i", payload[:4])[0]
                    out.token_ids.ids.append(token_id)
                else:
                    out.hidden_state.raw = payload
                    out.hidden_state.batch = 1
                    out.hidden_state.n_tokens = n_tokens

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
                if next_addr and self.mode != "last":
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
                        req_q, resp_iter = self._get_peer_stream(next_addr)
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
                            cached = self._peer_streams.pop(next_addr, None)
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
                    self._idem_put(step.session_id, step.step_id, peer_resp)
                    yield peer_resp
                    continue

                # v0.5 M0.5.2: store before yielding so a concurrent duplicate
                # gets the cached response. Only cache fully-formed successful
                # responses; error responses are not cached (the error was the
                # whole point of failure, replays should retry).
                self._idem_put(step.session_id, step.step_id, out)
                yield out
        except TimeoutError as e:
            sys.stderr.write(f"[inference] idle timeout: {e}\n")
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(str(e))
            return
        except Exception as e:
            sys.stderr.write(f"[inference] stream aborted: {e}\n")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Inference stream error: {e}")
            return


# Phase I8: persistent nonce slot for soft-attestation. Pillar issues a
# fresh nonce in each /peer response; we stash it here so the next
# heartbeat includes the matching signed attestation. Empty string on
# first contact — pillar accepts and returns a starter nonce.
_attestation_nonce: str = ""


def register_with_pillar(pillar_url: str, payload: dict, log_prefix: str = "[worker]",
                          priv_key: Optional[bytes] = None,
                          node_id: Optional[str] = None,
                          spki_hash: Optional[str] = None):
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
        context = None
        if full_url.startswith("https://"):
            if not _AUTH_AVAILABLE:
                print(f"{log_prefix} HTTPS pillar URL but auth module missing; "
                      f"refusing connection", flush=True)
                return False
            context = _wauth.build_pillar_ssl_context(spki_hash)
        with urlrequest.urlopen(req, timeout=5, context=context) as resp:
            # SPKI pinning happens *after* the handshake. If the pillar
            # served a cert whose SPKI doesn't match expected_hash, the
            # connection still succeeded at TCP/TLS but we refuse before
            # consuming the response.
            if (full_url.startswith("https://") and spki_hash
                    and _AUTH_AVAILABLE):
                sock = resp.fp.raw._sock if hasattr(resp.fp, "raw") else None
                if sock is not None and hasattr(sock, "getpeercert"):
                    der = sock.getpeercert(binary_form=True)
                    try:
                        _wauth.verify_pillar_cert_spki(der, spki_hash)
                    except _wauth.PillarSpkiMismatch as e:
                        print(f"{log_prefix} REFUSED: {e}", flush=True)
                        return False
            body = resp.read().decode()
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
                               url=pillar_url, log_prefix=log_prefix)
            except json.JSONDecodeError:
                pass
            print(f"{log_prefix} registered with pillar: {body}", flush=True)
            _audit("register_success", url=pillar_url, log_prefix=log_prefix)
            return True
    except (urlerror.URLError, OSError, TimeoutError) as e:
        print(f"{log_prefix} pillar registration failed ({pillar_url}): {e}", flush=True)
        _audit("register_failed", url=pillar_url, error=str(e),
               log_prefix=log_prefix)
        return False


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
                   spki_hash: Optional[str] = None):
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
        if stop_event.wait(timeout=next_interval):
            break
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
        daemon_alive = daemon is not None and daemon.proc.poll() is None
        started_at = _HEALTH_STATE.get("started_at", time.time())
        return {
            "status": "ok" if daemon_alive else "down",
            "mode": _HEALTH_STATE.get("mode", ""),
            "layer_start": _HEALTH_STATE.get("layer_start", -1),
            "layer_end": _HEALTH_STATE.get("layer_end", -1),
            "uptime_seconds": round(time.time() - started_at, 1),
            "protocol_version": "0.1.0",
        }

    def _health_payload(self):
        daemon = _HEALTH_STATE.get("daemon")
        daemon_alive = daemon is not None and daemon.proc.poll() is None
        started_at = _HEALTH_STATE.get("started_at", time.time())
        samples = list(daemon.recent_rpc_ms) if daemon is not None else []
        recent_avg = (sum(samples) / len(samples)) if samples else None
        gpu = daemon.gpu_offload_status() if daemon is not None else None
        return {
            "status": "ok" if daemon_alive else "down",
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
                           trusted_pubkeys: Optional[set] = None) -> str:
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
    _audit("package_provision_started", package_url=package_url,
           layer_start=layer_start, layer_end=layer_end, dest_path=dest_path)
    from packaging.fetch_package import fetch_and_assemble
    fetch_and_assemble(package_url, layer_start, layer_end, dest_path,
                       require_signature=require_signature,
                       trusted_pubkeys=trusted_pubkeys)
    _audit("package_provision_completed", dest=dest_path)
    return dest_path


def fetch_sub_gguf_from_peer(pillar_url: str, model_id: str,
                              layer_start: int, layer_end: int,
                              dest_path: str,
                              own_node_id: str = "") -> str:
    """Query Sthambha for a peer with the requested file and download it.

    Returns the local path on success; raises RuntimeError on failure.
    Verifies SHA-256 against the pillar's recorded hash if present.

    Skips candidates whose node_id matches `own_node_id` (don't fetch
    from yourself). Retries the next candidate on connection failure
    so a stale-but-still-listed peer doesn't block the bootstrap.
    """
    _audit("fetch_started", model_id=model_id,
           layer_start=layer_start, layer_end=layer_end,
           dest_path=dest_path)
    # 1. Ask the pillar for the file index
    files_url = f"{pillar_url.rstrip('/')}/files?model={model_id}"
    try:
        with urlrequest.urlopen(files_url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(f"could not query pillar at {files_url}: {e}")

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

    # Phase A7 (2026-05-20): refuse unverified fetch when the pillar
    # omits model_sha256. A malicious pillar can serve poisoned weights
    # by simply not declaring a hash. Default behaviour is to refuse;
    # Mode A/B operators can opt out with the env var.
    refuse_env = os.environ.get("STHAMBHA_REFUSE_UNVERIFIED_FETCH")
    verified_candidates = [
        c for c in candidates
        if not should_refuse_unverified_fetch(refuse_env, c.get("model_sha256", ""))
    ]
    if not verified_candidates:
        raise RuntimeError(
            f"refusing to fetch {model_id} layers [{layer_start},{layer_end})"
            f": no candidate peer has a pillar-attested model_sha256"
            f" (set STHAMBHA_REFUSE_UNVERIFIED_FETCH=false to allow"
            f" unverified Mode-A/B fetch)"
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
        expected_sha = chosen.get("model_sha256", "")
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

        # Verify SHA-256 if the pillar gave us one
        actual_sha = h.hexdigest()
        if expected_sha and actual_sha != expected_sha:
            print(f"[fetch] {chosen.get('node_id')} sha mismatch (got {actual_sha[:12]}, expected {expected_sha[:12]}); trying next", flush=True)
            try: os.unlink(tmp_path)
            except Exception: pass
            last_error = "sha mismatch"
            continue

        # Atomic move into place
        os.rename(tmp_path, dest_path)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5500)
    ap.add_argument("--sub-gguf", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["first", "middle", "last"], required=True)
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
    ap.add_argument("--pillar-url", type=str, default="",
                    help="Sthambha pillar URL (e.g. http://node-pi:7777). If set, "
                         "worker registers self + sends heartbeat every 30s.")
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
    ap.add_argument("--skip-sha256", action="store_true",
                    help="Skip SHA-256 of sub-GGUF (for fast restarts; cached_files lacks hash).")
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
    ap.add_argument("--package-url", type=str, default="",
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

    # Phase D5 (2026-05-20): initialise the audit log singleton early so
    # every event in main() (start, fetch, slice, register, auth fail)
    # has a place to land. Failure-soft: missing directory or permission
    # error leaves the worker running without forensic trail.
    if _AUDIT_AVAILABLE:
        try:
            _waudit.init_audit()
            _audit("worker_started",
                   model_id=args.model_id, mode=args.mode,
                   layer_start=args.layer_start, layer_end=args.layer_end,
                   port=args.port, pillar_url=args.pillar_url)
        except Exception as e:
            print(f"[worker] audit init failed (continuing without forensic log): {e}",
                  flush=True)

    # Phase 4: if sub-GGUF is missing AND we have a pillar to ask, fetch it
    # before doing anything else. This is what lets a fresh machine bootstrap
    # without manual scp.
    if not os.path.exists(args.sub_gguf):
        if args.package_url:
            print(f"[worker] sub-gguf missing at {args.sub_gguf}; provisioning layers "
                  f"[{args.layer_start},{args.layer_end}) from package {args.package_url}",
                  flush=True)
            trusted = (set(args.package_trusted_pubkey)
                       if args.package_trusted_pubkey else None)
            try:
                provision_from_package(
                    args.package_url, args.layer_start, args.layer_end, args.sub_gguf,
                    require_signature=args.package_require_signature,
                    trusted_pubkeys=trusted,
                )
            except Exception as e:
                sys.exit(f"[worker] package provisioning failed: {e}")
        elif args.auto_fetch and args.pillar_url:
            print(f"[worker] sub-gguf missing at {args.sub_gguf}; auto-fetching from peer", flush=True)
            own_node_id = args.node_id or f"{socket.gethostname()}-{args.port}"
            try:
                fetch_sub_gguf_from_peer(
                    args.pillar_url, args.model_id,
                    args.layer_start, args.layer_end,
                    args.sub_gguf,
                    own_node_id=own_node_id,
                )
            except Exception as e:
                sys.exit(f"[worker] auto-fetch failed: {e}")
        else:
            sys.exit(f"[worker] sub-gguf does not exist: {args.sub_gguf} "
                     f"(use --package-url to provision from a layer package, or "
                     f"--auto-fetch with --pillar-url to bootstrap from a peer)")

    print(f"[worker] spawning daemon: {args.daemon_bin} {args.sub_gguf} {args.mode} {args.n_ctx} threads={args.n_threads} gpu_layers={args.n_gpu_layers}", flush=True)
    daemon = DaemonClient(args.daemon_bin, args.sub_gguf, args.mode, args.n_ctx, args.n_threads, args.n_gpu_layers)

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
    auth_required = (
        _GRPC_AUTH_AVAILABLE
        and _wgrpcauth.resolve_auth_required(
            os.environ.get("NAKSHATRA_AUTH_REQUIRED"), args.pillar_url
        )
    )
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

    servicer = WorkerServicer(daemon, args.mode, args.layer_start, args.layer_end, args.model_id,
                              idem_max_entries=idem_max_entries,
                              idem_ttl_seconds=args.idempotency_cache_ttl,
                              auth_required=auth_required,
                              refuse_unregistered_peers=refuse_unregistered_peers,
                              refuse_unpinned_peers=refuse_unpinned_peers)
    print(f"[worker] idempotency cache: {args.idempotency_cache_mb} MB "
          f"(~{idem_max_entries} entries @ {IDEM_BYTES_PER_ENTRY//1024} KB), "
          f"ttl={args.idempotency_cache_ttl}s", flush=True)
    pb_grpc.add_NakshatraServicer_to_server(servicer, server)
    # 2026-05-21 SPKI Phase 2.4: bind TLS port when cert was prepared
    # above; otherwise legacy insecure path.
    if tls.required and tls.cert_path and tls.key_path:
        creds = _wtls.build_grpc_server_credentials(
            tls.cert_path, tls.key_path)
        server.add_secure_port(f"[::]:{args.port}", creds)
        listen_proto = "TLS"
    else:
        server.add_insecure_port(f"[::]:{args.port}")
        listen_proto = "plaintext"
    print(f"[worker] M5 listening on :{args.port} ({listen_proto})  "
          f"mode={args.mode}  layers=[{args.layer_start},{args.layer_end})  "
          f"model={args.model_id}", flush=True)
    server.start()

    # Phase 4: file server lets peers fetch this worker's sub-GGUF over HTTP
    # byte-range. New peers joining the cluster don't need their files
    # pre-shipped — they query the pillar's file index, find a peer that has
    # the file, and download it. Convention: file-server port = grpc port + 1000.
    file_server_port = args.file_server_port or (args.port + 1000)
    node_id = args.node_id or f"{socket.gethostname()}-{args.port}"
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
    if args.pillar_url:
        public_addr = args.public_address or f"{socket.gethostname()}:{args.port}"

        # Phase 3.5: compute SHA-256 of the sub-GGUF (one-time at startup)
        sub_gguf_sha256 = ""
        sub_gguf_size = 0
        try:
            sub_gguf_size = os.path.getsize(args.sub_gguf)
            if not args.skip_sha256:
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
        actual_backend = args.gpu_backend
        declared_gpu = (args.gpu_vram_gb > 0 and args.gpu_backend != "cpu")
        if declared_gpu and not offload_status["uses_gpu"]:
            print(f"[worker] WARNING: declared --gpu-backend={args.gpu_backend} "
                  f"but daemon offloaded {offload_status['n_offloaded']}/"
                  f"{offload_status['total_layers']} layers — daemon binary likely "
                  f"lacks {args.gpu_backend} support; downgrading registration to cpu",
                  flush=True)
            actual_backend = "cpu"
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
        if not args.no_cache_scan:
            cache_dir = args.cache_dir or str(Path(args.sub_gguf).resolve().parent)
            sha_seed = {str(Path(args.sub_gguf).resolve()): sub_gguf_sha256} if sub_gguf_sha256 else {}
            cached_files = scan_cache_dir(cache_dir, args.model_id, sha_cache=sha_seed)
            print(f"[worker] cache-scan found {len(cached_files)} sub-GGUF(s) in {cache_dir}", flush=True)

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

        # Phase F2: load or create the worker's persistent Ed25519
        # keypair; include public_key_hex in the FIRST registration so
        # the pillar locks it via TOFU. Subsequent heartbeats sign with
        # the same key but omit public_key_hex from the body.
        worker_priv: Optional[bytes] = None
        worker_pub_hex = ""
        if _AUTH_AVAILABLE:
            try:
                worker_priv, worker_pub_hex = _wauth.load_or_create_worker_key()
                register_payload["public_key_hex"] = worker_pub_hex
            except Exception as e:
                print(f"[worker] WARN: could not load worker key ({e}); "
                      f"requests will go unsigned. This will fail against "
                      f"any pillar with STHAMBHA_AUTH_REQUIRED=true.",
                      flush=True)
        else:
            print(f"[worker] WARN: nakshatra_auth import failed "
                  f"({_AUTH_IMPORT_ERR}); requests will go unsigned",
                  flush=True)
        # Phase F3: env-pinned pillar SPKI for HTTPS connections.
        # Phase A4 (2026-05-20): strict length + hex validation. Catches
        # the silent-disable-by-typo case where an operator sets the env
        # to "abc" and pinning silently turns off.
        try:
            spki_hash = validate_spki_hash_env(
                os.environ.get("STHAMBHA_PILLAR_SPKI_SHA256")
            )
        except ValueError as e:
            sys.exit(f"[worker] startup refused: {e}")
        if args.pillar_url.startswith("https://") and not spki_hash:
            print(f"[worker] WARN: HTTPS pillar URL but no SPKI hash pinned "
                  f"via STHAMBHA_PILLAR_SPKI_SHA256; cert identity NOT verified",
                  flush=True)
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
        register_with_pillar(args.pillar_url, register_payload,
                             priv_key=worker_priv, node_id=node_id,
                             spki_hash=spki_hash)

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
                    "spki_hash": spki_hash},
            daemon=True,
        )
        heartbeat_thread.start()
        print(f"[worker] heartbeat → {args.pillar_url} every 30s as {node_id} @ {public_addr}"
              + (f" (signed; pubkey={worker_pub_hex[:16]}…)" if worker_pub_hex else " (UNSIGNED)"),
              flush=True)

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
        )
        if fabric_backend.inbound_link is not None:
            fabric_thread = threading.Thread(
                target=fabric_backend.serve, daemon=True,
                name="fabric-serve",
            )
            fabric_thread.start()
            print("[worker] fabric serve loop started", flush=True)
        if args.mode == "first" and fabric_backend.forward_link is not None:
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
        if fabric_backend is not None:
            fabric_backend.stop()
        daemon.close()


if __name__ == "__main__":
    main()
