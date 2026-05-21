"""Worker-side gRPC Ed25519 verification helpers.

Phase B of the worker hardening sprint (2026-05-20). The HTTP side of
the worker is already covered by ``nakshatra_auth.py`` (Phase F, 2026-
05-19); this module applies the same canonical-string contract to
incoming gRPC calls and adds a peer-key resolver that pulls from the
pillar's ``/peers`` projection.

Tier model
----------

The worker's gRPC surface is split into two tiers:

- **ANONYMOUS_GRPC_METHODS** — discoverable without auth. Peer
  discovery and capability negotiation legitimately need this. Today:
  ``/nakshatra.Nakshatra/Info``.
- **AUTHENTICATED** — everything else. The caller must present a
  ``Sthambha-Ed25519`` authorization metadata pair signed by a key
  that the pillar has registered (we resolve the keyid against a
  cached projection of ``GET /peers``).

Streaming auth model
--------------------

For unary RPCs the body is one protobuf message; the signature covers
``sha256(serialized_message)``.

For streaming RPCs (``Inference``) the client signs the **first
frame**. Once the first frame's signature is accepted, subsequent
frames in the same stream are implicitly authorized — gRPC stream
identity = channel identity, so an attacker can't inject frames into
someone else's stream. The first-frame check binds the channel to a
keyid for the lifetime of the stream.

Wire format
-----------

Matches ``nakshatra_auth.AUTH_SCHEME`` byte-for-byte::

    authorization: Sthambha-Ed25519 keyid="<node_id>",sig="<b64>",ts="<unix>"

Canonical string::

    method "\\n" path "\\n" sha256_hex(first_message_bytes) "\\n" timestamp

Where ``method`` is ``"POST"`` for unary RPCs (matches the HTTP wire
shape — gRPC over HTTP/2 is always POST under the hood) and
``"STREAM"`` for streaming RPCs (different semantic body — only the
first frame is signed, not the cumulative stream).

ADR 0006 governs the wire format; this module's contract is a strict
extension of that ADR (new ``method`` values added; rest unchanged).
"""
from __future__ import annotations

import json
import re
import ssl
import sys
import threading
import time
from typing import Callable, Optional
from urllib import request as urlrequest, error as urlerror

import nakshatra_auth as auth


# Tier definitions
ANONYMOUS_GRPC_METHODS = frozenset([
    "/nakshatra.Nakshatra/Info",
])

# Default timestamp window — same 60s as the HTTP side (nakshatra_auth)
DEFAULT_TIMESTAMP_WINDOW_S = auth.DEFAULT_TIMESTAMP_WINDOW_S

# Canonical-string method tokens. Distinct from the HTTP "POST" / "GET"
# pattern so a captured HTTP signature can't be replayed against a gRPC
# endpoint (different method = different canonical string).
GRPC_METHOD_UNARY = "POST"        # matches HTTP/2 reality of unary RPCs
GRPC_METHOD_STREAM = "STREAM"     # first-frame-only signing model


# Cache shape: refresh interval, age cap (after which the resolver
# refuses to authorise anything — protects against operating off a
# stale cache when the pillar is unreachable for an extended period).
DEFAULT_REFRESH_INTERVAL_S = 60.0
DEFAULT_STALE_CACHE_DEADLINE_S = 300.0  # 5 minutes


# Parse the Sthambha-Ed25519 authorization header into (keyid, sig_b64, ts_int)
_AUTH_RE = re.compile(
    r'\s*Sthambha-Ed25519\s+keyid="([^"]*)",sig="([^"]*)",ts="([^"]*)"\s*'
)


class AuthError(Exception):
    """Raised when verification fails. Callers map to gRPC's
    ``UNAUTHENTICATED`` status code."""


def parse_auth_header(header: str) -> tuple[str, str, int]:
    """Parse a ``Sthambha-Ed25519`` authorization value into its parts.

    Returns ``(keyid, sig_b64, ts_unix)``. Raises ``AuthError`` on any
    malformed input — the caller is expected to abort the call.
    """
    if header is None:
        raise AuthError("missing authorization header")
    m = _AUTH_RE.fullmatch(header)
    if not m:
        raise AuthError("malformed authorization header")
    keyid, sig_b64, ts_str = m.groups()
    if not keyid:
        raise AuthError("empty keyid")
    if not sig_b64:
        raise AuthError("empty signature")
    try:
        ts = int(ts_str)
    except (TypeError, ValueError):
        raise AuthError("timestamp not an integer")
    return keyid, sig_b64, ts


def verify_grpc_call(
    method_path: str,
    auth_header: str,
    first_message_bytes: bytes,
    pubkey_resolver: Callable[[str], Optional[str]],
    *,
    now_seconds: Optional[int] = None,
    window_s: float = DEFAULT_TIMESTAMP_WINDOW_S,
    is_streaming: bool = False,
) -> str:
    """Verify the auth header against the call's first-message bytes.

    Returns the verified ``keyid`` on success; raises ``AuthError`` on
    any failure. ``pubkey_resolver(keyid)`` should return the
    public-key hex (64 chars) for the keyid, or ``None`` if the keyid
    is not registered.
    """
    keyid, sig_b64, ts = parse_auth_header(auth_header)
    current = now_seconds if now_seconds is not None else int(time.time())
    skew = abs(current - ts)
    if skew > window_s:
        raise AuthError(
            f"timestamp out of window (skew={skew}s, window={window_s}s)"
        )
    pub_hex = pubkey_resolver(keyid)
    if not pub_hex:
        raise AuthError(f"unknown keyid: {keyid!r}")
    method = GRPC_METHOD_STREAM if is_streaming else GRPC_METHOD_UNARY
    if not auth.verify_request(
        pub_hex, method, method_path, first_message_bytes, ts, sig_b64
    ):
        raise AuthError("signature mismatch")
    return keyid


def build_grpc_auth_header(
    priv_bytes: bytes,
    node_id: str,
    method_path: str,
    first_message_bytes: bytes,
    *,
    is_streaming: bool = False,
    timestamp_unix: Optional[int] = None,
) -> str:
    """Client-side helper. Build the ``authorization`` metadata value
    for a gRPC call by signing the canonical string with ``priv_bytes``.

    Symmetric with :func:`verify_grpc_call`; an envelope built here
    verifies there using the same ``method_path`` and message bytes.
    """
    method = GRPC_METHOD_STREAM if is_streaming else GRPC_METHOD_UNARY
    header, _ts = auth.build_signed_envelope(
        priv_bytes, node_id, method, method_path, first_message_bytes,
        timestamp_unix=timestamp_unix,
    )
    return header


# ── Pillar peer-key resolver ─────────────────────────────────────────


class PillarPeerKeyResolver:
    """Caches ``node_id → public_key_hex`` from the pillar's ``GET /peers``
    projection. Background thread refreshes every ``refresh_interval_s``.

    Also exposes :meth:`is_registered_address` for the Phase B5 SSRF
    defense — workers only push to peers the pillar has registered.

    Thread-safe. Each refresh is best-effort; consecutive failures log
    to stderr but don't crash the worker. A resolver whose cache is
    older than ``stale_cache_deadline_s`` returns ``None`` for every
    lookup — better to refuse calls than to authenticate against a
    cache that may have missed key rotations or evictions.
    """

    def __init__(
        self,
        pillar_url: str,
        *,
        refresh_interval_s: float = DEFAULT_REFRESH_INTERVAL_S,
        stale_cache_deadline_s: float = DEFAULT_STALE_CACHE_DEADLINE_S,
        priv_key: Optional[bytes] = None,
        own_node_id: Optional[str] = None,
        spki_hash: Optional[str] = None,
    ):
        self._pillar_url = (pillar_url or "").rstrip("/")
        self._refresh_interval = refresh_interval_s
        self._stale_deadline = stale_cache_deadline_s
        self._priv_key = priv_key
        self._own_node_id = own_node_id
        self._spki_hash = spki_hash
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        self._addresses: set[str] = set()
        # 2026-05-21 SPKI Phase 3.1: node_id → peer_spki_hash projection
        # from /peers. Empty value = peer has not declared a hash
        # (legacy / pre-Phase-2 worker). Phase 3 pinning treats empty
        # the same as unknown — refuse outbound when policy says so.
        self._spki_cache: dict[str, str] = {}
        # 2026-05-21 SPKI Phase 3.2: address → node_id reverse index so
        # expected_spki(address) can lookup the hash by the worker's
        # configured push target. Built from the same /peers payload as
        # _spki_cache; entries with empty address or empty node_id are
        # dropped (we don't index unidentifiable peers).
        self._addr_to_node: dict[str, str] = {}
        self._last_refresh: float = 0.0
        self._refresh_failures: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Background loop control ──────────────────────────────────────

    def start_background_refresh(self) -> None:
        if self._thread is not None:
            return
        try:
            self.refresh_once()
        except Exception as e:
            sys.stderr.write(
                f"[grpc-auth] initial pillar /peers refresh failed: {e}\n"
            )
        self._thread = threading.Thread(
            target=self._refresh_loop, daemon=True,
            name="grpc-auth-resolver",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(timeout=self._refresh_interval):
                break
            try:
                self.refresh_once()
            except Exception as e:
                sys.stderr.write(
                    f"[grpc-auth] pillar /peers refresh failed: {e}\n"
                )

    # ── Refresh ──────────────────────────────────────────────────────

    def refresh_once(self) -> None:
        """Synchronous single-shot refresh. Public for testing + force
        refresh from external callers."""
        url = f"{self._pillar_url}/peers"
        headers: dict[str, str] = {}
        if self._priv_key and self._own_node_id:
            header_val, _ts = auth.build_signed_envelope(
                self._priv_key, self._own_node_id, "GET", "/peers", b"",
            )
            headers["Authorization"] = header_val
        context: Optional[ssl.SSLContext] = None
        if url.startswith("https://"):
            context = auth.build_pillar_ssl_context(self._spki_hash)
        req = urlrequest.Request(url, headers=headers, method="GET")
        try:
            with urlrequest.urlopen(req, timeout=10, context=context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urlerror.URLError, OSError, TimeoutError, ValueError) as e:
            with self._lock:
                self._refresh_failures += 1
            raise

        new_cache: dict[str, str] = {}
        new_addrs: set[str] = set()
        new_spki: dict[str, str] = {}
        new_addr_to_node: dict[str, str] = {}
        for peer in (data or {}).get("peers", []):
            if not isinstance(peer, dict):
                continue
            node_id = str(peer.get("node_id") or "").strip()
            pub_hex = str(peer.get("public_key_hex") or "").strip()
            address = str(peer.get("address") or "").strip()
            # 2026-05-21 SPKI Phase 3.1: extract peer_spki_hash with
            # the same shape pillar's parse layer enforces (64 lowercase
            # hex). A pillar that drifted from the contract (or a
            # malicious /peers projection) could ship malformed values;
            # we apply the same defensive parse here.
            spki_in = str(peer.get("peer_spki_hash") or "").strip().lower()
            if spki_in and len(spki_in) == 64:
                try:
                    bytes.fromhex(spki_in)
                except ValueError:
                    spki_in = ""
            else:
                spki_in = ""
            # SSRF allowlist is independent of pubkey validity. A peer
            # with a malformed pubkey is still a registered peer from
            # the pillar's view — pushes to it stay allowed. Only auth
            # of incoming calls FROM them would fail (resolve() returns
            # None below).
            if address:
                new_addrs.add(address)
            if node_id and pub_hex and len(pub_hex) == 64:
                try:
                    bytes.fromhex(pub_hex)
                except ValueError:
                    continue
                new_cache[node_id] = pub_hex.lower()
            # SPKI cache + reverse address index: populated for any
            # peer with a node_id, regardless of pubkey validity.
            # Address-only or node_id-only entries don't enter the
            # reverse index — we'd have nothing to return if the
            # caller looked them up.
            if node_id:
                new_spki[node_id] = spki_in
                if address:
                    new_addr_to_node[address] = node_id
        with self._lock:
            self._cache = new_cache
            self._addresses = new_addrs
            self._spki_cache = new_spki
            self._addr_to_node = new_addr_to_node
            self._last_refresh = time.time()
            self._refresh_failures = 0

    # ── Queries ──────────────────────────────────────────────────────

    def resolve(self, node_id: str) -> Optional[str]:
        """Resolve a ``node_id`` to its registered ``public_key_hex``.
        Returns ``None`` if unknown, or if the cache is older than the
        stale deadline (refuses to authenticate against stale state)."""
        with self._lock:
            if self._is_stale_locked():
                return None
            return self._cache.get(node_id)

    def is_registered_address(self, address: str) -> bool:
        """Phase B5 SSRF defense. ``True`` iff ``address`` matches a
        registered peer's ``address`` field. Returns ``False`` on a
        stale cache — refusing to push beats pushing to attacker-
        supplied unverified endpoints."""
        with self._lock:
            if self._is_stale_locked():
                return False
            return address in self._addresses

    def expected_spki(self, address: str) -> Optional[str]:
        """2026-05-21 SPKI Phase 3.2 — return the SPKI hash the pillar
        has registered for the peer at ``address``, or ``None`` if:

        - the cache is stale (refuse-beats-stale: stale rosters may
          have missed a cert rotation; treating them as authoritative
          could pin against a hash the peer no longer serves)
        - the address is not in the pillar roster at all
        - the peer is in the roster but declared no SPKI (legacy /
          pre-Phase-2 worker; the caller's NAKSHATRA_REFUSE_UNPINNED_PEERS
          policy decides whether to refuse or fall through to plaintext)

        These three None cases are deliberately conflated at this
        level — every caller refuses, just under different policy
        names. Callers that need the distinction can read
        ``cache_age_seconds()`` and ``is_registered_address()``
        separately.
        """
        with self._lock:
            if self._is_stale_locked():
                return None
            node_id = self._addr_to_node.get(address)
            if not node_id:
                return None
            hex_hash = self._spki_cache.get(node_id, "")
            return hex_hash or None

    def expected_spki_for_node_id(self, node_id: str) -> Optional[str]:
        """Lookup variant indexed by node_id instead of address. Same
        None semantics as :meth:`expected_spki`. Convenient when the
        caller has the node_id directly (e.g. from an authenticated
        inbound stream context) rather than the push address."""
        with self._lock:
            if self._is_stale_locked():
                return None
            hex_hash = self._spki_cache.get(node_id, "")
            return hex_hash or None

    def known_node_ids(self) -> set[str]:
        with self._lock:
            return set(self._cache.keys())

    def known_addresses(self) -> set[str]:
        with self._lock:
            return set(self._addresses)

    def cache_age_seconds(self) -> float:
        """Seconds since last successful refresh; ``inf`` if never."""
        with self._lock:
            if self._last_refresh <= 0:
                return float("inf")
            return time.time() - self._last_refresh

    def stats(self) -> dict:
        with self._lock:
            return {
                "cached_node_ids": len(self._cache),
                "cached_addresses": len(self._addresses),
                # 2026-05-21 SPKI Phase 3.1: surfaces in /healthz for
                # operator observability — "how many peers in the
                # roster have declared a SPKI hash" is the rollout
                # signal during Phase 2 → Phase 3 migration.
                "cached_spki_pins": sum(
                    1 for v in self._spki_cache.values() if v),
                "cached_spki_total": len(self._spki_cache),
                "last_refresh_unix": self._last_refresh,
                "cache_age_seconds": (
                    time.time() - self._last_refresh
                    if self._last_refresh > 0 else None
                ),
                "consecutive_failures": self._refresh_failures,
                "is_stale": self._is_stale_locked(),
            }

    def _is_stale_locked(self) -> bool:
        if self._last_refresh <= 0:
            return True
        return (time.time() - self._last_refresh) > self._stale_deadline


# ── AUTH_REQUIRED env semantics (NAKSHATRA_AUTH_REQUIRED) ─────────────

# Default policy: if a pillar URL is configured, default to required.
# If no pillar URL is configured (legacy Mode A bringup with a static
# cluster YAML), default to not-required — the worker has no resolver
# anyway, so requiring auth would refuse all calls.
#
# Operators can force either way with the env var.

def resolve_auth_required(env_value: Optional[str], pillar_url: str) -> bool:
    """Decide whether the worker should require auth on its gRPC
    inbound surface, given the env value and the configured pillar URL.

    Truth table:

    ============ =========== ============================
    env_value    pillar_url  resolved
    ============ =========== ============================
    "true"       any         True
    "false"      any         False
    None/""      ""          False (Mode A legacy)
    None/""      set         True (Mode B/C default)
    ============ =========== ============================
    """
    val = (env_value or "").strip().lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    # Unset / blank: default depends on whether a pillar is configured.
    return bool((pillar_url or "").strip())
