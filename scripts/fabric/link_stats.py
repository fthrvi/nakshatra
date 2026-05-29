"""Fabric link_stats reporter — periodic counter snapshots from each
worker to the pillar's ``POST /fabric/link_stats`` endpoint.

Phase E of the fabric_lite sprint. Builds on Phase B (where each
``FabricLink`` already maintains the schema §9 counter set with the
exact field names the pillar reads via
``Sthambha.record_link_snapshots``) and Phase C (auth envelope shape).

What this module does:

  - Maintains a registry of ``FabricLink``s indexed by the
    ``other_peer_id`` of the pair (one entry per directional link).
  - Builds a snapshot bundle every ``interval_s`` seconds and POSTs
    it to ``<pillar>/fabric/link_stats``, signed under OWNER tier by
    the worker's Ed25519 key (per ADR 0006 §7 + server.py:1629 —
    the signer MUST equal ``peer_id`` so an attacker can't inject
    bogus counters under another worker's name).
  - Failure-soft: pillar outages log + retry on the next tick.
  - Silently no-ops when no pillar URL is configured (legacy Mode A
    bringups + the localhost smoke).

What this module deliberately doesn't do:

  - RTT measurement. Schema §9 lists ``rtt_ns_p50`` / ``rtt_ns_p99``
    as feedback-wormhole-only metrics; tracking them requires the
    first worker to timestamp outgoing seqs and match against
    incoming FEEDBACK packets — wiring that lands in Phase F when
    the feedback link's address is actually resolved. The reporter
    emits the fields as zero for now (the pillar's snapshot
    dataclass treats 0 as "not applicable" per
    ``sthambha/core.py:755`` comment).
  - Counter resets. The per-link counters are cumulative across the
    worker's lifetime; pillar snapshots track raw values. A rekey
    that resets ``send_seq`` does NOT reset these counters
    (operationally, counters carry through rekey — same stance
    spelled out for the pillar-side counter retention question in
    schema §10 OQ3).
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nakshatra_auth as auth  # noqa: E402

try:
    import nakshatra_audit as _audit_mod
    _AUDIT_AVAILABLE = True
except ImportError:
    _audit_mod = None
    _AUDIT_AVAILABLE = False


# ── Schema §9 — counter names + pillar contract ─────────────────────


# Names matched byte-for-byte by ``sthambha/core.py:LinkStatsSnapshot``
# (lines 747–756). Adding a private counter on the FabricLink side is
# fine; this list controls what gets surfaced to the pillar, so private
# debug-only counters never leak into the cross-repo wire contract.
SCHEMA_COUNTER_NAMES: tuple[str, ...] = (
    "sent_packets",
    "sent_bytes",
    "recv_packets",
    "recv_bytes",
    "recv_auth_fails",
    "recv_gaps",
    "recv_dropped_alloc",
    "recv_dropped_dtype",
    "rtt_ns_p50",
    "rtt_ns_p99",
)

# Defensive upper bound on links in one push. Pillar enforces 256
# (sthambha/core.py:MAX_LINK_STATS_PER_PUSH); reporter caps at the
# same value so a misconfigured worker doesn't ship a payload that
# the pillar would 400 anyway.
MAX_LINKS_PER_PUSH = 256


DEFAULT_INTERVAL_S = 30.0
ENV_INTERVAL = "NAKSHATRA_FABRIC_LINK_STATS_INTERVAL_S"


def _interval_from_env(default: float = DEFAULT_INTERVAL_S) -> float:
    """Same typo-safe fallback shape as ``nakshatra_tls._probe_timeout
    _from_env`` — empty / non-float / non-positive values fall back
    to the default rather than disable reporting silently."""
    raw = os.environ.get(ENV_INTERVAL, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    if v <= 0:
        return default
    return v


# ── Snapshot construction (testable without sockets) ───────────────


def _snapshot_link_counters(link) -> dict[str, int]:
    """Project a FabricLink's counter dict into the
    pillar-contract-recognised subset. RTT fields default to 0 — the
    FabricLink doesn't measure them yet (Phase F)."""
    out: dict[str, int] = {}
    for name in SCHEMA_COUNTER_NAMES:
        out[name] = int(link.counters.get(name, 0))
    return out


def build_snapshot(
    *,
    peer_id: str,
    plan_id: str,
    links: dict[str, object],
) -> dict:
    """Build the POST /fabric/link_stats body for the given (peer_id,
    plan_id, links registry). Each registry entry is a FabricLink
    (or anything exposing a ``.counters`` dict).

    Truncates to ``MAX_LINKS_PER_PUSH`` entries by sorted other_peer_id
    so the same subset gets reported every tick — operationally far
    more useful than a random truncation when the count actually
    crosses 256 (chain plans don't, but defensive)."""
    entries = []
    for other_peer_id in sorted(links.keys())[:MAX_LINKS_PER_PUSH]:
        link = links[other_peer_id]
        entry = {"other_peer_id": other_peer_id}
        entry.update(_snapshot_link_counters(link))
        entries.append(entry)
    body: dict = {"peer_id": peer_id, "links": entries}
    if plan_id:
        body["plan_id"] = plan_id
    return body


# ── LinkStatsReporter ──────────────────────────────────────────────


class LinkStatsReporter:
    """Periodic POST of FabricLink counters to the pillar.

    Lifecycle:

        reporter = LinkStatsReporter(
            pillar_url="http://pillar:7777",
            peer_id="worker-a",
            priv_key=worker_priv_bytes,
            plan_id=join_response.plan_id,
        )
        reporter.register("worker-b", forward_link)
        reporter.register("worker-z", backward_link)
        reporter.start()
        # ... worker runs ...
        reporter.stop()

    Thread-safety: ``register``, ``unregister``, ``set_plan_id`` take
    the internal lock. The background thread snapshots under the
    lock, releases it, and runs the HTTP POST without holding it
    (the lock guards bookkeeping; HTTP latency mustn't block
    rekey/unregister callers).
    """

    def __init__(
        self,
        pillar_url: str,
        peer_id: str,
        priv_key: bytes,
        *,
        plan_id: str = "",
        interval_s: Optional[float] = None,
        request_timeout_s: float = 5.0,
        pillar_spki_hash: Optional[str] = None,
    ):
        # Reject a malformed Ed25519 key up front — same shape as the
        # JoinClient guard. Don't refuse an empty pillar URL: that's
        # the silent-no-op path the legacy bringup relies on, so the
        # constructor accepts it and ``start()`` becomes a no-op.
        if pillar_url and len(priv_key) != 32:
            raise ValueError(
                f"worker Ed25519 priv key must be 32 bytes; got {len(priv_key)}"
            )
        self._pillar_url = (pillar_url or "").rstrip("/")
        self._peer_id = peer_id
        self._priv_key = priv_key
        self._plan_id = plan_id
        self._interval_s = (
            interval_s if interval_s is not None
            else _interval_from_env()
        )
        self._request_timeout_s = request_timeout_s
        self._pillar_spki_hash = pillar_spki_hash

        self._lock = threading.Lock()
        self._links: dict[str, object] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Coarse outcome counters for operator debug. Not surfaced to
        # the pillar — these are the reporter's own health, not link
        # health.
        self.push_count: int = 0
        self.push_failures: int = 0

    # ── Registry ─────────────────────────────────────────────────

    def register(self, other_peer_id: str, link) -> None:
        """Attach a FabricLink for periodic reporting. Re-registering
        an existing other_peer_id replaces the entry (e.g. rekey
        opened a fresh link to the same peer)."""
        if not other_peer_id:
            raise ValueError("register() requires a non-empty other_peer_id")
        with self._lock:
            self._links[other_peer_id] = link

    def unregister(self, other_peer_id: str) -> None:
        """Drop a registered link. Idempotent — unregistering an
        unknown peer is a no-op so the boot/teardown path doesn't have
        to track which links were registered."""
        with self._lock:
            self._links.pop(other_peer_id, None)

    def set_plan_id(self, plan_id: str) -> None:
        """Update the plan_id sent in future snapshots. Use after a
        rekey when /join returns a new plan."""
        with self._lock:
            self._plan_id = plan_id or ""

    # ── Snapshot ─────────────────────────────────────────────────

    def snapshot(self) -> Optional[dict]:
        """Build the POST body that would be sent right now. Returns
        ``None`` when no links are registered — there's nothing
        useful to report and a zero-link push wastes pillar cycles."""
        with self._lock:
            if not self._links:
                return None
            return build_snapshot(
                peer_id=self._peer_id,
                plan_id=self._plan_id,
                links=dict(self._links),
            )

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background reporting thread. No-ops when:
          - No pillar URL configured (legacy Mode A / localhost smoke)
          - The thread is already running (idempotent)
        """
        if not self._pillar_url:
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="fabric-link-stats",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the reporting thread to exit at the next tick.
        Idempotent + non-blocking."""
        self._stop.set()

    def _loop(self) -> None:
        """Sleep interval_s, snapshot, POST, repeat. Never raises."""
        while not self._stop.is_set():
            if self._stop.wait(timeout=self._interval_s):
                return
            body = self.snapshot()
            if body is None:
                continue   # nothing registered this tick
            try:
                self._post(body)
                self.push_count += 1
            except Exception as e:
                # Failure-soft — pillar outage shouldn't crash the
                # worker. Log + count + carry on; next tick may
                # succeed.
                self.push_failures += 1
                sys.stderr.write(
                    f"[fabric/link_stats] push failed: {e}\n"
                )
                if _AUDIT_AVAILABLE:
                    try:
                        _audit_mod.audit(
                            "fabric_link_stats_push_failed",
                            peer_id=self._peer_id,
                            pillar_url=self._pillar_url,
                            error=str(e),
                        )
                    except Exception:
                        pass

    # ── HTTP ─────────────────────────────────────────────────────

    def _post(self, body: dict) -> None:
        """Sign + POST the snapshot bundle. Raises on any error so the
        caller's failure-soft loop can count it; the caller never
        observes a partial success."""
        url = f"{self._pillar_url}/fabric/link_stats"
        body_bytes = json.dumps(body).encode("utf-8")
        header_val, _ts = auth.build_signed_envelope(
            self._priv_key, self._peer_id,
            "POST", "/fabric/link_stats", body_bytes,
        )
        context: Optional[ssl.SSLContext] = None
        if url.startswith("https://"):
            context = auth.build_pillar_ssl_context(self._pillar_spki_hash)
        req = urlrequest.Request(
            url, data=body_bytes,
            headers={
                "Authorization": header_val,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlrequest.urlopen(
            req, timeout=self._request_timeout_s, context=context,
        ) as resp:
            # Drain the body so the connection can be reused; we
            # don't need to inspect it (a non-2xx raises HTTPError
            # before we reach here).
            resp.read()
