"""model_router.py — the routing entry-proxy (v1.0 §6).

Turns any node into a valid entry point: a client addresses a model by name, and
if this node doesn't serve it, the request is routed to a peer that does —
discovered via P1 (discovery), ranked by measured compute Fᵢ, and pinned to the
peer's advertised Ed25519 key. The forward is *signed* with this node's mesh key
(behind the wall — NOT Mesh-LLM's `api-key:"mesh"` open default).

Composition:
    discovery (who serves what)  →  routing decision (LOCAL / ROUTE / NOT_FOUND)
    →  signed forward to the pinned peer.

This module is the decision + forward logic. The integration is one hook: when
nakshatra_serve's OpenAI handler would 404 a model it doesn't hold, it calls
route_or_local() and, on ROUTE, forwards instead of 404-ing.
"""
from __future__ import annotations

import enum
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib import request as urlrequest, error as urlerror

_log = logging.getLogger("nakshatra.router")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nakshatra_auth import build_signed_envelope  # noqa: E402
from discovery.nakshatra_listing import rank_listings  # noqa: E402
from discovery.relay import DiscoveryRelay, PinnedIdentity, pin_from_listing  # noqa: E402
from wire.version import is_compatible  # noqa: E402


@dataclass
class EntryProxyConfig:
    """Attach to a serving HTTPServer (as `server.entry_proxy`) to turn a
    model-not-served-here 404 into a routed forward. Absent ⇒ behaviour unchanged."""
    relay: "DiscoveryRelay"
    priv_bytes: bytes
    node_id: str
    mesh_id: Optional[str] = None


class Decision(enum.Enum):
    LOCAL = "local"          # this node serves the model — handle it here
    ROUTE = "route"          # a peer serves it — forward
    NOT_FOUND = "not_found"  # nobody (verified) serves it


@dataclass
class RouteTarget:
    decision: Decision
    peer: Optional[PinnedIdentity] = None    # set iff ROUTE
    endpoint: str = ""                       # peer dial target iff ROUTE
    score: float = 0.0


# Default listing freshness for routing decisions. Matches meshd's heartbeat
# discipline (publish every 30s, dead after a few missed beats — its auto TTL
# floor is 90s; nks-capacity drops peers at 90s too). Before this existed,
# resolve_serving_peer would happily route to a listing of ANY age — a node
# gone for a week still won routing as long as its file sat on the relay.
#
# Operator override: NKS_ROUTE_MAX_AGE_S (seconds; 0/negative disables the gate,
# same as max_age_s=None). Set it at or above meshd's effective TTL
# (max(90, 4×refresh)) so a peer meshd still considers live isn't 404'd here —
# a large --refresh widens meshd's TTL while this default stays 120s.
def _default_max_age() -> Optional[float]:
    raw = os.environ.get("NKS_ROUTE_MAX_AGE_S")
    if raw is None:
        return 120.0
    try:
        v = float(raw)
    except ValueError:
        return 120.0
    return None if v <= 0 else v


ROUTE_MAX_AGE_S = _default_max_age()


def resolve_serving_peer(relay: DiscoveryRelay, model: str, *,
                         mesh_id: Optional[str] = None,
                         exclude_node_id: str = "",
                         require_drift_class: Optional[str] = None,
                         max_age_s: Optional[float] = ROUTE_MAX_AGE_S
                         ) -> Optional[tuple[PinnedIdentity, str, float]]:
    """Discover the best *verified* peer serving `model`, ranked by measured
    compute. Returns (pinned_identity, endpoint_hint, score) or None.

    Only verified listings whose `serving` includes the model are considered;
    rank_listings already drops unsigned/unverifiable ones and self.

    `require_drift_class` (v1.1 §8.1): when set (e.g. this node's own gauge
    fingerprint), only peers advertising the SAME drift_class are eligible — a
    bit-deterministic chain must stay in one engine-build class
    (cross-machine-validation.md §2a). Leave None for throughput work where
    bit-identity isn't required.

    `max_age_s`: drop listings whose heartbeat is older than this before ranking
    (default ROUTE_MAX_AGE_S; None disables — caller owns the risk). An unstamped
    listing (created_unix=0) is treated as INFINITELY OLD and dropped: a signed
    listing with no timestamp is indistinguishable from an ancient one, and every
    real publisher stamps (meshd, discover.py). Each drop is logged (parity with
    meshd's 'skip … stale heartbeat')."""
    # §7: drop peers we can't speak to BEFORE pinning/forwarding — a clean
    # pre-join reject, never a silent attempt against an incompatible wire.
    # §8.1: when a deterministic class is required, drop out-of-class peers too.
    now = time.time()
    listings = []
    for l in relay.query(mesh_id=mesh_id):
        if model not in l.serving:
            continue
        if not is_compatible(l.supported_protocol):
            continue
        if require_drift_class is not None and l.drift_class != require_drift_class:
            continue
        if max_age_s is not None:
            age = now - (l.created_unix or 0)
            if age > max_age_s:
                _log.info("route: drop %s serving %s — stale (%.0fs > %.0fs)",
                          l.node_id, model, age, max_age_s)
                continue
        listings.append(l)
    ranked = rank_listings(listings, exclude_node_id=exclude_node_id,
                           want_mesh_id=mesh_id, want_model=model)
    for listing, score in ranked:
        if not listing.endpoint_hint:
            continue  # can't dial a peer with no endpoint; try the next best
        return pin_from_listing(listing), listing.endpoint_hint, score
    return None


def route_or_local(model: str, local_model_names: Iterable[str], relay: DiscoveryRelay,
                   *, mesh_id: Optional[str] = None, own_node_id: str = "",
                   require_drift_class: Optional[str] = None,
                   max_age_s: Optional[float] = ROUTE_MAX_AGE_S) -> RouteTarget:
    """The entry-proxy decision. LOCAL if we serve it; else ROUTE to the best
    discovered peer; else NOT_FOUND. `require_drift_class` (v1.1 §8.1) restricts
    ROUTE to same-drift-class peers for bit-deterministic chains. `max_age_s`
    drops stale-heartbeat peers before ranking (see resolve_serving_peer)."""
    if model in set(local_model_names):
        return RouteTarget(Decision.LOCAL)
    found = resolve_serving_peer(relay, model, mesh_id=mesh_id, exclude_node_id=own_node_id,
                                 require_drift_class=require_drift_class,
                                 max_age_s=max_age_s)
    if found is None:
        return RouteTarget(Decision.NOT_FOUND)
    peer, endpoint, score = found
    return RouteTarget(Decision.ROUTE, peer=peer, endpoint=endpoint, score=score)


def forward_chat(target: RouteTarget, body: bytes, priv_bytes: bytes, node_id: str,
                 *, path: str = "/v1/chat/completions", timeout: float = 120.0
                 ) -> tuple[int, bytes, dict]:
    """Forward an OpenAI chat body to the routed peer, SIGNED with this node's
    mesh key (the peer authenticates it against the same Ed25519 posture as the
    rest of the data plane). Returns (status, body_bytes, headers).

    The endpoint_hint is advisory; the peer still enforces admission. We sign so
    an open relay can never inject an unauthenticated request into the mesh."""
    if target.decision is not Decision.ROUTE or not target.endpoint:
        raise ValueError("forward_chat requires a ROUTE target with an endpoint")
    url = target.endpoint.rstrip("/") + path
    header, _ts = build_signed_envelope(priv_bytes, node_id, "POST", path, body)
    req = urlrequest.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": header,
        "X-Nakshatra-Routed-By": node_id,
        "X-Nakshatra-Pinned-Key": target.peer.ed25519_pubkey_hex if target.peer else "",
    })
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urlerror.HTTPError as e:
        return e.code, e.read(), dict(e.headers or {})


def not_found_body(model: str) -> bytes:
    """OpenAI-shaped 404 for a model nobody serves — keeps the surface honest."""
    return json.dumps({"error": {
        "message": f"model {model!r} is not served locally or by any discovered peer",
        "type": "model_not_found", "code": "model_not_found",
    }}).encode("utf-8")
