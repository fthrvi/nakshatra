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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib import request as urlrequest, error as urlerror

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


def resolve_serving_peer(relay: DiscoveryRelay, model: str, *,
                         mesh_id: Optional[str] = None,
                         exclude_node_id: str = "",
                         require_drift_class: Optional[str] = None
                         ) -> Optional[tuple[PinnedIdentity, str, float]]:
    """Discover the best *verified* peer serving `model`, ranked by measured
    compute. Returns (pinned_identity, endpoint_hint, score) or None.

    Only verified listings whose `serving` includes the model are considered;
    rank_listings already drops unsigned/unverifiable ones and self.

    `require_drift_class` (v1.1 §8.1): when set (e.g. this node's own gauge
    fingerprint), only peers advertising the SAME drift_class are eligible — a
    bit-deterministic chain must stay in one engine-build class
    (cross-machine-validation.md §2a). Leave None for throughput work where
    bit-identity isn't required."""
    # §7: drop peers we can't speak to BEFORE pinning/forwarding — a clean
    # pre-join reject, never a silent attempt against an incompatible wire.
    # §8.1: when a deterministic class is required, drop out-of-class peers too.
    listings = [
        l for l in relay.query(mesh_id=mesh_id)
        if model in l.serving
        and is_compatible(l.supported_protocol)
        and (require_drift_class is None or l.drift_class == require_drift_class)
    ]
    ranked = rank_listings(listings, exclude_node_id=exclude_node_id,
                           want_mesh_id=mesh_id, want_model=model)
    for listing, score in ranked:
        if not listing.endpoint_hint:
            continue  # can't dial a peer with no endpoint; try the next best
        return pin_from_listing(listing), listing.endpoint_hint, score
    return None


def route_or_local(model: str, local_model_names: Iterable[str], relay: DiscoveryRelay,
                   *, mesh_id: Optional[str] = None, own_node_id: str = "",
                   require_drift_class: Optional[str] = None) -> RouteTarget:
    """The entry-proxy decision. LOCAL if we serve it; else ROUTE to the best
    discovered peer; else NOT_FOUND. `require_drift_class` (v1.1 §8.1) restricts
    ROUTE to same-drift-class peers for bit-deterministic chains."""
    if model in set(local_model_names):
        return RouteTarget(Decision.LOCAL)
    found = resolve_serving_peer(relay, model, mesh_id=mesh_id, exclude_node_id=own_node_id,
                                 require_drift_class=require_drift_class)
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
