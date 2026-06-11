"""NakshatraListing — signed, compute-rankable discovery listings (v1.0 §4).

This is the discovery-layer payload that lets peers find each other without a
hand-edited YAML. It adapts Mesh-LLM's `MeshListing` but closes its two gaps:

  1. **Signed + identity-bound.** Mesh-LLM listings are effectively unsigned and
     can lie about who they are. A NakshatraListing carries the node's Ed25519
     mesh pubkey AND is self-signed by it, so the join handshake can *pin*
     against the advertised key (§4.2 admission). Discovery stays public; the
     mesh stays closed.
  2. **Compute-aware ranking.** Mesh-LLM's `score_mesh` ranks by RTT only. We
     carry `measured_decode_ms_per_layer` (the Sthambha planner's Fᵢ signal) and
     rank by *measured compute* (§4.3) — the move Mesh-LLM structurally can't
     make. Borrowed `score_mesh` factors (sticky mesh_id, capacity penalty,
     larger-mesh preference, serving/wanted match) are kept as secondary terms.

The listing is transport-agnostic (relay.py carries it over an in-memory/file
relay for tests, or a Nostr relay in production). Advertised capacity is a HINT
— §4.4 says verify with a liveness+capability probe before assigning real layers.
"""
from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

SCHEMA_VERSION = 1


class ListingError(Exception):
    """Listing is malformed or fails verification."""


@dataclass
class NakshatraListing:
    mesh_id: str                       # which mesh this node belongs to (sticky)
    node_id: str                       # stable node identifier
    ed25519_pubkey_hex: str            # mesh identity; admission pins against this
    serving: list[str] = field(default_factory=list)   # model ids served now
    wanted: list[str] = field(default_factory=list)    # models the mesh lacks GPU for
    total_vram_bytes: int = 0
    node_count: int = 1                # mesh size as this node sees it
    # The compute signal — lower is faster. None = unmeasured (ranked last).
    measured_decode_ms_per_layer: Optional[float] = None
    endpoint_hint: str = ""            # advisory dial hint (still pinned at transport)
    capacity_full: bool = False        # node is at client capacity (de-prioritise)
    created_unix: int = 0
    schema_version: int = SCHEMA_VERSION
    signature_b64: Optional[str] = None

    # ---- canonical bytes (excludes the signature) ----
    def _canonical_obj(self) -> dict:
        d = asdict(self)
        d.pop("signature_b64", None)
        d["serving"] = sorted(self.serving)
        d["wanted"] = sorted(self.wanted)
        return d

    def canonical_bytes(self) -> bytes:
        return json.dumps(self._canonical_obj(), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")

    # ---- sign / verify (self-signed by the advertised mesh key) ----
    def sign(self, priv_bytes: bytes) -> None:
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
        pub_hex = priv.public_key().public_bytes_raw().hex()
        if self.ed25519_pubkey_hex and self.ed25519_pubkey_hex != pub_hex:
            raise ListingError("signing key does not match advertised ed25519_pubkey_hex")
        self.ed25519_pubkey_hex = pub_hex
        self.signature_b64 = base64.b64encode(
            priv.sign(self.canonical_bytes())).decode("ascii")

    def verify(self) -> bool:
        """True iff self-signature is present and valid for the advertised key.
        This is what makes a discovered listing trustworthy enough to *pin*."""
        if not (self.ed25519_pubkey_hex and self.signature_b64):
            return False
        try:
            pub = ed25519.Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(self.ed25519_pubkey_hex))
            pub.verify(base64.b64decode(self.signature_b64, validate=True),
                       self.canonical_bytes())
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ListingError(f"unsupported schema_version {self.schema_version}")
        if not self.mesh_id or not self.node_id:
            raise ListingError("mesh_id and node_id are required")
        if not (isinstance(self.ed25519_pubkey_hex, str) and len(self.ed25519_pubkey_hex) == 64):
            raise ListingError("ed25519_pubkey_hex must be 64 hex chars")
        try:
            bytes.fromhex(self.ed25519_pubkey_hex)
        except ValueError:
            raise ListingError("ed25519_pubkey_hex is not hex")
        if self.node_count < 1:
            raise ListingError("node_count must be >= 1")
        if (self.measured_decode_ms_per_layer is not None and
                (not math.isfinite(self.measured_decode_ms_per_layer)
                 or self.measured_decode_ms_per_layer < 0)):
            raise ListingError("measured_decode_ms_per_layer must be a non-negative finite number")

    # ---- (de)serialization ----
    def to_json(self) -> str:
        obj = self._canonical_obj()
        if self.signature_b64:
            obj["signature_b64"] = self.signature_b64
        return json.dumps(obj, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "NakshatraListing":
        try:
            o = json.loads(text)
        except json.JSONDecodeError as e:
            raise ListingError(f"listing is not valid JSON: {e}")
        if not isinstance(o, dict):
            raise ListingError("listing must be a JSON object")
        return cls(
            mesh_id=o.get("mesh_id", ""),
            node_id=o.get("node_id", ""),
            ed25519_pubkey_hex=o.get("ed25519_pubkey_hex", ""),
            serving=list(o.get("serving", [])),
            wanted=list(o.get("wanted", [])),
            total_vram_bytes=int(o.get("total_vram_bytes", 0)),
            node_count=int(o.get("node_count", 1)),
            measured_decode_ms_per_layer=o.get("measured_decode_ms_per_layer"),
            endpoint_hint=o.get("endpoint_hint", ""),
            capacity_full=bool(o.get("capacity_full", False)),
            created_unix=int(o.get("created_unix", 0)),
            schema_version=int(o.get("schema_version", 0)),
            signature_b64=o.get("signature_b64"),
        )


# ── compute-aware ranking (§4.3) ──────────────────────────────────────
#
# Primary key: measured compute (Fᵢ ∝ 1/decode_ms_per_layer). Secondary: the
# useful score_mesh factors. We return a single float — HIGHER is better — so
# callers can sort(reverse=True). Unsigned/unmeasured listings are penalised,
# never silently trusted.

# Borrowed score_mesh weights (the *useful* ones; we drop RTT as primary).
W_STICKY_MESH = 500.0     # session continuity — prefer the mesh we're already in
W_SERVING_MATCH = 10.0    # node already serves the model we want
W_WANTED_MATCH = 15.0     # node wants what we can give (demand signal)
W_NODE_COUNT = 5.0        # larger meshes preferred (more capacity)
W_CAPACITY_FULL = -1000.0  # node is full — strongly de-prioritise
W_UNSIGNED = -1e9         # never rank an unverifiable listing above a verified one
COMPUTE_SCALE = 1000.0    # converts Fᵢ (1/ms) into a dominant score term


def score_listing(listing: NakshatraListing, *, want_mesh_id: Optional[str] = None,
                  want_model: Optional[str] = None,
                  offer_model: Optional[str] = None) -> float:
    """Compute-aware peer score; higher = better. §4.3.

    The compute term dominates (the Fᵢ move Mesh-LLM can't make); the borrowed
    factors break ties. An unsigned/invalid listing is forced to the bottom —
    discovery is public, but we only *act* on identity-bound listings."""
    if not listing.verify():
        return W_UNSIGNED
    score = 0.0
    # primary: measured compute (lower decode-ms/layer ⇒ higher Fᵢ ⇒ higher score)
    ms = listing.measured_decode_ms_per_layer
    if ms is not None and ms > 0:
        score += COMPUTE_SCALE * (1.0 / ms)
    else:
        score -= COMPUTE_SCALE  # unmeasured compute ranks below any measured peer
    # secondary: borrowed score_mesh factors
    if want_mesh_id is not None and listing.mesh_id == want_mesh_id:
        score += W_STICKY_MESH
    if want_model is not None and want_model in listing.serving:
        score += W_SERVING_MATCH
    if offer_model is not None and offer_model in listing.wanted:
        score += W_WANTED_MATCH
    score += W_NODE_COUNT * listing.node_count
    if listing.capacity_full:
        score += W_CAPACITY_FULL
    return score


def rank_listings(listings: list[NakshatraListing], *, exclude_node_id: str = "",
                  **score_kwargs) -> list[tuple[NakshatraListing, float]]:
    """Verify, score, and sort candidate listings best-first. Drops self and any
    listing that fails verification (score == W_UNSIGNED)."""
    scored = [
        (l, score_listing(l, **score_kwargs))
        for l in listings
        if l.node_id != exclude_node_id
    ]
    scored = [(l, s) for (l, s) in scored if s > W_UNSIGNED]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
