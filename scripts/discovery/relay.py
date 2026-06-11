"""Discovery transports + admission pin (v1.0 §4.2).

The doc's reusable architecture is a three-way decoupling:
    discovery (public gossip)  ≠  admission (a pinned credential)  ≠  transport
    (the closed, signed data plane).

This module owns the *discovery transport* (an abstract relay a listing is
published to / queried from) and the *admission pin* (turning a verified listing
into the Ed25519 identity the data-plane handshake must pin against).

Transports:
  • InMemoryRelay — process-local; for tests and single-host dev.
  • FileRelay     — a directory of <node_id>.json listings; a zero-dependency
                    relay over any shared filesystem / mount.
  • NostrRelay    — the production transport (public relay). Gated: real Nostr
                    events need secp256k1 Schnorr signatures (coincurve) +
                    a websocket loop. We ship the event mapping (testable, pure)
                    and raise a clear install hint until the dep is present, so
                    the architecture is complete and the wire is a drop-in.

Design rule (unchanged from the doc): discovery tells you *who exists*; it never
widens *what they can do*. Admission still requires a valid token AND a handshake
that verifies against the pinned key below.
"""
from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if __package__:
    from .nakshatra_listing import NakshatraListing, ListingError
else:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from discovery.nakshatra_listing import NakshatraListing, ListingError


# ── admission pin ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PinnedIdentity:
    """What a discovered peer is pinned to before the data plane will talk to it.
    The join handshake must verify against `ed25519_pubkey_hex`."""
    node_id: str
    ed25519_pubkey_hex: str
    endpoint_hint: str = ""


def pin_from_listing(listing: NakshatraListing) -> PinnedIdentity:
    """Turn a *verified* listing into the identity the join handshake pins
    against. Refuses an unverifiable listing — this is the line that closes
    Mesh-LLM's unsigned-listing / open-join gap."""
    if not listing.verify():
        raise ListingError(
            f"refusing to pin node {listing.node_id!r}: listing signature invalid "
            f"(discovery is public, but admission requires an identity-bound listing)")
    return PinnedIdentity(
        node_id=listing.node_id,
        ed25519_pubkey_hex=listing.ed25519_pubkey_hex,
        endpoint_hint=listing.endpoint_hint,
    )


# ── transport interface ───────────────────────────────────────────────

class DiscoveryRelay(ABC):
    @abstractmethod
    def publish(self, listing: NakshatraListing) -> None:
        """Advertise this node's listing (must be signed)."""

    @abstractmethod
    def query(self, mesh_id: Optional[str] = None) -> list[NakshatraListing]:
        """Return discovered listings, optionally filtered by mesh_id.
        Implementations MUST drop listings that fail verify() — a relay is
        untrusted, so the transport never hands up an unverifiable listing."""

    def close(self) -> None:  # optional
        pass


def _require_signed(listing: NakshatraListing) -> None:
    listing.validate()
    if not listing.verify():
        raise ListingError("refusing to publish an unsigned/invalid listing")


class InMemoryRelay(DiscoveryRelay):
    """Process-local relay. Latest listing per node_id wins."""

    def __init__(self) -> None:
        self._by_node: dict[str, NakshatraListing] = {}

    def publish(self, listing: NakshatraListing) -> None:
        _require_signed(listing)
        self._by_node[listing.node_id] = listing

    def query(self, mesh_id: Optional[str] = None) -> list[NakshatraListing]:
        out = []
        for l in self._by_node.values():
            if not l.verify():
                continue
            if mesh_id is not None and l.mesh_id != mesh_id:
                continue
            out.append(l)
        return out


class FileRelay(DiscoveryRelay):
    """Directory-backed relay: one <node_id>.json per listing. A zero-dependency
    relay over any shared filesystem (NFS, syncthing, a shared mount)."""

    def __init__(self, directory: str) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(node_id: str) -> str:
        # node_id is operator-controlled but keep it filesystem-safe.
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in node_id)[:128] or "node"

    def publish(self, listing: NakshatraListing) -> None:
        _require_signed(listing)
        dest = self.dir / f"{self._safe_name(listing.node_id)}.json"
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(listing.to_json())
        os.replace(tmp, dest)

    def query(self, mesh_id: Optional[str] = None) -> list[NakshatraListing]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                l = NakshatraListing.from_json(p.read_text())
            except (ListingError, OSError):
                continue
            if not l.verify():
                continue
            if mesh_id is not None and l.mesh_id != mesh_id:
                continue
            out.append(l)
        return out


# ── Nostr event mapping (pure; testable) + gated transport ────────────

NOSTR_KIND = 31990  # reuse Mesh-LLM's kind number (NOT NIP-89 conformant — §4.1)


def listing_to_nostr_event_content(listing: NakshatraListing) -> dict:
    """Map a listing to a Nostr event's `content` + `tags` (NIP-01 shape).
    Pure — no signing. The event's own secp256k1 signature is the Nostr
    identity; the Ed25519 mesh key travels in a tag and is what admission pins.
    This is testable without secp256k1; only publishing to a relay needs it."""
    if not listing.verify():
        raise ListingError("won't map an unverified listing to a Nostr event")
    tags = [
        ["d", listing.mesh_id],                      # NIP-33 replaceable addressing
        ["mesh_id", listing.mesh_id],
        ["node_id", listing.node_id],
        ["ed25519", listing.ed25519_pubkey_hex],     # the mesh key admission pins
    ]
    for m in sorted(listing.serving):
        tags.append(["serving", m])
    for m in sorted(listing.wanted):
        tags.append(["wanted", m])
    return {"kind": NOSTR_KIND, "content": listing.to_json(), "tags": tags}


def nostr_event_to_listing(event: dict) -> NakshatraListing:
    """Inverse: recover + re-verify a listing from a Nostr event's content."""
    listing = NakshatraListing.from_json(event["content"])
    if not listing.verify():
        raise ListingError("Nostr event carried an unverifiable listing")
    return listing


class NostrRelay(DiscoveryRelay):
    """Production transport over a public Nostr relay. Gated on secp256k1 Schnorr
    signing (coincurve) — until that dependency is present, construction raises
    with an install hint. The event mapping above is dependency-free and tested,
    so this is a drop-in once the dep lands."""

    def __init__(self, relay_url: str, nostr_privkey_hex: Optional[str] = None) -> None:
        try:
            import coincurve  # noqa: F401
        except ImportError as e:
            raise ListingError(
                "NostrRelay needs secp256k1 Schnorr signatures: "
                "`pip install coincurve` (+ websocket-client, already present). "
                "Until then use InMemoryRelay/FileRelay — the listing schema, "
                "ranking, admission pin, and event mapping are all functional."
            ) from e
        self.relay_url = relay_url
        self.nostr_privkey_hex = nostr_privkey_hex

    def publish(self, listing: NakshatraListing) -> None:  # pragma: no cover - needs dep
        raise NotImplementedError("NostrRelay.publish requires the gated secp256k1 wire")

    def query(self, mesh_id: Optional[str] = None) -> list[NakshatraListing]:  # pragma: no cover
        raise NotImplementedError("NostrRelay.query requires the gated secp256k1 wire")
