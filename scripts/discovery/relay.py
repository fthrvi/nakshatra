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
    """Production transport over a public Nostr relay (NIP-01 over websocket).

    Two keys, two jobs: the secp256k1 Nostr key signs the relay *event* (anti-spam
    at the relay); the Ed25519 mesh key inside the listing is what *admission*
    pins against. query() returns only listings that pass BOTH the Nostr event
    signature AND the inner Ed25519 self-signature — a relay is untrusted, so an
    unverifiable listing never reaches the ranker.

    Needs `coincurve` (BIP340 Schnorr) + `websocket-client`; __init__ raises a
    clear install hint if absent. The listing schema, ranking, admission pin, and
    event mapping all work without it — only this transport needs the dep."""

    def __init__(self, relay_url: str, nostr_privkey_hex: Optional[str] = None,
                 timeout: float = 10.0) -> None:
        try:
            from . import nostr as _nostr
            import websocket  # noqa: F401  (websocket-client)
        except ImportError as e:
            raise ListingError(
                "NostrRelay needs `pip install coincurve websocket-client`. "
                "Until then use InMemoryRelay/FileRelay — schema, ranking, "
                "admission pin, and event mapping are all functional without it."
            ) from e
        self._nostr = _nostr
        self.relay_url = relay_url
        self.timeout = timeout
        self.nostr_privkey_hex = nostr_privkey_hex or _nostr.keygen()[0]

    def _connect(self):
        import websocket
        return websocket.create_connection(self.relay_url, timeout=self.timeout)

    def publish(self, listing: NakshatraListing) -> None:
        _require_signed(listing)
        import time
        ev_content = listing_to_nostr_event_content(listing)  # also re-verifies
        event = self._nostr.build_event(
            self.nostr_privkey_hex, ev_content["kind"], ev_content["content"],
            ev_content["tags"], int(time.time()))
        ws = self._connect()
        try:
            ws.send(json.dumps(["EVENT", event]))
            # best-effort: read the relay's OK/NOTICE (don't hard-fail on quirks)
            try:
                resp = json.loads(ws.recv())
                if resp and resp[0] == "OK" and len(resp) >= 3 and not resp[2]:
                    raise ListingError(f"relay rejected event: {resp[3:]}")
            except (ValueError, IndexError):
                pass
        finally:
            ws.close()

    def query(self, mesh_id: Optional[str] = None) -> list[NakshatraListing]:
        import time
        sub = "nks-disc"
        filt: dict = {"kinds": [NOSTR_KIND]}
        if mesh_id is not None:
            filt["#mesh_id"] = [mesh_id]
        ws = self._connect()
        out: list[NakshatraListing] = []
        seen: set[str] = set()
        try:
            ws.send(json.dumps(["REQ", sub, filt]))
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                try:
                    msg = json.loads(ws.recv())
                except (ValueError, OSError):
                    break
                if not msg:
                    continue
                if msg[0] == "EOSE":
                    break
                if msg[0] == "EVENT" and len(msg) >= 3:
                    event = msg[2]
                    if event.get("id") in seen:
                        continue
                    seen.add(event.get("id"))
                    # both layers must verify: Nostr event sig + inner Ed25519
                    if not self._nostr.verify_event(event):
                        continue
                    try:
                        listing = nostr_event_to_listing(event)  # verifies Ed25519
                    except ListingError:
                        continue
                    if mesh_id is not None and listing.mesh_id != mesh_id:
                        continue
                    out.append(listing)
            try:
                ws.send(json.dumps(["CLOSE", sub]))
            except OSError:
                pass
        finally:
            ws.close()
        return out
