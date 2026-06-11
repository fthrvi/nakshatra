"""Tests for the real Nostr discovery transport (P1 §4 wire).

- nostr.py NIP-01 primitives (event id, Schnorr sign/verify, tamper) — pure.
- NostrRelay publish/query round-trip through a minimal in-process NIP-01 relay
  (no external egress — the box firewall blocks public relays anyway).

Guarded on coincurve + websockets; skips where the deps aren't installed.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

pytest.importorskip("coincurve", reason="Nostr wire needs coincurve (BIP340 Schnorr)")
pytest.importorskip("websockets", reason="local relay needs the websockets server lib")

from nakshatra_auth import generate_keypair  # noqa: E402
from discovery import nostr  # noqa: E402
from discovery.nakshatra_listing import NakshatraListing, rank_listings  # noqa: E402
from discovery.relay import NostrRelay, pin_from_listing, NOSTR_KIND  # noqa: E402


# ── NIP-01 primitives ─────────────────────────────────────────────────

def test_event_build_verify_tamper():
    priv, pub = nostr.keygen()
    ev = nostr.build_event(priv, NOSTR_KIND, "content", [["mesh_id", "m1"]], 1700000000)
    assert len(ev["id"]) == 64 and len(ev["sig"]) == 128 and ev["pubkey"] == pub
    assert nostr.verify_event(ev)
    for field, val in [("content", "x"), ("created_at", 1), ("kind", 1)]:
        bad = dict(ev); bad[field] = val
        assert not nostr.verify_event(bad)          # id no longer matches
    forged = dict(ev); forged["sig"] = "00" * 64
    assert not nostr.verify_event(forged)


def test_pubkey_of_matches_keygen():
    priv, pub = nostr.keygen()
    assert nostr.pubkey_of(priv) == pub


# ── minimal in-process NIP-01 relay ───────────────────────────────────

class _LocalRelay:
    """Stores EVENTs; answers REQ with matching stored events + EOSE."""

    def __init__(self):
        self.events: list[dict] = []
        self.port = None
        self._loop = None
        self._thread = None

    @staticmethod
    def _matches(ev, filt) -> bool:
        if "kinds" in filt and ev["kind"] not in filt["kinds"]:
            return False
        for key, vals in filt.items():
            if key.startswith("#"):
                tagname = key[1:]
                tagvals = [t[1] for t in ev["tags"] if len(t) >= 2 and t[0] == tagname]
                if not any(v in tagvals for v in vals):
                    return False
        return True

    async def _handler(self, ws):
        async for raw in ws:
            msg = json.loads(raw)
            if msg[0] == "EVENT":
                self.events.append(msg[1])
                await ws.send(json.dumps(["OK", msg[1]["id"], True, ""]))
            elif msg[0] == "REQ":
                sub, filt = msg[1], (msg[2] if len(msg) > 2 else {})
                for ev in self.events:
                    if self._matches(ev, filt):
                        await ws.send(json.dumps(["EVENT", sub, ev]))
                await ws.send(json.dumps(["EOSE", sub]))
            elif msg[0] == "CLOSE":
                pass

    def start(self):
        import websockets
        ready = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def main():
                server = await websockets.serve(self._handler, "127.0.0.1", 0)
                self.port = server.sockets[0].getsockname()[1]
                ready.set()
                await asyncio.Future()  # run forever

            try:
                self._loop.run_until_complete(main())
            except asyncio.CancelledError:
                pass

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait(timeout=5)
        return f"ws://127.0.0.1:{self.port}"


@pytest.fixture
def local_relay_url():
    relay = _LocalRelay()
    yield relay.start()


def _signed_listing(node_id, mesh, ms):
    priv, pub = generate_keypair()
    l = NakshatraListing(mesh_id=mesh, node_id=node_id, ed25519_pubkey_hex=pub,
                         serving=["llama-70b"], measured_decode_ms_per_layer=ms,
                         endpoint_hint="http://127.0.0.1:5530",
                         created_unix=1700000000)
    l.sign(priv)
    return l, pub


def test_nostr_relay_publish_query_roundtrip(local_relay_url):
    mesh = "roundtrip-mesh"
    relay = NostrRelay(local_relay_url, timeout=5.0)
    fast, fast_pub = _signed_listing("fast", mesh, 1.5)
    slow, _ = _signed_listing("slow", mesh, 7.0)
    relay.publish(fast)
    relay.publish(slow)
    time.sleep(0.3)

    got = relay.query(mesh_id=mesh)
    assert {l.node_id for l in got} == {"fast", "slow"}
    assert all(l.verify() for l in got)                 # inner Ed25519 verified
    ranked = rank_listings(got)
    assert ranked[0][0].node_id == "fast"               # compute-aware ranking survives the wire
    assert pin_from_listing(got[0]).ed25519_pubkey_hex == fast_pub  # admission pin intact


def test_nostr_relay_mesh_filter(local_relay_url):
    relay = NostrRelay(local_relay_url, timeout=5.0)
    a, _ = _signed_listing("a", "mesh-A", 2.0)
    b, _ = _signed_listing("b", "mesh-B", 2.0)
    relay.publish(a); relay.publish(b)
    time.sleep(0.3)
    assert {l.node_id for l in relay.query(mesh_id="mesh-A")} == {"a"}
