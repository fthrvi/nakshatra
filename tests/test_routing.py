"""Tests for v1.0 §6 routing entry-proxy (P3).

Proves: LOCAL/ROUTE/NOT_FOUND decisioning off discovery, ROUTE picks the
highest-Fᵢ verified peer, and forward_chat signs the forwarded request with the
mesh key (a peer can authenticate it — NOT an open `api-key:mesh` forward).
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from nakshatra_auth import generate_keypair, verify_request, canonical_string  # noqa: E402
from discovery.nakshatra_listing import NakshatraListing  # noqa: E402
from discovery.relay import InMemoryRelay  # noqa: E402
from routing.model_router import (  # noqa: E402
    Decision, route_or_local, resolve_serving_peer, forward_chat)


def _peer(node_id, serving, ms, endpoint="", mesh="m1", drift_class=None):
    priv, pub = generate_keypair()
    l = NakshatraListing(mesh_id=mesh, node_id=node_id, ed25519_pubkey_hex=pub,
                         serving=serving, measured_decode_ms_per_layer=ms,
                         endpoint_hint=endpoint, drift_class=drift_class)
    l.sign(priv)
    return l


# ── drift-class filtering (v1.1 §8.1) ─────────────────────────────────

def test_drift_class_filter_routes_same_class_only():
    relay = InMemoryRelay()
    # the fastest peer is a DIFFERENT drift class → must be skipped when a
    # deterministic class is required; the slower same-class peer is chosen.
    relay.publish(_peer("fast-otherclass", ["m"], ms=0.5, endpoint="http://f:8080",
                        drift_class="classB"))
    relay.publish(_peer("ok-sameclass", ["m"], ms=5.0, endpoint="http://ok:8080",
                        drift_class="classA"))
    t = route_or_local("m", ["local"], relay, mesh_id="m1", require_drift_class="classA")
    assert t.decision is Decision.ROUTE and t.peer.node_id == "ok-sameclass"


def test_drift_class_unset_routes_regardless():
    relay = InMemoryRelay()
    relay.publish(_peer("any", ["m"], ms=2.0, endpoint="http://a:8080", drift_class="classB"))
    t = route_or_local("m", ["local"], relay, mesh_id="m1")  # no class requirement
    assert t.decision is Decision.ROUTE and t.peer.node_id == "any"


def test_drift_class_no_match_is_not_found():
    relay = InMemoryRelay()
    relay.publish(_peer("p", ["m"], ms=1.0, endpoint="http://p:8080", drift_class="classB"))
    t = route_or_local("m", ["local"], relay, mesh_id="m1", require_drift_class="classA")
    assert t.decision is Decision.NOT_FOUND   # clean reject, never a cross-class chain


# ── decisioning ───────────────────────────────────────────────────────

def test_local_model_served_here():
    relay = InMemoryRelay()
    t = route_or_local("llama-70b", ["llama-70b"], relay)
    assert t.decision is Decision.LOCAL


def test_route_to_highest_fi_peer():
    relay = InMemoryRelay()
    relay.publish(_peer("slow", ["llama-70b"], ms=9.0, endpoint="http://slow:8080"))
    relay.publish(_peer("fast", ["llama-70b"], ms=1.5, endpoint="http://fast:8080"))
    t = route_or_local("llama-70b", ["other-model"], relay, mesh_id="m1")
    assert t.decision is Decision.ROUTE
    assert t.peer.node_id == "fast"          # ranked by measured compute
    assert t.endpoint == "http://fast:8080"


def test_unknown_model_not_found():
    relay = InMemoryRelay()
    relay.publish(_peer("p", ["llama-70b"], ms=2.0, endpoint="http://p:8080"))
    t = route_or_local("nonexistent", ["local"], relay)
    assert t.decision is Decision.NOT_FOUND


def test_peer_without_endpoint_is_skipped():
    relay = InMemoryRelay()
    relay.publish(_peer("noendpoint", ["m"], ms=1.0, endpoint=""))  # fastest but undialable
    relay.publish(_peer("dialable", ["m"], ms=5.0, endpoint="http://d:8080"))
    found = resolve_serving_peer(relay, "m", mesh_id="m1")
    assert found is not None and found[0].node_id == "dialable"


def test_unsigned_peer_not_routed():
    relay = InMemoryRelay()
    # publish requires signed; simulate a relay that somehow holds an unsigned one
    bad = NakshatraListing(mesh_id="m1", node_id="bad",
                           ed25519_pubkey_hex=generate_keypair()[1],
                           serving=["m"], measured_decode_ms_per_layer=0.1,
                           endpoint_hint="http://bad:8080")  # not signed
    relay._by_node["bad"] = bad  # bypass publish guard for the test
    t = route_or_local("m", ["local"], relay)
    assert t.decision is Decision.NOT_FOUND   # unverifiable peer is invisible


# ── signed forwarding against a loopback peer ─────────────────────────

class _CapturingHandler(BaseHTTPRequestHandler):
    captured: dict = {}

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        _CapturingHandler.captured = {
            "path": self.path,
            "auth": self.headers.get("Authorization", ""),
            "routed_by": self.headers.get("X-Nakshatra-Routed-By", ""),
            "pinned": self.headers.get("X-Nakshatra-Pinned-Key", ""),
            "body": body,
        }
        resp = json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


def test_forward_chat_is_signed_and_authenticates():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CapturingHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        relay = InMemoryRelay()
        priv, pub = generate_keypair()
        peer = NakshatraListing(mesh_id="m1", node_id="peer", ed25519_pubkey_hex=pub,
                                serving=["llama-70b"], measured_decode_ms_per_layer=2.0,
                                endpoint_hint=f"http://127.0.0.1:{port}")
        peer.sign(priv)
        relay.publish(peer)

        my_priv, my_pub = generate_keypair()
        target = route_or_local("llama-70b", [], relay, mesh_id="m1")
        assert target.decision is Decision.ROUTE

        body = json.dumps({"model": "llama-70b",
                           "messages": [{"role": "user", "content": "ping"}]}).encode()
        status, resp, _ = forward_chat(target, body, my_priv, "router-node")
        assert status == 200
        assert json.loads(resp)["choices"][0]["message"]["content"] == "pong"

        cap = _CapturingHandler.captured
        assert cap["path"] == "/v1/chat/completions"
        assert cap["routed_by"] == "router-node"
        assert cap["pinned"] == pub               # pinned to the discovered key
        # the forwarded request carries a VALID Ed25519 signature from the router
        scheme, rest = cap["auth"].split(" ", 1)
        assert scheme == "Sthambha-Ed25519"
        parts = dict(p.split("=", 1) for p in rest.replace('"', "").split(","))
        assert verify_request(my_pub, "POST", "/v1/chat/completions", body,
                              int(parts["ts"]), parts["sig"])
    finally:
        server.shutdown()
