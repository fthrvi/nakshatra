"""Tests for v1.0 §4 discovery (P1): signed listings, compute-aware ranking,
admission pin, and the relay transports.

Self-contained (no live relay, no secp256k1). Proves the two innovations over
Mesh-LLM: (1) listings are Ed25519-signed and pinned at admission, (2) ranking
is driven by measured compute Fᵢ, not RTT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from nakshatra_auth import generate_keypair  # noqa: E402
from discovery.nakshatra_listing import (  # noqa: E402
    NakshatraListing, ListingError, score_listing, rank_listings, W_UNSIGNED)
from discovery.relay import (  # noqa: E402
    InMemoryRelay, FileRelay, NostrRelay, pin_from_listing,
    listing_to_nostr_event_content, nostr_event_to_listing)


def _listing(node_id="n1", mesh_id="m1", ms=None, serving=None, wanted=None,
             node_count=1, capacity_full=False, sign=True):
    priv, pub = generate_keypair()
    l = NakshatraListing(
        mesh_id=mesh_id, node_id=node_id, ed25519_pubkey_hex=pub,
        serving=serving or [], wanted=wanted or [],
        measured_decode_ms_per_layer=ms, node_count=node_count,
        capacity_full=capacity_full, total_vram_bytes=16 * 2**30,
    )
    if sign:
        l.sign(priv)
    return l, priv, pub


# ── listing schema ────────────────────────────────────────────────────

def test_sign_verify_and_tamper():
    l, priv, pub = _listing()
    assert l.verify()
    l2 = NakshatraListing.from_json(l.to_json())
    assert l2.verify()
    l2.serving = ["sneaky-model"]           # tamper after signing
    assert not l2.verify()


def test_validate_rejects_bad_pubkey():
    l, _, _ = _listing()
    l.ed25519_pubkey_hex = "xyz"
    with pytest.raises(ListingError):
        l.validate()


def test_signing_key_must_match_advertised_pubkey():
    priv_a, pub_a = generate_keypair()
    _, pub_b = generate_keypair()
    l = NakshatraListing(mesh_id="m", node_id="n", ed25519_pubkey_hex=pub_b)
    with pytest.raises(ListingError):
        l.sign(priv_a)  # priv_a doesn't match advertised pub_b


# ── compute-aware ranking (§4.3) ──────────────────────────────────────

def test_faster_compute_ranks_higher():
    fast, *_ = _listing(node_id="fast", ms=2.0)    # lower ms = faster = higher Fᵢ
    slow, *_ = _listing(node_id="slow", ms=8.0)
    ranked = rank_listings([slow, fast])
    assert [l.node_id for l, _ in ranked] == ["fast", "slow"]


def test_measured_beats_unmeasured():
    measured, *_ = _listing(node_id="measured", ms=50.0)  # even a slow measured node…
    unmeasured, *_ = _listing(node_id="unmeasured", ms=None)
    ranked = rank_listings([unmeasured, measured])
    assert ranked[0][0].node_id == "measured"  # …outranks an unmeasured one


def test_unsigned_listing_is_dropped_from_ranking():
    good, *_ = _listing(node_id="good", ms=5.0)
    bad, *_ = _listing(node_id="bad", ms=1.0, sign=False)  # faster but unsigned
    assert score_listing(bad) == W_UNSIGNED
    ranked = rank_listings([bad, good])
    assert [l.node_id for l, _ in ranked] == ["good"]      # bad dropped entirely


def test_secondary_factors_break_ties():
    a, *_ = _listing(node_id="a", mesh_id="home", ms=5.0)
    b, *_ = _listing(node_id="b", mesh_id="other", ms=5.0)
    # same compute; sticky mesh_id should lift the in-mesh peer
    ranked = rank_listings([b, a], want_mesh_id="home")
    assert ranked[0][0].node_id == "a"


def test_capacity_full_deprioritised():
    full, *_ = _listing(node_id="full", ms=2.0, capacity_full=True)
    open_, *_ = _listing(node_id="open", ms=6.0)
    ranked = rank_listings([full, open_])
    assert ranked[0][0].node_id == "open"  # -1000 penalty outweighs the compute edge


def test_rank_excludes_self():
    a, *_ = _listing(node_id="self", ms=2.0)
    b, *_ = _listing(node_id="other", ms=9.0)
    ranked = rank_listings([a, b], exclude_node_id="self")
    assert [l.node_id for l, _ in ranked] == ["other"]


# ── admission pin (§4.2) ──────────────────────────────────────────────

def test_pin_from_verified_listing():
    l, _, pub = _listing(node_id="peer")
    pin = pin_from_listing(l)
    assert pin.node_id == "peer" and pin.ed25519_pubkey_hex == pub


def test_pin_refuses_unsigned():
    l, *_ = _listing(sign=False)
    with pytest.raises(ListingError):
        pin_from_listing(l)


# ── relays ────────────────────────────────────────────────────────────

def test_inmemory_relay_roundtrip_and_filter():
    r = InMemoryRelay()
    a, *_ = _listing(node_id="a", mesh_id="m1")
    b, *_ = _listing(node_id="b", mesh_id="m2")
    r.publish(a); r.publish(b)
    assert {l.node_id for l in r.query()} == {"a", "b"}
    assert {l.node_id for l in r.query(mesh_id="m1")} == {"a"}


def test_relay_refuses_unsigned_publish():
    r = InMemoryRelay()
    l, *_ = _listing(sign=False)
    with pytest.raises(ListingError):
        r.publish(l)


def test_file_relay_roundtrip(tmp_path):
    r = FileRelay(str(tmp_path))
    a, *_ = _listing(node_id="a", ms=3.0)
    r.publish(a)
    r2 = FileRelay(str(tmp_path))         # fresh handle, same dir
    got = r2.query()
    assert len(got) == 1 and got[0].node_id == "a" and got[0].verify()


def test_file_relay_drops_corrupt_listing(tmp_path):
    r = FileRelay(str(tmp_path))
    a, *_ = _listing(node_id="a")
    r.publish(a)
    # corrupt the on-disk listing
    f = next(tmp_path.glob("*.json"))
    f.write_text(f.read_text().replace('"a"', '"a-tampered"'))
    assert r.query() == []  # verification fails -> dropped


# ── Nostr event mapping (pure) + gated transport ──────────────────────

def test_nostr_event_mapping_roundtrip():
    l, _, pub = _listing(node_id="peer", serving=["llama-70b"], wanted=["qwen"])
    ev = listing_to_nostr_event_content(l)
    assert ev["kind"] == 31990
    assert ["ed25519", pub] in ev["tags"]
    back = nostr_event_to_listing(ev)
    assert back.verify() and back.node_id == "peer"


def test_nostr_relay_gated_without_secp256k1():
    pytest.importorskip  # noqa
    try:
        import coincurve  # noqa: F401
        pytest.skip("coincurve present — gate not exercised")
    except ImportError:
        pass
    with pytest.raises(ListingError, match="coincurve"):
        NostrRelay("wss://relay.example")
