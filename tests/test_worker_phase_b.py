"""Tests for Phase B of the worker hardening sprint (2026-05-20).

Covers nakshatra_grpc_auth (B1) + the wire-contract symmetry (B6) +
PillarPeerKeyResolver behaviour underpinning B2 and B5.

B2 (WorkerServicer auth wiring) and B5 (SSRF defense on chain[].address)
are exercised at integration depth by the cluster smoke test — these
unit tests cover the helper contracts they depend on.

The cross-repo wire-contract assertion (B6) is at the bottom: anything
signed via the HTTP path (nakshatra_auth.sign_request used in the
worker → pillar flow) verifies on the gRPC path's verify_grpc_call when
the method token matches, and DOES NOT verify when the method token
differs (preventing HTTP-replay-on-gRPC attacks).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nakshatra_auth as auth  # noqa: E402
import nakshatra_grpc_auth as ga  # noqa: E402


# ── B1: parse_auth_header ────────────────────────────────────────────


def test_b1_parse_auth_header_well_formed():
    h = 'Sthambha-Ed25519 keyid="node-a",sig="aGVsbG8=",ts="1700000000"'
    keyid, sig, ts = ga.parse_auth_header(h)
    assert keyid == "node-a"
    assert sig == "aGVsbG8="
    assert ts == 1700000000


def test_b1_parse_auth_header_missing_raises():
    with pytest.raises(ga.AuthError, match="missing"):
        ga.parse_auth_header(None)


def test_b1_parse_auth_header_malformed_scheme():
    with pytest.raises(ga.AuthError, match="malformed"):
        ga.parse_auth_header("Bearer some-token")


def test_b1_parse_auth_header_empty_keyid():
    h = 'Sthambha-Ed25519 keyid="",sig="aGVsbG8=",ts="1700000000"'
    with pytest.raises(ga.AuthError, match="empty keyid"):
        ga.parse_auth_header(h)


def test_b1_parse_auth_header_empty_sig():
    h = 'Sthambha-Ed25519 keyid="node-a",sig="",ts="1700000000"'
    with pytest.raises(ga.AuthError, match="empty signature"):
        ga.parse_auth_header(h)


def test_b1_parse_auth_header_non_int_timestamp():
    h = 'Sthambha-Ed25519 keyid="node-a",sig="aGVsbG8=",ts="not-a-number"'
    with pytest.raises(ga.AuthError, match="timestamp"):
        ga.parse_auth_header(h)


# ── B1: build_grpc_auth_header + verify_grpc_call roundtrip ───────────


def _make_keypair_and_resolver(node_id: str = "node-a"):
    priv, pub_hex = auth.generate_keypair()
    resolver = {node_id: pub_hex}.get
    return priv, pub_hex, resolver


def test_b1_unary_roundtrip_verified_keyid():
    priv, _pub, resolver = _make_keypair_and_resolver()
    body = b"forward-request-bytes"
    header = ga.build_grpc_auth_header(
        priv, "node-a", "/nakshatra.Nakshatra/Forward", body,
    )
    keyid = ga.verify_grpc_call(
        "/nakshatra.Nakshatra/Forward", header, body, resolver,
    )
    assert keyid == "node-a"


def test_b1_streaming_roundtrip_verified_keyid():
    priv, _pub, resolver = _make_keypair_and_resolver()
    first_frame = b"first-inference-step-bytes"
    header = ga.build_grpc_auth_header(
        priv, "node-a", "/nakshatra.Nakshatra/Inference", first_frame,
        is_streaming=True,
    )
    keyid = ga.verify_grpc_call(
        "/nakshatra.Nakshatra/Inference", header, first_frame, resolver,
        is_streaming=True,
    )
    assert keyid == "node-a"


def test_b1_unknown_keyid_rejected():
    priv, _pub, resolver = _make_keypair_and_resolver()
    header = ga.build_grpc_auth_header(
        priv, "rogue-node", "/nakshatra.Nakshatra/Forward", b"body",
    )
    with pytest.raises(ga.AuthError, match="unknown keyid"):
        ga.verify_grpc_call(
            "/nakshatra.Nakshatra/Forward", header, b"body", resolver,
        )


def test_b1_signature_mismatch_rejected():
    priv, _pub, resolver = _make_keypair_and_resolver()
    header = ga.build_grpc_auth_header(
        priv, "node-a", "/nakshatra.Nakshatra/Forward", b"original-body",
    )
    # Verify with a different body — signature won't match
    with pytest.raises(ga.AuthError, match="signature mismatch"):
        ga.verify_grpc_call(
            "/nakshatra.Nakshatra/Forward", header, b"tampered-body", resolver,
        )


def test_b1_method_path_mismatch_rejected():
    """A signature for /Forward must not verify against /Inference even
    with the same body bytes."""
    priv, _pub, resolver = _make_keypair_and_resolver()
    body = b"some-payload"
    header = ga.build_grpc_auth_header(
        priv, "node-a", "/nakshatra.Nakshatra/Forward", body,
    )
    with pytest.raises(ga.AuthError, match="signature mismatch"):
        ga.verify_grpc_call(
            "/nakshatra.Nakshatra/Inference", header, body, resolver,
        )


def test_b1_stale_timestamp_rejected():
    priv, _pub, resolver = _make_keypair_and_resolver()
    old_ts = int(time.time()) - 120  # 2 minutes old (window is 60s)
    header = ga.build_grpc_auth_header(
        priv, "node-a", "/nakshatra.Nakshatra/Forward", b"body",
        timestamp_unix=old_ts,
    )
    with pytest.raises(ga.AuthError, match="out of window"):
        ga.verify_grpc_call(
            "/nakshatra.Nakshatra/Forward", header, b"body", resolver,
        )


def test_b1_future_timestamp_rejected():
    """Symmetric window check — far-future timestamps fail too."""
    priv, _pub, resolver = _make_keypair_and_resolver()
    future_ts = int(time.time()) + 200
    header = ga.build_grpc_auth_header(
        priv, "node-a", "/nakshatra.Nakshatra/Forward", b"body",
        timestamp_unix=future_ts,
    )
    with pytest.raises(ga.AuthError, match="out of window"):
        ga.verify_grpc_call(
            "/nakshatra.Nakshatra/Forward", header, b"body", resolver,
        )


def test_b1_streaming_method_differs_from_unary():
    """A streaming-signed envelope must not verify as unary, and vice
    versa. The method token is what binds the signature to the call
    shape — a captured streaming sig replayed on a unary call must fail.
    """
    priv, _pub, resolver = _make_keypair_and_resolver()
    body = b"identical-bytes"
    # Sign as streaming
    streaming_header = ga.build_grpc_auth_header(
        priv, "node-a", "/nakshatra.Nakshatra/Inference", body,
        is_streaming=True,
    )
    # Attempt unary verify
    with pytest.raises(ga.AuthError, match="signature mismatch"):
        ga.verify_grpc_call(
            "/nakshatra.Nakshatra/Inference", streaming_header, body, resolver,
            is_streaming=False,
        )


# ── B1: resolve_auth_required truth table ────────────────────────────


def test_b1_resolve_auth_required_explicit_true():
    assert ga.resolve_auth_required("true", "") is True
    assert ga.resolve_auth_required("TRUE", "http://pillar") is True
    assert ga.resolve_auth_required("1", "http://pillar") is True
    assert ga.resolve_auth_required("yes", "") is True


def test_b1_resolve_auth_required_explicit_false():
    assert ga.resolve_auth_required("false", "http://pillar") is False
    assert ga.resolve_auth_required("0", "http://pillar") is False
    assert ga.resolve_auth_required("no", "http://pillar") is False


def test_b1_resolve_auth_required_unset_with_pillar():
    """Default secure: a configured pillar implies the worker is in a
    Mode-B/C posture, so default to required."""
    assert ga.resolve_auth_required(None, "http://pillar:7777") is True
    assert ga.resolve_auth_required("", "http://pillar:7777") is True


def test_b1_resolve_auth_required_unset_without_pillar():
    """Legacy Mode A bringup (no pillar configured) defaults to
    not-required — without a resolver, requiring auth would block
    every call."""
    assert ga.resolve_auth_required(None, "") is False
    assert ga.resolve_auth_required("", "") is False


# ── B1/B5: PillarPeerKeyResolver ─────────────────────────────────────


def test_b1_resolver_starts_with_stale_cache():
    """Before any successful refresh the cache is treated as stale —
    lookups return None and is_registered_address returns False."""
    r = ga.PillarPeerKeyResolver("http://pillar:7777")
    assert r.resolve("any") is None
    assert r.is_registered_address("any:5500") is False
    stats = r.stats()
    assert stats["cached_node_ids"] == 0
    assert stats["cache_age_seconds"] is None
    assert stats["is_stale"] is True


def _mock_peers_response(peers):
    body = json.dumps({"peers": peers}).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda *args: None
    return resp


def test_b1_resolver_caches_peers_on_refresh():
    pub_hex_a = "ab" * 32
    pub_hex_b = "cd" * 32
    peers = [
        {"node_id": "alpha", "public_key_hex": pub_hex_a,
         "address": "alpha-host:5500"},
        {"node_id": "beta", "public_key_hex": pub_hex_b,
         "address": "beta-host:5510"},
    ]
    r = ga.PillarPeerKeyResolver("http://pillar:7777")
    with patch.object(ga, "urlrequest") as ur:
        ur.urlopen.return_value = _mock_peers_response(peers)
        ur.Request = MagicMock()
        r.refresh_once()
    assert r.resolve("alpha") == pub_hex_a.lower()
    assert r.resolve("beta") == pub_hex_b.lower()
    assert r.resolve("unknown") is None


def test_b1_resolver_is_registered_address_post_refresh():
    peers = [
        {"node_id": "alpha", "public_key_hex": "ab" * 32,
         "address": "alpha-host:5500"},
        {"node_id": "beta", "public_key_hex": "cd" * 32,
         "address": "beta-host:5510"},
    ]
    r = ga.PillarPeerKeyResolver("http://pillar:7777")
    with patch.object(ga, "urlrequest") as ur:
        ur.urlopen.return_value = _mock_peers_response(peers)
        ur.Request = MagicMock()
        r.refresh_once()
    assert r.is_registered_address("alpha-host:5500") is True
    assert r.is_registered_address("beta-host:5510") is True
    # SSRF probe attempts:
    assert r.is_registered_address("127.0.0.1:22") is False
    assert r.is_registered_address("internal-secret:8080") is False


def test_b1_resolver_rejects_malformed_pubkey():
    """Peers whose public_key_hex is missing / wrong length / non-hex
    are silently skipped — not the resolver's job to refuse the pillar,
    but it must not poison the cache with garbage."""
    peers = [
        {"node_id": "good", "public_key_hex": "ab" * 32, "address": "good:5500"},
        {"node_id": "no_key", "address": "no-key:5500"},
        {"node_id": "short_key", "public_key_hex": "abc", "address": "short:5500"},
        {"node_id": "non_hex", "public_key_hex": "z" * 64, "address": "zzz:5500"},
    ]
    r = ga.PillarPeerKeyResolver("http://pillar:7777")
    with patch.object(ga, "urlrequest") as ur:
        ur.urlopen.return_value = _mock_peers_response(peers)
        ur.Request = MagicMock()
        r.refresh_once()
    assert r.resolve("good") == "ab" * 32
    assert r.resolve("no_key") is None
    assert r.resolve("short_key") is None
    assert r.resolve("non_hex") is None
    # Address entries still cached for the malformed-key peers — they're
    # the pillar's claim about who's registered, and SSRF defense rests
    # on the pillar's registration view, not on key parseability.
    assert r.is_registered_address("good:5500") is True
    assert r.is_registered_address("zzz:5500") is True


def test_b1_resolver_cache_goes_stale_past_deadline():
    """After stale_cache_deadline_s with no refresh, the resolver returns
    None for every lookup — refuses to authenticate against state that
    may have missed a key rotation or eviction."""
    pub_hex = "ab" * 32
    peers = [{"node_id": "alpha", "public_key_hex": pub_hex,
              "address": "alpha:5500"}]
    r = ga.PillarPeerKeyResolver(
        "http://pillar:7777", stale_cache_deadline_s=0.1,
    )
    with patch.object(ga, "urlrequest") as ur:
        ur.urlopen.return_value = _mock_peers_response(peers)
        ur.Request = MagicMock()
        r.refresh_once()
    # Immediately after refresh: fresh
    assert r.resolve("alpha") == pub_hex.lower()
    assert r.is_registered_address("alpha:5500") is True
    # After deadline passes: stale → refuse
    time.sleep(0.2)
    assert r.resolve("alpha") is None
    assert r.is_registered_address("alpha:5500") is False


def test_b1_resolver_records_refresh_failures():
    """When refresh raises, the failure counter ticks up. Operators
    grep stats() for consecutive_failures to detect a pillar outage."""
    import urllib.error as urlerror
    r = ga.PillarPeerKeyResolver("http://pillar:7777")
    with patch.object(ga, "urlrequest") as ur:
        ur.urlopen.side_effect = urlerror.URLError("connection refused")
        ur.Request = MagicMock()
        with pytest.raises(Exception):
            r.refresh_once()
    stats = r.stats()
    assert stats["consecutive_failures"] == 1


def test_b1_resolver_successful_refresh_clears_failure_counter():
    pub_hex = "ab" * 32
    peers = [{"node_id": "alpha", "public_key_hex": pub_hex,
              "address": "alpha:5500"}]
    import urllib.error as urlerror
    r = ga.PillarPeerKeyResolver("http://pillar:7777")
    with patch.object(ga, "urlrequest") as ur:
        # Fail once
        ur.urlopen.side_effect = urlerror.URLError("transient")
        ur.Request = MagicMock()
        with pytest.raises(Exception):
            r.refresh_once()
        assert r.stats()["consecutive_failures"] == 1
        # Succeed next
        ur.urlopen.side_effect = None
        ur.urlopen.return_value = _mock_peers_response(peers)
        r.refresh_once()
    assert r.stats()["consecutive_failures"] == 0


# ── B6: Cross-repo wire-contract symmetry ────────────────────────────


def test_b6_grpc_method_tokens_are_explicit():
    """The canonical-string method tokens for gRPC must differ from
    the HTTP "POST"/"GET" pattern to prevent cross-replay between the
    HTTP control plane (worker → pillar) and the gRPC data plane
    (worker → worker)."""
    assert ga.GRPC_METHOD_UNARY == "POST"
    assert ga.GRPC_METHOD_STREAM == "STREAM"
    # Both are distinct strings — accidental aliasing would silently
    # allow stream-replay-on-unary attacks.
    assert ga.GRPC_METHOD_UNARY != ga.GRPC_METHOD_STREAM


def test_b6_anonymous_methods_include_info():
    """Info is the discovery surface — operators rely on calling Info
    without credentials to negotiate capabilities."""
    assert "/nakshatra.Nakshatra/Info" in ga.ANONYMOUS_GRPC_METHODS
    assert "/nakshatra.Nakshatra/Forward" not in ga.ANONYMOUS_GRPC_METHODS
    assert "/nakshatra.Nakshatra/Inference" not in ga.ANONYMOUS_GRPC_METHODS


def test_b6_canonical_string_is_shared_between_http_and_grpc():
    """The exact canonical string nakshatra_auth.canonical_string emits
    is what nakshatra_grpc_auth.verify_grpc_call signs against —
    different method/path values, same byte-level layout. Regression
    guard for accidental schema drift between the two paths."""
    priv, _pub = auth.generate_keypair()
    body = b"some-message"
    ts = int(time.time())
    # Build via nakshatra_auth.sign_request with method=POST (gRPC unary)
    sig_via_http = auth.sign_request(
        priv, "POST", "/nakshatra.Nakshatra/Forward", body, ts
    )
    # Build via the gRPC helper
    header_via_grpc = ga.build_grpc_auth_header(
        priv, "node-a", "/nakshatra.Nakshatra/Forward", body,
        timestamp_unix=ts,
    )
    # Extract the signature out of the gRPC header
    _keyid, sig_via_grpc, _ts = ga.parse_auth_header(header_via_grpc)
    assert sig_via_http == sig_via_grpc, (
        "gRPC and HTTP paths must produce byte-identical signatures "
        "for the same (method, path, body, ts) — drift means one of "
        "them changed unilaterally"
    )


def test_b6_pubkey_resolver_callable_protocol():
    """The pubkey_resolver argument to verify_grpc_call accepts any
    callable returning Optional[str]. PillarPeerKeyResolver.resolve
    satisfies this; tests use bare dict.get; the contract is documented
    by this test."""
    resolver = lambda keyid: "ab" * 32 if keyid == "alpha" else None
    priv, _pub = auth.generate_keypair()
    # Sign with the actual key — resolver returns a DIFFERENT pubkey,
    # so verification should fail with signature mismatch (not "unknown
    # keyid"). This confirms the resolver path is consulted but the
    # signature check is independent.
    header = ga.build_grpc_auth_header(
        priv, "alpha", "/nakshatra.Nakshatra/Forward", b"body",
    )
    with pytest.raises(ga.AuthError, match="signature mismatch"):
        ga.verify_grpc_call(
            "/nakshatra.Nakshatra/Forward", header, b"body", resolver,
        )
