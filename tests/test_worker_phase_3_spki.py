"""Tests for Phase 3 of the 2026-05-21 SPKI federation sprint.

Phase 3 ships worker-side SPKI pinning on outbound gRPC streams. This
file covers items 3.1, 3.2, 3.6, and 3.7 — the resolver extension.
Items 3.3–3.5 (the pinned-channel mechanism) live in
test_pinned_channel.py once that lands.

  3.1 — PillarPeerKeyResolver extended with peer_spki_hash cache
  3.2 — expected_spki(address) lookup
  3.6 — stale-cache refuse semantics on the new lookup
  3.7 — worker-side cross-repo complement: a /peers response carrying
        peer_spki_hash parses correctly through the resolver's
        refresh path (Phase 1.8 was the pillar-side assertion of the
        same contract; this is the worker-side complement)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import nakshatra_grpc_auth as ga  # noqa: E402


VALID_SPKI_A = "a" * 64
VALID_SPKI_B = "b" * 64
SHORT_SPKI = "abc"
NON_HEX_SPKI = "z" * 64


def _mock_peers_response(peers):
    body = json.dumps({"peers": peers}).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda *args: None
    return resp


def _refresh_with(peers):
    r = ga.PillarPeerKeyResolver("http://pillar:7777")
    with patch.object(ga, "urlrequest") as ur:
        ur.urlopen.return_value = _mock_peers_response(peers)
        ur.Request = MagicMock()
        r.refresh_once()
    return r


# ── 3.1 — peer_spki_hash cache populated from /peers ─────────────────


def test_3_1_resolver_caches_spki_hashes_on_refresh():
    r = _refresh_with([
        {"node_id": "alpha", "public_key_hex": "ab" * 32,
         "address": "alpha-host:5500", "peer_spki_hash": VALID_SPKI_A},
        {"node_id": "beta", "public_key_hex": "cd" * 32,
         "address": "beta-host:5510", "peer_spki_hash": VALID_SPKI_B},
    ])
    assert r.expected_spki_for_node_id("alpha") == VALID_SPKI_A
    assert r.expected_spki_for_node_id("beta") == VALID_SPKI_B


def test_3_1_resolver_handles_missing_spki_field():
    """Pre-Phase-2 workers (or peers that explicitly haven't declared)
    have no peer_spki_hash. The cache slot exists but is empty;
    expected_spki returns None so the caller's refuse-unpinned policy
    fires."""
    r = _refresh_with([
        {"node_id": "alpha", "public_key_hex": "ab" * 32,
         "address": "alpha-host:5500"},  # no SPKI field
    ])
    assert r.expected_spki_for_node_id("alpha") is None
    # Still in the SSRF allowlist — pinning is independent of
    # registration.
    assert r.is_registered_address("alpha-host:5500") is True


def test_3_1_resolver_rejects_malformed_spki():
    """Defensive parse on the worker side too. A pillar that drifted
    from the canonical-hex contract (or an attacker-influenced /peers
    projection) can't poison the worker's pin cache with garbage."""
    r = _refresh_with([
        {"node_id": "good", "public_key_hex": "ab" * 32,
         "address": "good:5500", "peer_spki_hash": VALID_SPKI_A},
        {"node_id": "short", "public_key_hex": "cd" * 32,
         "address": "short:5500", "peer_spki_hash": SHORT_SPKI},
        {"node_id": "non_hex", "public_key_hex": "ef" * 32,
         "address": "nonhex:5500", "peer_spki_hash": NON_HEX_SPKI},
        {"node_id": "null_spki", "public_key_hex": "01" * 32,
         "address": "null:5500", "peer_spki_hash": None},
    ])
    assert r.expected_spki_for_node_id("good") == VALID_SPKI_A
    # Malformed values silently collapse to None (= unpinned).
    assert r.expected_spki_for_node_id("short") is None
    assert r.expected_spki_for_node_id("non_hex") is None
    assert r.expected_spki_for_node_id("null_spki") is None


def test_3_1_resolver_spki_normalised_to_lowercase():
    """The pillar canonicalises to lowercase, but defense-in-depth on
    the worker side too — if a future projection ever returns mixed
    case, the worker's hex-compare doesn't false-mismatch."""
    upper_hash = "ABCD" * 16
    r = _refresh_with([
        {"node_id": "alpha", "public_key_hex": "ab" * 32,
         "address": "alpha:5500", "peer_spki_hash": upper_hash},
    ])
    assert r.expected_spki_for_node_id("alpha") == upper_hash.lower()


# ── 3.2 — expected_spki(address) ─────────────────────────────────────


def test_3_2_expected_spki_by_address():
    r = _refresh_with([
        {"node_id": "alpha", "public_key_hex": "ab" * 32,
         "address": "alpha-host:5500", "peer_spki_hash": VALID_SPKI_A},
    ])
    assert r.expected_spki("alpha-host:5500") == VALID_SPKI_A


def test_3_2_expected_spki_unknown_address_returns_none():
    """Address not in the pillar roster. Caller's is_registered_address
    pre-check would catch this; expected_spki returning None is
    defense-in-depth so a single accidentally-skipped pre-check still
    refuses outbound."""
    r = _refresh_with([
        {"node_id": "alpha", "public_key_hex": "ab" * 32,
         "address": "alpha-host:5500", "peer_spki_hash": VALID_SPKI_A},
    ])
    assert r.expected_spki("not-in-roster:5500") is None


def test_3_2_expected_spki_for_address_with_empty_spki_returns_none():
    """The peer at this address is registered but hasn't declared a
    hash — same return as unknown-address. The caller can't distinguish
    by the return; both are 'refuse if unpinned-policy is true'."""
    r = _refresh_with([
        {"node_id": "alpha", "public_key_hex": "ab" * 32,
         "address": "alpha-host:5500"},  # no SPKI
    ])
    assert r.expected_spki("alpha-host:5500") is None
    # is_registered_address still True — needed for the caller's
    # refuse-unpinned decision (we treat "unknown address" and
    # "unpinned peer" as same return value but they're different
    # operator-facing reasons).
    assert r.is_registered_address("alpha-host:5500") is True


def test_3_2_expected_spki_address_mapping_survives_replace():
    """Heartbeat updates can replace the address for a node_id. The
    reverse index must follow — old address shouldn't keep resolving
    to the old SPKI after the peer moved."""
    r = ga.PillarPeerKeyResolver("http://pillar:7777")
    with patch.object(ga, "urlrequest") as ur:
        ur.Request = MagicMock()
        ur.urlopen.return_value = _mock_peers_response([
            {"node_id": "alpha", "public_key_hex": "ab" * 32,
             "address": "old-host:5500",
             "peer_spki_hash": VALID_SPKI_A},
        ])
        r.refresh_once()
        assert r.expected_spki("old-host:5500") == VALID_SPKI_A
        # Pillar's next /peers shows the peer at a new address.
        ur.urlopen.return_value = _mock_peers_response([
            {"node_id": "alpha", "public_key_hex": "ab" * 32,
             "address": "new-host:5500",
             "peer_spki_hash": VALID_SPKI_A},
        ])
        r.refresh_once()
        assert r.expected_spki("new-host:5500") == VALID_SPKI_A
        assert r.expected_spki("old-host:5500") is None


# ── 3.6 — stale-cache refuse on expected_spki ────────────────────────


def test_3_6_expected_spki_returns_none_when_stale():
    """A resolver whose cache is older than stale_cache_deadline_s
    refuses to authenticate the SPKI lookup — same shape as resolve()
    and is_registered_address. Refuse-beats-stale is the unified
    Phase B/Phase 3 contract."""
    r = ga.PillarPeerKeyResolver(
        "http://pillar:7777",
        stale_cache_deadline_s=0.01,  # 10ms — easy to step past in a test
    )
    with patch.object(ga, "urlrequest") as ur:
        ur.urlopen.return_value = _mock_peers_response([
            {"node_id": "alpha", "public_key_hex": "ab" * 32,
             "address": "alpha:5500", "peer_spki_hash": VALID_SPKI_A},
        ])
        ur.Request = MagicMock()
        r.refresh_once()
    assert r.expected_spki("alpha:5500") == VALID_SPKI_A  # fresh
    time.sleep(0.02)
    assert r.expected_spki("alpha:5500") is None  # now stale
    assert r.expected_spki_for_node_id("alpha") is None


def test_3_6_pre_first_refresh_returns_none():
    """No refresh has succeeded yet — the cache is at time 0 which is
    older than any deadline. Worker boot path may construct the
    resolver before the first refresh completes; SPKI lookups in
    that window MUST refuse, not fall through to a default."""
    r = ga.PillarPeerKeyResolver("http://pillar:7777")
    assert r.expected_spki("anything:5500") is None
    assert r.expected_spki_for_node_id("anything") is None


# ── 3.7 — cross-repo worker-side complement ──────────────────────────


def test_3_7_resolver_parses_real_sthambha_peers_projection_shape():
    """The exact JSON shape sthambha's server.py emits in /peers
    (post-Phase-1.5). Two peers — one with SPKI, one without — to
    exercise both arms of the cache. If sthambha drifts the field
    name or omits the field entirely, this assertion fails.

    Pinned alongside the sthambha-side Phase 1.8 round-trip test;
    that one runs in the sthambha test suite. This one is the
    nakshatra-side regression guard for the same wire contract.
    """
    # Shape mirrors sthambha/sthambha/server.py:870 (the /peers handler
    # post-Phase-1.5). Trimmed to the fields the resolver actually
    # consumes — the rest is operator-visible noise.
    sthambha_projection_shape = {
        "peers": [
            {
                "node_id": "worker-1",
                "node_type": "compute",
                "address": "worker-1:5530",
                "state": "online",
                "is_online": True,
                "went_offline_at": 0,
                "public_key_hex": "11" * 32,
                "peer_spki_hash": VALID_SPKI_A,
                "last_seen": 1700000000,
                "gpu_info": None,
                "layer_offerings": [],
                "hardware": None,
                "budget": None,
                "cached_files": [],
                "fabric": None,
                "recent_rpc_ms": 0.0,
                "sandbox_compliance": {},
            },
            {
                "node_id": "worker-legacy",
                "node_type": "compute",
                "address": "worker-legacy:5530",
                "state": "online",
                "is_online": True,
                "went_offline_at": 0,
                "public_key_hex": "22" * 32,
                "peer_spki_hash": "",  # pre-Phase-2 worker
                "last_seen": 1700000000,
            },
        ],
        "count": 2,
        "filter": None,
    }
    r = ga.PillarPeerKeyResolver("http://pillar:7777")
    with patch.object(ga, "urlrequest") as ur:
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            sthambha_projection_shape).encode("utf-8")
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda *args: None
        ur.urlopen.return_value = resp
        ur.Request = MagicMock()
        r.refresh_once()
    # The pinned worker resolves; the legacy one returns None.
    assert r.expected_spki("worker-1:5530") == VALID_SPKI_A
    assert r.expected_spki("worker-legacy:5530") is None
    # Both remain in the SSRF allowlist.
    assert r.is_registered_address("worker-1:5530") is True
    assert r.is_registered_address("worker-legacy:5530") is True


# ── stats() reflects the new cache ───────────────────────────────────


def test_stats_includes_spki_pin_counts():
    """Operators reading /healthz need a quick signal for the
    Phase 2 → Phase 3 rollout: how many peers in the roster have
    declared a hash vs. how many haven't."""
    r = _refresh_with([
        {"node_id": "pinned-1", "public_key_hex": "ab" * 32,
         "address": "p1:5500", "peer_spki_hash": VALID_SPKI_A},
        {"node_id": "pinned-2", "public_key_hex": "cd" * 32,
         "address": "p2:5500", "peer_spki_hash": VALID_SPKI_B},
        {"node_id": "unpinned", "public_key_hex": "ef" * 32,
         "address": "u1:5500"},
    ])
    s = r.stats()
    assert s["cached_spki_pins"] == 2
    assert s["cached_spki_total"] == 3


# ── 3.3 + 3.5 — WorkerServicer.open_outbound_channel ─────────────────


import worker  # noqa: E402  — at the bottom so the resolver tests
                # above don't pull in the heavy worker.py imports
                # unless needed.


class _StubDaemon:
    """Smallest viable DaemonClient stand-in for WorkerServicer
    construction. Real DaemonClient spawns a subprocess; we only need
    info() to return n_embd for the constructor."""

    def info(self):
        return {"n_embd": 4096, "n_layers": 64, "gpu_offload_status": {}}

    def gpu_offload_status(self):
        return {"uses_gpu": False, "n_offloaded": 0,
                "total_layers": 64, "backend_hints": []}


def _build_servicer(*, peer_resolver=None,
                    refuse_unpinned_peers=True):
    return worker.WorkerServicer(
        daemon=_StubDaemon(), mode="middle", layer_start=0, layer_end=8,
        model_id="test-model", idem_max_entries=8, idem_ttl_seconds=10.0,
        peer_resolver=peer_resolver,
        auth_required=False,
        refuse_unregistered_peers=False,
        refuse_unpinned_peers=refuse_unpinned_peers,
    )


def test_3_3_open_outbound_channel_refuses_unpinned_by_default(capsys):
    """No resolver attached + refuse_unpinned_peers=True → PinError
    on the unpinned-peer path. Counter increments + audit emit."""
    s = _build_servicer(peer_resolver=None, refuse_unpinned_peers=True)
    import nakshatra_tls as nt
    with pytest.raises(nt.PinError) as exc:
        s._open_outbound_channel("any-peer:5500")
    assert exc.value.reason == "unpinned_peer"
    assert s._spki_unpinned_refusals == 1


def test_3_3_open_outbound_channel_falls_through_when_policy_off():
    """refuse_unpinned_peers=False → no pin check; legacy insecure
    channel returned. Counter does NOT increment (no refusal)."""
    s = _build_servicer(peer_resolver=None, refuse_unpinned_peers=False)
    import grpc
    ch = s._open_outbound_channel("any-peer:5500")
    assert isinstance(ch, grpc.Channel)
    assert s._spki_unpinned_refusals == 0
    ch.close()


def test_3_3_open_outbound_channel_uses_resolver_lookup():
    """With a resolver attached and a peer_spki_hash on file, the
    channel-open path calls open_pinned_channel with the matching
    expected hash. Verify the wiring by mocking open_pinned_channel."""
    resolver = _refresh_with([
        {"node_id": "alpha", "public_key_hex": "ab" * 32,
         "address": "alpha-host:5500", "peer_spki_hash": VALID_SPKI_A},
    ])
    s = _build_servicer(peer_resolver=resolver, refuse_unpinned_peers=True)
    import nakshatra_tls as nt
    seen = {}

    def fake_open(address, expected_spki, *, refuse_unpinned,
                  probe_timeout_s=5.0):
        seen["address"] = address
        seen["expected"] = expected_spki
        seen["refuse_unpinned"] = refuse_unpinned
        # Return a sentinel — caller treats it as a channel object.
        return "mock-channel"

    with patch.object(nt, "open_pinned_channel", side_effect=fake_open):
        result = s._open_outbound_channel("alpha-host:5500")
    assert result == "mock-channel"
    assert seen == {
        "address": "alpha-host:5500",
        "expected": VALID_SPKI_A,
        "refuse_unpinned": True,
    }


def test_3_5_spki_mismatch_emits_audit_and_counts():
    """On spki_mismatch the worker:
      - increments _spki_mismatch_refusals
      - emits an spki_pin_mismatch audit event with both hashes
      - propagates the PinError so the caller emits push_failed.
    """
    resolver = _refresh_with([
        {"node_id": "alpha", "public_key_hex": "ab" * 32,
         "address": "alpha-host:5500", "peer_spki_hash": VALID_SPKI_A},
    ])
    s = _build_servicer(peer_resolver=resolver, refuse_unpinned_peers=True)
    import nakshatra_tls as nt

    def fake_open(address, expected_spki, **kwargs):
        raise nt.PinError(
            "spki_mismatch",
            address=address,
            expected=expected_spki,
            actual=VALID_SPKI_B,
        )

    audited = []

    def fake_audit(event, **payload):
        audited.append((event, payload))

    with patch.object(nt, "open_pinned_channel", side_effect=fake_open):
        with patch.object(worker, "_audit", side_effect=fake_audit):
            with pytest.raises(nt.PinError):
                s._open_outbound_channel("alpha-host:5500")
    assert s._spki_mismatch_refusals == 1
    # Exactly one audit event for the operator forensics path.
    mismatch_events = [a for a in audited if a[0] == "spki_pin_mismatch"]
    assert len(mismatch_events) == 1
    payload = mismatch_events[0][1]
    assert payload["peer"] == "alpha-host:5500"
    assert payload["expected"] == VALID_SPKI_A
    assert payload["actual"] == VALID_SPKI_B


def test_3_3_probe_failed_increments_counter_and_audits():
    """A peer that's down or refuses TLS surfaces as probe_failed;
    the worker counts it separately so operators can distinguish
    'peer unreachable' from 'peer has wrong cert'."""
    resolver = _refresh_with([
        {"node_id": "alpha", "public_key_hex": "ab" * 32,
         "address": "alpha-host:5500", "peer_spki_hash": VALID_SPKI_A},
    ])
    s = _build_servicer(peer_resolver=resolver, refuse_unpinned_peers=True)
    import nakshatra_tls as nt

    def fake_open(address, expected_spki, **kwargs):
        raise nt.PinError("probe_failed",
                          address=address, error="connection refused")

    audited = []
    with patch.object(nt, "open_pinned_channel", side_effect=fake_open):
        with patch.object(worker, "_audit",
                          side_effect=lambda e, **p: audited.append((e, p))):
            with pytest.raises(nt.PinError):
                s._open_outbound_channel("alpha-host:5500")
    assert s._spki_probe_failures == 1
    assert any(a[0] == "spki_probe_failed" for a in audited)


def test_auth_stats_surfaces_spki_counters():
    """Operators reading /healthz see the three SPKI counters alongside
    auth + ssrf counters. Each represents a different attacker class —
    the breakdown is operator-actionable."""
    s = _build_servicer(refuse_unpinned_peers=True)
    s._spki_unpinned_refusals = 3
    s._spki_mismatch_refusals = 1
    s._spki_probe_failures = 7
    stats = s.auth_stats()
    assert stats["refuse_unpinned_peers"] is True
    assert stats["spki_unpinned_refusals"] == 3
    assert stats["spki_mismatch_refusals"] == 1
    assert stats["spki_probe_failures"] == 7
