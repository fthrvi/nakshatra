"""Phase E tests for ``scripts/fabric/link_stats.py`` — periodic
counter snapshots to ``POST /fabric/link_stats``.

Covers the falsifiable checks from the sprint plan Phase E:

  * Snapshot shape: every per-link entry carries exactly the
    pillar-recognised counter names from schema §9, byte-identical
    to ``sthambha/core.py:LinkStatsSnapshot``
  * Counter values pulled from each registered FabricLink's ``.counters``
  * Private debug counters (recv_dropped_magic etc.) NOT in the snapshot
  * RTT fields default to 0 (Phase F wiring) — not omitted
  * Reporter silently no-ops when no pillar URL is configured
  * NAKSHATRA_FABRIC_LINK_STATS_INTERVAL_S env override (typo-safe)
  * Truncation at MAX_LINKS_PER_PUSH (sorted-stable, not random)
  * Empty registry returns None instead of a zero-link push
  * POST signed under OWNER tier with worker priv key
  * Failure-soft: HTTP errors counted, not raised; loop continues
  * register / unregister / set_plan_id ergonomics
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib import error as urlerror

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fabric import link_stats as ls  # noqa: E402


PRIV = b"\x07" * 32                        # any 32 bytes
PILLAR = "http://pillar:7777"
SELF_ID = "worker-a"


# ── helpers ──────────────────────────────────────────────────────────


class _FakeLink:
    """Stand-in for FabricLink — only ``.counters`` is read by the
    reporter. Keeps these tests focused on the link_stats logic
    without needing real UDP sockets."""
    def __init__(self, **counters):
        # Default to all-zero schema counters so most tests don't have
        # to spell out every field.
        self.counters = {name: 0 for name in ls.SCHEMA_COUNTER_NAMES}
        self.counters.update(counters)


def _mock_urlopen(body_dict: dict | None = None, status: int = 200):
    cm = MagicMock()
    payload = json.dumps(body_dict or {"stored": 1}).encode()
    cm.__enter__.return_value = io.BytesIO(payload)
    cm.__exit__.return_value = False
    return cm


def _http_error(code: int, body: str = "") -> urlerror.HTTPError:
    return urlerror.HTTPError(
        url=f"{PILLAR}/fabric/link_stats", code=code,
        msg=f"HTTP {code}", hdrs=None,
        fp=io.BytesIO(body.encode()),
    )


def _reporter(**overrides) -> ls.LinkStatsReporter:
    kwargs = dict(
        pillar_url=PILLAR, peer_id=SELF_ID, priv_key=PRIV,
        plan_id="plan-001",
    )
    kwargs.update(overrides)
    return ls.LinkStatsReporter(**kwargs)


# ── 1. _interval_from_env ────────────────────────────────────────────


def test_interval_default(monkeypatch):
    monkeypatch.delenv(ls.ENV_INTERVAL, raising=False)
    assert ls._interval_from_env() == ls.DEFAULT_INTERVAL_S


def test_interval_env_override(monkeypatch):
    monkeypatch.setenv(ls.ENV_INTERVAL, "10")
    assert ls._interval_from_env() == 10.0


def test_interval_env_typo_safe(monkeypatch):
    """Bad / non-positive values fall back rather than disable
    reporting silently. Same typo-safe stance as
    nakshatra_tls._probe_timeout_from_env."""
    for bad in ("", "   ", "abc", "30s", "-5", "0"):
        monkeypatch.setenv(ls.ENV_INTERVAL, bad)
        assert ls._interval_from_env() == ls.DEFAULT_INTERVAL_S


# ── 2. Constructor guard ─────────────────────────────────────────────


def test_constructor_rejects_short_priv_key():
    """Ed25519 priv keys are 32 bytes — same shape as JoinClient."""
    with pytest.raises(ValueError, match="32 bytes"):
        ls.LinkStatsReporter(
            pillar_url=PILLAR, peer_id=SELF_ID, priv_key=b"short")


def test_constructor_accepts_empty_pillar_with_any_key():
    """The silent-no-op path: an empty pillar URL means no /post will
    ever happen, so the priv_key isn't load-bearing. Accept any value
    so legacy bringups + localhost smokes don't need to fabricate a
    32-byte key just to instantiate the class."""
    r = ls.LinkStatsReporter(pillar_url="", peer_id=SELF_ID,
                              priv_key=b"")
    assert r._pillar_url == ""
    # And start() is a no-op — no thread spawned.
    r.start()
    assert r._thread is None


# ── 3. Snapshot shape ────────────────────────────────────────────────


def test_snapshot_returns_none_when_no_links():
    r = _reporter()
    assert r.snapshot() is None


def test_snapshot_includes_peer_id_and_plan_id():
    r = _reporter(plan_id="plan-XYZ")
    r.register("worker-b", _FakeLink(sent_packets=5))
    snap = r.snapshot()
    assert snap is not None
    assert snap["peer_id"] == SELF_ID
    assert snap["plan_id"] == "plan-XYZ"


def test_snapshot_omits_plan_id_when_empty():
    """Per pillar server.py:1636: plan_id is optional ("")."""
    r = _reporter(plan_id="")
    r.register("worker-b", _FakeLink())
    snap = r.snapshot()
    assert "plan_id" not in snap


def test_snapshot_entry_carries_schema_counter_names_byte_exact():
    """The pillar's record_link_snapshots reads these exact keys
    (sthambha/core.py:1899-1905). A drift here breaks the cross-repo
    wire contract on every push."""
    r = _reporter()
    r.register("worker-b", _FakeLink(
        sent_packets=10, sent_bytes=1024, recv_packets=8, recv_bytes=900,
        recv_auth_fails=1, recv_gaps=2, recv_dropped_alloc=0,
        recv_dropped_dtype=0,
    ))
    snap = r.snapshot()
    entry = snap["links"][0]
    assert entry["other_peer_id"] == "worker-b"
    # Every schema field is present, every value matches what the
    # link's counter dict held.
    for name in ls.SCHEMA_COUNTER_NAMES:
        assert name in entry, f"missing field: {name}"
    assert entry["sent_packets"] == 10
    assert entry["sent_bytes"] == 1024
    assert entry["recv_packets"] == 8
    assert entry["recv_auth_fails"] == 1
    assert entry["recv_gaps"] == 2


def test_snapshot_rtt_defaults_to_zero():
    """Phase F wiring — FabricLink doesn't measure RTT yet; the
    reporter emits 0 so the pillar's snapshot dataclass sees 'not
    applicable' rather than a missing field that triggers a parse
    fallback."""
    r = _reporter()
    r.register("worker-b", _FakeLink())
    snap = r.snapshot()
    entry = snap["links"][0]
    assert entry["rtt_ns_p50"] == 0
    assert entry["rtt_ns_p99"] == 0


def test_snapshot_excludes_private_counters():
    """FabricLink tracks recv_dropped_magic + recv_dropped_version +
    recv_dropped_truncated for operator debug — but these aren't in
    schema §9 and the pillar's parser wouldn't recognise them. They
    must NOT appear in the snapshot."""
    link = _FakeLink()
    link.counters["recv_dropped_magic"] = 99
    link.counters["recv_dropped_version"] = 7
    link.counters["recv_dropped_truncated"] = 3
    r = _reporter()
    r.register("worker-b", link)
    entry = r.snapshot()["links"][0]
    assert "recv_dropped_magic" not in entry
    assert "recv_dropped_version" not in entry
    assert "recv_dropped_truncated" not in entry


def test_snapshot_multiple_links_sorted_by_peer_id():
    """Sorting is a deterministic-output convenience for operators
    eyeballing the pillar's /fabric/links projection — same link
    ordering across ticks reduces visual churn."""
    r = _reporter()
    r.register("worker-z", _FakeLink(sent_packets=2))
    r.register("worker-a", _FakeLink(sent_packets=1))
    r.register("worker-m", _FakeLink(sent_packets=3))
    snap = r.snapshot()
    ids = [e["other_peer_id"] for e in snap["links"]]
    assert ids == ["worker-a", "worker-m", "worker-z"]


def test_snapshot_truncates_at_max_links_per_push():
    """Pillar enforces 256 (sthambha/core.py:MAX_LINK_STATS_PER_PUSH);
    reporter caps at the same value rather than ship a payload that
    would 400. Sorted-stable truncation: same first-N every tick."""
    r = _reporter()
    for i in range(ls.MAX_LINKS_PER_PUSH + 5):
        r.register(f"peer-{i:04d}", _FakeLink())
    snap = r.snapshot()
    assert len(snap["links"]) == ls.MAX_LINKS_PER_PUSH


# ── 4. register / unregister / set_plan_id ──────────────────────────


def test_register_rejects_empty_peer_id():
    r = _reporter()
    with pytest.raises(ValueError):
        r.register("", _FakeLink())


def test_unregister_is_idempotent():
    """Boot teardown calls unregister() unconditionally — must not
    raise on an already-removed peer."""
    r = _reporter()
    r.unregister("never-registered")     # no exception
    r.register("worker-b", _FakeLink())
    r.unregister("worker-b")
    r.unregister("worker-b")             # idempotent


def test_register_replaces_existing_link_for_same_peer():
    """Rekey may swap one link for a fresh one against the same peer.
    Reporter takes the new one rather than accumulating duplicates."""
    r = _reporter()
    old = _FakeLink(sent_packets=100)
    new = _FakeLink(sent_packets=0)
    r.register("worker-b", old)
    r.register("worker-b", new)
    snap = r.snapshot()
    assert snap["links"][0]["sent_packets"] == 0  # the new link


def test_set_plan_id_updates_future_snapshots():
    r = _reporter(plan_id="old-plan")
    r.register("worker-b", _FakeLink())
    assert r.snapshot()["plan_id"] == "old-plan"
    r.set_plan_id("new-plan")
    assert r.snapshot()["plan_id"] == "new-plan"


# ── 5. POST shape + signing ─────────────────────────────────────────


def test_post_signs_with_worker_priv_key_and_owner_tier():
    """Per server.py:1629, /fabric/link_stats is OWNER tier — signer
    must equal peer_id. Asserts the Authorization header carries the
    right scheme + keyid, and the body matches the on-wire contract."""
    captured = {}

    def fake_urlopen(req, *a, **kw):
        captured["url"] = req.full_url
        captured["method"] = req.method
        captured["body"] = json.loads(req.data.decode())
        captured["auth"] = req.headers.get("Authorization")
        captured["content_type"] = req.headers.get("Content-type")
        return _mock_urlopen()

    r = _reporter()
    r.register("worker-b", _FakeLink(sent_packets=42))
    with patch.object(ls.urlrequest, "urlopen", side_effect=fake_urlopen):
        r._post(r.snapshot())

    assert captured["url"] == f"{PILLAR}/fabric/link_stats"
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    # Auth scheme is Sthambha-Ed25519 (cross-repo contract).
    assert captured["auth"].startswith("Sthambha-Ed25519")
    assert f'keyid="{SELF_ID}"' in captured["auth"]
    # Body shape matches the pillar contract.
    assert captured["body"]["peer_id"] == SELF_ID
    assert captured["body"]["plan_id"] == "plan-001"
    assert len(captured["body"]["links"]) == 1
    assert captured["body"]["links"][0]["sent_packets"] == 42


# ── 6. Loop lifecycle + failure-soft ────────────────────────────────


def test_start_is_idempotent_and_noops_without_pillar():
    """start() with no pillar URL must not spawn a thread (silent
    no-op path); start() twice with a pillar URL is a no-op on the
    second call (idempotent)."""
    # Silent no-op:
    r_no_pillar = ls.LinkStatsReporter(pillar_url="", peer_id=SELF_ID,
                                         priv_key=b"")
    r_no_pillar.start()
    assert r_no_pillar._thread is None
    # Idempotent:
    r = _reporter(interval_s=10_000)              # loop sleeps forever
    r.register("b", _FakeLink())
    r.start()
    t1 = r._thread
    r.start()
    assert r._thread is t1
    r.stop()


def test_stop_is_idempotent_and_non_blocking():
    r = _reporter()
    r.stop()
    r.stop()        # no exception, no hang


def test_loop_failure_soft_on_pillar_outage():
    """A pillar 500 / network error must increment ``push_failures``,
    NOT raise. Next tick is independent — the loop survives the
    outage and resumes pushing when the pillar recovers."""
    r = _reporter(interval_s=0.05)
    r.register("worker-b", _FakeLink(sent_packets=1))

    call_count = {"n": 0}

    def flaky_urlopen(req, *a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _http_error(500, "internal error")
        return _mock_urlopen()

    with patch.object(ls.urlrequest, "urlopen", side_effect=flaky_urlopen):
        r.start()
        # Wait for at least 2 ticks.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if r.push_failures >= 1 and r.push_count >= 1:
                break
            time.sleep(0.05)
        r.stop()

    assert r.push_failures >= 1
    assert r.push_count >= 1
    # Worker is still alive — no exception propagated out of the loop.


def test_loop_skips_when_no_links_registered():
    """Empty registry → snapshot() returns None → loop skips the POST.
    A zero-link push wastes pillar cycles."""
    r = _reporter(interval_s=0.05)
    called = {"n": 0}

    def counting_urlopen(req, *a, **kw):
        called["n"] += 1
        return _mock_urlopen()

    with patch.object(ls.urlrequest, "urlopen", side_effect=counting_urlopen):
        r.start()
        time.sleep(0.25)
        r.stop()
    assert called["n"] == 0
