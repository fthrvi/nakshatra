"""
Tests for the unconscious lifecycle (scripts/fabric/unconscious_lifecycle.py). Pure — no GPU.

decide_placement gates THROUGH the real identity firewall (worker_join.eligible_workers), so these
drive it with an injected min_tier_fn/rank (no admission control plane needed) and prove: the 3
states, that the firewall agreement holds (foreign/cpu excluded, self admitted), the legacy
no-capabilities roster stays READY (can't demote the live service), and the toggle/status I/O.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "fabric"))
from worker_join import WorkerStanding  # noqa: E402
import unconscious_lifecycle as ul  # noqa: E402

# the admission tier ladder, supplied so eligible_workers runs without a control plane
RANK = {"stranger": 0, "known": 1, "trusted": 2, "self": 3}
SELF_FLOOR = lambda _m: "self"          # the unconscious model resolves to a self min-tier
MODEL = "prithvi-unconscious"


def _decide(enabled, standings, *, min_tier="self", min_vram_mb=0):
    return ul.decide_placement(enabled, standings, model_id=MODEL,
                               min_tier_fn=lambda _m: min_tier, rank=RANK, min_vram_mb=min_vram_mb)


def _node(node_id, tier="self", kind="gpu-worker", vram_mb=16000, admitted=True):
    caps = {"address": "127.0.0.1", "port": 5540}
    if kind is not None:
        caps["kind"] = kind
    if vram_mb is not None:
        caps["vram_mb"] = vram_mb
    return WorkerStanding(node_id=node_id, pubkey=node_id + "pk", admitted=admitted,
                          tier=tier, tenant=None, capabilities=caps)


def test_disabled_when_toggle_off():
    p = _decide(False, [_node("a")])
    assert p.state == ul.UnconsciousState.DISABLED
    assert p.nodes == [] and not p.is_ready


def test_idle_when_no_nodes():
    p = _decide(True, [])
    assert p.state == ul.UnconsciousState.IDLE_NO_HARDWARE
    assert p.nodes == []


def test_ready_with_owned_accelerators():
    nodes = [_node("gpu-a", vram_mb=16000), _node("gpu-b", vram_mb=16000)]
    p = _decide(True, nodes)
    assert p.state == ul.UnconsciousState.READY
    assert {w.node_id for w in p.nodes} == {"gpu-a", "gpu-b"}
    assert p.total_vram_mb == 32000          # advisory


def test_foreign_nodes_excluded_by_firewall():
    # a stranger's GPU must NEVER host the unconscious — the firewall's tier gate drops it.
    nodes = [_node("stranger-gpu", tier="stranger", vram_mb=24000)]
    p = _decide(True, nodes)
    assert p.state == ul.UnconsciousState.IDLE_NO_HARDWARE
    assert p.nodes == []
    assert any("stranger-gpu" == r.get("worker") for r in p.rejected)   # reason surfaced for the UI


def test_cpu_only_owned_node_rejected_by_floor():
    p = _decide(True, [_node("cpu-box", kind="cpu-worker", vram_mb=0)])
    assert p.state == ul.UnconsciousState.IDLE_NO_HARDWARE


def test_legacy_node_without_kind_stays_ready():
    # the LIVE roster's standings carry no `kind` — the firewall's floor passes them (back-compat),
    # so enabling the lifecycle must NOT demote the running service to idle.
    legacy = WorkerStanding(node_id="live-slot", pubkey="pk", admitted=True, tier="self",
                            tenant=None, capabilities={"address": "127.0.0.1", "port": 5540})
    p = _decide(True, [legacy])
    assert p.state == ul.UnconsciousState.READY
    assert p.nodes[0].node_id == "live-slot"


def test_agreement_with_firewall_on_mixed_pool():
    # self GPU + trusted GPU + stranger GPU + self CPU → only the self GPU is eligible.
    nodes = [_node("self-gpu", tier="self"), _node("trusted-gpu", tier="trusted"),
             _node("stranger-gpu", tier="stranger"), _node("self-cpu", kind="cpu-worker", vram_mb=0)]
    p = _decide(True, nodes)
    assert p.state == ul.UnconsciousState.READY
    assert [w.node_id for w in p.nodes] == ["self-gpu"]


def test_per_node_vram_floor_routes_through_firewall():
    # min_vram_mb is per-node (enforced inside eligible_workers), not a sum gate.
    small = _node("small-gpu", vram_mb=4000)
    big = _node("big-gpu", vram_mb=16000)
    p = _decide(True, [small, big], min_vram_mb=8000)
    assert p.state == ul.UnconsciousState.READY
    assert [w.node_id for w in p.nodes] == ["big-gpu"]    # small dropped by the per-node floor


def test_injected_eligible_fn_isolates_state_logic():
    # state machine in isolation, firewall stubbed
    n = _node("x")
    ready = ul.decide_placement(True, [n], model_id=MODEL, eligible_fn=lambda m, ws: ([n], []))
    idle = ul.decide_placement(True, [n], model_id=MODEL,
                               eligible_fn=lambda m, ws: ([], [{"worker": "x", "reason": "nope"}]))
    assert ready.state == ul.UnconsciousState.READY
    assert idle.state == ul.UnconsciousState.IDLE_NO_HARDWARE and idle.rejected[0]["reason"] == "nope"


def test_toggle_roundtrip_and_absent_default(tmp_path):
    flag = tmp_path / "unconscious.enabled"
    assert ul.read_enabled(flag) is True            # absent → default True (live unaffected)
    assert ul.read_enabled(flag, default=False) is False
    ul.set_enabled(flag, False)
    assert ul.read_enabled(flag) is False
    ul.set_enabled(flag, True)
    assert ul.read_enabled(flag) is True


def test_write_status_shape(tmp_path):
    nodes = [_node("gpu-a", vram_mb=16000)]
    p = _decide(True, nodes)
    doc = ul.write_status(tmp_path / "status.json", p, model_id=MODEL, min_tier="self", now=123.0)
    assert doc["state"] == "ready" and doc["min_tier"] == "self"
    assert doc["nodes"][0]["id"] == "gpu-a" and doc["nodes"][0]["vram_mb"] == 16000
    assert doc["nodes"][0]["kind"] == "gpu-worker"
    assert doc["updated"] == 123.0
    import json
    reread = json.loads((tmp_path / "status.json").read_text())
    assert reread["total_vram_mb"] == 16000


def test_write_status_idle_carries_rejected(tmp_path):
    p = _decide(True, [_node("stranger-gpu", tier="stranger")])
    doc = ul.write_status(tmp_path / "s.json", p, model_id=MODEL, min_tier="self")
    assert doc["state"] == "idle"
    assert doc["rejected"] and doc["rejected"][0]["worker"] == "stranger-gpu"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
