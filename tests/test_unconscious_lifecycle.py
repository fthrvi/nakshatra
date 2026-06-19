"""
Tests for the unconscious lifecycle (scripts/fabric/unconscious_lifecycle.py). Pure — no GPU.

Proves the 3 states (disabled / idle-no-hardware / ready), that placement is restricted to OWNED
(self-tier) accelerators, that the legacy no-capabilities roster stays READY (so enabling the
lifecycle can't demote the live service), the aggregate-VRAM floor, and the toggle/status I/O.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "fabric"))
from worker_join import WorkerStanding  # noqa: E402
import unconscious_lifecycle as ul  # noqa: E402


def _node(node_id, tier="self", kind="gpu-worker", vram_mb=16000, admitted=True):
    caps = {"address": "127.0.0.1", "port": 5540}
    if kind is not None:
        caps["kind"] = kind
    if vram_mb is not None:
        caps["vram_mb"] = vram_mb
    return WorkerStanding(node_id=node_id, pubkey=node_id + "pk", admitted=admitted,
                          tier=tier, tenant=None, capabilities=caps)


def test_disabled_when_toggle_off():
    p = ul.decide_placement(False, [_node("a")])
    assert p.state == ul.UnconsciousState.DISABLED
    assert p.nodes == [] and not p.is_ready


def test_idle_when_no_nodes():
    p = ul.decide_placement(True, [])
    assert p.state == ul.UnconsciousState.IDLE_NO_HARDWARE
    assert p.nodes == []


def test_ready_with_owned_accelerators():
    nodes = [_node("gpu-a", vram_mb=16000), _node("gpu-b", vram_mb=16000)]
    p = ul.decide_placement(True, nodes)
    assert p.state == ul.UnconsciousState.READY
    assert {w.node_id for w in p.nodes} == {"gpu-a", "gpu-b"}
    assert p.total_vram_mb == 32000


def test_foreign_nodes_excluded():
    # a stranger's GPU must NEVER host the unconscious — only tier=self is owned.
    nodes = [_node("stranger-gpu", tier="stranger", vram_mb=24000)]
    p = ul.decide_placement(True, nodes)
    assert p.state == ul.UnconsciousState.IDLE_NO_HARDWARE
    assert p.nodes == []


def test_cpu_only_owned_node_is_not_hardware():
    nodes = [_node("cpu-box", kind="cpu-worker", vram_mb=0)]
    p = ul.decide_placement(True, nodes)
    assert p.state == ul.UnconsciousState.IDLE_NO_HARDWARE


def test_legacy_node_without_kind_stays_ready():
    # the LIVE roster's standings carry no `kind` — enabling the lifecycle must keep them usable,
    # not demote the running service to idle.
    legacy = WorkerStanding(node_id="live-slot", pubkey="pk", admitted=True, tier="self",
                            tenant=None, capabilities={"address": "127.0.0.1", "port": 5540})
    p = ul.decide_placement(True, [legacy])
    assert p.state == ul.UnconsciousState.READY
    assert p.nodes[0].node_id == "live-slot"


def test_aggregate_vram_floor():
    # two 4GB owned GPUs = 8GB aggregate; an 8B model needing ~10GB across the chain → idle.
    nodes = [_node("small-a", vram_mb=4000), _node("small-b", vram_mb=4000)]
    idle = ul.decide_placement(True, nodes, min_vram_mb=10000)
    assert idle.state == ul.UnconsciousState.IDLE_NO_HARDWARE
    assert idle.total_vram_mb == 8000
    ready = ul.decide_placement(True, nodes, min_vram_mb=6000)
    assert ready.state == ul.UnconsciousState.READY


def test_require_tier_none_allows_any_admitted_accelerator():
    # caller that already firewalled by tier can pass require_tier=None
    nodes = [_node("trusted-gpu", tier="trusted", vram_mb=16000)]
    assert ul.decide_placement(True, nodes, require_tier="self").state == ul.UnconsciousState.IDLE_NO_HARDWARE
    assert ul.decide_placement(True, nodes, require_tier=None).state == ul.UnconsciousState.READY


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
    p = ul.decide_placement(True, nodes)
    doc = ul.write_status(tmp_path / "status.json", p, model_id="prithvi-unconscious", now=123.0)
    assert doc["state"] == "ready"
    assert doc["model_id"] == "prithvi-unconscious"
    assert doc["nodes"][0]["id"] == "gpu-a" and doc["nodes"][0]["vram_mb"] == 16000
    assert doc["updated"] == 123.0
    # and it actually landed on disk as valid json
    import json
    reread = json.loads((tmp_path / "status.json").read_text())
    assert reread["total_vram_mb"] == 16000


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
