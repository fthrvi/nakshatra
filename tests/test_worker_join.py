

# ── capability floor: a node needs an accelerator (GPU VRAM / unified memory) ──────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "scripts", "fabric"))
from worker_join import eligible_workers as _elig, WorkerStanding as _WS  # noqa: E402

def _w(nid, tier, kind=None, vram_mb=None):
    cap = {}
    if kind: cap["kind"] = kind
    if vram_mb is not None: cap["vram_mb"] = vram_mb
    return _WS(node_id=nid, pubkey=nid, admitted=True, tier=tier, tenant=None, capabilities=cap)

def test_capability_floor_rejects_cpu_only_node():
    rank = {"stranger":0,"known":1,"trusted":2,"self":3}
    ws = [_w("gpu", "known", kind="gpu-worker", vram_mb=16000),
          _w("mac", "known", kind="gpu-worker", vram_mb=24000),   # Apple unified memory = valid node
          _w("cpu", "known", kind="cpu-worker")]                  # no accelerator → rejected
    elig, rej = _elig("public-model", ws, min_tier_fn=lambda m:"known", rank=rank)
    ids = {w.node_id for w in elig}
    assert ids == {"gpu","mac"}                                    # both accelerator nodes in
    assert any(r["worker"]=="cpu" and "accelerator" in r["reason"] for r in rej)

def test_min_vram_floor_rejects_tiny_card():
    rank = {"stranger":0,"known":1,"trusted":2,"self":3}
    ws = [_w("big", "known", kind="gpu-worker", vram_mb=16000),
          _w("tiny","known", kind="gpu-worker", vram_mb=2000)]
    elig, rej = _elig("m", ws, min_tier_fn=lambda m:"known", rank=rank, min_vram_mb=8000)
    assert {w.node_id for w in elig} == {"big"}
    assert any(r["worker"]=="tiny" for r in rej)

def test_back_compat_worker_without_capabilities_passes_floor():
    rank = {"stranger":0,"known":1,"trusted":2,"self":3}
    ws = [_w("legacy", "known")]                                   # no capabilities dict → not rejected
    elig, _ = _elig("m", ws, min_tier_fn=lambda m:"known", rank=rank)
    assert {w.node_id for w in elig} == {"legacy"}
