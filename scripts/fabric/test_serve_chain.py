"""serve_chain — the from_roster generator, tested with everything injected (no GPU, no package,
no control plane). Proves: it resolves the package, runs the firewall-gated planner with a
package-backed slicer, writes a chain YAML, and propagates default-deny."""
import sys, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_chain as sc


class _FakeSlicer:
    def __init__(self, location, n_layers=32):
        self.location = location
        self.n_layers = n_layers
        self.sliced = []

    def slice_for(self, w, start, end, model):
        self.sliced.append((start, end))
        return f"/slices/{model}-L{start}-{end}.gguf"


def _roster():
    return {
        "pk-a": {"pubkey": "pk-a", "name": "self-a", "operator": "me", "tier": "self",
                 "tenant": "home", "coord": "127.0.0.1:5540"},
        "pk-b": {"pubkey": "pk-b", "name": "self-b", "operator": "me", "tier": "self",
                 "tenant": "home", "coord": "127.0.0.1:5541"},
        "pk-x": {"pubkey": "pk-x", "name": "opB", "operator": "opB", "tier": "stranger",
                 "tenant": "opB", "coord": "10.50.0.9:5540"},
    }


RANK = {"stranger": 0, "known": 1, "trusted": 2, "self": 3}
MT = lambda m: {"public": "stranger"}.get(m, "self")  # default self-only


def test_generates_firewall_gated_chain():
    captured = {}
    slicer = _FakeSlicer("/pkg")

    def planner(model, workers, **kw):
        captured["model"] = model
        captured["num_layers"] = kw["num_layers"]
        captured["hidden_size"] = kw["hidden_size"]
        captured["slice_for"] = kw["slice_for"]
        # emulate the real firewall: self-only model excludes the stranger
        import serve_planner as sp
        return sp.plan_chain(model, workers, num_layers=kw["num_layers"],
                             hidden_size=kw["hidden_size"], slice_for=kw["slice_for"],
                             min_tier_fn=MT, rank=RANK)

    d = Path(tempfile.mkdtemp())
    try:
        out = sc.build_chain_from_roster(
            "prithvi-private", hidden_size=4096, package_location="/pkg",
            roster_loader=_roster, slicer_factory=lambda loc: slicer,
            planner=planner, out_path=str(d / "gen.yaml"))
        import yaml
        chain = yaml.safe_load(Path(out).read_text())
        ids = [w["id"] for w in chain["workers"]]
        assert ids == ["self-a", "self-b"], "stranger must be firewalled out of a self-only model"
        assert captured["num_layers"] == 32           # pulled from the slicer manifest
        assert chain["model"]["hidden_size"] == 4096
        # slices were requested for the assigned contiguous ranges
        assert slicer.sliced == [(0, 16), (16, 32)]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_default_deny_propagates():
    # roster has only a stranger; a self-only model -> planner raises -> generator propagates
    def planner(model, workers, **kw):
        import serve_planner as sp
        return sp.plan_chain(model, workers, num_layers=kw["num_layers"],
                             hidden_size=kw["hidden_size"], slice_for=kw["slice_for"],
                             min_tier_fn=MT, rank=RANK)
    only_stranger = {"pk-x": {"pubkey": "pk-x", "name": "opB", "operator": "opB",
                              "tier": "stranger", "tenant": "opB", "coord": "10.50.0.9:5540"}}
    try:
        sc.build_chain_from_roster("prithvi-private", hidden_size=4096, package_location="/pkg",
                                   roster_loader=lambda: only_stranger,
                                   slicer_factory=lambda loc: _FakeSlicer("/pkg"), planner=planner)
        assert False, "expected PermissionError"
    except PermissionError:
        pass


def _smart_placement_roster():
    """Two GPU-slot workers on the SAME physical box — exactly today's live unconscious-tier
    shape: both on 127.0.0.1, roster names 'unconscious-a'/'unconscious-b', while the box's
    pillar telemetry is filed under its registrar identity 'hub'."""
    return {
        "pk-a": {"pubkey": "pk-a", "name": "unconscious-a", "operator": "me", "tier": "self",
                 "tenant": "home", "coord": "127.0.0.1:5540"},
        "pk-b": {"pubkey": "pk-b", "name": "unconscious-b", "operator": "me", "tier": "self",
                 "tenant": "home", "coord": "127.0.0.1:5541"},
    }


def _run_smart_placement(d):
    slicer = _FakeSlicer("/pkg")
    peers = [{"node_id": "hub", "budget": {"vram_offered_gb": 9.0}, "recent_rpc_ms": 10.0}]
    out = sc.build_chain_from_roster(
        "prithvi-unconscious", hidden_size=4096, package_location="/pkg",
        roster_loader=_smart_placement_roster, slicer_factory=lambda loc: slicer,
        model_size_gb=8.0, pillar_url="http://fake-pillar",
        peers_fetcher=lambda url, model_id: peers,
        out_path=str(d / "gen.yaml"))
    import yaml
    return yaml.safe_load(Path(out).read_text())


def test_smart_placement_without_host_map_falls_back_to_even_split():
    """Locks in the bug this branch fixes: roster names ('unconscious-a') never match the
    pillar's telemetry key ('hub'), so every node reads 0 vram_gb and placement can't place —
    it falls open to the plain even split, same as if the flag were off."""
    import os
    d = Path(tempfile.mkdtemp())
    had = os.environ.get("NKS_SMART_PLACEMENT")
    os.environ.pop("NKS_NODE_HOST_MAP", None)
    os.environ["NKS_SMART_PLACEMENT"] = "1"
    try:
        chain = _run_smart_placement(d)
        ids = [w["id"] for w in chain["workers"]]
        assert ids == ["unconscious-a", "unconscious-b"], f"expected the even split, got {ids}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
        if had is None:
            os.environ.pop("NKS_SMART_PLACEMENT", None)
        else:
            os.environ["NKS_SMART_PLACEMENT"] = had


def test_smart_placement_with_host_map_routes_whole():
    """With the host map in place, telemetry_of() resolves 'unconscious-a'/'-b' to 'hub', finds
    real VRAM there, and route-whole engages: ONE worker holding the whole model, zero
    inter-worker hops — the fix this branch makes."""
    import os
    d = Path(tempfile.mkdtemp())
    hostmap = d / "node-host-map.tsv"
    hostmap.write_text("unconscious-a\thub\nunconscious-b\thub\n")
    had_flag = os.environ.get("NKS_SMART_PLACEMENT")
    had_map = os.environ.get("NKS_NODE_HOST_MAP")
    os.environ["NKS_SMART_PLACEMENT"] = "1"
    os.environ["NKS_NODE_HOST_MAP"] = str(hostmap)
    try:
        chain = _run_smart_placement(d)
        ids = [w["id"] for w in chain["workers"]]
        assert len(ids) == 1 and ids[0] in ("unconscious-a", "unconscious-b"), \
            f"expected route-whole (1 worker), got {ids}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
        for k, had in (("NKS_SMART_PLACEMENT", had_flag), ("NKS_NODE_HOST_MAP", had_map)):
            if had is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = had


def test_no_package_location_errors():
    try:
        sc.build_chain_from_roster("unregistered-model", hidden_size=4096,
                                   roster_loader=_roster,
                                   registry_path="/nonexistent/packages.yaml")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"all serve_chain tests PASS ({len(fns)})")
