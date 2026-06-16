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
