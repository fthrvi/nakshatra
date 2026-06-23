"""
serve_chain — wire the firewall-gated planner + dynamic slicing INTO the serve path.

Slices 1+2 gave us `serve_planner.plan_chain` (roster → eligible_workers() firewall → layer
assignment → chain YAML) and `PackageSlicer` (assemble any range from a content-addressed package).
But `nakshatra_serve.py` still drove a STATIC hand-authored `chain_yaml`. This is the bridge: a single
call the serve makes at request time to GENERATE the chain from the live roster.

    build_chain_from_roster(model_id, hidden_size=…) → a chain YAML path client.py consumes,
        whose workers are the firewall-eligible rostered peers and whose slices materialise from
        the model's content-addressed package.

So the chain is DERIVED (who's admitted+eligible right now, sliced to match) instead of frozen. The
live model can stay on its static path; a model entry opts in with `from_roster: true`.

Injectable (roster_loader / slicer_factory / planner) so it tests without a control plane, a package,
or a GPU.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))         # fabric/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # scripts/
import serve_planner as sp
import package_slicer as ps


def _estimate_model_gb(slicer) -> Optional[float]:
    """Best-effort total model size in GB from the package manifest's artifact byte sizes.
    None if it can't be determined (caller then needs an operator-declared size, else skips
    smart placement). Never raises."""
    try:
        if hasattr(slicer, "_ensure_manifest"):
            slicer._ensure_manifest()
        arts = (getattr(slicer, "artifacts", None)
                or getattr(getattr(slicer, "manifest", None), "artifacts", None)
                or getattr(getattr(slicer, "package", None), "artifacts", None))
        if arts:
            total = sum(int(getattr(a, "size", 0) or 0) for a in arts)
            return round(total / 1e9, 2) if total else None
    except Exception:
        pass
    return None


def build_chain_from_roster(model_id: str, *, hidden_size: int,
                            num_layers: Optional[int] = None, wire_dtype: str = "f32",
                            package_location: Optional[str] = None,
                            registry_path: Optional[str] = None,
                            out_path: Optional[str] = None, cache_dir: Optional[str] = None,
                            require_signature: bool = False, trusted_pubkeys: Optional[set] = None,
                            roster_loader: Optional[Callable] = None,
                            slicer_factory: Optional[Callable] = None,
                            planner: Optional[Callable] = None,
                            min_tier_fn: Optional[Callable] = None,
                            rank: Optional[dict] = None,
                            model_size_gb: Optional[float] = None,
                            pillar_url: Optional[str] = None,
                            peers_fetcher: Optional[Callable] = None) -> str:
    """Generate a client.py chain YAML for `model_id` from the live roster, gated by the identity
    firewall, with slices provisioned from the model's content-addressed package. Returns the YAML
    path. Raises if no rostered worker is firewall-eligible (default-deny — the serve then 502s
    rather than serve a sensitive model on an ineligible peer)."""
    import yaml

    # 1) resolve the model's package (explicit arg wins, else the model→package registry).
    location = package_location or ps.resolve_package_location(model_id, registry_path=registry_path)
    if not location:
        raise FileNotFoundError(
            f"no package location for '{model_id}' — set it in ~/.nakshatra/packages.yaml "
            f"or pass package_location (from_roster needs a content-addressed package to slice)")

    # 2) the slicer (assembles the planner's assigned range on demand, content-addressed).
    if slicer_factory is not None:
        slicer = slicer_factory(location)
    else:
        slicer = ps.PackageSlicer(location, cache_dir=cache_dir,
                                  require_signature=require_signature, trusted_pubkeys=trusted_pubkeys)
    if num_layers is None:                       # the manifest knows the true layer count
        if hasattr(slicer, "_ensure_manifest"):
            slicer._ensure_manifest()
        num_layers = getattr(slicer, "n_layers", None)
    if not num_layers:
        raise ValueError("num_layers unknown — pass it or ensure the package manifest carries n_layers")

    # 3) the rostered workers → firewall → contiguous assignment → chain.
    # FEWER-HOP placement (WAN win): NAKSHATRA_MAX_STAGES caps the chain to N workers (each holds
    # more layers, fewer latency-bound round-trips). Unset = use all eligible.
    import os as _os
    try:
        _max_stages = int(_os.environ.get("NAKSHATRA_MAX_STAGES", "") or 0) or None
    except ValueError:
        _max_stages = None
    standings = sp.standings_from_roster(roster_loader=roster_loader)
    plan_fn = planner or sp.plan_chain
    _kw = dict(num_layers=num_layers, hidden_size=hidden_size, wire_dtype=wire_dtype,
               slice_for=slicer.slice_for, min_tier_fn=min_tier_fn, rank=rank)
    if _max_stages and plan_fn is sp.plan_chain:
        _kw["max_stages"] = _max_stages
    # SMART PLACEMENT (NKS_SMART_PLACEMENT): capacity-aware placement on MEASURED data — route the
    # whole model to the fastest node that fits (0 inter-worker hops, the route-don't-split win),
    # else a balanced split. Needs the model size (operator-declared model_size_gb, else estimated
    # from the package) + live pillar telemetry (recent_rpc_ms / VRAM offered). ANY gap → place_fn
    # returns None → plan_chain falls back to the even split. Fail-open; flag default off.
    if (_os.environ.get("NKS_SMART_PLACEMENT", "").strip().lower() in ("1", "true", "yes")
            and plan_fn is sp.plan_chain):
        try:
            import placement_feed as _pf
            _gb = model_size_gb or _estimate_model_gb(slicer)
            _purl = (pillar_url or _os.environ.get("NAKSHATRA_PILLAR_URL")
                     or _os.environ.get("NAKSHATRA_LIFECYCLE_PILLAR_URL"))
            _peers = (peers_fetcher or _pf.fetch_pillar_peers)(_purl, model_id) if _purl else []
            if _gb and _peers:
                _telem = _pf.telemetry_from_peers(_peers, model_id)
                _kw["place_fn"] = _pf.make_place_fn(
                    model_gb=_gb, telemetry_of=lambda w: _telem.get(w.node_id, {}))
        except Exception:
            pass   # fail-open → even split; placement must never break the serve
    plan = plan_fn(model_id, standings, **_kw)

    # 4) write the generated chain where the serve points client.py.
    dest = Path(out_path) if out_path else (Path(cache_dir or (Path.home() / ".nakshatra" / "slices"))
                                            / f"{model_id}.from-roster.chain.yaml")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(plan.chain, sort_keys=False))
    return str(dest)
