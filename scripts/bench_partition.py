#!/usr/bin/env python3
"""bench_partition.py — A/B benchmark: greedy vs compute-aware (water-filling)
layer partitioning on a Nakshatra chain.

This is the experiment behind trisul/research/compute-not-the-wire: does sizing
each worker's slice by *compute capacity* (Sthambha PlanConstraints.partition=
"compute") instead of *memory* ("greedy") move real tok/s on the heterogeneous
cluster, off the ~0.21 tok/s 70B baseline?

It builds BOTH plans for the SAME physical cluster, renders each to a client.py
cluster YAML, runs a fixed generation, and reports median steady-state tok/s and
the speedup.

────────────────────────────────────────────────────────────────────────────
USAGE
  # 1. Validate the two assignments without the cluster (planner only):
  python3 scripts/bench_partition.py --cluster bench_cluster.yaml --dry-run

  # 2. Real benchmark (cluster up, workers serving, slices pre-staged):
  python3 scripts/bench_partition.py --cluster bench_cluster.yaml

  # 3. Cut the per-plan sub-GGUF slices first (each mode needs its own ranges):
  python3 scripts/bench_partition.py --cluster bench_cluster.yaml --cut

CLUSTER DESCRIPTION (bench_cluster.yaml) — operator fills with REAL values:
  model:
    id: llama-3.3-70b
    num_layers: 80
    model_bytes: 42000000000      # on-disk bytes of the full Q4_K_M gguf
    hidden_size: 8192             # 70B → 16 KB/token fp16 on the wire
    wire_dtype: f16
    full_gguf_path: /models/llama-3.3-70b/llama-3.3-70b-Q4_K_M.gguf
  hub: home-pc                    # node id that holds the full gguf (cuts slices)
  slice_dir: /tmp/bench-slices
  prompt: "The capital of France is"
  tokens: 32                      # steady-state tokens to time
  repeats: 3                      # runs per mode; median reported
  nodes:
    - {id: home-pc, address: 100.x.x.x, port: 5530, vram_gb: 16, ram_gb: 32,
       rpc_ms: 120, vendor: amd, backend: rocm}
    - {id: node-b,  address: 100.x.x.x, port: 5531, vram_gb: 16, ram_gb: 64,
       rpc_ms: 480, vendor: amd, backend: metal}
    ...
  # rpc_ms = measured per-token round-trip for that node (the compute proxy the
  # water-filling planner sizes by). Get it from a 1-token warmup per node, or
  # from the pillar registry's recent_rpc_ms. Keep this file LOCAL/uncommitted —
  # it carries real addresses.

Steady-state tok/s is isolated from model-load + prompt-prefill by running each
plan at n=1 and n=tokens and reporting (tokens-1)/(T_n - T_1). The tok/s line
client.py prints is also parsed and reported as a cross-check.
"""
from __future__ import annotations

import argparse, os, re, statistics, subprocess, sys, time
from pathlib import Path

import yaml

# Cross-repo import: the planner lives in the sibling sthambha repo.
for cand in (os.environ.get("STHAMBHA_HOME"),
             str(Path.home() / "sthambha"),
             "/Users/bishwa/sthambha"):
    if cand and (Path(cand) / "sthambha" / "planner.py").exists():
        sys.path.insert(0, cand)
        break
try:
    from sthambha.core import (Budget, GpuInfo, Hardware, NodeState, PeerStatus)
    from sthambha.planner import PlanConstraints, plan_split
except ImportError as e:
    sys.exit(f"cannot import sthambha planner ({e}); set STHAMBHA_HOME to the "
             f"sthambha repo root.")

HERE = Path(__file__).resolve().parent
CLIENT = HERE / "client.py"
MODES = ["greedy", "compute"]


def _peer(n: dict) -> PeerStatus:
    """Build a PeerStatus from a node dict in the cluster description."""
    vram, ram = float(n["vram_gb"]), float(n.get("ram_gb", n["vram_gb"]))
    gpu = GpuInfo(
        vendor=n.get("vendor", "amd"),
        model=n.get("model", f"{n.get('vendor','amd')} gpu"),
        vram_total_gb=vram,
        backend=n.get("backend", "rocm"),
        actual_layers_offloaded=999,        # mark as a real GPU offloader
        total_layers_loaded=999,
        chain_status="ok",
    )
    return PeerStatus(
        node_id=n["id"], node_type="compute", last_seen=int(time.time()),
        state=NodeState.ONLINE, address=f"{n['address']}:{n['port']}",
        layer_offerings=[],
        hardware=Hardware(platform="linux", arch="x86_64", cpu_model="bench",
                          cpu_threads=8, ram_total_gb=ram, disk_avail_gb=500.0,
                          gpus=[gpu]),
        budget=Budget(vram_offered_gb=vram, ram_offered_gb=ram,
                      cpu_threads_offered=8, disk_for_cache_gb=200.0),
        cached_files=[],
        recent_rpc_ms=float(n.get("rpc_ms", 0.0)),
    )


def build_plan(cfg: dict, mode: str):
    m = cfg["model"]
    peers = [_peer(n) for n in cfg["nodes"]]
    return plan_split(
        model_id=m["id"], full_gguf_path=m["full_gguf_path"],
        num_layers=int(m["num_layers"]), model_bytes=int(m["model_bytes"]),
        candidate_peers=peers,
        constraints=PlanConstraints(hub_peer_id=cfg["hub"], partition=mode,
                                    max_chain_length=len(peers)),
    )


def slice_path(cfg: dict, mode: str, i: int, s: int, e: int) -> str:
    return str(Path(cfg["slice_dir"]) / f"{mode}_w{i}_L{s}-{e}.gguf")


def render_yaml(cfg: dict, mode: str, plan) -> dict:
    """ChainPlan -> client.py cluster-YAML dict."""
    m = cfg["model"]
    addr = {n["id"]: (n["address"], n["port"]) for n in cfg["nodes"]}
    workers = []
    for i, slot in enumerate(plan.slots):
        a, p = addr[slot.peer_id]
        w = {"id": slot.peer_id, "address": a, "port": p,
             "layer_range": [slot.layer_start, slot.layer_end],
             "sub_gguf_path": slice_path(cfg, mode, i, slot.layer_start, slot.layer_end),
             "mode": slot.mode}
        workers.append(w)
    return {"model": {"id": m["id"], "hidden_size": m["hidden_size"],
                      "num_blocks": m["num_layers"], "wire_dtype": m.get("wire_dtype", "f16")},
            "workers": workers}


def layer_split(plan) -> str:
    return ", ".join(f"{s.peer_id}=[{s.layer_start},{s.layer_end}) "
                     f"({s.layer_end - s.layer_start}L)" for s in plan.slots)


# ── client.py invocation + tok/s parse ────────────────────────────
_TOKS = re.compile(r"generated\s+(\d+)\s+tokens?\s+in\s+([\d.]+)\s*s.*?\(([\d.]+)\s*tok/s\)")


def run_client(cfg: dict, yaml_path: str, n_tokens: int) -> tuple[float, float]:
    """Run client.py for n_tokens. Returns (wall_seconds, parsed_tok_s|nan)."""
    m = cfg["model"]
    cmd = [sys.executable, str(CLIENT), "--config", yaml_path,
           "--model-path", m["full_gguf_path"], "--prompt", cfg.get("prompt", "Hello"),
           "-n", str(n_tokens), "--use-streaming"]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.get("timeout", 1800))
    wall = time.time() - t0
    if out.returncode != 0:
        sys.stderr.write(out.stdout + out.stderr)
        raise RuntimeError(f"client.py failed (rc={out.returncode}) for {yaml_path}")
    mobj = None
    for mm in _TOKS.finditer(out.stdout):
        mobj = mm  # last match = the full-run line
    parsed = float(mobj.group(3)) if mobj else float("nan")
    return wall, parsed


def bench_mode(cfg: dict, mode: str, plan, outdir: Path) -> dict:
    ydict = render_yaml(cfg, mode, plan)
    ypath = outdir / f"chain_{mode}.yaml"
    ypath.write_text(yaml.safe_dump(ydict, sort_keys=False))
    K = int(cfg.get("tokens", 32))
    reps = int(cfg.get("repeats", 3))

    # warm baseline (load + prefill + 1 token), once.
    t1, _ = run_client(cfg, str(ypath), 1)
    ss_rates, parsed_rates = [], []
    for r in range(reps):
        tK, parsed = run_client(cfg, str(ypath), K)
        ss = (K - 1) / (tK - t1) if tK > t1 else float("nan")
        ss_rates.append(ss)
        if parsed == parsed:  # not nan
            parsed_rates.append(parsed)
        print(f"    [{mode}] run {r+1}/{reps}: n={K} wall={tK:.1f}s "
              f"steady={ss:.3f} tok/s  client_reported={parsed:.3f} tok/s")
    return {"mode": mode, "split": layer_split(plan),
            "steady_median": statistics.median(ss_rates) if ss_rates else float("nan"),
            "client_median": statistics.median(parsed_rates) if parsed_rates else float("nan"),
            "yaml": str(ypath)}


def cut_slices(cfg: dict, plans: dict):
    """Cut each plan's sub-GGUF slices on the hub via partial_gguf.py.
    Adjust SLICER to match your slicing tool's CLI if it differs."""
    SLICER = HERE / "partial_gguf.py"
    m = cfg["model"]
    Path(cfg["slice_dir"]).mkdir(parents=True, exist_ok=True)
    for mode, plan in plans.items():
        for i, slot in enumerate(plan.slots):
            out = slice_path(cfg, mode, i, slot.layer_start, slot.layer_end)
            if Path(out).exists():
                print(f"  slice exists: {out}"); continue
            cmd = [sys.executable, str(SLICER), "--in", m["full_gguf_path"],
                   "--out", out, "--layer-start", str(slot.layer_start),
                   "--layer-end", str(slot.layer_end)]
            print("  CUT:", " ".join(cmd))
            subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster", required=True, help="cluster description YAML")
    ap.add_argument("--dry-run", action="store_true",
                    help="print both plans + required slices; no cluster needed")
    ap.add_argument("--cut", action="store_true",
                    help="cut per-plan sub-GGUF slices on the hub, then exit")
    ap.add_argument("--outdir", default=None, help="where to write chain_*.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cluster).read_text())
    plans = {mode: build_plan(cfg, mode) for mode in MODES}

    print(f"\nModel: {cfg['model']['id']}  ({cfg['model']['num_layers']} layers, "
          f"{int(cfg['model']['model_bytes'])/1e9:.0f} GB)   nodes: {len(cfg['nodes'])}\n")
    for mode, plan in plans.items():
        print(f"  partition={mode:8s}: {layer_split(plan)}")
        if plan.warnings:
            for w in plan.warnings:
                print(f"      warn: {w}")
    print()

    if args.dry_run:
        print("Slices each plan needs (cut on hub before a real run):")
        for mode, plan in plans.items():
            for i, slot in enumerate(plan.slots):
                print(f"  {slice_path(cfg, mode, i, slot.layer_start, slot.layer_end)}")
        print("\n(dry run — no generation performed)")
        return
    if args.cut:
        cut_slices(cfg, plans); print("\nslices ready."); return

    outdir = Path(args.outdir or Path(args.cluster).parent / "bench-out")
    outdir.mkdir(parents=True, exist_ok=True)
    results = [bench_mode(cfg, mode, plans[mode], outdir) for mode in MODES]

    print("\n" + "=" * 72)
    print(f"{'mode':10s} {'steady tok/s':>14s} {'client tok/s':>14s}   layer split")
    for r in results:
        print(f"{r['mode']:10s} {r['steady_median']:14.3f} {r['client_median']:14.3f}   {r['split']}")
    g = next(r for r in results if r["mode"] == "greedy")["steady_median"]
    c = next(r for r in results if r["mode"] == "compute")["steady_median"]
    if g and c and g == g and c == c:
        print(f"\ncompute-aware speedup: {c/g:.2f}x  ({g:.3f} -> {c:.3f} tok/s)")
    print("=" * 72)


if __name__ == "__main__":
    main()
