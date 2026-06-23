#!/usr/bin/env python3
"""
gen_chain_config.py — emit a client.py chain YAML for a FIXED topology, for the placement A/B.

bench_placement.py compares two ready chain configs; this generates them deterministically so
the A/B is turnkey: a ROUTE-WHOLE arm (one solo worker holds every layer — 0 inter-node hops)
and a SPLIT arm (an even contiguous split across N workers — the WAN-hop arm in cross-box). The
emitted shape matches serve_planner.plan_chain (the contract client.py --config enforces):

    {model: {id, hidden_size, num_blocks, wire_dtype},
     workers: [{id, address, port, layer_range:[a,b], mode, sub_gguf_path}]}

Modes follow the chain contract: 1 worker = "solo" (carries token_embd + lm_head); else
first / middle… / last. This is a CONTROLLED A/B (fixed topologies), distinct from
NKS_SMART_PLACEMENT (which *chooses* the topology) — here we pin both arms to measure the gap.

Usage:
  gen_chain_config.py route-whole --model <id> --hidden-size 4096 --num-layers 32 \
      --node hub,127.0.0.1,5540 --slices-dir ~/.nakshatra/slices --out route.yaml
  gen_chain_config.py split --model <id> --hidden-size 4096 --num-layers 32 \
      --node hub,127.0.0.1,5540 --node ijru,127.0.0.1,5571 --slices-dir ... --out split.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple


def _mode(i: int, n: int) -> str:
    if n == 1:
        return "solo"
    return "first" if i == 0 else ("last" if i == n - 1 else "middle")


def build_chain(model_id: str, hidden_size: int, num_layers: int, wire_dtype: str,
                workers: List[Tuple[str, str, int, int, int]], slices_dir: str,
                model_hash: str = "x") -> dict:
    """workers = [(node_id, address, port, layer_start, layer_end)] in layer order →
    the client.py chain dict. Validates contiguous full coverage [0, num_layers)."""
    if not workers:
        raise ValueError("no workers")
    cov = sorted((a, b) for (_, _, _, a, b) in workers)
    if cov[0][0] != 0 or cov[-1][1] != num_layers or any(cov[i][1] != cov[i + 1][0]
                                                         for i in range(len(cov) - 1)):
        raise ValueError(f"workers must contiguously cover [0,{num_layers}); got {cov}")
    n = len(workers)
    wy = [{"id": node, "address": addr, "port": int(port),
           "layer_range": [a, b], "mode": _mode(i, n),
           "sub_gguf_path": f"{slices_dir.rstrip('/')}/{model_id}@{model_hash}-L{a}-{b}.gguf"}
          for i, (node, addr, port, a, b) in enumerate(workers)]
    return {"model": {"id": model_id, "hidden_size": hidden_size,
                      "num_blocks": num_layers, "wire_dtype": wire_dtype},
            "workers": wy}


def route_whole(model_id, hidden_size, num_layers, node, slices_dir, wire_dtype="f32") -> dict:
    """One solo worker holds [0, num_layers) — the route-whole arm (0 inter-node hops)."""
    nid, addr, port = node
    return build_chain(model_id, hidden_size, num_layers, wire_dtype,
                       [(nid, addr, int(port), 0, num_layers)], slices_dir)


def even_split(model_id, hidden_size, num_layers, nodes, slices_dir, wire_dtype="f32") -> dict:
    """Even contiguous split across nodes (remainder to the front) — the split arm."""
    k = len(nodes)
    if k < 1:
        raise ValueError("need >=1 node")
    base, extra = divmod(num_layers, k)
    workers, cur = [], 0
    for i, (nid, addr, port) in enumerate(nodes):
        size = base + (1 if i < extra else 0)
        workers.append((nid, addr, int(port), cur, cur + size))
        cur += size
    return build_chain(model_id, hidden_size, num_layers, wire_dtype, workers, slices_dir)


def _parse_node(s: str) -> Tuple[str, str, int]:
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--node must be name,host,port (got {s!r})")
    return (parts[0], parts[1], int(parts[2]))


def main():
    ap = argparse.ArgumentParser(description="generate a client.py chain YAML for the placement A/B")
    ap.add_argument("mode", choices=["route-whole", "split"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--hidden-size", type=int, required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--node", action="append", type=_parse_node, required=True,
                    help="name,host,port — repeat for split; exactly one for route-whole")
    ap.add_argument("--slices-dir", default=str(Path.home() / ".nakshatra" / "slices"))
    ap.add_argument("--wire-dtype", default="f32")
    ap.add_argument("--out", help="write YAML here (default: stdout)")
    a = ap.parse_args()
    if a.mode == "route-whole":
        if len(a.node) != 1:
            ap.error("route-whole needs exactly one --node")
        chain = route_whole(a.model, a.hidden_size, a.num_layers, a.node[0], a.slices_dir, a.wire_dtype)
    else:
        chain = even_split(a.model, a.hidden_size, a.num_layers, a.node, a.slices_dir, a.wire_dtype)
    import yaml
    out = yaml.safe_dump(chain, sort_keys=False)
    if a.out:
        Path(a.out).write_text(out)
        print(f"wrote {a.out} ({len(chain['workers'])} worker(s), mode={a.mode})")
    else:
        print(out)


if __name__ == "__main__":
    main()
