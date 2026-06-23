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


def nodes_from_roster(path: str) -> List[Tuple[str, str, int]]:
    """Parse a roster TSV (pubkey·name·operator·tier·tenant·coord, TAB-sep, # = comment) →
    [(name, host, port)] in FILE ORDER — so you don't hand-type node addresses. Raises if a
    coord's port is non-numeric (e.g. an unsubstituted __IJRU_TUNNEL_PORT__ placeholder — fill
    it from the meshd tunnel first)."""
    out: List[Tuple[str, str, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) < 6:
            continue
        name, coord = c[1], c[5]
        host, _, port = coord.partition(":")
        if not port.isdigit():
            raise ValueError(f"roster node {name!r} coord {coord!r} has no numeric port "
                             f"— substitute the tunnel port first")
        out.append((name, host, int(port)))
    return out


def _gather_nodes(roster, extra_nodes) -> List[Tuple[str, str, int]]:
    nodes = nodes_from_roster(roster) if roster else []
    nodes += (extra_nodes or [])
    if not nodes:
        raise SystemExit("need --roster and/or --node")
    return nodes


def _pick(nodes, name):
    """The node that holds the whole model (route-whole/pair). Default = first."""
    if name:
        for n in nodes:
            if n[0] == name:
                return n
        raise SystemExit(f"--route-node {name!r} not among {[n[0] for n in nodes]}")
    return nodes[0]


def _write(chain, out, label):
    import yaml
    text = yaml.safe_dump(chain, sort_keys=False)
    if out:
        Path(out).write_text(text)
        print(f"wrote {out} ({label}: {len(chain['workers'])} worker(s))")
    else:
        print(text)


def main():
    ap = argparse.ArgumentParser(description="generate client.py chain YAML(s) for the placement A/B")
    ap.add_argument("mode", choices=["route-whole", "split", "pair"],
                    help="route-whole = 1 solo worker; split = even across nodes; pair = BOTH (A/B)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hidden-size", type=int, required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--node", action="append", type=_parse_node, default=[],
                    help="name,host,port — repeatable; appended after --roster")
    ap.add_argument("--roster", help="roster TSV to auto-fill nodes (name+coord), in file order")
    ap.add_argument("--route-node", help="node that holds the whole model (route-whole/pair); default=first")
    ap.add_argument("--slices-dir", default=str(Path.home() / ".nakshatra" / "slices"))
    ap.add_argument("--wire-dtype", default="f32")
    ap.add_argument("--out", help="output path (route-whole/split); default stdout")
    ap.add_argument("--out-route", default="route.yaml", help="pair: route-whole output path")
    ap.add_argument("--out-split", default="split.yaml", help="pair: split output path")
    a = ap.parse_args()
    nodes = _gather_nodes(a.roster, a.node)
    rw = lambda n: route_whole(a.model, a.hidden_size, a.num_layers, n, a.slices_dir, a.wire_dtype)
    sp = lambda: even_split(a.model, a.hidden_size, a.num_layers, nodes, a.slices_dir, a.wire_dtype)
    if a.mode == "route-whole":
        _write(rw(_pick(nodes, a.route_node)), a.out, "route-whole")
    elif a.mode == "split":
        _write(sp(), a.out, "split")
    else:  # pair — both arms of the A/B in one command
        _write(rw(_pick(nodes, a.route_node)), a.out_route, "route-whole")
        _write(sp(), a.out_split, "split")
        print(f"\nA/B ready:\n  bench_placement.py --model-path <gguf> "
              f"--route-config {a.out_route} --split-config {a.out_split}")


if __name__ == "__main__":
    main()
