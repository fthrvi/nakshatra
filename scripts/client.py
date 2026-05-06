#!/usr/bin/env python3
"""Nakshatra client stub (M1) — calls Info on each worker, prints capabilities,
verifies the chain forms a contiguous layer partition.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import yaml
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc


def call_info(addr: str):
    channel = grpc.insecure_channel(addr)
    stub = pb_grpc.NakshatraStub(channel)
    return stub.Info(pb.InfoRequest(), timeout=5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="cluster YAML config")
    ap.add_argument("--addr", type=str, action="append",
                    help="worker addr (e.g. localhost:5500); repeatable. Used if --config not given.")
    args = ap.parse_args()

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        workers = [(w["id"], f"{w['address']}:{w['port']}", w.get("layer_range")) for w in cfg["workers"]]
    elif args.addr:
        workers = [(f"w{i}", a, None) for i, a in enumerate(args.addr)]
    else:
        workers = [("w0", "localhost:5500", None)]

    print(f"[client] querying {len(workers)} worker(s)")
    rs = []
    for wid, addr, expected_range in workers:
        try:
            r = call_info(addr)
            rs.append((wid, addr, r))
            print(f"  {wid:12s} {addr:25s}  layers=[{r.layer_start},{r.layer_end})  backend={r.backend}  model={r.model_id}  hidden={r.hidden_size}  embd={r.has_token_embd}  lm_head={r.has_lm_head}")
            if expected_range and (r.layer_start != expected_range[0] or r.layer_end != expected_range[1]):
                print(f"    !! MISMATCH: config says {expected_range}, worker reports [{r.layer_start},{r.layer_end})")
        except grpc.RpcError as e:
            print(f"  {wid:12s} {addr:25s}  ERROR: {e.code()} {e.details()}")

    if len(rs) >= 2:
        print()
        print("[client] chain validation:")
        ranges = sorted([(r.layer_start, r.layer_end, wid) for wid, _, r in rs])
        prev_end = ranges[0][0]
        ok = True
        for s, e, w in ranges:
            if s != prev_end:
                print(f"  GAP between previous and {w}: prev_end={prev_end}, this_start={s}")
                ok = False
            prev_end = e
        if ok:
            print(f"  OK: contiguous coverage of [{ranges[0][0]}, {ranges[-1][1]})")


if __name__ == "__main__":
    main()
