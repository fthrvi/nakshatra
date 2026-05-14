#!/usr/bin/env python3
"""Pre-flight check for a Nakshatra cluster YAML.

Connects to each worker, calls Info, and validates:
  - YAML schema (required fields present, types ok)
  - layer ranges form a contiguous [0, num_blocks) partition
  - first worker has has_token_embd=True
  - last worker has has_lm_head=True
  - hidden_size and model_id agree across workers
  - mode (first/middle/last) matches each worker's position in the chain
  - reported layer_range matches the YAML's layer_range entries

Returns 0 if all checks pass, non-zero with a human-readable summary if any
fail. No models are loaded; this only exercises the gRPC Info endpoint.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import yaml
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc


GREEN = "\x1b[32m"
RED   = "\x1b[31m"
RESET = "\x1b[0m"


def ok(msg: str):    print(f"  {GREEN}OK{RESET}     {msg}")
def fail(msg: str):  print(f"  {RED}FAIL{RESET}   {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to cluster YAML")
    ap.add_argument("--timeout", type=float, default=5.0, help="per-worker Info timeout in seconds")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}")
        return 1

    cfg = yaml.safe_load(cfg_path.read_text())
    print(f"[validate] cluster: {cfg_path}")
    issues = 0

    # --- YAML schema sanity ---
    if "model" not in cfg or "workers" not in cfg:
        print(f"{RED}FAIL{RESET} top-level 'model' and 'workers' keys are required")
        return 1
    workers = cfg["workers"]
    if len(workers) < 1:
        print(f"{RED}FAIL{RESET} at least one worker required")
        return 1

    expected_num_blocks = cfg["model"].get("num_blocks")
    expected_hidden = cfg["model"].get("hidden_size")
    expected_model_id = cfg["model"].get("id")
    print(f"[validate] model {expected_model_id!r} num_blocks={expected_num_blocks} hidden_size={expected_hidden}")

    # --- per-worker Info ---
    print(f"[validate] connecting to {len(workers)} workers...")
    infos = []  # (worker_dict, info_response_or_None, error_or_None)
    for w in workers:
        addr = f"{w['address']}:{w['port']}"
        try:
            ch = grpc.insecure_channel(addr)
            stub = pb_grpc.NakshatraStub(ch)
            r = stub.Info(pb.InfoRequest(), timeout=args.timeout)
            infos.append((w, r, None))
            print(f"  {w['id']:14s} {addr:24s}  layers=[{r.layer_start},{r.layer_end})  embd={r.has_token_embd}  lm={r.has_lm_head}  hidden={r.hidden_size}")
        except grpc.RpcError as e:
            infos.append((w, None, e))
            print(f"  {w['id']:14s} {addr:24s}  ERROR: {e.code().name} {e.details()}")
            issues += 1

    if any(r is None for _, r, _ in infos):
        print()
        fail(f"{sum(1 for _,r,_ in infos if r is None)}/{len(infos)} workers unreachable; cannot continue")
        return 2

    print()
    print("[validate] checks:")

    # Hidden size consistent across workers + matches model.hidden_size
    hidden_sizes = {r.hidden_size for _, r, _ in infos}
    if len(hidden_sizes) != 1:
        fail(f"hidden_size disagrees across workers: {hidden_sizes}")
        issues += 1
    elif expected_hidden and list(hidden_sizes)[0] != expected_hidden:
        fail(f"hidden_size {list(hidden_sizes)[0]} differs from model.hidden_size {expected_hidden}")
        issues += 1
    else:
        ok(f"hidden_size = {list(hidden_sizes)[0]} consistent across workers")

    # Model id consistent
    model_ids = {r.model_id for _, r, _ in infos}
    if len(model_ids) != 1:
        fail(f"model_id disagrees: {model_ids}")
        issues += 1
    elif expected_model_id and list(model_ids)[0] != expected_model_id:
        fail(f"worker model_id {list(model_ids)[0]!r} != yaml model.id {expected_model_id!r}")
        issues += 1
    else:
        ok(f"model_id = {list(model_ids)[0]!r}")

    # Sort by layer_start to validate partition
    sorted_infos = sorted(infos, key=lambda x: x[1].layer_start)

    # Check contiguous partition
    prev_end = sorted_infos[0][1].layer_start
    if prev_end != 0:
        fail(f"chain does not start at layer 0 (first worker reports start={prev_end})")
        issues += 1
    for w, r, _ in sorted_infos:
        if r.layer_start != prev_end:
            fail(f"GAP: previous chain ends at {prev_end}, {w['id']} starts at {r.layer_start}")
            issues += 1
        prev_end = r.layer_end
    if expected_num_blocks and prev_end != expected_num_blocks:
        fail(f"chain ends at layer {prev_end} but model.num_blocks={expected_num_blocks}")
        issues += 1
    else:
        ok(f"chain partition is contiguous [0, {prev_end})")

    # First worker must have token_embd
    first_w, first_info, _ = sorted_infos[0]
    if not first_info.has_token_embd:
        fail(f"first worker {first_w['id']!r} reports has_token_embd=False (need True)")
        issues += 1
    else:
        ok(f"first worker {first_w['id']!r} has token_embd")

    # Last worker must have lm_head
    last_w, last_info, _ = sorted_infos[-1]
    if not last_info.has_lm_head:
        fail(f"last worker {last_w['id']!r} reports has_lm_head=False (need True)")
        issues += 1
    else:
        ok(f"last worker {last_w['id']!r} has lm_head")

    # Mode (yaml) vs position (computed) — check each yaml entry's mode is consistent
    for w, r, _ in sorted_infos:
        yaml_mode = w.get("mode", "?")
        is_first = (w == sorted_infos[0][0])
        is_last  = (w == sorted_infos[-1][0])
        expected_mode = "first" if is_first else ("last" if is_last else "middle")
        if yaml_mode != expected_mode:
            fail(f"{w['id']!r} yaml mode={yaml_mode!r} but position is {expected_mode!r}")
            issues += 1

    # Yaml layer_range vs reported layer_range
    for w, r, _ in sorted_infos:
        ylr = w.get("layer_range")
        if ylr is None:
            fail(f"{w['id']!r} yaml is missing layer_range")
            issues += 1
            continue
        if ylr[0] != r.layer_start or ylr[1] != r.layer_end:
            fail(f"{w['id']!r} yaml layer_range={ylr} but worker reports [{r.layer_start},{r.layer_end})")
            issues += 1

    print()
    if issues == 0:
        print(f"{GREEN}[validate] PASS{RESET} — cluster is ready")
        return 0
    else:
        print(f"{RED}[validate] {issues} issue(s) found{RESET}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
