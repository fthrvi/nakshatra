#!/usr/bin/env python3
"""v0.5 M0.5.2 acceptance test — idempotency cache.

Sends two InferenceStep messages with identical (session_id, step_id) on a
single bidirectional stream to one worker, then checks:

  1. The two responses are byte-identical (cache returned the same proto).
  2. /healthz idem_cache.hits incremented by exactly 1 between the two sends.
  3. The daemon's recent_rpc_ms_samples (from /healthz) incremented by exactly 1
     across both sends — confirming the second send did NOT touch the daemon.

Target worker is a first-mode worker (mode=first) since we send token_ids; any
v0.5-M0.5.1+ worker will do. Default target: bishwa:5563 (the 4-Mac Llama-70B
test cluster's last worker — actually that's mode=last; we want first).
Pass --address host:port and --file-server-port to match the target.

Usage:
  scripts/test_idempotency_cache.py --address 100.121.48.32:5560 --file-server-port 6560
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import uuid
from pathlib import Path
from urllib import request as urlrequest

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc


def fetch_healthz(host: str, file_port: int) -> dict:
    url = f"http://{host}:{file_port}/healthz"
    with urlrequest.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True, help="worker host:grpc-port (mode=first)")
    ap.add_argument("--file-server-port", type=int, required=True, help="worker's /healthz port")
    ap.add_argument("--prompt-tokens", type=str, default="128000 791 6864 315 9822 374",
                    help="space-separated int32 prompt tokens (default = Llama-3 tokenization of "
                         "'The capital of France is')")
    args = ap.parse_args()

    host, _, grpc_port = args.address.rpartition(":")
    if not grpc_port:
        sys.exit("--address must be host:port")

    tokens = [int(t) for t in args.prompt_tokens.split()]
    session_id = "idem-test-" + uuid.uuid4().hex[:8]
    step_id = "step-0"

    print(f"[test] target={args.address}, file_server=:{args.file_server_port}")
    print(f"[test] session_id={session_id} step_id={step_id} tokens={tokens}")

    # Snapshot /healthz BEFORE
    before = fetch_healthz(host, args.file_server_port)
    idem_before = before.get("idem_cache") or {}
    daemon_samples_before = before.get("recent_rpc_ms_samples") or 0
    print(f"[test] /healthz BEFORE: idem_cache={idem_before}, daemon_samples={daemon_samples_before}")

    # Open the stream and send the same step twice
    channel = grpc.insecure_channel(args.address)
    stub = pb_grpc.NakshatraStub(channel)

    def build_step():
        s = pb.InferenceStep(session_id=session_id, step_id=step_id, prefix_length=0)
        s.token_ids.ids.extend(tokens)
        return s

    # Use a generator that yields the same step twice
    sent = [build_step(), build_step()]
    received = []

    def request_gen():
        for s in sent:
            yield s

    print("[test] sending 2 identical InferenceSteps on one stream …")
    for i, resp in enumerate(stub.Inference(request_gen())):
        received.append(resp)
        which = "hidden_state" if resp.HasField("hidden_state") else (
            "token_ids" if resp.HasField("token_ids") else (
                "error" if resp.HasField("error") else "unknown"))
        print(f"[test]   recv #{i+1}: payload={which}, prefix_length={resp.prefix_length}")
        if len(received) == 2:
            break

    # Compare the two responses byte-for-byte
    r1_bytes = received[0].SerializeToString()
    r2_bytes = received[1].SerializeToString()
    if r1_bytes != r2_bytes:
        print(f"[test] FAIL: responses differ byte-for-byte ({len(r1_bytes)} vs {len(r2_bytes)} bytes)")
        sys.exit(1)
    print(f"[test] OK: both responses are byte-identical ({len(r1_bytes)} bytes)")

    # Snapshot /healthz AFTER (give heartbeat thread a moment)
    time.sleep(0.5)
    after = fetch_healthz(host, args.file_server_port)
    idem_after = after.get("idem_cache") or {}
    daemon_samples_after = after.get("recent_rpc_ms_samples") or 0
    print(f"[test] /healthz AFTER:  idem_cache={idem_after}, daemon_samples={daemon_samples_after}")

    hits_delta = idem_after.get("hits", 0) - idem_before.get("hits", 0)
    misses_delta = idem_after.get("misses", 0) - idem_before.get("misses", 0)
    samples_delta = daemon_samples_after - daemon_samples_before

    print()
    print(f"[test] cache hits delta:    {hits_delta}   (expected 1)")
    print(f"[test] cache misses delta:  {misses_delta} (expected 1)")
    print(f"[test] daemon RPC samples:  {samples_delta} (expected 1 — second call must NOT hit daemon)")

    ok = (hits_delta == 1 and misses_delta == 1 and samples_delta == 1)
    if ok:
        print()
        print("[test] PASS — M0.5.2 acceptance satisfied")
        sys.exit(0)
    print()
    print("[test] FAIL — cache counters or daemon-call accounting wrong")
    sys.exit(2)


if __name__ == "__main__":
    main()
