#!/usr/bin/env python3
"""Real-daemon integration smoke for Phase A.3 (C++ kernel-bypass sprint).

Drives the C++ ``llama-nakshatra-worker-fabric`` binary via
``ShmDaemonClient``, loading a real GGUF slice + sending real
CMD_INFO + CMD_EMBD_DECODE requests through the shared-memory rings.
Validates:

  - Python ↔ C++ ring header is byte-compatible (mismatch → C++ side
    raises in attach(), Python side sees daemon-died)
  - 20-byte request envelope round-trips intact through Python →
    shm → C++
  - 8-byte response envelope round-trips back through C++ → shm →
    Python
  - CMD_INFO response parses to the expected n_embd / n_layer
  - One EMBD_DECODE returns a non-zero-length response

This script lives in the nakshatra repo; it expects to be run on
home-pc (or any machine with the daemon binary + a slice). Usage:

  ~/nakshatra-v0/venv/bin/python smoke_fabric_cpp_daemon.py \\
      --daemon-bin ~/llama.cpp/build/bin/llama-nakshatra-worker-fabric \\
      --sub-gguf /tmp/cuts-smoke/w0_v2.gguf

Exit 0 on pass; non-zero on any check failure.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon-bin", required=True,
                    help="path to llama-nakshatra-worker-fabric")
    ap.add_argument("--sub-gguf", required=True,
                    help="path to a sub-GGUF slice (any small llama)")
    ap.add_argument("--mode", default="middle",
                    choices=("first", "middle", "last"))
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--ring-dir", default="/tmp",
                    help="where the shm ring files live (default /tmp)")
    ap.add_argument("--ready-timeout-s", type=float, default=60.0,
                    help="how long to wait for the daemon to load the model")
    ap.add_argument("--fabric-pkg-dir", default=None,
                    help="prepend this dir to sys.path so `from fabric import "
                         "shm_daemon_client` finds the deployed Python code "
                         "(default: assume already on sys.path)")
    args = ap.parse_args()

    if args.fabric_pkg_dir:
        sys.path.insert(0, args.fabric_pkg_dir)

    from fabric.shm_daemon_client import (
        ShmDaemonClient, CMD_EMBD_DECODE,
    )

    print(f"[smoke] daemon_bin = {args.daemon_bin}")
    print(f"[smoke] sub_gguf   = {args.sub_gguf}")
    print(f"[smoke] mode       = {args.mode}")
    print(f"[smoke] spawning C++ daemon — model load may take 20-60s",
          flush=True)

    t_boot = time.time()
    client = ShmDaemonClient(
        daemon_bin=args.daemon_bin,
        sub_gguf=args.sub_gguf,
        mode=args.mode,
        n_ctx=args.n_ctx,
        n_threads=0,
        n_gpu_layers=0,                  # CPU only — deterministic
        ring_dir=Path(args.ring_dir),
        ready_timeout_s=args.ready_timeout_s,
    )
    boot_s = time.time() - t_boot
    print(f"[smoke] ✓ daemon ready in {boot_s:.1f}s", flush=True)

    try:
        # CMD_INFO round trip.
        info = client.info()
        print(f"[smoke] ✓ info: n_embd={info['n_embd']} "
              f"layer_end={info['layer_end']} n_vocab={info['n_vocab']}",
              flush=True)
        if info["n_embd"] <= 0:
            print(f"[smoke] ✗ info returned bad n_embd")
            return 1

        # CMD_EMBD_DECODE round trip — send one token's worth of fp32
        # zeros; assert the response is non-empty.
        n_tokens = 1
        payload_in = b"\x00" * (n_tokens * info["n_embd"] * 4)
        t0 = time.time()
        status, payload_out = client.call(
            CMD_EMBD_DECODE, n_tokens=n_tokens, payload=payload_in,
            start_pos=0, flags=0,
        )
        elapsed_ms = (time.time() - t0) * 1000
        if status != 0:
            print(f"[smoke] ✗ EMBD_DECODE failed status={status}")
            return 1
        if len(payload_out) <= 4:
            print(f"[smoke] ✗ EMBD_DECODE returned too-short payload: "
                  f"{len(payload_out)}")
            return 1
        # Mode=middle returns hidden_state: 4-byte rtype prefix +
        # n_tokens × n_embd × 4 bytes fp32.
        expected_size = 4 + n_tokens * info["n_embd"] * 4
        if args.mode != "last" and len(payload_out) != expected_size:
            print(f"[smoke] ✗ EMBD_DECODE response size mismatch: "
                  f"got {len(payload_out)}, expected {expected_size}")
            return 1
        print(f"[smoke] ✓ EMBD_DECODE: status=0 response={len(payload_out)} "
              f"bytes elapsed={elapsed_ms:.1f}ms", flush=True)

        # Repeat — proves the ring cursor advances across calls.
        for i in range(5):
            t0 = time.time()
            status, _ = client.call(
                CMD_EMBD_DECODE, n_tokens=1, payload=payload_in,
                start_pos=0, flags=0,
            )
            assert status == 0, f"call {i}: status {status}"
            elapsed_ms = (time.time() - t0) * 1000
            print(f"[smoke]   call {i}: {elapsed_ms:.1f}ms", flush=True)

        print(f"\n[smoke] all checks passed (boot {boot_s:.1f}s)")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
