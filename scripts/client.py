#!/usr/bin/env python3
"""Nakshatra client (M5) — chain walker over a multi-worker gRPC chain.

Reads a cluster YAML, queries Info on each worker, validates the layer
partition is contiguous, then walks the chain:
  worker[0]  : tokens IN, hidden state OUT
  worker[1..N-2] (middle): hidden IN, hidden OUT
  worker[N-1] (last): hidden IN, top-1 token id OUT

For multi-token generation the loop replays from the prompt each step
(same brute-force as M2.5; KV-cache reuse arrives with the streaming
Inference RPC in a later milestone).
"""
import argparse
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import yaml
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc


LLAMA3_EOS_IDS = {128001, 128008, 128009}


def tokenize_local(model_path, prompt):
    from llama_cpp import Llama
    llama = Llama(model_path=model_path, vocab_only=True, verbose=False)
    return llama.tokenize(prompt.encode("utf-8"), add_bos=True, special=True), llama


def detok_one(llama, tid):
    try:
        return llama.detokenize([tid]).decode("utf-8", errors="replace")
    except Exception:
        return "?"


def call_forward(stub, payload, n_tokens, has_token_ids, worker_id="<unknown>"):
    req = pb.ForwardRequest(
        hidden_in=payload, batch=1, n_tokens=n_tokens, has_token_ids=has_token_ids,
    )
    try:
        resp = stub.Forward(req, timeout=300.0)
    except grpc.RpcError as e:
        sys.exit(
            f"[chain] Forward RPC to worker {worker_id!r} failed: "
            f"{e.code().name} — {e.details()}"
        )
    return resp.hidden_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="cluster YAML")
    ap.add_argument("--model-path", type=str, required=True, help="full GGUF path for tokenizer (must match the model that was split)")
    ap.add_argument("--prompt", type=str, default="The capital of France is")
    ap.add_argument("--max-tokens", "-n", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    workers = cfg["workers"]
    print(f"[chain] {len(workers)} workers in config")

    # Open channels + Info on each
    stubs = []
    n_embd = None
    for w in workers:
        addr = f"{w['address']}:{w['port']}"
        ch = grpc.insecure_channel(addr)
        stub = pb_grpc.NakshatraStub(ch)
        info = stub.Info(pb.InfoRequest(), timeout=10.0)
        print(f"  {w['id']:12s} {addr:25s}  layers=[{info.layer_start},{info.layer_end})  embd={info.has_token_embd}  lm={info.has_lm_head}  hidden={info.hidden_size}")
        if n_embd is None:
            n_embd = info.hidden_size
        elif info.hidden_size != n_embd:
            sys.exit(f"hidden_size mismatch across workers: {n_embd} vs {info.hidden_size}")
        stubs.append((w, stub, info))

    # Validate chain partition
    n_workers = len(stubs)
    sorted_stubs = sorted(stubs, key=lambda x: x[2].layer_start)
    prev_end = sorted_stubs[0][2].layer_start
    for w, _, info in sorted_stubs:
        if info.layer_start != prev_end:
            sys.exit(f"chain GAP between previous and {w['id']}")
        prev_end = info.layer_end
    if not sorted_stubs[0][2].has_token_embd:
        sys.exit("first worker must have token_embd")
    if not sorted_stubs[-1][2].has_lm_head:
        sys.exit("last worker must have lm_head")
    print(f"[chain] OK: contiguous coverage of [{sorted_stubs[0][2].layer_start}, {sorted_stubs[-1][2].layer_end})")

    # Tokenize
    print(f"[chain] tokenizing locally")
    tokens, llama = tokenize_local(args.model_path, args.prompt)
    print(f"[chain] {len(tokens)} prompt tokens: {tokens}")

    generated = []
    t0 = time.time()
    for step in range(args.max_tokens):
        ctx_tokens = tokens + generated
        n = len(ctx_tokens)

        # Step 1: tokens → first worker → hidden
        token_payload = struct.pack(f"<{n}i", *ctx_tokens)
        first_w, first_stub, _ = sorted_stubs[0]
        hidden = call_forward(first_stub, token_payload, n, has_token_ids=True,
                              worker_id=first_w["id"])
        if len(hidden) != n * n_embd * 4:
            sys.exit(f"[chain] first worker {first_w['id']!r} returned {len(hidden)} bytes, expected {n*n_embd*4} (n_tokens={n} hidden_size={n_embd})")

        # Steps 2..N-1: middle workers
        for w, stub, info in sorted_stubs[1:-1]:
            hidden = call_forward(stub, hidden, n, has_token_ids=False,
                                  worker_id=w["id"])
            if len(hidden) != n * n_embd * 4:
                sys.exit(f"[chain] middle worker {w['id']!r} returned {len(hidden)} bytes, expected {n*n_embd*4}")

        # Step N: last worker → token id
        last_w, last_stub, _ = sorted_stubs[-1]
        last_resp = call_forward(last_stub, hidden, n, has_token_ids=False,
                                 worker_id=last_w["id"])
        if len(last_resp) != 4:
            sys.exit(f"[chain] last worker {last_w['id']!r} returned {len(last_resp)} bytes, expected 4 (one int32 token id)")
        next_id = struct.unpack("<i", last_resp)[0]
        generated.append(next_id)

        if next_id in LLAMA3_EOS_IDS:
            print(f"[chain] step {step+1}: EOS {next_id} — stopping")
            break
        print(f"[chain] step {step+1}: id={next_id} '{detok_one(llama, next_id)}'", flush=True)

    elapsed = time.time() - t0
    full = llama.detokenize(tokens + generated).decode("utf-8", errors="replace")
    gen = llama.detokenize(generated).decode("utf-8", errors="replace") if generated else ""
    print(f"[chain] generated {len(generated)} tokens in {elapsed:.2f}s  ({len(generated)/elapsed:.2f} tok/s)")
    print(f"[chain] full: {full!r}")
    print(f"[chain] gen:  {gen!r}")
    print(f"TOPTOKS_CHAIN {' '.join(str(t) for t in generated)}")


if __name__ == "__main__":
    main()
