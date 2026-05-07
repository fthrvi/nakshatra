#!/usr/bin/env python3
"""Nakshatra client (M2) — Info + single-worker Forward end-to-end test.

Tokenizes prompt locally (vocab-only Llama instance for speed), calls Forward
on a single worker carrying the full model, prints the next-token id+string.
For multi-worker chain walk, see M5.
"""
import argparse
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc


def tokenize_local(model_path: str, prompt: str):
    """Use vocab-only Llama for fast tokenization without loading the full model."""
    from llama_cpp import Llama
    llama = Llama(model_path=model_path, vocab_only=True, verbose=False)
    return llama.tokenize(prompt.encode("utf-8"), add_bos=True, special=True), llama


def detokenize_one(llama, token_id: int) -> str:
    try:
        return llama.detokenize([token_id]).decode("utf-8", errors="replace")
    except Exception:
        return "?"


def call_forward(stub, token_ids):
    payload = struct.pack(f"={len(token_ids)}i", *token_ids)
    req = pb.ForwardRequest(hidden_in=payload, batch=1, n_tokens=len(token_ids), has_token_ids=True)
    resp = stub.Forward(req, timeout=120.0)
    next_id, = struct.unpack("=i", resp.hidden_out)
    return next_id


# Llama-3 family special tokens that should terminate generation.
LLAMA3_EOS_IDS = {128001, 128008, 128009}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", type=str, default="localhost:5500")
    ap.add_argument("--model-path", type=str, required=True, help="path used for tokenizer (must match worker's model)")
    ap.add_argument("--prompt", type=str, default="The capital of France is")
    ap.add_argument("--max-tokens", "-n", type=int, default=1, help="number of tokens to generate (greedy)")
    args = ap.parse_args()

    print(f"[client] tokenizing locally")
    token_ids, llama = tokenize_local(args.model_path, args.prompt)
    print(f"[client] {len(token_ids)} prompt tokens: {token_ids}")

    print(f"[client] connecting to {args.addr}")
    channel = grpc.insecure_channel(args.addr)
    stub = pb_grpc.NakshatraStub(channel)

    generated = []
    t0 = time.time()
    for i in range(args.max_tokens):
        # M2.5: re-send the entire context each step. M5 will use the streaming
        # Inference RPC with worker-side KV cache so we don't re-process the prompt.
        next_id = call_forward(stub, token_ids + generated)
        generated.append(next_id)
        if next_id in LLAMA3_EOS_IDS:
            print(f"[client] step {i+1}: EOS token {next_id} — stopping", flush=True)
            break
        next_str = detokenize_one(llama, next_id)
        print(f"[client] step {i+1}: id={next_id} '{next_str}'", flush=True)

    elapsed = time.time() - t0
    full_text = llama.detokenize(token_ids + generated).decode("utf-8", errors="replace")
    gen_text = llama.detokenize(generated).decode("utf-8", errors="replace") if generated else ""
    n_gen = len(generated)
    tps = n_gen / elapsed if elapsed > 0 else 0.0
    print(f"[client] generated {n_gen} tokens in {elapsed:.2f}s  ({tps:.1f} tok/s)")
    print(f"[client] full output: {full_text!r}")
    print(f"[client] generated only: {gen_text!r}")
    print(f"TOPTOKS {' '.join(str(t) for t in generated)}")


if __name__ == "__main__":
    main()
