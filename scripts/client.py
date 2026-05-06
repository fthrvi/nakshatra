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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", type=str, default="localhost:5500")
    ap.add_argument("--model-path", type=str, required=True, help="path used for tokenizer (must match worker's model)")
    ap.add_argument("--prompt", type=str, default="The capital of France is")
    args = ap.parse_args()

    print(f"[client] tokenizing locally")
    token_ids, llama = tokenize_local(args.model_path, args.prompt)
    print(f"[client] {len(token_ids)} tokens: {token_ids}")

    payload = struct.pack(f"={len(token_ids)}i", *token_ids)
    req = pb.ForwardRequest(hidden_in=payload, batch=1, n_tokens=len(token_ids), has_token_ids=True)

    print(f"[client] calling Forward on {args.addr}")
    channel = grpc.insecure_channel(args.addr)
    stub = pb_grpc.NakshatraStub(channel)

    t0 = time.time()
    resp = stub.Forward(req, timeout=120.0)
    elapsed = time.time() - t0

    top_id, = struct.unpack("=i", resp.hidden_out)
    top_str = detokenize_one(llama, top_id)
    print(f"[client] top-1 next token: id={top_id} '{top_str}'  rtt={elapsed*1000:.0f}ms")
    print(f"TOPTOK {top_id} {top_str}")


if __name__ == "__main__":
    main()
