#!/usr/bin/env python3
"""Nakshatra worker (M2) — full-model inference via llama-cpp-python.

M2 is the "cheating" worker: it loads the FULL GGUF (no patched-load support yet)
and runs a complete forward pass per request. Real partial-model loading arrives
at M3-M4 with the patched llama.cpp.

Forward behaviour:
  - request.has_token_ids = True: hidden_in carries int32-packed token IDs.
    Worker runs the full forward, samples top-1, returns the next-token ID
    in hidden_out (also int32-packed).
  - All other cases return UNIMPLEMENTED for now.

Inference (streaming) is still UNIMPLEMENTED in M2; arrives at M5.
"""
import argparse
import struct
import sys
import time
from concurrent import futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import grpc
import numpy as np
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc


class WorkerServicer(pb_grpc.NakshatraServicer):
    def __init__(self, model_path: str, layer_start: int, layer_end: int, model_id: str, n_ctx: int):
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.model_id = model_id
        self.n_ctx = n_ctx
        self.model_path = model_path

        # Load the model — M2 cheats by loading the full thing
        print(f"[worker] loading {model_path} (n_ctx={n_ctx}, CPU)", flush=True)
        t0 = time.time()
        from llama_cpp import Llama
        self.llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0,
            verbose=False,
            logits_all=True,   # required: with False, _scores[] stays zeros after .eval()
        )
        self.hidden_size = int(self.llama.n_embd())
        self.n_vocab = int(self.llama.n_vocab())
        print(f"[worker] loaded in {time.time()-t0:.1f}s  hidden_size={self.hidden_size}  vocab={self.n_vocab}", flush=True)

    def Info(self, request, context):
        return pb.InfoResponse(
            protocol_version="0.1.0",
            backend="llamacpp-cpu-python",
            model_id=self.model_id,
            model_content_hash=b"\x00" * 32,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            hidden_size=self.hidden_size,
            wire_dtype="f16",
            kv_cache_tokens_free=self.n_ctx,
            has_token_embd=(self.layer_start == 0),
            has_lm_head=False,
        )

    def Forward(self, request, context):
        if not request.has_token_ids:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("M2 worker only supports has_token_ids=True; full-pipeline hidden-state Forward arrives at M5")
            return pb.ForwardResponse()

        n = request.n_tokens
        if n <= 0 or len(request.hidden_in) != n * 4:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"hidden_in size mismatch: got {len(request.hidden_in)} bytes, expected {n*4}")
            return pb.ForwardResponse()

        token_ids = list(struct.unpack(f"={n}i", request.hidden_in))
        print(f"[worker] Forward: {n} tokens {token_ids}", flush=True)

        # Run full forward
        self.llama.reset()
        t0 = time.time()
        self.llama.eval(token_ids)
        elapsed = time.time() - t0

        # Last-token logits (shape [n_vocab])
        logits = np.asarray(self.llama._scores[-1])
        top_id = int(np.argmax(logits))
        top_v = float(logits[top_id])
        print(f"[worker] Forward done in {elapsed*1000:.0f}ms; top-1 id={top_id} logit={top_v:.4f}", flush=True)

        return pb.ForwardResponse(hidden_out=struct.pack("=i", top_id))

    def Inference(self, request_iterator, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Inference streaming arrives at M5")
        return iter([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5500)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=0, help="0 = autodetect from model block_count")
    ap.add_argument("--model-id", type=str, default="nakshatra-v0.1-m2")
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--n-ctx", type=int, default=256)
    args = ap.parse_args()

    # If layer_end is 0, auto-fill after model loads — but easier to just default to None and skip the chain check
    layer_end = args.layer_end if args.layer_end > 0 else 999  # placeholder; M2 is single-worker

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = WorkerServicer(args.model_path, args.layer_start, layer_end, args.model_id, args.n_ctx)
    pb_grpc.add_NakshatraServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{args.port}")
    print(f"[worker] M2 listening on :{args.port}  layers=[{args.layer_start},{layer_end})  model={args.model_id}", flush=True)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
