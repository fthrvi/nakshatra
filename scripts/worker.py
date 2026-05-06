#!/usr/bin/env python3
"""Nakshatra worker stub (M1) — implements Info only.

Forward + Inference return UNIMPLEMENTED. Real worker (M2+) links patched
llama.cpp and serves real compute. This stub exists to validate the
gRPC plumbing and the proto file shape.
"""
import argparse
import sys
from concurrent import futures
from pathlib import Path

# Add scripts/ to path so generated stubs are importable
sys.path.insert(0, str(Path(__file__).parent))

import grpc
import nakshatra_pb2 as pb
import nakshatra_pb2_grpc as pb_grpc


class WorkerServicer(pb_grpc.NakshatraServicer):
    def __init__(self, layer_start: int, layer_end: int, model_id: str):
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.model_id = model_id

    def Info(self, request, context):
        return pb.InfoResponse(
            protocol_version="0.1.0",
            backend="stub",
            model_id=self.model_id,
            model_content_hash=b"\x00" * 32,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            hidden_size=0,
            wire_dtype="f16",
            kv_cache_tokens_free=0,
            has_token_embd=(self.layer_start == 0),
            has_lm_head=False,  # caller decides this from the cluster config
        )

    def Forward(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Forward not implemented in M1 stub")
        return pb.ForwardResponse()

    def Inference(self, request_iterator, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Inference not implemented in M1 stub")
        return iter([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5500)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=14)
    ap.add_argument("--model-id", type=str, default="stub-model")
    args = ap.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_NakshatraServicer_to_server(
        WorkerServicer(args.layer_start, args.layer_end, args.model_id),
        server,
    )
    server.add_insecure_port(f"[::]:{args.port}")
    print(f"[worker] M1 stub listening on :{args.port}  layers=[{args.layer_start},{args.layer_end})  model={args.model_id}", flush=True)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
