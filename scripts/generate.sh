#!/usr/bin/env bash
# Regenerate protobuf + gRPC Python stubs into scripts/.
# Run from repo root: ./scripts/generate.sh
set -e
cd "$(dirname "$0")/.."
python -m grpc_tools.protoc \
    -I proto \
    --python_out=scripts \
    --grpc_python_out=scripts \
    proto/nakshatra.proto
echo "regenerated: scripts/nakshatra_pb2.py + scripts/nakshatra_pb2_grpc.py"
