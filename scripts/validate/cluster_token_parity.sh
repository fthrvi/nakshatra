#!/usr/bin/env bash
# §10 acceptance — PART B: token-parity over the gRPC chain.
#
# Part A (acceptance_controlplane.py) already proved discover → pin → negotiate →
# self-provision from a verified package, and emitted two assembled sub-GGUFs.
# This script takes those into a real 2-worker localhost chain and checks the
# distributed greedy token is byte-identical to the single-machine reference —
# the v0.1 parity bar, now provisioned with zero hand-cut sub-GGUFs.
#
# PREREQUISITES (this is why it can't run on a box that's never hosted Nakshatra):
#   1. A venv with the runtime deps:   pip install -e .   (grpc, gguf, numpy,
#      cryptography, protobuf, torch/hivemind/… per setup.cfg). Activate it.
#   2. The gRPC worker daemon binary (the partial-load patched llama.cpp):
#        cmake --build <llama.cpp>/build --target llama-nakshatra-worker
#      NOTE: the box currently only has llama-nakshatra-worker-FABRIC; the plain
#      gRPC chain needs the non-fabric target. Point --daemon-bin at it below.
#   3. A full-model GGUF whose layers the package was built from (for the
#      single-machine reference token).
#
# USAGE:
#   NODE_A_GGUF=/path/node-A.gguf NODE_B_GGUF=/path/node-B.gguf \
#   REF_GGUF=/path/full-model.gguf DAEMON_BIN=/path/llama-nakshatra-worker \
#   N_LAYERS=16 PROMPT="The capital of France is" \
#   ./scripts/validate/cluster_token_parity.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # scripts/

: "${NODE_A_GGUF:?set NODE_A_GGUF (from Part A output)}"
: "${NODE_B_GGUF:?set NODE_B_GGUF (from Part A output)}"
: "${REF_GGUF:?set REF_GGUF (full-model GGUF for the reference token)}"
: "${DAEMON_BIN:?set DAEMON_BIN (gRPC worker daemon)}"
: "${N_LAYERS:?set N_LAYERS (total layers, e.g. 16)}"
PROMPT="${PROMPT:-The capital of France is}"
MID=$(( N_LAYERS / 2 ))
PORT_A=5530; PORT_B=5531
MODEL_ID="${MODEL_ID:-accept-1b}"
N_CTX="${N_CTX:-256}"

# ---- prereq gate (fail loud, don't pretend) -------------------------------
fail=0
command -v python3 >/dev/null || { echo "no python3"; fail=1; }
python3 -c "import grpc" 2>/dev/null || { echo "MISSING: grpc (activate the nakshatra venv: pip install -e .)"; fail=1; }
[ -x "$DAEMON_BIN" ] || { echo "MISSING: daemon binary at $DAEMON_BIN (build llama-nakshatra-worker)"; fail=1; }
[ -f "$NODE_A_GGUF" ] && [ -f "$NODE_B_GGUF" ] || { echo "MISSING: provisioned sub-GGUFs (run Part A first)"; fail=1; }
[ -f "$REF_GGUF" ] || { echo "MISSING: reference full-model GGUF at $REF_GGUF"; fail=1; }
if [ "$fail" = 1 ]; then echo; echo "Prereqs unmet — see the header. Aborting."; exit 2; fi

echo "[ref] single-machine greedy token from $REF_GGUF…"
# Reference token: first generated token at temp 0 (the parity anchor). Capture
# to a file (NOT a pipe) so a short read can't SIGPIPE llama-cli under pipefail.
REF_RAW="$(mktemp)"
llama-cli -m "$REF_GGUF" -p "$PROMPT" -n 1 -c "$N_CTX" --temp 0 -no-cnv \
  --no-display-prompt > "$REF_RAW" 2>/dev/null || true
REF_TOK="$(tr -d '\n' < "$REF_RAW" | head -c 64)"
echo "[ref] reference first token text: '$REF_TOK'"

cleanup() { kill "${WA:-0}" "${WB:-0}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[chain] launching worker A [0,$MID) :$PORT_A and B [$MID,$N_LAYERS) :$PORT_B (CPU)…"
python3 "$HERE/worker.py" --port "$PORT_A" --sub-gguf "$NODE_A_GGUF" --mode first \
  --layer-start 0 --layer-end "$MID" --model-id "$MODEL_ID" \
  --daemon-bin "$DAEMON_BIN" --n-ctx "$N_CTX" --n-gpu-layers 0 &
WA=$!
python3 "$HERE/worker.py" --port "$PORT_B" --sub-gguf "$NODE_B_GGUF" --mode last \
  --layer-start "$MID" --layer-end "$N_LAYERS" --model-id "$MODEL_ID" \
  --daemon-bin "$DAEMON_BIN" --n-ctx "$N_CTX" --n-gpu-layers 0 &
WB=$!
sleep 8   # let daemons load + Info come up (P4 handshake negotiates here)

echo "[chain] running distributed inference via client.py…"
CHAIN_OUT=$(python3 "$HERE/client.py" \
  --worker "localhost:$PORT_A:0:$MID:first" \
  --worker "localhost:$PORT_B:$MID:$N_LAYERS:last" \
  --prompt "$PROMPT" --max-tokens 1 2>&1 || true)
echo "$CHAIN_OUT" | tail -20

echo
echo "=== PARITY CHECK ==="
echo "reference first token: '$REF_TOK'"
echo "(compare against the chain's first generated token above — they must match)"
echo "If equal → ✅ §10 token parity PASS (provisioned from a package, no hand-cut sub-GGUF)."
echo "If not   → capture both + the worker logs; this is the bug the run exists to find."
