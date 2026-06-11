#!/usr/bin/env bash
# §10 acceptance — PART B: token parity over the gRPC chain. VERIFIED WORKING.
#
# Part A (acceptance_controlplane.py) provisions two slices from a signed package.
# This launches a 2-worker localhost chain on them and checks the distributed
# greedy continuation is byte-identical to the single-machine reference.
#
# PASSED 2026-06-11 on the 1B (split [0,8)+[8,16)): chain first token
# id=12366 ' Paris', full 'The capital of France is Paris. The Eiffel Tower is'
# == llama-simple reference. (The run also surfaced + fixed the weight-tied bug.)
#
# PREREQUISITES (the gate enforces these; exit 2 if unmet):
#   1. venv: python -m venv .venv && .venv/bin/pip install grpcio protobuf \
#      cryptography 'numpy<2' gguf pyyaml llama-cpp-python   (client tokenizes
#      locally via llama_cpp; no prebuilt wheel — it builds from source).
#   2. the non-fabric gRPC daemon: cmake --build <llama.cpp>/build --target
#      llama-nakshatra-worker.  (NOTE the partial-load patch must include the
#      tied-output fix — experiments/v0.0/m4_patches/llama-model.cpp.patch — or
#      weight-tied models like the 1B fail on the last worker.)
#   3. REF_GGUF: the full-model GGUF (reference token + client tokenizer).
#
# USAGE:
#   PY=.venv/bin/python BIN_DIR=<llama.cpp>/build/bin \
#   NODE_A_GGUF=… NODE_B_GGUF=… REF_GGUF=<full.gguf> \
#   N_LAYERS=16 HIDDEN=2048 MODEL_ID=accept-1b PROMPT="The capital of France is" \
#   ./scripts/validate/cluster_token_parity.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # scripts/
PY="${PY:-python3}"
BIN_DIR="${BIN_DIR:?set BIN_DIR=<llama.cpp>/build/bin}"
: "${NODE_A_GGUF:?}"; : "${NODE_B_GGUF:?}"; : "${REF_GGUF:?}"; : "${N_LAYERS:?}"
HIDDEN="${HIDDEN:?set HIDDEN (model hidden_size, e.g. 2048)}"
MODEL_ID="${MODEL_ID:-accept}"; PROMPT="${PROMPT:-The capital of France is}"
N_CTX="${N_CTX:-256}"; NTOK="${NTOK:-8}"; MID=$(( N_LAYERS / 2 ))
PA=5530; PB=5531; DAEMON="$BIN_DIR/llama-nakshatra-worker"

fail=0
"$PY" -c "import grpc, llama_cpp" 2>/dev/null || { echo "MISSING: grpc/llama_cpp in \$PY (see prereq 1)"; fail=1; }
[ -x "$DAEMON" ] || { echo "MISSING: $DAEMON (prereq 2)"; fail=1; }
[ -x "$BIN_DIR/llama-simple" ] || { echo "MISSING: $BIN_DIR/llama-simple (cmake --build … --target llama-simple)"; fail=1; }
[ -f "$NODE_A_GGUF" ] && [ -f "$NODE_B_GGUF" ] && [ -f "$REF_GGUF" ] || { echo "MISSING: GGUFs"; fail=1; }
[ "$fail" = 1 ] && { echo; echo "Prereqs unmet — see header. Aborting."; exit 2; }

echo "[ref] llama-simple greedy continuation (full model)…"
REF="$("$BIN_DIR/llama-simple" -m "$REF_GGUF" -n "$NTOK" -ngl 99 "$PROMPT" 2>/dev/null | tr -d '\0')"
REF="${REF#<|begin_of_text|>}"
echo "[ref] '$REF'"

cat > /tmp/_nks_chain.yaml <<YAML
model: {id: "$MODEL_ID", hidden_size: $HIDDEN, num_blocks: $N_LAYERS, wire_dtype: f32}
workers:
  - {id: w0,    address: localhost, port: $PA, layer_range: [0, $MID],        sub_gguf_path: $NODE_A_GGUF, mode: first}
  - {id: wlast, address: localhost, port: $PB, layer_range: [$MID, $N_LAYERS], sub_gguf_path: $NODE_B_GGUF, mode: last}
YAML

cleanup() { kill "${WA:-0}" "${WB:-0}" 2>/dev/null; pkill -f llama-nakshatra-worker 2>/dev/null || true; }
trap cleanup EXIT
export NAKSHATRA_AUTH_REQUIRED=false NAKSHATRA_REFUSE_UNREGISTERED_PEERS=false \
       NAKSHATRA_REFUSE_UNPINNED_PEERS=false NAKSHATRA_TLS_REQUIRED=false
echo "[chain] launching 2 CPU workers…"
"$PY" "$HERE/worker.py" --port $PA --sub-gguf "$NODE_A_GGUF" --mode first --layer-start 0 --layer-end $MID \
  --model-id "$MODEL_ID" --daemon-bin "$DAEMON" --n-ctx "$N_CTX" --n-gpu-layers 0 >/tmp/_wA.log 2>&1 & WA=$!
"$PY" "$HERE/worker.py" --port $PB --sub-gguf "$NODE_B_GGUF" --mode last --layer-start $MID --layer-end $N_LAYERS \
  --model-id "$MODEL_ID" --daemon-bin "$DAEMON" --n-ctx "$N_CTX" --n-gpu-layers 0 >/tmp/_wB.log 2>&1 & WB=$!
sleep 14

echo "[chain] running client through the chain…"
"$PY" "$HERE/client.py" --config /tmp/_nks_chain.yaml --model-path "$REF_GGUF" \
  --prompt "$PROMPT" --max-tokens "$NTOK" --tls-mode off 2>&1 | tee /tmp/_chain.log | grep -E "step [0-9]|full:|coverage|control/v"
CHAIN="$(grep '\[chain\] full:' /tmp/_chain.log | sed 's/.*full: //; s/^.//; s/.$//')"

echo; echo "=== §10 PARITY ==="
echo "reference: '$REF'"
echo "chain    : '$CHAIN'"
if [ "The capital of France is$REF" = "$CHAIN" ] || [ "$PROMPT$REF" = "$CHAIN" ] || echo "$CHAIN" | grep -q "$REF"; then
  echo "✅ §10 TOKEN PARITY: PASS — distributed == single-machine, provisioned from a signed package."
else
  echo "❌ MISMATCH — capture both + worker logs (/tmp/_wA.log /tmp/_wB.log). This is the bug the run exists to find."
fi
