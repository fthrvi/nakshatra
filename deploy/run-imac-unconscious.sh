#!/usr/bin/env bash
# run-imac-unconscious.sh — bring Prithvi's unconscious (DSR1-8B) up across the iMacs, coordinated
# from this box (the GPU/hub). Proven 2026-06-20. CPU workers (Metal is broken on the Intel-iMac
# Radeons). Reach is via the relay (gpu-box-reach-clients.sh + the PRITHVI_RELAY stateful fix let
# this box walk the chain). Each iMac self-provisions its layer slice from ~/pkg.
#
# Two profiles (set PROFILE=relay|push, default relay):
#   relay  — 3-iMac distributed chain; the CLIENT relays each hop over the WAN (most distributed).
#   push   — 2 co-located stages on ONE iMac ($PUSH_HOST): the first worker pushes the hidden state
#            to the last over LOCALHOST, so only client<->first crosses the WAN. ~1.6x faster than a
#            2-iMac relay (measured 2.07 -> 3.37 tok/s on mac4, output identical). This is the
#            achievable form of worker-to-worker push: cross-iMac push is dead (the remote iMacs are
#            NAT'd relay clients and can't reach each other through the VPS reflector), but co-locating
#            the pair on one box turns the inter-stage hop into a free localhost round-trip.
#
#   run-imac-unconscious.sh             # start (PROFILE) + print the client command
#   run-imac-unconscious.sh stop        # stop workers on every iMac
#   run-imac-unconscious.sh chat "..."  # start (if needed) + run one prompt
# Each Mac must be provisioned (provision-worker.sh) + have ~/pkg (package_gguf.py of the model).
set -uo pipefail
PROFILE="${PROFILE:-relay}"
MODEL_GGUF="${MODEL_GGUF:-$HOME/.nakshatra/models/dsr1-llama8b/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf}"
ALL_HOSTS=("mac3" "mac4" "bishwa-mac")

launch(){ # host port mode start end node extra_env
  local h="$1" port="$2" mode="$3" s="$4" e="$5" node="$6" extra="${7:-}"
  ssh -o BatchMode=yes -o ConnectTimeout=12 "$h" "bash -lc '
    nohup env NAKSHATRA_ACT_QUANT=int8 $extra ~/nakshatra-venv/bin/python ~/nakshatra-scripts/worker.py --port $port --sub-gguf ~/slice-$mode.gguf \
      --package-url ~/pkg --mode $mode --layer-start $s --layer-end $e --model-id dsr1-8b \
      --daemon-bin ~/llama-nak/build/bin/llama-nakshatra-worker --n-ctx 2048 --n-gpu-layers 0 \
      --node-id $node --no-file-server --skip-sha256 >~/w-$mode.log 2>&1 & echo started $h $mode:$port'" 2>&1 | tail -1
}
stop_all(){ for h in "${ALL_HOSTS[@]}"; do ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'pkill -f worker.py 2>/dev/null' 2>/dev/null; done; echo "stopped workers on all iMacs"; }
[ "${1:-}" = "stop" ] && { stop_all; exit 0; }

if [ "$PROFILE" = "push" ]; then
  PUSH_HOST="${PUSH_HOST:-mac4}"; PUSH_IP="${PUSH_IP:-10.51.0.17}"
  CFG="$HOME/.nakshatra/cluster-push-$PUSH_HOST.yaml"
  echo "starting 2 co-located CPU workers on $PUSH_HOST (push profile)..."
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$PUSH_HOST" 'pkill -f worker.py 2>/dev/null; sleep 1' 2>/dev/null
  # the inter-worker push uses plaintext localhost -> allow unpinned peer (no SPKI between co-located stages)
  launch "$PUSH_HOST" 5540 first 0 16 "$PUSH_HOST-first" "NAKSHATRA_REFUSE_UNPINNED_PEERS=false"
  launch "$PUSH_HOST" 5541 last 16 32 "$PUSH_HOST-last"  "NAKSHATRA_REFUSE_UNPINNED_PEERS=false"
  cat > "$CFG" <<YAML
model: {id: dsr1-8b, hidden_size: 4096, num_blocks: 32, wire_dtype: f32}
workers:
  # the client reaches both on the mesh \`address\`; the FIRST worker forwards to the last on
  # \`internal_address\` (localhost) — a free hop instead of a WAN relay round-trip.
  - {id: $PUSH_HOST-first, address: $PUSH_IP, port: 5540, layer_range: [0, 16], mode: first}
  - {id: $PUSH_HOST-last,  address: $PUSH_IP, internal_address: 127.0.0.1, port: 5541, layer_range: [16, 32], mode: last}
YAML
  echo "waiting for workers to load (CPU, ~30-60s)..."
  for i in $(seq 1 24); do up=0; for p in 5540 5541; do timeout 5 bash -c "cat </dev/null >/dev/tcp/$PUSH_IP/$p" 2>/dev/null && up=$((up+1)); done; [ "$up" = 2 ] && break; sleep 6; done
  echo "2 co-located workers up on $PUSH_HOST. Config: $CFG"
  # push streams the hidden over localhost (the WAN win is fewer round-trips, not smaller bytes); the
  # iMacs run the pre-streaming-act-quant worker.py, so keep client act-quant OFF to match wire f32.
  CLIENT_ENV=""; STREAM_FLAGS="--use-streaming --use-streaming-push"
else
  CFG="$HOME/.nakshatra/cluster-3imac.yaml"
  WORKERS=("mac3:0:11:first" "mac4:11:22:middle" "bishwa-mac:22:32:last")
  IPS=("10.51.0.16" "10.51.0.17" "10.51.0.18")
  echo "starting CPU workers on the 3 iMacs (relay profile)..."
  for w in "${WORKERS[@]}"; do IFS=: read -r h s e mode <<<"$w"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'pkill -f worker.py 2>/dev/null; sleep 1' 2>/dev/null
    launch "$h" 5540 "$mode" "$s" "$e" "$h-$mode"
  done
  cat > "$CFG" <<YAML
model: {id: dsr1-8b, hidden_size: 4096, num_blocks: 32, wire_dtype: f32}
workers:
  - {id: mac3-first,  address: ${IPS[0]}, port: 5540, layer_range: [0, 11],  mode: first}
  - {id: mac4-middle, address: ${IPS[1]}, port: 5540, layer_range: [11, 22], mode: middle}
  - {id: bishwa-last, address: ${IPS[2]}, port: 5540, layer_range: [22, 32], mode: last}
YAML
  echo "waiting for workers to load (CPU, ~30-60s)..."
  for i in $(seq 1 24); do up=0; for ip in "${IPS[@]}"; do timeout 5 bash -c "cat </dev/null >/dev/tcp/$ip/5540" 2>/dev/null && up=$((up+1)); done; [ "$up" = 3 ] && break; sleep 6; done
  echo "3 iMac workers up. Config: $CFG"
  # relay: act-quant on the wire (int8 hidden = ~4x smaller, compounds across hops, measured
  # 0.98 -> 1.57 tok/s); UNARY (the iMacs' worker.py has act-quant on the unary path, not streaming).
  CLIENT_ENV="env NAKSHATRA_ACT_QUANT=int8"; STREAM_FLAGS=""
fi

CLIENT="$CLIENT_ENV $HOME/nakshatra/.venv/bin/python $HOME/nakshatra/scripts/client.py --config $CFG --model-path $MODEL_GGUF --tls-mode off $STREAM_FLAGS --prompt"
if [ "${1:-}" = "chat" ]; then
  exec $CLIENT "${2:?prompt}" --max-tokens "${3:-32}"
else
  echo "run a prompt:  $CLIENT \"your prompt\" --max-tokens 32"
fi
