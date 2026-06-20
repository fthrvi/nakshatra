#!/usr/bin/env bash
# run-imac-unconscious.sh — bring Prithvi's unconscious (DSR1-8B) up DISTRIBUTED across the 3 iMacs,
# coordinated from this box (the GPU/hub). Proven 2026-06-20. CPU workers (Metal is broken on the
# Intel-iMac Radeons). Reach is via the relay (the gpu-box-reach-clients.sh + PRITHVI_RELAY stateful
# fix make this box able to walk the chain). Each iMac self-provisions its layer slice from ~/pkg.
#
#   run-imac-unconscious.sh            # start the 3-iMac chain + print the client command
#   run-imac-unconscious.sh stop       # stop the workers on all 3 iMacs
#   run-imac-unconscious.sh chat "..."  # start (if needed) + run one prompt through it
# Each Mac must be provisioned (provision-worker.sh) + have ~/pkg (package_gguf.py of the model).
set -uo pipefail
MODEL_GGUF="${MODEL_GGUF:-$HOME/.nakshatra/models/dsr1-llama8b/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf}"
CFG="$HOME/.nakshatra/cluster-3imac.yaml"
# host ssh-alias : layer-range : mode  (3-way split of 32 blocks)
WORKERS=("mac3:0:11:first" "mac4:11:22:middle" "bishwa-mac:22:32:last")
IPS=("10.51.0.16" "10.51.0.17" "10.51.0.18")

stop_all(){ for w in "${WORKERS[@]}"; do h="${w%%:*}"; ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'pkill -f worker.py 2>/dev/null' 2>/dev/null; done; echo "stopped workers on all 3 iMacs"; }
[ "${1:-}" = "stop" ] && { stop_all; exit 0; }

echo "starting CPU workers on the 3 iMacs..."
for w in "${WORKERS[@]}"; do
  IFS=: read -r h s e mode <<<"$w"
  ssh -o BatchMode=yes -o ConnectTimeout=12 "$h" "bash -lc '
    pkill -f worker.py 2>/dev/null; sleep 1
    nohup env NAKSHATRA_ACT_QUANT=int8 ~/nakshatra-venv/bin/python ~/nakshatra-scripts/worker.py --port 5540 --sub-gguf ~/slice-$mode.gguf \
      --package-url ~/pkg --mode $mode --layer-start $s --layer-end $e --model-id dsr1-8b \
      --daemon-bin ~/llama-nak/build/bin/llama-nakshatra-worker --n-ctx 2048 --n-gpu-layers 0 \
      --node-id $h-$mode --no-file-server --skip-sha256 >~/w.log 2>&1 & echo started $h $mode'" 2>&1 | tail -1
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

# act-quant on the wire: int8 hidden states = ~4x smaller = ~1.6x faster over the high-latency relay
# (measured 0.98 -> 1.57 tok/s on the 3-iMac chain, output identical). Workers launched with it too.
CLIENT="env NAKSHATRA_ACT_QUANT=int8 $HOME/nakshatra/.venv/bin/python $HOME/nakshatra/scripts/client.py --config $CFG --model-path $MODEL_GGUF --tls-mode off --prompt"
if [ "${1:-}" = "chat" ]; then
  exec $CLIENT "${2:?prompt}" --max-tokens "${3:-32}"
else
  echo "run a prompt:  $CLIENT \"your prompt\" --max-tokens 32"
fi
