#!/usr/bin/env bash
# Start the 5-worker Nakshatra cluster (matches scripts/cluster_5worker.yaml).
#
# This is the *manual* bring-up path. You pre-cut sub-GGUFs with
# partial_gguf.py, scp them to each worker's /tmp/, then this script
# launches the workers in parallel.
#
# Alternative (Sthambha planner, shipped 2026-05-14):
#   sthambha-cli plan create --model <id> --num-layers N --model-bytes <bytes> --hub <peer>
#   sthambha-cli plan execute <plan_id> --wait-cached
# That path queries the live pillar registry, picks a vendor-grouped chain,
# runs partial_gguf.py on a designated hub peer via POST /slice on the
# worker's file-server port, and lets Phase-4 auto-fetch distribute slices
# to recipient workers. Reach for the YAML flow when you want a
# deterministic, fixed cluster shape (drift testing, regression repro,
# Nakshatra-engine debugging); for everything else the planner removes
# the hand-edited config + scp step. Full split in
# ~/sthambha/docs/layer-split-planner.md §10.
#
# Pre-requisite on the Linux home PC (one-time, no sudo needed):
#   loginctl enable-linger prithvi
# Without lingering, systemd-logind kills user processes when the SSH
# session closes — including the workers we nohup'd. macOS doesn't have
# this issue. The lab Macs work with plain nohup.
set -e

# host                       port  start  end  mode    sub_gguf            daemon_bin
WORKERS=(
  "prithvi-system-product-name 5530   0    6  first   /tmp/cuts5/w0.gguf  /home/prithvi/llama.cpp/build/bin/llama-nakshatra-worker"
  "mac3-2                       5531   6   12  middle  /tmp/w1.gguf        /Users/midev/llama.cpp/build/bin/llama-nakshatra-worker"
  "mentorings-imac-pro          5532  12   17  middle  /tmp/w2.gguf        /Users/mentoringinstitute/llama.cpp/build/bin/llama-nakshatra-worker"
  "mac4                         5533  17   23  middle  /tmp/w3.gguf        /Users/mi/llama.cpp/build/bin/llama-nakshatra-worker"
  "bishwa                       5534  23   28  last    /tmp/w4.gguf        /Users/MentoringInstitute/llama.cpp/build/bin/llama-nakshatra-worker"
)

MODEL_ID="prithvi-q8"
N_CTX="${N_CTX:-256}"
N_THREADS="${N_THREADS:-8}"

for w in "${WORKERS[@]}"; do
  read -r host port start end mode sub_gguf daemon_bin <<<"$w"
  echo "[start] $host:$port  mode=$mode  layers=[$start,$end)  sub_gguf=$sub_gguf"
  ssh "$host" "lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null; sleep 1; rm -f /tmp/worker_$port.log; cd ~/nakshatra-v0 && source venv/bin/activate && nohup python scripts/worker.py --port $port --sub-gguf $sub_gguf --mode $mode --layer-start $start --layer-end $end --model-id $MODEL_ID --daemon-bin $daemon_bin --n-ctx $N_CTX --n-threads $N_THREADS > /tmp/worker_$port.log 2>&1 < /dev/null &
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do grep -q 'M5 listening' /tmp/worker_$port.log 2>/dev/null && break; sleep 1; done
grep -q 'M5 listening' /tmp/worker_$port.log && echo \"[$host] READY\" || echo \"[$host] NOT_READY\"" &
done
wait

echo
echo "[start] running validator…"
ssh prithvi-system-product-name 'cd ~/nakshatra-v0 && source venv/bin/activate && python scripts/validate_cluster.py --config scripts/cluster_5worker.yaml'
