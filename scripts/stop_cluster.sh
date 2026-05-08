#!/usr/bin/env bash
# Cleanly stop all Nakshatra workers across the 5-machine cluster.
# Using lsof-by-port instead of pkill-by-pattern because pkill -f matches
# its own command line, killing the bash that runs it.
HOSTS_PORTS=(
  "node-a 5530"
  "node-b                       5531"
  "node-c          5532"
  "node-d                         5533"
  "node-e                       5534"
)
for hp in "${HOSTS_PORTS[@]}"; do
  read -r h port <<<"$hp"
  ssh "$h" "lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null; pgrep -f llama-nakshatra-worker 2>/dev/null | xargs -r kill -9 2>/dev/null; true" 2>/dev/null
  echo "[stop] $h (:$port)"
done
echo "[stop] done"
