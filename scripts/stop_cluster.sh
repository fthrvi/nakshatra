#!/usr/bin/env bash
# Cleanly stop all Nakshatra workers across the 5-machine cluster.
# Using lsof-by-port instead of pkill-by-pattern because pkill -f matches
# its own command line, killing the bash that runs it.
HOSTS_PORTS=(
  "prithvi-system-product-name 5530"
  "mac3-2                       5531"
  "mentorings-imac-pro          5532"
  "mac4                         5533"
  "bishwa                       5534"
)
for hp in "${HOSTS_PORTS[@]}"; do
  read -r h port <<<"$hp"
  ssh "$h" "lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null; pgrep -f llama-nakshatra-worker 2>/dev/null | xargs -r kill -9 2>/dev/null; true" 2>/dev/null
  echo "[stop] $h (:$port)"
done
echo "[stop] done"
