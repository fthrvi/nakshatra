#!/usr/bin/env bash
# Install the always-on Nakshatra mesh as systemd --user services (v1.1 capstone).
#
#   ./deploy/install-mesh.sh            # install + enable + start relay & meshd
#   ./deploy/install-mesh.sh status     # show service + mesh status
#   ./deploy/install-mesh.sh stop        # stop both services
#
# Idempotent. Creates ~/.nakshatra/{relay,meshd.env} on first run (edit meshd.env
# then `systemctl --user restart nakshatra-meshd`). For 24/7 across logout, run
# once (needs sudo):  loginctl enable-linger "$USER"
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
NKS_DIR="$HOME/.nakshatra"
ENV_FILE="$NKS_DIR/meshd.env"

cmd="${1:-install}"

case "$cmd" in
  status)
    echo "── services ──"
    systemctl --user --no-pager status nakshatra-relay nakshatra-meshd \
      | grep -E 'Active:|Main PID:|nakshatra-' || true
    echo; echo "── mesh status ($NKS_DIR/mesh-status.json) ──"
    if [ -f "$NKS_DIR/mesh-status.json" ]; then
      python3 -m json.tool "$NKS_DIR/mesh-status.json"
    else
      echo "(no status file yet — meshd may still be starting)"
    fi
    exit 0 ;;
  stop)
    systemctl --user stop nakshatra-meshd nakshatra-relay || true
    echo "stopped."; exit 0 ;;
  install) : ;;
  *) echo "usage: install-mesh.sh [install|status|stop]" >&2; exit 2 ;;
esac

echo "[install] repo=$REPO"
mkdir -p "$UNIT_DIR" "$NKS_DIR/relay"

# venv sanity
if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "[install] ERROR: $REPO/.venv/bin/python missing — create the venv first." >&2
  exit 1
fi

# seed the per-node env on first run
if [ ! -f "$ENV_FILE" ]; then
  cp "$REPO/deploy/meshd.env.example" "$ENV_FILE"
  echo "[install] seeded $ENV_FILE (edit DRIFT_CLASS / WORKER_ADDR, then restart meshd)"
fi

# install unit files (they reference %h, so no path rewriting needed)
install -m 0644 "$REPO/deploy/systemd/nakshatra-relay.service" "$UNIT_DIR/"
install -m 0644 "$REPO/deploy/systemd/nakshatra-meshd.service" "$UNIT_DIR/"

systemctl --user daemon-reload
systemctl --user enable --now nakshatra-relay.service
systemctl --user enable --now nakshatra-meshd.service

echo "[install] done. Check:  ./deploy/install-mesh.sh status"
echo "[install] 24/7 across logout:  sudo loginctl enable-linger $USER"
