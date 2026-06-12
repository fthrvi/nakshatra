#!/usr/bin/env bash
# Stand up Prithvi's "unconscious tier" — the Nakshatra mesh-backed
# /v1/chat/completions endpoint his think_deeper seam escalates to (fork B,
# connect L2→L4). Installs 2 layer-split workers + the OpenAI serve as
# always-on systemd --user services.
#
#   ./deploy/install-unconscious.sh           # install + enable + start
#   ./deploy/install-unconscious.sh status    # show services + a probe
#   ./deploy/install-unconscious.sh stop       # stop all three
#
# Prereqs: ~/.nakshatra/models/{node-A,node-B,whole-1b}.gguf (the split + tokenizer),
# the llama-nakshatra-worker daemon at ~/llama.cpp/build/bin/, and the venv.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
NKS="$HOME/.nakshatra"
PORT=11599

cmd="${1:-install}"

case "$cmd" in
  status)
    systemctl --user --no-pager status \
      nakshatra-unconscious-worker@a nakshatra-unconscious-worker@b nakshatra-unconscious \
      2>/dev/null | grep -E 'Active:|nakshatra-unconscious' || true
    echo; echo "── probe /v1/models ──"
    curl -s "http://127.0.0.1:${PORT}/v1/models" || echo "(unreachable)"; echo
    exit 0 ;;
  stop)
    systemctl --user stop nakshatra-unconscious \
      nakshatra-unconscious-worker@a nakshatra-unconscious-worker@b || true
    echo "stopped."; exit 0 ;;
  install) : ;;
  *) echo "usage: install-unconscious.sh [install|status|stop]" >&2; exit 2 ;;
esac

echo "[install] repo=$REPO"
mkdir -p "$UNIT_DIR" "$NKS/models"

# sanity: assets present
for f in "$NKS/models/node-A.gguf" "$NKS/models/node-B.gguf" "$NKS/models/whole-1b.gguf" \
         "$REPO/.venv/bin/python" "$HOME/llama.cpp/build/bin/llama-nakshatra-worker" \
         "$NKS/serve_models.unconscious.yaml"; do
  [ -e "$f" ] || { echo "[install] ERROR: missing $f" >&2; exit 1; }
done

# per-worker env (the layer split)
cat > "$NKS/unconscious-worker-a.env" <<EOF
PORT=5530
SUB_GGUF=$NKS/models/node-A.gguf
MODE=first
LAYER_START=0
LAYER_END=8
EOF
cat > "$NKS/unconscious-worker-b.env" <<EOF
PORT=5531
SUB_GGUF=$NKS/models/node-B.gguf
MODE=last
LAYER_START=8
LAYER_END=16
EOF

install -m 0644 "$REPO/deploy/systemd/nakshatra-unconscious-worker@.service" "$UNIT_DIR/"
install -m 0644 "$REPO/deploy/systemd/nakshatra-unconscious.service" "$UNIT_DIR/"

systemctl --user daemon-reload
systemctl --user enable --now nakshatra-unconscious-worker@a.service
systemctl --user enable --now nakshatra-unconscious-worker@b.service
sleep 6   # workers load their sub-GGUFs
systemctl --user enable --now nakshatra-unconscious.service

echo "[install] done. Probe:  ./deploy/install-unconscious.sh status"
echo "[install] then point Prithvi: PRITHVI_UNCONSCIOUS_ENABLED=1 + URL=http://127.0.0.1:${PORT}"
