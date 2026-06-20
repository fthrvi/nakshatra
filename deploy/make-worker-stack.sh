#!/usr/bin/env bash
# make-worker-stack.sh - (re)generate the vendored patched-llama.cpp SOURCE tarball that
# provision-worker.sh fetches. Run on a box that has the proven patched ~/llama.cpp checkout (the
# partial-load patch applied + the nakshatra-spike example present). Output is content-addressed.
#
# Hosting: place the resulting tarball where a freshly-joining worker can fetch it over ITS OWN
# internet (like the model download) - the public onboard host is the natural home:
#   scp worker-llama-stack.tgz  <vps>:/var/www/onboard/   (served at https://prithviloka.net/onboard/)
# (A relay client can't pull arbitrary mesh hosts - the PRITHVI_RELAY firewall whitelists it - so the
#  stack must come from a public URL or an allowed host, NOT an internal mesh IP.)
set -euo pipefail
SRC="${1:-$HOME/llama.cpp}"
OUT="${2:-$HOME/.nakshatra/worker-llama-stack.tgz}"
[ -f "$SRC/examples/nakshatra-spike/worker_daemon.cpp" ] || {
  echo "no patched llama.cpp at $SRC (need the partial-load patch + nakshatra-spike example)" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
tar czf "$OUT" -C "$SRC" \
  --exclude='./build' --exclude='./.git' --exclude='*.gguf' --exclude='*.o' --exclude='*.orig' \
  --exclude='*.metallib' --exclude='./models' .
printf 'wrote %s (%.1f MB)\nsha256 %s\n' "$OUT" \
  "$(echo "scale=1; $(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT")/1048576" | bc 2>/dev/null || echo '?')" \
  "$(sha256sum "$OUT" 2>/dev/null | cut -c1-16 || shasum -a256 "$OUT" | cut -c1-16)"

# the SERVE scripts bundle (worker.py + pb2 stubs + fabric/packaging) - public, non-secret. The model
# + roster are the gated parts; these are just the protocol so a dial-out worker can run worker.py.
SCRIPTS_SRC="${NAKSHATRA_SCRIPTS:-$HOME/nakshatra/scripts}"
SCRIPTS_OUT="$(dirname "$OUT")/worker-scripts.tgz"
if [ -f "$SCRIPTS_SRC/worker.py" ]; then
  tar czf "$SCRIPTS_OUT" -C "$SCRIPTS_SRC" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='*.gguf' --exclude='test_*' .
  printf 'wrote %s (%.1f MB)\n' "$SCRIPTS_OUT" \
    "$(echo "scale=1; $(stat -f%z "$SCRIPTS_OUT" 2>/dev/null || stat -c%s "$SCRIPTS_OUT")/1048576" | bc 2>/dev/null || echo '?')"
fi
