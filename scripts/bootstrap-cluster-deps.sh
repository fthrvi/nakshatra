#!/usr/bin/env bash
# Bootstrap the SPKI-arc runtime deps into each cluster machine's
# existing ~/nakshatra-v0/venv. Pip-installs cryptography>=41,<46 —
# the only runtime dep nakshatra_tls + Phase 3 SPKI added on top of
# the pre-2026-05-21 venv contents.
#
# Idempotent: skips machines where the cryptography import already
# succeeds at a version inside the supported range. Re-runnable
# after a cluster rebuild or after a python upgrade that rebuilt
# the venv.
#
# Drive-by from the 2026-05-21 retro list. Without this, every
# fresh cluster smoke has to manually pip-install cryptography on
# each machine before workers will boot.
#
# Usage:
#   scripts/bootstrap-cluster-deps.sh             # all 5 machines
#   scripts/bootstrap-cluster-deps.sh node-e node-d # subset
#   DRY_RUN=1 scripts/bootstrap-cluster-deps.sh   # print, don't exec
#
# Per-host overrides (rarely needed):
#   VENV_PATH=~/custom-venv scripts/bootstrap-cluster-deps.sh hostname
set -euo pipefail

DEFAULT_HOSTS=(
  node-a
  node-b
  node-c
  node-d
  node-e
)

VENV_PATH="${VENV_PATH:-~/nakshatra-v0/venv}"
DRY_RUN="${DRY_RUN:-0}"

if [[ $# -gt 0 ]]; then
  HOSTS=("$@")
else
  HOSTS=("${DEFAULT_HOSTS[@]}")
fi

# Single-quoted heredoc so it expands ONLY on the remote shell —
# $VENV_PATH below is the remote env var, not the local one.
# Exits 0 on already-installed, 0 on freshly-installed, non-zero
# on actual failure.
read -r -d '' REMOTE_CMD <<'EOF' || true
set -e
VENV="$1"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "  ✗ venv not found at $VENV — skipping"
  exit 2
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
CURRENT=$(python -c 'import cryptography; print(cryptography.__version__)' 2>/dev/null || true)
if [[ -n "$CURRENT" ]]; then
  # Already installed. Accept any version in the supported range; pip
  # itself enforces the bound on a fresh install, but a pre-existing
  # install might be older than what nakshatra_tls needs.
  if python -c "
import sys
from packaging.version import Version
v = Version('$CURRENT')
sys.exit(0 if Version('41') <= v < Version('46') else 1)
" 2>/dev/null; then
    echo "  ✓ cryptography $CURRENT already installed (in supported range)"
    exit 0
  fi
  echo "  · cryptography $CURRENT out of range; upgrading"
fi
pip install --quiet 'cryptography>=41,<46'
NEW=$(python -c 'import cryptography; print(cryptography.__version__)')
echo "  ✓ cryptography $NEW installed"
EOF

FAILED=()
for host in "${HOSTS[@]}"; do
  echo "[$host]"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (dry run) ssh $host bash -s -- $VENV_PATH"
    continue
  fi
  # Expand $VENV_PATH locally so ~ resolves on the remote shell
  # via passing the literal string; remote shell tilde-expands.
  if ! ssh -o ConnectTimeout=10 "$host" bash -s -- "$VENV_PATH" <<< "$REMOTE_CMD"; then
    FAILED+=("$host")
  fi
done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed on: ${FAILED[*]}"
  exit 1
fi
echo "All ${#HOSTS[@]} hosts bootstrapped."
