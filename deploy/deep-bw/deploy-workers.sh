#!/usr/bin/env bash
# deploy-workers.sh [--check] - put THIS checkout's worker code on the two chain nodes and PROVE it landed.
#
# ijru and blackwell run hand-copied worker.py trees, and they drifted (hub 3777 lines, blackwell 3357, ijru 3443 in one day):
# a worker fix landed on main and never reached the boxes that run the chain. The set copied is derived from what worker.py
# actually imports (nakshatra_*.py + wire/), not a hand list - a check keyed on worker.py alone would report "verified" while
# nakshatra_grpc_auth.py / the pb2 stubs were stale (the self-certifying-check trap).
#
#   deploy-workers.sh           copy every file, then verify sha256 of each on each node (exit 1 on ANY mismatch)
#   deploy-workers.sh --check   verify only, change nothing
#
# ijru:      ~/nakshatra/scripts/                 (ssh alias ijru-nks)
# blackwell: ~/.nakshatra-worker/nakshatra-scripts/  inside WSL (ssh alias blackwell; the login shell is PowerShell, so scripts
#            are piped into `wsl -e bash -l -s`, file bodies as base64)
# A running worker keeps its already-loaded code: deploying does not disturb a live chain; the NEXT start uses the new files.
set -euo pipefail
cd "$(dirname "$0")/../../scripts"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1

mapfile -t FILES < <({ echo worker.py; grep -ohE '^\s*(import|from) nakshatra_[a-z_0-9]+' worker.py | awk '{print $2".py"}'; echo wire/handshake.py; } | sort -u)
for f in "${FILES[@]}"; do [ -f "$f" ] || { echo "missing local file: $f" >&2; exit 2; }; done
echo "files: ${FILES[*]}"
declare -A LOCAL; for f in "${FILES[@]}"; do LOCAL[$f]=$(sha256sum "$f" | cut -d' ' -f1); done
bad=0

verify() { # node label, then lines "sha  file" from the node on stdin
  local label=$1 n=0 line sha f
  while read -r sha f; do
    [ -n "${f:-}" ] || continue
    f=${f#./}
    if [ "${LOCAL[$f]:-}" = "$sha" ]; then n=$((n+1)); else echo "  MISMATCH $label $f (node ${sha:0:10} vs here ${LOCAL[$f]:0:10})"; bad=1; fi
  done
  [ "$n" -eq "${#FILES[@]}" ] && echo "  $label: all ${#FILES[@]} files match" || { echo "  $label: only $n/${#FILES[@]} match"; bad=1; }
}

echo "== ijru"
if [ $CHECK -eq 0 ]; then
  ssh -o BatchMode=yes ijru-nks 'mkdir -p ~/nakshatra/scripts/wire'
  for f in "${FILES[@]}"; do
    ssh -o BatchMode=yes ijru-nks "[ -f ~/nakshatra/scripts/$f.bak-deploy ] || cp -n ~/nakshatra/scripts/$f ~/nakshatra/scripts/$f.bak-deploy 2>/dev/null; true"
    scp -q -o BatchMode=yes "$f" "ijru-nks:nakshatra/scripts/$f"
  done
fi
# process substitution, NOT a pipe: a pipe runs verify in a subshell and its `bad=1` never reached the final verdict, so a
# mismatching node still ended in "OK" (caught the first time this ran).
verify ijru < <(ssh -o BatchMode=yes ijru-nks "cd ~/nakshatra/scripts && sha256sum ${FILES[*]}")

echo "== blackwell"
if [ $CHECK -eq 0 ]; then
  {
    echo 'D=$HOME/.nakshatra-worker/nakshatra-scripts; mkdir -p $D/wire'
    for f in "${FILES[@]}"; do
      echo "[ -f \$D/$f.bak-deploy ] || cp -n \$D/$f \$D/$f.bak-deploy 2>/dev/null"
      echo "base64 -d > \$D/$f <<'B64_EOF'"; base64 -w0 "$f"; echo; echo "B64_EOF"
    done
  } | ssh blackwell 'wsl -e bash -l -s' >/dev/null
fi
verify blackwell < <({ echo 'cd $HOME/.nakshatra-worker/nakshatra-scripts && sha256sum '"${FILES[*]}"; } | ssh blackwell 'wsl -e bash -l -s' 2>&1 | tr -d '\r' | grep -E '^[0-9a-f]{64} ' || true)

[ $bad -eq 0 ] && echo "OK: both chain nodes run this checkout's worker code" || { echo "FAILED: node code differs from this checkout" >&2; exit 1; }
