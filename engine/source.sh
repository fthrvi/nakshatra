#!/usr/bin/env bash
# source.sh DEST — produce the exact Nakshatra engine source tree in DEST (a git checkout).
#
#   upstream llama.cpp @ COMMIT (engine/BASE)  +  engine/patches/*.patch  ==  TREE   (or refuse)
#
# The tree hash covers every file, so whatever served the upstream source (GitHub, a mirror, a local
# clone via ENGINE_UPSTREAM=/path) cannot change what gets built: a mismatch is refused, nothing is kept.
# Rebuilding the engine is: change the fork, `git format-patch COMMIT..HEAD -o engine/patches`, update TREE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:?usage: source.sh DEST}"
# shellcheck disable=SC1091
. "$HERE/BASE"
UP="${ENGINE_UPSTREAM:-$UPSTREAM}"
command -v git >/dev/null || { echo "[engine] REFUSING: git is required to build the verified engine source"; exit 1; }
# Threat model: the SOURCE (upstream, a mirror, the network, a tarball) and stale or damaged LOCAL state
# are untrusted. A local user who can already write these files could replace the built binary directly;
# that is out of scope. So: NO fast path — every run builds a FRESH tree, verifies it, and atomically
# REPLACES $DEST (any old checkout, git metadata and build cache go with it). (Codex review rounds 1+2.)
# Isolate git completely: no inherited GIT_* variables, no global/system config, no templates, no hooks.
while IFS= read -r v; do unset "$v"; done < <(compgen -e | grep '^GIT_' || true)
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
G() { git -c core.hooksPath=/dev/null -c core.fsmonitor=false "$@"; }
# One run at a time per DEST (mkdir is atomic everywhere, incl. macOS without flock); a UNIQUE staging
# dir per run; both cleaned up on any exit. (Codex round 3: a shared DEST.tmp let concurrent runs mix trees.)
mkdir -p "$(dirname "$DEST")"
LOCK="$DEST.lock"; STAGE=""; OLD=""; HAVE_LOCK=""
cleanup() {
  # An interrupted swap must never leave DEST missing: put the previous tree back.
  if [ -n "$OLD" ] && [ -e "$OLD" ] && [ ! -e "$DEST" ]; then mv "$OLD" "$DEST"; fi
  [ -n "$STAGE" ] && rm -rf "$STAGE"
  [ -n "$HAVE_LOCK" ] && rm -rf "$LOCK"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
take_lock() { mkdir "$LOCK" 2>/dev/null && echo $$ > "$LOCK/pid" && HAVE_LOCK=1; }
if ! take_lock; then
  # A lock whose owner is gone (crash, kill -9, reboot) is stale: take it over; a live owner wins.
  owner="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    echo "[engine] another engine build (pid $owner) is running for $DEST"; exit 1
  fi
  rm -rf "$LOCK"; take_lock || { echo "[engine] could not take $LOCK"; exit 1; }
fi
STAGE="$(mktemp -d "$DEST.stage.XXXXXX")"
G init -q --template= "$STAGE"
G -C "$STAGE" fetch -q --depth 1 "$UP" "$COMMIT" 2>/dev/null || G -C "$STAGE" fetch -q "$UP" "$COMMIT"
G -C "$STAGE" checkout -q --detach FETCH_HEAD
[ "$(G -C "$STAGE" rev-parse HEAD)" = "$COMMIT" ] || { echo "[engine] REFUSING: upstream gave a different commit"; exit 1; }
G -C "$STAGE" -c user.email=engine@nakshatra -c user.name=engine am -q "$HERE"/patches/*.patch
GOT="$(G -C "$STAGE" rev-parse 'HEAD^{tree}')"
DIRTY="$(G -C "$STAGE" status --porcelain --ignored --untracked-files=all)" || { echo "[engine] REFUSING: git status failed"; exit 1; }
if [ "$GOT" != "$TREE" ] || [ -n "$DIRTY" ]; then
  echo "[engine] REFUSING: source tree $GOT is not the engine $TREE"; exit 1
fi
# Commit: keep the old tree until the verified one is in place; put it back if the move fails.
if [ -e "$DEST" ] || [ -L "$DEST" ]; then
  OLD="$(mktemp -d "$DEST.old.XXXXXX")"; rmdir "$OLD"          # a unique, unused name (no PID reuse)
  mv "$DEST" "$OLD"
fi
if ! mv "$STAGE" "$DEST"; then
  echo "[engine] REFUSING: could not move the verified tree into place; previous tree restored"; exit 1
fi
STAGE=""
[ -n "$OLD" ] && rm -rf "$OLD"; OLD=""
echo "[engine] $DEST = upstream ${COMMIT:0:12} + $(ls "$HERE"/patches/*.patch | wc -l) patches (tree ${TREE:0:12}, verified)"
