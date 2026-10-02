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
rm -rf "$DEST.tmp"; mkdir -p "$DEST.tmp"
G init -q --template= "$DEST.tmp"
G -C "$DEST.tmp" fetch -q --depth 1 "$UP" "$COMMIT" 2>/dev/null || G -C "$DEST.tmp" fetch -q "$UP" "$COMMIT"
G -C "$DEST.tmp" checkout -q --detach FETCH_HEAD
[ "$(G -C "$DEST.tmp" rev-parse HEAD)" = "$COMMIT" ] || { echo "[engine] REFUSING: upstream gave a different commit"; rm -rf "$DEST.tmp"; exit 1; }
G -C "$DEST.tmp" -c user.email=engine@nakshatra -c user.name=engine am -q "$HERE"/patches/*.patch
GOT="$(G -C "$DEST.tmp" rev-parse 'HEAD^{tree}')"
DIRTY="$(G -C "$DEST.tmp" status --porcelain --ignored --untracked-files=all)" || { echo "[engine] REFUSING: git status failed"; rm -rf "$DEST.tmp"; exit 1; }
if [ "$GOT" != "$TREE" ] || [ -n "$DIRTY" ]; then
  echo "[engine] REFUSING: source tree $GOT is not the engine $TREE"; rm -rf "$DEST.tmp"; exit 1
fi
rm -rf "$DEST"; mv "$DEST.tmp" "$DEST"
echo "[engine] $DEST = upstream ${COMMIT:0:12} + $(ls "$HERE"/patches/*.patch | wc -l) patches (tree ${TREE:0:12}, verified)"
