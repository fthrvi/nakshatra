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
# Isolate git completely (Codex review 2026-10-03): no inherited GIT_* env, no global/system config, no
# templates, no hooks — nothing from the caller's environment may run during fetch/checkout/am or change files.
for v in $(env | grep -o '^GIT_[A-Z_]*' || true); do unset "$v"; done
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
G() { git -c core.hooksPath=/dev/null -c core.fsmonitor=false -c protocol.file.allow=always "$@"; }
# "Already the engine" means: the committed tree is TREE *and* the working tree is exactly that commit
# (no modified, deleted or untracked files; ignored build output is fine). Anything else is rebuilt.
is_engine() {
  [ -d "$1/.git" ] && [ "$(G -C "$1" rev-parse 'HEAD^{tree}' 2>/dev/null)" = "$TREE" ] \
    && [ -z "$(G -C "$1" status --porcelain 2>/dev/null)" ]
}
if is_engine "$DEST"; then
  echo "[engine] $DEST already is the engine (tree ${TREE:0:12}, working tree clean)"; exit 0
fi
rm -rf "$DEST.tmp"; mkdir -p "$DEST.tmp"
G init -q --template= "$DEST.tmp"
G -C "$DEST.tmp" fetch -q --depth 1 "$UP" "$COMMIT" 2>/dev/null || G -C "$DEST.tmp" fetch -q "$UP" "$COMMIT"
G -C "$DEST.tmp" checkout -q --detach FETCH_HEAD
[ "$(G -C "$DEST.tmp" rev-parse HEAD)" = "$COMMIT" ] || { echo "[engine] REFUSING: upstream gave a different commit"; rm -rf "$DEST.tmp"; exit 1; }
G -C "$DEST.tmp" -c user.email=engine@nakshatra -c user.name=engine am -q "$HERE"/patches/*.patch
GOT="$(G -C "$DEST.tmp" rev-parse 'HEAD^{tree}')"
if [ "$GOT" != "$TREE" ] || ! is_engine "$DEST.tmp"; then
  echo "[engine] REFUSING: source tree $GOT is not the engine $TREE"; rm -rf "$DEST.tmp"; exit 1
fi
rm -rf "$DEST"; mv "$DEST.tmp" "$DEST"
echo "[engine] $DEST = upstream ${COMMIT:0:12} + $(ls "$HERE"/patches/*.patch | wc -l) patches (tree ${TREE:0:12}, verified)"
