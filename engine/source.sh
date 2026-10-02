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
if [ -d "$DEST/.git" ] && [ "$(git -C "$DEST" rev-parse 'HEAD^{tree}' 2>/dev/null)" = "$TREE" ]; then
  echo "[engine] $DEST already is the engine (tree ${TREE:0:12})"; exit 0
fi
rm -rf "$DEST.tmp"; mkdir -p "$DEST.tmp"
git -C "$DEST.tmp" init -q
git -C "$DEST.tmp" fetch -q --depth 1 "$UP" "$COMMIT" 2>/dev/null || git -C "$DEST.tmp" fetch -q "$UP" "$COMMIT"
git -C "$DEST.tmp" checkout -q --detach FETCH_HEAD
[ "$(git -C "$DEST.tmp" rev-parse HEAD)" = "$COMMIT" ] || { echo "[engine] REFUSING: upstream gave a different commit"; rm -rf "$DEST.tmp"; exit 1; }
git -C "$DEST.tmp" -c user.email=engine@nakshatra -c user.name=engine am -q "$HERE"/patches/*.patch
GOT="$(git -C "$DEST.tmp" rev-parse 'HEAD^{tree}')"
if [ "$GOT" != "$TREE" ]; then
  echo "[engine] REFUSING: source tree $GOT is not the engine $TREE"; rm -rf "$DEST.tmp"; exit 1
fi
rm -rf "$DEST"; mv "$DEST.tmp" "$DEST"
echo "[engine] $DEST = upstream ${COMMIT:0:12} + $(ls "$HERE"/patches/*.patch | wc -l) patches (tree ${TREE:0:12}, verified)"
