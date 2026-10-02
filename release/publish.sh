#!/usr/bin/env bash
# publish.sh — mirror the LATEST signed release of a channel to a public release host.
#
#   release/publish.sh [channel] [ssh-host] [remote-dir]       (defaults: canary vpsjump /srv/nakshatra-dist)
#
# The host is UNTRUSTED: nodes verify every file against the release key their invite pinned, so the
# host can withhold an update but never forge one. Only the latest TWO versions are kept remotely
# (disk on the relay VPS is tight); older ones stay on the build machine.
set -euo pipefail
CH="${1:-canary}"; HOST="${2:-vpsjump}"; DIR="${3:-/srv/nakshatra-dist}"
DIST="${NAK_DIST:-$HOME/nakshatra-dist}"
VER=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['version'])" "$DIST/$CH/latest.json")
ssh "$HOST" "mkdir -p '$DIR/$CH'"
rsync -a --delete "$DIST/$CH/$VER/" "$HOST:$DIR/$CH/$VER/"
rsync -a "$DIST/$CH/latest.json" "$HOST:$DIR/$CH/latest.json.tmp"
ssh "$HOST" "mv '$DIR/$CH/latest.json.tmp' '$DIR/$CH/latest.json' && cd '$DIR/$CH' && ls -1d */ | sed 's#/##' | sort -V | head -n -2 | xargs -r rm -rf && chmod -R a+rX '$DIR'"
echo "published $CH $VER to $HOST:$DIR"
