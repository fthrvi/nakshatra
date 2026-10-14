#!/bin/sh
set -eu

name=${0##*/}
target="${NAK_NODE_PREFIX:-$HOME/.nakshatra-node}/bin/$name"
if [ -x "$target" ]; then
    exec "$target" "$@"
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    printf '%s\n' "$name is provided by the signed Nakshatra release." \
        "Join this container first with: nakc join '<invite line>'"
    exit 0
fi

printf '%s\n' "Nakshatra is not joined yet; run: nakc join '<invite line>'" >&2
exit 2
