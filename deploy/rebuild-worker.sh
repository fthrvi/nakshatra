#!/usr/bin/env bash
# Force-recompile the nakshatra worker daemon from CANONICAL source, reliably.
#
# Fixes the recurring "stale .o relink" gremlin: a cp + cmake build within the same
# wall-clock second left the copied source and the prior object file with identical
# second-granularity mtimes, so cmake/ninja concluded "nothing changed" and relinked
# the OLD object — shipping a daemon without the latest edit (cost us 3 debugging
# detours). This script removes all ambiguity: copy canonical -> examples, bump the
# mtimes, delete the object, build, and VERIFY the binary's mtime actually advanced
# (hard-fail if it didn't, so a silent stale build can never happen again).
#
# Usage:   bash rebuild-worker.sh                  # hub ROCm build (~/llama.cpp/build)
#          BUILD=~/llama.cpp/build-cuda bash rebuild-worker.sh   # ijru CUDA build
set -euo pipefail
LLAMA="${LLAMA:-$HOME/llama.cpp}"
BUILD="${BUILD:-$LLAMA/build}"
SRC="${SRC:-$HOME/nakshatra/experiments/v0.0}"
DST="$LLAMA/examples/nakshatra-spike"
CM="$(command -v cmake || echo "$HOME/.local/bin/cmake")"
BIN="$BUILD/bin/llama-nakshatra-worker"

before=$(stat -c %Y "$BIN" 2>/dev/null || echo 0)
cp "$SRC/worker_daemon.cpp" "$SRC/shm_ring.hpp" "$SRC/spike.cpp" "$SRC/m4_chain.cpp" "$DST/"
touch "$DST"/*.cpp "$DST"/*.hpp                       # guarantee source newer than any .o
find "$BUILD" -name 'worker_daemon.cpp.o' -delete 2>/dev/null || true   # belt + suspenders
"$CM" --build "$BUILD" --target llama-nakshatra-worker -j"$(nproc)"
after=$(stat -c %Y "$BIN")

if [ "$after" -gt "$before" ]; then
    echo "OK: worker daemon rebuilt — binary mtime advanced ($before -> $after) at $BIN"
else
    echo "FAIL: binary mtime did NOT advance ($before == $after) — stale build!" >&2
    exit 1
fi
