#!/usr/bin/env bash
# build-fabric.sh — Phase B cluster-wide builder for
# llama-nakshatra-worker-fabric. Run from a developer machine that
# has the nakshatra repo checked out + SSH to the cluster machines.
#
#   ./experiments/v0.0/build-fabric.sh                       # default cluster
#   ./experiments/v0.0/build-fabric.sh node-a node-e
#   ./experiments/v0.0/build-fabric.sh --local               # build on $HOSTNAME
#
# For each target machine:
#   1. rsync the new worker_daemon.cpp + shm_ring.hpp + the fabric
#      CMakeLists.txt into ~/llama.cpp/examples/nakshatra-spike/
#   2. SSH the target, set PATH to include /usr/local/bin (macOS
#      non-interactive SSH strips it), run cmake reconfigure +
#      build with -DNAKSHATRA_FABRIC_SHA + -DNAKSHATRA_FABRIC_BUILD_HOST
#   3. Verify the new binary via --version
#   4. Print summary
#
# Idempotent. Re-runs detect that the sibling target is already in
# CMakeLists.txt (the deployed file IS the patched one, so identical
# every time) + cmake's incremental build only relinks if needed.
#
# Designed to leave the existing llama-nakshatra-worker target's
# binary untouched on every cluster machine. The live v0.1 70B
# cluster on home-pc:5530 keeps using its May 8 binary.
set -euo pipefail

DEFAULT_HOSTS=(
    node-a
    node-e
    node-b
    node-c
    node-d
)

# Compute the nakshatra git SHA + this developer machine's hostname
# locally so every cluster machine gets the SAME stamp. THIS is the
# value smoke_daemon_version.py asserts is identical across the
# fleet — drift means someone built ad-hoc instead of going through
# this script.
HERE="$(cd "$(dirname "$0")/../.." && pwd)"  # ~/nakshatra
SHA="$(cd "$HERE" && git rev-parse --short HEAD)"
BUILD_HOST="$(hostname -s 2>/dev/null || hostname)"

if [[ $# -gt 0 && "$1" == "--local" ]]; then
    HOSTS=("local")
    shift
elif [[ $# -gt 0 ]]; then
    HOSTS=("$@")
else
    HOSTS=("${DEFAULT_HOSTS[@]}")
fi

# Files to deploy (worker_daemon.cpp + shm_ring.hpp + the fabric
# CMakeLists.txt that adds the sibling target). Local paths in this
# repo:
SRC_WORKER="$HERE/experiments/v0.0/worker_daemon.cpp"
SRC_SHM="$HERE/experiments/v0.0/shm_ring.hpp"
SRC_CMAKE="$HERE/experiments/v0.0/CMakeLists.fabric.txt"

for f in "$SRC_WORKER" "$SRC_SHM" "$SRC_CMAKE"; do
    if [[ ! -f "$f" ]]; then
        echo "missing source: $f" >&2
        exit 1
    fi
done

build_one() {
    local host="$1"
    local rsync_target ssh_cmd build_cmd
    if [[ "$host" == "local" ]]; then
        # Local build — just call cmake here.
        rsync_target=""
        ssh_cmd="bash -c"
    else
        rsync_target="$host:"
        ssh_cmd="ssh $host"
    fi
    echo "═══════════════════ $host ═══════════════════"

    # 1. Deploy. For local, just cp; for remote, rsync.
    local examples_dir='$HOME/llama.cpp/examples/nakshatra-spike'
    if [[ "$host" == "local" ]]; then
        eval "examples_dir_resolved=$examples_dir"
        cp "$SRC_WORKER" "$examples_dir_resolved/worker_daemon.cpp"
        cp "$SRC_SHM"    "$examples_dir_resolved/shm_ring.hpp"
        cp "$SRC_CMAKE"  "$examples_dir_resolved/CMakeLists.txt"
    else
        # Resolve the remote path via SSH first (depends on remote $HOME).
        local remote_examples
        remote_examples="$(ssh "$host" 'echo $HOME/llama.cpp/examples/nakshatra-spike')"
        rsync -a "$SRC_WORKER" "$host:$remote_examples/worker_daemon.cpp"
        rsync -a "$SRC_SHM"    "$host:$remote_examples/shm_ring.hpp"
        rsync -a "$SRC_CMAKE"  "$host:$remote_examples/CMakeLists.txt"
    fi

    # 2. Build. PATH override picks up /usr/local/bin on macOS where
    # non-interactive SSH strips it. cmake --version line proves the
    # right cmake is in scope.
    local build_script
    build_script=$(cat <<EOF
set -e
export PATH="\$PATH:/usr/local/bin"
cd \$HOME/llama.cpp/build
# Reconfigure to register the sibling target + capture the SHA +
# build-host defines. Idempotent — re-run noops if values unchanged.
cmake -S .. -B . \\
    -DNAKSHATRA_FABRIC_SHA=$SHA \\
    -DNAKSHATRA_FABRIC_BUILD_HOST=$BUILD_HOST \\
    2>&1 | tail -3
cmake --build . --target llama-nakshatra-worker-fabric -j 4 2>&1 | tail -5
echo "--- VERSION OUTPUT ---"
./bin/llama-nakshatra-worker-fabric --version
EOF
)
    if [[ "$host" == "local" ]]; then
        bash -c "$build_script"
    else
        ssh "$host" "$build_script"
    fi
    echo
}

for h in "${HOSTS[@]}"; do
    if ! build_one "$h"; then
        echo "BUILD FAILED on $h" >&2
        exit 1
    fi
done

echo "═══════════════════ summary ═══════════════════"
echo "SHA stamped: $SHA"
echo "BUILD_HOST stamped: $BUILD_HOST"
echo "hosts built: ${HOSTS[*]}"
echo
echo "Verify cluster-wide version consistency:"
echo "  python scripts/smoke_daemon_version.py"
