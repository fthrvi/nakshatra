#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
cd build
cmake .. > /dev/null
cmake --build . -j"$(nproc)"
echo
echo "built: $(pwd)/cb_eval_observe"
