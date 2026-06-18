#!/usr/bin/env bash
# build-ijru-cuda.sh — build the patched `llama-nakshatra-worker` daemon on a
# fresh Linux x86-64 + NVIDIA box (operator-B "ijru"), for the cross-box serve
# proof (see deploy/crossbox-ijru.md). RUN ON ijru, from inside its nakshatra
# checkout: `bash deploy/build-ijru-cuda.sh`.
#
# What it does (idempotent — safe to re-run):
#   1. Clone llama.cpp + pin to c46583b (tag b8445) — the EXACT base the
#      partial-load patches target. Verified 2026-06-17: all 5 patches apply
#      clean at -p4 against this commit.
#   2. Apply the 5 m4 partial-load patches (Llama arch — NOT the qwen3moe ones;
#      DeepSeek-R1-Distill-Llama is a Llama-arch model).
#   3. Drop the fabric example (worker_daemon.cpp + shm_ring.hpp + the fabric
#      CMakeLists) into examples/nakshatra-spike/ and register it.
#   4. Configure with CUDA (-DGGML_CUDA=ON) + build target llama-nakshatra-worker.
#   5. Print the binary path + --version.
#
# The partial-load patches are backend-agnostic — the ONLY difference from the
# hub's ROCm build is the GGML_CUDA configure flag. Prereqs on ijru: git, cmake
# (>=3.24 for CUDA_ARCHITECTURES=native), a C++17 toolchain, and the CUDA
# toolkit (nvcc). `nvcc --version` must work before running this.
set -euo pipefail

NKS="${NKS:-$(cd "$(dirname "$0")/.." && pwd)}"   # nakshatra repo root (this checkout)
LLAMA="${LLAMA:-$HOME/llama.cpp}"
BASE_COMMIT=c46583b
PATCHES="$NKS/experiments/v0.0/m4_patches"
SPIKE_SRC="$NKS/experiments/v0.0"
CUDA_ARCH="${CUDA_ARCH:-native}"                  # override e.g. CUDA_ARCH=86 for a 3090

command -v nvcc >/dev/null || { echo "FATAL: nvcc not found — install the CUDA toolkit first." >&2; exit 1; }
command -v cmake >/dev/null || { echo "FATAL: cmake not found." >&2; exit 1; }

# ── 1. base llama.cpp pinned to c46583b ──
if [[ ! -d "$LLAMA/.git" ]]; then
    echo "── cloning llama.cpp → $LLAMA ──"
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
cd "$LLAMA"
git fetch --tags origin >/dev/null 2>&1 || true
# Only hard-checkout if we're not already at the base (don't clobber a dirty re-run).
if [[ "$(git rev-parse --short HEAD)" != "$BASE_COMMIT" ]] && git diff --quiet 2>/dev/null; then
    git checkout --detach "$BASE_COMMIT"
fi
echo "llama.cpp at $(git rev-parse --short HEAD) (want $BASE_COMMIT)"

# ── 2. apply the 5 Llama partial-load patches (-p4, --forward = idempotent) ──
for pf in llama-graph.cpp.patch llama-model.cpp.patch llama-model.h.patch \
          llama-model-loader.cpp.patch models_llama.cpp.patch; do
    if patch -p4 --forward --reject-file=- --silent -i "$PATCHES/$pf" 2>/dev/null; then
        echo "  applied $pf"
    else
        # --forward exits non-zero when already applied; confirm that's the case.
        if patch -p4 --reverse --dry-run --silent -i "$PATCHES/$pf" >/dev/null 2>&1; then
            echo "  already applied $pf (skip)"
        else
            echo "FATAL: $pf did not apply and is not already applied — base drift?" >&2; exit 1
        fi
    fi
done

# ── 3. fabric example dir + registration ──
mkdir -p "$LLAMA/examples/nakshatra-spike"
cp "$SPIKE_SRC/worker_daemon.cpp"      "$LLAMA/examples/nakshatra-spike/worker_daemon.cpp"
cp "$SPIKE_SRC/shm_ring.hpp"           "$LLAMA/examples/nakshatra-spike/shm_ring.hpp"
cp "$SPIKE_SRC/CMakeLists.fabric.txt"  "$LLAMA/examples/nakshatra-spike/CMakeLists.txt"
grep -q "add_subdirectory(nakshatra-spike)" "$LLAMA/examples/CMakeLists.txt" \
    || echo "add_subdirectory(nakshatra-spike)" >> "$LLAMA/examples/CMakeLists.txt"

# ── 4. configure (CUDA) + build the worker target ──
SHA="$(cd "$NKS" && git rev-parse --short HEAD)"
HOST="$(hostname -s 2>/dev/null || hostname)"
cmake -S "$LLAMA" -B "$LLAMA/build" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNAKSHATRA_FABRIC_SHA="$SHA" -DNAKSHATRA_FABRIC_BUILD_HOST="$HOST"
cmake --build "$LLAMA/build" --target llama-nakshatra-worker -j"$(nproc)"

# ── 5. verify ──
BIN="$LLAMA/build/bin/llama-nakshatra-worker"
echo "── built: $BIN ──"
"$BIN" --version
echo
echo "✅ daemon built. Use it as --daemon-bin in the worker.py launch (deploy/crossbox-ijru.md §D)."
echo "   On a NVIDIA box, run the worker with GPU offload: --n-gpu-layers 99 (not 0)."
