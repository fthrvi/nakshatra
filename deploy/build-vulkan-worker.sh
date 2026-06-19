#!/usr/bin/env bash
# build-vulkan-worker.sh — build the patched `llama-nakshatra-worker` on the UNIVERSAL Vulkan
# backend, so a node of ANY vendor (Intel Arc, AMD, NVIDIA, … anything with a Vulkan driver) can
# serve layers. This is the load-bearing build for the heterogeneous-fleet thesis: our partial-load
# patch is backend-AGNOSTIC (frontend-only — loader/graph, zero ggml-backend code), so the SAME
# patched source compiles against Vulkan with no engine changes.
#
# Isolated from the live ROCm build: builds into $LLAMA/build-vulkan (a separate build dir), so the
# running ROCm `llama-nakshatra-worker` is never touched. Idempotent — safe to re-run.
#
# PREREQ (the only thing this needs that a ROCm/CUDA box may lack — the Vulkan SDK / shader toolchain):
#   sudo apt install vulkan-headers glslc glslang-tools shaderc   (Debian/Ubuntu)
#   # macOS: the Vulkan SDK from LunarG (MoltenVK) — but on a Mac you'd use the Metal build instead.
set -euo pipefail

NKS="${NKS:-$(cd "$(dirname "$0")/.." && pwd)}"
LLAMA="${LLAMA:-$HOME/llama.cpp}"
BASE_COMMIT=c46583b
PATCHES="$NKS/experiments/v0.0/m4_patches"
SPIKE_SRC="$NKS/experiments/v0.0"
BUILD="$LLAMA/build-vulkan"

# ── prereqs ──
command -v cmake >/dev/null || { echo "FATAL: cmake not found." >&2; exit 1; }
if ! command -v glslc >/dev/null && ! command -v glslangValidator >/dev/null; then
  echo "FATAL: no Vulkan shader compiler (glslc / glslangValidator). Install the Vulkan SDK first:" >&2
  echo "       sudo apt install vulkan-headers glslc glslang-tools shaderc" >&2
  exit 1
fi
[ -f /usr/include/vulkan/vulkan.h ] || echo "WARN: /usr/include/vulkan/vulkan.h not found — install vulkan-headers if cmake fails." >&2

# ── 1. base llama.cpp pinned to c46583b + the 5 Llama partial-load patches (idempotent) ──
if [[ ! -d "$LLAMA/.git" ]]; then
  echo "── cloning llama.cpp → $LLAMA ──"; git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
cd "$LLAMA"
git fetch --tags origin >/dev/null 2>&1 || true
if [[ "$(git rev-parse --short HEAD)" != "$BASE_COMMIT" ]] && git diff --quiet 2>/dev/null; then
  git checkout --detach "$BASE_COMMIT"
fi
echo "llama.cpp at $(git rev-parse --short HEAD) (want $BASE_COMMIT)"
for pf in llama-graph.cpp.patch llama-model.cpp.patch llama-model.h.patch \
          llama-model-loader.cpp.patch models_llama.cpp.patch; do
  if patch -p4 --forward --reject-file=- --silent -i "$PATCHES/$pf" 2>/dev/null; then
    echo "  applied $pf"
  elif patch -p4 --reverse --dry-run --silent -i "$PATCHES/$pf" >/dev/null 2>&1; then
    echo "  already applied $pf (skip)"
  else
    echo "FATAL: $pf did not apply and is not already applied — base drift?" >&2; exit 1
  fi
done

# ── 2. the nakshatra-spike daemon example (same source the ROCm build uses) ──
mkdir -p "$LLAMA/examples/nakshatra-spike"
cp "$SPIKE_SRC/worker_daemon.cpp"     "$LLAMA/examples/nakshatra-spike/worker_daemon.cpp"
cp "$SPIKE_SRC/shm_ring.hpp"          "$LLAMA/examples/nakshatra-spike/shm_ring.hpp"
cp "$SPIKE_SRC/CMakeLists.fabric.txt" "$LLAMA/examples/nakshatra-spike/CMakeLists.txt"
grep -q "add_subdirectory(nakshatra-spike)" "$LLAMA/examples/CMakeLists.txt" \
  || echo "add_subdirectory(nakshatra-spike)" >> "$LLAMA/examples/CMakeLists.txt"

# ── 3. configure (Vulkan) + build the worker target into the ISOLATED build dir ──
SHA="$(cd "$NKS" && git rev-parse --short HEAD)"; HOST="$(hostname -s 2>/dev/null || hostname)"
cmake -S "$LLAMA" -B "$BUILD" \
    -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
    -DNAKSHATRA_FABRIC_SHA="$SHA-vulkan" -DNAKSHATRA_FABRIC_BUILD_HOST="$HOST"
cmake --build "$BUILD" --target llama-nakshatra-worker -j"$(nproc)"

BIN="$BUILD/bin/llama-nakshatra-worker"
echo "── built (Vulkan): $BIN ──"; "$BIN" --version; echo
echo "✅ Vulkan worker built. Run a worker with: --daemon-bin $BIN --n-gpu-layers 99"
echo "   (any-vendor card: the Vulkan backend picks the device; set GGML_VK_VISIBLE_DEVICES to choose)."
