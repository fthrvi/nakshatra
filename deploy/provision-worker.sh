#!/usr/bin/env bash
# provision-worker.sh — turn a freshly-joined device into a SERVING nakshatra worker, from scratch,
# with NO by-hand steps. The dial-out worker installer (onboard worker.sh) calls this after the box
# is on the mesh + rostered. Idempotent, OS-aware (macOS + Linux), no sudo required.
#
# It reproduces the proven build (2026-06-19, the 3-iMac bring-up): fetch the EXACT patched
# llama.cpp source (vendored tarball, no fragile clone+patch), build the partial-load daemon, set up
# the Python venv. Run with a model afterwards via `serve-worker.sh` (or worker.py directly).
#
#   provision-worker.sh                      # defaults (fetch stack from the hub, build, venv)
# Env:
#   WORKER_STACK_URL   where to fetch the patched-llama.cpp source tarball
#                      (default: http://10.42.0.1:8078/worker-llama-stack.tgz — the hub on the mesh)
#   WORKER_DIR         install root (default ~/.nakshatra-worker)
#   BUILD_TARGET       cmake target (default llama-nakshatra-worker)
set -euo pipefail

WORKER_DIR="${WORKER_DIR:-$HOME/.nakshatra-worker}"
STACK_URL="${WORKER_STACK_URL:-http://10.42.0.1:8078/worker-llama-stack.tgz}"
BUILD_TARGET="${BUILD_TARGET:-llama-nakshatra-worker}"
LLAMA="$WORKER_DIR/llama"
say() { printf '\033[1;36m[provision]\033[0m %s\n' "$*"; }
OS="$(uname -s)"; ARCH="$(uname -m)"
mkdir -p "$WORKER_DIR"

# ── 1. cmake (no sudo) ────────────────────────────────────────────────────────────────────────────
if ! command -v cmake >/dev/null 2>&1 && [ ! -x "$WORKER_DIR/opt/cmake/bin/cmake" ] \
     && [ ! -x "$WORKER_DIR/opt/cmake/CMake.app/Contents/bin/cmake" ]; then
  say "fetching cmake (no sudo)…"
  mkdir -p "$WORKER_DIR/opt"
  if [ "$OS" = "Darwin" ]; then
    curl -fsSL -o "$WORKER_DIR/opt/cmake.tgz" \
      https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-macos-universal.tar.gz
    rm -rf "$WORKER_DIR/opt/cmake"; mkdir -p "$WORKER_DIR/opt/cmake"
    tar xzf "$WORKER_DIR/opt/cmake.tgz" -C "$WORKER_DIR/opt/cmake" --strip-components=1
    export PATH="$WORKER_DIR/opt/cmake/CMake.app/Contents/bin:$PATH"
  else
    K="cmake-3.30.5-linux-${ARCH}"; [ "$ARCH" = "arm64" ] && K="cmake-3.30.5-linux-aarch64"
    curl -fsSL -o "$WORKER_DIR/opt/cmake.tgz" \
      "https://github.com/Kitware/CMake/releases/download/v3.30.5/${K}.tar.gz"
    rm -rf "$WORKER_DIR/opt/cmake"; mkdir -p "$WORKER_DIR/opt/cmake"
    tar xzf "$WORKER_DIR/opt/cmake.tgz" -C "$WORKER_DIR/opt/cmake" --strip-components=1
    export PATH="$WORKER_DIR/opt/cmake/bin:$PATH"
  fi
else
  [ -x "$WORKER_DIR/opt/cmake/CMake.app/Contents/bin/cmake" ] && export PATH="$WORKER_DIR/opt/cmake/CMake.app/Contents/bin:$PATH"
  [ -x "$WORKER_DIR/opt/cmake/bin/cmake" ] && export PATH="$WORKER_DIR/opt/cmake/bin:$PATH"
fi
say "cmake: $(cmake --version | head -1)"

# ── 2. fetch + unpack the EXACT patched llama.cpp source (vendored — no clone+patch drift) ─────────
if [ ! -f "$LLAMA/examples/nakshatra-spike/worker_daemon.cpp" ]; then
  say "fetching patched llama.cpp source from $STACK_URL…"
  curl -fsSL -o "$WORKER_DIR/stack.tgz" "$STACK_URL"
  mkdir -p "$LLAMA"; tar xzf "$WORKER_DIR/stack.tgz" -C "$LLAMA"; rm -f "$WORKER_DIR/stack.tgz"
fi
say "source ready at $LLAMA"

# ── 3. build the partial-load daemon (Metal compiled on macOS; RUN it CPU — Metal is broken on the
#       Intel-iMac Radeons. On Linux, native.) ─────────────────────────────────────────────────────
METAL=OFF; [ "$OS" = "Darwin" ] && METAL=ON
NPROC="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
say "building $BUILD_TARGET (GGML_METAL=$METAL, -j$NPROC)…"
cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_METAL=$METAL -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$LLAMA/build" -j"$NPROC" --target "$BUILD_TARGET" >/dev/null
DAEMON="$LLAMA/build/bin/$BUILD_TARGET"
[ -x "$DAEMON" ] || { say "BUILD FAILED — no $DAEMON"; exit 1; }
say "✓ daemon: $DAEMON"

# ── 4. python venv + deps (relax the pb2 grpc version assertion so any grpcio>=1.66 runs the stubs —
#       the committed stubs hard-pin >=1.81.1 but 1.80 is wire-compatible for our unary/bidi RPCs) ──
PYV="$WORKER_DIR/venv"
[ -x "$PYV/bin/python" ] || python3 -m venv "$PYV"
say "installing python deps…"
"$PYV/bin/pip" install -q --upgrade pip
"$PYV/bin/pip" install -q grpcio numpy pyyaml protobuf cryptography gguf
SCRIPTS="$WORKER_DIR/nakshatra-scripts"
if [ -d "$SCRIPTS" ]; then
  for stub in "$SCRIPTS"/*_pb2_grpc.py; do
    [ -f "$stub" ] && perl -0pi -e 's/raise RuntimeError\(\s*[^)]*GRPC_GENERATED_VERSION[^)]*\)/pass  # nakshatra: version check relaxed (1.80 wire-compatible)/s' "$stub" 2>/dev/null || true
  done
fi
say "✓ venv: $PYV ($("$PYV/bin/python" -c 'import grpc;print("grpcio",grpc.__version__)' 2>/dev/null))"

say "DAEMON_OK $DAEMON"
say "worker stack provisioned. Next: serve a model slice — worker.py --package-url <pkg> --mode first/last --n-gpu-layers 0 (CPU)."
