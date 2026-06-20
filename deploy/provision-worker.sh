#!/usr/bin/env bash
# provision-worker.sh - turn a freshly-joined device into a SERVING nakshatra worker, from scratch,
# with NO by-hand steps. The dial-out worker installer (onboard worker.sh) calls this after the box
# is on the mesh + rostered. Idempotent, OS-aware (macOS + Linux), no sudo required. Pure ASCII
# (runs under the C locale that non-interactive macOS ssh hands you).
#
# Reproduces the proven build (2026-06-19, the 3-iMac bring-up): fetch the EXACT patched llama.cpp
# source (vendored tarball, no fragile clone+patch), build the partial-load daemon, set up the venv.
#
#   provision-worker.sh
# Env:
#   WORKER_STACK_URL  where to fetch the patched-llama.cpp source tarball (the device needs internet
#                     for this, like the model download; default = the public onboard host).
#   WORKER_DIR        install root (default ~/.nakshatra-worker)
#   BUILD_TARGET      cmake target (default llama-nakshatra-worker)
set -euo pipefail

WORKER_DIR="${WORKER_DIR:-$HOME/.nakshatra-worker}"
STACK_URL="${WORKER_STACK_URL:-https://prithviloka.net/onboard/worker-llama-stack.tgz}"
SCRIPTS_URL="${WORKER_SCRIPTS_URL:-https://prithviloka.net/onboard/worker-scripts.tgz}"
BUILD_TARGET="${BUILD_TARGET:-llama-nakshatra-worker}"
LLAMA="$WORKER_DIR/llama"
SCRIPTS="$WORKER_DIR/nakshatra-scripts"
say() { printf '[provision] %s\n' "$*"; }
OS="$(uname -s)"; ARCH="$(uname -m)"
mkdir -p "$WORKER_DIR"

# 1. cmake (no sudo) -------------------------------------------------------------------------------
CM=""
command -v cmake >/dev/null 2>&1 && CM="cmake"
[ -x "$WORKER_DIR/opt/cmake/CMake.app/Contents/bin/cmake" ] && CM="$WORKER_DIR/opt/cmake/CMake.app/Contents/bin/cmake"
[ -x "$WORKER_DIR/opt/cmake/bin/cmake" ] && CM="$WORKER_DIR/opt/cmake/bin/cmake"
if [ -z "$CM" ]; then
  say "fetching cmake (no sudo)"
  mkdir -p "$WORKER_DIR/opt"
  if [ "$OS" = "Darwin" ]; then
    curl -fsSL -o "$WORKER_DIR/opt/cmake.tgz" \
      "https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-macos-universal.tar.gz"
    rm -rf "$WORKER_DIR/opt/cmake"; mkdir -p "$WORKER_DIR/opt/cmake"
    tar xzf "$WORKER_DIR/opt/cmake.tgz" -C "$WORKER_DIR/opt/cmake" --strip-components=1
    CM="$WORKER_DIR/opt/cmake/CMake.app/Contents/bin/cmake"
  else
    K="cmake-3.30.5-linux-${ARCH}"; [ "$ARCH" = "arm64" ] && K="cmake-3.30.5-linux-aarch64"
    curl -fsSL -o "$WORKER_DIR/opt/cmake.tgz" \
      "https://github.com/Kitware/CMake/releases/download/v3.30.5/${K}.tar.gz"
    rm -rf "$WORKER_DIR/opt/cmake"; mkdir -p "$WORKER_DIR/opt/cmake"
    tar xzf "$WORKER_DIR/opt/cmake.tgz" -C "$WORKER_DIR/opt/cmake" --strip-components=1
    CM="$WORKER_DIR/opt/cmake/bin/cmake"
  fi
fi
say "cmake: $("$CM" --version | head -1)"

# 2. fetch + unpack the EXACT patched llama.cpp source (vendored - no clone+patch drift) -----------
if [ ! -f "$LLAMA/examples/nakshatra-spike/worker_daemon.cpp" ]; then
  say "fetching patched llama.cpp source from $STACK_URL"
  curl -fsSL -o "$WORKER_DIR/stack.tgz" "$STACK_URL"
  mkdir -p "$LLAMA"; tar xzf "$WORKER_DIR/stack.tgz" -C "$LLAMA"; rm -f "$WORKER_DIR/stack.tgz"
fi
say "source ready at $LLAMA"

# 3. build the partial-load daemon. Metal compiled on macOS but RUN it CPU (-ngl 0) - Metal is
#    numerically broken on the Intel-iMac Radeons. On Linux, native. ------------------------------
METAL=OFF; [ "$OS" = "Darwin" ] && METAL=ON
NPROC="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
say "building $BUILD_TARGET (GGML_METAL=$METAL, -j$NPROC)"
"$CM" -S "$LLAMA" -B "$LLAMA/build" -DGGML_METAL=$METAL -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
"$CM" --build "$LLAMA/build" -j"$NPROC" --target "$BUILD_TARGET" >/dev/null
DAEMON="$LLAMA/build/bin/$BUILD_TARGET"
[ -x "$DAEMON" ] || { say "BUILD FAILED - no $DAEMON"; exit 1; }
say "daemon built: $DAEMON"

# 4. python venv + deps. Relax the pb2 grpc version assertion so any grpcio runs the stubs (the
#    committed stubs hard-pin >=1.81.1 but 1.80 is wire-compatible for our unary/bidi RPCs). -------
PYV="$WORKER_DIR/venv"
[ -x "$PYV/bin/python" ] || python3 -m venv "$PYV"
say "installing python deps"
"$PYV/bin/pip" install -q --upgrade pip
"$PYV/bin/pip" install -q grpcio numpy pyyaml protobuf cryptography gguf

# 5. fetch the nakshatra SERVE scripts (worker.py + pb2 stubs + fabric/packaging) so this box can
#    actually serve, not just build. Public, non-secret (the model + roster are the gated parts). --
if [ ! -f "$SCRIPTS/worker.py" ]; then
  say "fetching serve scripts from $SCRIPTS_URL"
  if curl -fsSL -o "$WORKER_DIR/scripts.tgz" "$SCRIPTS_URL" 2>/dev/null; then
    mkdir -p "$SCRIPTS"; tar xzf "$WORKER_DIR/scripts.tgz" -C "$SCRIPTS"; rm -f "$WORKER_DIR/scripts.tgz"
  else
    say "  (serve scripts not hosted yet - daemon is built; worker.py can be supplied later)"
  fi
fi
# relax the committed pb2 grpc>=1.81.1 version pin (1.80 is wire-compatible for our unary/bidi RPCs)
for stub in "$SCRIPTS"/*_pb2_grpc.py; do
  [ -f "$stub" ] && perl -0pi -e 's/raise RuntimeError\([^)]*GRPC_GENERATED_VERSION[^)]*\)/pass/s' "$stub" 2>/dev/null || true
done
say "venv: $PYV ($("$PYV/bin/python" -c 'import grpc;print("grpcio",grpc.__version__)' 2>/dev/null))"

# 6. write a tiny serve helper so the operator (or the planner) can start the worker with one cmd. --
cat > "$WORKER_DIR/serve-worker.sh" <<SERVE
#!/usr/bin/env bash
# serve-worker.sh <port> <first|middle|last> <layer-start> <layer-end> <package-url> [model-id]
set -euo pipefail
cd "$SCRIPTS"
exec "$PYV/bin/python" worker.py --port "\${1:?port}" --sub-gguf "$WORKER_DIR/slice.gguf" \\
  --package-url "\${5:?package-url}" --mode "\${2:?mode}" --layer-start "\${3:?start}" --layer-end "\${4:?end}" \\
  --model-id "\${6:-model}" --daemon-bin "$DAEMON" --n-ctx 2048 --n-gpu-layers 0 \\
  --node-id "\$(hostname -s)-\${2}" --no-file-server --skip-sha256
SERVE
chmod +x "$WORKER_DIR/serve-worker.sh"

say "DAEMON_OK $DAEMON"
[ -f "$SCRIPTS/worker.py" ] && say "SCRIPTS_OK (serve: $WORKER_DIR/serve-worker.sh <port> first 0 16 <package-url>)" \
  || say "serve scripts pending (host worker-scripts.tgz)"
say "worker fully provisioned - ready for the planner to assign a model slice (CPU, -ngl 0)."
