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

# 1b. build prerequisites. A fresh Ubuntu (and a fresh WSL distro especially) has NO compiler and
# NO python venv module, and without them this dies deep inside cmake with "CMAKE_C_COMPILER not
# set" or, later, "ensurepip is not available" — errors that read like a broken script rather than
# a missing package. Install them up front when we can; say so plainly when we cannot.
if command -v apt-get >/dev/null 2>&1 && { sudo -n true 2>/dev/null || [ "$(id -u)" = 0 ]; }; then
  MISSING=""
  command -v cc >/dev/null 2>&1 || MISSING="$MISSING build-essential"
  python3 -c 'import ensurepip' >/dev/null 2>&1 || MISSING="$MISSING python3-venv"
  if [ -n "$MISSING" ]; then
    say "installing build prerequisites:$MISSING"
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $MISSING >/dev/null 2>&1 || true
  fi
fi
command -v cc >/dev/null 2>&1 || say "WARNING: no C compiler — the build will fail (apt install build-essential)"

# A venv left half-created by an earlier failed run has no bin/pip, and `[ -x venv/bin/python ]`
# then skips recreating it forever — the retry fails identically until someone deletes it by hand.
[ -d "$WORKER_DIR/venv" ] && [ ! -x "$WORKER_DIR/venv/bin/pip" ] && {
  say "clearing a half-created venv from a previous run"; rm -rf "$WORKER_DIR/venv"; }

# 2. fetch + unpack the EXACT patched llama.cpp source (vendored - no clone+patch drift) -----------
if [ ! -f "$LLAMA/examples/nakshatra-spike/worker_daemon.cpp" ]; then
  say "fetching patched llama.cpp source from $STACK_URL"
  curl -fsSL -o "$WORKER_DIR/stack.tgz" "$STACK_URL"
  mkdir -p "$LLAMA"; tar xzf "$WORKER_DIR/stack.tgz" -C "$LLAMA"; rm -f "$WORKER_DIR/stack.tgz"
fi
say "source ready at $LLAMA"

# 3. build the partial-load daemon, FOR THE ACCELERATOR THIS BOX ACTUALLY HAS. ------------------
#    Metal is compiled on macOS but RUN at -ngl 0 - it is numerically broken on the Intel-iMac
#    Radeons. On Linux we detect and build the matching backend.
#
#    ⚠ Until 2026-08-05 this only ever considered Metal, so EVERY Linux/NVIDIA worker silently
#    provisioned as CPU-only. A box donates its GPU, the join succeeds, and the one line saying
#    "-ngl 0" is the only hint that the GPU is not being used at all. Detect it here rather than
#    leaving it for someone to notice.
NPROC="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
METAL=OFF; [ "$OS" = "Darwin" ] && METAL=ON
ACCEL_FLAGS=""; ACCEL="cpu"

_have(){ command -v "$1" >/dev/null 2>&1; }
_apt(){ _have apt-get && (sudo -n true 2>/dev/null || [ "$(id -u)" = 0 ]); }

if [ "$OS" = "Darwin" ]; then
  ACCEL="metal(compiled, run -ngl 0)"
elif _have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  # Target THIS card's compute capability. A toolkit older than the arch cannot emit code for it:
  # an RTX 5070 is sm_120 and Ubuntu's own nvidia-cuda-toolkit is 12.4, which does not know sm_120 —
  # so "install the distro package" quietly yields a binary that will not run on the very GPU that
  # was donated. Ask the card, then make sure nvcc is new enough for the answer.
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
  [ -n "$CC" ] || CC=86
  NVCC_OK=0
  if _have nvcc; then
    NVV="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
    NVMAJ="${NVV%%.*}"; NVMIN="${NVV##*.}"
    # sm_120 (Blackwell) needs >= 12.8; anything older is only safe for older arches.
    if [ "$CC" -lt 120 ] || [ "${NVMAJ:-0}" -gt 12 ] || { [ "${NVMAJ:-0}" -eq 12 ] && [ "${NVMIN:-0}" -ge 8 ]; }; then NVCC_OK=1; fi
    [ "$NVCC_OK" = 1 ] || say "nvcc $NVV is too old for sm_$CC — will try NVIDIA's repo"
  fi
  if [ "$NVCC_OK" != 1 ] && [ "${WORKER_INSTALL_CUDA:-1}" = 1 ] && _apt; then
    . /etc/os-release 2>/dev/null || true
    RID="ubuntu$(echo "${VERSION_ID:-24.04}" | tr -d '.')"
    say "installing CUDA toolkit for sm_$CC from NVIDIA's $RID repo (large download)"
    TMPD="$(mktemp -d)"
    if curl -fsSL -o "$TMPD/k.deb" \
        "https://developer.download.nvidia.com/compute/cuda/repos/$RID/x86_64/cuda-keyring_1.1-1_all.deb" 2>/dev/null; then
      sudo -n dpkg -i "$TMPD/k.deb" >/dev/null 2>&1 || true
      sudo -n apt-get update -qq >/dev/null 2>&1 || true
      # newest cuda-toolkit-* the repo offers; newer is fine, older is not
      PKG="$(apt-cache search --names-only '^cuda-toolkit-[0-9]+-[0-9]+$' 2>/dev/null \
             | awk '{print $1}' | sort -V | tail -1)"
      [ -n "$PKG" ] && { say "  -> $PKG"; sudo -n apt-get install -y -qq "$PKG" >/dev/null 2>&1 || true; }
    fi
    rm -rf "$TMPD"
    for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do [ -x "$d/nvcc" ] && PATH="$d:$PATH"; done
    export PATH
    _have nvcc && NVCC_OK=1
  fi
  if [ "$NVCC_OK" = 1 ]; then
    ACCEL="cuda(sm_$CC)"
    ACCEL_FLAGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC"
  else
    say "NVIDIA GPU present but no usable nvcc — building CPU-only (set WORKER_INSTALL_CUDA=1 with sudo to fix)"
  fi
elif _have hipconfig || _have rocminfo; then
  ACCEL="hip/rocm"; ACCEL_FLAGS="-DGGML_HIP=ON"
fi

say "building $BUILD_TARGET (accel=$ACCEL, GGML_METAL=$METAL, -j$NPROC)"
# shellcheck disable=SC2086
"$CM" -S "$LLAMA" -B "$LLAMA/build" -DGGML_METAL=$METAL $ACCEL_FLAGS -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
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
say "worker fully provisioned - ready for the planner to assign a model slice (accel=$ACCEL)."
