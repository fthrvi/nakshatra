# ijru CUDA rebuild — attempt log, what's proven, what's blocked (2026-07-29 night)

Triggered by the rebalance bench finding: ijru's `llama-nakshatra-worker` was built
`GGML_CUDA=OFF`, so every "hub Vulkan / ijru CUDA" chain bench in this project's history
actually ran ijru's stage on **CPU** (127–140 ms/step). Rebuilding it with CUDA was ranked
the #1 speed action. This is the honest result of that attempt.

## PROVEN (good news)

1. **ijru's CUDA toolchain works.** CUDA 12.4 (`/usr/bin/nvcc`), driver 595.71.05,
   RTX 3060 compute capability 8.6, 11908 MiB. `cmake` available via pip
   (`~/.local/bin`, add to PATH — it is NOT in the system path).
2. **The host-compiler pin is mandatory and is the likely reason CUDA was off in the
   first place.** ijru's default `gcc` is 15.2, which CUDA 12.4 refuses. `gcc-13`/`g++-13`
   are installed; configure with
   `-DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`
   and the CUDA kernels compile clean (`-- Including CUDA backend`, `CMAKE_CUDA_ARCHITECTURES=86`).
3. **A STATIC build registers CUDA correctly.** `build-cuda-static`
   (`-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF`) produces a 198 MB binary, and
   `llama-bench --list-devices` from that tree prints:
   `ggml_cuda_init: found 1 CUDA devices … CUDA0: NVIDIA GeForce RTX 3060 (11908 MiB, 11368 MiB free)`.
   So the backend is live in that build.

## BLOCKED (the honest wall)

4. **The SHARED build silently drops CUDA.** With `-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON`
   the configure log says "Including CUDA backend" and `libggml-cuda.so` is produced — but
   `CMakeCache.txt` still reads `GGML_CUDA:BOOL=OFF` and, decisively,
   `ldd libggml.so` shows **only** `libggml-base` + `libggml-cpu`: the CUDA backend is never
   linked into the registry, so the worker runs on CPU with no error. **This is a silent
   trap: a clean "BUILD_OK" plus CUDA .so files on disk proves nothing.** Always verify with
   `ldd`/`--list-devices` and by watching `nvidia-smi` climb during a load.
5. **The fork's worker daemon does not offload even from the working static build.**
   Running `build-cuda-static/bin/llama-nakshatra-worker <slice> last 512 0 99` (n_gpu_layers=99):
   VRAM stays at the 429 MiB idle baseline, GPU util 0 %, and the daemon prints **no**
   `llama_model_loader:` / `ggml_cuda_init:` lines at all (it reports `ready` in ~0.3 s),
   whereas the CPU build of the same source *does* print full loader output. Meanwhile
   `llama-bench` from the same tree initializes CUDA fine — so the divergence is in the
   daemon's load path, not the build.
   For reference the hub, on the **same fork revision `3b160c30`**, offloads correctly with
   `GGML_HIP=ON` + `BUILD_SHARED_LIBS=ON` (its log shows
   `llama_model_load_from_file_impl: using device ROCm0 … 6236 MiB free`).

## HANDOFF — for whoever owns the engine (`examples/nakshatra-spike/worker_daemon.cpp`)

The source *looks* right (`mp.n_gpu_layers = n_gpu_layers` → `llama_model_load_from_file`,
worker_daemon.cpp ~:356). Next probes, in order:
1. Diff hub's working HIP build config against a **shared** CUDA config; check whether the
   fork's `ggml/src/CMakeLists.txt` adds `ggml-cuda` to `ggml`'s link libraries the way it
   adds `ggml-hip` (suspected wiring gap — explains #4 exactly).
2. In the static build, instrument the daemon: print `ggml_backend_dev_count()` /
   `ggml_backend_dev_name()` right before `load_mc()` to see whether CUDA is in the registry
   *in that process* (llama-bench says the tree can see it).
3. If the registry is empty in the daemon, the fix is likely calling `ggml_backend_load_all()`
   (or linking the backend) before model load; if it is populated, the partial-slice load path
   is choosing a CPU buffer type regardless of `n_gpu_layers`.

**Prize for closing this:** ijru's stage is currently 127–140 ms/step on CPU. On the 3060 it
should fall several-fold; the Q3 chain is 4.94 tok/s today and >10 tok/s is plausible.
Re-bench with `~/.nakshatra/qwen3-30b-q3-chain-v2.yaml` (ports 5562/5572) against the receipts
in `~/.nakshatra/receipt-rebalance.json` for an apples-to-apples number.

## Artifacts / state left behind

- `~/llama.cpp/build-cuda-static/` on ijru — working static CUDA tree (llama-bench + worker).
- `~/llama.cpp/build-cuda/` — the shared attempt that silently lost CUDA (keep as evidence or delete).
- `~/.nakshatra/llama-nakshatra-worker.cpu-backup` + `CMakeCache.cpu-backup.txt` — the original
  CPU binary and its config, so the live chain can always be restored.
- The original `~/llama.cpp/build/` (CPU) is **untouched**; the live serving path is unchanged.
- No stray processes: all bench workers killed on both boxes; ijru VRAM back to 429 MiB idle;
  the `eagle` conda env was never touched.
