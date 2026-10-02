# The Nakshatra engine (patched llama.cpp)

**The ONE source of truth for the worker engine** (unification U6c, 2026-10-03). Before this, the engine
lived in four places that disagreed:
- the hub-only fork branch `~/llama.cpp` `nakshatra/engine` (the newest one, with no remote copy);
- the older patch files `experiments/v0.0/m4_patches` (against b8445);
- a "canonical" `experiments/v0.0/worker_daemon.cpp`, stale: it lacks `solo` mode. `rebuild-worker.sh` would have copied it over the newer one;
- an unsigned source tarball on the VPS.

Now:

| File | What |
|---|---|
| `BASE` | upstream repo, pinned commit (tag b9992), and the expected result `TREE` |
| `patches/` | the engine as a patch series: partial-load/forward for llama, qwen3, qwen3moe; the worker daemon; solo mode |
| `source.sh DEST` | builds the exact source tree and **refuses unless its git tree hash == TREE** |

Build it for this machine's GPU with `deploy/provision-worker.sh`; it detects CUDA, ROCm, Vulkan, Metal or CPU, and needs no sudo. It ships in every signed release.

Change the engine: commit on the fork → `git format-patch <COMMIT>..HEAD -o engine/patches` → update `TREE` → release.
