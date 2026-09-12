# Can a Windows machine serve? — three routes, one recommendation

**T1.3. A comparison and a recommendation. Nothing here is built.**

## Why this matters more than it looks

Every other option in this repo widens the fleet across *server* hardware. Windows is where
the **idle consumer GPUs** are — the 3060s and 4070s in gaming machines that sit at 0%
utilisation most of the day. The mission sentence says *any machine with a GPU joins with one
command*. Today a Windows box cannot join at all, and that is the largest single population
excluded.

## The three routes

### 1. Native Windows build

Compile `llama-nakshatra-worker` with MSVC or clang-cl, ship a `.exe`.

- **For**: no dependency on anything Microsoft could change; best latency; a user double-clicks
  something.
- **Against**: a **second build matrix forever**. Our partial-load patch is frontend-only and
  backend-agnostic, so it *compiles*, but the surrounding daemon is POSIX-shaped —
  `os.kill(pid, 0)` for liveness, `start_new_session`, `/proc/<pid>/cmdline` for the argv check
  that catches the silent-CPU bug, `statvfs` for disk, `ss` for ports. Every one of those needs
  a Windows equivalent, and each is a place the two paths can silently diverge.
- ⚠️ **The argv check is the sharpest problem.** `procargs.py` reads `/proc` precisely *because*
  the config can lie. Windows has no `/proc`; the equivalent is WMI or `NtQueryInformationProcess`,
  which is a different failure surface. A Windows port that skipped that check would reintroduce
  the exact bug this repo spent a day catching.
- **Maintenance**: high and permanent. Two OSes, two CI targets, two sets of platform bugs.

### 2. WSL2 + CUDA passthrough  ⭐ **recommended**

The user installs WSL2 with a distro; the existing Linux path runs unmodified inside it.

- **For**: **zero new code.** `provision-worker.sh`, the daemon, `/proc`, `statvfs`, `ss` — all
  of it works because it *is* Linux. NVIDIA's WSL2 CUDA passthrough is mature and a first-class
  supported path. One codebase, one CI target, one set of bugs.
- **Against**: a real install step the user must take first (`wsl --install`, a reboot, and a
  recent Windows 10/11). Not literally one command until that exists.
- ⚠️ **NVIDIA only, in practice.** AMD's ROCm has no WSL2 story worth relying on. That is
  acceptable — the idle-consumer-GPU population we are actually chasing is overwhelmingly
  NVIDIA — but it should be stated rather than discovered.
- ⚠️ Memory is governed by `.wslconfig`, not the host: a 32 GB machine may present 16 GB to
  WSL2. `capability.py` reads what it is *given*, so it will size correctly — but an operator
  will be confused by the number unless the docs say why.
- **Maintenance**: near zero. It is the Linux path.

### 3. Prebuilt binary + a small native launcher

Ship a signed `.exe` that fetches the slice and runs a prebuilt worker.

- **For**: the best user experience by a distance — download, run, joined.
- **Against**: it needs everything route 1 needs *and* a code-signing certificate (~$200–400/yr,
  or SmartScreen scares every user away), plus a release/update pipeline for a binary that
  lives on strangers' machines.
- ⚠️ **A self-updating binary on a stranger's machine is a supply-chain surface.** It is the
  one component where a compromise of our build reaches every operator directly, and it is not
  where a project this size should spend its first security budget.
- **Maintenance**: highest. Build matrix, signing, updates, and a support burden in an OS
  nobody here runs.

## Recommendation: WSL2 (route 2)

Because it costs **no new code** and no second build matrix, while reaching the population that
matters. It trades a one-time user install step for permanent engineering simplicity — and the
step is one a gaming-PC owner can already follow.

Route 3 is the right *eventual* answer if this ever needs to reach people who will not open a
terminal, and it should be revisited then — deliberately, with a signing budget, not by drift.

Route 1 is the one to avoid: it carries most of route 3's cost with none of its benefit.

## What "supporting WSL2" would actually require

Not a port. A page of documentation and three checks:

1. **A prerequisites doc**: `wsl --install`, a distro, NVIDIA's WSL2 driver, and how to set
   `memory=` in `.wslconfig`.
2. **A detection line** in `provision-worker.sh` reading `/proc/sys/kernel/osrelease` for
   `microsoft` — ⚠️ not to change behaviour, but so the environment appears in the node's
   listing. A node that silently *is* WSL2 makes a support conversation start from zero.
3. **A capability note**: report the WSL2 memory ceiling alongside VRAM, because the operator's
   idea of their RAM and the daemon's will differ and only one of them is right.

None of that is a second code path, which is the entire argument.
