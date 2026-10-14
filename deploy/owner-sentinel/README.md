# Owner sentinel

The owner sentinel gives a shared Nakshatra worker a short, continuously renewed lease. The
owner's use always wins: pausing, opening a configured application, starting a fullscreen
Windows app, or using the GPU heavily makes the sentinel report busy. If the sentinel stops
for any reason, the worker treats silence as stop and frees the GPU within the lease TTL
(6 seconds by default).

The sentinel sends a heartbeat every 2 seconds to the worker on local port 5571 by default.
It does not decide how the worker unloads; it reports only whether the owner is clear or busy.

## Windows (PowerShell 5.1 or newer)

1. Copy the worker's `~/.nakshatra/owner-lease.env` from WSL into this directory.
   Windows-to-WSL localhost forwarding makes the local endpoint available as
   `127.0.0.1`; do not expose the lease port to the LAN.
2. Optionally put one game executable per line in `games.txt`. Names may include `.exe`.
3. In Windows PowerShell, run `./owner-sentinel.ps1 -Once -DryRun`. It should print one
   decision and send nothing.
4. Run `.\install.ps1 -Distro <WSL-distro> -LinuxUser <Linux-user>`, for example
   `.\install.ps1 -Distro Ubuntu-24.04 -LinuxUser biswa`. The installer copies the Windows
   files to `%LOCALAPPDATA%\nakshatra\owner-sentinel`, records `OWNER_WSL_DISTRO` and
   `OWNER_WSL_USER` in the installed `owner-lease.env`, and registers a hidden per-user
   Scheduled Task at logon. Administrator access and extra PowerShell modules are not
   required. On an existing installation whose installed `owner-lease.env` already contains
   both values, rerun the installer as `.\install.ps1`; it reuses the saved values. A fresh
   install still requires both parameters.

Run `./uninstall.ps1` to remove the task. It leaves the token and process list in place.

### Windows self-update

The Scheduled Task runs `sentinel-launcher.ps1`, which always starts the local installed
`owner-sentinel.ps1`. At startup and every six hours, the launcher first asks `wsl.exe` for
the running distributions. Only when the configured distribution is already running does it
use `wsl.exe` to read both scripts from the current signed worker release. It does not probe a
Windows WSL share, so an update check cannot boot a stopped distribution.

Each fetched script is written to a same-directory temporary file and hashed. The exact
temporary bytes being installed must have the required version header and pass a PowerShell
parse check. Launcher versions are integers and never move backward; an equal version may be
replaced when the just-fetched release bytes have a different hash. For an existing target,
the launcher uses Windows `ReplaceFile` semantics through `System.IO.File.Replace`, atomically
putting the previous file at the adjacent `.bak` path. It uses `System.IO.File.Move` only when
the target does not exist. A replacement exception leaves the installed file in place, is
logged, and is retried at the next six-hour interval. Before replacing the sentinel, the
launcher stops sentinel processes from this install directory; after a successful sentinel
update, it restarts its child.

After a successful launcher replacement, the running launcher stops its sentinel child,
releases and disposes its mutex, starts the new launcher hidden with the same PowerShell
arguments, and exits successfully. The successor waits up to 30 seconds for the mutex before
exiting successfully, so an overlap cannot run two sentinels. The primary mutex is named in
the global Windows namespace with the current user's SID; if that namespace is denied, the
launcher logs the fallback and uses the session-local namespace with the same SID suffix.
An unexpectedly exited sentinel child is restarted after five seconds. The launcher sends no
heartbeat itself, so a failed or stopped sentinel still means silence and the worker still
stops.

The task allows 999 one-minute restart attempts as crash insurance, has no execution time
limit, and runs on battery. Task Scheduler does not restart this task merely because its
PowerShell action returns a particular exit code; self-update therefore uses the direct
handoff above. Existing installations need one manual `.\install.ps1` rerun to install this
self-updating launcher and refresh the task settings; future launcher and sentinel updates
are automatic.

The WSL release tree is trusted because it is written only by the signed-release installer
inside the same Windows user's distro; it has the same trust boundary as the worker it
updates. The launcher never fetches from a network location. If WSL is stopped, the release
path is missing, hashing fails, or the release tree is temporarily unavailable, the launcher
keeps and runs the existing local sentinel. It never boots a stopped distro.

To verify updates, inspect
`%LOCALAPPDATA%\nakshatra\owner-sentinel\launcher.log`. Each update check is one timestamped
line naming the launcher or sentinel component, its local and release version/SHA-256, and
an action such as `updated` or `kept`; the file retains about 200 lines. It never
contains the lease token. You can also inspect the Scheduled Task action and confirm it names
`sentinel-launcher.ps1` in that directory.

The Windows sentinel reports busy for the pause switch, a non-Explorer foreground window
covering its monitor, a configured process, or an owner process holding more than
`OWNER_GPU_MEMORY_THRESHOLD_MIB` (default 1024 MiB) of dedicated GPU memory.

The GPU signal reads Windows' own `\GPU Process Memory(*)\Dedicated Usage` counters. It works for
any vendor, and it is the only usable per-process source on consumer drivers: there,
`nvidia-smi --query-compute-apps` reports `[N/A]` for every process (measured on an RTX 5070
host, 2026-10-13). By default `OWNER_GPU_IGNORE=vmwp,vmmem*,wsl*,dwm,csrss` excludes the WSL VM,
which is where our own worker's memory appears, and the desktop compositor's baseline (about 1.6 GB on
that host). Run `owner-sentinel.ps1 -Once -DryRun` to see the decision; a dry run also prints
why a decision failed. Normal runs stay silent, which the worker treats as stop.

## Linux desktop

The Python sentinel uses only the standard library. Install it and the user unit:

```sh
mkdir -p ~/.local/lib/nakshatra/owner-sentinel ~/.config/systemd/user ~/.config/nakshatra
cp owner_sentinel.py ~/.local/lib/nakshatra/owner-sentinel/
cp nakshatra-owner-sentinel.service ~/.config/systemd/user/
cp ~/.nakshatra/owner-lease.env ~/.config/nakshatra/owner-lease.env
cp games.txt ~/.config/nakshatra/games.txt
python3 ~/.local/lib/nakshatra/owner-sentinel/owner_sentinel.py \
  --env-file ~/.config/nakshatra/owner-lease.env \
  --process-list ~/.config/nakshatra/games.txt --once --dry-run
systemctl --user daemon-reload
systemctl --user enable --now nakshatra-owner-sentinel.service
```

Linux reports busy for the pause switch, a configured process, an external process using
more than `OWNER_GPU_MEMORY_THRESHOLD_MIB` on NVIDIA, or owner activity while AMD sysfs
`gpu_busy_percent` exceeds `OWNER_AMD_BUSY_THRESHOLD_PERCENT` (default 30). Known Nakshatra
worker processes are excluded. Missing `nvidia-smi` and missing AMD sysfs counters are
simply unavailable signals.

## Pause, resume, and verify

On Windows, run `./pause.ps1` and `./resume.ps1`. On Linux:

```sh
mkdir -p ~/.config/nakshatra && touch ~/.config/nakshatra/pause  # pause
rm ~/.config/nakshatra/pause                                    # resume
```

Verify locally with `-Once -DryRun` / `--once --dry-run`: clear state prints
`"clear":true`; after pausing it prints `"clear":false`. Dry-run never sends a heartbeat.
For an end-to-end check, start the installed sentinel, confirm the worker lease is live,
then pause or stop the sentinel and confirm the worker unloads within the configured TTL.

## Privacy and failure behavior

No window titles and no process lists leave the machine. The only request is a local HTTP
POST containing a Unix timestamp, `clear` or `busy`, and a short category such as
`owner paused` or `foreground fullscreen`. The bearer token is never logged. If reading any
available signal or sending the request fails, the sentinel emits nothing; silence makes the
worker stop.
