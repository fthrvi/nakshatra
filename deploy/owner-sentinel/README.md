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
4. Run `./install.ps1 -Distro <WSL-distro> -LinuxUser <Linux-user>`, for example
   `./install.ps1 -Distro Ubuntu-24.04 -LinuxUser biswa`. The installer copies the Windows
   files to `%LOCALAPPDATA%\nakshatra\owner-sentinel`, records `OWNER_WSL_DISTRO` and
   `OWNER_WSL_USER` in the installed `owner-lease.env`, and registers a hidden per-user
   Scheduled Task at logon. Administrator access and extra PowerShell modules are not
   required.

Run `./uninstall.ps1` to remove the task. It leaves the token and process list in place.

### Windows self-update

The Scheduled Task runs `sentinel-launcher.ps1`, which always starts the local installed
`owner-sentinel.ps1`. At startup and every six hours, the launcher looks for both scripts
in the current signed worker release under
`\\wsl.localhost\<Distro>\home\<LinuxUser>\.nakshatra-node\current\nakshatra\deploy\owner-sentinel`,
with `\\wsl$\...` as a compatibility fallback. If a SHA-256 differs, it copies the release
file to a temporary file, re-hashes it, and atomically replaces the installed file with
`Move-Item -Force`. A sentinel update restarts the sentinel child. A launcher update first
requires a `Launcher-Version` header and a clean PowerShell parse, then stops the child and
exits with code 75 so Task Scheduler restarts the updated launcher. Code 75 is an intentional
self-update restart, not a sentinel failure. If Windows refuses to replace the running script,
the validated copy is staged as `sentinel-launcher.ps1.next` and applied before the next
launcher starts its child. An unexpectedly exited child is restarted after five seconds.
The launcher sends no heartbeat itself, so a failed or stopped sentinel still means silence
and the worker still stops.

The task allows 999 one-minute restart attempts, has no execution time limit, and runs on
battery. Existing installations need one manual re-run of `install.ps1` to install this
self-updating launcher and register those restart-on-failure settings; future launcher and
sentinel updates are automatic.

The WSL release tree is trusted because it is written only by the signed-release installer
inside the same Windows user's distro; it has the same trust boundary as the worker it
updates. The launcher never fetches from a network location. If WSL is stopped, the release
path is missing, hashing/copying fails, or the release tree is temporarily unavailable, the
launcher keeps and runs the existing local sentinel. It never boots a stopped distro and
never deletes that copy. A per-user named mutex prevents a task restart and a stray launcher
copy from running concurrently.

To verify updates, inspect
`%LOCALAPPDATA%\nakshatra\owner-sentinel\launcher.log`. Each update check is one timestamped
line naming the launcher or sentinel component, its local and release version/SHA-256, and
an action such as `updated`, `staged`, or `kept`; the file retains about 200 lines. It never
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
