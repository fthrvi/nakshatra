# Owner sentinel

The owner sentinel gives a shared Nakshatra worker a short, continuously renewed lease. The
owner's use always wins: pausing, opening a configured application, starting a fullscreen
Windows app, or using the GPU heavily makes the sentinel report busy. If the sentinel stops
for any reason, the worker treats silence as stop and frees the GPU within the lease TTL
(6 seconds by default).

The sentinel sends a heartbeat every 2 seconds to the worker on local port 5571 by default.
It does not decide how the worker unloads; it reports only whether the owner is clear or busy.

## Windows (PowerShell 5.1 or newer)

1. Copy this directory to a stable location in the Windows user's account.
2. Copy the worker's `~/.nakshatra/owner-lease.env` from WSL next to
   `owner-sentinel.ps1`. Windows-to-WSL localhost forwarding makes the local endpoint
   available as `127.0.0.1`; do not expose the lease port to the LAN.
3. Optionally put one game executable per line in `games.txt`. Names may include `.exe`.
4. In Windows PowerShell, run `./owner-sentinel.ps1 -Once -DryRun`. It should print one
   decision and send nothing.
5. Run `./install.ps1`. This registers and starts a hidden, per-user Scheduled Task at
   logon. Administrator access and extra PowerShell modules are not required.

Run `./uninstall.ps1` to remove the task. It leaves the token and process list in place.

The Windows sentinel reports busy for the pause switch, a non-Explorer foreground window
covering its monitor, a configured process, or a non-WSL process holding more than
`OWNER_GPU_MEMORY_THRESHOLD_MIB` (default 1024 MiB) according to `nvidia-smi`. If
`nvidia-smi` is absent, that signal is skipped.

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
