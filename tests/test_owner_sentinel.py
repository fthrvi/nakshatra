from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SENTINEL_DIR = ROOT / "deploy" / "owner-sentinel"
sys.path.insert(0, str(SENTINEL_DIR))

import owner_sentinel as sentinel  # noqa: E402


def settings(tmp_path, **updates):
    values = dict(
        port=5571,
        token="lease-secret-token",
        pause_file=tmp_path / "pause",
        process_list=tmp_path / "games.txt",
        proc_root=tmp_path / "proc",
        drm_sysfs_root=tmp_path / "drm",
    )
    values.update(updates)
    values["proc_root"].mkdir(parents=True, exist_ok=True)
    values["drm_sysfs_root"].mkdir(parents=True, exist_ok=True)
    return sentinel.Settings(**values)


def test_pause_file_is_busy_without_probing_gpu(tmp_path, monkeypatch):
    config = settings(tmp_path)
    config.pause_file.touch()
    monkeypatch.setattr(sentinel, "snapshot_processes", lambda *_: pytest.fail("must not probe"))

    assert sentinel.decide(config) == sentinel.Decision(False, "owner paused")


def test_fake_nvidia_smi_marks_external_process_busy(tmp_path, monkeypatch):
    config = settings(tmp_path)
    monkeypatch.setattr(sentinel, "snapshot_processes", lambda *_: (sentinel.ProcessInfo(41, "render"),))
    monkeypatch.setattr(sentinel.shutil, "which", lambda name: "/fake/nvidia-smi")
    monkeypatch.setattr(
        sentinel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="41, /opt/game.exe, 2048 MiB\n"),
    )

    assert sentinel.decide(config) == sentinel.Decision(False, "owner GPU process")


def test_worker_gpu_allocation_is_ignored_and_clear(tmp_path, monkeypatch):
    config = settings(tmp_path)
    worker = sentinel.ProcessInfo(7, "python3", "/usr/bin/python3 /srv/scripts/worker.py")
    monkeypatch.setattr(sentinel, "snapshot_processes", lambda *_: (worker,))
    monkeypatch.setattr(sentinel.shutil, "which", lambda name: "/fake/nvidia-smi")
    monkeypatch.setattr(
        sentinel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="7, python3, 8192 MiB\n"),
    )

    assert sentinel.decide(config) == sentinel.Decision(True, "owner clear")


def test_configured_process_list_is_busy(tmp_path, monkeypatch):
    config = settings(tmp_path)
    config.process_list.write_text("# local names only\nSpaceGame.exe\n", encoding="utf-8")
    monkeypatch.setattr(
        sentinel,
        "snapshot_processes",
        lambda *_: (sentinel.ProcessInfo(12, "spacegame", "/games/SpaceGame.exe"),),
    )
    monkeypatch.setattr(sentinel.shutil, "which", lambda name: None)

    assert sentinel.decide(config) == sentinel.Decision(False, "configured process running")


def test_missing_gpu_probes_and_empty_process_list_are_clear(tmp_path, monkeypatch):
    config = settings(tmp_path)
    monkeypatch.setattr(sentinel, "snapshot_processes", lambda *_: ())
    monkeypatch.setattr(sentinel.shutil, "which", lambda name: None)

    assert sentinel.decide(config) == sentinel.Decision(True, "owner clear")


def test_amdgpu_busy_with_non_worker_drm_client_is_busy(tmp_path, monkeypatch):
    config = settings(tmp_path)
    busy = config.drm_sysfs_root / "card0" / "device" / "gpu_busy_percent"
    busy.parent.mkdir(parents=True)
    busy.write_text("88\n", encoding="ascii")
    owner = sentinel.ProcessInfo(55, "blender", "blender")
    monkeypatch.setattr(sentinel, "snapshot_processes", lambda *_: (owner,))
    monkeypatch.setattr(sentinel.shutil, "which", lambda name: None)
    monkeypatch.setattr(sentinel, "_process_uses_drm", lambda process, proc: process.pid == 55)

    assert sentinel.decide(config) == sentinel.Decision(False, "owner GPU busy")


class _LeaseHandler(BaseHTTPRequestHandler):
    request = None

    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).request = {
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "body": json.loads(body),
        }
        self.send_response(204)
        self.end_headers()


def test_heartbeat_posts_contract_and_bearer_token_without_printing(tmp_path, capsys):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LeaseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = settings(tmp_path, port=server.server_address[1])
    try:
        result = sentinel.heartbeat_iteration(
            config,
            decision_fn=lambda _: sentinel.Decision(False, "owner paused"),
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert result == sentinel.Decision(False, "owner paused")
    assert _LeaseHandler.request["path"] == "/lease"
    assert _LeaseHandler.request["authorization"] == f"Bearer {config.token}"
    assert _LeaseHandler.request["body"]["clear"] is False
    assert _LeaseHandler.request["body"]["reason"] == "owner paused"
    assert isinstance(_LeaseHandler.request["body"]["at"], (int, float))
    captured = capsys.readouterr()
    assert config.token not in captured.out and config.token not in captured.err


def test_probe_exception_sends_nothing_and_prints_no_token(tmp_path, capsys):
    config = settings(tmp_path)
    sent = []

    def explode(_settings):
        raise OSError("probe failed")

    assert sentinel.heartbeat_iteration(config, decision_fn=explode, send_fn=lambda *args: sent.append(args)) is None
    assert sent == []
    captured = capsys.readouterr()
    assert config.token not in captured.out and config.token not in captured.err


def test_once_dry_run_prints_decision_and_sends_nothing(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / "owner-lease.env"
    env_file.write_text("OWNER_LEASE_PORT=5571\nOWNER_LEASE_TOKEN=do-not-print\n", encoding="utf-8")
    pause = tmp_path / "pause"
    pause.touch()
    monkeypatch.setattr(sentinel, "send_heartbeat", lambda *_: pytest.fail("dry-run must not send"))

    assert sentinel.main(["--env-file", str(env_file), "--pause-file", str(pause), "--once", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == {"clear": False, "reason": "owner paused"}
    assert "do-not-print" not in output


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is not installed")
@pytest.mark.parametrize("script", sorted(SENTINEL_DIR.glob("*.ps1")), ids=lambda path: path.name)
def test_powershell_scripts_parse(script):
    command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$tokens,[ref]$errors) | Out-Null; "
        "if ($errors.Count) { $errors | ForEach-Object { [Console]::Error.WriteLine($_) }; exit 1 }"
    )
    completed = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", command, str(script)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_launcher_runs_only_the_local_sentinel_and_does_not_read_token():
    launcher = (SENTINEL_DIR / "sentinel-launcher.ps1").read_text(encoding="utf-8")

    assert "Join-Path $InstallDirectory 'owner-sentinel.ps1'" in launcher
    assert "Start-Process" in launcher
    assert "OWNER_LEASE_TOKEN" not in launcher


def test_launcher_has_no_network_download_surface():
    launcher = (SENTINEL_DIR / "sentinel-launcher.ps1").read_text(encoding="utf-8")
    lowered = launcher.lower()

    assert "invoke-webrequest" not in lowered
    urls = re.findall(r"https?://([^/\s'\"]+)", launcher, flags=re.IGNORECASE)
    assert all(host.split(":", 1)[0].lower() in {"127.0.0.1", "localhost", "[::1]"} for host in urls)


def test_launcher_self_update_has_required_safety_structure():
    launcher = (SENTINEL_DIR / "sentinel-launcher.ps1").read_text(encoding="utf-8")

    assert re.search(r"(?m)^# Launcher-Version: \d+$", launcher)
    assert "$LauncherPath = $PSCommandPath" in launcher
    assert "[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value" in launcher
    assert '"Global\\NakshatraOwnerSentinelLauncher-$currentUserSid"' in launcher
    assert '"Local\\NakshatraOwnerSentinelLauncher-$currentUserSid"' in launcher
    assert "Action 'mutex-fallback'" in launcher
    assert "[System.Threading.Mutex]::new" in launcher
    assert "[TimeSpan]::FromSeconds(30)" in launcher
    assert "[System.Management.Automation.Language.Parser]::ParseFile" in launcher
    assert "[System.IO.File]::Replace($TemporaryPath, $TargetPath, $backupPath)" in launcher
    assert "[System.IO.File]::Move($TemporaryPath, $TargetPath)" in launcher
    assert '$backupPath = "${TargetPath}.bak"' in launcher
    assert "Move-Item" not in launcher
    assert ".next" not in launcher
    assert "exit 75" not in launcher
    assert "$LauncherUpdateExitCode" not in launcher

    sync = launcher[launcher.index("function Invoke-LauncherSync") : launcher.index("function Invoke-SentinelSync")]
    parse_check = sync.index("Test-PowerShellFile -Path $temporaryPath")
    launcher_swap = sync.index("Install-VerifiedFile -TemporaryPath $temporaryPath -TargetPath $LauncherPath")
    assert parse_check < launcher_swap
    assert "release launcher lacks a Launcher-Version header" in sync
    assert "[Int64]::TryParse($localVersion" in sync
    assert "[Int64]::TryParse($releaseVersion" in sync
    assert "$releaseVersionNumber -lt $localVersionNumber" in sync

    sentinel_sync = launcher[
        launcher.index("function Invoke-SentinelSync") : launcher.index("function Start-SentinelChild")
    ]
    sentinel_parse = sentinel_sync.index("Test-PowerShellFile -Path $temporaryPath")
    stop_sentinels = sentinel_sync.index("Stop-LocalSentinels")
    sentinel_swap = sentinel_sync.index(
        "Install-VerifiedFile -TemporaryPath $temporaryPath -TargetPath $LocalSentinelPath"
    )
    assert sentinel_parse < stop_sentinels < sentinel_swap
    assert "release sentinel lacks a Sentinel-Version header" in sentinel_sync

    handoff = launcher[launcher.index("if (Invoke-LauncherSync)") : launcher.index("$updated = Invoke-SentinelSync")]
    assert handoff.index("Stop-SentinelChild") < handoff.index("ReleaseMutex")
    assert handoff.index("ReleaseMutex") < handoff.index("$launcherMutex.Dispose()")
    assert handoff.index("$launcherMutex.Dispose()") < handoff.index("Start-LauncherSuccessor")
    assert handoff.index("Start-LauncherSuccessor") < handoff.index("exit 0")
    successor = launcher[
        launcher.index("function Start-LauncherSuccessor") : launcher.index("$launcherMutex = $null")
    ]
    assert "Start-Process" in successor
    assert '-File `"$LauncherPath`"' in successor
    assert "-WindowStyle Hidden" in successor


def test_launcher_fetch_is_shared_and_wsl_never_boots_a_stopped_distro():
    launcher = (SENTINEL_DIR / "sentinel-launcher.ps1").read_text(encoding="utf-8")
    lowered = launcher.lower()

    assert "function Get-ReleaseFileCopy" in launcher
    assert "-FileName 'sentinel-launcher.ps1'" in launcher
    assert "-FileName 'owner-sentinel.ps1'" in launcher
    assert "wsl.localhost" not in lowered
    assert "wsl$" not in lowered
    wsl_fetch = launcher[
        launcher.index("function Save-ReleaseFileViaWsl") : launcher.index("function Get-ReleaseFileCopy")
    ]
    running_check = wsl_fetch.index("& wsl.exe -l --running -q")
    distro_read = wsl_fetch.index("& wsl.exe -d $Identity.Distro")
    assert running_check < distro_read
    assert "if ($names -notcontains $Identity.Distro) { return $false }" in wsl_fetch
    assert launcher.count("& wsl.exe") == 2


def test_installer_registers_launcher_without_an_execution_time_limit():
    installer = (SENTINEL_DIR / "install.ps1").read_text(encoding="utf-8")
    parameter_block = installer[: installer.index("$ErrorActionPreference")]

    assert "[string]$Distro" in installer
    assert "[string]$LinuxUser" in installer
    assert "Mandatory = $true" not in parameter_block
    assert "$hadExistingEnvironment" in installer
    assert "an existing installed owner-lease.env may supply them" in installer
    assert "sentinel-launcher.ps1" in installer
    assert "OWNER_WSL_DISTRO" in installer
    assert "OWNER_WSL_USER" in installer
    assert re.search(r"-ExecutionTimeLimit\s*\(New-TimeSpan\s+-Seconds\s+0\)", installer)
    assert re.search(r"-RestartCount\s+999\b", installer)
    assert re.search(r"-RestartInterval\s*\(New-TimeSpan\s+-Minutes\s+1\)", installer)
