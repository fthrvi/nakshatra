param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$Distro,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$LinuxUser
)

$ErrorActionPreference = 'Stop'
$taskName = 'Nakshatra Owner Sentinel'
$installDirectory = Join-Path $env:LOCALAPPDATA 'nakshatra\owner-sentinel'
$sourceEnvironmentPath = Join-Path $PSScriptRoot 'owner-lease.env'
$environmentPath = Join-Path $installDirectory 'owner-lease.env'
$scriptPath = Join-Path $installDirectory 'sentinel-launcher.ps1'
$powerShellPath = Join-Path $PSHOME 'powershell.exe'

New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null
foreach ($name in @('owner-sentinel.ps1', 'sentinel-launcher.ps1')) {
    $source = Join-Path $PSScriptRoot $name
    $destination = Join-Path $installDirectory $name
    if ([System.IO.Path]::GetFullPath($source) -ne [System.IO.Path]::GetFullPath($destination)) {
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
}
$installedGamesPath = Join-Path $installDirectory 'games.txt'
if (-not [System.IO.File]::Exists($installedGamesPath)) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'games.txt') -Destination $installedGamesPath
}
if (-not [System.IO.File]::Exists($environmentPath)) {
    if (-not [System.IO.File]::Exists($sourceEnvironmentPath)) {
        throw 'Copy owner-lease.env next to install.ps1 before installing'
    }
    if ([System.IO.Path]::GetFullPath($sourceEnvironmentPath) -ne [System.IO.Path]::GetFullPath($environmentPath)) {
        Copy-Item -LiteralPath $sourceEnvironmentPath -Destination $environmentPath
    }
}

$updatedEnvironment = @()
$foundDistro = $false
$foundLinuxUser = $false
foreach ($line in [System.IO.File]::ReadAllLines($environmentPath)) {
    if ($line -match '^\s*(?:export\s+)?OWNER_WSL_DISTRO\s*=') {
        $updatedEnvironment += "OWNER_WSL_DISTRO=$Distro"
        $foundDistro = $true
    } elseif ($line -match '^\s*(?:export\s+)?OWNER_WSL_USER\s*=') {
        $updatedEnvironment += "OWNER_WSL_USER=$LinuxUser"
        $foundLinuxUser = $true
    } else {
        $updatedEnvironment += $line
    }
}
if (-not $foundDistro) { $updatedEnvironment += "OWNER_WSL_DISTRO=$Distro" }
if (-not $foundLinuxUser) { $updatedEnvironment += "OWNER_WSL_USER=$LinuxUser" }
$temporaryEnvironmentPath = Join-Path $installDirectory ('owner-lease.{0}.tmp' -f [Guid]::NewGuid().ToString('N'))
try {
    $utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllLines($temporaryEnvironmentPath, [string[]]$updatedEnvironment, $utf8WithoutBom)
    Move-Item -LiteralPath $temporaryEnvironmentPath -Destination $environmentPath -Force
} finally {
    if ([System.IO.File]::Exists($temporaryEnvironmentPath)) {
        Remove-Item -LiteralPath $temporaryEnvironmentPath -Force -ErrorAction SilentlyContinue
    }
}

$arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`""
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $powerShellPath -Argument $arguments -WorkingDirectory $installDirectory
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
# No execution time limit: the default (72 h) would kill the sentinel after three days, and its
# silence would keep the worker 'away' until the next logon. Run on battery too.
$settings = New-ScheduledTaskSettingsSet -Hidden -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}
Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host "Installed and started the per-user Nakshatra Owner Sentinel task from $installDirectory."
