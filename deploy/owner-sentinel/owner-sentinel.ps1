param(
    [switch]$Once,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$SentinelDirectory = $PSScriptRoot
$EnvironmentFile = Join-Path $SentinelDirectory 'owner-lease.env'
$PauseFile = Join-Path $env:LOCALAPPDATA 'nakshatra\pause'
$GamesFile = Join-Path $SentinelDirectory 'games.txt'
$HeartbeatSeconds = 2

$NativeMethods = @'
using System;
using System.Runtime.InteropServices;

public static class OwnerSentinelNative {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public struct MONITORINFO {
        public int cbSize;
        public RECT rcMonitor;
        public RECT rcWork;
        public uint dwFlags;
    }

    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    public static extern IntPtr GetShellWindow();

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    public static extern IntPtr MonitorFromWindow(IntPtr hWnd, uint flags);

    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern bool GetMonitorInfo(IntPtr monitor, ref MONITORINFO info);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
}
'@

function Read-OwnerLeaseEnvironment {
    param([string]$Path)
    $values = @{}
    foreach ($rawLine in [System.IO.File]::ReadAllLines($Path)) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith('#')) { continue }
        if ($line.StartsWith('export ')) { $line = $line.Substring(7).TrimStart() }
        $separator = $line.IndexOf('=')
        if ($separator -le 0) { throw 'Invalid owner lease environment line' }
        $key = $line.Substring(0, $separator).Trim()
        $value = $line.Substring($separator + 1).Trim()
        if ($value.Length -ge 2) {
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        $values[$key] = $value
    }
    return $values
}

function Test-ForegroundFullscreen {
    $window = [OwnerSentinelNative]::GetForegroundWindow()
    if ($window -eq [IntPtr]::Zero -or $window -eq [OwnerSentinelNative]::GetShellWindow()) { return $false }

    [uint32]$processId = 0
    [void][OwnerSentinelNative]::GetWindowThreadProcessId($window, [ref]$processId)
    if ($processId -eq 0) { throw 'Could not identify the foreground process' }
    $process = Get-Process -Id $processId -ErrorAction Stop
    if ($process.ProcessName -ieq 'explorer') { return $false }

    $windowRect = New-Object OwnerSentinelNative+RECT
    if (-not [OwnerSentinelNative]::GetWindowRect($window, [ref]$windowRect)) {
        throw 'Could not read the foreground window rectangle'
    }
    $monitor = [OwnerSentinelNative]::MonitorFromWindow($window, 2)
    if ($monitor -eq [IntPtr]::Zero) { throw 'Could not identify the foreground monitor' }
    $monitorInfo = New-Object OwnerSentinelNative+MONITORINFO
    $monitorInfo.cbSize = [Runtime.InteropServices.Marshal]::SizeOf($monitorInfo)
    if (-not [OwnerSentinelNative]::GetMonitorInfo($monitor, [ref]$monitorInfo)) {
        throw 'Could not read the foreground monitor rectangle'
    }

    $tolerance = 2
    return (($windowRect.Left -le $monitorInfo.rcMonitor.Left + $tolerance) -and
            ($windowRect.Top -le $monitorInfo.rcMonitor.Top + $tolerance) -and
            ($windowRect.Right -ge $monitorInfo.rcMonitor.Right - $tolerance) -and
            ($windowRect.Bottom -ge $monitorInfo.rcMonitor.Bottom - $tolerance))
}

function Test-ConfiguredProcess {
    if (-not [System.IO.File]::Exists($GamesFile)) { return $false }
    $configured = @{}
    foreach ($rawLine in [System.IO.File]::ReadAllLines($GamesFile)) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith('#')) { continue }
        $configured[[System.IO.Path]::GetFileNameWithoutExtension($line).ToLowerInvariant()] = $true
    }
    foreach ($process in Get-Process -ErrorAction Stop) {
        if ($configured.ContainsKey($process.ProcessName.ToLowerInvariant())) { return $true }
    }
    return $false
}

function Test-OwnerGpuProcess {
    param([int]$ThresholdMiB)
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { $nvidiaSmi = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue }
    if ($null -eq $nvidiaSmi) { return $false }

    $output = & $nvidiaSmi.Source '--query-compute-apps=pid,process_name,used_memory' '--format=csv,noheader' 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
    if (-not $output) { return $false }
    $rows = $output | ConvertFrom-Csv -Header 'pid', 'process_name', 'used_memory'
    foreach ($row in $rows) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($row.process_name).ToLowerInvariant()
        if ($name -match '^(vmmem|vmmemwsl|wsl.*)$') { continue }
        $memoryMatch = [regex]::Match([string]$row.used_memory, '^\s*(\d+)\s*(?:MiB)?\s*$')
        if (-not $memoryMatch.Success) {
            throw 'Unexpected nvidia-smi memory value'
        }
        if ([int]$memoryMatch.Groups[1].Value -gt $ThresholdMiB) { return $true }
    }
    return $false
}

function Get-OwnerDecision {
    param([int]$GpuThresholdMiB)
    if ([System.IO.File]::Exists($PauseFile)) { return @{ clear = $false; reason = 'owner paused' } }
    if (Test-ForegroundFullscreen) { return @{ clear = $false; reason = 'foreground fullscreen' } }
    if (Test-ConfiguredProcess) { return @{ clear = $false; reason = 'configured process running' } }
    if (Test-OwnerGpuProcess -ThresholdMiB $GpuThresholdMiB) {
        return @{ clear = $false; reason = 'owner GPU process' }
    }
    return @{ clear = $true; reason = 'owner clear' }
}

function Send-OwnerLease {
    param([hashtable]$Decision, [int]$Port, [string]$Token)
    $body = @{ at = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); clear = $Decision.clear; reason = $Decision.reason } |
        ConvertTo-Json -Compress
    $headers = @{ Authorization = "Bearer $Token" }
    Invoke-WebRequest -UseBasicParsing -Method Post -Uri "http://127.0.0.1:$Port/lease" `
        -Headers $headers -ContentType 'application/json' -Body $body -TimeoutSec 1 | Out-Null
}

try {
    Add-Type -TypeDefinition $NativeMethods
    $leaseEnvironment = Read-OwnerLeaseEnvironment -Path $EnvironmentFile
    if (-not $leaseEnvironment.ContainsKey('OWNER_LEASE_PORT') -or
        -not $leaseEnvironment.ContainsKey('OWNER_LEASE_TOKEN') -or
        -not $leaseEnvironment['OWNER_LEASE_TOKEN']) {
        throw 'OWNER_LEASE_PORT and OWNER_LEASE_TOKEN are required'
    }
    $port = [int]$leaseEnvironment['OWNER_LEASE_PORT']
    if ($port -lt 1 -or $port -gt 65535) { throw 'OWNER_LEASE_PORT is outside its allowed range' }
    $gpuThresholdMiB = 1024
    if ($leaseEnvironment.ContainsKey('OWNER_GPU_MEMORY_THRESHOLD_MIB')) {
        $gpuThresholdMiB = [int]$leaseEnvironment['OWNER_GPU_MEMORY_THRESHOLD_MIB']
    }
    if ($gpuThresholdMiB -lt 0) { throw 'OWNER_GPU_MEMORY_THRESHOLD_MIB must not be negative' }
} catch {
    exit 1
}

do {
    $succeeded = $false
    try {
        $decision = Get-OwnerDecision -GpuThresholdMiB $gpuThresholdMiB
        if ($DryRun) {
            $decision | ConvertTo-Json -Compress
        } else {
            Send-OwnerLease -Decision $decision -Port $port -Token $leaseEnvironment['OWNER_LEASE_TOKEN']
        }
        $succeeded = $true
    } catch {
        # Deliberately silent: a missing heartbeat makes the worker release the GPU.
    }
    if ($Once) { if ($succeeded) { exit 0 } else { exit 1 } }
    Start-Sleep -Seconds $HeartbeatSeconds
} while ($true)
