$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$InstallDirectory = $PSScriptRoot
$EnvironmentFile = Join-Path $InstallDirectory 'owner-lease.env'
$LocalSentinelPath = Join-Path $InstallDirectory 'owner-sentinel.ps1'
$LogPath = Join-Path $InstallDirectory 'launcher.log'
$PowerShellPath = Join-Path $PSHOME 'powershell.exe'
$UpdateInterval = [TimeSpan]::FromHours(6)
$RestartDelaySeconds = 5
$MaximumLogLines = 200

function ConvertFrom-EnvironmentValue {
    param([string]$Value)
    $result = $Value.Trim()
    if ($result.Length -ge 2 -and
        (($result.StartsWith('"') -and $result.EndsWith('"')) -or
         ($result.StartsWith("'") -and $result.EndsWith("'")))) {
        $result = $result.Substring(1, $result.Length - 2)
    }
    return $result
}

function Get-OwnerWslIdentity {
    if (-not [System.IO.File]::Exists($EnvironmentFile)) {
        throw 'owner-lease.env is missing'
    }

    $distro = $null
    $linuxUser = $null
    foreach ($rawLine in [System.IO.File]::ReadLines($EnvironmentFile)) {
        if ($rawLine -match '^\s*(?:export\s+)?OWNER_WSL_DISTRO\s*=(.*)$') {
            $distro = ConvertFrom-EnvironmentValue -Value $Matches[1]
        } elseif ($rawLine -match '^\s*(?:export\s+)?OWNER_WSL_USER\s*=(.*)$') {
            $linuxUser = ConvertFrom-EnvironmentValue -Value $Matches[1]
        }
    }
    if (-not $distro -or -not $linuxUser) {
        throw 'OWNER_WSL_DISTRO and OWNER_WSL_USER are required'
    }
    if ($distro -notmatch '^[A-Za-z0-9._-]+$' -or $linuxUser -notmatch '^[A-Za-z0-9._-]+$') {
        throw 'OWNER_WSL_DISTRO and OWNER_WSL_USER contain unsupported characters'
    }
    return @{ Distro = $distro; LinuxUser = $linuxUser }
}

function Get-ReleaseSentinelPath {
    param([hashtable]$Identity)
    $relativePath = 'home\{0}\.nakshatra-node\current\nakshatra\deploy\owner-sentinel\owner-sentinel.ps1' -f $Identity.LinuxUser
    $candidates = @(
        ('\\wsl.localhost\{0}\{1}' -f $Identity.Distro, $relativePath),
        ('\\wsl$\{0}\{1}' -f $Identity.Distro, $relativePath)
    )
    foreach ($candidate in $candidates) {
        try {
            if ([System.IO.File]::Exists($candidate)) { return $candidate }
        } catch {
            # A stopped or unavailable WSL distro is an ordinary fail-open-for-the-sentinel case.
        }
    }
    return $null
}

function Get-SentinelVersion {
    param([string]$Path)
    if (-not $Path -or -not [System.IO.File]::Exists($Path)) { return 'missing' }
    try {
        $lineCount = 0
        foreach ($line in [System.IO.File]::ReadLines($Path)) {
            if ($line -match '^#\s*Sentinel-Version:\s*(\S+)') { return $Matches[1] }
            $lineCount++
            if ($lineCount -ge 40) { break }
        }
    } catch {
        return 'unknown'
    }
    return 'unknown'
}

function Get-SentinelHash {
    param([string]$Path)
    if (-not $Path -or -not [System.IO.File]::Exists($Path)) { return 'missing' }
    try { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
    catch { return 'unavailable' }
}

function Write-LauncherLog {
    param(
        [string]$Action,
        [string]$LocalVersion,
        [string]$LocalHash,
        [string]$ReleaseVersion,
        [string]$ReleaseHash,
        [string]$Detail = ''
    )
    try {
        $timestamp = [DateTimeOffset]::Now.ToString('o')
        $safeDetail = ($Detail -replace '[\r\n]+', ' ').Trim()
        $line = "$timestamp action=$Action local=$LocalVersion/$LocalHash release=$ReleaseVersion/$ReleaseHash"
        if ($safeDetail) { $line += " detail=$safeDetail" }
        $lines = @()
        if ([System.IO.File]::Exists($LogPath)) { $lines = @([System.IO.File]::ReadAllLines($LogPath)) }
        $lines += $line
        if ($lines.Count -gt $MaximumLogLines) {
            $lines = @($lines[($lines.Count - $MaximumLogLines)..($lines.Count - 1)])
        }
        [System.IO.File]::WriteAllLines($LogPath, [string[]]$lines)
    } catch {
        # Logging must never interrupt the local sentinel.
    }
}

function Stop-LocalSentinels {
    # A running sentinel holds owner-sentinel.ps1 open, so the replace below fails with
    # "being used by another process" (measured on blackwell). Stop every sentinel started from
    # THIS install directory: our own child, and orphans a previous launcher left behind.
    try {
        Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
            Where-Object { $_.CommandLine -and $_.CommandLine.Contains($LocalSentinelPath) } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Milliseconds 500
    } catch {
        # Best effort; the move below reports a failure and the current sentinel keeps running.
    }
}

function Save-ReleaseSentinelViaWsl {
    # Fallback when \\wsl.localhost is not visible to this Windows session (measured on blackwell,
    # 2026-10-13: the share was absent even in the interactive session). Reads the signed release copy
    # through wsl.exe, but ONLY when the distro is already running: wsl.exe would otherwise boot it,
    # and booting the distro starts the worker, which the owner's sentinel must never do.
    param([hashtable]$Identity, [string]$Destination)
    try {
        $running = ((& wsl.exe -l --running -q) -join "`n") -replace "`0", ''
        $names = $running -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        if ($names -notcontains $Identity.Distro) { return $false }
        $linuxPath = '/home/{0}/.nakshatra-node/current/nakshatra/deploy/owner-sentinel/owner-sentinel.ps1' -f $Identity.LinuxUser
        $encoded = (& wsl.exe -d $Identity.Distro -u $Identity.LinuxUser -- base64 -w0 $linuxPath) -join ''
        if ($LASTEXITCODE -ne 0 -or -not $encoded) { return $false }
        [System.IO.File]::WriteAllBytes($Destination, [Convert]::FromBase64String($encoded.Trim()))
        return $true
    } catch {
        return $false
    }
}

function Invoke-SentinelSync {
    $localVersion = Get-SentinelVersion -Path $LocalSentinelPath
    $localHash = Get-SentinelHash -Path $LocalSentinelPath
    $releaseVersion = 'unavailable'
    $releaseHash = 'unavailable'
    $temporaryPath = $null
    try {
        $identity = Get-OwnerWslIdentity
        $releasePath = Get-ReleaseSentinelPath -Identity $identity
        if (-not $releasePath) {
            $fetched = Join-Path $InstallDirectory ('owner-sentinel.{0}.wsl' -f [Guid]::NewGuid().ToString('N'))
            if (-not (Save-ReleaseSentinelViaWsl -Identity $identity -Destination $fetched)) {
                Write-LauncherLog -Action 'kept' -LocalVersion $localVersion -LocalHash $localHash `
                    -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash -Detail 'release copy unavailable'
                return $false
            }
            # The fetched copy is the release copy from here on; the copy step below re-hashes it.
            $releasePath = $fetched
            $temporaryPath = $fetched
        }

        $releaseVersion = Get-SentinelVersion -Path $releasePath
        $releaseHash = Get-SentinelHash -Path $releasePath
        if ($releaseHash -eq 'unavailable' -or $releaseHash -eq 'missing') {
            throw 'could not hash the release copy'
        }
        if ($localHash -eq $releaseHash) {
            Write-LauncherLog -Action 'kept' -LocalVersion $localVersion -LocalHash $localHash `
                -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash
            return $false
        }

        if ($temporaryPath -ne $releasePath) {
            $temporaryPath = Join-Path $InstallDirectory ('owner-sentinel.{0}.tmp' -f [Guid]::NewGuid().ToString('N'))
            Copy-Item -LiteralPath $releasePath -Destination $temporaryPath -Force
        }
        $temporaryHash = Get-SentinelHash -Path $temporaryPath
        if ($temporaryHash -ne $releaseHash) { throw 'release copy changed while it was being copied' }
        Stop-LocalSentinels
        Move-Item -LiteralPath $temporaryPath -Destination $LocalSentinelPath -Force
        $temporaryPath = $null
        $localVersion = Get-SentinelVersion -Path $LocalSentinelPath
        $localHash = Get-SentinelHash -Path $LocalSentinelPath
        Write-LauncherLog -Action 'updated' -LocalVersion $localVersion -LocalHash $localHash `
            -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash
        return $true
    } catch {
        Write-LauncherLog -Action 'kept' -LocalVersion $localVersion -LocalHash $localHash `
            -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash -Detail $_.Exception.Message
        return $false
    } finally {
        if ($temporaryPath -and [System.IO.File]::Exists($temporaryPath)) {
            Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Start-SentinelChild {
    $arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$LocalSentinelPath`""
    return Start-Process -FilePath $PowerShellPath -ArgumentList $arguments -WorkingDirectory $InstallDirectory `
        -WindowStyle Hidden -PassThru
}

$child = $null
$nextUpdate = [DateTime]::MinValue
while ($true) {
    if ([DateTime]::UtcNow -ge $nextUpdate) {
        $updated = Invoke-SentinelSync
        $nextUpdate = [DateTime]::UtcNow.Add($UpdateInterval)
        if ($updated -and $null -ne $child) {
            try {
                if (-not $child.HasExited) { Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue }
                $child.WaitForExit()
            } catch {
                # The child may have exited between the checks.
            }
            $child.Dispose()
            $child = $null
        }
    }

    if ($null -eq $child) {
        try { $child = Start-SentinelChild }
        catch {
            Start-Sleep -Seconds $RestartDelaySeconds
            continue
        }
    }

    $waitMilliseconds = [int][Math]::Max(1, [Math]::Min([int]::MaxValue, ($nextUpdate - [DateTime]::UtcNow).TotalMilliseconds))
    if ($child.WaitForExit($waitMilliseconds)) {
        $child.Dispose()
        $child = $null
        Start-Sleep -Seconds $RestartDelaySeconds
    }
}
