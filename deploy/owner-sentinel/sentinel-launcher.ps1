# Launcher-Version: 2026-10-13.1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$InstallDirectory = $PSScriptRoot
$EnvironmentFile = Join-Path $InstallDirectory 'owner-lease.env'
$LauncherPath = $PSCommandPath
$PendingLauncherPath = "${LauncherPath}.next"
$LocalSentinelPath = Join-Path $InstallDirectory 'owner-sentinel.ps1'
$LogPath = Join-Path $InstallDirectory 'launcher.log'
$PowerShellPath = Join-Path $PSHOME 'powershell.exe'
$UpdateInterval = [TimeSpan]::FromHours(6)
$RestartDelaySeconds = 5
$MaximumLogLines = 200
$LauncherUpdateExitCode = 75
$LauncherMutexName = 'Local\NakshatraOwnerSentinelLauncher'

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

function Get-ReleaseFilePath {
    param([hashtable]$Identity, [string]$FileName)
    if ([System.IO.Path]::GetFileName($FileName) -ne $FileName) {
        throw 'release file name must not contain a path'
    }
    $relativePath = 'home\{0}\.nakshatra-node\current\nakshatra\deploy\owner-sentinel\{1}' -f `
        $Identity.LinuxUser, $FileName
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

function Get-ScriptVersion {
    param([string]$Path, [string]$Header)
    if (-not $Path -or -not [System.IO.File]::Exists($Path)) { return 'missing' }
    try {
        $lineCount = 0
        $pattern = '^#\s*{0}:\s*(\S+)' -f [Regex]::Escape($Header)
        foreach ($line in [System.IO.File]::ReadLines($Path)) {
            if ($line -match $pattern) { return $Matches[1] }
            $lineCount++
            if ($lineCount -ge 40) { break }
        }
    } catch {
        return 'unknown'
    }
    return 'unknown'
}

function Get-ScriptHash {
    param([string]$Path)
    if (-not $Path -or -not [System.IO.File]::Exists($Path)) { return 'missing' }
    try { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
    catch { return 'unavailable' }
}

function Write-LauncherLog {
    param(
        [string]$Component,
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
        $line = "$timestamp component=$Component action=$Action local=$LocalVersion/$LocalHash release=$ReleaseVersion/$ReleaseHash"
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

function Test-PowerShellFile {
    param([string]$Path)
    try {
        $tokens = $null
        $parseErrors = $null
        [System.Management.Automation.Language.Parser]::ParseFile(
            $Path, [ref]$tokens, [ref]$parseErrors
        ) | Out-Null
        return @($parseErrors).Count -eq 0
    } catch {
        return $false
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
        # Best effort; the move below reports a failure and the current sentinel is restarted.
    }
}

function Stop-SentinelChild {
    param($Child)
    if ($null -eq $Child) { return }
    try {
        if (-not $Child.HasExited) { Stop-Process -Id $Child.Id -Force -ErrorAction SilentlyContinue }
        $Child.WaitForExit()
    } catch {
        # The child may have exited between the checks.
    } finally {
        try { $Child.Dispose() } catch { }
    }
}

function Save-ReleaseFileViaWsl {
    # Fallback when \\wsl.localhost is not visible to this Windows session (measured on blackwell,
    # 2026-10-13: the share was absent even in the interactive session). Reads the signed release copy
    # through wsl.exe, but ONLY when the distro is already running: wsl.exe would otherwise boot it,
    # and booting the distro starts the worker, which the owner's sentinel must never do.
    param([hashtable]$Identity, [string]$FileName, [string]$Destination)
    try {
        $running = ((& wsl.exe -l --running -q) -join "`n") -replace "`0", ''
        $names = $running -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        if ($names -notcontains $Identity.Distro) { return $false }
        $linuxPath = '/home/{0}/.nakshatra-node/current/nakshatra/deploy/owner-sentinel/{1}' -f `
            $Identity.LinuxUser, $FileName
        $encoded = (& wsl.exe -d $Identity.Distro -u $Identity.LinuxUser -- base64 -w0 $linuxPath) -join ''
        if ($LASTEXITCODE -ne 0 -or -not $encoded) { return $false }
        [System.IO.File]::WriteAllBytes($Destination, [Convert]::FromBase64String($encoded.Trim()))
        return $true
    } catch {
        return $false
    }
}

function Get-ReleaseFileCopy {
    # Both scripts use this exact share-first, already-running-WSL-second fetch and hash path.
    param([hashtable]$Identity, [string]$FileName)
    $downloadPath = $null
    $temporaryPath = Join-Path $InstallDirectory ('{0}.{1}.tmp' -f $FileName, [Guid]::NewGuid().ToString('N'))
    try {
        $releasePath = Get-ReleaseFilePath -Identity $Identity -FileName $FileName
        if (-not $releasePath) {
            $downloadPath = Join-Path $InstallDirectory ('{0}.{1}.wsl' -f $FileName, [Guid]::NewGuid().ToString('N'))
            if (-not (Save-ReleaseFileViaWsl -Identity $Identity -FileName $FileName -Destination $downloadPath)) {
                throw 'release copy unavailable'
            }
            $releasePath = $downloadPath
        }

        $releaseHash = Get-ScriptHash -Path $releasePath
        if ($releaseHash -eq 'unavailable' -or $releaseHash -eq 'missing') {
            throw 'could not hash the release copy'
        }
        Copy-Item -LiteralPath $releasePath -Destination $temporaryPath -Force
        $temporaryHash = Get-ScriptHash -Path $temporaryPath
        if ($temporaryHash -ne $releaseHash) { throw 'release copy changed while it was being copied' }
        return @{ Path = $temporaryPath; Hash = $releaseHash }
    } catch {
        if ([System.IO.File]::Exists($temporaryPath)) {
            Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
        }
        throw
    } finally {
        if ($downloadPath -and [System.IO.File]::Exists($downloadPath)) {
            Remove-Item -LiteralPath $downloadPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Invoke-PendingLauncherUpdate {
    if (-not [System.IO.File]::Exists($PendingLauncherPath)) { return $false }
    $localVersion = Get-ScriptVersion -Path $LauncherPath -Header 'Launcher-Version'
    $localHash = Get-ScriptHash -Path $LauncherPath
    $releaseVersion = Get-ScriptVersion -Path $PendingLauncherPath -Header 'Launcher-Version'
    $releaseHash = Get-ScriptHash -Path $PendingLauncherPath
    try {
        if ($releaseVersion -eq 'missing' -or $releaseVersion -eq 'unknown') {
            throw 'pending launcher lacks a Launcher-Version header'
        }
        if (-not (Test-PowerShellFile -Path $PendingLauncherPath)) {
            throw 'pending launcher has PowerShell parse errors'
        }
        if ($releaseHash -eq 'unavailable' -or $releaseHash -eq 'missing') {
            throw 'could not hash the pending launcher'
        }

        Move-Item -LiteralPath $PendingLauncherPath -Destination $LauncherPath -Force
        $installedHash = Get-ScriptHash -Path $LauncherPath
        if ($installedHash -ne $releaseHash) { throw 'pending launcher hash changed during replacement' }
        Write-LauncherLog -Component 'launcher' -Action 'updated' -LocalVersion $releaseVersion `
            -LocalHash $installedHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash `
            -Detail 'applied sentinel-launcher.ps1.next'
        return $true
    } catch {
        Write-LauncherLog -Component 'launcher' -Action 'kept' -LocalVersion $localVersion `
            -LocalHash $localHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash `
            -Detail ('pending update: ' + $_.Exception.Message)
        return $false
    }
}

function Invoke-LauncherSync {
    $localVersion = Get-ScriptVersion -Path $LauncherPath -Header 'Launcher-Version'
    $localHash = Get-ScriptHash -Path $LauncherPath
    $releaseVersion = 'unavailable'
    $releaseHash = 'unavailable'
    $temporaryPath = $null
    try {
        $identity = Get-OwnerWslIdentity
        $releaseCopy = Get-ReleaseFileCopy -Identity $identity -FileName 'sentinel-launcher.ps1'
        $temporaryPath = $releaseCopy.Path
        $releaseHash = $releaseCopy.Hash
        $releaseVersion = Get-ScriptVersion -Path $temporaryPath -Header 'Launcher-Version'
        if ($localHash -eq $releaseHash) {
            Write-LauncherLog -Component 'launcher' -Action 'kept' -LocalVersion $localVersion `
                -LocalHash $localHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash
            return $false
        }
        if ($releaseVersion -eq 'missing' -or $releaseVersion -eq 'unknown') {
            throw 'release launcher lacks a Launcher-Version header'
        }
        if (-not (Test-PowerShellFile -Path $temporaryPath)) {
            throw 'release launcher has PowerShell parse errors'
        }

        try {
            # PowerShell has already read this script. Verify that Windows permits the same-directory
            # atomic move over the running launcher's path before choosing the .next fallback.
            Move-Item -LiteralPath $temporaryPath -Destination $LauncherPath -Force
            $temporaryPath = $null
            $installedHash = Get-ScriptHash -Path $LauncherPath
            if ($installedHash -ne $releaseHash) { throw 'launcher hash changed during replacement' }
            Write-LauncherLog -Component 'launcher' -Action 'updated' -LocalVersion $releaseVersion `
                -LocalHash $installedHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash
            return $true
        } catch {
            $moveDetail = $_.Exception.Message
            if (-not $temporaryPath -or -not [System.IO.File]::Exists($temporaryPath)) { throw }
            Move-Item -LiteralPath $temporaryPath -Destination $PendingLauncherPath -Force
            $temporaryPath = $null
            $pendingHash = Get-ScriptHash -Path $PendingLauncherPath
            if ($pendingHash -ne $releaseHash) { throw 'staged launcher hash changed during replacement' }
            Write-LauncherLog -Component 'launcher' -Action 'staged' -LocalVersion $localVersion `
                -LocalHash $localHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash `
                -Detail ("direct replacement failed; staged .next: $moveDetail")
            return $true
        }
    } catch {
        Write-LauncherLog -Component 'launcher' -Action 'kept' -LocalVersion $localVersion `
            -LocalHash $localHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash `
            -Detail $_.Exception.Message
        return $false
    } finally {
        if ($temporaryPath -and [System.IO.File]::Exists($temporaryPath)) {
            Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Invoke-SentinelSync {
    $localVersion = Get-ScriptVersion -Path $LocalSentinelPath -Header 'Sentinel-Version'
    $localHash = Get-ScriptHash -Path $LocalSentinelPath
    $releaseVersion = 'unavailable'
    $releaseHash = 'unavailable'
    $temporaryPath = $null
    try {
        $identity = Get-OwnerWslIdentity
        $releaseCopy = Get-ReleaseFileCopy -Identity $identity -FileName 'owner-sentinel.ps1'
        $temporaryPath = $releaseCopy.Path
        $releaseHash = $releaseCopy.Hash
        $releaseVersion = Get-ScriptVersion -Path $temporaryPath -Header 'Sentinel-Version'
        if ($localHash -eq $releaseHash) {
            Write-LauncherLog -Component 'sentinel' -Action 'kept' -LocalVersion $localVersion `
                -LocalHash $localHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash
            return $false
        }

        Stop-LocalSentinels
        Move-Item -LiteralPath $temporaryPath -Destination $LocalSentinelPath -Force
        $temporaryPath = $null
        $localVersion = Get-ScriptVersion -Path $LocalSentinelPath -Header 'Sentinel-Version'
        $localHash = Get-ScriptHash -Path $LocalSentinelPath
        Write-LauncherLog -Component 'sentinel' -Action 'updated' -LocalVersion $localVersion `
            -LocalHash $localHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash
        return $true
    } catch {
        Write-LauncherLog -Component 'sentinel' -Action 'kept' -LocalVersion $localVersion `
            -LocalHash $localHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash `
            -Detail $_.Exception.Message
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

$launcherMutex = $null
$ownsLauncherMutex = $false
try {
    $launcherMutex = [System.Threading.Mutex]::new($false, $LauncherMutexName)
    try { $ownsLauncherMutex = $launcherMutex.WaitOne(0, $false) }
    catch [System.Threading.AbandonedMutexException] { $ownsLauncherMutex = $true }
} catch {
    Write-LauncherLog -Component 'launcher' -Action 'mutex-error' `
        -LocalVersion (Get-ScriptVersion -Path $LauncherPath -Header 'Launcher-Version') `
        -LocalHash (Get-ScriptHash -Path $LauncherPath) -ReleaseVersion 'unavailable' `
        -ReleaseHash 'unavailable' -Detail $_.Exception.Message
    exit 1
}

if (-not $ownsLauncherMutex) {
    Write-LauncherLog -Component 'launcher' -Action 'second-instance' `
        -LocalVersion (Get-ScriptVersion -Path $LauncherPath -Header 'Launcher-Version') `
        -LocalHash (Get-ScriptHash -Path $LauncherPath) -ReleaseVersion 'not-checked' `
        -ReleaseHash 'not-checked' -Detail 'another launcher owns the mutex; exiting'
    $launcherMutex.Dispose()
    exit 0
}

try {
    # A previous run uses .next only when Windows refused a move over the running script.
    # Apply it before starting a sentinel or checking the release again, then restart into it.
    if (Invoke-PendingLauncherUpdate) {
        Stop-LocalSentinels
        exit $LauncherUpdateExitCode
    }

    Write-LauncherLog -Component 'launcher' -Action 'started' `
        -LocalVersion (Get-ScriptVersion -Path $LauncherPath -Header 'Launcher-Version') `
        -LocalHash (Get-ScriptHash -Path $LauncherPath) -ReleaseVersion 'not-checked' `
        -ReleaseHash 'not-checked'

    $child = $null
    $nextUpdate = [DateTime]::MinValue
    while ($true) {
        if ([DateTime]::UtcNow -ge $nextUpdate) {
            if (Invoke-LauncherSync) {
                Stop-SentinelChild -Child $child
                $child = $null
                exit $LauncherUpdateExitCode
            }

            $updated = Invoke-SentinelSync
            $nextUpdate = [DateTime]::UtcNow.Add($UpdateInterval)
            if ($updated -and $null -ne $child) {
                Stop-SentinelChild -Child $child
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

        $waitMilliseconds = [int][Math]::Max(
            1, [Math]::Min([int]::MaxValue, ($nextUpdate - [DateTime]::UtcNow).TotalMilliseconds)
        )
        if ($child.WaitForExit($waitMilliseconds)) {
            $child.Dispose()
            $child = $null
            Start-Sleep -Seconds $RestartDelaySeconds
        }
    }
} finally {
    if ($ownsLauncherMutex) {
        try { $launcherMutex.ReleaseMutex() } catch { }
    }
    if ($null -ne $launcherMutex) { $launcherMutex.Dispose() }
}
