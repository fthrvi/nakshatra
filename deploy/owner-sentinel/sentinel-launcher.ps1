# Launcher-Version: 2026101302
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$InstallDirectory = $PSScriptRoot
$EnvironmentFile = Join-Path $InstallDirectory 'owner-lease.env'
$LauncherPath = $PSCommandPath
$LocalSentinelPath = Join-Path $InstallDirectory 'owner-sentinel.ps1'
$LogPath = Join-Path $InstallDirectory 'launcher.log'
$PowerShellPath = Join-Path $PSHOME 'powershell.exe'
$UpdateInterval = [TimeSpan]::FromHours(6)
$RestartDelaySeconds = 5
$MaximumLogLines = 200
$LauncherMutexWait = [TimeSpan]::FromSeconds(30)

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

function Install-VerifiedFile {
    param([string]$TemporaryPath, [string]$TargetPath)
    if ([System.IO.File]::Exists($TargetPath)) {
        $backupPath = "${TargetPath}.bak"
        [System.IO.File]::Replace($TemporaryPath, $TargetPath, $backupPath)
    } else {
        [System.IO.File]::Move($TemporaryPath, $TargetPath)
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
        # Best effort; the replacement below reports a failure and the current sentinel is restarted.
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
    # Read the signed release copy only through wsl.exe, and only when the distro is already running.
    # A distro-targeted wsl.exe call would otherwise boot it and start the worker that this sentinel
    # must never summon.
    param([hashtable]$Identity, [string]$FileName, [string]$Destination)
    try {
        if ([System.IO.Path]::GetFileName($FileName) -ne $FileName) { return $false }
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
    # Both scripts use this exact already-running-WSL fetch path.
    param([hashtable]$Identity, [string]$FileName)
    $temporaryPath = Join-Path $InstallDirectory ('{0}.{1}.tmp' -f $FileName, [Guid]::NewGuid().ToString('N'))
    try {
        if (-not (Save-ReleaseFileViaWsl -Identity $Identity -FileName $FileName -Destination $temporaryPath)) {
            throw 'release copy unavailable'
        }
        $releaseHash = Get-ScriptHash -Path $temporaryPath
        if ($releaseHash -eq 'unavailable' -or $releaseHash -eq 'missing') {
            throw 'could not hash the release copy'
        }
        return @{ Path = $temporaryPath; Hash = $releaseHash }
    } catch {
        if ([System.IO.File]::Exists($temporaryPath)) {
            Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
        }
        throw
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
        $localVersionNumber = 0L
        $releaseVersionNumber = 0L
        if (-not [Int64]::TryParse($localVersion, [ref]$localVersionNumber)) {
            throw 'installed launcher has a non-integer Launcher-Version header'
        }
        if (-not [Int64]::TryParse($releaseVersion, [ref]$releaseVersionNumber)) {
            throw 'release launcher has a non-integer Launcher-Version header'
        }
        if ($releaseVersionNumber -lt $localVersionNumber) {
            throw 'release launcher version is lower than the running launcher version'
        }

        # The temporary file is the release copy just fetched. Equal-version replacements are
        # permitted because its hash was computed from these exact bytes above.
        Install-VerifiedFile -TemporaryPath $temporaryPath -TargetPath $LauncherPath
        $temporaryPath = $null
        Write-LauncherLog -Component 'launcher' -Action 'updated' -LocalVersion $releaseVersion `
            -LocalHash $releaseHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash
        return $true
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

        if ($releaseVersion -eq 'missing' -or $releaseVersion -eq 'unknown') {
            throw 'release sentinel lacks a Sentinel-Version header'
        }
        if (-not (Test-PowerShellFile -Path $temporaryPath)) {
            throw 'release sentinel has PowerShell parse errors'
        }

        Stop-LocalSentinels
        Install-VerifiedFile -TemporaryPath $temporaryPath -TargetPath $LocalSentinelPath
        $temporaryPath = $null
        Write-LauncherLog -Component 'sentinel' -Action 'updated' -LocalVersion $releaseVersion `
            -LocalHash $releaseHash -ReleaseVersion $releaseVersion -ReleaseHash $releaseHash
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

function Start-LauncherSuccessor {
    $arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$LauncherPath`""
    Start-Process -FilePath $PowerShellPath -ArgumentList $arguments -WorkingDirectory $InstallDirectory `
        -WindowStyle Hidden | Out-Null
}

$launcherMutex = $null
$ownsLauncherMutex = $false
try {
    # One launcher per USER, across every logon session: a Global\ name carrying the user's SID.
    # Creating a named mutex in Global\ needs no privilege. There is deliberately no Local\ fallback:
    # Local\ is per-SESSION, so two sessions could each "win" and run two sentinels.
    $currentUserSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $launcherMutex = [System.Threading.Mutex]::new($false, "Global\NakshatraOwnerSentinelLauncher-$currentUserSid")
    try { $ownsLauncherMutex = $launcherMutex.WaitOne($LauncherMutexWait, $false) }
    catch [System.Threading.AbandonedMutexException] { $ownsLauncherMutex = $true }
} catch {
    # The sentinel is what tells the network the owner is busy, so liveness wins over exclusivity:
    # without a lock we still run (a duplicate sentinel only sends a second identical heartbeat).
    Write-LauncherLog -Component 'launcher' -Action 'mutex-unavailable' `
        -LocalVersion (Get-ScriptVersion -Path $LauncherPath -Header 'Launcher-Version') `
        -LocalHash (Get-ScriptHash -Path $LauncherPath) -ReleaseVersion 'not-checked' `
        -ReleaseHash 'not-checked' -Detail ('running unlocked: ' + $_.Exception.Message)
    if ($null -ne $launcherMutex) { $launcherMutex.Dispose() }
    $launcherMutex = $null
    $ownsLauncherMutex = $true
}

if (-not $ownsLauncherMutex) {
    Write-LauncherLog -Component 'launcher' -Action 'second-instance' `
        -LocalVersion (Get-ScriptVersion -Path $LauncherPath -Header 'Launcher-Version') `
        -LocalHash (Get-ScriptHash -Path $LauncherPath) -ReleaseVersion 'not-checked' `
        -ReleaseHash 'not-checked' -Detail 'could not acquire the launcher mutex within 30 seconds; exiting'
    if ($null -ne $launcherMutex) { $launcherMutex.Dispose() }
    exit 0
}

try {
    Write-LauncherLog -Component 'launcher' -Action 'started' `
        -LocalVersion (Get-ScriptVersion -Path $LauncherPath -Header 'Launcher-Version') `
        -LocalHash (Get-ScriptHash -Path $LauncherPath) -ReleaseVersion 'not-checked' `
        -ReleaseHash 'not-checked'

    $child = $null
    $nextUpdate = [DateTime]::MinValue
    while ($true) {
        if ([DateTime]::UtcNow -ge $nextUpdate) {
            if (Invoke-LauncherSync) {
                # Start the successor FIRST, while this launcher still holds the mutex and its sentinel
                # still runs: the successor waits up to 30 s for the mutex. If it cannot be started, this
                # launcher simply carries on (the new file is already in place for the next start).
                $handedOff = $false
                try { Start-LauncherSuccessor; $handedOff = $true }
                catch {
                    Write-LauncherLog -Component 'launcher' -Action 'handoff-failed' `
                        -LocalVersion (Get-ScriptVersion -Path $LauncherPath -Header 'Launcher-Version') `
                        -LocalHash (Get-ScriptHash -Path $LauncherPath) -ReleaseVersion 'not-checked' `
                        -ReleaseHash 'not-checked' -Detail ('keeping this launcher: ' + $_.Exception.Message)
                }
                if ($handedOff) {
                    Stop-SentinelChild -Child $child
                    $child = $null
                    if ($null -ne $launcherMutex) {
                        if ($ownsLauncherMutex) { try { $launcherMutex.ReleaseMutex() } catch { } }
                        $launcherMutex.Dispose()
                        $launcherMutex = $null
                    }
                    $ownsLauncherMutex = $false
                    exit 0
                }
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
