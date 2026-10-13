$ErrorActionPreference = 'Stop'
$taskName = 'Nakshatra Owner Sentinel'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue; Unregister-ScheduledTask -TaskName $taskName -Confirm:$false }
Write-Host 'Removed the Nakshatra Owner Sentinel task. Configuration files were left in place.'
