# Keep the Phase 3 queue running. A scheduled task runs this every 5 minutes; install with
#     powershell -File scripts\watchdog.ps1 -Install
# If the queue is not running (WSL shut down, reboot, crash), it is relaunched inside a hidden
# wsl.exe that stays alive for the queue's lifetime, which also keeps the distro from being
# shut down. Every stage of the queue resumes from its last checkpoint.
param([switch]$Install)
$task = "curated-vla-watchdog"
$distro = "cvla"  # WSL distro on C:\WSL\cvla (the original 'Ubuntu' lived on a failing USB drive)
$repo = Split-Path $PSScriptRoot -Parent
if ($Install) {
    # schtasks works without elevation (Register-ScheduledTask was denied here). Its default
    # 3-day execution limit would kill the queue's wsl.exe, so clear it through the task XML.
    schtasks.exe /Create /F /TN $task /SC MINUTE /MO 5 `
        /TR "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File \`"$PSCommandPath`"" | Out-Null
    $xml = (schtasks.exe /Query /TN $task /XML) -join "`n"
    $xml = $xml -replace '\s*<ExecutionTimeLimit>[^<]*</ExecutionTimeLimit>', ''
    $xml = $xml -replace '<Settings>', "<Settings>`n    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>"
    $tmp = Join-Path $env:TEMP "$task.xml"
    [IO.File]::WriteAllText($tmp, $xml, (New-Object Text.UnicodeEncoding))
    schtasks.exe /Create /F /TN $task /XML $tmp
    Remove-Item $tmp
    return
}
$drive = $repo.Substring(0, 1).ToLower()
$wslRepo = "/mnt/$drive" + ($repo.Substring(2) -replace '\\', '/')
# A disk error can leave the WSL filesystem unreadable (even /bin/bash fails to exec); only a
# restart of the VM clears it. Probe with plain reads before relying on the bash health check.
wsl.exe -d $distro -- cat /bin/cp /bin/bash /usr/bin/python3 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Add-Content (Join-Path $env:TEMP "curated-vla-watchdog.log") "$(Get-Date -Format s) WSL filesystem broken (exit $LASTEXITCODE), restarting WSL"
    wsl.exe --shutdown
    Start-Sleep 10
}
wsl.exe -d $distro -- bash "$wslRepo/scripts/watchdog.sh"
$code = $LASTEXITCODE
if ($code -ne 0) {
    # run_phase3.sh redirects its own output to the queue log.
    Start-Process wsl.exe -WindowStyle Hidden -ArgumentList "-d $distro -- bash $wslRepo/scripts/run_phase3.sh"
}
