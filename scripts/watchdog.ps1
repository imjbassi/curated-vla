# Keep WSL alive and the Phase 3 queue running. Registered as a scheduled task that
# runs every 5 minutes by: powershell -File scripts\watchdog.ps1 -Install
# (schtasks works without elevation; Register-ScheduledTask was denied here.)
param([switch]$Install)
$repo = Split-Path $PSScriptRoot -Parent
if ($Install) {
    schtasks.exe /Create /F /TN curated-vla-watchdog /SC MINUTE /MO 5 `
        /TR "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File \`"$PSCommandPath\`""
    return
}
# WSL shuts the distro down when no Windows process holds it open.
$alive = Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" |
    Where-Object { $_.CommandLine -like "*sleep infinity*" }
if (-not $alive) {
    Start-Process wsl.exe -ArgumentList '-d','Ubuntu','--','sleep','infinity' -WindowStyle Hidden
    Start-Sleep 5
}
$drive = $repo.Substring(0,1).ToLower()
$wslRepo = "/mnt/$drive" + ($repo.Substring(2) -replace '\\','/')
wsl.exe -d Ubuntu -- bash "$wslRepo/scripts/watchdog.sh"
