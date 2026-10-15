#!/usr/bin/env bash
# Phase 3 queue health check for scripts/watchdog.ps1. Exit codes:
#   0  queue running or finished
#   1  queue not running: relaunch it
#   2  the WSL filesystem is broken (reads fail after a disk error): restart WSL, then relaunch
# The launch itself happens from Windows, because a process started through a one-shot
# `wsl.exe -- ...` call is killed when that call returns.
LOGS="${CVLA_HOME:-$HOME/cvla}/outputs"
cat /bin/cp /usr/bin/python3 > /dev/null 2>&1 || exit 2
grep -q "end B_curated_seed1002" "$LOGS/phase3_queue.log" 2>/dev/null && exit 0
pgrep -f run_phase3.sh > /dev/null && exit 0
echo "$(date '+%F %T') watchdog: queue not running, relaunching" >> "$LOGS/phase3_queue.log"
exit 1
