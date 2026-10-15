#!/usr/bin/env bash
# Relaunch the Phase 3 queue if it is not running and not finished. Called every few
# minutes by scripts/watchdog.ps1 (Windows Task Scheduler); every stage resumes.
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${CVLA_HOME:-$HOME/cvla}/outputs"
grep -q "end B_curated_seed1002" "$LOGS/phase3_queue.log" 2>/dev/null && exit 0
pgrep -f run_phase3.sh > /dev/null && exit 0
echo "$(date '+%F %T') watchdog: queue not running, relaunching" >> "$LOGS/phase3_queue.log"
setsid nohup bash "$HERE/run_phase3.sh" >> "$LOGS/phase3_queue.log" 2>&1 < /dev/null &
