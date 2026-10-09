#!/usr/bin/env bash
# Phase 3 queue: A vs B, 3 seeds, interleaved so each finished pair gives an A-vs-B
# comparison early. Every stage is resumable, so rerunning this script after an
# interruption continues where it stopped.
#
# Usage: scripts/run_phase3.sh    (run detached; logs to ~/cvla/outputs/phase3_<name>.log)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${CVLA_HOME:-$HOME/cvla}/outputs"
for seed in 1000 1001 1002; do
    for cond in A_random B_curated; do
        echo "$(date '+%F %T') start ${cond}_seed${seed}"
        bash "$HERE/run_condition.sh" "$cond" "$seed" > "$LOGS/phase3_${cond}_seed${seed}.log" 2>&1
        echo "$(date '+%F %T') end ${cond}_seed${seed} (exit $?)"
    done
done
