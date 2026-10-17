#!/usr/bin/env bash
# Phase 3 queue: A vs B, 3 seeds, interleaved so each finished pair gives an A-vs-B
# comparison early. Every stage is resumable, so rerunning this script after an
# interruption continues where it stopped.
#
# Usage: scripts/run_phase3.sh    (normally launched by scripts/watchdog.ps1)
# Queue progress goes to ~/cvla/outputs/phase3_queue.log, per-model output to phase3_<name>.log.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${CVLA_HOME:-$HOME/cvla}/outputs"
exec >> "$LOGS/phase3_queue.log" 2>&1
for seed in 1000 1001 1002; do
    for cond in A_random B_curated; do
        # A failed stage (crash, WSL restart) is retried on the same model; every stage resumes
        # from its last checkpoint, so a retry only redoes the tail. Never skip to the next model.
        for attempt in 1 2 3 4 5; do
            echo "$(date '+%F %T') start ${cond}_seed${seed} (attempt $attempt)"
            if bash "$HERE/run_condition.sh" "$cond" "$seed" >> "$LOGS/phase3_${cond}_seed${seed}.log" 2>&1; then
                echo "$(date '+%F %T') end ${cond}_seed${seed}"
                break
            fi
            echo "$(date '+%F %T') ${cond}_seed${seed} failed (attempt $attempt), see phase3_${cond}_seed${seed}.log"
            [ $attempt -eq 5 ] && { echo "giving up on ${cond}_seed${seed}"; exit 1; }
            sleep 60
        done
    done
done
