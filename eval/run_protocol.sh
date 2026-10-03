#!/usr/bin/env bash
# Full LIBERO protocol for one policy and seed, then a summary table.
#
# Each task runs in its own lerobot-eval process: with hard resets the simulator
# and renderer are rebuilt every episode and memory accumulates within a process
# (a whole libero_10 suite in one process was OOM-killed at 15 GB in WSL).
#
# Usage: eval/run_protocol.sh [POLICY] [SEED]
#   SUITES="libero_10" eval/run_protocol.sh ...   # subset of suites
set -euo pipefail

POLICY="${1:-HuggingFaceVLA/smolvla_libero}"
SEED="${2:-1000}"
read -r -a SUITES <<< "${SUITES:-libero_object libero_spatial libero_goal libero_10}"

CVLA_HOME="${CVLA_HOME:-$HOME/cvla}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS=()
for suite in "${SUITES[@]}"; do
    for task_id in 0 1 2 3 4 5 6 7 8 9; do
        TASK_IDS="[$task_id]" bash "$HERE/eval_libero.sh" "$POLICY" "$SEED" "$suite" 10 > /dev/null 2>&1
        RUNS+=("$(ls -td "$CVLA_HOME"/outputs/eval/*/ | head -1)")
        echo "done: $suite task $task_id -> ${RUNS[-1]}"
    done
done

# shellcheck disable=SC1091
source "$CVLA_HOME/venv/bin/activate"
python "$HERE/summarize.py" "${RUNS[@]}" | tee "$CVLA_HOME/outputs/eval/summary_$(basename "$POLICY")_seed${SEED}.txt"
