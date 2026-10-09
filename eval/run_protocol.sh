#!/usr/bin/env bash
# Full LIBERO protocol for one policy and seed, then a summary table.
#
# Each task runs in its own lerobot-eval process: with hard resets the simulator
# and renderer are rebuilt every episode and memory accumulates within a process
# (a whole libero_10 suite in one process was OOM-killed at 15 GB in WSL).
#
# Robust to lerobot-eval exiting non-zero AFTER writing its results (seen
# intermittently at teardown, 2026-10-08): a task counts as done when its log has a
# result, is retried once otherwise, and finished tasks are recorded in a manifest so
# a rerun resumes where it stopped.
#
# Usage: eval/run_protocol.sh [POLICY] [SEED]
#   SUITES="libero_10" eval/run_protocol.sh ...   # subset of suites
#   TAG=name                                      # names eval dirs, manifest and summary (default: policy basename)
#   EXTRA_EVAL_ARGS="--rename_map=..."            # ONE extra lerobot-eval argument, passed unsplit
#                                                   (JSON values contain spaces)
set -uo pipefail

POLICY="${1:-HuggingFaceVLA/smolvla_libero}"
SEED="${2:-1000}"
read -r -a SUITES <<< "${SUITES:-libero_object libero_spatial libero_goal libero_10}"
export TAG="${TAG:-}"
EXTRA=()
[ -n "${EXTRA_EVAL_ARGS:-}" ] && EXTRA=("$EXTRA_EVAL_ARGS")

CVLA_HOME="${CVLA_HOME:-$HOME/cvla}"
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${TAG:-$(basename "$POLICY")}"
MANIFEST="$CVLA_HOME/outputs/eval/manifest_${NAME}_seed${SEED}.tsv"
mkdir -p "$(dirname "$MANIFEST")"
touch "$MANIFEST"

has_result() { grep -q "'pc_success'" "$1/eval.log" 2>/dev/null; }

for suite in "${SUITES[@]}"; do
    for task_id in 0 1 2 3 4 5 6 7 8 9; do
        if grep -qP "^${suite}\t${task_id}\t" "$MANIFEST"; then
            continue
        fi
        for attempt in 1 2; do
            TASK_IDS="[$task_id]" bash "$HERE/eval_libero.sh" "$POLICY" "$SEED" "$suite" 10 "${EXTRA[@]}" \
                > /dev/null 2>&1
            run="$(ls -td "$CVLA_HOME"/outputs/eval/*/ | head -1)"
            if has_result "$run"; then
                printf "%s\t%s\t%s\n" "$suite" "$task_id" "$run" >> "$MANIFEST"
                echo "done: $suite task $task_id -> $run"
                break
            fi
            echo "no result: $suite task $task_id (attempt $attempt) -> $run"
        done
    done
done

# shellcheck disable=SC1091
source "$CVLA_HOME/venv/bin/activate"
mapfile -t RUNS < <(cut -f3 "$MANIFEST")
python "$HERE/summarize.py" "${RUNS[@]}" | tee "$CVLA_HOME/outputs/eval/summary_${NAME}_seed${SEED}.txt"
