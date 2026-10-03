#!/usr/bin/env bash
# Full LIBERO protocol for one policy and seed: each suite as its own run
# (so partial progress survives), then a summary table.
#
# Usage: eval/run_protocol.sh [POLICY] [SEED]
set -euo pipefail

POLICY="${1:-HuggingFaceVLA/smolvla_libero}"
SEED="${2:-1000}"
SUITES=(libero_object libero_spatial libero_goal libero_10)

CVLA_HOME="${CVLA_HOME:-$HOME/cvla}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS=()
for suite in "${SUITES[@]}"; do
    bash "$HERE/eval_libero.sh" "$POLICY" "$SEED" "$suite" 10 > /dev/null 2>&1
    RUNS+=("$(ls -td "$CVLA_HOME"/outputs/eval/*/ | head -1)")
    echo "done: $suite -> ${RUNS[-1]}"
done

# shellcheck disable=SC1091
source "$CVLA_HOME/venv/bin/activate"
python "$HERE/summarize.py" "${RUNS[@]}" | tee "$CVLA_HOME/outputs/eval/summary_$(basename "$POLICY")_seed${SEED}.txt"
