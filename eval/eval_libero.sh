#!/usr/bin/env bash
# Closed-loop LIBERO evaluation using the published protocol:
# 4 suites x 10 tasks x 10 episodes, hard resets, one task at a time.
#
# Usage:
#   eval/eval_libero.sh [POLICY] [SEED] [SUITES] [N_EPISODES]
#   eval/eval_libero.sh HuggingFaceVLA/smolvla_libero 1000
#   eval/eval_libero.sh HuggingFaceVLA/smolvla_libero 1000 libero_object 1   # smoke test
#
# BATCH_SIZE (env var) runs that many episodes of a task in parallel. Keeping it
# equal to N_EPISODES runs each task in a single batch, so two policies evaluated
# with the same seed see the same initial states. TASK_IDS restricts to e.g. "[0]".
# Envs run in one process by default (ASYNC_ENVS=false): each async worker costs
# ~2 GB RAM, and policy inference, not simulation, is the bottleneck anyway.
set -euo pipefail

POLICY="${1:-HuggingFaceVLA/smolvla_libero}"
SEED="${2:-1000}"
SUITES="${3:-libero_spatial,libero_object,libero_goal,libero_10}"
N_EPISODES="${4:-10}"
BATCH_SIZE="${BATCH_SIZE:-$N_EPISODES}"
EXTRA_ARGS=()
if [ -n "${TASK_IDS:-}" ]; then
    EXTRA_ARGS+=("--env.task_ids=$TASK_IDS")
fi

CVLA_HOME="${CVLA_HOME:-$HOME/cvla}"
# shellcheck disable=SC1091
source "$CVLA_HOME/venv/bin/activate"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

RUN_NAME="$(basename "$POLICY")_seed${SEED}_$(date +%Y%m%d-%H%M%S)"
OUT="$CVLA_HOME/outputs/eval/$RUN_NAME"
mkdir -p "$OUT"

lerobot-eval \
    --policy.path="$POLICY" \
    --env.type=libero \
    --env.task="$SUITES" \
    --eval.batch_size="$BATCH_SIZE" \
    --eval.use_async_envs="${ASYNC_ENVS:-false}" \
    --eval.n_episodes="$N_EPISODES" \
    --env.max_parallel_tasks=1 \
    --seed="$SEED" \
    --output_dir="$OUT" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUT/eval.log"

echo "Results in $OUT"
