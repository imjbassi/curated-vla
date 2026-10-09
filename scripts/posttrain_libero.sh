#!/usr/bin/env bash
# Phase 2/3 post-training recipe on LIBERO, shared by every model (control and ours),
# followed by the full 4-suite LIBERO protocol.
#
# Reduced recipe (decided 2026-10-06): 30k steps x batch 32 (~1M samples, ~5-6 h on
# the 4070) instead of the SmolVLA paper's 100k x 64 (~33 h). The smolvla_base control
# is judged against our Phase 0 reproduction (72.2%); if it falls well short, the
# control is rerun once at the paper recipe to separate budget from bugs.
#
# Paper-budget control (2026-10-07): the paper's 100k x 64 does not fit in 12 GB and
# lerobot-train has no gradient accumulation, so it is approximated by the same number
# of samples at batch 32:
#   STEPS=200000 SAVE_FREQ=50000 scripts/posttrain_libero.sh control_smolvla_base_paperbudget lerobot/smolvla_base
#
# Usage:
#   scripts/posttrain_libero.sh NAME BASE_POLICY [SEED]
#   scripts/posttrain_libero.sh control_smolvla_base lerobot/smolvla_base 1000
#   scripts/posttrain_libero.sh baseline_random ~/cvla/outputs/pretrain/<run>/checkpoints/last/pretrained_model 1000
set -euo pipefail

NAME="$1"
BASE_POLICY="$2"
SEED="${3:-1000}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_SEED="${EVAL_SEED:-1000}"

CVLA_HOME="${CVLA_HOME:-$HOME/cvla}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$CVLA_HOME/venv/bin/activate"

# LIBERO cameras -> the camera names our models (and smolvla_base) use.
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
OUT="$CVLA_HOME/outputs/posttrain/${NAME}_seed${SEED}"
mkdir -p "$(dirname "$OUT")"

if [ -d "$OUT/checkpoints/last/pretrained_model" ] && [ -d "$OUT/checkpoints/$(printf '%06d' "$STEPS")" ]; then
    echo "post-training already done: $OUT"
else
rm -rf "$OUT"  # lerobot-train refuses an existing output dir; a partial run restarts
lerobot-train \
    --policy.path="$BASE_POLICY" \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/libero \
    --dataset.revision="${LIBERO_REVISION:-a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4}" \
    --dataset.video_backend=torchcodec \
    --rename_map="$RENAME" \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --save_freq="${SAVE_FREQ:-10000}" \
    --log_freq=200 \
    --num_workers=8 \
    --wandb.enable=false \
    --seed="$SEED" \
    --output_dir="$OUT" \
    2>&1 | tee "$OUT.log"
fi

CKPT="$OUT/checkpoints/last/pretrained_model"
# TAG names the eval run; run_protocol resumes from its manifest if interrupted.
TAG="$NAME" EXTRA_EVAL_ARGS="--rename_map=$RENAME" bash "$HERE/../eval/run_protocol.sh" "$CKPT" "$EVAL_SEED" \
    | tee "$OUT.eval.log"
