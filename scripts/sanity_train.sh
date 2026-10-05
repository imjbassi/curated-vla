#!/usr/bin/env bash
# Training-pipeline sanity check: short LIBERO post-training from the published
# SmolVLA base, then load the final checkpoint into the eval harness.
# Checks: loss decreases, checkpoints save, checkpoint runs closed-loop.
#
# Usage: scripts/sanity_train.sh [STEPS] [BATCH_SIZE]
set -euo pipefail

STEPS="${1:-3000}"
BATCH_SIZE="${2:-32}"
BASE_POLICY="${BASE_POLICY:-lerobot/smolvla_base}"  # or a local pretrained_model dir from pretrain/train.py
TASK_IDS_EVAL="${TASK_IDS_EVAL:-[0,1]}"
CVLA_HOME="${CVLA_HOME:-$HOME/cvla}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$CVLA_HOME/venv/bin/activate"

# smolvla_base expects cameras camera1..3; LIBERO provides image (agentview) and
# image2 (wrist). Providing a subset of the policy's cameras is allowed.
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

OUT="$CVLA_HOME/outputs/train/sanity_$(basename "$(dirname "$BASE_POLICY")" | tr '/' '_')_libero_${STEPS}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$OUT")"

lerobot-train \
    --policy.path="$BASE_POLICY" \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/libero \
    --dataset.revision="${LIBERO_REVISION:-a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4}" \
    --dataset.video_backend=torchcodec \
    --rename_map="$RENAME" \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --save_freq=$((STEPS / 2)) \
    --log_freq=50 \
    --num_workers=8 \
    --wandb.enable=false \
    --seed=1000 \
    --output_dir="$OUT" \
    2>&1 | tee "$OUT.log"

CKPT="$OUT/checkpoints/last/pretrained_model"
ls "$OUT/checkpoints"
TAG=sanity BATCH_SIZE=10 TASK_IDS="$TASK_IDS_EVAL" bash "$HERE/../eval/eval_libero.sh" "$CKPT" 1000 libero_object 10 \
    --rename_map="$RENAME"
