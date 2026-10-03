#!/usr/bin/env bash
# Phase 0: short SmolVLA training run (VLM-initialized, the config we will
# actually pretrain with) to measure samples/sec on this GPU.
#
# Usage: scripts/measure_throughput.sh [BATCH_SIZE] [STEPS] [extra lerobot-train args...]
#   scripts/measure_throughput.sh 64 150 --policy.use_amp=true
set -euo pipefail

BATCH_SIZE="${1:-32}"
STEPS="${2:-300}"

CVLA_HOME="${CVLA_HOME:-$HOME/cvla}"
# shellcheck disable=SC1091
source "$CVLA_HOME/venv/bin/activate"

OUT="$CVLA_HOME/outputs/throughput/bs${BATCH_SIZE}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$OUT")"

lerobot-train \
    --policy.type=smolvla \
    --policy.load_vlm_weights=true \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/libero \
    --dataset.revision="${LIBERO_REVISION:-a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4}" \
    --dataset.video_backend=pyav \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --log_freq=25 \
    --save_checkpoint=false \
    --num_workers="${NUM_WORKERS:-8}" \
    --wandb.enable=false \
    --seed=1000 \
    --output_dir="$OUT" \
    "${@:3}" \
    2>&1 | tee "$OUT.log"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "Log: $OUT.log"
