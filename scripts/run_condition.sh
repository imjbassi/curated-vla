#!/usr/bin/env bash
# One Phase 3 model end to end: pretrain on a condition's episode list, post-train on
# LIBERO with the shared 30k x 32 recipe, then the full 4-suite protocol (eval seed 1000,
# replan every step). Resumable stage by stage: a stage whose output exists is skipped.
#
# Usage: scripts/run_condition.sh CONDITION SEED
#   scripts/run_condition.sh A_random 1000
#   scripts/run_condition.sh B_curated 1001
set -euo pipefail

COND="$1"
SEED="$2"
CVLA_HOME="${CVLA_HOME:-$HOME/cvla}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
NAME="${COND}_seed${SEED}"
PRE="$CVLA_HOME/outputs/pretrain/$NAME"
# shellcheck disable=SC1091
source "$CVLA_HOME/venv/bin/activate"
cd "$REPO"

if [ ! -d "$PRE/checkpoints/last/pretrained_model" ] || ! grep -q '"step": 156250' "$PRE/train_log.jsonl" 2>/dev/null; then
    PYTHONPATH=. python -W ignore -m pretrain.train \
        --index "$CVLA_HOME/pool/index_all.pkl" \
        --video-root "$CVLA_HOME/pool_video" \
        --keys "conditions/${COND}_seed${SEED}.parquet" \
        --out "$PRE" \
        --samples 5000000 \
        --save-every 5000 \
        --seed "$SEED"
fi

STEPS=30000 bash "$HERE/posttrain_libero.sh" "$NAME" "$PRE/checkpoints/last/pretrained_model" "$SEED"
