#!/bin/bash
# =============================================================================
# Local training of 3 grokked models with frequent checkpointing
#
# M04: 425k params (d=128, H=4), p=113, 30/70 split, seed=42, 50k epochs
#      — First model to grok (~20k epochs on original run)
# M09: 3.6M params (d=384, H=12), p=113, 30/70 split, seed=42, 100k epochs
#      — Grokked on LUMI (test=0.988 at epoch 76k)
# M10: 5.6M params (d=480, H=15), p=113, 30/70 split, seed=456, 100k epochs
#      — Grokked on LUMI (test=0.992 at epoch 165k)
#
# Checkpoint every 500 steps, eval every 10, representation monitoring every 500
# Estimated time on Mac MPS: M04 ~45min, M09 ~3h, M10 ~5h
# =============================================================================

set -euo pipefail
cd /Users/davidsvensson/Desktop/dlp_grokking

PRIME=113
CKPT_INTERVAL=500
EVAL_INTERVAL=10
REP_INTERVAL=500
N_LAYERS=2

run_model() {
    local MODEL=$1
    local SEED=$2
    local EPOCHS=$3
    local TRAIN_FRAC=$4
    local OUTDIR="results/local_probe/${MODEL}_s${SEED}"

    echo ""
    echo "============================================================"
    echo "  Training ${MODEL} seed=${SEED} (${EPOCHS} epochs, frac=${TRAIN_FRAC})"
    echo "  Output: ${OUTDIR}"
    echo "  Time: $(date)"
    echo "============================================================"

    python3 src/trainer.py \
        --model-id "${MODEL}" \
        --prime "${PRIME}" \
        --n-layers "${N_LAYERS}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --train-frac "${TRAIN_FRAC}" \
        --output-dir "${OUTDIR}" \
        --eval-interval "${EVAL_INTERVAL}" \
        --checkpoint-interval "${CKPT_INTERVAL}" \
        --wd-max 0.30 \
        --wd-ramp-interval 1000 \
        --lr-drop-factor 1.0 \
        --monitor-reps \
        --rep-interval "${REP_INTERVAL}"

    echo ""
    echo "  ${MODEL} seed=${SEED} complete: $(date)"
    echo ""
}

# M04: 424k params, 30/70 split — grokked at ~20k epochs
run_model M04 42 50000 0.30

# M09: 3.6M params, 45/55 split (auto on LUMI) — grokked at ~76k epochs
run_model M09 42 100000 0.45

# M10: 5.6M params, 50/50 split (auto on LUMI) — grokked at ~165k epochs
# Note: may need >100k epochs to grok locally
run_model M10 456 200000 0.50

echo ""
echo "============================================================"
echo "  All training complete: $(date)"
echo "  Probe with: python3 src/probe_equations.py --output-dir results/local_probe/M04_s42"
echo "============================================================"
