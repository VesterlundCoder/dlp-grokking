#!/bin/bash
# =============================================================================
# Multi-seed M04 runs for confidence intervals
#
# M04 (427k params, d=128, H=4, p=113) with seeds 123 and 456
# Seed 42 already completed — this runs the remaining 2 seeds
#
# Usage:
#   bash run_multiseed.sh
#
# Each run: ~45 min on MPS (50k epochs, early stop at 99% test)
# Total: ~1.5h for 2 seeds
# =============================================================================

set -euo pipefail
cd /Users/davidsvensson/Desktop/dlp_grokking

PRIME=113
MODEL=M04
N_LAYERS=2
EPOCHS=50000
CKPT_INTERVAL=500
EVAL_INTERVAL=10
REP_INTERVAL=500
TRAIN_FRAC=0.30

for SEED in 123 456; do
    OUTDIR="results/local_probe/${MODEL}_s${SEED}"

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
done

echo ""
echo "============================================================"
echo "  Multi-seed training complete: $(date)"
echo "  Probe with:"
echo "    for s in 42 123 456; do"
echo "      python3 src/probe_equations.py --output-dir results/local_probe/M04_s\${s}"
echo "    done"
echo "============================================================"
