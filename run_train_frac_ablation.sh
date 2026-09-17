#!/bin/bash
# =============================================================================
# Train_frac ablation for M04
#
# M04 (427k params, d=128, H=4, p=113) with train_frac ∈ {0.10, 0.20, 0.30, 0.40, 0.50, 0.60}
# 3 seeds each: 42, 123, 456
#
# Total: 18 runs × ~45 min = ~13.5h on MPS
# Each run: 50k epochs, early stop at 99% test, checkpoint every 500
#
# Usage:
#   bash run_train_frac_ablation.sh
#
# Results stored in: results/ablation/M04_frac{frac}_s{seed}/
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

FRACS=(0.10 0.20 0.30 0.40 0.50 0.60)
SEEDS=(42 123 456)

TOTAL=$(( ${#FRACS[@]} * ${#SEEDS[@]} ))
COUNT=0

for FRAC in "${FRACS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        OUTDIR="results/ablation/M04_frac${FRAC}_s${SEED}"

        # Skip if already completed (summary.json exists with state E or G)
        if [ -f "${OUTDIR}/summary.json" ]; then
            STATE=$(python3 -c "import json; d=json.load(open('${OUTDIR}/summary.json')); print(d.get('state',''))" 2>/dev/null || echo "")
            if [ "${STATE}" = "E" ] || [ "${STATE}" = "G" ]; then
                echo "[${COUNT}/${TOTAL}] SKIP ${OUTDIR} (already completed, state=${STATE})"
                continue
            fi
        fi

        echo ""
        echo "============================================================"
        echo "  [${COUNT}/${TOTAL}] Training ${MODEL} frac=${FRAC} seed=${SEED}"
        echo "  Output: ${OUTDIR}"
        echo "  Time: $(date)"
        echo "============================================================"

        python3 src/trainer.py \
            --model-id "${MODEL}" \
            --prime "${PRIME}" \
            --n-layers "${N_LAYERS}" \
            --epochs "${EPOCHS}" \
            --seed "${SEED}" \
            --train-frac "${FRAC}" \
            --output-dir "${OUTDIR}" \
            --eval-interval "${EVAL_INTERVAL}" \
            --checkpoint-interval "${CKPT_INTERVAL}" \
            --wd-max 0.30 \
            --wd-ramp-interval 1000 \
            --lr-drop-factor 1.0 \
            --monitor-reps \
            --rep-interval "${REP_INTERVAL}"

        echo ""
        echo "  [${COUNT}/${TOTAL}] ${MODEL} frac=${FRAC} seed=${SEED} complete: $(date)"
        echo ""
    done
done

echo ""
echo "============================================================"
echo "  Train_frac ablation complete: $(date)"
echo "  Results in: results/ablation/"
echo ""
echo "  Generate ablation table:"
echo "    python3 paper/generate_ablation_table.py"
echo "============================================================"
