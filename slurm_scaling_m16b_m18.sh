#!/bin/bash
#SBATCH --job-name=dlp_scale3
#SBATCH --account=project_465002952
#SBATCH --partition=small-g
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/project_465002952/dlp_grokking/logs/scale3_%A_%a.out
#SBATCH --error=/scratch/project_465002952/dlp_grokking/logs/scale3_%A_%a.err

# =============================================================================
# DLP Grokking Scaling Study — LUMI M16B + M17 + M18 + NaN retries
#
# Array 0-14: 15 tasks total
#   0-2:   M16B (100M, d=2000, H=40, p=1009, frac=0.60, 30k epochs, batch=1024)
#          — Same model as M16 but DOUBLE the dataset (174k vs 87k train)
#   3-5:   M17  (150M, d=2400, H=40, p=2003, frac=0.10, 25k epochs, batch=1024)
#   6-8:   M18  (200M, d=2750, H=50, p=3001, frac=0.075, 15k epochs, batch=1024)
#   9-10:  M13 retry s42, s456 (NaN fix: bf16 AMP)
#   11:    M14 retry s42 (NaN fix: bf16 AMP)
#   12:    M15 retry s456 (NaN fix: bf16 AMP)
#   13-14: M16 retry s42, s456 (NaN fix: bf16 AMP)
#
# Key fix: switched from fp16 to bf16 AMP (MI250X native, no overflow)
# Each task: 4 GCDs with DDP, 48h, 1 seed
# =============================================================================

set -euo pipefail

SCRATCH="/scratch/project_465002952/dlp_grokking"
WORK="${SCRATCH}"
RESULTS_DIR="${SCRATCH}/results/scaling_v2"
SIF="/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif"

mkdir -p "${RESULTS_DIR}/logs"

# =============================================================================
# Task configuration
# =============================================================================
# M16B: 0-2, M17: 3-5, M18: 6-8, M13 retry: 9-10, M14 retry: 11, M15 retry: 12, M16 retry: 13-14
MODELS=("M16" "M16" "M16" "M17" "M17" "M17" "M18" "M18" "M18" "M13" "M13" "M14" "M15" "M16" "M16")
PRIMES=(1009 1009 1009 2003 2003 2003 3001 3001 3001 251 251 251 503 1009 1009)
EPOCHS=(30000 30000 30000 25000 25000 25000 15000 15000 15000 2000000 2000000 1000000 200000 60000 60000)
TRAIN_FRACS=(0.60 0.60 0.60 0.10 0.10 0.10 0.075 0.075 0.075 0.40 0.40 0.60 0.30 0.30 0.30)
BATCH_SIZES=("1024" "1024" "1024" "1024" "1024" "1024" "1024" "1024" "1024" "" "" "" "1024" "1024" "1024")
SEEDS=(42 123 456 42 123 456 42 123 456 42 456 42 456 42 456)
LABELS=("M16B" "M16B" "M16B" "M17" "M17" "M17" "M18" "M18" "M18" "M13" "M13" "M14" "M15" "M16" "M16")

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_ID=${MODELS[$TASK_ID]}
PRIME=${PRIMES[$TASK_ID]}
N_EPOCHS=${EPOCHS[$TASK_ID]}
TRAIN_FRAC=${TRAIN_FRACS[$TASK_ID]}
BATCH_SIZE=${BATCH_SIZES[$TASK_ID]}
SEED=${SEEDS[$TASK_ID]}
LABEL=${LABELS[$TASK_ID]}

# Build batch argument
BATCH_ARG=""
if [ -n "${BATCH_SIZE}" ]; then
    BATCH_ARG="--batch-size ${BATCH_SIZE}"
fi

# Output directory uses LABEL to distinguish M16B from M16
OUTPUT_DIR=${RESULTS_DIR}/${LABEL}_s${SEED}

echo ""
echo "============================================================"
echo "  DLP Scaling Study — LUMI"
echo "  Task:    ${TASK_ID} / 14"
echo "  Model:   ${MODEL_ID} (${LABEL})"
echo "  Prime:   ${PRIME}"
echo "  Epochs:  ${N_EPOCHS}"
echo "  TrainFrac: ${TRAIN_FRAC}"
echo "  Batch:   ${BATCH_SIZE:-full-batch}"
echo "  Seed:    ${SEED}"
echo "  AMP:     bf16 (fixed from fp16)"
echo "  Time:    $(date)"
echo "============================================================"

# Remove old results to force re-run
if [ -f "${OUTPUT_DIR}/summary.json" ]; then
    echo "  [PURGE] Removing old results for re-run"
    rm -rf "${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"

srun singularity exec \
    --bind "${SCRATCH}" \
    --env "NCCL_SOCKET_IFNAME=hsn" \
    --env "NCCL_IB_DISABLE=1" \
    --env "MIOPEN_DISABLE_CACHE=1" \
    --env "OMP_NUM_THREADS=7" \
    --env "PYTHONUNBUFFERED=1" \
    --env "ROCR_VISIBLE_DEVICES=0,1,2,3" \
    "${SIF}" \
    torchrun \
        --standalone \
        --nproc_per_node=4 \
        "${WORK}/src/trainer.py" \
        --model-id "${MODEL_ID}" \
        --prime "${PRIME}" \
        --n-layers 2 \
        --epochs "${N_EPOCHS}" \
        --wd-max 0.30 \
        --wd-ramp-interval 1000 \
        --lr-drop-factor 1.0 \
        --seed "${SEED}" \
        --train-frac "${TRAIN_FRAC}" \
        ${BATCH_ARG} \
        --output-dir "${OUTPUT_DIR}" \
        --eval-interval 10 \
        --checkpoint-interval 1000 \
        --ddp \
        --monitor-reps

echo ""
echo "============================================================"
echo "  ${LABEL} seed=${SEED} complete"
echo "  Finished: $(date)"
echo "============================================================"
