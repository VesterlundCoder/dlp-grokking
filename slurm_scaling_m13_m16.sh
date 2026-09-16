#!/bin/bash
#SBATCH --job-name=dlp_scale2
#SBATCH --account=project_465003364
#SBATCH --partition=small-g
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/project_465002952/dlp_grokking/logs/scale2_%A_%a.out
#SBATCH --error=/scratch/project_465002952/dlp_grokking/logs/scale2_%A_%a.err

# =============================================================================
# DLP Grokking Scaling Study — LUMI M13-M16 (Large Models)
#
# Array 0-11: 4 models × 3 seeds = 12 tasks (1 seed per task)
#   0-2:  M13 (25M,  d=1024, H=32, p=251,  2M epochs, full-batch)
#   3-5:  M14 (40M,  d=1280, H=40, p=251,  1M epochs, full-batch)
#   6-8:  M15 (64M,  d=1632, H=48, p=503,  200k epochs, batch=1024)
#   9-11: M16 (100M, d=2000, H=40, p=1009, 60k epochs, batch=1024)
#
# Epoch counts adjusted for 48h walltime (1 seed per task).
# With mini-batching, gradient steps per epoch = ceil(train_size/(batch*4)):
#   M13: 2M×1   = 2.0M grad steps (2.0x M12)
#   M14: 1M×1   = 1.0M grad steps (1.0x M12)
#   M15: 200k×9 = 1.8M grad steps (1.8x M12)
#   M16: 60k×21 = 1.3M grad steps (1.3x M12)
#
# Each task: 4 GCDs with DDP, 48h, 1 seed
# Problem: L6, L=2, WD max=0.30, wd_ramp_interval=1000, lr_drop_factor=1.0
# Larger primes for larger datasets to prevent overfitting
#
# Usage:
#   sbatch slurm_scaling_m13_m16.sh
#   squeue -u $USER
#   tail -f logs/scale2_<JOBID>_<task>.out
# =============================================================================

set -euo pipefail

SCRATCH=/scratch/project_465002952
WORK=${SCRATCH}/dlp_grokking
LOG_DIR=${WORK}/logs
RESULTS_DIR=${WORK}/results/scaling

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Container
SIF="/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif"
if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container not found: ${SIF}"
    ls /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2*.sif 2>/dev/null || echo "  (none matching)"
    exit 1
fi

# Per-task configuration
# Model | d_model | n_heads | Prime | Params  | Space  | train_frac | train_size | Epochs | Batch
# M13   | 1024    | 32      | 251   | 25M     | 25k    | 0.40       | 10k        | 2M     | full
# M14   | 1280    | 40      | 251   | 40M     | 25k    | 0.60       | 15k        | 1M     | full
# M15   | 1632    | 48      | 503   | 64M     | 125k   | 0.30       | 37.6k      | 200k   | 1024
# M16   | 2000    | 40      | 1009  | 100M    | 290k   | 0.30       | 87k        | 60k    | 1024

MODELS=("M13" "M13" "M13" "M14" "M14" "M14" "M15" "M15" "M15" "M16" "M16" "M16")
PRIMES=(251 251 251 251 251 251 503 503 503 1009 1009 1009)
EPOCHS=(2000000 2000000 2000000 1000000 1000000 1000000 200000 200000 200000 60000 60000 60000)
TRAIN_FRACS=(0.40 0.40 0.40 0.60 0.60 0.60 0.30 0.30 0.30 0.30 0.30 0.30)
BATCH_SIZES=("" "" "" "" "" "" "1024" "1024" "1024" "1024" "1024" "1024")
SEEDS=(42 123 456 42 123 456 42 123 456 42 123 456)

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_ID=${MODELS[$TASK_ID]}
PRIME=${PRIMES[$TASK_ID]}
N_EPOCHS=${EPOCHS[$TASK_ID]}
TRAIN_FRAC=${TRAIN_FRACS[$TASK_ID]}
BATCH_SIZE=${BATCH_SIZES[$TASK_ID]}
SEED=${SEEDS[$TASK_ID]}

echo "============================================================"
echo "  DLP Scaling Study (Large) — Array task ${TASK_ID}"
echo "  Job:      ${SLURM_JOB_ID}"
echo "  Model:    ${MODEL_ID}"
echo "  Prime:    ${PRIME}"
echo "  Epochs:   ${N_EPOCHS}"
echo "  Seed:     ${SEED}"
echo "  TrainFrac: ${TRAIN_FRAC}"
echo "  BatchSz:  ${BATCH_SIZE:-full}"
echo "  Problem:  L6 p=${PRIME}, L=2"
echo "  GPUs:     4 GCDs (DDP)"
echo "  Start:    $(date)"
echo "============================================================"

# Environment for ROCm + NCCL on LUMI
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export MIOPEN_USER_DB_PATH=/tmp/miopen_${USER}_${SLURM_JOB_ID}
export MIOPEN_DISABLE_CACHE=1
export ROCBLAS_LAYER=0
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1

# Build torchrun args
BATCH_ARG=""
if [ -n "${BATCH_SIZE}" ]; then
    BATCH_ARG="--batch-size ${BATCH_SIZE}"
fi

OUTPUT_DIR=${RESULTS_DIR}/${MODEL_ID}_s${SEED}

echo ""
echo "------------------------------------------------------------"
echo "  Starting ${MODEL_ID} seed=${SEED}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Time:   $(date)"
echo "------------------------------------------------------------"

# Remove old results to force re-run with new params
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
echo "  ${MODEL_ID} seed=${SEED} complete"
echo "  Finished: $(date)"
echo "============================================================"
