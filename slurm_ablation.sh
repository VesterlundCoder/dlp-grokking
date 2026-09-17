#!/bin/bash
#SBATCH --job-name=dlp_ablation
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=2
#SBATCH --time=48:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch/project_465002952/dlp_grokking/logs/ablation_%a.out
#SBATCH --error=/scratch/project_465002952/dlp_grokking/logs/ablation_%a.err

# =============================================================================
# Train_frac ablation on LUMI (M04, p=113)
#
# 18 tasks: 6 train_fracs × 3 seeds
#   frac 0.10, 0.20, 0.30, 0.40, 0.50, 0.60
#   seeds 42, 123, 456
#
# Array index mapping:
#   idx = frac_idx * 3 + seed_idx
#   frac_idx = idx / 3  →  0..5
#   seed_idx = idx % 3  →  0..2
#
# Usage:
#   sbatch slurm_ablation.sh
#   squeue -u $USER
# =============================================================================

set -euo pipefail

SCRATCH=/scratch/project_465002952
WORK=${SCRATCH}/dlp_grokking
SIF="/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif"

FRACS=(0.10 0.20 0.30 0.40 0.50 0.60)
SEEDS=(42 123 456)

FRAC_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

FRAC=${FRACS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

MODEL_ID=M04
PRIME=113
N_LAYERS=2
EPOCHS=100000
OUTPUT_DIR="${WORK}/results/ablation/M04_frac${FRAC}_s${SEED}"

echo "============================================================"
echo "  Ablation task ${SLURM_ARRAY_TASK_ID}: M04 frac=${FRAC} seed=${SEED}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Time: $(date)"
echo "============================================================"

export NCCL_DEBUG=WARN
export MIOPEN_USER_DB_PATH=/tmp/miopen_${USER}_${SLURM_JOB_ID}
export MIOPEN_DISABLE_CACHE=1
export ROCBLAS_LAYER=0
export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1

srun singularity exec \
    --bind "${SCRATCH}" \
    --env "NCCL_SOCKET_IFNAME=hsn" \
    --env "NCCL_IB_DISABLE=1" \
    --env "MIOPEN_DISABLE_CACHE=1" \
    --env "OMP_NUM_THREADS=7" \
    --env "PYTHONUNBUFFERED=1" \
    --env "ROCR_VISIBLE_DEVICES=0,1" \
    "${SIF}" \
    torchrun \
        --standalone \
        --nproc_per_node=2 \
        "${WORK}/src/trainer.py" \
        --model-id "${MODEL_ID}" \
        --prime "${PRIME}" \
        --n-layers "${N_LAYERS}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --train-frac "${FRAC}" \
        --output-dir "${OUTPUT_DIR}" \
        --eval-interval 10 \
        --checkpoint-interval 500 \
        --wd-max 0.30 \
        --wd-ramp-interval 1000 \
        --lr-drop-factor 1.0 \
        --ddp \
        --monitor-reps \
        --rep-interval 500

echo ""
echo "  Task ${SLURM_ARRAY_TASK_ID} complete: $(date)"
