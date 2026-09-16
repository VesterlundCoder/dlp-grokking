#!/bin/bash
#SBATCH --job-name=dlp_scale
#SBATCH --account=project_465003364
#SBATCH --partition=small-g
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/project_465002952/dlp_grokking/logs/scale_%A_%a.out
#SBATCH --error=/scratch/project_465002952/dlp_grokking/logs/scale_%A_%a.err

# =============================================================================
# DLP Grokking Scaling Study — LUMI M07-M12
#
# Array 0-5: 6 model sizes, each running 3 seeds sequentially
#   0: M07 (1.6M, d=256, H=8)
#   1: M08 (2.5M, d=320, H=10)
#   2: M09 (3.6M, d=384, H=12)
#   3: M10 (5.6M, d=480, H=15)
#   4: M11 (9.0M, d=608, H=19)
#   5: M12 (14.3M, d=768, H=24)
#
# Each task: 2 GCDs with DDP, 48h, 3 seeds sequential
# Problem: L6 p=113, L=2, WD max=0.30, wd_ramp_interval=1000, lr_drop_factor=1.0
# 1M epochs, auto train_frac
#
# Usage:
#   sbatch slurm_scaling_array.sh
#   squeue -u $USER
#   tail -f logs/scale_<JOBID>_<task>.out
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
MODELS=("M07" "M08" "M09" "M10" "M11" "M12")
SEEDS=(42 123 456)
# Train fractions: M07=0.40, M08=0.45, M09=0.50, M10=0.55, M11=0.60, M12=0.65
TRAIN_FRACS=(0.40 0.45 0.50 0.55 0.60 0.65)

MODEL_ID=${MODELS[$SLURM_ARRAY_TASK_ID]}
TRAIN_FRAC=${TRAIN_FRACS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "  DLP Scaling Study — Array task ${SLURM_ARRAY_TASK_ID}"
echo "  Job:      ${SLURM_JOB_ID}"
echo "  Model:    ${MODEL_ID}"
echo "  Seeds:    ${SEEDS[@]}"
echo "  TrainFrac: ${TRAIN_FRAC}"
echo "  Problem:  L6 p=113, L=2"
echo "  GPUs:     2 GCDs (DDP)"
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

# Run 3 seeds sequentially
for SEED in "${SEEDS[@]}"; do
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
        --env "ROCR_VISIBLE_DEVICES=0,1" \
        "${SIF}" \
        torchrun \
            --standalone \
            --nproc_per_node=2 \
            "${WORK}/src/trainer.py" \
            --model-id "${MODEL_ID}" \
            --prime 113 \
            --n-layers 2 \
            --epochs 1000000 \
            --wd-max 0.30 \
            --wd-ramp-interval 1000 \
            --lr-drop-factor 1.0 \
            --seed "${SEED}" \
            --train-frac "${TRAIN_FRAC}" \
            --output-dir "${OUTPUT_DIR}" \
            --eval-interval 10 \
            --checkpoint-interval 1000 \
            --ddp \
            --monitor-reps

    echo "  Finished ${MODEL_ID} seed=${SEED} at $(date)"
done

echo ""
echo "============================================================"
echo "  All seeds for ${MODEL_ID} complete"
echo "  Finished: $(date)"
echo "============================================================"
