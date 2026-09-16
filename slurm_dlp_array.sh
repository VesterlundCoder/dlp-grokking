#!/bin/bash
#SBATCH --job-name=dlp_grok
#SBATCH --account=project_465003364
#SBATCH --partition=small-g
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/project_465002952/dlp_grokking/logs/dlp_%A_%a.out
#SBATCH --error=/scratch/project_465002952/dlp_grokking/logs/dlp_%A_%a.err

# =============================================================================
# LUMI Cyclic DLP Solver — 4-problem array job
#
# Each array task runs one DLP type on 2 GCDs with DDP.
# Array 0-3: L6, L7, MDLP, PDLP
#
# Usage:
#   sbatch slurm_dlp_array.sh
#   squeue -u $USER
#   tail -f logs/dlp_<JOBID>_<task>.out
# =============================================================================

set -euo pipefail

SCRATCH=/scratch/project_465002952
WORK=${SCRATCH}/dlp_grokking
LOG_DIR=${WORK}/logs
RESULTS_DIR=${WORK}/results

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Container
SIF="/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif"
if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container not found: ${SIF}"
    ls /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2*.sif 2>/dev/null || echo "  (none matching)"
    exit 1
fi

# Per-problem configuration
PROBLEMS=("L6" "L7" "MDLP" "PDLP")
PROBLEM=${PROBLEMS[$SLURM_ARRAY_TASK_ID]}
PRIME=10007
D_MODEL=512
N_LAYERS=8
N_HEADS=16
EPOCHS=200000

# Per-problem train sizes (adjusted for solution space)
case $PROBLEM in
    L6)   TRAIN_SIZE=150000 ;;
    L7)   TRAIN_SIZE=3000 ;;
    MDLP) TRAIN_SIZE=150000 ;;
    PDLP) TRAIN_SIZE=90 ;;
    *)    TRAIN_SIZE=150000 ;;
esac

OUTPUT_DIR=${RESULTS_DIR}/${PROBLEM}_p${PRIME}

echo "============================================================"
echo "  LUMI Cyclic DLP Solver — Array task ${SLURM_ARRAY_TASK_ID}"
echo "  Job:      ${SLURM_JOB_ID}"
echo "  Problem:  ${PROBLEM} (prime=${PRIME})"
echo "  Model:    d=${D_MODEL}, L=${N_LAYERS}, H=${N_HEADS}"
echo "  Train:    ${TRAIN_SIZE} examples"
echo "  Epochs:   ${EPOCHS}"
echo "  GPUs:     2 GCDs (DDP)"
echo "  Output:   ${OUTPUT_DIR}"
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

# Launch training with torchrun DDP (2 GCDs)
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
        "${WORK}/lumi_cyclic_dlp_trainer.py" \
        --problem "${PROBLEM}" \
        --prime "${PRIME}" \
        --d-model "${D_MODEL}" \
        --n-layers "${N_LAYERS}" \
        --n-heads "${N_HEADS}" \
        --epochs "${EPOCHS}" \
        --train-size "${TRAIN_SIZE}" \
        --wd-start 0.05 \
        --wd-max 0.30 \
        --wd-step 0.05 \
        --wd-acc-threshold 0.05 \
        --ddp \
        --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?
echo ""
echo "============================================================"
echo "  Job finished: $(date)  |  Exit code: ${EXIT_CODE}"
echo "============================================================"
exit ${EXIT_CODE}
