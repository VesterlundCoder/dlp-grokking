#!/bin/bash -l
#SBATCH --job-name=dlp_batch
#SBATCH --account=project_465003364
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=72:00:00
#SBATCH --mem=60G

set -euo pipefail

# ============================================================================
# SLURM Batch Job — Run ALL jobs in a manifest sequentially on one GPU
# Use for manifests with many small jobs (e.g., CRT variants with 10 jobs)
# ============================================================================

PROJECT=project_465003364
CONTAINER="/projappl/${PROJECT}/mevo_train.sif"
PIP_PKG="/projappl/${PROJECT}/mevo_pip_packages"
HOST_LIBS="/projappl/${PROJECT}/mevo_host_libs"

MANIFEST="${MANIFEST:-jobs/crt_variants.yaml}"
SCRIPT_DIR="${SCRIPT_DIR:-/scratch/${PROJECT}/dlp_grokking}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/${PROJECT}/dlp_grokking/results}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID}"

# Container environment
export SINGULARITYENV_PYTHONPATH="${PIP_PKG}"
export SINGULARITYENV_LD_LIBRARY_PATH="${HOST_LIBS}:/opt/rocm/lib"
export SINGULARITYENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

mkdir -p "${OUTPUT_DIR}/logs"

singularity exec \
  -B "/projappl/${PROJECT}" \
  -B "/scratch/${PROJECT}" \
  "${CONTAINER}" \
  python3 "${SCRIPT_DIR}/lumi_dlp_trainer.py" \
    --manifest "${SCRIPT_DIR}/${MANIFEST}" \
    --all \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
  > "${OUTPUT_DIR}/logs/batch_${SLURM_JOB_ID}.log" 2>&1
