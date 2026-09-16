#!/bin/bash -l
#SBATCH --job-name=dlp_grok
#SBATCH --account=project_465003364
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=24:00:00
#SBATCH --mem=60G
#SBATCH --array=0-11

set -euo pipefail

# ============================================================================
# SLURM Array Job for DLP Grokking Study
# Each array task runs one job from a YAML manifest on a single GPU.
# ============================================================================

PROJECT=project_465003364
CONTAINER="/projappl/${PROJECT}/mevo_train.sif"
PIP_PKG="/projappl/${PROJECT}/mevo_pip_packages"
HOST_LIBS="/projappl/${PROJECT}/mevo_host_libs"

# Job configuration — override these at submission time
MANIFEST="${MANIFEST:-jobs/capacity_scaling.yaml}"
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

# Run the job
singularity exec \
  -B "/projappl/${PROJECT}" \
  -B "/scratch/${PROJECT}" \
  "${CONTAINER}" \
  python3 "${SCRIPT_DIR}/lumi_dlp_trainer.py" \
    --manifest "${SCRIPT_DIR}/${MANIFEST}" \
    --job-index "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
  > "${OUTPUT_DIR}/logs/array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
