#!/bin/bash -l
#SBATCH --job-name=dlp_fre2
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
# SLURM Batch Job — Run FRE/FCO DLP scaling experiments (Phase 2)
# Tests O(1) model size on very large primes with CRT-BOTH representation
# ============================================================================

PROJECT=project_465003364
CONTAINER="/projappl/${PROJECT}/mevo_train.sif"
PIP_PKG="/projappl/${PROJECT}/mevo_pip_packages"
HOST_LIBS="/projappl/${PROJECT}/mevo_host_libs"

SCRIPT_DIR="/scratch/${PROJECT}/dlp_grokking"
OUTPUT_DIR="/scratch/${PROJECT}/dlp_grokking/results/fre_fco"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID}"

# Container environment
export SINGULARITYENV_PYTHONPATH="${PIP_PKG}"
export SINGULARITYENV_LD_LIBRARY_PATH="${HOST_LIBS}:/opt/rocm/lib"
export SINGULARITYENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

mkdir -p "${OUTPUT_DIR}/logs"

# ============================================================================
# Experiments: FRE/FCO + CRT-BOTH scaling ladder (Phase 2)
# All use the same ~530k param model (O(1) in p)
# Using early-stop-patience=20 for faster turnaround
# ============================================================================

run_experiment() {
    local name=$1
    shift
    echo "=== Running ${name} ==="
    singularity exec \
      -B "/projappl/${PROJECT}" \
      -B "/scratch/${PROJECT}" \
      "${CONTAINER}" \
      python3 "${SCRIPT_DIR}/experiments/fre_fco/run_fre_fco.py" \
        --output-dir "${OUTPUT_DIR}" \
        --exp-name "${name}" \
        --early-stop-patience 20 \
        "$@" \
      > "${OUTPUT_DIR}/logs/${name}.log" 2>&1
    echo "=== Done ${name} ==="
}

# p=16411 (q=16410=30x547, M=2, B=256, ~3.3M examples)
run_experiment p16411_crt_B256 \
    --prime 16411 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 8192

# p=32771 (q=32770=145x226, M=2, B=256, ~6.5M examples)
run_experiment p32771_crt_B256 \
    --prime 32771 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

echo "=== All Phase 2 FRE/FCO experiments complete ==="
