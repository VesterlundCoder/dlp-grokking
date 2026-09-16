#!/bin/bash -l
#SBATCH --job-name=dlp_fb
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
# SLURM Batch Job — Fallback experiments for primes where base config fails
# Strategy: 2x model size (d_model=256) and/or 2x dataset (train_frac=0.60)
# Run ONLY for primes that failed with the base config (d_model=128, tf=0.30)
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

export SINGULARITYENV_PYTHONPATH="${PIP_PKG}"
export SINGULARITYENV_LD_LIBRARY_PATH="${HOST_LIBS}:/opt/rocm/lib"
export SINGULARITYENV_PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

mkdir -p "${OUTPUT_DIR}/logs"

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

# ============================================================================
# Fallback Strategy A: 2x model size (d_model=256, ~1.5M params)
# Use for primes where d_model=128 failed to grok
# ============================================================================

# Template — uncomment and fill in the prime that failed
# run_experiment p{PRIME}_crt_B256_d256 \
#     --prime {PRIME} --crt --base 256 --M {M} \
#     --d-model 256 --epochs 500000 --eval-interval 100 \
#     --train-frac 0.30 --batch-size 4096

# ============================================================================
# Fallback Strategy B: 2x dataset (train_frac=0.60)
# Use for primes where memorization succeeds but grokking fails
# ============================================================================

# Template — uncomment and fill in the prime that failed
# run_experiment p{PRIME}_crt_B256_tf60 \
#     --prime {PRIME} --crt --base 256 --M {M} \
#     --d-model 128 --epochs 500000 --eval-interval 100 \
#     --train-frac 0.60 --batch-size 4096

# ============================================================================
# Fallback Strategy C: 2x model + 2x data combined
# ============================================================================

# Template — uncomment and fill in the prime that failed
# run_experiment p{PRIME}_crt_B256_d256_tf60 \
#     --prime {PRIME} --crt --base 256 --M {M} \
#     --d-model 256 --epochs 500000 --eval-interval 100 \
#     --train-frac 0.60 --batch-size 4096

# ============================================================================
# Fallback Strategy D: RichFCO (bilinear digit interactions)
# ============================================================================

# Template — uncomment and fill in the prime that failed
# run_experiment p{PRIME}_crt_richfco_B256 \
#     --prime {PRIME} --crt --base 256 --M {M} \
#     --d-model 128 --epochs 500000 --eval-interval 100 \
#     --train-frac 0.30 --batch-size 4096 --rich-fco

echo "=== Fallback experiments complete ==="
echo "NOTE: This script contains templates. Uncomment the experiments for primes that failed."
echo "Check results in ${OUTPUT_DIR}/ for primes that did not reach 100% test accuracy."
