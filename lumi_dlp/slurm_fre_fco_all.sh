#!/bin/bash -l
#SBATCH --job-name=dlp_fre
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
# SLURM Batch Job — Run FRE/FCO DLP scaling experiments (all primes)
# Tests O(1) model size on large primes with CRT-BOTH representation
# Fixed: subset eval (10K examples) for large datasets
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
# Experiments: FRE/FCO + CRT-BOTH scaling ladder
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

# p=2003 (q=2002=26x77, M=2, B=256, ~1.1M examples)
run_experiment p2003_crt_B256 \
    --prime 2003 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 8192

# p=4001 (q=4000=32x125, M=2, B=256, ~6.4M examples)
run_experiment p4001_crt_B256 \
    --prime 4001 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 8192

# p=8209 (q=8208=27x304, M=2, B=256, ~18.9M examples)
run_experiment p8209_crt_B256 \
    --prime 8209 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 8192

# p=16411 (q=16410=30x547, M=2, B=256, ~71.7M examples)
run_experiment p16411_crt_B256 \
    --prime 16411 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 8192

# p=32771 (q=32770=145x226, M=2, B=256, ~6.5M examples)
run_experiment p32771_crt_B256 \
    --prime 32771 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

echo "=== All FRE/FCO experiments complete ==="
