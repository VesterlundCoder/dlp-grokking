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
# SLURM Batch Job — Run FRE/FCO DLP scaling experiments
# Tests O(1) model size on large primes with CRT-BOTH representation
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
# All use the same 464k-526k param model (O(1) in p)
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
        "$@" \
      > "${OUTPUT_DIR}/logs/${name}.log" 2>&1
    echo "=== Done ${name} ==="
}

# p=1201 (q=1200=25x48, M=2, B=256, ~1.4M examples)
run_experiment p1201_crt_B256 \
    --prime 1201 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 8192

# p=2003 (q=2002=26x77, M=2, B=256, ~3.9M examples)
run_experiment p2003_crt_B256 \
    --prime 2003 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 8192

# p=4001 (q=4000=32x125, M=2, B=256, ~16M examples)
run_experiment p4001_crt_B256 \
    --prime 4001 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 8192

# p=8209 (q=8208=27x304, M=2, B=256, ~67M examples)
run_experiment p8209_crt_B256 \
    --prime 8209 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 8192

echo "=== All FRE/FCO experiments complete ==="
