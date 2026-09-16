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
# SLURM Batch Job — Extended FRE/FCO DLP scaling (primes above 32771)
# Tests O(1) model size on very large primes with CRT-BOTH representation
# 10 new primes: 39953, 50021, 65539, 79999, 99991, 130003, 160001, 199999, 249973, 300017
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
# Phase 2: Extended scaling ladder (primes 12-21)
# All use the same ~530k-594k param model (O(1) in p)
# M=2 for primes up to 65536, M=3 for primes above 65536
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

# --- Phase 2a: M=2 primes (p < 65536) ---

# p=39953 (q=39952=176x227, M=2, B=256, ~1.6B examples)
run_experiment p39953_crt_B256 \
    --prime 39953 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

# p=50021 (q=50020=205x244, M=2, B=256, ~2.5B examples)
run_experiment p50021_crt_B256 \
    --prime 50021 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

# --- Phase 2b: M=3 primes (p > 65536) ---
# M=3 needed because 256^2=65536 is insufficient for values up to q-1

# p=65539 (q=65538=198x331, M=3, B=256, ~4.3B examples)
run_experiment p65539_crt_B256 \
    --prime 65539 --crt --base 256 --M 3 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

# p=79999 (q=79998=201x398, M=3, B=256, ~6.4B examples)
run_experiment p79999_crt_B256 \
    --prime 79999 --crt --base 256 --M 3 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

# p=99991 (q=99990=202x495, M=3, B=256, ~10B examples)
run_experiment p99991_crt_B256 \
    --prime 99991 --crt --base 256 --M 3 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

# p=130003 (q=130002=282x461, M=3, B=256, ~16.9B examples)
run_experiment p130003_crt_B256 \
    --prime 130003 --crt --base 256 --M 3 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

# p=160001 (q=160000=256x625, M=3, B=256, ~25.6B examples)
run_experiment p160001_crt_B256 \
    --prime 160001 --crt --base 256 --M 3 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

# p=199999 (q=199998=369x542, M=3, B=256, ~40B examples)
run_experiment p199999_crt_B256 \
    --prime 199999 --crt --base 256 --M 3 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

# p=249973 (q=249972=444x563, M=3, B=256, ~62.5B examples)
run_experiment p249973_crt_B256 \
    --prime 249973 --crt --base 256 --M 3 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

# p=300017 (q=300016=272x1103, M=3, B=256, ~90B examples)
run_experiment p300017_crt_B256 \
    --prime 300017 --crt --base 256 --M 3 \
    --d-model 128 --epochs 500000 --eval-interval 100 \
    --train-frac 0.30 --batch-size 4096

echo "=== All extended FRE/FCO experiments complete ==="
