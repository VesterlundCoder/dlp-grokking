#!/bin/bash
# Pack the LUMI DLP grokking package and transfer to LUMI
set -euo pipefail

PROJECT=project_465003364
LUMI_HOST="lumi"
LUMI_DIR="/scratch/${PROJECT}/dlp_grokking"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Packing LUMI DLP Grokking Package ==="
echo "Local: ${LOCAL_DIR}"
echo "Remote: ${LUMI_HOST}:${LUMI_DIR}"
echo ""

# Create remote directory
ssh "${LUMI_HOST}" "mkdir -p ${LUMI_DIR}/jobs"

# Transfer files
echo "Transferring trainer..."
scp "${LOCAL_DIR}/lumi_dlp_trainer.py" "${LUMI_HOST}:${LUMI_DIR}/"

echo "Transferring setup script..."
scp "${LOCAL_DIR}/setup_dlp_lumi.sh" "${LUMI_HOST}:${LUMI_DIR}/"

echo "Transferring SLURM scripts..."
scp "${LOCAL_DIR}/slurm_train_array.sh" "${LUMI_HOST}:${LUMI_DIR}/"
scp "${LOCAL_DIR}/slurm_train_batch.sh" "${LUMI_HOST}:${LUMI_DIR}/"

echo "Transferring job manifests..."
scp "${LOCAL_DIR}/jobs/"*.yaml "${LUMI_HOST}:${LUMI_DIR}/jobs/"

echo ""
echo "=== Transfer complete ==="
echo "Files on LUMI: ${LUMI_DIR}/"
echo ""
echo "To submit array jobs:"
echo "  ssh ${LUMI_HOST}"
echo "  cd ${LUMI_DIR}"
echo ""
echo "  # Capacity scaling (12 jobs, 1 GPU each)"
echo "  MANIFEST=jobs/capacity_scaling.yaml sbatch --array=0-11 slurm_train_array.sh"
echo ""
echo "  # CRT variants (10 jobs, sequential on 1 GPU)"
echo "  MANIFEST=jobs/crt_variants.yaml sbatch slurm_train_batch.sh"
echo ""
echo "  # Prime order (6 jobs, sequential)"
echo "  MANIFEST=jobs/prime_order.yaml sbatch slurm_train_batch.sh"
echo ""
echo "  # Large primes (6 jobs, 1 GPU each)"
echo "  MANIFEST=jobs/large_primes.yaml sbatch --array=0-5 slurm_train_array.sh"
