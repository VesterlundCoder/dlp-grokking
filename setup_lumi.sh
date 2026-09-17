#!/bin/bash
# =============================================================================
# One-time LUMI setup for DLP Grokking project
# Run on LUMI login node: bash setup_lumi.sh
# =============================================================================

set -euo pipefail

PROJECT=project_465002952
SCRATCH=/scratch/${PROJECT}
WORK=${SCRATCH}/dlp_grokking

echo "=== LUMI DLP Grokking Setup ==="
echo "Project: ${PROJECT}"
echo "Work dir: ${WORK}"
echo ""

# Create directories
mkdir -p "${WORK}/logs" "${WORK}/results"
echo "[1/4] Directories created: ${WORK}/{logs,results}"

# Verify container
SIF="/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif"
if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container not found: ${SIF}"
    echo "  Available containers:"
    ls /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2*.sif 2>/dev/null || echo "  (none matching)"
    exit 1
fi
echo "[2/4] Container OK: ${SIF}"

# Verify Python + PyTorch + NumPy inside container
echo "[3/4] Verifying Python environment inside container..."
singularity exec --bind "${SCRATCH}" "${SIF}" python3 -c "
import torch
import numpy as np
print(f'  Python:   {__import__(\"sys\").version}')
print(f'  PyTorch:  {torch.__version__}')
print(f'  NumPy:    {np.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  GPU count: {torch.cuda.device_count()}')
"

# Verify training script exists
if [ ! -f "${WORK}/lumi_cyclic_dlp_trainer.py" ]; then
    echo "WARNING: Training script not found at ${WORK}/lumi_cyclic_dlp_trainer.py"
    echo "  Run: scp lumi_cyclic_dlp_trainer.py ${WORK}/"
else
    echo "[4/4] Training script found: ${WORK}/lumi_cyclic_dlp_trainer.py"
fi

echo ""
echo "=== Setup complete ==="
echo "Next: sbatch ${WORK}/slurm_dlp_array.sh"
