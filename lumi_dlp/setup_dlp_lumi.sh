#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# DLP Grokking Container Setup on LUMI — project_465003364
# ═══════════════════════════════════════════════════════════════════════════
#
# Run ON LUMI:  bash setup_dlp_lumi.sh
#
# This script:
#   1. Pulls a ROCm PyTorch base image from Docker Hub
#   2. Installs Python packages (numpy, pyyaml) into a bind-mountable dir
#   3. Copies host's newer libstdc++ (needed for ROCm 6.x builds)
#   4. Verifies everything works
#
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT=project_465003364
PROJAPPL="/projappl/${PROJECT}"
SCRATCH="/scratch/${PROJECT}"
CONTAINER="${PROJAPPL}/mevo_train.sif"
PIP_PKG="${PROJAPPL}/mevo_pip_packages"
HOST_LIBS="${PROJAPPL}/mevo_host_libs"

# ── Fix OOM: redirect Singularity temp to scratch ──
export SINGULARITY_TMPDIR="${SCRATCH}/singularity_tmp"
export TMPDIR="${SCRATCH}/singularity_tmp"
mkdir -p "${SINGULARITY_TMPDIR}"

echo "═══════════════════════════════════════════════════════════════"
echo " DLP Grokking Container Setup — ${PROJECT}"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Pull base ROCm PyTorch image ──
echo ""
echo "=== Step 1: Pull ROCm PyTorch base image ==="
if [ -f "${CONTAINER}" ]; then
    echo "  Container already exists: ${CONTAINER}"
else
    echo "  Pulling rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0 ..."
    singularity pull "${CONTAINER}" \
        docker://rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
    echo "  Done: ${CONTAINER}"
    rm -rf "${SINGULARITY_TMPDIR}"/*
fi

# ── Step 2: Install Python packages ──
echo ""
echo "=== Step 2: Install Python packages ==="
mkdir -p "${PIP_PKG}"

singularity exec -B "${PROJAPPL}" "${CONTAINER}" \
    pip3 install --no-cache-dir --target="${PIP_PKG}" \
    numpy \
    pyyaml

echo "  Packages installed to: ${PIP_PKG}"

# ── Step 3: Copy host's newer libstdc++ ──
echo ""
echo "=== Step 3: Copy host libstdc++ (GLIBCXX_3.4.32) ==="
mkdir -p "${HOST_LIBS}"
cp /usr/lib64/libstdc++.so.6 "${HOST_LIBS}/"
echo "  Copied to: ${HOST_LIBS}/libstdc++.so.6"

# ── Step 4: Verify installation ──
echo ""
echo "=== Step 4: Verify ==="
singularity exec -B "${PROJAPPL}" "${CONTAINER}" \
    env PYTHONPATH="${PIP_PKG}" \
    python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  ROCm available: {torch.cuda.is_available()}')
import numpy; print(f'  numpy {numpy.__version__}')
import yaml; print(f'  pyyaml OK')
print('  All imports OK')
"

# ── Done ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " DLP Setup Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo " Container:   ${CONTAINER}"
echo " Packages:    ${PIP_PKG}"
echo " Host libs:   ${HOST_LIBS}"
echo ""
echo " To add more packages later:"
echo "   singularity exec -B ${PROJAPPL} ${CONTAINER} \\"
echo "     pip3 install --target=${PIP_PKG} <PACKAGE_NAME>"
echo ""
echo " Environment vars for sbatch scripts:"
echo "   SINGULARITYENV_PYTHONPATH=${PIP_PKG}"
echo "   SINGULARITYENV_LD_LIBRARY_PATH=${HOST_LIBS}:/opt/rocm/lib"
echo "═══════════════════════════════════════════════════════════════"
