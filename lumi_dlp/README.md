# LUMI DLP Grokking — Representation-Relative Grokking Study

Standalone training package for running DLP grokking experiments on LUMI-G (AMD MI250X).

## Contents

```
lumi_dlp/
├── lumi_dlp_trainer.py      # Standalone trainer (no external deps except PyTorch + PyYAML)
├── slurm_train_array.sh     # SLURM array job (1 GPU per task)
├── slurm_train_batch.sh     # SLURM batch job (all jobs sequential on 1 GPU)
├── pack_lumi.sh             # Transfer script: scp package to LUMI
├── jobs/
│   ├── capacity_scaling.yaml  # M01-M12 on p=113 (12 jobs)
│   ├── crt_variants.yaml      # RAW, CRT-BOTH, CRT-16, CRT-7, RAW+CRT, SCRAMBLED (10 jobs, 3 seeds)
│   ├── prime_order.yaml       # P113 subgroup DLP, IID + OOD (6 jobs)
│   └── large_primes.yaml      # M13-M18 on larger primes (6 jobs)
└── README.md
```

## Experiment Types

### 1. Capacity Scaling (`capacity_scaling.yaml`)
Models M01-M12 (56-768 d_model) on DLP in F_113*. Tests whether grokking onset scales with model capacity.

### 2. CRT Representation Variants (`crt_variants.yaml`)
Six representation variants on p=113, M04 architecture:
- **RAW**: Standard DLP (g, h) → x
- **CRT-BOTH**: Explicit CRT components (g16, g7, h16, h7) → x
- **CRT-16**: Only order-16 components → x
- **CRT-7**: Only order-7 components → x (information-theoretic ceiling ~6.25%)
- **RAW+CRT**: Raw inputs + CRT components → x
- **SCRAMBLED**: CRT components with scrambled mapping (control)

Multi-seed (42, 123, 7) for RAW and CRT-BOTH to measure grokking speedup statistics.

### 3. Prime-Order Subgroup DLP (`prime_order.yaml`)
DLP in the order-113 subgroup of F_227*. Tests grokking without smooth factorization of p-1.
- IID and OOD (disjoint generators) splits
- Multiple model sizes (M04, M06, M08)

### 4. Large-Prime Scaling (`large_primes.yaml`)
Models M13-M18 on primes 257-8209. Tests scaling to larger problem spaces.

## Deployment

### One-time setup

```bash
# From local machine — transfer files to LUMI
cd /Users/davidsvensson/Desktop/dlp_grokking/lumi_dlp
bash pack_lumi.sh

# SSH to LUMI and build the container (one-time, ~10 min)
ssh lumi
cd /scratch/project_465003364/dlp_grokking
bash setup_dlp_lumi.sh
```

This transfers all files to `/scratch/project_465003364/dlp_grokking/` on LUMI
and builds the ROCm PyTorch container at `/projappl/project_465003364/mevo_train.sif`.

### Submitting jobs

SSH to LUMI, then:

```bash
cd /scratch/project_465003364/dlp_grokking

# Capacity scaling: 12 jobs, each on 1 GPU
MANIFEST=jobs/capacity_scaling.yaml sbatch --array=0-11 slurm_train_array.sh

# CRT variants: 10 jobs, sequential on 1 GPU (~72h max)
MANIFEST=jobs/crt_variants.yaml sbatch slurm_train_batch.sh

# Prime-order: 6 jobs, sequential on 1 GPU
MANIFEST=jobs/prime_order.yaml sbatch slurm_train_batch.sh

# Large primes: 6 jobs, each on 1 GPU
MANIFEST=jobs/large_primes.yaml sbatch --array=0-5 slurm_train_array.sh
```

### Monitoring

```bash
# Check queue
squeue -u $USER

# Tail logs
tail -f /scratch/project_465003364/dlp_grokking/results/logs/*.log

# Check results
ls /scratch/project_465003364/dlp_grokking/results/
cat /scratch/project_465003364/dlp_grokking/results/<job_name>/summary.json
```

### Pulling results back

```bash
# From local machine
scp -r lumi:/scratch/project_465003364/dlp_grokking/results/ \
  /Users/davidsvensson/Desktop/dlp_grokking/lumi_results/
```

## Architecture

### Model
GrokkingTransformer: token + positional embedding → N transformer encoder layers (pre-norm, GELU) → unembed to vocab logits.

### Training
- Full-batch training (all training data per step)
- AdamW with weight-decay auto-scaling (WD ramps after memorization)
- BF16 autocast on MI250X
- Phase tracking: T_mem (memorization), T_10/T_50/T_90/T_99 (grokking thresholds)
- Early stopping after sustained high test accuracy

### Output Per Job
```
results/<job_name>/
├── config.json          # Full configuration
├── metrics.jsonl        # Per-eval-interval metrics (train/test acc, loss, WD, elapsed)
├── summary.json         # Final summary (best_test_acc, mem_epoch, total_epochs)
└── checkpoints/
    ├── epoch_XXXXXX.pt  # Periodic checkpoints
    └── final.pt         # Final model
```

## Dependencies

- PyTorch 2.3.0+ (ROCm 6.2 on LUMI)
- PyYAML
- NumPy

All available in the MEVO Singularity container at `/projappl/project_465003364/mevo_train.sif`.

## Key Results from Local Experiments

- **Phase A/B**: M04 and M05 grok DLP in F_113*, with spectral emergence preceding accuracy rise
- **Phase C**: CRT-BOTH groks ~30-250x faster than RAW, confirming representation-relative grokking
- **Phase D**: Prime-order DLP (P113) groks without smooth factorization
- **Phase F**: Double dissociation confirmed — period-7 and period-16 frequencies are causally linked to mod-7/mod-16 accuracy
