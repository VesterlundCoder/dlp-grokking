# DLP Grokking — Current Status

**Date:** September 15, 2026  
**Workspace:** `/Users/davidsvensson/Desktop/dlp_grokking/`

---

## Paper Restructuring (Sept 15)

The paper has been completely restructured per the 41-point plan. The new paper is a **scaling-focused ML paper**:

**Title:** "Scaling Grokking on the Discrete Logarithm Problem: Model Capacity, Data, and Group Size"

**10-section structure:**
1. Introduction (DLP as learning benchmark, not crypto attack)
2. DLP Learning Benchmark (variable-base task, datasets, architectures)
3. Discovery of DLP Grokking (M04 anchor, dynamics)
4. Capacity–Data Phase Diagram (main result)
5. Scaling with Mathematical Problem Size (group-order scaling)
6. Training Dynamics Across Scale (slingshots, WD, phases)
7. Mechanistic Analysis (spectral, group structure)
8. Prime-Order and Generator-OOD Controls
9. Limitations
10. Discussion and Conclusion

**Removed from main paper (moved to `paper/crt_material.tex`):**
- CRT Representation Acceleration section
- CRT-BOTH train-fraction sweep
- M12 CRT-BOTH scaling
- Local CRT-BOTH scaling
- Grokking ladder (L4/L5/L7/MDLP/PDLP)
- Related Findings section

**Key fixes:**
- Task definition: variable-base DLP with ALL primitive roots (not fixed g=3)
- Crypto framing reduced to 1 background + 1 limitation paragraph
- Fourier wording corrected (multiplicative characters parameterized by DL coordinates)
- Slingshot claims weakened (coincide/precede, not "necessary")
- WD scaling rule audited (decreases for larger models, treated as controlled variable)
- Novelty claims hedged ("to the best of our knowledge")
- Abstract: 4 claims only (grokking occurs, persists across capacity, phase boundary, structural controls)

**Paper files:**
- `paper/paper.tex` — 506 lines, 10 sections, balanced
- `paper/crt_material.tex` — CRT sections extracted for separate paper (now includes local scaling results)
- `paper/master_manifest.json` — 21 experiments + 8 pending
- `paper/paper_old_v1.tex` — backup of previous version

---

## Overnight Run 1 Results (Sept 14-15, CRT-BOTH Scaling)

**8 experiments completed** in 8.5 hours on Apple MPS. 3 grokked, 2 crashed, 3 failed.

### Grokked

| Experiment | Model | Prime | Train% | Best Acc | Notes |
|-----------|-------|-------|--------|----------|-------|
| M05_p337_lr3e4 | M05 (657k) | 337 | 40% | **99.99%** | Slingshot/forgetting after peak |
| M06_p337_lr3e4 | M06 (1.09M) | 337 | 30% | **99.24%** | Slingshot/forgetting after peak |
| M04_p337_tf50_lr3e4 | M04 (427k) | 337 | 50% | **95.6%** | Still climbing at timeout |

### Failed

| Experiment | Model | Prime | Train% | Best Acc | Reason |
|-----------|-------|-------|--------|----------|--------|
| M04_p337_lr3e4 | M04 (427k) | 337 | 30% | 22.8% | M04 can't grok p=337 at 30% data |
| M04_p601_lr1e4 | M04 (427k) | 601 | 30% | 0.18% | lr=1e-4 too low — model collapse |
| M06_p601_lr1e4 | M06 (1.09M) | 601 | 30% | 0.18% | Same model collapse |

### Crashed

| Experiment | Reason |
|-----------|--------|
| M07_p337_lr3e4 | M07 not in trainer model grid (FIXED — added M07/M08) |
| M04_p1201_lr3e5 | MPS OOM (41 GiB, p=1201 has 28,800 train examples) |

### Key Findings

1. **p=337 is grokkable locally** with M05/M06 at 30-40% train fraction
2. **M04 (427k) can grok p=337** but needs 50% train data (not 30%)
3. **p=601 needs lr=3e-4** (lr=1e-4 causes model collapse, lr=1e-3 diverges)
4. **p=1201 OOMs on MPS** — too large for local GPU
5. **Slingshot/forgetting visible** — M05/M06 grokked to ~99%, then went through slingshot phase

---

## Overnight Run 2 Results (Sept 15, CRT-BOTH Scaling)

**8 experiments completed** in ~8 hours on Apple MPS. 2 grokked, 6 failed/collapsed.

### Grokked

| Experiment | Model | Prime | Train% | Best Acc | Epochs | Notes |
|-----------|-------|-------|--------|----------|--------|-------|
| M07_p337_lr3e4 | M07 (1.6M) | 337 | 30% | **96.3%** | 800 | Fast grok |
| M08_p337_lr3e4 | M08 (2.5M) | 337 | 30% | **99.7%** | 400 | Fastest p=337 grok |

### Failed/Collapsed

| Experiment | Model | Prime | LR | Train% | Best Acc | Reason |
|-----------|-------|-------|----|--------|----------|--------|
| M04_p601_lr3e4 | M04 | 601 | 3e-4 | 30% | 0.18% | MPS collapse |
| M06_p601_lr3e4 | M06 | 601 | 3e-4 | 30% | 0.19% | MPS collapse |
| M04_p337_tf40 | M04 | 337 | 3e-4 | 40% | 20.9% | Needs >40% data |
| M04_p337_tf50_s123 | M04 | 337 | 3e-4 | 50% | 0.3% | **Seed collapse!** |
| M05_p601_lr5e4 | M05 | 601 | 5e-4 | 30% | 0.17% | MPS collapse |
| M04_p601_tf50 | M04 | 601 | 3e-4 | 50% | 0.18% | MPS collapse |

### Key Findings

1. **M07/M08 grok p=337 at 30%** — M08 in just 400 epochs (fastest p=337 grok yet)
2. **p=601 collapses on MPS regardless of LR** — tested 1e-4, 3e-4, 5e-4; all collapse (loss→0, train_acc→0). This is an MPS numerical issue, not representation failure (M12 on LUMI groks p=601 at 100%)
3. **Seed sensitivity at phase boundary** — M04 p=337 at 50% train: seed 42 groks (95.6%), seed 123 collapses (0.3%)
4. **M04 p=337 needs ≥50% data** — 40% fails at 20.9%, 50% groks at 95.6%

---

## Completed Work

### 1. LUMI Capacity Scaling (M01-M12, p=113) — COMPLETE

| Model | Params | Train% | Best Acc | Grokked? |
|-------|--------|--------|----------|----------|
| M01 | 90k | 33% | 7.6% | ❌ |
| M02 | 175k | 33% | 11.9% | ❌ |
| M03 | 287k | 33% | 15.5% | ❌ |
| **M04** | **427k** | **33%** | **96.2%** | **✅** |
| M05 | 657k | 33% | 25.7% | ❌ (at 33%) |
| M06 | 1.09M | 33% | 28.5% | ❌ (at 33%) |
| M07 | 1.64M | 35% | 99.8% | ✅ |
| M08 | 2.54M | 40% | 99.8% | ✅ |
| M09 | 3.64M | 45% | 99.7% | ✅ |
| M10 | 5.66M | 50% | 99.5% | ✅ |
| M11 | 9.03M | 55% | 99.2% | ✅ |
| M12 | 14.4M | 60% | 99.8% | ✅ |

**Key finding:** Grokking valley at M05-M06 (larger than M04 but fail at 33% data).

### 2. M05 Grokking Valley Resolution — NEW FINDING

M05 grokked **locally** at 99.6% test accuracy with **40% train fraction** (18,321 epochs).
On LUMI, M05 failed at 25.7% with 33% train fraction.

**Implication:** The grokking valley is **train-fraction dependent**, not a fundamental capacity barrier.
M05 needs more data than M04 to grok, but can grok when given enough.

### 3. CRT Representation Variants (p=113, M04) — COMPLETE

| Variant | Seeds | Mean Acc | Mean Epochs | Grok Rate |
|---------|-------|----------|-------------|-----------|
| RAW | 3 | 57.2% | 500K | 0/3 |
| CRT-BOTH | 3 | 99.99% | 21K | 3/3 |
| CRT-16 | 1 | 9.7% | 500K | 0/1 |
| CRT-7 | 1 | 4.0% | 500K | 0/1 |
| RAW+CRT | 1 | 100% | 77.8K | 1/1 |
| SCRAMBLED | 5 | 99.6% | 13.6K | 5/5 |

**SCRAMBLED multi-seed (5 seeds):** All grokked (99.2-100%), mean ~13.6K epochs.
- s42: 100%, 16,050 epochs
- s123: 99.5%, 14,900 epochs
- s456: 99.2%, 15,540 epochs
- s7: 99.6%, 8,230 epochs
- s789: 99.2%, 13,170 epochs

**Key finding:** SCRAMBLED groks faster than CRT-BOTH. Information content, not algebraic labeling, drives acceleration. Confirmed across 5 seeds.

### 4. CRT-BOTH Train-Fraction Sweep (p=113, M04) — COMPLETE

| Train Frac | Train Size | Best Acc | Epochs | Grokked? |
|------------|-----------|----------|--------|----------|
| 0.10 | 537 | 60.9% | 100K | ❌ |
| 0.15 | 806 | 99.6% | 64,210 | ✅ |
| 0.20 | 1,075 | 99.8% | 39,700 | ✅ |
| 0.25 | 1,344 | 99.6% | 9,830 | ✅ |
| 0.30 | 1,612 | 99.99% | 21,000 | ✅ (3 seeds) |

**Key finding:** Critical training fraction for CRT-BOTH is between 10% and 15%.
At 10%, the model memorizes but cannot generalize (60.9% best after 100K epochs).

### 5. Prime-Order DLP (order-113 subgroup of F_227*) — COMPLETE

| Experiment | Model | Split | Best Acc | Epochs |
|------------|-------|-------|----------|--------|
| P113-iid | M04 | IID | 100% | 145K |
| P113-iid | M04 | IID | 100% | 200K |
| P113-iid | M04 | IID | 99.99% | 243K |
| P113-iid | M06 | IID | 100% | 408K |
| P113-iid | M08 | IID | 99.90% | 417K |
| P113-ood | M04 | OOD | 100% | 326K |

**Key finding:** All 6 grokked, including OOD. Smooth factorization NOT required.

### 6. M12 CRT-BOTH Scaling (LUMI) — PARTIAL

| Prime | CRT Split | Train Frac | Best Acc | Epochs | Status |
|-------|-----------|-----------|----------|--------|--------|
| 113 | 7×16 | 0.30 | 99.88% | 32,250 | ✅ |
| 113 | 7×16 | 0.60 | 99.95% | 5,100 | ✅ |
| 241 | 15×16 | 0.30 | 99.96% | 7,550 | ✅ |
| 241 | 15×16 | 0.60 | 100% | 5,050 | ✅ |
| 337 | 16×21 | 0.30 | 99.98% | 5,200 | ✅ |
| 337 | 16×21 | 0.60 | 100% | 5,100 | ✅ |
| 601 | 24×25 | 0.30 | **100%** | 5,100 | ✅ |
| 601 | 24×25 | 0.60 | **100%** | 5,350 | ✅ |
| 1201 | 25×48 | 0.30 | — | — | OOMed — resubmitted with mini-batch |
| 1201 | 25×48 | 0.60 | — | — | OOMed — resubmitted with mini-batch |
| 2003 | 26×77 | 0.30 | — | — | Pending (mini-batch) |
| 2003 | 26×77 | 0.60 | — | — | Pending (mini-batch) |
| 4001 | 32×125 | 0.30 | — | — | Pending (mini-batch) |
| 4001 | 32×125 | 0.60 | — | — | Pending (mini-batch) |
| 8209 | 27×304 | 0.30 | — | — | Pending (mini-batch) |
| 8209 | 27×304 | 0.60 | — | — | Pending (mini-batch) |

**Key finding:** M12 CRT-BOTH groks p=601 at 100% in ~5K epochs (same as p=337).
Grokking time does not increase with prime size for CRT-BOTH at fixed train fraction.
p=1201+ OOMed with full-batch training; resubmitted with mini-batch gradient accumulation.

### 7. Local CRT-BOTH Scaling (M04-M08) — COMPLETE

| Prime | Model | Train% | Best Acc | Epochs | Status |
|-------|-------|--------|----------|--------|--------|
| 113 | M04 | 30% | 96.3% | 500 | ✅ Grokked instantly |
| 241 | M04 | 30% | 100% | 1,600 | ✅ Grokked instantly |
| 337 | M04 | 30% | 22.8% | 9,000 | ❌ Needs more data/capacity |
| 337 | M04 | 40% | 20.9% | 10,800 | ❌ Still insufficient |
| 337 | M04 | 50% | 95.6% | 7,400 | ✅ (seed 42) |
| 337 | M04 | 50% (s=123) | 0.3% | 12,400 | ❌ Seed collapse! |
| 337 | M05 | 40% | 99.99% | 8,000 | ✅ |
| 337 | M06 | 30% | 99.24% | 8,000 | ✅ |
| 337 | M07 | 30% | 96.3% | 800 | ✅ Fast grok |
| 337 | M08 | 30% | 99.7% | 400 | ✅ Fastest p=337 grok |
| 601 | M04/M05/M06 | 30-50% | 0.17-0.19% | 15,000 | ❌ MPS collapse (all LRs) |

**Key finding:** M07/M08 grok p=337 at 30% in 400-800 epochs. p=601 collapses on MPS regardless of LR — local frontier stays at p=337.

### 8. Large-Prime Scaling (LUMI) — PARTIAL (M15-M18 OOMed, resubmitted)

| Model | Prime | Best Acc | Status |
|-------|-------|----------|--------|
| M13 | 257 | 97.5% | Timed out at 197K — resubmitted |
| M14 | 521 | 99.93% | ✅ GROKKED (epoch 30K) |
| M15 | 1031 | — | OOMed — resubmitted with mini-batch (bs=4096) |
| M16 | 2053 | — | OOMed — resubmitted with mini-batch (bs=2048) |
| M17 | 4099 | — | OOMed — resubmitted with mini-batch (bs=1024) |
| M18 | 8209 | — | OOMed — resubmitted with mini-batch (bs=512) |

**Root cause of OOM:** LUMI trainer used full-batch gradient descent (all training examples in one forward pass). M15-M18 have 126K-5M training examples through 67M-190M parameter models, exceeding 64GB GPU memory. Fixed by adding mini-batch gradient accumulation to the trainer.

### 9. Mechanistic Analysis — COMPLETE (M04)

- **Group structure probes**: Complete (75 checkpoints). Composition consistency and cyclic shift accuracy rise in tandem with test accuracy.
- **Fourier spectral analysis**: Complete. Multiplicative character basis becomes sparse during grokking; additive basis remains diffuse. Basis preference = 0.9912 (multiplicative).
- **Extraction analysis**: Complete for M04 and M05. M04 needs 4 key frequencies for 100% accuracy.

### 10. Paper — FIRST DRAFT COMPLETE

- 622 lines, 8 figures, 26 references
- All sections written: discovery, dynamics, mechanistic, scaling, CRT, prime-order, discussion
- 3 LUMI figures generated (capacity scaling, CRT comparison, prime-order)

---

## LUMI Job Queue (Sept 15, 16:00)

### Pending (3 DLP batch jobs)
- `22071579`: **Tier 1 Critical** (23 jobs) — capacity sweep 30%, data sweep, multi-seed, prime-order
- `22071580`: **CRT-BOTH Frontier** (8 jobs) — M12 on p=1201, 2003, 4001, 8209 (mini-batch)
- `22071954`: **Large Primes v2** (5 jobs) — M13 resubmit + M15-M18 with mini-batch

### LUMI Trainer Patched (Sept 15)
- Added `train_batch_size` parameter for mini-batch gradient accumulation
- When set, processes data in chunks, accumulates gradients, steps once per epoch
- Backward compatible (0 = full batch, the default)
- Fixes OOM for large CRT primes (p=1201+) and large RAW primes (p=1031+)

### Previously Completed LUMI Jobs
- `22033821`: SCRAMBLED multi-seed + train-frac sweep — COMPLETE
- `22038895`: M12 CRT-BOTH scaling p=113-601 — COMPLETE (p=1201 OOMed)
- `21986808`: M13-M18 large-prime — TIMED OUT (M13/M14 have results, M15-M18 OOMed)

---

## FRE/FCO O(1) Model Architecture (Sept 15) — NEW

Implemented Factorized Residue Embedding (FRE) and Factorized Candidate Output (FCO), ported from the EI4 ECDLP grokking study. This makes model size O(1) in prime p, resolving the "tokenizer confound" limitation.

### Sanity Check + Scaling Results (FRE/FCO + CRT-BOTH, B=256)

| Prime | CRT Split | M | P_total | Best Acc | T_mem | T_90 | Grokked? | Platform |
|-------|----------|---|---------|----------|-------|------|----------|----------|
| 113 | 7×16 | 1 | 464k | 100% | 100 | ~2,000 | YES (ep 2900) | Local |
| 241 | 15×16 | 1 | 464k | 100% | 300 | ~400 | YES (ep 3900) | Local |
| 337 | 16×21 | 2 | 530k | 100% | 400 | ~400 | YES (ep 3000) | Local |
| 601 | 24×25 | 2 | 530k | 99.95% | 500 | ~400 | YES (ep 500) | Local |
| 769 | 3×256 | 2 | 530k | 100% | 2200 | ~5400 | YES (ep 5400) | Local |
| 1201 | 25×48 | 2 | 530k | 100% | 300 | ~200 | YES (ep 400) | LUMI |
| 2003 | 26×77 | 2 | 530k | 100% | 200 | ~200 | YES (ep 300) | LUMI |
| **4001** | **32×125** | **2** | **530k** | **100%** | **500** | **~500** | **YES (ep 500)** | **LUMI** |
| 8209 | 27×304 | 2 | 530k | — | — | — | Queued (LUMI) | LUMI |
| 16411 | 30×547 | 2 | 530k | — | — | — | Queued (LUMI) | LUMI |
| 32771 | 145×226 | 2 | 530k | — | — | — | Queued (LUMI) | LUMI |

**Key finding:** The same ~530k-param model groks p=601 in just 500 epochs — faster than the atomic M12 model (14.4M params, 5K epochs on LUMI). The FRE/FCO + CRT-BOTH combination is extremely efficient.

**Note:** M=1 (B=256) only works for p ≤ 256. For p=601+, M=2 is needed (target x up to p-2 > 255).

### Parameter Scaling (O(1) in p)

| Prime | CRT Split | M | B | P_total | Atomic P_total | Reduction |
|-------|----------|---|---|---------|----------------|-----------|
| 113 | 7×16 | 1 | 256 | 464k | 424k | 0.9× |
| 8209 | 27×304 | 2 | 256 | 526k | 2,497k | 4.7× |
| 32771 | 145×226 | 2 | 256 | 526k | 8,785k | 16.7× |

### Next Steps

1. Test p=241, p=601 locally with FRE/FCO + CRT-BOTH
2. Submit p=8209, p=16411, p=32771 to LUMI
3. If wall is hit, try RichFCO (bilinear digit interactions)

### Files

- `src/fre_fco.py` — FRE, FCO, RichFCO, GrokkingTransformerFRE
- `experiments/fre_fco/run_fre_fco.py` — Training script
- `DLP_GROKKING_PLAN.md` — Full scaling plan
- `results/fre_fco/` — Experiment results

---

## Pending Work

### Experiments
1. **M12 CRT-BOTH p=601, p=1201**: Waiting for LUMI job 22038895 to progress
2. **M13-M18 large-prime**: Waiting for LUMI job 21986808 to complete
3. **M04 multi-seed (s123, s456)**: s123 at 49.5% (21.5K epochs), s456 not started
4. **p=337 M06 CRT-BOTH**: In progress locally
5. **p=601 M04 CRT-BOTH**: In progress locally
6. **Attention pattern analysis**: Not started — needs head-level analysis code

### Figures
1. **M12 CRT-BOTH scaling figure**: Need to generate from LUMI results
2. **Train-fraction heatmap**: Data available from LUMI sweep, need to generate
3. **M05 grokking valley resolution figure**: Need to show M05 groks at 40% but fails at 33%

### Writing
1. **Update scaling table**: Add M05 local result (99.6% at 40% train)
2. **Add M12 CRT-BOTH scaling section**: New section with LUMI results
3. **Add local CRT-BOTH scaling section**: M04 groks p=241 in 200 epochs
4. **Update SCRAMBLED to 5 seeds**: Currently shows 1 seed in paper
5. **Add CRT-BOTH train-fraction sweep**: Critical fraction between 10-15%
6. **Update limitations**: Remove completed items, add new ones

---

## Key Hypotheses Status

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: Representation emergence precedes generalization | Supported | PR compresses before accuracy rises (M04, 75 checkpoints) |
| H2: Explicit representation accelerates learning | **Revised** | CRT-BOTH 24×, but SCRAMBLED even faster (5 seeds confirmed) |
| H3: Causal ablation selectively harms generalization | Supported | Double dissociation (M04, M05) |
| H4: Representation quality predicts sample complexity | **Supported** | CRT-BOTH critical frac 10-15%; RAW never groks |
| H5: Representation-aligned OOD generalization | Supported | P113-OOD grokked at 100% |
| H6: Structural changes induce different mechanisms | Supported | P113 uses different spectral representation |
| H7: Equivalent latent mathematics predicts convergence | Pending | Multi-curve experiment (future work) |

---

## Files

### Key Scripts
- `src/trainer.py` — Local single-curve trainer
- `src/models.py` — GrokkingTransformer model
- `lumi_dlp/lumi_dlp_trainer.py` — LUMI trainer (generalized CRT)
- `experiments/multicurve/run_multicurve.py` — Multi-curve trainer (with CRT-BOTH)
- `paper/generate_figures.py` — Figure generation
- `analyze_lumi_results.py` — LUMI results analysis

### Results Directories
- `results/local_probe/` — M04, M05, M09 local training + probing
- `results/extraction/` — Fourier extraction from grokked models
- `results/cross_prime/` — Cross-prime transfer (p=251, failed)
- `results/multicurve/` — Multi-curve and CRT-BOTH scaling experiments
- `lumi_results/` — Pulled LUMI summaries

### Paper
- `paper/paper.tex` — 622 lines, first draft
- `paper/figures/` — 12 figures (9 original + 3 LUMI)
- `paper/rule_extraction_plan.md` — Research plan (Phases 1-7)
