# Grokking the Discrete Logarithm Problem

Code and data for the preprint **"Grokking the Discrete Logarithm: Scaling Across Model Capacity and Mechanistic Analysis"** by David Vesterlund.

## Overview

This repository contains the training code, analysis pipelines, datasets, and figure-generation scripts for a systematic study of delayed generalization ("grokking") on the finite-field discrete logarithm problem (DLP).

Key results:
- A 427k-parameter, 2-layer transformer groks the full variable-base DLP at p=113, reaching 99.8% test accuracy after a prolonged memorization phase.
- A 12-model capacity sweep (90k–14.4M parameters) shows DLP grokking across multiple scales, with a non-monotonic, data-dependent region.
- Mechanistic analysis reveals spectral concentration in the multiplicative character basis during grokking.
- Symbolic regression on hidden states recovers multiplicative-character structure; free regression independently discovers Fourier harmonics.
- Prime-order and generator-OOD controls confirm grokking is not restricted to smooth composite-order groups.

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Regenerate the p=113 task and train a small model

```bash
# Generate the p=113 DLP dataset and train M04 for 1000 epochs
python3 reproduce/reproduce_anchor.py --epochs 1000

# Regenerate the phase-diagram figure from frozen results
python3 reproduce/reproduce_figures.py
```

### Verify the release

```bash
python3 reproduce/verify_release.py
```

## Repository Structure

```
dlp-grokking/
├── README.md
├── LICENSE                    # MIT License
├── CITATION.cff               # Citation metadata
├── requirements.txt
├── pyproject.toml
├── paper/
│   ├── paper.tex              # Manuscript (LaTeX)
│   ├── paper.pdf              # Compiled manuscript
│   └── figures/               # All manuscript figures
├── src/
│   ├── models.py              # GrokkingTransformer architecture
│   ├── trainer.py             # Training loop, tokenizer
│   ├── extract_algorithm.py   # Fourier/character extraction
│   ├── fourier_analysis.py    # Spectral analysis
│   └── symbolic_regression/  # SR pipeline (PySR-based)
├── experiments/
│   ├── multicurve/            # CRT-BOTH and multi-curve experiments
│   └── fre_fco/               # FRE/FCO O(1) model experiments
├── results/
│   ├── runs.csv               # Canonical result table
│   └── pending_runs.csv        # Experiments not yet complete
├── splits/                    # Frozen train/test split manifests
├── checkpoints/               # Checkpoint manifest (hashes only)
├── reproduce/
│   ├── reproduce_anchor.py    # End-to-end anchor reproduction
│   ├── reproduce_figures.py   # Regenerate all manuscript figures
│   └── verify_release.py      # Verify data/split/checkpoint hashes
└── analysis/
    ├── spectral/              # Fourier analysis scripts
    ├── group_probes/           # Group-structure probe scripts
    └── symbolic_regression/   # SR analysis scripts
```

## Datasets

The DLP datasets are deterministic finite mathematical objects. For p=113:
- Domain: all (g, h, x) triples where g is a primitive root of F*_113 and g^x ≡ h (mod 113)
- Size: φ(112) × 112 = 48 × 112 = 5,376 unique triples
- Split: 30% train (1,612 examples), 70% test (3,764 examples), pair-IID

Split manifests with SHA256 hashes are in `splits/`.

## Checkpoints

Selected checkpoints for the M04 anchor run (p=113, seed 42) are archived:
- Initialization (epoch 0)
- First memorization (epoch 90)
- Pre-grok (epoch 22,000)
- T50, T90, T99 milestones
- Final (epoch 37,481)

Checkpoint hashes are in `checkpoints/checkpoint_manifest.csv`.

## Citation

```bibtex
@misc{vesterlund2026grokking,
  title={Grokking the Discrete Logarithm: Scaling Across Model Capacity and Mechanistic Analysis},
  author={Vesterlund, David},
  year={2026},
  eprint={arXiv preprint (forthcoming)},
  url={https://github.com/VesterlundCoder/dlp-grokking}
}
```

## License

MIT License. See `LICENSE` for details.

## Acknowledgments

Compute resources provided by LUMI-G (EuroHPC JU) and local Apple MPS hardware.
