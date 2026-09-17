#!/usr/bin/env python3
"""Temporal trajectory symbolic regression: track complexity C(f_t) across checkpoints.

For each checkpoint, extract activations and run SR to find the simplest equation
that explains a selected latent dimension. Track how complexity changes over training.

Hypothesis (from the methodology):
    Pre-memorization:  No simple symbolic rule (C(f_t) >> 1)
    Memorization:      Fit is high but expression complex/unstable
    Pre-grok:          A simple algebraic rule starts to emerge
    Grok:              C(f_t) decreases while test accuracy increases

This would support: grokking = transition toward a simpler symbolic computation.

Usage:
    cd /Users/davidsvensson/Desktop/dlp_grokking
    python3 src/symbolic_regression/temporal_trajectory.py \
        --checkpoint-dir results/local_probe/M04_s42/checkpoints \
        --output-dir results/symbolic_regression/M04_s42_temporal \
        --p 113 --dim 54 --layer pre_unembed
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import re
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import GrokkingTransformer
from src.trainer import GrokkingTokenizer, _primitive_roots
from src.symbolic_regression.extract_activations import (
    load_model_from_checkpoint, build_dlp_domain_fast, extract_activations
)


def run_quick_pysr(X, y, feature_names, niterations=50, population_size=30):
    """Run a quick PySR fit and return the best equation's complexity and R²."""
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        population_size=population_size,
        maxsize=20,
        verbosity=0,
        progress=False,
        random_state=42,
        deterministic=True,
        parallelism='serial',
        model_selection='best',
    )

    try:
        model.fit(X, y, variable_names=feature_names)
    except Exception as e:
        return {'error': str(e), 'complexity': None, 'r2': None, 'equation': None}

    best = model.equations_.iloc[-1]

    # Compute R²
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'complexity': int(best['complexity']),
        'r2': float(r2),
        'loss': float(best['loss']),
        'equation': str(best['equation']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing epoch_*.pt checkpoints')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--dim', type=int, default=54,
                        help='Latent dimension to analyze')
    parser.add_argument('--layer', type=str, default='pre_unembed',
                        choices=['embed', 'block_0', 'block_1', 'pre_unembed'])
    parser.add_argument('--niterations', type=int, default=50)
    parser.add_argument('--max-checkpoints', type=int, default=15)
    args = parser.parse_args()

    p = args.p
    q = p - 1
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all checkpoints
    ckpt_pattern = re.compile(r'epoch_(\d+)\.pt')
    checkpoints = []
    for f in os.listdir(args.checkpoint_dir):
        m = ckpt_pattern.match(f)
        if m:
            epoch = int(m.group(1))
            checkpoints.append((epoch, os.path.join(args.checkpoint_dir, f)))
    # Also add final.pt
    final_path = os.path.join(args.checkpoint_dir, 'final.pt')
    if os.path.exists(final_path):
        checkpoints.append((999999, final_path))

    checkpoints.sort(key=lambda x: x[0])

    # Subsample to max_checkpoints
    if len(checkpoints) > args.max_checkpoints:
        indices = np.linspace(0, len(checkpoints) - 1, args.max_checkpoints, dtype=int)
        checkpoints = [checkpoints[i] for i in indices]

    print(f"Analyzing {len(checkpoints)} checkpoints for dim {args.dim}, layer {args.layer}")
    print(f"Checkpoints: {[e for e, _ in checkpoints]}")

    # Build DLP domain once
    tokenizer = GrokkingTokenizer(p)
    g0, pairs = build_dlp_domain_fast(p, q)
    print(f"Domain: {len(pairs)} pairs, g0={g0}")

    # Prepare algebra-aware features
    a_vals_all = np.array([p[3] for p in pairs]).astype(float)
    b_vals_all = np.array([p[4] for p in pairs]).astype(float)
    a_inv_all = np.array([pow(int(a), -1, q) if math.gcd(int(a), q) == 1 else 0
                         for a in a_vals_all])
    x_calc_all = (a_inv_all * b_vals_all) % q
    cos_x_all = np.cos(2 * np.pi * x_calc_all / q)
    sin_x_all = np.sin(2 * np.pi * x_calc_all / q)

    X_algebra = np.stack([a_vals_all, b_vals_all, a_inv_all, x_calc_all,
                          cos_x_all, sin_x_all], axis=1)
    feat_names = ['a', 'b', 'a_inv', 'x', 'cos_x_q', 'sin_x_q']

    results = []

    for epoch, ckpt_path in checkpoints:
        print(f"\n=== Epoch {epoch} ===")

        # Load model
        model, config = load_model_from_checkpoint(ckpt_path, device)

        # Extract activations
        acts = extract_activations(model, tokenizer, pairs, p, q, device, batch_size=512)
        z = acts[args.layer][:, args.dim]

        # Compute correlation with x
        x_vals = acts['x'].astype(float)
        corr_x = abs(np.corrcoef(z, x_vals)[0, 1]) if not np.isnan(np.corrcoef(z, x_vals)[0, 1]) else 0

        # Run algebra-aware SR
        sr_result = run_quick_pysr(X_algebra, z, feat_names,
                                    niterations=args.niterations)

        result = {
            'epoch': epoch if epoch < 999999 else 'final',
            'dim': args.dim,
            'layer': args.layer,
            'variance': float(np.var(z)),
            'corr_with_x': float(corr_x),
            'sr_complexity': sr_result.get('complexity'),
            'sr_r2': sr_result.get('r2'),
            'sr_loss': sr_result.get('loss'),
            'sr_equation': sr_result.get('equation'),
        }
        results.append(result)

        print(f"  Variance: {result['variance']:.4f}")
        print(f"  Corr with x: {result['corr_with_x']:.4f}")
        print(f"  SR complexity: {result['sr_complexity']}")
        print(f"  SR R²: {result['sr_r2']:.4f}" if result['sr_r2'] is not None else "  SR R²: None")
        print(f"  SR equation: {result['sr_equation']}")

    # Save results
    with open(os.path.join(args.output_dir, f'temporal_dim{args.dim}_{args.layer}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n=== Temporal Trajectory Summary ===")
    print(f"{'Epoch':>8} {'Variance':>10} {'Corr(x)':>8} {'Complexity':>10} {'R²':>8} {'Equation':>40}")
    for r in results:
        eq = (r['sr_equation'] or '')[:40]
        r2_str = f"{r['sr_r2']:.4f}" if r['sr_r2'] is not None else 'N/A'
        print(f"{str(r['epoch']):>8} {r['variance']:>10.4f} {r['corr_with_x']:>8.4f} "
              f"{str(r['sr_complexity']):>10} {r2_str:>8} {eq:>40}")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
