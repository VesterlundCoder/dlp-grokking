#!/usr/bin/env python3
"""Symbolic regression on DLP grokking model activations using PySR.

Implements the pipeline:
    activations -> sparse feature selection -> PySR (free + algebra-aware)

Levels of extraction:
    Level 2: latent-state extraction: (g, h) -> z_j
    Level 3: analysis-coordinate extraction: (a, b) -> z_j  (most promising)
    Level 4: subnetwork extraction: z_l -> z_{l+1}

Operator sets:
    Free SR: +, -, *, /, sin, cos, exp, log  (no modulo knowledge)
    Algebra-aware SR: +, -, *, /, mod, inv, sin(2πx/q), cos(2πx/q)

Usage:
    cd /Users/davidsvensson/Desktop/dlp_grokking
    python3 src/symbolic_regression/run_symbolic_regression.py \
        --activations results/symbolic_regression/M04_s42/activations.npz \
        --output-dir results/symbolic_regression/M04_s42 \
        --level 3 --layer block_1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# Linear Probe (baseline)
# ============================================================================

def linear_probe(X, y, feature_names, output_dir, name_prefix="probe"):
    """Linear probe baseline: how well does a linear model predict y from X?"""
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)

    print(f"\n  Linear Probe R²: {r2:.6f}")

    result = {
        'method': 'linear_probe',
        'r2': float(r2),
        'coefficients': dict(zip(feature_names, reg.coef_.tolist())),
        'intercept': float(reg.intercept_),
    }

    with open(os.path.join(output_dir, f'{name_prefix}_probe.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================================
# PySR Symbolic Regression
# ============================================================================

def run_pysr(X, y, feature_names, output_dir, name_prefix, binary_operators, unary_operators,
             extra_sympy_mapping=None, q=None, niterations=100, population_size=30):
    """Run PySR symbolic regression with given operator sets."""
    from pysr import PySRRegressor

    print(f"\n  PySR: {name_prefix}")
    print(f"    X: {X.shape}, y: {y.shape}")
    print(f"    Binary ops: {binary_operators}")
    print(f"    Unary ops: {unary_operators}")

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        population_size=population_size,
        maxsize=25,
        verbosity=1,
        progress=False,
        random_state=42,
        deterministic=True,
        parallelism='serial',
        model_selection='best',
    )

    try:
        model.fit(X, y, variable_names=feature_names)
    except Exception as e:
        print(f"    PySR failed: {e}")
        return {'method': name_prefix, 'error': str(e)}

    # Get the best equation (lowest loss = highest score)
    print(f"    Equations: {len(model.equations_)}")
    best = model.equations_.iloc[-1]
    print(f"    Best equation: {best['equation']}")
    print(f"    Best loss: {best['loss']:.6f}")
    print(f"    Best complexity: {best['complexity']}")

    # Compute R² manually
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print(f"    Best R²: {r2:.6f}")

    result = {
        'method': name_prefix,
        'equation': str(best['equation']),
        'r2': float(r2),
        'loss': float(best['loss']),
        'complexity': int(best['complexity']),
        'feature_names': feature_names,
    }
    if q is not None:
        result['q'] = q

    with open(os.path.join(output_dir, f'{name_prefix}_sr.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def run_free_sr(X, y, feature_names, output_dir, name_prefix="free"):
    """Free SR: generic operators, no modulo knowledge (anti-leakage control)."""
    return run_pysr(
        X, y, feature_names, output_dir, name_prefix,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log"],
        niterations=100,
        population_size=30,
    )


def run_algebra_aware_sr(X, y, feature_names, output_dir, p, q, name_prefix="algebra"):
    """Algebra-aware SR: modular operators + character functions."""
    # PySR doesn't support modular operators directly, but we can use
    # the modulo operator and custom functions via Julia
    # For now, use standard operators + mod-like approximations

    # We'll use the fact that PySR can handle custom operators defined in Julia
    # But for simplicity, let's use the standard operators + a "mod" function
    # that we define as a custom operator

    # Actually, PySR supports custom operators. Let's define modular ones.
    # But the simplest approach: use standard operators and let PySR find
    # the structure, then check if it matches modular arithmetic.

    # Alternative: precompute the modular features and add them as inputs
    a_vals = X[:, 0]
    b_vals = X[:, 1]

    # Add modular features
    a_inv = np.array([pow(int(a), -1, q) if math.gcd(int(a), q) == 1 else 0
                      for a in a_vals])
    b_inv = np.array([pow(int(b), -1, q) if math.gcd(int(b), q) == 1 else 0
                      for b in b_vals])
    x_calc = (a_inv * b_vals) % q

    # Character functions
    cos_a = np.cos(2 * np.pi * a_vals / q)
    sin_a = np.sin(2 * np.pi * a_vals / q)
    cos_b = np.cos(2 * np.pi * b_vals / q)
    sin_b = np.sin(2 * np.pi * b_vals / q)
    cos_x = np.cos(2 * np.pi * x_calc / q)
    sin_x = np.sin(2 * np.pi * x_calc / q)

    X_aug = np.stack([a_vals, b_vals, a_inv, b_inv, x_calc,
                      cos_a, sin_a, cos_b, sin_b, cos_x, sin_x], axis=1)
    feat_names = ['a', 'b', 'a_inv', 'b_inv', 'x',
                  'cos_a_q', 'sin_a_q', 'cos_b_q', 'sin_b_q',
                  'cos_x_q', 'sin_x_q']

    # Run SR with standard operators on augmented features
    return run_pysr(
        X_aug, y, feat_names, output_dir, name_prefix,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        niterations=100,
        population_size=30,
        q=q,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--level', type=int, default=3, choices=[2, 3, 4])
    parser.add_argument('--layer', type=str, default='block_1',
                        choices=['embed', 'block_0', 'block_1', 'pre_unembed'])
    parser.add_argument('--n-features', type=int, default=5)
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--skip-free', action='store_true')
    parser.add_argument('--skip-algebra', action='store_true')
    parser.add_argument('--niterations', type=int, default=100)
    args = parser.parse_args()

    p = args.p
    q = p - 1
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading activations from {args.activations}...")
    data = np.load(args.activations)
    Z = data[args.layer]
    a_vals = data['a']
    b_vals = data['b']
    g_vals = data['g']
    h_vals = data['h']
    x_vals = data['x']

    print(f"  Layer: {args.layer}, shape: {Z.shape}")

    results = {}

    if args.level == 3:
        print(f"\n--- Level 3: Analysis-Coordinate Extraction ---")
        print(f"  Mapping: (a, b) -> z_j")

        X_input = np.stack([a_vals, b_vals], axis=1).astype(float)
        input_names = ['a', 'b']

        # Feature selection: find dimensions with high variance and correlation with x
        variances = np.var(Z, axis=0)
        correlations = np.array([
            abs(np.corrcoef(Z[:, j], x_vals)[0, 1])
            if not np.isnan(np.corrcoef(Z[:, j], x_vals)[0, 1]) else 0
            for j in range(Z.shape[1])
        ])

        # Select top dims by correlation with x (the target)
        top_corr_dims = np.argsort(correlations)[::-1][:args.n_features]
        # Also get high-variance dims
        top_var_dims = np.argsort(variances)[::-1][:args.n_features]
        candidate_dims = sorted(set(top_corr_dims.tolist() + top_var_dims.tolist()))

        print(f"  Top dims by correlation with x: {top_corr_dims.tolist()}")
        print(f"  Correlations: {correlations[top_corr_dims]}")
        print(f"  Top dims by variance: {top_var_dims.tolist()}")
        print(f"  Testing dims: {candidate_dims[:5]}")

        selected_dims = []
        for j in candidate_dims[:5]:  # Test top 5
            z_j = Z[:, j]
            print(f"\n  === Dimension {j} ===")
            print(f"    Variance: {variances[j]:.4f}")
            print(f"    Correlation with x: {correlations[j]:.4f}")

            # Linear probe
            probe_result = linear_probe(X_input, z_j, input_names, args.output_dir,
                                        f"dim{j}_{args.layer}_L3")

            dim_result = {
                'dim': j,
                'variance': float(variances[j]),
                'corr_with_x': float(correlations[j]),
                'probe_r2': probe_result['r2'],
            }

            # Free SR
            if not args.skip_free:
                free_result = run_free_sr(X_input, z_j, input_names, args.output_dir,
                                          f"dim{j}_{args.layer}_L3_free")
                dim_result['free_sr'] = free_result

            # Algebra-aware SR
            if not args.skip_algebra:
                alg_result = run_algebra_aware_sr(X_input, z_j, input_names,
                                                  args.output_dir, p, q,
                                                  f"dim{j}_{args.layer}_L3_alg")
                dim_result['alg_sr'] = alg_result

            selected_dims.append(dim_result)

        results['level_3'] = selected_dims

    elif args.level == 2:
        print(f"\n--- Level 2: Latent-State Extraction ---")
        X_input = np.stack([g_vals, h_vals], axis=1).astype(float)
        input_names = ['g', 'h']

        variances = np.var(Z, axis=0)
        top_dims = np.argsort(variances)[::-1][:args.n_features]

        for j in top_dims:
            z_j = Z[:, j]
            print(f"\n  === Dimension {j} (var={variances[j]:.4f}) ===")
            linear_probe(X_input, z_j, input_names, args.output_dir,
                        f"dim{j}_{args.layer}_L2")
            if not args.skip_free:
                run_free_sr(X_input, z_j, input_names, args.output_dir,
                           f"dim{j}_{args.layer}_L2_free")
            if not args.skip_algebra:
                run_algebra_aware_sr(X_input, z_j, input_names,
                                    args.output_dir, p, q,
                                    f"dim{j}_{args.layer}_L2_alg")

    elif args.level == 4:
        print(f"\n--- Level 4: Subnetwork Extraction ---")
        layer_idx = ['embed', 'block_0', 'block_1', 'pre_unembed'].index(args.layer)
        if layer_idx == 0:
            print("  Cannot do Level 4 on embed")
            return
        prev_layer = ['embed', 'block_0', 'block_1', 'pre_unembed'][layer_idx - 1]
        Z_prev = data[prev_layer]
        print(f"  Previous layer: {prev_layer}, shape: {Z_prev.shape}")

        variances = np.var(Z, axis=0)
        top_dims = np.argsort(variances)[::-1][:args.n_features]

        for j in top_dims:
            z_j = Z[:, j]
            corr_prev = np.array([
                abs(np.corrcoef(Z_prev[:, k], z_j)[0, 1])
                if not np.isnan(np.corrcoef(Z_prev[:, k], z_j)[0, 1]) else 0
                for k in range(Z_prev.shape[1])
            ])
            top_prev = np.argsort(corr_prev)[::-1][:5]
            X_sel = Z_prev[:, top_prev]
            feat_names = [f"z_prev[{k}]" for k in top_prev]

            print(f"\n  === {prev_layer} -> {args.layer}[{j}] ===")
            print(f"    Top prev dims: {top_prev.tolist()}")
            linear_probe(X_sel, z_j, feat_names, args.output_dir,
                        f"dim{j}_{args.layer}_L4")
            if not args.skip_free:
                run_free_sr(X_sel, z_j, feat_names, args.output_dir,
                           f"dim{j}_{args.layer}_L4_free")

    with open(os.path.join(args.output_dir, f'sr_summary_{args.layer}_L{args.level}.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== Summary saved to {args.output_dir} ===")


if __name__ == '__main__':
    main()
