#!/usr/bin/env python3
"""Regenerate manuscript figures from frozen source data.

This script regenerates the key manuscript figures from the frozen results
in results/runs.csv and the checkpoint trajectories.

Usage:
    python3 reproduce/reproduce_figures.py
"""
import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'paper', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_runs():
    """Load the canonical runs.csv."""
    runs = []
    with open(os.path.join(REPO_ROOT, 'results', 'runs.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(row)
    return runs


def plot_capacity_sweep():
    """Regenerate the capacity sweep figure."""
    runs = load_runs()
    # Filter M01-M12 at p=113
    capacity_runs = [r for r in runs if r['p'] == '113' and r['run_classification'] in ('grokked', 'failed')]
    capacity_runs.sort(key=lambda r: int(r.get('P_total', 0) or 0))

    params = []
    accs = []
    labels = []
    for r in capacity_runs:
        p_total = int(r.get('P_total', 0) or 0)
        if p_total == 0:
            continue
        params.append(p_total / 1000)  # thousands
        accs.append(float(r.get('best_test_acc', 0) or 0))
        labels.append(r['experiment_id'].split('_')[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['green' if a > 0.9 else 'red' for a in accs]
    ax.bar(range(len(params)), accs, color=colors, alpha=0.7)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Capacity Sweep: M01-M12 at p=113')
    ax.axhline(y=0.9, color='blue', linestyle='--', alpha=0.5, label='Grokking threshold (90%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_capacity_scaling_repro.png'), dpi=150)
    plt.close()
    print(f"Saved fig_capacity_scaling_repro.png")


def plot_sr_comparison():
    """Regenerate the free vs algebra-aware SR comparison."""
    sr_path = os.path.join(REPO_ROOT, 'results', 'symbolic_regression', 'M04_s42',
                           'sr_summary_pre_unembed_L3.json')
    if not os.path.exists(sr_path):
        print(f"WARNING: SR results not found at {sr_path}")
        return

    with open(sr_path) as f:
        data = json.load(f)

    dims = []
    free_r2 = []
    alg_r2 = []
    for d in data.get('level_3', []):
        dims.append(d['dim'])
        free_r2.append(d.get('free_sr', {}).get('r2', 0))
        alg_r2.append(d.get('alg_sr', {}).get('r2', 0))

    x = np.arange(len(dims))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, free_r2, width, label='Free SR (no modulo)', color='steelblue')
    ax.bar(x + width/2, alg_r2, width, label='Algebra-aware SR', color='coral')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('R²')
    ax.set_title('Free SR vs Algebra-aware SR (pre_unembed layer)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'dim {d}' for d in dims])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_sr_comparison_repro.png'), dpi=150)
    plt.close()
    print(f"Saved fig_sr_comparison_repro.png")


def main():
    print("Regenerating manuscript figures from frozen data...")
    plot_capacity_sweep()
    plot_sr_comparison()
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    main()
