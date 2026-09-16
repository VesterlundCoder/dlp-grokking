#!/usr/bin/env python3
"""Generate figures for the symbolic regression analysis.

Creates:
1. temporal_trajectory.pdf — complexity and R² over training epochs
2. free_vs_algebra.pdf — R² comparison across dimensions
3. param_scaling.pdf — (already in paper, skip)
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = '/Users/davidsvensson/Desktop/dlp_grokking/results/symbolic_regression/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Figure 1: Temporal trajectory
# ============================================================================

def plot_temporal_trajectory():
    """Plot complexity and R² over training epochs for dim 54."""
    with open('/Users/davidsvensson/Desktop/dlp_grokking/results/symbolic_regression/M04_s42_temporal/temporal_dim54_pre_unembed.json') as f:
        data = json.load(f)

    epochs = []
    complexities = []
    r2s = []
    corrs = []
    for r in data:
        ep = r['epoch']
        if ep == 'final':
            ep = 37481
        epochs.append(ep)
        complexities.append(r['sr_complexity'] or 0)
        r2s.append(r['sr_r2'] or 0)
        corrs.append(r['corr_with_x'])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Mark grokking transition
    grok_epoch = 22650
    mem_epoch = 90

    # Plot 1: Complexity
    ax1.plot(epochs, complexities, 'bo-', markersize=6, label='SR Complexity')
    ax1.axvline(x=mem_epoch, color='green', linestyle='--', alpha=0.5, label=f'Memorization (ep {mem_epoch})')
    ax1.axvline(x=grok_epoch, color='red', linestyle='--', alpha=0.5, label=f'Grokking (ep {grok_epoch})')
    ax1.set_ylabel('Symbolic Complexity')
    ax1.set_title('Temporal Trajectory of Symbolic Structure (dim 54, pre_unembed)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: R²
    ax2.plot(epochs, r2s, 'rs-', markersize=6, label='Algebra-aware SR R²')
    ax2.axvline(x=mem_epoch, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(x=grok_epoch, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('R² (algebra-aware SR)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Correlation with x
    ax3.plot(epochs, corrs, 'g^-', markersize=6, label='|corr(z, x)|')
    ax3.axvline(x=mem_epoch, color='green', linestyle='--', alpha=0.5)
    ax3.axvline(x=grok_epoch, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('|corr(z_54, x)|')
    ax3.set_xlabel('Training Epoch')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_trajectory.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal_trajectory.pdf")

    # Also plot the presence of cos_x_q/sin_x_q in equations
    has_char = []
    for r in data:
        eq = (r.get('sr_equation') or '').lower()
        has_char.append(1 if ('cos_x_q' in eq or 'sin_x_q' in eq) else 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(epochs)), has_char, color=['blue' if h else 'gray' for h in has_char])
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([str(e) for e in epochs], rotation=45)
    ax.set_ylabel('Contains cos_x_q or sin_x_q')
    ax.set_title('Presence of Multiplicative Character Structure in SR Equations')
    ax.axvline(x=next((i for i, e in enumerate(epochs) if e >= grok_epoch), 0),
               color='red', linestyle='--', alpha=0.5, label='Grokking')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'character_presence.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved character_presence.pdf")


# ============================================================================
# Figure 2: Free SR vs Algebra-aware SR
# ============================================================================

def plot_free_vs_algebra():
    """Plot R² comparison across dimensions."""
    with open('/Users/davidsvensson/Desktop/dlp_grokking/results/symbolic_regression/M04_s42/sr_summary_pre_unembed_L3.json') as f:
        data = json.load(f)

    dims = []
    free_r2 = []
    alg_r2 = []

    for d in data['level_3']:
        dims.append(d['dim'])
        free_r2.append(d.get('free_sr', {}).get('r2', 0))
        alg_r2.append(d.get('alg_sr', {}).get('r2', 0))

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, free_r2, width, label='Free SR (no modulo)', color='steelblue')
    bars2 = ax.bar(x + width/2, alg_r2, width, label='Algebra-aware SR', color='coral')

    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('R²')
    ax.set_title('Free SR vs Algebra-aware SR (pre_unembed layer)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'dim {d}' for d in dims])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'free_vs_algebra.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved free_vs_algebra.pdf")


if __name__ == '__main__':
    plot_temporal_trajectory()
    plot_free_vs_algebra()
    print(f"\nAll figures saved to {OUTPUT_DIR}")
