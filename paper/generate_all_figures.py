#!/usr/bin/env python3
"""Generate all figures for the DLP Grokking paper from LUMI results."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# Data: Capacity scaling (M01-M12, p=113)
# ============================================================================
CAPACITY_DATA = {
    # model: (P_total, P_core, train_frac, best_acc, t_mem, t_50, t_90, t_95, t_99, total_ep, grokked)
    'M01': (90221, 76720, 0.33, 0.0757, 420, None, None, None, None, 199999, False),
    'M02': (174917, 155680, 0.33, 0.1194, 150, None, None, None, None, 199999, False),
    'M03': (287261, 262288, 0.33, 0.1554, 120, None, None, None, None, 199999, False),
    'M04': (427253, 396544, 0.33, 0.9625, 90, 386600, 467720, 491520, None, 499999, True),
    'M05': (656917, 618560, 0.33, 0.2565, 80, None, None, None, None, 499999, False),
    'M06': (1093573, 1043744, 0.33, 0.2850, 60, None, None, None, None, 499999, False),
    'M07': (1640821, 1579520, 0.35, 0.9983, 100, 515800, 652450, 662200, 678600, 687250, True),
    'M08': (2542517, 2465920, 0.40, 0.9985, 100, 358050, 420150, 439300, 451500, 480650, True),
    'M09': (3640821, 3548928, 0.45, 0.9966, 100, 281650, 372450, 388800, 405000, 437200, True),
    'M10': (5656917, 5542080, 0.50, 0.9952, 100, 132500, 236850, 248600, 289400, 294350, True),
    'M11': (9033173, 8887744, 0.55, 0.9921, 100, 133650, 226500, 236350, 247500, 257950, True),
    'M12': (14359413, 14175744, 0.60, 0.9981, 100, 70600, 121900, 133000, 145650, 150800, True),
}

# Local M05 at 40% (grokked)
M05_LOCAL = (656917, 618560, 0.40, 0.996, 80, None, 18321, None, 18321, 18321, True)

# ============================================================================
# Data: RAW train-fraction sweep (p=113, M04)
# ============================================================================
RAW_FRAC_DATA = [
    # (train_frac, best_acc, t_mem, t_90, t_99, total_ep)
    (0.30, 0.076, 90, None, None, 500000),  # from original M04 run (33%)
    (0.40, 0.9764, 120, 175710, None, 499999),
    (0.50, 0.9926, 150, 107310, 161340, 173770),
    (0.60, 0.9912, 210, 92580, 133650, 135960),
    (0.70, 0.9919, 300, 98810, 149790, 152830),
]

# ============================================================================
# Data: CRT-BOTH train-fraction sweep (p=113, M04)
# ============================================================================
CRT_FRAC_DATA = [
    (0.10, 0.6089, 60, None, None, 99999),
    (0.15, 0.9963, 70, 51870, 62100, 64210),
    (0.20, 0.9977, 90, 21450, 34510, 39700),
    (0.25, 0.9959, 90, 1600, 7840, 9830),
    (0.30, 0.9999, 90, 21000, 21000, 21000),  # from 3-seed mean
]

# ============================================================================
# Data: Prime-order DLP
# ============================================================================
PRIME_ORDER_DATA = [
    # (model, seed, split, best_acc, t_mem, t_90, t_99, total_ep)
    ('M04', 42, 'IID', 1.0000, 150, 130000, 132400, 145650),
    ('M04', 123, 'IID', 1.0000, 150, 177950, 182150, 199750),
    ('M04', 7, 'IID', 0.9999, 150, 184050, 210750, 243150),
    ('M06', 42, 'IID', 1.0000, 100, 344000, 368050, 407850),
    ('M08', 42, 'IID', 0.9990, 100, 369200, 400600, 416850),
    ('M04', 42, 'OOD', 1.0000, 200, 236100, 254950, 326150),
]

# ============================================================================
# Figure 1: Capacity scaling — T_90 vs model size
# ============================================================================
def fig_capacity_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = list(CAPACITY_DATA.keys())
    p_total = [CAPACITY_DATA[m][0] for m in models]
    best_acc = [CAPACITY_DATA[m][3] for m in models]
    grokked = [CAPACITY_DATA[m][10] for m in models]
    t_90 = [CAPACITY_DATA[m][6] for m in models]
    train_frac = [CAPACITY_DATA[m][2] for m in models]

    colors = ['#2ecc71' if g else '#e74c3c' for g in grokked]
    ax1.bar(range(len(models)), [a*100 for a in best_acc], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45)
    ax1.set_ylabel('Best Test Accuracy (%)')
    ax1.set_title('Capacity Scaling: Best Accuracy (p=113)')
    ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='Grokking threshold')
    ax1.legend()
    # Annotate train fractions
    for i, tf in enumerate(train_frac):
        ax1.text(i, 2, f'{int(tf*100)}%', ha='center', fontsize=7, color='white', fontweight='bold')

    # T_90 vs params (log scale)
    grokked_models = [m for m in models if CAPACITY_DATA[m][10]]
    grokked_params = [CAPACITY_DATA[m][0] for m in grokked_models]
    grokked_t90 = [CAPACITY_DATA[m][6] for m in grokked_models]
    grokked_tf = [CAPACITY_DATA[m][2] for m in grokked_models]

    ax2.scatter(grokked_params, grokked_t90, c=['#2ecc71']*len(grokked_models), s=100, zorder=5, edgecolors='black')
    for i, m in enumerate(grokked_models):
        ax2.annotate(f'{m}\n({int(grokked_tf[i]*100)}%)', (grokked_params[i], grokked_t90[i]),
                     textcoords="offset points", xytext=(10, 5), fontsize=8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Total Parameters')
    ax2.set_ylabel('$T_{90}$ (epochs to 90% test acc)')
    ax2.set_title('Grokking Time vs Model Capacity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_capacity_scaling.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_capacity_scaling.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated fig_capacity_scaling.pdf')

# ============================================================================
# Figure 2: Capacity x Data phase diagram
# ============================================================================
def fig_phase_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))

    models = list(CAPACITY_DATA.keys())

    # Y-axis range on log scale: 50K to 20M
    y_min, y_max = 50000, 20000000
    log_min, log_max = np.log10(y_min), np.log10(y_max)
    log_span = log_max - log_min

    # Divide into three equal zones on log scale (30% each, with small gaps)
    third = log_span / 3.0
    red_top = 10 ** (log_min + third)
    yellow_top = 10 ** (log_min + 2 * third)

    # Shade regions — equal thirds
    ax.axhspan(y_min, red_top, alpha=0.12, color='red', label='Underfitting zone')
    ax.axhspan(red_top, yellow_top, alpha=0.12, color='gold', label='Transition zone')
    ax.axhspan(yellow_top, y_max, alpha=0.12, color='green', label='Grokking zone')

    # Plot each model with staggered annotations to avoid overlap
    # Alternate annotation positions: above-right, below-right, above-left, below-left
    offsets = [(10, 12), (10, -18), (-45, 12), (-45, -18)]
    for i, m in enumerate(models):
        d = CAPACITY_DATA[m]
        p, tf, acc, grok = d[0], d[2], d[3], d[10]
        color = '#2ecc71' if grok else '#e74c3c'
        marker = 'o' if grok else 'X'
        size = 220 if grok else 180
        ax.scatter(tf * 100, p, c=color, marker=marker, s=size, zorder=5,
                   edgecolors='black', linewidth=1.2)
        ox, oy = offsets[i % len(offsets)]
        ax.annotate(f'{m}\n({acc*100:.0f}%)', (tf * 100, p),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    # Add M05 local (40%) — grokked with more data
    ax.scatter(40, M05_LOCAL[0], c='#2ecc71', marker='s', s=180, zorder=5,
               edgecolors='black', linewidth=1.2)
    ax.annotate('M05\n(40%, 99.6%)', (40, M05_LOCAL[0]),
                textcoords="offset points", xytext=(10, -20),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    # Zone boundary labels
    ax.text(72, red_top * 0.9, 'Underfitting', fontsize=10, color='red',
            alpha=0.6, ha='right', va='top', style='italic')
    ax.text(72, (red_top + yellow_top) / 2, 'Transition', fontsize=10, color='darkgoldenrod',
            alpha=0.7, ha='right', va='center', style='italic')
    ax.text(72, yellow_top * 1.1, 'Grokking', fontsize=10, color='green',
            alpha=0.6, ha='right', va='bottom', style='italic')

    ax.set_yscale('log')
    ax.set_xlabel('Training Fraction (%)', fontsize=12)
    ax.set_ylabel('Total Parameters', fontsize=12)
    ax.set_title('Capacity × Data Phase Diagram (p=113)\nGreen = Grokked, Red = Failed', fontsize=13)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(28, 75)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_phase_diagram.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_phase_diagram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated fig_phase_diagram.pdf')

# ============================================================================
# Figure 3: Data sweep — RAW vs CRT-BOTH
# ============================================================================
def fig_data_sweep():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # RAW
    raw_fracs = [r[0] for r in RAW_FRAC_DATA]
    raw_acc = [r[1] * 100 for r in RAW_FRAC_DATA]
    raw_t90 = [r[3] for r in RAW_FRAC_DATA]
    raw_grok = [a > 90 for a in raw_acc]

    # Accuracy vs train fraction
    colors = ['#2ecc71' if g else '#e74c3c' for g in raw_grok]
    ax1.scatter([f*100 for f in raw_fracs], raw_acc, c=colors, s=120,
                zorder=5, edgecolors='black', linewidth=1)
    ax1.plot([f*100 for f in raw_fracs], raw_acc, 'k--', alpha=0.3, zorder=4)
    ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='Grokking threshold')
    ax1.set_xlabel('Training Fraction (%)')
    ax1.set_ylabel('Best Test Accuracy (%)')
    ax1.set_title('Data Scaling: Accuracy vs Train Fraction (M04, p=113)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # T_90 vs train fraction
    raw_t90_valid = [(f*100, t) for f, t in zip(raw_fracs, raw_t90) if t is not None]
    if raw_t90_valid:
        ax2.plot([x[0] for x in raw_t90_valid], [x[1] for x in raw_t90_valid],
                 'ro-', markersize=8, linewidth=2)
    ax2.axhline(y=100000, color='gray', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Training Fraction (%)')
    ax2.set_ylabel('$T_{90}$ (epochs)')
    ax2.set_yscale('log')
    ax2.set_title('Grokking Time vs Train Fraction (M04, p=113)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_data_sweep.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_data_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated fig_data_sweep.pdf')

# ============================================================================
# Figure 4: Prime-order DLP results
# ============================================================================
def fig_prime_order():
    fig, ax = plt.subplots(figsize=(10, 5))

    models = [f'{d[0]}/{d[1]}' for d in PRIME_ORDER_DATA]
    accs = [d[3] * 100 for d in PRIME_ORDER_DATA]
    splits = [d[2] for d in PRIME_ORDER_DATA]
    colors = ['#2ecc71' if s == 'IID' else '#3498db' for s in splits]

    ax.bar(range(len(models)), accs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45)
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Prime-Order DLP (order-113 subgroup of $\\mathbb{F}_{227}^*$)')
    ax.axhline(y=99, color='gray', linestyle='--', alpha=0.5, label='99% threshold')
    ax.legend()

    # Legend for split types
    green_patch = mpatches.Patch(color='#2ecc71', label='IID split')
    blue_patch = mpatches.Patch(color='#3498db', label='OOD split (disjoint generators)')
    ax.legend(handles=[green_patch, blue_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_prime_order.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_prime_order.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated fig_prime_order.pdf')


# ============================================================================
# Figure 6: Five-phase stepping board
# ============================================================================
def fig_five_phases():
    fig, ax = plt.subplots(figsize=(14, 6))

    phases = [
        {
            'num': 1, 'name': 'Initialization',
            'epoch': '0–90', 'train': '~1%', 'test': '~1%',
            'event': 'Random weights',
            'color': '#95a5a6',  # gray
        },
        {
            'num': 2, 'name': 'Memorization\n+ WD Ramp',
            'epoch': '90–4,090', 'train': '100%', 'test': '~1%',
            'event': 'Train acc jumps\nWD: 0.05→0.30',
            'color': '#e74c3c',  # red
        },
        {
            'num': 3, 'name': 'Memorization\nPlateau',
            'epoch': '4,090–22,650', 'train': '100%', 'test': '7–15%',
            'event': '‖w‖: 148→70\nWD at max (0.30)',
            'color': '#f39c12',  # orange
        },
        {
            'num': 4, 'name': 'Circuit\nFormation',
            'epoch': '22,650–33,320', 'train': '100%', 'test': '→99%',
            'event': 'Test acc rises\nSlingshots occur',
            'color': '#3498db',  # blue
        },
        {
            'num': 5, 'name': 'Cleanup',
            'epoch': '33,320–37,480', 'train': '→99.8%', 'test': '99.8%',
            'event': '‖w‖→30\nMemorization pruned',
            'color': '#2ecc71',  # green
        },
    ]

    n = len(phases)
    step_width = 1.0
    step_heights = [0.15, 0.30, 0.45, 0.65, 0.85]  # increasing steps

    for i, ph in enumerate(phases):
        x = i * step_width
        h = step_heights[i]

        # Draw the step (rectangle)
        rect = plt.Rectangle((x, 0), step_width * 0.92, h,
                             facecolor=ph['color'], alpha=0.3,
                             edgecolor=ph['color'], linewidth=2)
        ax.add_patch(rect)

        # Draw step top line (thicker)
        ax.plot([x, x + step_width * 0.92], [h, h],
                color=ph['color'], linewidth=3, solid_capstyle='round')

        # Phase number in a circle
        circle = plt.Circle((x + step_width * 0.46, h + 0.06), 0.04,
                            color=ph['color'], zorder=5)
        ax.add_patch(circle)
        ax.text(x + step_width * 0.46, h + 0.06, str(ph['num']),
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white', zorder=6)

        # Phase name (above step)
        ax.text(x + step_width * 0.46, h + 0.13, ph['name'],
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=ph['color'])

        # Epoch range (on the step)
        ax.text(x + step_width * 0.46, h * 0.5, ph['epoch'],
                ha='center', va='center', fontsize=9, color='#333333',
                style='italic')

        # Train/Test accuracy (below step, inside)
        ax.text(x + step_width * 0.46, h * 0.25,
                f"Train: {ph['train']}\nTest: {ph['test']}",
                ha='center', va='center', fontsize=8, color='#555555')

        # Key event (below the step)
        ax.text(x + step_width * 0.46, -0.08, ph['event'],
                ha='center', va='top', fontsize=8, color='#444444',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=ph['color'], alpha=0.8))

    # Connect steps with arrows
    for i in range(n - 1):
        x1 = i * step_width + step_width * 0.92
        x2 = (i + 1) * step_width
        h1 = step_heights[i]
        h2 = step_heights[i + 1]
        ax.annotate('', xy=(x2 + step_width * 0.46, h2 + 0.02),
                    xytext=(x1, h1),
                    arrowprops=dict(arrowstyle='->', color='#777777',
                                  lw=1.5, connectionstyle='arc3,rad=0.2'))

    # Axis labels
    ax.set_xlim(-0.1, n * step_width + 0.1)
    ax.set_ylim(-0.20, 1.15)
    ax.set_aspect('equal')

    # Epoch axis at bottom
    ax.set_xlabel('Training Progress (epochs)', fontsize=12, labelpad=30)

    # Remove default axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title('Five-Phase Training Dynamics of DLP Grokking (M04, p=113)',
                 fontsize=13, fontweight='bold', pad=15)

    # Add accuracy scale on the right
    acc_labels = ['0%', '25%', '50%', '75%', '100%']
    acc_positions = [0.0, 0.25, 0.50, 0.75, 1.0]
    for label, pos in zip(acc_labels, acc_positions):
        ax.text(n * step_width + 0.15, pos, label,
                ha='left', va='center', fontsize=8, color='#888888')
    ax.text(n * step_width + 0.15, 0.5, 'Accuracy',
            ha='left', va='center', fontsize=9, color='#666666',
            rotation=90, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_five_phases.pdf'),
               dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_five_phases.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated fig_five_phases.pdf')


if __name__ == '__main__':
    fig_capacity_scaling()
    fig_phase_diagram()
    fig_data_sweep()
    fig_prime_order()
    fig_five_phases()
    print('\nAll figures generated in', FIGURES_DIR)
