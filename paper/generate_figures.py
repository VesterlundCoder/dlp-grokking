#!/usr/bin/env python3
"""Generate all figures for the DLP Grokking preprint from experiment data.

Usage:
    cd /Users/davidsvensson/Desktop/dlp_grokking
    python3 paper/generate_figures.py
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PAPER_DIR)
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Color palette
C_TRAIN = "#2196F3"
C_TEST = "#FF5722"
C_WD = "#4CAF50"
C_WNORM = "#9C27B0"
C_COMP = "#FF9800"
C_CYCLIC = "#795548"
C_INV = "#607D8B"
C_GROK = "#E91E63"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_metrics(result_dir):
    """Load training metrics from results directory."""
    path = os.path.join(result_dir, "metrics.jsonl")
    data = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("train_acc") is not None:
                data.append(d)
    return data


def load_probe_results(result_dir):
    """Load probe results."""
    path = os.path.join(result_dir, "probe_results.jsonl")
    results = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                results.append(json.loads(line))
    return results


def load_fourier_results(result_dir):
    """Load Fourier analysis results."""
    path = os.path.join(result_dir, "fourier_results.jsonl")
    results = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                results.append(json.loads(line))
    return results


# ============================================================================
# Figure 1: Training Curves (M04)
# ============================================================================

def fig1_training_curves():
    """Training/test accuracy + WD + weight norm over epochs."""
    m04 = load_metrics(os.path.join(PROJECT_DIR, "results/local_probe/M04_s42"))
    epochs = [d["epoch"] for d in m04]
    train = [d["train_acc"] for d in m04]
    test = [d["test_acc"] for d in m04]
    wd = [d["weight_decay"] for d in m04]
    wnorm = [d["weight_norm"] for d in m04]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # Top: accuracy
    ax1.plot(epochs, train, color=C_TRAIN, alpha=0.8, linewidth=0.8, label="Train")
    ax1.plot(epochs, test, color=C_TEST, alpha=0.8, linewidth=0.8, label="Test")
    ax1.axhline(0.99, color="gray", linestyle="--", alpha=0.4)
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="center right")
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_title("M04 (427k params, $p=113$, train\\_frac=0.30)")

    # Mark key milestones
    t_mem = 90
    t_90 = 26830
    t_99 = 33320
    for t, label in [(t_mem, "$T_{\\mathrm{mem}}$"), (t_90, "$T_{90}$"), (t_99, "$T_{99}$")]:
        ax1.axvline(t, color=C_GROK, linestyle="--", alpha=0.3, linewidth=0.8)
        ax1.text(t, 0.05, label, fontsize=8, color=C_GROK, ha="center")

    # Bottom: WD and weight norm
    ax2.plot(epochs, wd, color=C_WD, linewidth=0.8, label="WD")
    ax2_r = ax2.twinx()
    ax2_r.plot(epochs, wnorm, color=C_WNORM, linewidth=0.8, label="||w||")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Weight Decay", color=C_WD)
    ax2_r.set_ylabel("Weight Norm", color=C_WNORM)
    ax2.tick_params(axis="y", labelcolor=C_WD)
    ax2_r.tick_params(axis="y", labelcolor=C_WNORM)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_training_curves.pdf"))
    plt.close()
    print("  fig1_training_curves.pdf")


# ============================================================================
# Figure 2: Slingshot Events
# ============================================================================

def fig2_slingshots():
    """Zoom into slingshot events around grokking onset."""
    m04 = load_metrics(os.path.join(PROJECT_DIR, "results/local_probe/M04_s42"))

    # Focus on epochs 20k-35k (grokking transition)
    zoom = [d for d in m04 if 20000 <= d["epoch"] <= 35000]
    epochs = [d["epoch"] for d in zoom]
    train = [d["train_acc"] for d in zoom]
    test = [d["test_acc"] for d in zoom]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(epochs, train, color=C_TRAIN, linewidth=0.8, alpha=0.8, label="Train")
    ax.plot(epochs, test, color=C_TEST, linewidth=0.8, alpha=0.8, label="Test")

    # Mark slingshots
    slingshots = [
        (21020, 0.984), (23100, 0.951), (24780, 0.362),
        (26780, 0.849), (29070, 0.977), (31630, 0.974),
        (33520, 0.397), (35830, 0.444)
    ]
    for ep, drop in slingshots:
        ax.axvline(ep, color="red", alpha=0.15, linewidth=0.5)

    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Slingshot Events During Grokking Transition (M04)")
    ax.legend()
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_slingshots.pdf"))
    plt.close()
    print("  fig2_slingshots.pdf")


# ============================================================================
# Figure 3: Group Structure Emergence
# ============================================================================

def fig3_group_structure():
    """Group structure tests across checkpoints overlaid with test accuracy."""
    probes = load_probe_results(os.path.join(PROJECT_DIR, "results/local_probe/M04_s42"))
    if not probes:
        print("  fig3_group_structure.pdf (SKIPPED — no probe results yet)")
        return

    epochs = [p["epoch"] for p in probes]
    test = [p["test_accuracy"] for p in probes]
    comp = [p.get("composition_consistency", 0) for p in probes]
    cyclic = [p.get("cyclic_shift_accuracy", 0) for p in probes]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epochs, test, color=C_TEST, linewidth=1.5, label="Test Accuracy", zorder=3)
    ax.plot(epochs, comp, color=C_COMP, linewidth=1.2, label="Composition Consistency", alpha=0.8)
    ax.plot(epochs, cyclic, color=C_CYCLIC, linewidth=1.2, label="Cyclic Shift Accuracy", alpha=0.8)

    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy / Consistency")
    ax.set_title("Group Structure Emergence vs. Test Accuracy (M04)")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_group_structure.pdf"))
    plt.close()
    print("  fig3_group_structure.pdf")


# ============================================================================
# Figure 4: Fourier Spectrum Evolution
# ============================================================================

def fig4_fourier_spectrum():
    """Fourier spectrum at pre-grokking vs post-grokking checkpoints."""
    fourier = load_fourier_results(os.path.join(PROJECT_DIR, "results/local_probe/M04_s42"))
    if not fourier:
        print("  fig4_fourier_spectrum.pdf (SKIPPED — no Fourier results yet)")
        return

    # Find pre-grokking (test < 0.3) and post-grokking (test > 0.9) checkpoints
    probes = load_probe_results(os.path.join(PROJECT_DIR, "results/local_probe/M04_s42"))

    pre_idx = None
    post_idx = None
    for i, p in enumerate(probes):
        if p["test_accuracy"] < 0.3 and pre_idx is None:
            pre_idx = i
        if p["test_accuracy"] > 0.9:
            post_idx = i
            break

    if pre_idx is None or post_idx is None:
        print("  fig4_fourier_spectrum.pdf (SKIPPED — insufficient data)")
        return

    # Get layer data (prefer block_1)
    layer_name = "block_1"
    if layer_name not in fourier[pre_idx].get("layers", {}):
        layer_name = "unembed"
    if layer_name not in fourier[pre_idx].get("layers", {}):
        layer_name = list(fourier[pre_idx]["layers"].keys())[0]

    pre_mult = fourier[pre_idx]["layers"][layer_name]["mult_spectrum"]
    post_mult = fourier[post_idx]["layers"][layer_name]["mult_spectrum"]
    pre_add = fourier[pre_idx]["layers"][layer_name]["add_spectrum"]
    post_add = fourier[post_idx]["layers"][layer_name]["add_spectrum"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    freqs_mult = np.arange(len(pre_mult))
    freqs_add = np.arange(len(pre_add))

    # Multiplicative spectrum
    ax1.bar(freqs_mult, pre_mult, color="gray", alpha=0.4, label=f"Pre-grok (ep {probes[pre_idx]['epoch']})")
    ax1.bar(freqs_mult, post_mult, color=C_GROK, alpha=0.6, label=f"Post-grok (ep {probes[post_idx]['epoch']})")
    ax1.set_xlabel("Multiplicative Frequency $k$")
    ax1.set_ylabel("Energy")
    ax1.set_title("Multiplicative Character Spectrum")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=1e-6)

    # Additive spectrum
    ax2.bar(freqs_add, pre_add, color="gray", alpha=0.4, label=f"Pre-grok (ep {probes[pre_idx]['epoch']})")
    ax2.bar(freqs_add, post_add, color=C_GROK, alpha=0.6, label=f"Post-grok (ep {probes[post_idx]['epoch']})")
    ax2.set_xlabel("Additive Frequency $k$")
    ax2.set_ylabel("Energy")
    ax2.set_title("Additive Character Spectrum")
    ax2.legend(fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1e-6)

    plt.suptitle(f"Fourier Spectrum: Pre- vs Post-Grokking ({layer_name})", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_fourier_spectrum.pdf"))
    plt.close()
    print("  fig4_fourier_spectrum.pdf")


# ============================================================================
# Figure 5: Fourier Participation Ratio Evolution
# ============================================================================

def fig5_fourier_pr_evolution():
    """Participation ratio and concentration over training."""
    fourier = load_fourier_results(os.path.join(PROJECT_DIR, "results/local_probe/M04_s42"))
    if not fourier:
        print("  fig5_fourier_pr_evolution.pdf (SKIPPED — no Fourier results yet)")
        return

    epochs = [r["epoch"] for r in fourier]
    layer_name = "block_1"
    if layer_name not in fourier[0].get("layers", {}):
        layer_name = list(fourier[0]["layers"].keys())[0]

    mult_pr = [r["layers"][layer_name]["mult_pr"] for r in fourier]
    add_pr = [r["layers"][layer_name]["add_pr"] for r in fourier]
    mult_conc = [r["layers"][layer_name]["mult_concentration_4"] for r in fourier]
    add_conc = [r["layers"][layer_name]["add_concentration_4"] for r in fourier]

    # Load test accuracy from probes
    probes = load_probe_results(os.path.join(PROJECT_DIR, "results/local_probe/M04_s42"))
    test_epochs = [p["epoch"] for p in probes]
    test_acc = [p["test_accuracy"] for p in probes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Top: PR
    ax1.plot(epochs, mult_pr, color=C_GROK, linewidth=1.2, label="Mult PR")
    ax1.plot(epochs, add_pr, color=C_TRAIN, linewidth=1.2, label="Add PR")
    ax1_r = ax1.twinx()
    ax1_r.plot(test_epochs, test_acc, color=C_TEST, alpha=0.4, linewidth=0.8, label="Test Acc")
    ax1.set_ylabel("Participation Ratio")
    ax1_r.set_ylabel("Test Accuracy", color=C_TEST, alpha=0.6)
    ax1.legend(loc="upper left")
    ax1.set_title(f"Fourier Spectrum Sparsity Evolution ({layer_name})")

    # Bottom: Concentration
    ax2.plot(epochs, mult_conc, color=C_GROK, linewidth=1.2, label="Mult Conc-4")
    ax2.plot(epochs, add_conc, color=C_TRAIN, linewidth=1.2, label="Add Conc-4")
    ax2_r = ax2.twinx()
    ax2_r.plot(test_epochs, test_acc, color=C_TEST, alpha=0.4, linewidth=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-4 Energy Fraction")
    ax2_r.set_ylabel("Test Accuracy", color=C_TEST, alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_fourier_pr_evolution.pdf"))
    plt.close()
    print("  fig5_fourier_pr_evolution.pdf")


# ============================================================================
# Figure 6: WD Schedule and Weight Norm
# ============================================================================

def fig6_wd_schedule():
    """WD schedule and weight norm trajectory."""
    m04 = load_metrics(os.path.join(PROJECT_DIR, "results/local_probe/M04_s42"))
    epochs = [d["epoch"] for d in m04]
    wd = [d["weight_decay"] for d in m04]
    wnorm = [d["weight_norm"] for d in m04]

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    ax1.plot(epochs, wd, color=C_WD, linewidth=1.2, label="Weight Decay")
    ax1_r = ax1.twinx()
    ax1_r.plot(epochs, wnorm, color=C_WNORM, linewidth=1.2, label="Weight Norm")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weight Decay", color=C_WD)
    ax1_r.set_ylabel("Weight Norm ||w||", color=C_WNORM)
    ax1.tick_params(axis="y", labelcolor=C_WD)
    ax1_r.tick_params(axis="y", labelcolor=C_WNORM)
    ax1.set_title("Progressive WD Schedule and Weight Norm Trajectory (M04)")

    # Mark WD ramp events
    wd_changes = [(90, 0.1), (1090, 0.15), (2090, 0.2), (3090, 0.25), (4090, 0.3)]
    for ep, val in wd_changes:
        ax1.axvline(ep, color="gray", linestyle=":", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_wd_schedule.pdf"))
    plt.close()
    print("  fig6_wd_schedule.pdf")


# ============================================================================
# Figure 7: Grokking Phase Diagram
# ============================================================================

def fig7_phase_diagram():
    """Schematic phase diagram: train/test accuracy with labeled phases."""
    m04 = load_metrics(os.path.join(PROJECT_DIR, "results/local_probe/M04_s42"))
    epochs = [d["epoch"] for d in m04]
    train = [d["train_acc"] for d in m04]
    test = [d["test_acc"] for d in m04]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epochs, train, color=C_TRAIN, linewidth=0.8, alpha=0.8, label="Train")
    ax.plot(epochs, test, color=C_TEST, linewidth=0.8, alpha=0.8, label="Test")

    # Phase regions
    ax.axvspan(0, 90, alpha=0.05, color="blue", label="_")
    ax.axvspan(90, 4090, alpha=0.05, color="orange", label="_")
    ax.axvspan(4090, 22650, alpha=0.05, color="red", label="_")
    ax.axvspan(22650, 33320, alpha=0.05, color="green", label="_")
    ax.axvspan(33320, 37480, alpha=0.05, color="purple", label="_")

    # Phase labels
    phases = [
        (45, "Init", "blue"),
        (2000, "Memorization\n+ WD Ramp", "orange"),
        (13000, "Memorization\nPlateau", "red"),
        (28000, "Circuit\nFormation", "green"),
        (35500, "Cleanup", "purple"),
    ]
    for x, label, color in phases:
        ax.text(x, 1.02, label, fontsize=8, ha="center", color=color, fontweight="bold")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Five Phases of DLP Grokking (M04, $p=113$)")
    ax.legend(loc="center right")
    ax.set_ylim(-0.02, 1.12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_phase_diagram.pdf"))
    plt.close()
    print("  fig7_phase_diagram.pdf")


# ============================================================================
# Figure 8: Model Grid Table (as figure)
# ============================================================================

def fig8_model_grid():
    """Model grid table as a figure."""
    models = [
        ("M01", 56, 2, 13946, 8736),
        ("M02", 80, 4, 28426, 23104),
        ("M03", 104, 4, 48250, 43872),
        ("M04", 128, 4, 427253, 396544),
        ("M05", 160, 5, 656917, 618560),
        ("M06", 208, 8, 1116169, 1069056),
        ("M07", 256, 8, 1692993, 1632256),
        ("M08", 320, 10, 2632705, 2555904),
        ("M09", 384, 12, 3794241, 3695872),
        ("M10", 480, 15, 5938241, 5812224),
        ("M11", 608, 19, 9518401, 9371648),
        ("M12", 768, 24, 15197953, 14942208),
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    col_labels = ["Model", "$d_{\\mathrm{model}}$", "Heads", "Params", "$P_{\\mathrm{core}}$", "State"]
    rows = []
    for m in models:
        rows.append([m[0], str(m[1]), str(m[2]), f"{m[3]:,}", f"{m[4]:,}", "—"])

    # Mark M04 as grokked
    for r in rows:
        if r[0] == "M04":
            r[5] = "Grokked"
        if r[0] == "M05":
            r[5] = "Running"

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colWidths=[0.12, 0.15, 0.1, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color M04 row
    for j in range(len(col_labels)):
        if rows[3][5] == "Grokked ✓":
            table[4, j].set_facecolor("#C8E6C9")

    plt.title("Model Grid: Transformer Scaling Sweep (L=2, $p=113$)", fontsize=13, pad=20)
    plt.savefig(os.path.join(FIG_DIR, "fig8_model_grid.pdf"))
    plt.close()
    print("  fig8_model_grid.pdf")


# ============================================================================
# Figure 9: Grokking Ladder (schematic)
# ============================================================================

def fig9_grokking_ladder():
    """Grokking ladder: which problems grok and which don't."""
    problems = [
        ("L0\nIdentity", True, 1.0),
        ("L1\nPermutation", True, 1.0),
        ("L2\nAddition", True, 1.0),
        ("L3\nMultiplication", True, 1.0),
        ("L4\nExponent", False, 0.08),
        ("L5\nPermutation\nPower", False, 0.08),
        ("L6\nDLP", True, 0.998),
        ("L7\nMDLP", False, 0.08),
        ("MADD\nMulti-add", True, 1.0),
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    names = [p[0] for p in problems]
    grokked = [p[1] for p in problems]
    accs = [p[2] for p in problems]
    colors = ["#4CAF50" if g else "#F44336" for g in grokked]

    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Grokking Ladder: Which Operations Grok ($p=113$, M04-scale)")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.3)

    # Add check/cross marks
    for i, g in enumerate(grokked):
        symbol = "Y" if g else "N"
        ax.text(i, accs[i] + 0.03, symbol, ha="center", fontsize=14,
                color=colors[i], fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig9_grokking_ladder.pdf"))
    plt.close()
    print("  fig9_grokking_ladder.pdf")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Generating figures for DLP Grokking preprint...")
    print(f"  Output: {FIG_DIR}")
    print()

    fig1_training_curves()
    fig2_slingshots()
    fig3_group_structure()
    fig4_fourier_spectrum()
    fig5_fourier_pr_evolution()
    fig6_wd_schedule()
    fig7_phase_diagram()
    fig8_model_grid()
    fig9_grokking_ladder()

    print()
    print("Done. Figures saved to:", FIG_DIR)
