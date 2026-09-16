#!/usr/bin/env python3
"""Generate comparison charts for the DLP grokking experiments.

Produces:
1. Algebraic-pair comparison: test accuracy over epochs for all 6 variants
2. Prime-order vs composite-order: P113-A vs original M04 (q=112)
3. Component accuracy breakdown (when available)
4. Combined overview figure
"""
import json
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALG_DIR = os.path.join(EXPERIMENTS_DIR, "algebraic_pairs")
PRIME_DIR = os.path.join(EXPERIMENTS_DIR, "prime113")
M04_METRICS = os.path.join(EXPERIMENTS_DIR, "results", "local_probe", "M04_s42", "metrics.jsonl")

FIGURES_DIR = os.path.join(EXPERIMENTS_DIR, "experiments", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

VARIANT_COLORS = {
    "RAW":       "#1f77b4",
    "CRT-BOTH":  "#2ca02c",
    "CRT-16":    "#ff7f0e",
    "CRT-7":     "#d62728",
    "RAW+CRT":   "#9467bd",
    "SCRAMBLED": "#8c564b",
}

VARIANT_LABELS = {
    "RAW":       "RAW (original M04)",
    "CRT-BOTH":  "CRT-BOTH (g¹⁶, g⁷, h¹⁶, h⁷)",
    "CRT-16":    "CRT-16 only (order-16 component)",
    "CRT-7":     "CRT-7 only (order-7 component)",
    "RAW+CRT":   "RAW + CRT (full info)",
    "SCRAMBLED": "Scrambled pair (control)",
}


def load_metrics(path):
    """Load metrics.jsonl into arrays."""
    epochs, train_acc, test_acc, loss = [], [], [], []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            try:
                m = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            epochs.append(m["epoch"])
            train_acc.append(m.get("train_acc") or 0.0)
            test_acc.append(m.get("test_acc") or 0.0)
            loss.append(m.get("loss") or 0.0)
    if not epochs:
        return None
    return {
        "epoch": np.array(epochs),
        "train_acc": np.array(train_acc),
        "test_acc": np.array(test_acc),
        "loss": np.array(loss),
    }


def smooth(arr, window=5):
    """Simple moving average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


# ============================================================================
# Figure 1: Algebraic-Pair Comparison — Test Accuracy
# ============================================================================

def plot_algebraic_pairs():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: Test accuracy (log scale) ---
    ax = axes[0]
    for variant in ["RAW", "CRT-BOTH", "CRT-16", "CRT-7", "RAW+CRT", "SCRAMBLED"]:
        metrics_path = os.path.join(ALG_DIR, f"{variant}_s42", "metrics.jsonl")
        data = load_metrics(metrics_path)
        if data is None:
            continue
        ax.plot(data["epoch"], smooth(data["test_acc"], 3),
                color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant],
                linewidth=1.5, alpha=0.9)
        # Mark memorization point
        mem_mask = data["train_acc"] > 0.99
        if mem_mask.any():
            mem_epoch = data["epoch"][mem_mask][0]
            ax.axvline(mem_epoch, color=VARIANT_COLORS[variant],
                       linestyle=":", alpha=0.3, linewidth=1)

    # Theoretical ceilings
    ax.axhline(1/7, color="#ff7f0e", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(0.98, 1/7 + 0.005, "CRT-16 ceiling (1/7 ≈ 14.3%)",
            ha="right", va="bottom", fontsize=9, color="#ff7f0e", alpha=0.7)
    ax.axhline(1/16, color="#d62728", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(0.98, 1/16 + 0.003, "CRT-7 ceiling (1/16 ≈ 6.25%)",
            ha="right", va="bottom", fontsize=9, color="#d62728", alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Algebraic-Pair Representation: Test Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_yscale("logit" if ax.get_ylim()[1] <= 1 else "linear")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # --- Right: Train accuracy ---
    ax = axes[1]
    for variant in ["RAW", "CRT-BOTH", "CRT-16", "CRT-7", "RAW+CRT", "SCRAMBLED"]:
        metrics_path = os.path.join(ALG_DIR, f"{variant}_s42", "metrics.jsonl")
        data = load_metrics(metrics_path)
        if data is None:
            continue
        ax.plot(data["epoch"], smooth(data["train_acc"], 3),
                color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant],
                linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Accuracy")
    ax.set_title("Algebraic-Pair Representation: Train Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 2: Does CRT-Aligned Representation Accelerate Grokking?",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "algebraic_pairs_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Figure 2: Prime-Order vs Composite-Order
# ============================================================================

def plot_prime_vs_composite():
    fig, ax = plt.subplots(figsize=(10, 6))

    # P113-A (prime order)
    p113_data = load_metrics(os.path.join(PRIME_DIR, "P113-A_s42", "metrics.jsonl"))
    if p113_data is not None:
        ax.plot(p113_data["epoch"], smooth(p113_data["test_acc"], 5),
                color="#e377c2", label="P113-A: Prime order r=113 in F_227*",
                linewidth=2)

    # P113-B if available
    p113b_data = load_metrics(os.path.join(PRIME_DIR, "P113-B_s42", "metrics.jsonl"))
    if p113b_data is not None:
        ax.plot(p113b_data["epoch"], smooth(p113b_data["test_acc"], 5),
                color="#17becf", label="P113-B: Prime order (sample-matched)",
                linewidth=2, linestyle="--")

    # Original M04 (composite order q=112)
    m04_data = load_metrics(M04_METRICS)
    if m04_data is not None:
        ax.plot(m04_data["epoch"], smooth(m04_data["test_acc"], 5),
                color="#1f77b4", label="M04: Composite order q=112 = 16×7 in F_113*",
                linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Experiment 1: Prime-Order vs Composite-Order DLP Grokking")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(0.98, 0.02, "q=112: CRT decomposition 16×7\nr=113: prime (no decomposition)",
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "prime_vs_composite.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Figure 3: RAW vs CRT-BOTH — Direct Comparison
# ============================================================================

def plot_raw_vs_crt():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Test accuracy ---
    ax = axes[0]
    for variant in ["RAW", "CRT-BOTH"]:
        data = load_metrics(os.path.join(ALG_DIR, f"{variant}_s42", "metrics.jsonl"))
        if data is None:
            continue
        ax.plot(data["epoch"], smooth(data["test_acc"], 3),
                color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant],
                linewidth=2)
        # Mark grokking onset (test > 10%)
        grok_mask = data["test_acc"] > 0.10
        if grok_mask.any():
            grok_epoch = data["epoch"][grok_mask][0]
            ax.axvline(grok_epoch, color=VARIANT_COLORS[variant],
                       linestyle=":", alpha=0.5, linewidth=1.5)
            ax.annotate(f"grok @ {grok_epoch}", xy=(grok_epoch, 0.10),
                        fontsize=9, color=VARIANT_COLORS[variant])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("RAW vs CRT-BOTH: Test Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Loss ---
    ax = axes[1]
    for variant in ["RAW", "CRT-BOTH"]:
        data = load_metrics(os.path.join(ALG_DIR, f"{variant}_s42", "metrics.jsonl"))
        if data is None:
            continue
        ax.plot(data["epoch"], smooth(data["loss"], 3),
                color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant],
                linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("RAW vs CRT-BOTH: Training Loss")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Representation Matters: CRT Decomposition Accelerates Grokking",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "raw_vs_crt_both.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Figure 4: Combined Overview
# ============================================================================

def plot_overview():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- (0,0): All algebraic variants, test acc ---
    ax = axes[0, 0]
    for variant in ["RAW", "CRT-BOTH", "CRT-16", "CRT-7", "RAW+CRT", "SCRAMBLED"]:
        data = load_metrics(os.path.join(ALG_DIR, f"{variant}_s42", "metrics.jsonl"))
        if data is None:
            continue
        ax.plot(data["epoch"], smooth(data["test_acc"], 3),
                color=VARIANT_COLORS[variant], label=variant,
                linewidth=1.5)
    ax.axhline(1/7, color="#ff7f0e", linestyle="--", alpha=0.3)
    ax.axhline(1/16, color="#d62728", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Algebraic Pairs: All Variants")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (0,1): Prime vs composite ---
    ax = axes[0, 1]
    p113_data = load_metrics(os.path.join(PRIME_DIR, "P113-A_s42", "metrics.jsonl"))
    if p113_data is not None:
        ax.plot(p113_data["epoch"], smooth(p113_data["test_acc"], 5),
                color="#e377c2", label="P113-A (prime order 113)", linewidth=2)
    m04_data = load_metrics(M04_METRICS)
    if m04_data is not None:
        ax.plot(m04_data["epoch"], smooth(m04_data["test_acc"], 5),
                color="#1f77b4", label="M04 (composite order 112)", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Prime vs Composite Order")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,0): RAW vs CRT-BOTH detail ---
    ax = axes[1, 0]
    for variant in ["RAW", "CRT-BOTH"]:
        data = load_metrics(os.path.join(ALG_DIR, f"{variant}_s42", "metrics.jsonl"))
        if data is None:
            continue
        ax.plot(data["epoch"], smooth(data["test_acc"], 3),
                color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant],
                linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("RAW vs CRT-BOTH: Direct Comparison")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,1): Loss comparison ---
    ax = axes[1, 1]
    for variant in ["RAW", "CRT-BOTH", "CRT-16", "CRT-7"]:
        data = load_metrics(os.path.join(ALG_DIR, f"{variant}_s42", "metrics.jsonl"))
        if data is None:
            continue
        ax.plot(data["epoch"], smooth(data["loss"], 3),
                color=VARIANT_COLORS[variant], label=variant,
                linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Comparison")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("DLP Grokking Experiments: Representation and Group Structure",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "overview.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Generating comparison charts...")
    plot_algebraic_pairs()
    plot_prime_vs_composite()
    plot_raw_vs_crt()
    plot_overview()
    print("Done.")
