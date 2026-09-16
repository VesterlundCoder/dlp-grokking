#!/usr/bin/env python3
"""Generate figures from LUMI DLP grokking results.

Creates:
1. Capacity scaling plot (accuracy vs model size, showing grokking valley)
2. CRT variant comparison (grokking speed bar chart)
3. Prime-order DLP results
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LUMI_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lumi_results")

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def parse_summaries():
    with open(os.path.join(LUMI_RESULTS, "all_summaries.txt")) as f:
        text = f.read()
    summaries = {}
    current_name = None
    current_json = []
    for line in text.strip().split("\n"):
        if line.startswith("---") and line.endswith("---"):
            if current_name and current_json:
                try:
                    summaries[current_name] = json.loads("\n".join(current_json))
                except json.JSONDecodeError:
                    pass
            current_name = line.strip("---")
            current_json = []
        elif current_name:
            current_json.append(line)
    if current_name and current_json:
        try:
            summaries[current_name] = json.loads("\n".join(current_json))
        except json.JSONDecodeError:
            pass
    return summaries


def fig_capacity_scaling(summaries):
    """Plot accuracy vs model size, showing the grokking valley."""
    models = []
    for name, d in summaries.items():
        if "p113" in name and name.startswith("M0") and not name.startswith("M13"):
            model_id = name.split("_")[0]
            models.append({
                "model": model_id,
                "params": d["n_params"],
                "acc": d["best_test_acc"],
                "epochs": d["total_epochs"],
                "train_frac": d.get("train_size", 0) / (d.get("train_size", 0) + d.get("test_size", 1)),
            })

    models.sort(key=lambda x: x["params"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: accuracy vs params
    params = [m["params"] / 1e6 for m in models]
    accs = [m["acc"] * 100 for m in models]
    colors = ["#4CAF50" if a >= 80 else "#F44336" for a in accs]

    bars = ax1.bar(range(len(models)), accs, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m["model"] for m in models], rotation=45)
    ax1.set_ylabel("Best Test Accuracy (%)")
    ax1.set_title("Capacity Scaling: DLP Grokking (p=113)")
    ax1.axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="Grokking threshold")
    ax1.legend()

    # Add params annotation
    for i, m in enumerate(models):
        ax1.annotate(f"{m['params']/1e6:.1f}M", (i, accs[i]),
                     textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=7, color="gray")

    # Right: grokking time vs params (only grokked models)
    grokked = [m for m in models if m["acc"] >= 0.80]
    if grokked:
        g_params = [m["params"] / 1e6 for m in grokked]
        g_epochs = [m["epochs"] / 1000 for m in grokked]
        ax2.bar(range(len(grokked)), g_epochs, color="#2196F3", edgecolor="black", linewidth=0.5)
        ax2.set_xticks(range(len(grokked)))
        ax2.set_xticklabels([m["model"] for m in grokked], rotation=45)
        ax2.set_ylabel("Epochs to Grok (thousands)")
        ax2.set_title("Grokking Time vs Model Size")
        for i, m in enumerate(grokked):
            ax2.annotate(f"{m['epochs']/1000:.0f}k", (i, g_epochs[i]),
                         textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "lumi_capacity_scaling.pdf")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig_crt_comparison(summaries):
    """Bar chart comparing CRT variants."""
    variants = {}
    for name, d in summaries.items():
        v = d.get("variant", name.split("_")[0])
        if v not in variants:
            variants[v] = []
        variants[v].append({
            "acc": d["best_test_acc"] * 100,
            "epochs": d["total_epochs"],
            "seed": d.get("seed", 42),
        })

    # Order: RAW, CRT-BOTH, CRT-16, CRT-7, RAW+CRT, SCRAMBLED
    order = ["RAW", "CRT-BOTH", "CRT-16", "CRT-7", "RAW+CRT", "SCRAMBLED"]
    ordered = [(v, variants[v]) for v in order if v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: accuracy
    names = [v for v, _ in ordered]
    mean_accs = [np.mean([r["acc"] for r in runs]) for _, runs in ordered]
    colors = ["#F44336", "#4CAF50", "#FF9800", "#FF9800", "#2196F3", "#9C27B0"]

    bars = ax1.bar(range(len(names)), mean_accs, color=colors[:len(names)], edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Best Test Accuracy (%)")
    ax1.set_title("CRT Representation Variants (p=113, M04)")
    ax1.axhline(y=80, color="gray", linestyle="--", alpha=0.5)

    # Add ceiling lines
    ceilings = {"CRT-16": 14.3, "CRT-7": 6.25}
    for i, (v, _) in enumerate(ordered):
        if v in ceilings:
            ax1.plot([i-0.3, i+0.3], [ceilings[v], ceilings[v]], "r--", linewidth=2)
            ax1.annotate(f"ceiling={ceilings[v]}%", (i, ceilings[v]),
                         textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=7, color="red")

    # Right: grokking time (log scale)
    grok_times = []
    for v, runs in ordered:
        times = [r["epochs"] for r in runs]
        grok_times.append(np.mean(times))

    ax2.bar(range(len(names)), grok_times, color=colors[:len(names)], edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("Epochs to Grok (mean)")
    ax2.set_title("Grokking Speed by Representation")
    ax2.set_yscale("log")
    for i, t in enumerate(grok_times):
        ax2.annotate(f"{t:,.0f}", (i, t),
                     textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "lumi_crt_comparison.pdf")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig_prime_order(summaries):
    """Prime-order DLP results."""
    po = []
    for name, d in summaries.items():
        if name.startswith("P113"):
            po.append({
                "name": name,
                "model": name.split("_")[1],
                "split": "OOD" if "ood" in name else "IID",
                "seed": d["seed"],
                "acc": d["best_test_acc"] * 100,
                "epochs": d["total_epochs"],
            })

    po.sort(key=lambda x: (x["split"], x["model"], x["seed"]))

    fig, ax = plt.subplots(figsize=(8, 5))

    names = [f"{p['model']}\n{p['split']}\ns{p['seed']}" for p in po]
    accs = [p["acc"] for p in po]
    colors = ["#4CAF50" if p["split"] == "IID" else "#2196F3" for p in po]

    ax.bar(range(len(po)), accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(po)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("Prime-Order DLP (order-113 subgroup of F_227*)")
    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.5)

    for i, p in enumerate(po):
        ax.annotate(f"{p['acc']:.1f}%", (i, p["acc"]),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4CAF50", label="IID split"),
                       Patch(facecolor="#2196F3", label="OOD split")]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "lumi_prime_order.pdf")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def main():
    summaries = parse_summaries()
    print(f"Parsed {len(summaries)} summaries")

    fig_capacity_scaling(summaries)
    fig_crt_comparison(summaries)
    fig_prime_order(summaries)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
