#!/usr/bin/env python3
"""Generate the train_frac ablation table for the paper.

Reads results from results/ablation/M04_frac{frac}_s{seed}/summary.json
and outputs a LaTeX table + a heatmap figure.

Usage:
    cd /Users/davidsvensson/Desktop/dlp_grokking
    python3 paper/generate_ablation_table.py
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(PROJECT_DIR, "paper")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
ABLATION_DIR = os.path.join(PROJECT_DIR, "results/ablation")

FRACS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
SEEDS = [42, 123, 456]


def load_summary(frac, seed):
    path = os.path.join(ABLATION_DIR, f"M04_frac{frac}_s{seed}", "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_metrics_epoch(frac, seed, target_epoch):
    """Get test accuracy at a specific epoch from metrics."""
    path = os.path.join(ABLATION_DIR, f"M04_frac{frac}_s{seed}", "metrics.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("epoch") == target_epoch and d.get("test_acc") is not None:
                return d["test_acc"]
    return None


def generate_latex_table():
    """Generate LaTeX table for the paper."""
    rows = []
    for frac in FRACS:
        results = []
        for seed in SEEDS:
            s = load_summary(frac, seed)
            if s is None:
                results.append({"state": "—", "t_mem": "—", "t_95": "—", "best": "—"})
            else:
                results.append({
                    "state": s.get("state", "?"),
                    "t_mem": str(s.get("T_mem", "—")),
                    "t_95": str(s.get("T_95", "—")),
                    "best": f"{s.get('best_test_acc', 0):.4f}" if s.get("best_test_acc") else "—",
                })

        # Aggregate
        grokked = sum(1 for r in results if r["state"] in ("E", "G"))
        best_accs = [float(r["best"]) for r in results if r["best"] != "—"]
        mean_best = np.mean(best_accs) if best_accs else 0

        rows.append({
            "frac": frac,
            "n_grokked": grokked,
            "n_total": len(SEEDS),
            "mean_best": mean_best,
            "results": results,
        })

    # Generate LaTeX
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Train fraction ablation for M04 ($p=113$, 427k params). Each cell shows the best test accuracy. ``State'' indicates the final training state: E = grokked, G = grokking, M = memorization only.}")
    tex.append(r"\label{tab:train_frac_ablation}")
    tex.append(r"\begin{tabular}{rcccccc}")
    tex.append(r"\toprule")
    tex.append(r"$\rho$ & Seed 42 & Seed 123 & Seed 456 & Grokked & Mean Best Acc \\")
    tex.append(r"\midrule")

    for r in rows:
        cells = " & ".join([f"{rr['best']}" for rr in r["results"]])
        tex.append(f"{r['frac']:.2f} & {cells} & {r['n_grokked']}/{r['n_total']} & {r['mean_best']:.4f} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


def generate_heatmap():
    """Generate a heatmap figure of grokking outcome."""
    fig, ax = plt.subplots(figsize=(8, 4))

    data = np.zeros((len(SEEDS), len(FRACS)))
    for i, seed in enumerate(SEEDS):
        for j, frac in enumerate(FRACS):
            s = load_summary(frac, seed)
            if s is None:
                data[i, j] = -1
            else:
                data[i, j] = s.get("best_test_acc", 0)

    # Mask missing data
    masked = np.ma.masked_where(data < 0, data)

    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Labels
    ax.set_xticks(range(len(FRACS)))
    ax.set_xticklabels([f"{f:.2f}" for f in FRACS])
    ax.set_yticks(range(len(SEEDS)))
    ax.set_yticklabels([f"Seed {s}" for s in SEEDS])
    ax.set_xlabel(r"Train Fraction $\rho$")
    ax.set_ylabel("Seed")
    ax.set_title(r"Train Fraction Ablation: Best Test Accuracy (M04, $p=113$)")

    # Annotate cells
    for i in range(len(SEEDS)):
        for j in range(len(FRACS)):
            if data[i, j] < 0:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="gray")
            else:
                color = "white" if data[i, j] < 0.5 else "black"
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Best Test Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_ablation_heatmap.pdf"))
    plt.close()
    print(f"  fig_ablation_heatmap.pdf")


def main():
    print("Generating train_frac ablation table and heatmap...")
    print(f"  Reading from: {ABLATION_DIR}")
    print()

    # Check how many runs are complete
    complete = 0
    total = len(FRACS) * len(SEEDS)
    for frac in FRACS:
        for seed in SEEDS:
            s = load_summary(frac, seed)
            if s is not None:
                complete += 1
                state = s.get("state", "?")
                best = s.get("best_test_acc", 0)
                print(f"  frac={frac:.2f} seed={seed}: state={state}, best_acc={best:.4f}")
            else:
                print(f"  frac={frac:.2f} seed={seed}: NOT RUN YET")

    print()
    print(f"  Complete: {complete}/{total}")

    if complete == 0:
        print("  No results yet. Run the ablation first:")
        print("    bash run_train_frac_ablation.sh")
        return

    # Generate LaTeX table
    tex = generate_latex_table()
    tex_path = os.path.join(PAPER_DIR, "ablation_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"\n  LaTeX table: {tex_path}")
    print()
    print(tex)
    print()

    # Generate heatmap
    generate_heatmap()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
