#!/usr/bin/env python3
"""Scaling study monitor — shows 5-state classification, timing, and scaling curves.

Usage:
    python monitor_scaling.py
    python monitor_scaling.py /path/to/results/scaling
    python monitor_scaling.py --watch
"""
import json
import os
import sys
import time
import math
from pathlib import Path
from collections import defaultdict

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.analysis import analyze_run


MODEL_GRID = {
    "M01": {"d_model": 56,  "n_heads": 2,  "location": "local"},
    "M02": {"d_model": 80,  "n_heads": 4,  "location": "local"},
    "M03": {"d_model": 104, "n_heads": 4,  "location": "local"},
    "M04": {"d_model": 128, "n_heads": 4,  "location": "local"},
    "M05": {"d_model": 160, "n_heads": 5,  "location": "local"},
    "M06": {"d_model": 208, "n_heads": 8,  "location": "local"},
    "M07": {"d_model": 256, "n_heads": 8,  "location": "lumi"},
    "M08": {"d_model": 320, "n_heads": 10, "location": "lumi"},
    "M09": {"d_model": 384, "n_heads": 12, "location": "lumi"},
    "M10": {"d_model": 480, "n_heads": 15, "location": "lumi"},
    "M11": {"d_model": 608, "n_heads": 19, "location": "lumi"},
    "M12": {"d_model": 768, "n_heads": 24, "location": "lumi"},
}


def show_run(run_dir):
    name = os.path.basename(run_dir)
    parts = name.split("_s")
    model_id = parts[0] if len(parts) > 0 else name
    seed = parts[1] if len(parts) > 1 else "?"

    summary_path = os.path.join(run_dir, "summary.json")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    config_path = os.path.join(run_dir, "config.json")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        state = summary.get("state", "?")
        best = summary.get("best_test_acc", 0)
        t_mem = summary.get("T_mem", "N/A")
        t_50 = summary.get("T_50", "N/A")
        t_95 = summary.get("T_95", "N/A")
        g = summary.get("G", "N/A")
        epochs = summary.get("total_epochs", "?")
        p_total = summary.get("n_params", "?")
        p_core = summary.get("P_core", "?")

        state_emoji = {
            "A": "A.Under",
            "B": "B.Memo",
            "C": "C.Grok",
            "D": "D.ImmGen",
            "E": "E.Collapse",
            "unknown": "?",
        }.get(state, state)

        print(f"  {name:15s} | {state_emoji:10s} | best={best:.4f} | "
              f"T_mem={t_mem} T_50={t_50} T_95={t_95} G={g} | {epochs} ep")
        return

    # In-progress: read metrics
    if os.path.exists(metrics_path):
        analysis = analyze_run(metrics_path)
        state = analysis.get("state", "?")
        best = analysis.get("best_test_acc", 0)
        n_evals = analysis.get("n_evals", 0)
        total_ep = analysis.get("total_epochs", 0)
        t_mem = analysis.get("T_mem", "N/A")
        t_50 = analysis.get("T_50", "N/A")
        t_95 = analysis.get("T_95", "N/A")

        state_emoji = {
            "A": "A.Under",
            "B": "B.Memo",
            "C": "C.Grok",
            "D": "D.ImmGen",
            "E": "E.Collapse",
            "unknown": "?",
        }.get(state, state)

        print(f"  {name:15s} | {state_emoji:10s} | best={best:.4f} | "
              f"T_mem={t_mem} T_50={t_50} T_95={t_95} | {total_ep} ep (running)")
        return

    print(f"  {name:15s} | (no data yet)")


def show_scaling_table(results_dir):
    """Show aggregated scaling table across all models."""
    print(f"\n{'='*80}")
    print(f"  SCALING TABLE")
    print(f"{'='*80}")
    print(f"  {'Model':6s} | {'d_model':7s} | {'P_total':>10s} | {'P_core':>10s} | {'Seeds':5s} | {'State':10s} | {'Best Acc':>8s} | {'T_mem':>6s} | {'T_95':>6s} | {'G':>6s}")
    print(f"  {'-'*6} | {'-'*7} | {'-'*10} | {'-'*10} | {'-'*5} | {'-'*10} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6}")

    for model_id, cfg in MODEL_GRID.items():
        d_model = cfg["d_model"]
        # Find all runs for this model
        model_runs = []
        for d in Path(results_dir).iterdir():
            if d.is_dir() and d.name.startswith(f"{model_id}_s"):
                summary_path = d / "summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        model_runs.append(json.load(f))

        if not model_runs:
            # Check for in-progress
            for d in Path(results_dir).iterdir():
                if d.is_dir() and d.name.startswith(f"{model_id}_s"):
                    metrics_path = d / "metrics.jsonl"
                    if metrics_path.exists():
                        analysis = analyze_run(str(metrics_path))
                        model_runs.append(analysis)
                        break

        if not model_runs:
            print(f"  {model_id:6s} | {d_model:7d} | {'?':>10s} | {'?':>10s} | {0:5d} | {'---':10s} | {'---':>8s} | {'---':>6s} | {'---':>6s} | {'---':>6s}")
            continue

        # Aggregate
        n_seeds = len(model_runs)
        states = [r.get("state", "?") for r in model_runs]
        best_accs = [r.get("best_test_acc", 0) for r in model_runs]
        t_mems = [r.get("T_mem") for r in model_runs if r.get("T_mem") is not None]
        t_95s = [r.get("T_95") for r in model_runs if r.get("T_95") is not None]
        gs = [r.get("G") for r in model_runs if r.get("G") is not None]

        # Most common state
        from collections import Counter
        state_counter = Counter(states)
        dominant_state = state_counter.most_common(1)[0][0] if state_counter else "?"

        avg_best = sum(best_accs) / len(best_accs) if best_accs else 0
        avg_t_mem = sum(t_mems) / len(t_mems) if t_mems else None
        avg_t_95 = sum(t_95s) / len(t_95s) if t_95s else None
        avg_g = sum(gs) / len(gs) if gs else None

        p_total = model_runs[0].get("n_params", "?")
        p_core = model_runs[0].get("P_core", "?")

        p_total_str = f"{p_total:,}" if isinstance(p_total, int) else "?"
        p_core_str = f"{p_core:,}" if isinstance(p_core, int) else "?"

        t_mem_str = f"{avg_t_mem:.0f}" if avg_t_mem is not None else "---"
        t_95_str = f"{avg_t_95:.0f}" if avg_t_95 is not None else "---"
        g_str = f"{avg_g:.2f}" if avg_g is not None else "---"

        print(f"  {model_id:6s} | {d_model:7d} | {p_total_str:>10s} | {p_core_str:>10s} | {n_seeds:5d} | {dominant_state:10s} | {avg_best:>8.4f} | {t_mem_str:>6s} | {t_95_str:>6s} | {g_str:>6s}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "scaling")
    watch = "--watch" in sys.argv

    while True:
        os.system("clear" if os.name != "nt" else "cls")
        print(f"DLP Scaling Study Monitor")
        print(f"Results: {results_dir}")
        print(f"Time:    {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if os.path.exists(results_dir):
            run_dirs = sorted(
                d for d in Path(results_dir).iterdir()
                if d.is_dir() and (d / "metrics.jsonl").exists() or (d / "summary.json").exists()
            )

            if run_dirs:
                print(f"\n{'='*80}")
                print(f"  RUNS ({len(run_dirs)})")
                print(f"{'='*80}")
                for d in run_dirs:
                    show_run(str(d))

                show_scaling_table(results_dir)
            else:
                print("\n  (no run directories found)")
        else:
            print(f"\n  Results dir not found: {results_dir}")

        if not watch:
            break
        print(f"\n  (refreshing in 30s... Ctrl-C to stop)")
        time.sleep(30)


if __name__ == "__main__":
    main()
