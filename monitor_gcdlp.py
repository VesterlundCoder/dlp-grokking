#!/usr/bin/env python3
"""GCDLP metrics viewer for LUMI training.

Usage:
    python monitor_gcdlp.py
    python monitor_gcdlp.py /path/to/results/gcdlp/
    python monitor_gcdlp.py --watch
"""
import json
import os
import sys
import time
from pathlib import Path


def read_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        return []
    metrics = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return metrics


def show_run(run_dir):
    name = os.path.basename(run_dir)
    config_path = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"  Family: {config.get('family', '?')}, Order visible: {config.get('order_visible', '?')}")
        sgs = config.get("subgroups", [])
        for sg in sgs:
            print(f"    p={sg['p']}, d={sg['d']}")
        print(f"  Model: d={config.get('d_model', '?')}, L={config.get('n_layers', '?')}, H={config.get('n_heads', '?')}")
        print(f"  Params: {config.get('n_params', '?'):,}")
        print(f"  Train: {config.get('train_size', '?'):,}, Test: {config.get('test_size', '?'):,}")
        print(f"  Vocab: {config.get('vocab_size', '?')}, p_max: {config.get('p_max', '?')}")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n  --- SUMMARY ---")
        print(f"  Final train acc:  {summary.get('final_train_acc', 0):.4f}")
        print(f"  Final test acc:   {summary.get('final_test_acc', 0):.4f}")
        print(f"  Best test acc:    {summary.get('best_test_acc', 0):.4f}")
        print(f"  Grokking onset:   {summary.get('grokking_onset', 'N/A')}")
        print(f"  Grokking done:    {summary.get('grokking_completion', 'N/A')}")
        print(f"  Total epochs:     {summary.get('total_epochs', '?')}")
        print(f"  Wall time:        {summary.get('wall_time', '?')}")
        status = "GROKKED" if summary.get('best_test_acc', 0) >= 0.80 else \
                 "PARTIAL" if summary.get('best_test_acc', 0) >= 0.10 else "FAILED"
        print(f"  Status:           {status}")
        return

    metrics = read_metrics(metrics_path)
    if not metrics:
        print(f"  (no metrics yet)")
        return

    last = metrics[-1]
    first = metrics[0]
    print(f"\n  Epoch {first['epoch']} -> {last['epoch']} ({len(metrics)} evals)")
    print(f"  Train acc:  {first.get('train_acc', 0):.4f} -> {last.get('train_acc', 0):.4f}")
    print(f"  Test acc:   {first.get('test_acc', 0):.4f} -> {last.get('test_acc', 0):.4f}")
    print(f"  Chance:     {last.get('chance', 0):.4f}")
    print(f"  Chance-norm: {last.get('chance_normalized', 0):+.4f}")
    print(f"  Weight decay: {last.get('weight_decay', 0):.3f}")
    print(f"  Phase: {last.get('phase', '?')}")

    print(f"\n  Top-K Recall:")
    for k in [1, 5, 10, 100]:
        key = f"top{k}"
        val = last.get(key, 0)
        print(f"    R@{k:3d}: {val:.4f}")

    best = max(metrics, key=lambda m: m.get('test_acc', 0))
    print(f"\n  Best test acc: {best.get('test_acc', 0):.4f} (epoch {best['epoch']})")

    # BitsRemoved@90
    if last.get('top100', 0) >= 0.90:
        import math
        # K90 <= 100, so bits removed >= log2(d/100)
        # We don't know exact d from metrics, estimate from chance
        chance = last.get('chance', 0.001)
        if chance > 0:
            d_est = 1.0 / chance
            bits = math.log2(d_est / 100) if d_est > 100 else 0
            print(f"  BitsRemoved@90: ~{bits:.1f} bits (R@100 >= 90%)")

    if len(metrics) >= 5:
        recent = metrics[-5:]
        trend = recent[-1].get('test_acc', 0) - recent[0].get('test_acc', 0)
        print(f"  Recent trend: {'+' if trend >= 0 else ''}{trend:.4f} (last 5 evals)")

    test_acc = last.get('test_acc', 0)
    train_acc = last.get('train_acc', 0)
    if test_acc >= 0.80:
        print(f"  Status: GROKKED")
    elif train_acc >= 0.99:
        print(f"  Status: MEMORIZING (train {train_acc:.2%}, test {test_acc:.2%})")
    elif train_acc >= 0.50:
        print(f"  Status: LEARNING (train {train_acc:.2%})")
    else:
        print(f"  Status: EARLY (train {train_acc:.2%})")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else \
        "/scratch/project_465002952/dlp_grokking/results/gcdlp"
    watch = "--watch" in sys.argv

    while True:
        os.system("clear" if os.name != "nt" else "cls")
        print(f"GCDLP Monitor - {results_dir}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if os.path.exists(results_dir):
            run_dirs = sorted(
                d for d in Path(results_dir).iterdir()
                if d.is_dir() and ((d / "metrics.jsonl").exists() or (d / "summary.json").exists())
            )
            if run_dirs:
                for d in run_dirs:
                    show_run(str(d))
            else:
                print("\n  (no run directories with metrics found)")
        else:
            print(f"\n  Results dir not found: {results_dir}")

        if not watch:
            break
        print(f"\n  (refreshing in 30s... Ctrl-C to stop)")
        time.sleep(30)


if __name__ == "__main__":
    main()
