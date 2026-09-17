#!/usr/bin/env python3
"""Quick metrics viewer for DLP grokking training on LUMI.

Usage:
    python monitor.py                          # show all problems
    python monitor.py /path/to/results/         # specific results dir
    python monitor.py --watch                   # refresh every 30s
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


def show_problem(problem_dir):
    name = os.path.basename(problem_dir)
    config_path = os.path.join(problem_dir, "config.json")
    metrics_path = os.path.join(problem_dir, "metrics.jsonl")
    summary_path = os.path.join(problem_dir, "summary.json")

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"  Problem: {config.get('problem', '?')}, Prime: {config.get('prime', '?')}")
        print(f"  Model: d={config.get('d_model', '?')}, L={config.get('n_layers', '?')}, H={config.get('n_heads', '?')}")
        print(f"  Params: {config.get('n_params', '?'):,}")
        print(f"  Train: {config.get('train_size', '?')}, Test: {config.get('test_size', '?')}")
        print(f"  Space: {config.get('space_size', '?'):,}")
        print(f"  DDP: {config.get('ddp', False)}, AMP: {config.get('use_amp', False)}")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n  --- SUMMARY ---")
        print(f"  Final train acc: {summary.get('final_train_acc', 0):.4f}")
        print(f"  Final test acc:  {summary.get('final_test_acc', 0):.4f}")
        print(f"  Best test acc:   {summary.get('best_test_acc', 0):.4f}")
        print(f"  Grokking onset:  {summary.get('grokking_onset', 'N/A')}")
        print(f"  Grokking done:   {summary.get('grokking_completion', 'N/A')}")
        print(f"  Total epochs:    {summary.get('total_epochs', '?')}")
        print(f"  Wall time:       {summary.get('wall_time', '?')}")
        status = "GROKKED" if summary.get('final_test_acc', 0) >= 0.80 else \
                 "PARTIAL" if summary.get('final_train_acc', 0) >= 0.99 else "FAILED"
        print(f"  Status:          {status}")
        return

    metrics = read_metrics(metrics_path)
    if not metrics:
        print(f"  (no metrics yet)")
        return

    last = metrics[-1]
    first = metrics[0]
    print(f"\n  Epoch {first['epoch']} -> {last['epoch']} ({len(metrics)} evals)")
    print(f"  Train acc: {first.get('train_acc', 0):.4f} -> {last.get('train_acc', 0):.4f}")
    print(f"  Test acc:  {first.get('test_acc', 0):.4f} -> {last.get('test_acc', 0):.4f}")
    print(f"  Weight decay: {last.get('weight_decay', 0):.3f}")
    print(f"  Phase: {last.get('phase', '?')}")

    best = max(metrics, key=lambda m: m.get('test_acc', 0))
    print(f"  Best test acc: {best.get('test_acc', 0):.4f} (epoch {best['epoch']})")

    if len(metrics) >= 5:
        recent = metrics[-5:]
        trend = recent[-1].get('test_acc', 0) - recent[0].get('test_acc', 0)
        print(f"  Recent trend: {'+' if trend >= 0 else ''}{trend:.4f} (last 5 evals)")

    test_acc = last.get('test_acc', 0)
    train_acc = last.get('train_acc', 0)
    if test_acc >= 0.80:
        print(f"  Status: GROKKED")
    elif train_acc >= 0.99:
        print(f"  Status: MEMORIZING (train high, test {test_acc:.2%})")
    elif train_acc >= 0.50:
        print(f"  Status: LEARNING (train {train_acc:.2%})")
    else:
        print(f"  Status: EARLY (train {train_acc:.2%})")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else \
        "/scratch/project_465002952/dlp_grokking/results"
    watch = "--watch" in sys.argv

    while True:
        os.system("clear" if os.name != "nt" else "cls")
        print(f"DLP Grokking Monitor - {results_dir}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if os.path.exists(results_dir):
            problem_dirs = sorted(
                d for d in Path(results_dir).iterdir()
                if d.is_dir() and ((d / "metrics.jsonl").exists() or (d / "summary.json").exists())
            )
            if problem_dirs:
                for d in problem_dirs:
                    show_problem(str(d))
            else:
                print("\n  (no problem directories with metrics found)")
        else:
            print(f"\n  Results dir not found: {results_dir}")

        if not watch:
            break
        print(f"\n  (refreshing in 30s... Ctrl-C to stop)")
        time.sleep(30)


if __name__ == "__main__":
    main()
