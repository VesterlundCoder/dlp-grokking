#!/usr/bin/env python3
"""Local scaling sweep runner for M01-M06.

Runs each model with 3 seeds (anchors get 5 seeds), sequentially.
Resumes from completed runs (checks for summary.json).

Usage:
    python scripts/run_scaling_sweep.py
    python scripts/run_scaling_sweep.py --models M01 M02
    python scripts/run_scaling_sweep.py --epochs 50000
    python scripts/run_scaling_sweep.py --monitor-reps
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Model grid
MODEL_GRID = {
    "M01": {"d_model": 56,  "n_heads": 2},
    "M02": {"d_model": 80,  "n_heads": 4},
    "M03": {"d_model": 104, "n_heads": 4},
    "M04": {"d_model": 128, "n_heads": 4},
    "M05": {"d_model": 160, "n_heads": 5},
    "M06": {"d_model": 208, "n_heads": 8},
}

ANCHORS = ["M04"]  # M09 is LUMI-only
REGULAR_SEEDS = [42, 123, 456]
ANCHOR_SEEDS = [42, 123, 456, 789, 2026]


def is_completed(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, "summary.json"))


def run_model(model_id: str, seed: int, epochs: int, output_base: str,
              monitor_reps: bool = False, eval_interval: int = 10) -> bool:
    """Run a single model+seed. Returns True if completed successfully."""
    output_dir = os.path.join(output_base, f"{model_id}_s{seed}")

    if is_completed(output_dir):
        print(f"  [SKIP] {model_id} seed={seed} — already completed")
        return True

    print(f"\n{'='*60}")
    print(f"  Running {model_id} seed={seed}")
    print(f"  Output: {output_dir}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "src", "trainer.py"),
        "--model-id", model_id,
        "--prime", "113",
        "--n-layers", "2",
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--eval-interval", str(eval_interval),
    ]
    if monitor_reps:
        cmd.append("--monitor-reps")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  [DONE] {model_id} seed={seed} in {elapsed:.0f}s")
        return True
    else:
        print(f"\n  [FAIL] {model_id} seed={seed} (exit code {result.returncode})")
        return False


def main():
    ap = argparse.ArgumentParser(description="Local scaling sweep runner")
    ap.add_argument("--models", nargs="+", default=list(MODEL_GRID.keys()),
                    help="Models to run (default: M01-M06)")
    ap.add_argument("--epochs", type=int, default=100000)
    ap.add_argument("--output-base", type=str, default="results/scaling")
    ap.add_argument("--monitor-reps", action="store_true")
    ap.add_argument("--eval-interval", type=int, default=10)
    args = ap.parse_args()

    output_base = os.path.join(str(PROJECT_ROOT), args.output_base)
    os.makedirs(output_base, exist_ok=True)

    total_runs = 0
    completed = 0
    failed = 0
    skipped = 0

    print(f"DLP Scaling Sweep — Local Phase")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {output_base}")
    print(f"Monitor reps: {args.monitor_reps}")
    print()

    for model_id in args.models:
        if model_id not in MODEL_GRID:
            print(f"  [WARN] {model_id} not in local grid, skipping")
            continue

        seeds = ANCHOR_SEEDS if model_id in ANCHORS else REGULAR_SEEDS
        for seed in seeds:
            total_runs += 1
            output_dir = os.path.join(output_base, f"{model_id}_s{seed}")
            if is_completed(output_dir):
                print(f"  [SKIP] {model_id} seed={seed}")
                skipped += 1
                continue

            success = run_model(model_id, seed, args.epochs, output_base,
                                args.monitor_reps, args.eval_interval)
            if success:
                completed += 1
            else:
                failed += 1

    print(f"\n{'='*60}")
    print(f"  Sweep complete: {completed} done, {failed} failed, {skipped} skipped")
    print(f"  Total runs attempted: {total_runs}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
