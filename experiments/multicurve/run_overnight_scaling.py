#!/usr/bin/env python3
"""Overnight CRT-BOTH scaling sweep: test how large primes we can grok locally.

Runs a sequence of experiments with different model sizes (M04, M05, M06, M07)
and primes (337, 601, 1201) using CRT-BOTH representation.

Each experiment has a time budget. If it groks, we move on. If it doesn't grok
within the budget, we kill it and move to the next one.

Usage:
    python run_overnight_scaling.py
"""
import subprocess
import sys
import os
import time
import json
import signal
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/Users/davidsvensson/Desktop/rd-lumi-z3/venv/bin/python"
SCRIPT = str(REPO_ROOT / "experiments" / "multicurve" / "run_multicurve.py")
RESULTS_DIR = REPO_ROOT / "results" / "overnight_scaling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment matrix: (name, primes, model_id, lr, train_frac, epochs, time_budget_s, seed)
# Time budgets are estimates; the script will kill experiments that exceed them.
# Key learnings from run 1:
#   - p=337 groks with M05/M06 at tf=0.30, lr=3e-4 (99.2-99.99%)
#   - p=337 groks with M04 at tf=0.50, lr=3e-4 (95.6%)
#   - p=601 at lr=1e-4 causes model collapse (loss->0). Need lr=3e-4 or 5e-4.
#   - p=1201 OOMs on MPS (28,800 train examples). Skip locally.
#   - M07 now added to trainer.
EXPERIMENTS = [
    # Priority 1: p=601 with lr=3e-4 (the LR that works for p=337)
    ("M04_p601_lr3e4", [601], "M04", 3e-4, 0.30, 15000, 7200, 42),
    # Priority 2: p=601 with M06 (more capacity) at lr=3e-4
    ("M06_p601_lr3e4", [601], "M06", 3e-4, 0.30, 15000, 7200, 42),
    # Priority 3: p=337 M04 at tf=0.40 (between 0.30 fail and 0.50 success)
    ("M04_p337_tf40_lr3e4", [337], "M04", 3e-4, 0.40, 20000, 5400, 42),
    # Priority 4: p=337 M07 at tf=0.30 (test if larger model helps)
    ("M07_p337_lr3e4", [337], "M07", 3e-4, 0.30, 20000, 7200, 42),
    # Priority 5: p=337 M04 seed 123 (multi-seed verification)
    ("M04_p337_tf50_lr3e4_s123", [337], "M04", 3e-4, 0.50, 20000, 5400, 123),
    # Priority 6: p=601 M05 at lr=5e-4 (intermediate LR)
    ("M05_p601_lr5e4", [601], "M05", 5e-4, 0.30, 15000, 7200, 42),
    # Priority 7: p=337 M08 at tf=0.30 (even larger model)
    ("M08_p337_lr3e4", [337], "M08", 3e-4, 0.30, 20000, 7200, 42),
    # Priority 8: p=601 M04 at tf=0.50 (more data might help)
    ("M04_p601_tf50_lr3e4", [601], "M04", 3e-4, 0.50, 15000, 7200, 42),
]

LOG_FILE = RESULTS_DIR / "overnight_log.txt"
RESULTS_SUMMARY = RESULTS_DIR / "results_summary.json"


def log(msg):
    """Log a message to both stdout and the log file."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_experiment(name, primes, model_id, lr, train_frac, epochs, time_budget_s, seed=42):
    """Run a single experiment with a time budget. Returns (success, grokked, best_acc, epochs_run)."""
    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        best = summary.get("best_test_acc", 0)
        grokked = best > 0.95
        log(f"  SKIP {name} (already done: best={best:.4f}, grokked={grokked})")
        return True, grokked, best, summary.get("total_epochs", 0)

    cmd = [
        PYTHON, SCRIPT,
        "--primes", *[str(p) for p in primes],
        "--model-id", model_id,
        "--variant", "CRT-BOTH",
        "--epochs", str(epochs),
        "--train-frac", str(train_frac),
        "--seed", str(seed),
        "--eval-interval", "200",
        "--early-stop-patience", "50",
        "--wd-ramp-interval", "1000",
        "--lr-drop-factor", "1.0",
        "--lr", str(lr),
        "--output-dir", str(output_dir),
    ]

    log(f"  START {name}: primes={primes}, model={model_id}, lr={lr}, tf={train_frac}, "
        f"epochs={epochs}, budget={time_budget_s}s")

    start_time = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    # Monitor with time budget
    best_acc = 0.0
    grokked = False
    epochs_run = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget_s:
            log(f"  TIMEOUT {name} after {elapsed:.0f}s (budget={time_budget_s}s)")
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
            break

        # Check if process finished
        ret = proc.poll()
        if ret is not None:
            # Process finished, read remaining output
            output = proc.stdout.read().decode()
            with open(output_dir / "stdout.txt", "w") as f:
                f.write(output)
            log(f"  FINISH {name} (exit code {ret}) after {elapsed:.0f}s")
            break

        # Check for grokking by reading the summary file
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            best_acc = summary.get("best_test_acc", 0)
            epochs_run = summary.get("total_epochs", 0)
            grokked = best_acc > 0.95
            if grokked:
                log(f"  GROKKED {name}: best={best_acc:.4f}, epochs={epochs_run}")
                # Kill the process since it's done
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
                break

        # Also check metrics file for progress
        metrics_path = output_dir / "metrics.jsonl"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    lines = f.readlines()
                    if lines:
                        last = json.loads(lines[-1])
                        best_acc = max(best_acc, last.get("best_test_acc", 0))
                        epochs_run = last.get("epoch", 0)
                        # Check grokking from metrics (summary may not exist yet)
                        if best_acc > 0.95 and not grokked:
                            grokked = True
                            log(f"  GROKKED {name}: best={best_acc:.4f}, "
                                f"epochs={epochs_run} (from metrics)")
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                            proc.wait(timeout=10)
                            break
            except (json.JSONDecodeError, IndexError):
                pass

        time.sleep(30)

    # Final check
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        best_acc = max(best_acc, summary.get("best_test_acc", 0))
        epochs_run = summary.get("total_epochs", epochs_run)

    # Always check metrics for best (summary may not exist if killed)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                for line in f:
                    d = json.loads(line)
                    best_acc = max(best_acc, d.get("best_test_acc", 0))
                    epochs_run = max(epochs_run, d.get("epoch", 0))
        except (json.JSONDecodeError, IndexError):
            pass

    grokked = best_acc > 0.95

    elapsed = time.time() - start_time
    log(f"  DONE {name}: best={best_acc:.4f}, grokked={grokked}, "
        f"epochs={epochs_run}, elapsed={elapsed:.0f}s")

    return True, grokked, best_acc, epochs_run


def main():
    log("=" * 80)
    log("OVERNIGHT CRT-BOTH SCALING SWEEP")
    log(f"Total experiments: {len(EXPERIMENTS)}")
    log(f"Results dir: {RESULTS_DIR}")
    log("=" * 80)

    total_budget = 12 * 3600  # 12 hours
    elapsed_total = 0
    results = []

    for i, (name, primes, model_id, lr, train_frac, epochs, time_budget, seed) in enumerate(EXPERIMENTS):
        remaining = total_budget - elapsed_total
        if remaining < 600:  # Less than 10 min left, stop
            log(f"Stopping: only {remaining:.0f}s remaining of 12h budget")
            break

        # Cap time budget at remaining time
        actual_budget = min(time_budget, remaining)
        log(f"\nExperiment {i+1}/{len(EXPERIMENTS)}: {name} (budget={actual_budget}s)")

        start = time.time()
        done, grokked, best_acc, epochs_run = run_experiment(
            name, primes, model_id, lr, train_frac, epochs, actual_budget, seed
        )
        elapsed = time.time() - start
        elapsed_total += elapsed

        results.append({
            "name": name,
            "primes": primes,
            "model_id": model_id,
            "lr": lr,
            "train_frac": train_frac,
            "seed": seed,
            "grokked": grokked,
            "best_acc": best_acc,
            "epochs_run": epochs_run,
            "elapsed_s": elapsed,
        })

        # Save intermediate results
        with open(RESULTS_SUMMARY, "w") as f:
            json.dump(results, f, indent=2)

        log(f"  Cumulative elapsed: {elapsed_total:.0f}s / {total_budget}s "
            f"({elapsed_total/3600:.1f}h / 12h)")

    log("\n" + "=" * 80)
    log("OVERNIGHT SWEEP COMPLETE")
    log("=" * 80)
    log(f"\nResults summary ({len(results)} experiments):")
    for r in results:
        status = "GROKKED" if r["grokked"] else "FAILED"
        log(f"  {r['name']}: {r['best_acc']:.4f} ({status}), "
            f"{r['epochs_run']} epochs, {r['elapsed_s']:.0f}s")

    with open(RESULTS_SUMMARY, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nFull results saved to {RESULTS_SUMMARY}")


if __name__ == "__main__":
    main()
