#!/usr/bin/env python3
"""Comprehensive analysis of LUMI DLP grokking results."""

import json
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "lumi_results"
SUMMARIES_FILE = RESULTS_DIR / "all_summaries.txt"


def parse_summaries(text):
    """Parse the all_summaries.txt format: ---name--- followed by JSON."""
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


def categorize_results(summaries):
    """Categorize results into capacity scaling and prime-order."""
    categories = {
        "capacity_scaling": [],
        "prime_order": [],
    }

    for name, data in summaries.items():
        if name.startswith("M0") or name.startswith("M1") and "p113" in name:
            if "p113" in name and not name.startswith("M13") and not name.startswith("M14") \
               and not name.startswith("M15") and not name.startswith("M16") \
               and not name.startswith("M17") and not name.startswith("M18"):
                categories["capacity_scaling"].append((name, data))
        elif name.startswith("P113"):
            categories["prime_order"].append((name, data))

    return categories


def print_capacity_scaling(results):
    print("\n" + "=" * 80)
    print("CAPACITY SCALING (M01-M12, p=113)")
    print("=" * 80)
    print(f"{'Model':<8} {'Params':>10} {'Core':>10} {'TrainFrac':>10} {'BestAcc':>10} {'Epochs':>10} {'MemEp':>8} {'Grokked':>8}")
    print("-" * 80)

    grokked_models = []
    failed_models = []

    for name, d in sorted(results):
        model_id = name.split("_")[0]
        acc = d["best_test_acc"]
        grokked = acc >= 0.80
        train_frac = d.get("train_size", 0) / (d.get("train_size", 0) + d.get("test_size", 1))

        marker = "YES" if grokked else "NO"
        print(f"{model_id:<8} {d['n_params']:>10,} {d['n_core']:>10,} {train_frac:>10.1%} {acc:>10.4f} {d['total_epochs']:>10,} {d['mem_epoch']:>8} {marker:>8}")

        if grokked:
            grokked_models.append((model_id, acc, d['total_epochs'], d['n_params']))
        else:
            failed_models.append((model_id, acc, d['total_epochs'], d['n_params']))

    print(f"\nGrokked: {len(grokked_models)}/{len(results)}")
    if grokked_models:
        print("\nGrokking speed vs capacity:")
        for m, acc, ep, params in grokked_models:
            print(f"  {m}: {acc:.4f} in {ep:,} epochs ({params:,} params)")

    if failed_models:
        print("\nFailed to grok:")
        for m, acc, ep, params in failed_models:
            print(f"  {m}: {acc:.4f} in {ep:,} epochs ({params:,} params)")


def print_prime_order(results):
    print("\n" + "=" * 80)
    print("PRIME-ORDER SUBGROUP DLP (order-113 subgroup of F_227*)")
    print("=" * 80)
    print(f"{'Experiment':<25} {'Seed':>6} {'Model':>6} {'Split':>6} {'BestAcc':>10} {'Epochs':>10} {'MemEp':>8} {'Grokked':>8}")
    print("-" * 80)

    for name, d in sorted(results):
        parts = name.split("_")
        model_id = d.get("name", "").split("_")[1] if "iid" in name or "ood" in name else "?"
        split = "IID" if "iid" in name else "OOD"
        acc = d["best_test_acc"]
        grokked = "YES" if acc >= 0.80 else "NO"
        print(f"{name:<25} {d['seed']:>6} {model_id:>6} {split:>6} {acc:>10.4f} {d['total_epochs']:>10,} {d['mem_epoch']:>8} {grokked:>8}")


def main():
    if not SUMMARIES_FILE.exists():
        print(f"Error: {SUMMARIES_FILE} not found. Run the pull script first.")
        return

    with open(SUMMARIES_FILE) as f:
        text = f.read()

    summaries = parse_summaries(text)
    print(f"Parsed {len(summaries)} result summaries")

    categories = categorize_results(summaries)

    print_capacity_scaling(categories["capacity_scaling"])
    print_prime_order(categories["prime_order"])

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Capacity scaling findings
    cap = categories["capacity_scaling"]
    grokked_cap = [(n, d) for n, d in cap if d["best_test_acc"] >= 0.80]
    failed_cap = [(n, d) for n, d in cap if d["best_test_acc"] < 0.80]

    print(f"\n1. CAPACITY SCALING (p=113):")
    print(f"   - {len(grokked_cap)}/{len(cap)} models grokked")
    print(f"   - Grokked: {', '.join(n.split('_')[0] for n, _ in sorted(grokked_cap))}")
    print(f"   - Failed: {', '.join(n.split('_')[0] for n, _ in sorted(failed_cap))}")

    # Check for non-monotonicity
    cap_sorted = sorted(cap, key=lambda x: x[1]["n_params"])
    accs = [d["best_test_acc"] for _, d in cap_sorted]
    params = [d["n_params"] for _, d in cap_sorted]

    # Find the dip
    for i in range(1, len(accs) - 1):
        if accs[i] < accs[i-1] and accs[i] < accs[i+1] and accs[i] < 0.50:
            print(f"   - Non-monotonic dip at {cap_sorted[i][0].split('_')[0]} ({params[i]:,} params, acc={accs[i]:.4f})")
            print(f"     This is the 'grokking valley' — larger than M04 but fails to grok with 30% data")

    # Prime-order findings
    po = categories["prime_order"]
    po_grokked = [(n, d) for n, d in po if d["best_test_acc"] >= 0.80]
    print(f"\n2. PRIME-ORDER DLP (no smooth factorization):")
    print(f"   - {len(po_grokked)}/{len(po)} experiments grokked")
    ood = [(n, d) for n, d in po if "ood" in n]
    if ood:
        print(f"   - OOD split: GROKKED at {ood[0][1]['best_test_acc']:.4f} in {ood[0][1]['total_epochs']:,} epochs")


if __name__ == "__main__":
    main()
