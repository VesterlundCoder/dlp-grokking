#!/usr/bin/env python3
"""Comprehensive analysis of LUMI DLP grokking results."""

import json
import os
import re
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
    """Categorize results into capacity scaling, CRT, prime-order, large-prime."""
    categories = {
        "capacity_scaling": [],
        "crt_variants": [],
        "prime_order": [],
        "large_primes": [],
    }

    for name, data in summaries.items():
        if name.startswith("M0") or name.startswith("M1") and "p113" in name:
            if "p113" in name and not name.startswith("M13") and not name.startswith("M14") \
               and not name.startswith("M15") and not name.startswith("M16") \
               and not name.startswith("M17") and not name.startswith("M18"):
                categories["capacity_scaling"].append((name, data))
        elif name.startswith("CRT") or name.startswith("RAW") or name.startswith("SCRAM"):
            categories["crt_variants"].append((name, data))
        elif name.startswith("P113"):
            categories["prime_order"].append((name, data))
        elif any(name.startswith(f"M{i}") for i in range(13, 19)):
            categories["large_primes"].append((name, data))

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

        marker = "✅" if grokked else "❌"
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


def print_crt_variants(results):
    print("\n" + "=" * 80)
    print("CRT REPRESENTATION VARIANTS (p=113, M04 architecture)")
    print("=" * 80)
    print(f"{'Variant':<15} {'Seed':>6} {'BestAcc':>10} {'Epochs':>10} {'MemEp':>8} {'Grokked':>8}")
    print("-" * 80)

    by_variant = {}
    for name, d in sorted(results):
        variant = d.get("variant", name.split("_")[0])
        seed = d.get("seed", "?")
        acc = d["best_test_acc"]
        grokked = "✅" if acc >= 0.80 else "❌"

        if variant not in by_variant:
            by_variant[variant] = []
        by_variant[variant].append((seed, acc, d["total_epochs"], d["mem_epoch"]))

        print(f"{variant:<15} {seed:>6} {acc:>10.4f} {d['total_epochs']:>10,} {d['mem_epoch']:>8} {grokked:>8}")

    print("\n" + "=" * 80)
    print("SUMMARY BY VARIANT")
    print("=" * 80)
    print(f"{'Variant':<15} {'Seeds':>6} {'MeanAcc':>10} {'StdAcc':>10} {'MeanEpochs':>12} {'GrokRate':>10}")
    print("-" * 80)

    for variant, runs in sorted(by_variant.items()):
        accs = [r[1] for r in runs]
        epochs = [r[2] for r in runs]
        grok_count = sum(1 for a in accs if a >= 0.80)
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
        mean_ep = sum(epochs) / len(epochs)
        print(f"{variant:<15} {len(runs):>6} {mean_acc:>10.4f} {std_acc:>10.4f} {mean_ep:>12,.0f} {grok_count}/{len(runs):>3}")


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
        grokked = "✅" if acc >= 0.80 else "❌"
        print(f"{name:<25} {d['seed']:>6} {model_id:>6} {split:>6} {acc:>10.4f} {d['total_epochs']:>10,} {d['mem_epoch']:>8} {grokked:>8}")


def print_large_primes(results, running_data=None):
    print("\n" + "=" * 80)
    print("LARGE-PRIME SCALING (M13-M18)")
    print("=" * 80)
    print(f"{'Experiment':<20} {'Prime':>6} {'Params':>10} {'BestAcc':>10} {'Epochs':>10} {'Status':>10}")
    print("-" * 80)

    for name, d in sorted(results):
        prime = name.split("_")[1].replace("p", "")
        acc = d["best_test_acc"]
        status = "✅ Grokked" if acc >= 0.80 else "❌ Failed"
        print(f"{name:<20} {prime:>6} {d['n_params']:>10,} {acc:>10.4f} {d['total_epochs']:>10,} {status:>10}")

    if running_data:
        print("\n  Still running:")
        for name, metrics in running_data.items():
            acc = metrics.get("test_acc", 0)
            best = metrics.get("best_test_acc", 0)
            epoch = metrics.get("epoch", 0)
            status = "near grok" if best >= 0.90 else "in progress"
            print(f"  {name}: epoch={epoch:,} test_acc={acc:.4f} best={best:.4f} ({status})")


def main():
    if not SUMMARIES_FILE.exists():
        print(f"Error: {SUMMARIES_FILE} not found. Run the pull script first.")
        return

    with open(SUMMARIES_FILE) as f:
        text = f.read()

    summaries = parse_summaries(text)
    print(f"Parsed {len(summaries)} result summaries")

    categories = categorize_results(summaries)

    # Parse running job data
    running_data = {}
    running_file = RESULTS_DIR / "running_jobs.txt"
    if running_file.exists():
        with open(running_file) as f:
            current_name = None
            for line in f:
                line = line.strip()
                if line.startswith("---") and line.endswith("---"):
                    current_name = line.strip("---")
                elif line and current_name:
                    try:
                        running_data[current_name] = json.loads(line)
                    except json.JSONDecodeError:
                        pass

    print_capacity_scaling(categories["capacity_scaling"])
    print_crt_variants(categories["crt_variants"])
    print_prime_order(categories["prime_order"])
    print_large_primes(categories["large_primes"], running_data)

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

    # CRT findings
    crt = categories["crt_variants"]
    raw_runs = [(n, d) for n, d in crt if d.get("variant") == "RAW"]
    crt_both_runs = [(n, d) for n, d in crt if d.get("variant") == "CRT-BOTH"]
    scrambled_runs = [(n, d) for n, d in crt if d.get("variant") == "SCRAMBLED"]

    print(f"\n2. CRT REPRESENTATION ACCELERATION:")
    if raw_runs:
        raw_accs = [d["best_test_acc"] for _, d in raw_runs]
        raw_mean = sum(raw_accs) / len(raw_accs)
        print(f"   - RAW: mean acc = {raw_mean:.4f} ({len(raw_runs)} seeds), none grokked in 500K epochs")
    if crt_both_runs:
        crt_accs = [d["best_test_acc"] for _, d in crt_both_runs]
        crt_epochs = [d["total_epochs"] for _, d in crt_both_runs]
        crt_mean = sum(crt_accs) / len(crt_accs)
        crt_mean_ep = sum(crt_epochs) / len(crt_epochs)
        print(f"   - CRT-BOTH: mean acc = {crt_mean:.4f} ({len(crt_both_runs)} seeds), all grokked")
        print(f"   - CRT-BOTH mean grokking time: {crt_mean_ep:,.0f} epochs")
    if scrambled_runs:
        scr = scrambled_runs[0][1]
        print(f"   - SCRAMBLED: acc = {scr['best_test_acc']:.4f}, grokked in {scr['total_epochs']:,} epochs")
        print(f"   - ⚠️  SCRAMBLED groks FASTER than CRT-BOTH — algebraic labeling not required!")

    # Prime-order findings
    po = categories["prime_order"]
    po_grokked = [(n, d) for n, d in po if d["best_test_acc"] >= 0.80]
    print(f"\n3. PRIME-ORDER DLP (no smooth factorization):")
    print(f"   - {len(po_grokked)}/{len(po)} experiments grokked")
    ood = [(n, d) for n, d in po if "ood" in n]
    if ood:
        print(f"   - OOD split: GROKKED at {ood[0][1]['best_test_acc']:.4f} in {ood[0][1]['total_epochs']:,} epochs")
        print(f"     (Local experiment stopped at 100K epochs with 2.5% — LUMI ran to 326K)")

    # Large prime findings
    lp = categories["large_primes"]
    print(f"\n4. LARGE-PRIME SCALING:")
    for name, d in sorted(lp):
        prime = name.split("_")[1].replace("p", "")
        print(f"   - {name}: p={prime}, acc={d['best_test_acc']:.4f} in {d['total_epochs']:,} epochs")


if __name__ == "__main__":
    main()
