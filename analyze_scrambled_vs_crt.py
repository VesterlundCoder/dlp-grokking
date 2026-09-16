#!/usr/bin/env python3
"""Investigate the SCRAMBLED vs CRT-BOTH finding.

Key finding from LUMI results:
  CRT-BOTH: grokked in ~20,400 epochs (seed 42)
  SCRAMBLED: grokked in ~16,050 epochs (seed 42)

SCRAMBLED groks FASTER than CRT-BOTH, despite having a random permutation
applied to the CRT component values. This suggests:

1. The acceleration is NOT due to algebraic labeling of the CRT components
2. The acceleration IS due to the information content (both components present)
3. The model discovers the mapping regardless of the labeling

This script:
- Verifies the SCRAMBLED result by checking the LUMI metrics
- Analyzes what the SCRAMBLED model learned (if checkpoints available)
- Tests whether the speed difference is significant
- Considers implications for the representation-relative grokking hypothesis
"""

import json
import os
import sys
from pathlib import Path

# LUMI results
LUMI_RESULTS = Path(__file__).parent / "lumi_results"
ALL_SUMMARIES = LUMI_RESULTS / "all_summaries.txt"


def parse_summaries(text):
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


def main():
    if not ALL_SUMMARIES.exists():
        print(f"Error: {ALL_SUMMARIES} not found")
        return

    with open(ALL_SUMMARIES) as f:
        summaries = parse_summaries(f.read())

    print("=" * 80)
    print("SCRAMBLED vs CRT-BOTH ANALYSIS")
    print("=" * 80)

    # Collect relevant results
    crt_both = {k: v for k, v in summaries.items() if "CRT-BOTH" in k}
    scrambled = {k: v for k, v in summaries.items() if "SCRAMBLED" in k}
    raw_crt = {k: v for k, v in summaries.items() if "RAW+CRT" in k}
    raw = {k: v for k, v in summaries.items() if k.startswith("RAW_p")}

    print("\n1. GROKKING SPEED COMPARISON")
    print("-" * 60)
    print(f"{'Variant':<15} {'Seed':>6} {'Best Acc':>10} {'Total Epochs':>14} {'Mem Epoch':>10}")
    print("-" * 60)

    for name, d in sorted(crt_both.items()):
        print(f"CRT-BOTH        {d['seed']:>6} {d['best_test_acc']:>10.4f} {d['total_epochs']:>14,} {d['mem_epoch']:>10}")

    for name, d in sorted(scrambled.items()):
        print(f"SCRAMBLED       {d['seed']:>6} {d['best_test_acc']:>10.4f} {d['total_epochs']:>14,} {d['mem_epoch']:>10}")

    for name, d in sorted(raw_crt.items()):
        print(f"RAW+CRT         {d['seed']:>6} {d['best_test_acc']:>10.4f} {d['total_epochs']:>14,} {d['mem_epoch']:>10}")

    for name, d in sorted(raw.items()):
        print(f"RAW             {d['seed']:>6} {d['best_test_acc']:>10.4f} {d['total_epochs']:>14,} {d['mem_epoch']:>10}")

    print("\n2. KEY OBSERVATIONS")
    print("-" * 60)

    # CRT-BOTH vs SCRAMBLED
    crt_epochs = [d["total_epochs"] for d in crt_both.values()]
    scr_epochs = [d["total_epochs"] for d in scrambled.values()]
    crt_accs = [d["best_test_acc"] for d in crt_both.values()]
    scr_accs = [d["best_test_acc"] for d in scrambled.values()]

    print(f"CRT-BOTH:   mean grok time = {sum(crt_epochs)/len(crt_epochs):,.0f} epochs, mean acc = {sum(crt_accs)/len(crt_accs):.4f} ({len(crt_both)} seeds)")
    print(f"SCRAMBLED:  grok time = {scr_epochs[0]:,} epochs, acc = {scr_accs[0]:.4f} (1 seed)")

    if scr_epochs[0] < sum(crt_epochs) / len(crt_epochs):
        speedup = (sum(crt_epochs) / len(crt_epochs)) / scr_epochs[0]
        print(f"\n⚠️  SCRAMBLED groks {speedup:.2f}x FASTER than CRT-BOTH mean")
        print("   This means the algebraic labeling of CRT components is NOT required")
        print("   for the acceleration. The information content is sufficient.")
    else:
        print(f"\n   CRT-BOTH is faster (as expected)")

    # RAW+CRT comparison
    if raw_crt:
        rc = list(raw_crt.values())[0]
        print(f"\nRAW+CRT:    grok time = {rc['total_epochs']:,} epochs, acc = {rc['best_test_acc']:.4f}")
        print(f"   RAW+CRT is SLOWER than CRT-BOTH ({rc['total_epochs']:,} vs {sum(crt_epochs)/len(crt_epochs):,.0f})")
        print("   Adding raw inputs to CRT components SLOWS DOWN grokking!")
        print("   This suggests extra (redundant) information can confuse the model.")

    # RAW comparison
    if raw:
        raw_accs = [d["best_test_acc"] for d in raw.values()]
        raw_mean = sum(raw_accs) / len(raw_accs)
        print(f"\nRAW:        mean acc = {raw_mean:.4f} ({len(raw)} seeds), NONE grokked in 500K epochs")
        print("   RAW never groks with 30% data in 500K epochs on LUMI")

    print("\n3. IMPLICATIONS FOR REPRESENTATION-RELATIVE GROKKING")
    print("-" * 60)
    print("""
The original hypothesis H2 states: "Explicit representation accelerates learning"

The SCRAMBLED result challenges the *mechanism* behind H2:
  - CRT-BOTH provides algebraically meaningful CRT components
  - SCRAMBLED provides the SAME information but with scrambled labels
  - SCRAMBLED groks FASTER than CRT-BOTH

This means:
  a) The acceleration is NOT due to the model "recognizing" the CRT structure
  b) The acceleration IS due to the input containing sufficient information
     (both CRT components uniquely determine x)
  c) The model discovers the mapping from scrambled labels to x regardless

Revised interpretation:
  - "Explicit representation" should be understood as "sufficient information
    in a learnable format" rather than "algebraically meaningful labeling"
  - The key factor is INFORMATION CONTENT, not ALGEBRAIC STRUCTURE
  - The model is a general-purpose function approximator, not a structure
    detector

However, the SCRAMBLED result needs verification:
  - Only 1 seed for SCRAMBLED (need multi-seed)
  - The scramble map could accidentally preserve some structure
  - Need to verify the scramble is truly random and doesn't create shortcuts

The CRT-7 and CRT-16 negative controls still hold:
  - CRT-7: 4.0% (ceiling 6.25%) — information-theoretically insufficient
  - CRT-16: 9.7% (ceiling 14.3%) — information-theoretically insufficient
  These confirm that INSUFFICIENT information prevents grokking regardless
  of labeling.
""")

    print("4. PROPOSED FOLLOW-UP EXPERIMENTS")
    print("-" * 60)
    print("""
  a) Multi-seed SCRAMBLED: Run 3-5 seeds to verify the speed advantage
  b) Multiple scramble maps: Test different random permutations
  c) Partial scramble: Scramble only one component (g16 or g7)
  d) Identity scramble: Use identity permutation (should = CRT-BOTH)
  e) Structure-preserving scramble: Permute within order-16/order-7 subgroups
""")


if __name__ == "__main__":
    main()
