"""5-state classification and timing extraction for DLP scaling study.

States:
  A. Under-capacity: train_acc never reaches 99.5%
  B. Memorization-only: train_acc > 99.5% but test_acc stays < 10%
  C. Grokking: T_mem << T_95 (clear separation, T_95 > 2*T_mem)
  D. Immediate generalization: T_95 < 2*T_mem
  E. Collapse/instability: train_acc drops > 50pp after peak
"""
from __future__ import annotations

import json
import math
from typing import List, Dict, Optional


def _get(metrics: List[Dict], key: str, default=None) -> List[float]:
    """Extract a field from metrics list, skipping None values."""
    vals = []
    for m in metrics:
        v = m.get(key, default)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    return vals


def extract_timing(metrics: List[Dict]) -> Dict:
    """Extract T_mem, T_50, T_95, G from a metrics list.

    T_mem = min{epoch : train_acc > 0.995, sustained 3 evals}
    T_50  = min{epoch : test_acc > 0.50}
    T_95  = min{epoch : test_acc > 0.95}
    G     = (T_95 - T_mem) / T_mem (grokking delay ratio)
    """
    train_accs = [(m.get("epoch", i), m.get("train_acc")) for i, m in enumerate(metrics)]
    test_accs = [(m.get("epoch", i), m.get("test_acc")) for i, m in enumerate(metrics)]

    # T_mem: train_acc > 0.995 sustained for 3 consecutive evals
    t_mem = None
    for i in range(len(train_accs) - 2):
        ep, acc = train_accs[i]
        if acc is None or math.isnan(acc):
            continue
        if acc > 0.995:
            # Check sustained for 3 evals
            sustained = True
            for j in range(1, 3):
                if i + j >= len(train_accs):
                    sustained = False
                    break
                next_acc = train_accs[i + j][1]
                if next_acc is None or math.isnan(next_acc) or next_acc <= 0.995:
                    sustained = False
                    break
            if sustained:
                t_mem = ep
                break

    # T_50
    t_50 = None
    for ep, acc in test_accs:
        if acc is not None and not math.isnan(acc) and acc > 0.50:
            t_50 = ep
            break

    # T_95
    t_95 = None
    for ep, acc in test_accs:
        if acc is not None and not math.isnan(acc) and acc > 0.95:
            t_95 = ep
            break

    # G = (T_95 - T_mem) / T_mem
    g = None
    if t_95 is not None and t_mem is not None and t_mem > 0:
        g = (t_95 - t_mem) / t_mem

    return {
        "T_mem": t_mem,
        "T_50": t_50,
        "T_95": t_95,
        "G": g,
    }


def classify_5state(metrics: List[Dict]) -> str:
    """Classify a run into one of 5 states.

    A. Under-capacity: train_acc never reaches 99.5%
    B. Memorization-only: train_acc > 99.5% but test_acc stays < 10%
    C. Grokking: T_mem << T_95 (T_95 > 2*T_mem, i.e. G > 1)
    D. Immediate generalization: T_95 <= 2*T_mem (G <= 1) or T_95 < 2*T_mem
    E. Collapse/instability: train_acc drops > 50pp after peak
    """
    if not metrics:
        return "unknown"

    train_accs = _get(metrics, "train_acc")
    test_accs = _get(metrics, "test_acc")

    if not train_accs:
        return "unknown"

    max_train = max(train_accs) if train_accs else 0.0
    max_test = max(test_accs) if test_accs else 0.0

    # Check for collapse: train_acc drops > 50pp after peak
    if train_accs:
        peak_idx = train_accs.index(max_train)
        if peak_idx < len(train_accs) - 1:
            post_peak = train_accs[peak_idx:]
            min_post = min(post_peak) if post_peak else max_train
            if max_train - min_post > 0.50:
                return "E"

    # A: Under-capacity — never reaches 99.5%
    if max_train < 0.995:
        return "A"

    # B: Memorization-only — train > 99.5% but test stays < 10%
    if max_train >= 0.995 and max_test < 0.10:
        return "B"

    # C vs D: grokking vs immediate generalization
    timing = extract_timing(metrics)
    t_mem = timing["T_mem"]
    t_95 = timing["T_95"]

    if t_95 is None:
        # Never reached 95% test acc
        if max_test >= 0.10:
            return "B"  # partial memorization
        return "B"

    if t_mem is None:
        # Train never sustained 99.5% for 3 evals but test > 95%?
        return "D"  # immediate generalization (no memorization phase)

    if t_mem > 0 and t_95 > 2 * t_mem:
        return "C"  # grokking
    else:
        return "D"  # immediate generalization


def analyze_run(metrics_path: str) -> Dict:
    """Load metrics.jsonl and return full analysis."""
    metrics = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not metrics:
        return {"state": "unknown", "n_evals": 0}

    train_accs = _get(metrics, "train_acc")
    test_accs = _get(metrics, "test_acc")

    timing = extract_timing(metrics)
    state = classify_5state(metrics)

    return {
        "state": state,
        "n_evals": len(metrics),
        "total_epochs": metrics[-1].get("epoch", 0) if metrics else 0,
        "best_train_acc": max(train_accs) if train_accs else 0.0,
        "best_test_acc": max(test_accs) if test_accs else 0.0,
        "final_train_acc": train_accs[-1] if train_accs else 0.0,
        "final_test_acc": test_accs[-1] if test_accs else 0.0,
        "T_mem": timing["T_mem"],
        "T_50": timing["T_50"],
        "T_95": timing["T_95"],
        "G": timing["G"],
    }
