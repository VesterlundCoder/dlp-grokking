#!/usr/bin/env python3
"""Train-Fraction Sweep (H4): Measure N_crit for RAW vs CRT-BOTH.

For each variant and train fraction, train the M04 model and record:
- Whether grokking occurs (test acc >= 80%)
- The grokking epoch (if it occurs)
- Best test accuracy

This reveals the critical training fraction N_crit needed for grokking
under each representation, testing H4: "Representation quality predicts
sample complexity."

Sweep: train_frac ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70}
Variants: RAW, CRT-BOTH
Seed: 42
Epoch budget: 50,000 (enough for CRT-BOTH to grok, may not be enough for RAW at low fracs)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models import GrokkingTransformer
from src.trainer import L6Generator, ProblemSpec, split_specs

# ============================================================================
# Constants — Exact M04 Clone
# ============================================================================
M04_CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "max_len": 13,
    "vocab_size": 117,
    "lr": 0.001,
    "wd_start": 0.05,
    "wd_max": 0.30,
    "wd_step": 0.05,
    "wd_ramp_interval": 1000,
    "lr_drop_factor": 1.0,
    "eval_interval": 50,
    "grad_clip": 1.0,
}

PAD_ID = 0
SEP_ID = 1
BOS_ID = 2
EOS_ID = 3
INT_OFFSET = 4
P = 113
Q = 112


def project_16(y):
    return pow(y, 7, P)

def project_7(y):
    return pow(y, 16, P)


def encode(g, h, x, variant):
    g16, g7 = project_16(g), project_7(g)
    h16, h7 = project_16(h), project_7(h)
    tok = lambda v: v + INT_OFFSET
    target = tok(x)

    if variant == "RAW":
        tokens = [BOS_ID, tok(g), SEP_ID, tok(h), EOS_ID]
    elif variant == "CRT-BOTH":
        tokens = [BOS_ID, tok(g16), SEP_ID, tok(g7), SEP_ID, tok(h16), SEP_ID, tok(h7), EOS_ID]
    else:
        raise ValueError(variant)

    padded = tokens + [PAD_ID] * (M04_CONFIG["max_len"] - len(tokens))
    return padded[:M04_CONFIG["max_len"]], target


def build_dataset(specs, variant):
    examples = []
    for spec in specs:
        g, h, x = spec.metadata["g"], spec.metadata["h"], spec.target
        tokens, target = encode(g, h, x, variant)
        examples.append((torch.tensor(tokens, dtype=torch.long), target))
    return examples


def compute_accuracy(model, examples, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(examples), 512):
            batch = examples[i:i+512]
            tokens = torch.stack([b[0] for b in batch]).to(device)
            targets = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
            logits = model(tokens)
            if logits.dim() == 3:
                seq_lens = (tokens != PAD_ID).sum(dim=1) - 1
                idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
            else:
                logits_at_pos = logits
            preds = logits_at_pos.argmax(dim=-1)
            correct += (preds == targets).sum().item()
    return correct / max(len(examples), 1)


def train_one(variant, train_frac, seed, epochs, output_dir):
    cfg = M04_CONFIG.copy()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate all specs
    gen = L6Generator(p=P)
    all_specs = gen.generate_all()
    n_train = int(len(all_specs) * train_frac)
    train_specs, test_specs = split_specs(all_specs, n_train, seed=seed)

    train_examples = build_dataset(train_specs, variant)
    test_examples = build_dataset(test_specs, variant)

    model = GrokkingTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_len=cfg["max_len"],
        dropout=0.0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd_start"])
    current_wd = cfg["wd_start"]

    train_tokens = torch.stack([ex[0] for ex in train_examples]).to(device)
    train_targets = torch.tensor([ex[1] for ex in train_examples], dtype=torch.long).to(device)

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")

    memorized = False
    mem_epoch = -1
    best_test_acc = 0.0
    grok_epoch = -1
    start_time = time.time()

    n_train = len(train_examples)
    n_test = len(test_examples)

    print(f"[{variant} frac={train_frac:.2f}] train={n_train} test={n_test} epochs={epochs}", flush=True)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        xb = train_tokens[perm]
        yb = train_targets[perm]

        logits = model(xb)
        if logits.dim() == 3:
            seq_lens = (xb != PAD_ID).sum(dim=1) - 1
            idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
            logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
        else:
            logits_at_pos = logits
        loss = F.cross_entropy(logits_at_pos, yb)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

        if epoch % cfg["eval_interval"] == 0 or epoch == epochs - 1:
            train_acc = compute_accuracy(model, train_examples, device)
            test_acc = compute_accuracy(model, test_examples, device)

            if not memorized and train_acc > 0.99:
                memorized = True
                mem_epoch = epoch

            if memorized and current_wd < cfg["wd_max"]:
                ramps = (epoch - mem_epoch) // cfg["wd_ramp_interval"]
                target_wd = min(cfg["wd_start"] + (ramps + 1) * cfg["wd_step"], cfg["wd_max"])
                if target_wd > current_wd:
                    for pg in optimizer.param_groups:
                        pg["weight_decay"] = target_wd
                    current_wd = target_wd

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if grok_epoch < 0 and test_acc >= 0.80:
                grok_epoch = epoch
                print(f"[{variant} frac={train_frac:.2f}] GROKKED at epoch {epoch} (test={test_acc:.4f})", flush=True)

            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "train_acc": train_acc, "test_acc": test_acc,
                    "loss": loss.item(), "wd": current_wd,
                    "elapsed_s": time.time() - start_time,
                }) + "\n")

            if epoch % 500 == 0 or (test_acc > 0.01 and epoch % 100 == 0):
                elapsed = time.time() - start_time
                print(f"  ep={epoch:6d} train={train_acc:.4f} test={test_acc:.4f} wd={current_wd:.4f} t={elapsed:.1f}s", flush=True)

            # Early stop if grokked and stable
            if test_acc >= 0.99 and epoch > grok_epoch + 500 if grok_epoch >= 0 else False:
                print(f"[{variant} frac={train_frac:.2f}] Early stop at epoch {epoch}", flush=True)
                break

    summary = {
        "variant": variant,
        "train_frac": train_frac,
        "seed": seed,
        "total_epochs": epoch,
        "best_test_acc": best_test_acc,
        "mem_epoch": mem_epoch,
        "grok_epoch": grok_epoch,
        "grokked": best_test_acc >= 0.80,
        "train_size": n_train,
        "test_size": n_test,
        "elapsed_s": time.time() - start_time,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{variant} frac={train_frac:.2f}] DONE: best={best_test_acc:.4f} grok_ep={grok_epoch} t={summary['elapsed_s']:.1f}s", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train-fraction sweep for H4")
    parser.add_argument("--variants", nargs="+", default=["RAW", "CRT-BOTH"])
    parser.add_argument("--fracs", nargs="+", type=float,
                        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiments/train_frac_sweep")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.join(project_root, args.output)
    os.makedirs(base_dir, exist_ok=True)

    all_summaries = []

    for variant in args.variants:
        for frac in args.fracs:
            run_name = f"{variant}_f{frac:.2f}_s{args.seed}"
            output_dir = os.path.join(base_dir, run_name)

            if os.path.exists(os.path.join(output_dir, "summary.json")):
                print(f"Skipping {run_name} (already done)")
                with open(os.path.join(output_dir, "summary.json")) as f:
                    all_summaries.append(json.load(f))
                continue

            summary = train_one(variant, frac, args.seed, args.epochs, output_dir)
            all_summaries.append(summary)

    # Save combined results
    with open(os.path.join(base_dir, "all_summaries.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("TRAIN-FRACTION SWEEP RESULTS")
    print("=" * 80)
    print(f"{'Variant':<12} {'Frac':>6} {'TrainN':>7} {'BestAcc':>8} {'GrokEp':>8} {'Grokked':>8}")
    print("-" * 80)

    for s in sorted(all_summaries, key=lambda x: (x["variant"], x["train_frac"])):
        grok = "✅" if s.get("grokked", s["best_test_acc"] >= 0.80) else "❌"
        grok_ep = str(s.get("grok_epoch", -1))
        print(f"{s['variant']:<12} {s['train_frac']:>6.2f} {s['train_size']:>7} {s['best_test_acc']:>8.4f} {grok_ep:>8} {grok:>8}")

    # Find N_crit
    print("\n" + "=" * 80)
    print("CRITICAL TRAINING FRACTION (N_crit)")
    print("=" * 80)

    for variant in args.variants:
        var_runs = [s for s in all_summaries if s["variant"] == variant]
        var_runs.sort(key=lambda x: x["train_frac"])

        n_crit = None
        for s in var_runs:
            if s.get("grokked", s["best_test_acc"] >= 0.80):
                n_crit = s["train_frac"]
                break

        if n_crit:
            print(f"{variant}: N_crit = {n_crit:.2f} (first grok at train_frac={n_crit})")
        else:
            print(f"{variant}: N_crit > {var_runs[-1]['train_frac']:.2f} (never grokked in sweep)")


if __name__ == "__main__":
    main()
