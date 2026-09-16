#!/usr/bin/env python3
"""Experiment 2: M04 Algebraic-Pair DLP Representation Study.

Tests whether explicitly exposing the CRT-aligned algebraic decomposition of F_113*
changes grokking speed, sample efficiency, and learned mechanism.

For y ∈ F_113* (order q=112=16×7):
  y16 = y^7 mod 113   (order divides 16)
  y7  = y^16 mod 113  (order divides 7)

The pair (y16, y7) is an algebraic CRT representation of y. No discrete log used.

Representation variants:
  - RAW:        [BOS, g, SEP, h, EOS]
  - CRT-BOTH:   [BOS, g16, SEP, g7, SEP, h16, SEP, h7, EOS]
  - CRT-16:     [BOS, g16, SEP, h16, EOS]
  - CRT-7:      [BOS, g7, SEP, h7, EOS]
  - RAW+CRT:    [BOS, g, SEP, h, SEP, g16, SEP, g7, SEP, h16, SEP, h7, EOS]
  - SCRAMBLED:  Like CRT-BOTH but with scrambled token mapping

All variants use max_len=13 (longest = RAW+CRT) to keep parameter count identical.
Architecture: exact M04 clone (d_model=128, n_heads=4, n_layers=2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models import GrokkingTransformer
from src.trainer import L6Generator, GrokkingTokenizer, ProblemSpec, split_specs


# ============================================================================
# Constants — Exact M04 Clone
# ============================================================================
M04_CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "max_len": 13,  # Longest variant = RAW+CRT (13 tokens); all variants use same
    "vocab_size": 117,  # p + 4 = 113 + 4
    "lr": 0.001,
    "wd_start": 0.05,
    "wd_max": 0.30,
    "wd_step": 0.05,
    "wd_acc_threshold": 0.05,
    "wd_ramp_interval": 1000,
    "lr_drop_factor": 1.0,
    "eval_interval": 10,
    "checkpoint_interval": 500,
    "early_stop_patience": 100,
    "early_stop_threshold": 0.99,
    "grad_clip": 1.0,
}

PAD_ID = 0
SEP_ID = 1
BOS_ID = 2
EOS_ID = 3
INT_OFFSET = 4

P = 113
Q = 112  # p - 1 = 16 * 7


# ============================================================================
# Algebraic Projections
# ============================================================================

def project_16(y: int) -> int:
    """Project to order-16 component: y^7 mod 113. Order divides 16."""
    return pow(y, 7, P)

def project_7(y: int) -> int:
    """Project to order-7 component: y^16 mod 113. Order divides 7."""
    return pow(y, 16, P)


# ============================================================================
# Representation Variants
# ============================================================================

REPRESENTATIONS = ["RAW", "CRT-BOTH", "CRT-16", "CRT-7", "RAW+CRT", "SCRAMBLED"]

def encode_representation(g: int, h: int, x: int, variant: str,
                          scramble_map: Dict[int, int] = None) -> Tuple[List[int], int]:
    """Encode a (g, h, x) example in the given representation.

    Returns (tokens, target_token).
    """
    g16 = project_16(g)
    g7 = project_7(g)
    h16 = project_16(h)
    h7 = project_7(h)

    def tok(val: int) -> int:
        return val + INT_OFFSET

    target = tok(x)  # x ∈ {0,...,112} → token 4..116

    if variant == "RAW":
        tokens = [BOS_ID, tok(g), SEP_ID, tok(h), EOS_ID]

    elif variant == "CRT-BOTH":
        tokens = [BOS_ID, tok(g16), SEP_ID, tok(g7), SEP_ID, tok(h16), SEP_ID, tok(h7), EOS_ID]

    elif variant == "CRT-16":
        tokens = [BOS_ID, tok(g16), SEP_ID, tok(h16), EOS_ID]

    elif variant == "CRT-7":
        tokens = [BOS_ID, tok(g7), SEP_ID, tok(h7), EOS_ID]

    elif variant == "RAW+CRT":
        tokens = [BOS_ID, tok(g), SEP_ID, tok(h), SEP_ID,
                  tok(g16), SEP_ID, tok(g7), SEP_ID,
                  tok(h16), SEP_ID, tok(h7), EOS_ID]

    elif variant == "SCRAMBLED":
        # Apply random permutation to component values
        s_g16 = scramble_map.get(g16, g16)
        s_g7 = scramble_map.get(g7, g7)
        s_h16 = scramble_map.get(h16, h16)
        s_h7 = scramble_map.get(h7, h7)
        tokens = [BOS_ID, tok(s_g16), SEP_ID, tok(s_g7), SEP_ID, tok(s_h16), SEP_ID, tok(s_h7), EOS_ID]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return tokens, target


# ============================================================================
# Dataset
# ============================================================================

class AlgebraicPairDataset(Dataset):
    def __init__(self, specs: List[ProblemSpec], variant: str, max_len: int,
                 scramble_map: Dict[int, int] = None):
        self.examples = []
        for spec in specs:
            g, h, x = spec.metadata["g"], spec.metadata["h"], spec.target
            tokens, target = encode_representation(g, h, x, variant, scramble_map)
            padded = tokens + [PAD_ID] * (max_len - len(tokens))
            padded = padded[:max_len]
            self.examples.append((torch.tensor(padded, dtype=torch.long), target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, target = self.examples[idx]
        return tokens, torch.tensor(target, dtype=torch.long)


# ============================================================================
# Training (shared with Experiment 1, adapted)
# ============================================================================

def compute_accuracy(model, dataset, device, batch_size=512):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            tokens = torch.stack([b[0] for b in batch]).to(device)
            targets = torch.stack([b[1] for b in batch]).to(device)
            logits = model(tokens)
            if logits.dim() == 3:
                seq_lens = (tokens != PAD_ID).sum(dim=1) - 1
                idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
            else:
                logits_at_pos = logits
            preds = logits_at_pos.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(batch)
    return correct / max(total, 1), 0.0


def compute_weight_norm(model):
    total = 0.0
    for p in model.parameters():
        total += p.data.norm().item() ** 2
    return math.sqrt(total)


def train_variant(
    train_ds: AlgebraicPairDataset,
    test_ds: AlgebraicPairDataset,
    output_dir: str,
    variant: str,
    seed: int = 42,
    epochs: int = 100000,
    config: dict = None,
):
    """Train one representation variant with exact M04 regime."""
    cfg = {**M04_CONFIG, **(config or {})}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build model — exact M04 clone (same param count for all variants)
    model = GrokkingTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_len=cfg["max_len"],
        dropout=0.0,
    ).to(device)

    n_params = model.count_parameters()
    n_core = model.core_parameters()
    print(f"[{variant}] Model: {n_params:,} params, {n_core:,} core", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd_start"])
    current_wd = cfg["wd_start"]
    current_lr = cfg["lr"]

    train_tokens = torch.stack([train_ds.examples[i][0] for i in range(len(train_ds))]).to(device)
    train_targets = torch.tensor([train_ds.examples[i][1] for i in range(len(train_ds))], dtype=torch.long).to(device)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    full_config = {**cfg, "n_params": n_params, "n_core": n_core,
                   "train_size": len(train_ds), "test_size": len(test_ds),
                   "seed": seed, "epochs": epochs, "variant": variant}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=2)

    memorized = False
    mem_epoch = -1
    best_test_acc = 0.0
    consecutive_high = 0
    early_stopped = False
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    start_time = time.time()

    print(f"[{variant}] Starting: {epochs} epochs, train={len(train_ds)}, test={len(test_ds)}", flush=True)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_ds), device=device)
        epoch_loss = 0.0
        n_batches = 0

        idx = perm
        xb = train_tokens[idx]
        yb = train_targets[idx]

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
            epoch_loss = loss.item()
            n_batches = 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % cfg["eval_interval"] == 0 or epoch == epochs - 1:
            train_acc, _ = compute_accuracy(model, train_ds, device)
            test_acc, _ = compute_accuracy(model, test_ds, device)

            if not memorized and train_acc > 0.99:
                memorized = True
                mem_epoch = epoch
                print(f"[{variant}] MEMORIZED at epoch {epoch}", flush=True)

            if memorized and current_wd < cfg["wd_max"]:
                ramps = (epoch - mem_epoch) // cfg["wd_ramp_interval"]
                target_wd = min(cfg["wd_start"] + (ramps + 1) * cfg["wd_step"], cfg["wd_max"])
                if target_wd > current_wd:
                    for pg in optimizer.param_groups:
                        pg["weight_decay"] = target_wd
                        pg["lr"] = current_lr * cfg["lr_drop_factor"]
                    print(f"[{variant}] WD ramp: {current_wd:.4f} -> {target_wd:.4f} (epoch={epoch})", flush=True)
                    current_wd = target_wd

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if epoch % 100 == 0 or test_acc > 0.01:
                elapsed = time.time() - start_time
                print(f"[{variant}] Epoch {epoch:6d} | train={train_acc:.4f} test={test_acc:.4f} "
                      f"loss={avg_loss:.4f} wd={current_wd:.4f} t={elapsed:.1f}s", flush=True)

            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "train_acc": train_acc, "test_acc": test_acc,
                    "loss": avg_loss, "wd": current_wd,
                    "elapsed_s": time.time() - start_time, "best_test_acc": best_test_acc,
                }) + "\n")

            if test_acc > cfg["early_stop_threshold"]:
                consecutive_high += 1
                if consecutive_high >= cfg["early_stop_patience"]:
                    print(f"[{variant}] Early stopped at epoch {epoch}", flush=True)
                    early_stopped = True
            else:
                consecutive_high = 0

        if (epoch + 1) % cfg["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(output_dir, "checkpoints", f"epoch_{epoch:06d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "current_wd": current_wd,
                "best_test_acc": best_test_acc,
            }, ckpt_path)

        if early_stopped:
            break

    # Save final
    final_path = os.path.join(output_dir, "checkpoints", "final.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "current_wd": current_wd,
        "best_test_acc": best_test_acc,
    }, final_path)

    summary = {
        "variant": variant, "seed": seed, "total_epochs": epoch,
        "best_test_acc": best_test_acc, "mem_epoch": mem_epoch,
        "n_params": n_params, "n_core": n_core,
        "train_size": len(train_ds), "test_size": len(test_ds),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{variant}] Done. Best test acc: {best_test_acc:.4f}, epochs: {epoch}", flush=True)
    return summary


# ============================================================================
# Theoretical Ceiling Verification
# ============================================================================

def verify_ceiling(specs: List[ProblemSpec], variant: str) -> Dict:
    """Verify the information-theoretic ceiling for single-component variants."""
    if variant == "CRT-16":
        # x mod 16 is determined, but x mod 112 has 7 lifts → ceiling ≈ 1/7 = 14.3%
        mod_val = 16
        n_lifts = Q // mod_val  # 7
    elif variant == "CRT-7":
        # x mod 7 is determined, but x mod 112 has 16 lifts → ceiling ≈ 1/16 = 6.25%
        mod_val = 7
        n_lifts = Q // mod_val  # 16
    else:
        return {}

    # For each example, check if the CRT component uniquely determines x
    # Group by (g16, h16) or (g7, h7) and see how many x values map to same group
    groups = {}
    for spec in specs:
        g, h, x = spec.metadata["g"], spec.metadata["h"], spec.target
        if variant == "CRT-16":
            key = (project_16(g), project_16(h))
        else:
            key = (project_7(g), project_7(h))
        if key not in groups:
            groups[key] = []
        groups[key].append(x)

    # Average ambiguity
    avg_ambiguity = np.mean([len(set(xs)) for xs in groups.values()])
    max_acc = 1.0 / avg_ambiguity if avg_ambiguity > 0 else 0.0

    return {
        "variant": variant,
        "mod_val": mod_val,
        "n_lifts": n_lifts,
        "theoretical_ceiling": 1.0 / n_lifts,
        "avg_ambiguity": float(avg_ambiguity),
        "estimated_max_acc": float(max_acc),
        "n_groups": len(groups),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Algebraic-Pair DLP Representation Study")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--output-dir", type=str, default="experiments/algebraic_pairs")
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated variants or 'all'")
    parser.add_argument("--train-frac", type=float, default=0.30,
                        help="Train fraction (default 0.30, same as M04)")
    args = parser.parse_args()

    output_base = args.output_dir

    # Select variants
    if args.variants == "all":
        variants = REPRESENTATIONS[:]
    else:
        variants = args.variants.split(",")

    print(f"=== Algebraic-Pair DLP Representation Study ===")
    print(f"Variants: {variants}")
    print(f"Train fraction: {args.train_frac}")
    print(f"Seeds: {args.seed}")
    print(f"Epochs: {args.epochs}")

    # Generate the EXACT original M04 dataset
    print("\n=== Generating M04 dataset (F_113*, q=112) ===")
    gen = L6Generator(P, seed=args.seed)
    tokenizer = GrokkingTokenizer(P)
    space_size = gen.solution_space_size()
    n_train = int(space_size * args.train_frac)
    n_total = min(n_train * 3, space_size)
    print(f"Space size: {space_size}, n_train: {n_train}, n_total: {n_total}")

    specs = gen.generate(n_total) if n_total < space_size else gen.generate_all()
    train_specs, test_specs = split_specs(specs, n_train, seed=args.seed)
    print(f"Train: {len(train_specs)}, Test: {len(test_specs)}")

    # Verify all examples
    for s in train_specs + test_specs:
        assert gen.verify(s), f"Verification failed for g={s.metadata['g']}, h={s.metadata['h']}, x={s.target}"
    print("All examples verified.")

    # Build scrambled map (deterministic random permutation of {0,...,112})
    rng = random.Random(12345)  # Fixed seed for reproducibility
    perm = list(range(P))
    rng.shuffle(perm)
    scramble_map = {i: perm[i] for i in range(P)}
    # Verify it's a valid permutation
    assert len(set(scramble_map.values())) == P

    # Save scramble map
    with open(os.path.join(output_base, "data", "scramble_map.json"), "w") as f:
        json.dump(scramble_map, f, indent=2)

    # Verify theoretical ceilings
    print("\n=== Theoretical Ceiling Verification ===")
    for variant in ["CRT-16", "CRT-7"]:
        if variant in variants:
            ceiling = verify_ceiling(train_specs + test_specs, variant)
            print(f"{variant}: ceiling={ceiling['theoretical_ceiling']:.4f}, "
                  f"avg_ambiguity={ceiling['avg_ambiguity']:.2f}, "
                  f"estimated_max_acc={ceiling['estimated_max_acc']:.4f}")

    # Save split manifest
    splits_dir = os.path.join(output_base, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    with open(os.path.join(splits_dir, "split_manifest.json"), "w") as f:
        json.dump({
            "train_size": len(train_specs),
            "test_size": len(test_specs),
            "train_frac": args.train_frac,
            "seed": args.seed,
        }, f, indent=2)

    # Run training for each variant
    results = []
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"=== Training {variant} (seed={args.seed}) ===")
        print(f"{'='*60}")

        out_dir = os.path.join(output_base, f"{variant}_s{args.seed}")
        os.makedirs(out_dir, exist_ok=True)

        train_ds = AlgebraicPairDataset(train_specs, variant, M04_CONFIG["max_len"],
                                         scramble_map if variant == "SCRAMBLED" else None)
        test_ds = AlgebraicPairDataset(test_specs, variant, M04_CONFIG["max_len"],
                                        scramble_map if variant == "SCRAMBLED" else None)

        summary = train_variant(
            train_ds=train_ds,
            test_ds=test_ds,
            output_dir=out_dir,
            variant=variant,
            seed=args.seed,
            epochs=args.epochs,
        )
        results.append(summary)

    # Save comparison
    print(f"\n{'='*60}")
    print("=== RESULTS SUMMARY ===")
    print(f"{'='*60}")
    print(f"{'Variant':<15} {'Train':>6} {'Test':>6} {'MemEpoch':>8} {'BestAcc':>8} {'Epochs':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['variant']:<15} {r['train_size']:>6} {r['test_size']:>6} "
              f"{r['mem_epoch']:>8} {r['best_test_acc']:>8.4f} {r['total_epochs']:>8}")

    comparison_path = os.path.join(output_base, "representation_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved: {comparison_path}")


if __name__ == "__main__":
    main()
