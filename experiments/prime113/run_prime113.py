#!/usr/bin/env python3
"""Experiment 1: M04 Prime-Order DLP Control.

Tests whether the exact M04 architecture that groks DLP in F_113* (composite order 112)
can also grok DLP in a cyclic group of PRIME order 113.

Uses the unique order-113 subgroup of F_227* (quadratic residues, since 226 = 2 × 113).

Clones M04 exactly: d_model=128, n_heads=4, n_layers=2, vocab_size=117, max_len=5.
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

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models import GrokkingTransformer


# ============================================================================
# Constants — Exact M04 Clone
# ============================================================================
M04_CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "max_len": 5,
    "vocab_size": 117,
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

# Special tokens
PAD_ID = 0
SEP_ID = 1
BOS_ID = 2
EOS_ID = 3
INT_OFFSET = 4

# Experiment parameters
P_AMBIENT = 227  # Ambient field
R_PRIME = 113    # Prime order of subgroup


# ============================================================================
# Subgroup Construction
# ============================================================================

def build_subgroup() -> List[int]:
    """Build the unique order-113 subgroup of F_227* (quadratic residues)."""
    H = [y for y in range(1, P_AMBIENT) if pow(y, R_PRIME, P_AMBIENT) == 1]
    assert len(H) == R_PRIME, f"|H| = {len(H)}, expected {R_PRIME}"
    # Verify every non-identity element has order 113
    for g in H:
        if g == 1:
            continue
        assert pow(g, R_PRIME, P_AMBIENT) == 1
        # Check no smaller order
        for d in [1, R_PRIME]:
            if d < R_PRIME and pow(g, d, P_AMBIENT) == 1:
                raise AssertionError(f"Element {g} has order {d}, not {R_PRIME}")
    return sorted(H)


def build_token_map(H: List[int]) -> Dict[int, int]:
    """Map sorted subgroup elements to token IDs 4..116."""
    token_map = {}
    for i, h in enumerate(H):
        token_map[h] = i + INT_OFFSET
    assert len(token_map) == R_PRIME
    # Verify vocab_size = 117
    max_token = max(token_map.values())
    assert max_token == R_PRIME + INT_OFFSET - 1  # 116
    return token_map


def find_canonical_generator(H: List[int]) -> int:
    """Find the smallest generator of H (smallest element != 1)."""
    for g in H:
        if g != 1:
            # Verify it generates all of H
            powers = set()
            x = 1
            for _ in range(R_PRIME):
                powers.add(x)
                x = (x * g) % P_AMBIENT
            if len(powers) == R_PRIME:
                return g
    raise RuntimeError("No generator found")


# ============================================================================
# Dataset Generation
# ============================================================================

class PrimeOrderSpec:
    __slots__ = ["g", "h", "x", "g_field", "h_field"]
    def __init__(self, g_field: int, h_field: int, x: int):
        self.g_field = g_field  # Element of H (field residue mod 227)
        self.h_field = h_field  # Element of H (field residue mod 227)
        self.x = x              # Discrete log in {0, ..., 112}
        self.g = g_field
        self.h = h_field


def generate_all_examples(H: List[int]) -> List[PrimeOrderSpec]:
    """Generate all (g, x) pairs: 112 generators × 113 exponents = 12,656 examples."""
    specs = []
    generators = [g for g in H if g != 1]
    for g in generators:
        x = 1  # g^0 = 1
        for exp in range(R_PRIME):
            h = pow(g, exp, P_AMBIENT)
            specs.append(PrimeOrderSpec(g, h, exp))
    assert len(specs) == 112 * R_PRIME
    # Verify all
    for spec in specs:
        assert pow(spec.g_field, spec.x, P_AMBIENT) == spec.h_field
    return specs


class PrimeOrderDataset(Dataset):
    def __init__(self, specs: List[PrimeOrderSpec], token_map: Dict[int, int], max_len: int):
        self.examples = []
        for spec in specs:
            g_token = token_map[spec.g_field]
            h_token = token_map[spec.h_field]
            x_token = spec.x + INT_OFFSET  # Output: x ∈ {0,...,112} → token 4..116
            tokens = [BOS_ID, g_token, SEP_ID, h_token, EOS_ID]
            padded = tokens + [PAD_ID] * (max_len - len(tokens))
            padded = padded[:max_len]
            self.examples.append((torch.tensor(padded, dtype=torch.long), x_token))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, target = self.examples[idx]
        return tokens, torch.tensor(target, dtype=torch.long)


# ============================================================================
# Splits
# ============================================================================

def split_iid(specs: List[PrimeOrderSpec], n_train: int, seed: int = 42):
    """IID split: random shuffle, take first n_train for train."""
    rng = random.Random(seed)
    indices = list(range(len(specs)))
    rng.shuffle(indices)
    train_idx = set(indices[:n_train])
    train = [specs[i] for i in sorted(train_idx)]
    test = [specs[i] for i in indices[n_train:]]
    return train, test


def split_ood(specs: List[PrimeOrderSpec], train_frac: float = 0.6, seed: int = 42):
    """OOD split: disjoint generators for train/test."""
    generators = sorted(set(s.g_field for s in specs))
    rng = random.Random(seed)
    rng.shuffle(generators)
    n_gen_train = int(len(generators) * train_frac)
    train_gens = set(generators[:n_gen_train])
    test_gens = set(generators[n_gen_train:])
    train = [s for s in specs if s.g_field in train_gens]
    test = [s for s in specs if s.g_field in test_gens]
    return train, test


# ============================================================================
# Training Loop (Cloned from trainer.py, adapted)
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


def train_prime113(
    train_ds: PrimeOrderDataset,
    test_ds: PrimeOrderDataset,
    output_dir: str,
    seed: int = 42,
    epochs: int = 100000,
    config: dict = None,
    label: str = "",
):
    """Train with exact M04 regime."""
    cfg = {**M04_CONFIG, **(config or {})}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build model — exact M04 clone
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
    print(f"[{label}] Model: {n_params:,} params, {n_core:,} core", flush=True)

    # Optimizer — exact M04 regime
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd_start"])
    current_wd = cfg["wd_start"]
    current_lr = cfg["lr"]

    # Prepare full-batch tensors
    train_tokens = torch.stack([train_ds.examples[i][0] for i in range(len(train_ds))]).to(device)
    train_targets = torch.tensor([train_ds.examples[i][1] for i in range(len(train_ds))], dtype=torch.long).to(device)

    # Output dirs
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # Save config
    full_config = {
        **cfg,
        "n_params": n_params,
        "n_core": n_core,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "seed": seed,
        "epochs": epochs,
        "label": label,
        "P_ambient": P_AMBIENT,
        "R_prime": R_PRIME,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=2)

    # Training loop
    memorized = False
    mem_epoch = -1
    best_test_acc = 0.0
    consecutive_high = 0
    early_stopped = False
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    start_time = time.time()

    print(f"[{label}] Starting training: {epochs} epochs, train={len(train_ds)}, test={len(test_ds)}", flush=True)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_ds), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), len(perm)):
            idx = perm[i:i + len(perm)]
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

            if torch.isnan(loss) or torch.isinf(loss):
                n_batches += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate
        if epoch % cfg["eval_interval"] == 0 or epoch == epochs - 1:
            train_acc, _ = compute_accuracy(model, train_ds, device)
            test_acc, _ = compute_accuracy(model, test_ds, device)

            if not memorized and train_acc > 0.99:
                memorized = True
                mem_epoch = epoch
                print(f"[{label}] MEMORIZED at epoch {epoch} (train_acc={train_acc:.4f})", flush=True)

            # WD ramp
            if memorized and current_wd < cfg["wd_max"]:
                ramps = (epoch - mem_epoch) // cfg["wd_ramp_interval"]
                target_wd = min(cfg["wd_start"] + (ramps + 1) * cfg["wd_step"], cfg["wd_max"])
                if target_wd > current_wd:
                    new_wd = target_wd
                    new_lr = current_lr * cfg["lr_drop_factor"]
                    for pg in optimizer.param_groups:
                        pg["weight_decay"] = new_wd
                        pg["lr"] = new_lr
                    print(f"[{label}] WD ramp: {current_wd:.4f} -> {new_wd:.4f} (epoch={epoch})", flush=True)
                    current_wd = new_wd
                    current_lr = new_lr

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            weight_norm = compute_weight_norm(model) if epoch % 100 == 0 else 0.0

            # Log
            if epoch % 100 == 0 or test_acc > 0.01:
                elapsed = time.time() - start_time
                print(f"[{label}] Epoch {epoch:6d} | train={train_acc:.4f} test={test_acc:.4f} "
                      f"loss={avg_loss:.4f} wd={current_wd:.4f} t={elapsed:.1f}s", flush=True)

            # Save metrics
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "train_acc": train_acc, "test_acc": test_acc,
                    "loss": avg_loss, "wd": current_wd, "wnorm": weight_norm,
                    "elapsed_s": time.time() - start_time, "best_test_acc": best_test_acc,
                }) + "\n")

            # Early stopping
            if test_acc > cfg["early_stop_threshold"]:
                consecutive_high += 1
                if consecutive_high >= cfg["early_stop_patience"]:
                    print(f"[{label}] Early stopped at epoch {epoch}", flush=True)
                    early_stopped = True
            else:
                consecutive_high = 0

        # Checkpoint
        if (epoch + 1) % cfg["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(output_dir, "checkpoints", f"epoch_{epoch:06d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "current_wd": current_wd,
                "best_test_acc": best_test_acc,
            }, ckpt_path)

        if early_stopped:
            break

    # Save final checkpoint
    final_path = os.path.join(output_dir, "checkpoints", "final.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "current_wd": current_wd,
        "best_test_acc": best_test_acc,
    }, final_path)

    # Save summary
    summary = {
        "label": label,
        "seed": seed,
        "total_epochs": epoch,
        "best_test_acc": best_test_acc,
        "mem_epoch": mem_epoch,
        "final_wd": current_wd,
        "n_params": n_params,
        "n_core": n_core,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{label}] Done. Best test acc: {best_test_acc:.4f}, epochs: {epoch}", flush=True)
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prime-Order DLP Control Experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--output-dir", type=str, default="experiments/prime113")
    parser.add_argument("--split", type=str, default="A",
                        choices=["A", "B", "OOD", "all"],
                        help="A=coverage-matched(30%), B=sample-count-matched(1612), OOD=disjoint generators")
    args = parser.parse_args()

    output_base = args.output_dir

    # Build subgroup
    print("=== Building order-113 subgroup of F_227* ===")
    H = build_subgroup()
    print(f"|H| = {len(H)}")
    print(f"H = {H[:10]}... (showing first 10)")

    # Build token map
    token_map = build_token_map(H)
    print(f"Token map: {H[0]} -> {token_map[H[0]]}, {H[1]} -> {token_map[H[1]]}, ..., {H[-1]} -> {token_map[H[-1]]}")

    # Save token map with hash
    import hashlib
    token_map_json = json.dumps({str(k): v for k, v in token_map.items()}, sort_keys=True)
    token_map_hash = hashlib.sha256(token_map_json.encode()).hexdigest()
    with open(os.path.join(output_base, "data", "subgroup_token_map.json"), "w") as f:
        json.dump({"map": {str(k): v for k, v in token_map.items()}, "sha256": token_map_hash}, f, indent=2)
    print(f"Token map saved with SHA256: {token_map_hash[:16]}...")

    # Find canonical generator
    g0 = find_canonical_generator(H)
    print(f"Canonical generator: {g0}")

    # Generate all examples
    print("=== Generating all (g, x) examples ===")
    specs = generate_all_examples(H)
    print(f"Total examples: {len(specs)}")

    # Verify all
    print("Verifying all examples...")
    for s in specs:
        assert pow(s.g_field, s.x, P_AMBIENT) == s.h_field
    print("All verified.")

    # Save split manifests
    splits_dir = os.path.join(output_base, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    runs = []
    if args.split in ["A", "all"]:
        # P113-A: Coverage-matched (30% train)
        n_train_A = int(len(specs) * 0.30)
        train_A, test_A = split_iid(specs, n_train_A, seed=args.seed)
        runs.append(("P113-A", train_A, test_A, f"{output_base}/P113-A_s{args.seed}"))
        print(f"\nP113-A: {len(train_A)} train, {len(test_A)} test (30% coverage-matched)")

    if args.split in ["B", "all"]:
        # P113-B: Sample-count matched (1612 train, same as M04)
        train_B, test_B = split_iid(specs, 1612, seed=args.seed)
        runs.append(("P113-B", train_B, test_B, f"{output_base}/P113-B_s{args.seed}"))
        print(f"P113-B: {len(train_B)} train, {len(test_B)} test (1612 sample-count-matched)")

    if args.split in ["OOD", "all"]:
        # OOD: Disjoint generators
        train_OOD, test_OOD = split_ood(specs, train_frac=0.6, seed=args.seed)
        runs.append(("P113-OOD", train_OOD, test_OOD, f"{output_base}/P113-OOD_s{args.seed}"))
        print(f"P113-OOD: {len(train_OOD)} train, {len(test_OOD)} test (disjoint generators)")

    # Save split manifests
    for label, train, test, out_dir in runs:
        manifest = {
            "label": label,
            "train_size": len(train),
            "test_size": len(test),
            "train_gens": sorted(set(s.g_field for s in train)),
            "test_gens": sorted(set(s.g_field for s in test)),
        }
        with open(os.path.join(splits_dir, f"{label}_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    # Run training
    for label, train_specs, test_specs, out_dir in runs:
        print(f"\n{'='*60}")
        print(f"=== Training {label} (seed={args.seed}) ===")
        print(f"{'='*60}")

        train_ds = PrimeOrderDataset(train_specs, token_map, M04_CONFIG["max_len"])
        test_ds = PrimeOrderDataset(test_specs, token_map, M04_CONFIG["max_len"])

        train_prime113(
            train_ds=train_ds,
            test_ds=test_ds,
            output_dir=out_dir,
            seed=args.seed,
            epochs=args.epochs,
            label=label,
        )

    print("\n=== All runs complete ===")


if __name__ == "__main__":
    main()
