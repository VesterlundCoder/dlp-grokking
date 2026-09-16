#!/usr/bin/env python3
"""Multi-Curve DLP Grokking Trainer.

Trains a single transformer on DLP problems from multiple primes simultaneously.
Each problem is prefixed with a curve ID token so the model knows which prime/curve
it's operating on. Tests whether the model learns a *general* DLP algorithm or
curve-specific algorithms.

Token format: [BOS, CURVE_ID, tok(g), SEP, tok(h), EOS]
Target:       tok(x)  (the discrete log, in the shared integer token range)

Vocabulary layout:
  [PAD=0, SEP=1, BOS=2, EOS=3, CURVE_0=4, CURVE_1=5, ..., CURVE_{K-1},
   0, 1, 2, ..., max_p - 1]
  vocab_size = 4 + n_curves + max_p

Usage:
  python run_multicurve.py --primes 11 13 --model-id M04 --epochs 20000
  python run_multicurve.py --primes 11 13 17 23 --model-id M04 --epochs 50000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path for model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import GrokkingTransformer, compute_wd_max


# ============================================================================
# Number Theory
# ============================================================================

def _prime_factors(n: int) -> List[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _primitive_roots(p: int) -> List[int]:
    if p == 2:
        return [1]
    roots = []
    factors = set(_prime_factors(p - 1))
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            roots.append(g)
    return roots


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def coprime_factorization(q: int) -> Tuple[int, int]:
    """Factor q into two coprime factors (a, b) with a*b=q, gcd(a,b)=1,
    minimizing the ratio max(a,b)/min(a,b) for a balanced split."""
    from itertools import product as _product
    n = q
    unique_primes = []
    prime_powers = {}
    d = 2
    while d * d <= n:
        if n % d == 0:
            unique_primes.append(d)
            e = 0
            while n % d == 0:
                e += 1
                n //= d
            prime_powers[d] = e
        d += 1
    if n > 1:
        unique_primes.append(n)
        prime_powers[n] = 1
    if len(unique_primes) < 2:
        raise ValueError(f"q={q} has only one prime factor; no coprime split possible.")
    best = None
    for assignment in _product([0, 1], repeat=len(unique_primes)):
        a, b = 1, 1
        for i, pr in enumerate(unique_primes):
            if assignment[i] == 0:
                a *= pr ** prime_powers[pr]
            else:
                b *= pr ** prime_powers[pr]
        if a == 1 or b == 1:
            continue
        ratio = max(a, b) / min(a, b)
        if best is None or ratio < best[0]:
            best = (ratio, min(a, b), max(a, b))
    if best is None:
        raise ValueError(f"q={q}: could not find a coprime split.")
    return best[1], best[2]


def project_factor(y: int, p: int, factor: int) -> int:
    """Project y to the order-`factor` subgroup via y^((p-1)/factor) mod p."""
    return pow(y, (p - 1) // factor, p)


# ============================================================================
# Multi-Curve Problem Generation
# ============================================================================

class ProblemSpec:
    __slots__ = ["inputs", "target", "metadata"]
    def __init__(self, inputs: List[int], target: int, metadata: Dict):
        self.inputs = inputs
        self.target = target
        self.metadata = metadata


class SingleCurveGenerator:
    """L6: Finite field DLP — g^x = h mod p, for a single prime."""
    def __init__(self, p: int, seed: int = 42):
        self.p = p
        self.seed = seed
        self.rng = random.Random(seed)
        self._cached_roots = None

    def _get_roots(self):
        if self._cached_roots is None:
            self._cached_roots = _primitive_roots(self.p)
        return self._cached_roots

    def generate_all(self) -> List[ProblemSpec]:
        """Generate all unique (g, h) -> x specs."""
        roots = self._get_roots()
        specs = []
        for g in roots:
            x = 1
            for exp in range(1, self.p):
                h = pow(g, exp, self.p)
                specs.append(ProblemSpec(
                    inputs=[g, h],
                    target=exp,
                    metadata={"g": g, "h": h, "x": exp, "p": self.p},
                ))
                x = (x * g) % self.p
        return specs

    def solution_space_size(self) -> int:
        roots = self._get_roots()
        return (self.p - 1) * len(roots)


class MultiCurveGenerator:
    """Generates DLP problems from multiple primes, each tagged with a curve ID."""
    def __init__(self, primes: List[int], seed: int = 42):
        self.primes = list(primes)
        self.seed = seed
        self.n_curves = len(primes)
        self.generators = [SingleCurveGenerator(p, seed=seed + i)
                           for i, p in enumerate(primes)]
        self.curve_ids = list(range(self.n_curves))

    def generate_all(self) -> List[ProblemSpec]:
        """Generate all specs from all curves, tagged with curve_id in metadata."""
        all_specs = []
        for curve_id, gen in enumerate(self.generators):
            specs = gen.generate_all()
            for s in specs:
                s.metadata["curve_id"] = curve_id
                all_specs.append(s)
        return all_specs

    def solution_space_size(self, curve_id: int) -> int:
        return self.generators[curve_id].solution_space_size()

    def total_space_size(self) -> int:
        return sum(self.solution_space_size(i) for i in range(self.n_curves))


# ============================================================================
# Multi-Curve Tokenizer
# ============================================================================

class MultiCurveTokenizer:
    """Unified vocabulary for multi-curve DLP.

    Vocab layout:
      [PAD=0, SEP=1, BOS=2, EOS=3,
       CURVE_0=4, CURVE_1=5, ..., CURVE_{K-1}=4+K-1,
       0, 1, 2, ..., max_p-1]

    vocab_size = 4 + n_curves + max_p
    int_offset = 4 + n_curves

    Supports RAW and CRT-BOTH variants:
      RAW:       [BOS, CURVE_ID, tok(g), SEP, tok(h), EOS]                    (max_len=6)
      CRT-BOTH:  [BOS, CURVE_ID, tok(g_b), SEP, tok(g_a), SEP,
                  tok(h_b), SEP, tok(h_a), EOS]                               (max_len=10)
    """
    def __init__(self, primes: List[int], variant: str = "RAW",
                 crt_factors: Optional[Dict[int, Tuple[int, int]]] = None):
        self.primes = list(primes)
        self.n_curves = len(primes)
        self.max_p = max(primes)
        self.variant = variant
        self.pad_id = 0
        self.sep_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.curve_offset = 4
        self.int_offset = 4 + self.n_curves
        self.vocab_size = self.int_offset + self.max_p
        # CRT factors per curve: {curve_id: (a, b)} where a < b, a*b = p-1, gcd(a,b)=1
        self.crt_factors = crt_factors or {}
        if variant == "CRT-BOTH":
            self.max_len = 10  # BOS, CURVE_ID, g_b, SEP, g_a, SEP, h_b, SEP, h_a, EOS
        else:
            self.max_len = 6   # BOS, CURVE_ID, g, SEP, h, EOS

    def curve_token(self, curve_id: int) -> int:
        return self.curve_offset + curve_id

    def int_token(self, val: int) -> int:
        return self.int_offset + (val % self.max_p)

    def encode(self, spec: ProblemSpec) -> Tuple[List[int], int]:
        curve_id = spec.metadata["curve_id"]
        g, h, x = spec.metadata["g"], spec.metadata["h"], spec.target
        p = spec.metadata["p"]

        if self.variant == "CRT-BOTH":
            a, b = self.crt_factors[curve_id]  # (smaller, larger)
            g_a = project_factor(g, p, a)
            g_b = project_factor(g, p, b)
            h_a = project_factor(h, p, a)
            h_b = project_factor(h, p, b)
            tokens = [
                self.bos_id,
                self.curve_token(curve_id),
                self.int_token(g_b),
                self.sep_id,
                self.int_token(g_a),
                self.sep_id,
                self.int_token(h_b),
                self.sep_id,
                self.int_token(h_a),
                self.eos_id,
            ]
        else:  # RAW
            tokens = [
                self.bos_id,
                self.curve_token(curve_id),
                self.int_token(g),
                self.sep_id,
                self.int_token(h),
                self.eos_id,
            ]
        target = self.int_token(x)
        return tokens, target


# ============================================================================
# Dataset
# ============================================================================

class MultiCurveDataset:
    def __init__(self, specs: List[ProblemSpec], tokenizer: MultiCurveTokenizer):
        self.tokenizer = tokenizer
        self.max_len = tokenizer.max_len
        self.examples = []
        for spec in specs:
            tokens, target = tokenizer.encode(spec)
            padded = tokens + [tokenizer.pad_id] * (self.max_len - len(tokens))
            padded = padded[:self.max_len]
            curve_id = spec.metadata["curve_id"]
            self.examples.append((
                torch.tensor(padded, dtype=torch.long),
                target,
                curve_id,
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, target, curve_id = self.examples[idx]
        return tokens, torch.tensor(target, dtype=torch.long), curve_id


def split_multicurve(
    specs: List[ProblemSpec],
    train_frac: float,
    seed: int = 42,
) -> Tuple[List[ProblemSpec], List[ProblemSpec]]:
    """Split specs per-curve: each curve gets its own train/test split."""
    rng = random.Random(seed)
    train, test = [], []
    # Group by curve_id
    by_curve: Dict[int, List[ProblemSpec]] = {}
    for s in specs:
        cid = s.metadata["curve_id"]
        by_curve.setdefault(cid, []).append(s)
    for cid in sorted(by_curve.keys()):
        curve_specs = by_curve[cid]
        indices = list(range(len(curve_specs)))
        rng.shuffle(indices)
        n_train = max(1, int(len(curve_specs) * train_frac))
        train_idx = set(indices[:n_train])
        for i in indices:
            if i in train_idx:
                train.append(curve_specs[i])
            else:
                test.append(curve_specs[i])
    return train, test


# ============================================================================
# Per-Curve Accuracy
# ============================================================================

def compute_accuracy_per_curve(
    model: nn.Module,
    dataset: MultiCurveDataset,
    tokenizer: MultiCurveTokenizer,
    device,
    batch_size: int = 512,
) -> Tuple[float, Dict[int, float]]:
    """Compute overall accuracy and per-curve accuracy."""
    model.eval()
    correct_total = 0
    total_total = 0
    per_curve_correct: Dict[int, int] = {}
    per_curve_total: Dict[int, int] = {}

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset.examples[j] for j in range(i, min(i + batch_size, len(dataset)))]
            tokens = torch.stack([b[0] for b in batch]).to(device)
            targets = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
            curve_ids = [b[2] for b in batch]

            logits = model(tokens)
            if logits.dim() == 3:
                seq_lens = (tokens != tokenizer.pad_id).sum(dim=1) - 1
                idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
            else:
                logits_at_pos = logits
            preds = logits_at_pos.argmax(dim=-1)

            for j in range(len(batch)):
                cid = curve_ids[j]
                ok = (preds[j] == targets[j]).item()
                per_curve_correct[cid] = per_curve_correct.get(cid, 0) + (1 if ok else 0)
                per_curve_total[cid] = per_curve_total.get(cid, 0) + 1
                correct_total += (1 if ok else 0)
                total_total += 1

    model.train()
    overall = correct_total / total_total if total_total > 0 else 0.0
    per_curve = {cid: per_curve_correct.get(cid, 0) / per_curve_total.get(cid, 1)
                 for cid in per_curve_total}
    return overall, per_curve


def compute_weight_norm(model) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_sq += p.data.pow(2).sum().item()
    return math.sqrt(total_sq)


# ============================================================================
# Model Grid (subset)
# ============================================================================

MODEL_GRID = {
    "M01": {"d_model": 56,  "n_heads": 2},
    "M02": {"d_model": 80,  "n_heads": 4},
    "M03": {"d_model": 104, "n_heads": 4},
    "M04": {"d_model": 128, "n_heads": 4},
    "M05": {"d_model": 160, "n_heads": 5},
    "M06": {"d_model": 208, "n_heads": 8},
    "M07": {"d_model": 256, "n_heads": 8},
    "M08": {"d_model": 320, "n_heads": 10},
}


# ============================================================================
# Training
# ============================================================================

def train_multicurve(
    primes: List[int],
    model_id: str = "M04",
    n_layers: int = 2,
    epochs: int = 20000,
    lr: float = 1e-3,
    wd_start: float = 0.05,
    wd_step: float = 0.05,
    wd_max_override: Optional[float] = None,
    train_frac: float = 0.30,
    lr_drop_factor: float = 0.1,
    seed: int = 42,
    output_dir: str = "results/multicurve",
    checkpoint_interval: int = 5000,
    eval_interval: int = 50,
    early_stop_patience: int = 100,
    early_stop_threshold: float = 0.99,
    wd_ramp_interval: int = 1000,
    variant: str = "RAW",
):
    """Train a multi-curve DLP model."""

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    random.seed(seed)
    torch.manual_seed(seed)

    # Validate primes
    for p in primes:
        if not _is_prime(p):
            raise ValueError(f"{p} is not prime")
    n_curves = len(primes)
    max_p = max(primes)

    print(f"Device: {device}", flush=True)
    print(f"Curves: {primes} ({n_curves} curves, max_p={max_p})", flush=True)
    print(f"Variant: {variant}", flush=True)

    # Compute CRT factors per curve if using CRT-BOTH
    crt_factors: Optional[Dict[int, Tuple[int, int]]] = None
    if variant == "CRT-BOTH":
        crt_factors = {}
        for i, p in enumerate(primes):
            a, b = coprime_factorization(p - 1)
            crt_factors[i] = (a, b)
            print(f"  Curve {i} (p={p}): CRT factors {a} x {b}", flush=True)

    # Generate dataset
    gen = MultiCurveGenerator(primes, seed=seed)
    tokenizer = MultiCurveTokenizer(primes, variant=variant, crt_factors=crt_factors)
    max_len = tokenizer.max_len

    all_specs = gen.generate_all()
    total_space = gen.total_space_size()
    print(f"Total solution space: {total_space} across {n_curves} curves", flush=True)
    for i, p in enumerate(primes):
        space = gen.solution_space_size(i)
        n_roots = len(_primitive_roots(p))
        print(f"  Curve {i} (p={p}): space={space}, {n_roots} primitive roots", flush=True)

    train_specs, test_specs = split_multicurve(all_specs, train_frac, seed=seed)
    train_ds = MultiCurveDataset(train_specs, tokenizer)
    test_ds = MultiCurveDataset(test_specs, tokenizer)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)
    print(f"Vocab size: {tokenizer.vocab_size}, max_len: {max_len}", flush=True)

    # Build model
    cfg = MODEL_GRID[model_id]
    d_model = cfg["d_model"]
    n_heads = cfg["n_heads"]
    model = GrokkingTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
    ).to(device)

    n_params = model.count_parameters()
    n_core = model.core_parameters()

    if wd_max_override is not None:
        wd_max = wd_max_override
    else:
        wd_max = compute_wd_max(n_core)

    print(f"Model: {model_id}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}", flush=True)
    print(f"Params: {n_params:,} total, {n_core:,} core", flush=True)
    print(f"WD: start={wd_start}, max={wd_max:.4f}, step={wd_step}", flush=True)

    # Output dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    config = {
        "primes": primes,
        "n_curves": n_curves,
        "variant": variant,
        "crt_factors": {str(k): list(v) for k, v in crt_factors.items()} if crt_factors else None,
        "model_id": model_id,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "vocab_size": tokenizer.vocab_size,
        "max_len": max_len,
        "n_params": n_params,
        "n_core": n_core,
        "epochs": epochs,
        "lr": lr,
        "wd_start": wd_start,
        "wd_max": wd_max,
        "wd_step": wd_step,
        "wd_ramp_interval": wd_ramp_interval,
        "lr_drop_factor": lr_drop_factor,
        "train_frac": train_frac,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "seed": seed,
        "eval_interval": eval_interval,
        "checkpoint_interval": checkpoint_interval,
        "early_stop_patience": early_stop_patience,
        "early_stop_threshold": early_stop_threshold,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Optimizer
    current_wd = wd_start
    current_lr = lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_wd)

    # Pre-load data to device
    train_tokens = torch.stack([train_ds.examples[i][0] for i in range(len(train_ds))]).to(device)
    train_targets = torch.tensor([train_ds.examples[i][1] for i in range(len(train_ds))], dtype=torch.long).to(device)

    # Metrics
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    all_metrics: List[Dict] = []

    # Training loop
    start_time = time.time()
    memorized = False
    mem_epoch = -1
    best_test_acc = 0.0
    early_stopped = False
    consecutive_high = 0

    print(f"\nStarting training: {epochs} epochs", flush=True)
    print(f"{'='*80}", flush=True)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_ds), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), len(perm)):  # full-batch
            idx = perm
            xb = train_tokens[idx]
            yb = train_targets[idx]

            logits = model(xb)
            if logits.dim() == 3:
                seq_lens = (xb != tokenizer.pad_id).sum(dim=1) - 1
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Evaluate
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            train_acc, train_per_curve = compute_accuracy_per_curve(
                model, train_ds, tokenizer, device)
            test_acc, test_per_curve = compute_accuracy_per_curve(
                model, test_ds, tokenizer, device)

            # Detect memorization
            if not memorized and train_acc > 0.99:
                memorized = True
                mem_epoch = epoch
                print(f"  MEMORIZED at epoch {epoch} (train_acc={train_acc:.4f})", flush=True)

            # Post-memorization WD ramp
            if memorized and current_wd < wd_max:
                ramps_since_mem = (epoch - mem_epoch) // wd_ramp_interval
                target_wd_steps = ramps_since_mem + 1
                target_wd = min(wd_start + target_wd_steps * wd_step, wd_max)
                if target_wd > current_wd:
                    new_wd = target_wd
                    new_lr = current_lr * lr_drop_factor
                    for pg in optimizer.param_groups:
                        pg['weight_decay'] = new_wd
                        pg['lr'] = new_lr
                    print(f"  WD ramp: {current_wd:.4f} -> {new_wd:.4f} | "
                          f"LR: {current_lr:.2e} -> {new_lr:.2e} (epoch={epoch})",
                          flush=True)
                    current_wd = new_wd
                    current_lr = new_lr
        else:
            train_acc = float("nan")
            test_acc = float("nan")
            train_per_curve = {}
            test_per_curve = {}

        if not math.isnan(test_acc) and test_acc > best_test_acc:
            best_test_acc = test_acc

        # Log metrics
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            metric_entry = {
                "epoch": epoch,
                "train_acc": train_acc if not math.isnan(train_acc) else None,
                "test_acc": test_acc if not math.isnan(test_acc) else None,
                "train_loss": avg_train_loss,
                "weight_decay": current_wd,
                "elapsed_s": elapsed,
                "best_test_acc": best_test_acc,
                "per_curve_test": {str(k): v for k, v in test_per_curve.items()},
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metric_entry) + "\n")
            all_metrics.append(metric_entry)

        # Progress log
        if epoch % 100 == 0 or epoch == epochs - 1 or (
            not math.isnan(test_acc) and test_acc > 0.1 and epoch % eval_interval == 0
        ):
            elapsed = time.time() - start_time
            ta = train_acc if not math.isnan(train_acc) else 0.0
            te = test_acc if not math.isnan(test_acc) else 0.0
            curve_str = " | ".join(
                f"c{i}(p={primes[i]})={test_per_curve.get(i, 0):.2f}"
                for i in range(n_curves)
            )
            print(f"  Ep {epoch:6d} | tr={ta:.4f} te={te:.4f} "
                  f"loss={avg_train_loss:.4f} wd={current_wd:.4f} "
                  f"t={elapsed:.1f}s | {curve_str}", flush=True)

        # Checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(output_dir, "checkpoints", f"epoch_{epoch:06d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "current_wd": current_wd,
                "best_test_acc": best_test_acc,
            }, ckpt_path)

        # Early stopping
        if not math.isnan(test_acc):
            if test_acc > early_stop_threshold:
                consecutive_high += 1
                if consecutive_high >= early_stop_patience:
                    print(f"\nEarly stopped at epoch {epoch} "
                          f"(test_acc > {early_stop_threshold} for {consecutive_high} evals)",
                          flush=True)
                    early_stopped = True
            else:
                consecutive_high = 0

        if early_stopped:
            break

    # Final checkpoint
    final_path = os.path.join(output_dir, "checkpoints", "final.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "current_wd": current_wd,
        "best_test_acc": best_test_acc,
    }, final_path)

    # Summary
    elapsed = time.time() - start_time
    final_test_per_curve = {}
    if all_metrics:
        last = all_metrics[-1]
        if last.get("per_curve_test"):
            final_test_per_curve = {int(k): v for k, v in last["per_curve_test"].items()}

    summary = {
        "primes": primes,
        "n_curves": n_curves,
        "variant": variant,
        "model_id": model_id,
        "n_params": n_params,
        "n_core": n_core,
        "best_test_acc": best_test_acc,
        "total_epochs": epoch + 1,
        "final_wd": current_wd,
        "seed": seed,
        "elapsed_s": elapsed,
        "per_curve_best": final_test_per_curve,
        "all_grokked": all(v > 0.95 for v in final_test_per_curve.values()) if final_test_per_curve else False,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}", flush=True)
    print(f"Training complete: {epoch + 1} epochs in {elapsed:.1f}s", flush=True)
    print(f"Best test acc: {best_test_acc:.4f}", flush=True)
    print(f"Per-curve final:", flush=True)
    for i, p in enumerate(primes):
        acc = final_test_per_curve.get(i, 0.0)
        status = "GROKKED" if acc > 0.95 else "FAILED"
        print(f"  Curve {i} (p={p}): {acc:.4f} {status}", flush=True)
    all_grokked = all(v > 0.95 for v in final_test_per_curve.values()) if final_test_per_curve else False
    print(f"All curves grokked: {all_grokked}", flush=True)
    print(f"{'='*80}", flush=True)

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Multi-Curve DLP Grokking Trainer")
    ap.add_argument("--primes", type=int, nargs="+", required=True,
                    help="Prime moduli for each curve (e.g., --primes 11 13)")
    ap.add_argument("--model-id", type=str, default="M04",
                    choices=list(MODEL_GRID.keys()),
                    help="Model ID (default: M04, ~425k params)")
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-drop-factor", type=float, default=0.1)
    ap.add_argument("--wd-start", type=float, default=0.05)
    ap.add_argument("--wd-max", type=float, default=None)
    ap.add_argument("--wd-step", type=float, default=0.05)
    ap.add_argument("--train-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output dir (default: results/multicurve/<primes>)")
    ap.add_argument("--checkpoint-interval", type=int, default=5000)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--early-stop-patience", type=int, default=100)
    ap.add_argument("--early-stop-threshold", type=float, default=0.99)
    ap.add_argument("--wd-ramp-interval", type=int, default=1000)
    ap.add_argument("--variant", type=str, default="RAW",
                    choices=["RAW", "CRT-BOTH"],
                    help="Representation variant (default: RAW)")
    args = ap.parse_args()

    if args.output_dir is None:
        prime_str = "_".join(str(p) for p in args.primes)
        args.output_dir = f"results/multicurve/p{prime_str}_{args.model_id}_{args.variant}_s{args.seed}"

    train_multicurve(
        primes=args.primes,
        model_id=args.model_id,
        n_layers=args.n_layers,
        epochs=args.epochs,
        lr=args.lr,
        wd_start=args.wd_start,
        wd_max_override=args.wd_max,
        wd_step=args.wd_step,
        train_frac=args.train_frac,
        lr_drop_factor=args.lr_drop_factor,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        early_stop_patience=args.early_stop_patience,
        early_stop_threshold=args.early_stop_threshold,
        wd_ramp_interval=args.wd_ramp_interval,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
