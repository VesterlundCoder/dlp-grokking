"""Training script for FRE/FCO DLP grokking experiments.

Tests whether a fixed-size O(1) model can grok DLP at large primes using
Factorized Residue Embedding (FRE) and Factorized Candidate Output (FCO),
optionally combined with CRT-BOTH representation.

Usage:
    # Sanity check: p=113 with FRE/FCO (no CRT)
    python experiments/fre_fco/run_fre_fco.py --prime 113 --d-model 128 --epochs 100000

    # p=113 with FRE/FCO + CRT-BOTH
    python experiments/fre_fco/run_fre_fco.py --prime 113 --d-model 128 --crt --epochs 100000

    # Large prime: p=8209 with FRE/FCO + CRT-BOTH
    python experiments/fre_fco/run_fre_fco.py --prime 8209 --d-model 128 --crt \
        --train-frac 0.30 --epochs 500000 --batch-size 8192
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

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.fre_fco import (
    FREEmbedding,
    FactorizedCandidateOutput,
    RichFCO,
    FRETokenizer,
    GrokkingTransformerFRE,
    coprime_factorization,
    project_factor,
    encode_crt_both,
    encode_raw,
    estimate_fre_fco_params,
    auto_compute_M,
)


# ============================================================================
# Problem generation (ported from src/trainer.py)
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


def generate_dlp_dataset(p: int, n: int, seed: int = 42,
                         use_crt: bool = False,
                         crt_factors: Tuple[int, int] = None
                         ) -> List[Tuple[List[int], int, Dict]]:
    """Generate n DLP problems for prime p.

    Returns list of (tokens, target, metadata) tuples.
    tokens are vocab IDs (BOS=2, SEP=1, EOS=3, values=4+val).
    target is 0-indexed: x in {0, ..., p-2}.
    """
    rng = random.Random(seed)
    roots = _primitive_roots(p)
    specs = []
    seen = set()
    max_attempts = n * 10
    attempts = 0

    while len(specs) < n and attempts < max_attempts:
        g = rng.choice(roots)
        x = rng.randint(1, p - 1)
        h = pow(g, x, p)
        key = (g, h)
        if key not in seen:
            seen.add(key)
            if use_crt:
                tokens, target = encode_crt_both(g, h, x - 1, p, crt_factors)  # x-1 for 0-indexed
            else:
                tokens, target = encode_raw(g, h, x - 1, p)
            metadata = {"g": g, "h": h, "x": x}
            specs.append((tokens, target, metadata))
        attempts += 1

    return specs


def split_dataset(specs: List, n_train: int, seed: int = 42):
    """Split specs into train/test sets."""
    rng = random.Random(seed)
    indices = list(range(len(specs)))
    rng.shuffle(indices)
    train_idx = set(indices[:n_train])
    train = [specs[i] for i in sorted(train_idx)]
    test = [specs[i] for i in indices[n_train:]]
    return train, test


# ============================================================================
# Training
# ============================================================================

def compute_accuracy(model, specs, pad_id: int, device, batch_size: int = 8192,
                     max_eval: int = None):
    """Compute accuracy on a set of specs.

    Args:
        max_eval: If set, only evaluate on at most this many examples (random subset).
                  Useful for large datasets where full eval is too slow.
    """
    if max_eval is not None and len(specs) > max_eval:
        import random as _rng
        idxs = _rng.Random(42).sample(range(len(specs)), max_eval)
        specs = [specs[i] for i in idxs]

    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for i in range(0, len(specs), batch_size):
            batch = specs[i:i + batch_size]
            tokens_list = [t for t, _, _ in batch]
            targets = torch.tensor([tgt for _, tgt, _ in batch], dtype=torch.long, device=device)

            # Pad to same length
            max_len = max(len(t) for t in tokens_list)
            padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
            for j, t in enumerate(tokens_list):
                padded[j, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

            logits = model(padded)  # (B, p-1)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)
            loss_sum += F.cross_entropy(logits, targets).item() * len(targets)

    model.train()
    acc = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / total if total > 0 else 0.0
    return acc, avg_loss


def compute_weight_norm(model) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_sq += p.data.pow(2).sum().item()
    return math.sqrt(total_sq)


def train_fre_fco(
    prime: int = 113,
    base: int = 16,
    M: int = None,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    epochs: int = 100_000,
    lr: float = 1e-3,
    wd_start: float = 0.05,
    wd_step: float = 0.05,
    wd_max: float = 0.30,
    wd_ramp_interval: int = 1000,
    train_frac: float = 0.30,
    seed: int = 42,
    output_dir: str = "results/fre_fco",
    eval_interval: int = 10,
    early_stop_patience: int = 100,
    early_stop_threshold: float = 0.99,
    batch_size: int = 0,  # 0 = full batch
    use_crt: bool = False,
    use_rich_fco: bool = False,
    exp_name: str = None,
):
    """Train a FRE/FCO model on DLP."""

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Auto-compute M if not specified
    if M is None:
        M = auto_compute_M(prime, base)

    # CRT factors
    crt_factors = None
    if use_crt:
        try:
            crt_factors = coprime_factorization(prime - 1)
            print(f"CRT factors of q={prime-1}: {crt_factors[0]} x {crt_factors[1]}", flush=True)
        except ValueError as e:
            print(f"Warning: CRT not available for p={prime}: {e}", flush=True)
            use_crt = False

    # Tokenizer
    tokenizer = FRETokenizer(prime, base=base, M=M)
    max_len = 9 if use_crt else 5  # CRT-BOTH: 9 tokens, RAW: 5 tokens

    # Generate dataset
    roots = _primitive_roots(prime)
    space_size = (prime - 1) * len(roots)
    n_train = min(int(space_size * train_frac), space_size - 1)
    n_total = min(n_train * 3, space_size)

    print(f"Device: {device}", flush=True)
    print(f"Prime: {prime}, base: {base}, M: {M}, d_model: {d_model}", flush=True)
    print(f"Space size: {space_size}, n_train: {n_train}, n_total: {n_total}", flush=True)
    print(f"CRT: {use_crt}, RichFCO: {use_rich_fco}", flush=True)

    specs = generate_dlp_dataset(prime, n_total, seed=seed,
                                  use_crt=use_crt, crt_factors=crt_factors)
    train_specs, test_specs = split_dataset(specs, n_train, seed=seed)
    print(f"Train: {len(train_specs)}, Test: {len(test_specs)}", flush=True)

    # Build model
    model = GrokkingTransformerFRE(
        p=prime, base=base, M=M,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_len=max_len, use_rich_fco=use_rich_fco,
        use_crt=use_crt, crt_factors=crt_factors,
    ).to(device)

    n_params = model.count_parameters()
    n_core = model.core_parameters()
    breakdown = model.parameter_breakdown()

    print(f"\nModel parameters:", flush=True)
    print(f"  P_input:  {breakdown['P_input']:,}", flush=True)
    print(f"  P_core:   {breakdown['P_core']:,}", flush=True)
    print(f"  P_output: {breakdown['P_output']:,}", flush=True)
    print(f"  P_total:  {breakdown['P_total']:,}", flush=True)

    # Compare with atomic
    atomic_p = (prime + 4) * d_model + max_len * d_model + n_layers * 12 * d_model * d_model + d_model * (prime + 4)
    print(f"  Atomic P_total would be: {atomic_p:,} ({atomic_p / n_params:.1f}x larger)", flush=True)

    # WD auto-scaling
    wd_max_actual = min(wd_max, wd_max * math.sqrt(393_000 / max(n_core, 1)))
    wd_max_actual = max(0.15, wd_max_actual)
    print(f"  WD max: {wd_max_actual:.4f}", flush=True)

    # Optimizer
    current_wd = wd_start
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=current_wd)

    # Output directory
    if exp_name is None:
        rep = "crt" if use_crt else "raw"
        exp_name = f"p{prime}_{rep}_B{base}_M{M}_d{d_model}_L{n_layers}_f{train_frac}_s{seed}"
    out_dir = os.path.join(output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    config = {
        "prime": prime,
        "base": base,
        "M": M,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "max_len": max_len,
        "epochs": epochs,
        "lr": lr,
        "wd_start": wd_start,
        "wd_max": wd_max_actual,
        "wd_step": wd_step,
        "wd_ramp_interval": wd_ramp_interval,
        "train_frac": train_frac,
        "train_size": len(train_specs),
        "test_size": len(test_specs),
        "seed": seed,
        "use_crt": use_crt,
        "use_rich_fco": use_rich_fco,
        "crt_factors": list(crt_factors) if crt_factors else None,
        "batch_size": batch_size,
        "P_input": breakdown["P_input"],
        "P_core": breakdown["P_core"],
        "P_output": breakdown["P_output"],
        "P_total": breakdown["P_total"],
        "space_size": space_size,
        "device": str(device),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Prepare data tensors
    pad_id = 0
    def specs_to_tensors(specs):
        tokens_list = [t for t, _, _ in specs]
        targets = [tgt for _, tgt, _ in specs]
        max_l = max(len(t) for t in tokens_list)
        padded = torch.full((len(specs), max_l), pad_id, dtype=torch.long)
        for i, t in enumerate(tokens_list):
            padded[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        return padded.to(device), torch.tensor(targets, dtype=torch.long, device=device)

    train_tokens, train_targets = specs_to_tensors(train_specs)
    test_specs_tensors = specs_to_tensors(test_specs)

    # Metrics
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    best_test_acc = 0.0
    memorized = False
    mem_epoch = -1
    consecutive_high = 0
    early_stopped = False

    print(f"\nStarting training: {epochs} epochs", flush=True)
    print(f"WD schedule: start={wd_start}, max={wd_max_actual:.4f}, step={wd_step}", flush=True)
    print(f"Batch size: {batch_size if batch_size > 0 else len(train_specs)} (full batch)", flush=True)
    print("", flush=True)

    start_time = time.time()
    actual_batch = batch_size if batch_size > 0 else len(train_specs)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_specs), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), actual_batch):
            idx = perm[i:i + actual_batch]
            xb = train_tokens[idx]
            yb = train_targets[idx]

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

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
            # Use subset eval for large datasets (faster)
            max_eval = 10000 if len(test_specs) > 50000 else None
            train_acc, _ = compute_accuracy(model, train_specs, pad_id, device,
                                             max_eval=max_eval)
            test_acc, test_loss = compute_accuracy(model, test_specs, pad_id, device,
                                                    max_eval=max_eval)

            # Detect memorization
            if not memorized and train_acc > 0.99:
                memorized = True
                mem_epoch = epoch
                print(f"  MEMORIZED at epoch {epoch} (train_acc={train_acc:.4f})", flush=True)

            # Post-memorization WD ramp
            if memorized and current_wd < wd_max_actual:
                ramps_since_mem = (epoch - mem_epoch) // wd_ramp_interval
                target_wd_steps = ramps_since_mem + 1
                target_wd = min(wd_start + target_wd_steps * wd_step, wd_max_actual)
                if target_wd > current_wd:
                    current_wd = target_wd
                    for pg in optimizer.param_groups:
                        pg['weight_decay'] = current_wd
                    print(f"  WD ramp: {current_wd:.4f} at epoch {epoch}", flush=True)

            # Track best
            if test_acc > best_test_acc:
                best_test_acc = test_acc

            # Early stopping
            if test_acc > early_stop_threshold:
                consecutive_high += 1
                if consecutive_high >= early_stop_patience:
                    early_stopped = True
                    print(f"  EARLY STOP at epoch {epoch} (test_acc={test_acc:.4f} for {consecutive_high} evals)", flush=True)
            else:
                consecutive_high = 0

            # Log
            wnorm = compute_weight_norm(model)
            elapsed = time.time() - start_time
            metric = {
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": avg_train_loss,
                "test_loss": test_loss,
                "weight_decay": current_wd,
                "weight_norm": wnorm,
                "best_test_acc": best_test_acc,
                "elapsed_s": elapsed,
                "memorized": memorized,
                "mem_epoch": mem_epoch,
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metric) + "\n")

            if epoch % (eval_interval * 10) == 0 or test_acc > 0.5:
                print(f"  epoch {epoch:>6d} | train={train_acc:.4f} test={test_acc:.4f} "
                      f"loss={avg_train_loss:.4f} wd={current_wd:.3f} "
                      f"wnorm={wnorm:.1f} best={best_test_acc:.4f} "
                      f"t={elapsed:.0f}s", flush=True)

            if early_stopped:
                break

    # Summary
    elapsed = time.time() - start_time
    grokked = best_test_acc > 0.90
    summary = {
        "prime": prime,
        "use_crt": use_crt,
        "base": base,
        "M": M,
        "d_model": d_model,
        "P_total": n_params,
        "P_core": n_core,
        "best_test_acc": best_test_acc,
        "grokked": grokked,
        "mem_epoch": mem_epoch,
        "total_epochs": epoch,
        "elapsed_s": elapsed,
        "early_stopped": early_stopped,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"Summary: p={prime}, CRT={use_crt}, P_total={n_params:,}", flush=True)
    print(f"  Best test acc: {best_test_acc:.4f}", flush=True)
    print(f"  Grokked: {grokked}", flush=True)
    print(f"  Mem epoch: {mem_epoch}", flush=True)
    print(f"  Total epochs: {epoch}", flush=True)
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
    print(f"  Results: {out_dir}", flush=True)

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FRE/FCO DLP Grokking Trainer")
    parser.add_argument("--prime", type=int, default=113, help="Prime p")
    parser.add_argument("--base", type=int, default=16, help="Base B for digit decomposition")
    parser.add_argument("--M", type=int, default=None, help="Number of digits (auto if not set)")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-layers", type=int, default=2, help="Transformer layers")
    parser.add_argument("--epochs", type=int, default=100_000, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd-start", type=float, default=0.05, help="Initial weight decay")
    parser.add_argument("--wd-max", type=float, default=0.30, help="Max weight decay")
    parser.add_argument("--wd-step", type=float, default=0.05, help="WD ramp step")
    parser.add_argument("--wd-ramp-interval", type=int, default=1000, help="WD ramp interval (epochs)")
    parser.add_argument("--train-frac", type=float, default=0.30, help="Training fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/fre_fco", help="Output directory")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval (epochs)")
    parser.add_argument("--early-stop-patience", type=int, default=100, help="Early stop patience")
    parser.add_argument("--batch-size", type=int, default=0, help="Mini-batch size (0=full batch)")
    parser.add_argument("--crt", action="store_true", help="Use CRT-BOTH representation")
    parser.add_argument("--rich-fco", action="store_true", help="Use RichFCO (bilinear interactions)")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")

    args = parser.parse_args()

    train_fre_fco(
        prime=args.prime,
        base=args.base,
        M=args.M,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        epochs=args.epochs,
        lr=args.lr,
        wd_start=args.wd_start,
        wd_max=args.wd_max,
        wd_step=args.wd_step,
        wd_ramp_interval=args.wd_ramp_interval,
        train_frac=args.train_frac,
        seed=args.seed,
        output_dir=args.output_dir,
        eval_interval=args.eval_interval,
        early_stop_patience=args.early_stop_patience,
        batch_size=args.batch_size,
        use_crt=args.crt,
        use_rich_fco=args.rich_fco,
        exp_name=args.exp_name,
    )


if __name__ == "__main__":
    main()
