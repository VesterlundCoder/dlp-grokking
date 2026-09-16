"""Scaling study trainer for DLP grokking.

Adapted from lumi_cyclic_dlp_trainer.py with:
- WD auto-scaling (model-size-dependent cap)
- P_core reporting
- 5-state classification (src/analysis.py)
- T_mem/T_50/T_95 logging
- Optional representation monitoring (src/representation.py)
- Weight norm logging (fixes hardcoded 0.0 bug)
- Multi-seed support
- L6 p=113 only (scaling study)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import GrokkingTransformer, estimate_params, estimate_core_params, compute_wd_max
from src.analysis import classify_5state, extract_timing, analyze_run
from src.representation import DLPRepresentationMonitor


# ============================================================================
# Problem Generator (L6 only, adapted from lumi_cyclic_dlp_trainer.py)
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


class ProblemSpec:
    __slots__ = ["inputs", "target", "metadata"]
    def __init__(self, inputs: List[int], target: int, metadata: Dict):
        self.inputs = inputs
        self.target = target
        self.metadata = metadata


class L6Generator:
    """L6: Finite field DLP — g^x = h mod p."""
    problem_id = "L6"
    n_inputs = 2

    def __init__(self, p: int, seed: int = 42):
        self.p = p
        self.seed = seed
        self.rng = random.Random(seed)
        self._cached_roots = None

    def _get_roots(self):
        if self._cached_roots is None:
            self._cached_roots = _primitive_roots(self.p)
        return self._cached_roots

    def generate(self, n: int) -> List[ProblemSpec]:
        roots = self._get_roots()
        specs = []
        seen = set()
        max_attempts = n * 10
        attempts = 0
        while len(specs) < n and attempts < max_attempts:
            g = self.rng.choice(roots)
            x = self.rng.randint(1, self.p - 1)
            h = pow(g, x, self.p)
            key = (g, h)
            if key not in seen:
                seen.add(key)
                specs.append(ProblemSpec(
                    inputs=[g, h],
                    target=x,
                    metadata={"g": g, "h": h, "x": x},
                ))
            attempts += 1
        return specs

    def generate_all(self) -> List[ProblemSpec]:
        return self.generate(min(self.solution_space_size(), 500000))

    def solution_space_size(self) -> int:
        roots = self._get_roots()
        return (self.p - 1) * len(roots)

    def verify(self, spec: ProblemSpec) -> bool:
        g, h, x = spec.metadata["g"], spec.metadata["h"], spec.target
        return pow(g, x, self.p) == h


# ============================================================================
# Tokenizer and Dataset
# ============================================================================

class GrokkingTokenizer:
    """Vocab layout: [PAD=0, SEP=1, BOS=2, EOS=3, 0, 1, 2, ..., p-1]
    Vocab size = p + 4
    """
    def __init__(self, p: int):
        self.p = p
        self.pad_id = 0
        self.sep_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.int_offset = 4
        self.vocab_size = p + 4

    def encode(self, spec: ProblemSpec) -> tuple:
        tokens = [self.bos_id]
        for i, inp in enumerate(spec.inputs):
            tokens.append(self.int_offset + (inp % self.p))
            if i < len(spec.inputs) - 1:
                tokens.append(self.sep_id)
        tokens.append(self.eos_id)
        target = self.int_offset + (spec.target % self.p)
        return tokens, target


class GrokkingDataset(Dataset):
    def __init__(self, specs: List[ProblemSpec], tokenizer: GrokkingTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        for spec in specs:
            tokens, target = tokenizer.encode(spec)
            padded = tokens + [tokenizer.pad_id] * (max_len - len(tokens))
            padded = padded[:max_len]
            self.examples.append((torch.tensor(padded, dtype=torch.long), target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, target = self.examples[idx]
        return tokens, torch.tensor(target, dtype=torch.long)


def split_specs(specs: List[ProblemSpec], n_train: int, seed: int = 42):
    rng = random.Random(seed)
    indices = list(range(len(specs)))
    rng.shuffle(indices)
    train_idx = set(indices[:n_train])
    train = [specs[i] for i in sorted(train_idx)]
    test = [specs[i] for i in indices[n_train:]]
    return train, test


# ============================================================================
# Model Grid
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
    "M09": {"d_model": 384, "n_heads": 12},
    "M10": {"d_model": 480, "n_heads": 15},
    "M11": {"d_model": 608, "n_heads": 19},
    "M12": {"d_model": 768, "n_heads": 24},
    "M13": {"d_model": 1024, "n_heads": 32},
    "M14": {"d_model": 1280, "n_heads": 40},
    "M15": {"d_model": 1632, "n_heads": 48},
    "M16": {"d_model": 2000, "n_heads": 40},
    "M17": {"d_model": 2400, "n_heads": 40},
    "M18": {"d_model": 2750, "n_heads": 50},
}


# ============================================================================
# Accuracy Computation
# ============================================================================

def compute_accuracy(model, dataset: GrokkingDataset, tokenizer: GrokkingTokenizer,
                     device, batch_size: int = 512):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_tokens = []
            batch_targets = []
            for j in range(i, min(i + batch_size, len(dataset))):
                t, tgt = dataset.examples[j]
                batch_tokens.append(t)
                batch_targets.append(tgt)
            tokens = torch.stack(batch_tokens).to(device)
            targets = torch.tensor(batch_targets, dtype=torch.long).to(device)
            logits = model(tokens)
            if logits.dim() == 3:
                seq_lens = (tokens != tokenizer.pad_id).sum(dim=1) - 1
                idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
            else:
                logits_at_pos = logits
            preds = logits_at_pos.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)
            loss_sum += F.cross_entropy(logits_at_pos, targets).item() * len(targets)
    model.train()
    acc = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / total if total > 0 else 0.0
    return acc, avg_loss


def compute_weight_norm(model) -> float:
    """Compute L2 norm of all model weights (fixes hardcoded 0.0 bug)."""
    total_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_sq += p.data.pow(2).sum().item()
    return math.sqrt(total_sq)


# ============================================================================
# Training Loop
# ============================================================================

def auto_train_frac(model_id: str, base_frac: float = 0.3) -> float:
    """Auto-scale train_frac based on model size to prevent overfitting.

    Larger models get more training data. For p=113, solution space is 5376.
    M07+ (LUMI models) get progressively larger fractions.
    """
    model_idx = int(model_id[1:])  # M01 -> 1, M16 -> 16
    if model_idx <= 6:
        return base_frac
    if model_idx <= 12:
        # M07: 0.40, M08: 0.45, M09: 0.50, M10: 0.55, M11: 0.60, M12: 0.65
        frac = base_frac + 0.05 * (model_idx - 6)
        return min(frac, 0.7)
    # M13-M16: use larger primes, so smaller frac to get proportionally larger datasets
    # M13: 0.40, M14: 0.60, M15: 0.30, M16: 0.30
    large_fracs = {13: 0.40, 14: 0.60, 15: 0.30, 16: 0.30, 17: 0.10, 18: 0.075}
    return large_fracs.get(model_idx, 0.30)


def train_scaling(
    model_id: str,
    prime: int = 113,
    n_layers: int = 2,
    epochs: int = 1_000_000,
    lr: float = 1e-3,
    wd_start: float = 0.05,
    wd_step: float = 0.05,
    wd_acc_threshold: float = 0.05,
    wd_max_override: Optional[float] = None,
    train_frac: Optional[float] = None,
    lr_drop_factor: float = 0.1,
    seed: int = 42,
    output_dir: str = "results/scaling",
    checkpoint_interval: int = 1000,
    eval_interval: int = 10,
    early_stop_patience: int = 100,
    early_stop_threshold: float = 0.99,
    use_amp: bool = True,
    ddp: bool = False,
    monitor_reps: bool = False,
    rep_interval: int = 500,
    batch_size: Optional[int] = None,
    wd_ramp_interval: int = 1_000,
):
    """Train a scaling study model with WD auto-scaling and optional rep monitoring."""

    # Setup DDP
    local_rank = 0
    world_size = 1
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if ddp else "cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Set seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Model config from grid
    cfg = MODEL_GRID[model_id]
    d_model = cfg["d_model"]
    n_heads = cfg["n_heads"]

    if local_rank == 0:
        print(f"Device: {device}", flush=True)
        print(f"Model: {model_id}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}", flush=True)

    # Auto-scale train_frac if not explicitly set
    if train_frac is None:
        train_frac = auto_train_frac(model_id)

    # Generate dataset
    gen = L6Generator(prime, seed=seed)
    tokenizer = GrokkingTokenizer(prime)
    max_len = 2 + gen.n_inputs * 2 - 1  # BOS, g, SEP, h, EOS = 5

    space_size = gen.solution_space_size()
    n_train = min(int(space_size * train_frac), space_size - 1)
    n_total = min(n_train * 3, space_size)

    if local_rank == 0:
        print(f"Problem: L6, prime={prime}, space_size={space_size}", flush=True)
        print(f"n_inputs={gen.n_inputs}, max_len={max_len}, vocab_size={tokenizer.vocab_size}", flush=True)
        print(f"Generating dataset: n_train={n_train}, n_total~={n_total}...", flush=True)

    specs = gen.generate(n_total) if n_total < space_size else gen.generate_all()
    train_specs, test_specs = split_specs(specs, n_train, seed=seed)

    train_ds = GrokkingDataset(train_specs, tokenizer, max_len)
    test_ds = GrokkingDataset(test_specs, tokenizer, max_len)

    if local_rank == 0:
        print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)

    # Build model
    model = GrokkingTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
    ).to(device)

    if ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    raw_model = model.module if ddp else model
    n_params = raw_model.count_parameters()
    n_core = raw_model.core_parameters()

    # WD auto-scaling
    if wd_max_override is not None:
        wd_max = wd_max_override
    else:
        wd_max = compute_wd_max(n_core)

    if local_rank == 0:
        print(f"Model: {n_params:,} params (P_total), {n_core:,} core (P_core)", flush=True)
        print(f"WD auto-scale: wd_max={wd_max:.4f} (P_core={n_core:,})", flush=True)

    # Optimizer
    current_wd = wd_start
    current_lr = lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_wd)
    memorized = False
    mem_epoch = -1
    wd_ramp_interval = wd_ramp_interval  # from parameter

    # AMP — use bf16 on MI250X (same exponent range as fp32, no overflow)
    amp_dtype = torch.bfloat16 if (use_amp and device.type == "cuda") else None
    scaler = None  # bf16 doesn't need GradScaler

    # Output directory
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

        config = {
            "model_id": model_id,
            "prime": prime,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "P_total": n_params,
            "P_core": n_core,
            "epochs": epochs,
            "lr": lr,
            "wd_schedule": {
                "start": wd_start,
                "max": wd_max,
                "step": wd_step,
                "acc_threshold": wd_acc_threshold,
                "auto_scaled": wd_max_override is None,
                "lr_drop_factor": lr_drop_factor,
            },
            "train_frac": train_frac,
            "train_frac_auto": train_frac is not None and not hasattr(train_frac, '__call__'),
            "train_size": len(train_ds),
            "test_size": len(test_ds),
            "batch_size": batch_size,
            "prime": prime,
            "seed": seed,
            "device": str(device),
            "checkpoint_interval": checkpoint_interval,
            "eval_interval": eval_interval,
            "early_stop_patience": early_stop_patience,
            "early_stop_threshold": early_stop_threshold,
            "use_amp": use_amp and device.type == "cuda",
            "ddp": ddp,
            "world_size": world_size,
            "vocab_size": tokenizer.vocab_size,
            "max_len": max_len,
            "space_size": space_size,
            "monitor_reps": monitor_reps,
            "pytorch_version": torch.__version__,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # Representation monitor
    rep_monitor = None
    if monitor_reps and local_rank == 0:
        rep_monitor = DLPRepresentationMonitor(
            p=prime,
            device=device,
            output_dir=output_dir,
            rep_interval=rep_interval,
        )
        rep_monitor.register_hooks(raw_model)
        if local_rank == 0:
            print(f"Representation monitoring: enabled (interval={rep_interval})", flush=True)

    # Prepare tensors (full-batch)
    train_tokens = torch.stack([train_ds.examples[i][0] for i in range(len(train_ds))]).to(device)
    train_targets = torch.tensor([train_ds.examples[i][1] for i in range(len(train_ds))], dtype=torch.long).to(device)

    # Metrics
    metrics_path = os.path.join(output_dir, "metrics.jsonl") if local_rank == 0 else None
    all_metrics: List[Dict] = []

    # Training loop
    start_time = time.time()
    consecutive_high = 0
    best_test_acc = 0.0
    early_stopped = False
    full_batch_size = batch_size if batch_size is not None else len(train_ds)
    prev_test_acc = 0.0

    if local_rank == 0:
        print(f"\nStarting training: {epochs} epochs, batch_size={full_batch_size}", flush=True)
        print(f"WD schedule: start={wd_start}, max={wd_max:.4f}, step={wd_step}, threshold={wd_acc_threshold}", flush=True)
        print(f"AMP: {use_amp and device.type == 'cuda'}, DDP: {ddp} (world_size={world_size})", flush=True)
        print(f"Seed: {seed}", flush=True)
        print("", flush=True)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_ds), device=device)
        epoch_loss = 0.0
        n_batches = 0

        if ddp:
            gpu_indices = perm[local_rank::world_size]
        else:
            gpu_indices = perm

        for i in range(0, len(gpu_indices), full_batch_size):
            idx = gpu_indices[i:i + full_batch_size]
            xb = train_tokens[idx]
            yb = train_targets[idx]

            # Forward
            if amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(xb)
                    if logits.dim() == 3:
                        seq_lens = (xb != tokenizer.pad_id).sum(dim=1) - 1
                        idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                        logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
                    else:
                        logits_at_pos = logits
                    loss = F.cross_entropy(logits_at_pos, yb)
            else:
                logits = model(xb)
                if logits.dim() == 3:
                    seq_lens = (xb != tokenizer.pad_id).sum(dim=1) - 1
                    idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                    logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
                else:
                    logits_at_pos = logits
                loss = F.cross_entropy(logits_at_pos, yb)

            # NaN detection — skip optimizer step if loss is NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                n_batches += 1
                continue

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Evaluate
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            if ddp and local_rank != 0:
                train_acc = float("nan")
                test_acc = float("nan")
                test_loss = avg_train_loss
            else:
                train_acc, _ = compute_accuracy(model, train_ds, tokenizer, device, batch_size=512)
                test_acc, test_loss = compute_accuracy(model, test_ds, tokenizer, device, batch_size=512)

            # Detect memorization
            if not math.isnan(train_acc) and not memorized and train_acc > 0.99:
                memorized = True
                mem_epoch = epoch
                if local_rank == 0:
                    print(f"  MEMORIZED at epoch {epoch} (train_acc={train_acc:.4f})", flush=True)

            # Post-memorization WD ramp with LR drop
            if memorized and current_wd < wd_max:
                ramps_since_mem = (epoch - mem_epoch) // wd_ramp_interval
                target_wd_steps = ramps_since_mem + 1  # first ramp at memorization
                target_wd = min(wd_start + target_wd_steps * wd_step, wd_max)
                if target_wd > current_wd:
                    new_wd = target_wd
                    new_lr = current_lr * lr_drop_factor
                    for pg in optimizer.param_groups:
                        pg['weight_decay'] = new_wd
                        pg['lr'] = new_lr
                    if local_rank == 0:
                        print(f"  WD ramp: {current_wd:.4f} -> {new_wd:.4f} | LR: {current_lr:.2e} -> {new_lr:.2e} (epoch={epoch}, post-mem={epoch - mem_epoch})", flush=True)
                    current_wd = new_wd
                    current_lr = new_lr
        else:
            train_acc = float("nan")
            test_acc = float("nan")
            test_loss = avg_train_loss

        # DDP barrier
        if ddp and (epoch % eval_interval == 0 or epoch == epochs - 1):
            torch.distributed.barrier()

        # Track best
        if not math.isnan(test_acc):
            if test_acc > best_test_acc:
                best_test_acc = test_acc

        # Weight norm (fixes hardcoded 0.0 bug)
        weight_norm = 0.0
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            weight_norm = compute_weight_norm(model)

        # Representation monitoring
        if rep_monitor is not None and (epoch % rep_interval == 0 or epoch == 0):
            try:
                rep_metrics = rep_monitor.compute_metrics(raw_model, epoch)
                rep_monitor.save_metrics(rep_metrics)
                if rep_monitor.should_checkpoint(epoch, test_acc if not math.isnan(test_acc) else 0.0, prev_test_acc):
                    rep_monitor.checkpoint(raw_model, epoch)
            except Exception as e:
                if local_rank == 0:
                    print(f"  Rep monitor error: {e}", flush=True)

        prev_test_acc = test_acc if not math.isnan(test_acc) else prev_test_acc

        # Log metrics
        if local_rank == 0 and metrics_path:
            elapsed = time.time() - start_time
            metric_entry = {
                "epoch": epoch,
                "train_acc": train_acc if not math.isnan(train_acc) else None,
                "test_acc": test_acc if not math.isnan(test_acc) else None,
                "train_loss": avg_train_loss,
                "test_loss": test_loss if not math.isnan(test_loss) else None,
                "weight_decay": current_wd,
                "weight_norm": weight_norm,
                "elapsed_s": elapsed,
                "best_test_acc": best_test_acc,
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metric_entry) + "\n")
            all_metrics.append(metric_entry)

        # Checkpoint
        if local_rank == 0 and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(output_dir, "checkpoints", f"epoch_{epoch:06d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "current_wd": current_wd,
                "best_test_acc": best_test_acc,
            }, ckpt_path)

        # Progress log
        if local_rank == 0 and (epoch % 100 == 0 or epoch == epochs - 1 or (test_acc is not None and not math.isnan(test_acc) and test_acc > 0.1 and epoch % eval_interval == 0)):
            elapsed = time.time() - start_time
            ta = train_acc if not math.isnan(train_acc) else 0.0
            te = test_acc if not math.isnan(test_acc) else 0.0
            print(f"  Epoch {epoch:6d} | train={ta:.4f} test={te:.4f} "
                  f"loss={avg_train_loss:.4f} wd={current_wd:.4f} wnorm={weight_norm:.1f} "
                  f"t={elapsed:.1f}s", flush=True)

        # Early stopping (only on eval epochs where test_acc is valid)
        if not math.isnan(test_acc):
            if test_acc > early_stop_threshold:
                consecutive_high += 1
                if consecutive_high >= early_stop_patience:
                    if local_rank == 0:
                        print(f"\nEarly stopped at epoch {epoch} (test_acc > {early_stop_threshold} for {consecutive_high} evals)", flush=True)
                    early_stopped = True
            else:
                consecutive_high = 0

        if early_stopped:
            break

    # Final checkpoint
    if local_rank == 0:
        final_path = os.path.join(output_dir, "checkpoints", "final.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "current_wd": current_wd,
            "best_test_acc": best_test_acc,
        }, final_path)

        # 5-state classification
        state = classify_5state(all_metrics)
        timing = extract_timing(all_metrics)

        summary = {
            "model_id": model_id,
            "state": state,
            "best_test_acc": best_test_acc,
            "total_epochs": epoch + 1,
            "final_wd": current_wd,
            "n_params": n_params,
            "P_core": n_core,
            "T_mem": timing["T_mem"],
            "T_50": timing["T_50"],
            "T_95": timing["T_95"],
            "G": timing["G"],
            "seed": seed,
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}", flush=True)
        print(f"Training complete: {epoch + 1} epochs in {elapsed:.1f}s", flush=True)
        print(f"Model: {model_id} ({n_params:,} params, P_core={n_core:,})", flush=True)
        print(f"Best test acc: {best_test_acc:.4f}", flush=True)
        print(f"State: {state}", flush=True)
        print(f"T_mem={timing['T_mem']}, T_50={timing['T_50']}, T_95={timing['T_95']}, G={timing['G']}", flush=True)
        print(f"Final WD: {current_wd:.4f}", flush=True)
        print(f"{'='*60}", flush=True)

    if rep_monitor is not None:
        rep_monitor.remove_hooks()

    if ddp:
        torch.distributed.destroy_process_group()


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="DLP Scaling Study Trainer")
    ap.add_argument("--model-id", type=str, required=True,
                    choices=list(MODEL_GRID.keys()),
                    help="Model ID: M01-M18")
    ap.add_argument("--prime", type=int, default=113,
                    help="Prime modulus (default: 113)")
    ap.add_argument("--n-layers", type=int, default=2,
                    help="Transformer layers (default: 2)")
    ap.add_argument("--epochs", type=int, default=1_000_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-drop-factor", type=float, default=0.1,
                    help="Multiply LR by this factor on each WD ramp (default: 0.1 = 10%)")
    ap.add_argument("--wd-start", type=float, default=0.05)
    ap.add_argument("--wd-max", type=float, default=None,
                    help="Override auto-scaled WD max (default: auto)")
    ap.add_argument("--wd-step", type=float, default=0.05)
    ap.add_argument("--wd-acc-threshold", type=float, default=0.05)
    ap.add_argument("--train-frac", type=float, default=None,
                    help="Train fraction (default: auto-scaled by model size)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="results/scaling")
    ap.add_argument("--checkpoint-interval", type=int, default=1000)
    ap.add_argument("--eval-interval", type=int, default=10)
    ap.add_argument("--early-stop-patience", type=int, default=100)
    ap.add_argument("--early-stop-threshold", type=float, default=0.99)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--monitor-reps", action="store_true",
                    help="Enable representation monitoring")
    ap.add_argument("--rep-interval", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Mini-batch size (default: full-batch = len(train_ds))")
    ap.add_argument("--wd-ramp-interval", type=int, default=1000,
                    help="Epochs between WD ramps after memorization (default: 1000)")
    args = ap.parse_args()

    train_scaling(
        model_id=args.model_id,
        prime=args.prime,
        n_layers=args.n_layers,
        epochs=args.epochs,
        lr=args.lr,
        wd_start=args.wd_start,
        wd_max_override=args.wd_max,
        wd_step=args.wd_step,
        wd_acc_threshold=args.wd_acc_threshold,
        train_frac=args.train_frac,
        lr_drop_factor=args.lr_drop_factor,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        early_stop_patience=args.early_stop_patience,
        early_stop_threshold=args.early_stop_threshold,
        use_amp=not args.no_amp,
        ddp=args.ddp,
        monitor_reps=args.monitor_reps,
        rep_interval=args.rep_interval,
        batch_size=args.batch_size,
        wd_ramp_interval=args.wd_ramp_interval,
    )


if __name__ == "__main__":
    main()
