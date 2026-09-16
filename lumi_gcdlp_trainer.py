#!/usr/bin/env python3
"""
LUMI GCDLP Trainer — General Cyclic DLP Solver (standalone)

Trains a transformer to solve the General Cyclic DLP:
    Given (p, d, g, h), find x such that g^x = h mod p
    where d | p-1, ord(g) = d, x in {0, ..., d-1}.

The key generalization vs L6: d varies across instances (different subgroup orders,
different ambient primes, different generators). The model must learn the abstract
cyclic structure ax ≡ b mod d, not memorize a single group table.

Three experiment families:
  Family 1: Prime-order ladder (d=251, 509, 1013 — all prime, mixed in one dataset)
  Family 2: Prime + prime-power + smooth (d=503, 512, 504 — factorization effect)
  Family 3: Cross-field transfer (d=251, multiple ambient primes, hold out one for test)

Two input variants:
  Order-visible: [BOS] [P] p [D] d [G] g [H] h [EOS]  →  x
  Order-hidden:  [BOS] [P] p [G] g [H] h [EOS]        →  x

Data split: Generator-OOD (70% generators train, 15% val, 15% test).
Output masking: logits for x >= d masked to -inf.

Usage (single GPU):
    python lumi_gcdlp_trainer.py --family 1 --order-visible \\
        --d-model 512 --n-layers 8 --n-heads 16 \\
        --epochs 200000 --output-dir results/gcdlp/f1_vis

Usage (multi-GPU via torchrun):
    torchrun --standalone --nproc_per_node=4 lumi_gcdlp_trainer.py \\
        --family 1 --order-visible --ddp --output-dir results/gcdlp/f1_vis
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Number Theory Helpers
# ============================================================================

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


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


def _euler_totient(n: int) -> int:
    if n == 1:
        return 1
    factors = set(_prime_factors(n))
    result = n
    for p in factors:
        result = result // p * (p - 1)
    return result


def _mod_inverse(a: int, m: int) -> int:
    return pow(a, -1, m)


def _primitive_root(p: int) -> int:
    """Find the smallest primitive root of F_p*."""
    if p == 2:
        return 1
    n = p - 1
    factors = list(set(_prime_factors(n)))
    for g in range(2, p):
        is_primitive = True
        for q in factors:
            if pow(g, n // q, p) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    return -1


def _find_prime_congruent_1_mod(d: int, min_p: int = 0, max_p: int = 100000) -> int:
    """Find the smallest prime p such that p ≡ 1 mod d and p > min_p."""
    k = max(min_p // d, 1)
    while True:
        p = k * d + 1
        if p > max_p:
            return -1
        if p > d and _is_prime(p):
            return p
        k += 1


def _find_safe_prime_for_d(d: int) -> int:
    """For prime d, find p = 2d+1 if it's prime (safe prime). Otherwise search."""
    p = 2 * d + 1
    if _is_prime(p):
        return p
    return _find_prime_congruent_1_mod(d, min_p=d + 1)


# ============================================================================
# GCDLP Problem Specification
# ============================================================================

@dataclass
class GCDLPSpec:
    """A single GCDLP instance: (p, d, g, h) → x."""
    p: int
    d: int
    g: int
    h: int
    x: int  # target: g^x = h mod p, x in {0, ..., d-1}
    gen_idx: int  # which generator (for OOD split)


@dataclass
class SubgroupConfig:
    """Configuration for one (p, d) subgroup."""
    p: int
    d: int
    zeta: int   # primitive root of F_p*
    gamma: int  # zeta^((p-1)/d) mod p, canonical subgroup generator of order d


def build_subgroup(p: int, d: int) -> SubgroupConfig:
    """Build subgroup config: find primitive root, compute canonical generator."""
    zeta = _primitive_root(p)
    gamma = pow(zeta, (p - 1) // d, p)
    # Verify gamma has order exactly d
    assert pow(gamma, d, p) == 1, f"gamma^d != 1 mod p for p={p}, d={d}"
    if d > 1:
        for q in set(_prime_factors(d)):
            assert pow(gamma, d // q, p) != 1, f"gamma has order < d for p={p}, d={d}"
    return SubgroupConfig(p=p, d=d, zeta=zeta, gamma=gamma)


# ============================================================================
# GCDLP Generator with Generator-OOD Split
# ============================================================================

class GCDLPGenerator:
    """Generates GCDLP instances with generator-OOD split.

    For each (p, d):
    - Canonical generator gamma = zeta^((p-1)/d) has order d
    - All φ(d) generators: g_a = gamma^a for a in Z_d* (coprime to d)
    - All d elements: h_b = gamma^b for b in Z_d
    - DLP: g_a^x = h_b → a*x = b mod d → x = a^(-1)*b mod d

    Generator-OOD split: 70% of generators → train, 15% → val, 15% → test
    """

    def __init__(self, subgroups: List[SubgroupConfig], seed: int = 42,
                 gen_train_frac: float = 0.70, gen_val_frac: float = 0.15):
        self.subgroups = subgroups
        self.seed = seed
        self.rng = random.Random(seed)
        self.gen_train_frac = gen_train_frac
        self.gen_val_frac = gen_val_frac

        # Precompute generators and elements for each subgroup
        self._subgroup_data = []
        for sg in subgroups:
            # Generators: g_a = gamma^a for a in Z_d* (coprime to d)
            gen_exponents = [a for a in range(1, sg.d) if math.gcd(a, sg.d) == 1]
            generators = [(a, pow(sg.gamma, a, sg.p)) for a in gen_exponents]
            # Elements: h_b = gamma^b for b in Z_d
            elements = [(b, pow(sg.gamma, b, sg.p)) for b in range(sg.d)]
            self._subgroup_data.append({
                "config": sg,
                "gen_exponents": gen_exponents,
                "generators": generators,  # list of (a, g_a)
                "elements": elements,      # list of (b, h_b)
            })

    def generate_split(self) -> Tuple[List[GCDLPSpec], List[GCDLPSpec], List[GCDLPSpec]]:
        """Generate (train, val, test) splits with generator-OOD."""
        train_specs = []
        val_specs = []
        test_specs = []

        for sg_data in self._subgroup_data:
            sg = sg_data["config"]
            generators = sg_data["generators"]
            elements = sg_data["elements"]

            # Shuffle generators and split
            gen_indices = list(range(len(generators)))
            self.rng.shuffle(gen_indices)

            n_gen = len(gen_indices)
            n_train = int(n_gen * self.gen_train_frac)
            n_val = int(n_gen * self.gen_val_frac)

            train_gens = gen_indices[:n_train]
            val_gens = gen_indices[n_train:n_train + n_val]
            test_gens = gen_indices[n_train + n_val:]

            # For each generator, create all d (g, h) pairs
            for gen_list, target_specs in [
                (train_gens, train_specs),
                (val_gens, val_specs),
                (test_gens, test_specs),
            ]:
                for gi in gen_list:
                    a, g = generators[gi]
                    a_inv = _mod_inverse(a, sg.d)
                    for b, h in elements:
                        x = (a_inv * b) % sg.d
                        target_specs.append(GCDLPSpec(
                            p=sg.p, d=sg.d, g=g, h=h, x=x, gen_idx=gi,
                        ))

        return train_specs, val_specs, test_specs

    def solution_space_size(self, sg_idx: int) -> int:
        """N(d) = d * φ(d) for one subgroup."""
        sg = self.subgroups[sg_idx]
        return sg.d * _euler_totient(sg.d)

    def total_solution_space(self) -> int:
        return sum(self.solution_space_size(i) for i in range(len(self.subgroups)))


# ============================================================================
# Family Configurations
# ============================================================================

def build_family_1() -> List[SubgroupConfig]:
    """Prime-order ladder: d=251 (p=503), d=509 (p=1019), d=1013 (p=2027)."""
    pairs = [
        (251, 503),
        (509, 1019),
        (1013, 2027),
    ]
    subgroups = []
    for d, p_expected in pairs:
        p = _find_safe_prime_for_d(d)
        if p != p_expected:
            p = p_expected if _is_prime(p_expected) and (p_expected - 1) % d == 0 else p
        if p == -1 or not _is_prime(p) or (p - 1) % d != 0:
            p = _find_prime_congruent_1_mod(d, min_p=d + 1)
        assert p > 0, f"Could not find prime for d={d}"
        subgroups.append(build_subgroup(p, d))
    return subgroups


def build_family_2() -> List[SubgroupConfig]:
    """Prime + prime-power + smooth: d=503 (prime), d=512 (2^9), d=504 (smooth)."""
    configs = [
        ("prime", 503),
        ("prime_power", 512),
        ("smooth", 504),
    ]
    subgroups = []
    for label, d in configs:
        p = _find_prime_congruent_1_mod(d, min_p=d + 1, max_p=500000)
        assert p > 0, f"Could not find prime p ≡ 1 mod {d}"
        subgroups.append(build_subgroup(p, d))
    return subgroups


def build_family_3() -> List[SubgroupConfig]:
    """Cross-field transfer: d=251 with 3 different ambient primes.

    Train on first 2 primes, test on 3rd (held out).
    """
    d = 251
    primes = []
    k = 2
    while len(primes) < 3:
        p = k * d + 1
        if p > d and _is_prime(p):
            primes.append(p)
        k += 1
        if k > 100:
            break
    assert len(primes) >= 3, f"Could not find 3 primes ≡ 1 mod {d}"
    subgroups = []
    for p in primes[:3]:
        subgroups.append(build_subgroup(p, d))
    return subgroups


FAMILY_BUILDERS = {
    1: build_family_1,
    2: build_family_2,
    3: build_family_3,
}


# ============================================================================
# GCDLP Tokenizer
# ============================================================================

class GCDLPTokenizer:
    """Tokenizes GCDLP instances.

    Vocab layout: [PAD=0, BOS=1, EOS=2, P=3, D=4, G=5, H=6, 0, 1, 2, ..., p_max-1]
    Vocab size = p_max + 7
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    P_ID = 3
    D_ID = 4
    G_ID = 5
    H_ID = 6
    INT_OFFSET = 7

    def __init__(self, p_max: int):
        self.p_max = p_max
        self.vocab_size = p_max + self.INT_OFFSET

    def encode(self, spec: GCDLPSpec, order_visible: bool = True) -> Tuple[List[int], int]:
        tokens = [self.BOS_ID]
        tokens.append(self.P_ID)
        tokens.append(self.INT_OFFSET + spec.p)
        if order_visible:
            tokens.append(self.D_ID)
            tokens.append(self.INT_OFFSET + spec.d)
        tokens.append(self.G_ID)
        tokens.append(self.INT_OFFSET + spec.g)
        tokens.append(self.H_ID)
        tokens.append(self.INT_OFFSET + spec.h)
        tokens.append(self.EOS_ID)
        target = self.INT_OFFSET + spec.x
        return tokens, target

    def max_len(self, order_visible: bool = True) -> int:
        if order_visible:
            return 11  # BOS P p D d G g H h EOS
        return 9  # BOS P p G g H h EOS


# ============================================================================
# GCDLP Dataset
# ============================================================================

class GCDLPDataset:
    """Dataset wrapping encoded GCDLP specs."""

    def __init__(self, specs: List[GCDLPSpec], tokenizer: GCDLPTokenizer,
                 order_visible: bool = True, max_len: int = 11):
        self.tokenizer = tokenizer
        self.order_visible = order_visible
        self.max_len = max_len
        self.examples: List[Tuple[torch.Tensor, int, int]] = []  # (tokens, target, d)
        for spec in specs:
            tokens, target = tokenizer.encode(spec, order_visible)
            padded = tokens + [tokenizer.PAD_ID] * (max_len - len(tokens))
            padded = padded[:max_len]
            self.examples.append((
                torch.tensor(padded, dtype=torch.long),
                target,
                spec.d,
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, target, d = self.examples[idx]
        return tokens, torch.tensor(target, dtype=torch.long), d


# ============================================================================
# Grokking Transformer (same architecture as existing)
# ============================================================================

class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 16,
                 n_layers: int = 8, max_len: int = 11, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.unembed = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.size()
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.dropout(self.embed(tokens) + self.pos_embed(pos))
        h = self.transformer(h)
        logits = self.unembed(h)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Phase Classifier
# ============================================================================

class PhaseClassifier:
    def __init__(self):
        self.history: List[Tuple[int, str, float, float]] = []

    def classify(self, train_acc: float, test_acc: float, epoch: int) -> str:
        if math.isnan(train_acc):
            return "unknown"
        if train_acc < 0.5:
            return "warmup"
        if test_acc > 0.9:
            return "grokked"
        if train_acc > 0.99 and test_acc < 0.1:
            return "memorization"
        if train_acc > 0.99 and test_acc > 0.1:
            return "transition"
        return "learning"

    def update(self, train_acc: float, test_acc: float, epoch: int) -> str:
        phase = self.classify(train_acc, test_acc, epoch)
        self.history.append((epoch, phase, train_acc, test_acc))
        return phase

    def get_summary(self) -> Dict:
        if not self.history:
            return {}
        grokking_onset = None
        grokking_completion = None
        for epoch, phase, train_acc, test_acc in self.history:
            if phase == "transition" and grokking_onset is None:
                grokking_onset = epoch
            if phase == "grokked" and grokking_completion is None:
                grokking_completion = epoch
        return {
            "grokking_onset": grokking_onset,
            "grokking_completion": grokking_completion,
            "final_phase": self.history[-1][1],
        }


# ============================================================================
# Accuracy Computation with Output Masking and Top-K
# ============================================================================

def apply_output_mask(logits: torch.Tensor, ds: torch.Tensor, int_offset: int,
                      vocab_size: int) -> torch.Tensor:
    """Vectorized output masking: set logits outside [int_offset, int_offset+d-1] to -inf.

    Args:
        logits: (batch, vocab_size) logits at the output position
        ds: (batch,) tensor of subgroup orders d per instance
        int_offset: integer offset in vocab for target tokens
        vocab_size: total vocab size

    Returns:
        Masked logits (in-place modified copy)
    """
    # Create range tensor [0, 1, ..., vocab_size-1] and compare
    # Valid range for instance b: [int_offset, int_offset + ds[b] - 1]
    col_range = torch.arange(vocab_size, device=logits.device).unsqueeze(0)  # (1, V)
    d_expanded = ds.unsqueeze(1)  # (B, 1)
    # Mask: col < int_offset OR col >= int_offset + d
    mask = (col_range < int_offset) | (col_range >= int_offset + d_expanded)
    logits = logits.clone()
    logits[mask] = float('-inf')
    return logits


def compute_accuracy_masked(model, dataset: GCDLPDataset, tokenizer: GCDLPTokenizer,
                            device, batch_size: int = 512, top_ks: List[int] = None):
    """Compute accuracy with output masking (logits >= d → -inf) and Top-K recall."""
    if top_ks is None:
        top_ks = [1, 5, 10, 100]

    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    topk_correct = {k: 0 for k in top_ks}
    chance_sum = 0.0  # sum of 1/d for chance-normalized acc

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_tokens = []
            batch_targets = []
            batch_ds = []
            for j in range(i, min(i + batch_size, len(dataset))):
                t, tgt, d = dataset.examples[j]
                batch_tokens.append(t)
                batch_targets.append(tgt)
                batch_ds.append(d)

            tokens = torch.stack(batch_tokens).to(device)
            targets = torch.tensor(batch_targets, dtype=torch.long).to(device)
            ds = torch.tensor(batch_ds, dtype=torch.long).to(device)

            logits = model(tokens)
            if logits.dim() == 3:
                seq_lens = (tokens != tokenizer.PAD_ID).sum(dim=1) - 1
                idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
            else:
                logits_at_pos = logits

            # Output masking: logits for x >= d → -inf (vectorized)
            int_offset = tokenizer.INT_OFFSET
            vocab_size = tokenizer.vocab_size
            logits_at_pos = apply_output_mask(logits_at_pos, ds, int_offset, vocab_size)

            # Loss (only on valid classes)
            loss = F.cross_entropy(logits_at_pos, targets, reduction='sum')
            loss_sum += loss.item()

            # Top-1
            preds = logits_at_pos.argmax(dim=-1)
            correct += (preds == targets).sum().item()

            # Top-K
            for k in top_ks:
                if k <= logits_at_pos.size(1):
                    _, topk_indices = logits_at_pos.topk(k, dim=-1)
                    topk_correct[k] += (topk_indices == targets.unsqueeze(-1)).any(dim=-1).sum().item()

            total += len(targets)
            for d_val in batch_ds:
                chance_sum += 1.0 / d_val

    model.train()
    acc = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / total if total > 0 else 0.0
    chance = chance_sum / total if total > 0 else 0.0
    topk_acc = {k: topk_correct[k] / total if total > 0 else 0.0 for k in top_ks}
    chance_normalized = acc - chance

    return acc, avg_loss, chance, chance_normalized, topk_acc


# ============================================================================
# Training Loop
# ============================================================================

def train_gcdlp(
    family: int,
    order_visible: bool = True,
    d_model: int = 512,
    n_layers: int = 8,
    n_heads: int = 16,
    epochs: int = 200_000,
    lr: float = 1e-3,
    wd_start: float = 0.05,
    wd_max: float = 0.30,
    wd_step: float = 0.05,
    wd_acc_threshold: float = 0.05,
    seed: int = 42,
    output_dir: str = "results/gcdlp",
    checkpoint_interval: int = 1000,
    eval_interval: int = 10,
    early_stop_patience: int = 100,
    early_stop_threshold: float = 0.99,
    use_amp: bool = True,
    ddp: bool = False,
    max_train_size: int = 500_000,
):
    """Train a GCDLP solver with progressive weight decay."""

    # DDP setup
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

    if local_rank == 0:
        print(f"Device: {device}", flush=True)

    # Build family subgroups
    family_builder = FAMILY_BUILDERS[family]
    subgroups = family_builder()

    if local_rank == 0:
        print(f"\nFamily {family} subgroups:", flush=True)
        for sg in subgroups:
            phi_d = _euler_totient(sg.d)
            N = sg.d * phi_d
            print(f"  p={sg.p}, d={sg.d}, φ(d)={phi_d}, N(d)={N}, γ={sg.gamma}", flush=True)

    # Build generator and generate splits
    gen = GCDLPGenerator(subgroups, seed=seed)
    train_specs, val_specs, test_specs = gen.generate_split()

    # Cap training size
    if len(train_specs) > max_train_size:
        rng = random.Random(seed)
        train_specs = rng.sample(train_specs, max_train_size)

    if local_rank == 0:
        print(f"\nGenerator-OOD split:", flush=True)
        print(f"  Train: {len(train_specs)}", flush=True)
        print(f"  Val:   {len(val_specs)}", flush=True)
        print(f"  Test:  {len(test_specs)}", flush=True)

    # Tokenizer
    p_max = max(sg.p for sg in subgroups)
    tokenizer = GCDLPTokenizer(p_max)
    max_len = tokenizer.max_len(order_visible)

    if local_rank == 0:
        print(f"\nTokenizer: p_max={p_max}, vocab_size={tokenizer.vocab_size}, max_len={max_len}", flush=True)
        print(f"Order visible: {order_visible}", flush=True)

    # Build datasets
    train_ds = GCDLPDataset(train_specs, tokenizer, order_visible, max_len)
    test_ds = GCDLPDataset(test_specs, tokenizer, order_visible, max_len)

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

    n_params = model.count_parameters() if hasattr(model, 'count_parameters') else \
        sum(p.numel() for p in model.parameters())
    if local_rank == 0:
        print(f"Model: {n_params:,} params, d={d_model}, L={n_layers}, H={n_heads}", flush=True)

    # Optimizer
    current_wd = wd_start
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=current_wd)
    last_wd_acc_level = 0

    # AMP
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == "cuda" else None
    amp_dtype = torch.float16 if device.type == "cuda" else None

    # Output dir
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

        config = {
            "family": family,
            "order_visible": order_visible,
            "subgroups": [{"p": sg.p, "d": sg.d, "gamma": sg.gamma} for sg in subgroups],
            "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
            "n_params": n_params,
            "epochs": epochs, "lr": lr,
            "wd_schedule": {"start": wd_start, "max": wd_max, "step": wd_step,
                            "acc_threshold": wd_acc_threshold},
            "train_size": len(train_ds), "val_size": len(val_specs),
            "test_size": len(test_ds),
            "seed": seed, "device": str(device),
            "checkpoint_interval": checkpoint_interval,
            "early_stop_patience": early_stop_patience,
            "early_stop_threshold": early_stop_threshold,
            "use_amp": use_amp, "ddp": ddp, "world_size": world_size,
            "vocab_size": tokenizer.vocab_size, "max_len": max_len,
            "p_max": p_max,
            "gen_split": {"train_frac": 0.70, "val_frac": 0.15, "test_frac": 0.15},
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # Prepare tensors
    train_tokens = torch.stack([train_ds.examples[i][0] for i in range(len(train_ds))]).to(device)
    train_targets = torch.tensor([train_ds.examples[i][1] for i in range(len(train_ds))],
                                 dtype=torch.long).to(device)
    train_ds_vals = torch.tensor([train_ds.examples[i][2] for i in range(len(train_ds))],
                                 dtype=torch.long).to(device)

    # Metrics file
    metrics_path = os.path.join(output_dir, "metrics.jsonl") if local_rank == 0 else None
    phase_clf = PhaseClassifier()

    # Training loop
    start_time = time.time()
    consecutive_high = 0
    best_test_acc = 0.0
    early_stopped = False
    full_batch_size = len(train_ds)

    if local_rank == 0:
        print(f"\nStarting training: {epochs} epochs, batch_size={full_batch_size}", flush=True)
        print(f"WD schedule: start={wd_start}, max={wd_max}, step={wd_step}, threshold={wd_acc_threshold}", flush=True)
        print(f"AMP: {use_amp and device.type == 'cuda'}, DDP: {ddp} (world_size={world_size})", flush=True)
        if ddp:
            print(f"DDP: each GPU processes {full_batch_size // world_size} samples/epoch (strided)", flush=True)
        print("", flush=True)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_ds), device=device)
        epoch_loss = 0.0
        n_batches = 0

        # DDP strided partition
        if ddp:
            gpu_indices = perm[local_rank::world_size]
        else:
            gpu_indices = perm

        for i in range(0, len(gpu_indices), full_batch_size):
            idx = gpu_indices[i:i + full_batch_size]
            xb = train_tokens[idx]
            yb = train_targets[idx]
            db = train_ds_vals[idx]

            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(xb)
                    if logits.dim() == 3:
                        seq_lens = (xb != tokenizer.PAD_ID).sum(dim=1) - 1
                        idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                        logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
                    else:
                        logits_at_pos = logits

                    # Output masking
                    logits_at_pos = apply_output_mask(
                        logits_at_pos, db,
                        tokenizer.INT_OFFSET, tokenizer.vocab_size)

                    loss = F.cross_entropy(logits_at_pos, yb)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                if logits.dim() == 3:
                    seq_lens = (xb != tokenizer.PAD_ID).sum(dim=1) - 1
                    idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                    logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
                else:
                    logits_at_pos = logits

                # Output masking (vectorized)
                logits_at_pos = apply_output_mask(
                    logits_at_pos, db,
                    tokenizer.INT_OFFSET, tokenizer.vocab_size)

                loss = F.cross_entropy(logits_at_pos, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Evaluate (only on rank 0 when DDP)
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            if ddp and local_rank != 0:
                train_acc = float("nan")
                test_acc = float("nan")
                test_loss = avg_train_loss
                chance = 0.0
                chance_norm = 0.0
                topk = {}
            else:
                train_acc, _, _, _, _ = compute_accuracy_masked(
                    model, train_ds, tokenizer, device, batch_size=512)
                test_acc, test_loss, chance, chance_norm, topk = compute_accuracy_masked(
                    model, test_ds, tokenizer, device, batch_size=512)

            # Progressive weight decay ramp
            if not math.isnan(train_acc):
                acc_level = int(train_acc / wd_acc_threshold)
                if acc_level > last_wd_acc_level and current_wd < wd_max:
                    new_wd = min(current_wd + wd_step, wd_max)
                    for pg in optimizer.param_groups:
                        pg['weight_decay'] = new_wd
                    if local_rank == 0:
                        print(f"  WD ramp: {current_wd:.3f} -> {new_wd:.3f} "
                              f"(train_acc={train_acc:.4f}, level={acc_level})", flush=True)
                    current_wd = new_wd
                    last_wd_acc_level = acc_level
        else:
            train_acc = float("nan")
            test_acc = float("nan")
            test_loss = avg_train_loss
            chance = 0.0
            chance_norm = 0.0
            topk = {}

        # DDP barrier after eval
        if ddp and (epoch % eval_interval == 0 or epoch == epochs - 1):
            torch.distributed.barrier()

        # Track best
        if not math.isnan(test_acc):
            if test_acc > best_test_acc:
                best_test_acc = test_acc

        # Phase classification
        phase = phase_clf.update(
            train_acc if not math.isnan(train_acc) else 0.0,
            test_acc if not math.isnan(test_acc) else 0.0, epoch)

        # Logging
        if local_rank == 0 and (epoch % eval_interval == 0 or epoch == epochs - 1):
            elapsed = time.time() - start_time
            topk_str = " | ".join(f"R@{k}={topk.get(k, 0):.4f}" for k in [1, 5, 10, 100])
            print(f"Epoch {epoch:6d} | train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
                  f"chance={chance:.4f} cnorm={chance_norm:+.4f} | {topk_str} | "
                  f"loss={avg_train_loss:.4f} wd={current_wd:.3f} | "
                  f"phase={phase} t={elapsed:.0f}s", flush=True)

            # Write metrics
            if metrics_path:
                metric = {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "chance": chance,
                    "chance_normalized": chance_norm,
                    "test_loss": test_loss,
                    "train_loss": avg_train_loss,
                    "weight_decay": current_wd,
                    "phase": phase,
                    "elapsed": elapsed,
                    "top1": topk.get(1, 0),
                    "top5": topk.get(5, 0),
                    "top10": topk.get(10, 0),
                    "top100": topk.get(100, 0),
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(metric) + "\n")

        # Checkpoint
        if local_rank == 0 and epoch > 0 and epoch % checkpoint_interval == 0:
            ckpt_path = os.path.join(output_dir, "checkpoints", f"ckpt_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict() if not ddp else model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "current_wd": current_wd,
                "best_test_acc": best_test_acc,
            }, ckpt_path)

        # Early stopping
        if not math.isnan(train_acc) and train_acc >= early_stop_threshold:
            consecutive_high += 1
            if consecutive_high >= early_stop_patience:
                if not math.isnan(test_acc) and test_acc < early_stop_threshold:
                    if local_rank == 0:
                        print(f"\nEarly stop at epoch {epoch}: train_acc={train_acc:.4f} "
                              f"but test_acc={test_acc:.4f} < {early_stop_threshold}", flush=True)
                    early_stopped = True
                    break
        else:
            consecutive_high = 0

    # Save summary
    if local_rank == 0:
        elapsed = time.time() - start_time
        summary = phase_clf.get_summary()
        summary.update({
            "final_train_acc": train_acc if not math.isnan(train_acc) else 0.0,
            "final_test_acc": test_acc if not math.isnan(test_acc) else 0.0,
            "best_test_acc": best_test_acc,
            "total_epochs": epoch + 1,
            "wall_time": f"{elapsed:.1f}s ({elapsed/3600:.1f}h)",
            "early_stopped": early_stopped,
            "family": family,
            "order_visible": order_visible,
        })
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Save final checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict() if not ddp else model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "current_wd": current_wd,
            "best_test_acc": best_test_acc,
        }, os.path.join(output_dir, "checkpoints", "final.pt"))

        print(f"\n{'='*60}", flush=True)
        print(f"Training complete: {epoch + 1} epochs in {elapsed/3600:.1f}h", flush=True)
        print(f"Best test acc: {best_test_acc:.4f}", flush=True)
        print(f"Final: train={train_acc:.4f}, test={test_acc:.4f}", flush=True)
        print(f"Phase: {summary.get('final_phase', '?')}", flush=True)
        status = "GROKKED" if best_test_acc >= 0.80 else \
                 "PARTIAL" if best_test_acc >= 0.10 else "FAILED"
        print(f"Status: {status}", flush=True)
        print(f"{'='*60}", flush=True)

    # Cleanup DDP
    if ddp:
        torch.distributed.destroy_process_group()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GCDLP Trainer")
    parser.add_argument("--family", type=int, default=1, choices=[1, 2, 3],
                        help="Experiment family (1=prime ladder, 2=prime+power+smooth, 3=cross-field)")
    parser.add_argument("--order-visible", action="store_true", default=True,
                        help="Include subgroup order d in input")
    parser.add_argument("--order-hidden", action="store_true", default=False,
                        help="Hide subgroup order d from input")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd-start", type=float, default=0.05)
    parser.add_argument("--wd-max", type=float, default=0.30)
    parser.add_argument("--wd-step", type=float, default=0.05)
    parser.add_argument("--wd-acc-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/gcdlp")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=100)
    parser.add_argument("--early-stop-threshold", type=float, default=0.99)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--max-train-size", type=int, default=500_000)
    args = parser.parse_args()

    order_visible = not args.order_hidden

    train_gcdlp(
        family=args.family,
        order_visible=order_visible,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        epochs=args.epochs,
        lr=args.lr,
        wd_start=args.wd_start,
        wd_max=args.wd_max,
        wd_step=args.wd_step,
        wd_acc_threshold=args.wd_acc_threshold,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        early_stop_patience=args.early_stop_patience,
        early_stop_threshold=args.early_stop_threshold,
        use_amp=not args.no_amp,
        ddp=args.ddp,
        max_train_size=args.max_train_size,
    )


if __name__ == "__main__":
    main()
