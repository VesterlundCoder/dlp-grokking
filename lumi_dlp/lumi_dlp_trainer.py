#!/usr/bin/env python3
"""
LUMI DLP Grokking Trainer — Standalone training script for the
Representation-Relative Grokking study on LUMI-G (AMD MI250X).

Self-contained: bundles DLP problem generation, tokenizers, models,
CRT representation variants, prime-order subgroup experiments, and
the full training loop with WD auto-scaling.

No imports from the main codebase needed — only PyTorch + NumPy + PyYAML.

Usage (single job by index):
    python lumi_dlp_trainer.py --manifest jobs/capacity_scaling.yaml --job-index 0 \\
        --output-dir results/

Usage (all jobs sequentially):
    python lumi_dlp_trainer.py --manifest jobs/capacity_scaling.yaml --all \\
        --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ============================================================================
# Number Theory Utilities
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


# ============================================================================
# Problem Specs and Generators
# ============================================================================

@dataclass
class ProblemSpec:
    inputs: List[int]
    target: int
    metadata: Dict

    def __post_init__(self):
        self.metadata = self.metadata or {}


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
# Prime-Order Subgroup DLP (P113 experiments)
# ============================================================================

P_AMBIENT = 227
R_PRIME = 113
INT_OFFSET_P113 = 4


def build_subgroup(p_ambient: int, r_prime: int) -> List[int]:
    H = [y for y in range(1, p_ambient) if pow(y, r_prime, p_ambient) == 1]
    assert len(H) == r_prime, f"|H| = {len(H)}, expected {r_prime}"
    return sorted(H)


def find_canonical_generator(H: List[int], p_ambient: int, r_prime: int) -> int:
    for g in H:
        if g == 1:
            continue
        powers = set()
        x = 1
        for _ in range(r_prime):
            powers.add(x)
            x = (x * g) % p_ambient
        if len(powers) == r_prime:
            return g
    raise RuntimeError("No generator found")


class PrimeOrderGenerator:
    """DLP in the order-r_prime subgroup of F_p_ambient*."""

    def __init__(self, p_ambient: int, r_prime: int, seed: int = 42):
        self.p_ambient = p_ambient
        self.r_prime = r_prime
        self.seed = seed
        self.H = build_subgroup(p_ambient, r_prime)
        self.g = find_canonical_generator(self.H, p_ambient, r_prime)
        self.token_map = {h: i + INT_OFFSET_P113 for i, h in enumerate(self.H)}
        self.int_offset = INT_OFFSET_P113
        self.vocab_size = r_prime + INT_OFFSET_P113
        self.n_inputs = 2

    def solution_space_size(self) -> int:
        return self.r_prime * (self.r_prime - 1)

    def generate_all(self) -> List[ProblemSpec]:
        specs = []
        for g_val in self.H:
            if g_val == 1:
                continue
            x = 1
            for exp in range(1, self.r_prime):
                h_val = x
                specs.append(ProblemSpec(
                    inputs=[g_val, h_val],
                    target=exp,
                    metadata={"g": g_val, "h": h_val, "x": exp},
                ))
                x = (x * g_val) % self.p_ambient
        return specs

    def generate(self, n: int) -> List[ProblemSpec]:
        all_specs = self.generate_all()
        rng = random.Random(self.seed)
        rng.shuffle(all_specs)
        return all_specs[:n]

    def verify(self, spec: ProblemSpec) -> bool:
        g, h, x = spec.metadata["g"], spec.metadata["h"], spec.target
        return pow(g, x, self.p_ambient) == h


# ============================================================================
# CRT Representation Variants (Algebraic Pairs)
# ============================================================================

def coprime_factorization(q: int) -> Tuple[int, int]:
    """Factor q into two coprime factors (a, b) with a*b=q, gcd(a,b)=1,
    minimizing the ratio max(a,b)/min(a,b) for a balanced split.

    Raises ValueError if q has only one prime factor (no coprime split possible).
    """
    # Collect prime powers of q
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
        raise ValueError(
            f"q={q} has only one prime factor ({unique_primes[0]}); "
            "no coprime split into two factors both > 1 is possible."
        )

    # Try all assignments of prime powers to group A or B; pick the most balanced
    from itertools import product as _product
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
        raise ValueError(f"q={q}: could not find a coprime split with both factors > 1.")
    return best[1], best[2]


def project_factor(y: int, p: int, factor: int) -> int:
    """Project y to the order-`factor` subgroup of F_p* via y^((p-1)/factor) mod p.
    Generalizes project_16 (factor=16) and project_7 (factor=7) for arbitrary primes.
    """
    exp = (p - 1) // factor
    return pow(y, exp, p)


# Backward-compatible aliases for p=113 (q=112=7*16)
def project_16(y: int, p: int = 113) -> int:
    return project_factor(y, p, 16)

def project_7(y: int, p: int = 113) -> int:
    return project_factor(y, p, 7)


def encode_crt_representation(g: int, h: int, x: int, variant: str,
                              p: int = 113, scramble_map: Dict = None,
                              crt_factors: Tuple[int, int] = None) -> Tuple[List[int], int]:
    # Compute coprime factorization of q=p-1 if not provided
    if crt_factors is None:
        a, b = coprime_factorization(p - 1)
    else:
        a, b = crt_factors  # (smaller, larger)

    # Project to the two subgroups: a (smaller) and b (larger)
    g_a = project_factor(g, p, a)
    g_b = project_factor(g, p, b)
    h_a = project_factor(h, p, a)
    h_b = project_factor(h, p, b)

    PAD, SEP, BOS, EOS = 0, 1, 2, 3
    INT_OFFSET = 4

    def tok(val: int) -> int:
        return val + INT_OFFSET

    target = tok(x)

    if variant == "RAW":
        tokens = [BOS, tok(g), SEP, tok(h), EOS]
    elif variant == "CRT-BOTH":
        tokens = [BOS, tok(g_b), SEP, tok(g_a), SEP, tok(h_b), SEP, tok(h_a), EOS]
    elif variant == "CRT-16":
        # Project to the larger factor only (backward-compat name; uses b)
        tokens = [BOS, tok(g_b), SEP, tok(h_b), EOS]
    elif variant == "CRT-7":
        # Project to the smaller factor only (backward-compat name; uses a)
        tokens = [BOS, tok(g_a), SEP, tok(h_a), EOS]
    elif variant == "RAW+CRT":
        tokens = [BOS, tok(g), SEP, tok(h), SEP,
                  tok(g_b), SEP, tok(g_a), SEP,
                  tok(h_b), SEP, tok(h_a), EOS]
    elif variant == "SCRAMBLED":
        s_g_b = scramble_map.get(g_b, g_b)
        s_g_a = scramble_map.get(g_a, g_a)
        s_h_b = scramble_map.get(h_b, h_b)
        s_h_a = scramble_map.get(h_a, h_a)
        tokens = [BOS, tok(s_g_b), SEP, tok(s_g_a), SEP, tok(s_h_b), SEP, tok(s_h_a), EOS]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return tokens, target


# ============================================================================
# Tokenizer
# ============================================================================

class GrokkingTokenizer:
    """Vocab: [PAD=0, SEP=1, BOS=2, EOS=3, 0, 1, ..., p-1]"""
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


class PrimeOrderTokenizer:
    """Tokenizer for prime-order subgroup DLP.
    Uses token_map from PrimeOrderGenerator to map subgroup elements to tokens."""
    def __init__(self, token_map: Dict[int, int], vocab_size: int, int_offset: int = 4):
        self.token_map = token_map
        self.vocab_size = vocab_size
        self.int_offset = int_offset
        self.pad_id = 0
        self.sep_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def encode(self, spec: ProblemSpec) -> tuple:
        tokens = [self.bos_id]
        for i, inp in enumerate(spec.inputs):
            tokens.append(self.token_map[inp])
            if i < len(spec.inputs) - 1:
                tokens.append(self.sep_id)
        tokens.append(self.eos_id)
        target = self.int_offset + (spec.target % (self.vocab_size - self.int_offset))
        return tokens, target


# ============================================================================
# Dataset
# ============================================================================

class DLPDataset:
    def __init__(self, specs: List[ProblemSpec], tokenizer, max_len: int,
                 crt_variant: str = None, scramble_map: Dict = None,
                 p: int = 113, crt_factors: Tuple[int, int] = None):
        self.examples = []
        for spec in specs:
            g, h, x = spec.metadata["g"], spec.metadata["h"], spec.target
            if crt_variant:
                tokens, target = encode_crt_representation(
                    g, h, x, crt_variant, p, scramble_map, crt_factors)
            else:
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


def split_ood_by_generator(specs: List[ProblemSpec], train_frac: float, seed: int = 42):
    """OOD split: disjoint generators in train vs test."""
    rng = random.Random(seed)
    gens = list(set(s.metadata["g"] for s in specs))
    rng.shuffle(gens)
    n_train_gens = max(1, int(len(gens) * train_frac))
    train_gens = set(gens[:n_train_gens])
    train = [s for s in specs if s.metadata["g"] in train_gens]
    test = [s for s in specs if s.metadata["g"] not in train_gens]
    return train, test


# ============================================================================
# Model
# ============================================================================

class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 5, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
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

    def core_parameters(self) -> int:
        return sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)


def compute_wd_max(p_core: int, anchor_p_core: int = 393_000,
                   min_wd: float = 0.15, max_wd: float = 0.30) -> float:
    wd = min(max_wd, max_wd * math.sqrt(anchor_p_core / max(p_core, 1)))
    return max(min_wd, wd)


def estimate_params(vocab_size, d_model, n_heads, n_layers, max_len):
    embed = vocab_size * d_model + max_len * d_model
    per_layer = 12 * d_model * d_model
    unembed = d_model * vocab_size
    return embed + n_layers * per_layer + unembed


def estimate_core_params(d_model, n_layers):
    return n_layers * 12 * d_model * d_model


# ============================================================================
# Model Grid (M01-M18)
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


def auto_train_frac(model_id: str, base_frac: float = 0.3) -> float:
    model_idx = int(model_id[1:])
    if model_idx <= 6:
        return base_frac
    if model_idx <= 12:
        frac = base_frac + 0.05 * (model_idx - 6)
        return min(frac, 0.7)
    large_fracs = {13: 0.40, 14: 0.60, 15: 0.30, 16: 0.30, 17: 0.10, 18: 0.075}
    return large_fracs.get(model_idx, 0.30)


# ============================================================================
# Training Loop
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
                seq_lens = (tokens != 0).sum(dim=1) - 1
                idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
            else:
                logits_at_pos = logits
            preds = logits_at_pos.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(batch)
    return correct / max(total, 1)


def train_one_job(job, output_dir, device="cuda"):
    name = job["name"]
    exp_type = job.get("type", "standard")
    seed = job.get("seed", 42)
    epochs = int(job.get("epochs", 100000))
    lr = float(job.get("lr", 1e-3))
    wd_start = float(job.get("wd_start", 0.05))
    wd_step = float(job.get("wd_step", 0.05))
    wd_max_override = job.get("wd_max", None)
    wd_ramp_interval = int(job.get("wd_ramp_interval", 1000))
    lr_drop_factor = float(job.get("lr_drop_factor", 0.1))
    eval_interval = int(job.get("eval_interval", 10))
    checkpoint_interval = int(job.get("checkpoint_interval", 500))
    early_stop_patience = int(job.get("early_stop_patience", 100))
    early_stop_threshold = float(job.get("early_stop_threshold", 0.99))
    grad_clip = float(job.get("grad_clip", 1.0))
    use_amp = job.get("use_amp", True)
    n_layers = int(job.get("n_layers", 2))

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    outdir = Path(output_dir) / name
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Build dataset based on experiment type
    if exp_type == "standard":
        p = int(job.get("prime", 113))
        model_id = job.get("model_id", "M04")
        train_frac = float(job.get("train_frac", auto_train_frac(model_id)))

        gen = L6Generator(p, seed=seed)
        tokenizer = GrokkingTokenizer(p)
        max_len = 5
        space_size = gen.solution_space_size()
        n_train = min(int(space_size * train_frac), space_size - 1)
        n_total = min(n_train * 3, space_size)
        specs = gen.generate(n_total) if n_total < space_size else gen.generate_all()
        train_specs, test_specs = split_specs(specs, n_train, seed=seed)

        cfg = MODEL_GRID[model_id]
        d_model = cfg["d_model"]
        n_heads = cfg["n_heads"]
        vocab_size = tokenizer.vocab_size

    elif exp_type == "prime_order":
        p_ambient = int(job.get("p_ambient", 227))
        r_prime = int(job.get("r_prime", 113))
        train_frac = float(job.get("train_frac", 0.30))
        split_type = job.get("split", "iid")
        model_id = job.get("model_id", "M04")

        gen = PrimeOrderGenerator(p_ambient, r_prime, seed=seed)
        tokenizer = PrimeOrderTokenizer(gen.token_map, gen.vocab_size, INT_OFFSET_P113)
        max_len = 5
        space_size = gen.solution_space_size()
        n_train = min(int(space_size * train_frac), space_size - 1)
        specs = gen.generate_all()
        if split_type == "ood":
            train_specs, test_specs = split_ood_by_generator(specs, train_frac, seed=seed)
        else:
            train_specs, test_specs = split_specs(specs, n_train, seed=seed)

        cfg = MODEL_GRID[model_id]
        d_model = cfg["d_model"]
        n_heads = cfg["n_heads"]
        vocab_size = gen.vocab_size

    elif exp_type == "crt":
        p = int(job.get("prime", 113))
        variant = job.get("variant", "RAW")
        model_id = job.get("model_id", "M04")
        train_frac = float(job.get("train_frac", 0.30))

        gen = L6Generator(p, seed=seed)
        tokenizer = GrokkingTokenizer(p)
        max_len = 13
        space_size = gen.solution_space_size()
        n_train = min(int(space_size * train_frac), space_size - 1)
        n_total = min(n_train * 3, space_size)
        specs = gen.generate(n_total) if n_total < space_size else gen.generate_all()
        train_specs, test_specs = split_specs(specs, n_train, seed=seed)

        cfg = MODEL_GRID[model_id]
        d_model = cfg["d_model"]
        n_heads = cfg["n_heads"]
        vocab_size = tokenizer.vocab_size

        # Compute coprime factorization of q=p-1 for CRT projection
        crt_factors = coprime_factorization(p - 1)
        print(f"[{name}] CRT factors of q={p-1}: {crt_factors[0]} x {crt_factors[1]}",
              flush=True)

        scramble_map = None
        if variant == "SCRAMBLED":
            rng = random.Random(12345)
            perm = list(range(p))
            rng.shuffle(perm)
            scramble_map = {i: perm[i] for i in range(p)}
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

    # Build datasets
    if exp_type == "crt":
        train_ds = DLPDataset(train_specs, tokenizer, max_len, crt_variant=variant,
                              scramble_map=scramble_map, p=p, crt_factors=crt_factors)
        test_ds = DLPDataset(test_specs, tokenizer, max_len, crt_variant=variant,
                             scramble_map=scramble_map, p=p, crt_factors=crt_factors)
    else:
        train_ds = DLPDataset(train_specs, tokenizer, max_len)
        test_ds = DLPDataset(test_specs, tokenizer, max_len)

    # Build model
    model = GrokkingTransformer(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_len=max_len, dropout=0.0,
    ).to(device)

    n_params = model.count_parameters()
    n_core = model.core_parameters()

    if wd_max_override is not None:
        wd_max = float(wd_max_override)
    else:
        wd_max = compute_wd_max(n_core)

    print(f"[{name}] type={exp_type} params={n_params:,} core={n_core:,} "
          f"wd_max={wd_max:.4f} device={device}", flush=True)
    print(f"[{name}] train={len(train_ds)} test={len(test_ds)} "
          f"d_model={d_model} n_heads={n_heads} n_layers={n_layers}", flush=True)

    # Save config
    config = {
        "name": name, "type": exp_type, "seed": seed,
        "model_id": job.get("model_id", "M04"),
        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
        "vocab_size": vocab_size, "max_len": max_len,
        "n_params": n_params, "n_core": n_core,
        "epochs": epochs, "lr": lr,
        "wd_start": wd_start, "wd_max": wd_max, "wd_step": wd_step,
        "wd_ramp_interval": wd_ramp_interval, "lr_drop_factor": lr_drop_factor,
        "train_size": len(train_ds), "test_size": len(test_ds),
        "eval_interval": eval_interval, "checkpoint_interval": checkpoint_interval,
        "early_stop_patience": early_stop_patience,
        "early_stop_threshold": early_stop_threshold,
        "use_amp": use_amp and device.type == "cuda",
    }
    if exp_type == "crt":
        config["variant"] = variant
        config["crt_factors"] = list(crt_factors)
        config["prime"] = p
    if exp_type == "prime_order":
        config["p_ambient"] = p_ambient
        config["r_prime"] = r_prime
        config["split"] = split_type
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd_start)
    current_wd = wd_start
    current_lr = lr

    # AMP
    amp_dtype = torch.bfloat16 if (use_amp and device.type == "cuda") else None

    # Pre-load all training data to GPU
    train_tokens = torch.stack([train_ds.examples[i][0] for i in range(len(train_ds))]).to(device)
    train_targets = torch.tensor([train_ds.examples[i][1] for i in range(len(train_ds))], dtype=torch.long).to(device)

    # Training state
    memorized = False
    mem_epoch = -1
    best_test_acc = 0.0
    consecutive_high = 0
    early_stopped = False
    metrics_path = outdir / "metrics.jsonl"
    start_time = time.time()

    print(f"[{name}] Starting training: {epochs} epochs", flush=True)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_ds), device=device)
        xb = train_tokens[perm]
        yb = train_targets[perm]

        if amp_dtype is not None:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(xb)
        else:
            logits = model(xb)

        if logits.dim() == 3:
            seq_lens = (xb != 0).sum(dim=1) - 1
            idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
            logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
        else:
            logits_at_pos = logits
        loss = F.cross_entropy(logits_at_pos, yb)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        avg_loss = loss.item()

        if epoch % eval_interval == 0 or epoch == epochs - 1:
            train_acc = compute_accuracy(model, train_ds, device)
            test_acc = compute_accuracy(model, test_ds, device)

            if not memorized and train_acc > 0.99:
                memorized = True
                mem_epoch = epoch
                print(f"[{name}] MEMORIZED at epoch {epoch}", flush=True)

            if memorized and current_wd < wd_max:
                ramps = (epoch - mem_epoch) // wd_ramp_interval
                target_wd = min(wd_start + (ramps + 1) * wd_step, wd_max)
                if target_wd > current_wd:
                    for pg in optimizer.param_groups:
                        pg["weight_decay"] = target_wd
                        pg["lr"] = current_lr * lr_drop_factor
                    print(f"[{name}] WD ramp: {current_wd:.4f} -> {target_wd:.4f} "
                          f"(epoch={epoch})", flush=True)
                    current_wd = target_wd

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if epoch % 100 == 0 or test_acc > 0.01:
                elapsed = time.time() - start_time
                print(f"[{name}] Epoch {epoch:6d} | train={train_acc:.4f} test={test_acc:.4f} "
                      f"loss={avg_loss:.4f} wd={current_wd:.4f} t={elapsed:.1f}s", flush=True)

            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "train_acc": float(train_acc),
                    "test_acc": float(test_acc), "loss": float(avg_loss),
                    "wd": float(current_wd),
                    "elapsed_s": float(time.time() - start_time),
                    "best_test_acc": float(best_test_acc),
                }) + "\n")

            if test_acc > early_stop_threshold:
                consecutive_high += 1
                if consecutive_high >= early_stop_patience:
                    print(f"[{name}] Early stopped at epoch {epoch}", flush=True)
                    early_stopped = True
            else:
                consecutive_high = 0

        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:06d}.pt"
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "current_wd": current_wd, "best_test_acc": best_test_acc,
            }, ckpt_path)

        if early_stopped:
            break

    # Save final
    final_path = ckpt_dir / "final.pt"
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "current_wd": current_wd, "best_test_acc": best_test_acc,
    }, final_path)

    summary = {
        "name": name, "type": exp_type, "seed": seed,
        "total_epochs": epoch, "best_test_acc": float(best_test_acc),
        "mem_epoch": mem_epoch, "n_params": n_params, "n_core": n_core,
        "train_size": len(train_ds), "test_size": len(test_ds),
    }
    if exp_type == "crt":
        summary["variant"] = variant
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{name}] Done. Best test acc: {best_test_acc:.4f}, epochs: {epoch}", flush=True)
    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="LUMI DLP Grokking Trainer")
    ap.add_argument("--manifest", required=True, help="Path to YAML manifest")
    ap.add_argument("--job-index", type=int, default=None, help="Index of job in manifest")
    ap.add_argument("--output-dir", default="results", help="Output directory")
    ap.add_argument("--device", default="cuda", help="Device: cuda, cpu, mps")
    ap.add_argument("--all", action="store_true", help="Run all jobs sequentially")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", flush=True)
        device = torch.device("cpu")

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    jobs = manifest["jobs"]

    if args.all:
        for i, job in enumerate(jobs):
            print(f"\n{'='*60}", flush=True)
            print(f"Job {i+1}/{len(jobs)}: {job['name']}", flush=True)
            print(f"{'='*60}", flush=True)
            train_one_job(job, args.output_dir, device)
    else:
        if args.job_index is None:
            print("Error: --job-index or --all required", flush=True)
            sys.exit(1)
        if args.job_index >= len(jobs):
            print(f"Error: job_index {args.job_index} >= {len(jobs)} jobs", flush=True)
            sys.exit(1)
        job = jobs[args.job_index]
        train_one_job(job, args.output_dir, device)


if __name__ == "__main__":
    main()
