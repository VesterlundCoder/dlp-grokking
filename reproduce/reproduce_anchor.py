#!/usr/bin/env python3
"""Reproduce the M04 anchor experiment (p=113, seed 42, 30% train).

Generates the p=113 DLP dataset, builds the M04 model, and trains for a
configurable number of epochs. This is the end-to-end reproduction command.

Usage:
    python3 reproduce/reproduce_anchor.py --epochs 1000
    python3 reproduce/reproduce_anchor.py --epochs 50000  # full grokking run
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from models import GrokkingTransformer
from trainer import GrokkingTokenizer, _primitive_roots


def build_dataset(p=113, seed=42, train_frac=0.30):
    """Build the p=113 DLP dataset with pair-IID split."""
    q = p - 1
    prim_roots = _primitive_roots(p)
    triples = []
    for g in prim_roots:
        for x in range(1, p):
            h = pow(g, x, p)
            triples.append((g, h, x))
    triples.sort()

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(triples))
    n_train = int(len(triples) * train_frac)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return triples, train_idx.tolist(), test_idx.tolist()


def main():
    parser = argparse.ArgumentParser(description='Reproduce M04 DLP anchor experiment')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-frac', type=float, default=0.30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd-start', type=float, default=0.05)
    parser.add_argument('--wd-max', type=float, default=0.30)
    parser.add_argument('--wd-step', type=float, default=0.05)
    parser.add_argument('--wd-ramp-interval', type=int, default=1000)
    parser.add_argument('--eval-interval', type=int, default=100)
    args = parser.parse_args()

    p = args.p
    q = p - 1
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"p={p}, q={q}, seed={args.seed}, train_frac={args.train_frac}")

    # Build dataset
    print("Building dataset...")
    triples, train_idx, test_idx = build_dataset(p, args.seed, args.train_frac)
    print(f"  Total: {len(triples)}, Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Tokenize
    tokenizer = GrokkingTokenizer(p)
    train_tokens = []
    train_targets = []
    for i in train_idx:
        g, h, x = triples[i]
        tokens, target = tokenizer.encode((g, h, x))
        train_tokens.append(tokens)
        train_targets.append(target)

    test_tokens = []
    test_targets = []
    for i in test_idx:
        g, h, x = triples[i]
        tokens, target = tokenizer.encode((g, h, x))
        test_tokens.append(tokens)
        test_targets.append(target)

    train_tokens = torch.stack(train_tokens).to(device)
    train_targets = torch.tensor(train_targets, dtype=torch.long).to(device)
    test_tokens = torch.stack(test_tokens).to(device)
    test_targets = torch.tensor(test_targets, dtype=torch.long).to(device)

    # Build model (M04 config)
    d_model = 128
    n_heads = 4
    n_layers = 2
    vocab_size = p + 4  # PAD, SEP, BOS, EOS + 0..p-1
    model = GrokkingTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=5,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: M04 ({n_params:,} params, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers})")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd_start)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_test_acc = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Progressive WD schedule
        model.train()
        optimizer.zero_grad()
        logits = model(train_tokens)
        loss = criterion(logits, train_targets)
        loss.backward()
        optimizer.step()

        # WD ramp after memorization
        train_acc = (logits.argmax(dim=-1) == train_targets).float().mean().item()
        if train_acc > 0.99:
            wd = min(args.wd_start + args.wd_step * ((epoch - 1) // args.wd_ramp_interval),
                     args.wd_max)
            for g in optimizer.param_groups:
                g['weight_decay'] = wd

        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_tokens)
                test_acc = (test_logits.argmax(dim=-1) == test_targets).float().mean().item()
            best_test_acc = max(best_test_acc, test_acc)
            elapsed = time.time() - t0
            wd_val = optimizer.param_groups[0]['weight_decay']
            print(f"  epoch {epoch:>6d} | train={train_acc:.4f} test={test_acc:.4f} "
                  f"loss={loss.item():.4f} wd={wd_val:.4f} best={best_test_acc:.4f} t={elapsed:.1f}s")

    print(f"\nDone. Best test accuracy: {best_test_acc:.4f}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
