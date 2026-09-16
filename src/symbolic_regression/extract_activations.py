#!/usr/bin/env python3
"""Extract activations from grokked DLP transformer checkpoints for symbolic regression.

Implements the activation extraction pipeline:
    NN checkpoint -> activation extraction -> sparse feature selection -> PySR

Extracts hidden states at each layer (embed, block_0, block_1, pre-unembed)
for all (g, h) pairs, along with analysis coordinates (a, b) where g = g0^a, h = g0^b.

Levels of extraction (from the methodology):
    Level 1: output surrogate: (g, h) -> x_hat  (hardest, least likely to give simple formula)
    Level 2: latent-state extraction: (g, h) -> z_j  (what does a neuron/direction mean?)
    Level 3: analysis-coordinate extraction: (a, b) -> z_j  (most promising)
    Level 4: subnetwork extraction: z_l -> z_{l+1}  (circuit extraction)

Usage:
    cd /Users/davidsvensson/Desktop/dlp_grokking
    python3 src/symbolic_regression/extract_activations.py \
        --checkpoint results/local_probe/M04_s42/checkpoints/final.pt \
        --output-dir results/symbolic_regression/M04_s42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import GrokkingTransformer
from src.trainer import GrokkingTokenizer, _primitive_roots


# ============================================================================
# DLP Domain Construction
# ============================================================================

def build_dlp_domain(p: int, q: int):
    """Build the full DLP domain for prime p with group order q = p-1.

    Returns:
        g0: canonical primitive root
        pairs: list of (g, h, x, a, b) where g=g0^a, h=g0^b=g^x, x=a^{-1}*b mod q
    """
    roots = _primitive_roots(p)
    g0 = roots[0]

    # For each primitive root g and each h in F_p*, compute x = dlog_g(h)
    # g = g0^a, h = g0^b, x = a^{-1} * b mod q
    pairs = []
    for g in roots:
        a = pow(g0, 1, p)  # placeholder
        # Find a such that g0^a = g mod p
        a_val = None
        val = 1
        for i in range(1, q + 1):
            val = (val * g0) % p
            if val == g:
                a_val = i
                break
        if a_val is None:
            continue

        a_inv = pow(a_val, -1, q)  # a^{-1} mod q

        # For each h in F_p*, compute x = dlog_g(h)
        for h in range(1, p):
            # Find b such that g0^b = h mod p
            b_val = None
            val2 = 1
            for j in range(1, q + 1):
                val2 = (val2 * g0) % p
                if val2 == h:
                    b_val = j
                    break
            if b_val is None:
                continue

            x = (a_inv * b_val) % q
            if x == 0:
                x = q  # convention: x in {1, ..., q}

            pairs.append((g, h, x, a_val, b_val))

    return g0, pairs


def build_dlp_domain_fast(p: int, q: int):
    """Fast domain construction using precomputed discrete log table."""
    roots = _primitive_roots(p)
    g0 = roots[0]

    # Precompute dlog table: dlog[g0^i] = i for i in 1..q
    dlog = {}
    val = 1
    for i in range(1, q + 1):
        val = (val * g0) % p
        dlog[val] = i

    pairs = []
    for g in roots:
        a = dlog[g]  # g = g0^a
        a_inv = pow(a, -1, q)

        for h in range(1, p):
            b = dlog.get(h)
            if b is None:
                continue
            x = (a_inv * b) % q
            if x == 0:
                x = q
            pairs.append((g, h, x, a, b))

    return g0, pairs


# ============================================================================
# Activation Extraction
# ============================================================================

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a GrokkingTransformer from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config = ckpt.get("config", {})
    elif isinstance(ckpt, dict) and "embed.weight" in ckpt:
        state_dict = ckpt
        config = {}
    else:
        state_dict = ckpt
        config = {}

    vocab_size, d_model = state_dict["embed.weight"].shape
    n_layers = 0
    while f"transformer.layers.{n_layers}.self_attn.in_proj_weight" in state_dict:
        n_layers += 1

    max_len = state_dict["pos_embed.weight"].shape[0]

    model = GrokkingTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=config.get("n_heads", 4),
        n_layers=n_layers,
        max_len=max_len,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def extract_activations(model, tokenizer, pairs, p, q, device, batch_size=512):
    """Extract hidden states at each layer for all (g, h) pairs.

    Returns dict with:
        'embed': (N, d_model) - embedding output at EOS position
        'block_0': (N, d_model) - after first transformer block
        'block_1': (N, d_model) - after second transformer block
        'pre_unembed': (N, d_model) - final hidden state at EOS position
        'logits': (N, vocab_size) - output logits
        'g': (N,) - generator values
        'h': (N,) - target values
        'x': (N,) - discrete log (target)
        'a': (N,) - analysis coordinate (g = g0^a)
        'b': (N,) - analysis coordinate (h = g0^b)
    """
    hooks = {}
    activations = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            # output is (B, T, d_model) for transformer layers
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 3:
                # Take the EOS position (last non-pad token)
                # For [BOS, g, SEP, h, EOS], EOS is at position 4
                activations[name] = output[:, 4, :].detach().cpu()
            else:
                activations[name] = output.detach().cpu()
        return hook_fn

    # Register hooks
    hooks['embed'] = model.embed.register_forward_hook(make_hook('embed'))
    for i in range(model.n_layers):
        hooks[f'block_{i}'] = model.transformer.layers[i].register_forward_hook(
            make_hook(f'block_{i}')
        )

    all_embed = []
    all_block_0 = []
    all_block_1 = []
    all_pre_unembed = []
    all_logits = []
    all_g = []
    all_h = []
    all_x = []
    all_a = []
    all_b = []

    n = len(pairs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_pairs = pairs[start:end]

        # Tokenize: [BOS, g, SEP, h, EOS]
        tokens = []
        for g, h, x, a, b in batch_pairs:
            g_tok = tokenizer.int_offset + (g % p)
            h_tok = tokenizer.int_offset + (h % p)
            tok = [tokenizer.bos_id, g_tok, tokenizer.sep_id, h_tok, tokenizer.eos_id]
            tokens.append(tok)

        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(token_tensor)

        # Collect activations from hooks
        all_embed.append(activations['embed'].numpy())
        all_block_0.append(activations['block_0'].numpy())
        all_block_1.append(activations['block_1'].numpy())

        # Pre-unembed: run through transformer manually
        with torch.no_grad():
            B, T = token_tensor.shape
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            h_embed = model.embed(token_tensor) + model.pos_embed(pos)
            h_out = model.transformer(h_embed)
            all_pre_unembed.append(h_out[:, 4, :].detach().cpu().numpy())

        # Logits at EOS position
        if logits.dim() == 3:
            all_logits.append(logits[:, 4, :].detach().cpu().numpy())
        else:
            all_logits.append(logits.detach().cpu().numpy())

        for g, h, x, a, b in batch_pairs:
            all_g.append(g)
            all_h.append(h)
            all_x.append(x)
            all_a.append(a)
            all_b.append(b)

    # Remove hooks
    for h in hooks.values():
        h.remove()

    return {
        'embed': np.concatenate(all_embed, axis=0),
        'block_0': np.concatenate(all_block_0, axis=0),
        'block_1': np.concatenate(all_block_1, axis=0),
        'pre_unembed': np.concatenate(all_pre_unembed, axis=0),
        'logits': np.concatenate(all_logits, axis=0),
        'g': np.array(all_g),
        'h': np.array(all_h),
        'x': np.array(all_x),
        'a': np.array(all_a),
        'b': np.array(all_b),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to grokked checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--p', type=int, default=113,
                        help='Prime p')
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    p = args.p
    q = p - 1
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    print(f"  d_model={model.d_model}, n_layers={model.n_layers}, vocab={model.vocab_size}")
    print(f"  Parameters: {model.count_parameters():,}")

    tokenizer = GrokkingTokenizer(p)

    print(f"Building DLP domain for p={p}, q={q}...")
    g0, pairs = build_dlp_domain_fast(p, q)
    print(f"  g0 = {g0}")
    print(f"  Total (g, h) pairs: {len(pairs)}")
    print(f"  Primitive roots: {len(_primitive_roots(p))}")

    print(f"Extracting activations...")
    acts = extract_activations(model, tokenizer, pairs, p, q, device, args.batch_size)

    print(f"  embed: {acts['embed'].shape}")
    print(f"  block_0: {acts['block_0'].shape}")
    print(f"  block_1: {acts['block_1'].shape}")
    print(f"  pre_unembed: {acts['pre_unembed'].shape}")
    print(f"  logits: {acts['logits'].shape}")

    # Save activations
    print(f"Saving to {args.output_dir}...")
    np.savez(
        os.path.join(args.output_dir, 'activations.npz'),
        embed=acts['embed'],
        block_0=acts['block_0'],
        block_1=acts['block_1'],
        pre_unembed=acts['pre_unembed'],
        logits=acts['logits'],
        g=acts['g'], h=acts['h'], x=acts['x'],
        a=acts['a'], b=acts['b'],
    )

    # Save metadata
    meta = {
        'checkpoint': args.checkpoint,
        'p': p, 'q': q, 'g0': g0,
        'n_pairs': len(pairs),
        'd_model': model.d_model,
        'n_layers': model.n_layers,
        'vocab_size': model.vocab_size,
        'n_params': model.count_parameters(),
        'config': config,
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == '__main__':
    main()
