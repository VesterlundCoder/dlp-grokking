#!/usr/bin/env python3
"""Generate frozen train/test split manifests for the DLP datasets.

For each task/split, produces:
- split manifest (train/test indices)
- SHA256 hash
- metadata (p, q, split type, seed, counts)
"""
import json
import hashlib
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from trainer import GrokkingTokenizer, _primitive_roots

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def generate_p113_split():
    """Generate the p=113, 30% train, pair-IID split."""
    p = 113
    q = p - 1
    seed = 42
    train_frac = 0.30

    # Build full domain
    prim_roots = _primitive_roots(p)
    triples = []
    for g in prim_roots:
        for x in range(1, p):
            h = pow(g, x, p)
            triples.append((g, h, x))

    triples.sort()  # deterministic order
    n = len(triples)
    assert n == 5376, f"Expected 5376 triples, got {n}"

    # Pair-IID split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx = sorted(indices[:n_train].tolist())
    test_idx = sorted(indices[n_train:].tolist())

    # Build manifest
    manifest = {
        'task': 'DLP_p113',
        'p': p,
        'q': q,
        'group_type': 'composite (16x7)',
        'split_type': 'pair-IID',
        'seed': seed,
        'train_frac': train_frac,
        'n_total': n,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'generator_policy': 'all primitive roots of F*_p',
        'triples': triples,
        'train_indices': train_idx,
        'test_indices': test_idx,
    }

    # Compute SHA256 hash
    manifest_str = json.dumps(manifest, sort_keys=True)
    sha256 = hashlib.sha256(manifest_str.encode()).hexdigest()
    manifest['sha256'] = sha256

    return manifest


def generate_prime_order_split():
    """Generate the prime-order p=227 (order-113 subgroup) split."""
    p = 227
    q = 113  # subgroup order
    seed = 42
    train_frac = 0.30

    # Find a generator of the order-113 subgroup
    # 227 is prime, 227-1 = 226 = 2 * 113
    # Elements of order 113 are g^2 where g is a primitive root of F*_227
    prim_roots_227 = _primitive_roots(p)

    # The order-113 subgroup consists of quadratic residues
    # A generator of this subgroup is any primitive root squared
    g0 = prim_roots_227[0]
    g_sub = pow(g0, 2, p)  # generator of order-113 subgroup

    # Build all (g, h, x) triples in the subgroup
    # g ranges over all generators of the subgroup (elements of order 113)
    subgroup_gens = []
    for a in range(1, q):
        if np.gcd(a, q) == 1:
            subgroup_gens.append(pow(g_sub, a, p))

    triples = []
    for g in subgroup_gens:
        for x in range(1, q + 1):
            h = pow(g, x, p)
            triples.append((g, h, x))

    triples.sort()
    n = len(triples)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx = sorted(indices[:n_train].tolist())
    test_idx = sorted(indices[n_train:].tolist())

    manifest = {
        'task': 'DLP_prime_order_p227_q113',
        'p': p,
        'q': q,
        'group_type': 'prime-order subgroup',
        'split_type': 'pair-IID',
        'seed': seed,
        'train_frac': train_frac,
        'n_total': n,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'generator_policy': 'all generators of the order-113 subgroup of F*_227',
        'triples': triples,
        'train_indices': train_idx,
        'test_indices': test_idx,
    }

    manifest_str = json.dumps(manifest, sort_keys=True)
    sha256 = hashlib.sha256(manifest_str.encode()).hexdigest()
    manifest['sha256'] = sha256

    return manifest


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, 'splits'), exist_ok=True)

    # p=113 pair-IID
    print("Generating p=113 pair-IID split...")
    m1 = generate_p113_split()
    path1 = os.path.join(OUTPUT_DIR, 'splits', 'p113_pair_iid_s42.json')
    with open(path1, 'w') as f:
        json.dump(m1, f, indent=2)
    print(f"  Saved: {path1}")
    print(f"  SHA256: {m1['sha256']}")
    print(f"  Train: {m1['n_train']}, Test: {m1['n_test']}")

    # Prime-order p=227 q=113
    print("\nGenerating prime-order p=227 q=113 pair-IID split...")
    m2 = generate_prime_order_split()
    path2 = os.path.join(OUTPUT_DIR, 'splits', 'p227_q113_pair_iid_s42.json')
    with open(path2, 'w') as f:
        json.dump(m2, f, indent=2)
    print(f"  Saved: {path2}")
    print(f"  SHA256: {m2['sha256']}")
    print(f"  Train: {m2['n_train']}, Test: {m2['n_test']}")

    # Write summary
    summary = {
        'splits': [
            {
                'file': 'p113_pair_iid_s42.json',
                'task': m1['task'],
                'p': m1['p'],
                'q': m1['q'],
                'split_type': m1['split_type'],
                'seed': m1['seed'],
                'n_train': m1['n_train'],
                'n_test': m1['n_test'],
                'sha256': m1['sha256'],
            },
            {
                'file': 'p227_q113_pair_iid_s42.json',
                'task': m2['task'],
                'p': m2['p'],
                'q': m2['q'],
                'split_type': m2['split_type'],
                'seed': m2['seed'],
                'n_train': m2['n_train'],
                'n_test': m2['n_test'],
                'sha256': m2['sha256'],
            },
        ]
    }
    summary_path = os.path.join(OUTPUT_DIR, 'splits', 'split_manifest_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
