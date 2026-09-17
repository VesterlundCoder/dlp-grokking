#!/usr/bin/env python3
"""Generate checkpoint manifest with SHA256 hashes for the release.

Archives the key checkpoints used in the manuscript:
- M04 anchor (p=113, seed 42): init, memorization, pre-grok, T50, T90, T99, final
- Prime-order IID (P113-A): final
- Prime-order OOD (P113-OOD): final
"""
import hashlib
import os
import csv
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Key checkpoints to archive
CHECKPOINTS = [
    # M04 anchor (p=113, seed 42)
    {
        'run_id': 'M04_p113_s42',
        'checkpoint': 'epoch_000000.pt',
        'path': 'results/local_probe/M04_s42/checkpoints/epoch_000499.pt',  # earliest available
        'label': 'early (epoch 499)',
        'phase': 'memorization',
    },
    {
        'run_id': 'M04_p113_s42',
        'checkpoint': 'epoch_010000.pt',
        'path': 'results/local_probe/M04_s42/checkpoints/epoch_009999.pt',
        'label': 'memorization plateau',
        'phase': 'plateau',
    },
    {
        'run_id': 'M04_p113_s42',
        'checkpoint': 'epoch_020000.pt',
        'path': 'results/local_probe/M04_s42/checkpoints/epoch_019999.pt',
        'label': 'pre-grok',
        'phase': 'pre-grok',
    },
    {
        'run_id': 'M04_p113_s42',
        'checkpoint': 'epoch_022650.pt',
        'path': 'results/local_probe/M04_s42/checkpoints/epoch_022499.pt',
        'label': 'T50 vicinity',
        'phase': 'circuit-formation',
    },
    {
        'run_id': 'M04_p113_s42',
        'checkpoint': 'epoch_026830.pt',
        'path': 'results/local_probe/M04_s42/checkpoints/epoch_026999.pt',
        'label': 'T90 vicinity',
        'phase': 'circuit-formation',
    },
    {
        'run_id': 'M04_p113_s42',
        'checkpoint': 'epoch_033320.pt',
        'path': 'results/local_probe/M04_s42/checkpoints/epoch_033499.pt',
        'label': 'T99 vicinity',
        'phase': 'cleanup',
    },
    {
        'run_id': 'M04_p113_s42',
        'checkpoint': 'final.pt',
        'path': 'results/local_probe/M04_s42/checkpoints/final.pt',
        'label': 'final (epoch 37481)',
        'phase': 'final',
    },
    # Prime-order IID
    {
        'run_id': 'P113-iid_M04_s42',
        'checkpoint': 'final.pt',
        'path': 'experiments/prime113/P113-A_s42/checkpoints/final.pt',
        'label': 'prime-order IID final',
        'phase': 'final',
    },
    # Prime-order OOD
    {
        'run_id': 'P113-ood_M04_s42',
        'checkpoint': 'final.pt',
        'path': 'experiments/prime113/P113-OOD_s42/checkpoints/final.pt',
        'label': 'prime-order OOD final',
        'phase': 'final',
    },
]


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    output = os.path.join(REPO_ROOT, 'checkpoints', 'checkpoint_manifest.csv')
    os.makedirs(os.path.dirname(output), exist_ok=True)

    rows = []
    for ckpt in CHECKPOINTS:
        full_path = os.path.join(REPO_ROOT, ckpt['path'])
        if not os.path.exists(full_path):
            print(f"WARNING: Checkpoint not found: {full_path}")
            continue

        print(f"Hashing {ckpt['path']}...")
        sha256 = compute_sha256(full_path)
        size = os.path.getsize(full_path)

        rows.append({
            'run_id': ckpt['run_id'],
            'checkpoint': ckpt['checkpoint'],
            'label': ckpt['label'],
            'phase': ckpt['phase'],
            'path': ckpt['path'],
            'size_bytes': size,
            'sha256': sha256,
        })
        print(f"  SHA256: {sha256}")
        print(f"  Size: {size} bytes")

    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'checkpoint', 'label', 'phase', 'path', 'size_bytes', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest saved to {output}")
    print(f"Total checkpoints: {len(rows)}")


if __name__ == '__main__':
    main()
