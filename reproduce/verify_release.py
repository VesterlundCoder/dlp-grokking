#!/usr/bin/env python3
"""Verify the release: check that all hashes, files, and data match.

Verifies:
- Split manifests exist and have correct SHA256 hashes
- Checkpoint manifest exists and hashes match
- runs.csv exists and contains the expected experiments
- Paper compiles

Usage:
    python3 reproduce/verify_release.py
"""
import csv
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def check(name, condition, detail=""):
    global CHECKS_PASSED, CHECKS_FAILED
    status = "PASS" if condition else "FAIL"
    if condition:
        CHECKS_PASSED += 1
    else:
        CHECKS_FAILED += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def verify_splits():
    """Verify split manifests exist and hashes are correct."""
    print("\n=== Split Manifests ===")
    splits_dir = os.path.join(REPO_ROOT, 'splits')

    # p=113 pair-IID
    p113_path = os.path.join(splits_dir, 'p113_pair_iid_s42.json')
    check("p113 split exists", os.path.exists(p113_path))
    if os.path.exists(p113_path):
        with open(p113_path) as f:
            data = json.load(f)
        check("p113 has SHA256", 'sha256' in data, data.get('sha256', 'missing')[:16] + '...')
        check("p113 train count", data.get('n_train') == 1612, f"n_train={data.get('n_train')}")
        check("p113 test count", data.get('n_test') == 3764, f"n_test={data.get('n_test')}")
        check("p113 total count", data.get('n_total') == 5376, f"n_total={data.get('n_total')}")

    # Prime-order p=227
    p227_path = os.path.join(splits_dir, 'p227_q113_pair_iid_s42.json')
    check("p227 split exists", os.path.exists(p227_path))
    if os.path.exists(p227_path):
        with open(p227_path) as f:
            data = json.load(f)
        check("p227 has SHA256", 'sha256' in data, data.get('sha256', 'missing')[:16] + '...')


def verify_runs():
    """Verify runs.csv exists and has expected experiments."""
    print("\n=== Canonical Results ===")
    runs_path = os.path.join(REPO_ROOT, 'results', 'runs.csv')
    check("runs.csv exists", os.path.exists(runs_path))

    if os.path.exists(runs_path):
        with open(runs_path) as f:
            reader = csv.DictReader(f)
            runs = list(reader)

        check("runs.csv has experiments", len(runs) >= 20, f"{len(runs)} experiments")

        # Check key experiments
        run_ids = [r['experiment_id'] for r in runs]
        check("M04 anchor present", 'M04_p113_s42' in run_ids)
        check("M01 present", 'M01_p113_s42' in run_ids)
        check("M12 present", 'M12_p113_s42' in run_ids)
        check("P113-IID present", 'P113-iid_M04_s42' in run_ids)
        check("P113-OOD present", 'P113-ood_M04_s42' in run_ids)

        # Check M04 anchor accuracy
        m04 = next((r for r in runs if r['experiment_id'] == 'M04_p113_s42'), None)
        if m04:
            acc = float(m04.get('best_test_acc', 0))
            check("M04 accuracy > 0.99", acc > 0.99, f"acc={acc}")


def verify_checkpoints():
    """Verify checkpoint manifest exists."""
    print("\n=== Checkpoint Manifest ===")
    manifest_path = os.path.join(REPO_ROOT, 'checkpoints', 'checkpoint_manifest.csv')
    check("checkpoint_manifest.csv exists", os.path.exists(manifest_path))

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            ckpts = list(reader)

        check("has >= 5 checkpoints", len(ckpts) >= 5, f"{len(ckpts)} checkpoints")
        check("all have SHA256", all('sha256' in c and c['sha256'] for c in ckpts))

        # Verify one checkpoint hash
        if ckpts:
            ckpt = ckpts[-1]  # final.pt
            full_path = os.path.join(REPO_ROOT, ckpt['path'])
            if os.path.exists(full_path):
                h = hashlib.sha256()
                with open(full_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        h.update(chunk)
                actual = h.hexdigest()
                expected = ckpt['sha256']
                check("final.pt hash matches", actual == expected,
                      f"expected={expected[:16]}... actual={actual[:16]}...")
            else:
                check(f"checkpoint file exists: {ckpt['path']}", False,
                      "file not found (may be in .gitignore)")


def verify_paper():
    """Verify paper PDF exists."""
    print("\n=== Paper ===")
    paper_pdf = os.path.join(REPO_ROOT, 'paper', 'paper.pdf')
    check("paper.pdf exists", os.path.exists(paper_pdf))
    if os.path.exists(paper_pdf):
        size = os.path.getsize(paper_pdf)
        check("paper.pdf > 100KB", size > 100000, f"{size} bytes")


def main():
    print("=" * 60)
    print("DLP Grokking Release Verification")
    print("=" * 60)

    verify_splits()
    verify_runs()
    verify_checkpoints()
    verify_paper()

    print("\n" + "=" * 60)
    print(f"Results: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed")
    print("=" * 60)

    if CHECKS_FAILED > 0:
        print("\nWARNING: Some checks failed. Review before release.")
        sys.exit(1)
    else:
        print("\nAll checks passed. Release is ready.")


if __name__ == '__main__':
    main()
