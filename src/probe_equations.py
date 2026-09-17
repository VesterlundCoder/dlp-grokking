"""Probe DLP grokking checkpoints for equation details.

For each checkpoint, computes:
1. Full accuracy on all (g, h) pairs
2. Train/test split accuracy
3. Per-generator accuracy
4. Composition consistency: dlog_g(h1*h2) = dlog_g(h1) + dlog_g(h2) mod (p-1)
5. Cyclic shift test: dlog_g(g*h) = dlog_g(h) + 1
6. Inverse test: dlog_g(h^(-1)) = -dlog_g(h)
7. Error structure analysis
8. Representation metrics (character spectrum, effective rank, probe R^2, CKA)

Usage:
    python3 src/probe_equations.py --output-dir results/local_probe/M04_s42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import GrokkingTransformer
from src.trainer import (
    L6Generator, GrokkingTokenizer, GrokkingDataset,
    split_specs, MODEL_GRID, ProblemSpec,
)
from src.representation import DLPRepresentationMonitor
from src.fourier_analysis import FourierAnalyzer, generate_fourier_report


# ============================================================================
# Helpers
# ============================================================================

def load_config(output_dir: str) -> dict:
    with open(os.path.join(output_dir, "config.json")) as f:
        return json.load(f)


def build_model(config: dict, device: torch.device) -> GrokkingTransformer:
    model = GrokkingTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_len=config["max_len"],
    ).to(device)
    return model


def load_checkpoint(model: GrokkingTransformer, ckpt_path: str, device: torch.device) -> int:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt["epoch"]


def generate_full_space(p: int, seed: int):
    gen = L6Generator(p, seed=seed)
    specs = gen.generate_all()
    return specs


def build_lookup_tables(specs, p: int):
    """Build h_to_x lookup for each generator."""
    h_to_x = {}
    for spec in specs:
        g = spec.metadata["g"]
        h = spec.metadata["h"]
        x = spec.metadata["x"]
        if g not in h_to_x:
            h_to_x[g] = {}
        h_to_x[g][h] = x
    return h_to_x


def tokenize_batch(specs, tokenizer, max_len):
    """Tokenize a batch of ProblemSpecs into tensors."""
    tokens_list = []
    targets_list = []
    for spec in specs:
        tokens, target = tokenizer.encode(spec)
        padded = tokens + [tokenizer.pad_id] * (max_len - len(tokens))
        padded = padded[:max_len]
        tokens_list.append(padded)
        targets_list.append(target)
    return (
        torch.tensor(tokens_list, dtype=torch.long),
        torch.tensor(targets_list, dtype=torch.long),
    )


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, specs, tokenizer, max_len, device, batch_size=512):
    """Evaluate model on all specs. Returns (accuracy, predictions, targets)."""
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(specs), batch_size):
            batch = specs[i:i + batch_size]
            token_tensor, target_tensor = tokenize_batch(batch, tokenizer, max_len)
            token_tensor = token_tensor.to(device)
            logits = model(token_tensor)
            preds = logits[:, -1, :].argmax(dim=-1).cpu()
            targets = target_tensor

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

            correct += (preds == targets).sum().item()
            total += len(batch)

    return correct / total, all_preds, all_targets


def per_generator_accuracy(specs, predictions, targets, p):
    """Compute accuracy for each primitive root g."""
    per_gen = {}
    for spec, pred, target in zip(specs, predictions, targets):
        g = spec.metadata["g"]
        if g not in per_gen:
            per_gen[g] = {"correct": 0, "total": 0}
        if pred == target:
            per_gen[g]["correct"] += 1
        per_gen[g]["total"] += 1
    return {
        str(g): {"acc": v["correct"] / v["total"], "correct": v["correct"], "total": v["total"]}
        for g, v in sorted(per_gen.items())
    }


# ============================================================================
# Group structure tests
# ============================================================================

def composition_test(model, specs, h_to_x, tokenizer, max_len, device, p, n_samples=500):
    """Test dlog_g(h1*h2) = dlog_g(h1) + dlog_g(h2) mod (p-1).

    Uses ground-truth exponents from the solution space, not model predictions,
    to determine the expected answer. Then checks if the model predicts correctly.
    """
    rng = random.Random(42)
    by_gen = {}
    for spec in specs:
        g = spec.metadata["g"]
        by_gen.setdefault(g, []).append(spec)

    composed_specs = []
    expected_x = []
    for g, gen_specs in by_gen.items():
        if len(gen_specs) < 2:
            continue
        n_per_gen = max(1, n_samples // len(by_gen))
        for _ in range(n_per_gen):
            s1, s2 = rng.sample(gen_specs, 2)
            h1, x1 = s1.metadata["h"], s1.metadata["x"]
            h2, x2 = s2.metadata["h"], s2.metadata["x"]
            h3 = (h1 * h2) % p
            if h3 in h_to_x[g]:
                x3 = h_to_x[g][h3]
            else:
                continue
            composed_specs.append(ProblemSpec(
                inputs=[g, h3], target=x3, metadata={"g": g, "h": h3, "x": x3}
            ))
            expected_x.append(x3)

    if not composed_specs:
        return 0.0

    batch_size = 512
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(composed_specs), batch_size):
            batch = composed_specs[i:i + batch_size]
            token_tensor, target_tensor = tokenize_batch(batch, tokenizer, max_len)
            token_tensor = token_tensor.to(device)
            logits = model(token_tensor)
            preds = logits[:, -1, :].argmax(dim=-1).cpu()
            targets = target_tensor
            correct += (preds == targets).sum().item()
            total += len(batch)

    return correct / total


def cyclic_shift_test(model, specs, h_to_x, tokenizer, max_len, device, p, batch_size=512):
    """Test dlog_g(g*h) = dlog_g(h) + 1 mod (p-1).

    For each (g, h, x), compute h' = g*h mod p, look up x' = dlog_g(h'),
    and check if the model predicts x' for (g, h').
    """
    shifted_specs = []
    for spec in specs:
        g = spec.metadata["g"]
        h = spec.metadata["h"]
        h_prime = (g * h) % p
        if g in h_to_x and h_prime in h_to_x[g]:
            x_prime = h_to_x[g][h_prime]
            shifted_specs.append(ProblemSpec(
                inputs=[g, h_prime], target=x_prime,
                metadata={"g": g, "h": h_prime, "x": x_prime}
            ))

    if not shifted_specs:
        return 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(shifted_specs), batch_size):
            batch = shifted_specs[i:i + batch_size]
            token_tensor, target_tensor = tokenize_batch(batch, tokenizer, max_len)
            token_tensor = token_tensor.to(device)
            logits = model(token_tensor)
            preds = logits[:, -1, :].argmax(dim=-1).cpu()
            targets = target_tensor
            correct += (preds == targets).sum().item()
            total += len(batch)

    return correct / total


def inverse_test(model, specs, h_to_x, tokenizer, max_len, device, p, batch_size=512):
    """Test dlog_g(h^(-1)) = -dlog_g(h) mod (p-1).

    For each (g, h, x), compute h' = h^(-1) mod p, look up x' = dlog_g(h'),
    and check if the model predicts x' for (g, h').
    """
    inv_specs = []
    for spec in specs:
        g = spec.metadata["g"]
        h = spec.metadata["h"]
        h_inv = pow(h, -1, p)
        if g in h_to_x and h_inv in h_to_x[g]:
            x_inv = h_to_x[g][h_inv]
            inv_specs.append(ProblemSpec(
                inputs=[g, h_inv], target=x_inv,
                metadata={"g": g, "h": h_inv, "x": x_inv}
            ))

    if not inv_specs:
        return 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(inv_specs), batch_size):
            batch = inv_specs[i:i + batch_size]
            token_tensor, target_tensor = tokenize_batch(batch, tokenizer, max_len)
            token_tensor = token_tensor.to(device)
            logits = model(token_tensor)
            preds = logits[:, -1, :].argmax(dim=-1).cpu()
            targets = target_tensor
            correct += (preds == targets).sum().item()
            total += len(batch)

    return correct / total


# ============================================================================
# Error analysis
# ============================================================================

def error_analysis(specs, predictions, targets, p):
    """Analyze error structure when model is wrong."""
    int_offset = 4
    errors = []
    for spec, pred, target in zip(specs, predictions, targets):
        if pred != target:
            x = spec.metadata["x"]
            pred_x = pred - int_offset
            error = (pred_x - x) % (p - 1)
            abs_error = min(error, p - 1 - error)
            errors.append({
                "g": spec.metadata["g"],
                "h": spec.metadata["h"],
                "x": x,
                "pred_x": pred_x,
                "error": error,
                "abs_error": abs_error,
            })

    if not errors:
        return {"n_errors": 0, "error_rate": 0.0}

    n_errors = len(errors)
    error_counts = {}
    for e in errors:
        k = e["error"]
        error_counts[k] = error_counts.get(k, 0) + 1

    sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])[:10]
    near_miss = sum(1 for e in errors if e["abs_error"] <= 2)
    wild_preds = sum(1 for e in errors if e["pred_x"] < 0 or e["pred_x"] >= p)

    return {
        "n_errors": n_errors,
        "error_rate": n_errors / len(specs),
        "top_errors": [{"error": k, "count": v} for k, v in sorted_errors],
        "mean_abs_error": float(np.mean([e["abs_error"] for e in errors])),
        "near_miss_fraction": near_miss / n_errors,
        "wild_prediction_count": wild_preds,
    }


# ============================================================================
# Probe single checkpoint
# ============================================================================

def probe_checkpoint(model, ckpt_path, config, all_specs, train_set, test_set,
                     h_to_x, tokenizer, device, rep_monitor):
    """Probe a single checkpoint and return results dict."""
    epoch = load_checkpoint(model, ckpt_path, device)
    max_len = config["max_len"]
    p = config["prime"]

    # Full evaluation
    full_acc, all_preds, all_targets = evaluate_model(
        model, all_specs, tokenizer, max_len, device
    )

    # Train/test accuracy
    train_acc, _, _ = evaluate_model(model, train_set, tokenizer, max_len, device)
    test_acc, _, _ = evaluate_model(model, test_set, tokenizer, max_len, device)

    # Per-generator accuracy
    per_gen = per_generator_accuracy(all_specs, all_preds, all_targets, p)

    # Group structure tests
    comp = composition_test(model, all_specs, h_to_x, tokenizer, max_len, device, p)
    cyclic = cyclic_shift_test(model, all_specs, h_to_x, tokenizer, max_len, device, p)
    inverse = inverse_test(model, all_specs, h_to_x, tokenizer, max_len, device, p)

    # Error analysis
    errors = error_analysis(all_specs, all_preds, all_targets, p)

    # Representation metrics
    rep_metrics = None
    if rep_monitor is not None:
        try:
            rep_metrics = rep_monitor.compute_metrics(model, epoch)
        except Exception as e:
            rep_metrics = {"error": str(e)}

    return {
        "epoch": epoch,
        "full_accuracy": full_acc,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "per_generator_accuracy": per_gen,
        "composition_consistency": comp,
        "cyclic_shift_accuracy": cyclic,
        "inverse_accuracy": inverse,
        "error_analysis": errors,
        "representation_metrics": rep_metrics,
    }


# ============================================================================
# Summary report
# ============================================================================

def generate_summary(results, config, output_dir):
    """Generate a Markdown summary report."""
    p = config["prime"]
    lines = []
    lines.append(f"# Equation Probe Report: {config['model_id']} (p={p})\n")
    lines.append(f"- **Model**: {config['model_id']}, d_model={config['d_model']}, "
                  f"n_heads={config['n_heads']}, n_layers={config['n_layers']}")
    lines.append(f"- **Params**: {config['P_total']:,} (P_core={config['P_core']:,})")
    lines.append(f"- **Prime**: {p}, Train frac: {config['train_frac']}, "
                  f"Seed: {config['seed']}")
    lines.append(f"- **Train size**: {config['train_size']}, Test size: {config['test_size']}")
    lines.append(f"- **Checkpoints probed**: {len(results)}\n")

    # Key milestones
    lines.append("## Key Milestones\n")
    milestones = []

    for r in results:
        ta, te = r["train_accuracy"], r["test_accuracy"]
        comp = r["composition_consistency"]
        cyclic = r["cyclic_shift_accuracy"]
        epoch = r["epoch"]

        if te > 0.10 and not any(m[1] == "test>0.10" for m in milestones):
            milestones.append((epoch, "test>0.10", f"Test acc crossed 10% (epoch {epoch})"))
        if te > 0.50 and not any(m[1] == "test>0.50" for m in milestones):
            milestones.append((epoch, "test>0.50", f"Test acc crossed 50% (epoch {epoch})"))
        if te > 0.95 and not any(m[1] == "test>0.95" for m in milestones):
            milestones.append((epoch, "test>0.95", f"Test acc crossed 95% (epoch {epoch})"))
        if te > 0.99 and not any(m[1] == "test>0.99" for m in milestones):
            milestones.append((epoch, "test>0.99", f"Test acc crossed 99% (epoch {epoch})"))
        if comp > 0.50 and not any(m[1] == "comp>0.50" for m in milestones):
            milestones.append((epoch, "comp>0.50", f"Composition consistency > 50% (epoch {epoch})"))
        if comp > 0.90 and not any(m[1] == "comp>0.90" for m in milestones):
            milestones.append((epoch, "comp>0.90", f"Composition consistency > 90% (epoch {epoch})"))
        if cyclic > 0.50 and not any(m[1] == "cyclic>0.50" for m in milestones):
            milestones.append((epoch, "cyclic>0.50", f"Cyclic shift accuracy > 50% (epoch {epoch})"))
        if cyclic > 0.90 and not any(m[1] == "cyclic>0.90" for m in milestones):
            milestones.append((epoch, "cyclic>0.90", f"Cyclic shift accuracy > 90% (epoch {epoch})"))

    if milestones:
        for epoch, _, desc in milestones:
            lines.append(f"- {desc}")
    else:
        lines.append("- No milestones reached yet")

    # Per-checkpoint table
    lines.append("\n## Per-Checkpoint Summary\n")
    lines.append("| Epoch | Train | Test | Full | Comp | Cyclic | Inverse | Err Rate |")
    lines.append("|------:|------:|-----:|-----:|-----:|-------:|--------:|---------:|")
    for r in results:
        lines.append(
            f"| {r['epoch']} | {r['train_accuracy']:.4f} | {r['test_accuracy']:.4f} "
            f"| {r['full_accuracy']:.4f} | {r['composition_consistency']:.4f} "
            f"| {r['cyclic_shift_accuracy']:.4f} | {r['inverse_accuracy']:.4f} "
            f"| {r['error_analysis']['error_rate']:.4f} |"
        )

    # Representation evolution (if available)
    rep_data = [r for r in results if r.get("representation_metrics") and "layers" in (r["representation_metrics"] or {})]
    if rep_data:
        lines.append("\n## Representation Evolution (block_1 layer)\n")
        lines.append("| Epoch | Eff Rank | S_mult | S_add | cos_R2 | sin_R2 | CKA |")
        lines.append("|------:|---------:|-------:|-------:|-------:|-------:|----:|")
        for r in rep_data:
            rm = r["representation_metrics"]
            b1 = rm.get("layers", {}).get("block_1", {})
            lines.append(
                f"| {r['epoch']} | {b1.get('eff_rank', 0):.1f} | {b1.get('S_mult', 0):.3f} "
                f"| {b1.get('S_add', 0):.3f} | {b1.get('cos_probe_r2_mean', 0):.3f} "
                f"| {b1.get('sin_probe_r2_mean', 0):.3f} | {b1.get('cka_vs_prev', 1.0):.3f} |"
            )

    # Error evolution
    lines.append("\n## Error Evolution\n")
    lines.append("| Epoch | Err Rate | Mean Abs Err | Near Miss | Top Error |")
    lines.append("|------:|---------:|------------:|----------:|-----------|")
    for r in results:
        ea = r["error_analysis"]
        top = ea.get("top_errors", [])
        top_str = f"+{top[0]['error']} ({top[0]['count']}x)" if top else "N/A"
        lines.append(
            f"| {r['epoch']} | {ea['error_rate']:.4f} "
            f"| {ea.get('mean_abs_error', 0):.1f} "
            f"| {ea.get('near_miss_fraction', 0):.2f} | {top_str} |"
        )

    report_path = os.path.join(output_dir, "probe_summary.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Probe DLP grokking checkpoints for equation details")
    ap.add_argument("--output-dir", type=str, required=True, help="Model output directory")
    ap.add_argument("--device", type=str, default="auto", help="Device: auto, mps, cuda, cpu")
    ap.add_argument("--batch-size", type=int, default=512, help="Batch size for evaluation")
    ap.add_argument("--n-comp-samples", type=int, default=500, help="Composition test samples")
    args = ap.parse_args()

    config = load_config(args.output_dir)
    p = config["prime"]
    seed = config["seed"]

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Probing {config['model_id']} (p={p}, d={config['d_model']}, "
          f"seed={seed}, device={device})")

    # Build model
    model = build_model(config, device)

    # Generate full solution space
    all_specs = generate_full_space(p, seed)
    tokenizer = GrokkingTokenizer(p)
    max_len = config["max_len"]
    space_size = len(all_specs)
    n_train = min(int(space_size * config["train_frac"]), space_size - 1)
    train_specs, test_specs = split_specs(all_specs, n_train, seed=seed)
    print(f"Solution space: {space_size} | Train: {len(train_specs)} | Test: {len(test_specs)}")

    # Build lookup tables
    h_to_x = build_lookup_tables(all_specs, p)

    # Find checkpoints
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"No checkpoints directory found at {ckpt_dir}")
        return
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir)
         if f.startswith("epoch_") and f.endswith(".pt")],
        key=lambda f: int(f.split("_")[1].split(".")[0])
    )
    print(f"Found {len(ckpt_files)} checkpoints")

    # Representation monitor
    rep_monitor = DLPRepresentationMonitor(p=p, device=device, output_dir=args.output_dir)
    rep_monitor.register_hooks(model)

    # Fourier analyzer
    fourier_analyzer = FourierAnalyzer(p=p, device=device, output_dir=args.output_dir)
    fourier_analyzer.register_hooks(model)

    # Probe each checkpoint
    results_path = os.path.join(args.output_dir, "probe_results.jsonl")
    results = []
    t0 = time.time()

    with open(results_path, "w") as f:
        for i, ckpt_file in enumerate(ckpt_files):
            ckpt_path = os.path.join(ckpt_dir, ckpt_file)
            print(f"  [{i+1}/{len(ckpt_files)}] {ckpt_file}...", end=" ", flush=True)

            result = probe_checkpoint(
                model, ckpt_path, config,
                all_specs, train_specs, test_specs,
                h_to_x, tokenizer, device, rep_monitor,
            )

            # Fourier analysis
            fourier_result = fourier_analyzer.analyze_checkpoint(model, result["epoch"])
            fourier_analyzer.save_results(
                fourier_result,
                os.path.join(args.output_dir, "fourier_results.jsonl"),
            )
            result["fourier_analysis"] = fourier_result

            f.write(json.dumps(result) + "\n")
            f.flush()
            results.append(result)

            elapsed = time.time() - t0
            print(f"epoch={result['epoch']:6d} | train={result['train_accuracy']:.4f} "
                  f"test={result['test_accuracy']:.4f} | comp={result['composition_consistency']:.4f} "
                  f"cyclic={result['cyclic_shift_accuracy']:.4f} "
                  f"| {elapsed:.1f}s")

    rep_monitor.remove_hooks()
    fourier_analyzer.remove_hooks()

    print(f"\nProbe results saved to {results_path}")

    # Generate summary
    report_path = generate_summary(results, config, args.output_dir)
    print(f"Summary report saved to {report_path}")

    # Generate Fourier report
    fourier_results = [r["fourier_analysis"] for r in results if "fourier_analysis" in r]
    if fourier_results:
        fourier_report = generate_fourier_report(fourier_results, config, args.output_dir)
        print(f"Fourier report saved to {fourier_report}")


if __name__ == "__main__":
    main()
