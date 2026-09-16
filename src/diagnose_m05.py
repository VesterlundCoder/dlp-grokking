#!/usr/bin/env python3
"""Deep diagnostic: What is M05 actually computing?

Tests whether M05 uses:
1. A non-linear (gated) frequency model
2. MLP-mediated frequency interactions
3. A lookup-table-like representation
4. A quadratic-residue-based algorithm

Usage:
    cd /Users/davidsvensson/Desktop/dlp_grokking
    python3 src/diagnose_m05.py --checkpoint results/local_probe/M05_s42/checkpoints/final.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import GrokkingTransformer
from src.trainer import GrokkingTokenizer, L6Generator, _primitive_roots


def _primitive_root(p):
    return _primitive_roots(p)[0]


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    vocab_size, d_model = state_dict["embed.weight"].shape
    n_layers = 0
    while f"transformer.layers.{n_layers}.self_attn.in_proj_weight" in state_dict:
        n_layers += 1
    max_len = state_dict["pos_embed.weight"].shape[0]
    from src.trainer import MODEL_GRID
    n_heads = 4
    for mid, cfg in MODEL_GRID.items():
        if cfg["d_model"] == d_model:
            n_heads = cfg["n_heads"]
            break
    model = GrokkingTransformer(vocab_size, d_model, n_heads, n_layers, max_len, 0.0)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, {"vocab_size": vocab_size, "d_model": d_model, "n_heads": n_heads,
                   "n_layers": n_layers, "p": vocab_size - 4}


def get_all_logits(model, p, g, device, int_offset=4):
    """Get model logits for all h in F_p* with fixed generator g."""
    bos_id, sep_id, eos_id = 2, 1, 3
    h_values = list(range(1, p))
    tokens_list = []
    for h in h_values:
        tokens = [bos_id, int_offset + g, sep_id, int_offset + h, eos_id]
        tokens_list.append(tokens)
    token_tensor = torch.tensor(tokens_list, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(token_tensor)
    last_logits = logits[:, -1, :].cpu().numpy()
    # Extract logits for c in {1, ..., p-1}
    actual_logits = np.zeros((len(h_values), p - 1))
    for i in range(len(h_values)):
        for j, c in enumerate(range(1, p)):
            actual_logits[i, j] = last_logits[i, int_offset + c]
    return actual_logits, h_values


def test_linear_model(logits, p, g, key_freqs):
    """Test the linear cosine model (matched filter)."""
    q = p - 1
    dlog_table = {}
    gx = 1
    for x in range(q):
        dlog_table[gx] = x
        gx = (gx * g) % p

    h_values = list(range(1, p))
    c_values = list(range(1, p))
    n_pairs = len(h_values) * len(c_values)
    n_features = 2 * len(key_freqs)
    X = np.zeros((n_pairs, n_features))
    y = np.zeros(n_pairs)
    for i, h in enumerate(h_values):
        x_h = dlog_table[h]
        for j, c in enumerate(c_values):
            idx = i * len(c_values) + j
            y[idx] = logits[i, j]
            for fk_idx, k in enumerate(key_freqs):
                angle = 2 * np.pi * k * (x_h - c) / q
                X[idx, 2 * fk_idx] = np.cos(angle)
                X[idx, 2 * fk_idx + 1] = np.sin(angle)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    r2 = 1 - np.sum((y - y_pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    pred_logits = y_pred.reshape(len(h_values), len(c_values))
    actual_argmax = np.argmax(logits, axis=1)
    pred_argmax = np.argmax(pred_logits, axis=1)
    acc = np.mean(actual_argmax == pred_argmax)
    return r2, acc


def test_gated_model(logits, p, g, key_freqs):
    """Test a GATED frequency model: logit(c) = Σ_k α_k * sign(χ_56(h)) * cos(2πk(log_g(h)-c)/q).

    The parity bit (k=56) gates which frequencies contribute.
    Also test quadratic gating: logit(c) = Σ_k [α_k + γ_k * χ_56(h)] * cos(2πk(log_g(h)-c)/q)
    """
    q = p - 1
    dlog_table = {}
    gx = 1
    for x in range(q):
        dlog_table[gx] = x
        gx = (gx * g) % p

    h_values = list(range(1, p))
    c_values = list(range(1, p))
    n_pairs = len(h_values) * len(c_values)

    # Build features: for each freq k, we have:
    #   cos(2πk(x_h - c)/q)  [base]
    #   χ_56(h) * cos(2πk(x_h - c)/q)  [gated by parity]
    #   χ_56(h) * sin(2πk(x_h - c)/q)  [gated sine]
    n_freqs = len(key_freqs)
    n_features = 3 * n_freqs
    X = np.zeros((n_pairs, n_features))
    y = np.zeros(n_pairs)

    for i, h in enumerate(h_values):
        x_h = dlog_table[h]
        parity = (-1) ** x_h  # χ_56(h) = (-1)^x
        for j, c in enumerate(c_values):
            idx = i * len(c_values) + j
            y[idx] = logits[i, j]
            for fk_idx, k in enumerate(key_freqs):
                angle = 2 * np.pi * k * (x_h - c) / q
                cos_val = np.cos(angle)
                sin_val = np.sin(angle)
                X[idx, 3 * fk_idx] = cos_val
                X[idx, 3 * fk_idx + 1] = parity * cos_val
                X[idx, 3 * fk_idx + 2] = parity * sin_val

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    r2 = 1 - np.sum((y - y_pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    pred_logits = y_pred.reshape(len(h_values), len(c_values))
    actual_argmax = np.argmax(logits, axis=1)
    pred_argmax = np.argmax(pred_logits, axis=1)
    acc = np.mean(actual_argmax == pred_argmax)
    return r2, acc, coeffs, n_features


def test_quadratic_model(logits, p, g, key_freqs):
    """Test a QUADRATIC frequency model with pairwise interactions.

    logit(c) = Σ_k α_k cos(θ_k) + Σ_{k<k'} γ_{kk'} cos(θ_k) cos(θ_{k'})

    where θ_k = 2πk(log_g(h) - c)/q.

    This tests whether the MLP computes products of frequencies.
    """
    q = p - 1
    dlog_table = {}
    gx = 1
    for x in range(q):
        dlog_table[gx] = x
        gx = (gx * g) % p

    h_values = list(range(1, p))
    c_values = list(range(1, p))
    n_pairs = len(h_values) * len(c_values)
    n_freqs = len(key_freqs)

    # Base features: cos and sin for each freq
    # Plus pairwise products: cos(θ_k) * cos(θ_{k'})
    n_base = 2 * n_freqs
    n_interactions = n_freqs * (n_freqs - 1) // 2
    n_features = n_base + n_interactions
    X = np.zeros((n_pairs, n_features))
    y = np.zeros(n_pairs)

    for i, h in enumerate(h_values):
        x_h = dlog_table[h]
        for j, c in enumerate(c_values):
            idx = i * len(c_values) + j
            y[idx] = logits[i, j]
            cos_vals = []
            sin_vals = []
            for fk_idx, k in enumerate(key_freqs):
                angle = 2 * np.pi * k * (x_h - c) / q
                cv = np.cos(angle)
                sv = np.sin(angle)
                cos_vals.append(cv)
                sin_vals.append(sv)
                X[idx, 2 * fk_idx] = cv
                X[idx, 2 * fk_idx + 1] = sv
            # Pairwise interactions
            col = n_base
            for a in range(n_freqs):
                for b in range(a + 1, n_freqs):
                    X[idx, col] = cos_vals[a] * cos_vals[b]
                    col += 1

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    r2 = 1 - np.sum((y - y_pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    pred_logits = y_pred.reshape(len(h_values), len(c_values))
    actual_argmax = np.argmax(logits, axis=1)
    pred_argmax = np.argmax(pred_logits, axis=1)
    acc = np.mean(actual_argmax == pred_argmax)
    return r2, acc, n_features


def test_full_fourier_model(logits, p, g, max_k=None):
    """Test with ALL q frequencies (full Fourier basis) — is the model linear at all?"""
    q = p - 1
    if max_k is None:
        max_k = q
    dlog_table = {}
    gx = 1
    for x in range(q):
        dlog_table[gx] = x
        gx = (gx * g) % p

    h_values = list(range(1, p))
    c_values = list(range(1, p))
    n_pairs = len(h_values) * len(c_values)
    n_features = 2 * max_k
    X = np.zeros((n_pairs, n_features))
    y = np.zeros(n_pairs)

    for i, h in enumerate(h_values):
        x_h = dlog_table[h]
        for j, c in enumerate(c_values):
            idx = i * len(c_values) + j
            y[idx] = logits[i, j]
            for k in range(max_k):
                angle = 2 * np.pi * k * (x_h - c) / q
                X[idx, 2 * k] = np.cos(angle)
                X[idx, 2 * k + 1] = np.sin(angle)

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    r2 = 1 - np.sum((y - y_pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    pred_logits = y_pred.reshape(len(h_values), len(c_values))
    actual_argmax = np.argmax(logits, axis=1)
    pred_argmax = np.argmax(pred_logits, axis=1)
    acc = np.mean(actual_argmax == pred_argmax)
    return r2, acc, n_features


def analyze_mlp_nonlinearity(model, p, g, device, int_offset=4):
    """Analyze what the MLP layers do — are they linear or non-linear?"""
    q = p - 1
    dlog_table = {}
    gx = 1
    for x in range(q):
        dlog_table[gx] = x
        gx = (gx * g) % p

    # Register hooks to capture pre-MLP and post-MLP activations
    activations = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            activations[name] = (inp[0].detach().cpu().numpy(), out.detach().cpu().numpy())
        return hook_fn

    hooks = []
    for layer_idx in range(model.n_layers):
        layer = model.transformer.layers[layer_idx]
        # nn.TransformerEncoderLayer: linear1 -> activation -> linear2
        hooks.append(layer.linear1.register_forward_hook(make_hook(f"layer{layer_idx}_linear1_in")))
        hooks.append(layer.linear2.register_forward_hook(make_hook(f"layer{layer_idx}_linear2_out")))

    # Run model on all h values
    bos_id, sep_id, eos_id = 2, 1, 3
    h_values = list(range(1, p))
    tokens_list = []
    for h in h_values:
        tokens = [bos_id, int_offset + g, sep_id, int_offset + h, eos_id]
        tokens_list.append(tokens)
    token_tensor = torch.tensor(tokens_list, dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(token_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Analyze: is the MLP output a linear function of its input?
    results = []
    for layer_idx in range(model.n_layers):
        # linear1 input (pre-MLP) and linear2 output (post-MLP)
        key_in = f"layer{layer_idx}_linear1_in"
        key_out = f"layer{layer_idx}_linear2_out"
        if key_in not in activations or key_out not in activations:
            continue

        # We want the activation at the last token position (EOS)
        pre_mlp = activations[key_in][0][:, -1, :]  # (n_h, d_model)
        post_mlp = activations[key_out][1][:, -1, :]  # (n_h, d_model)

        # Test linearity: fit post = A @ pre + b
        # If R^2 is high, the MLP is effectively linear
        # If R^2 is low, the MLP is doing non-linear computation
        n_h, d = pre_mlp.shape
        # Use SVD-based least squares
        X = np.hstack([pre_mlp, np.ones((n_h, 1))])  # (n_h, d+1)
        coeffs, _, _, _ = np.linalg.lstsq(X, post_mlp, rcond=None)
        post_pred = X @ coeffs
        ss_res = np.sum((post_mlp - post_pred) ** 2)
        ss_tot = np.sum((post_mlp - post_mlp.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)

        # Also test: is the GELU activation actually firing (non-zero gradient)?
        # Check the distribution of pre-activation values
        pre_stats = {
            "mean": float(pre_mlp.mean()),
            "std": float(pre_mlp.std()),
            "min": float(pre_mlp.min()),
            "max": float(pre_mlp.max()),
            "frac_positive": float(np.mean(pre_mlp > 0)),
            "frac_large": float(np.mean(np.abs(pre_mlp) > 2.0)),
        }

        results.append({
            "layer": layer_idx,
            "mlp_linearity_r2": float(r2),
            "pre_activation_stats": pre_stats,
            "d_model": d,
        })

    return results


def test_logit_structure(logits, p, g):
    """Analyze the structure of the logit matrix directly."""
    q = p - 1
    dlog_table = {}
    gx = 1
    for x in range(q):
        dlog_table[gx] = x
        gx = (gx * g) % p

    h_values = list(range(1, p))
    c_values = list(range(1, p))

    # Is logit(h, c) a function of (log_g(h) - c) mod q? (circulant structure)
    # If yes, then logit is circulant and the Fourier model should work
    # If no, the model uses information beyond the difference

    # Check: does logit(h, c) = logit(h', c') when log_g(h) - c ≡ log_g(h') - c' (mod q)?
    logit_dict = {}
    circulant_violations = 0
    circulant_total = 0
    for i, h in enumerate(h_values):
        x_h = dlog_table[h]
        for j, c in enumerate(c_values):
            diff = (x_h - c) % q
            val = logits[i, j]
            if diff in logit_dict:
                # Compare with existing values at same diff
                existing = logit_dict[diff]
                if len(existing) < 5:  # Sample a few
                    existing.append(val)
                circulant_total += 1
                if len(existing) >= 2:
                    spread = np.std(existing)
                    if spread > 0.5:  # More than 0.5 logit units of spread
                        circulant_violations += 1
            else:
                logit_dict[diff] = [val]

    # Also: is the logit matrix low-rank?
    U, S, Vt = np.linalg.svd(logits)
    total_energy = np.sum(S ** 2)
    cumsum = np.cumsum(S ** 2) / total_energy
    rank_90 = int(np.searchsorted(cumsum, 0.90) + 1)
    rank_99 = int(np.searchsorted(cumsum, 0.99) + 1)
    rank_999 = int(np.searchsorted(cumsum, 0.999) + 1)

    return {
        "circulant_total": circulant_total,
        "circulant_violations": circulant_violations,
        "circulant_violation_rate": circulant_violations / max(circulant_total, 1),
        "singular_values_top10": [float(s) for s in S[:10]],
        "rank_90": rank_90,
        "rank_99": rank_99,
        "rank_999": rank_999,
        "logit_matrix_shape": list(logits.shape),
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    p = config["p"]
    q = p - 1
    g = _primitive_root(p)
    print(f"Model: d={config['d_model']}, heads={config['n_heads']}, layers={config['n_layers']}, p={p}, q={q}, g={g}")

    output_dir = args.output_dir or os.path.dirname(os.path.dirname(args.checkpoint))
    os.makedirs(output_dir, exist_ok=True)

    # Get all logits
    print("\n=== Computing model logits for all (h, c) pairs ===")
    logits, h_values = get_all_logits(model, p, g, device)
    print(f"Logit matrix shape: {logits.shape}")
    print(f"Logit range: [{logits.min():.3f}, {logits.max():.3f}], mean={logits.mean():.3f}, std={logits.std():.3f}")

    # Model accuracy
    dlog_table = {}
    gx = 1
    for x in range(q):
        dlog_table[gx] = x
        gx = (gx * g) % p
    actual_answers = np.array([dlog_table[h] for h in h_values])
    pred_answers = np.argmax(logits, axis=1) + 1  # c_values start at 1
    model_acc = np.mean(pred_answers == actual_answers)
    print(f"Model accuracy: {model_acc:.4f}")

    # Test 1: Linear matched filter (baseline)
    print("\n=== Test 1: Linear matched filter (top-20 freqs) ===")
    # Use the top-20 frequencies from the embedding spectrum
    key_freqs = [56, 88, 24, 104, 8, 40, 72, 14, 98, 42, 70, 48, 64, 16, 96, 28, 84, 32, 80, 21]
    r2, acc = test_linear_model(logits, p, g, key_freqs)
    print(f"  R² = {r2:.6f}, Argmax accuracy = {acc:.4f}")

    # Test 2: Full Fourier basis (all 112 frequencies)
    print("\n=== Test 2: Full Fourier basis (all 112 frequencies) ===")
    r2_full, acc_full, n_feat = test_full_fourier_model(logits, p, g, max_k=q)
    print(f"  R² = {r2_full:.6f}, Argmax accuracy = {acc_full:.4f}, n_features = {n_feat}")

    # Test 3: Gated model (parity-gated frequencies)
    print("\n=== Test 3: Parity-gated frequency model ===")
    print("  Model: logit(c) = Σ_k [α_k + γ_k * (-1)^x] * cos(2πk(log_g(h)-c)/q) + ...")
    r2_gated, acc_gated, coeffs_gated, n_gated = test_gated_model(logits, p, g, key_freqs[:15])
    print(f"  R² = {r2_gated:.6f}, Argmax accuracy = {acc_gated:.4f}, n_features = {n_gated}")

    # Test 4: Quadratic model (pairwise frequency interactions)
    print("\n=== Test 4: Quadratic frequency model (pairwise interactions) ===")
    print("  Model: logit(c) = Σ_k α_k cos(θ_k) + Σ_{k<k'} γ_{kk'} cos(θ_k)cos(θ_{k'})")
    # Use top-10 frequencies to keep feature count manageable
    r2_quad, acc_quad, n_quad = test_quadratic_model(logits, p, g, key_freqs[:10])
    print(f"  R² = {r2_quad:.6f}, Argmax accuracy = {acc_quad:.4f}, n_features = {n_quad}")

    # Test 5: MLP linearity analysis
    print("\n=== Test 5: MLP linearity analysis ===")
    mlp_results = analyze_mlp_nonlinearity(model, p, g, device)
    for r in mlp_results:
        print(f"  Layer {r['layer']}: MLP linearity R² = {r['mlp_linearity_r2']:.6f}")
        s = r["pre_activation_stats"]
        print(f"    Pre-activation: mean={s['mean']:.3f}, std={s['std']:.3f}, "
              f"frac>0={s['frac_positive']:.3f}, frac|.|>2={s['frac_large']:.3f}")

    # Test 6: Logit matrix structure
    print("\n=== Test 6: Logit matrix structure ===")
    structure = test_logit_structure(logits, p, g)
    print(f"  Circulant violation rate: {structure['circulant_violation_rate']:.4f}")
    print(f"  Singular values (top-10): {[f'{s:.2f}' for s in structure['singular_values_top10']]}")
    print(f"  Rank for 90% energy: {structure['rank_90']}")
    print(f"  Rank for 99% energy: {structure['rank_99']}")
    print(f"  Rank for 99.9% energy: {structure['rank_999']}")
    print(f"  Logit matrix shape: {structure['logit_matrix_shape']}")

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Model accuracy:           {model_acc:.4f}")
    print(f"Linear model (20 freqs):  R²={r2:.4f}, acc={acc:.4f}")
    print(f"Full Fourier (112 freqs): R²={r2_full:.4f}, acc={acc_full:.4f}")
    print(f"Gated model (15 freqs):   R²={r2_gated:.4f}, acc={acc_gated:.4f}")
    print(f"Quadratic model (10 freqs): R²={r2_quad:.4f}, acc={acc_quad:.4f}")
    print(f"MLP linearity R² (L0):    {mlp_results[0]['mlp_linearity_r2']:.4f}" if mlp_results else "N/A")
    print(f"MLP linearity R² (L1):    {mlp_results[1]['mlp_linearity_r2']:.4f}" if len(mlp_results) > 1 else "N/A")
    print(f"Logit matrix rank (99%):  {structure['rank_99']}")
    print(f"Circulant violations:     {structure['circulant_violation_rate']:.4f}")

    print("\nInterpretation:")
    if r2_full < 0.1:
        print("  → The model's logits are NOT a linear function of Fourier features.")
        print("  → The MLP is doing essential non-linear computation.")
    else:
        print("  → The model IS linear in Fourier features (just needs more frequencies).")

    if structure["circulant_violation_rate"] < 0.01:
        print("  → Logit matrix IS circulant (depends only on log_g(h) - c mod q).")
        print("  → The algorithm is a true convolution in the exponent domain.")
    else:
        print("  → Logit matrix is NOT circulant — the model uses info beyond log_g(h) - c.")
        print("  → The model may be using the specific value of h, not just its discrete log.")

    if mlp_results:
        mlp_r2 = mlp_results[0]["mlp_linearity_r2"]
        if mlp_r2 < 0.9:
            print(f"  → MLP layer 0 is strongly non-linear (R²={mlp_r2:.4f}).")
            print("  → The GELU activation is doing significant computation.")
        else:
            print(f"  → MLP layer 0 is approximately linear (R²={mlp_r2:.4f}).")

    if r2_gated > 0.5:
        print(f"  → Gated model works well (R²={r2_gated:.4f}) — parity gating is key!")
    elif r2_quad > 0.5:
        print(f"  → Quadratic model works well (R²={r2_quad:.4f}) — frequency interactions are key!")
    else:
        print("  → Neither gated nor quadratic model captures the algorithm.")
        print("  → The model may use higher-order non-linear interactions.")

    # Save results
    results = {
        "model_accuracy": float(model_acc),
        "linear_model": {"r2": r2, "accuracy": acc},
        "full_fourier": {"r2": r2_full, "accuracy": acc_full, "n_features": n_feat},
        "gated_model": {"r2": r2_gated, "accuracy": acc_gated, "n_features": n_gated},
        "quadratic_model": {"r2": r2_quad, "accuracy": acc_quad, "n_features": n_quad},
        "mlp_analysis": mlp_results,
        "logit_structure": structure,
    }
    results_path = os.path.join(output_dir, "m05_diagnostic.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
