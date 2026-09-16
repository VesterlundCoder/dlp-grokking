#!/usr/bin/env python3
"""Extract the learned DLP algorithm from a grokked transformer checkpoint.

Phase 1: Fourier Structure Extraction

Loads a grokked checkpoint, extracts embedding (W_E) and unembedding (W_U)
weight matrices, projects them onto the multiplicative character basis,
identifies key frequencies, and fits a closed-form trigonometric model:

    logit(c) ≈ Σ_{k ∈ K} α_k cos(2πk(log_g(h) - c)/q) + β_k sin(2πk(log_g(h) - c)/q)

Usage:
    cd /Users/davidsvensson/Desktop/dlp_grokking
    python3 src/extract_algorithm.py --checkpoint results/local_probe/M04_s42/checkpoints/final.pt
    python3 src/extract_algorithm.py --checkpoint results/local_probe/M04_s42/checkpoints/final.pt --top-k 20 --output-dir results/extraction/M04_s42
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


def _primitive_root(p: int) -> int:
    """Find the smallest primitive root modulo p."""
    roots = _primitive_roots(p)
    return roots[0] if roots else 1


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a GrokkingTransformer from a checkpoint, inferring config from weights."""
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

    # Infer architecture from state dict
    vocab_size, d_model = state_dict["embed.weight"].shape
    n_layers = 0
    while f"transformer.layers.{n_layers}.self_attn.in_proj_weight" in state_dict:
        n_layers += 1

    # Infer max_len from pos_embed weight shape
    max_len = state_dict["pos_embed.weight"].shape[0]

    # Infer n_heads from MODEL_GRID based on d_model
    from src.trainer import MODEL_GRID
    n_heads = config.get("n_heads", None)
    if n_heads is None:
        for mid, cfg in MODEL_GRID.items():
            if cfg["d_model"] == d_model:
                n_heads = cfg["n_heads"]
                break
        if n_heads is None:
            n_heads = 4  # fallback

    model = GrokkingTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
        dropout=0.0,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    config_inferred = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "p": vocab_size - 4,  # vocab = p + 4
    }

    return model, config_inferred


def compute_multiplicative_character_transform(
    W: np.ndarray, p: int, g: int, int_offset: int = 4
) -> np.ndarray:
    """Compute the multiplicative character transform of a weight matrix.

    W has shape (vocab_size, d) where vocab_size = p + 4.
    We extract rows corresponding to h ∈ {1, ..., p-1} (indices int_offset+1 .. int_offset+p-1)
    and project onto multiplicative characters χ_k(h) = exp(2πi·k·log_g(h)/q).

    Returns:
        W_hat: (q, d) complex — the multiplicative Fourier transform
    """
    q = p - 1
    d = W.shape[1]

    # Extract h-token rows: h ∈ {1, ..., p-1} → vocab indices int_offset+h
    # We exclude h=0 since 0 ∉ F_p*
    h_values = []
    x_values = []  # discrete logs
    for h in range(1, p):
        h_values.append(h)
        x_values.append(pow(g, 0, p))  # placeholder

    # Compute discrete logs: x = log_g(h)
    dlog_table = {}
    gx = 1
    for x in range(q):
        dlog_table[gx] = x
        gx = (gx * g) % p

    x_values = [dlog_table[h] for h in h_values]

    # Build multiplicative character matrix: χ_k(h) = exp(2πi·k·x/q)
    # Shape: (q, n_h) where n_h = p-1
    k_vals = np.arange(q)
    x_arr = np.array(x_values, dtype=np.float64)
    chi = np.exp(2j * np.pi * np.outer(k_vals, x_arr) / q)  # (q, n_h)

    # Extract weight rows for h ∈ {1, ..., p-1}
    W_h = W[int_offset + 1 : int_offset + p, :]  # (p-1, d)

    # Compute transform: W_hat[k, :] = Σ_h χ_k(h)* · W_h[h, :]
    W_hat = chi.conj() @ W_h  # (q, d) complex

    return W_hat


def identify_key_frequencies(W_hat: np.ndarray, top_k: int = 20) -> list:
    """Identify the key frequencies with highest energy.

    Args:
        W_hat: (q, d) complex multiplicative character transform
        top_k: number of top frequencies to return

    Returns:
        List of dicts: [{freq, energy, fraction, norm}]
    """
    energy = np.sum(np.abs(W_hat) ** 2, axis=1)  # (q,) per-frequency energy
    total = energy.sum()

    sorted_idx = np.argsort(energy)[::-1][:top_k]

    results = []
    for idx in sorted_idx:
        results.append({
            "freq": int(idx),
            "energy": float(energy[idx]),
            "fraction": float(energy[idx] / total) if total > 0 else 0.0,
            "norm": float(np.linalg.norm(W_hat[idx, :])),
        })

    return results, energy


def fit_closed_form_model(
    model: GrokkingTransformer,
    p: int,
    g: int,
    key_freqs: list,
    device: torch.device,
    int_offset: int = 4,
) -> dict:
    """Fit the closed-form trigonometric model to the model's logits.

    Model: logit(c) ≈ Σ_{k ∈ K} α_k cos(2πk(log_g(h) - c)/q) + β_k sin(2πk(log_g(h) - c)/q)

    We solve for α_k, β_k via least squares on the model's actual logits.

    Returns:
        Dict with coefficients, fit quality, and predictions
    """
    q = p - 1

    # Build dlog table
    dlog_table = {}
    gx = 1
    for x in range(q):
        dlog_table[gx] = x
        gx = (gx * g) % p

    # Generate all (h, c) pairs and compute model logits
    bos_id = 2
    sep_id = 1
    eos_id = 3

    h_values = list(range(1, p))  # h ∈ F_p*
    c_values = list(range(1, p))  # c ∈ F_p* (target classes are 1..p-1)

    # Run model on all h inputs (with fixed generator g)
    tokens_list = []
    for h in h_values:
        tokens = [bos_id, int_offset + g, sep_id, int_offset + h, eos_id]
        tokens_list.append(tokens)

    token_tensor = torch.tensor(tokens_list, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(token_tensor)  # (n_h, seq_len, vocab_size)

    # Use the last token position (EOS) logits
    last_logits = logits[:, -1, :].cpu().numpy()  # (n_h, vocab_size)

    # Extract logits for candidate outputs c ∈ {1, ..., p-1}
    # Target token = int_offset + c
    actual_logits = np.zeros((len(h_values), len(c_values)))
    for i, h in enumerate(h_values):
        for j, c in enumerate(c_values):
            actual_logits[i, j] = last_logits[i, int_offset + c]

    # Build design matrix: for each (h, c) pair, features are cos/sin at key frequencies
    n_pairs = len(h_values) * len(c_values)
    n_features = 2 * len(key_freqs)

    X = np.zeros((n_pairs, n_features))
    y = np.zeros(n_pairs)

    for i, h in enumerate(h_values):
        x_h = dlog_table[h]  # log_g(h)
        for j, c in enumerate(c_values):
            pair_idx = i * len(c_values) + j
            y[pair_idx] = actual_logits[i, j]
            for fk_idx, k in enumerate(key_freqs):
                angle = 2 * np.pi * k * (x_h - c) / q
                X[pair_idx, 2 * fk_idx] = np.cos(angle)
                X[pair_idx, 2 * fk_idx + 1] = np.sin(angle)

    # Solve least squares: y = X @ [α_1, β_1, α_2, β_2, ...]
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    # Compute predictions
    y_pred = X @ coeffs
    r2 = 1 - np.sum((y - y_pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    # Check if argmax matches (classification accuracy)
    pred_logits = y_pred.reshape(len(h_values), len(c_values))
    actual_argmax = np.argmax(actual_logits, axis=1)
    pred_argmax = np.argmax(pred_logits, axis=1)
    accuracy = np.mean(actual_argmax == pred_argmax)

    # Per-frequency contribution
    freq_contributions = []
    for fk_idx, k in enumerate(key_freqs):
        alpha_k = coeffs[2 * fk_idx]
        beta_k = coeffs[2 * fk_idx + 1]
        contribution = alpha_k ** 2 + beta_k ** 2
        freq_contributions.append({
            "freq": k,
            "alpha": float(alpha_k),
            "beta": float(beta_k),
            "contribution": float(contribution),
        })

    # Sort by contribution
    freq_contributions.sort(key=lambda x: -x["contribution"])

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "argmax_accuracy": float(accuracy),
        "n_key_freqs": len(key_freqs),
        "coefficients": freq_contributions,
        "n_samples": n_pairs,
        "rank": int(rank),
    }


def sweep_key_frequency_count(
    model: GrokkingTransformer,
    p: int,
    g: int,
    all_energy: np.ndarray,
    device: torch.device,
    max_k: int = 30,
) -> list:
    """Sweep over number of key frequencies to find the minimum |K| for high accuracy.

    Returns:
        List of dicts: [{n_freqs, r2, rmse, accuracy}]
    """
    q = p - 1
    sorted_freqs = np.argsort(all_energy)[::-1]

    results = []
    for n_k in range(1, min(max_k + 1, q)):
        key_freqs = sorted_freqs[:n_k].tolist()
        fit = fit_closed_form_model(model, p, g, key_freqs, device)
        results.append({
            "n_freqs": n_k,
            "r2": fit["r2"],
            "rmse": fit["rmse"],
            "accuracy": fit["argmax_accuracy"],
            "freqs": key_freqs,
        })
        if fit["argmax_accuracy"] >= 0.99 and n_k >= 3:
            # Found sufficient — but keep going a few more for the curve
            pass
        if fit["argmax_accuracy"] >= 0.999 and n_k >= 3:
            break

    return results


def extract_attention_head_frequencies(
    model: GrokkingTransformer,
    p: int,
    g: int,
    device: torch.device,
    int_offset: int = 4,
) -> list:
    """Extract per-head frequency specialization.

    For each attention head, compute the OV circuit and project onto
    multiplicative characters to identify which frequency it specializes in.

    Returns:
        List of dicts: [{layer, head, dominant_freq, freq_energy, freq_fraction}]
    """
    q = p - 1
    d_model = model.d_model
    n_heads = model.n_heads
    d_head = d_model // n_heads

    # Build dlog table
    dlog_table = {}
    gx = 1
    for x in range(q):
        dlog_table[gx] = x
        gx = (gx * g) % p

    # Build multiplicative character matrix
    k_vals = np.arange(q)
    x_arr = np.array([dlog_table[h] for h in range(1, p)], dtype=np.float64)
    chi = np.exp(2j * np.pi * np.outer(k_vals, x_arr) / q)  # (q, n_h)

    results = []

    for layer_idx in range(model.n_layers):
        layer = model.transformer.layers[layer_idx]

        # Extract attention weights
        # in_proj_weight has shape (3*d_model, d_model) = [Q; K; V]
        in_proj = layer.self_attn.in_proj_weight.detach().cpu().numpy()  # (3*d, d)
        out_proj = layer.self_attn.out_proj.weight.detach().cpu().numpy()  # (d, d)

        # Split into Q, K, V
        W_Q = in_proj[:d_model, :]       # (d, d)
        W_K = in_proj[d_model:2*d_model, :]  # (d, d)
        W_V = in_proj[2*d_model:, :]     # (d, d)

        for head_idx in range(n_heads):
            # Extract head's slice
            q_start = head_idx * d_head
            q_end = q_start + d_head

            W_V_h = W_V[q_start:q_end, :]  # (d_head, d_model)
            W_O_h = out_proj[:, q_start:q_end]  # (d_model, d_head)

            # OV circuit: W_O @ W_V, shape (d_model, d_model)
            # But we want the effect on the output logits
            # OV_circuit[h_out, h_in] = how input at position h_in affects output at h_out
            # For frequency analysis, we look at how the OV circuit maps
            # embedding(h) → contribution to output

            # Simpler: look at W_V_h @ W_E[h, :] for each h, then project onto chars
            W_E = model.embed.weight.detach().cpu().numpy()
            W_E_h = W_E[int_offset + 1 : int_offset + p, :]  # (p-1, d_model)

            # Compute V projections: V[h, :] = W_V_h @ W_E_h[h, :]
            V_projections = W_E_h @ W_V_h.T  # (p-1, d_head)

            # Project V projections onto multiplicative characters
            V_hat = chi.conj() @ V_projections  # (q, d_head) complex
            V_energy = np.sum(np.abs(V_hat) ** 2, axis=1)  # (q,)

            total_energy = V_energy.sum()
            if total_energy < 1e-12:
                continue

            dominant_freq = int(np.argmax(V_energy))
            freq_fraction = float(V_energy[dominant_freq] / total_energy)

            # Top-3 frequencies
            top3_idx = np.argsort(V_energy)[::-1][:3]
            top3 = [(int(idx), float(V_energy[idx] / total_energy)) for idx in top3_idx]

            results.append({
                "layer": layer_idx,
                "head": head_idx,
                "dominant_freq": dominant_freq,
                "freq_fraction": freq_fraction,
                "top3_freqs": top3,
                "total_energy": float(total_energy),
            })

    return results


def generate_extraction_report(
    config: dict,
    embed_key_freqs: list,
    embed_energy: np.ndarray,
    unembed_key_freqs: list,
    unembed_energy: np.ndarray,
    closed_form_fit: dict,
    freq_sweep: list,
    head_analysis: list,
    output_dir: str,
) -> str:
    """Generate a Markdown report of the extracted algorithm."""
    p = config["p"]
    q = p - 1

    lines = []
    lines.append(f"# DLP Algorithm Extraction Report\n")
    lines.append(f"## Model Configuration\n")
    lines.append(f"- **Prime**: p = {p}, q = p-1 = {q}")
    lines.append(f"- **d_model**: {config['d_model']}")
    lines.append(f"- **n_heads**: {config['n_heads']}")
    lines.append(f"- **n_layers**: {config['n_layers']}")
    lines.append(f"- **Vocab size**: {config['vocab_size']}")
    lines.append("")

    # Embedding analysis
    lines.append(f"## Embedding (W_E) Multiplicative Character Spectrum\n")
    lines.append(f"| Rank | Frequency k | Energy | Fraction | L2 Norm |")
    lines.append(f"|-----:|------------:|-------:|---------:|--------:|")
    for i, f in enumerate(embed_key_freqs[:15]):
        lines.append(f"| {i+1} | {f['freq']} | {f['energy']:.4f} | {f['fraction']:.4f} | {f['norm']:.4f} |")
    lines.append("")

    total_conc = sum(f["fraction"] for f in embed_key_freqs[:4])
    lines.append(f"Top-4 concentration: {total_conc:.4f}")
    lines.append(f"Top-8 concentration: {sum(f['fraction'] for f in embed_key_freqs[:8]):.4f}")
    lines.append("")

    # Unembedding analysis
    lines.append(f"## Unembedding (W_U) Multiplicative Character Spectrum\n")
    lines.append(f"| Rank | Frequency k | Energy | Fraction | L2 Norm |")
    lines.append(f"|-----:|------------:|-------:|---------:|--------:|")
    for i, f in enumerate(unembed_key_freqs[:15]):
        lines.append(f"| {i+1} | {f['freq']} | {f['energy']:.4f} | {f['fraction']:.4f} | {f['norm']:.4f} |")
    lines.append("")

    # Closed-form fit
    lines.append(f"## Closed-Form Model Fit\n")
    lines.append(f"Model: logit(c) ≈ Σ_{{k ∈ K}} α_k cos(2πk(log_g(h) - c)/q) + β_k sin(2πk(log_g(h) - c)/q)\n")
    lines.append(f"- **Number of key frequencies**: {closed_form_fit['n_key_freqs']}")
    lines.append(f"- **R²**: {closed_form_fit['r2']:.6f}")
    lines.append(f"- **RMSE**: {closed_form_fit['rmse']:.6f}")
    lines.append(f"- **Argmax accuracy**: {closed_form_fit['argmax_accuracy']:.6f}")
    lines.append(f"- **Rank**: {closed_form_fit['rank']}")
    lines.append("")

    lines.append(f"### Top Frequency Coefficients\n")
    lines.append(f"| Frequency k | α_k (cos) | β_k (sin) | Contribution |")
    lines.append(f"|------------:|----------:|----------:|-------------:|")
    for c in closed_form_fit["coefficients"][:10]:
        lines.append(f"| {c['freq']} | {c['alpha']:.6f} | {c['beta']:.6f} | {c['contribution']:.6f} |")
    lines.append("")

    # Frequency sweep
    lines.append(f"## Key Frequency Count Sweep\n")
    lines.append(f"How many frequencies are needed for high accuracy?\n")
    lines.append(f"| |K| | R² | RMSE | Argmax Acc | Key Frequencies |")
    lines.append(f"|-----:|---:|-----:|----------:|-----------------|")
    for s in freq_sweep:
        freqs_str = ", ".join(str(f) for f in s["freqs"][:8])
        if len(s["freqs"]) > 8:
            freqs_str += ", ..."
        lines.append(f"| {s['n_freqs']} | {s['r2']:.6f} | {s['rmse']:.6f} | {s['accuracy']:.6f} | {freqs_str} |")
    lines.append("")

    # Find minimum |K| for 99% accuracy
    min_k_99 = None
    min_k_999 = None
    for s in freq_sweep:
        if s["accuracy"] >= 0.99 and min_k_99 is None:
            min_k_99 = s["n_freqs"]
        if s["accuracy"] >= 0.999 and min_k_999 is None:
            min_k_999 = s["n_freqs"]

    lines.append(f"**Minimum |K| for 99% accuracy**: {min_k_99 if min_k_99 else 'not reached'}")
    lines.append(f"**Minimum |K| for 99.9% accuracy**: {min_k_999 if min_k_999 else 'not reached'}")
    lines.append("")

    # Attention head analysis
    lines.append(f"## Attention Head Frequency Specialization\n")
    lines.append(f"| Layer | Head | Dominant Freq | Fraction | Top-3 Frequencies |")
    lines.append(f"|------:|-----:|--------------:|---------:|-------------------|")
    for h in head_analysis:
        top3_str = ", ".join(f"k={k}({frac:.3f})" for k, frac in h["top3_freqs"])
        lines.append(f"| {h['layer']} | {h['head']} | {h['dominant_freq']} | {h['freq_fraction']:.4f} | {top3_str} |")
    lines.append("")

    # Summary
    lines.append(f"## Summary\n")
    lines.append(f"The grokked model uses **{min_k_99 or '?'} key frequencies** out of {q} total ")
    lines.append(f"to achieve 99% argmax accuracy on the DLP at p={p}.\n")

    if min_k_99:
        ratio = min_k_99 / q
        lines.append(f"Ratio |K|/q = {min_k_99}/{q} = {ratio:.4f}")
        if ratio < 0.1:
            lines.append(f"\nThis is a **highly sparse** representation — the model discovers ")
            lines.append(f"a small set of key frequencies, consistent with the Fourier circuit ")
            lines.append(f"theory of grokking.\n")

        import math as m
        sqrt_q = m.sqrt(q)
        log_q = m.log2(q)
        lines.append(f"\nFor comparison:")
        lines.append(f"- sqrt(q) = {sqrt_q:.2f} (baby-step giant-step complexity)")
        lines.append(f"- log2(q) = {log_q:.2f}")
        lines.append(f"- |K| = {min_k_99}")
        if min_k_99 < sqrt_q:
            lines.append(f"\n**|K| < sqrt(q)** — the extracted algorithm is sparser than BSGS!")
        elif min_k_99 < log_q:
            lines.append(f"\n**|K| < log2(q)** — the extracted algorithm is extremely efficient!")

    report_path = os.path.join(output_dir, "extraction_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Extract DLP algorithm from grokked checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: alongside checkpoint)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top frequencies to report")
    parser.add_argument("--max-sweep-k", type=int, default=30,
                        help="Max key frequencies to sweep in closed-form fit")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    p = config["p"]
    q = p - 1
    g = _primitive_root(p)
    print(f"Model loaded: d_model={config['d_model']}, n_heads={config['n_heads']}, "
          f"n_layers={config['n_layers']}, p={p}, q={q}, g={g}")

    # Output directory
    output_dir = args.output_dir or os.path.dirname(os.path.dirname(args.checkpoint))
    os.makedirs(output_dir, exist_ok=True)

    int_offset = 4  # tokenizer: [PAD=0, SEP=1, BOS=2, EOS=3, 0, 1, ..., p-1]

    # Step 1: Extract W_E and W_U
    print("\n--- Step 1: Extract weight matrices ---")
    W_E = model.embed.weight.detach().cpu().numpy()  # (vocab_size, d_model)
    W_U = model.unembed.weight.detach().cpu().numpy()  # (vocab_size, d_model)
    print(f"W_E shape: {W_E.shape}")
    print(f"W_U shape: {W_U.shape}")

    # Step 2: Compute multiplicative character transforms
    print("\n--- Step 2: Compute multiplicative character transforms ---")
    W_E_hat = compute_multiplicative_character_transform(W_E, p, g, int_offset)
    W_U_hat = compute_multiplicative_character_transform(W_U, p, g, int_offset)
    print(f"W_E_hat shape: {W_E_hat.shape} (q × d_model, complex)")
    print(f"W_U_hat shape: {W_U_hat.shape} (q × d_model, complex)")

    # Step 3: Identify key frequencies
    print("\n--- Step 3: Identify key frequencies ---")
    embed_key_freqs, embed_energy = identify_key_frequencies(W_E_hat, top_k=args.top_k)
    unembed_key_freqs, unembed_energy = identify_key_frequencies(W_U_hat, top_k=args.top_k)

    print("\nEmbedding key frequencies (top-10):")
    for i, f in enumerate(embed_key_freqs[:10]):
        print(f"  k={f['freq']:4d}  energy={f['energy']:.4f}  fraction={f['fraction']:.4f}")

    print("\nUnembedding key frequencies (top-10):")
    for i, f in enumerate(unembed_key_freqs[:10]):
        print(f"  k={f['freq']:4d}  energy={f['energy']:.4f}  fraction={f['fraction']:.4f}")

    # Step 4: Fit closed-form model
    print("\n--- Step 4: Fit closed-form trigonometric model ---")
    # Use top frequencies from embedding
    top_freqs_for_fit = [f["freq"] for f in embed_key_freqs[:args.top_k]]

    closed_form_fit = fit_closed_form_model(
        model, p, g, top_freqs_for_fit, device
    )
    print(f"\nClosed-form model with {len(top_freqs_for_fit)} frequencies:")
    print(f"  R² = {closed_form_fit['r2']:.6f}")
    print(f"  RMSE = {closed_form_fit['rmse']:.6f}")
    print(f"  Argmax accuracy = {closed_form_fit['argmax_accuracy']:.6f}")

    print("\nTop coefficients:")
    for c in closed_form_fit["coefficients"][:5]:
        print(f"  k={c['freq']:4d}  α={c['alpha']:+.6f}  β={c['beta']:+.6f}  contrib={c['contribution']:.6f}")

    # Step 5: Sweep key frequency count
    print("\n--- Step 5: Sweep key frequency count ---")
    freq_sweep = sweep_key_frequency_count(
        model, p, g, embed_energy, device, max_k=args.max_sweep_k
    )
    print(f"\n{'|K|':>5} {'R²':>10} {'RMSE':>10} {'Accuracy':>10}")
    print("-" * 40)
    for s in freq_sweep:
        print(f"{s['n_freqs']:5d} {s['r2']:10.6f} {s['rmse']:10.6f} {s['accuracy']:10.6f}")

    # Step 6: Attention head analysis
    print("\n--- Step 6: Attention head frequency specialization ---")
    head_analysis = extract_attention_head_frequencies(model, p, g, device)
    print(f"\n{'Layer':>5} {'Head':>5} {'Dom Freq':>8} {'Fraction':>10} {'Top-3':>30}")
    print("-" * 65)
    for h in head_analysis:
        top3 = ", ".join(f"k={k}({frac:.3f})" for k, frac in h["top3_freqs"])
        print(f"{h['layer']:5d} {h['head']:5d} {h['dominant_freq']:8d} {h['freq_fraction']:10.4f} {top3:>30}")

    # Generate report
    print("\n--- Generating report ---")
    report_path = generate_extraction_report(
        config, embed_key_freqs, embed_energy,
        unembed_key_freqs, unembed_energy,
        closed_form_fit, freq_sweep, head_analysis,
        output_dir,
    )
    print(f"Report saved: {report_path}")

    # Save raw results
    results = {
        "config": config,
        "embed_key_freqs": embed_key_freqs,
        "unembed_key_freqs": unembed_key_freqs,
        "closed_form_fit": closed_form_fit,
        "freq_sweep": freq_sweep,
        "head_analysis": head_analysis,
    }
    results_path = os.path.join(output_dir, "extraction_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved: {results_path}")

    print("\n=== Extraction complete ===")


if __name__ == "__main__":
    main()
