"""Fourier analysis of DLP grokking checkpoints.

Extends the representation monitoring with detailed Fourier spectral analysis:

1. Full DFT spectrum: project hidden states onto multiplicative and additive character bases
2. Per-frequency energy distribution (all modes, not just top-k entropy)
3. Dominant frequency identification with energy concentration
4. Phase coherence: alignment of hidden state phases with character frequencies
5. Spectrum transition metric: KL divergence between consecutive checkpoint spectra
6. Sparsity: effective number of Fourier modes (participation ratio)
7. Cross-spectrum: correlation between multiplicative and additive projections

Usage:
    from src.fourier_analysis import FourierAnalyzer
    analyzer = FourierAnalyzer(p=113, device=device)
    metrics = analyzer.analyze_checkpoint(model, epoch)
"""
from __future__ import annotations

import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def _primitive_root(p: int) -> int:
    """Find a primitive root modulo p."""
    if p == 2:
        return 1
    n = p - 1
    factors = set()
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            factors.add(d)
            temp //= d
        d += 1
    if temp > 1:
        factors.add(temp)
    for g in range(2, p):
        if all(pow(g, n // q, p) != 1 for q in factors):
            return g
    return 1


class FourierAnalyzer:
    """Detailed Fourier spectral analysis for DLP grokking checkpoints.

    Computes per-frequency energy distributions over both multiplicative
    (dlog-based) and additive (value-based) character bases, identifying
    which Fourier modes the model relies on at each training stage.
    """

    def __init__(self, p: int, device: torch.device,
                 output_dir: Optional[str] = None,
                 n_probe_layers: int = 8):
        self.p = p
        self.device = device
        self.output_dir = output_dir
        self.q = p - 1  # multiplicative group order
        self.g = _primitive_root(p)

        # Build probe dataset: full F_p* group
        self.probe_tokens = []
        self.probe_exponents = []
        for x in range(self.q):
            h = pow(self.g, x, p)
            self.probe_tokens.append(h)
            self.probe_exponents.append(x)

        # Precompute character basis matrices
        self._precompute_bases()

        # Track previous spectrum for transition metrics
        self._prev_spectrum: Optional[Dict[str, np.ndarray]] = None

        # Hooks
        self._hooks = []
        self._captured: Dict[str, torch.Tensor] = {}

    def _precompute_bases(self):
        """Precompute multiplicative and additive character basis matrices."""
        q = self.q
        p = self.p
        n = q

        # Multiplicative characters: chi_k(x) = exp(2*pi*i*k*x/q)
        # Shape: (n, q) complex — one column per frequency k
        k_vals = np.arange(q)
        x_vals = np.array(self.probe_exponents, dtype=np.float64)
        self.mult_basis = np.exp(2j * np.pi * np.outer(x_vals, k_vals) / q).astype(np.complex64)

        # Additive characters: psi_k(h) = exp(2*pi*i*k*h/p)
        # Shape: (n, p) complex
        h_vals = np.array(self.probe_tokens, dtype=np.float64)
        k_add = np.arange(p)
        self.add_basis = np.exp(2j * np.pi * np.outer(h_vals, k_add) / p).astype(np.complex64)

        # Normalization factors (Parseval)
        self.mult_norm = np.sqrt(n)
        self.add_norm = np.sqrt(n)

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on model layers."""
        if hasattr(model, 'get_hook_layers'):
            hook_layers = model.get_hook_layers()
        else:
            hook_layers = [("embed", model.embed), ("unembed", model.unembed)]

        for name, module in hook_layers:
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            self._captured[name] = output.detach()
        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _get_hidden_states(self, model: nn.Module) -> Dict[str, np.ndarray]:
        """Run probe dataset through model and extract hidden states at hooked layers."""
        self._captured = {}

        int_offset = 4
        bos_id = 2
        eos_id = 3

        tokens = []
        for h in self.probe_tokens:
            tokens.append([bos_id, int_offset + (h % self.p), eos_id])
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)

        with torch.no_grad():
            model(token_tensor)

        hidden = {}
        for name, output in self._captured.items():
            if output.dim() == 3:
                hidden[name] = output[:, -1, :].cpu().numpy()
            elif output.dim() == 2:
                hidden[name] = output.cpu().numpy()
            else:
                hidden[name] = output.cpu().numpy()

        return hidden

    def _compute_spectrum(self, H: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """Compute per-frequency energy spectrum.

        Args:
            H: (n, d) hidden states
            basis: (n, K) complex character basis

        Returns:
            energy: (K,) per-frequency energy (summed over d dimensions)
        """
        H_complex = H.astype(np.complex64)
        projections = H_complex.T @ basis  # (d, K) complex
        energy = np.sum(np.abs(projections) ** 2, axis=0)  # (K,) real
        return energy

    def _participation_ratio(self, energy: np.ndarray) -> float:
        """Effective number of active modes (participation ratio).

        PR = (sum(e_k))^2 / sum(e_k^2)
        """
        total = energy.sum()
        if total < 1e-12:
            return 0.0
        return float(total ** 2 / np.sum(energy ** 2))

    def _dominant_modes(self, energy: np.ndarray, k: int = 10) -> List[Dict]:
        """Identify top-k dominant frequency modes."""
        total = energy.sum()
        if total < 1e-12:
            return []

        sorted_idx = np.argsort(energy)[::-1][:k]
        modes = []
        for idx in sorted_idx:
            modes.append({
                "freq": int(idx),
                "energy": float(energy[idx]),
                "fraction": float(energy[idx] / total),
            })
        return modes

    def _concentration_ratio(self, energy: np.ndarray, k: int = 4) -> float:
        """Fraction of total energy in top-k modes."""
        total = energy.sum()
        if total < 1e-12:
            return 0.0
        sorted_energy = np.sort(energy)[::-1]
        return float(sorted_energy[:k].sum() / total)

    def _spectral_kl_divergence(self, current: np.ndarray, previous: np.ndarray) -> float:
        """KL divergence between two normalized spectra."""
        p = current / max(current.sum(), 1e-12)
        q_prev = previous / max(previous.sum(), 1e-12)
        # Smooth to avoid zeros
        eps = 1e-10
        p = p + eps
        q_prev = q_prev + eps
        p = p / p.sum()
        q_prev = q_prev / q_prev.sum()
        return float(np.sum(p * np.log(p / q_prev)))

    def _cross_spectrum_correlation(self, mult_energy: np.ndarray,
                                     add_energy: np.ndarray) -> float:
        """Correlation between multiplicative and additive energy distributions.

        High correlation suggests the model uses both representations similarly;
        low correlation suggests it has settled on one basis.
        """
        if len(mult_energy) != len(add_energy):
            min_len = min(len(mult_energy), len(add_energy))
            mult_energy = mult_energy[:min_len]
            add_energy = add_energy[:min_len]

        m = mult_energy / max(mult_energy.sum(), 1e-12)
        a = add_energy / max(add_energy.sum(), 1e-12)

        if np.std(m) < 1e-12 or np.std(a) < 1e-12:
            return 0.0
        return float(np.corrcoef(m, a)[0, 1])

    def _phase_coherence(self, H: np.ndarray, basis: np.ndarray) -> float:
        """Measure phase coherence: how aligned are hidden state projections.

        For each frequency k, compute the phase of the projection across
        hidden dimensions. High coherence means all dimensions agree on phase.
        """
        H_complex = H.astype(np.complex64)
        projections = H_complex.T @ basis  # (d, K) complex

        d, K = projections.shape
        if d < 2:
            return 0.0

        # Mean resultant length per frequency (circular statistics)
        coherence_per_k = np.zeros(K)
        for k in range(K):
            phases = np.angle(projections[:, k])
            mags = np.abs(projections[:, k])
            total_mag = mags.sum()
            if total_mag < 1e-12:
                continue
            weighted_mean_vector = np.sum(mags * np.exp(1j * phases)) / total_mag
            coherence_per_k[k] = np.abs(weighted_mean_vector)

        # Return mean coherence weighted by energy
        energy = np.sum(np.abs(projections) ** 2, axis=0)
        total_energy = energy.sum()
        if total_energy < 1e-12:
            return 0.0
        return float(np.sum(coherence_per_k * energy) / total_energy)

    def analyze_checkpoint(self, model: nn.Module, epoch: int) -> Dict:
        """Compute full Fourier analysis for a single checkpoint.

        Returns dict with per-layer spectral metrics.
        """
        hidden = self._get_hidden_states(model)

        result = {"epoch": epoch, "layers": {}}

        for name, H in hidden.items():
            layer_metrics = {}

            # Multiplicative spectrum
            mult_energy = self._compute_spectrum(H, self.mult_basis)
            layer_metrics["mult_spectrum"] = mult_energy.tolist()
            layer_metrics["mult_total_energy"] = float(mult_energy.sum())
            layer_metrics["mult_pr"] = self._participation_ratio(mult_energy)
            layer_metrics["mult_concentration_4"] = self._concentration_ratio(mult_energy, 4)
            layer_metrics["mult_concentration_8"] = self._concentration_ratio(mult_energy, 8)
            layer_metrics["mult_dominant_modes"] = self._dominant_modes(mult_energy, k=10)
            layer_metrics["mult_phase_coherence"] = self._phase_coherence(H, self.mult_basis)

            # Additive spectrum
            add_energy = self._compute_spectrum(H, self.add_basis)
            layer_metrics["add_spectrum"] = add_energy.tolist()
            layer_metrics["add_total_energy"] = float(add_energy.sum())
            layer_metrics["add_pr"] = self._participation_ratio(add_energy)
            layer_metrics["add_concentration_4"] = self._concentration_ratio(add_energy, 4)
            layer_metrics["add_concentration_8"] = self._concentration_ratio(add_energy, 8)
            layer_metrics["add_dominant_modes"] = self._dominant_modes(add_energy, k=10)
            layer_metrics["add_phase_coherence"] = self._phase_coherence(H, self.add_basis)

            # Cross-spectrum
            layer_metrics["cross_corr"] = self._cross_spectrum_correlation(
                mult_energy, add_energy
            )

            # Basis preference: mult vs add energy ratio
            total_mult = max(mult_energy.sum(), 1e-12)
            total_add = max(add_energy.sum(), 1e-12)
            layer_metrics["basis_preference"] = float(total_mult / total_add)

            # Transition metrics (vs previous checkpoint)
            if self._prev_spectrum is not None:
                key_mult = f"{name}_mult"
                key_add = f"{name}_add"
                if key_mult in self._prev_spectrum:
                    layer_metrics["mult_kl_div"] = self._spectral_kl_divergence(
                        mult_energy, self._prev_spectrum[key_mult]
                    )
                else:
                    layer_metrics["mult_kl_div"] = 0.0
                if key_add in self._prev_spectrum:
                    layer_metrics["add_kl_div"] = self._spectral_kl_divergence(
                        add_energy, self._prev_spectrum[key_add]
                    )
                else:
                    layer_metrics["add_kl_div"] = 0.0
            else:
                layer_metrics["mult_kl_div"] = 0.0
                layer_metrics["add_kl_div"] = 0.0

            result["layers"][name] = layer_metrics

        # Store spectra for next checkpoint
        self._prev_spectrum = {}
        for name, H in hidden.items():
            self._prev_spectrum[f"{name}_mult"] = self._compute_spectrum(H, self.mult_basis)
            self._prev_spectrum[f"{name}_add"] = self._compute_spectrum(H, self.add_basis)

        return result

    def save_results(self, metrics: Dict, path: str):
        """Append metrics to JSONL file."""
        with open(path, "a") as f:
            f.write(json.dumps(metrics) + "\n")


def generate_fourier_report(results: List[Dict], config: Dict, output_dir: str) -> str:
    """Generate a Markdown report from Fourier analysis results.

    Args:
        results: List of per-checkpoint Fourier analysis dicts
        config: Model config dict
        output_dir: Directory to save the report

    Returns:
        Path to the saved report
    """
    p = config["prime"]
    lines = []
    lines.append(f"# Fourier Analysis Report: {config['model_id']} (p={p})\n")
    lines.append(f"- **Model**: {config['model_id']}, d_model={config['d_model']}, "
                 f"n_heads={config['n_heads']}, n_layers={config['n_layers']}")
    lines.append(f"- **Params**: {config['P_total']:,} (P_core={config['P_core']:,})")
    lines.append(f"- **Prime**: {p}, Train frac: {config['train_frac']}, "
                 f"Seed: {config['seed']}")
    lines.append(f"- **Checkpoints analyzed**: {len(results)}\n")

    # Focus on block_1 ( deepest transformer block for L=2)
    target_layer = "block_1"
    layer_results = [r for r in results
                     if target_layer in r.get("layers", {})]

    if not layer_results:
        target_layer = "unembed"
        layer_results = [r for r in results
                         if target_layer in r.get("layers", {})]

    if not layer_results:
        lines.append("No layer data found.\n")
        report_path = os.path.join(output_dir, "fourier_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        return report_path

    lines.append(f"## Spectral Evolution (layer: {target_layer})\n")
    lines.append("| Epoch | Mult PR | Add PR | Mult Conc4 | Add Conc4 | "
                 "Basis Pref | Cross Corr | Mult KL | Add KL |")
    lines.append("|------:|--------:|-------:|-----------:|-----------:|"
                 "-----------:|-----------:|---------:|-------:|")
    for r in layer_results:
        lm = r["layers"][target_layer]
        lines.append(
            f"| {r['epoch']} | {lm['mult_pr']:.2f} | {lm['add_pr']:.2f} "
            f"| {lm['mult_concentration_4']:.4f} | {lm['add_concentration_4']:.4f} "
            f"| {lm['basis_preference']:.4f} | {lm['cross_corr']:.4f} "
            f"| {lm['mult_kl_div']:.6f} | {lm['add_kl_div']:.6f} |"
        )

    # Dominant modes at key epochs
    lines.append(f"\n## Dominant Multiplicative Modes ({target_layer})\n")
    lines.append("Show top-4 modes at selected epochs.\n")

    # Pick a few representative epochs: early, mid, late
    n = len(layer_results)
    sample_indices = sorted(set([
        0, n // 4, n // 2, 3 * n // 4, n - 1
    ]))
    sample_indices = [i for i in sample_indices if 0 <= i < n]

    for idx in sample_indices:
        r = layer_results[idx]
        lm = r["layers"][target_layer]
        modes = lm.get("mult_dominant_modes", [])[:4]
        mode_str = ", ".join(
            f"k={m['freq']} ({m['fraction']:.3f})" for m in modes
        )
        lines.append(f"- **Epoch {r['epoch']}**: {mode_str}")

    # Phase coherence evolution
    lines.append(f"\n## Phase Coherence Evolution ({target_layer})\n")
    lines.append("| Epoch | Mult Coherence | Add Coherence |")
    lines.append("|------:|---------------:|--------------:|")
    for r in layer_results:
        lm = r["layers"][target_layer]
        lines.append(
            f"| {r['epoch']} | {lm['mult_phase_coherence']:.4f} "
            f"| {lm['add_phase_coherence']:.4f} |"
        )

    # All layers summary at final epoch
    if layer_results:
        final = layer_results[-1]
        lines.append(f"\n## All-Layer Summary (epoch {final['epoch']})\n")
        lines.append("| Layer | Mult PR | Add PR | Conc4 Mult | Conc4 Add | "
                     "Basis Pref | Cross Corr |")
        lines.append("|-------|--------:|-------:|-----------:|-----------:|"
                     "-----------:|-----------:|")
        for name, lm in final["layers"].items():
            lines.append(
                f"| {name} | {lm['mult_pr']:.2f} | {lm['add_pr']:.2f} "
                f"| {lm['mult_concentration_4']:.4f} | {lm['add_concentration_4']:.4f} "
                f"| {lm['basis_preference']:.4f} | {lm['cross_corr']:.4f} |"
            )

    report_path = os.path.join(output_dir, "fourier_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path
