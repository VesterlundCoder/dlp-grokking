"""Representation monitoring for DLP grokking scaling study.

Metrics computed per checkpoint:
1. Effective rank (entropic) of hidden state covariance
2. Multiplicative character spectrum energy
3. Additive Fourier spectrum energy
4. Basis entropy gap (S_add - S_mult)
5. Top-k character energy (top 4/8/16)
6. Character probe R^2 (linear probe for cos/sin of 2*pi*k*x/q)
7. CKA vs previous checkpoint (representation drift)
"""
from __future__ import annotations

import json
import math
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


def _primitive_root(p: int) -> int:
    """Find a primitive root modulo p."""
    if p == 2:
        return 1
    # Factor p-1
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


class DLPRepresentationMonitor:
    """Monitor internal representations during DLP grokking training.

    Builds a probe dataset from the full F_p* group and computes
    spectral/structural metrics on hidden states.
    """

    def __init__(self, p: int, device: torch.device, output_dir: str,
                 rep_interval: int = 500, max_snapshots: int = 30):
        self.p = p
        self.device = device
        self.output_dir = output_dir
        self.rep_interval = rep_interval
        self.max_snapshots = max_snapshots
        self.metrics_path = os.path.join(output_dir, "representation_metrics.jsonl")
        self.snapshot_dir = os.path.join(output_dir, "representation_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Build probe dataset: full F_p* group
        # g = primitive root, for each x in [0, p-2]: h = g^x mod p
        self.g = _primitive_root(p)
        self.q = p - 1  # group order
        self.probe_tokens = []
        self.probe_exponents = []
        for x in range(self.q):
            h = pow(self.g, x, p)
            self.probe_tokens.append(h)
            self.probe_exponents.append(x)

        # Precompute character basis functions
        # Multiplicative: chi_k(h) = exp(2*pi*i*k*x/q) where x = dlog_g(h)
        # Additive: psi_k(h) = exp(2*pi*i*k*h/p)
        self._precompute_bases()

        # Previous hidden states for CKA
        self._prev_hidden: Optional[Dict[str, np.ndarray]] = None
        self._snapshot_count = 0

        # Hooks
        self._hooks = []
        self._captured: Dict[str, torch.Tensor] = {}

    def _precompute_bases(self):
        """Precompute multiplicative and additive character basis matrices."""
        q = self.q
        p = self.p
        n = q  # number of probe examples

        # Multiplicative characters: chi_k(x) = exp(2*pi*i*k*x/q)
        # Shape: (n, q) complex
        self.mult_basis = np.zeros((n, q), dtype=np.complex64)
        for i, x in enumerate(self.probe_exponents):
            for k in range(q):
                self.mult_basis[i, k] = np.exp(2j * np.pi * k * x / q)

        # Additive characters: psi_k(h) = exp(2*pi*i*k*h/p)
        # Shape: (n, p) complex
        self.add_basis = np.zeros((n, p), dtype=np.complex64)
        for i, h in enumerate(self.probe_tokens):
            for k in range(p):
                self.add_basis[i, k] = np.exp(2j * np.pi * k * h / p)

        # Real-valued probes for R^2: cos(2*pi*k*x/q), sin(2*pi*k*x/q)
        # For a subset of k values
        k_probe = list(range(min(8, q)))
        self.cos_probes = np.zeros((n, len(k_probe)), dtype=np.float32)
        self.sin_probes = np.zeros((n, len(k_probe)), dtype=np.float32)
        for i, x in enumerate(self.probe_exponents):
            for j, k in enumerate(k_probe):
                self.cos_probes[i, j] = np.cos(2 * np.pi * k * x / q)
                self.sin_probes[i, j] = np.sin(2 * np.pi * k * x / q)
        self.k_probe = k_probe

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on specified layers."""
        # Get hook layers from model
        if hasattr(model, 'get_hook_layers'):
            hook_layers = model.get_hook_layers()
        elif hasattr(model, 'module') and hasattr(model.module, 'get_hook_layers'):
            hook_layers = model.module.get_hook_layers()
        else:
            # Fallback: hook embed and unembed
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

        # Build input tokens: [BOS, h, EOS] for L6
        # Match the tokenizer format from lumi_cyclic_dlp_trainer.py
        # vocab: [PAD=0, SEP=1, BOS=2, EOS=3, int_offset=4, ...]
        int_offset = 4
        bos_id = 2
        eos_id = 3

        tokens = []
        for h in self.probe_tokens:
            tokens.append([bos_id, int_offset + (h % self.p), eos_id])
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)

        with torch.no_grad():
            model(token_tensor)

        # Extract hidden states at the last position (EOS) for each layer
        hidden = {}
        for name, output in self._captured.items():
            if output.dim() == 3:  # (batch, seq, d_model)
                # Take the last non-pad position
                hidden[name] = output[:, -1, :].cpu().numpy()
            elif output.dim() == 2:
                hidden[name] = output.cpu().numpy()
            else:
                hidden[name] = output.cpu().numpy()

        return hidden

    def _effective_rank(self, H: np.ndarray) -> float:
        """Entropic effective rank: exp(-sum(p_i * log(p_i)) of normalized eigenvalues."""
        if H.shape[0] < 2:
            return 0.0
        # Covariance
        H_centered = H - H.mean(axis=0, keepdims=True)
        cov = H_centered.T @ H_centered / max(H.shape[0] - 1, 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)  # numerical stability
        total = eigvals.sum()
        if total < 1e-12:
            return 0.0
        probs = eigvals / total
        probs = probs[probs > 1e-12]
        entropy = -np.sum(probs * np.log(probs))
        return float(np.exp(entropy))

    def _spectral_entropy(self, H: np.ndarray, basis: np.ndarray, name: str) -> Dict:
        """Compute spectral energy distribution over a character basis.

        Args:
            H: (n, d) hidden states
            basis: (n, K) complex character basis
            name: 'mult' or 'add'

        Returns:
            dict with entropy, top-k energy fractions
        """
        n, d = H.shape
        K = basis.shape[1]

        # Project: for each character k, compute energy = |<H_j, basis_k>|^2 summed over dims
        # basis_k is (n,) complex, H_j is (n,) real
        # projection_jk = sum_i H[i,j] * conj(basis[i,k])
        # energy_k = sum_j |projection_jk|^2
        H_complex = H.astype(np.complex64)
        projections = H_complex.T @ basis  # (d, K) complex
        energy = np.sum(np.abs(projections) ** 2, axis=0)  # (K,) real

        total_energy = energy.sum()
        if total_energy < 1e-12:
            return {
                f"S_{name}": 0.0,
                f"top4_{name}": 0.0,
                f"top8_{name}": 0.0,
                f"top16_{name}": 0.0,
            }

        energy_norm = energy / total_energy
        energy_sorted = np.sort(energy_norm)[::-1]

        # Entropy
        p = energy_norm[energy_norm > 1e-12]
        entropy = -np.sum(p * np.log(p))

        # Top-k energy fractions
        top4 = float(energy_sorted[:4].sum())
        top8 = float(energy_sorted[:8].sum())
        top16 = float(energy_sorted[:min(16, len(energy_sorted))].sum())

        return {
            f"S_{name}": float(entropy),
            f"top4_{name}": top4,
            f"top8_{name}": top8,
            f"top16_{name}": top16,
        }

    def _character_probe_r2(self, H: np.ndarray) -> Dict:
        """Linear probe R^2: can hidden state linearly predict cos/sin(2*pi*k*x/q)?

        Simple least squares: target = H @ beta, R^2 = 1 - SS_res/SS_tot
        """
        n, d = H.shape
        n_targets = self.cos_probes.shape[1]

        # Add bias term
        H_aug = np.hstack([H, np.ones((n, 1), dtype=np.float32)])

        results = {}
        cos_r2 = []
        sin_r2 = []

        for j in range(n_targets):
            # cos probe
            target = self.cos_probes[:, j]
            try:
                beta, _, _, _ = np.linalg.lstsq(H_aug, target, rcond=None)
                pred = H_aug @ beta
                ss_res = np.sum((target - pred) ** 2)
                ss_tot = np.sum((target - target.mean()) ** 2)
                r2 = 1 - ss_res / max(ss_tot, 1e-12)
                cos_r2.append(max(0.0, r2))
            except Exception:
                cos_r2.append(0.0)

            # sin probe
            target = self.sin_probes[:, j]
            try:
                beta, _, _, _ = np.linalg.lstsq(H_aug, target, rcond=None)
                pred = H_aug @ beta
                ss_res = np.sum((target - pred) ** 2)
                ss_tot = np.sum((target - target.mean()) ** 2)
                r2 = 1 - ss_res / max(ss_tot, 1e-12)
                sin_r2.append(max(0.0, r2))
            except Exception:
                sin_r2.append(0.0)

        results["cos_probe_r2_mean"] = float(np.mean(cos_r2))
        results["sin_probe_r2_mean"] = float(np.mean(sin_r2))
        results["cos_probe_r2_max"] = float(np.max(cos_r2))
        results["sin_probe_r2_max"] = float(np.max(sin_r2))
        return results

    def _cka(self, H1: np.ndarray, H2: np.ndarray) -> float:
        """Centered Kernel Alignment between two hidden state matrices."""
        if H1.shape != H2.shape or H1.shape[0] < 2:
            return 0.0

        n = H1.shape[0]
        H1c = H1 - H1.mean(axis=0, keepdims=True)
        H2c = H2 - H2.mean(axis=0, keepdims=True)

        K1 = H1c @ H1c.T
        K2 = H2c @ H2c.T

        # HSIC
        hsic = np.sum(K1 * K2) / (n - 1) ** 2
        hsic1 = np.sum(K1 * K1) / (n - 1) ** 2
        hsic2 = np.sum(K2 * K2) / (n - 1) ** 2

        denom = math.sqrt(max(hsic1 * hsic2, 1e-12))
        return float(hsic / denom) if denom > 1e-12 else 0.0

    def compute_metrics(self, model: nn.Module, epoch: int) -> Dict:
        """Compute all representation metrics at the current epoch."""
        hidden = self._get_hidden_states(model)

        all_metrics = {"epoch": epoch, "layers": {}}

        for name, H in hidden.items():
            layer_metrics = {}

            # 1. Effective rank
            layer_metrics["eff_rank"] = self._effective_rank(H)

            # 2-3. Spectral entropy (multiplicative + additive)
            layer_metrics.update(self._spectral_entropy(H, self.mult_basis, "mult"))
            layer_metrics.update(self._spectral_entropy(H, self.add_basis, "add"))

            # 4. Basis entropy gap
            s_mult = layer_metrics.get("S_mult", 0.0)
            s_add = layer_metrics.get("S_add", 0.0)
            layer_metrics["basis_entropy_gap"] = s_add - s_mult

            # 5. Top-k character energy (already in spectral entropy)

            # 6. Character probe R^2
            layer_metrics.update(self._character_probe_r2(H))

            # 7. CKA vs previous checkpoint
            if self._prev_hidden is not None and name in self._prev_hidden:
                layer_metrics["cka_vs_prev"] = self._cka(H, self._prev_hidden[name])
            else:
                layer_metrics["cka_vs_prev"] = 1.0

            all_metrics["layers"][name] = layer_metrics

        # Store for next CKA
        self._prev_hidden = hidden

        return all_metrics

    def save_metrics(self, metrics: Dict):
        """Append metrics to representation_metrics.jsonl."""
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def checkpoint(self, model: nn.Module, epoch: int):
        """Full snapshot at key epochs."""
        if self._snapshot_count >= self.max_snapshots:
            return
        hidden = self._get_hidden_states(model)
        path = os.path.join(self.snapshot_dir, f"snapshot_{epoch:06d}.npz")
        np.savez(path, **hidden)
        self._snapshot_count += 1

    def should_checkpoint(self, epoch: int, test_acc: float, prev_test_acc: float) -> bool:
        """Determine if we should do a full snapshot this epoch.

        Dense around T_mem and T_grok, sparse otherwise.
        """
        # Always snapshot at multiples of rep_interval
        if epoch > 0 and epoch % self.rep_interval == 0:
            return True
        # Snapshot when test_acc starts moving (adaptive)
        if prev_test_acc is not None and test_acc is not None:
            if test_acc - prev_test_acc > 0.01:
                return True
        return False
