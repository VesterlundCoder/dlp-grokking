"""Grokking Transformer model for DLP scaling study.

Adapted from lumi_cyclic_dlp_trainer.py with:
- core_parameters() method (P_core = transformer layers only)
- get_hook_layers() for representation monitoring
- Parameter counting helpers
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple


class GrokkingTransformer(nn.Module):
    """Transformer for cyclic DLP grokking.

    Architecture: token embedding + positional embedding → N transformer encoder layers
    → unembedding to vocab logits.
    """

    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 16,
                 n_layers: int = 8, max_len: int = 64, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.unembed = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.size()
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.dropout(self.embed(tokens) + self.pos_embed(pos))
        h = self.transformer(h)
        logits = self.unembed(h)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def core_parameters(self) -> int:
        """P_core: transformer layers only (excludes embedding + unembedding)."""
        return sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        """P_total: all trainable parameters."""
        return self.count_parameters()

    def get_hook_layers(self) -> List[Tuple[str, nn.Module]]:
        """Return list of (name, module) pairs for representation monitoring hooks.

        For L=2: embed, transformer.layers[0], transformer.layers[1], unembed.
        """
        layers = [("embed", self.embed)]
        n = self.n_layers
        if n == 1:
            layers.append(("block_0", self.transformer.layers[0]))
        elif n == 2:
            layers.append(("block_0", self.transformer.layers[0]))
            layers.append(("block_1", self.transformer.layers[1]))
        else:
            indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
            seen = set()
            for i in indices:
                if i not in seen:
                    seen.add(i)
                    layers.append((f"block_{i}", self.transformer.layers[i]))
        layers.append(("unembed", self.unembed))
        return layers


def estimate_params(vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int) -> int:
    """Estimate total parameter count."""
    embed = vocab_size * d_model + max_len * d_model
    per_layer = 12 * d_model * d_model
    unembed = d_model * vocab_size
    return embed + n_layers * per_layer + unembed


def estimate_core_params(d_model: int, n_layers: int) -> int:
    """Estimate P_core (transformer layers only)."""
    per_layer = 12 * d_model * d_model
    return n_layers * per_layer


def compute_wd_max(p_core: int, anchor_p_core: int = 393_000,
                   min_wd: float = 0.15, max_wd: float = 0.30) -> float:
    """Auto-scale WD max based on model size.

    wd_max = min(0.30, 0.30 * sqrt(anchor_p_core / p_core)), clamped to [0.15, 0.30]
    """
    import math
    wd = min(max_wd, max_wd * math.sqrt(anchor_p_core / max(p_core, 1)))
    return max(min_wd, wd)
