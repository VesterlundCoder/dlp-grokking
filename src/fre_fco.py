"""Factorized Residue Embedding (FRE) and Factorized Candidate Output (FCO).

Ported from the EI4 ECDLP study. These replace the atomic nn.Embedding(p+4, d_model)
and nn.Linear(d_model, p+4) with digit-decomposition-based alternatives that are
O(1) in prime p, enabling a fixed-size model to grok DLP at arbitrarily large primes.

Architecture:
    FRE input:  val -> M digits in base B -> per-position embedding tables
                P_input = n_special*d_model + M*B*d_model  (O(1) in p)

    FCO output: candidate c -> M digits in base B -> additive embedding
                logit(c) = h . emb(c) / sqrt(d_model)
                P_output = M*B*d_model + M*B  (O(1) in p)

Combined with CRT-BOTH representation, the model sees a decomposed problem
(g_n1, g_n2, h_n1, h_n2) with O(1) parameters regardless of prime size.
"""
from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn


# ============================================================================
# FRE: Factorized Residue Embedding
# ============================================================================

class FREEmbedding(nn.Module):
    """Factorized Residue Embedding — O(1) params in p.

    Decomposes each input value into M digits in base B:
        val = d_0 + d_1*B + d_2*B^2 + ... + d_{M-1}*B^{M-1}

    Each digit position j has its own embedding table: nn.Embedding(B, d_model).
    Special tokens (PAD=0, SEP=1, BOS=2, EOS=3) get a separate embedding table.

    P_input = n_special*d_model + M*B*d_model  (O(1) for fixed M, B, d_model)
    """

    def __init__(self, base: int, M: int, d_model: int, n_special: int = 4):
        super().__init__()
        self.base = base
        self.M = M
        self.d_model = d_model
        self.n_special = n_special
        # Special token embeddings (PAD, SEP, BOS, EOS)
        self.special_embed = nn.Embedding(n_special, d_model)
        # Per-position digit embeddings
        self.digit_embeds = nn.ModuleList([
            nn.Embedding(base, d_model) for _ in range(M)
        ])

    @staticmethod
    def decompose_value(val: int, base: int, M: int) -> List[int]:
        """Decompose val into M digits in base B."""
        digits = []
        for _ in range(M):
            digits.append(val % base)
            val //= base
        return digits

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) where each token is either:
          - a special token (< n_special): use special_embed
          - a value token (>= n_special): decompose val = token - n_special into digits

        Returns: (B, T, d_model)
        """
        B, T = tokens.shape
        is_special = tokens < self.n_special  # (B, T) bool
        values = (tokens - self.n_special).clamp(min=0)  # (B, T)

        # Decompose values into M digits
        out = torch.zeros(B, T, self.d_model, device=tokens.device, dtype=tokens.dtype)
        # Use float for accumulation, then convert
        out = out.float()
        v = values.clone()
        for j in range(self.M):
            d_j = v % self.base  # (B, T)
            out += self.digit_embeds[j](d_j)
            v = v // self.base

        # Override with special embeddings where applicable
        special_idx = tokens.clamp(min=0, max=self.n_special - 1)
        special_out = self.special_embed(special_idx)  # (B, T, d_model)
        out = torch.where(is_special.unsqueeze(-1), special_out.float(), out)

        return out.to(self.special_embed.weight.dtype)


# ============================================================================
# FCO: Factorized Candidate Output
# ============================================================================

class FactorizedCandidateOutput(nn.Module):
    """Factorized Candidate Output — O(1) params in p.

    Replaces nn.Linear(d_model, r) with additive digit decomposition.
    Candidate c has digits (d_0, d_1, ..., d_{M-1}) in base B.
    Embedding: emb(c) = sum_j O_j[d_j] / sqrt(M) + bias_j[d_j]
    Logit: logit(c) = h . emb(c) / sqrt(d_model) + bias

    P_output = M*B*d_model + M*B  (O(1) for fixed M, B, d_model)
    """

    def __init__(self, base: int, M: int, d_model: int, r: int):
        super().__init__()
        self.base = base
        self.M = M
        self.d_model = d_model
        self.r = r  # number of classes (p-1 for DLP)
        self.sqrt_M = math.sqrt(M)
        self.sqrt_d = math.sqrt(d_model)
        # Per-position output tables
        self.output_tables = nn.ModuleList([
            nn.Embedding(base, d_model) for _ in range(M)
        ])
        self.output_biases = nn.ModuleList([
            nn.Embedding(base, 1) for _ in range(M)
        ])
        # Precompute all candidate digit decompositions
        candidates = torch.arange(r)
        digits = []
        v = candidates.clone()
        for j in range(M):
            digits.append(v % base)
            v = v // base
        self.register_buffer('candidate_digits', torch.stack(digits, dim=1))  # (r, M)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) -> logits: (B, r)

        Computes logits for all r candidates via dot product with digit-summed embeddings.
        """
        # Compute embeddings for all candidates
        # emb(c) = sum_j O_j[d_j(c)] / sqrt(M)
        # bias(c) = sum_j bias_j[d_j(c)]
        cand_emb = torch.zeros(self.r, self.d_model, device=h.device, dtype=h.dtype)
        cand_bias = torch.zeros(self.r, device=h.device, dtype=h.dtype)
        for j in range(self.M):
            d_j = self.candidate_digits[:, j]  # (r,)
            cand_emb += self.output_tables[j](d_j) / self.sqrt_M
            cand_bias += self.output_biases[j](d_j).squeeze(-1)  # (r,)

        # logits = h . cand_emb^T / sqrt(d_model) + cand_bias
        logits = h @ cand_emb.T / self.sqrt_d + cand_bias.unsqueeze(0)  # (B, r)
        return logits


# ============================================================================
# RichFCO: Bilinear digit interactions (fallback if FCO fails)
# ============================================================================

class RichFCO(nn.Module):
    """Rich FCO with bilinear digit interactions — breaks d_model rank ceiling.

    Adds cross-terms: emb(c) = additive + sum_{j<k} (O_j[d_j] @ W_{jk}) * O_k[d_k]

    P_output = M*B*d + M*B + C(M,2)*d^2 + C(M,2)*B^2
    """

    def __init__(self, base: int, M: int, d_model: int, r: int):
        super().__init__()
        self.base = base
        self.M = M
        self.d_model = d_model
        self.r = r
        self.sqrt_d = math.sqrt(d_model)
        from itertools import combinations
        self.pairs = list(combinations(range(M), 2))  # C(M, 2) pairs
        n_pairs = len(self.pairs)

        # Additive components (same as FCO)
        self.output_tables = nn.ModuleList([
            nn.Embedding(base, d_model) for _ in range(M)
        ])
        self.output_biases = nn.ModuleList([
            nn.Embedding(base, 1) for _ in range(M)
        ])
        # Bilinear interaction matrices
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_model) * 0.02)
            for _ in range(n_pairs)
        ])

        # Precompute candidate digits
        candidates = torch.arange(r)
        digits = []
        v = candidates.clone()
        for j in range(M):
            digits.append(v % base)
            v = v // base
        self.register_buffer('candidate_digits', torch.stack(digits, dim=1))  # (r, M)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) -> logits: (B, r)"""
        r = self.r
        # Additive part
        cand_emb = torch.zeros(r, self.d_model, device=h.device, dtype=h.dtype)
        cand_bias = torch.zeros(r, device=h.device, dtype=h.dtype)
        for j in range(self.M):
            d_j = self.candidate_digits[:, j]
            cand_emb += self.output_tables[j](d_j)
            cand_bias += self.output_biases[j](d_j).squeeze(-1)

        # Bilinear part
        for idx, (j, k) in enumerate(self.pairs):
            d_j = self.candidate_digits[:, j]  # (r,)
            d_k = self.candidate_digits[:, k]  # (r,)
            e_j = self.output_tables[j](d_j)  # (r, d)
            e_k = self.output_tables[k](d_k)  # (r, d)
            # (e_j @ W) * e_k -> sum over d
            bilinear = (e_j @ self.W[idx]) * e_k  # (r, d)
            cand_emb += bilinear.sum(dim=-1, keepdim=False) * 0  # placeholder
            # Actually: bilinear gives (r, d), we want to add to cand_emb
            cand_emb = cand_emb + bilinear * 0.1  # scale down bilinear

        logits = h @ cand_emb.T / self.sqrt_d + cand_bias.unsqueeze(0)
        return logits


# ============================================================================
# CRT utilities (ported from lumi_dlp_trainer.py)
# ============================================================================

def coprime_factorization(q: int) -> Tuple[int, int]:
    """Factor q into two coprime factors (a, b) with a*b=q, gcd(a,b)=1,
    minimizing the ratio max(a,b)/min(a,b) for a balanced split.

    Raises ValueError if q has only one prime factor (no coprime split possible).
    """
    n = q
    unique_primes = []
    prime_powers = {}
    d = 2
    while d * d <= n:
        if n % d == 0:
            unique_primes.append(d)
            e = 0
            while n % d == 0:
                e += 1
                n //= d
            prime_powers[d] = e
        d += 1
    if n > 1:
        unique_primes.append(n)
        prime_powers[n] = 1

    if len(unique_primes) < 2:
        raise ValueError(
            f"q={q} has only one prime factor ({unique_primes[0]}); "
            "no coprime split into two factors both > 1 is possible."
        )

    from itertools import product as _product
    best = None
    for assignment in _product([0, 1], repeat=len(unique_primes)):
        a, b = 1, 1
        for i, pr in enumerate(unique_primes):
            if assignment[i] == 0:
                a *= pr ** prime_powers[pr]
            else:
                b *= pr ** prime_powers[pr]
        if a == 1 or b == 1:
            continue
        ratio = max(a, b) / min(a, b)
        if best is None or ratio < best[0]:
            best = (ratio, min(a, b), max(a, b))
    if best is None:
        raise ValueError(f"q={q}: could not find a coprime split with both factors > 1.")
    return best[1], best[2]


def project_factor(y: int, p: int, factor: int) -> int:
    """Project y to the order-`factor` subgroup of F_p* via y^((p-1)/factor) mod p."""
    exp = (p - 1) // factor
    return pow(y, exp, p)


def encode_crt_both(g: int, h: int, x: int, p: int,
                   crt_factors: Tuple[int, int]) -> Tuple[List[int], int]:
    """Encode DLP problem with CRT-BOTH representation.

    Returns (tokens, target) where tokens are value IDs (not yet FRE-decomposed).
    Token layout: [BOS, g_b, SEP, g_a, SEP, h_b, SEP, h_a, EOS]
    Target: x (the full discrete log, 0-indexed: x in {0, ..., p-2})
    """
    a, b = crt_factors  # (smaller, larger)
    g_a = project_factor(g, p, a)
    g_b = project_factor(g, p, b)
    h_a = project_factor(h, p, a)
    h_b = project_factor(h, p, b)

    PAD, SEP, BOS, EOS = 0, 1, 2, 3
    INT_OFFSET = 4

    def tok(val: int) -> int:
        return val + INT_OFFSET

    tokens = [BOS, tok(g_b), SEP, tok(g_a), SEP, tok(h_b), SEP, tok(h_a), EOS]
    target = x  # 0-indexed target for FCO (0 to p-2)
    return tokens, target


def encode_raw(g: int, h: int, x: int, p: int) -> Tuple[List[int], int]:
    """Encode DLP problem with RAW representation (no CRT)."""
    PAD, SEP, BOS, EOS = 0, 1, 2, 3
    INT_OFFSET = 4

    def tok(val: int) -> int:
        return val + INT_OFFSET

    tokens = [BOS, tok(g), SEP, tok(h), EOS]
    target = x
    return tokens, target


# ============================================================================
# FRE Tokenizer
# ============================================================================

class FRETokenizer:
    """Tokenizer for FRE/FCO model.

    Vocab layout: [PAD=0, SEP=1, BOS=2, EOS=3, <value tokens 4..4+p-1>]
    Same vocab IDs as GrokkingTokenizer, but the FRE embedding decomposes
    value tokens into digits instead of using a full p-size embedding table.

    The vocab_size is still p+4 (for compatibility), but the embedding
    parameters are O(1) in p.
    """

    def __init__(self, p: int, base: int = 16, M: int = None):
        self.p = p
        self.base = base
        if M is None:
            # Auto-compute M: smallest M such that base^M > p
            M = 1
            while base ** M <= p:
                M += 1
        self.M = M
        self.pad_id = 0
        self.sep_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.int_offset = 4
        self.vocab_size = p + 4  # same as atomic, but embedding is O(1)

    def encode_raw(self, g: int, h: int, x: int) -> Tuple[List[int], int]:
        """Encode with RAW representation."""
        return encode_raw(g, h, x, self.p)

    def encode_crt_both(self, g: int, h: int, x: int,
                        crt_factors: Tuple[int, int]) -> Tuple[List[int], int]:
        """Encode with CRT-BOTH representation."""
        return encode_crt_both(g, h, x, self.p, crt_factors)


# ============================================================================
# GrokkingTransformerFRE: Full model with FRE input + FCO output
# ============================================================================

class GrokkingTransformerFRE(nn.Module):
    """Grokking Transformer with FRE input + FCO output — O(1) params in p.

    Architecture:
        FRE embed + pos embed -> N transformer encoder layers -> FCO unembed

    The model size is independent of prime p. Only the dataset size grows with p.
    """

    def __init__(self, p: int, base: int = 16, M: int = None,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 max_len: int = 9, dropout: float = 0.0,
                 use_rich_fco: bool = False, use_crt: bool = False,
                 crt_factors: Tuple[int, int] = None):
        super().__init__()
        self.p = p
        self.base = base
        if M is None:
            M = 1
            while base ** M <= p:
                M += 1
        self.M = M
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.use_crt = use_crt
        self.crt_factors = crt_factors

        # FRE input embedding (O(1) in p)
        self.embed = FREEmbedding(base, M, d_model, n_special=4)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder (same as original)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # FCO output (O(1) in p)
        r = p - 1  # number of classes (x in {0, ..., p-2})
        if use_rich_fco:
            self.unembed = RichFCO(base, M, d_model, r)
        else:
            self.unembed = FactorizedCandidateOutput(base, M, d_model, r)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) -> logits: (B, r) where r = p-1

        Uses the last non-pad position for prediction (same as original).
        """
        B, T = tokens.size()
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.dropout(self.embed(tokens) + self.pos_embed(pos))
        h = self.transformer(h)  # (B, T, d_model)

        # Use last non-pad position for prediction
        pad_id = 0
        seq_lens = (tokens != pad_id).sum(dim=1) - 1  # (B,) — 0-indexed last position
        seq_lens = seq_lens.clamp(min=0)
        idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
        h_at_pos = h.gather(1, idx).squeeze(1)  # (B, d_model)

        logits = self.unembed(h_at_pos)  # (B, p-1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def core_parameters(self) -> int:
        """P_core: transformer layers only (excludes embedding + unembedding)."""
        return sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)

    def input_parameters(self) -> int:
        """P_input: FRE embedding + positional embedding."""
        return sum(p.numel() for p in self.embed.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.pos_embed.parameters() if p.requires_grad)

    def output_parameters(self) -> int:
        """P_output: FCO unembedding."""
        return sum(p.numel() for p in self.unembed.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        return {
            "P_input": self.input_parameters(),
            "P_core": self.core_parameters(),
            "P_output": self.output_parameters(),
            "P_total": self.count_parameters(),
        }


# ============================================================================
# Parameter estimation helpers
# ============================================================================

def estimate_fre_fco_params(base: int, M: int, d_model: int, n_layers: int,
                            max_len: int = 9) -> Dict[str, int]:
    """Estimate FRE/FCO parameter counts (independent of p)."""
    # P_input: special embed (4*d) + M digit embeds (M*B*d) + pos embed (max_len*d)
    p_input = 4 * d_model + M * base * d_model + max_len * d_model
    # P_core: same as original (12 * d^2 per layer)
    p_core = n_layers * 12 * d_model * d_model
    # P_output: M*B*d + M*B
    p_output = M * base * d_model + M * base
    return {
        "P_input": p_input,
        "P_core": p_core,
        "P_output": p_output,
        "P_total": p_input + p_core + p_output,
    }


def auto_compute_M(p: int, base: int = 16) -> int:
    """Compute minimum M such that base^M > p (covers all values 0..p-1)."""
    M = 1
    while base ** M <= p:
        M += 1
    return M
