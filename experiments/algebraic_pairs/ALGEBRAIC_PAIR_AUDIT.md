# ALGEBRAIC_PAIR_AUDIT.md — M04 Algebraic-Pair DLP Representation Study

## M04 Configuration (Frozen)

Exact clone of M04 (`results/local_probe/M04_s42/config.json`):

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 2 |
| max_len | 13 (longest variant = RAW+CRT; all variants use same) |
| vocab_size | 117 |
| lr | 0.001 |
| wd_start | 0.05 |
| wd_max | 0.30 |
| wd_step | 0.05 |
| wd_ramp_interval | 1000 |
| lr_drop_factor | 1.0 |
| optimizer | AdamW |
| activation | GELU |
| normalization | norm_first (pre-LN) |
| dropout | 0.0 |
| P_total | 428,277 (vs M04's 427,253; diff = 1,024 positional params) |
| P_core | 396,544 (identical to M04) |

**Note on max_len**: The original M04 used max_len=5. We use max_len=13 for ALL variants
to keep parameter count identical across representations. The 1,024 extra positional
parameters (8×128) are negligible (0.24% of total) and identical across all variants,
so they do not confound the comparison.

## Algebraic Projection

For y ∈ F_113* (order q = 112 = 16 × 7):

- **y16 = y^7 mod 113** — order divides 16 (Z_16 component)
- **y7 = y^16 mod 113** — order divides 7 (Z_7 component)

Because gcd(16, 7) = 1, the pair (y16, y7) uniquely determines y via CRT.
No discrete logarithm is used in constructing these projections.

## Representation Variants

| Variant | Input tokens | Sequence length | Description |
|---------|-------------|-----------------|-------------|
| RAW | [BOS, g, SEP, h, EOS] | 5 | Original M04 (control) |
| CRT-BOTH | [BOS, g16, SEP, g7, SEP, h16, SEP, h7, EOS] | 9 | Full CRT decomposition |
| CRT-16 | [BOS, g16, SEP, h16, EOS] | 5 | Order-16 component only |
| CRT-7 | [BOS, g7, SEP, h7, EOS] | 5 | Order-7 component only |
| RAW+CRT | [BOS, g, SEP, h, SEP, g16, SEP, g7, SEP, h16, SEP, h7, EOS] | 13 | Raw + CRT (maximum info) |
| SCRAMBLED | [BOS, π(g16), SEP, π(g7), SEP, π(h16), SEP, π(h7), EOS] | 9 | Random permutation of CRT tokens |

All variants padded to max_len=13 with PAD tokens.

## Theoretical Ceilings

| Variant | Determined info | Ambiguity | Max accuracy |
|---------|----------------|-----------|-------------|
| CRT-16 | x mod 16 | 7 lifts | 1/7 ≈ 14.29% |
| CRT-7 | x mod 7 | 16 lifts | 1/16 ≈ 6.25% |
| CRT-BOTH | x mod 112 (full) | 1 lift | 100% |
| RAW+CRT | x mod 112 (full) | 1 lift | 100% |

If single-component models substantially exceed these ceilings, AUDIT FOR LEAKAGE.

## Scrambled Control

- Deterministic random permutation π: {0,...,112} → {0,...,112} (seed=12345)
- Applied to CRT component values before tokenization
- Preserves: sequence length, vocabulary size, marginal frequencies
- Destroys: natural algebraic labeling (group structure)

## Dataset

- **Same underlying (g, h, x) examples as M04**: F_113*, q=112
- **Train fraction**: 30% (same as M04)
- **N_train**: 1,612, **N_test**: 3,224
- **All variants use identical train/test examples** — only tokenization differs

## Smoke Test Results (200 epochs)

| Variant | Mem epoch | Best test acc |
|---------|-----------|---------------|
| RAW | 110 | 5.52% |
| CRT-BOTH | 120 | **73.60%** |
| CRT-16 | — | 9.46% |
| CRT-7 | — | 3.97% |

**Key observation**: CRT-BOTH reaches 73.6% in just 200 epochs, while RAW is at 5.5%.
This suggests the CRT decomposition dramatically accelerates grokking.

## Training

- All variants: same optimizer, WD schedule, LR, seeds, epoch budget
- Epoch budget: 100,000
- Seeds: [42] (initial), [42, 123, 456, 789, 2026] for confirmation

## Status

- [x] Algebraic projections verified
- [x] Scrambled map saved
- [x] Theoretical ceilings computed
- [x] Smoke test passed (200 epochs)
- [ ] Full training (100k epochs, all 6 variants)
- [ ] Component-specific probes
- [ ] Causal ablation
- [ ] Comparison figure
