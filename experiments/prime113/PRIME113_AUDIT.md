# PRIME113_AUDIT.md — M04 Prime-Order DLP Control Experiment

## M04 Configuration (Frozen)

The following configuration is cloned exactly from the successful M04 run
(`results/local_probe/M04_s42/config.json`):

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 2 |
| max_len | 5 |
| vocab_size | 117 |
| lr | 0.001 |
| wd_start | 0.05 |
| wd_max | 0.30 |
| wd_step | 0.05 |
| wd_acc_threshold | 0.05 |
| wd_ramp_interval | 1000 |
| lr_drop_factor | 1.0 |
| eval_interval | 10 |
| checkpoint_interval | 500 |
| early_stop_patience | 100 |
| early_stop_threshold | 0.99 |
| grad_clip | 1.0 |
| optimizer | AdamW |
| activation | GELU |
| normalization | norm_first (pre-LN) |
| dropout | 0.0 |
| P_total | 427,253 |
| P_core | 396,544 |

## Group Construction

- **Ambient field**: F_227* (order 226 = 2 × 113)
- **Subgroup H**: Quadratic residues mod 227 (order 113, prime)
- **Construction**: H = {y ∈ F_227* : y^113 ≡ 1 (mod 227)}
- **Verified**: |H| = 113, every non-identity element has order 113
- **Canonical generator**: smallest element of H\{1}

## Tokenization

- **Vocab layout**: [PAD=0, SEP=1, BOS=2, EOS=3, rank(h₀), rank(h₁), ..., rank(h₁₁₂)]
- **Mapping**: sorted(H) → token IDs 4..116 (deterministic, generator-independent)
- **No discrete log in tokenization**: uses rank in sorted subgroup, not dlog
- **vocab_size = 117** (identical to M04)
- **Output**: x ∈ {0,...,112} → tokens 4..116 (same 113 output symbols)
- **Token map SHA256**: saved in `data/subgroup_token_map.json`

## Dataset

- **Complete domain**: 112 generators × 113 exponents = 12,656 examples
- **Verification**: every row satisfies pow(g, x, 227) == h

## Splits

### P113-A: Coverage-Matched
- Train fraction: 30% (same as M04)
- N_train ≈ 3,797, N_test ≈ 8,859
- Same split logic (random shuffle with same seed)

### P113-B: Sample-Count Matched
- N_train = 1,612 (exactly same as M04)
- N_test = 11,044
- Isolates effect of prime-order structure from dataset size

### P113-OOD: Disjoint Generators
- 60% generators for train, 40% for test
- Tests whether model learns group-wide DLP rule vs generator-specific maps

## Training

- Exact M04 regime: same optimizer, WD schedule, LR, grad clip, etc.
- Seeds: [42] (initial discovery), [42, 123, 456, 789, 2026] for confirmation
- Epoch budget: 100,000
- Metrics tracked: train/test accuracy, loss, WD, weight norm, timing

## Status

- [x] Group construction verified
- [x] Token map saved with SHA256
- [x] Dataset generated and verified
- [x] Split manifests saved
- [x] P113-A training launched (seed=42)
- [ ] P113-B training
- [ ] P113-OOD training
- [ ] Mechanistic analysis (Fourier extraction)
- [ ] Comparison with q=112 M04
