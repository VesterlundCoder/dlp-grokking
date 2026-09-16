# DLP Rule Extraction: Research Plan

## Goal

Extract the learned algorithm from grokked DLP transformers and determine whether it generalizes to larger primes.

## Background

The grokked M04 model (427k params, 2-layer transformer, $p=113$) achieves 99.8% test accuracy on the discrete logarithm problem. It generalizes from 30% of $\Fps^*$ to 100%, meaning it has learned a structured algorithm — not memorization. The question is: **what algorithm does the model implement, and can it be extracted and applied to larger primes?**

The discrete logarithm $\log_g(h)$ is the multiplicative Fourier transform: it maps from the "value" representation ($h$) to the "exponent" representation ($x$). The multiplicative characters $\chi_k(h) = e^{2\pi i k \log_g(h) / q}$ ($q = p-1$) form the natural Fourier basis for operations on $\Fps^*$.

## Phase 1: Fourier Structure Extraction

**Objective**: Extract the embedding and unembedding weight matrices, project them onto the multiplicative character basis, and identify the key frequencies the model uses.

### Steps

1. Load the grokked checkpoint (`final.pt`)
2. Extract $W_E$ (embedding: `model.embed.weight`, shape `(p+4, d_model)`) and $W_U$ (unembedding: `model.unembed.weight`, shape `(p+4, d_model)`)
3. Restrict to the $h$-token rows (indices 4..4+p-1, excluding PAD/SEP/BOS/EOS)
4. Compute the multiplicative character transform: $\hat{W}_E[k, :] = \sum_{h \in \Fps^*} \chi_k(h)^* \cdot W_E[h, :]$
5. Identify key frequencies: $k$ where $\|\hat{W}_E[k, :]\|$ is large
6. Verify the closed-form model: $\text{logit}(c) \approx \sum_{k \in K} \alpha_k \cos(2\pi k (\log_g(h) - c) / q) + \beta_k \sin(2\pi k (\log_g(h) - c) / q)$
7. Report: key frequency set $K$, coefficients $\{\alpha_k, \beta_k\}$, fit quality

### Expected Output

- A sparse set of key frequencies $K = \{k_1, k_2, \ldots, k_m\}$
- Per-frequency coefficients $\alpha_k, \beta_k$
- The number of key frequencies $|K|$ indicates algorithmic complexity
- If $|K| = O(\log q)$, the algorithm is efficient; if $|K| = O(\sqrt{q})$, it matches BSGS

### Implementation

`src/extract_algorithm.py` — loads checkpoint, extracts weights, computes multiplicative character transform, identifies key frequencies, fits closed-form model, reports extracted algorithm.

---

## Phase 2: Attention Head Decoding

**Objective**: Determine what each attention head computes in the Fourier basis, revealing the per-head algorithm specification.

### Steps

1. For each attention head $j$ in each layer:
   - Extract the OV circuit: $W_O^{(j)} \cdot W_V^{(j)}$ (value-output projection)
   - Extract the QK circuit: $W_Q^{(j)} \cdot W_K^{(j)}$ (query-key attention pattern)
2. Project the OV circuit onto multiplicative characters:
   - $\hat{W}_{OV}^{(j)}[k, :] = \sum_h \chi_k(h)^* \cdot W_{OV}^{(j)}[h, :]$
3. Identify which frequency each head specializes in:
   - Does head $j$ compute $\cos(k_j(a-b))$ or $\sin(k_j(a+b))$?
4. Check if heads partition the frequency space or redundantly compute the same frequencies
5. Map the full computation graph: input → embed → attention (frequency mixing) → MLP (trig identity) → unembed (inverse DFT)

### Expected Output

- Per-head frequency specialization table
- Trigonometric identity each head implements
- Whether the computation matches the "Discrete-Log Clock" algorithm

---

## Phase 3: Distillation to Interpretable Model

**Objective**: Build an explicit, minimal model that matches the grokked transformer's behavior, confirming the extracted algorithm.

### Steps

1. Build an explicit DFT+linear model:
   - Input: $h \in \Fps^*$
   - Compute multiplicative characters: $\chi_k(h)$ for $k \in K$ (key frequencies from Phase 1)
   - Output: $\text{logit}(c) = \sum_{k \in K} \alpha_k \cos(2\pi k (\log_g(h) - c) / q) + \beta_k \sin(2\pi k (\log_g(h) - c) / q)$
2. Train only the coefficients $\{\alpha_k, \beta_k\}$ to match the grokked model's logits on all 112 inputs
3. Find the minimum $|K|$ for 99% accuracy
4. Compare the distilled model's predictions with the original model

### Expected Output

- Minimal algorithm: key frequency set $K$ + coefficients
- Verification that the distilled model achieves 99%+ accuracy
- The number of key frequencies $|K|$ as a function of $q = p-1$

---

## Phase 4: Cross-Prime Transfer

**Objective**: Determine whether the extracted algorithm generalizes to larger primes.

### Steps

1. Train M04 on $p = 251$ ($q = 250$) and $p = 1009$ ($q = 1008$)
2. Extract Fourier structure from each grokked checkpoint
3. Compare key frequency sets across primes:
   - Are the same frequencies used? (prime-specific)
   - Is the frequency selection rule the same? (algorithmic transfer)
4. Fine-tune $p=251$ from the $p=113$ checkpoint — does it grok faster?
5. Zero-shot evaluation: apply the $p=113$ extracted algorithm to $p=251$

### Key Question: How does $|K|$ scale with $q$?

- $|K| = O(\log q)$: efficient algorithm, potential breakthrough
- $|K| = O(\sqrt{q})$: matches baby-step giant-step
- $|K| = O(q)$: no better than brute force

### Expected Output

- Cross-prime key frequency comparison
- Transfer learning results (fine-tuning speedup)
- Scaling law: $|K|$ vs $q$

---

## Phase 6: M12 CRT-BOTH Scaling (In Progress)

**Objective**: Test how the largest fast-grokking model (M12, 14.4M params) groks with CRT-BOTH representation across increasing primes and dataset sizes. This tests whether CRT-accelerated grokking scales beyond $p=113$ and whether larger models can handle larger curves.

### Design

10 runs: 5 power-of-2 primes × 2 train fractions (0.30 low, 0.60 high).

| Prime | $q=p-1$ | CRT split | Ratio | Solution space | Vocab |
|-------|---------|-----------|-------|---------------|-------|
| 113 | 112 | $7 \times 16$ | 2.29 | 6,272 | 117 |
| 241 | 240 | $15 \times 16$ | 1.07 | 28,800 | 245 |
| 337 | 336 | $16 \times 21$ | 1.31 | 56,448 | 341 |
| 601 | 600 | $24 \times 25$ | 1.04 | 180,000 | 605 |
| 1201 | 1200 | $25 \times 48$ | 1.92 | 720,000 | 1205 |

All jobs use M12 ($d_{model}=768$, $n_{heads}=24$, $n_{layers}=2$, ~14.4M params) with CRT-BOTH representation. Submitted as LUMI batch job 22038895.

### Key Questions

1. **Does CRT-BOTH accelerate grokking for larger primes?** At $p=113$, CRT-BOTH groks in ~21K epochs vs 500K+ for RAW. Does this acceleration hold for $p=241, 337, 601, 1201$?
2. **How does grokking time scale with prime size?** Does it scale as $O(\log p)$, $O(\sqrt{p})$, or $O(p)$?
3. **Does train fraction affect CRT-BOTH grokking?** At $p=113$, CRT-BOTH groks at 30%. Does the critical fraction $N_{crit}$ change with prime size?
4. **Can M12 handle the larger vocabulary?** For $p=1201$, the embedding layer has $1205 \times 768 \approx 926K$ params (vs $90K$ for $p=113$). Does this affect grokking?

### Technical Innovation

The CRT projection was generalized from the hardcoded $p=113$ case ($q=112=7\times16$) to support arbitrary primes via automatic coprime factorization of $q=p-1$. The `coprime_factorization(q)` function finds the most balanced split of $q$ into two coprime factors, and `project_factor(y, p, factor)` computes $y^{(p-1)/\text{factor}} \bmod p$ for any factor.

### Status

**Partially complete** — LUMI batch job 22038895 running. Results so far:

| Prime | Train Frac | Best Acc | Epochs | Status |
|-------|-----------|----------|--------|--------|
| 113 | 0.30 | 99.88% | 32,250 | ✅ |
| 113 | 0.60 | 99.95% | 5,100 | ✅ |
| 241 | 0.30 | 99.96% | 7,550 | ✅ |
| 241 | 0.60 | 100% | 5,050 | ✅ |
| 337 | 0.30 | 99.98% | 5,200 | ✅ |
| 337 | 0.60 | — | — | Running |
| 601 | 0.30 | — | — | Running |
| 601 | 0.60 | — | — | Pending |
| 1201 | 0.30 | — | — | Pending |
| 1201 | 0.60 | — | — | Pending |

**Key finding**: Grokking time *decreases* with prime size at fixed train fraction (p=337 groks 6× faster than p=113 at ρ=0.30). More training data at larger primes helps the model grok faster.

---

## Phase 8: Local CRT-BOTH Scaling with Small Models (In Progress)

**Objective**: Test how large primes we can grok locally with small models (M04-M07, 425k-1.6M params) using CRT-BOTH representation. This complements the M12 LUMI scaling by testing whether the representation alone (not model capacity) is sufficient for large-prime grokking.

### Motivation

The M12 LUMI results show CRT-BOTH groks p=241 in 7.5K epochs and p=337 in 5.2K epochs. But M12 has 14.4M params. Can a much smaller model (M04, 427k params) also grok these primes with CRT-BOTH? If so, the representation is the key factor, not model capacity.

### Local Results So Far

| Prime | CRT Split | Model | Best Acc | Epochs | Status |
|-------|-----------|-------|----------|--------|--------|
| 113 | 7×16 | M04 | 96.3% | 500 | ✅ Instant grokking |
| 241 | 15×16 | M04 | 100% | 1,600 | ✅ Grokked in 200 epochs |
| 337 | 16×21 | M04 | — | — | Overnight run (lr=3e-4) |
| 601 | 24×25 | M04 | — | — | Overnight run (lr=1e-4) |
| 1201 | 25×48 | M04 | — | — | Overnight run (lr=3e-5) |

**Key finding**: M04 (427k params) groks p=241 *faster* than M12 (14.4M params) on LUMI — 200 epochs vs 7,550 epochs. This suggests that for CRT-BOTH, model capacity is not the bottleneck; training data relative to solution space is.

### Overnight Run Design

12-hour local sweep testing M04, M05, M06, M07 on p=337, 601, 1201 with different learning rates:

1. M04 p=337 lr=3e-4 (1h budget)
2. M04 p=601 lr=1e-4 (2h budget)
3. M06 p=337 lr=3e-4 (1.5h budget)
4. M06 p=601 lr=1e-4 (2h budget)
5. M07 p=337 lr=3e-4 (2h budget)
6. M04 p=1201 lr=3e-5 (4h budget)
7. M05 p=337 lr=3e-4 tf=0.40 (1.5h budget)
8. M04 p=337 lr=3e-4 tf=0.50 (1.5h budget)

### Key Questions

1. **What is the largest prime M04 can grok with CRT-BOTH?** p=241 works instantly. Can p=337, 601, 1201 work?
2. **Does larger model capacity help for larger primes?** Compare M04 vs M06 vs M07 on p=337.
3. **Is the learning rate the bottleneck for larger primes?** p=337 diverges at lr=1e-3 but may work at lr=3e-4.
4. **Does train fraction matter for CRT-BOTH at larger primes?** Test tf=0.30 vs 0.50 for p=337.

---

## Cryptographic Significance

If the extracted algorithm is novel and has complexity $O(q^\epsilon)$ for $\epsilon < 1/2$, it would be better than baby-step giant-step. Known classical DLP algorithms:
- Baby-step giant-step: $O(\sqrt{q})$ time/space
- Pohlig-Hellman: reduces to prime factors of $q$
- Index calculus: $O(e^{O(\sqrt{\log p \log \log p})})$
- Number field sieve: $O(e^{O((\log p)^{1/3} (\log \log p)^{2/3})})$

The key insight is that the neural network may discover a **different algorithmic structure** than known methods — potentially exploiting regularities in the multiplicative group that classical algorithms don't use.

---

## Phase 7: Multi-Curve DLP (Future Work)

**Objective**: Train a single transformer on DLP problems from multiple primes/curves simultaneously, testing whether the model learns a *general* DLP algorithm or *curve-specific* algorithms.

### Motivation

All experiments so far train on a single prime $p$. If the model truly learns a general DLP algorithm, it should be able to solve DLP on multiple curves simultaneously without interference. If it learns curve-specific shortcuts, multi-curve training should degrade performance.

This directly tests **hypothesis H7** (cross-task generalization): does the grokked algorithm transfer across different group structures?

### Protocol

**Token format**: Add a "curve identifier" token to distinguish which prime/curve the problem is from:
$$\text{tokens} = [\text{BOS}, \text{CURSE\_ID}, \text{tok}(g), \text{SEP}, \text{tok}(h), \text{EOS}]$$

The curve ID is a special token (e.g., from a reserved vocabulary range) that tells the model which prime/curve the problem belongs to. This requires a unified vocabulary that covers all primes in the multi-curve set.

**Scaling protocol**:
1. **2 curves**: Start with $p=113$ and $p=241$ (both have balanced CRT structure: $7\times16$ and $15\times16$). Use M12 architecture.
2. **4 curves**: Add $p=337$ and $p=601$ ($16\times21$ and $24\times25$).
3. **8 curves**: Add $p=1201$, $p=2113$, $p=4801$, $p=12007$ (or similar primes with good CRT structure).
4. **16 curves**: Continue scaling to test the limit of multi-curve generalization.

**Vocabulary design**: The unified vocabulary must handle all primes. Two approaches:
- **Shared vocabulary**: Use a single integer token space covering $[0, \max(p)-1]$ for all curves. Curve ID disambiguates which prime is active.
- **Per-curve vocabulary**: Each curve has its own token range, with the curve ID implicitly selecting the range.

**Training data**: Mix problems from all curves in the training set. The train/test split is per-curve (each curve has its own train/test split). This tests whether the model can generalize on each curve independently.

### Success Criteria

- **General algorithm**: If the model groks on all curves simultaneously, it has learned a general DLP algorithm that works across different group structures.
- **Curve-specific**: If the model only groks on some curves (or performance degrades with more curves), it learns curve-specific algorithms.
- **Interference**: Measure whether adding more curves slows grokking on existing curves (negative transfer) or speeds it up (positive transfer via shared structure).

### Connection to Hypotheses

- **H7 (cross-task generalization)**: Directly tests whether the grokked algorithm transfers across different group structures.
- **H8 (non-DLP families)**: If the model learns a general DLP algorithm, it may also generalize to other cyclic group problems (elliptic curves, matrix groups).
- **Generality of extracted algorithm**: If Phase 3 (cross-prime transfer) shows that the extracted algorithm works on larger primes, multi-curve training tests whether it can be learned in a shared representation.

### Implementation Notes

- Requires extending the trainer to support multiple primes in a single dataset.
- The `L6Generator` needs to be generalized to produce problems from multiple primes with curve IDs.
- The tokenizer needs a unified vocabulary with curve ID tokens.
- The model architecture (M12) should be sufficient, but the embedding layer needs to handle the larger unified vocabulary.

### Status

**Exploratory experiments completed** — Multi-curve trainer implemented with CRT-BOTH support. Initial 2-curve experiments (p=113, p=127) showed the model can learn both curves simultaneously but does not fully grok either within 50K epochs (best: 80.5% aggregate, c0=89%, c1=70%). Single-curve p=113 with the same trainer groks instantly (96% in 500 epochs), confirming the multi-curve case is fundamentally harder.

The multi-curve experiment is paused in favor of the single-curve CRT-BOTH scaling sweep (Phase 8), which is more immediately productive. Multi-curve remains a future direction once we understand the single-curve scaling limits.
