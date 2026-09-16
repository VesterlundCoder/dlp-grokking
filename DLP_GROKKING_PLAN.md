# DLP Grokking Scaling Plan

**Created:** September 15, 2026
**Goal:** Find a model architecture that can grok DLP at arbitrarily large primes with O(1) model size.

## Motivation

The first DLP Grokking paper shows that small transformers grok the finite-field discrete logarithm problem. But the current architecture has a fundamental limitation: model size grows linearly with prime p because both the input embedding (`nn.Embedding(p+4, d_model)`) and output unembedding (`nn.Linear(d_model, p+4)`) scale with p. This is the "tokenizer confound" noted in the paper's limitations.

The EI4 ECDLP study solved this problem with two innovations:
1. **FRE (Factorized Residue Embedding)** — O(1) input parameters
2. **FCO (Factorized Candidate Output)** — O(1) output parameters

Combined with **CRT-BOTH** representation (which decomposes the DLP into simpler subproblems), this enables a fixed-size model to grok DLP at arbitrarily large primes.

## Key Insight from EI4

The EI4 study tested elliptic curve DLP (harder than finite-field DLP) and found:

| r | Approach | Result |
|---|---------|--------|
| 251 | FRE + FCO | Grokked (5/5 seeds) |
| 509 | FRE + FCO | Grokked (5/5 seeds) |
| 949 | FRE + FCO B=256 | Grokked (99.66%) |
| 949 | FRE + RichFCO B=16 | Grokked (98.78%, faster) |
| 1883 | All approaches | **HARD WALL** — no approach works |
| 3949 | All approaches | **HARD WALL** |

**Critical findings:**
- Base B matters far more than M (digit count). B=16 can't memorize for large r; B=256 can.
- Bigger models (d_model=256) HURT grokking — deeper memorization basin.
- More data at grokkable r accelerates 10×; more data at hard r HURTS.
- The r=1883 wall is structural, not capacity-related.
- RichFCO (bilinear interactions) helps at medium r but not at the wall.

## What Transfers to Finite-Field DLP

Finite-field DLP is simpler than ECDLP:
- Cyclic group (1 operation) vs elliptic curve (point addition + doubling)
- CRT decomposition available for most primes (q=p-1 is usually composite)
- The algebraic structure is more transparent

**Hypothesis:** The structural wall for finite-field DLP is further out than for ECDLP. The combination of CRT-BOTH (problem decomposition) + FRE/FCO (O(1) model) should push the frontier significantly beyond p=8209.

## Architecture: CRT-BOTH + FRE/FCO

### Input: CRT-BOTH + FRE

For prime p with q=p-1 = a × b (coprime):
```
Input: [BOS, g_b, SEP, g_a, SEP, h_b, SEP, h_a, EOS]
```
Each value is decomposed by FRE into M digits in base B:
- P_input = 4*d_model + M*B*d_model + max_len*d_model (O(1) in p)

### Output: FCO

Target x ∈ {0, ..., p-2} decomposed into M digits in base B:
- P_output = M*B*d_model + M*B (O(1) in p)

### Core: Same 2-layer transformer

- P_core = 12 * d_model² * n_layers ≈ 393k for d=128, L=2

### Total: ~422-526k params for ANY prime

| Prime | CRT Split | M | B | P_total | Atomic P_total | Reduction |
|-------|----------|---|---|---------|----------------|-----------|
| 113 | 7×16 | 1 | 256 | 464k | 424k | 0.9× |
| 8209 | 27×304 | 2 | 256 | 526k | 2,497k | 4.7× |
| 32771 | 145×226 | 2 | 256 | 526k | 8,785k | 16.7× |

## Sanity Check + Scaling Results (FRE/FCO + CRT-BOTH, B=256)

| Prime | CRT Split | M | P_total | Best Acc | T_mem | T_90 | Grokked? |
|-------|----------|---|---------|----------|-------|------|----------|
| 113 | 7×16 | 1 | 464k | 100% | 100 | ~2,000 | YES |
| 241 | 15×16 | 1 | 464k | 100% | 300 | ~400 | YES |
| 337 | 16×21 | 2 | 530k | 100% | 400 | ~400 | YES |
| 601 | 24×25 | 2 | 530k | 99.95% | 500 | ~400 | YES |
| 769 | 3×256 | 2 | 530k | 100% | 2200 | ~5400 | YES |
| 1201 | 25×48 | 2 | 530k | 100% | 300 | ~200 | YES |
| 2003 | 26×77 | 2 | 530k | 100% | 200 | ~200 | YES |
| **4001** | **32×125** | **2** | **530k** | **100%** | **500** | **~500** | **YES** |
| 8209 | 27×304 | 2 | 530k | — | — | — | Queued (LUMI) |
| 16411 | 30×547 | 2 | 530k | — | — | — | Queued (LUMI) |
| 32771 | 145×226 | 2 | 530k | — | — | — | Queued (LUMI) |

**Key findings:**
- B=256 is essential (B=16 fails even at p=113 with CRT-BOTH)
- M=1 works for p ≤ 256; M=2 needed for p > 256 (target x > 255)
- The same ~530k model groks p=1201 in 300 epochs on LUMI — faster than atomic M12 (14.4M params, 5K epochs)
- O(1) parameter count confirmed: p=113 (464k), p=241 (464k), p=337 (530k), p=601 (530k), p=1201 (530k) — all similar

## Scaling Ladder

### Phase 1: Verify the ladder (local + LUMI)

| Prime | CRT Split | M | B | P_total | Dataset Size | Platform |
|-------|----------|---|---|---------|-------------|----------|
| 113 | 7×16 | 1 | 256 | 464k | 5,376 | Local (done) |
| 241 | 15×16 | 1 | 256 | 464k | 46,080 | Local |
| 601 | 24×25 | 1 | 256 | 464k | 360,000 | Local/LUMI |
| 1201 | 25×48 | 2 | 256 | 526k | 1,440,000 | LUMI |
| 2003 | 26×77 | 2 | 256 | 526k | ~3,900,000 | LUMI |
| 4001 | 32×125 | 2 | 256 | 526k | ~16,000,000 | LUMI |
| 8209 | 27×304 | 2 | 256 | 526k | ~67,000,000 | LUMI |

### Phase 2: Push the frontier

| Prime | CRT Split | M | B | P_total | Platform |
|-------|----------|---|---|---------|----------|
| 16411 | 30×547 | 2 | 256 | 526k | LUMI |
| 32771 | 145×226 | 2 | 256 | 526k | LUMI |

### Phase 2b: Extended ladder (10 new primes above 32771)

| Prime | CRT Split | M | B | P_total | Atomic P_total | Reduction | Dataset Size | Status |
|-------|----------|---|---|---------|----------------|-----------|-------------|--------|
| 39953 | 176×227 | 2 | 256 | 530k | 10,628k | 20.1× | 1.6B | Planned |
| 50021 | 205×244 | 2 | 256 | 530k | 13,255k | 25.0× | 2.5B | Planned |
| 65539 | 198×331 | 3 | 256 | 594k | 17,479k | 29.4× | 4.3B | Planned |
| 79999 | 201×398 | 3 | 256 | 594k | 21,280k | 35.8× | 6.4B | Planned |
| 99991 | 202×495 | 3 | 256 | 594k | 26,598k | 44.8× | 10.0B | Planned |
| 130003 | 282×461 | 3 | 256 | 594k | 34,581k | 58.2× | 16.9B | Planned |
| 160001 | 256×625 | 3 | 256 | 594k | 42,560k | 71.6× | 25.6B | Planned |
| 199999 | 369×542 | 3 | 256 | 594k | 53,200k | 89.6× | 40.0B | Planned |
| 249973 | 444×563 | 3 | 256 | 594k | 66,493k | 111.9× | 62.5B | Planned |
| 300017 | 272×1103 | 3 | 256 | 594k | 79,805k | 134.3× | 90.0B | Planned |

**Note:** M=3 needed for primes above ~65536 (256² = 65536 insufficient for values up to q-1).
**Launcher:** `lumi_dlp/slurm_fre_fco_extended.sh`

### Phase 3: If wall is hit (fallback strategies)

If the base configuration (d_model=128, L=2, M=2-3, B=256) fails to grok at some prime p*:

1. **2x model size** — d_model=256, L=2 (~1.5M params, still O(1) in p)
   - Note: EI4 found d=256 can HURT grokking at medium scales (deeper memorization basin)
   - Try only after base config fails
2. **2x dataset** — train_frac=0.60 instead of 0.30
   - EI4 found 2x data accelerates grokking 10x at grokkable scales
   - But can HURT at hard scales (makes memorization harder)
3. **2x model + 2x data combined** — both together
4. **RichFCO** — bilinear digit interactions (breaks d_model rank ceiling)
5. **Larger base B** — B=512 or B=1024 (B matters more than M)
6. **Curriculum learning** — train on smaller primes first, transfer

**Launcher:** `lumi_dlp/slurm_fre_fco_fallback.sh` (templates for each strategy)

## Experiment Commands

```bash
# p=113 sanity check (DONE — grokked at 100% in 2900 epochs)
python experiments/fre_fco/run_fre_fco.py --prime 113 --crt --base 256 --M 1 \
    --d-model 128 --epochs 10000 --train-frac 0.30

# p=241 (local)
python experiments/fre_fco/run_fre_fco.py --prime 241 --crt --base 256 --M 1 \
    --d-model 128 --epochs 50000 --train-frac 0.30

# p=601 (local, may need mini-batch)
python experiments/fre_fco/run_fre_fco.py --prime 601 --crt --base 256 --M 1 \
    --d-model 128 --epochs 100000 --train-frac 0.30 --batch-size 8192

# p=8209 (LUMI, mini-batch required)
python experiments/fre_fco/run_fre_fco.py --prime 8209 --crt --base 256 --M 2 \
    --d-model 128 --epochs 500000 --train-frac 0.30 --batch-size 8192

# p=32771 (LUMI, mini-batch required)
python experiments/fre_fco/run_fre_fco.py --prime 32771 --crt --base 256 --M 2 \
    --d-model 128 --epochs 1000000 --train-frac 0.30 --batch-size 4096
```

## Success Criteria

1. **p=113 with CRT-BOTH + FRE/FCO** — DONE (100% in 2900 epochs, 464k params)
2. **p=8209 with CRT-BOTH + FRE/FCO** — Same 526k model groks p=8209
3. **p=32771 with CRT-BOTH + FRE/FCO** — Same 526k model groks p=32771
4. **O(1) parameter count verified** — P_total independent of p

## Risks

1. **The r=1883 wall**: EI4 hit a structural wall at r=1883. For finite-field DLP with CRT-BOTH, the effective problem is decomposed (e.g., max(27, 304) = 304 for p=8209), so the wall may be further out.

2. **Base B sensitivity**: B=16 fails, B=256 works. For larger primes, B=256 with M=2 should work (256² = 65536 > 32771), but may need testing.

3. **Training time**: Larger primes = larger datasets = longer training. Mini-batch training is essential for p>601.

4. **MPS stability**: p=601 collapsed on local MPS with atomic embeddings. FRE/FCO may behave differently — needs testing.

## File Map

- `src/fre_fco.py` — FRE, FCO, RichFCO, FRETokenizer, GrokkingTransformerFRE
- `experiments/fre_fco/run_fre_fco.py` — Training script
- `results/fre_fco/` — Experiment results
- `paper/crt_material.tex` — CRT-BOTH representation paper (separate preprint)
- `paper/paper.tex` — Main DLP scaling paper (tokenizer confound now resolved)
- `paper/fixed_size/paper.tex` — **Fixed-size grokking paper** (this scaling study)
- `lumi_dlp/slurm_fre_fco_all.sh` — LUMI launcher for primes 2003-32771
- `lumi_dlp/slurm_fre_fco_extended.sh` — LUMI launcher for primes 39953-300017
- `lumi_dlp/slurm_fre_fco_fallback.sh` — LUMI launcher for fallback experiments (2x model, 2x data)

## Connection to Papers

1. **Main DLP Scaling Paper** (`paper/paper.tex`): Shows grokking scales with model capacity and data. Limitations note the tokenizer confound — now resolved by FRE/FCO.

2. **Fixed-Size Grokking Paper** (`paper/fixed_size/paper.tex`): This scaling study. Shows that FRE/FCO + CRT-BOTH enables a fixed ~530k model to grok DLP from p=113 to p=300,017 (planned), with O(1) parameters. Includes the 21-prime scaling ladder and fallback strategies.

3. **Representation-Relative Grokking Paper** (`paper/crt_material.tex`): Shows CRT-BOTH changes the learning regime from delayed grokking to rapid generalization. FRE/FCO makes this scalable to large primes.

4. **EI4 ECDLP Study** (external): Source of FRE/FCO architecture. Hit a structural wall at r=1883. Finite-field DLP may have a different wall location.
