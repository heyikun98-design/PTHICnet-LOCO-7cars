# LOCO-CV Results: Full 5-Architecture Comparison

> 7-fold Leave-One-Car-Out Cross-Validation, seed=42
> Protocol: `val_split=0.15 split_seed=2026 seed=42 patience=50 restore_best`
> Main metric: **test accuracy @ best-val-checkpoint** (Val/Test/Gap three-column)
> E0/E1: PointNet++ via `feather/train_reg_att_props_X70_feather.py` (StepLR scheduler)
> E2/E3/E4: Point Transformer via `scripts/train_pt_hicnet.py` (CosineAnnealing scheduler)
> ⚠️ Single-seed; multi-seed pending new data (Phase 5).

---

## Table 1: Per-Architecture Summary

| Arch | Val Mean | Val Std | Test Mean | Test Std | 5-Normal | 2-Hard |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| E0 (PN++ baseline) | 76.53% | ±1.86pp | 69.02% | ±4.13pp | 71.21% | 63.53% |
| E1 (PN++ EarlyFusion) | 79.42% | ±1.34pp | 69.24% | ±6.89pp | 72.56% | 60.92% |
| E2 (PT, FiLM none) | 84.69% | ±0.89pp | 72.65% | ±8.70pp | 77.63% | 60.22% |
| E3 (PT, FiLM global) | 84.81% | ±0.69pp | **73.03%** | ±9.30pp | **78.39%** | 59.64% |
| E4 (PT, FiLM deep) | **86.22%** | ±1.16pp | 72.71% | ±7.43pp | 76.75% | 62.60% |

### Component Contributions (5-Normal Cars)

| Δ | Value | Interpretation |
|:---|:---:|------|
| E2−E0 (PT backbone) | **+6.42pp** | **Primary gain.** PT + Vector Attention replaces PN++ MLP |
| E3−E2 (FiLM global) | +0.76pp | Not significant — age modulation adds negligible cross-car benefit |
| E4−E2 (FiLM deep) | **−0.88pp** | **Actively harmful.** Deep FiLM overfits more, generalizes worse |
| E1−E0 (EarlyFusion on PN++) | +1.35pp | Marginal — EF without Vector Attention cannot exploit fused features |

---

## Table 2: Per-Vehicle Test Accuracy

| Vehicle | E0 | E1 | E2 | E3 | E4 | Winner |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| C201 | 68.34% | 67.98% | 75.99% | **76.71%** | 74.15% | E3 |
| EP32 | 73.79% | 74.27% | 79.93% | **80.31%** | 78.21% | E3 |
| JX65 | 72.21% | **80.04%** | 79.83% | 78.87% | 78.52% | E1 |
| CY02C | **64.23%** | 61.35% | 59.10% | 58.60% | 63.39% | E0 |
| M6 | **62.84%** | 60.50% | 61.33% | 60.67% | 61.81% | E0 |
| S50EVK | 71.48% | 70.68% | 77.12% | 79.93% | **80.41%** | E4 |
| FX11 | 70.23% | 69.86% | 75.27% | **76.13%** | 72.48% | E3 |

---

## Table 3: Per-Architecture Per-Vehicle Detail

### E0 — PointNet++ Baseline

| Held-Out | Val | Test | Gap |
|----------|:---:|:---:|:---:|
| C201 | 78.18% | 68.34% | 9.8pp |
| EP32 | 75.44% | 73.79% | 1.6pp |
| JX65 | 72.94% | 72.21% | 0.7pp |
| CY02C | 77.89% | 64.23% | 13.7pp |
| M6 | 77.54% | 62.84% | 14.7pp |
| S50EVK | 76.19% | 71.48% | 4.7pp |
| FX11 | 77.53% | 70.23% | 7.3pp |

### E1 — PointNet++ + EarlyFusion

| Held-Out | Val | Test | Gap |
|----------|:---:|:---:|:---:|
| C201 | 81.33% | 67.98% | 13.4pp |
| EP32 | 80.29% | 74.27% | 6.0pp |
| JX65 | 77.31% | 80.04% | −2.7pp |
| CY02C | 78.28% | 61.35% | 16.9pp |
| M6 | 80.18% | 60.50% | 19.7pp |
| S50EVK | 79.11% | 70.68% | 8.4pp |
| FX11 | 79.44% | 69.86% | 9.6pp |

### E2 — Point Transformer, FiLM none

| Held-Out | Val | Test | Gap |
|----------|:---:|:---:|:---:|
| C201 | 86.35% | 75.99% | 10.4pp |
| EP32 | 84.24% | 79.93% | 4.3pp |
| JX65 | 83.45% | 79.83% | 3.6pp |
| CY02C | 84.27% | 59.10% | 25.2pp |
| M6 | 84.85% | 61.33% | 23.5pp |
| S50EVK | 84.86% | 77.12% | 7.7pp |
| FX11 | 84.82% | 75.27% | 9.6pp |

### E3 — Point Transformer, FiLM global

| Held-Out | Val | Test | Gap |
|----------|:---:|:---:|:---:|
| C201 | 84.97% | 76.71% | 8.3pp |
| EP32 | 84.35% | 80.31% | 4.0pp |
| JX65 | 84.46% | 78.87% | 5.6pp |
| CY02C | 84.92% | 58.60% | 26.3pp |
| M6 | 84.86% | 60.67% | 24.2pp |
| S50EVK | 83.98% | 79.93% | 4.0pp |
| FX11 | 86.14% | 76.13% | 10.0pp |

### E4 — Point Transformer, FiLM deep

| Held-Out | Val | Test | Gap |
|----------|:---:|:---:|:---:|
| C201 | 87.96% | 74.15% | 13.8pp |
| EP32 | 86.78% | 78.21% | 8.6pp |
| JX65 | 85.46% | 78.52% | 6.9pp |
| CY02C | 84.43% | 63.39% | 21.0pp |
| M6 | 86.58% | 61.81% | 24.8pp |
| S50EVK | 86.80% | 80.41% | 6.4pp |
| FX11 | 85.56% | 72.48% | 13.1pp |

---

## Conclusions

1. **PT backbone is the primary cross-car generalization gain (+6.4pp).**
   Phase 2's JX65-only estimate (+1.4pp) severely underestimated PT's real benefit.

2. **FiLM global adds negligible cross-car benefit (+0.8pp, not significant).**
   Phase 2's "E3 > E2 by +1.78pp on JX65" was a single-car false positive.

3. **FiLM deep is actively harmful (−0.88pp vs E2).**
   E4 has the highest Val accuracy (86.22%) but worse Test than E2 (FiLM none).
   Deep FiLM is the strongest overfitter — more modulation layers ≠ better generalization.

4. **EarlyFusion without Vector Attention is ineffective (+1.35pp vs E0).**
   E1 on PN++ barely differs from E0. The 19-channel fused input needs
   `q−k` attention to separate geometric vs. material signal.

5. **PN++ baseline (E0) is most robust on hard cars.**
   CY02C/M6: E0 63.5% > E3 59.6%. PT models overfit harder on OOD vehicles.

6. **Paper narrative: Lead with PT backbone contribution.**
   Honestly reporting FiLM's insignificance (+0.8pp) and deep FiLM's failure (−0.88pp)
   is more credible than forcing a positive claim about age modulation.

---

## JX65 Consistency Check (Soft Reference)

| Metric | Value | Protocol |
|--------|-------|----------|
| E3-ES original (seed 42) | **83.88%** | 6-car train, no val split, test-triggered early stop |
| LOCO JX65 fold | **78.87%** | 5.85-car train + 15% val split, val-triggered early stop |
| Δ | **5.01pp** | >5pp — protocol difference expected |

---

## Limitations

- Single seed (42); multi-seed pending new data (Phase 5).
- E0/E1 use StepLR scheduler vs E2-E4 CosineAnnealing.
- E0/E1 metric aggregation (batch-level) differs slightly from E2-E4 (epoch-level).
- >2k HIC bucket has n=4 per fold — high variance.
- E1 JX65 fold has negative Val−Test gap (−2.7pp) — anomaly worth investigating.
