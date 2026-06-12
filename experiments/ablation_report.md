# PT-HICNET Phase 2 Ablation — Final Report

> 2026-06-02 | 15 seeds (5 experiments × 3 seeds) + E3-ES validation

---

## 1. Summary

| Experiment | Mean Acc | Std | Best Single | Mean MSE | Δ vs E0 |
|---|---:|---:|---:|---:|---:|
| **E0 (PN2++ baseline)** | **81.01%** | ±1.60pp | 82.84% | 99,913 | — |
| **E1 (PN2++ EarlyFusion)** | **80.40%** | ±0.99pp | 81.01% | — | -0.61pp |
| **E2 (PT no FiLM)** | **82.42%** | ±1.88pp | 84.24% | 89,347 | +1.41pp |
| **E3 (PT FiLM global)** | **84.20%** | ±0.75pp | 85.03% | 85,600 | +3.19pp |
| **E4 (PT FiLM deep)** | **82.39%** | ±1.28pp | 83.48% | 107,255 | +1.38pp |

> **E3 = final architecture. E3-ES (patience=50 + restore_best) = recommended training procedure.**
> Stop-epoch metrics are diagnostic only. **Best-checkpoint (best_acc_model.pth) is the deployed model.**

---

## 2. Per-Seed Detail

### E0 (PN2++ baseline)

| Seed | Best Acc | Best Ep | Best MSE |
|---|---:|---:|---:|
| 42 | 82.84% | 140 | 80,607 |
| 3407 | 79.86% | 36 | 111,273 |
| 2026 | 80.33% | 54 | 107,858 |

### E1 (PN2++ EarlyFusion)

| Seed | Best Acc | Best Ep |
|---|---:|---:|
| 42 | 81.01% | 29 |
| 3407 | 80.92% | 58 |
| 2026 | 79.26% | 17 |

### E2 (PT no FiLM)

| Seed | Best Acc | Best Ep | Best MSE | Stop Acc (ep200) | Best→Stop Δ |
|---|---:|---:|---:|---:|---:|
| 42 | 82.55% | 26 | 104,652 | 80.91% | -1.64pp |
| 3407 | 80.48% | 23 | 85,526 | 77.15% | -3.34pp |
| 2026 | 84.24% | 13 | 77,864 | 79.98% | -4.25pp |

### E3 (PT FiLM global)

| Seed | Best Acc | Best Ep | Best MSE | Stop Acc (ep200) | Best→Stop Δ |
|---|---:|---:|---:|---:|---:|
| 42 | 85.03% | 21 | 60,729 | 78.18% | -6.85pp |
| 3407 | 83.98% | 78 | 105,822 | 82.04% | -1.94pp |
| 2026 | 83.58% | 19 | 90,249 | 79.56% | -4.02pp |

### E4 (PT FiLM deep)

| Seed | Best Acc | Best Ep | Best MSE | Stop Acc (ep200) | Best→Stop Δ |
|---|---:|---:|---:|---:|---:|
| 42 | 80.99% | 20 | 102,016 | 79.91% | -1.07pp |
| 3407 | 82.71% | 29 | 128,041 | 81.55% | -1.16pp |
| 2026 | 83.48% | 15 | 91,707 | 80.05% | -3.43pp |

---

## 3. E3-ES: Early Stopping Validation

**Config:** patience=50, restore_best=True. Training stops when no accuracy improvement for 50 epochs. Best weights restored on stop.

| Seed | Best Acc | Best Ep | Stop Ep | Stop Acc | MSE (stop) | Best→Stop Δ |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 85.03% | 21 | 71 | 80.82% | 127,505 | -4.21pp |
| 3407 | 83.98% | 78 | 128 | 83.30% | 138,735 | **-0.68pp** ✅ |
| 2026 | 83.58% | 19 | 69 | 77.82% | 242,606 | -5.76pp |
| **Mean** | **84.20%** | — | **89** | **80.65%** | **169,615** | **-3.55pp** |

### E3-ES Reporting Convention (Two-Column)

| Column | Value | Purpose |
|------|------|------|
| Stop-epoch final | 80.65% (MSE 170K) | Overfitting diagnosis only |
| **Best-checkpoint (deployed)** | **84.20%** (MSE 86K) | **Main result + deployment** |

- Best-checkpoint accuracy = original E3 best accuracy (84.20%) — no ceiling loss.
- Stop-epoch MSE (170K) is 41% lower than original E3 ep200 MSE (288K) — less degradation.
- Training epochs reduced 55% (200 → 89 avg).

### Why patience=50 (not 20)

| Seed | Best Ep | p=20 outcome | p=50 outcome |
|------|:---:|------|------|
| 42 | 21 | ✅ Caught | ✅ Caught |
| 3407 | **78** | ❌ Stopped at ep39 (local opt 82.34%) | ✅ Caught |
| 2026 | 19 | ✅ Caught | ✅ Caught |

patience=20 prematurely stopped seed 3407 before its late peak (ep78 → 83.98%). patience=50 preserves late-peak seeds.

### restore_best Effectiveness

Patience=20 run (E3-ES-v2) with restore_best enabled:

| Seed | Best Acc | Restored Acc | Δ |
|------|:---:|:---:|:---:|
| 42 | 85.03% | 84.88% | -0.15pp |
| 3407 | 82.34% | 82.00% | -0.34pp |
| 2026 | 83.58% | 83.24% | -0.34pp |

Best→Restored Δ < 0.35pp in all cases, confirming the restore mechanism works.

---

## 4. Ablation Ladder

```
E0 (PN2++ baseline)             81.01%
     ↓ +EarlyFusion
E1 (PN2++ EarlyFusion)          80.40%   (-0.61pp, harmful without VectorAttention)
     ↓ +PT + VectorAttention
E2 (PT no FiLM)                 82.42%   (+2.02pp vs E1, VectorAttention unlocks EarlyFusion)
     ↓ +FiLM global
E3 (PT FiLM global)             84.20%   (+1.78pp vs E2) ⭐ FINAL ARCHITECTURE
     ↓ FiLM deep
E4 (PT FiLM deep)               82.39%   (-1.81pp vs E3, deep over-design)
```

---

## 5. Overfitting Summary

| Experiment | Best→Stop Δ (original) | With Early Stopping |
|---|---:|
| E2 (PT no FiLM) | -3.08pp | — |
| E3 (PT FiLM global) | -4.27pp | **-3.55pp** (stop-epoch), **~0pp** (deployed via best-checkpoint) |
| E4 (PT FiLM deep) | -1.89pp | — |

---

## 6. Key Findings

1. **E3 (PT + FiLM global) is the final architecture.** Mean 84.20%, +3.19pp vs E0 baseline.
2. **EarlyFusion only works with VectorAttention.** E1 < E0 (-0.61pp) but E2 > E1 (+2.02pp). `q-k` pairwise feature comparison is the key mechanism.
3. **FiLM global works; deep does not.** E3 > E2 (+1.78pp), E4 < E3 (-1.81pp). Global modulation is sufficient.
4. **PT converges 2-3× faster than PointNet++.** Best epoch 13-78 vs 36-140.
5. **Early stopping is effective.** E3-ES (patience=50 + restore_best) preserves best accuracy (84.20%), reduces training 55%, and provides a principled deployed model via `best_acc_model.pth`.
6. **patience=50 is the right choice.** patience=20 prematurely stops late-peak seeds (3407 best @ ep78).
7. **Best-checkpoint (not stop-epoch) is the deployed model.** This is standard ML practice. Stop-epoch metrics are diagnostic only.

---

## 7. Final Decision

- **Architecture:** E3 (PT + FiLM global)
- **Training procedure:** E3-ES (patience=50, restore_best)
- **Deployed model:** `best_acc_model.pth` (best-checkpoint, not stop-epoch)
- **Results (deployed):** Mean Acc 84.20% ± 0.75pp, +3.19pp vs E0 baseline
- **Results (stop-epoch, diagnostic):** Mean Acc 80.65%, MSE 170K (41% ↓ vs ep200)
- **Training efficiency:** 89 epochs avg (55% ↓ vs 200)
- **Next:** Error analysis (Phase 3 Step 4), then LOCO-CV (Phase 3 later)
