# PT-HICnet Evaluation Protocol

This document freezes the evaluation language used before the expanded-data runs.
It is the reference for comparing the current 7-car LOCO-CV results with future
larger-data experiments.

## Scope

- Dataset split: 7-fold leave-one-car-out cross-validation.
- Held-out vehicles: C201, EP32, JX65, CY02C, M6, S50EVK, FX11.
- Normal-car group: C201, EP32, JX65, S50EVK, FX11.
- Hard-car group: CY02C, M6.
- Current frozen seed: training seed 42, split seed 2026.

## Data Split

- One vehicle is held out as test in each fold.
- The remaining vehicles form the training pool.
- `val_split=0.15` is drawn from the training pool.
- Validation split is stratified by `(age_group, HIC bucket)`.
- Test data is never used for checkpoint selection.

## Checkpoint Selection

- Main checkpoint: `checkpoints/best_acc_model.pth`.
- Selection key: highest validation `accuracy_ratio`.
- If no validation loader exists, selection falls back to test accuracy only for
  legacy/non-LOCO runs. Do not use that fallback for main LOCO reporting.
- `checkpoints/best_model.pth` is MSE-selected and is not the reporting
  checkpoint for accuracy tables.
- `restore_best=True` means the training process restores `best_acc_model.pth`
  after early stopping, but reported LOCO numbers must still be extracted from
  the integer best epoch in `history.json`.

## Metrics

`accuracy_ratio` is defined per sample as:

```python
p = abs(pred)
t = abs(target)
denom = max(p, t)
score = min(p, t) / denom if denom > 0 else 0
```

Reporting rules:

- Store as a 0-1 scalar in `history.json`.
- Display as a percentage in tables.
- Main LOCO table columns: Val, Test, Gap.
- Gap is `Val - Test`, displayed in percentage points.
- Main result table must include 7-car mean, 5-normal mean, 2-hard mean, and
  every per-vehicle fold.
- Negative prediction rate (`neg_rate`) is diagnostic, not a selection metric.

## Aggregation Caveat

E0/E1 and E2-E4 use different training scripts. Both script families now use
StepLR, but their metric plumbing is not identical:

- E0/E1 are trained through `feather/train_reg_att_props_X70_feather.py`.
- E2/E3/E4 are trained through `scripts/train_pt_hicnet.py`.
- The main LOCO report must use `history.json` integer epochs and the same
  best-validation-accuracy selection rule for all architectures.
- E0/E1 legacy logs may contain batch-level helper metrics from the baseline
  script. Those are diagnostic only and must not replace the epoch-level
  `val_accuracy` / `test_accuracy` values in the main table.
- When comparing architectures, prefer paired per-vehicle deltas over
  cross-script internal training curves.
- Expanded-data aggregation must pass a strict architecture x vehicle x seed
  completeness check before any acceptance gate is evaluated.
- Multi-seed architecture claims must report seed-paired fleet deltas, not only
  a pooled mean across all folds.

## Current Frozen Baseline

Use these numbers as the pre-expansion reference:

| Arch | Meaning | 7-car Test | 5-Normal Test | 2-Hard Test |
|------|---------|:---:|:---:|:---:|
| E0 | PointNet++ baseline | 69.02% | 71.21% | 63.53% |
| E1 | PointNet++ + EarlyFusion | 69.24% | 72.56% | 60.92% |
| E2 | Point Transformer, FiLM none | 72.65% | 77.63% | 60.22% |
| E3 | Point Transformer, FiLM global | 73.03% | 78.39% | 59.64% |
| E4 | Point Transformer, FiLM deep | 72.71% | 76.75% | 62.60% |

Primary frozen claims:

- PT backbone contribution: E2-E0 = +6.42pp on 5-normal cars.
- FiLM global contribution: E3-E2 = +0.76pp on 5-normal cars.
- Deep FiLM overfits: E4 has the highest Val mean but lower 5-normal Test than E2.
- Hard cars remain the main failure mode: CY02C/M6 favor E0 over PT variants.

## JX65 Soft Check

JX65 single-car E3-ES is an upper-bound reference, not the LOCO reporting
protocol. It used 6-car train, no validation split, and test-triggered early
stopping. LOCO JX65 uses 5.85-car train plus a 15% validation split and
validation-triggered selection.

Interpretation bands for comparing LOCO JX65 to the E3-ES reference:

- <= 3pp difference: OK.
- > 3pp and <= 5pp: WARN.
- > 5pp: INVESTIGATE, but do not mix the protocols in the main table.

## Expanded-Data Acceptance Gates

Set these gates before inspecting future large-data results:

Hard gates:

- E2-E0 on 5-normal Test Mean must be >= +5.0pp to keep the PT-backbone claim.
- E2-E0 on 7-car Test Mean must be >= +3.0pp to keep the full-fleet PT claim.
- E3-E2 on 5-normal Test Mean must be >= +1.5pp and positive on at least 4/5
  normal vehicles before claiming FiLM as a real contributor.
- If E3-E2 is between 0 and +1.5pp, report E3 as the best checkpoint only, not
  as evidence that FiLM is a significant component.
- PT hard-car Gap should be <= 20.0pp for CY02C and M6, or improve by at least
  5.0pp against the current E3 gaps (CY02C 26.3pp, M6 24.2pp).
- No architecture claim may rely on a single favorable fold; per-vehicle tables
  and paired deltas must be shown.

Soft gates:

- 7-car Test Std should not increase above 10.0pp.
- Normal-car Test Std should stay near the current ~2pp band.
- Multi-seed reporting should use seeds 42, 3407, and 2026 once the expanded
  dataset is stable. Report per-seed fleet means and mean +/- std of paired
  E2-E0 and E3-E2 deltas.

## Expanded-Data QA Gates

Run these checks before starting large training jobs on new data:

- Geometry scale: report union and local-sample bbox diagonals per vehicle.
  Do not assume CY02C/M6 are hard because of scale; current diagnostics show
  their bbox scale is near the normal-car mean.
- Material features: report both normalized z-score shifts and raw-unit
  estimates from `normalization_params.pkl`. Never interpret `mat_02` z-score
  directly as raw Poisson's ratio.
- Material coverage: report MID overlap and rounded 15D material-vector overlap
  against the old 5-normal-car union. MID IDs can be vehicle-local, so vector
  overlap is the stronger physical check.
- Material lookup completeness: every MID used by a valid sample must resolve
  to a 15D material vector. Any unresolved MID is a blocking QA failure.
- HIC distribution: report per-vehicle mean, max, and `>2k` count. M6 currently
  contains a 140176 HIC outlier; MSE must be interpreted with this tail in mind.
- Invalid labels: report all HIC=0 samples excluded by `HICLoader_feather.py`.
- Hard-car per-sample deltas: include paired E0 vs E3 rows. The current top-20
  E3-E0 deltas are all negative, but the full paired table must be shown because
  not every sample degrades.

## Reproduction Commands

```bash
python scripts/aggregate_loco.py --architectures all --results_root experiments

python scripts/error_analysis_hard_cars.py \
  --architectures E0 E2 E3 E4 \
  --vehicles CY02C M6 \
  --run_data_diagnostics \
  --run_inference \
  --results_root experiments

python scripts/error_analysis_hard_cars.py \
  --architectures E0 E2 E3 E4 \
  --vehicles CY02C M6 \
  --run_data_diagnostics \
  --reuse_inference_csv \
  --results_root experiments
```

## Long-Running Jobs

Run long training or full inference jobs inside `screen` so they survive SSH
disconnects and can be inspected later. If `screen` is unavailable on the host,
use `tmux` with the same session naming and logging policy.

Recommended naming pattern:

```bash
screen -S pthicnet_<phase>_<arch>_<vehicle_or_fold>
```

Fallback when `screen` is not installed:

```bash
tmux new-session -s pthicnet_<phase>_<arch>_<vehicle_or_fold>
```

Inside the screen session, run Python with unbuffered output and tee a log:

```bash
/data1/user/yikun/.conda/envs/dl/bin/python -u <script> <args> 2>&1 | tee <log_path>
```

Use `screen -ls` to list sessions, `screen -r <name>` to reattach, and detach
with `Ctrl-a d`. For tmux fallback, use `tmux ls`, `tmux attach -t <name>`, and
detach with `Ctrl-b d`.
