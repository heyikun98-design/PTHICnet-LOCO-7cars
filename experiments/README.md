# Ablation Execution Guide

## 1) Run E0~E4 training matrix

```bash
python scripts/run_ablation.py --config configs/default.yaml
```

## 2) Evaluate PT-based checkpoints and aggregate

```bash
python scripts/eval_ablation.py --results_root experiments
python scripts/summarize_ablation.py --results_dir experiments
```

## 3) Output files

- `experiments/ablation_matrix.json`: experiment matrix (E0~E4 x seeds)
- `experiments/ablation_summary.json`: aggregated metrics
- `experiments/ablation_summary.md`: concise report for E0->E4
- `experiments/ablation_report.md`: markdown report scaffold

## 4) Interpretation rules

- E0->E1 isolates early-fusion gain.
- E1->E2 isolates PT backbone gain.
- E2->E3 isolates FiLM global gain.
- E3->E4 compares deep FiLM variant.
- Mark gain as non-significant if relative improvement < 1% and std overlaps.
