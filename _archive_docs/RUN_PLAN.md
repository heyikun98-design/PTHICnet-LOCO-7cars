# ⛔ DEPRECATED (2026-06-04) — 消融实验执行计划

> 本文档已归档。原内容为 Phase 2 消融实验（JX65 单车 × 5 架构 × 3 seeds）执行计划。
> 当前执行计划请参见项目根目录 `plan.md`。
> 项目上下文与实验结论请参见 `context_summary.md`。

---

# 消融实验执行计划（历史）

## 矩阵总览

| ID | 实验 | 骨干网络 | 早期融合 | FiLM | 训练脚本 | Seeds |
|----|------|---------|---------|------|---------|-------|
| E0 | baseline | PointNet++ + CrossAttn | 否 | N/A | `train_reg_att_props_X70_feather.py` | 42, 3407, 2026 |
| E1 | early_fusion_clean | PointNet++ (无 CrossAttn) | 是 | N/A | `train_reg_att_props_X70_feather.py` | 42, 3407, 2026 |
| E2 | pt_backbone | Point Transformer | 是 | none | `train_pt_hicnet.py` | 42, 3407, 2026 |
| E3 | film_global | Point Transformer | 是 | global | `train_pt_hicnet.py` | 42, 3407, 2026 |
| E4 | film_deep | Point Transformer | 是 | deep | `train_pt_hicnet.py` | 42, 3407, 2026 |

**总计**: 5 × 3 = 15 组训练，再加 1 次汇总评估

**已完成**: E2 seed42 ✅ | E3 seed42 ✅

---

## Step 1: 跑完剩余训练

### E0 (baseline) — 3 seeds

```bash
python feather/train_reg_att_props_X70_feather.py --config configs/default.yaml --ablation_mode baseline                          --seed 42
python feather/train_reg_att_props_X70_feather.py --config configs/default.yaml --ablation_mode baseline                          --seed 3407
python feather/train_reg_att_props_X70_feather.py --config configs/default.yaml --ablation_mode baseline                          --seed 2026
```

### E1 (early_fusion_clean) — 3 seeds

```bash
python feather/train_reg_att_props_X70_feather.py --config configs/default.yaml --ablation_mode early_fusion_clean --use_early_fusion --normalize_thickness --seed 42
python feather/train_reg_att_props_X70_feather.py --config configs/default.yaml --ablation_mode early_fusion_clean --use_early_fusion --normalize_thickness --seed 3407
python feather/train_reg_att_props_X70_feather.py --config configs/default.yaml --ablation_mode early_fusion_clean --use_early_fusion --normalize_thickness --seed 2026
```

### E2 (pt_backbone) — 剩余 2 seeds

```bash
python scripts/train_pt_hicnet.py --config configs/default.yaml --film_mode none --seed 3407
python scripts/train_pt_hicnet.py --config configs/default.yaml --film_mode none --seed 2026
```

### E3 (film_global) — 剩余 2 seeds

```bash
python scripts/train_pt_hicnet.py --config configs/default.yaml --film_mode global --seed 3407
python scripts/train_pt_hicnet.py --config configs/default.yaml --film_mode global --seed 2026
```

### E4 (film_deep) — 3 seeds

```bash
python scripts/train_pt_hicnet.py --config configs/default.yaml --film_mode deep --seed 42
python scripts/train_pt_hicnet.py --config configs/default.yaml --film_mode deep --seed 3407
python scripts/train_pt_hicnet.py --config configs/default.yaml --film_mode deep --seed 2026
```

> 每条命令末尾加 `--use_wandb` 可开启 wandb 可视化

---

## Step 2: 汇总评估

所有 15 组训练完成后：

```bash
python scripts/eval_ablation.py --results_root experiments --output_json experiments/ablation_summary.json --output_md experiments/ablation_summary.md
```

## Step 3: 查看结果

输出文件：

| 文件 | 内容 |
|------|------|
| `experiments/ablation_summary.json` | 各实验 MSE / Accuracy 均值±标准差 |
| `experiments/ablation_summary.md` | Markdown 表格，可直接查看 |
| `experiments/eval_<run_name>.json` | 每组的逐车评估明细 |

重点对比：
- **E0 → E1**: 早期融合的独立增益（消去 CrossAttention）
- **E1 → E2**: Point Transformer 替换 PointNet++ 的增益
- **E2 → E3**: FiLM 全局注入的增益
- **E3 → E4**: FiLM 深层注入 vs 全局注入

---

## 快速跑法（一键）

如果不想手动逐条执行，用编排脚本（会自动跳过已有 checkpoint 的实验）：

```bash
python scripts/run_ablation.py --config configs/default.yaml
```

> 注意：`run_ablation.py` 目前不支持 `--use_wandb` 透传。需要 wandb 的话，把 `configs/default.yaml` 里 `use_wandb: true` 改掉再跑。
