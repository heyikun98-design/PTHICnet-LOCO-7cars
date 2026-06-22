# Phase 5 Runbook — 新全量数据训练与评估

> 在集群上执行前完整阅读。每一步有明确的"通过/不通过"标准。

---

## 0. 前置准备

### 0.1 环境确认

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 预期: 2.x, True

pip list | grep -E "numpy|pandas|pyarrow|wandb"
```

### 0.2 代码同步

从 GitHub 拉最新代码（包含 Phase 4 所有产出）：

```bash
git clone https://github.com/heyikun98-design/PTHICnet-LOCO-7cars.git
# 或 git pull origin main
```

### 0.3 新数据挂载

```bash
# 新车型数据放入 车模型数据/ 目录，命名沿用 car{N}/ 格式
ls 车模型数据/
# 预期: car1 car2 ... car{N}  (N >= 8)
```

### 0.4 注册新车型

编辑 `feather/data_utils/HICLoader_feather.py`，在 `CAR_TO_VEHICLE` 字典中添加新车型映射：

```python
CAR_TO_VEHICLE = {
    "car1": "C201",
    "car2": "EP32",
    "car3": "JX65",
    "car4": "CY02C",
    "car5": "M6",
    "car6": "S50EVK",
    "car7": "FX11",
    # ↓ 新增车型（示例）
    "car8": "NEW_CAR_A",
    "car9": "NEW_CAR_B",
    # ...
}
```

---

## 1. 新数据 QA（训练前必做）

### 1.1 运行 QA 脚本

```bash
python scripts/qa_new_data.py \
  --data_dir 车模型数据 \
  --old_normal_union C201,EP32,JX65,S50EVK,FX11 \
  --output_dir experiments/phase5_qa
```

### 1.2 QA 检查清单

对照输出逐项确认：

| 检查项 | 通过标准 | 不通过时 |
|------|------|------|
| 车型数量 | 明确记录新旧车型总数 | 核对 car 目录是否遗漏 |
| 样本量 | 每车 >= 50 samples（最低）；记录分布 | 样本过少的车标记为 hard car 候选 |
| Bbox 尺度 | 每车 union diag 与 old-normal mean 比值在 0.8–1.2x | 超出范围的记录，不排除但需注意 |
| HIC 分布 | 无 HIC=0 样本（全部有效）；记录 >2k 数量 | HIC=0 样本需排除或溯源 |
| 材料向量覆盖 | 新 normal car 的 material vector overlap vs old 5-normal union >= 80% | 低于阈值的标记，Phase 5 训练时开启 material_dropout |
| ID 级别材料 | 记录 hard-only material 向量（仅在 hard car 中出现、normal car 未见的向量） | 作为 Phase 5 诊断参考 |
| 归一化参数 | 确认 `normalization_params.pkl` 基于新全量数据重新计算 | 不要用旧 7 车参数 |

### 1.3 车型分组

根据 QA 结果，将新车型分为 normal / hard 两组：

- **Normal car**: bbox 尺度正常、材料覆盖 >= 80%、HIC 分布无极端值
- **Hard car 候选**: 材料覆盖 < 80%、HIC 极端值、或 bbox 尺度偏离 > 20%

记录分组结果，Phase 5 实验结果按组报告。

---

## 2. 训练

### 2.1 总览

| 优先级 | 架构 | Seeds | 配置 | GPU 估算 |
|------|------|------|------|:---:|
| P0 | E3 (PT+FiLM global) | 42, 3407, 2026 | `--material_dropout_prob 0.15` | N folds × 3 |
| P0 | E2 (PT no FiLM) | 42, 3407, 2026 | `--material_dropout_prob 0.15` | N folds × 3 |
| P0 | E0 (PN++ baseline) | 42, 3407, 2026 | 默认 | N folds × 3 |
| P1 | E3 (ablation) | 42 | `--material_dropout_prob 0.0` | N folds × 1 |
| P2 | E1, E4 | 42 | 默认 | N folds × 2 |

**N = 新全量车型数。** 如果 N=12，P0 总计 12×3×3 = 108 折。

### 2.2 统一参数

所有实验共用以下参数（不做单折调参）：

```bash
--config configs/default.yaml
--val_split 0.15
--split_seed 2026
--patience 50
--restore_best
```

### 2.3 LOCO-CV 批量执行

在集群上写一个 shell 脚本，遍历所有车型作为 test fold：

```bash
#!/bin/bash
# run_phase5_loco.sh — 集群执行脚本

VEHICLES=(C201 EP32 JX65 CY02C M6 S50EVK FX11 NEW_CAR_A NEW_CAR_B)  # 填入全部车型
SEEDS=(42 3407 2026)
CONDA_ENV=/path/to/your/conda/env
PROJ_DIR=/path/to/PT-HICNET

for ARCH in E0 E2 E3; do
  # 确定 arch 特定参数
  if [ "$ARCH" = "E0" ]; then
    FILM="none"
    EXTRA=""
    SCRIPT="feather/train_reg_att_props_X70_feather.py"  # 或对应的 baseline 脚本
  elif [ "$ARCH" = "E2" ]; then
    FILM="none"
    EXTRA="--material_dropout_prob 0.15"
    SCRIPT="scripts/train_pt_hicnet.py"
  elif [ "$ARCH" = "E3" ]; then
    FILM="global"
    EXTRA="--material_dropout_prob 0.15"
    SCRIPT="scripts/train_pt_hicnet.py"
  fi

  for SEED in "${SEEDS[@]}"; do
    for VEHICLE in "${VEHICLES[@]}"; do
      echo "=== $ARCH | seed=$SEED | test=$VEHICLE ==="
      $CONDA_ENV/bin/python -u $SCRIPT \
        --config configs/default.yaml \
        --seed $SEED \
        --film_mode $FILM \
        --test_vehicles $VEHICLE \
        --val_split 0.15 \
        --split_seed 2026 \
        --patience 50 \
        --restore_best \
        $EXTRA \
        2>&1 | tee experiments/phase5_logs/${ARCH}_seed${SEED}_${VEHICLE}.log
    done
  done
done
```

**注意 E0/E1 使用 baseline 训练脚本**（`feather/train_reg_att_props_X70_feather.py`），需要确认其 CLI 参数兼容。如果 baseline 脚本不支持 `--test_vehicles` 等参数，单独写一个 E0/E1 的 loop。

### 2.4 集群执行策略

- **一个 job array 一个 fold**：把上面的双重循环拆成 job array，每个 array task 跑一个 (arch, seed, vehicle) 组合
- **单折 GPU 需求**：1 张 GPU（训练 ~1–3 小时/折，取决于数据量）
- **并行度**：按可用 GPU 数量决定同时跑多少折
- **失败重试**：每个 job 的 log 独立保存，失败的 fold 单独重跑

### 2.5 验证折（smoke test）

在跑全量 LOCO-CV 之前，先用单折验证训练链路：

```bash
# 选一辆熟悉的车（如 JX65）做 smoke test
python scripts/train_pt_hicnet.py \
  --config configs/default.yaml \
  --seed 42 \
  --film_mode global \
  --test_vehicles JX65 \
  --val_split 0.15 \
  --patience 50 \
  --restore_best \
  --material_dropout_prob 0.15

# 确认:
# 1. loss 正常下降
# 2. val_acc 在 0.75–0.90 区间
# 3. history.json 包含完整的 val_accuracy / test_accuracy 字段
# 4. best_acc_model.pth 正常保存
```

Smoke test 通过后再启动全量 LOCO-CV。

---

## 3. 结果聚合

所有 fold 跑完后，用 `aggregate_loco.py` 出表：

```bash
python scripts/aggregate_loco.py \
  --architectures E0 E2 E3 \
  --results_root experiments

# 多 seed 聚合（如有需要）
python scripts/aggregate_loco.py \
  --architectures E0 E2 E3 \
  --results_root experiments \
  --multi_seed
```

输出文件保存在 `experiments/loco_cv/`。

---

## 4. 验收（对照 eval_protocol.md）

### 4.1 硬门槛

拉出 aggregate 表后，逐项对表：

| 指标 | 门槛 | 当前 Phase 3 基线 |
|------|:---:|:---:|
| E2−E0 on 5-Normal Test Mean | >= +5.0pp | +6.42pp |
| E2−E0 on 全车型 Test Mean | >= +3.0pp | +3.63pp |
| E3−E2 on 5-Normal Test Mean | >= +1.5pp 且 >= 4/5 同向 | +0.76pp (不显著) |
| Hard-car Gap (E3) | <= 20pp 或 相比 Phase 3 ↓ >= 5pp | CY02C 26.3pp, M6 24.2pp |

### 4.2 软检查

| 检查项 | 通过标准 |
|------|------|
| 7-car Test Std | <= 10.0pp |
| Normal-car Test Std | ~2pp 附近 |
| Val-Test Gap | 每车记录，无明显恶化 |

---

## 5. 带回来的东西

从集群上下载以下文件到本地：

```
experiments/phase5_qa/              # QA 报告
experiments/phase5_logs/            # 训练日志（出问题时排查用）
experiments/pt_hicnet_loco_*/       # 每个 fold 的 history.json + best_acc_model.pth
experiments/loco_cv/                # aggregate 输出
```

### 最小必要集合（如果带宽有限）

1. **所有 `history.json`**（每个 fold 一个）：用于聚合和验证
2. **aggregate 输出 CSV/表**：直接对照验收门槛
3. **QA 报告**：新数据的材料/尺度/HIC 分布

---

## 6. 回来后的分析

Phase 5 结果回来后的分析清单：

1. **过验收门槛**：对照 eval_protocol.md 5 条硬门槛逐项判定
2. **Hard car 表现**：E3+matdrop vs E0 在每个 hard car 上的 per-car delta
3. **材料覆盖诊断（扩版 4.4a）**：在新全量数据上重跑 material overlap + delta 相关分析
4. **更新 frozen baseline**：eval_protocol.md 的 Frozen Baseline 表更新为新全量数据版
5. **论文叙事更新**：PT backbone 贡献、FiLM 贡献、material regularization 效果

---

## 附录 A：紧急排查

### 训练不收敛
- 降低 lr 到 0.0005
- 检查新数据归一化参数是否正确

### OOM
- 减小 batch_size 到 8 或 4
- 检查是否误将全部数据加载到单张卡

### history.json 字段缺失
- 确认训练脚本版本：`git log --oneline -1`
- Phase 4 收口 commit: `8713cc7`

### 某折 crash
- 查看该折独立 log: `experiments/phase5_logs/{arch}_seed{seed}_{vehicle}.log`
- 常见原因：该车数据格式不一致、样本量过少导致 val_split 为空
