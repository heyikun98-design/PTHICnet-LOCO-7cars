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

从 GitHub 拉最新代码（包含 Phase 4 收口 + Phase 5 prep）：

```bash
git clone https://github.com/heyikun98-design/PTHICnet-LOCO-7cars.git
# 或 cd PT-HICNET && git pull origin main
```

### 0.3 新数据挂载

```bash
# 数据不进入 Git。将集群已有数据挂载路径写入集群专用配置。
cp configs/default.yaml configs/phase5_cluster.yaml
# 编辑 data.data_root、material_lookup_path、normalization_params_path
# 三个字段都可以使用集群绝对路径。

# 新车型目录命名沿用 car{N}/ 格式
ls /CLUSTER/DATA/PATH/
# 预期: car1 car2 ... car{N}  (N >= 8)
```

不要上传 `.feather`、checkpoint 或原始材料数据库到 GitHub。Phase 5 命令统一使用
`--config configs/phase5_cluster.yaml`。集群网络不可用时统一传 `--no_wandb`。

### 0.4 注册新车型 + 检查材料 lookup

**Step 1**: 编辑 `feather/data_utils/HICLoader_feather.py`，在 `CAR_TO_VEHICLE` 字典中添加新车型映射：

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

**Step 2**: 确认新车在材料 lookup 表中有对应条目。`material_lookup_by_vehicle.pkl` 按 vehicle code 索引，新车缺失会静默用零材料矩阵继续训练——这是最隐蔽的污染。

```bash
python -c "
import pickle
with open('/CLUSTER/METADATA/material_lookup_by_vehicle.pkl', 'rb') as f:
    lookup = pickle.load(f)
print('Vehicles in lookup:', sorted(lookup.keys()))
expected = ['C201','EP32','JX65','CY02C','M6','S50EVK','FX11',
            'NEW_CAR_A','NEW_CAR_B']
missing = sorted(set(expected) - set(lookup))
print('Missing vehicles:', missing)
raise SystemExit(1 if missing else 0)
"
```

**如果新车不在 lookup 中**：用原始材料数据库重新生成 lookup，或至少用最相似的已有车型的材料表作为 fallback。不要用零矩阵。

---

## 1. 新数据 QA（训练前必做）

### 1.1 运行 QA 脚本

```bash
python scripts/qa_new_data.py \
  --data_dir /CLUSTER/DATA/PATH \
  --old_normal_vehicles C201,EP32,JX65,S50EVK,FX11 \
  --material_lookup_path /CLUSTER/METADATA/material_lookup_by_vehicle.pkl \
  --normalization_params_path /CLUSTER/METADATA/normalization_params.pkl \
  --output_dir experiments/phase5_qa
```

命令必须返回 `exit=0`，且报告末尾必须为 `Critical: 0`。任一车型缺 lookup、
存在未解析 MID 或材料向量数为 0 时，不得进入训练。

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

### 2.0 Cluster Preflight（跑全量之前必做）

在开 100+ folds 之前，先跑一个最小闭环验证链路。

```bash
# 选 1 辆 old normal + 1 辆 hard/new（共 2 folds）
PREFLIGHT_VEHICLES=(JX65 CY02C)

# E2
python -u scripts/train_pt_hicnet.py \
  --config configs/phase5_cluster.yaml \
  --seed 42 \
  --exp_name phase5_p0/pt_hicnet_loco_e2 \
  --film_mode none --test_vehicles JX65 \
  --val_split 0.15 --split_seed 2026 --patience 50 --restore_best \
  --material_dropout_prob 0.15 --no_wandb

# E3
python -u scripts/train_pt_hicnet.py \
  --config configs/phase5_cluster.yaml \
  --seed 42 \
  --exp_name phase5_p0/pt_hicnet_loco_e3 \
  --film_mode global --test_vehicles JX65 \
  --val_split 0.15 --split_seed 2026 --patience 50 --restore_best \
  --material_dropout_prob 0.15 --no_wandb

# ... 再跑 CY02C 的 E2/E3，加 E0
```

**Preflight 通过的 6 个条件**：

1. 每个 (arch, vehicle) 组合落在 `experiments/phase5_p0/` 下的独立子目录，**不互相覆盖**
2. 目录名含 vehicle + seed + film_mode + `_md0.15`（验证代码自动追加正确）
3. 每个 `history.json` 有完整的 `val_accuracy` / `test_accuracy` 字段
4. `aggregate_loco.py --results_root experiments/phase5_p0` 能读到这些 fold 并产出 per-vehicle 表
5. 旧 Phase 3 目录、单折 pilot（`pt_hicnet_loco_e3_matdrop015_*`）**不出现在主表中**（验证隔离有效）
6. E3+matdrop test acc 不出明显异常（vs Phase 4 frozen baseline: JX65 ~78%, CY02C ~62%）

**只有这个闭环全绿了，才开 2.5 的批量 LOCO-CV。**

### 2.1 总览

| 优先级 | 架构 | Seeds | 配置 | GPU 估算 |
|------|------|------|------|:---:|
| P0 | E3 (PT+FiLM global) | 42, 3407, 2026 | `--material_dropout_prob 0.15` | N folds × 3 |
| P0 | E2 (PT no FiLM) | 42, 3407, 2026 | `--material_dropout_prob 0.15` | N folds × 3 |
| P0 | E0 (PN++ baseline) | 42, 3407, 2026 | 默认 | N folds × 3 |
| P1 | E3 (ablation) | 42 | `--material_dropout_prob 0.0` | N folds × 1 |
| P2 | E2 (ablation) | 42 | `--material_dropout_prob 0.0` | N folds × 1 |
| P3 | E1, E4 | 42 | 默认 | N folds × 2 |

**N = 新全量车型数。** 如果 N=12，P0 总计 12×3×3 = 108 折。

### 2.2 统一参数

PT 脚本实验共用以下参数（不做单折调参）：

```bash
--config configs/phase5_cluster.yaml
--val_split 0.15
--split_seed 2026
--patience 50
--restore_best
```

E0 baseline 脚本不支持 `--restore_best`，但支持
`--test_vehicles` / `--val_split` / `--split_seed` / `--patience`，并在
early stopping 时恢复 `best_acc_model.pth`。

### 2.3 目录隔离规则（防止结果污染）

**P0 和 P1/P2 使用不同的 experiments 子目录，物理隔离。**

```bash
experiments/
├── phase5_p0/          # P0 主实验（aggregate 只扫这个）
├── phase5_ablation/    # P1/P2 no-dropout 对照（单独分析）
├── phase5_logs/        # 训练日志
├── phase5_qa/          # QA 报告
└── ...                 # 旧 Phase 3 目录（aggregate 不再扫描）
```

**聚合命令**：

```bash
python scripts/aggregate_loco.py \
  --architectures E0 E2 E3 \
  --results_root experiments/phase5_p0 \
  --vehicles JX65 CY02C \
  --normal_cars JX65 \
  --hard_cars CY02C \
  --expected_seeds 42 \
  --strict
```

这确保旧 Phase 3 目录、单折 pilot（`pt_hicnet_loco_e3_matdrop015_*`）、P1/P2 ablation 都不会混进 P0 主表。

### 2.4 命名规则

`train_pt_hicnet.py` 的 `run_name` **自动追加** `vehicle_tag`（CLI 显式传 `--test_vehicles` 时）和 `_md{prob}`（`material_dropout_prob > 0` 时）。因此 `--exp_name` **不需要**包含 vehicle/seed——代码会补。

E0 baseline 脚本**不会**自动追加，所以 E0 的 `--exp_name` 必须显式包含 vehicle/seed。

| 优先级 | 架构 | `--exp_name` | 代码自动追加 | 最终目录 |
|------|------|------|------|------|
| P0 | E0 | `phase5_p0/pt_hicnet_loco_e0_C201_seed42` | — | `phase5_p0/pt_hicnet_loco_e0_C201_seed42` |
| P0 | E2 | `phase5_p0/pt_hicnet_loco_e2` | `_C201_seed42_film-none_md0.15` | `phase5_p0/pt_hicnet_loco_e2_C201_seed42_film-none_md0.15` |
| P0 | E3 | `phase5_p0/pt_hicnet_loco_e3` | `_C201_seed42_film-global_md0.15` | `phase5_p0/pt_hicnet_loco_e3_C201_seed42_film-global_md0.15` |
| P1 | E3 | `phase5_ablation/pt_hicnet_loco_e3` | `_C201_seed42_film-global` | `phase5_ablation/pt_hicnet_loco_e3_C201_seed42_film-global` |
| P2 | E2 | `phase5_ablation/pt_hicnet_loco_e2` | `_C201_seed42_film-none` | `phase5_ablation/pt_hicnet_loco_e2_C201_seed42_film-none` |

**关键规则**：
- P0 的 `--exp_name` 以 `phase5_p0/` 开头 → 落进独立子目录
- PT 脚本 (E2/E3) 的 `--exp_name` 不含 vehicle/seed → 代码自动补
- E0 baseline 的 `--exp_name` 必须含 vehicle/seed → baseline 不自动补
- P1/P2 的 `--exp_name` 以 `phase5_ablation/` 开头 → 不匹配 aggregate 主表

### 2.5 LOCO-CV 批量执行

**E0 loop（baseline 脚本）**：

⚠️ E0 用 baseline 脚本，CLI 参数名完全不同。**先跑 `--help` 确认参数，再写 loop。**

```bash
python feather/train_reg_att_props_X70_feather.py --help
# 确认: --seed, --exp_name, --test_vehicles, --val_split, --split_seed, --patience
```

E0 baseline 的 `--exp_name` 直接决定输出目录 (`experiments/<exp_name>/`)，且**不会**自动追加 vehicle/seed。必须显式写全。

```bash
#!/bin/bash
VEHICLES=(C201 EP32 JX65 CY02C M6 S50EVK FX11 NEW_CAR_A NEW_CAR_B)
SEEDS=(42 3407 2026)

for SEED in "${SEEDS[@]}"; do
  for V in "${VEHICLES[@]}"; do
    EXP_NAME="phase5_p0/pt_hicnet_loco_e0_${V}_seed${SEED}"
    echo "=== E0 test=$V seed=$SEED ==="
    python -u feather/train_reg_att_props_X70_feather.py \
      --config configs/phase5_cluster.yaml \
      --seed $SEED \
      --exp_name $EXP_NAME \
      --test_vehicles $V \
      --val_split 0.15 \
      --split_seed 2026 \
      --patience 50 \
      --batch_size 15 \
      --no_wandb \
      2>&1 | tee experiments/phase5_logs/E0_seed${SEED}_${V}.log
  done
done
```

⚠️ E0 baseline 脚本**没有** `--film_mode`、`--restore_best`，不要传这两个参数。
但必须传 `--test_vehicles $V`，否则不同 `$V` 只会改变目录名，实际 held-out
vehicle 仍可能走 config 默认值。

**E2 + E3 P0 loop（PT 脚本，material_dropout_prob=0.15）**:

PT 脚本自动追加 vehicle/seed/dropout → `--exp_name` 不需要含 `${V}_seed${SEED}`。

```bash
#!/bin/bash
VEHICLES=(C201 EP32 JX65 CY02C M6 S50EVK FX11 NEW_CAR_A NEW_CAR_B)
SEEDS=(42 3407 2026)

for ARCH in E2 E3; do
  if [ "$ARCH" = "E2" ]; then FILM="none"; else FILM="global"; fi
  EXP_PREFIX="phase5_p0/pt_hicnet_loco_${ARCH,,}"  # e2 or e3
  for SEED in "${SEEDS[@]}"; do
    for V in "${VEHICLES[@]}"; do
      python -u scripts/train_pt_hicnet.py \
        --config configs/phase5_cluster.yaml \
        --seed $SEED \
        --exp_name $EXP_PREFIX \
        --film_mode $FILM \
        --test_vehicles $V \
        --val_split 0.15 --split_seed 2026 \
        --patience 50 --restore_best \
        --material_dropout_prob 0.15 \
        --no_wandb \
        2>&1 | tee experiments/phase5_logs/${ARCH}_seed${SEED}_${V}.log
    done
  done
done
```

实际目录示例：`experiments/phase5_p0/pt_hicnet_loco_e3_C201_seed42_film-global_md0.15`

**P1 对照：E3 + material_dropout_prob=0.0（seed=42 only）**:

```bash
for V in "${VEHICLES[@]}"; do
  python -u scripts/train_pt_hicnet.py \
    --config configs/phase5_cluster.yaml \
    --seed 42 \
    --exp_name phase5_ablation/pt_hicnet_loco_e3 \
    --film_mode global --test_vehicles $V \
    --val_split 0.15 --split_seed 2026 --patience 50 --restore_best \
    --material_dropout_prob 0.0 \
    --no_wandb \
    2>&1 | tee experiments/phase5_logs/E3_nodropout_seed42_${V}.log
done
```

**P2 建议：E2 + material_dropout_prob=0.0（seed=42 only）**:

```bash
for V in "${VEHICLES[@]}"; do
  python -u scripts/train_pt_hicnet.py \
    --config configs/phase5_cluster.yaml \
    --seed 42 \
    --exp_name phase5_ablation/pt_hicnet_loco_e2 \
    --film_mode none --test_vehicles $V \
    --val_split 0.15 --split_seed 2026 --patience 50 --restore_best \
    --material_dropout_prob 0.0 \
    --no_wandb \
    2>&1 | tee experiments/phase5_logs/E2_nodropout_seed42_${V}.log
done
```

**P2 的意义**：E2−E0 的提升混有 "PT backbone" 和 "material regularization" 两个因素。P2 对照可将二者分离。

**P3 可选**：E1/E4 只在 GPU 余量足够或需要补充消融时运行，不进入 P0 主表。

### 2.6 集群执行策略

- **一个 job array 一个 fold**：把上面的双重循环拆成 job array，每个 array task 跑一个 (arch, seed, vehicle) 组合
- **单折 GPU 需求**：1 张 GPU（训练 ~1–3 小时/折，取决于数据量）
- **并行度**：按可用 GPU 数量决定同时跑多少折
- **失败重试**：每个 job 的 log 独立保存，失败的 fold 单独重跑

### 2.7 验证折（smoke test）

在跑全量 LOCO-CV 之前，先用单折验证训练链路：

```bash
# 选一辆熟悉的车（如 JX65）做 smoke test
python scripts/train_pt_hicnet.py \
  --config configs/phase5_cluster.yaml \
  --seed 42 \
  --film_mode global \
  --test_vehicles JX65 \
  --val_split 0.15 \
  --patience 50 \
  --restore_best \
  --material_dropout_prob 0.15 \
  --no_wandb

# 确认:
# 1. loss 正常下降
# 2. val_acc 在 0.75–0.90 区间
# 3. history.json 包含完整的 val_accuracy / test_accuracy 字段
# 4. best_acc_model.pth 正常保存
```

Smoke test 通过后再启动全量 LOCO-CV。

---

## 3. 结果聚合

所有 fold 跑完后，只扫 P0 子目录：

```bash
python scripts/aggregate_loco.py \
  --architectures E0 E2 E3 \
  --results_root experiments/phase5_p0 \
  --vehicles C201 EP32 JX65 CY02C M6 S50EVK FX11 NEW_CAR_A NEW_CAR_B \
  --normal_cars C201 EP32 JX65 S50EVK FX11 NEW_NORMAL_CARS \
  --hard_cars CY02C M6 NEW_HARD_CARS \
  --expected_seeds 42 3407 2026 \
  --strict
```

车型和 normal/hard 分组通过 CLI 传入，不再需要改聚合脚本。`--strict` 会在任一
车型缺少任一 seed、或同一 `(vehicle, seed)` 出现重复结果时返回非零退出码。

P1/P2 ablation 需要单独聚合（可选，不同 `--results_root`）：

```bash
python scripts/aggregate_loco.py \
  --architectures E2 E3 \
  --results_root experiments/phase5_ablation \
  --vehicles "${VEHICLES[@]}" \
  --normal_cars "${NORMAL_CARS[@]}" \
  --hard_cars "${HARD_CARS[@]}" \
  --expected_seeds 42 \
  --strict
```

**注意 `aggregate_loco.py` 目前只在 stdout 打印，不写文件。** 用 `tee` 保存：

```bash
VEHICLES=(C201 EP32 JX65 CY02C M6 S50EVK FX11 NEW_CAR_A NEW_CAR_B)
NORMAL_CARS=(C201 EP32 JX65 S50EVK FX11 NEW_NORMAL_CARS)
HARD_CARS=(CY02C M6 NEW_HARD_CARS)

python scripts/aggregate_loco.py --architectures E0 E2 E3 \
  --results_root experiments/phase5_p0 \
  --vehicles "${VEHICLES[@]}" \
  --normal_cars "${NORMAL_CARS[@]}" \
  --hard_cars "${HARD_CARS[@]}" \
  --expected_seeds 42 3407 2026 --strict \
  2>&1 | tee experiments/phase5_aggregate_p0.txt
```

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
experiments/phase5_p0/              # P0 每个 fold 的 history.json + best_acc_model.pth
experiments/phase5_ablation/        # P1/P2 对照（如已运行）
experiments/phase5_aggregate_p0.txt # aggregate 主表 stdout 保存
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
