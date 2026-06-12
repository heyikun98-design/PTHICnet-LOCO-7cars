# ⛔ DEPRECATED (2026-06-04) — PT-HICnet 可执行重构蓝图

> 本文档已归档。原内容为 2026-05-29 的架构蓝图（Phase 0-5 任务分解）与实验进度表。
> 实验进度表已严重过期（E0/E1 标记 ❌ 但实际已完成 15-seed + 7-fold LOCO-CV）。
> 当前状态请参见 `plan.md`（实施计划）和 `context_summary.md`（上下文与结论）。
> 附录中的关键设计决策记录已同步至 `context_summary.md` 第 10 节。

---

# PT-HICnet 可执行重构蓝图（历史）

> **架构目标：** 物理引导的 Point Transformer HICnet
> **核心原则：** (1) 几何-本构早融合 (Early Injection) (2) Vector Attention 捕捉局部刚度突变 (3) 全局动能载荷（age_group）作为条件偏置 (FiLM)
>
> **状态更新 (2026-05-29)：** Phase 0-4 代码全部完成，E3 (film_global) 完成首轮 200 epoch 训练。实验暴露了 Ball Query 半径失配导致的严重 Train-Test 背离。详见 [BLOCKER_TODO.md](./BLOCKER_TODO.md) N1-N3。

---

## Phase 0 — 项目骨架搭建  [状态: ✅ 已完成]

- [x] **0.1** 创建新目录结构：`models/`、`data/`、`scripts/`、`configs/` ✅
- [x] **0.2** 创建 `configs/default.yaml`，统一管理超参 ✅
- [x] **0.3** 将绝对路径全部改为配置驱动或相对路径 ✅
- [x] **0.4** 迁移 `material_lookup_by_vehicle.pkl` 和 `normalization_params.pkl` 到 `data/` 目录 ✅（通过环境变量 + YAML 配置贯通）

---

## Phase 1 — 数据层：实现几何-本构早融合  [状态: ✅ 已完成]

### 1.1 DataLoader 输出重构

- [x] **1.1.1** 修改 `feather/data_utils/HICLoader_feather.py` 的 `_get_item()`：FPS 采样后将 `point_set [N,3]`、`thickness [N,1]`、`material_props [N,15]` 沿 `dim=1` 拼接为 `[N, 19]` 的 `fused_input` ✅
- [x] **1.1.2** 同步修改 `feather/data_utils/HICLoader.py`（旧版 JSON Loader）接口一致 ✅
- [x] **1.1.3** `HICDataLoader.__init__()` 新增 `early_fusion: bool = True` 参数 ✅
- [x] **1.1.4** `_get_item()` 中添加 `thickness` per-sample min-max 归一化 ✅
- [x] **1.1.5** 验证：fused_input.shape == (8192, 19)，age_group ∈ {0, 1} ✅（SmokeTest 通过）

### 1.2 训练/测试脚本适配

- [x] **1.2.1** `scripts/train_pt_hicnet.py`：使用新的 5 元组解包格式 `(fused_input, hic_point, category, age_group, target)` ✅
- [ ] **1.2.2** `feather/train_reg_att_props_X70_feather.py`：待适配（E0/E1 baseline 训练时需确认接口兼容）
- [ ] **1.2.3** 旧版 test/predict 脚本：保留向后兼容，暂不强制修改

---

## Phase 2 — Point Transformer 骨干网络替换 PointNet++  [状态: ✅ 代码已完成]

### 2.1 PT 核心算子实现

- [x] **2.1.1** `models/point_transformer_utils.py`：FPS / Ball Query / Index Points / Square Distance ✅
- [x] **2.1.2** `models/point_transformer_block.py`：PTv1 Vector Attention 层（q-k+delta_pos 注意力） ✅
- [x] **2.1.3** `models/point_transformer_layer.py`：Transition Down（FPS → Ball Query → PT Block → Pool → Proj） ✅

### 2.2 PT-HICnet 模型定义

- [x] **2.2.1** `models/pt_hicnet.py`：完整模型（input_proj → 4×TransitionDown → global feature → regressor） ✅
- [x] **2.2.2** 4 层 PT Block + Transition Down：8192→512→128→32→1，通道 64→128→256→512→1024 ✅
- [x] **2.2.3** 全局特征 `x_global [B, 1024]` 作为点云几何-物理融合表征 ✅

### 2.3 几何通道处理

- [x] **2.3.1** `use_normals` 逻辑：19ch（无 normals）vs 22ch（有 normals），forward 中 assert 验证 ✅
- [ ] **2.3.2** DataLoader 中法向拼接（若数据包含法向字段）：暂无需求，标记为将来扩展

---

## Phase 3 — 全局动能载荷注入：FiLM 条件偏置  [状态: ✅ 代码已完成]

### 3.1 FiLM 层实现

- [x] **3.1.1** `models/film.py`：`FiLM(h, c) = γ(c) * h + β(c)`，cond_dim=16, hidden_dim=1024 ✅
- [x] **3.1.2** 注入方案已实现：
  - **方案 A（全局注入 — E3 已测试）**：在 `x_global` 后用 FiLM 调制，然后送入回归头 ✅
  - **方案 B（深层注入 — E4 待测试）**：在每个 TransitionDown 后做 FiLM ✅

### 3.2 HIC 回归头

- [x] **3.2.1** 回归头：`hic_point_fc(3→64)` + `age_embedding(2→16)` + FiLM + `cat([x_global, hic_feat])` → MLP(1088→512→256→1) ✅

### 3.3 未移除旧架构（保留对照）

- [ ] **3.3.1** 旧 PointNet++ 架构 (`pointnet2_reg_att_props.py`) 保留作为 E0/E1 baseline，暂不标记 deprecated

---

## Phase 4 — 训练管线适配  [状态: ✅ 已完成]

### 4.1 Loss 与优化

- [x] **4.1.1** KDE 加权 Huber Loss → `models/losses.py` ✅
- [x] **4.1.2** `scripts/train_pt_hicnet.py`：完整训练入口，含 YAML 配置、Adam 优化、StepLR 调度、checkpoint 保存 ✅
- [x] **4.1.3** 数据增强 pipeline 补齐（B3 — random_point_dropout / scale / shift） ✅

### 4.2 配置与日志

- [x] **4.2.1** 训练脚本从 `configs/default.yaml` 读取全部参数 ✅
- [x] **4.2.2** Wandb 日志记录支持（`--use_wandb` 可选） ✅

---

## Phase 5 — 评估与验证  [状态: ⚠️ 部分完成]

- [x] **5.1** `scripts/eval_pt_hicnet.py`：加载 checkpoint + MSE/Accuracy 计算 + 逐车报告 ✅
- [x] **5.2** `scripts/eval_ablation.py` + `scripts/summarize_ablation.py`：消融汇总脚本 ✅
- [x] **5.3** `scripts/run_ablation.py`：一键编排 5×3=15 组实验 ✅

### 实验进度

| ID | 实验 | Seed 42 | Seed 3407 | Seed 2026 |
|----|------|:---:|:---:|:---:|
| E0 | baseline (PointNet++ + CrossAttn) | ❌ | ❌ | ❌ |
| E1 | early_fusion_clean (PointNet++, 无 CrossAttn) | ❌ | ❌ | ❌ |
| E2 | pt_backbone (PT + EarlyFusion, FiLM=none) | ⚠️ 中断 | ❌ | ❌ |
| E3 | film_global (PT + EarlyFusion + FiLM global) | ✅ | ❌ | ❌ |
| E4 | film_deep (PT + EarlyFusion + FiLM deep) | ❌ | ❌ | ❌ |

> **E2 seed42**：数据加载完成但训练被中断，无 history.json，需要重跑。
> **E3 seed42**：200 轮训练完成，test MSE 从 842K 恶化到 3,050K，详见下方 E3 结果摘要。

### E3 关键结果 (seed42, film_global, 200 epoch)

| 指标 | Epoch 1 | 最优 | Epoch 200 | 趋势 |
|------|:---:|:---:|:---:|:---:|
| Train Loss | 16,059 | 6,943 (ep166) | 8,508 | ↓ -47% |
| Test MSE | 842K | 608K (ep45) | 3,050K | ↑ +3.6× 恶化 |
| Test Accuracy | 0.530 | 0.711 (ep10) | 0.642 | → 无趋势 |

**结论：Train Loss 下降但 Test MSE 反而恶化，确认模型严重过拟合，PT 骨干网络的特征提取能力在实际数据上未生效。最可能的根因是 Ball Query 半径失配（见 BLOCKER_TODO.md N1）。**

---

## 附录：关键设计决策记录

| 决策 | 内容 | 力学依据 |
|------|------|---------|
| **输入通道: 19** | 3 (XYZ) + 1 (thickness) + 15 (材料属性) | 材料 15 维包含完整本构信息，厚度感知局部抗弯刚度 |
| **材料不预嵌入** | 15 维原始物理量直接作为通道，由 Conv1d 隐式编码 | 避免信息瓶颈 |
| **age_group 延迟注入** | 直到回归头才通过 FiLM 注入 | 头型年龄影响全局载荷，不参与逐点特征学习 |
| **PT Vector Attention** | `q - k` 操作编码相邻节点位移差 | 对应 FEA 应变张量的离散化近似 |
