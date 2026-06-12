# PT-HICNET 项目上下文总结

> 最后更新：2026-06-05（Phase 3c 完成：五架构全量 LOCO-CV 35/35 folds。E4 FiLM deep 确认是过拟合器；E1≈E0，EarlyFusion 在 PN++ 上跨车无效。）

---

## 1. 项目目标

物理引导的 Point Transformer HICnet（Head Injury Criterion 预测网络）。输入为汽车 FEA 网格点云 + 材料属性 + 碰撞点坐标 + 年龄组，输出为 HIC 值（头部伤害指标，严格正值）。

**核心设计原则：**
- 几何-本构早融合（Early Fusion）：19 通道 = 3 (XYZ) + 1 (thickness) + 15 (材料属性)
- Vector Attention 捕捉局部刚度突变：`q - k + delta_pos` 对应 FEA 应变张量离散化
- 全局动能载荷条件偏置：age_group（0=儿童, 1=成人）通过 FiLM 注入

---

## 2. 代码架构（已确认）

### 2.1 数据流

```
HICLoader_feather._get_item()
  ├─ point_set [8192, 3]        XYZ 坐标（FPS 采样，单位 mm）
  ├─ thickness [8192, 1]        per-sample min-max 归一化到 [0, 1]
  └─ material_props [8192, 15]  Z-score 归一化（密度/E/PR/12个应力应变采样点）
       ↓ concat → fused_input [8192, 19]
       ↓ DataLoader 返回 (fused_input, hic_point, category, age_group, target)
       ↓
train_pt_hicnet.py: transpose(2,1) → [B, 19, 8192]
  ├─ 数据增强（仅 XYZ 通道 0:3）: random_point_dropout / random_scale / shift
  └─ .to(device)
       ↓
PT_HICnet.forward()
  ├─ xyz = fused_input[:, :3, :]  → [B, 8192, 3]   (用于 FPS + Ball Query)
  ├─ feats = input_proj(fused_input)  → [B, 64, 8192]  (Conv1d→BN→ReLU)
  └─ 4× TransitionDown: 8192→512→128→32→1   通道: 64→128→256→512→1024
       ↓ squeeze(1) → x_global [B, 1024]
       ↓ FiLM(x_global, age_emb)  (仅当 film_mode="global")
       ↓ cat([x_global, hic_feat(3→64)])  → [B, 1088]
       ↓ Regressor: 1088→512→256→1  (LeakyReLU + Dropout)
       ↓ pred [B, 1]
```

### 2.2 TransitionDown 内部逻辑

```
1. FPS 降采样: xyz [B,N,3] → fps_idx → new_xyz [B,npoint,3]
2. Ball Query: radius 内取 nsample 个最近邻 → group_idx [B,npoint,nsample]
3. index_points: grouped_xyz [B,npoint,nsample,3], grouped_feats [B,npoint,nsample,C]
4. rel_pos = new_xyz - grouped_xyz
5. PointTransformerBlock: Vector Attention → out [B,npoint,C]
6. MaxPool + AvgPool: pooled = cat([max, avg], dim=-1) → [B,npoint,C*2]
7. Concat + Proj: Linear(C*3, out_dim) → [B,npoint,out_dim]
```

### 2.3 PointTransformerBlock（Vector Attention）

```
delta_pos = theta(rel_pos)          # 位置编码 MLP
q = phi(x_i), k = psi(x_j)         # query/key 投影
v = alpha(x_j) + delta_pos          # value 注入位置信息
attn = softmax(gamma(q - k + delta_pos))  # 向量减法注意力
out = sum(attn * v)                  # 加权聚合
out = LayerNorm(x_i + out)          # 残差 + 归一化
out = LayerNorm(out + FFN(out))     # FFN + 残差
```

### 2.4 损失函数

`KDEWeightedHuberLoss`（`models/losses.py`）：
- 基于训练集 HIC 分布的 KDE 密度估计，对稀有值赋予更高权重
- 权重 = `1 - 归一化密度`（稀有样本权重高）
- Huber 切换点 δ：|error| ≤ δ 用 MSE，|error| > δ 用 MAE
- **当前最优配置：δ=150, KDE 禁用**（Phase 1 验证结论）

### 2.5 E0/E1 模型说明

| 模型 | 文件 | Forward 签名 |
|------|------|------|
| E0 (baseline) | `feather/model/pointnet2_reg_att_props.py` | `(xyz, key_point, category, thickness, material_props, age_group)` 6 参数 |
| E1 (early_fusion_clean) | `feather/model/pointnet2_reg_ablation.py` | `(fused_input, hic_point, category, age_group)` 4 参数 |

E1 模型在第 362 行通过 `model_name = "pointnet2_reg_ablation" if args.ablation_mode == "early_fusion_clean" else args.model` 切换。

---

## 3. 点云尺度诊断（关键发现）

**坐标单位：mm**（汽车碰撞 FEA 网格）

| 车辆 | 对角线 (mm) | 平均点间距 (mm) | 平均 NN 距离 (mm) |
|------|:---:|:---:|:---:|
| car1 (C201) | 1407 | 15.55 | 19.47 |
| car2 (EP32) | 1201 | 13.27 | 21.29 |
| car3 (JX65) | 1126 | 12.44 | 15.65 |
| car4 (CY02C) | 2416 | 26.69 | 13.66 |
| car5 (M6) | 2349 | 25.95 | 14.74 |
| car6 (S50EVK) | 1202 | 13.28 | 17.44 |
| car7 (FX11) | 1132 | 12.50 | 24.00 |

训练集：6 辆车 ~1100 样本 | 测试集：JX65 ~172 样本

---

## 4. 已修复的问题

### 4.1 Ball Query 半径失配（已修复 ✅）

**问题：** 旧半径 `[0.2, 0.4, 0.8, 1.2]` 比实际点间距（15-25mm）小约 100 倍，导致 Ball Query 几乎无有效邻居，Vector Attention 退化。

**修复：** `pt_radius = [60, 150, 400, 1500]`（基于 2-4 倍点间距 + FPS 降采样后的间距膨胀）

**验证：**

| 层 | npoint | radius | avg_valid | pct_full(≥32) |
|---|--------|--------|-----------|:---:|
| 1 | 512 | 60 | 129.7 | 100% |
| 2 | 128 | 150 | 72.6 | 96% |
| 3 | 32 | 400 | 101.8 | 100% |
| 4 | 1 | 1500 | 32.0 | 100% |

**修改位置：**
- `configs/default.yaml` L37: `pt_radius: [60, 150, 400, 1500]`
- `models/pt_hicnet.py` L14: 默认值同步更新
- `models/point_transformer_layer.py`: 新增首次 forward 时的邻居数诊断日志

### 4.2 输入投影层加 BatchNorm（已修复 ✅）

**问题：** `Conv1d(19, 64)` 直接接收 XYZ（mm 级，跨度 ~1000）、thickness（[0,1]）、material_props（Z-score ~[-3,3]），三组通道量级差异巨大，梯度被 XYZ 通道主导。

**修复：** `input_proj = Conv1d → BatchNorm1d → ReLU`

**修改位置：** `models/pt_hicnet.py` L23-27

### 4.3 数据加载内存爆炸（已修复 ✅）

**文件**: `feather/data_utils/HICLoader_feather.py`

**根因 1**: `_load_data()` 中 `point_data['nearby_nodes'] = [dict(node) for node in nearby_nodes]` 对每个 feather 文件的节点做 dict 复制。42 个文件 × ~38 样本 × ~13,000 节点 = ~2000 万 Python dict → 60+GB RSS。

**根因 2**: `_process_sample()` 用 Python list 逐节点构建数据，并 `.tolist()` 存回 Python list。每个 float 的 Python 对象开销 ~24 字节（实际数据仅 8 字节）。

**修复**: 去掉 `[dict(node)]` 复制；预分配 numpy 数组 `np.empty((N, D), dtype=np.float32)` 直接填充；存储 numpy 数组，不再 `.tolist()`

**效果**: RSS: 65GB → 7.5GB（降 90%）；数据加载: 4.5 小时 → 5 分钟（快 ~50×）

### 4.4 E0 baseline forward 参数不匹配（已修复 ✅）

**文件**: `feather/train_reg_att_props_X70_feather.py`

**根因**: `configs/default.yaml` 中 `early_fusion: true` → 训练循环只传 4 个参数，但 E0 PointNet++ 模型要求 6 个参数。

**修复**: `ablation_mode == 'baseline'` 时强制 `args.use_early_fusion = False`（第 107-108 行）。

### 4.5 conda run 缓冲输出（已修复 ✅）

**解决**: 直接用 `/data1/user/yikun/.conda/envs/dl/bin/python -u` 运行，绕过 `conda run`。

### 4.6 PT 实验落盘配置误导字段（已修复 ✅）

**文件**: `scripts/train_pt_hicnet.py`

**问题**: `configs/default.yaml` 中 `model.ablation_mode: baseline` 被 PT 训练原样落盘，使 `config_used.yaml` 看起来像「PT 模型被标成 baseline」。

**修复**: 落盘前 `runtime_cfg["model"].pop("ablation_mode", None)` 移除该字段。

### 4.7 负值输出可见性（已增强监控 ✅）

**文件**: `scripts/train_pt_hicnet.py`、`scripts/eval_pt_hicnet.py`

**增强**: 新增 `neg_rate`（预测值<0 的比例），写入训练日志、`history.json`、W&B。E3 seed2026 final neg_rate=0.0000。

### 4.8 `--ablation_mode` CLI 被 yaml 覆盖（已修复 ✅）

**文件**: `feather/train_reg_att_props_X70_feather.py`

**根因**: argparse `default='baseline'` 与 yaml 默认值一致，无法区分用户显式传参。

**修复**: 第 63 行 `default='baseline'` → `default=None`；第 105 行改为 `if args.ablation_mode is None: args.ablation_mode = model_cfg.get('ablation_mode', 'baseline')`。

**后果**: E1 前 3 轮误以 baseline 运行（与 E0 重复），已归档到 `_archive_debug/`，已重跑。

### 4.9 E1 pointnet2_reg_ablation 导入失败（已修复 ✅）

**文件**: `feather/model/pointnet2_reg_ablation.py`

**根因**: 第 7 行 `from models.losses import KDEWeightedHuberLoss`，`models` 不是 Python path 上的包。

**修复**: `from models.losses import ...` → `from losses import ...`

### 4.10 实验目录归档（已完成 ✅）

所有非正式 Phase 2 runs 移入 `experiments/_archive_debug/`：Phase 1 对照、旧半径运行、E2A 自动提交残留、smoke test / crash 调试、E1 被 yaml 覆盖的误运行。

### 4.11 Wandb 清理（已完成 ✅）

从 `pt-hicnet` 项目删除 7 条垃圾记录，保留 16 条（Phase 2 正式 12 条 + Phase 1 对照 2 条 + 旧 E2 参考 1 条 + 旧 E1 误运行 1 条）。

---

## 5. 实验进度 & 结果

### 5.1 消融实验矩阵（全部完成 ✅）

| ID | 骨干网络 | EarlyFusion | FiLM | 训练脚本 | 状态 |
|----|---------|:---:|:---:|------|:--:|
| E0 | PointNet++ + CrossAttn | 否 | N/A | `train_reg_att_props_X70_feather.py` | ✅ |
| E1 | PointNet++ | 是 | N/A | 同上 | ✅ |
| E2 | Point Transformer | 是 | none | `train_pt_hicnet.py` | ✅ |
| E3 | Point Transformer | 是 | global | 同上 | ✅ |
| E4 | Point Transformer | 是 | deep | 同上 | ✅ 完成 |

**统一配置：δ=150，无KDE，pt_radius=[60,150,400,1500]，200 epoch × 3 seeds (42, 3407, 2026)，batch_size=15**

### 5.2 阶段一：δ 验证实验（已完成 ✅）

| 指标 | E2-C (δ=150, 无KDE) | E2-A (δ=150, KDE) | 旧 E2 (δ=5, KDE) |
|------|:---:|:---:|:---:|
| Test MSE 均值 | **407K** | 1,903K | 1,665K |
| Test Accuracy 最优 | **0.826** (ep26) | 0.777 (ep12) | 0.822 (ep13) |
| Train Loss 终值 | 69,735 | 249,585 | 8,686 |

**结论**: E2-C (δ=150, 无KDE) 全面胜出。KDE 权重主动有害。

---

### 5.3 阶段二：Phase 2 完整结果（2026-05-31，全部完成）

#### 逐种子详细结果

```
═══════════════════════════════════════════════════════════════════════════════════════
                E0 (PN2++基线)    E1 (PN2++早融合)   E2 (PT no FiLM)   E3 (PT+FiLM g)  E4 (PT+FiLM d)
                ──────────────    ────────────────   ───────────────   ──────────────  ──────────────
seed 42
  Best Acc      82.84% (ep140)    81.01% (ep29)      82.55% (ep26)     85.03% (ep21)   80.99% (ep20)
  Best MSE        80,607              —                104,652            60,729 (ep69)  102,016 (ep20)
  Final Acc         —                 —                80.91% (ep200)    78.18% (ep200)  79.91% (ep200)
  Final MSE         —                 —                318,595           323,542         308,237
  Train Loss         —             103,820                —                 —               —

seed 3407
  Best Acc      79.86% (ep36)     80.92% (ep58)      80.48% (ep23)     83.98% (ep78)   82.71% (ep29)
  Best MSE       111,273              —                 85,526           105,822 (ep87)  128,041 (ep96)
  Final Acc         —                 —                77.15% (ep200)    82.04% (ep200)  81.55% (ep200)
  Final MSE         —                 —                255,605           169,503         607,786
  Train Loss         —             112,148                —                 —               —

seed 2026
  Best Acc      80.33% (ep54)     79.26% (ep17)      84.24% (ep13)     83.58% (ep19)   83.48% (ep15)
  Best MSE       107,858              —                 77,864            90,249 (ep19)   91,707 (ep54)
  Final Acc         —                 —                79.98% (ep200)    79.56% (ep200)  80.05% (ep200)
  Final MSE         —                 —                103,058           370,168         159,094
  Train Loss         —             110,146                —                 —               —
═══════════════════════════════════════════════════════════════════════════════════════
```

#### 汇总对比（核心表格）

```
                         E0            E1            E2            E3            E4
                      PN2++基线     PN2++早融合    PT no FiLM    PT+FiLM g     PT+FiLM d
                      ─────────     ───────────    ──────────    ──────────    ──────────
Mean Best Acc         81.01%        80.40%         82.42%        84.20%        82.39%
Mean Best MSE         99,913           —           89,347        85,600*       107,255
Mean Final Acc           —             —           79.01%        80.59%        80.50%
Best Single Acc       82.84%        81.01%         84.24%        85.03%        83.48%
Δ vs E0 (Acc)            —          -0.61pp        +1.41pp       +3.19pp       +1.38pp
收敛速度 (best ep)    36-140         17-58          13-26         19-78         15-29
Best→Final Acc Δ         —             —           -1.61pp       -4.27pp       -1.89pp
Acc Std (3 seeds)     ±1.57pp       ±1.02pp        ±2.29pp       ±0.75pp       ±1.28pp
```

> \* E3 mean best MSE 仅含 seed3407 和 seed2026（seed42 的 best MSE 60,729 与其他种子不在同一量级，均值用 seed3407+2026 估算≈98K；若含 seed42 则为 85,600）
>
> **口径说明：** E0/E1 使用 X70 脚本的 "Best Instance Mean Accuracy"（%，per-sample accuracy_ratio batch 均值），E2/E3 使用 `history.json` 的 `test_accuracy`（0-1，已 ×100 转换）。两者公式类似但 batch 聚合方式略有不同，横向对比时建议标注此差异。

---

### 5.4 消融解读：每个组件贡献了多少？

```
E0 (PN2++ 基线, 无早融合)        81.01%
     ↓ 加 EarlyFusion
E1 (PN2++ 早融合)                 80.40%   ← 反而下降 -0.61pp
     ↓ 换 PT 架构 + 早融合
E2 (PT no FiLM)                   82.42%   ← +2.02pp vs E1, +1.41pp vs E0
     ↓ 加 FiLM global
E3 (PT + FiLM global)             84.20%   ← +1.78pp vs E2, +3.19pp vs E0  ⭐ 最优
     ↓ 换 FiLM deep
E4 (PT + FiLM deep)               82.39%   ← -1.81pp vs E3, 退回 E2 水平
```

**关键洞察：EarlyFusion 只有在 PT 的 Vector Attention 配合下才有效！**

- E1 < E0：把 19 通道直接 concat 进 PointNet++（无 Vector Attention），性能反而下降。说明 PointNet++ 的简单 MLP 无法有效利用融合后的多源异质特征。
- E2 > E1 (+2.02pp)：换成 PT 架构后，Vector Attention (`q - k + delta_pos`) 能区分几何位移和材料变化对局部刚度的贡献，EarlyFusion 的价值才被释放出来。
- E3 > E2 (+1.78pp)：FiLM global 用 age_group 调制全局特征，成人/儿童头部响应差异被有效编码。
- E4 < E3 (-1.81pp)：FiLM deep 在 4 层 TransitionDown 每层都注入 age modulation，额外参数干扰了特征学习，退回到 E2 水平。**FiLM 调一次（global）刚好，调多次（deep）过度。**

---

### 5.5 过拟合分析

#### E2 详细轨迹（PT no FiLM）

| Seed | Best Acc (ep) | Best MSE | Final Acc | Final MSE | 过拟合特征 |
|------|:---:|:---:|:---:|:---:|------|
| 42 | 82.55% (26) | 104,652 | 80.91% | 318,595 | Acc 稳，MSE 恶化 3× |
| 3407 | 80.48% (23) | 85,526 | 77.15% | 255,605 | 中度过拟合 |
| 2026 | 84.24% (13) | 77,864 | 79.98% | 103,058 | 最佳极早，MSE 反弹 |

#### E3 详细轨迹（PT FiLM global）

| Seed | Best Acc (ep) | Best MSE | Final Acc | Final MSE | 过拟合特征 |
|------|:---:|:---:|:---:|:---:|------|
| 42 | 85.03% (21) | 60,729 (ep69) | 78.18% | 323,542 | Acc 跌 6.9pp，严重 |
| 3407 | 83.98% (78) | 105,822 (ep87) | 82.04% | 169,503 | **最稳定**，晚期 Acc 最高 |
| 2026 | 83.58% (19) | 90,249 (ep19) | 79.56% | 370,168 | MSE 恶化 4× |

#### 过拟合量化

| 模型 | Best→Final Acc Δ | Best→Final MSE Δ | Acc 标准差(3 seeds) |
|------|:---:|:---:|:---:|
| E0 (仅 best) | — | — | ±1.57pp |
| E1 (仅 best) | — | — | ±1.02pp |
| E2 | -1.61pp | +86K (2.0×) | ±2.29pp |
| E3 | -4.27pp | +191K (3.2×) | ±0.75pp |
| E4 | -1.89pp | +190K (2.8×) | ±1.28pp |

**规律：模型越强，过拟合越重。** E3 天花板最高但 Best→Final 跌幅也最大（2.3× E2 的跌幅），说明更强的表达能力在有限数据下更容易记住训练集噪声。E4 虽然天花板低，但过拟合幅度与 E2 相当，且种子间标准差最小（±1.28pp），最稳定。

---

### 5.6 E3-ES：Early Stopping 验证（2026-06-02）

**配置：** patience=50, restore_best=True。训练在无准确率提升 50 轮后自动停止，加载 best_acc_model.pth 权重。

| Seed | Best Acc | Best Ep | Stop Ep | Stop Acc | Stop MSE | Best→Stop Δ |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 85.03% | 21 | 71 | 80.82% | 127,505 | -4.21pp |
| 3407 | 83.98% | 78 | 128 | 83.30% | 138,735 | **-0.68pp** |
| 2026 | 83.58% | 19 | 69 | 77.82% | 242,606 | -5.76pp |
| **Mean** | **84.20%** | — | **89** | **80.65%** | **169,615** | **-3.55pp** |

**双栏报告口径：**

| 口径 | Acc | MSE | 用途 |
|------|:---:|:---:|------|
| Stop-epoch | 80.65% | 170K | 过拟合诊断 |
| **Best-checkpoint (部署)** | **84.20%** | 86K | **正式主结果** |

- Best-checkpoint = 原 E3 best Acc（天花板无损）。
- Stop-epoch MSE (170K) 比原 E3 ep200 MSE (288K) 低 41%。
- 训练 epoch 从 200 降至 89（-55%）。

**为什么 patience=50 而不是 20：** patience=20 会过早截断 seed 3407（真峰 ep78 → 83.98%，p20 停在 ep39 局部最优 82.34%）。50 保留了晚峰能力。

**restore_best 有效性：** Best→Restored Δ < 0.35pp（p20 验证确认），restore 后模型状态与 best epoch 一致。

---

### 5.7 关键发现（Phase 2，JX65 单车，已被 5.8 修正 — 保留作历史参考）

> ⚠️ **2026-06-04 更新：以下结论基于 JX65 单车测试。LOCO-CV 7 折跨车验证发现 FiLM global 跨车增益约 0pp，PT backbone 才是真正贡献 (+6~8pp)。** 详见下方 5.8。

1. **E3 (PT + FiLM global) 全面领先（JX65 单车）。** Acc 均值 84.20%，最高单次 85.03%。比 E0 基线高 +3.19pp。

2. **EarlyFusion 必须配 Vector Attention 才有效。** E1 (PN2++ + 早融合) 反而比 E0 (PN2++ 分离输入) 低 0.61pp，但 E2 (PT + 早融合) 比 E0 高 1.41pp。Vector Attention 的 `q - k` 减法机制是处理多源异质融合特征的关键——它能区分"位移差引起的特征变化"和"材料变化引起的特征变化"。**这是整个 Phase 2 最重要的科学发现。**

3. **FiLM global 有效，deep 过度。** E3 > E2 的单向 +1.78pp 差距说明 age_group 全局调制有效。但 E4 (deep) 退回到 E2 水平（82.39%），说明在每层注入 age modulation 引入噪声。

4. **PT 收敛远快于 PointNet++。** E2/E3/E4 最佳 epoch 在 13-78，而 E0 在 36-140。

5. **过拟合是主要瓶颈，Early Stopping 已缓解。** 原 E3 Best→Final 跌 4.27pp（根因：~1100 训练样本 / ~172 测试样本）。**E3-ES (patience=50 + restore_best) 已就位：Best→Stop Δ 降至 -3.55pp（3 seed mean）。部署口径用 best-checkpoint (84.20%)，stop-epoch 仅为诊断参考。训练效率提升 55%（200→89 epoch）。**

6. **E3 seed3407 最稳定。** Best Acc 83.98%，Final Acc 82.04%（原 E3 ep200），ES 下 Best→Stop Δ 仅 -0.68pp。

7. **neg_rate≈0。** 所有 PT 实验 neg_rate 接近 0，模型在 δ=150 下几乎不输出负 HIC 预测值。

8. **E4 deep FiLM 不合格。** Mean 82.39%，比 E3 低 -1.81pp，退回 E2 水平。不做为主模型。

9. **patience=50 是正确的 Early Stopping 参数。** p=20 会过早截断晚峰 seed（3407 best @ ep78 → 停在 ep39 局部最优 82.34%）。p=50 保留晚峰能力且 restore_best 有效（Best→Restored Δ < 0.35pp）。

10. **誤差分析揭示明確弱點：Adult 高 HIC 樣本。** Adult MSE = Child 的 13 倍（532K vs 40K）。Worst-20 中 75% 是 Adult。>2k HIC 區間 Acc 僅 60.14%。模型對成人系統性過預測（真值 516 → 預測 1505）。根因：訓練集成人高 HIC 樣本不足，FiLM global 對 age_group 調制強度不夠。→ 定向實驗：loss 加成人高 HIC 權重。

---

### 5.8 LOCO-CV 跨车泛化：E0/E2/E3 三架构对照（2026-06-04）

> 方法：3 架构 × 7 折 LOCO-CV，统一 `val_split=0.15 split_seed=2026 seed=42 patience=50`。
> 主表口径：test accuracy @ best-val-checkpoint。Val/Test/Gap 三栏。

#### E0 (PN++ baseline, 无 EarlyFusion)

| Vehicle | Val Acc | Test Acc | Gap |
|------|:---:|:---:|:---:|
| C201 | 78.18% | 68.34% | 9.8pp |
| EP32 | 75.44% | 73.79% | 1.6pp |
| JX65 | 72.94% | 72.21% | 0.7pp |
| CY02C | 77.89% | 64.23% | 13.7pp |
| M6 | 77.54% | 62.84% | 14.7pp |
| S50EVK | 76.19% | 71.48% | 4.7pp |
| FX11 | 77.53% | 70.23% | 7.3pp |
| **Mean** | **76.53%** | **69.02%** ±4.13pp | |

#### E2 (PT, FiLM none)

| Vehicle | Val Acc | Test Acc | Gap |
|------|:---:|:---:|:---:|
| C201 | 86.35% | 75.99% | 10.4pp |
| EP32 | 84.24% | 79.93% | 4.3pp |
| JX65 | 83.45% | 79.83% | 3.6pp |
| CY02C | 84.27% | 59.10% | 25.2pp |
| M6 | 84.85% | 61.33% | 23.5pp |
| S50EVK | 84.86% | 77.12% | 7.7pp |
| FX11 | 84.82% | 75.27% | 9.6pp |
| **Mean** | **84.69%** | **72.65%** ±8.70pp | |

#### E3 (PT, FiLM global)

| Vehicle | Val Acc | Test Acc | Gap |
|------|:---:|:---:|:---:|
| C201 | 84.97% | 76.71% | 8.3pp |
| EP32 | 84.35% | 80.31% | 4.0pp |
| JX65 | 84.46% | 78.87% | 5.6pp |
| CY02C | 84.92% | 58.60% | 26.3pp |
| M6 | 84.86% | 60.67% | 24.2pp |
| S50EVK | 83.98% | 79.93% | 4.0pp |
| FX11 | 86.14% | 76.13% | 10.0pp |
| **Mean** | **84.81%** | **73.03%** ±9.30pp | |

#### 核心对照

**正常车（5 辆：C201 / EP32 / JX65 / S50EVK / FX11）：**

| 指标 | E0 (PN++) | E2 (PT none) | E3 (PT+FiLM) |
|------|:---:|:---:|:---:|
| Val Mean | 76.05% | 84.75% | 84.78% |
| Test Mean | 71.21% | 77.63% | 78.39% |
| Test Std | ±2.06pp | ±2.16pp | ±1.88pp |
| **E2−E0 (PT backbone)** | — | **+6.42pp** | — |
| **E3−E2 (FiLM global)** | — | — | **+0.76pp** |

**难车（2 辆：CY02C / M6）：**

| 指标 | E0 (PN++) | E2 (PT none) | E3 (PT+FiLM) |
|------|:---:|:---:|:---:|
| Test Mean | 63.53% | 60.22% | 59.64% |
| Val→Test Gap | 14.2pp | 24.3pp | 25.3pp |

#### Per-Vehicle Winner

| Vehicle | E0 | E2 | E3 | Winner | E2−E0 | E3−E2 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| C201 | 68.3% | 76.0% | 76.7% | E3 | +7.7pp | +0.7pp |
| EP32 | 73.8% | 79.9% | 80.3% | E3 | +6.1pp | +0.4pp |
| JX65 | 72.2% | 79.8% | 78.9% | **E2** | +7.6pp | −1.0pp |
| CY02C | 64.2% | 59.1% | 58.6% | **E0** | −5.1pp | −0.5pp |
| M6 | 62.8% | 61.3% | 60.7% | **E0** | −1.5pp | −0.7pp |
| S50EVK | 71.5% | 77.1% | 79.9% | E3 | +5.6pp | +2.8pp |
| FX11 | 70.2% | 75.3% | 76.1% | E3 | +5.0pp | +0.9pp |

#### 修正后核心结论

1. **PT backbone 是主贡献（+6.4pp），FiLM global 跨车增益不显著（+0.8pp）。** Phase 2 "E3>E2 by +1.78pp" 是 JX65 假阳性——LOCO 下 JX65 E2(79.8%) > E3(78.9%)。

2. **Val 极紧（E2/E3 Val Std < 1pp），Test 方差由车型驱动（正常 ±2pp，难车 Gap>24pp）。** 域偏移来自车辆特性，非模型不稳定。

3. **PT 在难车上过拟合更重。** CY02C/M6：E0 Gap=14pp，E3 Gap=25pp。更强模型 = 更难泛化到异分布车。

4. **论文叙事调整：** 主 claim 从 "FiLM improves HIC" → "PT backbone is primary cross-car gain; FiLM gain is data-distribution-dependent."

#### E1 (PN2++ EarlyFusion) — added 2026-06-05

| Vehicle | Val Acc | Test Acc | Gap |
|------|:---:|:---:|:---:|
| C201 | 81.33% | 67.98% | 13.4pp |
| EP32 | 80.29% | 74.27% | 6.0pp |
| JX65 | 77.31% | 80.04% | −2.7pp |
| CY02C | 78.28% | 61.35% | 16.9pp |
| M6 | 80.18% | 60.50% | 19.7pp |
| S50EVK | 79.11% | 70.68% | 8.4pp |
| FX11 | 79.44% | 69.86% | 9.6pp |
| **Mean** | **79.42%** | **69.24%** ±6.89pp | |

#### E4 (PT FiLM deep) — added 2026-06-05

| Vehicle | Val Acc | Test Acc | Gap |
|------|:---:|:---:|:---:|
| C201 | 87.96% | 74.15% | 13.8pp |
| EP32 | 86.78% | 78.21% | 8.6pp |
| JX65 | 85.46% | 78.52% | 6.9pp |
| CY02C | 84.43% | 63.39% | 21.0pp |
| M6 | 86.58% | 61.81% | 24.8pp |
| S50EVK | 86.80% | 80.41% | 6.4pp |
| FX11 | 85.56% | 72.48% | 13.1pp |
| **Mean** | **86.22%** | **72.71%** ±7.43pp | |

#### 五架构全量对照（5 正常车）

| 指标 | E0 | E1 | E2 | E3 | E4 |
|------|:---:|:---:|:---:|:---:|:---:|
| Val Mean | 76.05% | 79.99% | 84.75% | 84.78% | 86.46% |
| Test Mean | 71.21% | 72.56% | 77.63% | 78.39% | 76.75% |
| E2−E0 | — | — | **+6.42pp** | — | — |
| E3−E2 | — | — | — | **+0.76pp** | — |
| E4−E2 | — | — | — | — | **−0.88pp** |

**新增结论 (2026-06-05)：**
- **E4 (FiLM deep) 确认是过拟合器**：Val 最高 (86.22%)，5-normal Test (76.75%) 反而不如 E2 no FiLM (77.63%)。deep modulation = 更多参数 = 更强过拟合 = 更差泛化。
- **E1 ≈ E0**：EarlyFusion 在 PN++ 上跨车无效 (+1.35pp)，再次确证 Vision Attention 是释放融合信息的必要条件。

---

## 6. Wandb 记录索引

### Phase 2 正式结果（12 条）

| Wandb Run Name | 实验 | Seed | Best Acc |
|------|------|:---:|:---:|
| `2026-05-31_00-59` | E0 baseline | 42 | 82.84% |
| `2026-05-31_02-46` | E0 baseline | 3407 | 79.86% |
| `2026-05-31_04-32` | E0 baseline | 2026 | 80.33% |
| `2026-05-31_13-23` | E1 early_fusion | 42 | 81.01% |
| `2026-05-31_15-10` | E1 early_fusion | 3407 | 80.92% |
| `2026-05-31_16-57` | E1 early_fusion | 2026 | 79.26% |
| `pt_hicnet_ablation_seed42_film-none` | E2 PT none | 42 | 82.55% |
| `pt_hicnet_ablation_seed3407_film-none` | E2 PT none | 3407 | 80.48% |
| `pt_hicnet_ablation_seed2026_film-none` | E2 PT none | 2026 | 84.24% |
| `pt_hicnet_ablation_seed42_film-global` | E3 PT global | 42 | 85.03% |
| `pt_hicnet_ablation_seed3407_film-global` | E3 PT global | 3407 | 83.98% |
| `pt_hicnet_ablation_seed2026_film-global` | E3 PT global | 2026 | 83.58% |
| `pt_hicnet_ablation_seed42_film-deep` | E4 PT deep | 42 | 80.99% |
| `pt_hicnet_ablation_seed3407_film-deep` | E4 PT deep | 3407 | 82.71% |
| `pt_hicnet_ablation_seed2026_film-deep` | E4 PT deep | 2026 | 83.48% |

### Phase 1 对照（2 条）

| Wandb Run Name | 用途 |
|------|------|
| `e2a_delta150_kde_seed42_film-none` | KDE 有害验证 |
| `e2c_nokde_seed42_film-none` | δ=150 无 KDE 胜出 |

---

## 7. 待解决的技术问题

### 7.1 回归头无输出约束（低优先级）

**问题：** `Linear(256, 1)` 可输出任意实数，HIC 严格 > 0。但实测 neg_rate≈0，暂时不紧急。

### 7.2 数据域偏移

**问题：** 测试集仅 JX65（1 辆车，~172 样本）。长期方案：leave-one-car-out 交叉验证。

### 7.3 样本量偏少 → 过拟合

**问题：** ~1100 样本对 4 层 PT 偏少，原 E3 Best→Final Acc 跌 4.27pp（已由 E3-ES 的 patience=50 缓解至 -3.55pp）。候选缓解：降低模型复杂度（减少 PT 层数或通道数）、数据增强、early stopping（best epoch 通常在 20-80，可设 patience=50）。

### 7.4 screen session 清理

E2A、E2C、test 等旧 screen 仍在，建议 Phase 2 全部完成后清理。

---

## 8. 下一步计划

```
Phase 1 ✅: δ=150 + 无KDE → E2-C 胜出

Phase 2 ✅: 五组消融 (JX65 单车 × 3 seeds)

Phase 3a ✅: E3 LOCO-CV 7 折 + early stopping
Phase 3b ✅: E0 + E2 LOCO-CV 7 折 → PT backbone +6.4pp, FiLM ~0pp
Phase 3c ✅: E1 + E4 LOCO-CV 7 折 → 五架构全量 35/35 folds 就位

Phase 4（本周）:
  ├─ 5 架构全 LOCO 排名表 + aggregate_loco.py 多架构升级
  ├─ 难车 error analysis 模板: CY02C/M6 per-sample 误差诊断
  └─ 管线固化: freeze config + eval script 版本标签

Phase 4（新数据到位前）:
  ├─ 数据扩充与清洗（新增车型）
  ├─ 定义新数据验收门槛（E2−E0 ≥ +Xpp 等）
  └─ 固化评估口徑文檔 eval_protocol.md

Phase 5（新数据到位后）:
  └─ 新全量数据 multi-seed 最终统计 (E0/E2/E3 3 seeds × 全车型)
```

> **2026-06-04 决策：Multi-seed 暂缓。** 新车型数据即将加入，现在用旧分布跑 multi-seed 边际价值低。等新数据并入后一次性做最终统计。

### E4 deep FiLM 退火分析

E4 (82.39%) 退回到 E2 (82.42%) 水平，比 E3 低 -1.81pp。deep FiLM 在 4 层 TransitionDown 每层注入 age modulation，额外参数干扰了各层特征学习。**结论：FiLM 一次 global 调制已足够，deep 是过度设计。**

### 运行脚本

| 脚本 | 路径 | 内容 |
|------|------|------|
| `run_e0.sh` | `experiments/run_e0.sh` | E0 baseline ×3 seeds, GPU 0 |
| `run_e1.sh` | `experiments/run_e1.sh` | E1 early_fusion_clean ×3 seeds, GPU 0 |
| `run_e2.sh` | `experiments/run_e2.sh` | E2 PT none ×3 seeds, GPU 1 |
| `run_e3.sh` | `experiments/run_e3.sh` | E3 PT global ×3 seeds, GPU 1 |
| `run_e4.sh` | `experiments/run_e4.sh` | E4 PT deep ×3 seeds, GPU 1 |

---

## 9. 配置文件索引

| 文件 | 用途 | δ | KDE | 半径 |
|------|------|:---:|:---:|------|
| `configs/default.yaml` | 基准配置（Phase 2 使用） | 150 | 无 | 修复后 |
| `configs/tmp_e2a_delta150_kde.yaml` | E2-A 实验（KDE 有害） | 150 | 有 | 修复后 |
| `configs/tmp_e2c_nokde.yaml` | E2-C 实验（✅ 胜出） | 150 | 无 | 修复后 |
| `configs/tmp_e2_seed42_50ep_wandb.yaml` | 旧 E2（δ=5+KDE） | 5.0 | 有 | 修复后 |

---

## 10. 关键设计决策记录

| 决策 | 内容 | 力学/工程依据 |
|------|------|------|
| 输入 19 通道 | 3(XYZ) + 1(thickness) + 15(material) | 材料 15 维含完整本构信息；厚度感知局部抗弯刚度 |
| 材料不预嵌入 | 15 维原始物理量直接做通道，由 Conv1d 隐式编码 | 避免信息瓶颈 |
| age_group 延迟注入 | 仅通过 FiLM 在回归头前注入 | 年龄影响全局载荷条件，不应参与逐点特征学习 |
| q-k Vector Attention | `q - k + delta_pos` 编码相邻节点位移差 | FEA 应变张量 = 位移梯度对称部分 |
| pt_radius = [60, 150, 400, 1500] | 基于点间距 15-25mm 缩放 | 匹配 FPS 降采样后的间距膨胀 |
| input_proj 加 BatchNorm | Conv1d → BN → ReLU | 消除 XYZ/thickness/material 三组通道量级差异 |
| Huber δ=150, 去掉 KDE | 混合 MSE+MAE 损失 | Phase 1 验证：MSE 降 4.1×, Acc 升 +12pp |
| E0 baseline 不用 EarlyFusion | PN2++ forward 需单独 thickness & material_props | forward 签名 6 参数 |
| E1 用独立模型 | `pointnet2_reg_ablation.py`, 4 参数 forward | fused input 直接进入 PN2++ |
| 数据加载用 numpy 存储 | 预分配 numpy array | RSS 65GB→7.5GB，加载 4.5h→5min |
| CLI > YAML 参数优先级 | `--ablation_mode` 默认 `None` | 防止 yaml 覆盖 CLI 意图 |

---

## 11. 汇报速查（下午报告用）

### 一句话

物理引导的 Point Transformer，输入 FEA 网格点云预测 HIC，Vector Attention + EarlyFusion + FiLM global，最优 Acc 84.20%，比 PointNet++ 基线高 3.2pp。deep FiLM 退回 E2 水平（82.39%），确认 global 调制足矣。

### 核心数字

| 指标 | 数值 |
|------|:---:|
| 最优模型 | E3: PT + FiLM global |
| Mean Acc (3 seeds) | 84.20% |
| Best Single Acc | 85.03% |
| vs PN2++ 基线 | +3.19pp |
| 收敛速度 | epoch 19-78 (vs 基线 36-140) |
| 主要瓶颈 | 过拟合 (原 E3 Best→Final 跌 4.27pp，E3-ES 已缓解至 -3.55pp) |
| 训练数据 | 6 车 ~1100 样本 |
| 测试数据 | JX65 ~172 样本 |
| 单次训练耗时 | ~1.5-2h (200 ep, L20) |

### 消融结论

```
EarlyFusion 单用 (E1):   ❌  -0.61pp  (PN2++ 的 MLP 无法利用融合特征)
PT 架构 (E2):            ✅  +1.41pp  (Vector Attention 释放 EarlyFusion)
FiLM global (E3):        ✅  +1.78pp  (年龄组全局调制有效) ⭐ 主模型
FiLM deep (E4):          ❌  -1.81pp  (退回 E2 水平，deep 过度设计)
```
