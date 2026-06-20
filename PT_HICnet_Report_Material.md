# PT-HICnet 技术素材：架构、参数与 LOCO 战绩

> **数据来源**：`configs/default.yaml` · `models/pt_hicnet.py` · `models/point_transformer_layer.py` · `models/point_transformer_block.py` · `feather/data_utils/HICLoader_feather.py` · `experiments/loco_cv/loco_cv_report.md`  
> **说明**：本文档仅提取与归纳，不修改任何代码。

---

## 与原版 Point Transformer（PT）的对应关系概览

| 维度 | 原版 Point Transformer（文献/通用实现） | PT-HICnet（本项目） |
|------|--------------------------------------|---------------------|
| **任务** | 点云语义分割 / 分类（如 S3DIS、ShapeNet） | **HIC 回归**（FEA 碰撞 surrogate） |
| **输入坐标** | 通常归一化到单位球或米级场景 | **FEA 网格 mm 坐标**，不做全局归一化 |
| **输入特征** | 常见 xyz (+ rgb/normal) | **19 维 Early Fusion**：xyz + thickness + 15 维材料本构 |
| **邻域构建** | FPS + Ball Query / k-NN，半径与数据尺度匹配 | **FPS + Ball Query**，半径 **[60, 150, 400, 1500] mm**（针对 FEA 点间距 13–27 mm 标定） |
| **Attention 核心** | Vector Attention：`softmax(γ(q−k+δ_pos)) · (α(x_j)+δ_pos)` | **同构实现**（`point_transformer_block.py`） |
| **降采样** | Transition Down：FPS → 分组 → PT Block → Pool → Proj | **同构 4 层**，npoint **8192→512→128→32→1** |
| **通道演进** | 逐层扩宽（实现各异） | **64→128→256→512→1024**（`dims` in `pt_hicnet.py`） |
| **输入投影** | 常见 Linear/Conv 直接映射 | **Conv1d(19→64) + BatchNorm1d + ReLU**（平衡 mm 级 xyz 与 Z-score 材料通道） |
| **条件调制** | 通常无 | **FiLM（age_group）**：global / deep 两种模式（E3/E4 消融） |
| **输出头** | 逐点 / 全局分类头 | **全局 1024 维 + 碰撞点 64 维 → MLP → 标量 HIC** |
| **损失** | Cross-Entropy 等 | **Huber Loss，δ=150，KDE 权重禁用** |

**一句话**：PT-HICnet **保留了原版 PT 的 FPS + Ball Query + Vector Attention + Transition Down 骨架**，针对 FEA 汽车网格做了 **尺度标定（mm 半径）、19 通道早融合、BatchNorm 输入投影、HIC 回归头与 FiLM 条件分支** 的领域化改造。

---

## 模块一：宏观数据流与特征演进

### 1.1 数据加载 → 8192 点输入

**来源**：`HICLoader_feather._get_item()`（`feather/data_utils/HICLoader_feather.py`）

| 步骤 | 操作 | 输出形状 / 说明 |
|------|------|----------------|
| 读取 FEA 节点 | 从 `.feather` 加载单样本网格 | 原始 N 点（约 8000–13000） |
| 点采样 | N ≥ 8192：无放回随机采样；N < 8192：有放回补齐 | `point_set [8192, 3]`，单位 **mm** |
| 板厚 | per-sample min-max | `thickness [8192, 1]` → **[0, 1]** |
| 材料 | Z-Score（`normalization_params.pkl`） | `material_props [8192, 15]` |
| Early Fusion | `concat` 点级拼接 | **`fused_input [8192, 19]`** |
| 其他输入 | 碰撞点坐标、年龄组、HIC 标签 | `hic_point [3]`，`age_group ∈ {0,1}`，`label [1]` |

**19 维 Early Fusion 构成**：

| 通道 | 维数 | 内容 | 归一化 |
|------|:---:|------|--------|
| XYZ | 3 | FEA 节点空间坐标 (mm) | 无（保留物理尺度） |
| Thickness | 1 | 板厚 | per-sample min-max → [0,1] |
| Material | 15 | 密度 ρ + 杨氏模量 E + 泊松比 PR + 12 点应力-应变曲线采样 | Z-Score |

15 维材料明细（代码 L517–532）：
- [0] 密度 RO/R0
- [1] 杨氏模量 E
- [2] 泊松比 PR
- [3–8] 0.001 应变水平下 6 个应力采样点（0, 0.05, 0.1, 0.15, 0.2, 0.5）
- [9–14] 1.0 应变水平下 6 个应力采样点（同上）

### 1.2 Input Projection（进入 PT Block 前）

**来源**：`models/pt_hicnet.py` L23–27, L66–67

```
DataLoader 输出:  fused_input [B, 8192, 19]
train 脚本转置:   fused_input [B, 19, 8192]
                         │
                         ▼
              input_proj: Conv1d(19 → 64, kernel=1)
                         BatchNorm1d(64)
                         ReLU
                         │
                         ▼
              feats [B, 8192, 64]   （同时 xyz = fused_input[:,:3] → [B,8192,3] 供 FPS/Ball Query）
```

**与原版 PT 差异**：原版通常输入 3–6 维同质特征；本项目 **19 维异质通道（mm 级 xyz + [0,1] 厚度 + Z-score 材料）**，故增加 **BatchNorm1d** 防止 xyz 梯度主导（`context_summary.md` §4.2）。

### 1.3 四层 Transition Down 逐层演进

**来源**：`models/pt_hicnet.py` L29–33, L13–14；`configs/default.yaml` L39–41

| 层 | 模块 | 输入点数 N | FPS 后 npoint | Ball Query 半径 (mm) | nsample | 输入通道 C_in | 输出通道 C_out |
|:--:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| — | Input Projection | 8192 | — | — | — | **19** | **64** |
| 1 | `down1` | 8192 | **512** | **60** | 32 | 64 | **128** |
| 2 | `down2` | 512 | **128** | **150** | 32 | 128 | **256** |
| 3 | `down3` | 128 | **32** | **400** | 32 | 256 | **512** |
| 4 | `down4` | 32 | **1** | **1500** | 32 | 512 | **1024** |

**点数缩减链**：`8192 → 512 → 128 → 32 → 1`（全局单点特征 `x_global [B, 1024]`）

**通道扩增链**：`19 → 64 → 128 → 256 → 512 → 1024`

### 1.4 Transition Down 内部数据流（单层）

**来源**：`models/point_transformer_layer.py` L23–50

```
xyz [B,N,3], feats [B,N,C_in]
  │
  ├─ FPS(xyz, npoint)           → new_xyz [B,npoint,3], fps_idx
  ├─ BallQuery(radius, nsample) → group_idx [B,npoint,32]
  ├─ grouped_xyz, grouped_feats, rel_pos = new_xyz - grouped_xyz
  │
  ├─ PointTransformerBlock(center_feats, grouped_feats, rel_pos)
  │     → out [B,npoint,C_in]
  │
  ├─ MaxPool + AvgPool on grouped_feats → pooled [B,npoint,C_in*2]
  │
  └─ proj: Linear(C_in*3 → C_out) + ReLU
        concat[out, pooled] 其中 C_in*3 = C_in(attn_out) + C_in(max) + C_in(avg)
        → out [B,npoint,C_out]
```

### 1.5 回归头（PT 骨架之后）

**来源**：`models/pt_hicnet.py` L35–36, L51–58, L83–88

```
x_global [B, 1024]
  ├─ [E3/E4] FiLM(x_global, age_emb)     age_emb: Embedding(2→16)
  ├─ hic_feat = Linear(3→64)(hic_point)
  └─ cat → [B, 1088] → MLP(1088→512→256→1) → pred HIC [B,1]
```

---

## 模块二：物理参数与邻域构建（针对 CAERI 专家）

### 2.1 Ball Query 半径：[60, 150, 400, 1500] mm

**配置来源**：`configs/default.yaml` L40 · `models/pt_hicnet.py` L14

| 层 | npoint | radius (mm) | 物理尺度含义（汇报用语） | 诊断统计（首次 forward） |
|:--:|:---:|:---:|------|:---:|
| L1 | 512 | **60** | ~**2–4×** FEA 平均点间距（13–27 mm）→ 覆盖**单焊点 / 局部加强筋**邻域 | avg_valid=129.7, pct_full=**100%** |
| L2 | 128 | **150** | FPS 后点间距膨胀 → 覆盖**子结构 / 局部板件** | avg_valid=72.6, pct_full=**96%** |
| L3 | 32 | **400** | 覆盖**区域级**变形模式 | avg_valid=101.8, pct_full=**100%** |
| L4 | 1 | **1500** | 接近**整车碰撞区域**尺度（对角线 1100–2400 mm） | avg_valid=32.0, pct_full=**100%** |

**FEA 点云尺度基准**（`context_summary.md` §3，单位 mm）：

| 车辆 | 对角线 | 平均点间距 | 平均 NN 距离 |
|------|:---:|:---:|:---:|
| C201 | 1407 | 15.55 | 19.47 |
| EP32 | 1201 | 13.27 | 21.29 |
| CY02C | **2416** | 26.69 | 13.66 |
| M6 | **2349** | 25.95 | 14.74 |

### 2.2 为什么必须用大半径，而不是「极小半径」或纯 k-NN？

**历史教训（本项目实测）**：
- 旧配置 `pt_radius = [0.2, 0.4, 0.8, 1.2]`（`context_summary.md` §4.1）
- 比 FEA 点间距（15–25 mm）**小约 100 倍**
- Ball Query 几乎找不到有效邻居 → Vector Attention **退化为孤立点 MLP** → E2/E4 训练无效

**与原版 PT 的差异逻辑**：

| | 原版 PT（典型设置） | PT-HICnet |
|--|-------------------|-----------|
| 坐标尺度 | 物体归一化到 [-1,1] 或米级室内场景 | **mm 级 FEA 网格，保留物理坐标** |
| 半径含义 | 与归一化尺度匹配的小球（如 0.1–0.5 in normalized space） | **必须与 mm 点间距同量级**（60–1500 mm） |
| 邻域策略 | Ball Query 或 k-NN，均可 | **Ball Query + 固定 nsample=32**（`query_ball_point` in `point_transformer_utils.py` L46–57） |
| 若误用极小半径 | 在归一化点云上仍可能有足够邻居 | 在 mm FEA 网格上**邻居数 ≈ 0**，attention 失效 |

**Ball Query 实现要点**（`point_transformer_utils.py` L46–57）：
```python
sqrdists = square_distance(new_xyz, xyz)       # 欧氏距离平方
group_idx[sqrdists > radius ** 2] = sentinel   # 超出半径的点剔除
group_idx = sort & take top nsample            # 每中心点最多 32 邻居
# 邻居不足时用最近点填充
```

### 2.3 降采样策略：FPS（Farthest Point Sampling）

**来源**：`point_transformer_utils.py` L27–43 · `TransitionDown.forward` L24–25

- **算法**：迭代选取距已选点集最远的点（标准 PointNet++ / PT 同款 FPS）
- **作用**：每层 Transition Down 先将 N 点减至 npoint，再在 FPS 中心点的 Ball Query 邻域内做 Vector Attention
- **与原版 PT 关系**：**一致**——原版 PT 的 Transition Down 同样采用 FPS 作为几何降采样骨干

**本项目 FPS 调用链**：
```
8192 原始点 ──FPS(512)──→ 512 中心点 ──FPS(128)──→ … ──FPS(1)──→ 1 全局点
每层中心点独立做 Ball Query(radius) 取 32 邻居
```

---

## 模块三：Vector Attention 底层实现代码快照

**源文件**：`models/point_transformer_block.py` L6–31

### 3.1 符号与张量形状

| 符号 | 代码变量 | 形状 | 含义 |
|------|----------|------|------|
| 中心点特征 | `x_i` = `center_feats` | [B, N, C] | FPS 采样后的中心点 |
| 邻居特征 | `x_j` = `grouped_feats` | [B, N, K, C] | Ball Query 邻域，K=32 |
| 相对位置 | `rel_pos` | [B, N, K, 3] | `new_xyz - grouped_xyz`（mm） |

### 3.2 运算逻辑（与原版 PT 公式对应）

**原版 PT 论文形式**：
$$\text{Attention}(i,j) = \text{softmax}\big(\gamma(\mathbf{q}_i - \mathbf{k}_j + \delta_{ij})\big) \cdot \big(\alpha(\mathbf{x}_j) + \delta_{ij}\big)$$

**本项目代码实现（伪代码）**：

```python
# Step 1: 位置编码
delta_pos = MLP_theta(rel_pos)          # Linear(3→C) → ReLU → Linear(C→C)
                                          # theta: point_transformer_block.py L8

# Step 2: Query / Key / Value
q = Linear_phi(x_i).unsqueeze(2)         # [B, N, 1, C]   — 中心点 query
k = Linear_psi(x_j)                      # [B, N, K, C]   — 邻居 key
v = Linear_alpha(x_j) + delta_pos        # [B, N, K, C]   — value 注入位置

# Step 3: Vector subtraction attention（核心创新点）
logits = MLP_gamma(q - k + delta_pos)    # [B, N, K, C]   — 逐通道向量减法
attn   = softmax(logits, dim=K)            # 在邻居维度 K 上归一化

# Step 4: 加权聚合 + 残差 FFN
out = sum(attn * v, dim=K)                 # [B, N, C]
out = LayerNorm(x_i + out)
out = LayerNorm(out + FFN(out))
```

### 3.3 物理动机（面向 CAERI 专家）

| 项 | 物理类比 |
|----|----------|
| `q - k` | 邻域特征差分 → 对应 **应变梯度 / 刚度突变** |
| `delta_pos = θ(rel_pos)` | 相对位移编码 → **FEA 节点间几何关系** |
| `v = α(x_j) + delta_pos` | 值向量注入位置 → 聚合时保留 **空间几何信息** |
| `softmax` over 邻居 | 在物理邻域内自适应加权 → **关注高梯度区域**（焊点、加强筋） |

### 3.4 与原版 PT 的实现差异

| 项目 | 原版 PT | PT-HICnet |
|------|---------|-----------|
| Attention 公式 | `q−k+δ_pos` | **相同** |
| 邻域来源 | Ball Query on FPS centers | **相同** |
| 特征维度 C | 随层 64→128→… | **相同通道 schedule** |
| 输入到 Block 的特征 | 通常 xyz embedding + 语义特征 | **19 维融合后经 Conv1d+BN 投影的 64 维特征** |

---

## 模块四：消融实验与泛化大考（核心战绩）

**实验协议**（`loco_cv_report.md` 头部）：
- 7-fold LOCO-CV，seed=42，val_split=0.15，patience=50，restore_best
- 指标：**accuracy_ratio**（回归相对精度，非分类 accuracy）
- 选 checkpoint：Val Acc 最高 epoch → 报告 Test Acc
- **5 辆正常车**：C201, EP32, JX65, S50EVK, FX11  
- **2 辆难车**：CY02C, M6（车身尺度大、domain shift 明显）

### 4.1 五架构在 5-Normal 车上的 Mean Test Acc

**来源**：`experiments/loco_cv/loco_cv_report.md` Table 1, "5-Normal" 列

| 架构 | 含义 | 5-Normal Mean Test Acc | vs E0 |
|------|------|:---:|:---:|
| **E0** | PointNet++ baseline | **71.21%** | — |
| **E1** | PN++ + Early Fusion (19ch) | **72.56%** | +1.35pp |
| **E2** | PT backbone, FiLM=none | **77.63%** | **+6.42pp** |
| **E3** | PT + FiLM global | **78.39%** | +7.18pp |
| E4 | PT + FiLM deep（对照） | 76.75% | +5.54pp |

**组件隔离结论**：

| Δ | Value | 解读 |
|---|:---:|------|
| E2−E0 | **+6.42pp** | **PT + Vector Attention 是唯一显著贡献** |
| E3−E2 | +0.76pp | FiLM global 跨车不显著 |
| E1−E0 | +1.35pp | 早融合 alone 在 PN++ 上无效 |

### 4.2 5-Normal 逐车 Test Acc（五架构完整版）

**来源**：`history.json` per-fold best-val checkpoint，`scripts/aggregate_loco.py --architectures E0 E1 E2 E3 E4`

| Vehicle | E0 | E1 | E2 | E3 | E4 | E3−E0 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| C201 | 68.34% | 67.98% | 75.99% | **76.71%** | 74.15% | +8.37pp |
| EP32 | 73.79% | 74.27% | 79.93% | **80.31%** | 78.21% | +6.52pp |
| JX65 | 72.21% | **80.04%** † | 79.83% | 78.87% | 78.52% | +6.66pp |
| S50EVK | 71.48% | 70.68% | 77.12% | 79.93% | **80.41%** | +8.45pp |
| FX11 | 70.23% | 69.86% | 75.27% | **76.13%** | 72.48% | +5.90pp |
| **Mean** | **71.21%** | **72.56%** | **77.63%** | **78.39%** | 76.75% | **+7.18pp** |

> † **JX65 E1 异常标注**：E1 fold2_JX65 Val=77.31% → Test=80.04%，Gap=−2.7pp，是 35 折中**唯一** Test 超过 Val 的一折。可能原因：(1) val_split=0.15 下 JX65 的 val 子集碰巧偏难；(2) E1 与 PT 变体虽同为 StepLR 口径，但训练脚本和模型族不同，val 曲线波动更大。此 80.04% 为单折现象，**不作为 E1 优于 PT 的证据**——E1 的 5-normal mean 仅 72.56%，低于所有 PT 变体。

### 4.3 难车 CY02C / M6：Test 掉点与 Val-Test Gap

**用途**：分析 **尺度失配 / domain shift / 过拟合** 的汇报素材。

#### Test Acc 对比（越低越差）

| Vehicle | E0 | E1 | E2 | E3 | E4 | 现象 |
|---------|:---:|:---:|:---:|:---:|:---:|------|
| **CY02C** | **64.23%** | 61.35% | 59.10% | 58.60% | 63.39% | PT 路线 **低于** PN++ baseline；E4 反而略好于 E2/E3 |
| **M6** | **62.84%** | 60.50% | 61.33% | 60.67% | 61.81% | 同上；E4 与 E2/E3 接近 |

**2-Hard Mean Test**：E0 **63.53%** > E4 62.60% > E2 60.22% > E3 **59.64%**

#### Val − Test Gap（pp）：过拟合 / 分布偏移信号

**来源**：`loco_cv_report.md` Table 3（E2/E3 为重点）

| Vehicle | E0 Gap | E1 Gap | E2 Gap | E3 Gap | E4 Gap |
|---------|:---:|:---:|:---:|:---:|:---:|
| **CY02C** | 13.7 | 16.9 | **25.2** | **26.3** | 21.0 |
| **M6** | 14.7 | 19.7 | **23.5** | **24.2** | 24.8 |

**解读要点（可直接用于汇报）**：
1. CY02C/M6 车身对角线 ~**2400 mm**，约为 C201/EP32（~1200 mm）的 **2 倍**——训练分布中少见的大尺度结构
2. PT 路线（E2/E3/E4）在难车上 **Val 84–87%** 但 **Test 58–63%** → Gap **>21pp**，典型 **「高 Val、低 Test」** 过拟合 + domain shift。E4 在 CY02C 上 Test=63.39% 略好于 E2/E3，但仍 < E0（64.23%）
3. E0 baseline Gap 仅 13–15pp，Test 反而最高 → **更简单模型在 OOD 难车上更鲁棒**
4. 这不否定 PT 在正常车上的 +6.4pp 增益，但说明 **泛化边界** 需在 Phase 4 做失效分析与数据扩充

### 4.4 LOCO 全 7 车汇总（补充）

| 架构 | Val Mean | Test Mean (7-car) | Test Std (7-car) |
|------|:---:|:---:|:---:|
| E0 | 76.53% | 69.02% | ±4.13pp |
| E1 | 79.42% | 69.24% | ±6.89pp |
| E2 | 84.69% | 72.65% | ±8.70pp |
| **E3** | 84.81% | **73.03%** | ±9.30pp |
| E4 | **86.22%** | 72.71% | ±7.43pp |

> E4：**Val 最高（86.22%）、全车 Test（72.71%）低于 E3（73.03%）** → deep FiLM 整体是过拟合器，非泛化增益。  
> **但注意 CY02C 反例**：E4 在 CY02C 上 Test=63.39%，高于 E2（59.10%）和 E3（58.60%）——deep FiLM 在该 OOD 车上反而略好，不过仍低于 E0 baseline（64.23%）。汇报话术：「E4 在 Val 上系统性过拟合，全车队 Test 略低于 E3；难车表现因车而异，CY02C 上 E4 略优于 E2/E3，但不改变 deep FiLM 整体有害的结论。」

---

## 附录：关键配置参数速查

| 参数 | 值 | 文件位置 |
|------|:---:|----------|
| `num_point` | 8192 | `default.yaml` L13 |
| `in_channels` | 19 | `default.yaml` L36 |
| `pt_radius` | [60, 150, 400, 1500] mm | `default.yaml` L40 |
| `pt_nsample` | [32, 32, 32, 32] | `default.yaml` L41 |
| `pt_npoints` | (512, 128, 32, 1) | `pt_hicnet.py` L13 |
| `film_mode` | none / global / deep | 消融变量 |
| `Huber delta` | 150 | `default.yaml` L46 |
| `patience` | 50 | LOCO 脚本 |
| `batch_size` | 15 | `default.yaml` L20 |
| `learning_rate` | 0.001 | `default.yaml` L22 |

---

> **文件路径**：`PT-HICNET/PT_HICnet_Report_Material.md`  
> **聚合命令**：`python scripts/aggregate_loco.py --architectures all --results_root experiments`
