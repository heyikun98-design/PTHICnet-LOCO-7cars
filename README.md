# PT-HICnet — Point Transformer for HIC Surrogate Modeling

基于 Point Transformer 的汽车碰撞 HIC（Head Injury Criterion）深度学习代理模型。

## 物理动机：从刚度矩阵到 Vector Attention

### 问题背景

汽车碰撞 HIC 的传统评估依赖整车 FEA 仿真（单车型 ~4–8 小时），无法支撑设计迭代。深度学习代理模型的目标是：给定一辆车的 FEA 网格节点（坐标 + 材料 + 厚度），直接预测 HIC 值。

现有工作的瓶颈不在特征提取，而在**邻域聚合算子缺乏物理归纳偏置**。

### FEA 告诉我们什么

有限元法中，单元刚度矩阵的组装是碰撞模拟的核心计算：

$$
\mathbf{K}_e = \int_{\Omega_e} \mathbf{B}^\top \mathbf{D} \mathbf{B} \, d\Omega
$$

离散化为高斯积分：

$$
\mathbf{k}_e = \sum_{g=1}^{n_g} w_g \cdot \mathbf{B}(\boldsymbol{\xi}_g)^\top \cdot \mathbf{D} \cdot \mathbf{B}(\boldsymbol{\xi}_g) \cdot \det(\mathbf{J}_g)
$$

其中：

| 符号 | 物理含义 | 尺寸 |
|------|----------|:---:|
| **B** | 应变-位移矩阵（形状函数导数） | 6 × 24 |
| **D** | 材料本构矩阵（弹性模量、泊松比） | 6 × 6 |
| **B**ᵀ**DB** | 单点刚度贡献 | 24 × 24 |
| w_g | 高斯积分权重 | 标量 |
| **k_e** | 单元刚度矩阵 | 24 × 24 |

关键观察：**每个高斯积分点的刚度贡献都被加权累加，一个不丢。** 单元的行为由所有积分点共同决定——不是最"硬"的那个点，不是简单平均，而是本构关系（**D**）和变形模式（**B**）的物理耦合。

### PointNet++ 的问题

PointNet++ 的 Transition Down 使用 Max Pooling 聚合邻域特征：

$$
\mathbf{f}_{\text{out}} = \max(\mathbf{f}_1, \mathbf{f}_2, \dots, \mathbf{f}_K)
$$

从刚度组装角度看，这等价于只保留邻域中"响应最大"的那一个点，丢弃其余 K−1 个点的所有贡献。在 FEA 中，这意味着：

$$
\mathbf{k}_e \stackrel{?}{=} \max_g \left( w_g \mathbf{B}_g^\top \mathbf{D} \mathbf{B}_g \right) \quad \leftarrow \text{物理上不成立}
$$

Avg Pooling 同样有问题——它假设所有邻域点等权贡献，忽略了节点间的选择性耦合（刚度矩阵是稀疏的：两个节点不相邻 = 刚度项为零）。

### Vector Attention 作为可学习的刚度组装

PT-HICnet 的核心假设是：**Vector Attention 的 q−k 减法天然对偶于位移梯度，整个 attention 聚合可以理解为一种学习的、稀疏的刚度组装**。

设邻域点 j 的特征为 f_j（编码了坐标、材料、厚度），中心点为 i。Vector Attention 的计算为：

$$
\mathbf{a}_{ij} = \text{softmax}\left( \gamma(\mathbf{q}_i - \mathbf{k}_j + \boldsymbol{\delta}_{\text{pos}}) \right)
$$

$$
\mathbf{f}_i^{\text{out}} = \sum_{j \in \mathcal{N}(i)} a_{ij} \cdot \mathbf{v}_j
$$

与 FEA 刚度组装的对应：

| FEA 刚度组装 | Vector Attention | 对应关系 |
|------|------|------|
| **B**(ξ_g) — 应变-位移 | **q − k** — 向量减法 | Δu → ∂u/∂x |
| **∂N/∂x** — 形状函数导数 | **δ_pos** — 相对位置编码 | 节点间距 → 应变分配 |
| **D** — 材料本构 | **v_j** (材料通道) — value 投影 | 材料响应 |
| softmax → 选择性耦合 | 稀疏刚度矩阵 (K_ij ≠ 0 iff 节点相邻) | 非零刚度项 |
| Σ_g → 加权累加 | Σ_j → attention 聚合 | 所有邻域点参与，按物理相关性加权 |
| **k_e** → 单元刚度 | **f_out** → 聚合特征 | 局部力学响应 |

**归纳偏置**：模型不需要从数据中"学到"位移梯度的概念——q−k 减法直接提供了它。不需要学到"材料决定响应强度"——19 维 Early Fusion 已经逐点拼接了材料本构。模型只需要学到：在碰撞这个物理过程里，哪些节点对之间的耦合是重要的。

### 四层物理尺度分层

Ball Query 半径按 FEA 特征尺度标定：

| 层 | 半径 | 覆盖范围 | FEA 对应 |
|------|:---:|------|------|
| L1 | 60 mm | 单焊点 / 加强筋 | 局部应力集中 |
| L2 | 150 mm | 子结构（A 柱、门槛） | 截面力传递 |
| L3 | 400 mm | 区域变形 | 弯折铰链 / 吸能区 |
| L4 | 1500 mm | 整车碰撞区域 | 全局应变能分布 |

每层的 Transition Down 做 FPS 降采样 + Ball Query 构图 + Vector Attention + 聚合，形成从局部到全局的层级化力学特征提取。

### 19 维逐点输入

每个 FEA 节点拼接为 19 维向量：

```
[ x, y, z (mm) | thickness (min-max normalized) | 15-ch material ]
                                              ├── 密度
                                              ├── 弹性模量
                                              ├── 泊松比
                                              ├── 屈服强度
                                              └── 11-pt 应力-应变曲线
```

坐标保留物理尺度（mm），不归一化到单位球——因为 Ball Query 半径依赖绝对物理尺度。材料通道按车型 Z-Score 归一化，训练时可通过 `--material_dropout_prob 0.15` 对材料通道施加 dropout 以缓解材料分布偏移（见 Phase 4 结论）。

## 环境依赖

Python 3.10+，CUDA 12.1+（GPU 训练推荐双卡 NVIDIA L20 或同等）。

```bash
pip install -r requirements.txt
```

## 快速开始

**训练（单 fold 示例）**：

```bash
python scripts/train_pt_hicnet.py \
  --config configs/default.yaml \
  --seed 42 \
  --film_mode global \
  --val_split 0.15 \
  --patience 50 \
  --restore_best \
  --test_vehicles JX65
```

**评估（LOCO-CV 聚合）**：

```bash
python scripts/aggregate_loco.py \
  --architectures all \
  --results_root experiments
```

**单 checkpoint 评估**：

```bash
python scripts/eval_pt_hicnet.py \
  --checkpoint experiments/<run_dir>/checkpoints/best_acc_model.pth \
  --config configs/default.yaml
```

## 实验结果

7-fold LOCO-CV，seed=42，val_split=0.15，patience=50，restore_best。指标为 accuracy_ratio（回归相对精度，±15% 误差容限）。5 辆正常车（C201, EP32, JX65, S50EVK, FX11）的 Test Mean：

| 架构 | 含义 | 5-Normal Test Mean | vs E0 |
|------|------|:---:|:---:|
| E0 | PointNet++ baseline | 71.21% | — |
| E1 | PN++ + Early Fusion (19ch) | 72.56% | +1.35pp |
| E2 | PT backbone, FiLM=none | 77.63% | +6.42pp |
| **E3** | **PT + FiLM global** | **78.39%** | **+7.18pp** |
| E4 | PT + FiLM deep（对照） | 76.75% | +5.54pp |

- **PT 骨架是唯一显著贡献**：E2−E0 = **+6.42pp**
- **FiLM global 跨车不显著**：E3−E2 = +0.76pp
- **E1 早融合在 PN++ 上收益微弱**：+1.35pp
- **E4 deep FiLM 系统过拟合**：Val 86.22%（最高）但 Test 低于 E3

> 详细 per-fold 数据、难车（CY02C/M6）分析见 `PT_HICnet_Report_Material.md` 与 `experiments/loco_cv/loco_cv_report.md`。
> 
> **Phase 4 难车诊断（收口）：** CY02C hard fold 主要退化与 material-vector distribution shift 有关。Hard-only 材料向量在 Ball Query 邻域富集 2×，PT Vector Attention 对未见材料过拟合。Material-channel dropout (prob=0.15) 在 held-out CY02C 单折上将 E3−E0 gap 从 −5.63pp 收窄至 −1.72pp，方向验证通过。Phase 5 将采用 `--material_dropout_prob 0.15` 作为标准训练配置。详见 `eval_protocol.md`、`experiments/error_analysis/hard_cars/hard_car_analysis.md`。

## 项目结构

```
PT-HICNET/
├── configs/                     # 实验配置
│   └── default.yaml             # 主配置（模型/训练/数据/损失超参）
├── data/                        # 数据目录（说明）
│   └── README.md
├── 车模型数据/                   # FEA 碰撞数据（.feather，不入库）
│   └── car{1..7}/               # 7 车型 × 6 batch/batch
├── feather/                     # 数据加载与 PointNet++ 基线
│   ├── data_utils/              # HICLoader（数据读取、早融合、归一化）
│   │   └── HICLoader_feather.py # 核心 DataLoader：8192 点采样 + 19ch 早融合
│   ├── model/                   # PointNet++ 基线模型
│   ├── material_lookup_by_vehicle.pkl  # 材料查找表
│   └── normalization_params.pkl       # Z-Score 归一化参数
├── materials/                   # 材料字典（外部来源）
├── models/                      # PT-HICnet 模型定义
│   ├── pt_hicnet.py             # 主模型：4 层 PT → 全局特征 + FiLM → HIC 回归
│   ├── point_transformer_block.py    # Vector Attention 核心：q−k+δ_pos
│   ├── point_transformer_layer.py    # Transition Down：FPS + BallQuery + PTBlock + Pool + Proj
│   ├── point_transformer_utils.py    # FPS / Ball Query / index_points
│   ├── film.py                  # FiLM 层：global / deep 双模式
│   └── losses.py                # Huber Loss（δ=150）+ KDE 权重（可选）
├── scripts/                     # 训练与评估脚本
│   ├── train_pt_hicnet.py       # 训练入口（支持 LOCO split、early stop、restore_best）
│   ├── eval_pt_hicnet.py        # 单 checkpoint 评估（per-vehicle 精度）
│   ├── aggregate_loco.py        # LOCO-CV 聚合（多架构 per-fold / per-vehicle / paired delta）
│   ├── run_ablation.py          # 消融实验批量启动
│   ├── eval_ablation.py         # 消融评估
│   ├── error_analysis.py        # per-sample 误差分析
│   └── error_analysis_hard_cars.py  # 难车诊断：尺度/材料/推断/delta
├── experiments/                 # 实验输出（不入库）
│   ├── loco_cv/                 # LOCO-CV 报告
│   │   └── loco_cv_report.md    # 五架构完整报告
│   └── pt_hicnet_loco_*/        # 35 个 LOCO 折目录（history.json + checkpoints/）
├── requirements.txt             # 依赖
├── .gitignore                   # Git 忽略规则
├── PT_HICnet_Report_Material.md # 中汽研技术素材（架构、物理对应、战绩）
├── context_summary.md           # 项目上下文摘要
├── plan.md                      # 当前阶段计划
└── CLAUDE.md                    # 本文件（Claude Code 项目配置）
```

---

