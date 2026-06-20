# PT-HICnet — Point Transformer for HIC Surrogate Modeling

基于 Point Transformer 的汽车碰撞 HIC（Head Injury Criterion）深度学习代理模型。

## 核心思路

传统 FEA 碰撞仿真耗时长（单车型 ~4–8 小时），PointNet++ 等点云回归模型使用 Max Pooling 聚合邻域信息，本质上是"黑箱函数拟合器"——它无法显式感知邻域特征梯度，因此泛化能力受限。

PT-HICnet 用 **Vector Attention（q−k 向量减法）替换 Max Pooling**。q−k 天然对偶于连续介质力学的**位移梯度**（应变），#delta#_pos 编码了**形状函数导数**（B 矩阵），softmax 在邻域内实现了**选择性节点耦合**（非零刚度项）。这让模型结构本身偏置向学习"离散化的应变-位移-材料耦合"，输出头用于标量 HIC 回归。

19 维 Early Fusion 将 FEA 节点坐标（mm，保留物理尺度）、板厚（per-sample min-max）、15 维材料本构（密度、弹性模量、泊松比、多点应力-应变曲线）逐点拼接注入。四层 Ball Query 半径标定至 FEA 物理尺度 [60, 150, 400, 1500] mm，分别覆盖焊点/加强筋 → 子结构 → 区域变形 → 整车碰撞区域。

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
│   └── error_analysis.py        # per-sample 误差分析
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

