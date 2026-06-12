# ⛔ DEPRECATED (2026-06-04) — 消融实验启动阻塞项

> 本文档已归档。原内容为 2026-05-29 的阻塞项追踪（B1-B3 数据/增强路径问题，N1-N3 Ball Query 半径/BN/E2 缺失）。
> 所有阻塞项均已解决。当前状态请参见 `plan.md` 和 `context_summary.md`。

---

# 消融实验启动阻塞项（历史）

> **状态更新 (2026-05-29)：三个阻塞项已全部解决。** E3 (film_global, seed42) 成功完成 200 轮训练。
> E2 (film_none, seed42) 训练被中断（数据加载完成后进程终止，无 history.json）。
>
> E3 实验暴露了新的严重问题（详见下方「⚠️ E3 实验发现的新阻塞项」）。

---

## B1 — 数据资产路径配置  [状态: ✅ 已解决]

### 问题
`configs/default.yaml` L7-8 指向 `data/train` / `data/test`，但 `data/` 目录下只有 `README.md`。

### 解决方案
`configs/default.yaml` 已配置 `data_root: "车模型数据"`，`collect_data_files()` 中 `train_data_dir=""` 时直接搜索 `data_root`。E3 实验成功加载了 6 组训练车 + 1 组测试车（JX65）共 42 个 `.feather` 文件。

---

## B2 — Material .pkl 路径 YAML→DataLoader 打通  [状态: ✅ 已解决]

### 解决方案
`scripts/train_pt_hicnet.py` L278-284 在 import HICDataLoader 之前通过环境变量 `PT_HICNET_MATERIAL_LOOKUP_PATH` / `PT_HICNET_NORMALIZATION_PARAMS_PATH` 注入 `.pkl` 路径。E3 实验日志确认各车材料查找表均加载成功。

---

## B3 — `train_pt_hicnet.py` 补齐 3D 数据增强  [状态: ✅ 已解决]

### 解决方案
`scripts/train_pt_hicnet.py` L154-160 已加入与 baseline 一致的三段增强（random_point_dropout / random_scale_point_cloud / shift_point_cloud），仅作用于 XYZ 通道。

---

## ⚠️ E3 实验发现的新阻塞项（替换原 B1-B3）

### N1 — Ball Query 半径与点云尺度失配（最高优先级）

**现象：** E3 训练 200 轮后，Test MSE 从 epoch 1 的 842K 恶化到 epoch 200 的 3,050K，且全程剧烈震荡（608K ↔ 5.3M）。Train Loss 则正常下降（16K → 7K），呈现严重的 Train-Test 背离。

**根因推断：** `pt_radius = [0.2, 0.4, 0.8, 1.2]` 依赖点云坐标单位。若 FEA 模型单位为 mm（汽车碰撞仿真常见），则 radius=0.2mm 远小于典型网格间距（~10-30mm），导致 Ball Query 几乎无有效邻居，Vector Attention 退化为恒等映射。

**修改建议：**
1. 在 DataLoader 中或训练脚本开头，统计一批点云的平均点间距（≈ `diagonal / sqrt(N)`），据此自动计算或至少打印参考半径
2. 若点云单位为 mm，建议初始半径设为 `[30, 60, 120, 240]` 或类似量级（3-5 倍点间距）
3. 在 `TransitionDown.forward()` 中添加有效邻居计数日志，确认 `nsample=32` 的邻居中有 >50% 落在查询球内

### N2 — 输入投影层缺少归一化

**现象：** `input_proj = Conv1d(19, 64, 1)` 直接接收 XYZ（可能 10³-10⁴ 量级）、thickness（[0,1]）、material_props（Z-score ~[-3, 3]）三组不同量级的数据。

**修改建议：** 在 `input_proj` 后添加 `BatchNorm1d(64)` 或 `GroupNorm`，消除通道间量级差异。

### N3 — E2 对照实验缺失

E3（film_global）的直接对照 E2（film_none）未产生结果文件，导致无法判断 FiLM 是否带来任何收益。需要优先补跑 E2 seed42。
