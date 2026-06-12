# PT-HICnet 实施计划（当前执行版）

> 更新日期：2026-06-05
> **Phase 3c 完成！** 五架构全量 LOCO-CV (35/35 folds) 就位。下一阶段：Phase 4 管线固化 + 失效分析。

---

## 0) 当前里程碑状态

- ✅ Phase 1 已完成：损失策略锁定为 `delta=150 + 无 KDE`。
- ✅ Phase 2 已完成：E0/E1/E2/E3/E4 全部 15 个 seed 结果可比。
- ✅ Phase 3 Early Stopping 验证通过（patience=50 + restore_best）
- ✅ Phase 3 误差分析完成 → 暴露 Adult 高 HIC 弱点
- ✅ AHW (Adult HIC Weighted) → NO-GO
- ✅ **E3 LOCO-CV 7 折完成**
- ✅ **E0+E2 LOCO-CV 7/7 完成**
- ✅ **Phase 3c: E1+E4 LOCO-CV 7/7 完成** — 五架构全量 LOCO-CV (35/35 folds) 就位
  - E4 Val 最高 (86.22%) 但 Test 不如 E2 (76.75% vs 77.63% 5-normal) — deep FiLM 是过拟合器
  - E1 ≈ E0 (69.24% vs 69.02%) — EarlyFusion 在 PN++ 上跨车无效
- ⏳ 难车 error analysis：CY02C/M6 失效分析模板
- ⏳ 管线固化：freeze config + eval script + 版本标签
- ⏳ 定义新数据验收门槛（E2−E0 ≥ +Xpp 等）
- 🚫 **Multi-seed: 暂缓至 Phase 5**（新数据并入后一次性做最终统计）
- ⏳ Phase 4: 数据扩充与清洗（新增车型）
- ⏳ Phase 5: 在新全量数据上 multi-seed 最终统计

---

## 0.1) AHW 决策门槛（go/no-go）

> Baseline: E3-ES 三 seed error_analysis 均值（详见下方）

| 指标 | E3-ES Baseline | AHW 通过门槛 | 说明 |
|------|:---:|:---:|------|
| **Adult MSE** | **340,703** | **≤ 289,598** (↓15%) | 主指标，必须达标 |
| **>2k Bucket Acc** | **64.93%** | **≥ 69.93%** (+5pp) | 辅指标，注意 n=4 高方差 |
| **Overall Best Acc** | **84.20%** | **≥ 83.70%** (≤0.5pp↓) | 约束条件：不可明显恶化 |

**判定规则：**
- Adult MSE ↓ ≥ 15% **且** Overall Acc ≥ 83.70% → **GO**，AHW 写入主模型，接 LOCO-CV
- Adult MSE ↓ ≥ 15% **但** Overall Acc < 83.70% → 权衡（可能降 weight）
- Adult MSE ↓ < 15% → **NO-GO**，直接用 E3-ES 接 LOCO-CV
- >2k Bucket Acc 因 n=4，作为参考不硬卡

**E3-ES per-seed baseline（error_analysis）：**

| Seed | Overall Acc | Adult MSE | Child MSE | >2k Acc | >2k Adult Acc |
|------|:---:|:---:|:---:|:---:|:---:|
| 42 | 83.88% | 103,081 | 71,254 | 64.37% | 69.96% |
| 3407 | 84.02% | 753,995 | 40,271 | 57.82% | 40.89% |
| 2026 | 83.13% | 165,034 | 46,364 | 72.60% | 73.64% |
| **Mean** | **83.68%** | **340,703** | **52,630** | **64.93%** | **61.50%** |

⚠️ Adult MSE 跨 seed 方差极大（103K–754K），因 >2k bucket 仅 4 样本 (2 Adult)。AHW 对比时优先看 seed-for-seed 而非均值。

---

## 1) 已完成（Done）

### 1.1 数据与训练链路

- [x] `HICLoader_feather` 早融合输出稳定（`fused_input [N,19]`）
- [x] `train_pt_hicnet.py` 支持 smoke test、日志、checkpoint、config hash
- [x] baseline 脚本与 PT 脚本都可稳定启动

### 1.2 架构与配置一致性

- [x] PT 主线完成：`PointTransformerBlock` + `TransitionDown` + `PT_HICnet`
- [x] Ball Query 半径修复：`[60, 150, 400, 1500]`
- [x] 输入投影加 `BatchNorm1d`
- [x] PT 落盘移除误导字段 `model.ablation_mode`
- [x] `--ablation_mode` 修复为 **CLI 显式优先于 YAML**

### 1.3 记录与实验管理

- [x] 无效实验已归档至 `experiments/_archive_debug/`
- [x] W&B 无效 run 已清理，当前保留为报告可用集合

---

## 2) 当前结果快照（用于决策）

### 2.1 已验证结论（Phase 2 JX65 单车 → Phase 3 LOCO-CV 修正）

**Phase 2 (JX65 单车):**
- E1（PN2++ 早融合）不如 E0（-0.61pp），但 E2（PT + 相同早融合）比 E1 高 +2.02pp。**EarlyFusion 只有在 Vector Attention 配合下才有效。**
- E2（PT no FiLM）明显优于 E0（+1.41pp）。
- E3（PT + FiLM global）JX65 最佳（+3.19pp vs E0）。
- E4（PT + FiLM deep）退回 E2 水平（-1.81pp vs E3）。

**Phase 3 LOCO-CV 修正（E0/E2/E3 7 折 × seed42，2026-06-04）:**
- **PT backbone 是真正增益：** 正常车 E2−E0 = +6.4pp（远大于 Phase 2 的 +1.41pp，因为 LOCO 下 baseline 更弱）
- **FiLM global 跨车收益 ≈ 0：** 正常车 E3−E2 = +0.8pp（统计不显著），Phase 2 的 JX65 +1.78pp 是假阳性
- **PT 在难车（CY02C/M6）上反而劣于 PN++ baseline：** E0 > E2 > E3，更强模型 = 更难泛化到异分布车
- **论文叙事调整：** 主 claim 从 "FiLM improves HIC" → "PT backbone is primary cross-car gain; FiLM gain is data-distribution-dependent"

### 2.2 主要风险

- LOCO 下 normal cars Test Std ~2pp（可接受），但 hard cars Gap >24pp（严重域偏移）
- 当前仅 seed=42；multi-seed 可能改变 E3 vs E2 的 0.8pp 微弱差异方向
- E1/E4 LOCO 运行中（~12h），若 E1 跨车也不如 E0，则 EarlyFusion 在 PN++ 上的负面结论更稳固

---

## 3) Phase 3 执行计划（3a → 3b → 3c）

### Phase 3a: E3 LOCO-CV ✅
- [x] E3-ES 验证通过（patience=50 + restore_best）
- [x] 误差分析完成（暴露 Adult 高 HIC 弱点）
- [x] AHW 验证 → NO-GO（Overall Acc 暴跌 11pp）
- [x] LOCO-CV 代码落地
- [x] E0/E1 feather 脚本 LOCO-CV 支援完成
- [x] E3 7 折全跑完成 + aggregate（Val 84.8% ± 0.6pp, Test 73.0% ± 8.6pp）
- [x] E0/E1 smoke 通过验证

### Phase 3b: E0 + E2 LOCO-CV ✅
- [x] E0 7 折（PN++ baseline）→ Test 69.0% ± 4.1pp
- [x] E2 7 折（PT FiLM none）→ Test 72.7% ± 8.7pp
- [x] 回答：PT 骨架正常车稳定 +6.4pp vs baseline；FiLM 收益跨车不显著（+0.8pp）

### Phase 3c: E1 + E4 LOCO-CV ✅
- [x] E1 7/7 完成 — Test 69.24% ±6.89pp (≈E0), EarlyFusion 在 PN++ 上跨车无效
- [x] E4 7/7 完成 — Test 72.71% ±7.43pp, Val 最高(86.22%)但 5-normal(76.75%) < E2(77.63%)
- [x] aggregate_loco.py 多架构升级完成（向后兼容 + 三表输出）
- [x] loco_cv_report.md 五架构版完成

---

## 4) DoD（阶段验收标准）

### Phase 3a DoD ✅

- [x] E3-ES 过拟合缓解（Best→Stop Δ 降至 -3.55pp，原 -4.27pp）
- [x] E3 LOCO 7 折全部跑通，跨车均值 ± std 产出
- [x] E0/E1 smoke 通过（history.json + val 早停 + best_acc_model.pth）

### Phase 3b DoD ✅

- [x] E0 + E2 各 7 折全部跑通
- [x] E3 vs E0 per-vehicle paired Δ 产出（PT backbone +6.4pp normal cars）
- [x] E3 vs E2 per-vehicle paired Δ 产出（FiLM global +0.8pp，不显著）

### Phase 3c DoD ✅

- [x] E1 + E4 各 7 折全部跑通 (E1 69.24%, E4 72.71%)
- [x] 5 架构全 LOCO 排名表产出（`aggregate_loco.py --architectures all`）
- [x] `loco_cv_report.md` 五架构版完成
- [ ] 难车 error analysis 模板（→ Phase 4）
- [ ] 固化 config + eval 脚本版本标签（→ Phase 4）

---

## 4) Phase 4: 数据扩充与管线固化（3c 完成后）

### 目标
在新车数据到位前，把所有"跑一次就够"的基础设施做好，确保新旧数据可比。

### DoD

- [ ] **管线固化**：freeze 一版 `configs/default.yaml` + eval script + 版本标签（如 `v1.0-pre-expansion`）
- [ ] **失效分析模板**：CY02C/M6 per-sample error analysis → 可复用脚本，新数据到了直接套
- [ ] **验收门槛定义**：
  - E2−E0 (PT backbone gain) ≥ +Xpp on full-car mean（X 基于当前 5-normal-car 给出下限）
  - Val Std ≤ Ypp（跨车收敛一致性）
  - Hard-car Gap ≤ Zpp（不过度恶化）
- [ ] **数据清单**：明确新车型数量、预期样本量、HIC 分布概况
- [ ] **评估口径文档**：best-checkpoint vs stop-epoch、Val/Test/Gap 三栏定义、JX65 soft check 规则 —— 写入单页 `eval_protocol.md`

---

## 5) Phase 5: 新全量数据 Multi-seed 最终统计

### 触发条件
Phase 4 数据扩充完成后启动。

### 范围
- E0/E2/E3: 3 seeds (42/3407/2026) × 全车型
- E1/E4: 视资源决定全量或抽样（基于 Phase 3c 单 seed 结论）
- 统一输出：mean ± std × seed、per-car ranking、val/test gap 跨种子统计

---

## 6) 风险与对策（当前版）

- **过拟合风险（高）**
  - 对策：early stopping + 正则扫描 + 降低模型复杂度（必要时）
- **泛化风险（高）**
  - 对策：LOCO-CV 替代单车测试作为主结论依据
- **实验口径风险（中）**
  - 对策：报告中固定统一口径；附录列出脚本差异
- **运行环境风险（中）**
  - 对策：固定脚本入口、固定 run 命名规则、保留清理策略

---

## 7) 立即可执行命令（备忘）

```bash
# 检查 Phase 3c 进度
ps aux | grep -E "train_pt|train_reg" | grep -v grep
screen -ls | grep -E "E[14]"

# E1/E4 完成后：多架构聚合
python scripts/aggregate_loco.py --results_root experiments  # 需先升级多架构支持
```

---

## 8) 后续展望：可视化与物理解释（论文核心 evidence）

Phase 4/5 把数据与统计做完后，下一步是**证明网络学到了物理上有意义的东西**：

- **Attention 热力图**：4 层点云渲染，验证高 attention 区域是否落在焊点/加强筋/碰撞区
- **量化验证**：attention vs material gradient 相关性 + 关键点消融（因果证据）
- **FiLM 调制分析**：同车成人 vs 儿童 feature Δ
- **逐层物理尺度对应**：L1(60mm) 单焊点 → L4(1500mm) 整车应变能
- **难车诊断**：CY02C/M6 的 attention 分布是否与 easy cars 不同（结构差异 or 标签噪声?）

目标：不只是说"我们用了 Vector Attention"，而是**证明它确实在学物理上有意义的东西**——这是审稿人最在意的问题。

---

## 9) 计划维护规则

- 每次新增实验后，优先更新 `context_summary.md` 的结果区。
- `plan.md` 只保留“下一步与验收标准”，避免堆叠历史细节。
- 若状态变化（如 E4 完成、LOCO-CV 启动），当天同步更新本文件顶部状态。
