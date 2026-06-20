# PT-HICnet 实施计划（当前执行版）

> 更新日期：2026-06-20
> **Phase 3 全部完成。Phase 4 收口（主体诊断完成）。**
> Phase 5 触发条件：新数据到位 + `eval_protocol.md` QA checklist 通过后启动。

---

## 0) 当前里程碑状态

### Phase 1–3（全部完成 ✅）
- ✅ Phase 1：损失策略锁定 `delta=150 + 无 KDE`
- ✅ Phase 2：E0–E4 全部 15 seed 结果可比
- ✅ Phase 3 Early Stopping 验证通过（patience=50 + restore_best）
- ✅ AHW → NO-GO
- ✅ Phase 3a–3c：五架构 35/35 LOCO-CV 全量完成
  - E3 5-Normal 78.39% 当前最优；E2−E0 = **+6.42pp**（PT backbone 唯一显著贡献）
  - FiLM global +0.76pp 不显著；FiLM deep −0.88pp 有害
  - Hard car Gap >24pp — PT 路线在 CY02C/M6 上弱于 PN++ baseline

### Phase 4（收口，主体诊断完成 ✅）
- ✅ **4.1 冻结基线**：`eval_protocol.md` 补全聚合差异声明 + 具体验收门槛；`aggregate_loco.py` 表头已修
- ✅ **4.2 难车诊断**：full inference 完成（exit=0）；**尺度假说排除**（hard car bbox ≈ normal）；**材料漂移成立**：mat_02 raw PR 0.315→0.289（Cohen d=−1.09），mat_03-08 应力-应变 0.001 段 raw 差 48–78 MPa；ID overlap 低（6/28）但向量 overlap 高（14/18），M6 材料全在 normal 中出现过；E0 vs E3 paired delta：246 负 219 正 — CY02C 系统性变差、M6 tail/outlier 混合；M6 单样本 HIC=140k 驱动 MSE 爆炸
- ✅ **4.3 验收门槛**：已写入 `eval_protocol.md`（E2−E0 ≥ +5.0pp 5N / ≥ +3.0pp 7C，E3−E2 ≥ +1.5pp + 4/5，Hard Gap ≤ 20pp 或 ↓5pp）
- ✅ **4.4a material-aware failure localization（CPU 诊断）**
  - 已给每个 CY02C 样本打标：hard-only material vector 标志、hard-only 节点占比、hard-only 节点与 HIC point 距离
  - 相关分析完成：连续暴露强度与 E3−E0 delta 负相关（r_frac=-0.314, r_top32=-0.245）
  - 结论：材料 OOD 是合理方向，优先做不泄漏 CY02C test-domain 的 material regularization
- ✅ **4.4b material dropout pilot 验证通过**
  - E3 global + material_dropout_prob=0.15，CY02C held-out 单折（seed=42，patience=50，restore_best）
  - 结果（best-val-acc 口径）：
    | 配置 | CY02C Test Acc | E3−E0 Δ |
    |------|:---:|:---:|
    | E0 baseline | 64.23% | — |
    | E3 global | 58.60% | −5.63pp |
    | E3 + matdrop 0.15 | 62.51% | −1.72pp |
  - **结论**：material dropout 将 E3 相对 E0 的 gap 从 −5.63pp 收窄到 −1.72pp（追回约 3.9pp），且 Val Acc 未受损（84.92% → 84.88%），Test MSE 减半。材料正则化方向成立，`material_dropout_prob=0.15` 写入 Phase 5 标准配置。
  - ⚠️ 叙事口径：material dropout 显著收窄了 PT 在 CY02C 上的退化，但未完全追平 E0。不声称"解决"，只声称"方向验证通过，证据链足够强，足以进入 Phase 5"。
- ➡️ **4.5 分档实验（跳过）**：不再单独扩到 M6/normal folds。Material dropout 全量验证合并入 Phase 5。
- ⏳ **数据清单**：明确新车型数量、预期样本量、HIC 分布概况（待新数据到位后补充）

### 暂停项
- 🚫 E5 physical pooling：等难车诊断结论后再定方向
- 🚫 Multi-seed：Phase 5 第三档执行

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
- [x] 难车 error analysis 模板（→ Phase 4）
- [x] **固化 config + eval 脚本版本标签**（→ Phase 4 已完成）

---

## 4) Phase 4: 数据扩充与管线固化（3c 完成后）

### 目标
在新车数据到位前，把所有"跑一次就够"的基础设施做好，确保新旧数据可比。

### DoD

- [x] **管线固化**：freeze 一版 `configs/default.yaml` + eval script + 版本标签 `v1.0-pre-expansion`；`material_dropout_prob=0.0` 保证基线复现不变
- [x] **失效分析模板**：CY02C/M6 per-sample error analysis → `scripts/error_analysis_hard_cars.py`
- [x] **验收门槛定义**：
  - E2−E0 ≥ +5.0pp on 5-normal Test Mean
  - E2−E0 ≥ +3.0pp on 7-car Test Mean
  - E3−E2 ≥ +1.5pp and positive on at least 4/5 normal vehicles before claiming FiLM
  - Hard-car Gap ≤ 20pp, or improve by ≥5pp vs current E3 hard gaps
- [x] **数据 QA checklist**：新增材料 z-score/raw-unit 分布、MID/材料向量覆盖、bbox 尺度、HIC tail、HIC=0 排除样本检查
- [ ] **数据清单**：明确新车型数量、预期样本量、HIC 分布概况
- [x] **评估口径文档**：best-checkpoint vs stop-epoch、Val/Test/Gap 三栏定义、JX65 soft check 规则 —— `eval_protocol.md`

---

## 5) Phase 5: 新全量数据 Multi-seed 最终统计

### 触发条件
新数据到位 + `eval_protocol.md` Expanded-Data QA Gates 全部通过后启动。

### 配置
- 主架构 E3：`material_dropout_prob=0.15` + `material_jitter_std=0.0`（Phase 5 标准配置）
- 对照架构 E2：同配置，验证 material dropout 在无 FiLM 场景下的效果
- E0 baseline：不变（PN++ 无材料通道 dropout 概念）
- 可选项：E3 + material_dropout_prob=0.0 作为消融对照（1 seed × 全车型）

### 范围
- E0/E2/E3: 3 seeds (42/3407/2026) × 全车型
- E1/E4: 视资源决定全量或抽样（基于 Phase 3c 单 seed 结论）
- 统一输出：mean ± std × seed、per-car ranking、val/test gap 跨种子统计
- 验收：按 `eval_protocol.md` 5 条硬门槛逐项对表

---

## 6) 风险与对策（当前版）

- **过拟合风险（高）**
  - 对策：early stopping + 正则扫描 + 降低模型复杂度（必要时）
- **泛化风险（高）**
  - 对策：LOCO-CV 替代单车测试作为主结论依据
- **实验口径风险（中）**
  - 对策：报告中固定统一口径；附录列出脚本差异
- **运行环境风险（中）**
  - 对策：固定脚本入口、固定 run 命名规则、耗时实验统一放在 `screen` 中运行并保留日志、保留清理策略

---

## 7) 立即可执行命令（备忘）

```bash
# 耗时任务统一进入 screen
screen -S pthicnet_phase4_hardcars
/data1/user/yikun/.conda/envs/dl/bin/python -u scripts/error_analysis_hard_cars.py \
  --architectures E0 E2 E3 E4 \
  --vehicles CY02C M6 \
  --run_data_diagnostics \
  --run_inference \
  2>&1 | tee experiments/error_analysis/hard_cars/run_inference.log

# 已有 hard_car_per_sample.csv 后，刷新 CPU 诊断/报告即可
/data1/user/yikun/.conda/envs/dl/bin/python -u scripts/error_analysis_hard_cars.py \
  --architectures E0 E2 E3 E4 \
  --vehicles CY02C M6 \
  --run_data_diagnostics \
  --reuse_inference_csv

# 多架构聚合
python scripts/aggregate_loco.py --architectures all --results_root experiments
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
