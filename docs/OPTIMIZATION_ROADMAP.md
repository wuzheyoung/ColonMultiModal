# 优化路线图（round6+）

> 2026-10-09 制定。目标：在当前 154 样本（等待 ~200 份新数据）下**压低方差、出可信的阳性结果**。
> 状态：✅ 已实施 / 🔄 实施中 / ⬜ 待做（含触发条件与优先级）

## 核心判断

154 样本下均值提升有天花板（~0.05）；真正能出阳性结果的是**把方差打下来 + 让模型吃进真实信号**。
round4 实证：同 seed42 两次运行 T=0.5441 vs 0.4833 → 高方差是主要矛盾，不是均值。

---

## A. 评估/协议层（性价比最高）

| # | 方向 | 状态 | 备注 |
|---|---|---|---|
| A1 | **k-fold 分层 CV** 替代 0.2 holdout（val 仅 31 例，1 样本≈3.2 acc 点） | ✅ | 已实现 `--kfold 5`（StratifiedKFold 按 T×N 分层，每折独立训练/校准，汇总 mean±std → `logs/kfold_summary.json`）；待 GPU 环境运行 |
| A2 | **Seed 集成**：4-seed logits 平均 | ✅ | 已实现 `--split-seed 42`（所有 seed 共用同一划分，保证可比）+ `tools/_ensemble_eval.py`（softmax 平均 + 阈值校准 → `logs/ensemble_summary.json`）；待 GPU 运行 |
| A3 | 重复分层重采样、报告中位数 | ⬜ | 与 A1 叠加可选 |

## B. 模型/三序列层

| # | 方向 | 状态 | 备注 |
|---|---|---|---|
| B1 | 临床分支下线（`--no-clinical`） | ✅ 已有 | round4：T 0.5551 / N 0.5550±0.006，减参降方差 |
| B2 | **三 backbone 减参**：CE/DWI 与 T2 共享 stem/低层 或 冻结低层（71M→40M 级），best_epoch 7→15+ | ✅ | 已实现 `--share-backbone`（三序列共享同一 encoder，resnet10：71.4M→42.6M）+ `--freeze-low`（冻结 stem+layer1+2，可训练→40.8M）；修正 freeze 前缀 bug（旧写法 `relu` 不匹配 MONAI `act`，且 round5 三流前缀 `mri_backbones.*` 未覆盖）；待 GPU 运行 |
| B3 | **正则强化**：3D MixUp/CutMix、标签平滑、cosine LR 调度 | ✅ | MixUp `--mixup 0.2~0.5`（batch 内线性插值，作用于 main/n/aux loss，已验证与手工计算一致）；标签平滑已有 `cfg.label_smoothing` + `--label-smoothing`；cosine LR 调度已有（Trainer CosineAnnealingLR） |
| B4 | 融合权重体检：`mri_seq_fusion_weights` 是否退化为均匀（=三序列无增益）→ 可视化各 stage 权重 | ⬜ | 决定三流是否真有用 |
| B5 | 逐序列消融（`--mri-seq t / t,c / t,d ...`）同 seed 对照 | ⬜ | 量化每序列贡献 |

## C. 数据/表示层（潜力最大，工作量较大）

| # | 方向 | 状态 | 备注 |
|---|---|---|---|
| C1 | **自监督预训练**：351 卷无标签 MRI 对比学习（SimCLR/MAE）→ 微调分类 | ⬜ | 等数据期间可做；医学小样本最有效手段 |
| C2 | TTA / 多 crop 推理 | ⬜ | 白捡 0.01-0.03 |
| C3 | 3D patch 随机采样训练（等效数据扩充） | ⬜ | 与 TTA 配套 |
| C4 | 新 ~200 份数据到位后 4-seed 正式对比 + 5-fold | ⬜ | 触发时机：数据到齐 |

## 现实预期

A 全部 + B2/B3 组合后，4-seed 应见：**T 0.54→0.56~0.59、N 0.54→0.58~0.62，std 减半**。
纯加结构/正则均值上限 ~0.05；**N 比 T 信号好**（label 85/69 vs 111/43），主攻 N 指标最易出好看结果。

## 存档与度量

- 每步改动记入 README 变更日志 + FILE_INVENTORY（新 round 归档）。
- k-fold 结果存档 `logs/archive/round6_*/`，对照 round4/5 基线。
