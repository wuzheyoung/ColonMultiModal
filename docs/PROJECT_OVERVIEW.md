# ColonMultiModal 项目总览

> **阶段快照：2026-09-29 / round4 归档后** — 本文记录该阶段状态，内容不代表当前状态。
> 当前状态见 [README.md](../README.md) 与 [FIXES_2026-11-20.md](FIXES_2026-11-20.md)；数据规模与口径已变更，文中数字不可直接引用。
> 2026-11-21 已删除全部实验权重（`checkpoints_*` / `archive` / `logs`），涉及 `.pth` 的命令需重跑生成。
> 相关快照：[文件清单](FILE_INVENTORY.md) · [项目总览](PROJECT_OVERVIEW.md) · [进度汇报](PROGRESS_REPORT_HOS.md) · [优化路线图](OPTIMIZATION_ROADMAP.md)

---

## 1. 项目定位

结直肠癌（CRC）术前多模态分期系统：融合 **CT + MRI 影像 + 临床血液指标**，双任务预测病理分期：

| 任务 | 标签 | 类别 | 分布（round4 时期 154 有效样本；当前见 README） |
|---|---|---|---|
| 主任务 T | `label_T` | pT1-2 (0) vs pT3-4 (1) | 43 : 111 |
| 次任务 N | `label_N` | pN0 (0) vs pN+ (1) | 85 : 69 |

- **数据模态**：155 行 csv（154 有效）：双模态 39 / 仅 CT 24 / 仅 MRI 91，缺失模态用 imputer 补全
- **核心指标**：验证集 macro-F1（T 主 / N 次），阈值校准后报告；多 seed 取均值±std
- **基线参考**：T 四分类不可行（T1:6/T2:37/T3:106/T4:5）；N 三分可作扩展分析

## 2. 目录地图

```
ColonMultiModal/
├── config/              配置类（default → hos 继承覆盖）
│   ├── default.py       TCGA 时代基础配置
│   └── hos.py           当前 Hos 多模态配置（resnet18/multi/双头/冻结 layer3）
├── data_loader/
│   └── loader.py        确定性预处理(CT/MRI) + 磁盘缓存(_DetDiskCache/cache_hos) + 随机增强
├── models/
│   ├── backbone.py      ResNet3D(resnet10/18/50, MONAI ResNetFeatures) / Swin3D
│   ├── fusion.py        SelfAttentionEncoder / ResidualMLPImputer / BSAFusion(cross-attn)
│   ├── network.py       CRCMultiModalNet（双 backbone + 缺失补全 + 多尺度融合 + 双头）
│   └── losses.py        FocalLoss
├── trainers/
│   └── trainer.py       Trainer：模态 dropout / 多任务 loss / N-best 独立跟踪 / 阈值校准接口
├── utils/
│   ├── metrics.py       compute_metrics + calibrate_binary_threshold
│   ├── logger.py        setup_logger
│   └── ...              可视化/报告辅助（数据构建/格式转换/扫描工具实际在 tools/）
├── tools/
│   ├── build_det_cache.py   确定性预处理磁盘缓存构建（cache_hos/*.npy）
│   ├── train_*.py           各类实验脚本（radiomics / MIL / 2.5D / ROI-DL / CRCFound）
│   └── _probe_*.py          临时验证探针（可删）
├── train_hos_seeds.py   多 seed 训练 + 校准 + 汇总（当前主入口）
├── data/                hos_pairs.csv + 各 seed 分层划分（hos_{branch}_{train/val}_s{seed}.csv）
├── cache_hos/           确定性预处理缓存（当前 544 个 .npy）
├── checkpoints_hos_full/  空（权重已于 2026-11-21 删除）
├── plots_hos_full/      训练曲线 / 混淆矩阵图
├── logs/                训练日志 + eval_report_*.json/txt（300+ 文件）
├── results/round9/      每轮实验归档 run_NN_<tag>/（编号原子占位）
├── archive/             旧分支/早期权重/TCGA 时代产物
│   ├── scripts_legacy/scripts_legacy/train_hos.py   旧单 seed 入口
│   ├── checkpoints_early/checkpoints_hos/           round5 及更早权重（已删除）
│   └── tcga/_archive_tcga/                          TCGA 外部验证数据
```

> 文件级逐文件索引（分类/状态/可清理项）见 [`FILE_INVENTORY.md`](FILE_INVENTORY.md)。

## 3. 数据管线（data_loader/loader.py）

```
CSV(row) ──ct_available?──> 有: CT 确定性预处理 ──┐
                          无: zero_volume(1,96³) ─┤──> 随机增强(train only) ──> batch
                            MRI 同构               ┘        ↓
                                                       disk cache (cache_hos)
```

- **CT 确定性管线**：LoadImage → EnsureChannelFirst → Spacing(1mm) → HU 窗裁剪(-150~250) → CropForeground → Resize(96³) → 前景 z-score
- **MRI 确定性管线**：同上，无 HU 窗（直接归一化）
- **磁盘缓存** `_DetDiskCache`：key=sha1(path+mtime+spatial_size)，版本化 .npy，原子写盘，线程安全。194 卷命中 100%，epoch ~7s
- **随机增强**（train only，逐模态独立，作用于缓存输出）：Flip(axis0/1) + Affine(rotate0.15rad/prob0.7/shear0.05) + Zoom(0.9–1.1) + ShiftIntensity(0.15) + ScaleIntensity(0.2) + GaussianNoise(std0.02) + CoarseDropout(3×12³)
- **样本输出**：ct/mri(1×96³), clinical(7维), label_T, label_N, 可用性 mask

## 4. 模型架构（models/network.py）

```
CT ResNet3D ──> 多尺度特征[24³,12³,6³,3³]
MRI ResNet3D ─> 多尺度特征[...]
      ↓ 模态内 Self-Attention（提纯）
      ↓ 缺失模态 Imputer（MLP 跨模态生成 + mask token）   ← 重建 loss 正则
      ↓ BSAFusion（多尺度 cross-attention, num_fuse_stages=2: 12³+6³）
      ↓ 拼接临床向量(7) + 池化
      ↓ 双头分类: main(T, 逆频率加权) + N 头
```

- **backbone**：MONAI `ResNetFeatures`：resnet10=42.6M / resnet18=79.9M / resnet50；或 SwinUNETR encoder
- **fusion_mode**：`multi`（多尺度 cross-attn 融合）/ `single`（单尺度）
- **缺失模态**：`modality_dropout={ct_only:0.20, mri_only:0.15, both:0.65}` 训练时随机采样 + imputer 补全
- **损失**：主 T CE（label_smoothing 0.1, 逆频率权重）+ aux N CE + 重建 MSE/cosine（`loss_weights={aux:0.3, rec:0.5}`，已验证不可削减）

## 5. 训练流程（train_hos_seeds.py）

```
main --seeds 42,0,1,7 --epochs 30 [--fusion multi|single] [--backbone resnet18|resnet10]
  └ run_seed(s): 分层划分(按 T_N 组合, random_state=seed)
        → Trainer.fit:
            - best_state_dict (T) / best_state_dict_n (N) 独立跟踪
            - 每次 save: best_model[_n]_hos_s{seed}.pth
        → predict_proba(best_key="t"/"n") 阈值扫描校准 (0.05~0.95)
        → seed_summary.json + 控制台汇总表
```

- 默认配置：resnet18 + multi + 冻结 conv1~layer3（63.5M 可训练）；`--backbone resnet10` 时冻结为空（全量 42.6M 可训练）
- **N-best 独立跟踪**：校准 N 用 N-best 权重，不被 T 的 best epoch 绑架（round3 修复）

## 6. 实验脉络

| Round | 配置 | seed | 结果（校准 macro-F1） | 结论 |
|---|---|---|---|---|
| R1 | resnet18 multi | 多 | N mean 0.5198 | 基线 |
| R3 | resnet18 multi 30ep | 42/0/1 | T 0.4833/0.6086/0.4973；N 0.6039/0.6744/0.4423（mean 0.5735±0.097） | 修复 N-best；方差大 |
| 消融 | resnet18 **single** | 42 | T 0.5141（+0.031）、N 0.5643（-0.040） | single 益 T 损 N |
| R4 | **resnet10** multi 不冻结 | 42/0/1/7 | T mean 0.5405；N mean 0.5437±0.057 | N 更稳但降精度；方差是主要矛盾 |

**关键事实**：同一 seed42+resnet10 两次运行 T=0.5441 vs 0.4833 —— 154 样本训练轨迹对初始化/顺序极敏感。

## 7. 当前结论与下一步

1. **主要矛盾**：154 样本上的高方差，非 backbone 参数量
2. **已就位**：确定性缓存（快）+ 强化数据增强（抗过拟合）+ resnet10 支持 + N-best 独立跟踪
3. **下一步**：数据扩充（用户收集中）→ 重建缓存 → 对比增强前后 4-seed 方差；备选：mixup / 调融合 / 扩 seed 求均值
4. **工程注意**：checkpoints_hos 同名覆盖 → 每轮归档备份；`cache_hos` 新样本自动增量（key 含 mtime）

## 8. 已知限制

- 训练/运行完全用户手动，agent 不代跑训练
- val 划分固定 0.2（31 样本），单 seed 噪声大；未做 k-fold
- `seed_summary.json` 只保留最近一次运行汇总，历史在 archive
- 无 CI/自动测试，临时探针在 `tools/_probe_*.py`
