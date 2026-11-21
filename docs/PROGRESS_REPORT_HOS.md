# Hos 数据多模态分期项目 — 进度汇报

> **阶段快照：2026-10-09 / 154 样本阶段** — 本文记录该阶段状态，内容不代表当前状态。
> 当前状态见 [README.md](../README.md) 与 [FIXES_2026-11-20.md](FIXES_2026-11-20.md)；数据规模与口径已变更，文中数字不可直接引用。
> 2026-11-21 已删除全部实验权重（`checkpoints_*` / `archive` / `logs`），涉及 `.pth` 的命令需重跑生成。
> 相关快照：[文件清单](FILE_INVENTORY.md) · [项目总览](PROJECT_OVERVIEW.md) · [进度汇报](PROGRESS_REPORT_HOS.md) · [优化路线图](OPTIMIZATION_ROADMAP.md)

---

## 1. 任务与数据基线

- **任务**：术前 T / N 双任务二分类 —— T1-2 vs T3-4、N0 vs N+（M 分期因数据缺失排除）
- **样本**：154 有效（双模态 39 / 仅 CT 24 / 仅 MRI 91）；label_T 43:111、label_N 85:69
- **模态**：CT 影像（当前单序列，增强 z 优先）、MRI 影像（T2/CE/DWI 三序列）、临床 7 维（age/sex/ca199/cea/albumin/prealbumin/hb）
- **核心指标**：验证集校准后 macro-F1（T 主 / N 次），多 seed 取均值±std

---

## 2. 拿到 Hos 数据后的尝试与修改（按轮次）

### [2026-09-28] 课题定位与 Hos 训练配置修复
- **定位**：多分类（Stage I-IV）改为 T/N 双任务二分类；数据审计确认 pT=T1:6/T2:37/T3:106/T4:5（T 四分类不可行）、pN=N0:85/N1:47/N2:22
- **配置修复**：`loss_weights` aux 0.3→0.1、rec 0.5→0.1（重建/辅助不再挤压分类头）；lr 1e-4→3e-4；epochs 30→60

### [2026-09-29] round4：resnet10 消融 + 性能基建 + 增强强化 + N-best 修复
- **确定性磁盘缓存**：`_DetDiskCache`（cache_hos，194 卷 100% 命中，epoch ~7s），增强作用于缓存输出 → 增强改动无需重建缓存
- **N-best 独立跟踪**：T/N 各自保存 best checkpoint，校准 N 用 N-best 权重
- **resnet10 支持**（42.6M）+ `--fusion/--backbone/--no-clinical` CLI
- **数据增强强化**：Zoom/强度偏移/强度缩放/高斯噪声/CoarseDropout + 仿射温和化
- **结果**：resnet10 4-seed T 0.5405±0.062 / N 0.5437±0.057；no-clinical 消融 T 0.5551 / N 0.5550±0.006；同 seed42 两次运行 T=0.5441 vs 0.4833

### [2026-10-09] round5：MRI 三序列（T2/CE/DWI）整合
- **数据**：`collect_mri_seqs` 每患者导出全部 MRI 序列，csv 新增 `mri_path/available_{t,c,d}`；131 例 MRI 中 **93 例三序列齐全（71%）、127 例（97%）≥2 序列**
- **对齐管线**：同患者 t/c/d 的 spacing/origin/direction/FOV 完全不同 → 以 T2 为锚构建 1mm 各向同性参考网格，CE/DWI 重采样对齐 + 统一 bbox 裁剪 + Resize(96³)，输出 [3,96,96,96]；磁盘缓存升 v2（key 含三序列 mtime）
- **模型**：MRI 单 backbone → 三 backbone（`mri_backbones` t/c/d），stage 级可学习 softmax 融合，缺失序列 mask 后重归一化（71.4M）
- **结果**：seed42 T 0.4833 / N 0.6026（落在 round4 同 seed 波动区间，无法判定三序列增益）

### [2026-10-09] round6：k-fold CV + seed 集成 + 共享减参 + 正则强化（代码就绪，GPU 已验证）
- **A1 k-fold**：`--kfold 5` StratifiedKFold（按 T×N 分层，每折独立训练/校准 → `logs/kfold_summary.json`）
- **A2 seed 集成**：`--split-seed 42`（所有 seed 共用同一划分）+ `tools/_ensemble_eval.py`（softmax 平均 + 阈值校准）
- **B2 共享减参**：`--share-backbone` 三序列共享 encoder（71.4M→**42.6M**）+ `--freeze-low`（可训练→40.8M）；**修复 freeze 前缀 bug**（旧 `relu` 不匹配 MONAI `act`，round5 三流前缀未覆盖 → 此前冻结从未生效）
- **B3 正则**：`--mixup 0.2~0.5`（已对照手工计算验证）、标签平滑、cosine LR
- **验证**：py_compile / 结构冒烟 / GPU 1-epoch 整链路全通过（项目环境 `Pytorch`：torch 2.5.1+cu124 / RTX 4060）

### [2026-10-09] 数据满足性探针（异构多网络前置分析）
- MRI 三齐全 93/154；CT 平扫 p / 增强 z 双齐全仅 34/154（22%），**97 例（63%）无 CT**
- 结论写入 ROADMAP D 段：MRI 异构三网络可行；CT 双序列等数据补齐

---

## 3. 当前成果

### 已落地系统
| 层 | 成果 |
|---|---|
| 数据 | 154 样本 csv（CT/MRI 路径 + 三序列可用性 + 临床 + T/N）；确定性预处理磁盘缓存（194 npy v2，epoch ~7s，增量自更新） |
| 预处理 | T2 锚定 1mm 各向同性三序列物理对齐管线（round5）；CT/MRI 确定性预处理 + 强化随机增强 |
| 模型 | CT backbone + MRI 三 backbone（stage 级可学习融合、缺失 mask 重归一化）+ 模态内自注意力 + 缺失 imputer + 多尺度 cross-attention 融合 + 双头（T/N）；可切换 resnet10/18/50/swin、共享/独立、多/单尺度 |
| 训练 | 多 seed 训练 + 阈值校准 + k-fold（5 折分层）+ seed 集成评估 + MixUp/标签平滑/cosine LR/冻结策略 |
| 文档 | PROJECT_OVERVIEW / FILE_INVENTORY / OPTIMIZATION_ROADMAP（A/B/C/D 计划表）+ 每轮归档 `logs/archive/roundN_*` |

### 量化成果（校准 macro-F1）
| 配置 | T | N | 备注 |
|---|---|---|---|
| round4 resnet10 4-seed（42/0/1/7） | 0.5405±0.062 | 0.5437±0.057 | 当前最稳基线 |
| round4 no-clinical 4-seed | 0.5551±0.095 | 0.5550±0.006 | 临床分支为净噪声源 |
| round4 resnet18 3-seed | 0.5297±0.055 | 0.5735±0.097 | 参数量大、N 方差大 |
| round5 MRI三序列 seed42 | 0.4833 | 0.6026 | 结构升级；单 seed 无法判定增益 |
| round5 校准细节 | acc 0.6129（N）| N 类别均衡（85/69）优于 T（43/111）| N 是更容易出结果的目标 |

### 关键结论（实证）
1. **154 样本高方差是主要矛盾**：同 seed42+resnet10 两次运行 T=0.5441 vs 0.4833，均值提升被方差淹没
2. **临床分支是净噪声**：ca199/cea 62.6% 缺失、单列相关均不显著、纯临床基线仅 0.47-0.49；no-clinical 无损失反降 N 方差
3. **MRI 三序列物理信息可独立建模**：T2（解剖）/ CE（血供）/ DWI（弥散受限）本质不同，三流 + 缺失 mask 机制就绪
4. **CT 双序列当前数据不足**：双齐全仅 34/154，63% 无 CT → 平扫+增强双流模型暂缓

---

## 4. 等待数据补充后的尝试与改进计划

> 新 ~200 份数据（多为 2/3 期）到位后执行，优先级从高到低。

### 第一阶段：数据到位后立即执行（C4 + A 系列收尾）
| # | 事项 | 说明 |
|---|---|---|
| C4 | **4-seed 正式对比 + 5-fold** | 数据量翻倍后跑 round4/5/6 同协议对照，T/N 均值±std 正式结论 |
| A1-A2 | round6 k-fold / seed 集成正式运行 | 代码已就绪（当前 154 样本也可先跑出基线） |
| A3 | 重复分层重采样、报告中位数 | 与 k-fold 叠加，进一步压低估计方差 |
| 重建 | 新样本自动增量缓存 + 归档 round6/7 | 缓存 key 含路径+mtime，增量自更新 |

### 第二阶段：模型结构升级（ROADMAP D 段）
| # | 事项 | 说明 |
|---|---|---|
| D1 | **MRI 异构三网络** | T2→ResNet18 / CE→ResNet10 / DWI→ResNet10；不同 backbone 需 stage 级 1×1×1 投影对齐通道 |
| D2 | **CT 双序列（平扫+增强）** | 数据补齐后扩展 `ct_path_{p,z}` + 对齐管线（同 MRI 三序列策略）+ CT 双流 stage 融合 |
| D3 | CT 序列选择优化 | 有 z 用 z、无 z 用 p 把 CT 覆盖从 62 扩到 ~90%（代价：平扫/增强混用，需消融验证） |

### 第三阶段：表示/评估层（ROADMAP B/C）
| # | 事项 | 说明 |
|---|---|---|
| B4 | 融合权重体检 | 可视化 `mri_seq_fusion_weights`，若退化为均匀 → 三序列无增益的直接证据 |
| B5 | 逐序列消融 | `--mri-seq t / t,c / t,d ...` 同 seed 对照，量化每序列贡献（CLI 待实现） |
| C1 | 自监督预训练 | 351 卷无标签 MRI 对比学习（SimCLR/MAE）→ 微调分类，医学小样本最有效手段 |
| C2 | TTA / 多 crop 推理 | 白捡 0.01-0.03 |
| C3 | 3D patch 随机采样训练 | 等效数据扩充，与 TTA 配套 |

### 现实预期
- A 全部 + round6 B2/B3 组合后 4-seed：**T 0.54→0.56~0.59、N 0.54→0.58~0.62，std 减半**
- 纯加结构/正则均值上限 ~0.05；**N 比 T 信号好**，主攻 N 指标最易出好看结果
- 每步改动记入 README 变更日志 + FILE_INVENTORY，实验归档 `logs/archive/roundN_*`

---

## 5. 环境与运行

- 项目环境：conda **`Pytorch`**（`D:\Anaconda\envs\Pytorch\python.exe`，torch 2.5.1+cu124，**RTX 4060 CUDA 可用**）；当前 shell 的 `python` 是 base（CPU torch），务必用绝对路径或 `conda activate Pytorch`
- 主入口：`python train_hos_seeds.py --seeds 42,0,1,7 --epochs 60 --split-seed 42 --share-backbone --freeze-low --mixup 0.3`
- k-fold：`python train_hos_seeds.py --kfold 5 --epochs 60 --share-backbone --freeze-low --mixup 0.3`
- 集成评估：`python tools/_ensemble_eval.py --ckpts checkpoints_hos/best_model_hos_s42.pth,... --names s42,... --val-csv data/hos_val_s42.csv --share-backbone`
> 注（2026-11-21 权重清理）：该命令引用的权重已删除，不可直接执行；需先重跑生成。
