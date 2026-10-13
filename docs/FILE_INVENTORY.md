# 文件清单与分类（FILE INVENTORY）

> 最后更新: 2026-09-29。与 `PROJECT_OVERVIEW.md`（项目级梳理）配套，本文是**文件级**索引。
> 分类标记: ✅核心代码 ｜ 🛠工具/辅助 ｜ 📦产物 ｜ 📜历史 ｜ 🗑临时可清理 ｜ ⚠️注意

---

## 0. 统计概览

| 范围 | 数量 | 说明 |
|---|---|---|
| Python 源文件 | 34 | 核心 5 包 25 + 入口/工具 9；一次性脚本×5 已归档 `logs/archive/_tmp/20260929_cleanup/` |
| Notebook | 0（根目录） | 4 个历史探索已归档 `_tmp/20260929_cleanup/`（均未进主线） |
| 数据 csv | 13 | `data/`（划分含 4-seed 版本） |
| 缓存 npy | 194 | `cache_hos/`（约 1.6GB 级） |
| checkpoint | 8 | `checkpoints_hos/`（resnet10 T/N×8；其中 s42 为 round5 三序列 285MB，s0/1/7 为 round4 单序列 170MB） |
| 训练图 | 12 | `plots_hos/`（T/N 曲线 + 混淆矩阵） |
| 日志 | 55 + 27 | `logs/` 55 文件 + `logs/archive/_tmp/20260929_cleanup/` 27 文件（含 2026-09-29 消融分析脚本×2 + round5 旧 csv 备份 + 冒烟脚本） |
| 归档目录 | 5 + 1 | `round1-5` + `_tmp/`（2026-09-29 清理桶） |
| TCGA 历史 | 19 | `_archive_tcga/`（2.19GB；cmb 影像 93GB 已删，仅留 metadata 0.9MB） |

---

## 1. 核心代码（当前主线）

### 1.1 config/ — 配置 ✅
| 文件 | 用途 |
|---|---|
| `default.py` | `Config` 基类（TCGA 时代字段：train/val csv、backbone、fusion、loss、训练超参） |
| `hos.py` | `HosConfig(Config)` 当前有效配置：hos_pairs.csv、label_T/N、CT+MRI+临床、冻结 layer3、多任务权重 |
| `__init__.py` | 导出 `Config, HosConfig` |

### 1.2 data_loader/ — 数据管线 ✅
| 文件 | 用途 |
|---|---|
| `loader.py` | CT 确定性预处理(`_ct_deterministic`) + **MRI 三序列对齐**(`_mri_seqs_deterministic`, T2 锚定 1mm 网格, round5) → 磁盘缓存(`_DetDiskCache` v2, cache_hos) → 训练随机增强 → `CRCMultiModalDataset`(mri [3,96³] + `mri_seq_available`) / `build_dataloaders` |
| `__init__.py` | 导出 `build_dataloaders, CRCMultiModalDataset` |

### 1.3 models/ — 网络 ✅
| 文件 | 关键符号 | 用途 |
|---|---|---|
| `backbone.py` | `ResNet3DBackbone`(resnet10/18/50), `SwinTransformer3DBackbone`, `build_backbone` | 影像特征提取 |
| `fusion.py` | `SelfAttentionEncoder`, `ResidualMLPImputer`, `CrossAttentionBlock`, `BSAFusionModule` | 模态内提纯/缺失补全/跨模态融合 |
| `network.py` | `CRCMultiModalNet` | CT backbone + **MRI 三 backbone**(t/c/d, stage 级 softmax 融合, round5) + 双头(T/N)；round6 `mri_share_backbone=True` 时三序列共享同一 encoder（resnet10：71.4M→42.6M）；`_replace_missing_features` 处理缺失 |
| `losses.py` | `FocalLoss` | 可选焦点损失 |
| `__init__.py` | 导出 `CRCMultiModalNet, build_backbone, BSAFusionModule, CrossAttentionBlock, FocalLoss` | |

### 1.4 trainers/ — 训练循环 ✅
| 文件 | 关键符号 | 用途 |
|---|---|---|
| `trainer.py` | `Trainer` | fit/validate/predict_proba；模态 dropout；多任务+重建 loss；**T/N 双 best 独立跟踪**(round3 修复)；阈值校准接口；round6：MixUp（`cfg.mixup`，batch 内插值作用于 main/n/aux）+ freeze 前缀修正（兼容 `mri_backbones.*` 与共享模式，`relu`→`act`） |
| `__init__.py` | 导出 `Trainer` | |

### 1.5 utils/ — 通用 ✅
| 文件 | 关键符号 | 用途 | 状态 |
|---|---|---|---|
| `metrics.py` | `compute_metrics`, `calibrate_binary_threshold` | 指标 + 阈值扫描校准 | ✅在用 |
| `logger.py` | `setup_logger` | 文件+控制台日志 | ✅在用 |
| `visualize.py` | `plot_curves`, `plot_confusion_matrix` | 训练曲线/混淆矩阵 | ✅在用 |
| `build_hos_dataset.py` | — | 构建 `hos_pairs.csv`（DICOM→NIfTI + xlsx 临床/病理对齐） | 🛠重跑需确认规则 |
| `build_hos_testset.py` | — | 构建 CT-only 外部测试集 csv | 🛠历史 |
| `convert_dcm_to_nifti.py` | — | DICOM→NIfTI 转换 | 🛠被 build_hos_dataset 调用 |
| `build_tcga_index.py` / `split_tcga.py` / `make_kfold.py` | — | TCGA 配对/划分/K 折 | 📜TCGA 时代 |
| `scan_modalities.py` | — | CMB-CRC 元数据配对 | 📜早期 CMB |
| `filter_cmb_data.py` / `filter_patients.py` | — | 数据清洗 | 📜一次性 |
| `__init__.py` | 导出 metrics/logger/visualize 的 5 个函数 | | |

## 2. 入口与工具脚本 ✅/🛠
| 文件 | 用途 | 状态 |
|---|---|---|
| `train_hos_seeds.py` | **当前主入口**：多 seed 分层划分+训练+校准+汇总；CLI `--epochs --val-ratio --seeds --fusion --backbone --no-clinical`；round6 新增 `--kfold N`（StratifiedKFold CV）、`--split-seed`（seed 共用划分）、`--mixup`、`--label-smoothing`、`--freeze-low`、`--share-backbone` | ✅在用 |
| `train_hos.py` | 旧单 seed 入口（被 seeds 版取代） | 🛠保留/可弃 |
| `tools/build_det_cache.py` | 确定性预处理缓存构建（`cache_hos/*.npy`，含 mtime 版本 key，增量友好） | ✅在用 |
| `tools/_ensemble_eval.py` | **round6 A2**：seed 集成评估（多 checkpoint 在固定 val 上平均 softmax + 阈值校准 → `logs/ensemble_summary.json`） | ✅新增 |
| `tools/_probe_monai.py` | 验证 MONAI 组件可用性 | 🗑临时探针 |
| `tools/_probe_resnet10.py` | 验证 resnet10 backbone | 🗑临时探针 |
| `tools/_probe_augment.py` | 验证强化增强（round4） | 🗑临时探针（可留作参考） |
| `_eval_best.py` | 评估 `best_model_hos.pth`（epoch11）T/N | 🗑历史临时 |
| `_eval_fast.py` | 轻量 GPU 评估指定 checkpoint + 校准 | 🗑临时（可并入主流程） |
| `test_hos_ct.py` | CT 外部数据测试（stage2/3） | 🗑历史临时 |

## 3. 一次性数据处理脚本（2026-09-29 已归档 `logs/archive/_tmp/20260929_cleanup/`）📜
| 文件 | 当时用途 |
|---|---|
| `classify_wzy.py` | 解析 `data/Hos/wzy` 序列类型 |
| `move_wzy_classify.py` | wzy 分类移动到 `data/Hos/{CT,MRI}`（编号+100 等规则） |
| `delete_non_dicom_dirs.py` | 删除无 .dcm 文件夹 |
| `rename_ct.py` / `rename_mri.py` | 统一 CT/MRI 命名（p/z/c/t/d 映射） |

## 4. Notebook（2026-09-29 已归档 `logs/archive/_tmp/20260929_cleanup/`）📜
`DataLoader-test.ipynb`、`dataloader.ipynb`、`debug.ipynb`、`models.ipynb` — 早期数据管线/模型调试，已被正式代码取代。

---

## 5. 产物目录（勿手改，由脚本生成）📦

### data/ — 数据与划分
| 文件 | 说明 |
|---|---|
| `hos_pairs.csv` | **唯一数据源**：155 行（154 有效），CT/MRI 路径+可用性+临床 7 维+label_T/N；**round5 新增 `mri_path/available_{t,c,d}`**（T2/CE/DWI 三序列，缺失序列路径为空） |
| `hos_pairs_raw.csv` | 临床原始值备份（复查用） |
| `hos_train/val.csv` | 默认 seed42 划分（train_test_split 0.2） |
| `hos_train/val_s{0,1,7}.csv` | round4 用 4-seed 划分（42/0/1/7，分层按 T×N） |
| `hos_ct_test.csv` | CT-only 外部测试（stage2/3，label=-1 表示 OOD） |

### cache_hos/ — 确定性预处理缓存
194 个 `.npy`（CT: 1×96³ / MRI: **3×96³** float32，round5 三序列对齐卷）。key=sha1(路径+mtime+size)；MRI key 含 t/c/d 三路径+mtime。训练只读缓存输出再增强 → **增强改动无需重建缓存**；新增样本自动增量。

### checkpoints_hos/ — 权重
| 文件 | 归属 |
|---|---|
| `best_model_hos_s{42,0,1,7}.pth` | resnet10 T-best（162.8MB×4，round4） |
| `best_model_n_hos_s{42,0,1,7}.pth` | resnet10 N-best（round3 修复后新增） |

⚠️ **同名覆盖**：每次训练覆盖 `s{seed}`，重要轮次先归档备份。
⚠️ **round5 结构变更**：MRI 端改三 backbone，round4 及之前的权重（单 MRI backbone）与新结构**不兼容**，重训后覆盖。

### plots_hos/ — 训练图
`training_curves_hos[_s{seed}].png` ×6 + `confusion_matrix_hos[_s{seed}].png` ×6。

### logs/ — 日志与运行输出（55 文件 0.31MB）
| 类别 | 文件 | 状态 |
|---|---|---|
| 汇总 | `seed_summary.json` | ✅最近一次多 seed 汇总（历史在 archive） |
| 消融汇总 | `seed_summary_noclinc.json` | ✅no-clinical 消融 4-seed（已移入 round4 归档，见下） |
| Hos 主日志 | `crc_20260918_*`(9) / `crc_20260928_*`(5) / `crc_seed{42,0,1,7}_20260928/29_*`(28) / `crc_seed42_20261009_151822`(1, round5 三序列) | 📜对应 round1-5，重要日志已进 archive |
| TCGA/早期 | `crc_20260602~20260716_*`(9) | 📜历史 |
| 已清理(2026-09-29) | 运行输出 txt×14 + `hos_.log` 已移入 `_tmp/20260929_cleanup/`；3 个 0 字节日志已删除 | ✅ |

### logs/archive/ — 轮次归档 📜
| 目录 | 内容 |
|---|---|
| `round1_20260928` | resnet18 基线 + 早期 Hos 训练 |
| `round2_20260928` | 多任务 loss 权重消融等 |
| `round3_20260928` | resnet18 3-seed + N-best 修复 |
| `round4_20260929` | resnet10 4-seed + single 消融 + resnet18 复跑确认 + **no-clinical 消融**（`seed_summary_noclinc.json`）+ 8 权重 + 8 划分 + 8 图 + summary.json |
| `round5_20261009` | **MRI 三序列整合（T2/CE/DWI）**：三 backbone + stage 级 softmax 融合 + T2 锚定对齐管线；seed42 单 seed 验证（T 0.4833 / N 0.6026）；summary + 日志 + seed_summary + plots×2 + csv 快照 |
| `_tmp/20260929_cleanup/` | 2026-09-29 清理桶：一次性脚本×5 + notebook×4 + 运行输出 txt×14 + 消融分析脚本×2（`_ablate_clinical_baseline.py`/`_quick_clin_stats.py`）+ **round5 旧单序列 csv 备份 + 冒烟脚本** |

## 6. 历史代码（整体冻结）📜

### _archive_tcga/ — TCGA 时代（19 项，2.19GB，2026-09-29 已瘦身）
- 代码：`main.py`(启动) / `evaluate.py`(单模态对比) / `cross_validate.py`(K 折) / `ml_stage_clinical.py`(GSE39582 临床 ML) / `machine_learning/`(特征工程+传统 ML: extract_features, train_models, run_all)
- 数据：`tcga/`(CT nifti 2.24GB，27 例带分期 → **外部验证可用**) / `cmb_metadata_keep/`(0.9MB，CMB 元数据+digest，重下钥匙) / `tcga_*.csv`×4 / `GSE39582_clinical.csv`(0.1MB)
- 产物：`plots/` `kfold/` `data/` + `PROGRESS_summary.md/txt` + `train_tcga.log` + `1.py`(soft 解析，临时)
- 已清理：`cmb/` 原始影像 93GB（仅 6 例有分期，无评估价值） / `GSE39582_family.soft` 771MB / `checkpoints/` 1.86GB(kfold 权重) + cmb 散件

## 7. 使用注意事项 ⚠️
1. **入口**：训练只走 `train_hos_seeds.py`；`train_hos.py`/`_eval_*.py` 已退役
2. **缓存**：改确定性管线（非增强）后必须重跑 `tools/build_det_cache.py`
3. **权重覆盖**：`s{seed}` 系列会被新训练覆盖，重要轮次前先归档备份；旧 resnet18 权重已于 2026-09-29 删除（archive round1/3 有备份）
4. **编码**：源文件 UTF-8；PowerShell 5.1 控制台读取中文 JSON/日志会乱码（用 `python -c` 或 VS Code 查看）
5. **2026-09-29 已清理**：`_archive_tcga` 释放 ~96GB（cmb 影像 93GB / soft 771MB / kfold checkpoints 1.86GB）+ 旧 resnet18 权重 310MB + 临时文件归档 `_tmp/20260929_cleanup/`。**剩余 B 级可删项**：`logs/archive/round3_20260928/`（2.44GB resnet18 备份，确认后删）
6. **消融开关**：`train_hos_seeds.py --no-clinical` 将临床特征置 0（结构不变），用于临床贡献验证。2026-09-29 resnet10 4-seed 结果 T 0.5551 / N 0.5550±0.006——与有临床版（0.5405/0.5437）无差甚至微升 → **当前临床分支为净噪声（62.6% 缺失占位 + 弱相关），后续不要为提高临床权重调整 loss**
7. **round5 MRI 三序列**：loader 用 `_mri_seqs_deterministic`（T2 锚定 1mm 网格对齐，输出 [3,96³]）；模型 MRI 端为三 backbone（t/c/d）+ stage 级 softmax 融合；缓存 v2 已重建；**round4 及之前的单序列权重与新结构不兼容，勿用于评估**；seed42 单 seed 验证结果见 `logs/archive/round5_20261009/round5_summary.json`，权重备份 `backup_round5_mri3/`
8. **同步义务**：本文档与 PROJECT_OVERVIEW.md 需随轮次更新（新 round 归档时同步改第 0/5 节统计）
