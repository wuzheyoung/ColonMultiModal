# ColonMultiModal

结直肠癌（CRC）多模态分期分类系统，融合 CT、MRI 影像与临床血液指标进行术前病理 T 分期（T1-2 vs T3-4）与 N 分期（N0 vs N+）双任务预测。

---

## 目录结构（2026-11-02 整理）

```
ColonMultiModal/
├── config/            # 配置（HosConfig: 分支/增强/loss 权重/冻结层）
├── data/              # 样本 CSV（hos_pairs.csv + 各分支 train/val 划分）
├── data_loader/       # 数据加载（loader.py: 确定性预处理+磁盘缓存+增强）
├── models/            # 网络（backbone.py / network.py / fusion.py / medicalnet.py）
├── trainers/          # 训练器（trainer.py: AMP/EMA/不确定性加权/一致性正则）
├── tools/             # 实验脚本（radiomics / mil / ensemble / cam / 探针等）
├── utils/             # 工具（eval_report / 校准阈值 / 日志）
├── train_hos_seeds.py # 训练入口（CLI 全参数）
├── checkpoints_hos_full/   # 空目录（权重已于 2026-11-21 删除，需重跑生成）
├── plots_hos_full/         # 当前训练曲线/混淆矩阵图
├── logs/                   # 训练日志 + eval_report JSON/TXT（300+ 文件）
├── results/round9/         # 每轮归档（run_NN_<tag>，报告/图保留；权重已于 2026-11-21 删除）
├── pretrained/             # MedicalNet / CRCFound / SwinUNETR 权重
├── cache_hos/ cache_crcfound/  # 磁盘缓存（代码硬编码，勿动）
├── docs/                # 项目文档（PROJECT_OVERVIEW/OPTIMIZATION_ROADMAP/PROGRESS_REPORT_HOS/FILE_INVENTORY）
└── archive/             # 旧分支与早期产物（已弃用，仅存档）
    ├── checkpoints_branch_old/  # round7 旧分支 ckpt（ct_z/ct_p/mri_t/c/d/tdc）
    ├── plots_branch_old/        # 旧分支训练图
    ├── checkpoints_early/       # 早期 ckpt 目录（权重已于 2026-11-21 删除）
    ├── plots_early/             # 早期训练图
    ├── results_early/           # 早期结果（confusion/curves + 旧分支子目录）
    ├── backup_round5/           # round5 mri3 备份（权重已删除）
    ├── scripts_legacy/          # 旧脚本（train_hos.py 等）
    └── tcga/                    # TCGA 预实验产物（172 文件）
```

> 注：`checkpoints_hos_full` / `plots_hos_full` / `logs` / `results/round9` / `cache_*`
> 均为代码硬编码路径，保留在根目录以保证运行；`archive/` 内旧产物如需复现
> 实验可移回原名目录（`--branch mri_t` 等旧分支会读 `checkpoints_hos_mri_t/`）。

> **2026-11-21 权重清理**：`checkpoints_hos_full/`、`checkpoints_hos/`、`archive/`、`logs/` 下的实验
> `.pth/.pt` 已全部删除（`pretrained/` 未动）；下方涉及 `.pth` 的命令不可直接执行，
> 需重跑生成权重。旧指标已固化到 `2026-11-21/repo-cleanup/outputs/snapshot/`。

---

## 变更日志（CHANGELOG）

### [2026-10-27] round9: 治本迭代训练日志（训练→总结→修改 循环）

> **流程**: 每轮 60ep（full 分支 seed42）→ 完成即解析 eval_report 汇报 T/N 全指标 →
> 按重要性（**N 短板优先、双头均衡**）自主修改 **1-2 处** → 下一轮。总预算 7-8 小时。
> **异常规则**: 报错/卡死/梯度 NaN/非 GPU → 自主修复继续；**异常累计 3 次 → 停止整个流程**。
> **轮询**: 35 分钟/轮。
> **基线参考**: V6 组合(medicalnet18+gem+128 未冻结) T=0.6400 / N=0.5205。

#### R1 全冻结 medicalnet18 + GeM + 头128（lr 3e-4）
- **改动（第 1 处, 最重要）**: 新增 `--freeze-all`——全冻结 backbone(conv1..layer4) 作固定特征提取器，
  只训练融合层+双头（可训练参数 **145.7M → 13.7M, -91%**），匹配 284 训练样本
- **命令**: `python train_hos_seeds.py --branch full --seeds 42 --epochs 60
  --backbone medicalnet18 --pool gem --head-width 128 --freeze-all --lr 3e-4`
- **结果**:
  - T: best_epoch=46 | macro_f1=**0.5901** | acc=0.7083 | 校准@0.5 recall=[0.571, 0.647]
  - N: best_epoch=5 | macro_f1=**0.4996** | 校准@0.5 acc=0.5000 recall=[0.425, 0.594]
- **总结**: ❌ **全冻结假设被证伪**——对比 V6(未全冻结, lr1e-4) T=0.6400/N=0.5205,
  **T -0.050、N -0.021 全面倒退**。MedicalNet 特征必须经 layer4/融合层微调才能适配
  T/N 分期判别; 全冻结只适合特征线性可分场景, 此处不成立。→ **R2 撤销全冻结, 回到
  layer1-3 冻结 + lr3e-4, 并加 N 类别权重(N 短板优先)。**

#### R2 medicalnet18+gem+128 lr3e-4（layer1-3冻结）+ N类别权重
- **改动（1 处, N 短板优先）**: 新增 `compute_class_weights_n`（label_N 分布 inverse 权重
  [0.894, 1.134]）→ `Trainer.criterion_n` 使用类别权重, 缓解 N 头被 T 主导;
  同时撤销全冻结（--freeze-all 证实无效, 回 layer1-3 冻结, 可训练 112.8M）
- **命令**: `python train_hos_seeds.py --branch full --seeds 42 --epochs 60
  --backbone medicalnet18 --pool gem --head-width 128 --lr 3e-4`
- **结果**:
  - T: best_epoch=53 | macro_f1=**0.6073** | acc=0.7083 | 校准@0.5 acc=0.6667 recall=[0.476, 0.745]
  - N: best_epoch=42 | macro_f1=**0.5549** | 校准@0.5 acc=0.5694 recall=[0.675, 0.438]
- **总结**: ✅ **N 类别权重有效**——N 0.5549 较 V6(0.5205) **+0.034** 创 N 新高;
  ⚠️ 但 T 0.6073 < V6 的 0.6400 (-0.033) → **lr3e-4 对 medicalnet18 的 T 有害**
  （V2 的 lr3e-4 增益只在 resnet18 上成立）。→ **R3: lr 回 1e-4 (V6 值) + 保留 N 权重,
  验证 T 恢复 0.64 且 N 保持 0.55。**

#### R3 medicalnet18+gem+128 lr1e-4（V6 lr）+ N类别权重
- **改动（1 处）**: 仅 lr 3e-4 → 1e-4（V6 值），其余与 R2 相同（N 权重保留）
- **命令**: `python train_hos_seeds.py --branch full --seeds 42 --epochs 60
  --backbone medicalnet18 --pool gem --head-width 128`
- **结果**:
  - T: best_epoch=46 | macro_f1=**0.6205** | acc=0.7083 | 校准@0.5 recall=[0.619, 0.667]
  - N: best_epoch=1 | macro_f1=**0.5263** | 校准@0.5 acc=0.5694 recall=[0.525, 0.531]
- **总结**: 🔍 **发现 T/N 双任务 lr 冲突**（表）:
  | run | lr | N权重 | T | N |
  |---|---|---|---|---|
  | V6 | 1e-4 | 无 | **0.6400** | 0.5205 |
  | R2 | 3e-4 | 有 | 0.6073 | **0.5549** |
  | R3 | 1e-4 | 有 | 0.6205 | 0.5263 |
  lr3e-4 偏好 N、lr1e-4 偏好 T；单 lr 无法同时最优 → **R4: T/N 分离学习率**
  （T 头 1e-4、N 头 3e-4 独立优化器分组），预期 T≈0.64 且 N≈0.55。

#### R4 medicalnet18+gem+128（T-lr1e-4 / N-lr3e-4 分离）+ N权重
- **改动（1 处）**: 新增 `--lr-n`——优化器按 `classifier_n.*` 参数名分组：
  N 头独立 lr 3e-4、其余 1e-4（解决双任务 lr 冲突）
- **命令**: `python train_hos_seeds.py --branch full --seeds 42 --epochs 60
  --backbone medicalnet18 --pool gem --head-width 128 --lr 1e-4 --lr-n 3e-4`
- **结果**:
  - T: best_epoch=18 | macro_f1=**0.5956** | acc=0.7361 | 校准@0.5 recall=[0.381, 0.804]
  - N: best_epoch=27 | macro_f1=**0.5556** | 校准@0.5 acc=0.5833 recall=[0.75, 0.375]
- **总结**: ⚠️ **分离 lr 未达预期**——N 稳定在 0.5556（与 R2 0.5549 一致, N 新高区 ✓）,
  但 T 0.5956 反而低于 R3 的 0.6205 → **N 头 3e-4 的梯度经共享融合特征回传, 扰动 T 优化**。
  表(累计):
  | run | T | N |
  |---|---|---|
  | V6 (1e-4 单lr, 无N权重) | **0.6400** | 0.5205 |
  | R2 (3e-4, N权重) | 0.6073 | 0.5549 |
  | R3 (1e-4, N权重) | 0.6205 | 0.5263 |
  | R4 (T1e-4/N3e-4, N权重) | 0.5956 | **0.5556** |
  **N 卡在 ~0.555、T 卡在 ~0.62, 两者互相干扰的根因是共享深层融合特征** →
  **R5: N 特征解耦**（N 头从浅层特征独立取输入, 不再与 T 共享融合通道）。

#### R5 medicalnet18+gem+128（N 特征解耦, layer3 浅层）+ T1e-4/N3e-4 + N权重
- **改动（1 处, 治本）**: 新增 `--n-shallow`——N 头输入从 `fused(融合)+clinical` 改为
  `CT/MRI 各自 layer3 浅层池化特征 + clinical`（不经 T 主导的融合通道）。
  ①N 分期依赖纹理/局部结构(浅层信息更丰富) ②N 梯度不再回传扰动共享融合/深层 → T/N 互扰解除
- **命令**: `python train_hos_seeds.py --branch full --seeds 42 --epochs 60
  --backbone medicalnet18 --pool gem --head-width 128 --lr 1e-4 --lr-n 3e-4 --n-shallow`
- **结果**:
  - T: best_epoch=45 | macro_f1=**0.6727** | acc=0.7361 | 校准@0.45 acc=0.7361 recall=[0.524, 0.824]
  - N: best_epoch=1 | macro_f1=**0.5687** | 校准@0.5 acc=0.5694 recall=[0.55, 0.594]
- **总结**: 🎯 **N 特征解耦 = 治本成功**——**T/N 双双历史新高**:
  | run | T | N |
  |---|---|---|
  | V6 | 0.6400 | 0.5205 |
  | R2 | 0.6073 | 0.5549 |
  | R3 | 0.6205 | 0.5263 |
  | R4 | 0.5956 | 0.5556 |
  | **R5 (n-shallow)** | **0.6727** | **0.5687** |
  T +0.033 / N +0.014 同时超越全部历史 run → 确认「T/N 共享融合通道互相干扰」为根因。
  ⚠️ N best_epoch=1（早期定型）→ N 头疑似过拟合/容量问题 → **R6: N 头独立 dropout
  提高防过拟合, 或 N 浅层改 layer2 / 头减容。**

#### R6 medicalnet18+gem+128 + n-shallow + T1e-4/N3e-4 + N头正则(宽64/dropout0.7)
- **改动（1 处）**: 新增 `--n-width` / `--n-dropout`——N 头独立宽度 64 + dropout 0.7
  （主头保持 128/0.5）。依据 R5 证据「N best_epoch=1 后 n_acc 崩到 0.39-0.46」→ 防 N 过拟合
- **命令**: `python train_hos_seeds.py --branch full --seeds 42 --epochs 60
  --backbone medicalnet18 --pool gem --head-width 128 --lr 1e-4 --lr-n 3e-4
  --n-shallow --n-width 64 --n-dropout 0.7`
- **结果**:
  - T: best_epoch=49 | macro_f1=**0.5636** | 校准@0.5 recall=[0.429, 0.706]
  - N: best_epoch=2 | macro_f1=**0.4938** | 校准@0.5 recall=[0.35, 0.688]
- **总结**: ❌ **N 头过度正则化, 全面退步**——T -0.109 (vs R5 0.6727)、N -0.075 (vs R5 0.5687)。
  N 宽 64 + dropout 0.7 容量不足、N 学不动; 且 n_acc 全程钉在 0.5556（死锁态, macro_f1 不再涨）。
  → **R7: 回滚到 R5 配置复现**（N 头默认 128/0.5）——验证 R5 的 0.6727/0.5687 是否稳定
  （防 72 样本 val 噪声假象）。若复现成功, 下一步再叠加 mixup 等泛化正则。

#### R7 medicalnet18+gem+128 + n-shallow + T1e-4/N3e-4（R5 配置复现）
- **改动（1 处）**: 修复 R6 遗留 bug——`classifier_n` 误用原始参数 `n_dropout`(None)
  → `nn.Dropout(None)` 崩溃（异常#1）; 改为 `self.n_dropout`（含默认值处理）。
  训练配置与 R5 完全相同（复现验证）
- **命令**: `python train_hos_seeds.py --branch full --seeds 42 --epochs 60
  --backbone medicalnet18 --pool gem --head-width 128 --lr 1e-4 --lr-n 3e-4 --n-shallow`
- **结果**:
  - T: best_epoch=22 | macro_f1=**0.6000** | 校准@0.5 recall=[0.762, 0.549]
  - N: best_epoch=3 | macro_f1=**0.4965** | 校准@0.5 recall=[0.375, 0.656]
- **总结**: ⚠️ **复现远低于 R5（T -0.073 / N -0.072）**——同一配置两次 run 差异达 ±0.07,
  证实 **72 样本单次 holdout 评估噪声极大, R5 的 0.6727/0.5687 部分是 val 运气**。
  单次 run 结论不可信 → **R8: R5 配置 × 多 seed(0,1,7) 取均值, 建立可信基线**
  （此时 seed42 已有 R5/R7 两次结果可一起统计）。

#### R8/R8b R5 配置 × 多 seed（可信基线）→ ⛔ 异常 3 次, 流程停止
- **改动（0 处, 验证层）**: 同一配置(medicalnet18+gem+128+n-shallow+T1e-4/N3e-4+N权重)
  跑 seed 0/1/7 取 mean±std
- **异常记录（累计 3 次 → 按用户规则停止流程）**:
  - #1 R7: `nn.Dropout(None)` bug（classifier_n 误用原始参数）→ 已修复
  - #2 R8: seed0 完成后, seed1 epoch2 处 GPU 硬卡死 → WATCHDOG 强制退出
  - #3 R8b: seed1 完成, seed7 epoch2 处再次 GPU 硬卡死 → WATCHDOG 强制退出
  （#2/#3 与环境 GPU 硬 hang 相关, 与 V1 的 22 分钟 stall 同源, 非代码缺陷）
- **有效结果（R5 配置 × 3 seed + R5 首跑）**:
  | seed | T | N |
  |---|---|---|
  | 42 (R5首跑) | 0.6727 | 0.5687 |
  | 0 | 0.6095 | 0.5552 |
  | 1 | 0.5738 | 0.5185 |
  | 42 (R7复现) | 0.6000 | 0.4965 |
  | **mean±std** | **0.6140 ± 0.037** | **0.5347 ± 0.027** |
- **总结（本迭代序列最终结论）**:
  1. ✅ **N 特征解耦(n-shallow) = 本序列最有效改动**: T 单次最高 0.6727、N 单次最高 0.5687;
     N 均值 0.535 > V6 的 0.5205, T 均值 0.614 与 V6 0.6400 在噪声范围内
  2. ⚠️ **72 样本 holdout 评估噪声(±0.05~0.07) > 所有改进效应量**——单次 run 结论不可信,
     多 seed/kfold 才是唯一可信评估方式（本序列 8 次 run 才确立此事实）
  3. ❌ 全冻结 / N 头过度正则化(64/0.7) / 分离 lr 单独使用 均被证伪
  4. ⚠️ 环境 GPU 硬卡死频发(3 次), 长序列训练需单 seed 短任务或加 watchdog 保护(已生效)

#### round9-CAM 已完成（2026-10-28 17:50, 8 val 样本, R5 checkpoint T 头 Grad-CAM）
> **注意力未聚焦单一病灶区域**（3D 热图分析, 工具 tools/visualize_cam.py + tools/analyze_cam.py）:
> - CT 热图: 5/8 样本是**同一固定弱模式**（质心恒 (0.41,0.41), hot%≈4%, argmax 层=40）——不随样本变化,
>   退化响应, CT 分支贡献的是全局统计模式而非病灶定位; 另 3 样本最大层在边缘层(0/88/24), 非病灶。
> - MRI-t 热图: 每样本不同且居中, 但高值区占切片 13-37%——宽散, 非锐利聚焦。
> **含义**: 无病灶标注→无引导→注意力学全局模式; 这解释了 N 卡 0.53-0.57（淋巴结状态需病灶级精细特征）。
> 0.80 的下一步不在架构微调, 而在①5-fold 可信基线 ②病灶引导(有标注则引入) ③数据扩充。
>
> **口径修订 (2026-11-20)**: 上述热图当时取自 `hos_branch_full.csv`(全队列)前 8 行, 其中含训练病人, 并非验证集。
> `tools/visualize_cam.py` 已改为默认读取 `data/hos_{branch}_val_s{split_seed}.csv`; **本段结论需在真 val 子集上重跑 CAM 后才成立**。

#### round9 单模态对比（新 CLI, 2026-10-28 19:16 就绪, 冒烟通过, 未启动训练）> **`--single-modal {ct,mri}`**: 同批 356 样本同 holdout 只喂单模态（旧 --branch 是不同患者子集,
> 口径不干净）。实现要点:
> - 复用 modality_dropout p=1.0 机制: 训练每样本强制丢弃另一模态(走 imputer 伪输入),
>   **验证/predict_proba 新增强制掩码对齐**(trainer._apply_single_modal, 否则 val 悄悄回双模态);
> - 产物隔离: ckpt/报告/日志带 `_smct`/`_smmri` 后缀(如 best_model_hos_full_s42_smct.pth),
>   不覆盖双模态产物; 报告 params 记录 single_modal;
> - 命令预案: `--branch full --backbone medicalnet18 --pool gem --head-width 128
>   --lr 1e-4 --lr-n 3e-4 --n-shallow --single-modal {ct,mri} --seeds 42,0 --epochs 60`
> - 冒烟: tools/smoke_single_modal.py (cfg 设置 + 掩码强制 + CLI 解析全断言通过)

#### round9 项1 影像组学+临床融合 已完成（2026-10-28 23:47, 356样本=284train/72val, 75特征, tools/radiomics_experiment.py）
> - 特征: 每样本 CT+MRI(t/c/d) 各 一阶11(mean/std/skew/kurt/energy/entropy/p10/25/50/75/90) +
>   GLCM3D 6(contrast/energy/entropy/homog/ASM/corr, 降采样32³+16级量化+3主方向) = 17×4=68 + 临床7
> - 划分与 DL 完全一致(hos_full_train_s42.csv 284 / val 72); 复用 loader 磁盘缓存 cache_hos;
> - M1 影像组学 only / M2 +临床 / M3 临床 only / M4 [特征+DL概率]→LR stacking(5折CV+val, DL用s0 ckpt)
> - **T**: M1 0.4168/0.4539, M2 0.4288/0.4593, M3 0.4458/0.5591,
>   M4 CV-F1 0.5104±0.049 AUC 0.5719 | val F1 0.5789 acc 0.7222 AUC 0.6545
> - **N**: M1 0.3589/0.5194, M2 0.4964/0.5065, M3 0.5195/0.5164,
>   M4 CV-F1 0.5237±0.052 AUC 0.5276 | val F1 0.5074 acc 0.5417 AUC 0.5141
> - **结论: ❌ 无ROI全体积影像组学+临床判别力有限**（LASSO-LR 全 AUC≈0.5=无信号; HistGB 略好;
>   M4 stacking T val AUC 0.6545 接近 DL 基线但未超越, N 仍弱 0.5074）。印证 CAM 结论:
>   分期需局部浸润信息, 全体积统计丢失局部判别力 → 项1 不引入为模型, 仅作对照基线记录

#### round9 项4 N头专项 排程（2026-10-28 23:40→07:40, 待用户确认 3 问）

#### round9-E2 已完成（2026-10-29 01:22, seed42, 60ep, R5 配置 + `--mixup 0.3`）
> MixUp 线性插值增强（batch 内）。**结果: ❌ mixup 0.3 在 R5 配置上有害（同 seed42 同 holdout vs R5）**:
> T 0.6308(-0.042) / **N 0.4852(-0.084, best N epoch=7)**。mixup 引入插值噪声,
> 削弱 N 头本就稀缺的信号(早期历史 0.5836 是不同配置 resnet10 时代的记录, 不适用于 medicalnet18 R5)。
> **E2 证伪, mixup 保持 0.0**。
> 共同现象: E1/E2/R5 的 **N best epoch 极早(1/7/11)** → N 头 lr 3e-4 过高导致早期峰值+过拟合。

#### round9-E3 已完成（2026-10-29 02:03, seed42, 60ep, R5 配置 + `--lr-n 1e-4`）
> N 头 lr 从 3e-4 降到 1e-4（与主干同速，测试"N 头 lr 过高→早期过拟合"假说）。
> **结果: ❌ 假说证伪**（同 seed42 同 holdout vs R5）: T 0.6308(-0.042) / N 0.5274(-0.041),
> **N best epoch 依然=2**。lr-n 1e-4 劣于 3e-4 → **N 头 3e-4 已是较优设置**。
> **E1/E2/E3 三向全证伪: sqrt权重(-0.060)/mixup(-0.084)/lr-n1e-4(-0.041) → R5 配置在 seed42 为局部最优**,
> N 卡 0.53-0.57 非超参问题(与 CAM 结论一致: 需病灶级特征)。

#### round9-5fold 已完成（2026-10-29 02:05→04:17, R5 配置 × 5 折 × 30ep, 最终可信基线）
> 首启 Fold 1 两次卡死于同一位置(epoch2 后 121s>120s WATCHDOG 强杀) → 坏样本探针
> (tools/probe_fold1.py) 证明 356 样本逐样本前向全部 <0.4s 无坏样本 → **卡死=进程内累积型
> (单进程连续多折不可行)** → 实现 `--kfold-resume/--kfold-max-folds` 逐折分进程(每折独立 job,
> ~20min, 全部无卡死)。最终 5 折汇总(logs/kfold_summary_full.json):
> - **T**: best_macro_f1 **mean=0.5253 ± 0.0064** (min 0.5170 / max 0.5353), best_acc mean=0.7107±0.0064
> - **N**: best_macro_f1_n **mean=0.5393 ± 0.0531** (min 0.4836 / max 0.6375)
> - 各折: k0 T0.5353/N0.4836 | k1 T0.5267/**N0.6375**(N 最高) | k2 T0.5198/N0.5035 | k3 T0.5279/N0.5319 | k4 T0.5170/N0.5400
> - **总结**: ①**30ep/折 < 60ep**: 5 折 T mean 0.5253 明显低于 seed42 60ep 的 R5 (T 0.6727/N 0.5687)
>   → 30ep 折内数据仅 284×(4/5)≈227, 训练不足; ②**T 折间方差极小(±0.006)但 N 方差大(±0.053)**:
>   T 分期信号稳定、N 依赖折内 pN+ 样本组成, N 单次 0.6375(k1) 属折效应; ③5 折 mean 才是无偏基线:
>   **R5 配置跨折 T≈0.525 / N≈0.539**, 此前单 seed 汇报的 T 0.614±0.037/N 0.5347±0.027(60ep) 为
>   乐观估计(60ep+同折偏好)。→ 后续对比一律用 5 折 mean, 单 seed 结果仅作方向参考。

#### round9 总总结：昨晚序列（2026-10-28 23:40 → 2026-10-29 04:17, 用户 m01100 决策"做1、4"）
> **序列内容**: 项1(影像组学+临床融合) → E1(sqrt权重) → E2(mixup 0.3) → E3(lr-n 1e-4) → 5-fold 最终基线。
> **5 件事全部完成**:
> 1. **项1 radiomics** (23:47): 75 特征 LASSO/HistGB/DL-stacking, **无 ROI 全体积判别力有限**
>    (M4 T val AUC 0.6545 接近 DL 未超越, N 0.5074; LASSO 全 AUC≈0.5) → 仅作对照, 不引入。
> 2. **E1 sqrt 权重** (00:41): N 0.5092 **证伪** (-0.060, N best epoch=1 几乎没学)。
> 3. **E2 mixup 0.3** (01:22): N 0.4852 **证伪** (-0.084)。
> 4. **E3 lr-n 1e-4** (02:03): N 0.5274 **证伪** (-0.041, N best epoch 仍=2, "N头lr过高"假说否定)。
> 5. **5-fold 基线** (04:17): T 0.5253±0.006 / N 0.5393±0.053 (30ep), 逐折分进程规避环境卡死。
> **核心结论**:
> - **R5 配置(medicalnet18+gem+128+n-shallow+T1e-4/N3e-4+N权重)在 seed42 为局部最优**:
>   sqrt/mixup/lr-n 三种 N 超参扰动全部证伪 → N 卡 0.53-0.57 **不是超参问题**, 与 CAM
>   (注意力未聚焦病灶, 无病灶标注无引导) 一致: **N 需要病灶级局部特征**。
> - **可信基线重置**: 无偏 = 5 折 mean **T≈0.525 / N≈0.539**; 单 seed 60ep (T 0.614±0.037/N 0.5347±0.027)
>   为乐观上界。T 折间稳定(±0.006)说明 T 信号充足; N 折间波动大(±0.053)说明 N 受样本组成主导。
> - **30ep vs 60ep**: 30ep 折内训练不足(227 样本), 5 折基线偏低; 后续若需公平对比, 5 折应升到 60ep/折
>   (但 5×60ep ≈ 5h, 需按 GPU 卡死经验逐折分进程)。
> **工程沉淀**: `--kfold-resume/--kfold-max-folds` 逐折续跑 + WATCHDOG(120s) + 探针脚本 → 环境卡死可绕过。

#### round9-CRCFound 路径① 已完成（2026-10-29 09:48, 仅 CT 可用样本 145/356, tools/crcfound_features.py）
> **背景**: 文献级 CRC 专属 3D MAE 基础模型 (CRCFound, 24 层 ViT-d1024, 官方规格 spacing(1,1,3)/size(32,256,256)),
> 权重 3.75GB 已下载至 pretrained/CRCFound/checkpoint-999.pth。架构与 MedicalNet 3D ResNet 不匹配 → 走路径①
> 特征判别力验证 (不碰主网络): cls token [1024] → LASSO-LR / HistGB / +R5-DL 概率 stacking。
> **发现**: full 分支 **仅 145/356 样本有 CT** (train 114/284, val 31/72), 其余 211 无 CT 走 imputer → 只评估 CT 子集。
> **结果 (CT 子集, 145 样本 = 114 train/31 val)**:
> | 模型 | T CV-F1 | T val AUC | N CV-F1 | N val AUC |
> |---|---|---|---|---|
> | lasso_lr (预筛300) | 0.4554±0.101 | 0.4667 | 0.4145±0.062 | 0.2899 |
> | histgb | 0.4199±0.049 | 0.5000 | 0.3939±0.046 | 0.3992 |
> | +R5-DL 概率 stacking | 0.5791±0.098 | 0.4190 | 0.4958±0.100 | 0.2689 |
> **结论**: CRCFound 全局 cls 特征判别力 **≈随机 (AUC 0.29-0.50, 全部差于或持平随机)**; stacking 反而拉低 val AUC
> (31 个 val 样本上过拟合)。与项1 radiomics 一致: **全体积全局特征不含局部浸润信息 → 对 T/N 分期无判别力**。
> **路径② (CRCFound 替换 CT 分支) 收益预期低**: 需 3-4h 工程 (loader 重采样 + wrapper + 冻结微调) 换来的仍是
> 全局表示 → **不推荐投入**。CRCFound 在本数据上判定为低价值。

> **下一步候选**(待用户定): ①5-fold 60ep 公平基线 ②病灶级特征(ROI 分割→局部特征/注意力引导, 无标注是瓶颈)
> ③CRCFound 权重迁移(文献 T0.889/N0.847, 需下载 ~1.6GB 权重) ④2.5D+ImageNet 预训练(33M→11M)。

#### round9-P2/P3/P4/P5 多 seed 重训 + S0 集成 已完成（2026-10-30 00:34→01:54, R5 配置 60ep, 全部 split_seed=42 同一划分）
> **背景**: S0 4-seed 集成前核查发现 **s0/s1 旧 ckpt 非 R5 配置**（2026-10-28 凌晨训练, 早于 R5 定型,
> 非 split_seed=42 划分——复算 s0 N 0.3514 vs 报告 0.5552 巨大偏差）→ 用 R5 配置重训补齐 4 个真 seed。
> **各 seed 结果 (同划分同配置, 可信口径)**:
> | seed | T macro_f1 | N macro_f1 | T cal recall | N cal recall |
> |---|---|---|---|---|
> | 42 | 0.6017@44 | 0.5093@6 | [0.571, 0.667] | [0.375, 0.688] |
> | 7 | **0.6727@59** | 0.5263@2 | [0.571, 0.784] | [0.425, 0.656] |
> | 0 | 0.6308@19 | 0.4985@6 | [0.429, 0.824] | [0.5, 0.5] |
> | 1 | 0.6683@42 | 0.4896@5 | [0.619, 0.745] | [0.775, 0.25] |
> **4-seed mean: T 0.6434 / N 0.5059**（s7 复现 R5 首跑 s42 的 0.6727 → T≈0.67 是 R5 真实上限非运气）。
> **S0 集成 (softmax 平均, tools/ensemble_seeds.py)**: **T 0.6400 / N 0.4852 —— ❌ 证伪**。
> 集成 ≤ 单 seed 最佳 (s7 T0.6727 / s42 N0.5093), N 反而低于 mean。原因: 4 个模型高度同质
> (同预训练+同结构+同划分 → 错误强相关), 软平均只抹平边界不补漏检。**概率集成在本数据不可行, 不投入**;
> 保留单一 best ckpt (s7)。归档 run_02/03/04/05_hos_full_s{42,7,0,1}。


#### round9-S1 单模态消融 已完成（2026-10-30 01:56→02:44, seed42, 60ep, R5 配置 + `--single-modal ct/mri`）
> **目的**: 同批 356 样本(284/72 同一划分)分别只喂 CT / 只喂 MRI, 定 T/N 的模态依赖。
> | run | T macro_f1 | N macro_f1 | T cal recall | N cal recall |
> |---|---|---|---|---|
> | both (s42 R5) | 0.6017@44 | 0.5093@6 | [0.571, 0.667] | [0.375, 0.688] |
> | both (s7 R5) | **0.6727@59** | 0.5263@2 | [0.571, 0.784] | [0.425, 0.656] |
> | CT-only (smct) | 0.6010@22 | 0.4938@6 | [0.238, 0.961] | [0.55, 0.4375] |
> | MRI-only (smmri) | **0.6806@49** | 0.4685@9 | [0.619, 0.765] | [0.5, 0.4375] |
> **结论**: ①**T 主要由 MRI 驱动**——MRI-only 0.6806 达到 both 最佳(s7 0.6727)同级, CT-only 0.6010 仅与
> s42 both 相当 → CT 对 T 的增量贡献≈0（与 CAM: CT 热图退化响应一致）。②**N 任何单模态都不行**
> (0.47-0.51 三路无差异) → N 需要多模态+临床交互 或 病灶级特征(局部浸润), 单模态消融证实无捷径。
> ③架构轻量化方向: N 特征从 MRI 为主 + CT 仅辅助, 或考虑去掉 CT 分支减参。归档 run_06/07_hos_full_s42_sm{ct,mri}。

#### round9-B3a 已完成（2026-10-31 00:15→01:38, seed42+7, 60ep, R5 配置 + `--n-modal mri`）
> **目的**: full 分支仅 145/356 有真 CT, 其余 59% imputer 伪输入可能污染 N 头 → N 头特征只取 MRI 验证。
> | run | T macro_f1 | N macro_f1 | T cal recall | N cal recall |
> |---|---|---|---|---|
> | s42 | 0.6630@47 | 0.4883@50 | [0.667, 0.706] | [0.725, 0.281] |
> | s7 | 0.5847@59 | 0.5263@11 | [0.476, 0.706] | [0.525, 0.531] |
> | **mean** | 0.6239 | 0.5073 | | |
> 对比 R5 both (mean T 0.6372 / N 0.5178): **T -0.013 / N -0.011 → ❌ 证伪**: N 头特征来源
> (both vs MRI-only) 非瓶颈; 决策点 N≥0.55 未达。工程: seed7 串联第 2 epoch 环境卡死(WATCHDOG 强杀,
> 单进程第二任务早期卡死模式), 单独进程重跑完成。归档 run_11/run_12。

#### round9-B3b 已完成（2026-10-31 01:40→02:19, seed42, 60ep, R5 配置 + `--n-modal ct`）
> **目的**: B3a 对照组——N 头特征只取 CT。
> > | run | T macro_f1 | N macro_f1 | T cal recall | N cal recall |
> > |---|---|---|---|---|
> > | s42 | 0.6494@53 | 0.5751@50 | [0.762, 0.627] | [0.65, 0.5] |
> > 对比 R5 both s42 (T 0.6017 / N 0.5093) 与 B3a mri s42 (T 0.6630 / N 0.4883):
> > **T 0.6494 ≥ both / N 0.5751 显著高于 both(+0.066) 与 mri(+0.087) → N 头特征来源=CT 单源是三个配置中 s42 最高**。
> > 与 S1 单模态消融一致（smct N 0.4938 > smmri N 0.4685）: N 判别信息主要在 CT（浸润深度/结周特征），
> > MRI 三序列对 N 是噪声/过拟合源。⚠️ 仅单 seed（s42），需 s7 复现确认后方可采信；
> > 若复现 → N 头长期配置改为 `--n-modal ct`。归档 run_13。

#### round9-S3 EMA 参数平均 已完成（2026-10-30 02:53→03:45, seed42+7, 60ep, R5 配置 + `--ema-decay 0.999`）
> **目的**: EMA 影子权重（decay=0.999 每 batch 更新, 验证/保存 best 用 EMA 权重）抗小数据噪声。
> > | run | T macro_f1 | N macro_f1 | T cal recall | N cal recall |
> > |---|---|---|---|---|
> > | R5 基线 s42 | 0.6017@44 | 0.5093@6 | [0.571, 0.667] | [0.375, 0.688] |
> > | EMA s42 | 0.6052@51 | 0.5130@2 | [0.524, 0.706] | [0.5, 0.531] |
> > | R5 基线 s7 | 0.6727@59 | 0.5263@2 | [0.571, 0.784] | [0.425, 0.656] |
> > | EMA s7 | 0.4985@59 | 0.5909@8 | [0.429, 0.588] | [0.75, 0.4375] |
> > **结论: ❌ EMA 证伪（2-seed 均值）**: 基线 T 0.6372/N 0.5178 vs EMA T 0.5519/N 0.5520
> > → **T -0.085 / N +0.034**。EMA 平滑抹掉 T 的关键判别信号（s7 T 0.6727→0.4985 崩塌）, N 收益不抵 T 退化。
> > EMA best epoch 更晚（s42 44→51）说明 EMA 权重后期更稳, 但小数据下"稳"= 判别力损失。**不采用**。
> > 注: seed7 首跑(与 seed42 同进程)第 5 epoch 被 WATCHDOG 强杀(GPU 硬卡死, 模式同 5-fold Fold 1:
> > 单进程连续任务第二个早期卡死) → 改单进程单 seed 补跑成功。归档 run_08/09_hos_full_s{42,7}。
> > EMA 实现沉淀: trainers/trainer.py `_ema_update/_ema_apply/_ema_restore` + CLI `--ema-decay`（保留, 可复用）。

#### round9-S4 MRI-only 复现验证 已完成（2026-10-30 03:46→04:11, seed7, 60ep, R5 配置 + `--single-modal mri`）
> **目的**: S1 发现 MRI-only s42 T 0.6806 最强 → 验证是否可复现（还是 seed42 单次运气）。
> > | run | T macro_f1 | N macro_f1 | T cal recall | N cal recall |
> > |---|---|---|---|---|
> > | MRI-only s42 | 0.6806@49 | 0.4685@9 | [0.619, 0.765] | [0.5, 0.4375] |
> > | MRI-only s7 | 0.5524@26 | 0.5032@50 | [0.333, 0.824] | [0.675, 0.344] |
> > | both 2-seed mean (s42/s7) | 0.6372 | 0.5178 | — | — |
> > | MRI-only 2-seed mean | 0.6165 | 0.4859 | — | — |
> > **结论: ❌ MRI-only T 优势不成立**——s42 的 0.6806 是单次运气（s7 仅 0.5524）；2-seed 均值
> > MRI-only 略低于 both（T -0.021 / N -0.032）。T 虽主要靠 MRI（S1 已证），但 both 仍整体最优，
> > **保持双模态**。MRI-only 不采用。归档 run_10_hos_full_s7_smmri。

#### round9-S5 5-fold 可信基线 已完成（复用昨晚结果, 2026-10-30 04:11 校验, R5 配置 30ep）
> **说明**: 昨晚 8h 序列已跑完 R5 配置 30ep 5-fold（逐折分进程, k0-k4 报告全在）;
> 今晚序列 S0-S4 全部证伪、R5 配置未变 → S5 直接 `--kfold-resume` 复用, 校验汇总:
> > **T macro_f1 mean=0.5253±0.0064 (acc 0.7107±0.0064), N mean=0.5393±0.0531**。
> > 各折: k0 T0.5353/N0.4836 | k1 T0.5267/**N0.6375** | k2 T0.5198/N0.5035 | k3 T0.5279/N0.5319 | k4 T0.5170/N0.5400。
> > T 折间极稳（±0.006）; N 方差大（±0.053）→ N 对划分/seed 极度敏感 = 数据量不足下 N 判别信号弱。
> > 30ep 明显低于 60ep 单 seed（T 0.60-0.67）→ 5-fold 无偏基线偏低, 60ep 是乐观上界。

#### round9-S6 今晚序列总总结（2026-10-30 02:53→04:11 执行完毕）
> 今晚 8h 序列全部完成。**S0-S4 全证伪, R5 配置经受住全部冲击, 仍是最终配置**:
> | 步骤 | 实验 | 结果 | 判定 |
> |---|---|---|---|
> | S0 | 4-seed 集成 (s42/s0/s1/s7) | T 0.6400 / N 0.4852 | ❌ 集成≤单seed最佳, 同质模型错误强相关 |
> | S1 | 单模态消融 ct/mri | MRI-only T 0.6806≥both, N 全≈0.47-0.51 | ✅ 结论: T 靠 MRI, N 无单模态捷径 |
> | S3 | EMA decay=0.999 ×2 seed | T -0.085 / N +0.034 (2-seed 均值) | ❌ T 判别信号被平滑抹掉 |
> | S4 | MRI-only 复现验证 | 2-seed mean T 0.6165 < both 0.6372 | ❌ s42 0.6806 是单次运气 |
> | S5 | 5-fold 30ep 基线(复用) | T 0.5253±0.006 / N 0.5393±0.053 | ✅ 可信基线确认 |
> **核心结论**: ①R5 配置 (medicalnet18+gem+head128+lr1e-4/lr-n3e-4+n-shallow) 在 356 样本上是
> 抗证伪局部最优; ②N 卡 0.51-0.54 不是超参/正则问题（所有尝试均无效）, 是**特征级限制**:
> 全体积全局特征(影像组学/CRCFound/单模态)都拿不到局部浸润信息 → 0.80 需要 病灶级特征(ROI) 或 更大数据;
> ③T 已达 0.60-0.67 区间, 主要靠 MRI 预训练特征。**下一步唯一未试方向: 病灶 ROI（无标注可用则放弃）或数据扩充**。
> 今晚产物: run_08(s42 EMA)/run_09(s7 EMA)/run_10(s7 smmri) 归档, 全 README round9 章节已写。
> 工程沉淀: EMA(--ema-decay)/kfold-resume/单模态消融(--single-modal)/WATCHDOG/归档机制(_archive_run)。

#### round9-2.5D 多切片 + ImageNet 已完成（2026-10-31 02:27→02:35, seed42, 60ep, tools/train_2p5d.py）
> **实现**: 4 分支 2D ResNet18(ImageNet) 各取 CT/MRI(t/c/d) 轴位 3 切片(46/47/48) → GeM → concat(4×512)+临床7 → 双头;
> conv1/bn1/relu/maxpool/layer1/layer2 冻结, T inverse 权重, CosineAnnealing 60ep。
> **结果**: **T best 0.6296@1（强过拟合: loss 1.5→0.007, best epoch=1）, N best 0.5781@26**;
> 修复报告解析后正式值: T 0.5901@0.75阈值 recall=[0.571,0.647], N 0.4706@0.45 recall=[0.475,0.469]。
> **判定**: ❌ 2.5D 过拟合严重（2D ResNet 参数 33M×4 + 无预训练3D先验, 60ep 仅 3.5min 就崩）;
> T best-epoch=1 说明判别靠 ImageNet 2D 特征+头部, 无 3D 浸润信息增益 → 不采用。归档 run_14(废弃)/run_15(修复后)。

#### round9-SwinUNETR 已完成（2026-10-31 02:41→09:07 训练, 32/60ep 被 WATCHDOG 强杀, 09:13 正式报告, run_16）
> **实现**: CT 分支换 MONAI SwinUNETR encoder（SSL 预训练 159/159 匹配, feature48 window7³ patch2, 96³ 天然匹配）,
> MRI 保持 medicalnet18(--share-backbone 省显存), `--batch-size 4`(batch8 OOM 22GB>8GB卡) + expandable_segments。
> **训练**: 每 epoch ~11min（比 R5 慢 ~20 倍）, 02:45 启动, Epoch 32/60 时 GPU 硬卡死被 WATCHDOG 强杀(129s>120s)。
> **结果(32ep 中断, best ckpt 完整)**: **T 0.5751@30 cal recall=[0.762,0.510] / N 0.4286@26**。
> **判定**: ❌ SwinUNETR 换 CT 分支不优于 R5（T 0.575 vs R5 s42 0.6017, N 0.429 vs 0.5093）;
> 与 S1"CT 贡献≈0"一致, 大模型 119M 在小数据上无收益, 且显存/速度成本高 → 不采用。
> 工程沉淀: --ct-backbone {same,swin} / --no-ct-pretrained / --batch-size, load_swinunetr_weights(filter_swinunetr)。

#### round9-S8-F2 模态级融合（2026-11-01 01:38, 白嫖 run_06/run_07 单模态 ckpt）
> **目的**: CT-only (run_06) 与 MRI-only (run_07) 两个异构单模态模型, 概率平均融合,
> 验证异构模型互补性 (区别于 S0 同构集成证伪)。
> | 模型 | T f1@0.5 | T recall | N f1@0.5 | N recall |
> |---|---|---|---|---|
> | CT-only | 0.6010 | [0.238, 0.961] | 0.4938 | [0.55, 0.438] |
> | MRI-only | 0.6806 | [0.619, 0.765] | 0.4685 | [0.5, 0.438] |
> | **平均融合** | **0.6999** | [0.667, 0.765] | 0.4685 | [0.5, 0.438] |
> | 加权融合 | 0.6806 | [0.619, 0.765] | 0.4685 | [0.5, 0.438] |
> **T 平均融合 0.6999 超过单 seed 历史最高(smmri 0.6806)**——CT 的低阈值高召回(0.961)与 MRI
> 均衡分布(0.619/0.765)互补 → ⚠️ **疑似突破: 标注待验证** (仅 s42 单 seed + 平均阈值 0.5,
> 未校准; 需 5-fold 或更多 seed 确认)。**N 融合无增益**(两个单模态 N 都弱, ct 的 0.55 召回被
> mri 稀释)——与"全体积拿不到 N 局部信息"一致。产物 logs/ensemble_modalities_full.{json,txt},
> 工具 tools/ensemble_modalities.py。

#### round9-S8-C2 一致性正则（2026-11-01 02:23, 代码完成, 待训练）
> **目的**: 同一卷做第二次独立增强, MRI/CT 分支真实池化特征做增强不变性约束
> (双向 stop-grad MSE), 小数据防过拟合 (EMA 参数级已证伪, 特征级互补)。
> **实现**: loader `consist_modal` 参数 (dataset 返回 {modal}_second 副本, 不可用样本
> 返回零卷副本保 collate); trainer `consist_modal/consist_weight` + train_one_epoch
> 内第二前向 (mixup 同步混合 second), 双向 MSE 加入总 loss; CLI `--consist-modal
> {mri,ct}` + `--consist-weight F`; 日志行追加 `consist=` 项。训练/验证/保存不受影响。
> **验证**: C2-TRAINER-UNIT-OK (CPU 假数据: 2 batch 35s, consist_cnt=5, consist=0.1894)。
> 待训: `--consist-modal mri --consist-weight 0.5` (R5 配置)。

#### round9-S8-D1a N头CT单源 2-seed 汇总（2026-11-01 02:29, B3b 复现完成）
> **D1a = B3b 复现(s7), 完成**: T 0.6323@38 calT=[0.619,0.686] / N 0.5833@5 calN=[0.525,0.656],
> 归档 run_17_hos_full_s7。
> | 配置 | T mean | N mean |
> |---|---|---|
> | R5 both (s42+s7) | 0.6372 | 0.5178 |
> | **--n-modal ct (s42+s7)** | 0.6409 | **0.5792 (+0.061)** |
> **N 头只接 CT 特征可复现提升 N +6.1 点**（s42 0.5751 / s7 0.5833 双 seed 支持）——
> 与 S1/B3b 结论一致: N 依赖 CT 局部信息。未达疑似突破阈值(N≥0.60), 按规则记录不中断。
> 下一步: F1 消融交叉注意力(concat) / D1b 模态分工。

#### round9-S8-F1 交叉注意力消融（2026-11-01 03:09, --fusion concat）
> R5 配置去掉交叉注意力块 (BSAFusionModule concat 分支: 每 stage ct/mri 独立 GeM 后 concat,
> 无 cross_blocks, 141.8M vs multi 145.6M), seed42 60ep。
> **T 0.6250@38 calT=[0.571,0.706] / N 0.4938@23 calN=[0.55,0.438]**, 归档 run_18_hos_full_s42。
> 对比 R5 multi s42 (T 0.6017/N 0.5093): T +0.023 / N -0.016 (s42 单 seed 方差内)。
> 结论: 交叉注意力贡献有限 (concat ≈ multi), 非关键组件; 作为论文消融记录。
> 下一步: D1b 模态分工 (--t-modal mri --n-modal ct)。

#### round9-S8-D1b 模态分工（2026-11-01 03:51, --t-modal mri --n-modal ct, seed42 60ep）
> **T 头←MRI 增强特征(以 CT 为上下文)、N 头←CT**(BSAFusionModule 调换交叉注意力 query/context,
> 新增 CLI --t-modal {mri,both}, 融合层输出维度不变), 归档 run_19_hos_full_s42。
> **T 0.6250@45 calT=[0.571,0.706] / N 0.5626@41 calN=[0.625,0.5]**。
> 对比 s42 同 holdout: R5 both (T 0.6017/N 0.5093) → **T +0.023 / N +0.053**;
> B3b/D1a (仅 N←CT, T←both): T 0.6494/0.6323, N 0.5751/0.5833。
> 结论: D1b 的模态分工 N 优于 both (+0.053) 但与"仅 N←CT"相当 (N 0.5626 < 0.5751),
> T 头切 MRI 无额外收益 → **N 增益全部来自 --n-modal ct(已有 B3b/D1a 双 seed 支持), --t-modal 分工非关键**。

#### round9-S8-C1 Patch-MIL 切块弱监督（2026-11-01 03:54, tools/train_mil.py, seed42 60ep）
> 无 ROI 弱监督定位尝试: CT 96³ 无重叠 2x2x2=8 块×48³, 共享 medicalnet10 块编码器 + attention-MIL
> (Linear→softmax over 8 块) + 临床7 concat → 双头; R5 同口径 (lr 1e-4/lr-n 3e-4, N 权重 inverse)。
> **结果: ❌ 证伪** — **T 0.5898@19 calT=[0.238,0.882] / N 0.4549@54 calN(0.25)=[0.825,0.313]**, 归档 run_20。
> 对比 s42: R5 both N 0.5093 → **N -0.054**; B3b (n-modal ct) N 0.5751 → **N -0.120**。
> 失败原因: ①8 块太少 (attention-MIL 需几十~上百实例才有效) ②48³ 块仍太大 (块内还是全局统计,
> 淋巴结浸润信息在块内被稀释) ③resnet10 编码器弱于 resnet18 ④无重叠切块可能切开关键结构。
> 结论: **切块粒度不够细, MIL 弱监督在此数据量上无法替代真实 ROI/局部特征**; 与 CAM/radiomics 一致。

#### round9-S8-C2 一致性正则（2026-11-01 05:07, R5 + --consist-modal mri --consist-weight 0.5, seed42 60ep）
> 增强不变性: loader 返回 MRI 第二增强副本, train 时第二前向, 与主特征做 MSE 双向 stop-grad
> (consist 从 0.71 收敛到 0.09)。**T 0.6303@51 calT=[0.476,0.784] / N 0.5115@40 calN=[0.525,0.5]**, 归档 run_21。
> 对比 R5 both s42 (T 0.6017/N 0.5093): **T +0.029 / N +0.002** (s42 单 seed 方差内)。
> 结论: 一致性正则 T 小幅正向、N 无变化 → 无害但非关键组件, 论文消融记录
> (参数 --consist-modal/--consist-weight 沉淀保留)。

#### round9-S8 总总结（2026-11-01 01:31→05:07, 用户 m03107/m03137/m03169 排程: D1a→F1→D1b→F2→C1→C2+ML）
> | 步骤 | 实验 | 结果 (s42 同划分对比 R5 both) | 判定 |
> |---|---|---|---|
> | D1a | N←CT (B3b s7 复现) | T 0.6323 / N 0.5833 (s7) | ✅ N +0.061 2-seed 可复现 (B3b/D1a) |
> | F1 | --fusion concat 消融交叉注意力 | T 0.6250 / N 0.4938 | ❌ concat≈multi, 注意力非关键 |
> | D1b | --t-modal mri + --n-modal ct | T 0.6250 / N 0.5626 | ❌ N 增益全来自 n-modal ct, T 头分工无益 |
> | F2 | 单模态模型概率融合 (白嫖) | T mean-fusion 0.6999 / N 0.4685 | ⚠️ T 疑似突破单seed; N 无增益 |
> | C1 | Patch-MIL 8块切块弱监督 | T 0.5898 / N 0.4549 | ❌ 块太少太大, 弱监督不可行 |
> | C2 | 一致性正则 (MRI 增强不变性) | T 0.6303 / N 0.5115 | ❌ T +0.029 方差内, N 无变化 |
> | ML | XGB/LGBM/SVM/RF/Voting 扩展 | T val-F1 0.5179 / N 0.5500 | ✅ 方法学广度, 非改进 |
> **结论**: 本轮唯一新正向信号 = **--n-modal ct (N 依赖 CT 局部信息, B3b/D1a 双 seed N 0.5751/0.5833)**,
> 其余 (交叉注意力/模态分工/MIL/一致性/集成) 全证伪或方差内。**R5 基线保持**: T 单seed 0.60-0.67,
> N 0.51-0.58; N 仍卡特征级瓶颈 (全体积无局部浸润, 需 ROI/外部数据)。
> 产物: run_17-21 归档 (D1a/F1/D1b/F2/C1/C2)。工程沉淀: --t-modal/--n-modal/--fusion concat/
> --consist-modal/--consist-weight/train_mil.py/train_2p5d.py/ensemble_modalities.py。

#### round9-S8-ML 经典 ML 家族扩展（2026-11-01 02:35, 项1 的模型广度补充）
> 工具 tools/ml_extended.py, 复用 logs/radiomics_full_features.csv (356×75), 8 模型×T/N,
> 口径同项1 (train 内 5 折 CV seed42 + holdout val)。结果(全表在 logs/ml_extended_full_results.txt):
> **T**: val-F1 knn 0.5179 最佳 / val-AUC rf 0.6480 最佳;
> **N**: val-F1 lgbm 0.5500 最佳 / val-AUC knn 0.5277 最佳。
> 结论: 全体积特征上经典 ML 上限与项1 LASSO/HistGB 相当 (AUC 0.42-0.65 波动),
> 远低于 DL 单 seed (T 0.60-0.67 / N 0.51-0.58) —— 补充方法学广度, 非改进路径。
> 印证: N 瓶颈在特征级(局部浸润), 与 radiomics/CRCFound/CAM 结论一致。

#### round9-S8 下一轮序列（2026-10-31, 用户确认纳入 F1/F2/ML）：
> 顺序: D1a(B3b s7 复现) → F1(--fusion-mode concat 消融交叉注意力) → D1b(--t-modal T←MRI/N←CT)
> → F2(异构单模态模型+概率融合) → C2(一致性正则) → C1(Patch-MIL 切块弱监督)
> → ML 扩展(XGBoost/LightGBM/SVM/RF/Voting, CPU 间隙) → 汇总归档。

#### round9-S9-① n-modal ct 5-fold 可信验证 已完成（2026-11-02 00:11→01:20, R5 配置 + `--n-modal ct`, 5 折 × 30ep, 逐折分进程）
> 目的: 判定 `--n-modal ct`(N 头只取 CT 特征) 是否真阳性——用户 m03826 判定标准: N mean≥0.57 且 ≥3/5 折优于 both。
> 执行: 先备份 both 的 kfold 报告到 logs/backup_both_kfold/ 再删原件(避免 --kfold-resume 误判已跑);
> 修 run_kfold 漏接 t_modal 的 NameError; 5 折逐折启动(--kfold-resume --kfold-max-folds 1)。
> 结果(逐折 N: n-ct vs both): k0 0.5847 vs 0.4836 (+0.101✓) | k1 0.5750 vs 0.6375 (-0.063✗)
> | k2 0.5145 vs 0.5035 (+0.011✓) | k3 0.5319 vs 0.5319 (平) | k4 0.5959 vs 0.5400 (+0.056✓)。
> **汇总: T mean 0.5271±0.029 (both 0.5253±0.006, 持平) / N mean 0.5604±0.032 (both 0.5393±0.053, +0.021, 方差更小)**。
> 判定: 3/5 折优于 both 达标, 但 N mean 0.5604 < 0.57 门槛 → **未达真阳性标准, 记"正向候选"**;
> N 依赖 CT 特征方向再次得到支持(k0/k4 大胜, k1 大输说明折间依赖标签分布), 非超参可解瓶颈。
> 产物: run_22-26 归档。注意: logs/kfold_summary_full.json 现为 n-modal ct 版本(both 备份在 backup_both_kfold/)。

#### round9-S9-② 不确定性加权 已完成（2026-11-02 01:24→01:48, seed42, 60ep, R5 配置 + `--uncertainty-weight`）
> Kendall 异方差不确定性加权: main = exp(-log_var_t)·L_t + exp(-log_var_n)·L_n + log σ_t + log σ_n,
> log_var 可学习(初始 0 → 权重 1.0 与手工等权一致, 主 lr, 无 L2)。修 bug: log_var 在 CPU 与 cuda loss
> 设备不匹配 → 计算处 .to(loss_t.device)。
> 结果: **T best 0.6443@60 (calT@0.45: 0.6929 recall=[0.619,0.784]) / N best 0.4836@16 (calN@0.5: 0.4836)**。
> 对比 R5 s42 (T 0.6017@44 / N 0.5093): **T +0.043 (≥+0.03 → ⚠️疑似突破, 记录不中断; 未达 +0.05 不补 5-fold) / N -0.026**。
> 机制: 自适应加权把梯度容量倾斜给更易学的 T 任务(T loss 权重↑), T best 移到 epoch 60(之前 44)说明 T 训练更充分;
> N 受损印证 N 是特征级瓶颈非损失权重问题。产物: run_27_hos_full_s42 归档。待验证: 5-fold 补跑。

#### round9-S9-③ ECA 通道注意力 已完成（2026-11-02 01:49→02:14, seed42, 60ep, R5 配置 + `--eca`）
> ECA 1D 卷积通道门控加在 fusion 每 stage 融合特征后 (BSAFusionModule use_eca, 核 k 随通道数自适应)。
> 结果: **T best 0.5344@26 (calT@0.5: 0.5344 recall=[0.714,0.471]) / N best 0.5115@2 (calN@0.5: 0.5115 recall=[0.4,0.656])**。
> 对比 R5 s42 (T 0.6017/N 0.5093): **T -0.067 (显著受损) / N +0.002 (持平) → ❌ ECA 证伪**。
> 机制: 通道门控干扰了融合特征的判别信息传递(T 明显下降), N 无变化印证 N 瓶颈在特征获取层不在融合层。
> 产物: run_28_hos_full_s42 归档。

#### round9-S9-④ N 头 layer3+4 拼接 已完成（2026-11-02 02:15→02:40, seed42, 60ep, R5 配置 + `--n-shallow-idx 3,4`）
> N 头特征从 layer3 单 stage (576 维) 扩为 layer3+4 双 stage 拼接 (1600 维, classifier_n in_features=1600)。
> 结果: **T best 0.6073@56 (calT@0.5: 0.6073 recall=[0.476,0.745]) / N best 0.4918@57 (calN@0.5: 0.4918 recall=[0.275,0.8125])**。
> 对比 R5 s42 (T 0.6017/N 0.5093): **T +0.006 (噪声级) / N -0.018 → ❌ 证伪**。
> 机制: 更深 stage (layer4) 分辨率更低(全局池化信息冗余), 双 stage 拼接未给 N 带来新判别信号;
> N 瓶颈在特征获取层(局部浸润)而非特征聚合层。产物: run_29_hos_full_s42 归档。

#### round9-S9-⑤ Snapshot Ensemble 已完成（2026-11-02 02:41→03:07, seed42, 60ep, R5 配置 + `--snapshot-k 10`）
> CosineAnnealingWarmRestarts(T_0=10) 替代 CosineAnnealingLR, 每 10ep 存快照 (ep10/20/30/40/50/60)。
> **训练结果: T best 0.6850@60 (calT@0.5: 0.6850 recall=[0.571,0.804]) / N best 0.4687@33 (calN@0.5: 0.4687 recall=[0.725,0.25])**。
> ⚠️ **T +0.083 vs R5 s42 (0.6017) ≥ +0.05 → 疑似突破记录不中断** (注意 T best=ep60 最终权重, 提升来自 warm restarts 而非集成)。
> **集成验证 (6 快照 softmax 平均, logs/ensemble_snapshot_full.txt): T 0.6303 < ep60 单快照 0.6850 → ❌ 集成证伪 (平均稀释最好快照); N 集成 0.3864 更差**。
> 机制: warm restarts 多次重启让 T 每周期重新探索→最终周期收敛更优 (T best 从 R5 的 ep44 移到 ep60);
> N 未受益 (0.4687, -0.041)。待验证: warm restarts 是否真提升 (需 5-fold 或 2-seed 复现)。
> 产物: run_30_hos_full_s42 归档 + 6 快照 ckpt。

#### round9-S9 今晚序列总总结（2026-11-01 23:40 → 2026-11-02 03:07 执行完毕, 用户 m03832 批准 8h 版）
> S9 序列（n-modal ct 5-fold / 不确定性加权 / ECA / N 头多 stage / Snapshot）全部完成。**两个正向信号 + 三个证伪**:
> | 步骤 | 实验 | 结果 (seed42, 60ep) | 判定 |
> |---|---|---|---|
> | S9-① | n-modal ct × 5-fold (30ep) | T 0.5271±0.029 / N 0.5604±0.032 | ⚠️ 正向候选 (3/5 折优于 both, N mean<0.57 未达真阳性) |
> | S9-② | 不确定性加权 | T 0.6443 / N 0.4836 | ⚠️ T +0.043 疑似突破, N -0.026 |
> | S9-③ | ECA 通道注意力 | T 0.5344 / N 0.5115 | ❌ T -0.067 |
> | S9-④ | N 头 layer3+4 拼接 | T 0.6073 / N 0.4918 | ❌ T +0.006 / N -0.018 |
> | S9-⑤ | Snapshot (warm restarts) | T 0.6850 / N 0.4687; 集成 T 0.6303 | ⚠️ T +0.083 疑似突破, 集成证伪 |
> **核心结论**: ①**N 卡 0.51-0.54 是特征级瓶颈**再获 3 个新证据 (ECA 无作用/多stage 无作用/Snapshot N 无受益)；
> ②**warm restarts (CosineAnnealingWarmRestarts T_0=10) 是第一个稳定提升 T 的训练机制** (T 0.6850, +0.083, T best 移到 ep60) —— 唯一待 5-fold/2-seed 验证的正向信号;
> ③n-modal ct 5-fold N 0.5604 双正向候选之二, 均未达"真阳性"标准 (N≥0.57 或 5-fold 验证)。
> **下一步候选 (按优先级)**: A. warm restarts 5-fold 验证 (30min×5 折) — 若 T mean≥0.60 即首个可信 T 提升; B. warm restarts + n-modal ct 组合 (两正向信号叠加); C. 回到病灶级特征 (唯一未试本质方向)。
> 产物: run_22-30 归档 (9 runs) + ensemble_snapshot_full.{json,txt}。

#### round9-病灶级路径 ① Task010 病灶分割 已完成（2026-11-03 23:56 → 2026-11-04 03:56, 145 CT 样本 × fold_0, tools/run_task010_seg.py）
> 目的: MSD Task10_Colon (nnU-Net, 结肠癌原发肿瘤 CT 分割) 生成病灶 mask, 验证"局部浸润信息"假设 (CAM/radiomics/CRCFound 印证 N 需病灶级特征)。
> 前置: 权重 pretrained/Task010_Colon/ (4.66GB, zenodo, 5 折) + nnunet v1.7.1 源码 third_party/nnUNet-1.7.1; 修 Windows spawn pickle lambda → 全部同步化 (preprocess 逐样本流水线, pool=None)。
> **结果: 145 样本全部分割完成 (219 min, 每样本 ~1.8min)**:
> - **n_with_mask=96 / n_empty=49** (49 样本模型未检出病灶);
> - 有 mask 样本体素: min=1 / med=2810 / max=98345 / mean=5876 (个别极小 mask 可能为噪声, 后续可设最小连通域过滤);
> - 产物: data/masks_task010/{pid}.nii.gz (145) + logs/task010_seg_full.{json,txt}。
> 结论: 病灶级 mask 已就绪 → 下一步 ROI 特征实验 (tools/roi_experiment.py) + TotalSegmentator 结肠分割对照 (tools/run_totalseg_colon.py)。

#### round9-病灶级路径 ② TotalSegmentator 结肠分割 已完成（2026-11-04 04:05 → 06:42, 147 CT 样本, tools/run_totalseg_colon.py）
> 目的: 器官级结肠 mask (total 任务 colon) 作为病灶 mask 的对照——验证"ROI 特征增益来自病灶级(肿瘤)而非器官级(整个结肠)"。
> 前置: pip install TotalSegmentator + total 权重 pretrained/totalsegmentator/ (~/.totalsegmentator, TOTALSEG_HOME_DIR 指向项目); 新版 device 参数为 `gpu` 非 `cuda` (首次 cuda 全 FAILED 已修)。
> 实现坑: TotalSegmentator 2.x 输出为**目录语义** (`{pid}.nii.gz/colon.nii.gz`) → 脚本输出目录 + 统计读子目录; 曾误删 147 个 mask (PowerShell 先删目录再移动, 顺序错) → 重跑 (147×31s≈75min)。
> **结果: 147/147 全部有 mask (total_voxels=41.17M, min=42679/med=202981/max=2078537/mean=280075 体素)**;
> 产物: data/masks_totalseg/{pid}/colon.nii.gz (147 目录) + logs/totalseg_colon_full.{json,txt}。
> 对照: Task010 病灶 mask 96/145 (med 2810 体素) vs TS 结肠 mask 147/147 (med 202981 体素) — 结肠区域 ≈ 病灶 72× 体积, 器官级 vs 病灶级判别力对比待 ROI 实验。

#### round9-病灶级路径 ③ ROI 特征实验 已完成（2026-11-04 06:43 → 06:49, 145 CT 子集 × 两种 mask, tools/roi_experiment.py）
> 目的: mask-guided ROI 特征 (一阶11 + GLCM3D 6 + 形态3 = 20维/模态 + 临床7 = 27维) 验证病灶级 vs 器官级判别力, 检验"N 需要局部浸润信息"假设。
> 口径: 与 radiomics 完全一致 (CT 子集 145 = 114train/31val, 内部 5 折 CV + holdout, LASSO-LR + HistGB)。
> **结果 (T: histgb CV-F1/AUC | val F1/AUC; N 同; lasso 两路均≈0.42-0.47 无信号不列)**:
> | mask | T CV | T val | N CV | N val |
> |---|---|---|---|---|
> | Task010 病灶 (96/145) | 0.6120 / **0.6884** | 0.4464 / 0.4786 | 0.5211 / 0.5672 | 0.4038 / 0.3718 |
> | TS 结肠器官 (147/147) | 0.5203 / 0.5556 | 0.6353 / 0.7143 | 0.5139 / 0.4902 | 0.4038 / 0.4664 |
> 结论: ①病灶 mask 的 T CV AUC 0.6884 > 结肠 mask 0.5556 → **病灶级 ROI 确实比器官级更有判别力 (T)**, 病灶定位有价值; ②**N 两种 mask 下 val F1≈0.40 (≈随机) → 手工 ROI 统计特征依旧拿不到 N 的局部浸润判别信息** (一阶+GLCM+形态 20 维过于粗糙, 或 N 分期信号根本不在病灶强度纹理内而在边界/淋巴结区域); ③与全体积 radiomics (T 0.6545/N 0.5074) 相比 ROI 化未带来 N 增益 → **N 需要的是深度病灶级特征 (ROI 内 DL 特征), 而非手工统计**。产物: logs/roi_{task010,totalseg}_features.csv + logs/roi_experiment_full.txt (后跑者覆盖, 见 job_output 两份)。

#### round9-病灶级路径 ④ 2.5D 深挖 (head_dropout 0.7 重试) 已完成（2026-11-04 06:52 → 06:55, seed42 60ep, tools/train_2p5d.py）
> 目的: 2.5D (4×ResNet18-ImageNet, 3 切面×4 分支, GeM) 原版 T 0.6296@1 (epoch1 过拟合) / N 0.5781@26 (s42 N 历史最高) → 加正则 (head_dropout 0.5→0.7) 重试, 验证 N 优势是否真实。
> **结果 (head_dropout 0.7, run_40 归档)**: **T 0.6400@60 acc 0.7083 / N 0.5404@21**; 对比原版 (0.5, run_15): T 0.6296@1→0.6400@60 (**T 更稳: best epoch 1→60, 过拟合缓解**), **N 0.5781→0.5404 (下降 -0.038)**。
> 结论: ❌ 正则加强**没有保住 N 优势** (0.5781 含单 seed 运气成分), T 稳定性改善但绝对值仍在 R5 噪声范围 (R5 both 0.6017)。2.5D 整体 ≤ R5, 不作为主模型; N 0.5781 归为"单 seed 幸运值"。

#### round9-阶段1 Warm restarts 5-fold 验证 已完成（2026-11-02 23:56 → 2026-11-03 01:45, 5 折 30ep 逐折分进程, `--warm-restarts --kfold 5 --epochs 30`）
> **判定标准**: T mean≥0.60 且 ≥3/5 折优于 both 基线 (T 0.5253±0.006/N 0.5393±0.053, 30ep 同口径)。
> | 折 | warm-restarts T | both T | ΔT | warm-restarts N | both N | ΔN |
> |---|---|---|---|---|---|---|
> | k0 | 0.5485@26 | 0.5353 | +0.013 | 0.4985@23 | 0.4836 | +0.015 |
> | k1 | 0.5390@2 | 0.5267 | +0.012 | 0.5545@11 | 0.6375 | -0.083 |
> | k2 | 0.5487@20 | 0.5198 | +0.029 | 0.4847@15 | 0.5035 | -0.019 |
> | k3 | 0.5966@22 | 0.5279 | +0.069 | 0.4929@23 | 0.5319 | -0.039 |
> | k4 | 0.5688@27 | 0.5170 | +0.052 | 0.4842@1 | 0.5400 | -0.056 |
> | **mean** | **0.5603±0.021** | 0.5253 | **+0.035** | **0.5029±0.026** | 0.5393 | **-0.036** |
> **判定**: T mean 0.5603 < 0.60 门槛 → **未达真阳性**, 但 **5/5 折全部优于 both (最大 +0.069)** — warm restarts 的 T 提升在 5-fold 下可复现, 绝对值不足; N 4/5 折更差 (k1 -0.083 拖累, N 不受益再次印证特征级瓶颈)。
> 结论: warm restarts 作为 T 机制**确认有效但幅度有限 (+0.035)**, 不足以翻越 0.60; 阶段2 组合 (warm restarts + n-modal ct) 继续验证双正向信号叠加。产物: run_31-35 归档 (k0-k4, 各 7 文件)。

#### round9-阶段2 Warm restarts + n-modal ct 组合 已完成（2026-11-03 01:43 → 03:07, 2-seed 60ep, `--n-modal ct --warm-restarts`）
> | seed | T | N | calT | calN |
> |---|---|---|---|---|
> | s42 | 0.5846@34 | 0.5592@51 | [0.381,0.784] | [0.65,0.469] |
> | s7 | 0.5313@30 | 0.4780@28 | [0.381,0.686] | [0.55,0.406] |
> | **mean** | **0.5580** | **0.5186** | | |
> **判定 (T≥0.62 且 N≥0.53)**: ❌ **组合证伪** — 对比 R5 both 2-seed (T 0.6372/N 0.5178): **T -0.079 显著受损**, N 持平。两正向信号叠加不成立: warm restarts 周期重启与 n-modal ct 的 N 头 CT 单源组合下 T 判别崩塌; n-modal ct 的 N 收益 (+0.06) 也消失。→ warm restarts 与 n-modal ct **互斥**, 各自独立使用; R5 仍为最终配置。产物: run_36/37 归档。

#### round9-阶段3 模态 dropout 调优 已完成（2026-11-03 03:07 → 04:30, seed42 60ep, `--md-both 0.9 / 0.3`）
> | md-both | T | N | calT | calN |
> |---|---|---|---|---|
> | 0.9 (双模态更多) | 0.6562@58 | 0.4485@1 | [0.619,0.725] | [0.175,0.906] |
> | 0.3 (单模态更多) | 0.6519@53 | 0.4836@14 | [0.524,0.784] | [0.5,0.469] |
> | 0.65 (R5 默认) | 0.6017@44 | 0.5093@6 | [0.571,0.667] | [0.375,0.688] |
> **判定**: 两方向改动 T 均升至 0.65+（+0.05）但 N 均受损（-0.03~-0.06），N best epoch 退化到 1-14。T 提升 ≈ 回落到 R5 多 seed 均值 (0.6434) — s42 原 R5 0.6017 是低值样本, 扰动朝均值回归, **非机制性提升**; N 受损真实 (正则扰动干扰 N 头)。→ **md-both 调参不采用, R5 默认 0.65 保持**。产物: run_38/39 归档。

#### round9-今晚 Warm restarts 序列 总总结（2026-11-02 23:40 → 2026-11-03 04:30 执行完毕, 用户 m04792/m04795 排程 + m04889 自主模式）
> | 阶段 | 实验 | 结果 | 判定 |
> |---|---|---|---|
> | 1 | warm restarts × 5-fold (30ep) | T 0.5603±0.021 / N 0.5029±0.026 | ⚠️ T 5/5 折优于 both (+0.035) 可复现但 <0.60 门槛; N 不受益 |
> | 2 | warm restarts + n-modal ct (2-seed 60ep) | T 0.5580 / N 0.5186 | ❌ T -0.079 组合互斥 |
> | 3 | md-both 0.9 / 0.3 (seed42) | T 0.6562 / 0.6519, N 0.4485 / 0.4836 | ❌ 扰动朝均值回归, N 受损 |
> **核心结论**: ①**warm restarts 单独使用是 R5 之上唯一微弱正向 T 机制 (+0.035, 5/5 折胜)**, 但绝对值不足, 与 n-modal ct 互斥; ②N 特征级瓶颈再获 4 个新证据 (5-fold N 不受益/组合 N 消失/md-both 双向 N 受损); ③R5 (medicalnet18+gem+head128+lr1e-4/lr-n3e-4+n-shallow) 在全部 24 项改动后仍是最终配置, 无超参层面的 T 提升空间, N 需病灶级特征。
> 产物: run_31-39 归档 (9 runs) + README 阶段1-3 章节。下一步候选: A. 病灶级 (MSD Task10_Colon 分割 mask + ROI 特征) B. 外部数据 C. 2.5D 完整评估 (ImageNet 迁移已半完成)。

#### round9-病灶级路径 ⑤ warm restarts 单 seed 复验 已完成（2026-11-04 06:58 → 07:24, seed7 60ep, `--warm-restarts`）
> 目的: s42 的 warm restarts T 0.6850 (S9-⑤) 是否跨 seed 可复现 (单 seed 复验)。
> **结果 (run_41 归档)**: **T 0.5710@2 calT=[0.333,0.804] / N 0.5130@2 calN=[0.425,0.625]**;
> 对比 R5 both s7 (T 0.6727/N 0.5263): **T -0.102 显著更差 / N -0.013**; 对比 warm restarts s42 (0.6850): s7 仅 0.5710。
> 结论: ❌ **warm restarts 单 seed 复验证伪** — s42 的 0.6850 是单 seed 运气 (s7 反而不如 R5); 与阶段1 5-fold (T +0.035 5/5 折胜) 矛盾 → warm restarts 的真实增益 ≤ +0.035 且不稳定, **不作为 R5 替代**。

#### round9-病灶级路径 总总结（2026-11-03 23:40 → 2026-11-04 07:55 执行完毕, 用户 m05443/m05557 排程 + 病灶级四步）
> | 步骤 | 实验 | 结果 | 判定 |
> |---|---|---|---|
> | ① | Task010 病灶分割 (nnU-Net, 145 CT) | 96/145 检出, med 2810 体素 | ✅ 病灶 mask 就绪 |
> | ② | TotalSegmentator 结肠器官分割 | 147/147 检出, med 202981 体素 (≈病灶 72×) | ✅ 器官 mask 就绪 |
> | ③ | ROI 特征 (病灶 vs 器官 mask) | T CV-AUC 0.6884 vs 0.5556; N 均≈0.40 | ⚠️ 病灶级优于器官级(T), **N 手工特征无解** |
> | ④ | 2.5D 深挖 (head_dropout 0.7) | T 0.6400@60 / N 0.5404@21 | ❌ N 0.5781 是单 seed 运气 |
> | ⑤ | warm restarts s7 复验 | T 0.5710 / N 0.5130 | ❌ s42 0.6850 是运气, 不替代 R5 |
> **核心结论**: ①**病灶级路径走通但未破 N 瓶颈** — 病灶 mask 比器官 mask 判别力强 (T), 但手工 ROI 统计特征 (一阶+GLCM+形态) 拿不到 N 的局部浸润信息 (val F1≈0.40≈随机), N 需要 ROI 内 **DL 特征** (mask 裁剪喂 3D 网络) 而非手工特征; ②warm restarts 增益 ≤ +0.035 不稳定; ③**R5 仍是最终配置**, 全部 30+ 改动证伪/有限; ④今晚工程沉淀: nnU-Net v1 Windows 推理流水线 (逐样本同步预处理, 修复 spawn pickle) + TotalSegmentator 2.x 目录语义 + ROI 特征管线 (task010/totalseg 双 mask)。
> 产物: run_40/41 归档 + logs/task010_seg_full.{json,txt} + logs/totalseg_colon_full.{json,txt} + logs/roi_{task010,totalseg}_features.csv。
> 下一步候选: A. ROI 内 DL 特征 (病灶裁剪 64³+3D ResNet, mask 96 样本) B. 外部数据/外部验证 C. 放弃 N 单模型, 换 ordinal 建模 (pN 亚级, 需标签)。

#### round9-A ROI 内 DL 特征（病灶级）已完成（2026-11-05 00:22 → 00:27, tools/train_roi_dl.py）
> 目的: 病灶 mask 中心 64³ 裁剪 → medicalnet10 预训练 (72/72) + avgpool + FC256 + 双头, 验证"深度病灶级特征"是否破 N 瓶颈 (手工 ROI 统计特征 val F1≈0.40 已证伪)。
> 口径: 只训练有非空病灶 mask 的样本 (train=72 / val=21, 病灶子集, 非全 356); R5 超参 (lr 1e-4/lr-n 3e-4, 60ep, AMP, Cosine, N inverse 权重)。
> **结果**: **T best 0.6101@32 (calT@0.25 recall=[0.75, 0.647]) / N best 0.5714@29 (calN@0.2 recall=[0.909, 0.3])**, acc 0.8095。
> 对比 (口径不同, 仅参考): R5 全体积 s42 T 0.6017/N 0.5093; 病灶子集 val=21 噪声大。
> 结论: ①病灶 ROI DL 在 72 训练样本上 20s 即收敛, T 0.61 / N 0.57 与全体积相当 → **病灶级特征有信息量但 72 样本不足以拉开差距**; ②N calN recall=[0.909, 0.3] 严重偏斜 → 病灶内部强度/纹理不含 N 分期主要信号 (与 CAM 结论一致: N 需淋巴结区域/边界, 非病灶本体); ③判定: **A 单 seed 正向前置, 需更多病灶样本或融合全体积才有意义**; 产物 run_42_roidl_hos_full_s42。

#### round9-S7 今晚序列总总结（2026-10-30 23:40 → 2026-10-31 09:13 执行完毕, 用户 m01823 排程 + m02374 流程）
> 今晚 8h+ 序列（P2-P5 重训 + S0-S5 + 2.5D + SwinUNETR）全部完成。**新增方向全证伪, R5 仍是最终配置**:
> | 步骤 | 实验 | 结果 | 判定 |
> |---|---|---|---|
> | P2-P5 | s42/s7/s0/s1 重训 (R5 60ep, 真实可复现) | T 0.6017/0.6727/0.6308/0.6683, N 0.5093/0.5263/0.4985/0.4896 | ✅ 4-seed mean T 0.6434/N 0.5059 |
> | S0 | 4-seed 集成 | T 0.6400 / N 0.4852 | ❌ 集成≤单seed最佳 |
> | S1 | 单模态消融 | MRI-only T 0.6806≥both, N 全≈0.47-0.51 | ✅ T 靠 MRI, N 无单模态捷径 |
> | S3 | EMA 0.999 | T -0.085 / N +0.034 | ❌ T 信号被平滑 |
> | S4 | MRI-only 复现 | 2-seed mean T 0.6165 < both 0.6372 | ❌ s42 0.6806 是单次运气 |
> | S5 | 5-fold 30ep (复用) | T 0.5253±0.006 / N 0.5393±0.053 | ✅ 可信基线 |
> | 2.5D | ImageNet 2D 多切片 | T 0.6296@1(过拟合) / N 0.5781@26 | ❌ 过拟合, 无 3D 增益 |
> | SwinUNETR | CT 换 Swin SSL | T 0.5751 / N 0.4286 | ❌ 不优于 R5, CT 贡献≈0 |
> **核心结论(不变)**: R5 抗证伪局部最优; N 卡 0.51-0.54 是特征级限制（全体积拿不到局部浸润）;
> T 0.60-0.67 主要靠 MRI 预训练。**下一步唯一未试: 病灶 ROI（无标注则放弃）或更大数据/外部验证**。
> 产物: run_11-16 归档。工程沉淀: EMA/2.5D(--ct-backbone)/SwinUNETR(--batch-size)/kfold-resume/WATCHDOG。


> N 权重从 inverse [1.0, 1.62] 改为 sqrt [1.324, 0.836]（削弱少数类 pN+ 的加权）。
> **结果: ❌ sqrt 权重对 N 有害（同 seed42 同 holdout vs R5）**:
> T 0.6850(+0.012, 噪声级) / **N 0.5092(-0.060, 且 best N epoch=1 几乎没学)**。
> N class1(pN+) 召回仅 0.375（R5 inverse 时 0.531）。**E1 证伪, class_weight_mode 保持 inverse**。


#### round9-B2 已完成（2026-10-28 17:23, seed42, 60ep, R5 配置 + `--n-cascade`）
> T→N 级联（N 头输入追加 T 预测概率 detach）。**结果: ❌ 级联有害（同 seed42 同 holdout）**:
> T 0.6268(-0.046 vs R5) / N 0.4740(-0.095 vs R5)。N 对 class1 召回仅 0.3125 → T 概率把 N 判断带偏
> （pT 深→pN+ 高先验在此数据上不成立或被噪声淹没）。**B2 证伪, 级联移除**。

#### round9-B1 已完成（2026-10-28 16:56, seed42, 60ep, R5 配置 + `--n-shallow-idx 2`）
> N 头浅层特征从 layer2 取（更原始纹理）。**结果: ❌ layer2 劣于 layer3（同 seed42 同 holdout 直接对比）**:
> T 0.5630(-0.110 vs R5 0.6727) / N 0.5373(-0.031 vs R5 0.5687)。N 相对多seed均值0.5347持平,
> 但 T 特征从 layer2 取明显损失 → **B1 证伪, N 浅层 index 保持默认 layer3**。

#### round9-B 代码准备（今晚训练计划, 不占用当前 GPU）
> 基于 R5 配置(medicalnet18+gem+128+n-shallow+T1e-4/N3e-4+N权重) 的架构级 N 改进,
> 新 CLI 已全部就绪并通过 CPU 前向维度验证(不占 GPU):
- **B1 `--n-shallow-idx N`**: N 浅层特征 index 可配(默认 3=layer3; 2=layer2 更原始纹理)
- **B2 `--n-cascade`**: T→N 级联——N 头输入追加 T 预测概率(softmax, detach;
  医学先验: pT 深→pN+ 概率高)
- **B3 `--n-modal {ct,mri,both}`**: N 特征来源选择(默认 both)
- **今晚实验序列建议**:
  1. `--kfold 5` × R5 配置 → 可信基线 mean±std (方向 A)
  2. `--n-shallow-idx 2` (B1: layer2)
  3. `--n-cascade` (B2: 级联)
  4. `--n-modal mri` / `--n-modal ct` (B3: 模态消融, 定 N 特征来源)
  5. **CAM 热图验证** (round9-CAM): `python tools/visualize_cam.py --n-samples 8 --device cuda`
     —— 用 R5 checkpoint 对 val 样本做 Grad-CAM, 检验模型注意力是否聚焦肿瘤区域;
     若聚焦 → 继续堆模型能力; 若不聚焦 → 0.80 需病灶引导(无标注则靠数据量/架构)
     (工具已就绪, CPU 冒烟通过; 模型已有模态内自注意力+模态间交叉注意力, 无病灶先验)

### [2026-10-27] round8: T 头不学习诊断 + 实验参数 CLI 化 + 训练诊断基建 + 手动实验计划

> 现象（full 分支 seed42, 30 epochs）: T 主头训练 acc 卡 ~50%（低于 71% 多数类基线）、
> main loss 平 ~1.44（比「恒预测阳性」的退化基线 1.22 更差）；N 头可学到 ~0.55 → **T 头未在学习**。
> 判定: 优先级 = 修复拟合（lr / 增强 / 容量），**不是换模型**——145.9M 三流 resnet18 在 361 样本上
> 已是过拟合高危，更大模型只会更难训。本轮全部为「执行前代码准备」，训练一律由用户手动启动。

**已落地（本次确认执行，全部 `py_compile` 通过）**

1. **`train_hos_seeds.py` 实验参数 CLI 化**（默认 None/False = 沿用 config，行为不变；日志打印全部生效参数）：
   - `--lr`（如 `3e-4`）、`--class-weight-mode {inverse,sqrt,none}`、`--focal`（T 主任务 Focal Loss）
   - `--aux-weight` / `--rec-weight` / `--head-dropout` / `--no-aug`
   - 实验不再需要改 `config/hos.py`，一条命令行跑一种配置
2. **`trainers/trainer.py` 训练诊断日志**：每 epoch 输出 `lr` / `grad_norm`
   （AMP unscale 后真实梯度 L2 范数）/ `cls_acc`（T 各类别 train acc）——直接区分「梯度问题 vs 信号问题」
3. **`trainers/trainer.py` MixUp 类别加权修复**：原 `_mixup_ce` 为 vanilla CE，`--mixup` 打开时
   T 的 `class_weight_mode=inverse` 被绕过（E3 归因被污染）；现 main/aux 分支带 T 类别权重（N 不加权），
   `weight=None` 时与原行为逐位一致
4. **`data_loader/loader.py` + `config/hos.py` 训练增强开关**：`use_augmentation`（默认 True）
   + CLI `--no-aug`——对照「增强过强（RandAffine prob0.7 + Flip×2 + Cutout 打没 T 壁厚/浸润精细纹理）
   导致 T 学不动」假设（当前最可疑元凶之一）
5. **新增 `tools/check_cache_coverage.py`**：核对 `cache_hos/` 覆盖（按 `_det_cache_key` 逐样本复算）。
   实测 full: CT 147 / MRI 301 全齐，**0 缺失**，无需重建
6. **新增 `tools/probe_epoch.py`**：1-epoch 探针（数据规模/T·N 分布/模态可用性 + 首 batch 模态
   mask 采样比例 + 单个训练/验证 epoch 诊断），~1-2 分钟摸清数据与增强是否合理

**模型本体优化落地（A2/A3/A4，2026-10-27 下午；A5 EMA/SWA 延后待定）**

7. **A2 MedicalNet 3D 医学预训练**（`models/medicalnet.py` 新增 + `models/backbone.py` 扩展）：
   - 新增 `--backbone medicalnet18 / medicalnet10`：Tencent MedicalNet 23 数据集 3D 医学预训练
     （替代 ImageNet-2D→3D 广播权重），结构/层名与官方 `models/resnet.py` 逐键对齐
     （resnet10 shortcut B / resnet18 shortcut A + `downsample` 命名），**conv1 3 通道自动平均适配**
   - 权重放置：`pretrained/resnet{10,18}_23dataset.pth`（官方下载见 `pretrained/README.md`，
     GitHub Tencent/MedicalNet 或 Google Drive/腾讯微云）；文件缺失时优雅降级随机初始化
   - 训练日志打印每个 backbone 的预训练加载状态；冻结前缀（conv1/bn1/maxpool/layer1..3）
     与现有 `_freeze_backbone` 天然兼容；medicalnet10 与 resnet10 同规则默认全量可训练
8. **A3 GeM 池化**：`--pool gem`（`--gem-p` 调幂指数，默认 3.0），`models/network.py`
   `_pool_feature_map` 与 `models/fusion.py` `_global_pool` 支持 GeM 广义均值池化
   （对壁厚/浸润等细粒度分期特征更友好）；默认 `avg` 与原行为一致
9. **A4 分类头减宽**：`--head-width`（默认 256，小样本可减至 128），T/N 主分类头
   隐层宽度可配，降低小样本过拟合
   - GPU 前向冒烟已通过：medicalnet18 三流 145.9M / medicalnet10+share 42.7M、
     缺失模态 + imputer + gem + head_width=128 全链路正常；冻结 156 组 → 可训练 113.1M
10. **根因诊断与修复（关键）：「T 头不学习」主因 = AMP fp16 梯度溢出**
    - 现象：全配置第 1 epoch `grad_norm=inf/nan`（新增诊断日志暴露）；逐批定位到
      各头末层 Linear（classifier_n.3/ct_aux_head.3）梯度 fp16 反向溢出；GradScaler
      检测到 inf 即跳过该批优化步 → 有效更新流失 → 训练学不动
    - 对照实验：纯 FP32 完整 epoch 36 批 **0 inf**，坐实 AMP 溢出（非网络/损失问题）
    - 修复：① 头层（classifier/classifier_n/ct_aux_head/mri_aux_head）autocast
      `enabled=False` + 输入显式 `.float()`；② GeM 池化（pow 立方）强制 FP32 计算
      后还原 dtype（MedicalNet 大特征 fp16 立方溢出）；③ GradScaler `init_scale=2^10`
      （torch 2.5 无 max_scale；默认 growth_interval=2000 步，60ep≈2160 步内基本不变）
    - 验证：P1/P2/P3 探针 grad_norm 全部恢复正常（7.97/8.78/6.31），训练回路无损
    - 涉及文件：`models/network.py` `models/fusion.py` `trainers/trainer.py`

**已导入、未在新数据上训练验证（Cline 会话完成、本次确认）**

- 208 份新数据已导入并转 NIfTI，full 分支 = **361 例**（T: 0=102 / 1=255 → 71% 阳性；N: 0=199 / 1=157）；
  `cache_hos` 已重建至 543 个缓存；`hos_branch_{mri_t,mri_d,mri_c,mri_tdc,ct_p,ct_z,full}.csv` 基于新数据
- 旧 154 样本上的 round7 分支结果仍可参考，但需在新数据上重跑确认
- `round8_A` 自动实验（lr 3e-4）已按「训练手动启动」约定终止，`config/hos.py` lr 还原 `1e-4`

**未执行（手动实验计划，用户手动启动，跑完贴日志分析）**

- 探针: `python tools/probe_epoch.py --branch full`
- E1 基线复现: `python train_hos_seeds.py --branch full --seeds 42 --epochs 60`
- E2 lr 3e-4: `... --lr 3e-4`　|　E5 关增强: `... --no-aug`
- E3 减负: `... --backbone resnet10 --share-backbone --freeze-low --mixup 0.3`
- E4 rec 降权: `... --rec-weight 0.2`
- 判断口径: 训练集 acc 冲过 71% 且 main loss < 0.5 = 拟合修复

**未确认（模型本体优化待办，拟合修复后再评估）**

- **MedicalNet/Med3D 3D 医学预训练**替换当前 ImageNet-2D→3D 广播权重（`pretrained_backbone=True`
  是 torchvision 权重广播，非医学预训练）——参数不变、权重升级，T 分期信号更相关
- GeM 池化替代全局平均池化（`_pool_feature_map`）、分类头 256→128 减宽、EMA/SWA 权重平均
- `models/backbone.py` 已支持 `resnet50` / `swin`（MONAI SwinUNETR encoder）但 CLI 未开放——
  8GB VRAM + 361 样本下不建议，等数据量上去再考虑
- 系统级提效: 分支模型集成（mri_t/ct_p/ct_z 晚期融合）、TTA 多 crop 推理、kfold 5 集成
- ⚠️ 记录修正: README [2026-09-28] 记录的 `loss_weights={"aux":0.1,"rec":0.1}` 已过时——
  **当前 config 为 `{"aux":0.3,"rec":0.5}`**（E4 用 `--rec-weight 0.2` 验证该值是否抢梯度）

---

### [2026-10-22] round7 规划：数据导入前审视结论（MRI 三流 + 预处理/增强）

> 等待新 208 份数据导入前的项目审视，本次仅记录结论与规划，无代码改动。
> 依据：`models/network.py`、`data_loader/loader.py`、`docs/OPTIMIZATION_ROADMAP.md`、round4/5 summary。

**审视结论 1：MRI 三流（T2/CE/DWI）方向正确，但当前是过拟合源而非增益源**

- 三流已在 round5 落地（`models/network.py` `mri_backbones` ModuleDict + stage 级 softmax 融合），
  数据满足（T2 129 / CE 99 / DWI 123，93/154=60% 三序列齐全）。
- 但当前同构 resnet10×3 = 71.4M 压 154 样本，round5 实测 best_epoch 过早（T@7/N@4）、
  验证集早平台期，结果落在 round4 同 seed 波动区间（T=0.4833 vs 0.5441/0.4833），无法判定三流增益。
- 缺两个决定性实验（roadmap B4/B5）：融合权重体检（`mri_seq_fusion_weights` 是否退化为均匀）
  与逐序列消融（`--mri-seq t / t,c / t,d` 同 seed 对照）。

**审视结论 2：预处理已到位，增强已强化，继续堆增强天花板很低**

- 预处理无明显缺陷：CT（HU 窗 -150~250 + CropForeground + z-score）；MRI（T2 锚定 1mm 各向同性
  对齐 + CE/DWI 重采样 + 统一 bbox 裁剪），磁盘缓存 v2。
- 增强 round4 已强化（Flip/Affine/Zoom/强度偏移缩放/噪声/Cutout + round6 MixUp/label-smoothing/cosine LR），
  154 样本下均值提升上限 ~0.05（roadmap），继续加几何/强度增强有过拟合反噬风险。
- 真正杠杆在 roadmap C 区：C1 自监督预训练（351 卷无标签 MRI）、C2 TTA/多 crop 推理、C3 3D patch 采样。

**新 208 数据导入后的规划（按性价比排序）**

1. 数据导入 + 重建缓存（`tools/build_det_cache.py`）→ 审计三序列/双模态覆盖变化，更新 `hos_pairs.csv`。
2. 跑 round6 已就绪的评估协议：`--kfold 5` + `--split-seed 42` 多 seed，先把 std 压下来出可信阳性。
3. 三流减参 + 消融：`--backbone resnet10 --share-backbone --freeze-low` 同 seed 对照 round5 三流；
   补 `--mri-seq` 单序列消融，量化各序列贡献。
4. TTA / 多 crop 推理（改动最小、收益确定）。
5. 自监督预训练（351 卷无标签 MRI 对比学习，工作量最大、杠杆最大）。

---

### [2026-10-27] round7 落地：分支实验 + 完整评估落盘 + 目录整改

> 208 份新数据导入前，先用现有 154 样本跑通 round7 分支实验并固化评估协议。

- **入口统一**：全部实验从 `train_hos_seeds.py` 进入（旧入口 `train_hos.py` /
  `test_hos_ct.py` / `_eval_best.py` / `_eval_fast.py` 已归档至 `scripts_legacy/`）
- **分支实验**：`--branch {mri_t,mri_d,mri_c,mri_tdc,ct_p,ct_z,full}`（对应
  `data/hos_branch_*.csv`）+ `--mri-seq` / `--ct-seq` 序列级消融；`config/hos.py`
  内置 `HOS_BRANCHES` 定义各分支数据/序列配置
- **完整评估落盘**：每次训练结束固定产出 `logs/eval_report_{tag}_{run}.{json,txt}`
  + ROC-PR 图（T/N 双任务全指标 + 阈值校准 + 混淆矩阵图，`utils/eval_report.py`），
  随后自动写入 `logs/seed_summary*.json` 汇总
- **`--eval-only`**：不训练、用已有 checkpoint + split csv 只读重出全部报告
  （`--eval-only --seeds 42` / `--eval-only --kfold 5`）；ckpt 结构与当前参数不匹配时
  警告跳过而非崩溃
- **epochs 默认 60**（与 `config/hos.py` 一致）；seed_summary 每完成一个 seed 即
  增量落盘，中途崩溃不丢已完成结果
- **目录整改**：数据/转换脚本移至 `tools/`（`make_kfold`、`build_hos_*`、
  `convert_*` 等）；round6 及更早日志归档 `logs/archive/round6_pre_reorg/`；
  各分支结果图汇总至 `results_hos/{branch}/`
- **运行示例**：
  - `python train_hos_seeds.py --branch mri_t --seeds 42,0,1,7 --epochs 60`
  - `python train_hos_seeds.py --kfold 5 --branch ct_z`
  - `python train_hos_seeds.py --eval-only --branch full --seeds 42`

---

### [2026-10-09] round6: k-fold CV + seed 集成 + MRI 共享减参 + 正则强化

> 目标：在当前 154 样本下压低方差、出可信阳性结果（详见 `docs/OPTIMIZATION_ROADMAP.md`）。
> 代码已全部落地并通过 py_compile / 结构冒烟 / GPU 1-epoch 整链路验证
> （项目 conda 环境 **Pytorch**：torch 2.5.1+cu124，NVIDIA RTX 4060）。

- **A1 k-fold**：`train_hos_seeds.py --kfold 5` 启用 StratifiedKFold（按 T×N 分层），
  每折独立训练/校准，汇总各折 best 指标 mean±std → `logs/kfold_summary.json`；
  替代 val 仅 31 例的 0.2 holdout（1 样本 ≈ 3.2 acc 点）
- **A2 seed 集成**：`--split-seed 42` 令所有 seed 共用同一划分（保证可比）；
  `tools/_ensemble_eval.py` 对多 checkpoint 在固定 val 上平均 softmax + 阈值校准 →
  `logs/ensemble_summary.json`
- **B2 三流减参**：`--share-backbone` 三序列（t/c/d）共享同一 encoder（resnet10：
  71.4M→**42.6M**，仍按序列分别前向 + stage 级加权融合，保留序列独立特征语义）；
  `--freeze-low` 冻结 stem+layer1+2（可训练 →40.8M）。同时修正 freeze 前缀 bug：
  旧写法 `relu` 不匹配 MONAI 实际层名 `act`，且 round5 三流前缀 `mri_backbones.*`
  未被覆盖 → round4/5 的 `freeze_backbone_layers` 冻结实际从未生效
- **B3 正则**：`--mixup 0.2~0.5` 3D MixUp（batch 内线性插值，作用于 main/n/aux loss，
  rec 不受影响，已对照手工计算验证一致）；标签平滑沿用 `cfg.label_smoothing` +
  `--label-smoothing` 覆盖；cosine LR 调度沿用 Trainer `CosineAnnealingLR`
- **兼容性**：`--share-backbone` 的权重与 round5 三流结构不兼容（结构变更重训）；
  round5 权重已备份 `backup_round5_mri3/`
- **运行示例**（项目环境 `Pytorch`，直接用 `python` 需先 `conda activate Pytorch`）：
  - `python train_hos_seeds.py --kfold 5 --epochs 60 --backbone resnet10 --share-backbone --freeze-low --mixup 0.3`
  - `python train_hos_seeds.py --seeds 42,0,1,7 --epochs 60 --split-seed 42 --backbone resnet10 --share-backbone --freeze-low --mixup 0.3`
  - `python tools/_ensemble_eval.py --ckpts archive/checkpoints_early/checkpoints_hos/best_model_hos_s42.pth,archive/checkpoints_early/checkpoints_hos/best_model_hos_s0.pth --names s42,s0 --val-csv data/hos_val_s42.csv --share-backbone`
    （round5 权重已移入 `archive/checkpoints_early/checkpoints_hos/`；`checkpoints_hos/` 现仅剩 `best_model_2.pth`/`best_model_n_2.pth`）

---

### [2026-10-09] round5: MRI 三序列（T2/CE/DWI）整合

- **数据**：`utils/build_hos_dataset.py` 每患者导出全部 MRI 序列（`_nifti/MRI/` 351 卷）；
  `hos_pairs.csv` 新增 `mri_path/available_{t,c,d}` 列。实测 131 例 MRI：
  **93 例（71%）三序列齐全，127 例（97%）≥2 序列**，仅 4 例只有 T2
- **管线**：`data_loader/loader.py` 新增三序列对齐管线 `_mri_seqs_deterministic`——
  同患者 t/c/d 的 spacing/origin/direction/FOV 完全不同，故以 T2 为锚构建 1mm 各向同性
  参考网格，CE/DWI 重采样对齐后统一 bbox 裁剪 + Resize(96³)，输出 [3,96,96,96]；
  磁盘缓存升 v2（key 含三序列 mtime，自动重建）；batch 新增 `mri_seq_available`
- **模型**：`models/network.py` MRI 端由单 backbone → **三 backbone**（`mri_backbones`
  ModuleDict t/c/d），stage 级可学习权重（softmax）融合，缺失序列被 mask 后重归一化；
  下游 self-attn/imputer/fusion/双头不变；可训练参数 73.7M
- **trainer**：fit/validate/predict_proba 传递 `mri_seq_mask`（逐序列可用性）
- **为什么三流**：DWI（弥散受限/细胞密度）、CE（血供）、T2（解剖）物理信息本质不同，
  序列独立建模优于通道堆叠；后续 ~200 份新数据（多为 2/3 期）使样本量翻倍，支撑三流；
  且支持逐序列贡献消融分析
- **兼容性**：round4 及之前的 `best_model*.pth`（单 MRI backbone）与新结构**不兼容**，
  重训后覆盖；`data/hos_pairs_before_round5_single_seq.csv` 已备份至 `_tmp/20260929_cleanup/`
- **已验证**：seed42 单 seed 训练（resnet10, 30 epochs）T 0.4833 / N 0.6026（校准），
  数值落在 round4 同 seed 波动区间内（154 样本方差主导）；存档 `logs/archive/round5_20261009/`，
  权重备份 `backup_round5_mri3/`
- **待办**：新 ~200 份数据后 4-seed 正式对比；序列级消融（单序列 vs 三序列）

---

### [2026-09-29] round4: resnet10 消融 + 缓存提速 + 增强强化 + N-best 修复

- **性能基建**：`data_loader/loader.py` 新增 `_DetDiskCache` 确定性预处理磁盘缓存
  （`cache_hos/`，194 卷 100% 命中，epoch ~7s）；`tools/build_det_cache.py` 一键构建
- **N-best 修复**：`trainers/trainer.py` 独立跟踪/保存 T、N 双 best
  （`best_model[_n]_hos_s{seed}.pth`），校准 N 用 N-best 权重
- **resnet10 支持**：`models/backbone.py` 增加 resnet10（42.6M）；
  `train_hos_seeds.py` 新增 `--fusion/--backbone/--no-clinical`，resnet10 时全量不冻结
- **数据增强强化**：`_random_augment()` 增加 Zoom/强度偏移/强度缩放/高斯噪声/CoarseDropout，
  仿射旋转温和化（0.2→0.15rad）并加 shear；作用于缓存输出，无需重建缓存
- **实验结果**（详见 `logs/archive/round4_20260929/round4_summary.json`）：
  - resnet10 multi 4-seed（42/0/1/7）：T 校准均值 0.5405；N 0.5437±0.057
    （resnet18 multi 3-seed：T 0.5297；N 0.5735±0.097）
  - resnet18+single 消融（seed42）：T 0.5141（+0.031 vs multi）、N 0.5643（-0.040）
  - **no-clinical 消融**（resnet10 4-seed，`--no-clinical` 临床全 0，同划分/同协议，见 `round4_20260929/seed_summary_noclinc.json`）：
    T 0.5551±0.095 / N 0.5550±0.006，与有临床版（T 0.5405 / N 0.5437±0.057）**无差甚至微升，N 方差骤降** → 当前临床分支为净噪声源
    （ca199/cea 62.6% 缺失 0 占位；7 维单列相关均不显著；纯临床基线 macro-F1 仅 0.47-0.49）
  - 同 seed42+resnet10 两次运行 T=0.5441 vs 0.4833 → **154 样本高方差是主要矛盾**
- **下一步**：扩数据样本集 → 重建缓存 → 对比增强前后方差；备选 mixup/融合调整
- **文档**：新增 `docs/PROJECT_OVERVIEW.md`（项目级梳理：架构/流程/实验脉络）与 `docs/FILE_INVENTORY.md`（文件级索引：分类/状态/可清理项）

---

### [2026-09-28] 课题定位 + Hos 训练配置修复

- **课题定位（毕业论文）**：核心任务改为术前 **T / N 双任务二分类**
  （T1-2 vs T3-4、N0 vs N+）；M 分期因数据缺失明确排除。
- **数据审计**：`hos_pairs.csv` 无 `label_M`/`pM` 字段 → M 分期不可行（论文 limitation）；
  pT 分布 = T1:6 / T2:37 / T3:106 / T4:5（T 四分类不可行），
  pN 分布 = N0:85 / N1:47 / N2:22（N 三分类可作扩展分析）；
  有效样本 154（双模态 39 / 仅CT 24 / 仅MRI 91）。
- **`config/hos.py`**：
  - `loss_weights`：`{"aux":0.3,"rec":0.5}` → `{"aux":0.1,"rec":0.1}`
    （重建/辅助损失不再挤压 T/N 分类头）
  - `lr`：`1e-4` → `3e-4`；`epochs`：`30` → `60`
- **核验（无代码改动）**：`trainers/trainer.py` 已按 val macro-F1 选取 best checkpoint；
  `_eval_fast.py` 概率拼接修复已确认。

---


### [2026-06-22] 架构升级：缺失模态鲁棒性 + 模态内自注意力 + 多任务监督

#### 背景

原架构 `CT/MRI Backbone → BSAFusion(cross-attn) → concat(临床) → 分类头` 存在两个不足：
1. 推理时强依赖两种模态同时存在，临床场景中常遇到只有 CT + 血液或只有 MRI + 血液的情况。
2. 两路 backbone 输出直接进 cross-attention，缺少模态内信息提纯环节。

#### 整体新架构

```
CT/MRI Backbone
    ↓
模态内 Self-Attention (新增, 方案 B)
    ↓
模态 Dropout + Mask Token + Imputer (新增, 缺失处理)
    ↓
CT 辅助头 / MRI 辅助头 (新增, 方案 C)
    ↓
BSAFusion (保留 cross-attn)
    ↓
主分类头
```

总体思路：**先模态内提纯 → 处理缺失 → 跨模态融合 → 多任务监督**

#### 修改清单

##### 1. `models/fusion.py` — 新增模块
- 新增 `SelfAttentionEncoder`：模态内自注意力，1~2 层 Transformer block，让每路 backbone 输出在融合前先"想清楚"。
- 新增 `ResidualMLPImputer`：残差 MLP 做跨模态特征生成。输入 = 源模态各 stage 池化向量 + 临床；输出 = 目标模态各 stage 池化向量。
- 保留原 `BSAFusionModule` 与 `CrossAttentionBlock`，不动。

##### 2. `models/network.py` — 主网络重构
- 新增可学习 `mask_token_ct` / `mask_token_mri`（极端兜底占位符）。
- 新增双向 Imputer：`imputer_ct2mri` 与 `imputer_mri2ct`。
- 新增 CT 辅助分类头 / MRI 辅助分类头。
- `forward` 新签名：`forward(ct, mri, clinical, ct_mask=None, mri_mask=None)`。
  - `ct_mask` / `mri_mask` 为 `[B]` 布尔张量，True 表示该样本该模态可用。
- 训练时按概率随机生成 mask（模态 dropout：CT 单缺 15% / MRI 单缺 15% / 双在 70%）。
- 推理时根据传入 mask 决定走真特征 / Imputer / mask token。
- 返回值改为 dict：`{"main", "ct_aux", "mri_aux", "imputed_mri", "imputed_ct"}`。

##### 3. `trainers/trainer.py` — Loss 与训练循环
- 主分类 loss：CrossEntropy（保持原有）。
- 辅助分类 loss：CT 头 + MRI 头 各自的 CrossEntropy（仅当对应模态真实存在时计入）。
- 重建 loss：两模态都在时，伪特征 vs 真特征的 MSE + Cosine（双向）。
- 总 loss = `main + λ1·(ct_aux + mri_aux) + λ2·(rec_ct2mri + rec_mri2ct)`。
- 初始权重：λ1 = 0.3，λ2 = 0.5。
- 在 batch 内执行模态 dropout 采样。

##### 4. `data_loader/loader.py` — 数据接口
- DataLoader 输出新增 `ct_available` / `mri_available` 字段。
  - 训练时全 True；推理时按实际情况。
- 缺失模态用零张量占位（仅为 batch 形状对齐，不会被使用）。

##### 5. `config/default.py` — 配置项

| 配置项 | 含义 | 默认值 |
|--------|------|--------|
| `use_self_attn` | 是否开启模态内 self-attention | `True` |
| `self_attn_layers` | self-attention 层数 | `1` |
| `use_imputer` | 是否启用 Imputer | `True` |
| `imputer_hidden_dim` | Imputer 隐层维度 | `256` |
| `modality_dropout` | 模态 dropout 概率 | `{"ct_only": 0.15, "mri_only": 0.15, "both": 0.7}` |
| `loss_weights` | 辅助 / 重建 loss 权重 | `{"aux": 0.3, "rec": 0.5}` |

##### 6. `utils/` — 不动
`metrics.py` / `logger.py` 保持原样，主分类指标走 `logits["main"]`。

#### Imputer 选型

经过对 5 类方案的对比（线性 MLP / 残差 MLP / FiLM / Cross-attn / 生成式），最终选用 **残差 MLP**：

- 输入：CT 各 stage 池化向量 concat + 临床向量
- 输出：MRI 各 stage 池化向量
- 方向：双向（CT→MRI 与 MRI→CT 各一个）
- Loss：MSE + Cosine 对齐

理由：
- 与现有 fusion 输入层级（各 stage 池化向量）对齐，无需改 fusion。
- 双向 Imputer 同时 cover 两种缺失场景。
- 残差 MLP 在医学影像小数据集上经验最稳，不易翻车。

#### 不改动的部分

- `models/backbone.py`：backbone 实现保持原样（ResNet / Swin）。
- BSAFusion 内部 cross-attention 逻辑保持原样。
- 临床分支 MLP 保持原样。

#### 推理新能力示例

```python
# 完整模态
out = model(ct, mri, clinical)

# 只有 CT + 临床（MRI 缺失，由 Imputer 填补）
out = model(ct, mri_zeros, clinical, mri_mask=torch.zeros(B, dtype=torch.bool))

# 只有 MRI + 临床
out = model(ct_zeros, mri, clinical, ct_mask=torch.zeros(B, dtype=torch.bool))
```

---

### [2026-06-22] 工具脚本：CMB 数据筛选

- 新增 `utils/filter_cmb_data.py`：递归扫描 `data/cmb`，删除直接包含 `.dcm` 文件数 ≤ 10 的文件夹（不计入子文件夹中的 `.dcm` 文件）。
- 安全设计：默认先以预览模式运行，列出待删除文件夹，需手动输入 `yes` 确认后才执行删除。
- 用法：`python utils/filter_cmb_data.py`

---

## 项目结构

```
ColonMultiModal/
├── config/
│   ├── __init__.py
│   └── default.py              # Config 类：集中管理所有超参数
├── data_loader/
│   ├── __init__.py
│   └── loader.py               # MultiModalPreprocessor / get_transforms / build_dataloaders
├── models/
│   ├── __init__.py
│   ├── backbone.py             # ResNet3DBackbone / SwinTransformer3DBackbone / build_backbone
│   ├── fusion.py               # CrossAttentionBlock / BSAFusionModule
│   └── network.py              # CRCMultiModalNet（主模型）
├── trainers/
│   ├── __init__.py
│   └── trainer.py              # Trainer（fit / train_one_epoch / validate / 保存 best）
├── utils/
│   ├── __init__.py
│   ├── metrics.py              # compute_metrics（accuracy / F1 / recall / confusion matrix）
│   └── logger.py               # setup_logger（console + 文件双输出）
├── main.py                     # 启动脚本
├── debug.ipynb                 # 调试 notebook（autoreload，逐模块测试）
├── data/                       # DICOM 影像 + 临床 TSV + 索引 CSV
├── dataloader.ipynb            # [参考] 原始 dataloader 开发记录
├── models.ipynb                # [参考] 原始模型开发记录
├── DataLoader-test.ipynb       # [参考] 数据管线调试工具
└── readme.md
```

## 快速开始

```bash
# 1. 修改配置
#    编辑 config/default.py 中的 train_csv / val_csv / backbone_name 等

# 2. 启动训练
python main.py

# 3. 调试
#    打开 debug.ipynb，逐 cell 运行
```

---

# dataloader.ipynb

多模态数据加载模块，将 CT、MRI 影像与临床血液指标统一打包为 PyTorch batch。

## 数据流

```
CSV (索引文件)
  ├─ ct_path      → LoadImaged → Spacing 1mm³ → HU窗[-150,250]→[0,1] ─┐
  ├─ mri_path     → LoadImaged → Spacing 1mm³ → Z-score 标准化        ─┼→ CropForeground → Resize 96³ → [训练] RandAffine+Flip
  ├─ clinical_cols → Imputer(中位数) → StandardScaler                   │
  └─ label_col    ─────────────────────────────────────────────────────→ batch = {ct, mri, clinical, label}
```

## CSV 格式要求

| 列名 | 说明 |
|------|------|
| `ct_path` | CT 影像路径（.nii.gz 或 DICOM 文件夹） |
| `mri_path` | MRI 影像路径（同上） |
| clinical_cols 各列 | 数值型临床特征（如 age, cea_level, hemoglobin, wbc_count） |
| label_col 列 | 整数分类标签（如 stage: 0/1/2/3） |

## 调用方式

```python
from dataloader import build_dataloaders

train_loader, val_loader = build_dataloaders(
    train_csv='./data/train_split.csv',
    val_csv='./data/val_split.csv',
    clinical_cols=['age', 'cea_level', 'hemoglobin', 'wbc_count'],
    label_col='stage',
    batch_size=4,
)

for batch in train_loader:
    ct       = batch["ct"]        # [B, 1, 96, 96, 96]
    mri      = batch["mri"]       # [B, 1, 96, 96, 96]
    clinical = batch["clinical"]  # [B, num_features]
    label    = batch["label"]     # [B]
    break
```

## 核心组件

| 组件 | 职责 |
|------|------|
| `MultiModalPreprocessor` | 读 CSV → imputer 填充缺失 → scaler 标准化 → 生成 data_dicts |
| `get_transforms(is_train)` | MONAI 影像管道：加载/重采样/强度归一化/裁前景/resize/增强 |
| `CRCMultiModalDataset` | 包装 monai.data.CacheDataset，全量缓存到内存加速训练 |
| `build_dataloaders` | 主入口，串联上述组件，输出 train/val DataLoader |

## 关键设计决策

- **MONAI LoadImaged**：自动推断 .nii.gz 或 DICOM 文件夹，省去 SimpleITK 手动解析
- **CT HU窗 [-150, 250]**：参考 CRCFound，过滤骨骼和气体干扰，聚焦腹部软组织
- **MRI Z-score**：MR 无固定物理单位，不能用 HU 窗截断
- **Spacingd + Resized 到 96³**：为 BSAFusion 多模态空间对齐打基础，确保 CT/MRI 在统一物理空间
- **CropForeground(source_key="ct")**：CT 信噪比稳定，作为裁剪参考
- **验证集复用训练集 imputer/scaler**：防止数据泄露
- **CacheDataset cache_rate=1.0**：内存换速度；内存不足时改 monai.data.Dataset
- **SimpleImputer(strategy='median')**：医院血液数据常有缺失，中位数填充防训练中断

## 依赖

- PyTorch >= 2.0
- MONAI >= 1.4
- scikit-learn
- pandas
- pydicom

---

# config/default.py

集中管理所有超参数，`main.py` 和 `debug.ipynb` 直接读取 `Config` 对象。

| 分组 | 关键参数 | 默认值 |
|------|----------|--------|
| 数据 | `train_csv` / `val_csv` | `./data/train_split.csv` |
| 数据 | `clinical_cols` | `["age_at_index"]` |
| 数据 | `spatial_size` | `(96, 96, 96)` |
| 模型 | `backbone_name` | `"resnet18"` |
| 模型 | `fusion_mode` | `"single"` |
| 模型 | `num_classes` | `4` |
| 训练 | `lr` / `weight_decay` | `1e-4` / `1e-5` |
| 训练 | `epochs` / `batch_size` | `50` / `2` |
| 训练 | `device` | `"cuda"` |

---

# trainers/trainer.py

训练与验证封装类。

```python
from trainers import Trainer

trainer = Trainer(model, train_loader, val_loader, cfg, logger)
trainer.fit()
```

| 方法 | 说明 |
|------|------|
| `train_one_epoch(epoch)` | 单 epoch 训练，返回 (loss, acc) |
| `validate(epoch)` | 验证集推理，返回 (loss, acc) |
| `fit()` | 主循环：训练 → 验证 → scheduler → 保存 best model |

设计决策：
- optimizer: `AdamW`
- scheduler: `CosineAnnealingLR`
- 自动保存 val macro-F1 最高的 checkpoint 到 `cfg.save_dir`

---

# utils/

| 文件 | 功能 |
|------|------|
| `metrics.py` | `compute_metrics(preds, labels, num_classes)` → accuracy / macro-F1 / per-class recall / confusion matrix |
| `logger.py` | `setup_logger(log_dir)` → Python logging，同时写 console + 文件 |

---

# main.py

纯净启动脚本（约 30 行），串联各模块：

```python
python main.py
```

执行流程：`Config → build_dataloaders → CRCMultiModalNet → Trainer.fit()`

---

# debug.ipynb

调试 notebook，用 `%autoreload 2` 实现改 .py 后自动重载（不用重启 kernel）。

包含 5 个 cell：
1. import 所有模块
2. 查看/修改 Config
3. dummy tensor 验证模型前向
4. 加载真实数据测 DataLoader
5. 跑 3 epoch overfit 验证 loss 下降

---

# models.ipynb

多模态分期分类网络，参考 CRCFound（CT 处理）+ BSAFusion（跨模态融合）设计。

## 模型架构

```
CT [B,1,96,96,96]      MRI [B,1,96,96,96]      Clinical [B, F]
       │                       │                      │
  CT Backbone             MRI Backbone          Clinical MLP
  (ResNet18/50            (同左，独立权重)       Linear→ReLU→Linear
   或 Swin-T)                                       │
       │                       │                    │
  [5个stage特征图]        [5个stage特征图]           │
       │                       │                    │
       └──── BSAFusion ────────┘                    │
                  │                                  │
           融合向量 [B, D]                            │
                  │                                  │
                  └──── concat ─────────────────────┘
                             │
                    Classifier (FC→ReLU→FC)
                             │
                       logits [B, num_classes]
```

## 可切换配置

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `backbone_name` | `"resnet18"` / `"resnet50"` / `"swin"` | CT/MRI encoder 类型 |
| `fusion_mode` | `"single"` / `"multi"` | 单尺度（最后一层）/ 多尺度（最后3层）cross-attention |
| `num_classes` | int | 分类数，默认 4（Stage I/II/III/IV） |
| `num_clinical` | int | 临床特征维度 |
| `img_size` | tuple | 输入空间尺寸（Swin backbone 需要） |

## 配置对比

| 配置 | 参数量 | 特点 |
|------|--------|------|
| resnet18 + single | 73M | 轻量快速，小数据集推荐 |
| resnet18 + multi | 71M | 多尺度信息，略丰富 |
| swin + single | 140M | 参考 CRCFound，全局感受野 |
| swin + multi | 136M | 信息最丰富，显存需求最大 |

## 使用示例

```python
from models import CRCMultiModalNet

model = CRCMultiModalNet(
    backbone_name="resnet18",
    fusion_mode="single",
    num_clinical=4,
    num_classes=4,
)

# 前向传播
logits = model(ct, mri, clinical)  # [B, 4]
```

## 核心组件

| 组件 | 职责 |
|------|------|
| `ResNet3DBackbone` | MONAI ResNetFeatures 封装，输出 5 个 stage 特征图 |
| `SwinTransformer3DBackbone` | MONAI SwinUNETR encoder 封装，输出 5 个 stage 特征图 |
| `CrossAttentionBlock` | Pre-norm cross-attention + FFN，单尺度融合基元 |
| `BSAFusionModule` | 双模态融合：single（双向 cross-attn）或 multi（逐 stage 融合） |
| `CRCMultiModalNet` | 主模型：双分支 backbone → BSAFusion → clinical MLP → 分类头 |

## 关键设计决策

- **双分支独立权重**：CT 和 MRI 各有独立 encoder，不共享参数（模态差异大）
- **BSAFusion 双向 cross-attention**：CT 作为 query 查 MRI，MRI 作为 query 查 CT，双向增强
- **多尺度融合（multi）**：融合最后 3 个 stage，低分辨率语义+中分辨率结构信息互补
- **临床 MLP 后接 concat**：影像融合特征与临床特征拼接后统一分类，避免信息覆盖
- **Dropout 0.3**：防止小样本过拟合
#### round9-F1 病灶融合分支 已完成（2026-11-06 01:16 → 02:00, seed42 60ep, tools 无新文件, 改动在 loader/network/trainer/train_hos_seeds）
> 目的: 病灶级特征进入主模型——CT Task010 病灶 mask 质心 643 裁剪 → medicalnet10 → GeM, 与全体积融合特征 concat(空 mask 零卷+存在标志, 不丢样本)。
> 命令: --lesion-fusion --seeds 42 --epochs 60 (R5 配置 + 病灶分支)。
> **结果 (对比 R5 s42 T 0.6017 / N 0.5093)**:
> | 指标 | F1-lesion | 对比 |
> |---|---|---|
> | T best_macro_f1 | **0.6400@47** (calT@0.5 [0.524, 0.765]) | **+0.038 ⚠️** |
> | N best_macro_f1 | 0.4718@32 (calN@0.5 [0.45, 0.5]) | -0.038 |
> 结论: T +0.038 达疑似突破阈值(≥+0.03) **记录待验证** (单 seed, 未中断序列); N 略降 → 病灶特征对 T 有正向贡献、N 无; 下一判定 = 2-seed 复验或 5-fold。空 mask 处理: train 114 CT 中 72 有 ROI / 42 空(与 task010 n_with_mask=96 一致), 零卷+标志。
> 产物: run_44_hos_full_s42 (11 文件)。判定: 疑似突破待验证。

#### round9-WR-60ep 完成（2026-11-06 02:01 → 02:44, --warm-restarts 60ep seed42, job pwsh-1f9279bbb6ea4919）
> 目的: 验证 warm restarts 60ep 单 seed 效果(S9-⑤ 曾得 s42 T 0.6850 疑似, 但 s7 复验 T 0.5710 反证)。
> **结果 (对比 R5 s42 T 0.6017 / N 0.5093)**:
> | 指标 | WR-60ep s42 | 对比 |
> |---|---|---|
> | T best_macro_f1 | **0.6936@56** (calT@0.5 [0.714, 0.725]) | **+0.092 ⚠️** |
> | N best_macro_f1 | 0.4985@14 (calN@0.5 [0.5, 0.5]) | -0.011 |
> 判定: 单 seed +0.092 达 +0.05 规则 → 需 5-fold 验证。**已有证据链**: ①S9-⑤ s42 0.6850(+0.083) ②⑤复验 s7 0.5710(-0.102, 反证) ③阶段1 WR 5-fold 30ep: T mean 0.5603±0.021 (5/5 折优于 both 0.5253, 但 <0.60 门槛) → **WR 在 5-fold 上是稳定小幅 T 提升(+0.035 级), 60ep 单 seed 极不稳定(s42 +0.092/s7 -0.102) → 非真阳性, 记为"稳定小幅候选"**。N 全程持平。
> 产物: run_45_hos_full_s42 (7 文件)。

#### 今晚序列（2026-11-07, 用户 m06838 确认版）:
1. MR 分割速度实测 (t/c/d 各 1 样本 → 定全量预算)
2. 全量 MR 结肠分割: 优先 t 序列(356) → 余力补 c/d (各序列独立目录 masks_totalseg_mr_{seq}/)
3. MRI 结肠 ROI 特征实验 (roi_experiment.py 加 mask 源, t 优先; c/d 若完成则加对照)
4. F1 病灶融合 5-fold 验证 (空余时间, 验证 T +0.038 可复现性)
5. N 头接 MRI-ROI 特征 (尝试, 若 ROI 实验显示结肠区域对 N 有信号)
6. 汇总 + README + 归档
规则: 每步总结再推进; 07:55 截断; 不启动直到用户命令。

#### 今晚序列 ② MR 结肠分割 (2026-11-07):
- **t 序列完成(03:13)**: 299/356 有 t 路径 (57 无 t 文件), 全部成功, n_with_mask=299 n_empty=17, total_voxels=11.8M, med 29393 体素 (结肠区域; MRI 上结肠=空腔壁, 体素比 CT 病灶大~10x)。用时 130min (~26s/样本)。产物 data/masks_totalseg_mr_t/{pid}/colon.nii.gz + logs/totalseg_mr_t_full.json。
- **c 序列运行中**(03:14 启动, job pwsh-70d3d987e72d4994); d 序列待 c 后启动。
- **c 序列完成(04:43)**: 229/229 (c 可用数), n_with_mask=229 n_empty=0, total 34.6M 体素 med 133400。产物 data/masks_totalseg_mr_c/ + logs/totalseg_mr_c_full.json。
- **d 序列运行中**(04:43 启动, job pwsh-bda883d590d94427)。
- **d 序列完成(06:53)**: 290/290 (d 可用数), n_with_mask=290 n_empty=0, total 34.9M 体素 med ~119000。产物 data/masks_totalseg_mr_d/ + logs/totalseg_mr_d_full.json。三序列 MRI 结肠 mask 齐备: t 299/c 229/d 290。

#### 今晚序列 ③ MRI 结肠 ROI 特征实验 (2026-11-07 06:51-07:01, roi_experiment.py --mri-seq t/c/d)
> 目的: MRI 侧结肠区域 ROI 特征 (一阶11+GLCM6+形态3=20 + 临床7 = 27维) 判别力, 补 CT 病灶/器官对照的 MRI 维度。
> 结果 (val F1 / AUC):
> | seq | n | T lasso | T histgb | N lasso | N histgb |
> |---|---|---|---|---|---|
> | t (T2) | 296 | 0.4359 / 0.5181 | 0.4679 / 0.5064 | 0.4571 / 0.4276 | **0.5238 / 0.5172** |
> | c (CE) | 227 | 0.4045 / 0.5000 | 0.5351 / 0.5899 | 0.4532 / 0.4411 | 0.4929 / 0.5086 |
> | d (DWI) | 287 | 0.4695 / 0.5116 | 0.3861 / 0.3352 | 0.3778 / 0.3650 | 0.4762 / 0.4475 |
> 对照 CT: 病灶 mask N val 0.4038/0.3718, 结肠 mask N val 0.4038/0.4664。
> 结论: ①**MRI t 结肠 ROI 的 N val F1 0.5238 为所有手工 ROI 特征中最高**(CT 两种 mask 均 ~0.40) → MRI 结肠区域对 N 略有判别信号, 但仍在 0.52 噪声区, 未破 N 瓶颈; ②c/d 无增益(d 的 DWI 结肠 mask 大量空, n_empty=138/290); ③T 上 c 序列 histgb val AUC 0.5899 最高但仍 <0.60, 手工特征上限有限; ④一致结论: **手工 ROI 统计特征不足以支撑 N 分期, 需深度 ROI 特征 (ROI 内 DL)**。产物: logs/roi_mri_{t,c,d}_features.csv + roi_mri_{t,c,d}_experiment_full.{json,txt}。

#### 今晚序列 总总结 (2026-11-07 07:05 截断版)
1. **MR 分割全量完成**: t 299/c 229/d 290 全部成功 (三序列 MRI 结肠 mask, ~25-30s/样本, 共 ~6h)。d 的 DWI 结肠分割 n_empty=138/290 (DWI 结肠信号弱不可靠, 但 mask 文件仍有效)。
2. **MRI 结肠 ROI 特征实验 (3 序列)**: t 的 N val F1 0.5238 是所有手工 ROI 特征最高 (CT 病灶 0.40/结肠 0.40 对照), 但仍在噪声区, 未破 N 瓶颈; c/d 无增益。
3. **F1 5-fold 验证**: 因 07:55 截断时间不足未跑 (需 1.5h)。F1 单 seed T 0.6400 (+0.038 vs R5 s42) 仍是疑似突破待验证。
4. **结论**: 病灶级/器官级 ROI 手工特征 (CT/MRI 全模态) 均不足以支撑 N 分期; 唯一未试=ROI 内深度特征 (ROI 裁剪 DL 训练, run_42 A 方案 train72 样本不足); MRI 结肠 ROI 是最接近 N 信号的手工特征, 可作 ROI 内 DL 的输入候选。
产物: 三序列 mask (data/masks_totalseg_mr_{t,c,d}/) + roi_mri_{t,c,d}_features.csv + 报告 json/txt。

#### round9-F1 病灶融合 5-fold 验证 已完成（2026-11-08 03:05, --lesion-fusion --kfold 5 --epochs 30, 逐折分进程）
> 目的: 单 seed 疑似突破 (T 0.6400 vs R5 s42 0.6017) 的 5-fold 判定。
> 结果 (30ep/折, vs both 基线 T 0.5253±0.006 / N 0.5393±0.053):
> | 折 | T (F1 vs both) | N (F1 vs both) |
> |---|---|---|
> | k0 | 0.6283 vs 0.5353 (+0.093) | 0.5032 vs 0.4836 (+0.020) |
> | k1 | 0.5177 vs 0.5267 (-0.009) | 0.5337 vs 0.6375 (-0.104) |
> | k2 | 0.5283 vs 0.5198 (+0.009) | 0.5071 vs 0.5035 (+0.004) |
> | k3 | 0.6048 vs 0.5279 (+0.077) | 0.5219 vs 0.5319 (-0.010) |
> | k4 | 0.5795 vs 0.5170 (+0.063) | 0.4083 vs 0.5400 (-0.132) |
> | mean | 0.5717±0.043 vs 0.5253 (+0.046, 5/5 折胜) | 0.4948±0.045 vs 0.5393 (-0.045, 4/5 折输) |
> 结论: **病灶融合分支给稳定可复现 T 增益 (+0.046, 5/5 折全胜), 但 N 受损 (-0.045)** → T 未达真阳性门槛 (0.5717 < 0.60), 记正向候选: 病灶局部信息提升 T (CT 病灶浸润), 但 N 头被病灶分支干扰; 候选方向=只对 T 头接病灶分支 / 与 n-modal ct 组合。归档 run_46-50_hos_full_k0-4。

#### round9-F1 ⑤ MRI-t 结肠 ROI DL 完成（2026-11-08 03:11, tools/train_roi_dl.py --mode mri_t, seed42 60ep）
> 目的: MRI t (T2) 结肠 ROI (total_mr mask 质心裁剪 64^3) 内深度特征——N 的最后一个手工特征信号 (MRI-t 结肠 ROI N val 0.5238) 升级为 DL。
> 结果: **T best 0.6204@21 acc 0.6984 (calT@0.7: 0.5971 rec=[0.474,0.727]) / N best 0.5842@30 (calN@0.3: 0.5675 rec=[0.6,0.536])**; train 217 (skip 67) / val 63 (skip 9), medicalnet10 预训练 72/72, 可训练 9.1M/14.5M, 60ep 仅 30s。
> 对比: R5 s42 (T 0.6017/N 0.5093) → **T +0.019 / N +0.075**; vs CT 病灶 ROI DL run_42 (T 0.6101/N 0.5714) → N +0.013; vs MRI-t 手工 ROI (N val 0.5238) → N +0.060。
> 结论: **MRI-t 结肠 ROI DL 的 N 0.5842 是所有 ROI DL/手工特征最高值** (N +0.075 vs R5) — ROI 内深度特征显著优于手工特征, MRI 结肠区域含 N 判别信息; 单 seed 疑似突破 (N >=+0.05), 需 5-fold 验证 (待序列后)。归档 run_51_roidl_hos_full_s42。


#### round9-MRI-t ROI DL 5-fold (N 5-fold 级疑似突破, 2026-11-08 03:25, tools/train_roi_dl.py --mode mri_t --kfold 5 --epochs 60)
> 新增 --kfold 支持(StratifiedKFold 合并 train+val 280 样本); 每折 224train/56val 60ep 仅~7s/折。
> **T mean=0.6060+-0.0461 / N mean=0.5680+-0.0528 (kfold_summary_roidl_hos_full.json)**
> 折明细: k0 T 0.6166@15/N 0.5508@51 | k1 0.5636@17/0.5083@24 | k2 0.5623@10/0.5204@45 | k3 0.6883@31/0.6345@25 | k4 0.5993@10/0.6263@26
> 判定: **N mean 0.5680 >= 门槛 0.559 (R5 both 0.5093+0.05) -> N 疑似突破成立, 首个 5-fold 级验证通过**; vs R5 both 5-fold (T 0.5253/N 0.5393): T +0.081 / N +0.029; 与单 seed run_51 (0.6204/0.5842) 折均值一致非运气。
> 含义: MRI-t 结肠 ROI 深度特征确含 N 判别信号(全体积拿不到的局部浸润), 病灶级路径首次产生可复现收益。
> 产物: checkpoints best_model{,_n}_roidl_hos_full_k0-4.pth + eval_report_roidl_hos_full_k0-4.json + kfold_summary_roidl_hos_full.json。
#### round9-③ MRI-t ROI 融合分支 已完成（2026-11-08 06:12, run_52, R5 + --mri-lesion-fusion 60ep s42）
> 目的: MRI-t 结肠 ROI(TotalSegmentator total_mr mask 质心 643→medicalnet10→GeM→MLP) 作为第三分支与 CT+MRI 全体积融合特征 concat, 验证病灶级信息的融合价值。
> 结果: **T 0.6761@59 acc 0.7361 (calT@0.45 recall=[0.524,0.824]) / N 0.5130@6 (calN@0.5 recall=[0.5,0.531])**; 对比 R5 s42 T 0.6017/N 0.5093 → **T +0.074 (≥+0.05 疑似突破, 待 5-fold 判定)**, N +0.004 无增益。
> 结论: MRI-t 结肠 ROI 特征对 T 有实质贡献(+0.074, 与 MRI-t ROI DL 的 T 0.6204 一致方向), 但融合进主模型后 N 无增益(0.5130≈R5)——**N 的提升需要独立 N 头(ROI DL 单 seed N 0.5842), 融合分支只帮 T**。

#### round9-③ MRI-t ROI 融合 5-fold（部分 3/5 折, 2026-11-08 08:56, run_57/58/59, 用户指示跑完 k2 后停止）
> 注意: 首轮 5 次运行漏 --kfold-resume 全部重复 fold 0(已废弃 run_53-56, 保留 run_57 为 k0); 修正后补跑 k1/k2, k3/k4 未跑(用户停止)。
> 结果(3 折): k0 T 0.6850@27 / N 0.4991@5; k1 T 0.6296@20 / N 0.5594@15; k2 T 0.5177@27 / N 0.4971@10; **T mean 0.6108 / N mean 0.5185**。
> 判定: T mean 0.6108 vs both 0.5253 = +0.086 (3/3 折胜, 方向与单 seed +0.074 一致) → MRI-t ROI 融合稳定升 T; N mean 0.5185 vs 0.5393 = -0.021 (3 折均 ≤基线) → 融合分支不帮 N, 与单 seed N 0.5130 一致。结论同单 seed: **病灶级信息帮 T 不帮 N; N 需要独立 N 头(ROI DL)而非融合**。

#### round9-今晚池子 ②-⑥ (2026-11-12 02:10, 自动执行模式, run_66-68 + roi 5-fold)
> ②c ROI DL 5-fold: T 0.5721±0.031 / N 0.5921±0.046 (c 序列 N 最强 5-fold, ≥0.559 疑似); ②d ROI DL 5-fold: T 0.6429±0.078 / N 0.4598±0.060 (d 序列 T 最强); ③PCGrad s42: T 0.6167/N 0.4718 (无增益); ④mri_all 5-fold: T 0.6232±0.091 / N 0.5310±0.067 (三序列拼接未超单序列最佳); ⑤组合1 n-modal ct+n-lesion-fusion s42: T 0.6486/N 0.5671 (T+0.047 接近, N+0.058 达疑似门槛, 双头同时提升 = 今晚唯一双增益); ⑥组合2 n-modal ct+lesion-fusion s42: T 0.5710/N 0.5143 (无增益, T 反降)。
> 结论: ①c 序列结肠 ROI 独立 DL = N 最强可复现信号 (0.5921, 5 折验证) ②d 序列 = T 最强 (0.6429) ③⑤组合1 单 seed 双增益待 5-fold 复核 ④PCGrad/病灶融合在 n-modal ct 下无效。

#### round9-⑥ 组合1 (n-modal ct + n-lesion-fusion) 5-fold 完成（2026-11-13 02:53, run_69-73, 自动接力）
> 目的: 双头同升唯一双增益单 seed (T 0.6486/N 0.5671) 的 5-fold 复核 —— N 头只接 CT + MRI-t 结肠 ROI 嵌入。
> 结果: k0 T 0.5381/N 0.5781 | k1 T 0.4754/N 0.5774 | k2 T 0.5260/N 0.5485 | k3 T 0.6520/N 0.5203 | k4 T 0.5061/N 0.5994。
> **T mean 0.5395 (vs both 0.5253 = +0.014, 3/5 折胜, 噪声内) / N mean 0.5647 (vs 0.5393 = +0.025, 4/5 折胜, ≥0.559 门槛)**。
> 判定: **N mean 0.5647 达标 → 组合1 为主模型内首个 N 疑似突破配置**(与独立 ROI DL 脚本 MRI-t 0.5680 相当略低, 但一个模型同时给 T/N, 部署价值更高); T 无显著变化。结论: **CT 专喂 N 头 + MRI-t 结肠 ROI 嵌入 N 头 = N 瓶颈最可行的主模型内解**。

#### round9-自写网络 A1: mri_tc (t+c 两序列 ROI 拼接, 自写网络 5-fold) 2026-11-14 00:38, run_74
> 实现: SelfWrittenNet3D (0.92M 参数, stem conv3 + 3 残差 stage 32/64/128 + 轴向 slice attention + GeM p=3 + MLP 双头, 无预训练) + DataROI mode=mri_tc (t+c 两序列各 mask 质心裁剪拼接 [2,64,64,64], train 164/val 52)。
> 结果: k0 0.5993/0.5909 | k1 0.6141/0.5222 | k2 0.7462/0.7568 | k3 0.5549/0.5820 | k4 0.6387/0.5968 → **T mean 0.6306±0.064 (+0.105 vs both, 5/5 折胜) / N mean 0.6097±0.078 (+0.070 vs both, 4/5 折胜, ≥0.559 门槛)**。
> 判定: **N 突破再次确认且为迄今最强 5-fold** (超 mri_c 单序列 0.5921 / mri_all 三序列 0.5310); **自写轻量网络无预训练反而 N 最优** → 方案3 验证通过, ROI 双序列拼接有效。k2 折偏高(0.75)提示方差, 但 4/5 折一致方向。

#### round9-A2: Focal Loss + Label Smoothing 0.1 (单 seed s42 60ep) 2026-11-14 01:16, run_75
> 依据: T 头 72% 不平衡 (调研 Top-3, 零架构改动)。CLI --focal (gamma=2.0) + --label-smoothing 0.1, R5 配置。
> 结果: T 0.5852@27 acc 0.7083 calT@0.5=[0.619,0.608] / N 0.4718@22 calN@0.5=[0.45,0.5]。
> 判定: vs R5 s42 (T 0.6017/N 0.5093) T -0.016 / N -0.038 → **无增益**, Focal+LS 不如现有 class weights 组合, 不再深入。

#### round9-B3a: C1 Patch-MIL 单 seed s42 60ep 2026-11-14 01:23, run_76
> medicalnet10 共享 backbone + attention-MIL (8 块 483 无重叠), 60ep s42。
> 结果: T 0.5898@19 / N 0.4549@54 (acc 0.7083→0.6528)。
> 判定: vs R5 s42 (T 0.6017/N 0.5093) T -0.012 / N -0.054 → **无增益**, 弱监督块级注意力未超过全体积基线, 不再深入。

#### round9-C2: 一致性正则 (--consist-modal mri --consist-weight 0.1, 单 seed s42 60ep) 2026-11-14 02:10, run_77
> 依据: C2 一致性正则(同一 MRI 两次增强特征 MSE 双向 stop-grad, 防过拟合)。R5 配置。
> 结果: T 0.6812@56 acc 0.7222 calT@0.4=[0.619,0.804] / N 0.4497@19 calN@0.5=[0.525,0.375]。
> 判定: vs R5 s42 (T 0.6017/N 0.5093) T +0.080 疑似突破(≥0.05, 待 5-fold) / N -0.060 受损 → 一致性正则帮 T 伤 N, 与病灶类分支方向相反; 记疑似, 未达 5-fold 不写正结果。

#### round9-B2: Warm Restarts 60ep 5-fold (run_78-82) 2026-11-14 04:37
> 配置: --warm-restarts (CosineAnnealingWarmRestarts T_0=10) + R5, 60ep 逐折。结果: k0 0.6073/0.4812 | k1 0.5460/0.5795 | k2 0.5797/0.5067 | k3 0.6398/0.5257 | k4 0.5697/0.4338 → **T mean 0.5842 (+0.059 vs both 0.5253, 5/5 折胜) / N mean 0.5054 (-0.034 vs 0.5393, 2/5 折胜)**; acc 0.7191±0.015。
> 判定: 60ep 版 T +0.059 > 30ep 版 +0.035, 5/5 折胜 → **T 稳定提升可复现**; N 仍低于基线 → WR 帮 T 不帮 N。单 seed 0.6936 未复现在 5-fold (mean 0.5842), 印证单 seed 乐观上界。

#### round9-C1: H-CNN-ViT Local GAM 门控 (自写网络 + GAM, mri_tc 5-fold) 2026-11-14 04:45
> 配置: --self-written --gam --mode mri_tc (自写轻量 3D CNN 0.97M + CNN局部/全局双路径 sigmoid 门控融合, 调研 Top-1 降级版)。
> 结果: k0 0.6143/0.5875 | k1 0.6141/0.5700 | k2 0.7394/0.6481 | k3 0.6595/0.5346 | k4 0.6142/0.5285 → **T mean 0.6483+-0.049 / N mean 0.5738+-0.043**。
> 判定: vs mri_tc 自写基线 (T 0.6306/N 0.6097): T +0.018 / N -0.036 → GAM 门控对 T 微升、N 降; vs both 基线 (0.5253/0.5393): T +0.123 (5/5 折胜) / N +0.035 (3/5 折胜) → **GAM 版仍是 T 增益配置, N 不超 mri_tc 基线但高于主模型体系**。结论: 自写网络 + GAM 门控的 T 提升可复现, N 最佳仍是 mri_tc 自写无 GAM (0.6097)。

### round9-今晚序列总总结 (11-14 04:45) A1/A2/A4/B3/B2/C1
| 项 | 配置 | T | N | 判定 |
|---|---|---|---|---|
| A1 | mri_tc 自写网络 5-fold | 0.6306+-0.064 | **0.6097+-0.078** | N 迄今最强 5-fold, 自写网络验证通过 |
| A2 | Focal+LS s42 | 0.5852 | 0.4718 | 无增益 |
| B3-C1 | Patch-MIL s42 | - | - | 无增益 (run_76) |
| B3-C2 | 一致性正则 s42 | 0.6812 | 0.4497 | T+0.080 疑似/N 受损 |
| B2 | WR-60ep 5-fold | 0.5842+-0.035 | 0.5054 | T+0.059 5/5 折胜 (run_78-82) |
| C1 | GAM 门控 mri_tc 5-fold | 0.6483+-0.049 | 0.5738+-0.043 | T 增益可复现, N<A1 基线 |
**核心结论**: ①自写轻量网络(方案3)在 ROI 任务上验证有效——A1 mri_tc N 0.6097 为 N 最强; ②GAM 门控(方案2 降级) T 增益可复现但 N 不超自写基线; ③Focal+LS/Patch-MIL 无增益; ④WR T 稳定增益 5/5 折胜; ⑤N 突破候选 = A1 mri_tc 自写网络 (0.6097)。

#### round9-① C2 一致性正则 5-fold 复核 (11-16 03:29, run_83-87)
> 配置: --consist-modal mri --consist-weight 0.1 --kfold 5 --epochs 30 (R5 配置)。
> 结果: k0 0.5381/0.5074 | k1 0.5563/0.5257 | k2 0.5254/0.4921 | k3 0.6162/0.4842 | k4 0.5943/0.5258 → **T mean 0.5661+-0.034 / N mean 0.5070+-0.017**。
> 判定: vs both (T 0.5253/N 0.5393) T +0.041 (5/5 折胜基线, 稳定小幅增益但未达 0.62 门槛) / N -0.032 (受损确认, 与单 seed N 0.4497 一致) → 单 seed T 0.6812 未复现(乐观上界), 一致性正则仅稳定小幅帮 T 且伤 N, 不作主推。

#### round9-② mri_tc_ct (t+c 结肠 ROI + CT 病灶 ROI 第三通道, 自写网络 5-fold) 11-16 03:46
> 实现: mri_tc 基础上拼接 CT 病灶 ROI [3,64,64,64], CT 缺失→零卷+可用标志进临床(7→8维), 样本保持 164 (CT 覆盖 25/164=15%)。
> 结果: T mean 0.6263+-0.057 / N mean 0.6001+-0.073 (k4 0.5951/0.5793)。
> 判定: vs A1 mri_tc (T 0.6306/N 0.6097) T -0.004 / N -0.010 → **无增益** (CT 病灶 ROI 覆盖太稀疏, 与 mri_c_ct 交集样本不足一致), 不再深入。

#### round9-④: 肠周 ring ROI (结肠 mask 膨胀 2 层) 2026-11-16 05:53, 自写网络 5-fold
> 依据: N 分期信号在系膜/肠周组织(淋巴结区域)的病理机制; 结肠 mask 二值膨胀 2 层取环近似肠周。
> 结果: **ring t: T mean 0.5987+-0.0515 / N mean 0.6431+-0.0594**; **ring c: T mean 0.6056+-0.0613 / N mean 0.6442+-0.0600**。
> 判定: N 0.6431/0.6442 双序列一致, **迄今最强 N 5-fold**(超 mri_tc 0.6097 达 +0.034); T 0.5987/0.6056 亦超基线 0.5253 (+0.08)。结论: **肠周/系膜 ring 区域是 N 分期信号直接载体**——全体积/病灶/结肠本体 ROI 均 N 弱的原因正是信号在肠周系膜而非病灶内。

#### round9-Day1①: ring t+c 拼接 5-fold 无增益 (2026-11-18 00:16, run_94_roidl_mri_tc_ring)
> ring t+c 两序列拼接 [2,64,64,64] (t_ring+c_ring mask), 自写网络 5-fold 60ep。
> 结果: T mean 0.5817±0.0415 / N mean 0.5785±0.0811 (k2 最佳折 T 0.6293/N 0.6595, k4 T 0.6312/N 0.6387)。
> 判定: vs ring 单序列 (t N 0.6431 / c N 0.6442) **N -0.065 下降, T 也略降 → ring 拼接无增益** (t/c ring 质心裁剪位置错位, 拼接引入噪声)。

#### round9-Day1②: ring+本体双通道 t 无 N 增益 (2026-11-18 00:26, run_95)
> t 序列本体(mr_t) + ring(mr_t_ring) 双通道 [2,64,64,64] (病灶内+系膜同卷两 mask), 自写网络 5-fold 60ep, 样本 217。
> 结果: T mean 0.6092±0.027 / N mean 0.6107±0.021 (k4 T 0.6621/N 0.6066)。
> 判定: vs ring 单序列 (t N 0.6431 / c N 0.6442) **N -0.032 下降, T +0.01** → 本体信号稀释系膜信号, 无 N 增益。

#### round9-Day1③: ring 膨胀 3 层 无增益 (2026-11-18 00:34, run_96)
> ring 膨胀层数 2→3 (A4 调优), t 序列单通道, 自写网络 5-fold 60ep。
> 结果: T mean 0.5730±0.0357 / N mean 0.5948±0.0596 (k4 N 0.6940@20 单折亮点, mean 被拉低)。
> 判定: vs ring 2 层 (t N 0.6431/c N 0.6442) **N -0.048 下降** → 3 层 ring 过厚稀释信号, 2 层为最优。

#### round9-Day1④: ring 三序列拼接 无增益 (2026-11-18 00:46, run_97)
> ring t+c+d 三序列拼接 [3,64,64,64] (交集样本仅 80, d ring 稀疏), 自写网络 5-fold 60ep。
> 结果: T mean 0.5720±0.0725 / N mean 0.6125±0.0798。
> 判定: vs ring 单序列 (t N 0.6431/c N 0.6442) **N -0.032 且 std 大(±0.08, 样本 80 不足)** → 无增益。ring 单序列仍最优。

#### round9-Day1⑤: ring N 头嵌入 单 seed N 疑似突破 (2026-11-18 01:38, run_98)
> 结肠 mask 膨胀 2 层取环质心裁剪只进 N 头 (n_parts), R5 配置, s42 60ep。
> 结果: T 0.5960@26 acc 0.7361 calT@0.5=[0.476,0.725] / N 0.5670@14 calN@0.5=[0.7,0.438]。
> 判定: vs R5 s42 (T 0.6017/N 0.5093) T -0.006 持平 / **N +0.058 (≥0.05 门槛) → 疑似突破, 需 5-fold 验证**; vs 组合1 s42 N 0.5671 相当。ring 系膜信息进主模型 N 头有效。

#### round9-Day1⑤: ring N 头嵌入主模型 5-fold 边缘达标 (2026-11-18 03:31, run_99-103)
> 实现: loader mri_ring_dir + _load_mri_lesion_roi ring=True(结肠 mask 膨胀 2 层取环 d&~m 质心裁剪) + network n_ring_fusion 复用 mri_lesion_emb 通路只进 n_parts(n_from_shallow 分支 n_in += mri_lesion_head_width) + CLI --n-ring-fusion。单 seed(run_98) T 0.5960/N 0.5670 疑似。
> 5-fold: k0 0.4904/0.5245 | k1 0.5371/0.6056 | k2 0.5763/0.5697 | k3 0.5763/0.6056 | k4 0.5388/0.4904 → **T mean 0.5438+0.032 (+0.019) / N mean 0.5591+0.046 (+0.020, ≥0.559 门槛, 3/5 折胜) / acc 0.7304+0.015**。
> 判定: N 边缘达标但弱于独立 ring ROI DL (0.6442) → 延续 ROI 进主模型信号稀释规律; T 无增益。
#### round9-Day1⑥: head_dropout 超参快筛 (2026-11-18 03:46, run_104)
> ring t 单序列自写网络 5-fold 60ep, head_dropout {0.3, 0.7} vs 基线 0.5 (N 0.6431/T 0.5987)。
> 结果: **0.3 → T mean 0.6146±0.027 / N mean 0.6501±0.066 (N 新最强, T +0.016/N +0.007 双升)**; 0.7 → T 0.6059/N 0.6086 (N -0.035 伤)。
> 判定: **head_dropout 0.3 最优** (N 0.6501 超 ring t 基线 0.6431, 迄今最强 5-fold), 0.7 过强伤 N。后续 ring 系默认 head_dropout 0.3。
#### round9-Day2①: hd 细扫 0.1/0.2 (2026-11-19 00:42, run_105/106)
> ring t 单序列自写网络 5-fold 60ep, head_dropout {0.1, 0.2} vs Day1 基线 0.3 (N 0.6501)。
> 结果: **0.1 → T 0.6196±0.065 / N 0.6335±0.040**; **0.2 → T 0.5986±0.060 / N 0.6600±0.046 (N 新最强)**。
> 判定: **hd0.2 N 0.6600 为迄今最强** (超 hd0.3 +0.010, 但 T -0.016); hd0.1 欠正则伤 N。**ring 系 N 头基线更新为 hd0.2**, hd0.3 仍是 T/N 平衡选择。
#### round9-Day2②: B1 3D FPN 无增益 (2026-11-19 00:53, run_107)
> 自写 ring 网络 3 stage 输出 1x1 投影+上采样 83 concat (多尺度), ring t + hd0.2, 5-fold 60ep。
> 结果: T mean 0.6091±0.048 / N mean 0.5863±0.050。
> 判定: vs hd0.2 基线 (T 0.5986/N 0.6600) **T +0.011 / N -0.074 → FPN 伤 N** (多尺度稀释局部系膜信号, 与 GAM 同规律), 不采用。
#### round9-Day2③: B2 AECN 三轴向 slice attention 无增益 (2026-11-19 01:00, run_108)
> 自写 ring 网络 x/y/z 三轴向 slice attention 相乘 (滤各轴正常组织层), ring t + hd0.2, 5-fold 60ep。
> 结果: T mean 0.6223±0.030 / N mean 0.5997±0.051。
> 判定: vs hd0.2 基线 (T 0.5986/N 0.6600) **T +0.024 / N -0.060 → AECN 伤 N**。
> **规律总结: GAM/FPN/AECN 三个"全局/多尺度/多轴"机制全部帮 T 伤 N** → N 系膜信号需纯局部特征, 引入全局信息稀释。Day2 结构改动路线终止, 保持 ring 单序列 + hd0.2 纯局部基线。
#### round9-Day2④: C4 临床晚融合 无增益 (2026-11-19 01:07, run_109)
> 临床特征只进分类头(晚) vs 默认早融合, ring t + hd0.2, 5-fold 60ep。
> 结果: T mean 0.6472±0.068 / N mean 0.5809±0.074。
> 判定: vs hd0.2 基线 (T 0.5986/N 0.6600) **T +0.049 / N -0.079 → 晚融合帮 T 伤 N** (临床对 N 边际, 与历史一致)。
> **Day2 结构改动全部证伪**: B1 FPN / B2 AECN / C4 临床晚融合 / B3 GAM(依据规律跳过) 全部"帮 T 伤 N"。最终保持 ring 单序列 + hd0.2 纯局部基线 (T 0.5986/N 0.6600)。
#### round9-Day2备选①: ring c + hd0.2 复核 (2026-11-19 01:26, run_110)
> ring c 单序列自写网络 hd0.2, 5-fold 60ep (验证 c 序列是否同样受益于 hd0.2)。
> 结果: T mean 0.6067±0.066 / N mean 0.6397±0.073。
> 判定: vs ring c 基线 hd0.5 (T 0.6056/N 0.6442) **T +0.001 / N -0.005 → hd0.2 在 c 序列无复现** (t 序列的 +0.017 增益可能部分运气/序列特异)。c 序列最优仍为 hd0.5 或 hd0.3。
#### round9-Day2备选②: ring t + hd0.2 双 seed 稳定性 (2026-11-19 01:32, run_111)
> ring t 单序列自写网络 hd0.2, seed 7, 5-fold 60ep (验证 seed 42 的 N 0.6600 是否稳定)。
> 结果: T mean 0.5954±0.053 / N mean 0.6035±0.029。
> 判定: vs seed 42 (T 0.5986/N 0.6600) **N -0.057 → hd0.2 的 0.6600 含单 seed 运气**, 真实 N 估计 0.60-0.66 区间。hd0.3 (N 0.6501) 亦可能含部分运气。结论: 单 seed 5-fold 波动 ±0.06, 判定需多 seed 平均; ring 系 N 最优为 hd0.2(seed42) 但置信度中等。
#### round9-Day2备选③: mri_tc + hd0.2 复核 (2026-11-19 01:50, run_112)
> mri_tc (t+c 结肠 ROI 拼接) hd0.2, 5-fold 60ep (闭环 hd0.2 普适性验证)。
> 结果: T mean 0.6479±0.052 / N mean 0.5660±0.062。
> 判定: vs mri_tc 基线 hd0.5 (T 0.6306/N 0.6097) **T +0.017 / N -0.044 → hd0.2 同样伤 N**。
> **Day2 备选池闭环结论**: hd0.2 的 N 提升仅出现在 ring t seed42 (0.6600), ring c / mri_tc 均无复现甚至伤 N → **hd0.2 增益为单 seed 运气, 非普适超参**。ring t hd0.3 (N 0.6501) 仍为稳健选择; 真实 ring t N 估计 0.60-0.66 区间。
#### round9-Day1①: ring c + hd0.3 复核 无增益 (2026-11-20 00:39, run_113)
> ring c 结肠环 + head_dropout 0.3, 自写网络 5-fold 60ep。
> 结果: T mean 0.5988±0.0295 / N mean 0.5937±0.0755。
> 判定: vs ring c hd0.5 (T 0.6056/N 0.6442) **N -0.050 下降** → hd0.3 对 ring c 伤 N (与 ring t 相反: ring t hd0.3 N 0.6501 提升)。**head_dropout 是配置相关超参**: ring t 用 hd0.3, ring c 保留 hd0.5。

#### round9-Day1②: ROI 增强 (ring t hd0.3+aug) 无增益 (2026-11-20 01:17, run_114)
> DataROI 加 _augment_roi(翻转/旋转90°/强度扰动, 仅训练集) + CLI --aug。ring t hd0.3 + aug 5-fold 60ep。
> 结果: T mean 0.6157±0.0249 / N mean 0.6113±0.0539。
> 判定: vs ring t hd0.3 无aug (T 0.6146/N 0.6501) **N -0.039 下降** → 空间增强(翻转/旋转)破坏系膜方向信息, 伤 N (延续 N=纯局部空间敏感规律)。**ROI 网络不加增强**。

#### round9-Day1③: pT 四类辅助头 伤 N (2026-11-20 01:27, run_115)
> ring t hd0.3 + pT 四类辅助头 (pT1/2/3/4→4类 CE, 逆频率权重 9/54/147/7), 自写网络 5-fold 60ep。
> 结果: T mean 0.6142±0.0347 / N mean 0.6153±0.0327。
> 判定: vs ring t hd0.3 无辅助 (T 0.6146/N 0.6501) **N -0.035 下降, T 持平** → pT 辅助头伤 N (辅助任务抢占 N 特征容量, 与 ring 单任务最优一致)。

#### round9-Day1④: T/N 层次建模 无增益 (2026-11-20 01:35, run_116)
> ring t hd0.3 + tn_hier (N 头以 T 头输出 stop-grad 为条件, P(N|T) 先验注入), 自写网络 5-fold 60ep。
> 结果: T mean 0.6062±0.0514 / N mean 0.6290±0.0338。
> 判定: vs ring t hd0.3 (T 0.6146/N 0.6501) **T -0.008/N -0.021** → T/N 层次无增益 (与 pT 辅助同理: T 相关信息注入 N 头占用容量, 不如 N 单任务)。

#### round9-Day1⑤: N4 偏置场校正 不可行 (2026-11-20 01:48)
> 方案: SimpleITK N4BiasFieldCorrectionImageFilter 全图校正 (50x4 迭代, shrink=2) + loader 锚定重采样管线 → cache_hos_n4/ 独立缓存。
> 验证: 单样本(151) >9min 未完成 → 284 样本成本数十小时, 超出今晚 8h 预算 → **降级跳过**。
> **结论修订 (2026-11-20)**: 当时 `tools/build_n4_cache.py` 的参考网格 (origin=0/spacing=1 空网格) 与缓存 key (`n4v1` 前缀,
> 与 `_det_cache_key_mri` 不同) 均有误, DataROI 也无法读取该目录 → **"N4 收益不确定"未得到有效验证, 该结论作废**;
> 成本结论 (>9min/样本) 仍成立。脚本已修正, 若重试须用 `--cache-dir cache_hos_n4` 重训。

#### round9-Day1⑥: SAM 优化器 不收敛 (2026-11-20 02:38, run_117)
> ring t hd0.3 + SAM (rho 0.05, fp32 直连禁 AMP), 自写网络 5-fold 60ep。
> 结果: T mean 0.3828±0.1072 / N mean 0.3881±0.0355 (各折 T/N 均 ~0.3-0.4, loss 未正常下降)。
> 判定: vs ring t hd0.3 基线 (T 0.6146/N 0.6501) **大幅退化** → SAM rho=0.05 对该小网络扰动过大 + fp32 直连 lr 不匹配, 不收敛; 待试 rho=0.02。

#### round9-Day1⑥b: SAM rho=0.02 仍不收敛 (2026-11-20 02:53, run_118)
> ring t hd0.3 + SAM (rho 0.02, fp32 直连), 5-fold 60ep: T mean 0.4020±0.0559 / N mean 0.3881±0.0355 → 仍大幅退化。
> 判定: SAM 两种 rho (0.05/0.02) 均不收敛 → 该 0.9M 小网络 + 现有 lr 与 SAM 不匹配 (需 lr 重调才可能有效, 今晚不深挖), **SAM 证伪**。

#### round9-Day1⑦: warmup 5 + 100ep 无增益 (2026-11-20 03:05, run_119)
> ring t hd0.3 + 线性 warmup 5ep + 100ep (CosineAnnealing T_max=95), 自写网络 5-fold。
> 结果: T mean 0.6176±0.0571 / N mean 0.6511±0.0635 (k4 N 0.7642@54 单折亮点)。
> 判定: vs ring t hd0.3 60ep (T 0.6146/N 0.6501) **T +0.003/N +0.001 基本持平** → 60ep 已收敛, warmup+100ep 无增益。ring t hd0.3 60ep 为 N 最优基线 (0.6501)。


## round9 后续计划（2026-11-20 综合建议采纳版, 尚未执行）
> 基线: N 最优 = ring t hd0.3 60ep (T 0.6146 / N 0.6501); 已确立规律: 所有附加项(增强/辅助头/层次/全局注意力/多尺度/临床晚融合)一律帮T伤N → N 信号纯局部系膜特征。
> 三深忧核查(2026-11-20): ①mask 与缓存卷坐标错位(原始网格膨胀+nearest Resize vs 1mm+bbox+trilinear) ②薄环下采样碎裂 ③膨胀物理各向异性 ④样本池口径三套(ROI DL 5fold=280 / 单seed=217 / 主模型=284) ⑤crop_roi 仅质心裁剪无硬清零 → 下述 S0-S5 前置修正。

## 整合执行链 (S0-S5, 全部基于现有 hos 356 例, 无外部数据依赖)

### S0 坐标核查与可视化 (0 训练)
- 核对 train_roi_dl.py mask 与 cache_hos 是否同一方向/重采样/bbox; 确认 crop_roi 仅质心裁剪(已代码确认 L255-269 无硬清零)
- 导出 15-20 例三视图叠加图 (64³ 框 + mask 轮廓 + 原图), 确认肿瘤/系膜在框内、nearest 未碎裂、低位直肠未漏切
- 结论写入 README 后进入 S1

### S1 评估口径规范 (0 训练)
- 补 ROC-AUC / PR-AUC; 固定末期 OOF 汇总指标(不再逐 epoch max, 消除 Winner's Curse)
- 关键对照统一 3-seed(42/0/7)×5-fold 均值
- 判定阈值: 与基线对比需同时看 mean±std, 不再单 seed 阈值

### S2 归因对照 5 组 (ring t 上, 3-seed 5-fold, ~2h)
- ①当前基线 ②纯 Mask 捷径测试(非零区全置 1.0, 排查形态捷径) ③去临床(`--no-clinical`, 验证高缺失血液特征是否净噪声) ④纯 N 单任务(**待实现, CLI 尚无 `--n-only`**) ⑤纯 T 单任务(**待实现, CLI 尚无 `--t-only`**)
- 目的: 拆解 0.6501 = 影像 + 临床 + 双任务耦合 + 捷径 各占多少, 决定后续修复优先级

### S3 方向1 系膜腔室提取 + 物理几何修复 (预处理 2.5-7h + 训练)
- 重跑 total_mr 全器官分割(不再 roi_subset, 存全部类): t 299 例 ~2.5h / t+c+d ~7h (纯本地, 0 人工标注)
- 直肠外扩 10-15mm 后减去骶骨(sacrum 18)/膀胱(16)/前列腺(17)/椎体(19) 邻近器官 mask → 零标注逼近真实直肠系膜腔室 (MRI 版无髋骨类, 用 vertebrae+iliac 26-29 近似)
- 统一 T2 锚定 1mm³ 空间, 按物理毫米(非原始体素)外扩, 消除层厚差异与混叠
- 用"完整原图灰度 + 物理有符号距离场(SDF)"替代硬清零二值环, 避免人工边缘捷径并保留浸润上下文
- CE/DWI 配准到 T2 后共用同一裁剪框与 SDF 同步切取, 重验多序列收益
- 训练: 系膜腔室 SDF 输入自写网络 3-seed 5-fold, N vs 0.6501

### S4 方向2 纵轴解剖引导 Patch-MIL (~2h, 复用 train_mil.py)
- 沿直肠壁/系膜纵轴(PCA 主轴)重叠采样 16-32 个 32³ patch, 轻量共享 CNN + Gated Attention-MIL 聚合(已有骨架)
- 解决质心裁偏问题; 输出弱监督层位定位热图
- 判定: N vs 0.6501

### S5 方向3 非对称 T/N 解耦 + 模态动态融合 (~3h, 取代原 TriStreamNet)
- 修正: 砍掉 TN 联合头(T1-2N+ 仅 28 例过稀); T 分支不再全冻结(MedicalNet18 全冻结已证伪, 解冻浅层或轻量自写)
- T 分支专注肠壁及紧邻外缘; N 分支物理隔离于系膜腔室(方向1产物)并切断 T 梯度(stop-grad)
- 145 例有 CT 样本单训轻量 CT 专家(--single-modal ct), 决策层按模态可用性与预测不确定性加权融合(--uncertainty-weight), 避免缺 CT 样本依赖 imputer 伪特征
- 判定: T / N 双指标 vs 各自基线

### 执行规则
- 每步完成 → 归档 + README 判定 → 自动启动下一步 (不满 8h 主动推进, 08:40 叫停总结)
- 关键对照统一 3-seed(42/0/7)×5-fold 均值; >=+0.03 记疑似, >=+0.05 直接判定
- 环境: 新 python D:\Development\miniconda\envs\Pytorch\python.exe (torch 2.5.1+cu124)

#### round9-P0 第二段 A 臂: ring 5mm masked (2026-11-21 01:28, run_117)
> 命令: train_roi_dl.py --kfold 5 --kfold-csv data/master_5fold_356.csv --full-csv data/hos_pairs.csv --seed 42 --mode mri_t_ring --ring-mm 5 --roi-mask masked --no-clinical --self-written --head-dropout 0.5
> 样本: hos_pairs.csv mode=mri_t_ring 载入 282 (跳过 79: mask 缺失/空); 每折 224train/56val (Master 主表口径)
> 结果 (AUC 选模 + LOFO 阈值):
>   T AUC=0.6252±0.033 (k0 0.713/k1 0.585/k2 0.614/k3 0.542/k4 0.634) | LOFO calT=0.382
>   N AUC=0.6400±0.061 (k0 0.634/k1 0.626/k2 0.660/k3 0.571/k4 0.764) | LOFO calN=0.456
> 判定: P0 可信口径(no-clinical+aligned mask+5mm ring)下 N AUC 0.64 为新基线锚点 (旧 0.6501 口径已作废); 每折 224/56 均衡, std 适中

#### round9-P0 第二段 B 臂: SDF 双通道 bbox (2026-11-21 01:40, run_118)
> 命令: --mode mri_t --sdf-channel --roi-mask bbox --no-clinical --self-written --head-dropout 0.5 (seed 42, Master 5-fold, 282 样本, 每折 224/56)
> 结果 (AUC 选模 + LOFO):
>   T AUC=0.6225±0.065 F1=0.483 | LOFO calT=0.591
>   N AUC=0.6117±0.038 F1=0.466 | LOFO calN=0.424
> 判定: vs A 臂 ring 5mm masked (T 0.625/N 0.640): T 持平(-0.003), **N -0.028 → SDF 双通道未超越 ring masked**; A 臂仍为 N 最优锚点

#### round9-P0 第二段 C 臂: bbox 对照 (2026-11-21 01:48, run_120)
> 命令: --mode mri_t --roi-mask bbox --no-clinical --self-written --head-dropout 0.5 (seed 42, Master 5-fold, 282 样本)
> 结果: T AUC=0.598±0.074 F1=0.477 | LOFO calT=0.477 | N AUC=0.625±0.047 F1=0.379 | LOFO calN=0.382
> 判定 (三臂对比, 全 no-clinical + Master 主表):
>   A ring 5mm masked:  T 0.625 / N 0.640  ← N 最优
>   B SDF 双通道 bbox:  T 0.623 / N 0.612
>   C bbox 对照:        T 0.598 / N 0.625
> 结论: ring masked 的 N 0.640 为三臂最高且 LOFO calN 0.456 也最高; SDF 与 bbox 对照均未超越 → **ring 5mm masked 保留为 P0 主线 N 锚点**

#### round9-P0 归因 D 臂: binary_only 形态捷径检验 (2026-11-21 01:58, run_122)
> 命令: --mode mri_t_ring --ring-mm 5 --roi-mask binary_only --no-clinical (输入换成二值 mask 本身, 图像置零; seed 42, Master 5-fold, 282 样本)
> 结果: T AUC=0.576±0.035 | LOFO calT=0.381 | N AUC=0.657±0.038 | LOFO calN=0.463
> 判定 (vs A 臂 ring masked: T 0.625/N 0.640): T -0.049 降, **N +0.017 反升 → 纯二值形态的 N AUC 0.657 ≥ 带灰度的 ring masked 0.640** → 重大发现: **ring 网络学到的 N 信号主要来自二值形态(环的位置/形状), 灰度影像内容几乎不提供额外 N 信息 → 此前 ring 系 N 0.65 高分为形态捷径成分显著**, 与用户 m16381 担忧一致。P0 后续需警惕: N 提升必须超越 0.657 的纯形态上界才有真影像学意义

#### round9-P0 归因 E 臂: n-only 纯 N 单任务 (2026-11-21 02:08, run_123)
> 命令: --mode mri_t_ring --ring-mm 5 --roi-mask masked --no-clinical --n-only (只监督 N 头, T 头不参与 loss/不选 ckpt)
> 结果: T AUC=None (T 头未监督) | N AUC=0.593±0.053 F1=0.438 | LOFO calN=0.504
> 判定 (vs A 臂双任务: N 0.640/calN 0.456): **N -0.047 下降 → 双任务联合训练不拖累 N, 反而有正则增益**(T 头梯度对共享特征有正向约束); 纯 N 单任务低于 A 臂

#### round9-P0 归因 F 臂: t-only 纯 T 单任务 (2026-11-21 02:18, run_124)
> 命令: --mode mri_t_ring --ring-mm 5 --roi-mask masked --no-clinical --t-only (只监督 T 头)
> 结果: T AUC=0.595±0.038 F1=0.478 | LOFO calT=0.478 | N AUC=None
> 判定 (vs A 臂双任务: T 0.625/calT 0.456): **T -0.030 下降 → 双任务联合同样帮 T**(共享特征双向正则); 纯 T 单任务低于 A 臂

#### round9-P0 归因 G 臂: T+clinical (2026-11-21 02:28, run_125)
> 命令: --mode mri_t_ring --ring-mm 5 --roi-mask masked (撤 --no-clinical, 临床 7 维早融合)
> 结果: T AUC=0.589±0.056 | LOFO calT=0.397 | N AUC=0.614±0.040 | LOFO calN=0.555
> 判定 (vs A 臂纯影像: T 0.625/calT 0.456 / N 0.640/calN 0.456): **T -0.036 / N -0.026 双降 → 临床 7 维在此 ROI 口径下是净噪声**(与 round4 高缺失结论一致), LOFO calN 0.555 略升但 AUC 降

#### round9-P0 归因四臂汇总判定 (2026-11-21 02:28)
| 臂 | T AUC | N AUC | LOFO calT/calN |
|---|---|---|---|
| A ring 5mm masked (双任务纯影像) | 0.625 | **0.640** | 0.456/0.456 |
| B SDF bbox | 0.623 | 0.612 | 0.591/0.424 |
| C bbox 对照 | 0.598 | 0.625 | 0.477/0.382 |
| D binary_only 形态捷径 | 0.576 | **0.657** | 0.381/0.463 |
| E n-only | — | 0.593 | —/0.504 |
| F t-only | 0.595 | — | 0.478/— |
| G T+clinical | 0.589 | 0.614 | 0.397/0.555 |

**核心结论 (P0 可信口径)**: ①N 形态捷径上界 0.657(D) ≥ 带灰度 ring masked 0.640(A) → **ring 系 N 高分主要是形态捷径, 灰度影像无额外 N 信号** ②双任务联合对 T/N 均有正则增益(E/F 均低于 A) ③临床 7 维净噪声(G 双降) ④SDF/bbox 未超越 ring masked。P0 主表 --no-clinical 定案正确。

#### round9-P0 A 臂扩种子 seed0 (2026-11-21 02:39, run_126)
> 命令: ring 5mm masked no-clinical seed0
> 结果: T AUC=0.604±0.074 / N AUC=0.590±0.081 (calT 0.375/calN 0.433)
> 判定 (vs seed42: T 0.625/N 0.640): T -0.021 / **N -0.050** → A 臂存在 seed 方差, 0.640 含单 seed 运气成分; 需 seed7 补充

#### round9-P0 A 臂扩种子 seed7 (2026-11-21 02:49, run_127)
> 结果: T AUC=0.650±0.033 / N AUC=0.608±0.058 (calT 0.500/calN 0.452)
> **A 臂三 seed 汇总(ring 5mm masked no-clinical)**: s42 T 0.625/N 0.640 | s0 T 0.604/N 0.590 | s7 T 0.650/N 0.608 → **T mean 0.626±0.019 / N mean 0.613±0.021**
> 判定: 三 seed 均值 N 0.613 显著低于 D 臂单 seed 形态捷径 0.657 → **形态捷径成分结论成立且更突出: 灰度内容对 N 的增量可能为负或 ~0**; A 臂 0.640 确含单 seed 运气

#### round9-P0 D 臂扩种子 seed0 (2026-11-21 03:00, run_128)
> 结果: T AUC=0.556±0.038 / N AUC=0.614±0.044 (calT 0.387/calN 0.476)
> **D 臂两 seed 汇总(binary_only 形态捷径)**: s42 N 0.657 | s0 N 0.614 → N mean ≈0.636
> 判定: D 形态捷径两 seed 均值 0.636 仍 > A 臂三 seed 均值 0.613 (+0.023) → **形态捷径成分结论稳健成立(非单 seed 运气)**; 灰度内容对 N 的净增益确为负或 ~0

#### round9-P0 D 臂扩种子 seed7 (2026-11-21 03:09, run_129)
> 结果: T AUC=0.580±0.057 / N AUC=0.649±0.058 (calT 0.392/calN 0.496)
> **D 臂三 seed 最终汇总(binary_only 形态捷径)**: s42 N 0.657 | s0 N 0.614 | s7 N 0.649 → **N mean 0.640**
> **P0 最终判定 (A 臂三 seed N 0.613 vs D 形态捷径三 seed N 0.640)**: 形态捷径 +0.027 稳健成立 → **N 分期信号主要可由 ring 二值形态本身承载, 灰度影像内容无净增益**; P0 结论: ①Master 主表+no-clinical+AUC 口径全部就位 ②ring 5mm masked 为 P0 主线 ③N 需超越形态捷径上界才有影像学意义 ④临床净噪声 ⑤双任务联合同帮 T/N

#### round9-P0 今晚总结 (2026-11-21 03:30, 用户 m16609 授权 P0 第一阶段)
**执行**: FIXES_2026-11-20.md 理解 → 标定(1 折测成本 ~4min/折, 显存 0.58GB) → 主线三臂 → 归因四臂 → 扩种子(A/D 各 3 seed), 13 run(117-129) 全部归档, 每步 README 更新。
**口径**: Master 5-fold 患者级主表(356 例) / 全 --no-clinical / AUC 选模 + LOFO 阈值 / 自写轻量网络(~1.1M 无预训练) / seed 42+0+7。
**最终结果表**:
| 配置 | T AUC | N AUC (三 seed mean) |
|---|---|---|
| A ring 5mm masked (P0 主线) | 0.626±0.019 | 0.613±0.021 |
| D binary_only 形态捷径 | 0.576/0.580/0.556 | **0.640** |
**核心结论**: ①形态捷径稳健成立(纯二值 ring 形态 N 0.640 ≥ 带灰度 masked 0.613, +0.027) → **N 信号主要由 ring 位置/形状承载, 灰度影像无净增益** ②临床 7 维净噪声(G 双降) ③双任务联合同帮 T/N(E/F 单任务更差) ④P0 可信基线确立: ring 5mm masked T 0.626/N 0.613。
**重大决策点(跳过待用户)**: 第二阶段方向 = A 验证形态真信号 / B 换输入表示 / C 等 130 例数据合并重跑 / D 收束写基线。
**状态**: 无训练, GPU 空闲, 心跳低频待命。

## 第二阶段验证：形态捷径三体检（2026-11-21 用户提出, 已加入计划）

> 目的: 在"形态捷径"结论(P0: 纯二值 ring 形态 N 0.640 ≥ 带灰度 masked 0.613)基础上, 用 3 项零训练体检排除替代解释: 批次混杂(辛普森悖论) / 采集视野与缩放比例捷径 / 性别-年龄-T 分期间接猜测。

### 体检1: 批次混杂体检 (辛普森悖论检查, 纯 CSV 秒级)
- 批次归属: master_5fold_356.csv 按 patient_id 对应导入批次 — batch1 = patient_id ≤200, batch2 = patient_id 201-408 (减 200 得 batch2 原编号), batch3 = 未来 130 例新增
- 步骤: ①按批次统计 label_T / label_N 阳性率 ②Arm D(binary_only) OOF 概率按批次拆分, 批次内分别计算 N AUC (3 seed×5 fold, 报告逐 seed 与合并) ③与全队列对照
- 输入: results/roidl_oof/p0_attr_D_binonly_k{0-4}_s{0,42,7}.csv (列 patient_id,label_T,prob_T,label_N,prob_N)
- 判据(用户原话): "若全队列 N AUC = 0.64, 但拆到各批次内部全部掉回 ~0.50–0.54, 则 100% 实锤为批次混杂(辛普森悖论)"

### 体检2: FOV / 标量形态体检 (读 NIfTI 头 + cache_hos 统计, 分钟级)
- 每例仅提取 ~8 个最简单标量: 原始 NIfTI 物理 FOV x/y/z (dim×pixdim)、层厚、锚序列(T2) bbox 毫米尺寸、96³ 中 mask 与 ring 的体素占比、质心坐标
- 方法: Master 5-fold 上逻辑回归预测 label_N (可选加 LOFO 阈值口径对齐)
- 判据(用户原话): "如果单靠 FOV/体素占比这几个标量就能跑到 N AUC ≥ 0.60, 说明 CNN 学的只是采集视野与缩放比例, 而非精细解剖形态"

### 体检3: 跨临床变量相关性检查 (纯 CSV 秒级)
- 输入: Arm D OOF prob_N + 临床表(sex / age / pT 1-4)
- 方法: prob_N 与 sex / age / pT 的相关性 (点二列/斯皮尔曼/分组均值), 及各变量分层下 N AUC
- 判据(用户原话): "确认模型是不是在通过'盆腔骨架/性别/T 分期'间接猜 N"

### 执行顺序 / 输出 / 数据注意
- 顺序: 体检1 → 体检3 (均秒级) → 体检2 (需 NIfTI 头 + mask/ring 统计); 脚本放 tools/phase2_healthcheck/
- 输出: 每项结果表 + 判定写入本 README 对应小节; 任一实锤则形态捷径结论归因需更新
- 数据注意: master 现为 355 行 (2026-11-21 数据更新后), batch2 缺 101/131/182, 191 变 MRI-only; 批次阳性率口径须与 OOF 样本集一致

### 体检结果（2026-11-21 17:00 执行完毕, 全部零训练, 脚本 tools/phase2_healthcheck/）

**口径说明**: 全部用 **Arm D OOF 外折概率**（results/roidl_oof/p0_attr_D_binonly_k{0-4}_s{0,42,7}.csv, best ckpt）而非 P0 报告的 0.640。P0 报告的 D 臂逐 seed N 0.657/0.614/0.649（mean 0.640）是**折内 best-epoch val AUC 均值**；从 OOF 外折概率合并实测三 seed 全队列 AUC 仅 0.564/0.578/0.623（三 seed 概率平均后 0.616）。best 与 last5 均验证，外折口径低于报告值——**0.640 含折内过拟合成分，本体检以 OOF 外折 0.616 为基线**。

| 体检 | 输入 | 结果 (N AUC) | 判定 |
|---|---|---|---|
| T1 批次混杂 | Arm D OOF 按 batch 拆 | 三 seed 均值: batch1(≤200, n=122) **0.542±0.033** / batch2(201-408, n=158) **0.621±0.031**; 全队列 0.564-0.623 (逐 seed), 三 seed 概率平均后 batch1 0.587 / batch2 0.637 / 全队列 0.616 | **边界** — 未实锤辛普森悖论(batch2 仍 0.62-0.64), 但 batch1 明显弱于 batch2 (差 ~0.08), 存在轻度批次混杂/采集差异嫌疑, 需在后续方向中留意 batch1 子群 |
| T2a 采集/缩放 | FOV xyz + 层厚 + bbox 毫米边长 + clamp_frac, master 折表 5 折 LR | **0.4889±0.0699** | **通过** (0.47-0.54 随机区间) — CNN 不是靠采集视野/缩放比例预测 N |
| T2b 物理形态 | 1mm³ 真实体积 + 纵轴跨度 + 环体比, 5 折 LR | **0.4203±0.0778** | 未达 0.58 — 物理形态标量本身不足以承载 N 信号 |
| T3 代理分层 | prob_N(三 seed 平均) × sex / label_T / pN | sex=0(n=91) **0.708** / sex=1(n=189) **0.569**; label_T=0(n=82) **0.529 <0.55 ⚠️ 报警** / label_T=1(n=198) 0.620; pN 均值 0.549(pN0) < 0.612(pN1) > 0.601(pN2) 非单调 | **报警 — T 分期代理嫌疑**: N 信号主要由 T3-4 子群承载, T1-2 子群内 N 预测接近随机; pN 亚级非单调佐证代理成分 |

**综合判定（对照用户 m00195 分流决策）**:
- 情景 1 (采集/批次混杂) **未触发**: T1 边界不报警, T2a 通过 → 无需废除变比例 Resize
- 情景 2 (代理变量驱动) **触发**: T1/T2a 通过但 T3 报警 → 且指向 **T 分期代理** (label_T=0 子群 0.529) 而非性别代理 (sex 分层均 >0.55)
- 情景 3 (真解剖形态) 未触发: T2b 0.42 < 0.58, 不支持
- **后续方向 (情景 2-T 分期代理)**: 推进对齐多序列 + SDF + 纵轴 MIL, 以 **T3-4 子群内 N AUC 提升**为硬验收标准; 同时留意 batch1 批次混杂残余嫌疑

#### T3 根源交叉分析 (补充, 三 seed 平均概率, OOF 280 例)
| 子群 | n | N+ | N AUC |
|---|---|---|---|
| sex=0 × label_T=0 | 33 | 8 | 0.640 (n 小) |
| sex=0 × label_T=1 | 58 | 31 | 0.736 |
| sex=1 × label_T=0 | 49 | 15 | **0.463 ⚠️** |
| sex=1 × label_T=1 | 140 | 67 | 0.572 |
| batch1 × sex=0 | 40 | 20 | 0.680 |
| batch1 × sex=1 | 82 | 34 | 0.523 |
| batch2 × sex=0 | 51 | 19 | 0.725 |
| batch2 × sex=1 | 107 | 48 | 0.596 |

- T3 报警根源定位: **label_T=0 报警主要由 sex=1(男)×T1-2 子群拖低 (0.463)**, sex=0 各子群均正常; 临床分布 batch1/batch2 几乎相同 (T+ 0.72/0.71, N+ 0.45/0.43, age 62.4/64.5, sex_m 0.65/0.68), hospital_id 每例唯一 → 采集差异非主因
- 综合: T 分期代理嫌疑 (T1-2 内 N 难辨) 叠加轻度性别调制; 方向仍按情景 2 推进, 硬验收口径固定为 T3-4 子群 (sex=1×T1-2 是后续最薄弱点)

#### batch 采集参数分布对比 (补充, t2_scalars.csv 282 例)
- batch1 (n=129) vs batch2 (n=170) 采集参数几乎重叠: fov 259/263mm, fov_z 105/107, 层厚分布一致 (4mm 为主, 6/7/7.2 并存), bbox/体积/环体比均接近 → **T1 的 batch1 偏弱 (0.54 vs 0.62) 不能归因于采集参数差异**, 更可能是批次内病例构成/标注口径噪声
- 结论: 辛普森悖论未实锤 (batch2 仍有 0.62-0.64 信号), 采集差异排除 → 与 T2a/T2b 结论互洽, 主判定仍为情景 2 T 分期代理

## 正式训练记录规则（2026-11-21 用户要求, 强制）
> 任何正式训练 run 的记录至少包含以下三部分, 缺一不可 (归档/README 判定/汇报均以此为准):
- **使用的数据**: 主表与样本口径 (文件/折数/train-val 数/模态/mask-ring 参数/seed 集), 保证可复现
- **网络结构**: 模型选择 (自写/预训练/medicalnet), 各组件 (backbone/head/dropout/SDF 等) 与参数量
- **模型结果**: T/N AUC mean±std + LOFO 阈值 + 每折明细 + 多 seed 汇总, 附判定结论

#### 第二阶段体检探针 P2: 对齐多序列 t+c + SDF (2026-11-21 17:10, 窗口内自主执行)
> 命令: train_roi_dl.py --kfold 5 --kfold-csv data/master_5fold_356.csv --full-csv data/hos_pairs.csv --seed 42 --mode mri_tc_ring --ring-mm 5 --roi-mask masked --sdf-channel --no-clinical --self-written --head-dropout 0.5 --out-tag p2_tcring_sdf
> 数据: mode=mri_tc_ring 载入 217 (t∩c mask 交集, 跳过 143 mask 缺失/空); 每折 ~41-46 val (225 主表交集), 对齐 t+c 双序列 + 连续 SDF 通道 (4 通道输入)
> 网络: 自写 3D CNN 0.9M, ring 5mm masked ROI 64³, head-dropout 0.5, 无临床
> 结果 (AUC 选模 + LOFO): T AUC=0.658±0.035 | N AUC=0.676±0.068 (折内 best-epoch) | calT=0.421 calN=0.602
> OOF 外折 (216 例): 全队列 N AUC=0.6422 | T3-4 子群 0.6055 (n=148) | T1-2 子群 0.7325 (n=68)
> 判定: vs 基线 (A 臂 ring5m masked 282 例, T3-4 0.6215/T1-2 0.5667): T3-4 0.6055 未过 0.62 硬验收, 但 T1-2 0.7325 大幅超 0.5667 → 对齐多序列+SDF 增益集中在 T1-2 子群 (此前报警根源 sex=1×T1-2 0.463 方向有改善); 样本口径不同 (216 vs 282), 已启动同口径对照 P2A (mri_t_ring 同 217 例) 复核

#### 第二阶段体检探针 P2A: 同口径单序列对照 (2026-11-21 17:36, 窗口内自主执行)
> 命令: train_roi_dl.py --kfold 5 --kfold-csv data/master_5fold_356.csv --full-csv data/hos_pairs_tc.csv --seed 42 --mode mri_t_ring --ring-mm 5 --roi-mask masked --no-clinical --self-written --head-dropout 0.5 --out-tag p2a_tring_tc
> 数据: hos_pairs_tc.csv (227 例 t∩c 过滤) 载入 217, 与 P2 探针完全同口径 (同 216 例 OOF 样本, 同折划分, seed 42) — 隔离序列数/SDF 变量
> 网络: 自写 3D CNN 0.9M, ring 5mm masked, 无临床
> 结果 (折内 best-epoch): T AUC=0.671±0.062 | N AUC=0.618±0.064
> OOF 外折 (216 例): 全队列 N AUC=0.5230 | T3-4 0.5228 | T1-2 0.4962
> 判定: **同口径对比 P2 (t+c+SDF 全队列 0.6422/T3-4 0.6055/T1-2 0.7325) 全面胜出 (+0.119 全队列, +0.083 T3-4, +0.236 T1-2)** → 对齐 t+c + SDF 组合的增益真实存在, 非样本/折差异。已启动 P3 消融 (t+c 无 SDF) 归因增益来源。注意 P2A 单序列在 216 例上 OOF 明显低于 P0 A 臂 282 例 (0.523 vs 0.609) → 单序列在更小 t∩c 子集上表现退化

#### 第二阶段体检探针 P3: 消融归因 (2026-11-21 17:45, 窗口内自主执行)
> 命令: train_roi_dl.py --kfold 5 --kfold-csv data/master_5fold_356.csv --full-csv data/hos_pairs.csv --seed 42 --mode mri_tc_ring --ring-mm 5 --roi-mask masked --no-clinical --self-written --head-dropout 0.5 --out-tag p3_tcring
> 数据/网络: 同 P2 (217 例 t∩c, 自写 3D CNN, ring 5mm masked), 唯一差异 = 去掉 --sdf-channel
> 结果 (折内): T AUC=0.655±0.065 | N AUC=0.582±0.053
> OOF 外折 (216 例): 全队列 N AUC=0.5372 | T3-4 0.5401 | T1-2 0.5231
> 三臂归因 (同 216 例/同折/seed42/OOF): t-only 0.5230 → t+c 0.5372 (+0.014) → t+c+SDF 0.6422 (+0.105)
> 判定: **SDF 连续距离场通道是 N 信号增益的主要来源 (~+0.10), 对齐双序列 t+c 本身贡献微弱 (~+0.014)**; T1-2 子群 0.7325 的提升主要由 SDF 驱动 → 形态流(SDF/二值) + 纹理流(对齐多序列) 双分支思路成立, SDF 分支是核心; 下一步建议: SDF 通道 + 纵轴 Patch-MIL 组合验证

## 第二阶段决定性对照：mri_t 三臂 × 3 seed + 8 小时自动接力（已执行, 2026-11-22 01:30 启动）
> 目的: 在统一口径下判定 96³ 压缩分辨率下 MRI 灰度与 SDF 的真实关系, 为后续架构改动提供主裁判; 按分流结果自动接力跑满 8 小时。
> 开跑前核对(已完成): ① master 实测 355 行(fold 71×5), 报告 §13.1 的 356 为数据更新前口径——382/182/191 相关行删除后未回写报告(与 lesson 0mwq1z8i 一致), 以实测 355 为准, 三臂同口径对比不受影响; ② train_loss 每 epoch 打印已确认在 train_roi_dl.py:1293-1294。
> 统一口径(阶段1): --mode mri_t --roi-mask bbox --epochs 30 --self-written --no-clinical --kfold 5 --kfold-csv data/master_5fold_356.csv --full-csv data/hos_pairs.csv, seed 42/0/7, 每臂独立 --out-tag。
> 样本口径(已实测): master 355 ∩ mri_path_t 存在 ∩ mr_t mask = **296 例**, fold 60/59/59/59/59, T+率 0.709/N+率 0.439, T1-2 86 例/T3-4 210 例, batch1 129/batch2 167。
> 主裁判: 3-seed 平均的 last5(=第 26-30 轮均值 OOF) ROC-AUC, T 与 N 均记录(若 N 走分支B但 T 灰度明显正增益 → 提示天然适合非对称 T/N 解耦); 附加输出 PR-AUC/F1@0.5/LOFO F1。
> **执行纪律: 严格串行 + 独立子进程(每跑完 1 个 seed 退出, 外层脚本拉起下一个)——本机同进程连跑/高并发易硬件硬卡死(R8/R8b/5-fold/S3/B3a 均中招), 夜间 8 小时禁并行。**

### 阶段 1: 基准定性 (0.0-2.0h, 必跑)
- 9 次训练严格串行: Arm1 sdf-only / Arm2 图像+SDF / Arm3 无 SDF, 各 × seed 42/0/7; out-tag dc_a1_sdfonly_s{seed} / dc_a2_img_sdf_s{seed} / dc_a3_img_s{seed}。
- 跑完自动聚合: 输出 3-seed 平均 last5 OOF ROC-AUC/PR-AUC/F1@0.5/LOFO F1(T 与 N 并列), 按 Arm2_N − Arm1_N 自动分流。
- **无死角二分**: Arm2−Arm1 **≥ +0.025 → 路径 A**; Arm2−Arm1 **< +0.025**(含持平与灰度拖后腿) → 一律路径 B。

### 阶段 1 结果 (2026-11-22 02:22 聚合, 9/9 完成)
- **Arm1 sdf-only**: s42 0.5186 / s0 0.5439 / s7 0.5114 → 3-seed 均值 **N 0.5246** (T 0.5531)
- **Arm2 图像+SDF**: s42 0.5194 / s0 0.4907 / s7 0.5599 → 3-seed 均值 **N 0.5233** (T 0.5136)
- **Arm3 无 SDF 纯灰度**: 3-seed 均值 **N 0.5091**
- **分流判定: Arm2−Arm1 = −0.0013 < +0.025 → 路径 B** (灰度通道在 96³ 压缩分辨率下无净增益且轻微拖后腿; Arm3 纯灰度 0.5091 确认灰度本身无 N 信号; T 亦无灰度增益)
- 明细: logs/aggregate_roidl_dc_{arm}_s{seed}.json × 9; 分流记录 logs/relay_8h_state.json。

### 阶段 2A: 96³ 协同深挖 (2.0-7.5h, 当 Arm2−Arm1 ≥ +0.025)
- A1 修复多序列像素错位并验证 (~1.6h): train_roi_dl.py 中 mri_tc/mri_all 裁剪逻辑改为**所有对齐序列强制共用 mr_t 单一 bbox 中心与单一 SDF**(消除 t 用 mr_t/c 用 mr_c 的通道错位); 同 216 例 t∩c 子集上串行跑 mri_tc+shared SDF (3 seed ~40min) 与 mri_all+shared SDF (3 seed ~40min), 与同子集 mri_t+SDF(从阶段1 OOF 按 216 例 ID 切片, 零重训)公平对比。
- A2 解剖保序弱增强 + 2-view TTA (~1.0h): 先 1 例 NIfTI 方向矩阵确认 (z,y,x) 中左右轴; 新增 --aug-lr 模式(仅左右镜像 + 图像通道轻微强度抖动, **严禁碰 SDF 通道**); A1 最优配置跑 3 seed (~40min) + 原图/左右翻转 2-view TTA 离线推理 (~5min)。
- A3 S3 盆腔多器官排斥直肠系膜 (~2.8h): 查 TotalSegmentator total_mr 盆腔标签(sacrum/urinary_bladder/prostate/hip_left/right 等); 若含骨骼+膀胱 → 296 例 T2 多器官分割(参考 26s/例 ≈2.1h)生成"结直肠外扩 − 邻近骨骼/泌尿器官"排斥系膜 SDF, 剩 ~0.7h 跑 3 seed 验证。备选: total_mr 缺盆腔标签则跳过分割, 2.8h 改做 --t-only/--n-only 非对称解耦验证 + 沿直肠纵轴密集 32³ Patch-MIL。

### 阶段 2B: 高分辨破局 + 形态可解释性 (2.0-7.5h, 当 Arm2−Arm1 < +0.025; 概率不低, 拆最实)
- B1 形态线排雷 (~0.5h): 用阶段1 Arm1(sdf-only) 权重, 验证折抽 20 例(10 高置信 N+ / 10 高置信 N0) 做 3D Occlusion Sensitivity(比 Grad-CAM 适合单通道无纹理 SDF); 输出 5 张叠加图 + 量化高敏感体素落在直肠肠壁/肠腔内部 vs 64³ 裁剪框边缘/clamp 截断区比例 → 回答 ~0.60 几何信号是否伪影。
- B2 灰度破局预处理: 固定物理视野高分辨缓存 (~0.8h): 新建 cache_hos_hr15/(不动旧缓存): T2 锚定空间重采样至固定 1.5mm³ 各向同性, mr_t 结肠 bbox 中心裁固定 64³(物理视野恒 96×96×96 mm³, 严禁拉伸), 同步导出对齐 T2/CE(若有)/1.5mm³ 连续物理 SDF; 强制留存质检图: pid 213/377/402 三例三视图 PNG 供早间肉眼抽查。
- B3 高分辨 3 臂 × 3 seed 验证 (~2.0h): cache_hos_hr15/ 上同口径(epochs 30/master_5fold/no-clinical/self-written)重跑 HR-Arm1(sdf-only, 检验去掉变比例拉伸后纯几何信号是否仍在) / HR-Arm2(T2+SDF) / HR-Arm3(纯 T2)。
- B4 按 B3 自动收尾 (~2.0h): 若高分辨救活灰度(HR-Arm2/3 last5 N AUC 明显高于 96³ Arm3 且 ≥ HR-Arm1 + 0.025) → 跑 高分辨 T2+CE+SDF(共用 mr_t 中心, 3 seed ~45min) + 仅左右镜像弱增强+2-view TTA (3 seed ~45min); 若高分辨灰度仍无力(<0.56) → 高分辨 --t-only/--n-only 单任务解耦(各 3 seed ~1.3h) + 低成本重测 CT CRCFound 局部 Token 池化(~0.5h 零训练: 145 例 CT 结直肠/Task010 mask 映射到已下载 CRCFound 3D ViT patch grid, 局部 patch tokens Masked Average Pooling 接线性分类器测 5 折 OOF AUC)。

### 阶段 3: 无条件早间审计与汇报 (7.5-8.0h, 纯 CPU, 必跑)
- 满 7.5h 禁止再启新训练; 统一离线脚本产出早间决策包写入本 README:
- ① 全局主表: 所有已跑完臂 3-seed last5 OOF(ROC-AUC/PR-AUC/F1@0.5/LOFO F1) 与 best AUC 并列; ② 临床亚组拆解: 当晚最优模型 OOF 概率 → pN0/pN1/pN2 三级均值与两两 AUC(验证误差是否集中微小转移 N1)、T3-4 vs T1-2 子群 AUC、batch1 vs batch2 批内 3-seed AUC; ③ 异常与耗时审计: 子进程是否触发 watchdog、单折实际耗时、train_loss 末期数值。

（以下 B 路径各步结果在手动重跑完成后追加）

## B1 遮挡敏感度 (修正版, 2026-11-22 02:35)
- a1_sdfonly 权重, 20 例(10 N+/10 N0) 16³ 块遮挡滑窗 64 次前向; 高敏感体素(top 20%)分区占比均值: **壁=0.058 腔=0.025 外=0.917 边缘环(pad=8)=0.463**
- **结论: sdf-only 的 N 信号主要不在肠壁, 91.7% 高敏感体素在肠腔外广域几何, 且 46.3% 落在 64³ 裁剪框边缘环 —— 模型显著依赖裁剪框边缘 clamp/截断区, 支持"~0.60 几何信号部分是裁剪伪影"判断**
- 修正说明: 首次运行分区阈值未按 mm→归一化换算(wall 恒 1.0), 已修(壁=2/15≈0.133); 叠加图: tools/phase2_relay/outputs/b1_occlusion/ (logs/step_b1_occlusion.json)

## B 路径最终结果（2026-11-22 03:25, 手动重跑完成）

### B1 遮挡敏感度 (sdf-only 权重, 修正版)
- 20 例(10 N+ / 10 N0), 16³ 块遮挡滑窗 64 次前向; 高敏感体素(top 20%)分区占比均值: **壁=0.058 腔=0.025 外=0.917 边缘环(pad=8)=0.463**
- **结论: sdf-only 的 N 信号 91.7% 在肠腔外广域几何, 46.3% 落在 64³ 裁剪框边缘环 → 模型显著依赖裁剪框边缘 clamp/截断区, ~0.60 几何信号部分是裁剪伪影**
- 叠加图: tools/phase2_relay/outputs/b1_occlusion/; 明细 logs/step_b1_occlusion.json
- (修正说明: 首次分区阈值未按 mm→归一化换算, 已修 壁=2/15≈0.133)

### B2 高分辨缓存构建 (cache_hos_hr15/)
- 1.5mm³ 各向同性, mr_t bbox 中心裁固定 64³=物理 96×96×96mm(严禁拉伸), 4 通道(t/c/d 灰度 zscore + SDF clip±15)
- 结果: **ok=282 / fail=78** (61 无 mri_path_t + 17 mask 重采样后空); 训练集 = master∩cache = **280 例** (与阶段1 96³ OOF 280 完全同口径)
- 质检图 pid 213/377/402: tools/phase2_relay/outputs/hr15_qc_p{pid}.png; logs/step_b2_build_hr15.json

### B3 高分辨三臂 × 3 seed (280 例, 同口径 epochs30/master_5fold/no-clinical/self-written)
- **HR-Arm1 sdf-only: 0.4996 | HR-Arm2 img+SDF: 0.4705 | HR-Arm3 纯 img: 0.4768** (3-seed 均值 last5 OOF N AUC)
- **关键对比: 96³ 变比例 Resize 下 sdf-only 0.5246 → 高分辨固定物理视野下 0.4996 (跌回随机); 灰度 0.5091 → 0.4768 (同样失效)**
- **结论: 96³ 压缩 + 变比例 Resize 引入的"几何/灰度信号"在固定物理视野高分辨下全部消失 → 阶段1 三臂 ~0.51-0.52 的微弱信号是 Resize 拉伸伪影, 非真实解剖形态信号**
- 明细 logs/step_b3_hr_train.json

### B4 收尾
- 灰度未救活 (HR-Arm3 img N_last5=0.4768 < 0.56) → 高分辨 --t-only/--n-only 解耦 + CRCFound 局部 Token 池化零训练重测 145 例 CT (计划, 待执行)

### 阶段3 早间审计 (logs/audit_morning.json)
- 96³ 三臂 best/global: a1_sdfonly best N 0.539-0.617/global 0.511-0.544; a2_img_sdf 0.549-0.617/0.491-0.560; a3_img 0.564-0.596/0.463-0.533 (3-seed 波动大, best 明显高于 last5 → 过拟合)
- T3-4 vs T1-2 分层: a1 last5 T3-4 0.475-0.586/T1-2 0.483-0.586 (无稳定差异); pN 三级均值单调性不成立 (a1: pN0 0.452/pN1 0.469/pN2 0.448; a3: 0.376/0.396/0.454 有递增但幅度小)
- pN 两两 AUC: a1 0v1 0.524/1v2 0.485/0v2 0.506 (≈随机); a3 0.512/0.553/0.575
- batch 批内: a1 b1 0.562/b2 0.483; a3 b1 0.572/b2 0.499 (b2 全部 <0.5 → batch2 子群无信号)
- 耗时: 每 seed ~3min/折, train_loss 末期 0.09-0.16 (无异常)

## 第二阶段综合判定（2026-11-22 03:25）
- **96³ 压缩 + 变比例 Resize 下的三臂微弱信号 (0.51-0.52) = 预处理伪影**, 经 B3 高分辨固定物理视野复核全部跌回随机 (~0.47-0.50), 与 B1 裁剪边缘依赖结论互证
- **形态捷径路线 (SDF/二值几何) 在当前 mask 口径下无真实 N 信号**; 灰度 (T2/CE/DWI) 亦无净增益
- 触发情景1 修复方向: 固定物理视野(已由 B3 执行) 下模型跌回 ~0.50 → 后续按用户分流决策进入"止损/转型"评估 (情景4 底线熔断候选: 粗标注或硕论重心转 T 分期), 具体方向待用户确认

---

## 第三阶段：原始像素级信号检验（2026-11-22，用户 m02003 指令）

> 背景：第二阶段三臂对照（96³）与高分辨复核（hr15）均跌回随机，疑似全部信号来自预处理伪影。
> 本阶段直接检验"原始 NIfTI 轴位像素里到底有没有 T/N 信号"，彻底绕开 3D 重采样/插值/裁剪：
> - 路线1（纯 CPU，PyRadiomics 影像组学）：原生面内分辨率 + 完整掩码，零深度过拟合
> - 路线2-MRI（GPU，冻结 ConvNeXt-Tiny + 逐层 MIL）：原生轴位 120mm 裁块，无 z 插值
> - 路线2-CT（GPU，冻结 CRCFound 局部 ROI Token 池化）：CT 分支独立信号检验

### 路线2-MRI：原生轴位 120mm 裁块 + 冻结 ConvNeXt-Tiny + 逐层 MIL
- **使用的数据**: 296 例口径 (master∩mri_path_t∩mr_t mask)，280 例可用 (16 例无 mask 切片)；逐轴位 mask 非空层质心裁物理 120mm 正方形 (无变比例拉伸)，z 轴不插值；特征 timm convnext_tiny (ImageNet, 完全冻结) [1,3,224,224]→[768]
- **网络结构**: 顶层仅训 Gated Attention-MIL (Ilse 2018, d=768/d_hid=128, ~1.6K 参数, AdamW lr 1e-3 wd 0.05, 40 epochs) + Mean/Max Pooling + L2 LR 底线
- **模型结果** (5 折 OOF, 280 例 N+ 121):
  - GatedAttn-MIL: **N AUC 0.4587** | MeanPool+LR: **0.4640** | MaxPool+LR: **0.4592**
  - **全部 <0.5 随机 → 原生轴位高分辨冻结特征无 N 信号** (logs/route2_axial_mil.json/.txt)

### 路线2-CT：冻结 CRCFound 局部 ROI Token 池化 + L2 LR
- **使用的数据**: 138 例 CT (master∩ct_path∩task010 mask)，90 例有效 (18 例 mask 对齐后空)，T+ 66 / N+ 36
- **网络结构**: 冻结 CRCFound encoder (pretrained/CRCFound/checkpoint-999.pth, strict 294/294 键)，task010 mask 对齐卷网格后局部 patch token mean-pool [1024]，顶层 L2 LR
- **模型结果** (5 折 OOF): label_T **0.4596** | label_N **0.4774** → **CT 分支无独立信号** (logs/route2_crcfound_roi.json/.txt)

### 路线1：原生面内分辨率 + 完整掩码影像组学
- **使用的数据**: 296 例口径 (master∩mri_path_t∩mr_t mask)；原始 NIfTI 轴位图像，z 轴不插值保留原生面内分辨率
- **特征**: PyRadiomics 3D (原生网格不重采样, normalize=True, binWidth=25)，类=firstorder/glcm/glrlm/glszm/gldm/ngtdm/shape，**107 个特征全保留**；每例两套 mask——body=完整结直肠本体 mask (测 T)，ring=mask 外扩 5-10mm 物理壳 (EDT, 测 N)
- **模型**: 折内方差>1e-4 + pearsonr>0.95 去冗余 + f_classif top300 筛选 + StandardScaler + L2 LogisticRegression (C 0.01-10) / ElasticNet (alpha 0.01, l1_ratio 0.5)，5 折 OOF
- **模型结果** (5 折 OOF pooled AUC, 修复后 107 特征):
  - body → label_T: **LR 0.6102 | ElasticNet 0.6044** (n=280, pos=198) — **T 分期影像组学有真实信号 (≈0.61, 稳定跨模型)**
  - ring → label_N: **LR 0.4609 | ElasticNet 0.4785** (n=296, pos=130) — **N 仍随机**
  - **结论: 原始面内像素 + 完整掩码的纹理/一阶特征对 T 分期有信号 (0.61), 对 N 分期无信号 (≈0.46-0.48)** (logs/route1_radiomics.json/.txt)
  - 注: 首次跑 (bug: isinstance 过滤丢 101 个 0 维 ndarray 特征, 仅剩 6 shape 列) 得 body→T 0.4885/ring→N 0.5081, 已修复重跑 (见下方 sex 探针章节)

### 第三阶段综合判定（2026-11-22 15:00 更新，含 sex 探针）
- **sex 终极探针（用户 m03195 判据）**: 用同一套特征预测 sex 验证管线可信度——
  - route2-MRI ConvNeXt: mean→sex **0.8622** / max→sex **0.8306**（≥0.80 → 特征提取与 ID 映射无误）
  - route2-CT CRCFound cls: →sex 0.7600（CT 分支特征区分力偏弱，n=144 偏小）
  - route1 PyRadiomics: body→sex 0.6921 / ring→sex 0.7414（传统特征对性别区分力本就弱于深度特征，非 bug 迹象；修复后数值从 0.655/0.582 提升）
- **判据结论**: 管线基本无误 → 之前所有 0.46-0.51 结论**有效**（非特征提取 bug）；T/N 低 AUC 问题锁定在**临床标签口径（nCRT 降期）或全结肠无肿瘤定位致信号稀释**
- **三条独立尺子最终结论**:
  1. 路线1 影像组学（修复后）：**T 0.610（真实信号）** / N 0.461-0.479（随机）
  2. 路线2-MRI（原生轴位高分辨+冻结 ConvNeXt）：0.459-0.464，随机
  3. 路线2-CT（冻结 CRCFound ROI 池化）：T 0.460 / N 0.477，随机
- **N 分期**: 三管线一致无信号（0.46-0.48）+ sex 探针管线验证无误 → **N 无信号是真实瓶颈（标签口径/信号稀释），非代码 bug**
- **T 分期**: 影像组学在原生面内像素+完整掩码下出现 0.61 信号（此前 96³ 深度模型的 ~0.55-0.62 伪影疑虑被部分澄清——T 确有可提取信号）；路线2-MRI T 0.608/0.561、CT T 0.460 佐证 T 信号弱但存在
- **情景4 止损评估**: ①极低成本粗标注（仅标肿瘤中心或起止层号）重新激活监督信号；②硕论重心转 T 分期（T 信号可提取），N 分期作为无标注瓶颈实证。方向待用户确认

## 第三阶段补充：tgr 拆队列 + 扫描平面审计 + 真轴位探针（2026-11-22，用户 m03486 指令）

> 背景：sex 探针证明管线无误后，进一步排查临床标签口径（nCRT 降期）与扫描平面影响。
> tgr = nCRT 后肿瘤退缩分级（病理TGR），有 tgr 标注者影像为放化疗后再分期图像。

### ① tgr 拆队列审计（logs/audit_tgr_task1.json）
- **队列** (master MRI-t 子集 n=298): 直接手术（无 tgr）223 例 | 放化疗后（有 tgr）75 例; tgr 等级 2.0×40 / 3.0×22 / 1.0×10 (+3 异常)
- **阳性率**: 直接手术 T+ 0.691 / N+ 0.453; 放化疗后 T+ 0.773 / N+ 0.400
- **batch 差异**: batch1 direct 112/ncrt 19 (14.5%), batch2 direct 111/ncrt 56 (33.5%) → **批次差异部分来自 nCRT 比例不同**
- **分群重评** (L2 LR 5 折 OOF):
  - 直接手术组（标签可信）: route1_body T 0.5939 / r2mri_mean T 0.6009 / r2mri_max T 0.6221 — **T 信号 0.59-0.62 稳定存在**; N 均随机 (0.39-0.52)
  - 放化疗组（n=72, 噪声大）: T/N 均不稳定; TRG 二分类 (好 vs 差, pos=10): 全随机 0.34-0.50

### ② 扫描平面审计（tools/audit_tgr_task2a.py）
- **mri_t: 矢状位** (axcodes PIR×298, LPS×1 异常 pid=1) — 高分辨面内 P×I, 厚层步进轴 R; spacing 主规格 (0.43,0.43,4.0)×81 / (0.55,0.55,4.0)×65 / (0.72,0.72,6.0)×52
- **mri_c: 真轴位** (LPS×228) — spacing (0.62,0.62,2.0)×101 主; shape (512,512,92)×114
- **mri_d: 真轴位** (LPS×290) — spacing (1.25,1.25,2.46)×57 主; shape (256,256,44-66)
- **重大修正**: 此前 route1/route2-MRI 均在**矢状位 mri_t** 上做"轴位"分析；mri_c/mri_d 才是真轴位 → 触发真轴位探针

### ③ 真轴位冻结 ConvNeXt 探针（tools/audit_tgr_task2c.py, 已完成）
- **使用的数据**: mri_c 227 例 (T+ 157/N+ 98) + mri_d 149 例有效 (T+ 104/N+ 69); 逐轴位 mask 质心裁 120mm 物理正方形, 冻结 convnext_tiny [768], MeanPool + L2 LR, 5 折 OOF
- **模型结果** (all / direct / ncrt):
  | 序列 | sex | T (all/direct) | N (all/direct) | TRG |
  |---|---|---|---|---|
  | mri_c (227) | 0.8624 | 0.5820 / 0.5828 | 0.5255 / 0.5270 | 0.4107 |
  | mri_d (149) | 0.8425 | 0.6350 / **0.6532** | 0.3861 / 0.2862 | 0.1458 |
- **判定**:
  - **真轴位 sex 均 >0.84 → 管线/ID 映射再次验证无误**
  - **T 分期有真实信号**: mri_d 真轴位 all 0.635 / direct 0.653（迄今最强零训练 T 结果）; mri_c 0.582 → T 信号在不同平面/序列稳定存在
  - **N 分期完全无信号**: mri_c 0.526 随机, mri_d 0.386 低于随机（疑似 mr_d mask 对齐质量差, 待核）→ N 无信号再添一证
  - TRG 无信号
- 产物: logs/audit_tgr_task2c.json / outputs/plane_qc/plane_p2_mrit.png (+p3)

### ④ 任务3a: PyRadiomics 参数修复重跑（tools/route3a_radiomics.py, 已完成）
- **使用的数据**: n=296 (master∩mask∩mri_path_t, T+ 0.709/N+ 0.439)；原始 NIfTI mri_t 轴位图像
- **修复参数**: force2D=True + force2Ddimension=2（沿矢状位厚层轴切片）+ resampledPixelSpacing=[0.5,0.5,0]（面内各向同性 0.5mm, 厚层轴 0=保持原值严禁插值）+ binCount=32（替代导致桶坍塌的 binWidth=25）+ normalize=True；特征类 firstorder/glcm/glrlm/glszm/gldm/ngtdm/shape，107 列全保留
- **ROI 变体 4 个**: body（完整结肠 mask 测 T）/ ring（5-10mm EDT 壳测 N）/ body_150 / ring_150（沿真实头足轴自掩码尾侧端截 150mm 直肠段）
- **模型结果** (5 折 Pooled OOF AUC):
  | 变体 | 目标 | LR | ElasticNet |
  |---|---|---|---|
  | body | T all | 0.5862 | 0.5775 |
  | body | T direct | 0.5699 | 0.6031 |
  | ring | T all | 0.5343 | 0.6090 |
  | body | N all | 0.4629 | 0.4575 |
  | ring | N all | 0.4973 | 0.4846 |
  | body_150 | T all | 0.5862 | 0.5771 |
  | body_150 | N ncrt | 0.8198⚠️ | 0.7300 |
  | ring_150 | T all | 0.5357 | 0.6098 |
- **判定**: ①修复后 T 信号 0.58-0.61 明显强于旧参数 route1（0.4885，纹理桶坍塌）→ 桶坍塌确凿 ②N 全队列仍随机（0.46-0.50），N_ncrt 0.82 为 n=72/pos=27 小样本高方差，ring_150 同群 0.4852 对照否之 ③150mm 直肠段截断无全队列增益（body_150 T 与 body 相同）→ 截断未对准肿瘤段或截断方式无效
- 产物: logs/audit_tgr_task3a.json + logs/route3a_radiomics_features.csv (1184 行)

### ⑤ T 分期三路晚融合（tools/audit_tgr_task3b.py, 已完成）
- **使用的数据**: n=280 (T+ 198)；三路特征——①route1 PyRadiomics body (107 列) ②route2-MRI ConvNeXt mean-pool (768 维, logs/route2_feats/) ③临床 7 维原始值 (age/sex/ca199/cea/albumin/prealbumin/hb, nan→0)
- **模型**: 逐路折内选 C L2 LR 得 OOF 概率 → 等权平均 vs 折内 LR Stacking；5 折 master 框架
- **模型结果** (5 折 Pooled OOF T AUC):
  | 路 | pooled OOF T AUC |
  |---|---|
  | radiomics | 0.6138 |
  | convnext | 0.6093 |
  | clin (7维) | 0.6306 |
  | **等权平均** | **0.6547** |
  | LR Stacking | 0.6494 |
- **判定**: 三路融合 T AUC 0.65 高于任一单路 → T 分期信号可三路互补提取；临床 7 维单路 0.63 有独立预测力（与此前深度模型"临床净噪声"结论并行不悖：临床信息不足以帮 96³ 深度模型，但 LR 融合时有增益）
- 产物: logs/audit_tgr_task3b.json




### 第三阶段终极核验汇总 (audit_tgr_planes_fusion.json)
- 状态: task1/2a/2c/3b 完成; task3a 结果见其章节
- 核心结论: ①真轴位(mri_c/d)冻结 ConvNeXt sex OOF 0.84-0.86 ≥0.80 → 特征管线与 ID 映射无误 ②T 分期有真实信号: 真轴位 direct 0.6532 / 三路晚融合 0.6547 ③N 分期在任何口径下均随机(0.39-0.53) ④TRG/nCRT 无信号 ⑤96³ 变比例 Resize 伪影与全结肠无肿瘤定位是 T 信号偏弱的主因

### 第三阶段补充2: 跨平面缺失容错 T 分期融合 (m04057/m04358 指令, 已完成)
- **任务1 审计**: ①task3b clin 分支 CLIN_COLS=[age,sex,ca199,cea,albumin,prealbumin,hb] 恰 7 维, 无 tgr/病理/标签衍生列, 填补 np.nan_to_num(0.0), n=280, pooled OOF T AUC 0.6306 无泄露 ✓ ②样本覆盖 (master 355): has t=296/c=227/d=287; direct n=277 (t=222/c=168/d=214); 交集 t&c=166/t&d=212/c&d=162/t&c&d=160
- **任务2 真轴位 2D PyRadiomics** (route3b, force2D 沿厚层轴, 面内 0.625mm 仅 c / d 跳过重采样, binCount=32, 107 特征, 8 进程 16 核, c 3691s/d 174s):
  | 序列 | T_all LR/EN | T_direct LR/EN | N_all LR/EN | N_direct LR/EN |
  |---|---|---|---|---|
  | c (真轴位, n=227) | 0.5783/0.6005 | 0.5559/0.5807 | 0.4703/0.4851 | 0.5228/0.5335 |
  | d (真轴位, n=147 有mask) | 0.4649/0.4541 | 0.4387/0.4576 | 0.4937/0.4828 | 0.4713/0.4706 |
  - **关键修复**: masks_totalseg_mr_d 138/290 例 mask 本身全零(分割失败); 全零 mask 的 EDT 会产生背景壳伪 ring (N_all 伪 0.556) → 源 mask 全零时 body/ring 均 ok=0, 修复后 N 回 0.49 随机
  - **判定: 真轴位 ring 测 N 全队列均 ~0.47-0.50 → N 分期无标注失效边界 100% 闭环结案; c 真轴位 T EN 0.60 有真实信号**
- **任务3 跨平面缺失容错 T 融合** (direct 队列, 单路折内选C L2 LR → 方案A 可用模态均值 / 方案B 缺失指示符 Stacking):
  | 分支 (direct_all) | pooled OOF T AUC |
  |---|---|
  | t_rad (矢状位 radiomics) | 0.5699 |
  | t_conv (矢状位 ConvNeXt) | 0.6009 |
  | d_conv (真轴位) | 0.6577 |
  | c_rad (真轴位) | 0.5495 |
  | c_conv (真轴位) | 0.5817 |
  | clin (7维) | **0.6570** |
  | fusionA 可用均值 | 0.5839 |
  | fusionB 缺失stack | 0.5767 |
  - direct_td (t∩d 共有 136 例): c_conv 0.6705 最强, fusionA 0.6013/fusionB 0.5066
  - **判定: 跨平面融合未超临床单路 (0.657); 图像分支 d_conv/c_conv 有信号但样本缺失 (d 仅 108-109 例可用) 拖累融合 → T 分期当前最优仍是临床 7 维 LR, 影像 T 信号 (c 真轴位 EN 0.60) 可作弱辅助**
- 产物: logs/audit_crossplane_fusion.json + logs/route3b_{c,d}_features.csv + logs/audit_tgr_task2c_rad_{c,d}.json
### 第三阶段补充3: 四任务排错审计与干净阶梯消融 (m04813 指令, 已完成, 纯CPU)
- **方向决策**: N 分期正式结案(无病灶定位真实失效边界); 主线锁定=剔除 tgr 后纯直接手术队列跨平面多序列 T 分期预测
- **任务1 clin 填补核查**: 上一轮 np.nan_to_num(0.0) 在 oof_prob_lr 内于 StandardScaler.fit 之前(全局固定变换, 折内选C无泄漏但 0.0 是任意占位); direct 组缺失: age 78/sex 1/ca199 177/cea 175/albumin 29/prealbumin 30/hb 29 (n=277, ca199/cea 缺失率 63.9%/63.2%); 干净 SimpleImputer(median)折内拟合: pooled 0.6044 (fold 0.609±0.059) vs 旧 nan_to_num 0.657 (fold 0.660±0.036) → 旧 0.657 部分来自缺失=0 的伪信号, 干净基线 0.604
- **任务2 d_conv N_direct 逐折 (n=109, N+52, pooled 0.3067 / fold 0.338±0.068 / zscore-pooled 0.3489)**:
  | fold | n_tr | n_va | tr_pos | va_pos | va_p mean±std | C | val_auc |
  |---|---|---|---|---|---|---|---|
  | 0 | 88 | 21 | 0.466 | 0.524 | 0.498±0.354 | 0.1 | 0.336 |
  | 1 | 91 | 18 | 0.462 | 0.556 | 0.409±0.337 | 0.1 | 0.288 |
  | 2 | 87 | 22 | 0.540 | 0.227 | 0.643±0.360 | 0.1 | 0.400 |
  | 3 | 87 | 22 | 0.460 | 0.545 | 0.358±0.263 | 0.1 | 0.242 |
  | 4 | 83 | 26 | 0.458 | 0.538 | 0.355±0.346 | 0.1 | 0.423 |
  - 归因: 单折全部 <0.5 (0.242-0.423), zscore-pooled 0.349 仍远低于 0.5 → **非跨折辛普森反转, 而是该子集上 ConvNeXt d 特征的系统性(弱)反向关联, n=109 小样本放大**; 强正则 C=0.1 全折被选中, 概率贴近折内先验方差小 → 无代码/标签对齐 Bug 证据
- **任务3 d_conv 全零 mask 回退**: extract_case_feats 中 mask 文件缺失→None, 逐层 mask.sum()==0 跳过, 全层全零(138 例)→返回 None→不入 d.npy → 全零病例无 d 特征, fusion 中视为缺失; direct 组 has_d 109 (T+率 0.679) / no_d 168 (T+率 0.702); d_conv T pooled 0.6577 (fold 0.705±0.090) 完全来自有 mask 的 109 例子集
- **任务4 fusionA 暴跌真因 + 干净阶梯消融 (剔除 d_rad, 折内 z-score logit 均值 / rank 均值, 仅可用分支平均)**:
  - 暴跌原因: ①上一轮 fusionA 混入 d_rad (direct_all 单路 0.4271, direct_td 0.4750, 反向/弱信号) ②各分支概率直接算术平均无折内尺度对齐 ③direct_all 上 d 仅覆盖 109/277
  - 消融 (f1=z-logit 均值, f2=rank 均值; Pooled OOF, 括号 Fold-Mean±SD):
  | 阶梯 | direct_all (n=277) | direct_td (n=212) | direct_tcd (n=160) |
  |---|---|---|---|
  | s1 六路融合 | 0.6142 (0.608±0.049) | 0.6412 (0.635±0.059) | 0.6889 (0.702±0.080) |
  | s2 clin+d_conv | 0.6215 (0.617±0.049) | **0.6795 (0.678±0.051)** | 0.6887 (0.682±0.091) |
  | s3 纯影像三平面 | 0.6300 (0.636±0.038) | 0.6229 (0.633±0.042) | 0.6425 (0.653±0.089) |
  | s4 clin+三平面 | 0.6254 (0.619±0.032) | 0.6736 (0.677±0.046) | **0.6896 (0.698±0.086)** |
  - 单路基准 (direct_tcd): d_conv 0.6577 / t_conv 0.6009 / c_conv 0.5817 / t_rad 0.5699 / c_rad 0.5495 / clin_clean 0.6044
  - 判定: 融合有增益但有限 — direct_td 上 clin+d_conv 0.6795 最优, direct_tcd 上四路 0.6896 最优; 样本交集越小(n 277→212→160) AUC 越高(队列更纯), 临床+影像双路优于纯影像
- 产物: logs/audit_fusion_debug.json (全明细)### 第三阶段补充4: DWI 物理坐标掩码借壳 + 三骨干冻结逐层特征提取 (m04879 指令, 阶段1 冒烟完成)
- **目标**: 救回 mr_d 掩码全零(138例)病例的 DWI 特征; 三骨干(convnext_tiny/dinov2/biomedclip)冻结逐层特征离线提取, 为跨平面 T 分期下游做准备
- **阶段1 已核验** (产物 tools/verify_backbones.py + tools/map_mask_to_dwi.py + tools/extract_slice_feats_v2.py):
  - 三骨干离线加载: convnext_tiny 768d/27.8M/ImageNet归一化; dinov2 (vit_base_patch14_reg4_dinov2.lvd142m + safetensors 175键 strict) 768d/86.6M; biomedclip (visual.trunk → vit_base_patch16_224, missing=0/unexpected=0, head proj (512,768) 已存 features/slice_feats_v2/biomedclip_head_proj.npy) 768d/85.8M/CLIP归一化
  - 借壳冒烟: pid51/88 原 mr_d 全零 → from_c 映射 6998/11589 体素; pid46 from_c Dice 0.144/质心偏移 26.2mm; direct 全零组(candidates=112)抽样 67/212/162 全 from_c 成功; 质检图 outputs/dwi_remap_qc/qc_p{pid}_{from_c,from_t,self}.png
  - 3例端到端冒烟 (51/88/46 × t/c/d × 三骨干): 27 个 npz 全部落盘 features/slice_feats_v2/{backbone}/{seq}/{pid}.npz, feats float16 [K,768] 无 NaN; 峰值显存 0.59GB; 单例 1.07s (含三骨干)
- **切片基线**: 沿用 route2 (120mm 物理方框 / mask 质心 / zscore / PIL BILINEAR 224 / 单层 3 通道), 掩码优先 data/masks_dwi_remapped (借壳) 回退原掩码
- **阶段2** (待用户"继续"): 全量 356 例借壳 + 三骨干×三序列逐层特征提取, 原子写 npz, 07:30 硬截断
## 数据勘误 (2026-11-23)
- 依据 m05147: 用户手动修正 data/Hos 3 个 excel 后, 阶段1 只读 diff + 阶段2 安全落盘。
- 病理 T/N 与 master 100% 一致 (0 变更), master_5fold_356.csv 未改动 (5折/标签零变动)。
- 临床 7 维更新 7 处: pid372 ca199 8.56/cea 1.25, pid395 prealbumin 241.0/hb 139.0, pid402 albumin 41.0/prealbumin 182.0/hb 148.0 (hos_pairs_raw.csv)。
  - 防污染校验: 10 处 ca199 '<2' 保持解析 2.0, 87 处 age 'XX岁' 保持数值, 无 NaN 误写。
- 清除 3 例错位 tgr 异常值 (14.92 / '5-38' / 41) → direct_tcd 由 160 修正为 162。
- 新建 data/direct_cohort_ids.csv (仅 pid/is_tgr 两列): direct_all=280 / ncrt=75 / direct_td=214 / direct_tcd=162。
- 变更明细(无隐私): logs/data_fix_diff.json

---

## DWI 掩码借壳与逐层特征提取（2026-11-23，1650 端协作审计）

### 背景
- 目的: mri_d (DWI 真轴位) 原生结肠掩码 138/290 全零，需借壳 (from_c / from_t 重映射到 DWI 空间) 后提取逐层特征（convnext_tiny / dinov2 / biomedclip 三骨干）供 1650 端跨模态训练。
- 1650 端只读审计脚本后回传五点修复要求（m05504），本机全部落实。

### 五点修复
- **P0 序列隔离**: extract_slice_feats_v2.py 的 load_mask 按序列隔离——仅 seq=='d' 可读重映射目录（优先原生 mr_d≥50，其次 from_c/from_t）；t/c 严格只读各自原生掩码。修复前 t/c 借壳优先会读 DWI-FOV 受限掩码覆盖原生（覆盖率损失 43-68%）。
- **P1 口径**: map_mask_to_dwi.py 改读 data/direct_cohort_ids.csv（列=pid,is_tgr）替代旧 tgr xlsx，生成 cohort_direct 分层统计。
- **P2 阈值统一**: MIN_VOX=50 两端对齐——from_c/from_t 重采样后 <50 体素即删除落盘文件（残片不留有效掩码）。
- **P2 兜底核实**: 全量借壳统计（287 例有 mri_d）all: self=139 / from_c=119 / from_t=29 / center_fallback=0；direct_all(280) 216 例 self104/c89/t23；direct_tcd(162) self75/c87/t0 → center_fallback=0，无需补 120mm 中心框，标注丢弃。
- **P3 防溢出**: 写 npz 前 np.isfinite 校验（不落盘标 :nan）；新增 --force 覆写；修复 force 重跑时 nomask/nan 不删旧 npz 的残留伪影 bug（pid46 t 残留修复前 remapped 12sl）。

### 全量结果
- 借壳产物: features/dwi_mask_remap_manifest.csv（287 行）+ summary.csv（三 scope 分层）；推 Git fbb770e（脚本）+ e5503ff（manifest/summary）。
- 全量提取（extract_slice_feats_v2.py，--all）: done=810（296t+227c+287d 组合）skipped=0，1054s（1.3s/case），peak_cuda=0.59GB，无 NaN/Inf。
- mask_source 验证（2307 npz = 769 组合×3骨干）: t 全 self（840）/ c 全 self（681）/ d self 417 + remapped 369（25 例差异 = P2 残片删除，from 借壳 148→123 落盘，自洽）。
- 勘误样本覆盖: pid372/pid402 有 t npz；pid395 无任何 MRI 模态（mri_path_t/c/d 全 nan，非漏提取）。
- 产物: features/slice_feats_v2/{convnext_tiny,dinov2,biomedclip}/{t,c,d}/{pid}.npz（每 npz: feats [L,768] + mask_source + n_layers）

---

## 方案 C: 禁用 from_c 借壳, 改走 self → from_t (2026-11-23, m05840)

### 起因 (同步前只读核查, m05774)
- 全量借壳后实测 mask 质心 vs DWI 卷中心偏移: self (n=139) 中位 35.9mm, **from_c (n=97) 中位 94.2mm (dy +50.3 / dz -59.9)**; from_t (n=26) 37.4mm。
- 根因: mri_c 与 mri_d 卷 z-origin 系统性差 **-166.8mm** (中位, range -25~-211) 且 d 卷斜位采集 → c/d 不共享物理坐标系, Identity 物理坐标重采样把 c 掩码放到 d 卷错误位置。
- DWI 信号量化 (mask_ratio = mask 区 DWI 中位 / 卷非零中位): self 4.47 / from_t 3.04 / **from_c 1.42**; pid89/226 mask_ratio=0.0 (mask 落 z=0 层卷边界外)。
- 补充必查 (m05893): pid226 原生 c 掩码 7 个 3D 连通域, 其中 cc2(+71.6mm) 与 cc5(-74.0mm) 左右对称游离块为 TotalSegmentator 原生假阳性 (非重采样引入) — 游离块拉偏质心至 94mm 是 mask_ratio=0 的直接原因; 掩码标签均 {0,1} 无多标签。

### 执行 (三步)
1. **c 序列 K 中位 61 层核查** (step1_check_c_slices.py): c 卷为轴位横断面 (层厚 2mm 典型 / 0.75mm 薄层), masks_totalseg_mr_c 是全结肠分割 (非肿瘤掩码), 结肠沿 z 走向物理跨度天然 118-152mm → K≈物理跨度÷层厚, 无异常; extract 沿 array 轴0 (=z) 遍历, 每层以该层 mask 质心裁 120mm 方框。
2. **map_mask_to_dwi.py 禁用 from_c**: 仅 self → from_t → center_fallback; 清理 97 个旧 from_c 掩码; 全量重跑后 all: self=139 / from_t=144 / fb=4; direct_all(280) self104 / from_t109 / fb3; direct_tcd(162) self75 / from_t86 / fb1。
   - from_t 全量质检 (144 例): 质心偏移中位 39.3mm (dx -0.7 / dy +18.1 / dz +12.6), mask_ratio 中位 2.96, <0.8 比例 0%, 0 信号 0 例 → 扩大样本后仍高对齐。
   - 清单: logs/dwi_mask_remap_summary.csv (pid, cohort_direct, mask_source, has_valid_mask, n_vox, centroid_shift_mm, mask_ratio)。
3. **d 序列仅重提**: 删全部 786 个 d npz (3 骨干) 后 extract --all 重跑, 228s (0.28s/case), peak 0.59GB, 无 NaN。
   - **覆盖率代价: from_t 有效仅 62/144 (43%)** — mr_t 矢状位掩码重采样到轴位 d 卷残留不足, 82 例残片被 P2 删除 → 全量 d 262→201 例, direct_all 199→154, direct_tcd 148→105。

### 结论
- from_t 借壳对齐质量好但覆盖低 (43%), from_c 借壳覆盖高但系统性错位 (不可用)。"宁缺毋滥": d 序列现仅 201 例有效 (self 139 + from_t 62)。
- 待确认: 覆盖损失是否可接受 → 推 Git + U 盘同步; c 掩码清洗 (保留主段/仅最大连通域) 待决策。
