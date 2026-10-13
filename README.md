# ColonMultiModal

结直肠癌（CRC）多模态分期分类系统，融合 CT、MRI 影像与临床血液指标进行术前病理 T 分期（T1-2 vs T3-4）与 N 分期（N0 vs N+）双任务预测。

---

## 变更日志（CHANGELOG）

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
  - `python tools/_ensemble_eval.py --ckpts checkpoints_hos/best_model_hos_s42.pth,checkpoints_hos/best_model_hos_s0.pth --names s42,s0 --val-csv data/hos_val_s42.csv --share-backbone`

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