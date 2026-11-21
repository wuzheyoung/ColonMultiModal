# 仓库盘点与整理方案（只读盘点 · 未执行任何删除/移动）

> 盘点时间：2026-11-21 15:03（本地）｜范围：`d:\PyFile\ColonMultiModal` 全仓｜口径：体积按 GiB（1024³）计。
> 约束：本轮仅盘点与出方案；未删除、移动、重命名、复制任何文件，未写 CSV，未训练/推理。
> 原始数据：`work/inventory.json`、`work/tables.md`、`work/empty_dirs.txt`、`work/zero_byte.txt`、`work/pyc_list.txt`；脚本：`work/inventory.py`、`work/make_tables.py`。

## 0. 结论摘要

| 事实 | 数值 |
|---|---|
| 全仓文件数 | 37,796 |
| 全仓体积 | 144,622,323,978 B = **134.69 GiB** |
| 前 5 项占比 | **93.1%**（results / data / checkpoints_hos_full / pretrained / archive） |
| 字节级重复组 | 12 组，全部位于 `archive/`；可释放 545.7 MiB |
| 根目录 `_wire_*` / `_smoke_*` 与 archive 的字节级重复 | **0 组** |
| 删除候选天花板（不含需特批的 logs/archive/round3） | **546.8 MiB（0.40%）**，156 个文件 |
| 其中零风险可自动执行 | **0.92 MiB**（`__pycache__`）+ 12 个零字节日志（0 B） |
| 受保护目录（data/results/checkpoints/logs/cache） | 129.99 GiB（96.5%） |

**核心判断**：仓库体积由受保护的影像、缓存、权重与正式结果决定；在不触碰硬约束的前提下，整理只能回收 <0.5% 体积。真正有价值的动作是**结构整理**（归档一次性脚本、扁平化双层 archive、合并同构工具/旧文档），而不是删体积。

## 1. 全仓体积与文件数分布（按顶层项）

| # | 顶层项 | 类型 | 体积 | 文件数 | 占比 |
|---|---|---|---|---|---|
| 1 | `results` | 目录 | 52.73 GB | 1044 | 39.15% |
| 2 | `data` | 目录 | 30.40 GB | 33992 | 22.57% |
| 3 | `checkpoints_hos_full` | 目录 | 17.78 GB | 148 | 13.20% |
| 4 | `pretrained` | 目录 | 16.22 GB | 198 | 12.04% |
| 5 | `archive` | 目录 | 8.28 GB | 243 | 6.14% |
| 6 | `cache_hos` | 目录 | 3.78 GB | 544 | 2.81% |
| 7 | `logs` | 目录 | 3.67 GB | 579 | 2.72% |
| 8 | `cache_crcfound` | 目录 | 1.13 GB | 145 | 0.84% |
| 9 | `checkpoints_hos` | 目录 | 0.64 GB | 2 | 0.48% |
| 10 | `plots_hos_full` | 目录 | 0.06 GB | 65 | 0.04% |
| 11 | `third_party` | 目录 | 2.97 MB | 356 | 0.002% |
| 12 | `.git` | 目录 | 1.70 MB | 257 | 0.001% |
| 13 | `2026-11-21` | 目录 | 1.01 MB | 48 | 0.001% |
| 14 | `tools` | 目录 | 0.63 MB | 79 | 0.0005% |
| 15 | `plots_roi_check_20261120` | 目录 | 0.39 MB | 5 | 0.0003% |
| 16 | `README.md` | 文件 | 0.13 MB | 1 | 0.0001% |
| — | 其余 21 项（docs/models/utils/trainers/config/根脚本等） | — | 0.9 MB | 90 | 0.001% |

### 占用最大的 10 个文件

| # | 文件 | 体积 |
|---|---|---|
| 1 | `pretrained/nnUNet/Task010_Colon.zip` | 4,772.1 MB |
| 2 | `pretrained/CRCFound/checkpoint-999.pth` | 3,829.9 MB |
| 3 | `pretrained/SwinUNETR/ssl_pretrained_weights.pth` | 719.0 MB |
| 4 | `results/round9/run_44_hos_full_s42/best_model_n_hos_full_s42.pth` | 611.4 MB |
| 5 | `results/round9/run_52_hos_full_s42/best_model_hos_full_s42.pth` | 611.4 MB |
| 6 | `results/round9/run_44_hos_full_s42/best_model_hos_full_s42.pth` | 611.4 MB |
| 7 | `results/round9/run_43_hos_full_s42/best_model_hos_full_s42.pth` | 611.4 MB |
| 8 | `results/round9/run_59_hos_full_k2/best_model_hos_full_k2.pth` | 611.4 MB |
| 9 | `results/round9/run_58_hos_full_k1/best_model_hos_full_k1.pth` | 611.4 MB |
| 10 | `results/round9/run_57_hos_full_k0/best_model_hos_full_k0.pth` | 611.4 MB |

说明：`results/round9` 单目录 932 文件 / 52.73 GiB；`checkpoints_hos_full` 148 文件 / 17.78 GiB，主要是 611.4 MB 级双头 ckpt。二者均属硬约束保护区。

## 2. 四类分档

### 2.1 保留（原地不动）

> 保留档按“目录 / 关键入口文件”列示，不逐条展开 3.7 万个文件；逐文件体积见 `work/inventory.json`。

| 项 | 体积 | 文件数 | 说明 |
|---|---|---|---|
| `data/` | 30.40 GiB | 33,992 | 影像 + CSV + mask；按用户要求仅整体统计，不给 `data/Hos` 内部逐病人方案 |
| `results/` | 52.73 GiB | 1,044 | 正式实验产物（round9 的 ckpt/report/OOF） |
| `checkpoints_hos_full/` | 17.78 GiB | 148 | 主模型正式权重 |
| `pretrained/` | 16.22 GiB | 198 | nnUNet / CRCFound / SwinUNETR / TotalSegmentator 预训练权重 |
| `cache_hos/`、`cache_crcfound/` | 4.91 GiB | 689 | 确定性缓存，重生成成本高 |
| `logs/` | 3.67 GiB | 579 | 正式运行日志与归档（含 round3 resnet18 备份，见 2.4-G） |
| `checkpoints_hos/` | 0.64 GiB | 2 | `best_model_2.pth` / `best_model_n_2.pth`，README 明确引用 |
| `train_hos_seeds.py` | 67,697 B | 1 | 主训练入口 |
| `tools/`（47 个 .py） | 0.63 MB | 79 | 当前实验脚本主链路（含本轮新增 `aggregate_roidl_kfold.py`） |
| `models/`、`trainers/`、`data_loader/`、`utils/`、`config/` | 0.40 MB | 54 | 核心代码与配置（`config/hos.py:124` 仍以 `./plots_hos` 为默认输出） |
| `third_party/` | 2.97 MB | 356 | nnUNet 依赖副本；其中 **29 个零字节 `__init__.py` 不可删**（包语义） |
| `docs/FIXES_2026-11-20.md` | 86,039 B | 1 | 硬约束，明确不得列入删除候选 |
| `docs/WORKLOG_2026-11-21.md` | 9,478 B | 1 | 数据侧住院号校验记录，与 FIXES 不同范围，保留 |
| `README.md`、`.gitignore`、`skills/`、`.clinerules`、`.idea/` | 0.18 MB | — | 入口文档与本地配置 |
| `plots_hos_full/`、`plots_roi_check_20261120/` | 58.4 MB | 70 | 正式图与上一轮 ROI 复核证据 |
| `2026-11-21/` | 1.01 MB | 48 | 本轮盘点交付物 + 另一路 `hos-data-inventory` 记录 |
| `plots_hos/`（空目录） | 0 B | 0 | 被 `config/hos.py:124`、`utils/eval_report.py:234` 引用为默认输出目录，**不删** |
| `.git/` | 1.70 MB | 257 | 硬约束 |

### 2.2 合并（同类型可整合）

| # | 对象 | 体积/规模 | 重复或同构证据 | 建议 |
|---|---|---|---|---|
| M1 | `docs/FILE_INVENTORY.md`、`PROJECT_OVERVIEW.md`、`PROGRESS_REPORT_HOS.md`、`OPTIMIZATION_ROADMAP.md` | 12,480 + 8,380 + 9,317 + 5,050 = 35,227 B | 四份均带“已过时（2026-11-20）”横幅；FILE_INVENTORY 自述 2026-09-29 快照；PROGRESS/ROADMAP 同为 2026-10-09 的 154 样本叙事，轮次叙述重叠 | 合并为 `docs/archive/HISTORY_2026-09-29.md`（只读快照），`docs/` 只留 FIXES + WORKLOG + 新索引；**原文件移入 archive，不做内容覆写** |
| M2 | `archive/scripts_legacy/scripts_legacy/` | 4 文件 / 19,889 B（`test_hos_ct.py` 4,134；`train_hos.py` 7,481；`_eval_best.py` 3,203；`_eval_fast.py` 5,071） | 双层同名嵌套；FIXES D10（L169）已记录 | 拍平为 `archive/scripts_legacy/`；**不改文件内容** |
| M3 | `archive/tcga/_archive_tcga/` | 172 文件 / 2,240.3 MiB | 双层嵌套；内含 `data/`、`cga/`、`machine_learning/`、`kfold/` 等完整旧项目 | 拍平为 `archive/tcga/`（仅少一层路径，**不释放体积**）；需同步文档路径 |
| M4 | 工具同构组 A：`tools/_ensemble_eval.py` 6,371 B、`ensemble_seeds.py` 8,889 B、`ensemble_modalities.py` 8,155 B、`ensemble_snapshot.py` 5,104 B | 28,519 B | 同一模式：加载 ckpt 列表 → val 前向 softmax → 平均 → 校准阈值 → 输出 T/N F1；仅“来源”不同（seed / 模态 / 快照） | 合并为 `tools/evaluate_ensemble.py --source seeds\|modalities\|snapshots`；风险中高（README:882-883 有命令），建议先加新脚本并保留旧入口一个版本 |
| M5 | `tools/build_hos_batch2_csv.py` 8,795 B、`build_hos_batch3_csv.py` 7,185 B | 15,980 B | 文件头均声明“与 batch2 一致的管线”，仅批号/输入目录/输出规则不同 | 合并为 `tools/build_hos_batch_csv.py --batch 2\|3`；属数据构建链，需单独验证后再动 |
| M6 | `tools/train_2p5d.py`、`train_mil.py`、`train_roi_dl.py` | 19,213 + 18,187 + 92,736 B | 各自维护 `_bin_metrics` / 阈值校准 / 归档占位等近同代码 | **不建议合并脚本**；抽取 `tools/train_common.py` 去重，降低“改一处漏两处”风险 |
| M7 | 根目录 4 个一次性脚本 vs `archive/` | 11,848 B | 哈希比对：**无字节级副本**；仅语义上同属旧接线/冒烟 | 不做“合并”；按 2.3 归档 |

### 2.3 归档（移入 `archive/`，不必删除）

| # | 路径 | 体积 | 说明 | 建议目标 |
|---|---|---|---|---|
| A1 | `_wire_hos.py` | 1,067 B | 一次性给 `config/hos.py` 接线；已幂等化，无运行时引用 | `archive/scripts_legacy/oneoff_20261120/` |
| A2 | `_wire_mri_lesion.py` | 5,152 B | 一次性给 mri_lesion 接线；同上 | 同上 |
| A3 | `_smoke_mri_lesion.py` | 2,547 B | 单样本冒烟 | 同上 |
| A4 | `_smoke_n_lesion.py` | 3,082 B | 单样本冒烟 | 同上 |
| A5 | `README.md.gbk_backup` | 124,105 B | 旧编码备份，与 README 字节不同；无脚本引用 | `archive/docs/`（或确认后删除，见 2.4-F） |
| A6 | 4 份过时文档（M1） | 35,227 B | 2026-09-29/10-09 快照 | `docs/archive/` |
| A7 | `logs/archive/_tmp/20260929_cleanup/*.ipynb`（4 个） | 11,617 + 5,817 + 7,686 + 11,864 B | 已在归档目录内，非活动 notebook | **无需动作**（已在 archive） |
| A8 | `archive/backup_round5/backup_round5_mri3/` | 544.9 MiB / 2 文件 | round5 权重备份；与 `archive/checkpoints_early/checkpoints_hos/` 字节相同 | 保留为冷备；若执行去重见 2.4-D |

### 2.4 删除候选

> 每条给出“删除后果”与“是否影响脚本可运行性”。未执行任何删除。

| 编号 | 候选 | 规模 | 删除后果 | 影响脚本可运行性 | 建议档位 |
|---|---|---|---|---|---|
| P1 | `__pycache__`（23 个目录，含 `tools/`、根目录、`archive/tcga/.../machine_learning/`） | 131 个 `.pyc` / 969,348 B | Python 首次 import 时自动重建；仅损失首次启动速度 | **否** | 自动（零风险） |
| P2 | 零字节失败日志：`logs/c2_unit_20261101_022222.log`、`logs/roidl_hos_full_s42_20261105_002252.log`、`..._002320.log`、`..._20261119_235052.log`、`logs/archive/round6_pre_reorg/batch2_convert.err.log`、`logs/archive/round6_pre_reorg/hos_20260918_150108.log`、`logs/archive/_tmp/20260929_cleanup/full_out.txt`、`.../r10_out.txt`，以及 `results/round9/run_42_*`、`run_51_*` 下 4 个同名空日志 | 12 个文件 / 0 B | 丢失“曾启动但未产出”的时间戳证据 | **否** | 自动（零体积收益） |
| P3 | 空目录：`.agents`、`.screenshots/.tmp`、`plots_hos`、`pretrained/Task010_Colon`、`pretrained/_nnunet_tmp/{nnUNet_cropped_data,nnUNet_raw_data,preprocessed}`、`.git/objects/{info,pack}`、`.git/refs/tags`、本轮 `repo-cleanup/outputs` | 11 个 / 0 B | `plots_hos` 是脚本默认输出目录；`pretrained/*` 可能是解压/预处理占位；`.git/*` 不可动 | **删除 `plots_hos` 可能影响**未 `makedirs` 的写图脚本（`config/hos.py:124`、`utils/eval_report.py:234`）；其余无影响 | **不建议删**（0 收益、有风险） |
| P4 | `archive/backup_round5/backup_round5_mri3/best_model[_n]_hos_s42.pth` | 2 个 / 571,355,840 B（544.9 MiB） | 删除后 round5 仍有 `archive/checkpoints_early/checkpoints_hos/` 的同哈希副本；README:878/907 与 `docs/FILE_INVENTORY.md:160` 的 `backup_round5_mri3/` 路径失效 | **否**（无脚本引用该路径；评测脚本用 `checkpoints_early`，README:882-883） | 需确认（建议保留副本并改文档，或改用硬链接去重） |
| P5 | `archive/results_early/results_hos/*.png` 中与 `archive/plots_early/plots_hos/` 同哈希的 10 张 | 10 个 / 891,005 B | 早期曲线/混淆矩阵仍可从 `plots_early` 取得；README:31 的目录说明仍在 | **否**（无脚本引用 `results_early` 图片） | 需确认（收益仅 0.85 MiB） |
| P6 | `README.md.gbk_backup` | 1 个 / 124,105 B | 丢失旧编码备份；README 正文不受影响 | **否**（无引用） | 需确认 |
| P7 | `logs/archive/round3_20260928/`（19 文件，resnet18 3-seed + N-best 备份） | 2,440.5 MiB | `docs/FILE_INVENTORY.md:158` 早已标注“B 级可删项，确认后删”；删除后 round3 旧结果基本无法复现 | **否**（无脚本引用） | **需特批**：与“logs/ 正式产物不得删”硬约束冲突，默认不列入执行 |

**删除候选规模汇总**

| 档位 | 文件数 | 体积 | 说明 |
|---|---|---|---|
| 零风险自动 | 131 `.pyc` + 12 空日志 = 143（另 23 个空目录） | 969,348 B = **0.92 MiB** | 不影响任何脚本 |
| 需确认 | 2 ckpt + 10 PNG + 1 gbk 备份 = 13 | 571,355,840 + 891,005 + 124,105 = **545.9 MiB** | 需同步文档路径/指定保留副本 |
| 需特批 | 19 | **2,440.5 MiB** | 违反 logs 硬约束，默认不做 |
| **合计天花板（不含 P7）** | **156** | **546.8 MiB（0.40%）** | — |

## 3. 硬约束（不得列入删除候选）

| 对象 | 原因 |
|---|---|
| `data/` 下全部 CSV（含 `*.bak_*`）与影像 | 训练/评估唯一数据源；另有 `hos_pairs*.bak_20261121_114536` 为当日备份 |
| `cache_hos/`、`cache_crcfound/` | 重生成成本高，且 cache key 与当前代码绑定 |
| `checkpoints_hos*/` | 当前与历史权重；`checkpoints_hos_full` 17.78 GiB |
| `results/` | 正式产物（含 OOF、逐折报告、归档 ckpt） |
| `logs/` 内正式产物 | 运行证据；含 `logs/archive/` 历史备份 |
| `.git/` | 版本库 |
| `docs/FIXES_2026-11-20.md` | 用户明确保护 |
| `third_party/**/__init__.py`（零字节） | 包结构必需，**不得按零字节文件删除** |

## 4. 风险与回退

### 4.1 回退基础

| 事实 | 影响 |
|---|---|
| `.git` 仅 **36 个文件被跟踪**；`git status` 有 85 项变更 | 绝大多数文件未纳入 git；删除/移动**无法用 git 恢复** |
| 唯一提交 | `e9215f5 first commit: ColonMultiModal project` |
| 可回退范围 | 仅 36 个已跟踪代码文件；archive、docs、根脚本、图片均未跟踪 |

**强制前置**：任何删除/移动前，先把 `work/inventory.json` 与目标文件清单复制到仓库外，并对拟删/拟移文件做带 SHA256 的清单；移动动作保留“原路径 → 新路径”映射表。否则不可回退。

### 4.2 会破坏硬编码/文档路径的动作（点名位置）

| 动作 | 失效引用 |
|---|---|
| 删除 `archive/backup_round5/...` | `README.md:878`、`README.md:907`、`docs/FILE_INVENTORY.md:160`（`backup_round5_mri3/`） |
| 删除 `archive/results_early/...png` | `README.md:31`（目录说明）；无脚本引用 |
| 扁平化 `archive/scripts_legacy/scripts_legacy/` | `README.md` 未引用；`docs/PROJECT_OVERVIEW.md:6`、`docs/FILE_INVENTORY.md:8`、`:77`、`docs/FIXES_2026-11-20.md:169` 仍写双层路径 |
| 扁平化 `archive/tcga/_archive_tcga/` | `docs/PROJECT_OVERVIEW.md:7`、`:59`、`docs/FILE_INVENTORY.md:147`、`docs/FIXES_2026-11-20.md:169`；`tools/build_tcga_index.py` 读的是 `data/tcga/`，**不读 archive** |
| 移动根目录 `_wire_*` / `_smoke_*` | 无代码 import/路径引用（grep 仅命中 `docs/FIXES_2026-11-20.md:61`、`:203`）；移动后需更新这两处叙述 |
| 删除 `plots_hos/`（空目录） | `config/hos.py:124`、`utils/eval_report.py:234` 以 `./plots_hos` 为默认输出；可能让未建目录的脚本写图失败 |
| 合并 4 份旧文档 | 文档互相引用：`FILE_INVENTORY.md:4` 指向 `PROJECT_OVERVIEW.md`；`PROGRESS_REPORT_HOS.md:5` 指向 `FILE_INVENTORY.md` 与 `OPTIMIZATION_ROADMAP.md` |
| 合并 ensemble 工具组 | `README.md:882-883` 的 `tools/_ensemble_eval.py` 命令；`docs/FIXES_2026-11-20.md:55-56` 也提到该入口 |
| 删除 `logs/archive/round3_20260928/` | `docs/FILE_INVENTORY.md:140`、`:158`；无脚本引用 |

### 4.3 会让旧结果无法复现的动作

| 动作 | 复现影响 |
|---|---|
| P4 去重（保留 `checkpoints_early` 副本） | 不影响：同 SHA256 另一副本仍在 |
| P6 删除 gbk 备份 | 不影响实验，仅丢失文档旧编码 |
| P7 删除 round3 备份 | **会**：round3 resnet18 结果失去权重与日志证据 |
| 合并/改写旧文档 | 会丢失逐条修改的原始叙述；**必须先归档原文**再建索引 |
| 移动 archive 路径 | 不影响数值复现；影响按文档路径执行的命令 |

## 5. 分阶段落地顺序

| 阶段 | 内容 | 文件数变化 | 释放体积 | 风险 |
|---|---|---|---|---|
| 0 冻结与清单 | 生成 SHA256 清单、原→新路径映射；仓库外备份拟动文件 | 0 | 0 | 无 |
| 1 零风险清理 | 删除 23 个 `__pycache__`（131 `.pyc`）+ 12 个零字节失败日志；`.screenshots/.tmp` 可留可删 | -143（+23 目录） | **0.92 MiB** | 无；首次 import 变慢 |
| 2 归档（移动） | 4 个根一次性脚本 → `archive/scripts_legacy/oneoff_20261120/`；`README.md.gbk_backup` → `archive/docs/`；4 份旧文档 → `docs/archive/` 并建索引 | 10 个文件改路径 | 0（移动） | 低；只需改文档 |
| 3 去重（需确认） | 删除 `backup_round5` 的 2 个 ckpt 副本与 `results_early` 的 10 张重复 PNG；同步 README/FILE_INVENTORY | -12 | **545.8 MiB** | 中；文档路径失效但可改；建议先做硬链接方案评估 |
| 4 结构合并（需确认） | 扁平化 2 处双层 archive；合并 ensemble 4→1、batch CSV 2→1；抽取 train_common | 7 个脚本合并为 2 + 8 个文件改路径 | 0（不释放） | 中高；CLI/文档需回归 |
| 5 特批项（默认不做） | `logs/archive/round3_20260928/` | -19 | 2,440.5 MiB | 高；违反 logs 硬约束且破坏旧结果复现 |

**累计**：阶段 1 = 0.92 MiB；阶段 1+3 = 546.8 MiB（0.40%）；若批准阶段 5 再 +2.38 GiB。阶段 2/4 的收益是**可维护性**，不是体积。

## 6. 盘点事实与缺口

| 事实 | 说明 |
|---|---|
| `data/Hos` 内部 | 按用户要求未展开逐病人方案；`data` 仅整体统计 |
| `run_93_*` 重复目录 | 当前 `results/round9` 下已不存在；FIXES D1 提到的是历史现象 |
| 根脚本与 archive 重复 | 哈希比对无字节级副本；不能以“重复”为由删除 |
| 零字节文件 41 个 | 其中 29 个是 `third_party` 的 `__init__.py`，**必须保留**；真正可删的是 12 个空日志 |
| 重复组 12 组 | 全部在 `archive/`：2 组 ckpt（4 个文件，两两同哈希）+ 10 组 PNG（20 个文件）；可释放 545.7 MiB |
| 本轮新增文件 | 仅 `2026-11-21/repo-cleanup/work/` 下盘点脚本与清单；`outputs/` 下本报告；未触碰其他文件 |
| 不可量化项 | `data/Hos` 内部整理、外部冷存储/压缩策略不在本方案；若允许把 `pretrained` / `results` 冷备到仓库外，体积可降一个数量级，但超出本轮授权 |

## 附录：盘点命令与产物

| 项 | 内容 |
|---|---|
| 体积/文件数 | `work/inventory.py`（`os.walk` + `os.path.getsize`，`.git` 仅统计） |
| 重复检测 | 先按（大小, 文件名）分组，再对组内文件做 SHA256；范围：根文件 + `archive/tools/docs/utils/config/models/data_loader` |
| 特殊清单 | `empty_dirs.txt`、`zero_byte.txt`、`pyc_list.txt` |
| 表格生成 | `work/make_tables.py` → `work/tables.md` |
| 排除 | `data/Hos` 内部逐病人明细；GPU/训练/推理 |
