# 仓库整理全流程日志（2026-11-21）

> 本文只汇总已发生的动作与已有日志，不补写未执行步骤；时间取自日志/文件时间戳，个别条目为时间范围。

> 口径：体积按 GiB（1024³）；“释放”仅指删除，“移动”单独标注。

## 1. 时间线

| 时间 | 动作 | 文件数 | 字节 | 校验结果 | 执行脚本 | 原始日志 |
|---|---|---|---|---|---|---|
| 15:04:01 | 仓库盘点（全仓体积/文件数、重复、空目录、零字节、pyc） | 37,796 文件 | 144,622,323,978 B（134.69 GiB） | 生成 inventory.json / tables.md；发现 12 组重复、23 个 __pycache__、41 个零字节文件 | work/inventory.py；work/make_tables.py | work/inventory.json（generated 字段） |
| 15:19:43–15:21:29 | results 文本/指标快照导出（共 4 次，权威版本 _152127） | 123 run / 228 fold；5 文件 | 555,765 B | JSON 严格合法；805 个非权重文件 SHA256 入库；失败 run 2（run_93 ring_c/t，保留空行） | work/export_results_snapshot.py | snapshot/export_20261121_152127.log |
| 15:52:49–15:52:53 | 删除 results/ 全部权重（先 dry-run 后执行） | 127 个 .pth | 56,595,615,574 B（52.71 GiB） | 清单与磁盘 127/127 一致；805/805 非权重哈希通过；剩余权重 0 | work/delete_results_weights.py | snapshot/delete_weights_20261121_155253_exec.log |
| 16:28:58–16:28:59 | 删除 checkpoints_hos_full / checkpoints_hos / archive / logs 权重 | 188 个 .pth | 30,231,267,468 B（28.15 GiB） | 非权重 784 个数量/字节/mtime 未变；pretrained 13 文件未变；146 个 ckpt 无法映射 run | work/delete_ckpt_weights.py | snapshot/delete_ckpt_weights_20261121_162858_exec.log |
| 16:29:27–16:29:29 | 阶段 1：删除 .pyc 与零字节失败日志 | 144 个（132 pyc + 12 日志） | 984,013 B | 清理空 __pycache__ 24 个；无错误 | work/cleanup_stages123.py | cleanup_20261121/stage1_20261121_162927.log |
| 16:29:31 | 阶段 2：根目录一次性脚本与 gbk 备份归档 | 5 个 | 135,953 B（移动，未释放） | 移至 archive/cleanup_20261121/20261121_162927/；无 import/运行引用，仅 FIXES 历史提及 | work/cleanup_stages123.py | cleanup_20261121/stage2_20261121_162927.log |
| 16:29:31 | 阶段 3：archive 内重复 PNG 去重 | 11 组 / 11 个 | 920,777 B（移动，未释放） | 保留 plots_early（及 TCGA fold1），移走 results_early（fold2）到 duplicates/；2 组重复 ckpt 已随权重删除跳过 | work/cleanup_stages123.py | cleanup_20261121/stage3_20261121_162927.log |
| 16:30:43 | 路径引用改写（README/docs/tools/config） | 26 条 | 0 B | 26 条改写；残留 26 条全部标注故意保留；待改写 0；py_compile 通过 | work/rewrite_refs.py | cleanup_20261121/rewrite_refs_20261121_163043.log |
| 16:33:48 | 阶段 4：四份历史文档顶部统一横幅（方案 B，不合并） | 4 份文档 | 0 B（仅文本替换） | 旧横幅合并为 4 行；正文 SHA256 4/4 未变；相对链接 25/25 有效 | work/stage4_banners.py | cleanup_20261121/stage4_20261121_163348.log |

## 2. 体积汇总

| 项 | 数值 |
|---|---|
| 初始仓库 | 144,622,323,978 B（134.69 GiB）/ 37,796 文件 |
| 累计删除释放 | **86,827,867,055 B（80.86 GiB）** |
| 当前仓库 | **57,797,953,692 B（53.83 GiB）/ 37,415 文件**（16:36 测量，含本日志与构建脚本） |
| 净减少 | 86,824,370,286 B（80.85 GiB）；与释放差 3,496,769 B 为本轮新增清单/快照/日志/脚本 |
| 移动未释放 | 1,056,730 B（阶段2 135,953 + 阶段3 920,777） |

### 当前分类体积（按顶层）

| 分类 | 体积 | 文件数 |
|---|---|---|
| `data` | 30.402 GiB | 33,992 |
| `pretrained` | 16.215 GiB | 198 |
| `cache_hos` | 3.784 GiB | 544 |
| `archive` | 2.192 GiB | 224 |
| `cache_crcfound` | 1.133 GiB | 145 |
| `plots_hos_full` | 0.057 GiB | 65 |
| `results` | 0.026 GiB | 913 |
| `logs` | 0.011 GiB | 553 |
| `2026-11-21（本轮产物）` | 0.004 GiB | 111 |
| `third_party / .git / docs / tools / 其余` | 约 0.007 GiB | 608 |

## 3. 校验记录汇总

| 校验 | 结果 |
|---|---|
| results 非权重 SHA256 | 805/805 通过；不一致 0；缺失 0 |
| results 删除后残留权重 | 0 |
| checkpoints/archive/logs 非权重 | 784 个，数量/字节/mtime 未变；缺失/新增/变化均 0 |
| pretrained 前后 | 13 文件 / 7,023,152,510 B（6.54 GiB），完全一致；顶层 4 个为 4.62 GiB，另有 9 个 TotalSegmentator 嵌套权重 |
| 快照 JSON | 严格合法（allow_nan=False）；123 run / 228 fold |
| 阶段4 正文 | 4/4 SHA256 一致，正文改动 0 |
| 文档链接 | 25/25 相对链接有效 |

## 4. 未执行与遗留

| 项 | 状态 |
|---|---|
| P0 正式训练（3 seed × 5 fold） | 本轮未启动；仓库内 P0 OOF 为此前产物 |
| Hos 数据内部处置 | 待单独批准（数据侧负责） |
| git 提交 | 未提交；仓库仅 36 个文件受跟踪，当前 81 行未提交状态 |
| 无法映射到 run 的 ckpt | 146 / 188（已随权重删除，manifest 留路径/时间） |
| 文档合并、archive 拍平、脚本同构合并 | 未做（阶段4 方案 B 仅加横幅） |
| 重复 PNG / 阶段2 文件 | 已移至 `archive/cleanup_20261121/`，未删除 |

## 5. 附录：日志与清单索引（按时间排序）

共 51 个文件。

| # | 时间 | 路径 | 字节 | 用途 |
|---|---|---|---|---|
| 1 | 15:04:01 | `2026-11-21/repo-cleanup/work/inventory.json` | 34735 | 盘点原始数据 |
| 2 | 15:05:48 | `2026-11-21/repo-cleanup/work/tables.md` | 3868 | 盘点表格 |
| 3 | 15:07:23 | `2026-11-21/repo-cleanup/outputs/repo-cleanup-plan.md` | 19146 | 仓库整理方案报告 |
| 4 | 15:12:16 | `2026-11-21/repo-cleanup/outputs/results-reusability-plan.md` | 35972 | results 可复用性报告 |
| 5 | 15:19:43 | `2026-11-21/repo-cleanup/outputs/snapshot/run_summary_20261121_151941.csv` | 74307 | run 级指标 CSV |
| 6 | 15:19:43 | `2026-11-21/repo-cleanup/outputs/snapshot/fold_detail_20261121_151941.csv` | 108999 | fold 级指标 CSV |
| 7 | 15:19:43 | `2026-11-21/repo-cleanup/outputs/snapshot/sha256_20261121_151941.csv` | 129574 | 非权重文件 SHA256 清单 |
| 8 | 15:19:43 | `2026-11-21/repo-cleanup/outputs/snapshot/export_20261121_151941.log` | 1036 | 快照导出日志 |
| 9 | 15:19:43 | `2026-11-21/repo-cleanup/outputs/snapshot/snapshot_20261121_151941.json` | 425687 | 快照主文件（run/fold/失败/provenance） |
| 10 | 15:20:27 | `2026-11-21/repo-cleanup/outputs/snapshot/run_summary_20261121_152024.csv` | 76238 | run 级指标 CSV |
| 11 | 15:20:27 | `2026-11-21/repo-cleanup/outputs/snapshot/fold_detail_20261121_152024.csv` | 83781 | fold 级指标 CSV |
| 12 | 15:20:27 | `2026-11-21/repo-cleanup/outputs/snapshot/sha256_20261121_152024.csv` | 129574 | 非权重文件 SHA256 清单 |
| 13 | 15:20:27 | `2026-11-21/repo-cleanup/outputs/snapshot/snapshot_20261121_152024.json` | 361373 | 快照主文件（run/fold/失败/provenance） |
| 14 | 15:20:27 | `2026-11-21/repo-cleanup/outputs/snapshot/export_20261121_152024.log` | 1033 | 快照导出日志 |
| 15 | 15:20:44 | `2026-11-21/repo-cleanup/outputs/snapshot/run_summary_20261121_152042.csv` | 76342 | run 级指标 CSV |
| 16 | 15:20:44 | `2026-11-21/repo-cleanup/outputs/snapshot/fold_detail_20261121_152042.csv` | 85498 | fold 级指标 CSV |
| 17 | 15:20:44 | `2026-11-21/repo-cleanup/outputs/snapshot/sha256_20261121_152042.csv` | 129574 | 非权重文件 SHA256 清单 |
| 18 | 15:20:44 | `2026-11-21/repo-cleanup/outputs/snapshot/export_20261121_152042.log` | 952 | 快照导出日志 |
| 19 | 15:20:44 | `2026-11-21/repo-cleanup/outputs/snapshot/snapshot_20261121_152042.json` | 366401 | 快照主文件（run/fold/失败/provenance） |
| 20 | 15:21:29 | `2026-11-21/repo-cleanup/outputs/snapshot/run_summary_20261121_152127.csv` | 76313 | run 级指标 CSV |
| 21 | 15:21:29 | `2026-11-21/repo-cleanup/outputs/snapshot/fold_detail_20261121_152127.csv` | 52986 | fold 级指标 CSV |
| 22 | 15:21:29 | `2026-11-21/repo-cleanup/outputs/snapshot/sha256_20261121_152127.csv` | 129574 | 非权重文件 SHA256 清单 |
| 23 | 15:21:29 | `2026-11-21/repo-cleanup/outputs/snapshot/snapshot_20261121_152127.json` | 295940 | 快照主文件（run/fold/失败/provenance） |
| 24 | 15:21:29 | `2026-11-21/repo-cleanup/outputs/snapshot/export_20261121_152127.log` | 952 | 快照导出日志 |
| 25 | 15:52:49 | `2026-11-21/repo-cleanup/outputs/snapshot/weights_to_delete_20261121_155249.csv` | 15870 | results 权重删除清单 |
| 26 | 15:52:49 | `2026-11-21/repo-cleanup/outputs/snapshot/delete_weights_20261121_155249_dryrun.log` | 660 | results 权重删除日志 |
| 27 | 15:52:53 | `2026-11-21/repo-cleanup/outputs/snapshot/weights_to_delete_20261121_155253.csv` | 15870 | results 权重删除清单 |
| 28 | 15:52:53 | `2026-11-21/repo-cleanup/outputs/snapshot/delete_weights_20261121_155253_exec.log` | 1151 | results 权重删除日志 |
| 29 | 16:28:01 | `2026-11-21/repo-cleanup/outputs/snapshot/weights_ckpt_archive_logs_to_delete_20261121_162801.csv` | 36650 | 四目录权重删除清单 |
| 30 | 16:28:01 | `2026-11-21/repo-cleanup/outputs/snapshot/nonweights_predelete_20261121_162801.csv` | 63803 | 删除前非权重状态 |
| 31 | 16:28:01 | `2026-11-21/repo-cleanup/outputs/snapshot/delete_ckpt_weights_20261121_162801_dryrun.log` | 848 | 四目录权重删除日志 |
| 32 | 16:28:45 | `2026-11-21/repo-cleanup/outputs/snapshot/weights_ckpt_archive_logs_to_delete_20261121_162845.csv` | 40872 | 四目录权重删除清单 |
| 33 | 16:28:45 | `2026-11-21/repo-cleanup/outputs/snapshot/nonweights_predelete_20261121_162845.csv` | 63803 | 删除前非权重状态 |
| 34 | 16:28:45 | `2026-11-21/repo-cleanup/outputs/snapshot/delete_ckpt_weights_20261121_162845_dryrun.log` | 848 | 四目录权重删除日志 |
| 35 | 16:28:59 | `2026-11-21/repo-cleanup/outputs/snapshot/weights_ckpt_archive_logs_to_delete_20261121_162858.csv` | 40872 | 四目录权重删除清单 |
| 36 | 16:28:59 | `2026-11-21/repo-cleanup/outputs/snapshot/nonweights_predelete_20261121_162858.csv` | 63803 | 删除前非权重状态 |
| 37 | 16:28:59 | `2026-11-21/repo-cleanup/outputs/snapshot/delete_ckpt_weights_20261121_162858_exec.log` | 1276 | 四目录权重删除日志 |
| 38 | 16:29:28 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/stage1_delete_manifest_20261121_162927.csv` | 10554 | 阶段1 删除清单 |
| 39 | 16:29:29 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/stage1_20261121_162927.log` | 429 | 阶段1 日志/git 状态 |
| 40 | 16:29:29 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/stage1_gitstatus_20261121_162927.txt` | 4515 | 阶段1 日志/git 状态 |
| 41 | 16:29:31 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/stage2_gitstatus_20261121_162927.txt` | 4401 | 阶段2 日志/git 状态 |
| 42 | 16:29:31 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/stage2_20261121_162927.log` | 3306 | 阶段2 日志/git 状态 |
| 43 | 16:29:31 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/stage3_20261121_162927.log` | 2246 | 阶段3 日志/git 状态 |
| 44 | 16:29:31 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/stage3_gitstatus_20261121_162927.txt` | 4287 | 阶段3 日志/git 状态 |
| 45 | 16:30:43 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/path_rewrite_table_20261121_163043.csv` | 5734 | 路径改写对照表 |
| 46 | 16:30:58 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/gitstatus_final_20261121_163043.txt` | 2160 | 最终 git status |
| 47 | 16:31:07 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/residual_refs_20261121_163043.csv` | 5965 | 残留引用分类表 |
| 48 | 16:31:23 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/rewrite_refs_20261121_163043.log` | 735 | 路径改写日志 |
| 49 | 16:33:48 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/stage4_banner_table_20261121_163348.csv` | 601 | 阶段4 横幅对照表 |
| 50 | 16:34:07 | `2026-11-21/repo-cleanup/outputs/cleanup_20261121/stage4_20261121_163348.log` | 903 | 阶段4 日志 |
| 51 | 16:36:35 | `2026-11-21/repo-cleanup/outputs/CLEANUP_LOG.md` | 12146 | 本汇总日志（字节数为追加本行前） |
