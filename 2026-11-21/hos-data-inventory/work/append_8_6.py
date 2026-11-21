# -*- coding: utf-8 -*-
"""追加 §8.6：CSV 同步执行记录与验证。"""
P = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\outputs\hos-inventory-report.md"

SEC = """### 8.6 CSV 同步执行记录（2026-11-21 11:10，已执行）

用户删除 `_nifti/CT` 下 6 个孤儿文件后（现 `_nifti/CT` 92 个文件、无孤儿），按授权同步全部 CSV。备份目录：`data/_csv_backup_20261121_111046/`（42 个文件，原样时间戳副本）。执行脚本：`work/patch_csvs.py`（`--apply` 前先跑过 dry-run）。

处理规则：

| 文件类别 | 处理 |
|---|---|
| `hos_pairs.csv`、`hos_pairs_raw.csv`、`hos_branch_full.csv`、`hos_full_*`、`hos_mri_*`、`hos_train*`、`hos_val*` | 保留行；患者 1/2/4/5/6 的 `ct_path`、`ct_path_plain`、`ct_path_enhanced`、`ct_series` 置空，`ct_available`、`ct_avail_plain`、`ct_avail_enhanced` 置 False |
| `hos_branch_ct_p.csv`、`hos_ct_p_train_s42.csv`、`hos_ct_p_val_s42.csv` | 直接剔除这 5 行（CT 分支样本已不存在） |
| `master_5fold_356.csv` | `ct_available=0`，`modality_sig` 重算为 `C0T{...}`，`stratum` 同步；**fold 保持不变** |

变更量：

| 文件 | 行数变化 | 变更行 |
|---|---|---|
| `hos_pairs.csv` / `hos_pairs_raw.csv` | 361 → 361 | 5 |
| `master_5fold_356.csv` | 356 → 356 | 5 |
| `hos_branch_ct_p.csv` | 142 → 137 | 删 5 |
| `hos_ct_p_train_s42.csv` | 112 → 108 | 删 4 |
| `hos_ct_p_val_s42.csv` | 28 → 27 | 删 1 |
| `hos_branch_{full,mri_c,mri_d,mri_t,mri_tdc}.csv` | 不变 | 3–5 |
| `hos_full_kfold_f{0..4}_{train,val}.csv` | 不变 | 1–5/文件 |
| `hos_full_{train,val}_s{0,1,42,7}.csv` | 不变 | 2–5/文件 |
| `hos_mri_{c,d,t,tdc}_*_s42.csv` | 不变 | 1–4/文件 |
| `hos_{train,val}*.csv`（旧格式） | 不变 | 1–4/文件 |

验证结果：

- 全部 CSV 的所有路径列（ct/mri/plain/enhanced）逐行检查：**0 个指向不存在文件**。
- `hos_pairs.csv`：`ct_available` 147 → **142**；1/2/4/5/6 均为 False、`ct_series` 空、`mri_series` 保留（1_t…6_t）。
- `master_5fold_356.csv`：`ct_available` 145 → **140**；例如 patient 1 = `0 / C0T1C1D1 / 1_1_C0T1C1D1 / fold1`，patient 4 = `0 / C0T1C1D0 / 1_0_C0T1C1D0 / fold2`。
- 分折人数仍为 72/71/71/71/71（fold 未重排，保证与既有实验可比）。
- 缓存：`cache_hos/*.npy` 以影像路径为键，CT 路径置空后**不会命中旧缓存**，无需清理；旧条目成为孤儿但不会被读取。
- 实验影响：含 1/2/4/5/6 的 CT 分支、全模态分支结果需重跑；纯 MRI 分支不受影响。

仍未处理（需确认）：

1. **编号 97**：CT 为 10900337 王宏明，表为贺万珍 QG12339632，且无 MRI——目前仍按原样保留，属唯一剩余的身份冲突。
2. `QG12497657`：仍未提取（4 个就诊/4 序列）。
3. 表记 CT 但无 CT 数据 11 例（12、15、18、50、65、75、117、120、129、136、144），其中编号 6 的 `12.20 下腹ct` 明确有记录无数据。
4. 旧结果文件（`results/`、`checkpoints_hos*/`、`plots_hos*/`）未改动，但引用这 5 例的指标已不可用于最终结论。

"""

s = open(P, encoding="utf-8").read()
marker = "## 附录（逐条清单）"
assert marker in s and "### 8.6 " not in s
s = s.replace(marker, SEC + marker)
open(P, "w", encoding="utf-8").write(s)
print("added 8.6")
