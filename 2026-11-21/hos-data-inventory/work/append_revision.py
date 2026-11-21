# -*- coding: utf-8 -*-
"""在报告 §7 后插入 §8 复核修订（不改动其他内容）。"""
P = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\outputs\hos-inventory-report.md"

SEC = """## 8. 复核修订（2026-11-21，用户删除 6 个特殊命名 CT 目录后）

用户已删除 `Hos/CT` 下 6 个“编号_住院号”目录（`1_QG12455983`、`2_N01022751`、`3_QG12479734`、`4_QG12425897`、`5_QG12447866`、`6_QG12446259`），并说明收集表中的空位属正常现象。以下为复核后的实测状态，本节覆盖前文 §1–§5 中与之冲突的数字。

| 项目 | 修订前 | 修订后（本次实测） |
|---|---|---|
| `Hos/CT` 目录数 / dcm 数 | 99 / 14,426 | **93 / 13,964**（另 12 个 txt 仍在） |
| CT 真实住院号 | 65 | **59**（58 个编号 + `QG12497657`） |
| `_nifti/CT` 文件数 | 98 | **98（未变，产生 6 个孤儿）** |
| 有原始无 NIfTI | `QG12497657` | `QG12497657`（未变，4 个就诊/4 序列） |
| 有 NIfTI 无原始 | 0 | **6 个**：`1_QG12455983`、`2_N01022751`、`3_QG12479734`、`4_QG12425897`、`5_QG12447866`、`6_QG12446259` |
| `Hos/MRI` | 351 目录 / 131 人 | 未变，351/351 对齐 |

### 8.1 已解决的项

- §5.3 的 5 例索引错配（编号 1/2/4/5/6）：错配的 CT 原始数据已删除，这 5 例此后应为 **MRI-only**（表内标签与 MRI 保留）。
- §3.2 的 7 个“表里有目录没有”（9、12、20、65、75、117、120）确认为正常空位，不需处理；§3.3 的 101、131 同理。
- 编号 3 的目录碰撞（`3_p/3_z` vs `3_QG12479734`）随 `3_QG12479734` 删除而消失，`3_z`（=QG12261993）成为编号 3 的唯一 CT。

### 8.2 残留（尚未处理，需授权）

| 残留 | 位置 | 影响 |
|---|---|---|
| 6 个孤儿 NIfTI（其原始已删） | `data/Hos/_nifti/CT/` 上述 6 个 `*.nii.gz`（约 122 MiB） | 按文件名探测会误判 CT 可用；`expand_ct_series_csv.py` 的 `_find_special_plain()` 仍会把它们当作平扫 |
| `hos_pairs.csv` | patient_id 1、2、4、5、6 | `ct_available=True`、`ct_series=1_QG12455983` 等、`ct_path`/`ct_path_plain` 仍指向孤儿文件；当前 361 行中 `ct_available=True` 计 147 行，其中 5 行为失效引用 |
| `hos_pairs_raw.csv` | 同上 5 行 | 同 `hos_pairs.csv` |
| `master_5fold_356.csv` | 同上 5 行 | `ct_available=1`、`modality_sig` 含 `C1`（如 patient 1 = `C1T1C1D1`）；`ct_available=True` 计 145 行，其中 5 行应改 0 |
| `QG12497657` | `Hos/CT/QG12497657/`（4 个就诊） | 仍未提取；且无编号，无法入表 |
| 备注 | `3_QG12479734` 对应的真实病人 QG12479734（2026-04-29）已随目录删除 | 若该病人本应入组，属数据丢失；仅剩 `_nifti/CT/3_QG12479734.nii.gz` 可回溯 |

### 8.3 建议的收尾顺序（未执行）

1. 确认 6 个孤儿 NIfTI 的处理：移入隔离目录（如 `_nifti/_orphan_20261121/`）或删除；二者都需你批准。
2. 隔离后重新生成 `hos_pairs.csv` / `hos_pairs_raw.csv`（`build_hos_dataset.py --no-convert` 按现有 raw 扫描，1/2/4/5/6 自动变为 MRI-only），再重跑 batch2 合并与 `master_5fold_356.csv` 生成。
3. 重跑前备份 4 个 CSV（带时间戳），并按 §6 方案 B 追加 `pair_status`/`batch`/真实住院号列，避免下次再按编号误配。
4. 受影响实验：仅含 patient_id 1/2/4/5/6 的 CT 分支与多模态分支结论需重跑；纯 MRI 分支及其余 351 例样本不受影响。

"""

s = open(P, encoding="utf-8").read()
marker = "## 附录（逐条清单）"
assert marker in s and "## 8. 复核修订" not in s
s = s.replace(marker, SEC + marker)
open(P, "w", encoding="utf-8").write(s)
print("inserted")
