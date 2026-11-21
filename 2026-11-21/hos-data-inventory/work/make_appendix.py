# -*- coding: utf-8 -*-
"""生成报告附录用的 Markdown 片段（数据来自 facts.json / alignment.json）。"""
import json
import os
import re
from collections import OrderedDict

WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"
facts = json.load(open(os.path.join(WORK, "facts.json"), encoding="utf-8"))
alig = json.load(open(os.path.join(WORK, "alignment.json"), encoding="utf-8"))

L = []


def sec(t):
    L.append("\n" + t + "\n")


def ids(lst):
    return "、".join(str(x) for x in lst) if lst else "（无）"


b2 = set(facts["b2_mods"])
b1 = set(facts["ct_mods"]) | set(facts["mr_mods"])
b1n = {x for x in b1 if x.isdigit()}
ov = sorted(b1n & b2, key=int)
sec("### A. 第一批影像目录的模态覆盖（按 NIfTI 文件）")
L.append("| 组合 | 人数 | 编号 |")
L.append("|---|---|---|")
for combo, lst in facts["ct_mod_groups"].items():
    L.append(f"| CT `{combo}` | {len(lst)} | {ids(lst)} |")
for combo, lst in facts["mr_mod_groups"].items():
    L.append(f"| MRI `{combo}` | {len(lst)} | {ids(lst)} |")
L.append("")
L.append(f"CT 数字编号 {len([k for k in facts['ct_mods'] if k.isdigit()])} 个、批量 CT 目录 6 个（特殊命名）；MRI 编号 {len(facts['mr_mods'])} 个。")
both = sorted(set(facts["ct_mods"]) & set(facts["mr_mods"]), key=int)
L.append("双模态编号：" + ids(both))
ct_only = sorted(set(facts["ct_mods"]) - set(facts["mr_mods"]), key=lambda s: int(s) if s.isdigit() else 10**9)
mr_only = sorted(set(facts["mr_mods"]) - set(facts["ct_mods"]), key=int)
L.append("")
L.append(f"仅 CT（无 MRI）{len(ct_only)} 例：" + ids(ct_only))
L.append("")
L.append(f"仅 MRI（无 CT）{len(mr_only)} 例：" + ids(mr_only))

sec("### B. 第二批 batch2 模态覆盖")
L.append("| 组合 | 人数 | 编号 |")
L.append("|---|---|---|")
for combo, lst in facts["b2_mod_groups"].items():
    L.append(f"| `{combo}` | {len(lst)} | {ids(lst)} |")
L.append("")
L.append(f"batch2 编号 1–208，缺号：{ids(sorted(set(range(1, 209)) - {int(x) for x in b2}, key=int))}")
L.append("")
L.append(f"batch2 与第一批同号（编号空间冲突）：{len(ov)} 个 → " + ids(ov))
L.append(f"batch2 独有编号：{len(b2 - b1n)} 个 → " + ids(sorted(b2 - b1n, key=int)))

sec("### C. 表格差集")
L.append("- 收集表（163 行，编号 1–162，另 1 行编号为空）中无任何影像目录的 7 例："
         + ids(alig["t1_vs_batch1"]["table_only"]))
L.append("- 影像目录中但收集表没有的编号：无")
L.append("- 收集表9.29（208 行，编号 1–208）中 batch2 无 NIfTI 的 2 例："
         + ids(alig["t2_vs_batch2"]["table_only"]))
L.append("- batch2 有 NIfTI 但 9.29 表没有的编号：无")

sec("### D. 索引—真实 PatientID 不一致清单（DICOM 头核对）")
L.append("| 编号 | CT 目录 | CT 真实住院号 | MRI 目录 | MRI 真实住院号 |")
L.append("|---|---|---|---|---|")
for x in facts["index_mismatch"]:
    L.append(f"| {x['index']} | {', '.join(x['ct_dirs'])} | {', '.join(x['ct_real'])} | "
             f"{', '.join(x['mri_dirs'])} | {', '.join(x['mri_real'])} |")
L.append("")
L.append("CT 编号 3 的目录碰撞：" + str(facts["ct_index_collision"]))

sec("### E. 27 例 CT 有、MRI 无（按真实住院号）")
L.append("CT-only 真实住院号（31 个，含无编号的 QG12497657）："
         + ids(facts["real_ct_only"]))

with open(os.path.join(WORK, "appendix.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("\n".join(L))
