# -*- coding: utf-8 -*-
"""从 inventory.json 生成报告用 Markdown 表格 (只读)。"""
import json
import os

R = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(R, "inventory.json"), encoding="utf-8"))
t = d["top_level"]
lines = []
lines.append("| # | 顶层项 | 类型 | 体积 | 文件数 | 占比 |")
lines.append("|---|---|---|---|---|---|")
for i, x in enumerate(t[:16], 1):
    lines.append(f"| {i} | `{x['name']}` | {'目录' if x['type'] == 'dir' else '文件'} | "
                 f"{x['bytes'] / 1024 ** 3:.2f} GB | {x['files']} | {x['pct']:.2f}% |")
rest = t[16:]
lines.append(f"| — | 其余 {len(rest)} 项合计 | — | {sum(x['bytes'] for x in rest) / 1024 ** 2:.1f} MB | "
             f"{sum(x['files'] for x in rest)} | "
             f"{sum(x['pct'] for x in rest):.2f}% |")
lines.append("")
lines.append("| # | 文件 | 体积 |")
lines.append("|---|---|---|")
for i, f in enumerate(d["largest_files"][:10], 1):
    lines.append(f"| {i} | `{f['path']}` | {f['mb']:.1f} MB |")
lines.append("")
lines.append("| 组 | 单份体积 | 副本数 | 可释放 | 路径 |")
lines.append("|---|---|---|---|---|")
for i, g in enumerate(d["duplicates"], 1):
    paths = "<br>".join(f"`{p}`" for p in g["paths"])
    waste = g["bytes_each"] * (len(g["paths"]) - 1) / 1024 ** 2
    lines.append(f"| D{i} | {g['bytes_each'] / 1024 ** 2:.1f} MB | {len(g['paths'])} | "
                 f"{waste:.1f} MB | {paths} |")
open(os.path.join(R, "tables.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
