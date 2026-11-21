# -*- coding: utf-8 -*-
"""对 G:\数据\第二批 全量做 性别/年龄 交叉核对（只读头信息）。"""
import glob
import os
import re

import pandas as pd
import SimpleITK as sitk

RAW = r"G:\数据\第二批"
OFF = {"1-100": 0, "101-200": 100, "201-": 200}
PAT = re.compile(r"^(\d+)(c\+|c|\+|d|t)?$")
TAB = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work\b2_table.csv"


def first(d):
    for f in sorted(glob.glob(os.path.join(d, "*.dcm"))):
        if os.path.getsize(f) > 0:
            return f
    return ""


def meta(f):
    r = sitk.ImageFileReader()
    r.SetFileName(f)
    r.ReadImageInformation()
    g = lambda k: r.GetMetaData(k) if r.HasMetaDataKey(k) else ""
    return g("0010|0040").strip(), g("0010|0030").strip(), g("0008|0020").strip()


tab = pd.read_csv(TAB, dtype=str)
tmap = {int(float(r["编号"])): r for _, r in tab.iterrows()}
rows = []
for seg, off in OFF.items():
    base = os.path.join(RAW, seg)
    if not os.path.isdir(base):
        continue
    for name in os.listdir(base):
        m = PAT.match(name)
        if not m:
            continue
        f = first(os.path.join(base, name))
        if not f:
            continue
        sex, birth, date = meta(f)
        aid = int(m.group(1)) + off
        rows.append({"编号": aid, "dir": f"{seg}/{name}", "sex": sex, "birth": birth, "date": date})
df = pd.DataFrame(rows)
df["sex_set"] = df.groupby("编号")["sex"].transform(lambda s: "/".join(sorted(set(s))))
df["birth_set"] = df.groupby("编号")["birth"].transform(lambda s: "/".join(sorted(set(s))))

issues = []
for aid, g in df.groupby("编号"):
    t = tmap.get(aid)
    if t is None:
        issues.append((aid, "表中无此编号", g["sex_set"].iloc[0], g["birth_set"].iloc[0]))
        continue
    if "/" in g["sex_set"].iloc[0]:
        issues.append((aid, f"序列性别不一致 {g['sex_set'].iloc[0]}", "", ""))
        continue
    if "/" in g["birth_set"].iloc[0]:
        issues.append((aid, f"序列生日不一致 {g['birth_set'].iloc[0]}", "", ""))
        continue
    ts = str(t.get("性别", "")).strip()
    ts = {"男": "M", "女": "F"}.get(ts, ts)
    if ts and ts != g["sex_set"].iloc[0]:
        issues.append((aid, f"性别不符 表{ts} 影像{g['sex_set'].iloc[0]}", "", ""))
        continue
    try:
        age = int(float(str(t.get("年龄", "")).replace("岁", "")))
        b = g["birth_set"].iloc[0]
        d = g["date"].iloc[0]
        if len(b) == 8 and len(d) == 8:
            edad = int(d[:4]) - int(b[:4]) - (1 if d[4:] < b[4:] else 0)
            if abs(edad - age) > 3:
                issues.append((aid, f"年龄差异 表{age} 影像{edad}", b, d))
    except (ValueError, TypeError):
        pass

print("编号数", df["编号"].nunique(), "目录数", len(df))
print("问题数", len(issues))
for i in issues:
    print("  ", i)
