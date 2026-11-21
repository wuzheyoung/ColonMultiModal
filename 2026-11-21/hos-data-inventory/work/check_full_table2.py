# -*- coding: utf-8 -*-
"""复核: 完整版收集表的影像信息解析(ct/c/mri) 与现有数据对齐。只读。"""
import json
import os
import re

import pandas as pd

SRC = r"c:\Users\wzy\.trae-cn\attachments\cc6db7db-c2b1-4a67-af45-b67546fc8f46_e67f5806-9202-4a5f-a975-aca895171bb4_副本结直肠癌多模态患者数据收集表.xlsx"
HOS = r"d:\PyFile\ColonMultiModal\data\Hos"
WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"

meta = json.load(open(os.path.join(WORK, "dicom_meta.json"), encoding="utf-8"))


def norm(x):
    return str(x).strip().upper().replace(" ", "")


ct_real, mr_real = {}, {}
for side, d in (("CT", ct_real), ("MRI", mr_real)):
    for r in meta[side]:
        pid = (r.get("meta_first") or {}).get("PatientID")
        if pid:
            d.setdefault(norm(pid), []).append(r["dir"])

ct_now = {f[:-7] for f in os.listdir(os.path.join(HOS, "_nifti", "CT")) if f.endswith(".nii.gz")}
mr_now = {f[:-7] for f in os.listdir(os.path.join(HOS, "_nifti", "MRI")) if f.endswith(".nii.gz")}
ct_nums = {int(x.split("_")[0]) for x in ct_now if re.fullmatch(r"\d+_[pz]", x)}
mr_nums = {int(x.split("_")[0]) for x in mr_now if re.fullmatch(r"\d+_[cdt]", x)}
special = {1: "1_QG12455983", 2: "2_N01022751", 3: "3_QG12479734",
           4: "4_QG12425897", 5: "5_QG12447866", 6: "6_QG12446259"}

df = pd.read_excel(SRC, dtype=str)
df.insert(0, "编号", range(1, len(df) + 1))
info = df["影像信息"].fillna("").str.lower().str.replace(" ", "").str.replace("，", ",")
df["表CT"] = info.str.contains(r"ct") | info.str.contains(r"(?<![a-z])c(?![a-z])")
df["表MRI"] = info.str.contains(r"mri") | info.str.contains(r"(?<![a-z])m(?![a-z])")

print("行数", len(df), "含影像信息", int((info != "").sum()),
      "表CT", int(df["表CT"].sum()), "表MRI", int(df["表MRI"].sum()))

print("\n=== 编号 1-6 行（完整版表）===")
print(df.loc[0:5, ["编号", "姓名", "住院号", "影像信息", "表CT", "表MRI"]].to_string(index=False))

print("\n=== 已删 6 个 CT 病人的住院号是否在表中 ===")
for hid in ["QG12455983", "N01022751", "QG12479734", "QG12425897", "QG12447866", "QG12446259"]:
    print(hid, "命中", int((df["住院号"].map(norm) == hid).sum()))

print("\n=== 表记 CT 但当前无 CT 数据 ===")
for _, r in df.iterrows():
    if r["表CT"]:
        num = int(r["编号"])
        has = num in ct_nums or any(s in ct_now for s in [special.get(num, "")])
        if not has:
            print(f"编号{r['编号']:>3} {r['姓名']} {r['住院号']} {r['影像信息']}")

print("\n=== 有 CT 数据但表未记 CT ===")
for num in sorted(ct_nums):
    row = df[df["编号"] == num]
    if len(row) and not row.iloc[0]["表CT"]:
        print(f"编号{num} {row.iloc[0]['姓名']} {row.iloc[0]['住院号']} {row.iloc[0]['影像信息']!r}")
for hid, dirs in sorted(ct_real.items()):
    row = df[df["住院号"].map(norm) == hid]
    if len(row) and not row.iloc[0]["表CT"]:
        print(f"住院号{hid} 编号{row.iloc[0]['编号']} {row.iloc[0]['姓名']} {row.iloc[0]['影像信息']!r} {dirs}")

print("\n=== 表记 MRI 但当前无 MRI 数据 ===")
for _, r in df.iterrows():
    if r["表MRI"] and int(r["编号"]) not in mr_nums:
        print(f"编号{r['编号']:>3} {r['姓名']} {r['住院号']} {r['影像信息']}")

print("\n=== 有 MRI 数据但表未记 MRI ===")
for num in sorted(mr_nums):
    row = df[df["编号"] == num]
    if len(row) and not row.iloc[0]["表MRI"]:
        print(f"编号{num} {row.iloc[0]['姓名']} {row.iloc[0]['住院号']} {row.iloc[0]['影像信息']!r}")

print("\n=== 表住院号 vs MRI DICOM 住院号 校验 ===")
bad, notin = [], []
for _, r in df.iterrows():
    if pd.isna(r["住院号"]):
        continue
    num, hid = int(r["编号"]), norm(r["住院号"])
    if hid not in mr_real:
        notin.append((num, hid, r["姓名"], r["影像信息"]))
    elif not any(d.split("_")[0] == str(num) for d in mr_real[hid]):
        bad.append((num, hid, r["姓名"], mr_real[hid]))
print("住院号不在 MRI 中", len(notin), notin)
print("住院号在 MRI 但编号不同", len(bad), bad)

print("\n=== 表住院号 vs CT DICOM 住院号 校验（仅编号制 CT）===")
for num in sorted(ct_nums):
    row = df[df["编号"] == num]
    if not len(row):
        continue
    hid = norm(row.iloc[0]["住院号"])
    if hid not in ct_real:
        print(f"编号{num} CT 有影像, 表住院号 {hid} 不在 CT DICOM: {row.iloc[0]['姓名']}")
    elif not any(d.split("_")[0] == str(num) for d in ct_real[hid]):
        print(f"编号{num} CT 有影像, 表住院号 {hid} 对应目录 {ct_real[hid]}")
