# -*- coding: utf-8 -*-
"""用完整版收集表(25列, 含影像信息/住院号)核对: 表记录 CT 的病例 vs 现有 CT 数据。只读。"""
import json
import os
import re

import pandas as pd

SRC = r"c:\Users\wzy\.trae-cn\attachments\cc6db7db-c2b1-4a67-af45-b67546fc8f46_e67f5806-9202-4a5f-a975-aca895171bb4_副本结直肠癌多模态患者数据收集表.xlsx"
HOS = r"d:\PyFile\ColonMultiModal\data\Hos"
WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"

meta = json.load(open(os.path.join(WORK, "dicom_meta.json"), encoding="utf-8"))
# 历史扫描(含已删目录)的真实住院号
ct_real = {}
for r in meta["CT"]:
    pid = (r.get("meta_first") or {}).get("PatientID")
    if pid:
        ct_real.setdefault(pid.strip(), []).append(r["dir"])
mr_real = {}
for r in meta["MRI"]:
    pid = (r.get("meta_first") or {}).get("PatientID")
    if pid:
        mr_real.setdefault(pid.strip(), []).append(r["dir"])

# 现有 NIfTI
ct_now = {f[:-7] for f in os.listdir(os.path.join(HOS, "_nifti", "CT")) if f.endswith(".nii.gz")}
mr_now = {f[:-7] for f in os.listdir(os.path.join(HOS, "_nifti", "MRI")) if f.endswith(".nii.gz")}
raw_ct = {d for d in os.listdir(os.path.join(HOS, "CT")) if os.path.isdir(os.path.join(HOS, "CT", d))}

df = pd.read_excel(SRC, dtype=str)
df.insert(0, "编号", range(1, len(df) + 1))
info = df["影像信息"].fillna("").str.lower().str.replace(" ", "")
df["表记CT"] = info.str.contains("ct")
df["表记MRI"] = info.str.contains("mri")

# 现有 CT 对应编号(按 编号_p / 编号_z)
ct_nums = {int(x.split("_")[0]) for x in ct_now if re.fullmatch(r"\d+_[pz]", x)}
mr_nums = {int(x.split("_")[0]) for x in mr_now if re.fullmatch(r"\d+_[cdt]", x)}
ct_any = ct_nums | ct_nums  # 编号制 CT
# 特殊名 CT -> 编号
special = {1: "1_QG12455983", 2: "2_N01022751", 3: "3_QG12479734",
           4: "4_QG12425897", 5: "5_QG12447866", 6: "6_QG12446259"}

print("总行数", len(df), "表记CT", int(df["表记CT"].sum()), "表记MRI", int(df["表记MRI"].sum()),
      "含影像信息", int((info != "").sum()))

# 1) 6 个已删住院号在表中的记录
del_ids = ["QG12455983", "N01022751", "QG12479734", "QG12425897", "QG12447866", "QG12446259"]
print("\n=== 已删 CT 病人与表对照 ===")
for hid in del_ids:
    hit = df[df["住院号"].astype(str).str.strip() == hid]
    if len(hit) == 0:
        print(hid, "表中无此住院号")
    else:
        for _, r in hit.iterrows():
            print(hid, "编号", r["编号"], r["姓名"], r["影像信息"], "表记CT=", r["表记CT"],
                  "表记MRI=", r["表记MRI"])

# 2) 表记 CT 但当前无 CT 数据的编号
print("\n=== 表记 CT 但当前无 CT NIfTI ===")
miss = []
for _, r in df.iterrows():
    if r["表记CT"]:
        num = int(r["编号"])
        has = num in ct_nums or any(s in ct_now for s in special.get(num, ""))
        if not has:
            miss.append((num, r["住院号"], r["姓名"], r["影像信息"]))
for x in miss:
    print(x)

# 3) 当前有 CT 但表未记 CT
print("\n=== 有 CT NIfTI 但表未记 CT（按编号/住院号） ===")
rows_by_mrn = {str(r["住院号"]).strip(): r for _, r in df.iterrows()}
for num in sorted(ct_nums):
    row = df[df["编号"] == num].iloc[0] if (df["编号"] == num).any() else None
    if row is not None and not row["表记CT"]:
        print(num, row["住院号"], row["姓名"], repr(row["影像信息"]))
for hid, dirs in sorted(ct_real.items()):
    if hid in rows_by_mrn and not rows_by_mrn[hid]["表记CT"]:
        print("按住院号:", hid, rows_by_mrn[hid]["编号"], rows_by_mrn[hid]["姓名"], repr(rows_by_mrn[hid]["影像信息"]), dirs)
# 4) 表记MRI 但无 MRI
print("\n=== 表记 MRI 但当前无 MRI NIfTI ===")
for _, r in df.iterrows():
    if r["表记MRI"] and int(r["编号"]) not in mr_nums:
        print(int(r["编号"]), r["住院号"], r["姓名"], r["影像信息"])
# 5) 有 MRI 但表未记
print("\n=== 有 MRI NIfTI 但表未记 MRI ===")
for num in sorted(mr_nums):
    row = df[df["编号"] == num]
    if len(row) and not row.iloc[0]["表记MRI"]:
        print(num, row.iloc[0]["住院号"], row.iloc[0]["姓名"], repr(row.iloc[0]["影像信息"]))
# 6) 住院号与编号是否一致(表 vs MRI DICOM)
print("\n=== 表住院号 vs MRI DICOM 住院号 不一致 ===")
bad = []
for _, r in df.iterrows():
    num = int(r["编号"])
    hid = str(r["住院号"]).strip()
    dirs = mr_real.get(hid, [])
    ok = any(d.split("_")[0] == str(num) for d in dirs)
    if dirs and not ok:
        bad.append((num, hid, r["姓名"], dirs))
print("不一致数", len(bad))
for x in bad[:20]:
    print(x)
print("\n=== 表中住院号在 MRI 中不存在 ===")
for _, r in df.iterrows():
    hid = str(r["住院号"]).strip()
    if hid not in mr_real:
        print(int(r["编号"]), hid, r["姓名"], r["影像信息"])
