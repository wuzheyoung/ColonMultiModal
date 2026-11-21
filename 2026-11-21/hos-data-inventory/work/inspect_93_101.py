# -*- coding: utf-8 -*-
"""编号 93-101 区域的三方对照：表 / MRI DICOM / CT DICOM。只读。"""
import json

import pandas as pd

WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"
meta = json.load(open(WORK + r"\dicom_meta.json", encoding="utf-8"))

a = pd.read_excel(r"c:\Users\wzy\.trae-cn\attachments\cc6db7db-c2b1-4a67-af45-b67546fc8f46_e67f5806-9202-4a5f-a975-aca895171bb4_副本结直肠癌多模态患者数据收集表.xlsx", dtype=str)
a.insert(0, "编号", range(1, len(a) + 1))

mr = {}
for r in meta["MRI"]:
    p = r["dir"].split("_")[0]
    if p.isdigit():
        mr.setdefault(int(p), []).append(r["meta_first"])
ct = {}
for r in meta["CT"]:
    p = r["dir"].split("_")[0]
    if p.isdigit():
        ct.setdefault(int(p), []).append((r["dir"], r["meta_first"]))

print(f"{'编号':<4}{'表姓名':<8}{'表住院号':<14}{'性':<3}{'龄':<4}{'影像信息':<18}{'MRI:ID/姓名':<34}{'CT:目录/ID/姓名'}")
for n in range(93, 102):
    row = a[a["编号"] == n].iloc[0]
    m = mr.get(n, [{}])[0]
    ms = f"{m.get('PatientID')}/{str(m.get('PatientName','')).split('=')[0]}" if m else "-"
    cs = "; ".join(f"{d}/{x.get('PatientID')}/{str(x.get('PatientName','')).split('=')[0]}" for d, x in ct.get(n, [])) or "-"
    print(f"{n:<4}{row['姓名']:<8}{str(row['住院号']):<14}{row['性别']:<3}{row['年龄']:<4}{str(row['影像信息']):<18}{ms:<34}{cs}")

print("\n--- 详细头信息 ---")
for n in range(93, 102):
    for d, x in ct.get(n, []):
        print(f"CT {d}: ID={x.get('PatientID')} Name={x.get('PatientName')} Sex={x.get('PatientSex')} "
              f"Birth={x.get('PatientBirthDate')} Date={x.get('StudyDate')} Desc={x.get('SeriesDescription')}")
    if n in mr:
        x = mr[n][0]
        print(f"MRI {n}: ID={x.get('PatientID')} Name={x.get('PatientName')} Sex={x.get('PatientSex')} "
              f"Birth={x.get('PatientBirthDate')} Date={x.get('StudyDate')}")
print("\n表 93-101 全字段:")
print(a[a["编号"].between(93, 101)][["编号", "姓名", "住院号", "性别", "年龄", "手术日期", "影像信息", "病理T", "病理N"]].to_string(index=False))
