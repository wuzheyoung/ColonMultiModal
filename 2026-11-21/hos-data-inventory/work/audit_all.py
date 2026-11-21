# -*- coding: utf-8 -*-
"""全量一致性审计：表 vs CT/MRI 身份、性别年龄、标签、批次、NIfTI 完整性。只读。"""
import glob
import json
import os
import re
from datetime import datetime

import pandas as pd

DATA = r"d:\PyFile\ColonMultiModal\data"
WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"
HOS = os.path.join(DATA, "Hos")
T25 = r"c:\Users\wzy\.trae-cn\attachments\cc6db7db-c2b1-4a67-af45-b67546fc8f46_e67f5806-9202-4a5f-a975-aca895171bb4_副本结直肠癌多模态患者数据收集表.xlsx"

meta = json.load(open(os.path.join(WORK, "dicom_meta.json"), encoding="utf-8"))


def norm(x):
    return str(x).strip().upper().replace(" ", "")


def cname(s):
    s = str(s or "")
    m = re.search(r"=([^=]+)$", s)  # 形如 'WANG HONG MING=中文名-Latin'
    return (m.group(1) if m else s.split("=")[0]).replace("^", " ").strip()


t = pd.read_excel(os.path.join(HOS, "结直肠癌多模态患者数据收集表.xlsx"), dtype=str)
t["编号"] = pd.to_numeric(t["编号"], errors="coerce").astype("Int64")
t25 = pd.read_excel(T25, dtype=str)
t25.insert(0, "编号", range(1, len(t25) + 1))
tbl = t.merge(t25[["编号", "姓名", "住院号", "影像信息"]], on="编号", how="left")

ct_meta, mr_meta = {}, {}
for r in meta["CT"]:
    p = r["dir"].split("_")[0]
    if p.isdigit():
        ct_meta.setdefault(int(p), []).append(r)
for r in meta["MRI"]:
    p = r["dir"].split("_")[0]
    if p.isdigit():
        mr_meta.setdefault(int(p), []).append(r)

ct_files = {f[:-7] for f in os.listdir(os.path.join(HOS, "_nifti", "CT")) if f.endswith(".nii.gz")}
mr_files = {f[:-7] for f in os.listdir(os.path.join(HOS, "_nifti", "MRI")) if f.endswith(".nii.gz")}
b2_files = {f[:-7] for f in os.listdir(os.path.join(HOS, "_nifti", "batch2")) if f.endswith(".nii.gz")}

find = {"id": [], "sex": [], "age": [], "label": [], "series": [], "other": []}

for _, r in tbl.iterrows():
    if pd.isna(r["编号"]):
        continue
    num = int(r["编号"])
    hid = norm(r["住院号"]) if pd.notna(r["住院号"]) else ""
    sex = str(r["性别"]).strip()
    age = pd.to_numeric(r["年龄"], errors="coerce")
    for side, mp, files in (("CT", ct_meta, ct_files), ("MRI", mr_meta, mr_files)):
        recs = mp.get(num, [])
        if not recs:
            continue
        m = recs[0]["meta_first"]
        did, dnm = norm(m.get("PatientID")), cname(m.get("PatientName"))
        if hid and did != hid:
            find["id"].append(f"{side} 编号{num} 表={r['姓名']}/{hid} 实际={dnm}/{did}")
        dsex = {"M": "男", "F": "女"}.get(str(m.get("PatientSex")).strip(), "?")
        if sex and dsex != "?" and dsex != sex:
            find["sex"].append(f"{side} 编号{num} 表={sex} 实际={dsex} ({dnm})")
        bd = str(m.get("PatientBirthDate") or "")
        sd = str(m.get("StudyDate") or "")
        if len(bd) == 8 and len(sd) == 8 and pd.notna(age):
            a = (datetime.strptime(sd, "%Y%m%d") - datetime.strptime(bd, "%Y%m%d")).days / 365.25
            if abs(a - float(age)) > 3:
                find["age"].append(f"{side} 编号{num} 表年龄={int(age)} 实际={a:.0f} ({dnm} {sd})")
        # 序列描述与模态/部位
        desc = str(m.get("SeriesDescription") or "")
        if side == "MRI" and str(m.get("Modality")) != "MR":
            find["other"].append(f"MRI 编号{num} Modality={m.get('Modality')}")
        if side == "CT" and str(m.get("Modality")) != "CT":
            find["other"].append(f"CT 编号{num} Modality={m.get('Modality')}")

# CT 平扫/增强 与描述一致性
for num, recs in ct_meta.items():
    for rec in recs:
        d = rec["dir"][-1]
        desc = str(rec["meta_first"].get("SeriesDescription") or "").upper()
        if d == "p" and ("C+" in desc or "ENHANC" in desc):
            find["series"].append(f"CT {rec['dir']} 标为平扫但描述含增强: {desc}")
        if d == "z" and ("C+" in desc or "ENHANC" in desc):
            find["series"].append(f"CT {rec['dir']} 标为增强且描述含 C+: {desc}")

# 标签一致性: pairs/master/表
p = pd.read_csv(os.path.join(DATA, "hos_pairs.csv"))
mst = pd.read_csv(os.path.join(DATA, "master_5fold_356.csv"))
j = p.merge(mst, on="patient_id", suffixes=("_p", "_m"))
bad_lab = j[(j.label_T_p != j.label_T_m) | (j.label_N_p != j.label_N_m)]
find["label"].append(f"pairs vs master 标签不一致: {len(bad_lab)} 行 {bad_lab.patient_id.tolist()[:10]}")

# 表病理 vs pairs 标签（第一批）
lbl = {}
for _, r in tbl.iterrows():
    if pd.isna(r["编号"]):
        continue
    pt, pn = pd.to_numeric(r["病理T"], errors="coerce"), pd.to_numeric(r["病理N"], errors="coerce")
    lbl[int(r["编号"])] = (pt, pn)
mm = []
for _, r in p[p.patient_id <= 162].iterrows():
    pt, pn = lbl.get(int(r.patient_id), (None, None))
    if pd.notna(pt) and pd.notna(pn):
        if int(pt) != int(r.label_T) or int(pn) != int(r.label_N):
            mm.append((int(r.patient_id), (pt, pn), (r.label_T, r.label_N)))
find["label"].append(f"表病理 vs pairs 标签不一致: {len(mm)} 行 {mm[:6]}")

# 影像信息 vs 实际影像
for _, r in tbl.iterrows():
    if pd.isna(r["编号"]):
        continue
    num, info = int(r["编号"]), str(r["影像信息"] or "").lower().replace(" ", "")
    has_ct = any(x in ct_files for x in (f"{num}_p", f"{num}_z")) or any(s in ct_files for s in [f"{num}_QG12455983", f"{num}_N01022751", f"{num}_QG12479734", f"{num}_QG12425897", f"{num}_QG12447866", f"{num}_QG12446259"])
    has_mr = any(x in mr_files for x in (f"{num}_c", f"{num}_t", f"{num}_d"))
    rec_ct = "ct" in info or bool(re.search(r"(?<![a-z])c(?![a-z])", info))
    rec_mr = "mri" in info or bool(re.search(r"(?<![a-z])m(?![a-z])", info))
    if rec_ct and not has_ct:
        find["other"].append(f"编号{num} 表记CT无数据")
    if rec_mr and not has_mr:
        find["other"].append(f"编号{num} 表记MRI无数据")
    if has_ct and not rec_ct:
        find["other"].append(f"编号{num} 有CT但表未记")
    if has_mr and not rec_mr and "mir" not in info:
        find["other"].append(f"编号{num} 有MRI但表未记({info})")

# 重复住院号
dup = tbl[tbl["住院号"].astype(str).str.strip().duplicated(keep=False)]
find["other"].append(f"表内重复住院号: {len(dup)} 行 {dup[['编号','姓名','住院号']].values.tolist()[:6]}")

# DICOM 同一 ID 出现在多个编号
for side, mp in (("CT", ct_meta), ("MRI", mr_meta)):
    ids = {}
    for num, recs in mp.items():
        ids.setdefault(norm(recs[0]["meta_first"].get("PatientID")), []).append(num)
    d = {k: v for k, v in ids.items() if len(v) > 1}
    if d:
        find["other"].append(f"{side} 同一住院号占多个编号: {d}")

# 批次: batch2 vs 第一批 住院号不重叠（无原始DICOM，仅编号）
b2_rows = pd.read_excel(os.path.join(HOS, "结直肠癌多模态患者数据收集表9.29.xlsx"), dtype=str)
b2_rows["编号"] = pd.to_numeric(b2_rows["编号"], errors="coerce").astype("Int64")
dup2 = b2_rows[b2_rows["住院号"].astype(str).str.strip().duplicated(keep=False)]
find["other"].append(f"9.29 表内重复住院号: {len(dup2)} 行 {dup2[['编号','姓名','住院号']].values.tolist()[:6]}")
find["other"].append(f"9.29 住院号与第一批重叠: {sorted(set(b2_rows['住院号'].dropna().map(norm)) & set(tbl['住院号'].dropna().map(norm)))}")

# NIfTI 完整性
zero = [f for f in glob.glob(os.path.join(HOS, "_nifti", "*", "*.nii.gz")) if os.path.getsize(f) < 1024]
find["other"].append(f"小于1KB的NIfTI: {[os.path.basename(x) for x in zero]}")

for k, v in find.items():
    print(f"\n### {k}  ({len(v)})")
    for x in v:
        print(" ", x)
