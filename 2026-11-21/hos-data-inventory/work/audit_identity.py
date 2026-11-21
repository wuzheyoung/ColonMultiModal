# -*- coding: utf-8 -*-
"""以完整版收集表的住院号为准，审计全部 163 行的编号↔CT/MRI 真实身份。只读。"""
import json
import os

import pandas as pd

SRC = r"c:\Users\wzy\.trae-cn\attachments\cc6db7db-c2b1-4a67-af45-b67546fc8f46_e67f5806-9202-4a5f-a975-aca895171bb4_副本结直肠癌多模态患者数据收集表.xlsx"
WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"

meta = json.load(open(os.path.join(WORK, "dicom_meta.json"), encoding="utf-8"))
ct, mr = {}, {}
for r in meta["CT"]:
    m = r["meta_first"]
    num = r["dir"].split("_")[0]
    if num.isdigit():
        ct[int(num)] = (m.get("PatientID"), m.get("PatientName"), r["dir"])
for r in meta["MRI"]:
    m = r["meta_first"]
    num = r["dir"].split("_")[0]
    if num.isdigit():
        mr.setdefault(int(num), (m.get("PatientID"), m.get("PatientName")))


def norm(s):
    return str(s).strip().upper().replace(" ", "")


def cname(s):
    """取拼音名 (DICOM 里形如 'HUANG YONG=...')。"""
    return str(s or "").split("=")[0].replace("^", " ").strip()


df = pd.read_excel(SRC, dtype=str)
df.insert(0, "编号", range(1, len(df) + 1))

rows = []
for _, r in df.iterrows():
    num = int(r["编号"])
    hid = norm(r["住院号"]) if pd.notna(r["住院号"]) else ""
    name = str(r["姓名"]) if pd.notna(r["姓名"]) else ""
    info = str(r["影像信息"]) if pd.notna(r["影像信息"]) else ""
    cs = ct.get(num)
    ms = mr.get(num)
    ct_ok = bool(cs and hid and norm(cs[0]) == hid)
    mr_ok = bool(ms and hid and norm(ms[0]) == hid)
    if (cs and not ct_ok) or (ms and not mr_ok):
        rows.append({
            "编号": num, "表姓名": name, "表住院号": hid, "影像信息": info,
            "CT编号对应": f"{cs[2]}={cs[0]}/{cname(cs[1])}" if cs else "无",
            "CT一致": ct_ok,
            "MRI编号对应": f"{ms[0]}/{cname(ms[1])}" if ms else "无",
            "MRI一致": mr_ok,
        })

out = pd.DataFrame(rows)
print("存在编号-住院号不一致的编号数:", len(out))
print(out.to_string(index=False))
out.to_csv(os.path.join(WORK, "id_mismatch_audit.txt"), sep="\t", index=False, encoding="utf-8")
