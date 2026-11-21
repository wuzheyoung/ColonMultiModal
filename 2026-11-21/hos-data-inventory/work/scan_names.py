# -*- coding: utf-8 -*-
"""重扫 MRI/CT 头(含性别/生日)，正确解码 GBK 姓名，定位表-数据错位范围。只读。"""
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work")
import dicom_meta as dm  # noqa: E402

dm.WANT.update({(0x0010, 0x0030): "Birth", (0x0010, 0x0040): "Sex"})
HOS = r"d:\PyFile\ColonMultiModal\data\Hos"


def cname(s):
    raw = str(s or "")
    parts = raw.split("=")
    txt = parts[1] if len(parts) >= 2 and parts[1] else raw.split("=")[0].replace("^", " ")
    try:
        txt = txt.encode("latin-1", "ignore").decode("gb18030", "ignore")
    except Exception:
        pass
    txt = re.sub(r"\x1b[^a-zA-Z]*[A-Za-z]|\x1b\(B", "", txt)
    return txt.strip()


def scan(side):
    out = {}
    for d in sorted(os.listdir(os.path.join(HOS, side))):
        p = os.path.join(HOS, side, d)
        if not os.path.isdir(p):
            continue
        f = sorted(glob.glob(os.path.join(p, "**", "*.dcm"), recursive=True))
        if not f:
            continue
        m = dm.parse(f[0])
        num = d.split("_")[0]
        if num.isdigit():
            out.setdefault(int(num), []).append({
                "dir": d, "name": cname(m.get("PatientName")), "id": m.get("PatientID"),
                "sex": {"M": "男", "F": "女"}.get(str(m.get("Sex")), "?"),
                "birth": m.get("BirthDate") or m.get("Birth"), "date": m.get("StudyDate"),
            })
    return out


mr, ct = scan("MRI"), scan("CT")
a = pd.read_excel(r"c:\Users\wzy\.trae-cn\attachments\cc6db7db-c2b1-4a67-af45-b67546fc8f46_e67f5806-9202-4a5f-a975-aca895171bb4_副本结直肠癌多模态患者数据收集表.xlsx", dtype=str)
a.insert(0, "编号", range(1, len(a) + 1))
tn = {int(r["编号"]): str(r["姓名"]).strip() for _, r in a.iterrows()}
tsex = {int(r["编号"]): str(r["性别"]).strip() for _, r in a.iterrows()}

print("== 姓名对位统计 ==")
eq0 = eqp1 = eqm1 = 0
rows = []
for n in sorted(set(mr) | set(ct)):
    recs = (mr.get(n) or ct.get(n) or [{}])
    nm = recs[0].get("name", "")
    if nm == tn.get(n):
        eq0 += 1
    elif nm == tn.get(n + 1):
        eqp1 += 1
        rows.append(n)
    elif nm == tn.get(n - 1):
        eqm1 += 1
    else:
        rows.append(("?", n, nm, tn.get(n)))
print(f"与同行姓名一致={eq0}  与下一行姓名一致={eqp1}  与上一行一致={eqm1}")
shift_up = [x for x in rows if isinstance(x, int)]
print("DICOM 姓名 = 表下一行姓名 的编号(前 60):", shift_up[:60])
print("无法归类(前 30):", [x for x in rows if not isinstance(x, int)][:30])

print("\n== 性别比对(同行) ==")
bad_sex = [(n, tsex.get(n), (mr.get(n) or ct.get(n))[0]["sex"], (mr.get(n) or ct.get(n))[0]["name"]) for n in sorted(mr)
           if tsex.get(n) and (mr.get(n) or ct.get(n))[0]["sex"] != "?" and tsex[n] != (mr.get(n) or ct.get(n))[0]["sex"]]
print("不一致数", len(bad_sex), bad_sex[:15])

print("\n== 93-101 明细 ==")
for n in range(93, 102):
    m = (mr.get(n) or [{}])[0]
    print(n, "表:", tn.get(n), tsex.get(n), "| MRI:", m.get("id"), m.get("name"), m.get("sex"), m.get("birth"), m.get("date"))
