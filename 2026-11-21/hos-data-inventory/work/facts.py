# -*- coding: utf-8 -*-
"""汇总事实：真实 PatientID 与管线索引的一致性、模态覆盖、表格差集。只读。"""
import json
import os
import re
from collections import defaultdict

import pandas as pd

DATA = r"d:\PyFile\ColonMultiModal\data"
ROOT = os.path.join(DATA, "Hos")
WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"
MODS = "cdtpz"

meta = json.load(open(os.path.join(WORK, "dicom_meta.json"), encoding="utf-8"))

real = {"CT": defaultdict(set), "MRI": defaultdict(set)}  # index -> realIDs
dirs_by_real = {"CT": defaultdict(list), "MRI": defaultdict(list)}
for side in ("CT", "MRI"):
    for r in meta[side]:
        pid = r["meta_first"].get("PatientID")
        if not pid:
            continue
        m = re.match(r"^(\d+)(?:_([A-Za-z0-9]+))?$", r["dir"])
        if m:
            real[side][int(m.group(1))].add(pid)
        dirs_by_real[side][pid].append(r["dir"])

pairs = pd.read_csv(os.path.join(DATA, "hos_pairs.csv"))
master = pd.read_csv(os.path.join(DATA, "master_5fold_356.csv"))

mismatch = []
for i in range(1, 201):
    c = real["CT"].get(i, set())
    m = real["MRI"].get(i, set())
    if not c or not m:
        continue
    if not (c & m):
        mismatch.append({
            "index": i, "ct_real": sorted(c), "mri_real": sorted(m),
            "ct_dirs": [d for x in c for d in dirs_by_real["CT"][x]],
            "mri_dirs": [d for x in m for d in dirs_by_real["MRI"][x]],
        })

# 各索引的模态覆盖
cover = {}
for side in ("CT", "MRI"):
    for i, ids in real[side].items():
        cover.setdefault(i, {"ct": set(), "mri": set()})
        cover[i][side.lower()] = ids

# 文件级模态
def files(sub):
    b = os.path.join(ROOT, "_nifti", sub)
    d = defaultdict(set)
    for f in os.listdir(b):
        if f.endswith(".nii.gz"):
            s = f[:-7]
            mm = re.match(r"^(.+)_([" + MODS + r"])$", s)
            if mm:
                d[mm.group(1)].add(mm.group(2))
            else:
                d[s].add("-")
    return d

ctf, mrf, b2f = files("CT"), files("MRI"), files("batch2")

t1 = pd.read_excel(os.path.join(ROOT, "结直肠癌多模态患者数据收集表.xlsx"), dtype=str)
t2 = pd.read_excel(os.path.join(ROOT, "结直肠癌多模态患者数据收集表9.29.xlsx"), dtype=str)
t1_ids = [str(x).strip() for x in t1["编号"].tolist()]
t2_ids = [str(x).strip() for x in t2["编号"].tolist()]

facts = {
    "ct_dirs": len(meta["CT"]), "mr_dirs": len(meta["MRI"]),
    "ct_real_ids": len(dirs_by_real["CT"]), "mr_real_ids": len(dirs_by_real["MRI"]),
    "real_in_both": len(set(dirs_by_real["CT"]) & set(dirs_by_real["MRI"])),
    "real_ct_only": sorted(set(dirs_by_real["CT"]) - set(dirs_by_real["MRI"])),
    "real_mr_only_cnt": len(set(dirs_by_real["MRI"]) - set(dirs_by_real["CT"])),
    "index_mismatch": mismatch,
    "ct_cover": {str(k): {"p" if "p" in v else "", "z" if "z" in v else ""}
                 for k, v in [(k, ct) for k, ct in []]},
    "ct_mods": {k: "".join(sorted(v)) for k, v in ctf.items()},
    "mr_mods": {k: "".join(sorted(v)) for k, v in mrf.items()},
    "b2_mods": {k: "".join(sorted(v)) for k, v in b2f.items()},
    "t1_row_nan_id": int(sum(1 for x in t1_ids if x in ("nan", ""))),
    "t2_row_nan_id": int(sum(1 for x in t2_ids if x in ("nan", ""))),
    "t1_ids_range": [min(int(x) for x in t1_ids if x.isdigit()), max(int(x) for x in t1_ids if x.isdigit())],
    "t2_ids_range": [min(int(x) for x in t2_ids if x.isdigit()), max(int(x) for x in t2_ids if x.isdigit())],
    "pairs_ids_range": [int(pairs["patient_id"].min()), int(pairs["patient_id"].max())],
    "master_ids_range": [int(master["patient_id"].min()), int(master["patient_id"].max())],
}
# 同一索引多真实ID（CT 内部的索引碰撞）
collide = {str(k): sorted(v) for k, v in real["CT"].items() if len(v) > 1}
facts["ct_index_collision"] = collide

# 模态缺失清单
def by_combo(d):
    out = defaultdict(list)
    for k, v in d.items():
        out["".join(sorted(v)) if v else "none"].append(k)
    return {k: sorted(v, key=lambda s: int(s) if s.isdigit() else 10**9) for k, v in out.items()}

facts["ct_mod_groups"] = by_combo(ctf)
facts["mr_mod_groups"] = by_combo(mrf)
facts["b2_mod_groups"] = by_combo(b2f)

with open(os.path.join(WORK, "facts.json"), "w", encoding="utf-8") as f:
    json.dump(facts, f, ensure_ascii=False, indent=1)

print("索引不一致(CT真实ID≠MRI真实ID) 数量:", len(mismatch))
for x in mismatch:
    print("  idx", x["index"], "CT", x["ct_real"], x["ct_dirs"], "<-> MRI", x["mri_real"], x["mri_dirs"])
print("CT 索引碰撞:", collide)
for k in ("ct_mod_groups", "mr_mod_groups", "b2_mod_groups"):
    print(k)
    for c, ids in facts[k].items():
        print("  ", c, len(ids), ids[:40] if len(ids) > 40 else ids)
print("t1 nan id rows:", facts["t1_row_nan_id"], " t1 range:", facts["t1_ids_range"])
print("t2 nan id rows:", facts["t2_row_nan_id"], " t2 range:", facts["t2_ids_range"])
print("pairs range:", facts["pairs_ids_range"], " master range:", facts["master_ids_range"])
