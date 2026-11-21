# -*- coding: utf-8 -*-
"""只读对齐分析：表格 vs 目录 vs 管线 CSV。不写 data 目录。"""
import json
import os
import re

import pandas as pd

ROOT = r"d:\PyFile\ColonMultiModal\data\Hos"
DATA = r"d:\PyFile\ColonMultiModal\data"
WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"
MODS = "cdtpz"


def stems(sub):
    base = os.path.join(ROOT, "_nifti", sub)
    return sorted(f[:-7] for f in os.listdir(base) if f.endswith(".nii.gz"))


def split_stem(s):
    m = re.match(r"^(.+)_([" + MODS + r"])$", s)
    return (m.group(1), m.group(2)) if m else (s, "")


def ids_of(sub):
    """返回 {patient_key: set(mods)}"""
    out = {}
    for s in stems(sub):
        pid, mod = split_stem(s)
        out.setdefault(pid, set()).add(mod)
    return out


ct = ids_of("CT")
mr = ids_of("MRI")
b2 = ids_of("batch2")

t1 = pd.read_excel(os.path.join(ROOT, "结直肠癌多模态患者数据收集表.xlsx"), dtype=str)
t2 = pd.read_excel(os.path.join(ROOT, "结直肠癌多模态患者数据收集表9.29.xlsx"), dtype=str)

t1_ids = set(t1["编号"].dropna().astype(str).str.strip())
t2_ids = set(t2["编号"].dropna().astype(str).str.strip())
t2_mrn = set(t2["住院号"].dropna().astype(str).str.strip())

batch1 = {k for k in ct if re.fullmatch(r"\d+", k)} | {k for k in mr}
# 特殊命名的 CT 患者（1_QG..., QG...）
ct_special = sorted({k for k in ct if not re.fullmatch(r"\d+", k)})

pairs = pd.read_csv(os.path.join(DATA, "hos_pairs.csv"))
master = pd.read_csv(os.path.join(DATA, "master_5fold_356.csv"))
master_ids = set(master["patient_id"].astype(str))
pair_ids = set(pairs["patient_id"].astype(str))

all_nii = set(ct) | set(mr) | set(b2)

res = {
    "counts": {
        "nifti_CT_files": len(stems("CT")), "nifti_CT_patients": len(ct),
        "nifti_MRI_files": len(stems("MRI")), "nifti_MRI_patients": len(mr),
        "nifti_b2_files": len(stems("batch2")), "nifti_b2_patients": len(b2),
        "table1_rows": len(t1), "table1_unique_ids": len(t1_ids),
        "table2_rows": len(t2), "table2_unique_ids": len(t2_ids),
        "pairs_rows": len(pairs), "master_rows": len(master),
    },
    "ct_modality_dist": {k: sorted(v) for k, v in sorted(ct.items())},
    "mr_modality_dist": {k: sorted(v) for k, v in sorted(mr.items())},
    "b2_modality_dist": {k: sorted(v) for k, v in sorted(b2.items())},
    "ct_special_keys": ct_special,
    "t1_vs_batch1": {
        "table_only": sorted(t1_ids - batch1, key=lambda x: int(x) if x.isdigit() else 10**9),
        "dir_only": sorted(batch1 - t1_ids, key=lambda x: int(x) if x.isdigit() else 10**9),
        "both": len(t1_ids & batch1),
    },
    "t2_vs_batch2": {
        "table_only": sorted(t2_ids - set(b2), key=lambda x: int(x) if x.isdigit() else 10**9),
        "dir_only": sorted(set(b2) - t2_ids, key=lambda x: int(x) if x.isdigit() else 10**9),
        "both": len(t2_ids & set(b2)),
        "mrn_only": sorted(t2_mrn - set(b2)),
    },
    "batch_overlap": {
        "batch2_and_ct": sorted(set(b2) & set(ct), key=int),
        "batch2_and_mr": sorted(set(b2) & set(mr), key=int),
        "batch2_and_batch1": sorted(set(b2) & batch1, key=int),
    },
    "pairs_vs_nifti": {
        "pairs_not_found": sorted(pair_ids - all_nii, key=lambda x: int(x) if x.isdigit() else 10**9),
        "nifti_not_in_pairs": sorted(all_nii - pair_ids, key=lambda x: int(x) if x.isdigit() else 10**9),
        "found": len(pair_ids & all_nii),
    },
    "master_vs_nifti": {
        "master_not_found": sorted(master_ids - all_nii, key=lambda x: int(x) if x.isdigit() else 10**9),
        "found": len(master_ids & all_nii),
    },
    "pairs_minus_master": sorted(pair_ids - master_ids, key=lambda x: int(x) if x.isdigit() else 10**9),
    "master_minus_pairs": sorted(master_ids - pair_ids, key=lambda x: int(x) if x.isdigit() else 10**9),
}

# 5 例额外病人的详情
extra = []
pairs_idx = pairs.set_index(pairs["patient_id"].astype(str))
for pid in res["pairs_minus_master"]:
    r = pairs_idx.loc[pid]
    extra.append({
        "patient_id": pid,
        "batch": "batch2" if pid in b2 else ("batch1" if pid in batch1 else "缺失"),
        "ct": sorted(ct.get(pid, [])) or None,
        "mr": sorted(mr.get(pid, [])) or None,
        "b2": sorted(b2.get(pid, [])) or None,
        "label_T": r["label_T"], "label_N": r["label_N"],
        "ct_available": bool(r["ct_available"]), "mri_available": bool(r["mri_available"]),
        "ct_path": str(r["ct_path"]), "mri_path": str(r["mri_path"]),
    })
res["extra_5"] = extra

# 表格行数 vs 唯一编号
res["t1_dup_ids"] = {str(k): int(v) for k, v in t1["编号"].value_counts()[lambda s: s > 1].items()}
res["t2_dup_ids"] = {str(k): int(v) for k, v in t2["编号"].value_counts()[lambda s: s > 1].items()}

with open(os.path.join(WORK, "alignment.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))

print(json.dumps({k: v for k, v in res.items()
                  if k not in ("ct_modality_dist", "mr_modality_dist", "b2_modality_dist")},
                 ensure_ascii=False, indent=1,
                 default=lambda o: o.item() if hasattr(o, "item") else str(o)))
# 模态缺失统计
for name, d in (("CT", ct), ("MRI", mr), ("batch2", b2)):
    cnt = {}
    for k, v in d.items():
        cnt["".join(sorted(v))] = cnt.get("".join(sorted(v)), 0) + 1
    print(name, "modality combos:", cnt)
