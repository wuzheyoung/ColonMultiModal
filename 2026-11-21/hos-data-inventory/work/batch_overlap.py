# -*- coding: utf-8 -*-
"""检查 batch2 与 batch1(_nifti/CT,MRI) 同名文件是否同一份数据；并核对 pairs 的 200 偏移映射。"""
import hashlib
import json
import os

import pandas as pd

ROOT = r"d:\PyFile\ColonMultiModal\data\Hos\_nifti"
DATA = r"d:\PyFile\ColonMultiModal\data"
WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"


def files(sub):
    b = os.path.join(ROOT, sub)
    return {f[:-7]: os.path.join(b, f) for f in os.listdir(b) if f.endswith(".nii.gz")}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ct, mr, b2 = files("CT"), files("MRI"), files("batch2")
common = sorted(set(b2) & (set(ct) | set(mr)))
print("同名 stem 交集:", len(common))

same_size, diff_size = [], []
for s in common:
    a = b2[s]
    b = ct.get(s) or mr.get(s)
    (same_size if os.path.getsize(a) == os.path.getsize(b) else diff_size).append(s)
print("size 相同:", len(same_size), " size 不同:", len(diff_size))

# 对 size 相同的做 md5 抽查
ident, notident = [], []
for s in same_size:
    if md5(b2[s]) == md5(ct.get(s) or mr[s]):
        ident.append(s)
    else:
        notident.append(s)
print("md5 相同(完全重复):", len(ident), " md5 不同:", len(notident))
print("md5 不同的 stem 样例:", notident[:10])
print("md5 相同的 stem 样例:", ident[:10])
print("size 不同 stem 样例:", diff_size[:20])

# pairs 的 201..408 -> batch2 偏移校验
pairs = pd.read_csv(os.path.join(DATA, "hos_pairs.csv"))
rows = []
bad = []
for _, r in pairs.iterrows():
    pid = int(r["patient_id"])
    mp = str(r["mri_path"])
    cp = str(r["ct_path"])
    if pid > 200:
        for col, p in (("mri", mp), ("ct", cp)):
            if p != "nan" and "batch2" in p:
                stem = os.path.basename(p)[:-7]
                expect = f"{pid - 200}_{stem.split('_')[-1]}"
                rows.append({"pid": pid, "col": col, "file": stem, "expect": expect,
                             "ok": stem == expect})
                if stem != expect:
                    bad.append((pid, col, stem, expect))
print("pairs>200 且指向 batch2 的路径条数:", len(rows), " 偏移不一致:", len(bad))
print("不一致样例:", bad[:10])

# batch2 编号范围/缺口
b2ids = sorted(int(s.split("_")[0]) for s in b2)
print("batch2 id 范围:", b2ids[0], "-", b2ids[-1], "缺失:", [i for i in range(b2ids[0], b2ids[-1] + 1) if i not in set(b2ids)])

# 输出中间清单
out = {
    "same_size": same_size, "diff_size": diff_size,
    "identical_md5": ident, "different_md5_same_size": notident,
    "offset_rows": len(rows), "offset_bad": bad,
}
with open(os.path.join(WORK, "batch2_vs_batch1.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
