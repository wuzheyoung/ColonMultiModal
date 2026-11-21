# -*- coding: utf-8 -*-
"""只读扫描 G:\数据\第二批 的 DICOM 头，按编号核验第二批住院号。"""
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, r"d:\PyFile\ColonMultiModal\tools")
import hos_identity as hi  # noqa: E402

RAW = r"G:\数据\第二批"
HOS = r"d:\PyFile\ColonMultiModal\data\Hos"
OFF = {"1-100": 0, "101-200": 100, "201-": 200}
PAT = re.compile(r"^(\d+)(c\+|c|\+|d|t)?$")
OUT = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work\batch2_raw_ids.json"

tbl = pd.read_excel(os.path.join(HOS, "结直肠癌多模态患者数据收集表9.29.xlsx"), dtype=str)
tab = {}
for _, r in tbl.iterrows():
    try:
        aid = int(float(str(r["编号"]).strip()))
    except (ValueError, TypeError):
        continue
    tab[aid] = {"hid": hi.norm(r.get("住院号")), "name": str(r.get("姓名", "")).strip(),
                "imaging": str(r.get("影像信息", "")).strip()}

recs = {}
empty, bad = [], []
for seg, off in OFF.items():
    segdir = os.path.join(RAW, seg)
    if not os.path.isdir(segdir):
        continue
    for name in sorted(os.listdir(segdir), key=lambda s: (int(re.match(r"\d+", s).group()) if re.match(r"\d+", s) else 9999, s)):
        d = os.path.join(segdir, name)
        if not os.path.isdir(d):
            continue
        m = PAT.match(name)
        if not m:
            bad.append(f"{seg}/{name}")
            continue
        aid = int(m.group(1)) + off
        seq = m.group(2) or "?"
        pid = hi.patient_id_of_dir(d)
        recs.setdefault(aid, {})[seq] = pid
        if not pid:
            empty.append(f"{seg}/{name}")

rows = []
for aid in sorted(recs):
    ids = {v for v in recs[aid].values() if v}
    t = tab.get(aid, {})
    rows.append({
        "编号": aid, "表住院号": t.get("hid", ""), "表姓名": t.get("name", ""),
        "影像住院号": "/".join(sorted(ids)), "一致": len(ids) == 1 and (not t.get("hid") or next(iter(ids)) == t.get("hid")),
        "键": ";".join(f"{k}={v}" for k, v in sorted(recs[aid].items())),
        "影像信息": t.get("imaging", ""),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT.replace(".json", ".csv"), index=False, encoding="utf-8-sig")
json.dump({"raw_dir": RAW, "records": {str(k): v for k, v in recs.items()}}, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print(f"扫描目录 {sum(len(v) for v in recs.values())} 个序列目录, 覆盖编号 {len(recs)} 个")
print(f"无 DICOM 头的目录 {len(empty)}: {empty[:10]}")
print(f"无法解析命名 {len(bad)}: {bad[:10]}")
print(f"多住院号冲突(同编号内) {int((df['影像住院号'].str.contains('/')).sum())}")
print(f"与 9.29 表不一致 {int((~df['一致']).sum())}")
print(f"表中有、原始目录无: {sorted(set(tab) - set(recs))}")
print(f"原始目录有、表中无: {sorted(set(recs) - set(tab))}")
print("\n不一致明细:")
print(df[~df["一致"]].to_string(index=False))
