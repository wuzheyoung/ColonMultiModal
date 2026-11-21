# -*- coding: utf-8 -*-
"""把 hospital_id / ct_patient_id / mri_patient_id / pair_status 追加到全部 CSV（不改其他值）。"""
import glob
import os
import shutil
import sys
import time

import pandas as pd

sys.path.insert(0, r"d:\PyFile\ColonMultiModal\tools")
import hos_identity as hi  # noqa: E402

BASE = r"d:\PyFile\ColonMultiModal"
DATA = os.path.join(BASE, "data")
HOS = os.path.join(DATA, "Hos")
BK = os.path.join(DATA, "_csv_backup_20261121_114432_pre_rebuild")
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT_BK = os.path.join(DATA, f"_csv_backup_{STAMP}_identity")
COLS = ["hospital_id", "ct_patient_id", "mri_patient_id", "pair_status"]

ident, _ = hi.load_table_identity()
b2 = pd.read_excel(os.path.join(HOS, "结直肠癌多模态患者数据收集表9.29.xlsx"), dtype=str)
b2_map = {}
for _, r in b2.iterrows():
    try:
        aid = int(float(str(r["编号"]).strip()))
    except (ValueError, TypeError):
        continue
    b2_map[aid] = str(r.get("住院号", "") or "").strip().upper().replace(" ", "")


def identity_of(pid):
    pid = int(pid)
    if pid > 200:
        hid = b2_map.get(pid - 200, "")
        return {"hospital_id": hid, "ct_patient_id": hid, "mri_patient_id": hid,
                "pair_status": "unverified"}
    hid = ident.get(pid, {}).get("hospital_id", "")
    return {"hospital_id": hid, "ct_patient_id": "", "mri_patient_id": "", "pair_status": ""}


def put_cols(df, mapping):
    for c in COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    for c in COLS:
        df[c] = [mapping.get(int(p), {}).get(c, "") for p in df["patient_id"]]
    order = list(df.columns)
    for c in COLS:
        order.remove(c)
    pos = order.index("cohort") + 1 if "cohort" in order else len(order)
    order[pos:pos] = COLS
    return df[order]


def main():
    # 1) 还原被重建覆盖的 3 个文件
    for name in ("hos_pairs.csv", "hos_pairs_raw.csv", "master_5fold_356.csv"):
        shutil.copy2(os.path.join(BK, name), os.path.join(DATA, name))
    os.makedirs(OUT_BK, exist_ok=True)

    # 2) pairs: 逐行走 DICOM 头核验
    pairs_path = os.path.join(DATA, "hos_pairs.csv")
    pairs = pd.read_csv(pairs_path)
    mapping = {}
    for _, r in pairs.iterrows():
        pid = int(r["patient_id"])
        m = identity_of(pid)
        if pid <= 200:
            def clean(v):
                s = str(v).strip() if v is not None else ""
                return "" if s.lower() in ("nan", "none", "") else s

            ct_series = clean(r.get("ct_series"))
            mri_series = clean(r.get("mri_series"))
            if not ct_series:
                for c in ("ct_path", "ct_path_plain", "ct_path_enhanced"):
                    v = clean(r.get(c))
                    if v.endswith(".nii.gz"):
                        ct_series = os.path.basename(v)[:-7]
                        break
            if not mri_series:
                for c in ("mri_path_t", "mri_path_c", "mri_path_d"):
                    v = clean(r.get(c))
                    if v.endswith(".nii.gz"):
                        mri_series = os.path.basename(v)[:-7]
                        break
            ct_pid = hi.patient_id_of_dir(os.path.join(HOS, "CT", ct_series)) if ct_series else ""
            mri_pid = hi.patient_id_of_dir(os.path.join(HOS, "MRI", mri_series)) if mri_series else ""
            m.update({"ct_patient_id": ct_pid, "mri_patient_id": mri_pid,
                      "pair_status": hi.classify(ct_pid, mri_pid, m["hospital_id"])})
        mapping[pid] = m

    log = []
    for f in sorted(glob.glob(os.path.join(DATA, "hos_*.csv"))) + [os.path.join(DATA, "master_5fold_356.csv")]:
        name = os.path.basename(f)
        df = pd.read_csv(f)
        if "patient_id" not in df.columns:
            continue
        shutil.copy2(f, os.path.join(OUT_BK, name))
        new = put_cols(df, mapping)
        new.to_csv(f, index=False, encoding="utf-8-sig")
        log.append(f"{name}: rows={len(new)} {new['pair_status'].value_counts().to_dict()}")
        print(log[-1])
    with open(os.path.join(BASE, "2026-11-21", "hos-data-inventory", "work", f"identity_{STAMP}.log"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + f"\nbackup={OUT_BK}\n")
    print("backup:", OUT_BK)


if __name__ == "__main__":
    main()
