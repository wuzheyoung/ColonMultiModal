# -*- coding: utf-8 -*-
"""按第二批 DICOM 头核验结果修正数据：删除错人 NIfTI，剔除 382，清空 391 的 CT。无备份。"""
import glob
import os

import pandas as pd

BASE = r"d:\PyFile\ColonMultiModal"
DATA = os.path.join(BASE, "data")
B2 = os.path.join(DATA, "Hos", "_nifti", "batch2")

# 1) 删除错人 NIfTI
for name in ("182_p.nii.gz", "191_z.nii.gz"):
    p = os.path.join(B2, name)
    if os.path.exists(p):
        os.remove(p)
        print("removed", p)
    else:
        print("missing(already)", p)

# 2) 第二批状态映射
STATUS = {24: "typo_table", 61: "typo_table", 167: "ok_format", 174: "ok_format", 204: "ok_format"}
DROP = {382}
CT_CLEAR = {391}


def fix(df):
    if "patient_id" not in df.columns:
        return df
    df = df[~df.patient_id.isin(DROP)].copy()
    for c in df.columns:
        if c.startswith("ct_"):
            if c.startswith("ct_avail"):
                if df[c].dtype == bool:
                    df.loc[df.patient_id.isin(CT_CLEAR), c] = False
                else:
                    df.loc[df.patient_id.isin(CT_CLEAR), c] = 0
            else:
                df.loc[df.patient_id.isin(CT_CLEAR), c] = ""
    if "pair_status" in df.columns:
        for pid in df.patient_id:
            if pid <= 200:
                continue
            if pid in CT_CLEAR:
                df.loc[df.patient_id == pid, "pair_status"] = "ok"
                if "ct_patient_id" in df.columns:
                    df.loc[df.patient_id == pid, "ct_patient_id"] = ""
            else:
                st = STATUS.get(pid - 200, "ok")
                df.loc[df.patient_id == pid, "pair_status"] = st
                if "ct_patient_id" in df.columns:
                    df.loc[df.patient_id == pid, "ct_patient_id"] = df.loc[df.patient_id == pid, "hospital_id"]
                if "mri_patient_id" in df.columns:
                    df.loc[df.patient_id == pid, "mri_patient_id"] = df.loc[df.patient_id == pid, "hospital_id"]
    if "modality_sig" in df.columns:
        for i, r in df.iterrows():
            sig = f"C{int(r.get('ct_available', 0) or 0)}T{int(r.get('mri_t', 0) or 0)}" \
                  f"C{int(r.get('mri_c', 0) or 0)}D{int(r.get('mri_d', 0) or 0)}"
            df.at[i, "modality_sig"] = sig
            if "stratum" in df.columns:
                df.at[i, "stratum"] = f"{int(r['label_T'])}_{int(r['label_N'])}_{sig}"
    if "n_mri" in df.columns:
        df["n_mri"] = df[["mri_t", "mri_c", "mri_d"]].sum(axis=1).astype(int)
    return df


files = sorted(glob.glob(os.path.join(DATA, "hos_*.csv"))) + [os.path.join(DATA, "master_5fold_356.csv")]
for f in files:
    d0 = pd.read_csv(f)
    if "patient_id" not in d0.columns:
        continue
    d1 = fix(d0)
    d1.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"{os.path.basename(f):<30} {len(d0)} -> {len(d1)}")
