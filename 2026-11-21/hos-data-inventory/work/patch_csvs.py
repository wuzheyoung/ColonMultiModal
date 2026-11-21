# -*- coding: utf-8 -*-
"""同步 CSv: patient 1/2/4/5/6 的 CT 已删除。
- pairs/raw/派生全集: 清空 CT 路径/序列, ct_available=False
- CT 分支专用 CSV: 直接剔除这 5 行
- master_5fold: ct_available=0, 重算 modality_sig/stratum
用法: python patch_csvs.py [--apply]
"""
import glob
import os
import shutil
import sys
import time

import pandas as pd

DATA = r"d:\PyFile\ColonMultiModal\data"
WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"
AFFECTED = {1, 2, 4, 5, 6}
APPLY = "--apply" in sys.argv
STAMP = time.strftime("%Y%m%d_%H%M%S")
BK = os.path.join(DATA, f"_csv_backup_{STAMP}")

CT_BRANCH = ("ct_p", "ct_z", "branch_ct_")


def is_ct_branch(name):
    return any(k in name for k in CT_BRANCH)


def patch(df, name):
    mask = df["patient_id"].astype(int).isin(AFFECTED)
    if not mask.any():
        return df, 0, 0
    if is_ct_branch(name):
        n = int(mask.sum())
        return df.loc[~mask].reset_index(drop=True), n, 0
    for c in ("ct_path", "ct_path_plain", "ct_path_enhanced", "ct_series"):
        if c in df:
            df.loc[mask, c] = ""
    for c in ("ct_available", "ct_avail_plain", "ct_avail_enhanced"):
        if c in df:
            df[c] = df[c].astype(bool)
            df.loc[mask, c] = False
    return df, 0, int(mask.sum())


def patch_master(df):
    mask = df["patient_id"].astype(int).isin(AFFECTED)
    df.loc[mask, "ct_available"] = 0
    for i in df.index[mask]:
        sig = f"C0T{int(df.at[i,'mri_t'])}C{int(df.at[i,'mri_c'])}D{int(df.at[i,'mri_d'])}"
        df.at[i, "modality_sig"] = sig
        df.at[i, "stratum"] = f"{int(df.at[i,'label_T'])}_{int(df.at[i,'label_N'])}_{sig}"
    return df, 0, int(mask.sum())


def main():
    targets = sorted(glob.glob(os.path.join(DATA, "hos_*.csv")))
    targets.append(os.path.join(DATA, "master_5fold_356.csv"))
    log = []
    if APPLY:
        os.makedirs(BK, exist_ok=True)
    for f in targets:
        name = os.path.basename(f)
        df = pd.read_csv(f)
        if "patient_id" not in df:
            continue
        if name == "master_5fold_356.csv":
            new, dropped, patched = patch_master(df.copy())
        else:
            new, dropped, patched = patch(df.copy(), name)
        if dropped == 0 and patched == 0:
            continue
        log.append(f"{name}: rows {len(df)}->{len(new)}, dropped={dropped}, patched={patched}")
        print(log[-1])
        if APPLY:
            shutil.copy2(f, os.path.join(BK, name))
            new.to_csv(f, index=False, encoding="utf-8-sig")
    if APPLY:
        with open(os.path.join(WORK, f"patch_csvs_{STAMP}.log"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(log) + f"\nbackup={BK}\n")
        print("备份目录:", BK)
    else:
        print("[dry-run] 未写文件")


if __name__ == "__main__":
    main()
