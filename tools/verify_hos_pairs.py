# -*- coding: utf-8 -*-
"""训练前数据自检：路径存在性 + 住院号配对状态。只读。

用法:
  python tools/verify_hos_pairs.py            # 校验 hos_pairs/master 及派生 CSV
  python tools/verify_hos_pairs.py --strict   # identity_conflict / 路径缺失 直接非零退出
"""
import argparse
import glob
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
PATH_COLS = ["ct_path", "ct_path_plain", "ct_path_enhanced",
             "mri_path", "mri_path_t", "mri_path_c", "mri_path_d"]
BAD_STATUS = {"identity_conflict"}


def check_file(path, strict):
    df = pd.read_csv(path)
    problems = []
    for c in PATH_COLS:
        if c not in df.columns:
            continue
        for pid, v in df[c].dropna().items():
            s = str(v).strip()
            if s and s.lower() != "nan" and not os.path.exists(s):
                problems.append(f"{os.path.basename(path)} pid={df.at[pid,'patient_id']} {c} 缺失: {s}")
    if "pair_status" in df.columns:
        for st in sorted(set(df.pair_status.dropna()) & BAD_STATUS):
            for pid in df.loc[df.pair_status == st, "patient_id"].tolist():
                problems.append(f"{os.path.basename(path)} pid={pid} pair_status={st}")
        if "hospital_id" in df.columns:
            ok = (df.pair_status == "ok") & df.ct_patient_id.notna() & (df.ct_patient_id != "")
            mismatch = df[ok & (df.ct_patient_id.str.upper() != df.hospital_id.str.upper())]
            for pid in mismatch.patient_id.tolist():
                problems.append(f"{os.path.basename(path)} pid={pid} ct_patient_id != hospital_id")
    return df, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    files = [os.path.join(DATA, "hos_pairs.csv"), os.path.join(DATA, "hos_pairs_raw.csv"),
             os.path.join(DATA, "master_5fold_356.csv")]
    files += sorted(glob.glob(os.path.join(DATA, "hos_branch_*.csv")))
    all_problems = []
    for f in files:
        if not os.path.exists(f):
            continue
        df, probs = check_file(f, args.strict)
        all_problems += probs
        status = df["pair_status"].value_counts().to_dict() if "pair_status" in df else {}
        print(f"{os.path.basename(f):<28} rows={len(df):<4} {status}")
    print("-" * 60)
    if all_problems:
        print(f"[FAIL] 发现 {len(all_problems)} 个问题:")
        for p in all_problems:
            print("  -", p)
        if args.strict:
            sys.exit(1)
    else:
        print("[OK] 路径与住院号配对检查全部通过")


if __name__ == "__main__":
    main()
