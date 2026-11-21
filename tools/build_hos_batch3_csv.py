# -*- coding: utf-8 -*-
"""
A4: 合并第三批新数据到 hos_pairs.csv (数据重构建, 130 例待用户交付)。

输入(用户交付格式, 与 build_hos_batch2_csv.py 一致的管线):
  - data/hos_pairs_raw.csv               历史原始值(全量, 标准化前)
  - data/Hos/new_batch.xlsx | .csv       新表, 列: patient_id/pT/pN/age/sex/
                                          ca199/cea/albumin/prealbumin/hb/
                                          has_ct/has_mri_t/has_mri_c/has_mri_d
  - data/Hos/_nifti/CT/{pid}_ct.nii.gz   新 CT (has_ct 标 True 的必须有文件)
  - data/Hos/_nifti/MRI/{pid}_{t,c,d}.nii.gz  新 MRI (has_mri_* 标 True 必须有)

标签规则(与现有一致):
  pT: 1-2 -> label_T=0, 3-4 -> label_T=1
  pN: 0 -> label_N=0, 1/2/3 -> label_N=1, 其他 -> -1

用法:
  python tools/build_hos_batch3_csv.py --new-table data/Hos/new_batch.xlsx [--dry-run]
"""
import os
import re
import sys
import glob
import argparse
import warnings

warnings.simplefilter("ignore")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOS = os.path.join(BASE, "data", "Hos")
RAW_CSV = os.path.join(BASE, "data", "hos_pairs_raw.csv")
OUT_CSV = os.path.join(BASE, "data", "hos_pairs.csv")
OUT_RAW = os.path.join(BASE, "data", "hos_pairs_raw.csv")
CT_DIR = os.path.join(HOS, "_nifti", "CT")
MRI_DIR = os.path.join(HOS, "_nifti", "MRI")

CLINICAL_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]


def parse_less_than(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    m = re.match(r"^<([\d.]+)$", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_tnm(t, n):
    """原始 pT/pN 数值 -> (label_T, label_N, pT, pN)。解析失败返回 (-1,-1,nan,nan)。"""
    try:
        t = int(float(t))
        n = int(float(n))
    except Exception:
        return -1, -1, np.nan, np.nan
    label_T = 0 if t in (1, 2) else (1 if t in (3, 4) else -1)
    label_N = 0 if n == 0 else (1 if n in (1, 2, 3) else -1)
    return label_T, label_N, float(t), float(n)


def exists(p):
    return isinstance(p, str) and bool(p) and os.path.exists(p)


def _backup(path):
    """写前备份 (带时间戳, 不覆盖既有备份)。"""
    import shutil
    import time as _t
    if os.path.exists(path):
        dst = f"{path}.bak_{_t.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(path, dst)
        print(f"[backup] {os.path.basename(dst)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-table", required=True, help="新表路径 (xlsx/csv)")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计, 不写文件")
    args = ap.parse_args()

    if args.new_table.lower().endswith(".csv"):
        new = pd.read_csv(args.new_table)
    else:
        new = pd.read_excel(args.new_table)
    print(f"新表: {len(new)} 行, 列: {list(new.columns)}")

    # 1. 读历史 raw (标准化前原始值)
    df_raw = pd.read_csv(RAW_CSV)
    hist_ids = set(df_raw["patient_id"].astype(str))
    print(f"历史 raw: {len(df_raw)} 行 (patient_id {df_raw['patient_id'].min()}~{df_raw['patient_id'].max()})")

    # 2. 生成新 rows
    rows, skip = [], []
    for _, r in new.iterrows():
        pid = str(r.get("patient_id", "")).strip()
        if not pid or pid in hist_ids:
            skip.append((pid, "id 重复或空" if pid in hist_ids else "空 id"))
            continue
        # 影像路径
        ct = os.path.join(CT_DIR, f"{pid}_ct.nii.gz")
        mri_t = os.path.join(MRI_DIR, f"{pid}_t.nii.gz")
        mri_c = os.path.join(MRI_DIR, f"{pid}_c.nii.gz")
        mri_d = os.path.join(MRI_DIR, f"{pid}_d.nii.gz")
        ct_ok = exists(ct)
        mri_ok_t, mri_ok_c, mri_ok_d = exists(mri_t), exists(mri_c), exists(mri_d)
        if not ct_ok and not (mri_ok_t or mri_ok_c or mri_ok_d):
            skip.append((pid, "无影像文件"))
            continue
        # 标签
        label_T, label_N, pT, pN = parse_tnm(r.get("pT"), r.get("pN"))
        # 临床原始值
        raw = {
            "patient_id": pid,
            "cohort": "hos",
            "ct_path": ct if ct_ok else "",
            "mri_path": mri_t if mri_ok_t else "",
            "mri_path_t": mri_t if mri_ok_t else "",
            "mri_path_c": mri_c if mri_ok_c else "",
            "mri_path_d": mri_d if mri_ok_d else "",
            "ct_available": ct_ok,
            "mri_available": mri_ok_t or mri_ok_c or mri_ok_d,
            "mri_available_t": mri_ok_t,
            "mri_available_c": mri_ok_c,
            "mri_available_d": mri_ok_d,
            "label_T": label_T,
            "label_N": label_N,
            "pT": pT,
            "pN": pN,
            "ct_series": f"{pid}_ct" if ct_ok else "",
            "mri_series": f"{pid}_t" if mri_ok_t else "",
        }
        for col in CLINICAL_COLS:
            if col == "sex":
                v = r.get(col)
                s = "" if v is None else str(v).strip()
                raw[col] = 1.0 if s in ("男", "M", "m", "1") else (0.0 if s in ("女", "F", "f", "0") else np.nan)
            else:
                raw[col] = parse_less_than(r.get(col))
        rows.append(raw)

    if skip:
        print(f"⚠️ 跳过 {len(skip)} 例:")
        for pid, why in skip[:20]:
            print(f"    {pid}: {why}")

    df2 = pd.DataFrame(rows)
    print(f"新有效: {len(df2)} 行")

    # 3. 合并 raw + 全量 z-score (重建, 与 build_hos_batch2 一致)
    df_raw2 = pd.concat([df_raw, df2], ignore_index=True)
    df = df_raw2.copy()
    for c in CLINICAL_COLS:
        m = df[c].mean()
        s = df[c].std()
        if not np.isfinite(s) or s == 0:
            s = 1.0
        if not np.isfinite(m):
            m = 0.0
        df[c] = df[c].fillna(m)
        df[c] = (df[c] - m) / s

    print("=" * 60)
    print(f"合并后全量: {len(df)} 行 (历史 {len(df_raw)} + 新 {len(df2)})")
    print("双模态:", int((df["ct_available"] & df["mri_available"]).sum()),
          " CT-only:", int((df["ct_available"] & ~df["mri_available"]).sum()),
          " MRI-only:", int((~df["ct_available"] & df["mri_available"]).sum()))
    print("有 CT:", int(df["ct_available"].sum()), " 有 MRI:", int(df["mri_available"].sum()))
    print("label_T 分布:", df["label_T"].value_counts().to_dict())
    print("label_N 分布:", df["label_N"].value_counts().to_dict())
    print("label -1 (待过滤):", int((df["label_T"] == -1).sum()), "T /", int((df["label_N"] == -1).sum()), "N")

    if args.dry_run:
        print("\n[dry-run] 未写文件。")
        return

    _backup(OUT_CSV)
    _backup(OUT_RAW)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    df_raw2.to_csv(OUT_RAW, index=False, encoding="utf-8-sig")
    print(f"已生成: {OUT_CSV} ({len(df)} 行) + {OUT_RAW} ({len(df_raw2)} 行)")
    print("⚠️ 注意: hos_pairs.csv 已更新, 需重新生成 kfold csv 与缓存 (见 skill 文件读取/缓存章节)")


if __name__ == "__main__":
    main()
