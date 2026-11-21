# -*- coding: utf-8 -*-
"""
合并第二批 208 患者到 hos_pairs.csv (数据重构建)。

输入:
  - data/hos_pairs_raw.csv          第一批原始值(154 行)
  - data/Hos/结直肠癌多模态患者数据收集表9.29.xlsx   第二批临床+病理
  - data/Hos/_nifti/batch2/{编号}_{seq}.nii.gz       第二批 NIfTI

标签规则(用户确认):
  新表『病理学诊断』 = T-N-M 三位组合编码 (如 30x=T3N0Mx, 第一位T第二位N)
    label_T = 0 (pT1-2) / 1 (pT3-4)
    label_N = 0 (pN0) / 1 (pN+)

患者编号:
  第二批实际编号 1~208 -> patient_id = 编号 + OFFSET (OFFSET=200, 避开第一批 1~162)

序列映射(与 convert_batch2_to_nifti.py 一致):
  mri_path_t = {id}_t   mri_path_c = {id}_c(+增强MRI)   mri_path_d = {id}_d
  ct_path: 优先 {id}_z(增强CT c+) > {id}_p(平扫CT c)

用法:
  python tools/build_hos_batch2_csv.py [--dry-run] [--force]
  (重复执行有保护: 检测到第二批 patient_id 已在 raw 中会中止; 写前自动备份 .bak_时间戳)
"""
import os
import re
import sys
import warnings

warnings.simplefilter("ignore")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOS = os.path.join(BASE, "data", "Hos")
NII = os.path.join(HOS, "_nifti", "batch2")
XLSX = os.path.join(HOS, "结直肠癌多模态患者数据收集表9.29.xlsx")
RAW_CSV = os.path.join(BASE, "data", "hos_pairs_raw.csv")
OUT_CSV = os.path.join(BASE, "data", "hos_pairs.csv")
OUT_RAW = os.path.join(BASE, "data", "hos_pairs_raw.csv")

OFFSET = 200

CLINICAL_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
XLSX_MAP = {
    "age": "年龄", "sex": "性别", "ca199": "ca199", "cea": "癌胚抗原",
    "albumin": "白蛋白", "prealbumin": "前白蛋白", "hb": "血红蛋白",
}


def parse_less_than(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    m = re.match(r"^<([\d.]+)$", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_tnm(s):
    """'30x' -> (label_T, label_N, pT, pN)。解析失败返回 (-1, -1, nan, nan)。"""
    if s is None:
        return -1, -1, np.nan, np.nan
    s = str(s).strip()
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) >= 2:
        t, n = int(digits[0]), int(digits[1])
    elif len(digits) == 1:
        t, n = int(digits[0]), -1
    else:
        return -1, -1, np.nan, np.nan
    label_T = 0 if t in (1, 2) else 1
    label_N = 0 if n == 0 else (1 if n in (1, 2, 3) else -1)
    return label_T, label_N, float(t), float(n) if n >= 0 else np.nan


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印统计, 不写文件")
    ap.add_argument("--force", action="store_true",
                    help="检测到 patient_id 重复仍继续 (默认中止, 防止重复执行把第二批追加两次)")
    args = ap.parse_args()
    # 1. 读第一批 raw
    df1 = pd.read_csv(RAW_CSV)
    print(f"第一批 raw: {len(df1)} 行, patient_id {df1['patient_id'].min()}~{df1['patient_id'].max()}")

    # 2. 读新表
    xl = pd.read_excel(XLSX)
    print(f"新表 9.29: {len(xl)} 行, 编号 {xl['编号'].min()}~{xl['编号'].max()}")

    # 3. 生成第二批 rows
    rows = []
    parse_fail = []
    for _, r in xl.iterrows():
        aid = int(r["编号"])          # 第二批实际编号 1~208
        pid = aid + OFFSET            # 合并后的 patient_id

        # 序列路径 (nifti 文件名用实际编号 aid)
        mri_t = os.path.join(NII, f"{aid}_t.nii.gz")
        mri_c = os.path.join(NII, f"{aid}_c.nii.gz")
        mri_d = os.path.join(NII, f"{aid}_d.nii.gz")
        ct_z = os.path.join(NII, f"{aid}_z.nii.gz")
        ct_p = os.path.join(NII, f"{aid}_p.nii.gz")

        ct_path = ct_z if exists(ct_z) else (ct_p if exists(ct_p) else "")
        mri_path_t = mri_t if exists(mri_t) else ""
        mri_path_c = mri_c if exists(mri_c) else ""
        mri_path_d = mri_d if exists(mri_d) else ""

        ct_ok = bool(ct_path)
        mri_avail_t = bool(mri_path_t)
        mri_avail_c = bool(mri_path_c)
        mri_avail_d = bool(mri_path_d)
        mri_ok = mri_avail_t or mri_avail_c or mri_avail_d

        if not ct_ok and not mri_ok:
            continue  # 无影像 (101/131)

        # 标签
        label_T, label_N, pT, pN = parse_tnm(r.get("病理学诊断"))
        if label_T == -1:
            parse_fail.append((aid, r.get("病理学诊断")))

        # 住院号: 第二批原始 DICOM 已不存在, 只能取表内值, 标记 unverified
        hosp_id = str(r.get("住院号", "") or "").strip().upper().replace(" ", "")

        # 临床
        raw = {
            "patient_id": pid,
            "cohort": "hos",
            "hospital_id": hosp_id,
            "ct_patient_id": hosp_id,
            "mri_patient_id": hosp_id,
            "pair_status": "unverified",
            "ct_path": ct_path,
            "mri_path": mri_path_t,
            "mri_path_t": mri_path_t,
            "mri_path_c": mri_path_c,
            "mri_path_d": mri_path_d,
            "ct_available": ct_ok,
            "mri_available": mri_ok,
            "mri_available_t": mri_avail_t,
            "mri_available_c": mri_avail_c,
            "mri_available_d": mri_avail_d,
            "label_T": label_T,
            "label_N": label_N,
            "pT": pT,
            "pN": pN,
            "ct_series": os.path.basename(ct_path).replace(".nii.gz", "") if ct_path else "",
            "mri_series": f"{aid}_t" if mri_avail_t else "",
        }
        for col, xl_col in XLSX_MAP.items():
            if col == "sex":
                v = r.get(xl_col)
                s = "" if v is None else str(v).strip()
                raw[col] = 1.0 if s == "男" else (0.0 if s == "女" else np.nan)
            else:
                raw[col] = parse_less_than(r.get(xl_col))
        rows.append(raw)

    df2 = pd.DataFrame(rows)
    print(f"第二批有效: {len(df2)} 行 (无影像被剔除)")

    # 幂等保护: 第二批 pid 已在 raw 中 → 说明本脚本已执行过, 重复执行会追加重复患者
    dup = sorted(set(df2["patient_id"].astype(int)) & set(df1["patient_id"].astype(int)))
    if dup:
        msg = f"patient_id 已存在于 raw {len(dup)} 个 (疑似重复执行): {dup[:8]}"
        if not args.force:
            raise SystemExit(f"[abort] {msg}; 如确认要重建请加 --force (会先备份)")
        print(f"[warn] {msg} (--force 继续)")

    if parse_fail:
        print(f"⚠️ 病理学诊断解析失败 {len(parse_fail)} 例:")
        for a, v in parse_fail:
            print(f"    编号 {a}: {v!r}")

    # 4. 合并 raw
    df_raw = pd.concat([df1, df2], ignore_index=True)
    print(f"合并 raw: {len(df_raw)} 行")

    # 5. 临床特征 z-score (全量合并样本上统计)
    df = df_raw.copy()
    for c in CLINICAL_COLS:
        m = df[c].mean()
        s = df[c].std()
        if not np.isfinite(s) or s == 0:
            s = 1.0
        if not np.isfinite(m):
            m = 0.0
        df[c] = df[c].fillna(m)
        df[c] = (df[c] - m) / s

    # 6. 输出 (先备份; dry-run 不落盘)
    if args.dry_run:
        print("\n[dry-run] 未写文件。")
        return
    _backup(OUT_CSV)
    _backup(OUT_RAW)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    df_raw.to_csv(OUT_RAW, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"已生成: {OUT_CSV}  ({len(df)} 行)")
    print(f"已生成: {OUT_RAW}  ({len(df_raw)} 行)")
    print("双模态:", int((df["ct_available"] & df["mri_available"]).sum()),
          " CT-only:", int((df["ct_available"] & ~df["mri_available"]).sum()),
          " MRI-only:", int((~df["ct_available"] & df["mri_available"]).sum()))
    print("有 CT:", int(df["ct_available"].sum()), " 有 MRI:", int(df["mri_available"].sum()))
    print("label_T 分布:", df["label_T"].value_counts().to_dict())
    print("label_N 分布:", df["label_N"].value_counts().to_dict())


if __name__ == "__main__":
    main()
