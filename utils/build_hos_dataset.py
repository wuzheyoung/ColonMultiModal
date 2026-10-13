# -*- coding: utf-8 -*-
"""
构建 data/Hos 的多模态训练集 (CT + MRI + 临床 + 病理 T/N 标签)。

流程:
  1. 将 data/Hos/CT 与 data/Hos/MRI 下每个 DICOM 目录转换为 NIfTI
     (SimpleITK ImageSeriesReader, 输出到 data/Hos/_nifti/{CT,MRI}/)
  2. 读取 结直肠癌多模态患者数据收集表.xlsx
  3. 按「编号」列与 CT/MRI 目录对应，生成 data/hos_pairs.csv:
       patient_id, cohort, ct_path, mri_path, ct_available, mri_available,
       age, sex, ca199, cea, albumin, prealbumin, hb,      # 临床特征(z-score)
       label_T, label_N, pT, pN                            # 病理标签
  4. 同时输出 data/hos_pairs_raw.csv (临床特征原始值, 便于复查)

序列选择(用户确认过的命名规则):
  CT : 优先 z(增强) > p(平扫) > 特殊名(如 1_QG12455983, DE_Abd_NC 平扫)
  MRI: 优先 t(T2)   > c(CE增强) > d(DWI)

标签:
  label_T: 病理T 二分类  pT1-2 -> 0, pT3-4 -> 1, 缺失 -> -1
  label_N: 病理N 二分类  pN0   -> 0, pN+  -> 1, 缺失 -> -1

用法:
  python utils/build_hos_dataset.py [--no-convert]
"""
import os
import re
import sys
import glob
import argparse
import warnings

warnings.simplefilter("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import SimpleITK as sitk

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOS = os.path.join(BASE, "data", "Hos")
CT_DIR = os.path.join(HOS, "CT")
MRI_DIR = os.path.join(HOS, "MRI")
NIFTI_DIR = os.path.join(HOS, "_nifti")
XLSX = os.path.join(HOS, "结直肠癌多模态患者数据收集表.xlsx")

CT_PRIORITY = ["z", "p", "other"]
MRI_PRIORITY = ["t", "c", "d"]

CLINICAL_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
XLSX_CLINICAL_MAP = {
    "age": "年龄", "sex": "性别", "ca199": "ca199",
    "cea": "癌胚抗原", "albumin": "白蛋白", "prealbumin": "前白蛋白", "hb": "血红蛋白",
}


def convert_dir_to_nifti(dcm_dir, out_path):
    """单个 DICOM 目录 -> NIfTI。返回 True 成功。"""
    reader = sitk.ImageSeriesReader()
    sids = reader.GetGDCMSeriesIDs(dcm_dir)
    if not sids:
        return False
    files = reader.GetGDCMSeriesFileNames(dcm_dir, sids[0])
    if not files:
        return False
    reader.SetFileNames(files)
    img = reader.Execute()
    sitk.WriteImage(img, out_path)
    return True


def is_valid_gzip(path):
    """快速校验 gzip 流完整性（读到 EOF 并触发 CRC 检查）。"""
    import gzip
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def convert_all():
    os.makedirs(os.path.join(NIFTI_DIR, "CT"), exist_ok=True)
    os.makedirs(os.path.join(NIFTI_DIR, "MRI"), exist_ok=True)
    done, fail = 0, []
    for root, sub in [(CT_DIR, "CT"), (MRI_DIR, "MRI")]:
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            if not any(f.lower().endswith(".dcm") for f in os.listdir(d)):
                continue
            out = os.path.join(NIFTI_DIR, sub, name + ".nii.gz")
            if os.path.exists(out) and is_valid_gzip(out):
                continue
            if os.path.exists(out):
                # 损坏/不完整文件（如转换被中断残留）：删除后重新转换
                os.remove(out)
                print(f"[reconvert] 损坏文件: {out}")
            try:
                if convert_dir_to_nifti(d, out):
                    done += 1
                    print(f"[convert] {sub}/{name} -> {out}")
                else:
                    fail.append(f"{sub}/{name}")
            except Exception as e:
                fail.append(f"{sub}/{name} ({e})")
    print(f"转换完成: {done} 个成功, {len(fail)} 个失败")
    if fail:
        print("失败列表:", fail)
    return fail


def parse_dir_type(name):
    """从目录名解析 编号 与 类型。返回 (int编号, 类型 或 None)。"""
    m = re.match(r"^(\d+)_([a-zA-Z]+)$", name)
    if m:
        return int(m.group(1)), m.group(2).lower()
    m = re.match(r"^(\d+)_", name)  # 特殊名: 1_QG12455983 -> 编号1, 类型 other
    if m:
        return int(m.group(1)), "other"
    return None, None


def pick_series(dir_root, priority):
    """每个编号选择优先级最高的目录名。返回 {编号: 目录名}。"""
    avail = {}
    for name in sorted(os.listdir(dir_root)):
        d = os.path.join(dir_root, name)
        if not os.path.isdir(d) or not any(f.lower().endswith(".dcm") for f in os.listdir(d)):
            continue
        pid, typ = parse_dir_type(name)
        if pid is None:
            continue
        avail.setdefault(pid, []).append((typ, name))
    picked = {}
    for pid, lst in avail.items():
        # 按优先级排序: 数字小优先; other 放最后
        lst.sort(key=lambda x: priority.index(x[0]) if x[0] in priority else 99)
        picked[pid] = lst[0][1]
    return picked


def collect_mri_seqs(dir_root):
    """每个编号收集全部可用序列目录。返回 {编号: {seq: 目录名}}。"""
    avail = {}
    for name in sorted(os.listdir(dir_root)):
        d = os.path.join(dir_root, name)
        if not os.path.isdir(d) or not any(f.lower().endswith(".dcm") for f in os.listdir(d)):
            continue
        pid, typ = parse_dir_type(name)
        if pid is None or typ in (None, "other"):
            continue
        avail.setdefault(pid, {})[typ] = name
    return avail


def parse_less_than(v):
    """'<2' -> 2.0 ; 其他数值字符串 -> float ; 无法解析 -> nan。"""
    if v is None:
        return np.nan
    s = str(v).strip()
    m = re.match(r"^<([\d.]+)$", s)
    if m:
        return float(m.group(1))
    try:
        f = float(s)
        return f
    except (ValueError, TypeError):
        return np.nan


def build_csv(fail_ok=False):
    ct_picked = pick_series(CT_DIR, CT_PRIORITY)
    mri_seqs_map = collect_mri_seqs(MRI_DIR)

    xl = pd.read_excel(XLSX)
    xl["编号"] = xl["编号"].astype(float)

    rows_raw = []
    for _, r in xl.iterrows():
        pid = r["编号"]
        if pd.isna(pid):
            continue
        pid = int(pid)
        ct_name = ct_picked.get(pid)
        mri_seqs = mri_seqs_map.get(pid, {})

        ct_path = os.path.join(NIFTI_DIR, "CT", ct_name + ".nii.gz") if ct_name else ""
        mri_path_t = os.path.join(NIFTI_DIR, "MRI", mri_seqs["t"] + ".nii.gz") if "t" in mri_seqs else ""
        mri_path_c = os.path.join(NIFTI_DIR, "MRI", mri_seqs["c"] + ".nii.gz") if "c" in mri_seqs else ""
        mri_path_d = os.path.join(NIFTI_DIR, "MRI", mri_seqs["d"] + ".nii.gz") if "d" in mri_seqs else ""
        mri_avail_t = bool(mri_path_t) and os.path.exists(mri_path_t)
        mri_avail_c = bool(mri_path_c) and os.path.exists(mri_path_c)
        mri_avail_d = bool(mri_path_d) and os.path.exists(mri_path_d)
        ct_ok = bool(ct_name) and os.path.exists(ct_path)
        mri_ok = mri_avail_t or mri_avail_c or mri_avail_d
        if not ct_ok and not mri_ok:
            continue  # 无任何影像的病例剔除

        # 病理 T/N 标签
        pT = r["病理T"]
        pN = r["病理N"]
        label_T = -1
        label_N = -1
        if pd.notna(pT):
            pT = float(pT)
            label_T = 0 if pT in (1.0, 2.0) else 1
        if pd.notna(pN):
            pN = float(pN)
            label_N = 0 if pN == 0.0 else 1

        raw = {
            "patient_id": pid,
            "cohort": "hos",
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
            "ct_series": ct_name or "",
            "mri_series": mri_seqs.get("t", "") or "",
        }
        for col, xl_col in XLSX_CLINICAL_MAP.items():
            if col == "sex":
                v = r.get(xl_col)
                s = "" if v is None else str(v).strip()
                raw[col] = 1.0 if s == "男" else (0.0 if s == "女" else np.nan)
            else:
                raw[col] = parse_less_than(r.get(xl_col))
        rows_raw.append(raw)

    df_raw = pd.DataFrame(rows_raw)

    # 保留全部有影像样本: 双模态 / CT-only / MRI-only 都用于训练
    # （缺失模态在训练时填零张量 + imputer 特征补全）
    df = df_raw.copy()
    df = df.reset_index(drop=True)

    # 临床特征填充 + z-score 标准化（在全量有影像样本上统计）
    df_clin = df[CLINICAL_COLS].copy()
    for c in CLINICAL_COLS:
        m = df_clin[c].mean()
        s = df_clin[c].std()
        if not np.isfinite(s) or s == 0:
            s = 1.0
        df_clin[c] = df_clin[c].fillna(m)
        df[c] = (df_clin[c] - m) / s

    out_path = os.path.join(BASE, "data", "hos_pairs.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    raw_path = os.path.join(BASE, "data", "hos_pairs_raw.csv")
    df_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    print(f"已生成: {out_path}  ({len(df)} 行, 双模态/CT-only/MRI-only)")
    print(f"已生成: {raw_path}  ({len(df_raw)} 行, 有 CT 或 MRI)")
    print("双模态(CT+MRI):", int((df["ct_available"] & df["mri_available"]).sum()),
          " CT-only:", int((df["ct_available"] & ~df["mri_available"]).sum()),
          " MRI-only:", int((~df["ct_available"] & df["mri_available"]).sum()))
    print("有 CT 例数:", int(df["ct_available"].sum()), " 有 MRI 例数:", int(df["mri_available"].sum()))
    print("label_T 分布:", df["label_T"].value_counts().to_dict())
    print("label_N 分布:", df["label_N"].value_counts().to_dict())
    print("序列选择情况(CT):", df["ct_series"].str.rsplit("_", n=1).str[-1].value_counts().to_dict())
    print("序列选择情况(MRI):", df["mri_series"].str.rsplit("_", n=1).str[-1].value_counts().to_dict())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-convert", action="store_true", help="跳过 DICOM->NIfTI 转换")
    args = ap.parse_args()
    if not args.no_convert:
        convert_all()
    build_csv()
    print("DONE")

