"""
扫描 TCGA nifti，为每个患者挑选主 CT 体积（体素数最大者），生成配对 CSV。

输入:
  data/tcga/tcga_coad/_nifti/<patient>/*.nii.gz
  data/tcga/tcga_read/_nifti/<patient>/*.nii.gz

输出:
  data/tcga_ct_pairs.csv  (dataloader 直接读，复用 CMB 多模态列结构)
    列: patient_id, cohort, label, ct_path, mri_path,
        ct_available, mri_available, n_ct_voxels, ct_shape

TCGA 为单模态 CT: mri_path 恒空、mri_available=False。
label 取自 clinical 表的 AJCC 分期(四分类): I=0, II=1, III=2, IV=3。
  - patient_id 取前 12 字符与 clinical 的 Patient ID 匹配
  - 无分期记录或分期为空的患者 label 留空
stage 列保留原始 AJCC 字符串（如 STAGE IIA）便于核对。
"""

import os
import glob
import re
import numpy as np
import nibabel as nib
import pandas as pd

CLINICAL_TSV = os.path.join(
    "tcga", "metadata", "coadread_tcga_pan_can_atlas_2018_clinical_data.tsv")
STAGE_COL = "Neoplasm Disease Stage American Joint Committee on Cancer Code"


def _stage_to_label(stage):
    """AJCC 分期字符串 -> 四分类 label(I=0,II=1,III=2,IV=3)；无法解析返回 None。"""
    if not isinstance(stage, str):
        return None
    m = re.search(r"STAGE\s+(IV|III|II|I)", stage.upper())
    if not m:
        return None
    return {"I": 0, "II": 1, "III": 2, "IV": 3}[m.group(1)]


def _label_to_binary(label4):
    """四分类 label -> 二分类: 早期(I/II)=0, 晚期(III/IV)=1；None 透传。"""
    if label4 is None:
        return None
    return 0 if label4 <= 1 else 1


def _load_stage_map(base):
    """返回 {patient_id_12: (raw_stage, label)}。"""
    path = os.path.join(base, "data", CLINICAL_TSV)
    if not os.path.isfile(path):
        return {}
    c = pd.read_csv(path, sep="\t")
    out = {}
    for _, r in c.iterrows():
        pid = str(r["Patient ID"])
        raw = r.get(STAGE_COL, "")
        out[pid] = (raw if isinstance(raw, str) else "", _stage_to_label(raw))
    return out


DIAG_MIN_SLICES = 20   # 诊断级 CT 的最小层数（低于此视为定位片/无效重建）


def _list_volumes(nii_dir):
    """返回该患者目录下所有 nii.gz 的 (path, voxels, shape, n_slices)，按体素降序。"""
    vols = []
    for p in sorted(glob.glob(os.path.join(nii_dir, "*.nii.gz"))):
        try:
            shape = tuple(int(x) for x in nib.load(p).header.get_data_shape())
        except Exception:
            continue
        voxels = int(np.prod(shape)) if shape else 0
        n_slices = shape[2] if len(shape) >= 3 else 1
        vols.append((p, voxels, "x".join(map(str, shape)), n_slices))
    vols.sort(key=lambda x: x[1], reverse=True)
    return vols


def _pick_main_volume(nii_dir):
    """在患者目录下按体素数选最大的 nii.gz（定位片通常仅 1~few 层）。"""
    vols = _list_volumes(nii_dir)
    if not vols:
        return "", -1, ""
    p, voxels, shape, _ = vols[0]
    return p, voxels, shape


def build(base):
    nifti_root = {
        "coad": os.path.join(base, "data", "tcga", "tcga_coad", "_nifti"),
        "read": os.path.join(base, "data", "tcga", "tcga_read", "_nifti"),
    }
    stage_map = _load_stage_map(base)
    rows = []
    for cohort, root in nifti_root.items():
        if not os.path.isdir(root):
            continue
        for pid in sorted(os.listdir(root)):
            pdir = os.path.join(root, pid)
            if not os.path.isdir(pdir):
                continue
            ct_path, voxels, shape = _pick_main_volume(pdir)
            raw_stage, label = stage_map.get(pid[:12], ("", None))
            label_bin = _label_to_binary(label)
            rows.append({
                "patient_id": pid,
                "cohort": cohort,
                "stage": raw_stage,
                "label": label if label is not None else "",
                "label_bin": label_bin if label_bin is not None else "",
                "ct_path": ct_path,
                "mri_path": "",
                "ct_available": bool(ct_path),
                "mri_available": False,
                "n_ct_voxels": voxels if voxels > 0 else 0,
                "ct_shape": shape,
            })

    df = pd.DataFrame(rows)
    out = os.path.join(base, "data", "tcga_ct_pairs.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    n_label = (df["label"] != "").sum()
    print(f"患者配对(主体积) -> {out}  ({len(df)} 行, {n_label} 个有分期 label)")
    print(df[["patient_id", "cohort", "stage", "label", "ct_available", "ct_shape"]].to_string(index=False))

    build_series_table(base, stage_map, nifti_root)
    return df


def build_series_table(base, stage_map, nifti_root):
    """展开每个患者的所有诊断级 series（每个一行），供训练集同患者增强用。

    仅收录 >= DIAG_MIN_SLICES 层的 series（排除定位片/无效重建）。
    输出 data/tcga_ct_series.csv，列与主体积表一致，另加 n_slices。
    """
    rows = []
    for cohort, root in nifti_root.items():
        if not os.path.isdir(root):
            continue
        for pid in sorted(os.listdir(root)):
            pdir = os.path.join(root, pid)
            if not os.path.isdir(pdir):
                continue
            raw_stage, label = stage_map.get(pid[:12], ("", None))
            label_bin = _label_to_binary(label)
            for path, voxels, shape, n_slices in _list_volumes(pdir):
                if n_slices < DIAG_MIN_SLICES:
                    continue
                rows.append({
                    "patient_id": pid,
                    "cohort": cohort,
                    "stage": raw_stage,
                    "label": label if label is not None else "",
                    "label_bin": label_bin if label_bin is not None else "",
                    "ct_path": path,
                    "mri_path": "",
                    "ct_available": True,
                    "mri_available": False,
                    "n_ct_voxels": voxels,
                    "n_slices": n_slices,
                    "ct_shape": shape,
                })
    df = pd.DataFrame(rows)
    out = os.path.join(base, "data", "tcga_ct_series.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    n_pat = df["patient_id"].nunique()
    print(f"\n诊断级 series 展开表 -> {out}  ({len(df)} series / {n_pat} 患者)")
    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build(base)
