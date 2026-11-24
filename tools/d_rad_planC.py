# -*- coding: utf-8 -*-
"""m05965 任务1a: 干净版 DWI 影像组学 d_rad (方案 C 掩码: 仅 self + 有效 from_t)。

基于 route3b_radiomics.py 的提取逻辑 (force2D 轴位 / d 跳过重采样 / body+ring / binCount=32),
但掩码来源改为方案 C 干净集合 (读 features/dwi_mask_remap_manifest.csv):
  - mask_source==self:  data/masks_totalseg_mr_d/{pid}/colon.nii.gz
  - mask_source==from_t: data/masks_dwi_remapped/{pid}/colon.nii.gz (has_valid_mask==1)
输出: features/d_rad_planC.csv (含 patient_id/mask_type/ok/label_T/label_N/fold + 107 特征列)
用法: pyrad39 python tools/d_rad_planC.py [--smoke]
"""
import os, time, argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PAIRS_RAW = "data/hos_pairs_raw.csv"
MASTER = "data/master_5fold_356.csv"
MANIFEST = "features/dwi_mask_remap_manifest.csv"
MASK_D = "data/masks_totalseg_mr_d"
MASK_REMAP = "data/masks_dwi_remapped"
PATH_D = "mri_path_d"
OUT_CSV = "features/d_rad_planC.csv"


def build_case_list(smoke=False):
    man = pd.read_csv(MANIFEST)
    clean = man[(man.mask_source.isin(["self", "from_t"])) & (man.has_valid_mask == 1)]
    pairs = pd.read_csv(PAIRS_RAW)
    master = pd.read_csv(MASTER)
    df = clean.rename(columns={"pid": "patient_id"})
    df = df.merge(pairs[["patient_id", PATH_D]], on="patient_id", how="left")
    df = df.merge(master[["patient_id", "label_T", "label_N", "fold"]], on="patient_id", how="left")
    df["mri_ok"] = df[PATH_D].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    df = df[df["mri_ok"]].reset_index(drop=True)
    print(f"[d_rad_planC] 干净 d 例: {len(df)} (self {int((df.mask_source=='self').sum())} + from_t {int((df.mask_source=='from_t').sum())})")
    if smoke:
        df = df.head(4)
    return df


def load_img_mask(row):
    import SimpleITK as sitk
    pid = int(row["patient_id"])
    if row["mask_source"] == "self":
        mask_p = os.path.join(MASK_D, str(pid), "colon.nii.gz")
    else:
        mask_p = os.path.join(MASK_REMAP, str(pid), "colon.nii.gz")
    if not os.path.exists(mask_p):
        return None, None
    img = sitk.ReadImage(row[PATH_D])
    mask = sitk.ReadImage(mask_p)
    ref = sitk.ResampleImageFilter()
    ref.SetReferenceImage(img)
    ref.SetInterpolator(sitk.sitkNearestNeighbor)
    ref.SetOutputPixelType(sitk.sitkUInt8)
    mask = ref.Execute(mask)
    return img, mask


def make_ring_mask(mask_img, in_mm=5.0, out_mm=10.0):
    import SimpleITK as sitk
    import scipy.ndimage as ndi
    arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
    sp = mask_img.GetSpacing()
    sxy, sz = sp[0], sp[2]
    d = ndi.distance_transform_edt((arr == 0).astype(np.uint8), sampling=(sz, sxy, sxy))
    ring = ((d > in_mm) & (d <= out_mm)).astype(np.uint8)
    out = sitk.GetImageFromArray(ring)
    out.CopyInformation(mask_img)
    return out


def extract_one(args):
    row_dict, mname = args
    import SimpleITK as sitk
    from radiomics import featureextractor
    row = pd.Series(row_dict)
    try:
        img, mask = load_img_mask(row)
        if img is None or mask is None:
            return {"patient_id": int(row["patient_id"]), "mask_type": mname, "ok": 0,
                    "label_T": int(row["label_T"]), "label_N": int(row["label_N"]), "fold": int(row["fold"])}
        m = mask if mname == "body" else make_ring_mask(mask)
        m_arr = sitk.GetArrayFromImage(m)
        src_arr = sitk.GetArrayFromImage(mask)
        if m_arr.sum() == 0 or m_arr.ndim < 2 or src_arr.sum() == 0:
            return {"patient_id": int(row["patient_id"]), "mask_type": mname, "ok": 0,
                    "label_T": int(row["label_T"]), "label_N": int(row["label_N"]), "fold": int(row["fold"])}
        extractor = featureextractor.RadiomicsFeatureExtractor()
        settings = {
            "normalize": True, "binCount": 32, "force2D": True, "force2Ddimension": 0,
            "preCrop": True, "padDistance": 5, "interpolator": "sitkLinear",
            "correctMask": True, "label": 1,
        }
        extractor.settings = settings
        for cls in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"):
            extractor.enableFeatureClassByName(cls)
        feats = extractor.execute(img, m)
        r = {"patient_id": int(row["patient_id"]), "mask_type": mname, "ok": 1,
             "label_T": int(row["label_T"]), "label_N": int(row["label_N"]), "fold": int(row["fold"])}
        for k, v in feats.items():
            if k.startswith("diagnostics_"):
                continue
            kk = k[len("original_"):] if k.startswith("original_") else k
            if isinstance(v, (int, float, np.integer, np.floating)):
                r[kk] = float(v)
            elif isinstance(v, np.ndarray) and v.ndim == 0:
                r[kk] = float(v)
        return r
    except Exception as e:
        return {"patient_id": int(row["patient_id"]), "mask_type": mname, "ok": 0,
                "label_T": int(row["label_T"]), "label_N": int(row["label_N"]),
                "fold": int(row["fold"]), "err": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    df = build_case_list(smoke=args.smoke)
    tasks = [(r.to_dict(), mn) for _, r in df.iterrows() for mn in ("body", "ring")]
    t0 = time.time()
    rows = []
    if args.smoke:
        for t in tasks:
            rows.append(extract_one(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(extract_one, tasks, chunksize=8)):
                rows.append(r)
                if (i + 1) % 100 == 0:
                    print(f"[d_rad_planC] {i+1}/{len(tasks)} tasks ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    ok = out["ok"] == 1
    print(f"[d_rad_planC] rows={len(out)} ok={int(ok.sum())} ({time.time()-t0:.0f}s) → {OUT_CSV}")


if __name__ == "__main__":
    main()
