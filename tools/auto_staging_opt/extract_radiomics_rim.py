# -*- coding: utf-8 -*-
"""extract_radiomics_rim.py — 阶段4: 原始影像 3D 形态 + 瘤周浸润环 (3mm/5mm) 放射组学
TRIPOD 1b: 区域定义只依赖分割掩码, 严禁使用标签信息。

区域: 1) 3D 瘤内 ROI (intra) 2) 2D 轴位向肠周形态学膨胀 3mm 环带 (rim_3mm) 3) 5mm 环带 (rim_5mm)。
特征: pyradiomics 缺失 → SimpleITK + scikit-image/scipy 稳定实现:
  - 3D 形态: 体积 mm3、最大径、球形度、扁平度、表面积体积比
  - 强度: 分位数(10/25/50/75/90)、偏度、峰度、均值、std、能量、熵
  - GLCM 纹理: 对比度、能量、同质性、相关性、ASM、熵 (2D 轴位合并)
输出: features/auto_staging_opt/radiomics_rim_features.csv (pid, seq, roi, 特征列)
评估在 stage4_radiomics_ablation.py。
"""
import os, sys, json, datetime, glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, marching_cubes
from multiprocessing import Pool

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

MRI_T = "data/Hos/_nifti/MRI"
MASK_T = "data/masks_totalseg_mr_t"
MASK_D = "data/masks_totalseg_mr_d"
REMAPPED = "data/masks_dwi_remapped"
OUT = "features/auto_staging_opt/radiomics_rim_features.csv"
LOG = "logs/auto_staging_opt/extract_radiomics_rim_log.json"

RIM_MM = {"rim_3mm": 3.0, "rim_5mm": 5.0}

def mask_path_for(seq, pid):
    if seq == "t":
        p = os.path.join(MASK_T, str(pid), "colon.nii.gz")
        return (p, "self") if os.path.exists(p) else (None, None)
    # d: 优先原生 mr_d, 否则 remapped
    p1 = os.path.join(MASK_D, f"{pid}.nii.gz")
    if os.path.exists(p1):
        return p1, "self"
    p2 = os.path.join(REMAPPED, str(pid), "colon.nii.gz")
    if os.path.exists(p2):
        return p2, "from_t"
    return None, None

def load_nii(p):
    img = sitk.ReadImage(p)
    arr = sitk.GetArrayFromImage(img)          # (Z,Y,X)
    sp = img.GetSpacing()                       # (x,y,z)
    return arr, (sp[2], sp[1], sp[0]), img.GetDirection()

def phys_dilate(mask3d, spacing, mm):
    """物理空间膨胀 mm 后减原掩码 → 环带掩码 (3D)。"""
    rz, ry, rx = spacing
    se = ndimage.generate_binary_structure(3, 1)
    se = ndimage.iterate_structure(se, 1)
    # 各向异性结构元近似: 每轴半径按 mm/spacing 换算
    nz = max(1, int(round(mm / rz)))
    ny = max(1, int(round(mm / ry)))
    nx = max(1, int(round(mm / rx)))
    se_aniso = np.zeros((2 * nz + 1, 2 * ny + 1, 2 * nx + 1), dtype=bool)
    zz, yy, xx = np.mgrid[-nz:nz + 1, -ny:ny + 1, -nx:nx + 1]
    se_aniso = (zz / nz) ** 2 + (yy / ny) ** 2 + (xx / nx) ** 2 <= 1.0
    dil = ndimage.binary_dilation(mask3d, structure=se_aniso)
    ring = dil & ~mask3d
    return ring

def roi_crop_2d_slices(mask3d, rim, spacing):
    """2D 轴位切片与环带的公共层: 对每层 ring>0 的 z, 返回 (intra_slice, rim_slice)。"""
    zs = np.where(rim.any(axis=(1, 2)))[0]
    return zs

def feats_3d_shape(mask3d, spacing):
    """3D 形态特征 (掩码体素) — 体素数×spacing 体积。"""
    v = mask3d.sum()
    vol_mm3 = float(v * spacing[0] * spacing[1] * spacing[2])
    props = regionprops(mask3d.astype(np.uint8), spacing=spacing)
    if props:
        pr = props[0]
        maxd = float(pr.axis_major_length)
        spher = float(pr.sphericity) if hasattr(pr, "sphericity") else 0.0
        sa = float(pr.surface_area) if hasattr(pr, "surface_area") else 0.0
        amin, amaj = pr.axis_minor_length, pr.axis_major_length
        flat = float(amin / max(amaj, 1e-9))
    else:
        maxd = spher = flat = sa = 0.0
    return {
        "vol_mm3": round(vol_mm3, 2),
        "max_diameter_mm": round(maxd, 2),
        "sphericity": round(spher, 4),
        "flatness": round(flat, 4),
        "surface_volume_ratio": round(float(sa / max(vol_mm3, 1e-9)), 5),
    }

def feats_intensity(roi, img):
    """ROI 内灰度强度特征 (2D 合并所有层)。"""
    vals = img[roi > 0]
    if vals.size == 0:
        return {}
    q = np.percentile(vals, [10, 25, 50, 75, 90])
    mu, sd = vals.mean(), vals.std()
    z = (vals - mu) / (sd + 1e-9)
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean()) - 3.0
    e = float((vals ** 2).sum())
    p = vals / (vals.sum() + 1e-9)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "int_q10": round(float(q[0]), 2), "int_q25": round(float(q[1]), 2),
        "int_median": round(float(q[2]), 2), "int_q75": round(float(q[3]), 2),
        "int_q90": round(float(q[4]), 2), "int_mean": round(mu, 2), "int_std": round(sd, 2),
        "int_skew": round(skew, 4), "int_kurt": round(kurt, 4),
        "int_energy": round(e, 2), "int_entropy": round(ent, 4),
    }

def feats_glcm_2d(img, roi, levels=32):
    """2D 轴位 GLCM (对比度/能量/同质性/相关性/ASM/熵), 合并所有层取均值。"""
    out = {}
    for name in ("contrast", "energy", "homogeneity", "correlation", "ASM"):
        out[f"glcm_{name}"] = []
    out["glcm_entropy"] = []
    zs = np.where(roi.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        return {}
    for z in zs:
        sl = img[z]
        m = roi[z] > 0
        if m.sum() < 5:
            continue
        minv, maxv = sl[m].min(), sl[m].max()
        if maxv - minv < 1e-6:
            continue
        q = np.round((sl - minv) / (maxv - minv) * (levels - 1)).astype(np.int64)
        q[~m] = -1
        # 只统计掩码内像素对 (忽略背景 -1): 用 roi 内局部 GLCM
        qm = q.copy()
        qm[~m] = levels  # 背景标签
        P = graycomatrix(qm, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=levels + 1, symmetric=True, normed=True)
        # 去掉背景行/列
        Pbg = P[:-1, :-1, :, :].sum(axis=(2, 3))
        if Pbg.sum() < 1e-9:
            continue
        Pbg = Pbg / Pbg.sum()
        for name in ("contrast", "energy", "homogeneity", "correlation", "ASM"):
            out[f"glcm_{name}"].append(float(graycoprops(Pbg[:, :, None, None], name)[0, 0]))
        p = Pbg
        out["glcm_entropy"].append(float(-(p * np.log(p + 1e-12)).sum()))
    if not out["glcm_contrast"]:
        return {}
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}

def process_one(args):
    seq, pid, mri_path = args
    try:
        mp, src = mask_path_for(seq, pid)
        if not mp:
            return None
        m_arr, m_sp, _ = load_nii(mp)
        mbin = m_arr > 0
        if mbin.sum() < 5:
            return None
        if not os.path.exists(mri_path):
            return None
        i_arr, i_sp, _ = load_nii(mri_path)
        # 重采样掩码到 MRI 网格
        if tuple(np.round(i_sp, 3)) != tuple(np.round(m_sp, 3)):
            # 用 SimpleITK Resample 重采样掩码到 MRI 空间
            mi = sitk.ReadImage(mp)
            mri = sitk.ReadImage(mri_path)
            ri = sitk.Resample(mi, mri, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mi.GetPixelID())
            m_arr = sitk.GetArrayFromImage(ri)
            mbin = m_arr > 0
        rows = []
        # intra (3D 全掩码)
        intra = mbin
        f3 = feats_3d_shape(intra, i_sp)
        fi = feats_intensity(intra, i_arr)
        fg = feats_glcm_2d(i_arr, intra)
        rows.append({"seq": seq, "pid": pid, "roi": "intra", "mask_source": src,
                     **f3, **fi, **fg})
        # rim 3mm/5mm
        for rname, rmm in RIM_MM.items():
            ring = phys_dilate(intra, i_sp, rmm)
            if ring.sum() < 5:
                continue
            f3r = feats_3d_shape(ring, i_sp)
            fir = feats_intensity(ring, i_arr)
            fgr = feats_glcm_2d(i_arr, ring)
            rows.append({"seq": seq, "pid": pid, "roi": rname, "mask_source": src,
                         **f3r, **fir, **fgr})
        return rows
    except Exception as e:
        return [{"seq": seq, "pid": pid, "roi": "ERROR", "mask_source": str(e)[:100]}]

def main():
    # 目标 pid 集: direct 队列中 t 与 d(self+from_t) 有影像的
    md = pd.read_csv("data/master_5fold_356.csv")
    dci = pd.read_csv("data/direct_cohort_ids.csv")
    direct = set(md.loc[md["patient_id"].isin(dci["pid"][dci["is_tgr"] == 0]), "patient_id"])
    raw = pd.read_csv("data/hos_pairs_raw.csv")
    jobs = []
    for seq in ("t", "d"):
        col = f"mri_path_{seq}"
        for pid in sorted(direct):
            sub = raw.loc[raw["patient_id"] == pid, col]
            if sub.empty or pd.isna(sub.iloc[0]):
                continue
            mp, src = mask_path_for(seq, pid)
            if mp and os.path.exists(sub.iloc[0]):
                jobs.append((seq, int(pid), sub.iloc[0]))
    print("jobs:", len(jobs))
    all_rows = []
    with Pool(8) as pool:
        for res in pool.imap_unordered(process_one, jobs):
            if res:
                all_rows.extend(res)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)
    print("rows:", len(df), "->", OUT)
    print("ERROR rows:", (df["roi"] == "ERROR").sum())
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump({"jobs": len(jobs), "rows": len(df),
                   "errors": int((df["roi"] == "ERROR").sum()),
                   "time": datetime.datetime.now().isoformat(timespec="seconds")}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
