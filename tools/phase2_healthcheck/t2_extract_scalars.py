# -*- coding: utf-8 -*-
"""
T2a/T2b 标量提取: 每例从 NIfTI 头 + colon mask 物理几何计算采集/形态标量。
- 输入: hos_pairs_raw (mri_path_t) + data/masks_totalseg_mr_t/{pid}/colon.nii.gz
- 对齐 train_roi_dl aligned 路径: 以 mri_t 为锚构建 1mm 参考网格, mask nearest 重采样,
  bbox+15mm 带宽内 EDT → 5mm 环 (_ring_sdf 同款逻辑)
- 输出标量 (T2a 采集/缩放): fov_x/fov_y/fov_z (锚序列物理范围 mm), slice_thick (层厚 mm),
  bbox_lx/ly/lz (mask bbox 物理边长 mm), clamp_frac (64³ 裁剪框越界体素占比)
- (T2b 物理形态): vol_mm3 (本体 1mm 体积), z_span_mm (纵轴跨度), ring_ratio (5mm环/本体),
  ring_vol_mm3
- 输出 CSV: tools/phase2_healthcheck/outputs/t2_scalars.csv (UTF-8)
"""
import os, sys, glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CLIN = os.path.join(ROOT, "data", "hos_pairs_raw.csv")
MR_MASK_T = os.path.join(ROOT, "data", "masks_totalseg_mr_t")
OUT_DIR = os.path.join(ROOT, "tools", "phase2_healthcheck", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RING_MM = 5.0
PAD_BBOX = 5          # _mri_bbox pad
BAND = 15             # SDF_CLIP_MM 带宽
ROI = 64              # 裁剪边长 (train_roi_dl)


def _read_mask(pid):
    p = os.path.join(MR_MASK_T, str(pid), "colon.nii.gz")
    if not os.path.exists(p):
        return None
    return sitk.ReadImage(p)


def scalars_for(pid, t_path):
    out = {"patient_id": pid}
    if not (isinstance(t_path, str) and t_path and os.path.exists(t_path)):
        return out
    try:
        t_img = sitk.ReadImage(t_path)
    except Exception:
        return out
    sz, sp = t_img.GetSize(), t_img.GetSpacing()
    out["fov_x"], out["fov_y"], out["fov_z"] = [s * p for s, p in zip(sz, sp)]
    out["slice_thick"] = sp[2]

    mask = _read_mask(pid)
    if mask is None:
        return out
    try:
        m_arr = sitk.GetArrayFromImage(mask)  # [D,H,W]
    except Exception:
        return out
    if m_arr.sum() <= 0:
        return out

    # 1mm 参考网格 (以 t 为锚, 同 loader._mri_reference_grid)
    phys = np.array(t_img.GetSize()) * np.array(t_img.GetSpacing())
    new_size = np.ceil(phys / 1.0).astype(int).tolist()  # xyz
    ref = sitk.Image(new_size, sitk.sitkFloat32)
    ref.SetSpacing((1.0, 1.0, 1.0))
    ref.SetDirection(t_img.GetDirection())
    ref.SetOrigin(t_img.GetOrigin())

    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(ref)
    rf.SetInterpolator(sitk.sitkNearestNeighbor)
    rf.SetDefaultPixelValue(0.0)
    try:
        m_1mm = rf.Execute(mask)              # 本体在 1mm 网格
        m_1mm = (sitk.GetArrayFromImage(m_1mm) > 0).astype(np.uint8)  # [D,H,W]
    except Exception:
        return out
    if m_1mm.sum() <= 0:
        return out

    # bbox (pad=5, 同 _mri_bbox)
    idx = np.argwhere(m_1mm)
    d0, h0, w0 = idx.min(0)
    d1, h1, w1 = idx.max(0) + 1
    D, H, W = m_1mm.shape
    bd0, bd1 = max(d0 - PAD_BBOX, 0), min(d1 + PAD_BBOX, D)
    bh0, bh1 = max(h0 - PAD_BBOX, 0), min(h1 + PAD_BBOX, H)
    bw0, bw1 = max(w0 - PAD_BBOX, 0), min(w1 + PAD_BBOX, W)
    out["bbox_lx"] = bd1 - bd0      # 1mm 网格 → mm
    out["bbox_ly"] = bh1 - bh0
    out["bbox_lz"] = bw1 - bw0
    out["vol_mm3"] = float(m_1mm.sum())
    out["z_span_mm"] = float(d1 - d0)

    # clamp 贴边: 以 bbox 中心裁 64³, 越界体素占比 (训练时 clamp+零填充)
    cx, cy, cz = (bd0 + bd1) / 2.0, (bh0 + bh1) / 2.0, (bw0 + bw1) / 2.0
    half = ROI // 2
    x0, x1 = int(cx) - half, int(cx) + half
    y0, y1 = int(cy) - half, int(cy) + half
    z0, z1 = int(cz) - half, int(cz) + half
    total = ROI ** 3
    oob = (max(0, -x0) + max(0, x1 - D) + max(0, -y0) + max(0, y1 - H)
           + max(0, -z0) + max(0, z1 - W))
    out["clamp_frac"] = float(min(oob, total) / total)

    # 5mm 环: bbox+带宽内, mask 外的物理距离场 (同 _ring_sdf: 0<dist<=5 & ~body)
    d0e, d1e = max(d0 - BAND, 0), min(d1 + BAND, D)
    h0e, h1e = max(h0 - BAND, 0), min(h1 + BAND, H)
    w0e, w1e = max(w0 - BAND, 0), min(w1 + BAND, W)
    body = m_1mm[d0e:d1e, h0e:h1e, w0e:w1e]
    # EDT 输入应为"到 mask 的距离": 补集为前景 → 背景(mask 内)距离 0, mask 外为到边缘距离
    dist = distance_transform_edt((body == 0).astype(np.uint8))
    ring = ((dist > 0) & (dist <= RING_MM) & (body == 0)).astype(np.uint8)
    if ring.sum() <= 0:  # 极窄环采样抹掉 → 保留最近一层 (同 _ring_sdf 兜底)
        pos = dist[(body == 0) & (dist > 0)]
        if len(pos):
            ring = ((dist > 0) & (dist <= pos.min() + 1e-6) & (body == 0)).astype(np.uint8)
    ring_vol = float(ring.sum())
    out["ring_vol_mm3"] = ring_vol
    out["ring_ratio"] = ring_vol / out["vol_mm3"] if out["vol_mm3"] else np.nan
    return out


def main():
    clin = pd.read_csv(RAW_CLIN, encoding="utf-8-sig")
    cols = ["patient_id", "mri_path_t"]
    clin = clin[cols].dropna(subset=["mri_path_t"])
    rows = []
    n_ok = 0
    for _, r in clin.iterrows():
        pid = int(r["patient_id"])
        sc = scalars_for(pid, r["mri_path_t"])
        if {"vol_mm3", "ring_ratio", "bbox_lx"} <= set(sc):
            n_ok += 1
        rows.append(sc)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "t2_scalars.csv"), index=False, encoding="utf-8-sig")
    print(f"已落盘: {OUT_DIR}\\t2_scalars.csv  行数={len(df)} 完整标量={n_ok}")
    print("样例:")
    print(df.head(8).to_string(index=False))
    print("\n数值列统计:")
    num = df.select_dtypes(include=[np.number]).describe().T
    print(num.to_string())


if __name__ == "__main__":
    main()
