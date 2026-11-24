# -*- coding: utf-8 -*-
"""m05943 两组只读诊断 (不写任何文件):
1) 82 例 from_t 失败真实原因: n_vox==0 vs 1-49 分布; n_vox==0 例检查原生 mr_t 有效性与物理包围盒交集。
2) 三级兜底可行性模拟 (仅内存): from_t 失败例 → remap from_c → ①dwi>0 前景交集 ②剔除 |dx|>50mm 连通域 → 质检。
"""
import os, json
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MR_T = "data/masks_totalseg_mr_t"
MR_C = "data/masks_totalseg_mr_c"
RAW = "data/hos_pairs_raw.csv"
SUMMARY = "logs/dwi_mask_remap_summary.csv"
MIN_VOX = 50


def mask_ok(p):
    if not os.path.exists(p):
        return False
    try:
        a = sitk.GetArrayFromImage(sitk.ReadImage(p))
        return int(a.sum()) >= MIN_VOX
    except Exception:
        return False


def bbox_mm(img, arr):
    """物理包围盒 (min..max 三轴, mm)。"""
    nz = np.nonzero(arr)
    if len(nz[0]) == 0:
        return None
    pts = []
    for z, y, x in zip(*nz):
        pts.append(img.TransformIndexToPhysicalPoint((int(x), int(y), int(z))))
    pts = np.array(pts)
    return [float(pts[:, i].min()) for i in range(3)], [float(pts[:, i].max()) for i in range(3)]


def inter_vol(bb1, bb2):
    """两物理包围盒交集体积 (mm^3)。"""
    lo = np.maximum(bb1[0], bb2[0]); hi = np.minimum(bb1[1], bb2[1])
    d = hi - lo
    if np.any(d <= 0):
        return 0.0
    return float(np.prod(d))


def centroid_mm(img, arr):
    nz = np.nonzero(arr)
    if len(nz[0]) == 0:
        return None
    pts = []
    for z, y, x in zip(*nz):
        pts.append(img.TransformIndexToPhysicalPoint((int(x), int(y), int(z))))
    return np.array(pts).mean(axis=0)


def main():
    sm = pd.read_csv(SUMMARY)
    raw = pd.read_csv(RAW).set_index("patient_id")
    # ---- 诊断 1: from_t 失败 82 例 ----
    fail = sm[(sm.mask_source == "from_t") & (sm.has_valid_mask == 0)]
    print(f"== 诊断1: from_t 失败 {len(fail)} 例 (n_vox 分布) ==")
    nv = fail["n_vox"]
    print(f"  n_vox==0: {(nv==0).sum()} | 1-49: {((nv>0)&(nv<50)).sum()}")
    zeros = fail[nv == 0]["pid"].astype(int).tolist()
    # n_vox==0 例: 原生 mr_t 有效性与物理包围盒
    t_ok_cnt, t_missing_cnt, no_overlap, overlap_ok = 0, 0, 0, 0
    zero_details = []
    for pid in zeros[:40]:
        tp = os.path.join(MR_T, str(pid), "colon.nii.gz")
        t_ok = mask_ok(tp)
        mp = raw.loc[pid, "mri_path_d"]
        d_ok = isinstance(mp, str) and os.path.exists(mp)
        ov = None
        if t_ok and d_ok:
            t_img = sitk.ReadImage(tp); t_arr = sitk.GetArrayFromImage(t_img)
            d_img = sitk.ReadImage(mp); d_arr = sitk.GetArrayFromImage(d_img)
            t_bb = bbox_mm(t_img, t_arr); d_bb = bbox_mm(d_img, d_arr > 0)
            ov = inter_vol(t_bb, d_bb) if d_bb else 0.0
            no_overlap += (ov == 0)
            overlap_ok += (ov > 0)
            zero_details.append({"pid": pid, "t_vox": int(t_arr.sum()), "d_vox_total": int((d_arr > 0).sum()),
                                "t_bbox": [[round(v,1) for v in t_bb[0]], [round(v,1) for v in t_bb[1]]],
                                "d_bbox": [[round(v,1) for v in d_bb[0]], [round(v,1) for v in d_bb[1]]],
                                "inter_mm3": round(ov, 0)})
        else:
            t_missing_cnt += (not t_ok)
        if t_ok and d_ok: t_ok_cnt += 1
    print(f"  抽查 n_vox==0 例: t_mask_ok={t_ok_cnt} d_exists={t_ok_cnt} (无交集={no_overlap} 有交集={overlap_ok})")

    # ---- 诊断 2: 三级兜底模拟 (from_t 失败 且 direct_tcd 内) ----
    tcd = set()
    for _, r in sm.iterrows():
        pid = int(r.pid)
        if pid not in raw.index:
            continue
        if all(isinstance(raw.loc[pid, c], str) and os.path.exists(raw.loc[pid, c])
               for c in ("mri_path_t", "mri_path_c", "mri_path_d")):
            tcd.add(pid)
    fail_tcd = fail[fail["pid"].astype(int).isin(tcd)]
    print(f"== 诊断2: from_t 失败且 direct_tcd 内 {len(fail_tcd)} 例, 模拟清洗版 from_c 兜底 ==")
    rescued = []
    for pid in fail_tcd["pid"].astype(int):
        c_path = os.path.join(MR_C, str(pid), "colon.nii.gz")
        mp = raw.loc[pid, "mri_path_d"]
        if not os.path.exists(c_path) or not isinstance(mp, str) or not os.path.exists(mp):
            continue
        mimg = sitk.ReadImage(c_path)
        vimg = sitk.ReadImage(mp)
        res = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                            vimg.GetOrigin(), vimg.GetSpacing(), vimg.GetDirection(), 0, sitk.sitkUInt8)
        ra = sitk.GetArrayFromImage(res)
        va = sitk.GetArrayFromImage(vimg).astype(np.float32)
        # ① 与 DWI 前景交集 (dwi>0)
        ra = (ra > 0) & (va > 0)
        if ra.sum() == 0:
            continue
        # ② 连通域, 剔除质心 |dx|>50mm 的域
        lab, nlab = ndimage.label(ra, structure=np.ones((3, 3, 3)))
        keep = np.zeros_like(ra, dtype=bool)
        sz = vimg.GetSize()
        c_vol = vimg.TransformIndexToPhysicalPoint((sz[0] // 2, sz[1] // 2, sz[2] // 2))
        for li in range(1, nlab + 1):
            m_i = lab == li
            cm = centroid_mm(vimg, m_i)
            if cm is None:
                continue
            dx = cm[0] - c_vol[0]
            if abs(dx) > 50.0:
                continue
            keep |= m_i
        if keep.sum() < MIN_VOX:
            continue
        # 质检
        cm = centroid_mm(vimg, keep)
        shift = float(np.linalg.norm(cm - np.array(c_vol))) if cm is not None else None
        d_xyz = (cm - np.array(c_vol)) if cm is not None else None
        nz = np.nonzero(keep)
        nz_all = va[va > 0]
        ratio = float(np.median(va[nz])) / float(np.median(nz_all)) if len(nz_all) else None
        rescued.append({"pid": int(pid), "n_vox": int(keep.sum()),
                        "shift_mm": round(shift, 1) if shift else None,
                        "dx": round(float(d_xyz[0]), 1) if d_xyz is not None else None,
                        "dy": round(float(d_xyz[1]), 1) if d_xyz is not None else None,
                        "dz": round(float(d_xyz[2]), 1) if d_xyz is not None else None,
                        "mask_ratio": round(ratio, 2) if ratio else None})
    print(f"  清洗后救回 {len(rescued)}/{len(fail_tcd)} 例 (体素>=50)")
    if rescued:
        df = pd.DataFrame(rescued)
        print(f"  救回质检: shift中位 {df.shift_mm.median():.1f}mm | dx中位 {df.dx.median():+.1f} dy {df.dy.median():+.1f} dz {df.dz.median():+.1f} | mask_ratio中位 {df.mask_ratio.median():.2f} | <0.8: {((df.mask_ratio<0.8).sum())/len(df)*100:.0f}%")
        print(f"  n_vox 中位 {df.n_vox.median():.0f}")
    # 诊断1 细节落盘
    with open("logs/step2_diag.json", "w", encoding="utf-8") as f:
        json.dump({"fail_nv0": len(zeros), "fail_1_49": int(((nv>0)&(nv<50)).sum()),
                   "zero_details": zero_details,
                   "rescued": rescued}, f, ensure_ascii=False, indent=2)
    print("DONE → logs/step2_diag.json")


if __name__ == "__main__":
    main()
