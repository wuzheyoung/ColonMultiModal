# -*- coding: utf-8 -*-
"""B2: 构建固定物理视野高分辨缓存 cache_hos_hr15/
规则(用户 m00745 B2): 1.5mm³ 各向同性, mr_t 结肠 bbox 中心裁固定 64³=物理 96×96×96mm,
严禁拉伸(不做变比例 Resize); 同步导出 T2(t)/CE(c)/DWI(d) 三通道 + SDF 通道;
质检图 pid 213/377/402 三视图 PNG。
输出: cache_hos_hr15/{key}.npy  = [4,64,64,64] (ch0-2=t/c/d 图像, ch3=SDF)
      cache_hos_hr15/{key}.json = meta (pid, bbox, origin, spacing, mask_sum)
      tools/phase2_relay/outputs/hr15_qc_p{pid}.png (三视图)
      logs/step_b2_build_hr15.json (汇总)
用法: python tools/phase2_relay/step_b2_build_hr15.py [--pids "213,377,402" 仅质检]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import SimpleITK as sitk

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, "cache_hos_hr15")
QC_DIR = os.path.join(ROOT, "tools", "phase2_relay", "outputs")
HR_SPACING = (1.5, 1.5, 1.5)
HR_SIZE = 64  # 64³ = 物理 96mm
HALF = HR_SIZE // 2
SDF_CLIP_MM = 15.0

sys.path.insert(0, os.path.join(ROOT, "data_loader"))
from loader import _mri_bbox, _zscore_nonzero  # noqa: E402


def _ref_grid_1p5(anchor):
    """以 anchor 物理范围构建 1.5mm 各向同性参考网格。"""
    rf = sitk.ResampleImageFilter()
    phys = np.array(anchor.GetSize()) * np.array(anchor.GetSpacing())
    new_size = np.ceil(phys / np.array(HR_SPACING)).astype(int).tolist()
    rf.SetOutputSpacing(HR_SPACING)
    rf.SetOutputDirection(anchor.GetDirection())
    rf.SetOutputOrigin(anchor.GetOrigin())
    rf.SetSize(new_size)
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(0.0)
    return rf.Execute(anchor)


def _resample_to(img, ref, interp=sitk.sitkLinear):
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(ref)
    rf.SetInterpolator(interp)
    rf.SetDefaultPixelValue(0.0)
    return rf.Execute(img)


def _crop64(arr, center):
    """arr [D,H,W] 任意 → 以 center 为中心裁 64³ (clamp+pad0)。"""
    roi = np.zeros((HR_SIZE,) * 3, dtype=np.float32)
    d0 = max(0, center[0] - HALF); d1 = min(arr.shape[0], center[0] + HALF)
    h0 = max(0, center[1] - HALF); h1 = min(arr.shape[1], center[1] + HALF)
    w0 = max(0, center[2] - HALF); w1 = min(arr.shape[2], center[2] + HALF)
    src = arr[d0:d1, h0:h1, w0:w1]
    od0 = max(0, HALF - center[0]); oh0 = max(0, HALF - center[1]); ow0 = max(0, HALF - center[2])
    roi[od0:od0 + src.shape[0], oh0:oh0 + src.shape[1], ow0:ow0 + src.shape[2]] = src
    return roi


def _sdf64(mask64, spacing_mm=1.5):
    """64³ mask → 物理距离场 SDF (采样间距 1.5mm), clip ±15mm 归一 [-1,1]。"""
    from scipy.ndimage import distance_transform_edt
    m = mask64 > 0
    if not m.any():
        return None
    sdf = (distance_transform_edt(~m, sampling=spacing_mm)
           - distance_transform_edt(m, sampling=spacing_mm)).astype(np.float32)
    return np.clip(sdf, -SDF_CLIP_MM, SDF_CLIP_MM) / SDF_CLIP_MM


def build_one(pid, pairs_row, full_csv_df):
    t_path = str(pairs_row.get("mri_path_t", ""))
    c_path = str(pairs_row.get("mri_path_c", ""))
    d_path = str(pairs_row.get("mri_path_d", ""))
    mask_t = os.path.join(ROOT, "data", "masks_totalseg_mr_t", str(pid), "colon.nii.gz")
    if not t_path or not os.path.exists(t_path):
        return {"pid": pid, "ok": False, "err": "no t_path"}
    if not os.path.exists(mask_t):
        return {"pid": pid, "ok": False, "err": "no mr_t mask"}

    t_img = sitk.ReadImage(t_path)
    # 1) 锚: mr_t 结肠 mask 重采样到 1.5mm 网格 → bbox 中心
    ref = _ref_grid_1p5(t_img)
    t15 = _resample_to(t_img, ref)
    if os.path.exists(mask_t):
        m_img = sitk.ReadImage(mask_t)
        m15 = sitk.GetArrayFromImage(_resample_to(m_img, ref, sitk.sitkNearestNeighbor))
    else:
        m15 = np.zeros(sitk.GetArrayFromImage(t15).shape, dtype=np.uint8)
    if float((m15 > 0).sum()) <= 0:
        return {"pid": pid, "ok": False, "err": "empty mask after resample"}
    cen = _mri_bbox(m15 > 0, pad=5)
    center = tuple(((cen[0] + cen[1]) // 2, (cen[2] + cen[3]) // 2, (cen[4] + cen[5]) // 2))

    # 2) t/c/d 图像通道: 全部重采样到 ref (t 已是 ref), 各向同性 1.5mm, 归一化后同中心裁 64³
    vols = []
    for p in (t_path, c_path, d_path):
        if p and os.path.exists(p):
            a = sitk.GetArrayFromImage(_resample_to(sitk.ReadImage(p), ref))  # [D,H,W]
            vols.append(_crop64(_zscore_nonzero(a), center))
        else:
            vols.append(np.zeros((HR_SIZE,) * 3, dtype=np.float32))
    img4 = np.stack(vols, axis=0)  # [3,64,64,64]

    # 3) SDF 通道: mr_t mask 在 1.5mm 网格的 64³ 二值 → 物理距离场
    m64 = _crop64((m15 > 0).astype(np.float32), center)
    sdf = _sdf64(m64 > 0.5)
    if sdf is None:
        return {"pid": pid, "ok": False, "err": "sdf empty"}
    vol4 = np.concatenate([img4, sdf[None].astype(np.float32)], axis=0)  # [4,64,64,64]

    import hashlib
    key = hashlib.sha1(f"hr15v1|{pid}".encode("utf-8")).hexdigest()[:20]
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, f"{key}.npy"), vol4)
    meta = {
        "pid": pid, "key": key, "ok": True, "center": [int(x) for x in center],
        "mask_bbox": [int(x) for x in cen], "spacing_mm": list(HR_SPACING),
        "size": HR_SIZE, "mask_sum": int((m64 > 0.5).sum()),
        "t/c/d": all(bool(v.sum() > 0) for v in vols),
    }
    with open(os.path.join(CACHE_DIR, f"{key}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return meta


def qc(pids, full_csv_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for pid in pids:
        row = full_csv_df[full_csv_df["patient_id"].astype(str).str.strip() == str(pid)]
        if row.empty:
            print(f"[qc] pid {pid} 不在 pairs 表, 跳过")
            continue
        meta = build_one(str(pid), row.iloc[0], full_csv_df)
        if not meta.get("ok"):
            print(f"[qc] pid {pid} build 失败: {meta.get('err')}")
            continue
        vol = np.load(os.path.join(CACHE_DIR, f"{meta['key']}.npy"))
        os.makedirs(QC_DIR, exist_ok=True)
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        mid = np.array(meta["center"]) // (HR_SPACING[0] / 1.0)  # 网格坐标近似
        z, h, w = vol.shape[2] // 2, vol.shape[1] // 2, vol.shape[2] // 2
        for i, ax in enumerate(axes[:, 0]):
            ax.imshow(vol[i, z, :, :], cmap="gray"); ax.set_title(f"ch{i} axial"); ax.axis("off")
        for i, ax in enumerate(axes[:, 1]):
            ax.imshow(vol[i, :, h, :], cmap="gray"); ax.set_title(f"ch{i} coronal"); ax.axis("off")
        for i, ax in enumerate(axes[:, 2]):
            ax.imshow(vol[i, :, :, w], cmap="gray"); ax.set_title(f"ch{i} sagittal"); ax.axis("off")
        fig.suptitle(f"pid {pid} hr15 qc (mask_sum={meta['mask_sum']})")
        fig.tight_layout()
        out = os.path.join(QC_DIR, f"hr15_qc_p{pid}.png")
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print(f"[qc] {out} saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default=None, help="仅质检, 逗号分隔, 如 213,377,402")
    ap.add_argument("--limit", type=int, default=0, help="全量构建上限 0=全部")
    ap.add_argument("--only-missing", action="store_true", help="仅构建尚未有缓存的样本")
    ap.add_argument("--tag-prefix", default="dc", help="由 relay 传入, 本步不使用 (兼容统一调用)")
    args = ap.parse_args()

    pairs = os.path.join(ROOT, "data", "hos_pairs.csv")
    df = pd.read_csv(pairs, encoding="utf-8-sig")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    if args.pids:
        qc([p.strip() for p in args.pids.split(",") if p.strip()], df)
        return 0

    done_keys = set()
    if args.only_missing and os.path.isdir(CACHE_DIR):
        done_keys = {json.load(open(os.path.join(CACHE_DIR, f))).get("pid")
                     for f in os.listdir(CACHE_DIR) if f.endswith(".json")}

    results = []
    n_ok = n_fail = 0
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        pid = str(row["patient_id"]).strip()
        if args.limit and i >= args.limit:
            break
        if args.only_missing and pid in done_keys:
            continue
        r = build_one(pid, row, df)
        results.append(r)
        if r.get("ok"):
            n_ok += 1
        else:
            n_fail += 1
        if (i + 1) % 50 == 0:
            print(f"[{i+1}] ok={n_ok} fail={n_fail} ({time.time()-t0:.0f}s)")
    print(f"done: ok={n_ok} fail={n_fail} total={len(results)} ({time.time()-t0:.0f}s)")
    out = os.path.join(ROOT, "logs", "step_b2_build_hr15.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_ok": n_ok, "n_fail": n_fail, "results": results},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "README.md"), "a", encoding="utf-8") as f:
        f.write(f"\n## B2 高分辨缓存构建 (自动接力 {time.strftime('%Y-%m-%d %H:%M')})\n"
                f"- cache_hos_hr15/ 1.5mm³ 各向同性, mr_t bbox 中心裁 64³=物理96mm, "
                f"4通道(t/c/d灰度+SDF), 严禁拉伸; 质检图 pid 213/377/402 见 outputs/\n"
                f"- 结果: ok={n_ok} fail={n_fail} (logs/step_b2_build_hr15.json)\n")
    return 0


if __name__ == "__main__":
    import pandas as pd
    sys.exit(main())
