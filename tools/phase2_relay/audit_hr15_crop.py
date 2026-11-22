# -*- coding: utf-8 -*-
"""审计 cache_hos_hr15 固定 96mm 裁剪框对 mr_t 结肠 mask 的截断损失。
对每例: 1.5mm 网格上 mask 总体积 vs 以 center 裁 64³ 后框内保留体积,
输出损失比例分布 + 按轴统计 + 典型样本, 判断直肠段是否被裁出框外。
纯 CPU, 不占 GPU。"""
import os, sys, json, glob
import numpy as np
import SimpleITK as sitk

ROOT = r"D:\PyFile\ColonMultiModal"
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "phase2_relay"))
sys.path.insert(0, os.path.join(ROOT, "data_loader"))
from loader import _mri_bbox  # noqa: E402

HR_SPACING = (1.5, 1.5, 1.5)
HR_SIZE = 64
HALF = HR_SIZE // 2
CACHE_DIR = os.path.join(ROOT, "cache_hos_hr15")
MASK_ROOT = os.path.join(ROOT, "data", "masks_totalseg_mr_t")


def ref_grid_1p5(anchor):
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


def resample_to(img, ref, interp=sitk.sitkNearestNeighbor):
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(ref)
    rf.SetInterpolator(interp)
    rf.SetDefaultPixelValue(0.0)
    return rf.Execute(img)


def main():
    metas = []
    for jf in sorted(glob.glob(os.path.join(CACHE_DIR, "*.json"))):
        m = json.load(open(jf, encoding="utf-8"))
        if m.get("ok"):
            metas.append(m)
    print(f"ok 样本: {len(metas)}", flush=True)

    rows = []
    for i, m in enumerate(metas):
        pid = m["pid"]
        mask_t = os.path.join(MASK_ROOT, str(pid), "colon.nii.gz")
        t_path = os.path.join(ROOT, "data", "Hos", "_nifti", "MRI", f"{pid}_t.nii.gz")
        # batch2 路径兜底
        if not os.path.exists(t_path):
            import pandas as pd
            df = pd.read_csv(os.path.join(ROOT, "data", "hos_pairs_raw.csv"), encoding="utf-8-sig")
            r = df[df["patient_id"].astype(str).str.strip() == str(pid)]
            if not r.empty and isinstance(r.iloc[0].get("mri_path_t"), str):
                t_path = r.iloc[0]["mri_path_t"]
        if not os.path.exists(t_path):
            print(f"[skip] pid {pid} no t_path"); continue
        if not os.path.exists(mask_t):
            print(f"[skip] pid {pid} no mask"); continue
        t_img = sitk.ReadImage(t_path)
        ref = ref_grid_1p5(t_img)
        m15 = sitk.GetArrayFromImage(resample_to(sitk.ReadImage(mask_t), ref))
        m15 = (m15 > 0).astype(np.float32)
        total = float(m15.sum())
        if total <= 0:
            print(f"[skip] pid {pid} empty mask"); continue
        # 重建 center (与 step_b2 相同)
        cen = _mri_bbox(m15 > 0, pad=5)
        center = ((cen[0] + cen[1]) // 2, (cen[2] + cen[3]) // 2, (cen[4] + cen[5]) // 2)
        # 以 center 裁 64³ (clamp+pad0)
        D, H, W = m15.shape
        d0 = max(0, center[0] - HALF); d1 = min(D, center[0] + HALF)
        h0 = max(0, center[1] - HALF); h1 = min(H, center[1] + HALF)
        w0 = max(0, center[2] - HALF); w1 = min(W, center[2] + HALF)
        kept = float(m15[d0:d1, h0:h1, w0:w1].sum())
        loss = 1.0 - kept / total
        # 各轴 mask 分布超出裁剪框的比例
        ax_loss = {}
        for name, (lo, hi, a0, a1) in {
            "x": (cen[0], cen[1], center[0] - HALF, center[0] + HALF),
            "y": (cen[2], cen[3], center[1] - HALF, center[1] + HALF),
            "z": (cen[4], cen[5], center[2] - HALF, center[2] + HALF),
        }.items():
            ext = 0.0
            if hi > a1:
                sl = [slice(None)] * 3
                sl[{"x": 2, "y": 1, "z": 0}[name]] = slice(max(0, a1), hi)
                ext += float(m15[tuple(sl)].sum())
            if lo < a0:
                sl = [slice(None)] * 3
                sl[{"x": 2, "y": 1, "z": 0}[name]] = slice(lo, min(D, a0))
                ext += float(m15[tuple(sl)].sum())
            ax_loss[name] = ext / total
        rows.append({"pid": pid, "total_vox": total, "kept": kept, "loss": loss,
                     "x": ax_loss["x"], "y": ax_loss["y"], "z": ax_loss["z"]})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(metas)} done", flush=True)

    n = len(rows)
    losses = np.array([r["loss"] for r in rows])
    print(f"\n=== 结果 (n={n}) ===")
    print(f"mask 总体积被裁出框外比例 loss: mean={losses.mean():.4f} "
          f"median={np.median(losses):.4f} max={losses.max():.4f}")
    for th in (0.05, 0.10, 0.20, 0.50):
        c = int((losses > th).sum())
        print(f"  loss>{th}: {c} 例 ({100*c/n:.1f}%)")
    print(f"\n各轴单独被裁比例: x mean={np.mean([r['x'] for r in rows]):.4f} "
          f"y mean={np.mean([r['y'] for r in rows]):.4f} "
          f"z mean={np.mean([r['z'] for r in rows]):.4f}")
    print(f"z 轴(纵轴/直肠走行)被裁>10%: {sum(1 for r in rows if r['z']>0.10)} 例 "
          f"({100*sum(1 for r in rows if r['z']>0.10)/n:.1f}%)")
    print("\n=== 损失最大 15 例 ===")
    for r in sorted(rows, key=lambda r: -r["loss"])[:15]:
        print(f"pid {r['pid']}: loss={r['loss']:.3f} "
              f"(x={r['x']:.3f} y={r['y']:.3f} z={r['z']:.3f}) total={r['total_vox']:.0f}")

    out = os.path.join(ROOT, "logs", "audit_hr15_crop.json")
    json.dump({"n": n, "loss_mean": float(losses.mean()), "loss_median": float(np.median(losses)),
               "loss_max": float(losses.max()), "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n落盘: {out}")


if __name__ == "__main__":
    main()
