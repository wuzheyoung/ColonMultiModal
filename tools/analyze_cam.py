"""分析 3D Grad-CAM 热图 (npy): 每样本报告
- 热图最大值所在轴向层 argmax_slice (0-95)
- 最大层上高值区占比 hot% / 质心 / 离中心距离 / 集中度 conc
用法: python tools/analyze_cam.py [npy目录] [n样本]
"""
import sys, glob, os
import numpy as np


def analyze(cam):
    """cam: [96,96,96] 非负热图"""
    vol_max = cam.max()
    if vol_max < 1e-8:
        return None
    z = cam.argmax() // (96 * 96)          # 最大值所在轴向层
    sl = cam[z]
    thr = 0.3 * sl.max()                    # 高值阈值
    hot = sl > thr
    frac = hot.mean()
    if frac < 0.001:
        return (z, frac, None, None, 0.0)
    ys, xs = np.nonzero(hot)
    cy, cx = ys.mean() / 96, xs.mean() / 96
    center_d = np.sqrt((cy - 0.5) ** 2 + (cx - 0.5) ** 2)
    k = max(1, int(0.2 * 96 * 96))
    flat = hot.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    conc = flat[idx].sum() / k / (frac + 1e-9)
    return (z, frac, (cx, cy), center_d, conc)


def main():
    npy_dir = sys.argv[1] if len(sys.argv) > 1 else "plots_hos_full/cam_npy"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ct_files = sorted(glob.glob(os.path.join(npy_dir, "cam_ct_*.npy")))[:n]
    print(f"{'file':<12} {'panel':<5} {'argmax_slice':>12} {'hot%':>7} {'centroid':>12} {'center_d':>8} {'conc':>6}")
    for f in ct_files:
        base = os.path.basename(f)
        idx = base.replace("cam_ct_", "").replace(".npy", "")
        for panel, key in (("CT", "ct"), ("MRI", "mri")):
            p = os.path.join(npy_dir, f"cam_{key}_{idx}.npy")
            if not os.path.exists(p):
                continue
            cam = np.load(p)
            r = analyze(cam)
            if r is None:
                print(f"cam_{idx}.png  {panel:<5}  (全零热图)")
                continue
            z, frac, cent, cd, conc = r
            cstr = f"({cent[0]:.2f},{cent[1]:.2f})" if cent else "-"
            print(f"cam_{idx}.png  {panel:<5} {z:>12} {frac * 100:>6.1f}% {cstr:>12} {cd if cd is not None else -1:>8.3f} {conc:>6.2f}")


if __name__ == "__main__":
    main()
