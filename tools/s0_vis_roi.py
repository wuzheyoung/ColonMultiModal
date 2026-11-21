"""S0: 坐标核查可视化 — 验证 ring mask 质心 64³ 裁剪与缓存卷解剖对齐。

对每个 pid:
  1. 读缓存卷 MRI t 通道 [96,96,96] (模型实际输入灰度, 1mm+bbox+trilinear)
  2. 读 ring mask → Resize 96³ nearest (模型实际用于质心定位)
  3. crop_roi 质心 (与 train_roi_dl.py 完全同逻辑)
  4. 三视图 (轴向 z=cen_z / 冠状 y=cen_y / 矢状 x=cen_x) 叠加:
     灰度 + ring mask 轮廓(红) + 本体结肠 mask 轮廓(绿) + 64³ 裁剪框(黄)

用法:
  python tools/s0_vis_roi.py --pids 151,153,155 --outdir plots_hos_full/s0_vis
  python tools/s0_vis_roi.py --n 5 --random --outdir plots_hos_full/s0_vis
"""
import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loader.loader import _det_cache_key_mri, _DetDiskCache  # noqa: E402
from tools.train_roi_dl import load_mask_96, crop_roi, SPATIAL, MASK_DIR_MR, RING_ITERATIONS  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
# 中文字体 (Windows 黑体)
from matplotlib import font_manager  # noqa: E402
for _fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = "Microsoft YaHei" if "msyh" in _fp else "SimHei"
        break
plt.rcParams["axes.unicode_minus"] = False


def _vol_t(pid, disk):
    """读缓存卷并取 t 通道 [96,96,96]。"""
    t_path = c_path = d_path = None
    # 从 CSV 找 pid 的 mri_path_t (hos_pairs.csv 有全量列)
    import pandas as pd
    for csv in ("data/hos_pairs.csv", "data/hos_full_train_s42.csv", "data/hos_full_val_s42.csv", "data/hos.csv"):
        p = os.path.join(ROOT, csv)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        hit = df[df["patient_id"].astype(str).str.strip() == str(pid)]
        if len(hit) > 0 and "mri_path_t" in df.columns:
            t_path = hit.iloc[0].get("mri_path_t", "")
            c_path = hit.iloc[0].get("mri_path_c", "")
            d_path = hit.iloc[0].get("mri_path_d", "")
            if isinstance(t_path, str) and t_path and os.path.exists(t_path):
                break
    if not t_path or not os.path.exists(t_path):
        return None, None, None, None
    # 缓存 key 是 t+c+d 三序列一起生成的 (与 train_roi_dl 一致)
    key = _det_cache_key_mri(t_path, c_path, d_path, SPATIAL)
    vol, hit = disk.get_vol(key)
    if vol is None or not hit:
        return None, None, None, None
    v = vol.numpy()
    return (v[0] if v.ndim == 4 else v), t_path, c_path, d_path


def vis_one(pid, disk, outdir):
    vol, t_path, c_path, d_path = _vol_t(pid, disk)
    if vol is None:
        print(f"[{pid}] 缓存卷缺失 (t_path={t_path})")
        return False
    # 用与训练一致的 aligned 几何绘制 mask 轮廓 (旧版直接 Resize, 会与缓存卷错位)
    seq_paths = (t_path, c_path, d_path)
    ring = load_mask_96(pid, "mr_t_ring", aligned=True, seq_paths=seq_paths)
    body = load_mask_96(pid, "mr_t", aligned=True, seq_paths=seq_paths)
    if ring is None:
        print(f"[{pid}] ring mask 为空/缺失")
        return False
    roi = crop_roi(vol, ring)
    if roi is None:
        print(f"[{pid}] crop_roi 返回 None")
        return False
    cen = np.argwhere(ring > 0).mean(axis=0).astype(int)  # (z,y,x) 与 train_roi_dl 一致
    z, y, x = int(cen[0]), int(cen[1]), int(cen[2])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [f"axial z={z}", f"coronal y={y}", f"sagittal x={x}"]
    slices = [(vol[z], 0), (vol[:, y, :], 1), (vol[:, :, x], 2)]
    ring_sl = [(ring[z], 0), (ring[:, y, :], 1), (ring[:, :, x], 2)]
    body_sl = [(body[z], 0) if body is not None else (None, 0),
               (body[:, y, :], 1) if body is not None else (None, 1),
               (body[:, :, x], 2) if body is not None else (None, 2)]
    for ax, (sl, ax_), title, (rs, _), (bs, _) in zip(axes, slices, titles, ring_sl, body_sl):
        ax.imshow(sl.T, cmap="gray", origin="lower")
        ax.contour(rs.T, levels=[0.5], colors="red", linewidths=1.0)          # ring 轮廓
        if bs is not None:
            ax.contour(bs.T, levels=[0.5], colors="lime", linewidths=0.8)      # 本体结肠轮廓
        # 64³ 裁剪框 (在 96³ 坐标系中)
        rng = [(max(0, z - 32), min(96, z + 32)),
               (max(0, y - 32), min(96, y + 32)),
               (max(0, x - 32), min(96, x + 32))]
        if ax_ == 0:
            ax.add_patch(plt.Rectangle((rng[1][0] - 0.5, rng[2][0] - 0.5),
                                       rng[1][1] - rng[1][0], rng[2][1] - rng[2][0],
                                       fill=False, edgecolor="yellow", linewidth=1.2))
        elif ax_ == 1:
            ax.add_patch(plt.Rectangle((rng[2][0] - 0.5, rng[0][0] - 0.5),
                                       rng[2][1] - rng[2][0], rng[0][1] - rng[0][0],
                                       fill=False, edgecolor="yellow", linewidth=1.2))
        else:
            ax.add_patch(plt.Rectangle((rng[1][0] - 0.5, rng[0][0] - 0.5),
                                       rng[1][1] - rng[1][0], rng[0][1] - rng[0][0],
                                       fill=False, edgecolor="yellow", linewidth=1.2))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"pid={pid}  T={t_path.split(os.sep)[-1] if t_path else '?'}  "
                 f"ring质心=({z},{y},{x})  roi均值={roi.mean():.3f}")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"s0_{pid}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[{pid}] 已出图 (ring质心 {z},{y},{x})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default="", help="逗号分隔 pid 列表")
    ap.add_argument("--n", type=int, default=5, help="随机取 n 例")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "plots_hos_full", "s0_vis"))
    args = ap.parse_args()

    disk = _DetDiskCache(os.path.join(ROOT, "cache_hos"), SPATIAL)
    if args.pids:
        pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    else:
        import glob
        import random
        dirs = sorted(glob.glob(os.path.join(ROOT, "data", "masks_totalseg_mr_t", "*")))
        pids = [os.path.basename(d) for d in dirs]
        if args.random:
            random.seed(42)
            random.shuffle(pids)
        pids = pids[:args.n]
    ok = 0
    for pid in pids:
        if vis_one(pid, disk, args.outdir):
            ok += 1
    print(f"\n完成: {ok}/{len(pids)} 例 → {args.outdir}")


if __name__ == "__main__":
    main()
