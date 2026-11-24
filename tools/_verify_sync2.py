# -*- coding: utf-8 -*-
"""m05774 核查1a: 三轴偏移分解 + FOV 对比; 核查2: K 分布"""
import numpy as np, pandas as pd, os, nibabel as nib, glob

FEAT = "features/slice_feats_v2"
MASK_BASE = f"data/masks_totalseg_mr_d"
MASK_REMAP = "data/masks_dwi_remapped"
raw = pd.read_csv("data/hos_pairs_raw.csv").set_index("patient_id")
man = pd.read_csv("features/dwi_mask_remap_manifest.csv").set_index("pid")

def load_mask(pid, ms):
    mp = os.path.join(MASK_BASE if ms == "self" else MASK_REMAP, str(pid), "colon.nii.gz")
    return nib.load(mp) if os.path.exists(mp) else None

rows = []
for p in sorted(glob.glob(f"{FEAT}/dinov2/d/*.npz")):
    pid = int(os.path.basename(p)[:-4])
    ms = man.loc[pid, "mask_source"]
    grp = "self" if ms == "self" else "remapped"
    mimg = load_mask(pid, ms)
    if mimg is None:
        continue
    md = mimg.get_fdata()
    nz = np.argwhere(md > 0)
    if len(nz) == 0:
        continue
    dp = raw.loc[pid, "mri_path_d"]
    if not isinstance(dp, str) or not os.path.exists(dp):
        continue
    vimg = nib.load(dp)
    vshape = np.asarray(vimg.shape)
    vc = (vshape - 1) / 2.0
    aff = mimg.affine
    pc = aff @ np.r_[np.mean(nz, axis=0), 1.0]
    pv = aff @ np.r_[vc, 1.0]
    d = pc[:3] - pv[:3]
    zoom = np.asarray(vimg.header.get_zooms())[:3]
    fov = vshape * zoom
    rows.append(dict(pid=pid, grp=grp, ms=ms, dx=d[0], dy=d[1], dz=d[2],
                     off=np.linalg.norm(d), fov_x=fov[0], fov_y=fov[1], fov_z=fov[2]))

df = pd.DataFrame(rows)
print("=" * 70)
print("[核查1a] 三轴质心偏移 (mask质心 - DWI卷中心, mm) + FOV 对比")
for g in ("self", "remapped"):
    sub = df[df.grp == g]
    print(f"\n--- {g} (n={len(sub)}) ---")
    print(f"  三轴偏移中位数: dx={np.median(sub.dx):.1f}  dy={np.median(sub.dy):.1f}  dz={np.median(sub.dz):.1f}")
    print(f"  总偏移中位数: {np.median(sub.off):.1f}  最大: {sub.off.max():.1f} (pid {sub.loc[sub.off.idxmax(),'pid']})")
    print(f"  FOV中位数(x,y,z mm): {np.median(sub.fov_x):.0f} / {np.median(sub.fov_y):.0f} / {np.median(sub.fov_z):.0f}")
    print(f"  FOV 最大(z): {sub.fov_z.max():.0f}mm")
print("\n  self vs remapped FOV-z 对比: "
      f"self {np.median(df[df.grp=='self'].fov_z):.0f}mm vs remapped {np.median(df[df.grp=='remapped'].fov_z):.0f}mm")

print("=" * 70)
print("[核查2] dinov2/{t,c,d} 切片数 K (feats.shape[0]) 分布")
for s in "tcd":
    ks = []
    for p in glob.glob(f"{FEAT}/dinov2/{s}/*.npz"):
        ks.append(int(np.load(p)["feats"].shape[0]))
    ks = np.array(sorted(ks))
    print(f"  {s}: n={len(ks)}  min={ks.min()}  p50={np.median(ks):.0f}  p95={np.percentile(ks,95):.0f}  max={ks.max()}")
# c/1 具体
d1 = np.load(f"{FEAT}/dinov2/c/1.npz")
print(f"  c/1.npz: K={d1['feats'].shape[0]}, rel_z[0..2]={d1['rel_z'][:3]}, rel_z[-2:]={d1['rel_z'][-2:]}")

# 保存质心数据供叠图用
df.to_csv("outputs/_dwi_offset_axis.csv", index=False)
print("\n已存 outputs/_dwi_offset_axis.csv")
# 输出 from_c / from_t 中位数附近 pid
for ms in ("from_c", "from_t"):
    sub = df[(df.ms == ms)]
    if len(sub):
        med = sub.off.median()
        idx = (sub.off - med).abs().argsort().iloc[:2]
        print(f"  {ms} 中位数附近2例: {sorted(sub.iloc[idx].pid.tolist())} (偏移 {sorted(sub.iloc[idx].off.round(1).tolist())})")
