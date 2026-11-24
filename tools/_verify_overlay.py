# -*- coding: utf-8 -*-
"""m05774 核查1b: 5 例 DWI+mask 叠图 (pid89 最大偏移 + from_c 2例 + from_t 2例 中位数附近)"""
import numpy as np, pandas as pd, os, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASK_BASE = "data/masks_totalseg_mr_d"
MASK_REMAP = "data/masks_dwi_remapped"
raw = pd.read_csv("data/hos_pairs_raw.csv").set_index("patient_id")
man = pd.read_csv("features/dwi_mask_remap_manifest.csv").set_index("pid")
out_dir = "outputs/dwi_remap_audit_highshift"
os.makedirs(out_dir, exist_ok=True)

cases = [(89, "from_c"), (226, "from_c"), (336, "from_c"), (263, "from_t"), (380, "from_t")]

for pid, ms in cases:
    dp = raw.loc[pid, "mri_path_d"]
    if not isinstance(dp, str) or not os.path.exists(dp):
        print(f"pid{pid}: mri_path_d 缺失, 跳过"); continue
    vol = nib.load(dp)
    vd = vol.get_fdata().astype(np.float32)  # (x,y,z)
    mp = os.path.join(MASK_BASE if ms == "self" else MASK_REMAP, str(pid), "colon.nii.gz")
    mimg = nib.load(mp)
    md = mimg.get_fdata() > 0.5
    nz = np.argwhere(md)
    if len(nz) == 0:
        print(f"pid{pid}: mask 空, 跳过"); continue
    # mask 体素最多的 z 层
    zc = np.bincount(nz[:, 2])
    z = int(np.argmax(zc))
    # 物理质心偏移 (相对卷中心)
    aff = mimg.affine
    vc = (np.asarray(vol.shape) - 1) / 2.0
    pc = aff @ np.r_[np.mean(nz, axis=0), 1.0]; pv = aff @ np.r_[vc, 1.0]
    off = np.linalg.norm(pc[:3] - pv[:3])
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a in ax:
        a.imshow(vd[:, :, z].T, cmap="gray", origin="lower", aspect="equal")
        a.contour(md[:, :, z].T, levels=[0.5], colors="red", linewidths=1.2)
        a.set_title(f"pid {pid} | z={z} | src={ms} | offset={off:.0f}mm | vox={len(nz)}", fontsize=9)
        a.axis("off")
    fig.suptitle(f"DWI slice (axial, max-mask layer) pid {pid}", fontsize=10)
    fig.tight_layout()
    out = os.path.join(out_dir, f"{pid}_{ms}.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"已导出 {out}  (mask层z={z}, mask体素={len(nz)}, 质心偏移={off:.1f}mm)")
print("完成")
