# -*- coding: utf-8 -*-
"""验证 from_c 系统性 z 偏移根因: 对比 mri_c 与 mri_d 卷 origin z / spacing / shape"""
import numpy as np, pandas as pd, os, nibabel as nib
raw = pd.read_csv("data/hos_pairs_raw.csv").set_index("patient_id")
man = pd.read_csv("features/dwi_mask_remap_manifest.csv").set_index("pid")
fc = man[man.mask_source == "from_c"].index.tolist()
print(f"from_c 例数: {len(fc)}")
rows = []
for pid in fc[:40]:  # 抽前 40 例
    cp = raw.loc[pid, "mri_path_c"]
    dp = raw.loc[pid, "mri_path_d"]
    if not (isinstance(cp, str) and os.path.exists(cp) and isinstance(dp, str) and os.path.exists(dp)):
        continue
    cv = nib.load(cp); dv = nib.load(dp)
    rows.append(dict(pid=pid,
                     c_origin_z=cv.affine[2, 3], d_origin_z=dv.affine[2, 3],
                     dz_origin=cv.affine[2, 3] - dv.affine[2, 3],
                     c_shape_z=cv.shape[2], d_shape_z=dv.shape[2],
                     c_sp_z=cv.header.get_zooms()[2], d_sp_z=dv.header.get_zooms()[2]))
df = pd.DataFrame(rows)
print(f"n={len(df)}  c与d的 z-origin 差 (c-d): 中位={np.median(df.dz_origin):.1f}mm  min={df.dz_origin.min():.1f} max={df.dz_origin.max():.1f}")
print(f"c卷z层数中位: {np.median(df.c_shape_z):.0f} (spacing {np.median(df.c_sp_z):.3f}mm)  d卷z层数中位: {np.median(df.d_shape_z):.0f} (spacing {np.median(df.d_sp_z):.3f}mm)")
print("\n前 12 例明细:")
print(df.head(12).to_string(index=False))
