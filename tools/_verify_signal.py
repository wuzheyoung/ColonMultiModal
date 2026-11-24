# -*- coding: utf-8 -*-
"""量化验证借壳对齐质量: mask 区域 DWI 灰度 vs 卷整体 (self 为基准)。mask 落在组织(高b信号)上应显著高于背景"""
import numpy as np, pandas as pd, os, nibabel as nib
raw = pd.read_csv("data/hos_pairs_raw.csv").set_index("patient_id")
man = pd.read_csv("features/dwi_mask_remap_manifest.csv").set_index("pid")

def mask_stats(pid, grp):
    dp = raw.loc[pid, "mri_path_d"]
    if not (isinstance(dp, str) and os.path.exists(dp)):
        return None
    vol = nib.load(dp)
    vd = vol.get_fdata().astype(np.float32)
    mp = os.path.join("data/masks_totalseg_mr_d" if grp == "self" else "data/masks_dwi_remapped", str(pid), "colon.nii.gz")
    if not os.path.exists(mp):
        return None
    md = nib.load(mp).get_fdata() > 0.5
    n = int(md.sum())
    if n == 0:
        return None
    inside = vd[md]
    # 卷整体信号: 用非零体素 (避免背景0污染)
    nz = vd[vd > 0]
    return dict(pid=pid, n=n,
                mask_med=float(np.median(inside)),
                vol_med=float(np.median(nz)),
                mask_pct=float(np.percentile(nz, (np.percentile(inside, 50) / nz.max()) * 100) if nz.max() else 0),
                mask_ratio=float(np.median(inside) / (np.median(nz) + 1e-6)))

# self 基准 (抽样 40) + from_c 全部 + from_t 全部
res = []
for pid, ms in [(int(p), "self") for p in man[man.mask_source == "self"].index[:40]] + \
               [(int(p), "from_c") for p in man[man.mask_source == "from_c"].index] + \
               [(int(p), "from_t") for p in man[man.mask_source == "from_t"].index]:
    r = mask_stats(pid, ms)
    if r:
        r["grp"] = ms
        res.append(r)
df = pd.DataFrame(res)
print("mask区域DWI信号中位 / 卷非零体素中位 之比 (mask_ratio):  >1.5 高信号组织(肿瘤/壁) | ~1 同水平 | <0.8 低信号(背景/空气)")
for g in ("self", "from_c", "from_t"):
    s = df[df.grp == g]
    if len(s):
        print(f"  {g:8s} n={len(s):3d}  mask_ratio 中位={np.median(s.mask_ratio):.2f}  p25={np.percentile(s.mask_ratio,25):.2f} p75={np.percentile(s.mask_ratio,75):.2f}  <0.8比例={(s.mask_ratio<0.8).mean()*100:.0f}%  mask体素中位={np.median(s.n):.0f}")
# from_c 里 ratio 最低的 5 例
fc = df[df.grp == "from_c"].sort_values("mask_ratio")
print("\nfrom_c mask_ratio 最低 5 例 (疑似错位):")
print(fc[["pid", "mask_ratio", "n"]].head(5).to_string(index=False))
print("\nself 最低 3 例 (参照):")
print(df[df.grp == "self"].sort_values("mask_ratio")[["pid", "mask_ratio", "n"]].head(3).to_string(index=False))
