# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
df = pd.read_csv("outputs/_dwi_offset_axis.csv")
for ms in ("self", "from_c", "from_t"):
    s = df[df.ms == ms]
    if len(s):
        print(f"{ms:8s} n={len(s):3d}  dx={np.median(s.dx):6.1f} dy={np.median(s.dy):6.1f} dz={np.median(s.dz):6.1f} | off中位={np.median(s.off):5.1f} max={s.off.max():5.1f}")
fc = df[df.ms == "from_c"]
print()
print("from_c dz<-50 例数:", (fc.dz < -50).sum(), "/", len(fc))
print("from_c |dz|>50 例数:", (fc.dz.abs() > 50).sum())
print("pid89:", df.loc[df.pid == 89, ["dx", "dy", "dz", "off"]].to_dict("records"))
# z 质心相对卷底部比例 (dz 大负值 → mask 贴底)
print("\nfrom_c dz 分位:", np.percentile(fc.dz, [5, 25, 50, 75, 95]).round(1))
ft = df[df.ms == "from_t"]
print("from_t dz 分位:", np.percentile(ft.dz, [5, 25, 50, 75, 95]).round(1))
